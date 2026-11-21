"""White-box lowered-op measured-binding, node-shape, and timed-node parity verification.

[Role] Verify that the lowered ``mindie-sim`` replay target (L1 single-op, L2
  subgraph, or L3 1-layer model step) matches the measured silicon profiler
  capture's recipe, paged layout, KV length, page count, indexer ``topk`` and
  dtypes, GEMM/grouped-GEMM per-node ``(m, n, k, matrix_groups)``, DCP
  ``attention_merge`` dtype/``partial_count``, RMSNorm width, and exact
  topologically ordered ``timed_nodes`` sequence.
[Invariants]
  * Multi-GEMM subgraphs with ``actual_shape`` MUST provide per-node
    ``actual_shape["node_shapes"][op_id]`` for EVERY GEMM (preventing shared
    top-level scalar masking).
  * Planned L2/L3 captures MUST provide explicit ``timed_nodes`` matching the
    exact lowered op set and dependency DAG order.
  * Never silently falls back to planned dimensions when writer-reported
    ``actual_shape`` fields conflict or are missing on required nodes.
[Timeouts & Locks] Pure in-memory contract verification; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.schema.common import ModelingError

from ._profiler_geometry import _norm_timed_op_id, validate_timed_nodes_sequence

_INDEXER_SHAPE_MIN_RANK = 3
_GEMM_SHAPE_MIN_RANK = 3
_BLOCK_TABLE_PAIR_LEN = 2
_MIN_DCP_PARTIALS = 2
_GEMM_FAMILIES = frozenset({"gemm", "gate_up_projection", "down_projection"})
_MLA_RECIPES = frozenset({
    "ascend_paged_mla_decode_arch22",
    "ascend_paged_mla_decode_arch35",
    "latent_attention0",
    "latent_attention_batch",
    "paired_mla_attention_analytic",
})
_GQA_RECIPES = frozenset({
    "fused_infer_attention",
    "flash_attention",
    "flash_attention_prefill",
    "flash_attention_decode",
})


def _opt_pos_int(*values: Any) -> int | None:
    for val in values:
        if val is not None and not isinstance(val, bool):
            try:
                ival = int(val)
                if ival > 0:
                    return ival
            except (TypeError, ValueError):
                continue
    return None


def _extract_lowered_attention_contract(
    op: Any, sim_case: Any
) -> tuple[int, int, str, int, int]:
    """Return ``(latent_or_v_dim, cache_head_dim, query_layout, kv_len, page_count)``."""
    meta = dict(getattr(op, "metadata", ()) or ())
    nabi = meta.get("native_attention_abi") if isinstance(meta.get("native_attention_abi"), Mapping) else {}
    pgeo = meta.get("paged_mla_geometry") if isinstance(meta.get("paged_mla_geometry"), Mapping) else {}
    loc = getattr(op, "local_shape", None)
    loc_seq = loc if isinstance(loc, Sequence) and not isinstance(loc, (str, bytes)) else ()

    if nabi.get("latent_width"):
        lat = int(nabi["latent_width"])
        rope = int(nabi.get("rope_width") or 0)
        primary, cache_dim = lat, lat + rope
    else:
        loc_dim = int(loc_seq[-1]) if loc_seq else 0
        atv = meta.get("attention_tensor_view")
        cache_dim = 0
        if isinstance(atv, Mapping):
            csh = atv.get("cache_head_shape")
            if isinstance(csh, Sequence) and not isinstance(csh, (str, bytes)) and csh:
                cache_dim = int(csh[-1])
        fallback = int(getattr(sim_case.model, "head_dim", 0) or 0)
        primary = loc_dim or cache_dim or fallback
        cache_dim = cache_dim or primary

    q_layout = str(nabi.get("query_layout") or "").strip().upper()
    if not q_layout and str(getattr(op, "recipe", "")) == "latent_attention0":
        q_layout = "TND"

    kv_len = 0
    if isinstance(pgeo.get("kv_lengths"), Sequence) and pgeo["kv_lengths"]:
        kv_len = int(pgeo["kv_lengths"][0])
    elif str(getattr(op, "recipe", "")) == "latent_attention0" and len(loc_seq) >= _BLOCK_TABLE_PAIR_LEN:
        kv_len = int(loc_seq[1])

    page_count = 0
    if isinstance(pgeo.get("page_counts"), Sequence) and pgeo["page_counts"]:
        page_count = int(pgeo["page_counts"][0])
    elif nabi.get("page_entries"):
        page_count = int(nabi["page_entries"])
    return primary, cache_dim, q_layout, kv_len, page_count


def _extract_block_table_pages(mapping: Mapping[str, Any]) -> int | None:
    bt = mapping.get("block_table")
    if (
        isinstance(bt, Sequence)
        and not isinstance(bt, (str, bytes))
        and len(bt) >= _BLOCK_TABLE_PAIR_LEN
    ):
        return _opt_pos_int(bt[1])
    return None


def _verify_attention_op_binding(
    target_op: Any,
    sim_case: Any,
    binding: Mapping[str, Any],
    actual: Mapping[str, Any],
    plan_key: Mapping[str, Any],
    *,
    local_kv_heads: int,
    case_id: str,
) -> None:
    meas_rec = str(binding.get("recipe") or plan_key.get("recipe") or "").strip()
    if meas_rec in _GQA_RECIPES and target_op.recipe in _MLA_RECIPES:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:recipe:measured={meas_rec}:lowered={target_op.recipe}"
        )
    if meas_rec in _MLA_RECIPES and target_op.recipe not in _MLA_RECIPES:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:recipe:measured={meas_rec}:lowered={target_op.recipe}"
        )
    if binding.get("recipe") and str(binding["recipe"]).strip() != target_op.recipe:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:recipe:measured={binding['recipe']}:lowered={target_op.recipe}"
        )
    if binding.get("paged") is False and (target_op.recipe in _MLA_RECIPES or "paged" in target_op.op_id):
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:layout:measured_unpaged_vs_lowered_paged"
        )
    lowered_kv = max(1, int(local_kv_heads))
    tp = max(1, int(getattr(sim_case.runtime.parallel, "tp", 1) or 1))
    lowered_heads = max(1, int(getattr(sim_case.model, "num_heads", 1) or 1) // tp)
    lowered_dim, lowered_cache_dim, lowered_layout, lowered_kv_len, lowered_pages = (
        _extract_lowered_attention_contract(target_op, sim_case)
    )
    meas_kv = _opt_pos_int(actual.get("kv_heads"), binding.get("kv_heads"))
    meas_dim = _opt_pos_int(actual.get("latent"), actual.get("head_dim"), binding.get("latent"), binding.get("head_dim"))
    meas_rope = _opt_pos_int(actual.get("rope"), binding.get("rope")) or 0
    meas_heads = _opt_pos_int(actual.get("heads"), binding.get("heads"))
    meas_layout = str(
        actual.get("query_layout") or actual.get("input_layout")
        or binding.get("query_layout") or binding.get("input_layout") or ""
    ).strip().upper()
    meas_kv_len = _opt_pos_int(actual.get("seq_len"), actual.get("kv_len"), binding.get("kv_len"))
    actual_explicit_pages = _opt_pos_int(actual.get("page_count"))
    actual_bt_pages = _extract_block_table_pages(actual)
    binding_pages = _opt_pos_int(binding.get("page_count"))
    if (
        actual_explicit_pages is not None
        and actual_bt_pages is not None
        and actual_explicit_pages != actual_bt_pages
    ):
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:page_count_block_table_conflict"
            f":page_count={actual_explicit_pages}:block_table_pages={actual_bt_pages}"
        )
    if meas_kv is not None and meas_kv != lowered_kv:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:kv_heads:measured={meas_kv}:lowered={lowered_kv}"
        )
    if meas_dim is not None and lowered_dim > 0 and meas_dim != lowered_dim:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:head_dim:measured={meas_dim}:lowered={lowered_dim}"
        )
    if meas_dim is not None and meas_rope > 0 and lowered_cache_dim > lowered_dim and (meas_dim + meas_rope) != lowered_cache_dim:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:cache_dim:measured={meas_dim + meas_rope}:lowered={lowered_cache_dim}"
        )
    if meas_heads is not None and meas_heads != lowered_heads:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:heads:measured={meas_heads}:lowered={lowered_heads}"
        )
    if meas_layout and lowered_layout and meas_layout != lowered_layout:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:query_layout:measured={meas_layout}:lowered={lowered_layout}"
        )
    if meas_kv_len is not None and lowered_kv_len > 0 and meas_kv_len != lowered_kv_len:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:kv_len:measured={meas_kv_len}:lowered={lowered_kv_len}"
        )
    for meas_pages in (actual_explicit_pages, actual_bt_pages, binding_pages):
        if meas_pages is not None and lowered_pages > 0 and meas_pages != lowered_pages:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:page_count:measured={meas_pages}:lowered={lowered_pages}"
            )


def _verify_indexer_op_binding(
    target_op: Any,
    binding: Mapping[str, Any],
    actual: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    case_id: str,
) -> None:
    loc = tuple(getattr(target_op, "local_shape", ()) or ())
    if len(loc) >= _INDEXER_SHAPE_MIN_RANK:
        lowered_heads, lowered_dim = int(loc[1]), int(loc[2])
        meas_heads = _opt_pos_int(actual.get("heads"), binding.get("heads"))
        meas_dim = _opt_pos_int(actual.get("head_dim"), binding.get("head_dim"))
        meas_kv = _opt_pos_int(actual.get("kv_heads"), binding.get("kv_heads"))
        if meas_heads is not None and meas_heads != lowered_heads:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:heads:measured={meas_heads}:lowered={lowered_heads}"
            )
        if meas_dim is not None and meas_dim != lowered_dim:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:head_dim:measured={meas_dim}:lowered={lowered_dim}"
            )
        if meas_kv is not None and meas_kv != 1:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:kv_heads:measured={meas_kv}:lowered=1"
            )
    meta = dict(getattr(target_op, "metadata", ()) or ())
    subgraphs = meta.get("structure_subgraphs")
    if not isinstance(subgraphs, Sequence) or not subgraphs or not isinstance(subgraphs[0], Mapping):
        return
    tv = subgraphs[0].get("tensor_view")
    if not isinstance(tv, Mapping):
        return
    lowered_topk = _opt_pos_int(tv.get("topk"))
    meas_topk = _opt_pos_int(
        actual.get("sparse_count"), actual.get("top_k"), actual.get("topk"),
        binding.get("sparse_count"), binding.get("topk"), payload.get("topk"),
    )
    if meas_topk is not None and lowered_topk is not None and meas_topk != lowered_topk:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:topk:measured={meas_topk}:lowered={lowered_topk}"
        )
    for dt_field in ("query_dtype", "cache_dtype"):
        lowered_dt = str(tv.get(dt_field) or "").strip().lower()
        meas_dt = str(actual.get(dt_field) or binding.get(dt_field) or "").strip().lower()
        if meas_dt and lowered_dt and meas_dt != lowered_dt:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:{dt_field}:measured={meas_dt}:lowered={lowered_dt}"
            )
    seq_parts = tv.get("sequence_partition")
    if isinstance(seq_parts, Sequence) and seq_parts and isinstance(seq_parts[0], Mapping):
        lowered_kv = _opt_pos_int(seq_parts[0].get("kv_length"))
        meas_kv_len = _opt_pos_int(actual.get("seq_len"), actual.get("kv_len"), binding.get("kv_len"))
        if meas_kv_len is not None and lowered_kv is not None and meas_kv_len != lowered_kv:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:kv_len:measured={meas_kv_len}:lowered={lowered_kv}"
            )


def _verify_gemm_node_binding(
    op: Any,
    nb: Mapping[str, Any],
    n_actual: Mapping[str, Any],
    *,
    case_id: str,
    op_id: str,
    is_per_node_actual: bool,
    is_last_grouped_gemm: bool,
    is_sole_gemm: bool = False,
    has_actual: bool = False,
) -> None:
    meta = dict(getattr(op, "metadata", ()) or ())
    lowered_mg = max(1, int(meta.get("matrix_groups") or 1))
    use_case_mg = is_per_node_actual or lowered_mg > 1 or is_sole_gemm
    act_mg = _opt_pos_int(n_actual.get("matrix_groups"), n_actual.get("e")) if use_case_mg else None
    nb_mg = _opt_pos_int(nb.get("matrix_groups"), nb.get("e"))
    if (is_per_node_actual or (has_actual and lowered_mg > 1 and is_last_grouped_gemm)) and lowered_mg > 1 and act_mg is None:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:matrix_groups_missing:{op_id}"
        )
    meas_mg = act_mg if act_mg is not None else nb_mg
    if meas_mg is not None and meas_mg != lowered_mg:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:matrix_groups:{op_id}:measured={meas_mg}:lowered={lowered_mg}"
        )
    loc = tuple(getattr(op, "local_shape", ()) or ())
    if len(loc) < _GEMM_SHAPE_MIN_RANK:
        return
    use_actual_dims = is_per_node_actual or (lowered_mg > 1 and is_last_grouped_gemm) or is_sole_gemm
    require_act_dims = is_per_node_actual or (has_actual and lowered_mg > 1 and is_last_grouped_gemm)
    for dim_idx, dim_key in enumerate(("m", "n", "k")):
        act_d = _opt_pos_int(n_actual.get(dim_key)) if use_actual_dims else None
        nb_d = _opt_pos_int(nb.get(dim_key))
        if require_act_dims and act_d is None:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:gemm_{dim_key}_missing:{op_id}"
            )
        if act_d is not None and nb_d is not None and act_d != nb_d:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:gemm_{dim_key}_conflict:{op_id}:actual={act_d}:planned={nb_d}"
            )
        meas_d = act_d if act_d is not None else nb_d
        if meas_d is not None and meas_d != int(loc[dim_idx]):
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:weight_{dim_key}:{op_id}:measured={meas_d}:lowered={loc[dim_idx]}"
            )


def _verify_attention_merge_node_binding(
    op: Any,
    sim_case: Any,
    nb: Mapping[str, Any],
    n_actual: Mapping[str, Any],
    *,
    case_id: str,
    op_id: str,
    require_writer_fields: bool = False,
) -> None:
    lowered_dt = str(getattr(op, "dtype", "") or "").strip().lower()
    act_dt = str(n_actual.get("dtype") or "").strip().lower()
    nb_dt = str(nb.get("dtype") or "").strip().lower()
    if require_writer_fields and not act_dt:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:dtype_missing:{op_id}"
        )
    if act_dt and nb_dt and act_dt != nb_dt:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:dtype_conflict:{op_id}:actual={act_dt}:planned={nb_dt}"
        )
    meas_dt = act_dt or nb_dt
    if meas_dt and lowered_dt and meas_dt != lowered_dt:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:dtype:{op_id}:measured={meas_dt}:lowered={lowered_dt}"
        )
    lowered_partials = max(_MIN_DCP_PARTIALS, int(getattr(sim_case.runtime.parallel, "cp", 1) or 1))
    act_partials = _opt_pos_int(n_actual.get("partial_count"))
    nb_partials = _opt_pos_int(nb.get("partial_count"))
    if require_writer_fields and act_partials is None:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:partial_count_missing:{op_id}"
        )
    if act_partials is not None and nb_partials is not None and act_partials != nb_partials:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:partial_count_conflict:{op_id}:actual={act_partials}:planned={nb_partials}"
        )
    meas_partials = act_partials if act_partials is not None else nb_partials
    if meas_partials is not None and meas_partials != lowered_partials:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:partial_count:{op_id}:measured={meas_partials}:lowered={lowered_partials}"
        )


def _verify_node_bindings(
    chosen_ops: tuple[Any, ...],
    sim_case: Any,
    binding: Mapping[str, Any],
    actual: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    local_kv_heads: int,
    expected_level: str,
    case_id: str,
) -> None:
    ops_by_id = {str(op.op_id): op for op in chosen_ops}
    has_payload_timed = "timed_nodes" in payload and payload.get("timed_nodes") is not None
    if expected_level in {"L2", "L3"} and has_payload_timed:
        lowered_edges = [
            (dep, str(op.op_id))
            for op in chosen_ops
            for dep in (getattr(op, "deps", ()) or ())
            if dep in ops_by_id
        ]
        try:
            validate_timed_nodes_sequence(
                payload.get("timed_nodes"),
                cid=case_id,
                level=expected_level,
                expected_ops=list(ops_by_id),
                expected_edges=lowered_edges,
                error_prefix=f"whitebox_{expected_level.lower()}_timed_nodes",
            )
        except ModelingError as exc:
            detail = str(exc)
            tag = f"{expected_level.lower()}_op_set_mismatch" if "plan_mismatch" in detail else detail
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:{tag}"
            ) from exc

    raw_node_bindings = binding.get("node_bindings")
    if isinstance(raw_node_bindings, Sequence) and not isinstance(raw_node_bindings, (str, bytes)) and raw_node_bindings:
        node_bindings: Sequence[Mapping[str, Any]] = [nb for nb in raw_node_bindings if isinstance(nb, Mapping)]
    elif expected_level in {"L2", "L3"} and actual:
        node_bindings = [{"op_id": oid} for oid in ops_by_id]
    else:
        node_bindings = ()

    raw_ns = actual.get("node_shapes")
    node_shapes: dict[str, Mapping[str, Any]] = {}
    if raw_ns is not None and not isinstance(raw_ns, Mapping):
        raise ModelingError(f"whitebox_measured_binding_mismatch:{case_id}:node_shapes_invalid")
    if isinstance(raw_ns, Mapping):
        for k, v in raw_ns.items():
            if not isinstance(v, Mapping):
                raise ModelingError(
                    f"whitebox_measured_binding_mismatch:{case_id}:node_shapes_entry_invalid:{k}"
                )
            node_shapes[_norm_timed_op_id(k, case_id, error_prefix="whitebox_node_shapes")] = v
    all_gemm_ids = [
        str(nb.get("op_id") or "").strip()
        for nb in node_bindings
        if str(nb.get("op_id") or "").strip() in ops_by_id
        and ops_by_id[str(nb.get("op_id") or "").strip()].family in _GEMM_FAMILIES
    ]
    grouped_gemm_ids = [
        oid for oid in all_gemm_ids
        if int(dict(getattr(ops_by_id[oid], "metadata", ()) or ()).get("matrix_groups") or 1) > 1
    ]
    last_grouped_id = grouped_gemm_ids[-1] if grouped_gemm_ids else ""
    sole_gemm_id = all_gemm_ids[0] if len(all_gemm_ids) == 1 else ""
    multi_gemm = len(all_gemm_ids) > 1
    has_actual = bool(actual)
    for nb in node_bindings:
        op_id = str(nb.get("op_id") or "").strip()
        if not op_id:
            continue
        op = ops_by_id.get(op_id)
        if op is None:
            raise ModelingError(f"whitebox_measured_binding_mismatch:{case_id}:node_op_id_missing:{op_id}")
        per_node = node_shapes.get(op_id)
        is_per_node = per_node is not None
        n_actual = per_node if is_per_node else actual
        if op.family == "attention":
            _verify_attention_op_binding(
                op, sim_case, nb, n_actual, {}, local_kv_heads=local_kv_heads, case_id=case_id
            )
        elif op.family in _GEMM_FAMILIES:
            if has_actual and multi_gemm and not is_per_node:
                raise ModelingError(
                    f"whitebox_measured_binding_mismatch:{case_id}:node_shapes_missing:{op_id}"
                )
            _verify_gemm_node_binding(
                op, nb, n_actual, case_id=case_id, op_id=op_id,
                is_per_node_actual=is_per_node,
                is_last_grouped_gemm=(op_id == last_grouped_id),
                is_sole_gemm=(op_id == sole_gemm_id),
                has_actual=has_actual,
            )
        elif op.family == "attention_merge":
            _verify_attention_merge_node_binding(
                op, sim_case, nb, n_actual, case_id=case_id, op_id=op_id,
                require_writer_fields=(has_actual or is_per_node),
            )
        elif op.family == "rmsnorm":
            loc = getattr(op, "local_shape", None)
            lowered_w = int(loc[-1]) if isinstance(loc, Sequence) and loc else 0
            meas_w = _opt_pos_int(n_actual.get("width") if is_per_node else None, nb.get("width"))
            if meas_w is not None and lowered_w > 0 and meas_w != lowered_w:
                raise ModelingError(
                    f"whitebox_measured_binding_mismatch:{case_id}:width:{op_id}:measured={meas_w}:lowered={lowered_w}"
                )

    has_planned_context = bool(binding.get("timed_op_ids") or actual or binding.get("node_bindings"))
    if expected_level in {"L2", "L3"} and has_planned_context and not has_payload_timed:
        raise ModelingError(
            f"whitebox_measured_binding_mismatch:{case_id}:{expected_level.lower()}_timed_nodes_missing"
        )


def _verify_measured_binding(
    chosen_ops: tuple[Any, ...],
    sim_case: Any,
    payload: Mapping[str, Any],
    *,
    local_kv_heads: int,
    expected_level: str,
    case_id: str,
) -> None:
    """Verify the lowered replay target matches the measured operator's recipe, layout, and geometry."""
    binding = payload.get("measured_binding") if isinstance(payload.get("measured_binding"), Mapping) else {}
    actual = payload.get("actual_shape") if isinstance(payload.get("actual_shape"), Mapping) else {}
    plan_key = payload.get("plan_key") if isinstance(payload.get("plan_key"), Mapping) else {}
    if not binding and not actual and not plan_key and not payload.get("timed_nodes"):
        return

    _verify_node_bindings(
        chosen_ops, sim_case, binding, actual, payload,
        local_kv_heads=local_kv_heads, expected_level=expected_level, case_id=case_id,
    )

    wanted_id = str(binding.get("target_op_id") or "").strip()
    target_op: Any = None
    if expected_level == "L1" and chosen_ops:
        target_op = chosen_ops[0]
        if wanted_id and target_op.op_id != wanted_id:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:target_op_id:measured={wanted_id}:lowered={target_op.op_id}"
            )
    elif wanted_id:
        matches = [op for op in chosen_ops if op.op_id == wanted_id]
        if not matches:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:target_op_id_missing:{wanted_id}"
            )
        target_op = matches[0]
    else:
        wanted_fam = str(binding.get("family") or plan_key.get("family") or payload.get("target_family") or "").strip()
        if wanted_fam:
            matches = [op for op in chosen_ops if op.family == wanted_fam]
            target_op = matches[0] if matches else None
    if target_op is None:
        return

    t_actual: Mapping[str, Any] = actual
    raw_ns = actual.get("node_shapes")
    if isinstance(raw_ns, Mapping):
        for k, v in raw_ns.items():
            if isinstance(v, Mapping) and _norm_timed_op_id(k, case_id, error_prefix="whitebox_node_shapes") == str(target_op.op_id):
                t_actual = v
                break

    if target_op.family == "attention":
        _verify_attention_op_binding(
            target_op, sim_case, binding, t_actual, plan_key,
            local_kv_heads=local_kv_heads, case_id=case_id,
        )
    elif target_op.family == "rmsnorm":
        loc = getattr(target_op, "local_shape", None)
        lowered_w = int(loc[-1]) if isinstance(loc, Sequence) and loc else int(getattr(sim_case.model, "hidden_size", 0) or 0)
        meas_w = _opt_pos_int(t_actual.get("width"), binding.get("width"))
        if meas_w is not None and lowered_w > 0 and meas_w != lowered_w:
            raise ModelingError(
                f"whitebox_measured_binding_mismatch:{case_id}:width:measured={meas_w}:lowered={lowered_w}"
            )
    elif target_op.family == "indexer":
        _verify_indexer_op_binding(target_op, binding, t_actual, payload, case_id=case_id)
    elif target_op.family in _GEMM_FAMILIES:
        _verify_gemm_node_binding(
            target_op, binding, t_actual, case_id=case_id, op_id=str(target_op.op_id),
            is_per_node_actual=True, is_last_grouped_gemm=True, is_sole_gemm=True,
        )
    elif target_op.family == "attention_merge":
        _verify_attention_merge_node_binding(
            target_op, sim_case, binding, t_actual, case_id=case_id, op_id=str(target_op.op_id),
        )


__all__ = [
    "_opt_pos_int",
    "_verify_measured_binding",
]
