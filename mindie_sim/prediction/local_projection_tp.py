"""Semantic dense-subgraph local-TP region rewrite.

[Role]
  Insert node-local AllGather/AllToAll/ReduceScatter boundaries around declared
  dense projection operators and restore each original public output identity.
[Invariants]
  * Column projections gather rows, compute an output-feature shard and exchange
    slices; row projections exchange input shards and reduce-scatter partial rows;
    MLA grouped output pairs shard whole groups.  A replicated packed slice (the
    low-rank ``f_a``/``g_a`` factors) keeps its full width; ``f_b``/``g_b`` carry
    local width one and never enter a collective.
  * A row projection that already owns a pre-GEMM activation quantizer is
    rebuilt as BF16 exchange -> received-row quantize -> GEMM -> reduce-scatter;
    the pre-exchange quantizer is retired instead of quantizing the full axis.
  * Per-rank matrix FLOPs remain ``2*m*n*k``; internal GEMMs carry unique IDs
    while the final restore collective preserves the original public ``op_id``.
  * Transport collectives carry clean transport-only metadata and model uniform
    buffer padding across ragged/idle participants; completely empty groups emit
    zero-work pass-throughs with no phantom GEMM or weight reads.
  * Source-bound implementations invoke their registered retargeting adapter or
    raise ``local_tp_source_placement_unsupported``.
[Timeouts & Locks]
  Pure bounded graph transformation; no Engine, I/O or locks.
[Side-Effect Policy]
  Returns new operator tuples/mappings; never mutates input records.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from mindie_sim.catalog.ops import comm_op
from mindie_sim.catalog.projection_placement import (
    ROLE_ATTENTION_INPUT, ROLE_DENSE_MLP_DOWN, ROLE_DENSE_MLP_UP,
    ROLE_DRAFT_LOGITS, ROLE_LINEAR_INPUT, ROLE_LOGITS,
    ROLE_SHARED_EXPERT_DOWN, ROLE_SHARED_EXPERT_UP,
    _padded_local, role_from_category, role_width,
)
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec

ZERO = 0
ONE = 1
TWO = 2
ZERO_FLOAT = 0.0
FIRST = 0
SECOND = 1
ALIGN_ELEMENT = 16
FALLBACK_ACTIVATION = "bf16"
FALLBACK_OUTPUT = "bf16"
RETARGET_ADAPTERS: dict[str, Any] = {}
ROLE_COLUMN = "column"
ROLE_ROW = "row"
ROLE_GROUP_A = "group_pair_a"
ROLE_GROUP_B = "group_pair_b"
SDK_PLACEMENT = "local_tp"
_TRANSPORT_KEEP_META = frozenset({
    "layer_scope", "producer_scope", "draft_step", "structure_block",
    "structure_role", "graph_replay_scope", "graph_bucket", "fusion_boundary",
})


def register_retarget_adapter(implementation: str, adapter: Any) -> None:
    if not implementation or not callable(adapter):
        raise ModelingError("local_tp_retarget_adapter_invalid")
    RETARGET_ADAPTERS[str(implementation)] = adapter


def _width(runtime: Any, role: str) -> int:
    return role_width(role, runtime.parallel)


def _role(metadata: Mapping[str, Any]) -> str:
    explicit = str(metadata.get("projection_role", ""))
    if explicit:
        return explicit
    ffn_role = str(metadata.get("ffn_role", ""))
    ffn_cat = str(metadata.get("ffn_category", ""))
    if ffn_role == "gate_up" and ffn_cat == "dense_mlp":
        return ROLE_DENSE_MLP_UP
    if ffn_role == "down" and ffn_cat == "dense_mlp":
        return ROLE_DENSE_MLP_DOWN
    if ffn_role == "gate_up" and ffn_cat == "shared_expert":
        return ROLE_SHARED_EXPERT_UP
    if ffn_role == "down" and ffn_cat == "shared_expert":
        return ROLE_SHARED_EXPERT_DOWN
    return role_from_category(
        str(metadata.get("quant_category", metadata.get("catalog_category", ""))))


def _is_globally_replicated(
    op: OpSpec, role: str, metadata: Mapping[str, Any], runtime: Any, model: Any,
) -> bool:
    cat = str(metadata.get("quant_category", metadata.get("catalog_category", ""))).lower()
    if role == ROLE_ATTENTION_INPUT:
        return cat in {"mla.q_a", "mla.kv_a"} or op.op_id.endswith((".q_a", ".kv_a", ".wkv"))
    if role in {ROLE_SHARED_EXPERT_UP, ROLE_SHARED_EXPERT_DOWN}:
        return bool(dict(getattr(runtime, "metadata", ()) or ()).get("enable_shared_expert_dp", False))
    if role in {"logits", "draft_logits"} and model is not None:
        vocab = int(getattr(model, "vocab_size", ZERO) or ZERO)
        return vocab > ZERO and len(op.local_shape) > ONE and int(op.local_shape[ONE]) >= vocab
    return False


def _effective_factor(
    op: OpSpec, role: str, metadata: Mapping[str, Any], runtime: Any, model: Any,
) -> tuple[int, int]:
    """Return ``(factor, stride)`` for one matrix op under global TP + Local-TP."""
    width = _width(runtime, role)
    if width <= ONE:
        return ONE, ONE
    tp = max(ONE, int(getattr(runtime.parallel, "tp", ONE) or ONE))
    if tp <= ONE or _is_globally_replicated(op, role, metadata, runtime, model):
        return width, ONE
    if width == tp:
        return ONE, ONE
    if width < tp or width % tp != ZERO:
        raise ModelingError(f"local_tp_width_not_multiple_of_tp:{width}%{tp}")
    return width // tp, tp


def _quant_for_op(model: Any, metadata: Mapping[str, Any]) -> Any:
    if model is None:
        return None
    if str(metadata.get("layer_scope", "")) == "draft":
        aux = getattr(model, "auxiliary", None)
        draft = getattr(aux, "mtp", None) or getattr(aux, "dspark", None)
        if draft is not None and getattr(draft, "quant", None) is not None:
            return draft.quant
    return getattr(model, "quant", None)


def _local_dimension(size: int, width: int, quant: Any = None) -> int:
    if width <= ONE:
        return int(size)
    # A row-parallel split cuts the input axis, so the local hint is padded to
    # the weight-scale granule (512/TP8 with a 128-column block becomes 128).
    term = SimpleNamespace(quant=quant)
    return max(ONE, int(_padded_local(int(size), int(width), term)))


def _local_output_width(size: int, width: int) -> int:
    """Exact per-rank output features of one packed column split."""
    if width <= ONE:
        return int(size)
    return max(ONE, (int(size) + int(width) - ONE) // int(width))


def _slice_local(item: Mapping[str, Any], width: int, quant: Any = None) -> int:
    """Local output extent of one packed slice; replicated and KV-head slices stay whole."""
    shape = tuple(item.get("shape", ()))
    size = int(shape[-ONE]) if shape else ZERO
    if item.get("replicated"):
        return size
    if str(item.get("shard", "")) == "kv_out":
        units = int(item.get("units", ZERO) or ZERO)
        unit_width = int(item.get("unit_width", ZERO) or ZERO)
        if units > ZERO and unit_width > ZERO:
            local_units = max(ONE, (units + int(width) - ONE) // int(width))
            return local_units * unit_width
    return _local_output_width(size, width)


def _replicated_local_width(metadata: Mapping[str, Any], width: int) -> int:
    """Local output columns every owner already holds without a collective."""
    slices = metadata.get("packed_projection_slices") or ()
    return sum(int(tuple(item.get("shape", (0, 0)))[-ONE])
               for item in slices if item.get("replicated"))


def _packed_local_width(
    metadata: Mapping[str, Any], width: int, total: int, quant: Any = None,
    *, role: str = "",
) -> int:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        if role in {
            ROLE_LOGITS, ROLE_DRAFT_LOGITS, ROLE_ATTENTION_INPUT,
            ROLE_LINEAR_INPUT, ROLE_DENSE_MLP_UP, ROLE_SHARED_EXPERT_UP,
        }:
            return _local_output_width(total, width)
        return _local_dimension(total, width, quant)
    return max(ONE, sum(_slice_local(s, width, quant) for s in slices))


def _localize_packed_slices(
    metadata: dict[str, Any], width: int, quant: Any = None, *, rows: int | None = None,
) -> None:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        return
    # Every slice is a column view into the one localized packed output, so a
    # row advances by the parent buffer's full local width, not by the slice's
    # own width.  Using the slice width makes ``Q[1, 0]`` alias ``K[0, 0]``.
    local_sizes = [_slice_local(item, width, quant) for item in slices]
    packed_width = sum(local_sizes)
    offset, local_slices = ZERO, []
    for item, local_size in zip(slices, local_sizes):
        shape = tuple(item.get("shape", ()))
        out_rows = int(rows) if rows is not None else (int(shape[ZERO]) if shape else ZERO)
        local_slices.append({**dict(item), "shape": (out_rows, local_size),
                             "strides": (packed_width, ONE), "offset_elements": offset})
        offset += local_size
    metadata["packed_projection_slices"] = tuple(local_slices)


def _dtype(metadata: Mapping[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key)
    return str(value) if value else fallback


def _groups_for(
    machine: Any, runtime: Any, replica: int, stage: int, width: int,
    *, stride: int = ONE,
) -> tuple[int, ...]:
    from mindie_sim.prediction.local_tp_groups import local_tp_groups
    from mindie_sim.prediction.topology import Topology
    ranks = Topology(machine, runtime.parallel).ranks(replica, stage)
    if not ranks:
        raise ModelingError(f"local_tp_stage_has_no_ranks:{stage}")
    anchor = int(ranks[ZERO])
    for group in local_tp_groups(
            machine, runtime.parallel, stage=stage, width=width, stride=stride):
        if anchor in group:
            return group
    raise ModelingError(f"local_tp_group_missing:{anchor}")


def _row_counts(cohort_views: Mapping[int, Sequence[Any]], group: Sequence[int],
                fallback: int, peer_rows: Mapping[int, int] | None = None) -> list[int]:
    counts: list[int] = []
    for rk in (int(rank) for rank in group):
        if peer_rows is not None and rk in peer_rows:
            counts.append(max(ZERO, int(peer_rows[rk])))
        elif cohort_views and rk in cohort_views and cohort_views[rk] is not None:
            counts.append(sum(int(view.logical_q_len) for view in cohort_views[rk]))
        else:
            counts.append(int(fallback))
    return counts


def _transport_metadata(op: OpSpec, *, role: str, group: Sequence[int], width: int,
                        family: str, scope: str,
                        stride: int = ONE) -> tuple[tuple[str, Any], ...]:
    src = dict(op.metadata)
    kept = {k: src[k] for k in _TRANSPORT_KEEP_META if k in src}
    kept.update({
        "collective": family, "group_axis": SDK_PLACEMENT,
        "group_ranks": tuple(int(rank) for rank in group),
        "local_tp_width": int(width), "projection_role": role,
        "placement_scope": str(src.get("placement_scope") or src.get("layer_scope") or scope or "target"),
        "local_tp_applied": True, "payload_dtype": str(src.get("payload_dtype", "")),
    })
    if stride > ONE:
        kept["local_tp_stride"] = int(stride)
    return tuple(sorted(kept.items(), key=lambda pair: pair[FIRST]))


def _collective(op: OpSpec, *, suffix: str, family: str, recipe: str, payload: float,
                shape: tuple[int, ...], role: str, group: Sequence[int], width: int,
                deps: tuple[str, ...], scope: str, dtype: str,
                public_id: str | None = None, stride: int = ONE) -> OpSpec:
    built = comm_op(
        op_id=f"{op.op_id}:local_tp:{suffix}", deps=deps, stage=op.stage,
        layer=op.layer, requests=op.requests, family=family, phase=op.phase,
        recipe=recipe, link_bytes=payload, shape=shape,
        metadata=_transport_metadata(op, role=role, group=group, width=width,
                                     family=family, scope=scope, stride=stride),
        notes=(f"local_tp_{role}", f"local_tp_{suffix}"),
    )
    return replace(built, op_id=public_id or built.op_id, dtype=dtype)


def _rewrite_matrix(op: OpSpec, *, gemm_id: str, rows: int, cols: int, reduced: int,
                    role: str, width: int, dep_ids: tuple[str, ...],
                    weight_ratio: float, quant: Any, groups: int = ONE,
                    extra_metadata: Mapping[str, Any] | None = None,
                    scale_ratio: float | None = None) -> OpSpec:
    metadata = {**dict(op.metadata), **dict(extra_metadata or ())}
    original_weight = float(metadata.get("packed_weight_bytes", ZERO_FLOAT) or ZERO_FLOAT)
    cat = str(metadata.get("quant_category", metadata.get("catalog_category", "")) or "")
    if quant is not None and cat and rows > ZERO:
        local_weight = float(quant.matrix_bytes(
            cols if groups <= ONE else (groups * cols), reduced, cat))
    else:
        local_weight = original_weight * float(weight_ratio) if rows > ZERO else ZERO_FLOAT
    act = _dtype(metadata, "activation_format", op.dtype or FALLBACK_ACTIVATION)
    out = _dtype(metadata, "output_format", FALLBACK_OUTPUT)
    m, n, k = int(rows), int(cols), int(reduced)
    in_b, out_b = float(m * k) * dtype_bytes(act), float(m * n) * dtype_bytes(out)
    orig_m = max(ONE, int(op.local_shape[ZERO]))
    ratio = float(m) / float(orig_m) if scale_ratio is None else float(scale_ratio)
    scale_shape, geometry_bytes = _local_scale_geometry(metadata, m, k)
    scale_b = (geometry_bytes if scale_shape else
               float(metadata.get("activation_scale_bytes", ZERO_FLOAT) or ZERO_FLOAT)
               * ratio)
    if scale_shape:
        metadata["activation_scale_shape"] = scale_shape
    metadata.update({
        "packed_weight_bytes": local_weight, "weight_bytes": local_weight,
        "activation_scale_bytes": scale_b, "placement_role": role,
        "local_tp_width": int(width), "local_tp_applied": True,
        "matrix_groups": max(ONE, int(groups)),
    })
    pad_m = align_up(m, ALIGN_ELEMENT) if m > ZERO else ZERO
    return replace(
        op, op_id=gemm_id, local_shape=(m, n, k),
        padded_shape=(pad_m, align_up(n, ALIGN_ELEMENT), align_up(k, ALIGN_ELEMENT)),
        launch_shape=(m, n, k), flops=float(TWO * m * n * k),
        hbm_bytes=in_b + local_weight + out_b + scale_b,
        mte2_bytes=in_b + local_weight + scale_b, mte3_bytes=out_b,
        deps=dep_ids, metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[FIRST])),
    )


def _partition_kind(role: str, metadata: Mapping[str, Any]) -> str:
    cat = str(metadata.get("quant_category", metadata.get("catalog_category", ""))).lower()
    if cat == "mla.o_a":
        return ROLE_GROUP_A
    if cat == "mla.o_b":
        return ROLE_GROUP_B
    if role in {"attention_output", "linear_output", ROLE_DENSE_MLP_DOWN, ROLE_SHARED_EXPERT_DOWN}:
        return ROLE_ROW
    return ROLE_COLUMN


def _empty_passthrough(op: OpSpec, metadata: Mapping[str, Any], width: int) -> OpSpec:
    meta = {**dict(metadata), "local_tp_applied": True, "local_tp_width": int(width),
            "packed_weight_bytes": ZERO_FLOAT, "weight_bytes": ZERO_FLOAT}
    return replace(op, flops=ZERO_FLOAT, hbm_bytes=ZERO_FLOAT, mte2_bytes=ZERO_FLOAT,
                   mte3_bytes=ZERO_FLOAT,
                   metadata=tuple(sorted(meta.items(), key=lambda pair: pair[FIRST])))


def _local_scale_geometry(
    metadata: Mapping[str, Any], rows: int, local_k: int,
) -> tuple[tuple[int, ...], float]:
    """Received-row scale shape and bytes after the input axis is sharded."""
    shape = tuple(metadata.get("activation_scale_shape") or ())
    total = float(metadata.get("activation_scale_bytes", ZERO_FLOAT) or ZERO_FLOAT)
    group = int(metadata.get("activation_group_size") or ZERO)
    if not shape or total <= ZERO_FLOAT or group <= ZERO:
        return (), ZERO_FLOAT
    per_row = total / float(max(ONE, int(shape[ZERO])))
    old_tail = tuple(int(extent) for extent in shape[ONE:])
    if not old_tail:
        # One per-row scale: the received row count is the only change.
        return (int(rows),), per_row * float(rows)
    old_elements = ONE
    for extent in old_tail:
        old_elements *= extent
    if len(old_tail) == TWO:
        pack = max(ONE, old_tail[ONE])
        groups = (int(local_k) + group - ONE) // group
        new_tail = ((groups + pack - ONE) // pack, pack)
    else:
        new_tail = ((int(local_k) + group - ONE) // group,)
    elements = ONE
    for extent in new_tail:
        elements *= extent
    per_scale = per_row / float(max(ONE, old_elements))
    return (int(rows), *new_tail), per_scale * float(elements) * float(rows)


def _received_quantizer(op: OpSpec, *, rows: int, local_k: int, full_k: int,
                        exchange_id: str, width: int) -> OpSpec:
    """Move one pre-exchange activation quantizer onto the received local rows."""
    meta = dict(op.metadata)
    in_dtype = _dtype(meta, "input_format", FALLBACK_ACTIVATION)
    out_dtype = _dtype(meta, "output_format", op.dtype or in_dtype)
    old_rows = max(ONE, int(meta.get("local_tp_template_rows") or op.local_shape[ZERO]))
    old_k = max(ONE, int(op.local_shape[ONE]) if len(op.local_shape) > ONE else full_k)
    ratio = float(rows) * float(local_k) / (float(old_rows) * float(old_k))
    scale_shape, scale_bytes = _local_scale_geometry(meta, rows, local_k)
    read = float(rows * local_k) * dtype_bytes(in_dtype)
    write = float(rows * local_k) * dtype_bytes(out_dtype) + scale_bytes
    meta.update({
        "local_tp_applied": True, "local_tp_width": int(width),
        "local_tp_quantization_scope": "received_expert_input_rows",
        "local_tp_pre_exchange_dtype": in_dtype,
        "source_rows": int(rows),
    })
    if scale_shape:
        meta["activation_scale_shape"] = scale_shape
        meta["activation_scale_bytes"] = scale_bytes
    pad_rows = align_up(rows, ALIGN_ELEMENT)
    return replace(
        op, deps=(exchange_id,),
        local_shape=(int(rows), int(local_k)), launch_shape=(int(rows), int(local_k)),
        padded_shape=(pad_rows, align_up(int(local_k), ALIGN_ELEMENT)),
        semantic_shape=(int(rows), int(local_k)),
        vector_ops=float(op.vector_ops) * ratio,
        mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write, flops=ZERO_FLOAT,
        metadata=tuple(sorted(meta.items(), key=lambda pair: pair[FIRST])),
    )


def _row_pre_quantizer(op: OpSpec, by_id: Mapping[str, OpSpec],
                       users: Mapping[str, list[str]]) -> OpSpec | None:
    """The single pre-GEMM activation quantizer this row matrix consumes."""
    if len(op.deps) != ONE:
        return None
    producer = by_id.get(op.deps[ZERO])
    if producer is None or producer.plan_required:
        return None
    if producer.family not in {"quantize", "rmsnorm_quant"} or len(producer.deps) != ONE:
        return None
    if users.get(producer.op_id) != [op.op_id]:
        return None
    if str(dict(producer.metadata).get("quantization_scope", "")) == "moe_dispatch_expert_input":
        return None
    return producer


def _match_paired_ffn(
    op: OpSpec, role: str, by_id: Mapping[str, OpSpec],
    users: Mapping[str, list[str]],
) -> tuple[OpSpec, OpSpec | None, OpSpec] | None:
    """Match ``gate_up -> activation [-> quant] -> down`` for single-block Megatron Local-TP."""
    if role not in {ROLE_DENSE_MLP_UP, ROLE_SHARED_EXPERT_UP}:
        return None
    meta = dict(op.metadata)
    if meta.get("ffn_role") != "gate_up":
        return None
    cat = str(meta.get("ffn_category", ""))
    act_users = users.get(op.op_id) or []
    if len(act_users) != ONE:
        return None
    act_op = by_id.get(act_users[ZERO])
    if act_op is None or dict(act_op.metadata).get("ffn_role") != "activation":
        return None
    next_users = users.get(act_op.op_id) or []
    if len(next_users) != ONE:
        return None
    nxt = by_id.get(next_users[ZERO])
    if nxt is None:
        return None
    pre_quant: OpSpec | None = None
    down = nxt
    if dict(nxt.metadata).get("ffn_role") != "down":
        if nxt.family not in {"quantize", "rmsnorm_quant"}:
            return None
        down_users = users.get(nxt.op_id) or []
        if len(down_users) != ONE:
            return None
        pre_quant = nxt
        down = by_id.get(down_users[ZERO])
        if down is None or dict(down.metadata).get("ffn_role") != "down":
            return None
    if str(dict(down.metadata).get("ffn_category", "")) != cat:
        return None
    return act_op, pre_quant, down


def _scale_activation_op(
    act_op: OpSpec, *, rows: int, local_w: int, old_rows: int, old_w: int,
    dep_id: str, width: int,
) -> OpSpec:
    ratio = (float(rows) * float(local_w)) / float(max(ONE, old_rows * old_w))
    meta = {**dict(act_op.metadata), "local_tp_applied": True, "local_tp_width": int(width)}
    return replace(
        act_op, deps=(dep_id,),
        local_shape=(int(rows), int(local_w)),
        padded_shape=(align_up(int(rows), ALIGN_ELEMENT), align_up(int(local_w), ALIGN_ELEMENT)),
        launch_shape=(int(rows), int(local_w)),
        semantic_shape=(int(rows), int(local_w)),
        vector_ops=float(act_op.vector_ops) * ratio,
        hbm_bytes=float(act_op.hbm_bytes) * ratio,
        mte2_bytes=float(act_op.mte2_bytes) * ratio,
        mte3_bytes=float(act_op.mte3_bytes) * ratio,
        metadata=tuple(sorted(meta.items(), key=lambda pair: pair[FIRST])),
    )


def _rewrite_paired_ffn(
    gate_up: OpSpec, act_op: OpSpec, pre_quant: OpSpec | None, down: OpSpec,
    *, role: str, down_role: str, factor: int, stride: int,
    group: Sequence[int], total_tok: int, pad_tok: int,
    quant: Any, op_scope: str,
) -> tuple[OpSpec, ...]:
    """Emit one Megatron-style FFN block: AllGather -> gate_up -> act -> [quant] -> down -> RS."""
    g_meta = dict(gate_up.metadata)
    d_meta = dict(down.metadata)
    m, _, hidden = (int(gate_up.local_shape[ZERO]), int(gate_up.local_shape[ONE]),
                    int(gate_up.local_shape[TWO]))
    inter_w = int(down.local_shape[TWO])
    local_w = _local_output_width(inter_w, factor)
    act_dt = _dtype(g_meta, "activation_format", gate_up.dtype or FALLBACK_ACTIVATION)
    out_dt = _dtype(d_meta, "output_format", FALLBACK_OUTPUT)
    gathered_tok = factor * pad_tok
    gather = _collective(
        gate_up, suffix="gather", family="allgather", recipe="ring_allgather",
        payload=float((factor - ONE) * pad_tok * hidden) * dtype_bytes(act_dt),
        shape=(gathered_tok, hidden), role=role, group=group, width=factor,
        deps=gate_up.deps, scope=op_scope, dtype=act_dt, stride=stride,
    )
    g_meta["local_tp_partition"] = ROLE_COLUMN
    gate_gemm = _rewrite_matrix(
        gate_up, gemm_id=f"{gate_up.op_id}:local_tp:gemm",
        rows=gathered_tok, cols=TWO * local_w, reduced=hidden,
        role=role, width=factor, dep_ids=(gather.op_id,),
        weight_ratio=float(local_w) / float(max(ONE, inter_w)),
        quant=quant, extra_metadata=g_meta,
    )
    act_old_rows = max(ONE, int(dict(act_op.metadata).get("local_tp_template_rows") or m))
    scaled_act = _scale_activation_op(
        act_op, rows=gathered_tok, local_w=local_w, old_rows=act_old_rows, old_w=inter_w,
        dep_id=gate_gemm.op_id, width=factor,
    )
    emitted: list[OpSpec] = [gather, gate_gemm, scaled_act]
    down_dep = scaled_act.op_id
    scale_ratio: float | None = None
    if pre_quant is not None:
        requant = _received_quantizer(
            pre_quant, rows=gathered_tok, local_k=local_w, full_k=inter_w,
            exchange_id=scaled_act.op_id, width=factor,
        )
        emitted.append(requant)
        down_dep = requant.op_id
        pq_old_rows = max(ONE, int(dict(pre_quant.metadata).get("local_tp_template_rows") or m))
        scale_ratio = (float(gathered_tok) / float(pq_old_rows)
                       * float(local_w) / float(max(ONE, inter_w)))
    d_meta["local_tp_partition"] = ROLE_ROW
    down_gemm = _rewrite_matrix(
        down, gemm_id=f"{down.op_id}:local_tp:gemm",
        rows=gathered_tok, cols=hidden, reduced=local_w,
        role=down_role, width=factor, dep_ids=(down_dep,),
        weight_ratio=float(local_w) / float(max(ONE, inter_w)),
        quant=quant, extra_metadata=d_meta, scale_ratio=scale_ratio,
    )
    restore = _collective(
        down, suffix="reduce_scatter", family="reduce_scatter",
        recipe="ring_reduce_scatter",
        payload=float((factor - ONE) * pad_tok * hidden) * dtype_bytes(out_dt),
        shape=(pad_tok, hidden), role=down_role, group=group, width=factor,
        deps=(down_gemm.op_id,), scope=op_scope, dtype=out_dt,
        public_id=down.op_id, stride=stride,
    )
    emitted.extend((down_gemm, restore))
    return tuple(emitted)


def _rewrite_single_projection(
    op: OpSpec, *, role: str, partition: str, factor: int, stride: int,
    group: Sequence[int], total_tok: int, pad_tok: int,
    quant: Any, op_scope: str, by_id: Mapping[str, OpSpec],
    users: Mapping[str, list[str]], metadata: dict[str, Any],
) -> tuple[OpSpec, ...]:
    m, n, k = (int(op.local_shape[ZERO]), int(op.local_shape[ONE]), int(op.local_shape[TWO]))
    act = _dtype(metadata, "activation_format", op.dtype or FALLBACK_ACTIVATION)
    out = _dtype(metadata, "output_format", FALLBACK_OUTPUT)
    metadata["local_tp_partition"] = partition
    gemm_id = f"{op.op_id}:local_tp:gemm"
    groups = max(ONE, int(metadata.get("matrix_groups", ONE))) if partition == ROLE_GROUP_A else ONE
    if partition == ROLE_GROUP_A:
        lg = groups // factor
        ex = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                         payload=float((factor - ONE) * pad_tok * lg * k) * dtype_bytes(act),
                         shape=(total_tok * lg, k), role=role, group=group,
                         width=factor, deps=op.deps, scope=op_scope, dtype=act, stride=stride)
        gemm = _rewrite_matrix(op, gemm_id=op.op_id, rows=total_tok * lg, cols=n, reduced=k,
                               role=role, width=factor, dep_ids=(ex.op_id,),
                               weight_ratio=ONE / float(factor), quant=quant, groups=lg, extra_metadata=metadata)
        return (ex, gemm)
    if partition == ROLE_GROUP_B:
        gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=n, reduced=k // factor,
                               role=role, width=factor, dep_ids=op.deps, weight_ratio=ONE / float(factor),
                               quant=quant, groups=ONE, extra_metadata=metadata)
        restore = _collective(op, suffix="reduce_scatter", family="reduce_scatter", recipe="ring_reduce_scatter",
                              payload=float((factor - ONE) * pad_tok * n) * dtype_bytes(out), shape=(m, n),
                              role=role, group=group, width=factor, deps=(gemm.op_id,), scope=op_scope,
                              dtype=out, public_id=op.op_id, stride=stride)
        return (gemm, restore)
    if partition == ROLE_COLUMN:
        local_n = _packed_local_width(metadata, factor, n, quant, role=role)
        replicated_n = _replicated_local_width(metadata, factor)
        _localize_packed_slices(metadata, factor, quant, rows=total_tok)
        gather = _collective(op, suffix="gather", family="allgather", recipe="ring_allgather",
                             payload=float((factor - ONE) * pad_tok * k) * dtype_bytes(act), shape=(total_tok, k),
                             role=role, group=group, width=factor, deps=op.deps, scope=op_scope, dtype=act,
                             stride=stride)
        gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=local_n, reduced=k,
                               role=role, width=factor, dep_ids=(gather.op_id,),
                               weight_ratio=float(local_n) / float(max(ONE, n)), quant=quant, extra_metadata=metadata)
        exchanged_n = max(ZERO, local_n - replicated_n)
        restore = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                              payload=float((factor - ONE) * pad_tok * exchanged_n)
                                      * dtype_bytes(out), shape=(m, n),
                              role=role, group=group, width=factor, deps=(gemm.op_id,), scope=op_scope,
                              dtype=out, public_id=op.op_id, stride=stride)
        return (gather, gemm, restore)
    local_k = _local_dimension(k, factor, quant)
    source = op.deps
    pre = _row_pre_quantizer(op, by_id, users)
    exchange_dtype = act
    if pre is not None:
        exchange_dtype = _dtype(dict(pre.metadata), "input_format", act)
        source = pre.deps
    ex = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                     payload=float((factor - ONE) * pad_tok * local_k)
                             * dtype_bytes(exchange_dtype),
                     shape=(total_tok, local_k), role=role, group=group, width=factor,
                     deps=source, scope=op_scope, dtype=exchange_dtype, stride=stride)
    out_ops: list[OpSpec] = [ex]
    gemm_dep, scale_ratio = ex.op_id, None
    if pre is not None:
        requantized = _received_quantizer(
            pre, rows=total_tok, local_k=local_k, full_k=k,
            exchange_id=ex.op_id, width=factor)
        out_ops.append(requantized)
        gemm_dep = requantized.op_id
        scale_ratio = (float(total_tok) / float(max(ONE, m))
                       * float(local_k) / float(max(ONE, k)))
    gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=n, reduced=local_k,
                           role=role, width=factor, dep_ids=(gemm_dep,),
                           weight_ratio=float(local_k) / float(max(ONE, k)),
                           quant=quant, extra_metadata=metadata,
                           scale_ratio=scale_ratio)
    restore = _collective(op, suffix="reduce_scatter", family="reduce_scatter", recipe="ring_reduce_scatter",
                          payload=float((factor - ONE) * pad_tok * n) * dtype_bytes(out), shape=(m, n),
                          role=role, group=group, width=factor, deps=(gemm.op_id,), scope=op_scope,
                          dtype=out, public_id=op.op_id, stride=stride)
    out_ops.extend((gemm, restore))
    return tuple(out_ops)


def lower_local_projection_tp(
    ops: Sequence[OpSpec], *, model: Any, runtime: Any, machine: Any,
    replica: int, cohort_views: Mapping[int, Sequence[Any]], scope: str,
    peer_rows: Mapping[tuple[int, str], Mapping[int, int]] | None = None,
) -> tuple[OpSpec, ...]:
    """Rewrite declared dense projection regions for node-local TP."""
    if not runtime.parallel.has_local_tp:
        return tuple(ops)
    from mindie_sim.prediction.local_tp_groups import validate_local_projection_tp

    validate_local_projection_tp(SimpleNamespace(runtime=runtime, machine=machine))
    by_id = {op.op_id: op for op in ops}
    users: dict[str, list[str]] = {}
    for consumer in ops:
        for dependency in consumer.deps:
            users.setdefault(dependency, []).append(consumer.op_id)
    paired_blocks: dict[str, tuple[OpSpec, OpSpec | None, OpSpec]] = {}
    paired_members: set[str] = set()
    retire: set[str] = set()
    for candidate in ops:
        candidate_meta = dict(candidate.metadata)
        candidate_role = _role(candidate_meta)
        if not candidate_role or candidate_meta.get("local_tp_applied"):
            continue
        factor, _ = _effective_factor(candidate, candidate_role, candidate_meta, runtime, model)
        if factor <= ONE:
            continue
        matched = _match_paired_ffn(candidate, candidate_role, by_id, users)
        if matched is not None:
            paired_blocks[candidate.op_id] = matched
            act_op, pre_q, down_op = matched
            paired_members.update((act_op.op_id, down_op.op_id))
            if pre_q is not None:
                paired_members.add(pre_q.op_id)
            continue
        if (candidate.op_id not in paired_members
                and _partition_kind(candidate_role, candidate_meta) == ROLE_ROW):
            pre = _row_pre_quantizer(candidate, by_id, users)
            if pre is not None:
                retire.add(pre.op_id)
    rewritten: list[OpSpec] = []
    for op in ops:
        if op.op_id in retire or op.op_id in paired_members:
            continue
        metadata = dict(op.metadata)
        role = _role(metadata)
        if not role or metadata.get("local_tp_applied"):
            rewritten.append(op)
            continue
        factor, stride = _effective_factor(op, role, metadata, runtime, model)
        if factor <= ONE:
            rewritten.append(op)
            continue
        partition = _partition_kind(role, metadata)
        group = _groups_for(machine, runtime, replica, op.stage, factor, stride=stride)
        if op.source_boundary or metadata.get("implementation"):
            impl = str(metadata.get("implementation", "") or metadata.get("kernel", "source"))
            adapter = RETARGET_ADAPTERS.get(impl)
            if adapter is None:
                raise ModelingError(f"local_tp_source_placement_unsupported:{impl}")
            op = adapter(op, width=factor, role=role, group=group, partition=partition)
            metadata = dict(op.metadata)
        quant = _quant_for_op(model, metadata)
        m, k = int(op.local_shape[ZERO]), int(op.local_shape[TWO])
        op_scope = str(metadata.get("layer_scope") or scope or "target")
        op_peers = (peer_rows or {}).get((int(op.stage), op.op_id))
        groups = max(ONE, int(metadata.get("matrix_groups", ONE))) if partition == ROLE_GROUP_A else ONE
        if (partition == ROLE_GROUP_A and groups % factor != ZERO) or (
                partition == ROLE_GROUP_B and k % factor != ZERO):
            raise ModelingError("local_tp_group_geometry_indivisible")
        counts = _row_counts(cohort_views, group, m // groups, op_peers)
        total_tok, pad_tok = sum(counts), max(counts, default=ZERO)
        if total_tok <= ZERO:
            rewritten.append(_empty_passthrough(op, metadata, factor))
            if op.op_id in paired_blocks:
                act_op, pre_q, down_op = paired_blocks[op.op_id]
                rewritten.append(_empty_passthrough(down_op, dict(down_op.metadata), factor))
            continue
        if op.op_id in paired_blocks:
            act_op, pre_q, down_op = paired_blocks[op.op_id]
            down_role = (ROLE_DENSE_MLP_DOWN if role == ROLE_DENSE_MLP_UP
                         else ROLE_SHARED_EXPERT_DOWN)
            rewritten.extend(_rewrite_paired_ffn(
                op, act_op, pre_q, down_op, role=role, down_role=down_role,
                factor=factor, stride=stride, group=group,
                total_tok=total_tok, pad_tok=pad_tok, quant=quant,
                op_scope=op_scope,
            ))
            continue
        rewritten.extend(_rewrite_single_projection(
            op, role=role, partition=partition, factor=factor, stride=stride,
            group=group, total_tok=total_tok, pad_tok=pad_tok, quant=quant,
            op_scope=op_scope, by_id=by_id, users=users, metadata=metadata,
        ))
    return tuple(rewritten)


def _synth_idle_block(
    tmpl: OpSpec, paired: tuple[OpSpec, OpSpec | None, OpSpec] | None,
    dep_ids: tuple[str, ...],
) -> list[OpSpec]:
    """Build zero-row template ops for an idle peer, keeping paired FFN structure intact."""
    z_local = (ZERO, int(tmpl.local_shape[ONE]), int(tmpl.local_shape[TWO]))
    z_pad = (ZERO, int(tmpl.padded_shape[ONE]), int(tmpl.padded_shape[TWO]))
    z_launch = (ZERO, int(tmpl.launch_shape[ONE]), int(tmpl.launch_shape[TWO]))
    gate = replace(tmpl, requests=(), deps=dep_ids, local_shape=z_local, padded_shape=z_pad,
                   launch_shape=z_launch, flops=ZERO_FLOAT, hbm_bytes=ZERO_FLOAT,
                   mte2_bytes=ZERO_FLOAT, mte3_bytes=ZERO_FLOAT)
    if paired is None:
        return [gate]
    act_tmpl, pq_tmpl, down_tmpl = paired
    act_rows = max(ONE, int(act_tmpl.local_shape[ZERO]))
    act_meta = tuple(sorted({
        **dict(act_tmpl.metadata), "local_tp_template_rows": act_rows,
    }.items(), key=lambda pair: pair[FIRST]))
    act_shape = (ZERO, *tuple(int(d) for d in act_tmpl.local_shape[ONE:]))
    synth_act = replace(act_tmpl, requests=(), deps=(gate.op_id,),
                        local_shape=act_shape, padded_shape=act_shape,
                        launch_shape=act_shape, semantic_shape=act_shape,
                        metadata=act_meta)
    chain: list[OpSpec] = [gate, synth_act]
    down_dep = synth_act.op_id
    if pq_tmpl is not None:
        pq_rows = max(ONE, int(pq_tmpl.local_shape[ZERO]))
        pq_meta = tuple(sorted({
            **dict(pq_tmpl.metadata), "local_tp_template_rows": pq_rows,
        }.items(), key=lambda pair: pair[FIRST]))
        pq_shape = (ZERO, *tuple(int(d) for d in pq_tmpl.local_shape[ONE:]))
        synth_pq = replace(pq_tmpl, requests=(), deps=(synth_act.op_id,),
                           local_shape=pq_shape, padded_shape=pq_shape,
                           launch_shape=pq_shape, semantic_shape=pq_shape,
                           metadata=pq_meta)
        chain.append(synth_pq)
        down_dep = synth_pq.op_id
    dz_local = (ZERO, int(down_tmpl.local_shape[ONE]), int(down_tmpl.local_shape[TWO]))
    dz_pad = (ZERO, int(down_tmpl.padded_shape[ONE]), int(down_tmpl.padded_shape[TWO]))
    dz_launch = (ZERO, int(down_tmpl.launch_shape[ONE]), int(down_tmpl.launch_shape[TWO]))
    synth_down = replace(down_tmpl, requests=(), deps=(down_dep,),
                         local_shape=dz_local, padded_shape=dz_pad, launch_shape=dz_launch,
                         flops=ZERO_FLOAT, hbm_bytes=ZERO_FLOAT,
                         mte2_bytes=ZERO_FLOAT, mte3_bytes=ZERO_FLOAT)
    chain.append(synth_down)
    return chain


def lower_local_projection_tp_cohort(
    graphs: Mapping[int, Sequence[OpSpec]], *, model: Any, runtime: Any, machine: Any,
) -> dict[int, tuple[OpSpec, ...]]:
    """Cohort-level local-TP rewrite using actual per-rank operator rows and idle synthesis."""
    if not runtime.parallel.has_local_tp or not graphs:
        return {int(r): tuple(ops) for r, ops in graphs.items()}
    from mindie_sim.prediction.local_tp_groups import validate_local_projection_tp
    from mindie_sim.prediction.topology import Topology

    validate_local_projection_tp(SimpleNamespace(runtime=runtime, machine=machine))
    topo = Topology(machine, runtime.parallel)
    identifiers = [tuple(op.op_id for op in ops) for ops in graphs.values()]
    homogeneous = all(ids == identifiers[ZERO] for ids in identifiers)
    ordered_occ: list[tuple[int, str]] = []
    template_by_occ: dict[tuple[int, str], tuple[OpSpec, frozenset[str], int, int, Any]] = {}
    paired_down_keys: set[tuple[int, str]] = set()
    peer_rows: dict[tuple[int, str], dict[int, int]] = {}
    op_maps = {int(r): {op.op_id: op for op in ops} for r, ops in graphs.items()}
    for r, ops in sorted(graphs.items()):
        seq = tuple(ops)
        r_by_id = op_maps[int(r)]
        r_users: dict[str, list[str]] = {}
        for consumer in seq:
            for dep in consumer.deps:
                r_users.setdefault(dep, []).append(consumer.op_id)
        for idx, op in enumerate(seq):
            meta = dict(op.metadata)
            role = _role(meta)
            if not role or meta.get("local_tp_applied"):
                continue
            factor, stride = _effective_factor(op, role, meta, runtime, model)
            if factor <= ONE:
                continue
            key = (int(op.stage), op.op_id)
            if key not in template_by_occ:
                paired = _match_paired_ffn(op, role, r_by_id, r_users)
                if paired is not None:
                    paired_down_keys.add((int(op.stage), paired[TWO].op_id))
                ordered_occ.append(key)
                template_by_occ[key] = (
                    op, frozenset() if homogeneous else
                    frozenset(x.op_id for x in seq[idx + ONE:]),
                    factor, stride, paired)
            grp_div = max(ONE, int(meta.get("matrix_groups", ONE))) if _partition_kind(role, meta) == ROLE_GROUP_A else ONE
            tok_m = int(op.local_shape[ZERO]) // grp_div
            for rk in topo.ranks(int(r), int(op.stage)):
                peer_rows.setdefault(key, {})[int(rk)] = tok_m
    ext_roots = tuple(dict.fromkeys(d for r0, ops in sorted(graphs.items()) for item in ops
                                    for d in item.deps if d not in op_maps[int(r0)]))
    augmented: dict[int, list[OpSpec]] = {int(r): list(ops) for r, ops in graphs.items()}
    prev_synth: dict[tuple[int, int], str] = {}
    for stage, op_id in ordered_occ:
        if (stage, op_id) in paired_down_keys:
            continue
        tmpl, after_ids, factor, stride, paired = template_by_occ[(stage, op_id)]
        for r in sorted(augmented):
            if op_id in op_maps[r]:
                prev_synth[(r, stage)] = paired[TWO].op_id if paired is not None else op_id
                continue
            group = _groups_for(machine, runtime, r, stage, factor, stride=stride)
            for rk in topo.ranks(r, stage):
                peer_rows.setdefault((stage, op_id), {}).setdefault(int(rk), ZERO)
                if paired is not None:
                    peer_rows.setdefault((stage, paired[TWO].op_id), {}).setdefault(int(rk), ZERO)
            if sum(peer_rows[(stage, op_id)].get(int(rk), ZERO) for rk in group) <= ZERO:
                continue
            existing_ids = {item.op_id for item in augmented[r]}
            dep_ids = tuple(d for d in tmpl.deps if d in existing_ids) or (
                (prev_synth[(r, stage)],) if (r, stage) in prev_synth
                else ("iteration_input",) if "iteration_input" in existing_ids else ext_roots[:ONE])
            chain = _synth_idle_block(tmpl, paired, dep_ids)
            min_idx = max((i + ONE for i, x in enumerate(augmented[r]) if x.op_id in dep_ids), default=ZERO)
            insert_at = next((i for i in range(min_idx, len(augmented[r]))
                              if augmented[r][i].op_id in after_ids), len(augmented[r]))
            for offset, synth in enumerate(chain):
                augmented[r].insert(insert_at + offset, synth)
                op_maps[r][synth.op_id] = synth
            prev_synth[(r, stage)] = chain[-ONE].op_id
    return {r: lower_local_projection_tp(ops, model=model, runtime=runtime, machine=machine,
                                         replica=r, cohort_views={}, scope="target", peer_rows=peer_rows)
            for r, ops in augmented.items()}
