"""Profiler observation geometry, timed-node DAG order, and writer-clock verification.

[Role] Parse profiler run-IDs/arms and quantiles, validate ``timed_nodes``
  sequences (zero blanks, duplicates, foreign prefixes, op-set drift, or
  dependency-order inversions), verify receipt/builder-reported ``actual_shape``
  and per-node ``node_shapes`` against planned geometry/bindings BEFORE
  sealing, derive ``clock_operator`` strictly from writer timing evidence,
  select clock-matched multi-mode arms, and extract canonical executed/ABI
  attempt geometry without accepting ``requested_shape``-only substitutions.
[Invariants]
  * Never overwrites writer-measured ``clock_operator`` or receipt-reported
    ``actual_shape`` with plan fields; fails closed before sealing.
  * Multi-GEMM subgraphs (e.g., all L3 step cases) MUST provide per-node
    ``actual_shape["node_shapes"][op_id]`` with exact ``(m, n, k, matrix_groups)``.
  * Rejects alias conflicts (e.g., ``topk=2049`` + ``sparse_count=2048``) and
    ``page_count`` vs ``block_table`` conflicts.
  * Zero magic numbers, zero I/O, zero global state mutations.
[Timeouts & Locks] Pure in-memory validation helpers; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from mindie_sim.schema.common import ModelingError

from .contracts import VALID_CLOCK_OPERATORS

_BLOCK_TABLE_PAIR_LEN = 2
_TRIAL_SUFFIX_RE = re.compile(r"\.[^.]+\.t\d+$")
_KNOWN_ARMS = frozenset({"base", "aclgraph"})
_BUSY_UNION_TIMING_MODES = frozenset({
    "stream_saturated", "busy_union", "event_busy_union", "eager_device", "aclgraph_device",
})
_ELAPSED_SPAN_TIMING_MODES = frozenset({"eager", "aclgraph", "elapsed_span", "event_elapsed_span"})
_BUSY_UNION_BOUNDARIES = frozenset({"busy_union", "event_busy_union"})
_GEMM_BINDING_FAMILIES = frozenset({"gemm", "grouped_gemm", "gate_up_projection", "down_projection"})
_COLLECTIVE_BINDING_FAMILIES = frozenset({"allreduce", "dcp_allgather"})


def quantile_ns(sorted_ns: Sequence[int], q: float) -> int:
    """Compute linear-interpolated quantile over sorted integer ns samples."""
    if not sorted_ns:
        return 0
    if len(sorted_ns) == 1:
        return int(sorted_ns[0])
    pos = q * (len(sorted_ns) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return int(sorted_ns[lo])
    return int(round(sorted_ns[lo] * (1.0 - (pos - lo)) + sorted_ns[hi] * (pos - lo)))


def parse_run_id_and_arm(
    run_id: str,
    timing_mode: str = "",
    trial: Any = None,
    source: str = "",
    execution: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Strip ``.<arm>.<mode>.t<trial>`` or ``.<mode>.t<trial>`` to recover ``(base_run_id, arm)``."""
    clean = str(run_id or "run-0").strip()
    if timing_mode and trial is not None and not isinstance(trial, bool):
        suffix = f".{timing_mode}.t{int(trial)}"
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    else:
        clean = _TRIAL_SUFFIX_RE.sub("", clean)
    arm = ""
    for candidate in _KNOWN_ARMS:
        marker = f".{candidate}"
        if clean.endswith(marker):
            clean = clean[: -len(marker)]
            arm = candidate
            break
    if not arm and isinstance(execution, Mapping):
        if str(execution.get("graph") or "").strip().lower() == "aclgraph":
            arm = "aclgraph"
    if not arm and source:
        prefix = source.rsplit("/", 1)[-1].split(".", 1)[0].strip().lower()
        if prefix in _KNOWN_ARMS:
            arm = prefix
    return clean or "run-0", arm or "base"


def _int_from(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        val = mapping.get(key)
        if val is not None and not isinstance(val, bool):
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def _unique_alias_int(
    mapping: Mapping[str, Any], keys: Sequence[str], *, cid: str, field: str
) -> int | None:
    """Resolve an integer across alias keys and fail closed on any conflicting values."""
    found: list[tuple[str, int]] = []
    for key in keys:
        if key not in mapping or mapping[key] is None:
            continue
        raw = mapping[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ModelingError(f"profiler_shape_alias_invalid:{cid}:{field}:{key}={raw!r}")
        found.append((key, int(raw)))
    if not found:
        return None
    distinct = {val for _k, val in found}
    if len(distinct) > 1:
        detail = ",".join(f"{k}={v}" for k, v in found)
        raise ModelingError(f"profiler_shape_alias_conflict:{cid}:{field}:{detail}")
    return found[0][1]


def extract_shape_dims(
    shape: Mapping[str, Any],
    plan_key: Mapping[str, Any],
    *,
    plan_pt: Mapping[str, Any] | None = None,
    expected_shape: Mapping[str, Any] | None = None,
    measured_binding: Mapping[str, Any] | None = None,
    cid: str = "",
) -> dict[str, Any]:
    """Extract canonical integer onboarding coordinates strictly from executed shape + plan key."""
    dims: dict[str, Any] = {}
    axis = str((plan_pt or {}).get("axis") or plan_key.get("axis") or "").strip()
    eff_shape: dict[str, Any] = dict(shape)
    primary = _primary_geometry_source(shape, measured_binding, cid=cid or "case")
    if primary is not shape:
        eff_shape.update(primary)
    elif isinstance(shape.get("node_shapes"), Mapping):
        for nval in shape["node_shapes"].values():
            if isinstance(nval, Mapping) and any(
                nval.get(mk) is not None
                for mk in ("heads", "kv_heads", "block_table", "top_k", "sparse_count", "paged")
            ):
                eff_shape.update(nval)
                break
    merged = {**dict(plan_key), **eff_shape}

    q_val = _int_from(eff_shape, "q_len", "query_tokens", "q")
    if q_val is None:
        q_val = _int_from(plan_key, "q_len", "query_tokens", "q")
    dims["query_tokens"] = q_val if q_val is not None else 1

    ctx_explicit = _int_from(eff_shape, "context_len", "context", "kv_len_prior")
    seq_len_val = _int_from(eff_shape, "seq_len", "kv_len", "scanned_rows")
    exp_seq_len = _int_from(expected_shape or {}, "seq_len", "kv_len")
    exp_ctx_len = _int_from(expected_shape or {}, "context_len")
    if seq_len_val is not None:
        dims["kv_len"] = seq_len_val
    if ctx_explicit is not None:
        dims["context_len"] = ctx_explicit
    elif seq_len_val is not None and exp_seq_len is not None and exp_ctx_len is not None and seq_len_val == exp_seq_len:
        dims["context_len"] = exp_ctx_len
    elif seq_len_val is not None:
        dims["context_len"] = seq_len_val
    else:
        pk_ctx = _int_from(plan_key, "context_len", "seq_len", "context")
        if pk_ctx is not None:
            dims["context_len"] = pk_ctx

    tok_explicit = _int_from(eff_shape, "tokens", "num_tokens")
    m_val = _int_from(eff_shape, "m")
    exp_m = _int_from(expected_shape or {}, "m")
    exp_tok = _int_from(expected_shape or {}, "tokens")
    if tok_explicit is not None:
        tok_val: int | None = tok_explicit
    elif m_val is not None and exp_m is not None and exp_tok is not None and m_val == exp_m:
        tok_val = exp_tok
    elif m_val is not None:
        tok_val = m_val
    else:
        tok_val = _int_from(plan_key, "tokens", "m", "num_tokens")
    if tok_val is not None:
        dims["tokens"] = tok_val

    seq_val = _int_from(eff_shape, "sequences", "batch_size", "batch", "b")
    kv_val = _int_from(eff_shape, "kv_heads")
    bxkv_val = _int_from(eff_shape, "batch_x_kv_heads")
    if bxkv_val is None and seq_val is not None and kv_val is not None and axis == "batch_x_kv_heads":
        bxkv_val = seq_val * kv_val
    if bxkv_val is not None:
        dims["batch_x_kv_heads"] = bxkv_val

    if seq_val is not None:
        dims["sequences"] = seq_val
    if axis == "batch_x_kv_heads":
        dims["batch_size"] = _int_from(plan_pt or {}, "batch_size") or _int_from(plan_key, "batch_size") or 1
    elif seq_val is not None:
        dims["batch_size"] = seq_val
    else:
        dims["batch_size"] = _int_from(plan_key, "batch_size", "batch", "b", "sequences") or 1

    tk_val = _int_from(eff_shape, "sparse_count", "top_k", "topk", "topk_items")
    if tk_val is None:
        tk_val = _int_from(plan_key, "sparse_count", "top_k", "topk", "topk_items")
    if tk_val is not None:
        dims["topk"] = tk_val

    dims["parallel_topology"] = str(
        merged.get("parallel_topology") or (plan_pt or {}).get("parallel_topology") or "tp1"
    )
    return dims


def _norm_timed_op_id(
    raw_id: Any,
    cid: str,
    *,
    error_prefix: str = "profiler_timed_nodes",
    require_case_prefix: bool = False,
) -> str:
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ModelingError(f"{error_prefix}_invalid_entry:{cid}:entry={raw_id!r}")
    clean = raw_id.strip()
    if ":" in clean:
        pfx, op_part = clean.split(":", 1)
        if pfx != cid or not op_part.strip():
            raise ModelingError(f"{error_prefix}_foreign_case_prefix:{cid}:entry={clean}")
        return op_part.strip()
    if require_case_prefix:
        raise ModelingError(f"{error_prefix}_missing_case_prefix:{cid}:entry={clean}")
    return clean


def validate_timed_nodes_sequence(
    raw_timed: Any,
    *,
    cid: str,
    level: str,
    expected_ops: Sequence[str] = (),
    expected_edges: Sequence[Sequence[str]] = (),
    error_prefix: str = "profiler_timed_nodes",
    require_case_prefix: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate non-empty timed_nodes with zero blanks/duplicates/foreign prefixes and topo dep order."""
    if not isinstance(raw_timed, Sequence) or isinstance(raw_timed, (str, bytes)) or not raw_timed:
        raise ModelingError(f"{error_prefix}_missing:{cid}:level={level}")
    items: list[str] = []
    op_list: list[str] = []
    for raw_entry in raw_timed:
        op_id = _norm_timed_op_id(
            raw_entry, cid, error_prefix=error_prefix, require_case_prefix=require_case_prefix
        )
        items.append(raw_entry.strip())
        op_list.append(op_id)
    if len(op_list) != len(set(op_list)):
        raise ModelingError(
            f"{error_prefix}_duplicate:{cid}:level={level}:count={len(op_list)}:unique={len(set(op_list))}"
        )
    exp_set = {_norm_timed_op_id(str(x), cid, error_prefix=error_prefix) for x in expected_ops if str(x).strip()}
    if exp_set and set(op_list) != exp_set:
        raise ModelingError(
            f"{error_prefix}_plan_mismatch:{cid}:level={level}:got={len(op_list)}:expected={len(exp_set)}"
        )
    pos = {oid: i for i, oid in enumerate(op_list)}
    for edge in expected_edges:
        if len(edge) < _BLOCK_TABLE_PAIR_LEN:
            continue
        src = _norm_timed_op_id(str(edge[0]), cid, error_prefix=error_prefix)
        dst = _norm_timed_op_id(str(edge[1]), cid, error_prefix=error_prefix)
        if src in pos and dst in pos and pos[src] >= pos[dst]:
            raise ModelingError(
                f"{error_prefix}_order_violation:{cid}:level={level}:edge={src}->{dst}:pos={pos[src]}>={pos[dst]}"
            )
    return items, op_list


def _verify_node_bindings_actual_geometry(
    actual_shape: Mapping[str, Any],
    measured_binding: Mapping[str, Any] | None,
    *,
    cid: str,
) -> None:
    if not isinstance(measured_binding, Mapping):
        return
    raw_bindings = measured_binding.get("node_bindings")
    if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)) or not raw_bindings:
        return
    bindings = [nb for nb in raw_bindings if isinstance(nb, Mapping)]
    raw_ns = actual_shape.get("node_shapes")
    node_shapes: dict[str, Mapping[str, Any]] = {}
    if raw_ns is not None and not isinstance(raw_ns, Mapping):
        raise ModelingError(f"profiler_node_shapes_invalid:{cid}")
    if isinstance(raw_ns, Mapping):
        for k, v in raw_ns.items():
            if not isinstance(v, Mapping):
                raise ModelingError(f"profiler_node_shapes_entry_invalid:{cid}:{k}")
            node_shapes[_norm_timed_op_id(k, cid, error_prefix="profiler_node_shapes")] = v
    gemm_ids = [
        str(nb.get("op_id") or "").strip()
        for nb in bindings
        if str(nb.get("family") or "") in _GEMM_BINDING_FAMILIES and str(nb.get("op_id") or "").strip()
    ]
    grouped_ids = [
        str(nb.get("op_id") or "").strip()
        for nb in bindings
        if str(nb.get("family") or "") in _GEMM_BINDING_FAMILIES
        and int(nb.get("matrix_groups") or 1) > 1
        and str(nb.get("op_id") or "").strip()
    ]
    last_grouped = grouped_ids[-1] if grouped_ids else ""
    sole_gemm = gemm_ids[0] if len(gemm_ids) == 1 else ""
    multi_gemm = len(gemm_ids) > 1
    for nb in bindings:
        oid = str(nb.get("op_id") or "").strip()
        fam = str(nb.get("family") or "").strip()
        per_node = node_shapes.get(oid)
        is_per_node = per_node is not None
        n_act = per_node if is_per_node else actual_shape
        if fam == "attention_merge":
            exp_dt = str(nb.get("dtype") or "").strip().lower()
            got_dt = str(n_act.get("dtype") or "").strip().lower()
            if exp_dt and not got_dt:
                raise ModelingError(f"profiler_actual_geometry_missing:{cid}:dtype:{oid}")
            if exp_dt and got_dt != exp_dt:
                raise ModelingError(
                    f"profiler_actual_geometry_mismatch:{cid}:dtype:{oid}:expected={exp_dt}:actual={got_dt}"
                )
            exp_p = _int_from(nb, "partial_count")
            got_p = _int_from(n_act, "partial_count")
            if exp_p is not None and got_p is None:
                raise ModelingError(f"profiler_actual_geometry_missing:{cid}:partial_count:{oid}")
            if exp_p is not None and got_p != exp_p:
                raise ModelingError(
                    f"profiler_actual_geometry_mismatch:{cid}:partial_count:{oid}:expected={exp_p}:actual={got_p}"
                )
        elif fam in _GEMM_BINDING_FAMILIES:
            exp_mg = _int_from(nb, "matrix_groups") or 1
            if multi_gemm and not is_per_node:
                raise ModelingError(f"profiler_actual_geometry_missing:{cid}:node_shapes:{oid}")
            check_dims = is_per_node or (exp_mg > 1 and oid == last_grouped) or (oid == sole_gemm)
            if not check_dims:
                continue
            for dim in ("m", "n", "k"):
                exp_d, got_d = _int_from(nb, dim), _int_from(n_act, dim)
                if exp_d is not None and got_d is None:
                    raise ModelingError(f"profiler_actual_geometry_missing:{cid}:{dim}:{oid}")
                if exp_d is not None and got_d != exp_d:
                    raise ModelingError(
                        f"profiler_actual_geometry_mismatch:{cid}:{dim}:{oid}:expected={exp_d}:actual={got_d}"
                    )
            if exp_mg > 1 or is_per_node:
                got_mg = _unique_alias_int(n_act, ("matrix_groups", "e"), cid=cid, field=f"matrix_groups:{oid}")
                if exp_mg > 1 and got_mg is None:
                    raise ModelingError(f"profiler_actual_geometry_missing:{cid}:matrix_groups:{oid}")
                if got_mg is not None and got_mg != exp_mg:
                    raise ModelingError(
                        f"profiler_actual_geometry_mismatch:{cid}:matrix_groups:{oid}:expected={exp_mg}:actual={got_mg}"
                    )
            exp_dt = str(nb.get("dtype") or "").strip().lower()
            exp_q = str(nb.get("quant_mode") or "none").strip().lower()
            got_dt = str(n_act.get("dtype") or n_act.get("input_dtype") or "").strip().lower()
            got_wdt = str(n_act.get("weight_dtype") or "").strip().lower()
            got_q = str(n_act.get("quant_mode") or n_act.get("quantization") or "").strip().lower()
            if exp_dt and not got_dt:
                raise ModelingError(f"profiler_actual_geometry_missing:{cid}:dtype:{oid}")
            if exp_dt and got_dt != exp_dt:
                raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:dtype:{oid}:expected={exp_dt}:actual={got_dt}")
            if exp_dt and got_wdt and got_wdt != exp_dt:
                raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:weight_dtype:{oid}:expected={exp_dt}:actual={got_wdt}")
            if got_q and got_q not in {"", "none", "dense", exp_q, exp_dt}:
                raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:quant_mode:{oid}:expected={exp_q}:actual={got_q}")
            if (n_act.get("has_scales") is True or str(n_act.get("scale_dtype") or "").strip()) and exp_q in {"", "none", "dense"}:
                raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:unexpected_quant_scales:{oid}")
        elif fam in _COLLECTIVE_BINDING_FAMILIES and is_per_node:
            exp_dt = str(nb.get("dtype") or "").strip().lower()
            got_dt = str(n_act.get("dtype") or "").strip().lower()
            if exp_dt and got_dt and got_dt != exp_dt:
                raise ModelingError(
                    f"profiler_actual_geometry_mismatch:{cid}:dtype:{oid}:expected={exp_dt}:actual={got_dt}"
                )
            for ckey in ("nbytes", "group_size"):
                exp_c, got_c = _int_from(nb, ckey), _int_from(n_act, ckey)
                if exp_c is not None and got_c is not None and got_c != exp_c:
                    raise ModelingError(
                        f"profiler_actual_geometry_mismatch:{cid}:{ckey}:{oid}:expected={exp_c}:actual={got_c}"
                    )
        elif fam == "rmsnorm" and is_per_node:
            exp_w, got_w = _int_from(nb, "width"), _int_from(n_act, "width")
            if exp_w is not None and got_w is not None and got_w != exp_w:
                raise ModelingError(
                    f"profiler_actual_geometry_mismatch:{cid}:width:{oid}:expected={exp_w}:actual={got_w}"
                )


def _verify_l1_gemm_actual_geometry(
    n_act: Mapping[str, Any],
    mb: Mapping[str, Any],
    *,
    cid: str,
) -> None:
    oid = str(mb.get("target_op_id") or mb.get("builder") or "op0").strip()
    for dim in ("m", "n", "k"):
        exp_d, got_d = _int_from(mb, dim), _int_from(n_act, dim)
        if exp_d is not None and got_d is None:
            raise ModelingError(f"profiler_actual_geometry_missing:{cid}:{dim}:{oid}")
        if exp_d is not None and got_d != exp_d:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:{dim}:{oid}:expected={exp_d}:actual={got_d}"
            )
    exp_mg = _int_from(mb, "matrix_groups") or 1
    got_mg = _unique_alias_int(n_act, ("matrix_groups", "e"), cid=cid, field=f"matrix_groups:{oid}")
    if exp_mg > 1 and got_mg is None:
        raise ModelingError(f"profiler_actual_geometry_missing:{cid}:matrix_groups:{oid}")
    if got_mg is not None and got_mg != exp_mg:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:matrix_groups:{oid}:expected={exp_mg}:actual={got_mg}"
        )
    exp_dt = str(mb.get("dtype") or "").strip().lower()
    exp_wdt = str(mb.get("weight_dtype") or exp_dt).strip().lower()
    exp_q = str(mb.get("quant_mode") or "none").strip().lower()
    got_dt = str(n_act.get("dtype") or n_act.get("input_dtype") or "").strip().lower()
    got_wdt = str(n_act.get("weight_dtype") or "").strip().lower()
    got_q = str(n_act.get("quant_mode") or n_act.get("quantization") or "").strip().lower()
    if exp_dt and not got_dt:
        raise ModelingError(f"profiler_actual_geometry_missing:{cid}:dtype:{oid}")
    if exp_dt and got_dt != exp_dt:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:dtype:{oid}:expected={exp_dt}:actual={got_dt}")
    if exp_wdt and got_wdt and got_wdt != exp_wdt:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:weight_dtype:{oid}:expected={exp_wdt}:actual={got_wdt}"
        )
    if got_q and got_q not in {"", "none", "dense", exp_q, exp_dt}:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:quant_mode:{oid}:expected={exp_q}:actual={got_q}")
    if (n_act.get("has_scales") is True or str(n_act.get("scale_dtype") or "").strip()) and exp_q in {"", "none", "dense"}:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:unexpected_quant_scales:{oid}")


def _primary_geometry_source(
    actual_shape: Mapping[str, Any],
    measured_binding: Mapping[str, Any] | None,
    *,
    cid: str,
) -> Mapping[str, Any]:
    raw_ns = actual_shape.get("node_shapes")
    if not isinstance(raw_ns, Mapping) or not isinstance(measured_binding, Mapping):
        return actual_shape
    norm_ns = {
        _norm_timed_op_id(k, cid, error_prefix="profiler_node_shapes"): v
        for k, v in raw_ns.items()
        if isinstance(v, Mapping)
    }
    for nb in measured_binding.get("node_bindings") or ():
        if isinstance(nb, Mapping) and str(nb.get("family") or "") in {"attention", "indexer"}:
            oid = str(nb.get("op_id") or "").strip()
            if oid and oid in norm_ns:
                return norm_ns[oid]
    return actual_shape


def verify_actual_geometry_matches_plan(
    actual_shape: Mapping[str, Any],
    plan_pt: Mapping[str, Any],
    *,
    expected_shape: Mapping[str, Any],
    cid: str,
    measured_binding: Mapping[str, Any] | None = None,
) -> None:
    """Compare receipt/builder-reported executed geometry against the plan BEFORE sealing."""
    if not actual_shape:
        raise ModelingError(f"profiler_actual_shape_missing:{cid}")
    geo_src = _primary_geometry_source(actual_shape, measured_binding, cid=cid)
    axis = str(plan_pt.get("axis") or "").strip()
    exp_val = int(plan_pt["value"])
    exp_q = max(1, int(plan_pt.get("query_tokens", 1)))
    exp_seqs = _int_from(expected_shape, "sequences")
    exp_tokens = _int_from(expected_shape, "tokens")
    is_attn_or_idx = bool(
        expected_shape.get("paged")
        or expected_shape.get("seq_len") is not None
        or expected_shape.get("kv_len") is not None
    )

    got_q = _unique_alias_int(geo_src, ("q_len", "query_tokens"), cid=cid, field="q_len")
    if is_attn_or_idx and got_q is None:
        raise ModelingError(f"profiler_actual_geometry_missing:{cid}:q_len")
    if got_q is not None and got_q != exp_q:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:q_len:expected={exp_q}:actual={got_q}")

    got_seqs = (
        _int_from(geo_src, "sequences")
        if (is_attn_or_idx or axis == "batch_x_kv_heads")
        else _unique_alias_int(geo_src, ("sequences", "batch_size"), cid=cid, field="sequences")
    )
    if (is_attn_or_idx or axis == "batch_x_kv_heads") and exp_seqs is not None and got_seqs is None:
        raise ModelingError(f"profiler_actual_geometry_missing:{cid}:sequences")
    if exp_seqs is not None and got_seqs is not None and got_seqs != exp_seqs:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:sequences:expected={exp_seqs}:actual={got_seqs}")

    got_toks = _int_from(geo_src, "tokens")
    if axis == "tokens" and exp_tokens is not None and got_toks is None and _int_from(geo_src, "m") is None:
        raise ModelingError(f"profiler_actual_geometry_missing:{cid}:tokens")
    if exp_tokens is not None and got_toks is not None and got_toks != exp_tokens:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:tokens:expected={exp_tokens}:actual={got_toks}")

    exp_seq_len = _int_from(expected_shape, "seq_len", "kv_len")
    got_seq_len = _int_from(geo_src, "seq_len", "kv_len")
    if is_attn_or_idx and exp_seq_len is not None and got_seq_len is None:
        raise ModelingError(f"profiler_actual_geometry_missing:{cid}:seq_len")
    if exp_seq_len is not None and got_seq_len is not None and got_seq_len != exp_seq_len:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:seq_len:expected={exp_seq_len}:actual={got_seq_len}")
    if axis == "context_len":
        got_ctx = _int_from(geo_src, "context_len")
        if got_ctx is not None and got_ctx != exp_val:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:context_len:expected={exp_val}:actual={got_ctx}")

    exp_pages = _int_from(expected_shape, "page_count")
    explicit_pages = _int_from(geo_src, "page_count")
    bt = geo_src.get("block_table")
    bt_seqs: int | None = None
    bt_pages: int | None = None
    if isinstance(bt, Sequence) and not isinstance(bt, (str, bytes)) and len(bt) >= _BLOCK_TABLE_PAIR_LEN:
        if not isinstance(bt[0], bool) and isinstance(bt[0], int):
            bt_seqs = int(bt[0])
        if not isinstance(bt[1], bool) and isinstance(bt[1], int):
            bt_pages = int(bt[1])
    if bt_seqs is not None and exp_seqs is not None and bt_seqs != exp_seqs:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:block_table_sequences:expected={exp_seqs}:actual={bt_seqs}"
        )
    if bt_seqs is not None and got_seqs is not None and bt_seqs != got_seqs:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:sequences_block_table_conflict"
            f":sequences={got_seqs}:block_table_sequences={bt_seqs}"
        )
    if explicit_pages is not None and bt_pages is not None and explicit_pages != bt_pages:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:page_count_block_table_conflict"
            f":page_count={explicit_pages}:block_table_pages={bt_pages}"
        )
    for got_pages in (explicit_pages, bt_pages):
        if exp_pages is not None and got_pages is not None and got_pages != exp_pages:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:page_count:expected={exp_pages}:actual={got_pages}"
            )

    if axis == "batch_x_kv_heads":
        got_bxkv = _int_from(geo_src, "batch_x_kv_heads")
        got_kv = _int_from(geo_src, "kv_heads")
        if got_bxkv is not None and got_bxkv != exp_val:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:batch_x_kv_heads:expected={exp_val}:actual={got_bxkv}")
        if got_seqs is not None and got_kv is not None and got_seqs * got_kv != exp_val:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:batch_x_kv_heads:expected={exp_val}:actual={got_seqs * got_kv}")

    got_tk = _unique_alias_int(geo_src, ("topk", "top_k", "sparse_count"), cid=cid, field="topk")
    if plan_pt.get("topk") is not None and got_tk is not None and got_tk != int(plan_pt["topk"]):
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:top_k:expected={plan_pt['topk']}:actual={got_tk}")

    exp_layout = str(expected_shape.get("query_layout") or expected_shape.get("input_layout") or "").strip().upper()
    got_layout = str(geo_src.get("query_layout") or geo_src.get("input_layout") or "").strip().upper()
    if exp_layout and got_layout and got_layout != exp_layout:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:query_layout:expected={exp_layout}:actual={got_layout}")

    for field in ("kv_heads", "heads", "head_dim", "latent", "width"):
        exp_f = _int_from(expected_shape, field)
        got_f = _int_from(geo_src, field)
        if exp_f is not None and got_f is not None and got_f != exp_f:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:{field}:expected={exp_f}:actual={got_f}")
    if (
        isinstance(measured_binding, Mapping)
        and not measured_binding.get("node_bindings")
        and str(measured_binding.get("family") or "").strip() in _GEMM_BINDING_FAMILIES
    ):
        _verify_l1_gemm_actual_geometry(geo_src, measured_binding, cid=cid)
    _verify_node_bindings_actual_geometry(actual_shape, measured_binding, cid=cid)


def derive_writer_clock_operator(
    group: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    cid: str,
    tmode: str,
) -> str:
    """Derive clock operator strictly from profiler writer timing evidence (never plan)."""
    explicit: set[str] = set()
    for row in group:
        exec_map = row.get("execution") if isinstance(row.get("execution"), Mapping) else {}
        ident_map = row.get("identity") if isinstance(row.get("identity"), Mapping) else {}
        ident_exec = ident_map.get("execution") if isinstance(ident_map.get("execution"), Mapping) else {}
        for cand in (
            row.get("clock_operator"), exec_map.get("clock_operator"),
            ident_map.get("clock_operator"), ident_exec.get("clock_operator"),
            summary.get("clock_operator"),
        ):
            val = str(cand or "").strip()
            if val:
                if val not in VALID_CLOCK_OPERATORS:
                    raise ModelingError(f"profiler_writer_clock_operator_invalid:{cid}:{val}")
                explicit.add(val)
    if len(explicit) > 1:
        raise ModelingError(f"profiler_writer_clock_operator_conflict:{cid}:{sorted(explicit)}")
    if explicit:
        return next(iter(explicit))
    boundary_name = str(group[0].get("boundary") or "device_kernel").strip().lower()
    mode_low = str(tmode or "").strip().lower()
    if boundary_name in _BUSY_UNION_BOUNDARIES:
        return "busy_union"
    if mode_low == "stream_saturated" and boundary_name in {"fx_region", "model_step"}:
        return "elapsed_span"
    if mode_low in _BUSY_UNION_TIMING_MODES:
        return "busy_union"
    if mode_low in _ELAPSED_SPAN_TIMING_MODES:
        return "elapsed_span"
    return "busy_union" if boundary_name == "device_kernel" else "elapsed_span"


def _attach_top_level_node_shapes(shape: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    top_ns = src.get("node_shapes")
    if isinstance(top_ns, Mapping) and top_ns:
        existing = shape.get("node_shapes")
        if isinstance(existing, Mapping) and existing and dict(existing) != dict(top_ns):
            raise ModelingError("profiler_node_shapes_conflict")
        shape["node_shapes"] = dict(top_ns)
    return shape


def extract_observed_shape(
    summary: Mapping[str, Any],
    first: Mapping[str, Any],
    comp: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    *,
    allow_requested_fallback: bool = True,
) -> dict[str, Any]:
    """Return the richest non-placeholder receipt/builder-reported executed shape."""
    for rcpt in receipts:
        if isinstance(rcpt, Mapping):
            for key in ("actual_shape", "executed_shape"):
                cand = rcpt.get(key)
                if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
                    return _attach_top_level_node_shapes(dict(cand), rcpt)
            top_ns = rcpt.get("node_shapes")
            if isinstance(top_ns, Mapping) and top_ns:
                return {"node_shapes": dict(top_ns)}
    comp_map = comp if isinstance(comp, Mapping) else {}
    writer_cands: list[tuple[Any, Mapping[str, Any]]] = [
        (comp_map.get("actual_shape"), comp_map),
        (summary.get("actual_shape"), summary),
        (first.get("actual_shape"), first),
        (comp_map.get("executed_shape"), comp_map),
        (summary.get("executed_shape"), summary),
        (first.get("executed_shape"), first),
    ]
    if allow_requested_fallback:
        ident = first.get("identity") if isinstance(first.get("identity"), Mapping) else {}
        key_map = first.get("key") if isinstance(first.get("key"), Mapping) else {}
        writer_cands.extend([
            (summary.get("requested_shape"), summary),
            (first.get("requested_shape"), first),
            (ident.get("shape"), ident),
            (key_map.get("shape"), key_map),
        ])
    for cand, parent in writer_cands:
        if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
            return _attach_top_level_node_shapes(dict(cand), parent)
    return {}


def extract_abi_attempted_shape(
    rec: Mapping[str, Any], planned_pt: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Extract writer-reported attempted geometry for an ABI rejection; reject ``requested_shape``-only & alias conflicts."""
    cid = str(rec.get("case_id") or rec.get("point_id") or "abi").strip()
    hint = rec.get("unmodeled_hint") if isinstance(rec.get("unmodeled_hint"), Mapping) else {}
    raw: Mapping[str, Any] | None = None
    for cand in (
        rec.get("executed_shape"), rec.get("attempted_shape"), rec.get("actual_shape"),
        hint.get("executed_shape"), hint.get("actual_shape"),
    ):
        if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
            raw = cand
            break
    if raw is None:
        return None
    out = dict(raw)
    axis = str((planned_pt or {}).get("axis") or "").strip()
    tk = _unique_alias_int(out, ("topk", "top_k", "sparse_count"), cid=cid, field="topk")
    if tk is not None:
        out["topk"] = tk
    q_val = _unique_alias_int(out, ("query_tokens", "q_len"), cid=cid, field="query_tokens")
    if q_val is not None:
        out["query_tokens"] = q_val
    ctx_val = _unique_alias_int(out, ("context_len",), cid=cid, field="context_len")
    if ctx_val is None:
        ctx_val = _unique_alias_int(out, ("seq_len",), cid=cid, field="context_len")
    if ctx_val is not None:
        out["context_len"] = ctx_val
    if axis != "batch_x_kv_heads":
        b_val = _unique_alias_int(out, ("batch_size", "sequences"), cid=cid, field="batch_size")
        if b_val is not None:
            out["batch_size"] = b_val
    return out


def select_mode_candidates(
    candidates: Sequence[tuple[str, str, str, list[Mapping[str, Any]]]],
    *,
    cid: str,
    plan_pt: Mapping[str, Any] | None,
) -> list[tuple[str, str, str, list[Mapping[str, Any]]]]:
    """Filter multi-mode groups to the plan's clock/timing mode; fail closed on mismatch."""
    plan_clk = str((plan_pt or {}).get("clock_operator") or "").strip()
    if not plan_clk:
        return list(candidates)
    matched = [item for item in candidates if item[2] == plan_clk]
    if not matched:
        found = sorted({item[2] for item in candidates})
        raise ModelingError(
            f"clock_operator_plan_mismatch:{cid}:plan={plan_clk}:measured={','.join(found)}"
        )
    if len(matched) == 1:
        return matched
    pref_mode = "eager" if plan_clk == "elapsed_span" else "stream_saturated"
    scored = sorted(
        matched,
        key=lambda it: (0 if it[0] == "base" else 1, 0 if it[1].lower() == pref_mode else 1, it[0], it[1]),
    )
    return [scored[0]]


__all__ = [
    "_int_from",
    "_norm_timed_op_id",
    "_unique_alias_int",
    "derive_writer_clock_operator",
    "extract_abi_attempted_shape",
    "extract_observed_shape",
    "extract_shape_dims",
    "parse_run_id_and_arm",
    "quantile_ns",
    "select_mode_candidates",
    "validate_timed_nodes_sequence",
    "verify_actual_geometry_matches_plan",
]
