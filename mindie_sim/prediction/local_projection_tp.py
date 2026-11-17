"""Semantic dense-subgraph local-TP region rewrite.

[Role]
  Insert node-local AllGather/AllToAll/ReduceScatter boundaries around declared
  dense projection operators and restore each original public output identity.
[Invariants]
  * Column projections gather rows, compute an output-feature shard and exchange
    slices; row projections exchange input shards and reduce-scatter partial rows;
    MLA grouped output pairs shard whole groups.
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
from mindie_sim.catalog.projection_placement import _padded_local, role_from_category
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
    if role == "attention_output":
        return int(runtime.parallel.o_proj_tp)
    if role in {"linear_input", "linear_gate", "linear_output"}:
        return int(runtime.parallel.linear_proj_tp)
    return int(runtime.parallel.lm_head_tp) if role in {"logits", "draft_logits"} else ONE


def _role(metadata: Mapping[str, Any]) -> str:
    return str(metadata.get("projection_role", "")) or role_from_category(
        str(metadata.get("quant_category", metadata.get("catalog_category", ""))))


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
    # The placement ledger pads the local hint to the weight-scale granule
    # (for example 512/TP8 with a 128-column block becomes 128, not 64).
    term = SimpleNamespace(quant=quant)
    return max(ONE, int(_padded_local(int(size), int(width), term)))


def _packed_local_width(metadata: Mapping[str, Any], width: int, total: int, quant: Any = None) -> int:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        return _local_dimension(total, width, quant)
    return max(ONE, sum(
        _local_dimension(int(tuple(s.get("shape", ()))[-ONE]) if s.get("shape") else ZERO, width, quant)
        for s in slices))


def _localize_packed_slices(metadata: dict[str, Any], width: int, quant: Any = None) -> None:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        return
    offset, local_slices = ZERO, []
    for item in slices:
        shape = tuple(item.get("shape", ()))
        local_size = _local_dimension(int(shape[-ONE]) if shape else ZERO, width, quant)
        local_slices.append({**dict(item), "shape": (int(shape[ZERO]) if shape else ZERO, local_size),
                             "strides": (local_size, ONE), "offset_elements": offset})
        offset += local_size
    metadata["packed_projection_slices"] = tuple(local_slices)


def _dtype(metadata: Mapping[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key)
    return str(value) if value else fallback


def _groups_for(machine: Any, runtime: Any, replica: int, stage: int, width: int) -> tuple[int, ...]:
    from mindie_sim.prediction.local_tp_groups import local_tp_groups
    from mindie_sim.prediction.topology import Topology
    ranks = Topology(machine, runtime.parallel).ranks(replica, stage)
    if not ranks:
        raise ModelingError(f"local_tp_stage_has_no_ranks:{stage}")
    anchor = int(ranks[ZERO])
    for group in local_tp_groups(machine, runtime.parallel, stage=stage, width=width):
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
                        family: str, scope: str) -> tuple[tuple[str, Any], ...]:
    src = dict(op.metadata)
    kept = {k: src[k] for k in _TRANSPORT_KEEP_META if k in src}
    kept.update({
        "collective": family, "group_axis": SDK_PLACEMENT,
        "group_ranks": tuple(int(rank) for rank in group),
        "local_tp_width": int(width), "projection_role": role,
        "placement_scope": str(src.get("placement_scope") or src.get("layer_scope") or scope or "target"),
        "local_tp_applied": True, "payload_dtype": str(src.get("payload_dtype", "")),
    })
    return tuple(sorted(kept.items(), key=lambda pair: pair[FIRST]))


def _collective(op: OpSpec, *, suffix: str, family: str, recipe: str, payload: float,
                shape: tuple[int, ...], role: str, group: Sequence[int], width: int,
                deps: tuple[str, ...], scope: str, dtype: str,
                public_id: str | None = None) -> OpSpec:
    built = comm_op(
        op_id=f"{op.op_id}:local_tp:{suffix}", deps=deps, stage=op.stage,
        layer=op.layer, requests=op.requests, family=family, phase=op.phase,
        recipe=recipe, link_bytes=payload, shape=shape,
        metadata=_transport_metadata(op, role=role, group=group, width=width,
                                     family=family, scope=scope),
        notes=(f"local_tp_{role}", f"local_tp_{suffix}"),
    )
    return replace(built, op_id=public_id or built.op_id, dtype=dtype)


def _rewrite_matrix(op: OpSpec, *, gemm_id: str, rows: int, cols: int, reduced: int,
                    role: str, width: int, dep_ids: tuple[str, ...],
                    weight_ratio: float, quant: Any, groups: int = ONE,
                    extra_metadata: Mapping[str, Any] | None = None) -> OpSpec:
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
    scale_b = float(metadata.get("activation_scale_bytes", ZERO_FLOAT) or ZERO_FLOAT) * (float(m) / float(orig_m))
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
    return ROLE_ROW if role in {"attention_output", "linear_output"} else ROLE_COLUMN


def _empty_passthrough(op: OpSpec, metadata: Mapping[str, Any], width: int) -> OpSpec:
    meta = {**dict(metadata), "local_tp_applied": True, "local_tp_width": int(width),
            "packed_weight_bytes": ZERO_FLOAT, "weight_bytes": ZERO_FLOAT}
    return replace(op, flops=ZERO_FLOAT, hbm_bytes=ZERO_FLOAT, mte2_bytes=ZERO_FLOAT,
                   mte3_bytes=ZERO_FLOAT,
                   metadata=tuple(sorted(meta.items(), key=lambda pair: pair[FIRST])))


def lower_local_projection_tp(
    ops: Sequence[OpSpec], *, model: Any, runtime: Any, machine: Any,
    replica: int, cohort_views: Mapping[int, Sequence[Any]], scope: str,
    peer_rows: Mapping[tuple[int, str], Mapping[int, int]] | None = None,
) -> tuple[OpSpec, ...]:
    """Rewrite declared dense projection regions for node-local TP."""
    if max(int(runtime.parallel.o_proj_tp), int(runtime.parallel.linear_proj_tp),
           int(runtime.parallel.lm_head_tp)) <= ONE:
        return tuple(ops)
    from mindie_sim.prediction.local_tp_groups import validate_local_projection_tp

    validate_local_projection_tp(SimpleNamespace(runtime=runtime, machine=machine))
    rewritten: list[OpSpec] = []
    for op in ops:
        metadata = dict(op.metadata)
        role, width = _role(metadata), _width(runtime, _role(metadata))
        if not role or width <= ONE or metadata.get("local_tp_applied"):
            rewritten.append(op)
            continue
        partition = _partition_kind(role, metadata)
        group = _groups_for(machine, runtime, replica, op.stage, width)
        if op.source_boundary or metadata.get("implementation"):
            impl = str(metadata.get("implementation", "") or metadata.get("kernel", "source"))
            adapter = RETARGET_ADAPTERS.get(impl)
            if adapter is None:
                raise ModelingError(f"local_tp_source_placement_unsupported:{impl}")
            op = adapter(op, width=width, role=role, group=group, partition=partition)
            metadata = dict(op.metadata)
        quant = _quant_for_op(model, metadata)
        m, n, k = (int(op.local_shape[ZERO]), int(op.local_shape[ONE]), int(op.local_shape[TWO]))
        op_scope = str(metadata.get("layer_scope") or scope or "target")
        op_peers = (peer_rows or {}).get((int(op.stage), op.op_id))
        act = _dtype(metadata, "activation_format", op.dtype or FALLBACK_ACTIVATION)
        out = _dtype(metadata, "output_format", FALLBACK_OUTPUT)
        metadata["local_tp_partition"] = partition
        gemm_id = f"{op.op_id}:local_tp:gemm"
        groups = max(ONE, int(metadata.get("matrix_groups", ONE))) if partition == ROLE_GROUP_A else ONE
        if (partition == ROLE_GROUP_A and groups % width != ZERO) or (
                partition == ROLE_GROUP_B and k % width != ZERO):
            raise ModelingError("local_tp_group_geometry_indivisible")
        counts = _row_counts(cohort_views, group, m // groups, op_peers)
        total_tok, pad_tok = sum(counts), max(counts, default=ZERO)
        if total_tok <= ZERO:
            rewritten.append(_empty_passthrough(op, metadata, width))
            continue
        if partition == ROLE_GROUP_A:
            lg = groups // width
            ex = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                             payload=float((width - ONE) * pad_tok * lg * k) * dtype_bytes(act),
                             shape=(total_tok * lg, k), role=role, group=group,
                             width=width, deps=op.deps, scope=op_scope, dtype=act)
            gemm = _rewrite_matrix(op, gemm_id=op.op_id, rows=total_tok * lg, cols=n, reduced=k,
                                   role=role, width=width, dep_ids=(ex.op_id,),
                                   weight_ratio=ONE / float(width), quant=quant, groups=lg, extra_metadata=metadata)
            rewritten.extend((ex, gemm))
        elif partition == ROLE_GROUP_B:
            gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=n, reduced=k // width,
                                   role=role, width=width, dep_ids=op.deps, weight_ratio=ONE / float(width),
                                   quant=quant, groups=ONE, extra_metadata=metadata)
            restore = _collective(op, suffix="reduce_scatter", family="reduce_scatter", recipe="ring_reduce_scatter",
                                  payload=float((width - ONE) * pad_tok * n) * dtype_bytes(out), shape=(m, n),
                                  role=role, group=group, width=width, deps=(gemm.op_id,), scope=op_scope,
                                  dtype=out, public_id=op.op_id)
            rewritten.extend((gemm, restore))
        elif partition == ROLE_COLUMN:
            local_n = _packed_local_width(metadata, width, n, quant)
            _localize_packed_slices(metadata, width, quant)
            gather = _collective(op, suffix="gather", family="allgather", recipe="ring_allgather",
                                 payload=float((width - ONE) * pad_tok * k) * dtype_bytes(act), shape=(total_tok, k),
                                 role=role, group=group, width=width, deps=op.deps, scope=op_scope, dtype=act)
            gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=local_n, reduced=k,
                                   role=role, width=width, dep_ids=(gather.op_id,),
                                   weight_ratio=float(local_n) / float(max(ONE, n)), quant=quant, extra_metadata=metadata)
            restore = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                                  payload=float((width - ONE) * pad_tok * local_n) * dtype_bytes(out), shape=(m, n),
                                  role=role, group=group, width=width, deps=(gemm.op_id,), scope=op_scope,
                                  dtype=out, public_id=op.op_id)
            rewritten.extend((gather, gemm, restore))
        else:
            local_k = _local_dimension(k, width, quant)
            ex = _collective(op, suffix="exchange", family="alltoall", recipe="alltoall",
                             payload=float((width - ONE) * pad_tok * local_k) * dtype_bytes(act),
                             shape=(total_tok, local_k), role=role, group=group, width=width,
                             deps=op.deps, scope=op_scope, dtype=act)
            gemm = _rewrite_matrix(op, gemm_id=gemm_id, rows=total_tok, cols=n, reduced=local_k,
                                   role=role, width=width, dep_ids=(ex.op_id,),
                                   weight_ratio=float(local_k) / float(max(ONE, k)), quant=quant, extra_metadata=metadata)
            restore = _collective(op, suffix="reduce_scatter", family="reduce_scatter", recipe="ring_reduce_scatter",
                                  payload=float((width - ONE) * pad_tok * n) * dtype_bytes(out), shape=(m, n),
                                  role=role, group=group, width=width, deps=(gemm.op_id,), scope=op_scope,
                                  dtype=out, public_id=op.op_id)
            rewritten.extend((ex, gemm, restore))
    return tuple(rewritten)


def lower_local_projection_tp_cohort(
    graphs: Mapping[int, Sequence[OpSpec]], *, model: Any, runtime: Any, machine: Any,
) -> dict[int, tuple[OpSpec, ...]]:
    """Cohort-level local-TP rewrite using actual per-rank operator rows and idle synthesis."""
    if max(int(runtime.parallel.o_proj_tp), int(runtime.parallel.linear_proj_tp),
           int(runtime.parallel.lm_head_tp)) <= ONE or not graphs:
        return {int(r): tuple(ops) for r, ops in graphs.items()}
    from mindie_sim.prediction.local_tp_groups import validate_local_projection_tp
    from mindie_sim.prediction.topology import Topology

    validate_local_projection_tp(SimpleNamespace(runtime=runtime, machine=machine))
    topo = Topology(machine, runtime.parallel)
    identifiers = [tuple(op.op_id for op in ops) for ops in graphs.values()]
    homogeneous = all(ids == identifiers[ZERO] for ids in identifiers)
    ordered_occ: list[tuple[int, str]] = []
    template_by_occ: dict[tuple[int, str], tuple[OpSpec, frozenset[str]]] = {}
    peer_rows: dict[tuple[int, str], dict[int, int]] = {}
    op_maps = {int(r): {op.op_id: op for op in ops} for r, ops in graphs.items()}
    for r, ops in sorted(graphs.items()):
        seq = tuple(ops)
        for idx, op in enumerate(seq):
            meta = dict(op.metadata)
            role, width = _role(meta), _width(runtime, _role(meta))
            if not role or width <= ONE or meta.get("local_tp_applied"):
                continue
            key = (int(op.stage), op.op_id)
            if key not in template_by_occ:
                ordered_occ.append(key)
                template_by_occ[key] = (
                    op, frozenset() if homogeneous else
                    frozenset(x.op_id for x in seq[idx + ONE:]))
            grp_div = max(ONE, int(meta.get("matrix_groups", ONE))) if _partition_kind(role, meta) == ROLE_GROUP_A else ONE
            tok_m = int(op.local_shape[ZERO]) // grp_div
            for rk in topo.ranks(int(r), int(op.stage)):
                peer_rows.setdefault(key, {})[int(rk)] = tok_m
    ext_roots = tuple(dict.fromkeys(d for r0, ops in sorted(graphs.items()) for item in ops
                                    for d in item.deps if d not in op_maps[int(r0)]))
    augmented: dict[int, list[OpSpec]] = {int(r): list(ops) for r, ops in graphs.items()}
    prev_synth: dict[tuple[int, int], str] = {}
    for stage, op_id in ordered_occ:
        tmpl, after_ids = template_by_occ[(stage, op_id)]
        width = _width(runtime, _role(dict(tmpl.metadata)))
        for r in sorted(augmented):
            if op_id in op_maps[r]:
                prev_synth[(r, stage)] = op_id
                continue
            group = _groups_for(machine, runtime, r, stage, width)
            for rk in topo.ranks(r, stage):
                peer_rows.setdefault((stage, op_id), {}).setdefault(int(rk), ZERO)
            if sum(peer_rows[(stage, op_id)].get(int(rk), ZERO) for rk in group) <= ZERO:
                continue
            existing_ids = {item.op_id for item in augmented[r]}
            dep_ids = tuple(d for d in tmpl.deps if d in existing_ids) or (
                (prev_synth[(r, stage)],) if (r, stage) in prev_synth
                else ("iteration_input",) if "iteration_input" in existing_ids else ext_roots[:ONE])
            z_local = (ZERO, int(tmpl.local_shape[ONE]), int(tmpl.local_shape[TWO]))
            z_pad = (ZERO, int(tmpl.padded_shape[ONE]), int(tmpl.padded_shape[TWO]))
            z_launch = (ZERO, int(tmpl.launch_shape[ONE]), int(tmpl.launch_shape[TWO]))
            synth = replace(tmpl, requests=(), deps=dep_ids, local_shape=z_local, padded_shape=z_pad,
                            launch_shape=z_launch, flops=ZERO_FLOAT, hbm_bytes=ZERO_FLOAT,
                            mte2_bytes=ZERO_FLOAT, mte3_bytes=ZERO_FLOAT)
            min_idx = max((i + ONE for i, x in enumerate(augmented[r]) if x.op_id in dep_ids), default=ZERO)
            insert_at = next((i for i in range(min_idx, len(augmented[r]))
                              if augmented[r][i].op_id in after_ids), len(augmented[r]))
            augmented[r].insert(insert_at, synth)
            op_maps[r][op_id], prev_synth[(r, stage)] = synth, op_id
    return {r: lower_local_projection_tp(ops, model=model, runtime=runtime, machine=machine,
                                         replica=r, cohort_views={}, scope="target", peer_rows=peer_rows)
            for r, ops in augmented.items()}
