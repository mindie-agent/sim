"""Semantic dense-subgraph local-TP region rewrite.

[Role]
  Insert the node-local AllGather/AllToAll/ReduceScatter boundaries around
  declared dense projection operators and restore each original public output
  identity.  Work is rewritten before mutable execution state exists and is
  idempotent.

[Invariants]
  * Column projections gather activation rows, compute a local output-feature
    shard and exchange slices back to row owners.
  * Row projections exchange input-feature shards, compute full outputs and
    reduce-scatter partial rows.
  * Per-rank matrix FLOPs remain ``2*m*n*k``; no free division by the local
    width and no partial output escapes to an external consumer.
  * Original deps, request/state metadata, stream, stage and phase are
    preserved; the final collective keeps the original public op id.
  * A source-bound implementation without a registered retargeting adapter
    raises ``local_tp_source_placement_unsupported``.

[Timeouts & Locks]
  Pure bounded graph transformation; no Engine, I/O or locks.

[Side-Effect Policy]
  Returns a new operator tuple; never mutates input records.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from mindie_sim.catalog.ops import comm_op
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec


ZERO = 0
ONE = 1
TWO = 2
FALLBACK_ACTIVATION = "bf16"
FALLBACK_OUTPUT = "bf16"
ZERO_FLOAT = 0.0
FIRST = 0
ALIGN_ELEMENT = 16
RETARGET_ADAPTERS: dict[str, Any] = {}
ROLE_COLUMN = "column"
ROLE_ROW = "row"
SDK_PLACEMENT = "local_tp"


def register_retarget_adapter(implementation: str, adapter: Any) -> None:
    """Register a source implementation that can accept the new placement."""
    if not implementation or not callable(adapter):
        raise ModelingError("local_tp_retarget_adapter_invalid")
    RETARGET_ADAPTERS[str(implementation)] = adapter


def _width(metadata: Mapping[str, Any], runtime: Any, role: str) -> int:
    if role == "attention_output":
        return int(runtime.parallel.o_proj_tp)
    if role in {"linear_input", "linear_gate", "linear_output"}:
        return int(runtime.parallel.linear_proj_tp)
    if role in {"logits", "draft_logits"}:
        return int(runtime.parallel.lm_head_tp)
    return ONE


def _role(metadata: Mapping[str, Any]) -> str:
    declared = str(metadata.get("projection_role", ""))
    if declared:
        return declared
    from mindie_sim.catalog.projection_placement import role_from_category

    return role_from_category(str(
        metadata.get("quant_category", metadata.get("catalog_category", ""))))


def _local_dimension(size: int, width: int) -> int:
    if width <= ONE:
        return int(size)
    padded = int(size) if int(size) % width == ZERO else align_up(int(size), width)
    return max(ONE, padded // width)


def _packed_local_width(metadata: Mapping[str, Any], width: int,
                        total: int) -> int:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        return _local_dimension(total, width)
    local = ZERO
    for item in slices:
        shape = tuple(item.get("shape", ()))
        size = int(shape[-ONE]) if shape else ZERO
        local += _local_dimension(size, width)
    return max(ONE, local)


def _localize_packed_slices(metadata: dict[str, Any], width: int) -> None:
    slices = metadata.get("packed_projection_slices")
    if not slices:
        return
    offset = ZERO
    local_slices = []
    for item in slices:
        shape = tuple(item.get("shape", ()))
        rows = int(shape[ZERO]) if shape else ZERO
        size = int(shape[-ONE]) if shape else ZERO
        local_size = _local_dimension(size, width)
        local_slices.append({**dict(item),
                             "shape": (rows, local_size),
                             "strides": (local_size, ONE),
                             "offset_elements": offset})
        offset += local_size
    metadata["packed_projection_slices"] = tuple(local_slices)


def _dtype(metadata: Mapping[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key)
    return str(value) if value else fallback


def _groups_for(machine: Any, runtime: Any, replica: int, stage: int, width: int) -> tuple[int, ...]:
    from mindie_sim.prediction.local_tp_groups import local_tp_groups
    from mindie_sim.prediction.topology import Topology

    topology = Topology(machine, runtime.parallel)
    ranks = topology.ranks(replica, stage)
    if not ranks:
        raise ModelingError(f"local_tp_stage_has_no_ranks:{stage}")
    anchor = int(ranks[ZERO])
    for group in local_tp_groups(machine, runtime.parallel, stage=stage, width=width):
        if anchor in group:
            return group
    raise ModelingError(f"local_tp_group_missing:{anchor}")


def _row_counts(cohort_views: Mapping[int, Sequence[Any]], group: Sequence[int],
                fallback: int) -> list[int]:
    counts: list[int] = []
    for rank in group:
        views = cohort_views.get(int(rank)) if cohort_views else None
        if views is not None:
            counts.append(sum(int(view.logical_q_len) for view in views))
        else:
            counts.append(int(fallback))
    return counts


def _common_metadata(op: OpSpec, *, role: str, group: Sequence[int], width: int,
                     family: str) -> tuple[tuple[str, Any], ...]:
    metadata = dict(op.metadata)
    metadata.update({
        "collective": family,
        "group_axis": SDK_PLACEMENT,
        "group_ranks": tuple(int(rank) for rank in group),
        "local_tp_width": int(width),
        "projection_role": role,
        "placement_scope": str(metadata.get("placement_scope", "target")),
        "local_tp_applied": True,
        "payload_dtype": str(metadata.get("payload_dtype", "")),
    })
    if family in {"alltoall", "alltoallv"}:
        metadata.setdefault("steps", max(ONE, len(group) - ONE))
    return tuple(sorted(metadata.items(), key=lambda pair: pair[FIRST]))


def _collective(op: OpSpec, *, suffix: str, family: str, recipe: str,
                payload: float, shape: tuple[int, ...], role: str,
                group: Sequence[int], width: int, deps: tuple[str, ...]) -> OpSpec:
    return comm_op(
        op_id=f"{op.op_id}:local_tp:{suffix}", deps=deps, stage=op.stage,
        layer=op.layer, requests=op.requests, family=family, phase=op.phase,
        recipe=recipe, link_bytes=payload, shape=shape,
        metadata=_common_metadata(op, role=role, group=group, width=width,
                                  family=family),
        notes=(f"local_tp_{role}", f"local_tp_{suffix}"),
    )


def _rewrite_matrix(op: OpSpec, *, rows: int, cols: int, reduced: int,
                    role: str, group: Sequence[int], width: int,
                    dep: str, weight_ratio: float,
                    extra_metadata: Mapping[str, Any] | None = None) -> OpSpec:
    metadata = dict(op.metadata)
    metadata.update(dict(extra_metadata or ()))
    original_weight = float(metadata.get("packed_weight_bytes", ZERO_FLOAT) or ZERO_FLOAT)
    local_weight = original_weight * float(weight_ratio)
    act = _dtype(metadata, "activation_format", op.dtype or FALLBACK_ACTIVATION)
    out = _dtype(metadata, "output_format", FALLBACK_OUTPUT)
    m, n, k = int(rows), int(cols), int(reduced)
    input_bytes = float(m * k) * dtype_bytes(act)
    output_bytes = float(m * n) * dtype_bytes(out)
    scale_bytes = float(metadata.get("activation_scale_bytes", ZERO_FLOAT) or ZERO_FLOAT)
    metadata.update({
        "packed_weight_bytes": local_weight,
        "weight_bytes": local_weight,
        "placement_role": role,
        "local_tp_width": int(width),
        "local_tp_applied": True,
        "matrix_groups": max(ONE, int(metadata.get("matrix_groups", ONE))),
    })
    hbm = input_bytes + local_weight + output_bytes + scale_bytes
    return replace(
        op,
        local_shape=(m, n, k),
        padded_shape=(align_up(m, ALIGN_ELEMENT), align_up(n, ALIGN_ELEMENT), align_up(k, ALIGN_ELEMENT)),
        launch_shape=(m, n, k),
        flops=float(TWO * m * n * k),
        hbm_bytes=hbm,
        mte2_bytes=input_bytes + local_weight + scale_bytes,
        mte3_bytes=output_bytes,
        deps=(dep,),
        metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[FIRST])),
    )


def _source_supported(metadata: Mapping[str, Any]) -> bool:
    implementation = str(metadata.get("implementation", "") or "")
    return bool(implementation) and implementation in RETARGET_ADAPTERS


def lower_local_projection_tp(
    ops: Sequence[OpSpec], *, model: Any, runtime: Any, machine: Any,
    replica: int, cohort_views: Mapping[int, Sequence[Any]], scope: str,
) -> tuple[OpSpec, ...]:
    """Rewrite declared dense projection regions for node-local TP."""
    del model
    widths = (int(runtime.parallel.o_proj_tp),
              int(runtime.parallel.linear_proj_tp),
              int(runtime.parallel.lm_head_tp))
    if max(widths) <= ONE:
        return tuple(ops)
    from types import SimpleNamespace
    from mindie_sim.prediction.local_tp_groups import validate_local_projection_tp

    validate_local_projection_tp(SimpleNamespace(runtime=runtime, machine=machine))
    rewritten: list[OpSpec] = []
    for op in ops:
        metadata = dict(op.metadata)
        role = _role(metadata)
        width = _width(metadata, runtime, role)
        if not role or width <= ONE:
            rewritten.append(op)
            continue
        if metadata.get("local_tp_applied"):
            rewritten.append(op)
            continue
        if (op.source_boundary or metadata.get("implementation")) \
                and not _source_supported(metadata):
            implementation = str(metadata.get("implementation", "") or
                                 metadata.get("kernel", "source"))
            raise ModelingError(
                f"local_tp_source_placement_unsupported:{implementation}")
        group = _groups_for(machine, runtime, replica, op.stage, width)
        m, n, k = (int(op.local_shape[ZERO]), int(op.local_shape[ONE]),
                   int(op.local_shape[TWO]))
        rows = _row_counts(cohort_views, group, m)
        act = _dtype(metadata, "activation_format", op.dtype or FALLBACK_ACTIVATION)
        out = _dtype(metadata, "output_format", FALLBACK_OUTPUT)
        row_parallel = role in {"attention_output", "linear_output"} or (
            role == "attention_output" and int(metadata.get("matrix_groups", ONE)) > ONE)
        partition = ROLE_ROW if row_parallel else ROLE_COLUMN
        metadata["local_tp_partition"] = partition
        if partition == ROLE_COLUMN:
            local_n = _packed_local_width(metadata, width, n)
            weight_ratio = float(local_n) / float(max(ONE, n))
            _localize_packed_slices(metadata, width)
            gather = _collective(
                op, suffix="gather", family="allgather",
                recipe="ring_allgather",
                payload=float((width - ONE) * m * k) * dtype_bytes(act),
                shape=(sum(rows), k), role=role, group=group, width=width,
                deps=op.deps)
            gather = replace(gather, dtype=act)
            gemm = _rewrite_matrix(
                op, rows=sum(rows), cols=local_n, reduced=k, role=role,
                group=group, width=width, dep=gather.op_id,
                weight_ratio=weight_ratio, extra_metadata=metadata)
            restore = _collective(
                op, suffix="exchange", family="alltoall", recipe="alltoall",
                payload=float((width - ONE) * m * local_n) * dtype_bytes(out),
                shape=(m, n), role=role, group=group, width=width,
                deps=(gemm.op_id,))
            restore = replace(restore, op_id=op.op_id, dtype=out)
            rewritten.extend((gather, gemm, restore))
        else:
            local_k = _local_dimension(k, width)
            weight_ratio = float(local_k) / float(max(ONE, k))
            exchange = _collective(
                op, suffix="exchange", family="alltoall", recipe="alltoall",
                payload=float((width - ONE) * m * local_k) * dtype_bytes(act),
                shape=(sum(rows), local_k), role=role, group=group, width=width,
                deps=op.deps)
            exchange = replace(exchange, dtype=act)
            gemm = _rewrite_matrix(
                op, rows=sum(rows), cols=n, reduced=local_k, role=role,
                group=group, width=width, dep=exchange.op_id,
                weight_ratio=weight_ratio, extra_metadata=metadata)
            restore = _collective(
                op, suffix="reduce_scatter", family="reduce_scatter",
                recipe="ring_reduce_scatter",
                payload=float((width - ONE) * m * n) * dtype_bytes(out),
                shape=(m, n), role=role, group=group, width=width,
                deps=(gemm.op_id,))
            restore = replace(restore, op_id=op.op_id, dtype=out)
            rewritten.extend((exchange, gemm, restore))
    return tuple(rewritten)
