"""DeepEP two-stage deduplicated communication and shared-expert overlap.

[Role] Price DeepEP's inter-node group exchange plus intra-node scatter against
  a flat AllToAll ledger, expose the EPLB load ratio, and run the causal engine
  with the shared-expert stream enabled or disabled.
[Invariants] Node-level dispatch deduplicates a token to one remote delivery;
  intra-node traffic counts each of its top-k assignments. Link work is
  capacity times bytes, and startup latency is charged once per collective
  stage. Shared-expert overlap changes scheduling only: total FLOPs and HBM
  bytes are conserved. Unsupported topology or ratio declarations raise.
[Timeouts & Locks] Pure arithmetic; the causal-engine calls are bounded by the
  supplied case and start no device work, I/O, locks or remote execution.
[Side-Effect Policy] Returns immutable reports; never mutates the input case.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any, Literal

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from .deepep_exchange import (
    _DOWN_OVERLAP_COLUMN_ALIGNMENT,
    _UNSCALED_MATRIX_KEYS,
    _allocate_rows,
    _deepep_peer_traffic_matrix,
    _scale_matrix,
    _scale_op_portion,
    _validate_eplb,
    _validate_overlap,
    declare_received_value,
    deepep_exchange_factor,
    deepep_group_volume,
    deepep_intra_volume,
    down_overlap_columns,
    overlap_row_cut,
    received_value_contract,
    resolve_exchange_options,
    scale_exchange_operator,
    scale_exchange_stages,
    split_deepep_combine_overlap,
    split_deepep_dispatch_overlap,
    split_peer_traffic_rows,
    standard_alltoall_volume,
)
from mindie_sim.prediction.implementations import append_audit, typed_value

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_REVISION = "175ef4d49c2072f9233619636f0309f4f88b4e2c"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-ascend",
        revision=_VLLM_ASCEND_REVISION,
        relative_path="vllm_ascend/ops/fused_moe/token_dispatcher.py",
        source_sha256="6126304014959e8c52d87284ccac078f7622a6d72c7a7fc8be25929dfe644cfb",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision=_VLLM_ASCEND_REVISION,
        relative_path="vllm_ascend/ops/fused_moe/moe_comm_method.py",
        source_sha256="b785ade12a81349d01d4806c95ab17d37537db9548b3aae492135c8f4318fe39",
    ),
)

DEFAULT_TOKENS = 1
DEFAULT_CONTEXT = 4096
DEFAULT_SEQUENCES = 16
_MATRIX_SHAPE_ALIGNMENT = 16


def expert_bank_segments(bank_rows, first: int):
    """Partition one destination bank in slot order at a complete row boundary.

    Both results keep the full bank length: a slot whose rows span the cut
    contributes to both segments, while an untouched slot stays zero.  This is
    the per-segment expert bank handed to the selected Grouped program; it is
    not a global expert ledger and never reorders the local weight slot.
    """
    remaining = int(first)
    main, tail = [], []
    for count in bank_rows:
        count = int(count)
        taken = min(count, max(0, remaining))
        main.append(taken)
        tail.append(count - taken)
        remaining -= taken
    return tuple(main), tuple(tail)


def _view_contract(source: Mapping, *, fallback_dtype: str, rows: int, width: int):
    """Validate the source value a row view aliases; no ancestry probing."""
    if not source:
        return None
    dtype = str(source.get("dtype") or fallback_dtype)
    shape = tuple(int(value) for value in source.get("shape") or (rows, width))
    if len(shape) != 2 or shape[1] != width:
        raise ModelingError(
            "deepep_activation_view_requires_matching_received_row_width:"
            f"{shape}!={(rows, width)}")
    if shape[0] != rows:
        raise ModelingError(
            "deepep_activation_view_requires_matching_received_rows:"
            f"{shape[0]}!={rows}")
    return {
        "producer": str(source.get("producer") or ""),
        "output": str(source.get("output") or "output"),
        "dtype": dtype,
        "shape": shape,
        "row_bytes": float(source.get("row_bytes")
                           or width * dtype_bytes(dtype)),
        "storage": str(source.get("storage") or ""),
        "alias_of": str(source.get("alias_of") or source.get("producer") or ""),
        "state_version": source.get("state_version"),
        "binding": str(source.get("binding") or "static"),
        "binding_name": str(source.get("binding_name") or ""),
    }


def _received_row_view(
    up, orig_meta, source, source_dtype, source_deps, dispatch_overlap_id,
    idx, physical_span, row_cut, full_shape, declared_active,
):
    physical_start, physical_stop = physical_span
    first, second = row_cut
    m, _, k = full_shape
    start = min(physical_start, declared_active)
    stop = min(physical_stop, declared_active)
    rows = max(0, stop - start)
    byte_offset = start * k * dtype_bytes(source_dtype)
    view_id = f"{up.op_id}.rows{start}_{stop}"
    view_deps = tuple(dict.fromkeys(
        (*source_deps, dispatch_overlap_id) if idx == 1 else source_deps))
    source_row_bytes = (source["row_bytes"] if source else float(k * dtype_bytes(source_dtype)))
    source_shape = source["shape"] if source else (m, k)
    view_meta = {
        "source_producer": (source["producer"] if source else ""),
        "source_output": (source["output"] if source else ""),
        "source_alias": (source["alias_of"] if source else ""),
        "source_state_version": (source["state_version"] if source else None),
        "source_storage_shape": tuple(source_shape),
        "source_storage": (source["storage"] if source else ""),
        "source_binding": (source["binding"] if source else "static"),
        "source_binding_name": (source["binding_name"] if source else ""),
        "named_input": (f"{source['producer']}.{source['output']}"
                        if source else ""),
        "row_start": start, "row_stop": stop,
        "physical_row_start": physical_start,
        "physical_row_stop": physical_stop,
        "full_rows": m, "full_active_rows": declared_active,
        "full_expected_rows": float(orig_meta.get("expected_rows", m)),
        "byte_offset": byte_offset,
        "element_offset": start * k,
        "source_storage_bytes": (int(math.prod(source_shape))
                                 * dtype_bytes(source_dtype)),
        "deepep_row_cut": (first, second),
        "deepep_chunk_row_start": physical_start,
        "deepep_chunk_row_stop": physical_stop,
        "deepep_chunk_rows": rows,
        "deepep_chunk_payload_bytes": float(rows) * float(source_row_bytes),
        "deepep_activation_view": "zero_materialization_received_rows",
        "output_format": source_dtype, "materialized": False,
        "output_layout": {
            "logical_shape": [rows, k],
            "storage_shape": list(source_shape),
            "offset": start * k,
            "strides": [k, 1]},
        "memory_accesses": ({
            "producer": (source["producer"] if source else ""),
            "output": (source["output"] if source else "output"),
            "direction": "alias", "position": "local",
            "storage": (source["storage"] if source else ""),
            "bytes": 0.0,
            "span": ((physical_start, physical_stop),),
            "last_use": ((up.op_id if idx == 0 else up.op_id + "_overlap")
                         + (".quantize" if orig_meta.get("activation_scale_bytes")
                            else "")),
            "source": (source["producer"] if source else ""),
            "purpose": "received_activation_rows",
        },),
    }
    view = vector_op(
        op_id=view_id, deps=view_deps, stage=up.stage, layer=up.layer,
        requests=up.requests, elements=0, ops_per_element=0,
        vector_ops_override=0, dtype=source_dtype, family="tensor_view",
        phase=up.phase, recipe="row_view", shape=(rows, k),
        read_bytes_override=0, write_bytes_override=0,
        metadata=tuple(view_meta.items()))
    view_meta = dict(view.metadata)
    append_audit(
        view_meta, kind="source_value_contract", op_id=view.op_id,
        boundary={"typed_outputs": (typed_value(
            producer=view_id, dtype=source_dtype, shape=(rows, k),
            storage_shape=tuple(source_shape),
            output=(source["output"] if source else "output"),
            role="activation",
            source=(f"{source['producer']}.{source['output']}"
                    if source else f"{view_id}:row_view"),
            materialized=False, write_bytes=0,
            storage=(source["storage"] if source else ""),
            alias_of=(source["alias_of"] if source else ""),
            state_version=(source["state_version"] if source else None),
            binding=(source["binding"] if source else "static"),
            binding_name=(source["binding_name"] if source else ""),
            contract="zero_materialization_received_view",
            layout={"shape": [rows, k], "strides": [k, 1],
                    "offset": start * k,
                    "storage_shape": list(source_shape)}),)},
        layer=up.layer, stage=up.stage)
    view = replace(view, metadata=tuple(view_meta.items()))
    return view, view_id, start, stop, source_shape


_MATRIX_SHAPE_RANK = 3


def _build_deepep_up_part(
    up, idx, shape, frac, *, view_id, start, stop, physical_start, physical_stop,
    first, second, m, n, k, declared_active, source_shape, activation_id,
):
    meta = dict(up.metadata)
    meta["expected_rows"] = float(shape[0])
    meta["deepep_row_cut"] = (first, second)
    meta["deepep_gate_row_start"] = start
    meta["deepep_gate_row_stop"] = stop
    meta["deepep_physical_row_start"] = physical_start
    meta["deepep_physical_row_stop"] = physical_stop
    meta["deepep_full_rows"] = m
    meta["deepep_full_active_rows"] = declared_active
    meta["deepep_activation_view"] = view_id
    meta["deepep_received_storage"] = tuple(source_shape)
    if "activation_scale_bytes" in meta:
        meta["activation_scale_bytes"] = (float(meta["activation_scale_bytes"]) / m) * shape[0]
    if "activation_scale_shape" in meta and meta["activation_scale_shape"]:
        meta["activation_scale_shape"] = (shape[0], *tuple(meta["activation_scale_shape"])[1:])
    if idx == 1:
        meta["deepep_overlap_branch"] = True
    weight_scale = 1.0
    realization = meta.get("routing_realization")
    if isinstance(realization, Mapping) and realization.get("basis") == "explicit_token_routing_integer":
        bank = tuple(int(value) for value in realization.get("local_expert_rows") or ())
        if bank and sum(bank) == m:
            seg_main, seg_tail = expert_bank_segments(bank, first)
            segment = seg_main if idx == 0 else seg_tail
            active_full = sum(1 for value in bank if value)
            active_seg = sum(1 for value in segment if value)
            weight_scale = (active_seg / float(active_full)) if active_full > 0 else 1.0
            meta["routing_realization"] = {
                **dict(realization), "local_expert_rows": segment,
                "rows": sum(segment), "active": active_seg}
            meta["matrix_group_rows"] = segment
            meta["matrix_groups"] = active_seg
            meta["deepep_expert_bank_rows"] = bank
            meta["deepep_segment_expert_bank_rows"] = segment
    else:
        # A chunk of shape[0] rows can touch at most min(active_full, shape[0])
        # local expert banks; clamp weight reads accordingly.
        active_full = float(
            meta.get("expected_active_experts")
            or meta.get("active_local_experts")
            or 0.0
        )
        if active_full > float(shape[0]) > 0.0:
            weight_scale = float(shape[0]) / active_full
    meta["deepep_weight_scale"] = float(weight_scale)
    # Row-splitting along M keeps full participating expert weight bytes;
    # cross-chunk weight reuse is handled strictly by L2 residency.
    for key in (
        "matrix_weight_bytes", "packed_weight_bytes", "weight_scale_bytes",
        "weight_bytes", "routed_weight_read_bytes",
    ):
        if key in meta and isinstance(meta[key], (int, float)):
            meta[key] = float(meta[key]) * weight_scale
    if "matrix_epilogue" in meta:
        ep_m = dict(meta["matrix_epilogue"])
        ep_m["row_scale_bytes"] = float(ep_m.get("row_scale_bytes", 0.0)) * frac
        ep_m["column_scale_bytes"] = float(ep_m.get("column_scale_bytes", 0.0)) * weight_scale
        ep_m["operations_per_output"] = (up.vector_ops * frac) / (shape[0] * shape[1])
        meta["matrix_epilogue"] = ep_m
    gate_input_format = str(meta.get("activation_format") or "int8")
    gate_read = shape[0] * k * dtype_bytes(gate_input_format) + float(meta.get("activation_scale_bytes", 0.0))
    gate_mte2 = float(meta.get("packed_weight_bytes", 0.0)) + gate_read
    gate_mte3 = shape[0] * n * dtype_bytes(str(meta.get("output_format") or "bf16"))
    part = replace(
        up, op_id=up.op_id if idx == 0 else up.op_id + "_overlap", recipe=up.recipe,
        deps=(view_id,), local_shape=shape,
        semantic_shape=shape if len(up.semantic_shape) == _MATRIX_SHAPE_RANK else up.semantic_shape,
        launch_shape=shape if len(up.launch_shape) == _MATRIX_SHAPE_RANK else up.launch_shape,
        padded_shape=tuple(math.ceil(v / _MATRIX_SHAPE_ALIGNMENT) * _MATRIX_SHAPE_ALIGNMENT for v in shape),
        flops=up.flops * frac, matrix_flops=tuple((dt, val * frac) for dt, val in up.matrix_flops),
        hbm_bytes=gate_mte2 + gate_mte3, vector_ops=up.vector_ops * frac,
        mte2_bytes=gate_mte2, mte3_bytes=gate_mte3, metadata=tuple(meta.items()),
    )
    part_meta = dict(part.metadata)
    out_dtype = str(part_meta.get("output_format") or "bf16")
    out_bytes = shape[0] * n * dtype_bytes(out_dtype)
    part_meta["memory_accesses"] = ({
        "producer": part.op_id, "output": part_meta.get("primary_output") or "output",
        "direction": "write", "position": "local", "storage": f"moe.act:{part.op_id}",
        "bytes": float(out_bytes), "span": ((0, shape[0]),),
        "last_use": str(activation_id or ""), "purpose": "gate_output_rows",
    },)
    append_audit(
        part_meta, kind="source_value_contract", op_id=part.op_id,
        boundary={"typed_outputs": (typed_value(
            producer=part.op_id, dtype=out_dtype, shape=(shape[0], n), storage_shape=(shape[0], n),
            output=part_meta.get("primary_output") or "output", role="activation",
            source=part.op_id + ":output_format", materialized=True, write_bytes=out_bytes),)},
        layer=up.layer, stage=up.stage)
    return replace(part, metadata=tuple(part_meta.items()))


def split_deepep_up_overlap(up, options, *, dispatch_overlap_id: str | None,
                            allow_split: bool = True, active_rows: int | None = None,
                            source_contract: Mapping | None = None,
                            row_cut: tuple[int, int] | None = None,
                            activation_id: str | None = None):
    """Split expert_up by the same integer rows the transport chunk uses."""
    overlap = float(options.get("overlap", 0.0))
    orig_meta = dict(up.metadata)
    if (not options.get("deepep") or overlap <= 0.0 or dispatch_overlap_id is None
            or len(up.local_shape) < 3):
        return (up,), up.op_id
    m, n, k = up.local_shape[-3:]
    if (not allow_split or m < 2 or "matrix_group_rows" in orig_meta
            or "moe_transport_kind" in orig_meta):
        gated = replace(up, deps=tuple(dict.fromkeys((*up.deps, dispatch_overlap_id))))
        return (gated,), gated.op_id
    declared_active = orig_meta.get("graph_active_query_rows")
    if declared_active is None:
        declared_active = active_rows
    if declared_active is None:
        declared_active = m
    declared_active = min(m, max(0, int(declared_active)))
    if declared_active < m or declared_active < 2:
        # A padded graph has physical rows that no active token reaches.  Do
        # not invent a second active interval for padding: keep one materialized
        # gate and make its conversion wait for both dispatch parts.
        gated = replace(up, deps=tuple(dict.fromkeys((*up.deps, dispatch_overlap_id))))
        gated = replace(gated, metadata=tuple({
            **dict(gated.metadata), "deepep_unsplit_padded": True,
            "deepep_full_active_rows": declared_active,
        }.items()))
        return (gated,), gated.op_id
    if row_cut is None:
        first, second = overlap_row_cut(m, overlap)
    else:
        first, second = int(row_cut[0]), int(row_cut[1])
        if first + second != m or first < 1 or second < 1:
            raise ModelingError(
                "deepep_activation_row_cut_must_match_received_rows:"
                f"{row_cut}!={(m, m)}")
    if first < 1 or second < 1:
        return (up,), up.op_id
    shapes = ((first, n, k), (second, n, k))
    fracs = (first / float(m), second / float(m))
    source = _view_contract(
        source_contract or {}, fallback_dtype=str(
            orig_meta.get("input_format") or up.dtype or "bf16"),
        rows=m, width=k)
    source_dtype = str(source["dtype"]) if source else str(
        orig_meta.get("input_format") or up.dtype or "bf16")
    source_deps = tuple(up.deps)
    result = []
    last_gate = up.op_id
    offset = 0
    for idx, (shape, frac) in enumerate(zip(shapes, fracs)):
        physical_start = offset
        physical_stop = physical_start + shape[0]
        view, view_id, start, stop, source_shape = _received_row_view(
            up, orig_meta, source, source_dtype, source_deps, dispatch_overlap_id,
            idx, (physical_start, physical_stop), (first, second), (m, n, k),
            declared_active,
        )
        part = _build_deepep_up_part(
            up, idx, shape, frac, view_id=view_id, start=start, stop=stop,
            physical_start=physical_start, physical_stop=physical_stop,
            first=first, second=second, m=m, n=n, k=k, declared_active=declared_active,
            source_shape=source_shape, activation_id=activation_id)
        result.extend((view, part))
        last_gate = part.op_id
        offset = physical_stop
    return tuple(result), last_gate


def split_deepep_down_overlap(down, options, *, has_combine: bool = True):
    """Split expert_down along output columns N so combine_overlap sends complete outputs."""
    overlap = float(options.get("overlap", 0.0))
    if (not has_combine or not options.get("deepep") or overlap <= 0.0
            or options.get("ep") == 1 or len(down.local_shape) < 3):
        return (down,), down.op_id, down.op_id
    m, n, k = down.local_shape[-3:]
    n1, n2 = down_overlap_columns(n, overlap)
    if n < 2 or n1 < 1 or n2 < 1:
        return (down,), down.op_id, down.op_id
    shapes = ((m, n1, k), (m, n2, k))
    fracs = (n1 / n, n2 / n)
    base_meta = dict(down.metadata)
    full_weight_bytes = float(base_meta.get("packed_weight_bytes", 0.0))
    full_act_read_bytes = max(0.0, float(down.mte2_bytes) - full_weight_bytes)
    parts = []
    for idx, (shape, frac) in enumerate(zip(shapes, fracs)):
        meta = dict(down.metadata)
        if idx == 1:
            meta.pop("input_conversion_subgraphs", None)
            meta["activation_producer"] = "explicit_quantization"
            meta["deepep_overlap_branch"] = True
        meta["deepep_column_cut"] = (n1, n2)
        meta["deepep_column_span"] = (0, n1) if idx == 0 else (n1, n)
        meta["deepep_weight_scale"] = float(frac)
        for key in (
            "matrix_weight_bytes", "packed_weight_bytes", "weight_scale_bytes",
            "weight_bytes", "routed_weight_read_bytes",
        ):
            if key in meta and isinstance(meta[key], (int, float)):
                meta[key] = float(meta[key]) * frac
        if "matrix_epilogue" in meta:
            ep_m = dict(meta["matrix_epilogue"])
            ep_m["column_scale_bytes"] = float(ep_m.get("column_scale_bytes", 0.0)) * frac
            meta["matrix_epilogue"] = ep_m
        # Column-splitting along N splits weights and outputs, while each chunk
        # reads the full activation rows (reuse is governed by L2 residency).
        part_mte2 = full_weight_bytes * frac + full_act_read_bytes
        part_mte3 = float(down.mte3_bytes) * frac
        parts.append(replace(
            down,
            op_id=down.op_id if idx == 0 else down.op_id + "_overlap",
            recipe=down.recipe,
            deps=(down.deps if idx == 0
                  else tuple(dict.fromkeys((*down.deps, down.op_id)))),
            local_shape=shape,
            semantic_shape=shape if len(down.semantic_shape) == 3 else down.semantic_shape,
            launch_shape=shape if len(down.launch_shape) == 3 else down.launch_shape,
            padded_shape=tuple(math.ceil(v / _MATRIX_SHAPE_ALIGNMENT) * _MATRIX_SHAPE_ALIGNMENT
                               for v in shape),
            flops=down.flops * frac,
            matrix_flops=tuple((dt, val * frac) for dt, val in down.matrix_flops),
            hbm_bytes=part_mte2 + part_mte3,
            vector_ops=down.vector_ops * frac,
            mte2_bytes=part_mte2,
            mte3_bytes=part_mte3,
            metadata=tuple(meta.items()),
        ))
    return (parts[0], parts[1]), parts[0].op_id, parts[1].op_id


def analyze_shared_expert_overlap(
    case, *, phase: str = "decode", tokens: int = DEFAULT_TOKENS,
    context: int = DEFAULT_CONTEXT,
    sequences: int = DEFAULT_SEQUENCES, scope: str = "body",
) -> dict[str, Any]:
    """Evaluate the causal engine with and without the shared-expert stream."""
    from mindie_sim.prediction.efficiency import analyze_network_efficiency
    from mindie_sim.prediction.structure import parse_case

    parsed = parse_case(case)

    def evaluate(enabled):
        metadata = {**dict(parsed.runtime.metadata),
                    "overlap_shared_expert": bool(enabled)}
        runtime = replace(
            parsed.runtime,
            multistream_overlap_shared_expert=bool(enabled),
            metadata=tuple(sorted(metadata.items())))
        report = analyze_network_efficiency(
            replace(parsed, runtime=runtime), phase=phase, tokens=tokens,
            context=context, sequences=sequences, scope=scope)
        return report

    baseline = evaluate(False)
    overlapped = evaluate(True)
    baseline_ms = float(baseline["graph_duration_ms"])
    overlapped_ms = float(overlapped["graph_duration_ms"])
    baseline_flops = float(baseline["total_flops"])
    overlapped_flops = float(overlapped["total_flops"])
    baseline_hbm = float(baseline["total_hbm_bytes"])
    overlapped_hbm = float(overlapped["total_hbm_bytes"])
    return {
        "schema": "mindie-sim.shared-expert-overlap/1",
        "phase": phase, "tokens": int(tokens), "context": int(context),
        "sequences": int(sequences), "scope": scope,
        "baseline_step_ms": baseline_ms,
        "overlapped_step_ms": overlapped_ms,
        "saved_ms": baseline_ms - overlapped_ms,
        "speedup_ratio": baseline_ms / max(overlapped_ms, 1e-12),
        "baseline_operator_sum_ms": float(baseline["operator_sum_ms"]),
        "overlapped_operator_sum_ms": float(overlapped["operator_sum_ms"]),
        "baseline_total_flops": baseline_flops,
        "overlapped_total_flops": overlapped_flops,
        "baseline_total_hbm_bytes": baseline_hbm,
        "overlapped_total_hbm_bytes": overlapped_hbm,
        "flops_conserved": math.isclose(
            baseline_flops, overlapped_flops, rel_tol=1e-12, abs_tol=0.0),
        "hbm_bytes_conserved": math.isclose(
            baseline_hbm, overlapped_hbm, rel_tol=1e-12, abs_tol=0.0),
    }


__all__ = [
    "deepep_group_volume", "deepep_intra_volume", "standard_alltoall_volume",
    "resolve_exchange_options", "deepep_exchange_factor",
    "scale_exchange_operator", "scale_exchange_stages",
    "analyze_shared_expert_overlap",
]
