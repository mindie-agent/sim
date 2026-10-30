"""Explicitly selected per-row INT8 MoE dispatch and BF16 return protocol.

[Role] Build the histogram, dynamic INT8 producer, split scale/payload exchanges
  and transposed BF16 return for the selected ALLTOALL transport branch.
[Invariants] The selector is an explicit physical branch, never a model name.
  Tokens are counted once per local assignment; an explicit TP caller already
  owns its local rows, otherwise ``min(tp, ep)`` partitions them once.
  Histogram entry bytes are never TP-shrunk. The
  return ledger transposes source row counts and then applies BF16 row width.
  Unsupported selectors or expert representations raise; no fallback is used.
[Timeouts & Locks] Bounded O(experts + source partitions) construction, no I/O.
[Side-Effect Policy] Returns immutable operators and metadata; no device work.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.ops import comm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes


_SELECTOR = "moe_transport"
_CONFIG = {
    "kind": "separate_quantized_payload",
    "payload_dtype": "int8",
    "scale_dtype": "fp32",
    "scale_granularity": "row",
    "histogram_dtype": "int64",
}
_EXPERT_ROW_FORMAT = "peer_traffic_row_format"


@dataclass(frozen=True)
class Transport:
    """Validated orthogonal representation choices for one physical transport."""

    kind: str
    payload_dtype: str
    scale_dtype: str
    scale_granularity: str
    histogram_dtype: str

    @property
    def payload_bytes(self):
        return dtype_bytes(self.payload_dtype)

    @property
    def return_bytes(self):
        return dtype_bytes("bf16")


def select(runtime, *, layer_scope="body"):
    """Return the explicit transport declaration or ``None`` when absent."""
    kernels = getattr(runtime, "kernels", None) or {}
    raw = kernels.get(_SELECTOR)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ModelingError("moe_transport_requires_mapping")
    unknown = set(raw) - set(_CONFIG) - {"layer_scopes"}
    missing = set(_CONFIG) - set(raw)
    if unknown:
        raise ModelingError("moe_transport_unknown_fields:" + ",".join(sorted(unknown)))
    if missing:
        raise ModelingError("moe_transport_missing_fields:" + ",".join(sorted(missing)))
    for key, expected in _CONFIG.items():
        allowed = {"int32", "int64"} if key == "histogram_dtype" else {expected}
        if raw[key] not in allowed:
            raise ModelingError(
                f"moe_transport_unsupported:{key}={raw[key]!r};expected={expected!r}")
    scopes = raw.get("layer_scopes")
    if scopes is not None:
        if (not isinstance(scopes, (list, tuple)) or not scopes
                or any(scope not in {"body", "draft"}
                       for scope in scopes)):
            raise ModelingError("moe_transport_layer_scopes_require_declared_model_scopes")
        if layer_scope not in scopes:
            return None
    return Transport(**{key: raw[key] for key in _CONFIG})


def validate_contract(model, plan):
    """Reject expert representations this initial row-scale branch cannot honor."""
    if model.activation_dtype != "bf16":
        raise ModelingError("moe_transport_initial_branch_requires_bf16_activations")
    quant = model.quant.for_category("routed_expert")
    weight_format = quant.format_for("routed_expert")
    # W8A8 and W4A8 both expose INT8 dynamic activations; the packed weight
    # differs but the communicated expert input does not.
    if weight_format not in {"int8", "w8a8", "int4", "w4a8"}:
        raise ModelingError(
            "moe_transport_requires_int8_activations:"
            f"{weight_format}")
    width = plan.expert_latent_size or model.hidden_size
    group = quant.activation_group_size or width
    scale_format = quant.activation_scale_format or "fp32"
    if group < width:
        raise ModelingError(
            "moe_transport_requires_row_scale_width:"
            f"group={group}<row={width}")
    if scale_format not in {"fp32", "float32"}:
        raise ModelingError(
            "moe_transport_requires_fp32_row_scales:"
            f"{scale_format}")


def source_partitions(parallel):
    """One outgoing assignment partition, independent of ETP expert copies."""
    return min(parallel.tp, parallel.ep)


def source_rows(tokens, top_k, partitions):
    """Logical assignment rows owned by one physical source rank."""
    if tokens < 0 or top_k < 1 or partitions < 1:
        raise ModelingError("moe_transport_source_rows_invalid")
    return tokens * top_k / partitions


def _ceil_rows(rows):
    return int(math.ceil(rows - 1e-12))


def _count_basis(realization):
    if realization is None:
        return "probability_moments"
    return getattr(realization, "basis", "declared_integer_counts")


def histogram(*, prefix, plan, stage, requests, rows, ep, deps,
              ids_producer, realization, mode, transport):
    """Typed local expert histogram, then one count allgather per source."""
    basis = _count_basis(realization)
    count_dtype = transport.histogram_dtype
    count_bytes = dtype_bytes(count_dtype)
    # Global expert totals do not identify any source's histogram. In the
    # supported EP1 explicit realization the source owns the complete count
    # vector; balanced cross-rank destination counts are separate provenance.
    counts = (tuple(realization.expert_rows)
              if basis == "declared_integer_counts" and ep == 1 else None)
    if counts is not None:
        if len(counts) != plan.num_experts or sum(counts) != rows:
            raise ModelingError(
                "moe_transport_explicit_counts_disagree_with_source_rows:"
                f"{sum(counts)}!={rows}")
    count_rows = sum(counts) if counts is not None else rows
    op = vector_op(
        op_id=prefix + ".moe_histogram", deps=(ids_producer,) if ids_producer else (deps,),
        stage=stage, layer=plan.index, requests=requests,
        elements=_ceil_rows(count_rows), ops_per_element=0,
        vector_ops_override=count_rows, dtype=count_dtype, family="histogram",
        phase=mode, recipe="moe_expert_count_histogram", shape=(plan.num_experts,),
        read_bytes_override=count_rows * 4,
        write_bytes_override=plan.num_experts * count_bytes,
        returns=("expert_counts",),
        metadata=(("histogram_dtype", count_dtype), ("topk_ids_producer", ids_producer),
                  ("input_format", "int32"), ("output_format", count_dtype),
                  ("source_assignments", count_rows), ("num_experts", plan.num_experts),
                  ("routing_basis", basis), ("counts", counts),
                  ("count_values", "declared_source" if counts is not None
                   else "not_declared"),
                  ("global_expert_counts", tuple(realization.expert_rows)
                   if basis == "forced_balanced_integer_counts" else None),
                  ("count_work", "one_increment_per_source_assignment"),
                  ("includes_zero_count_experts", True)),
    )
    ops = [op]
    gather = None
    if ep > 1:
        gather = replace(comm_op(
            op_id=prefix + ".moe_histogram_gather", deps=(op.op_id,), stage=stage,
            layer=plan.index, requests=requests, family="allgather", phase=mode,
            recipe="ring_allgather", link_bytes=(ep - 1) * plan.num_experts * count_bytes,
            shape=(ep, plan.num_experts),
            metadata=(("group_axis", "ep"), ("steps", ep - 1),
                      ("collective_id", prefix + ".moe_histogram_gather"),
                      ("histogram_dtype", count_dtype),
                      ("traffic_model", "local_expert_count_allgather"),
                      ("histogram_tp_partitioned", False)),
        ), dtype=count_dtype)
        ops.append(gather)
    return ops, gather.op_id if gather else None


def quantize(*, prefix, plan, stage, requests, pack, transport, rows, width,
             partitions, mode):
    """One dynamic per-row INT8 payload and FP32 scale producer after pack."""
    rows_int = _ceil_rows(rows)
    groups = rows  # one scale group covers each complete row
    vector = 4 * rows * width - groups + 3 * groups
    op = vector_op(
        op_id=prefix + ".moe_transport_quantize", deps=(pack.op_id,), stage=stage,
        layer=plan.index, requests=requests, elements=rows_int * width,
        ops_per_element=0, vector_ops_override=vector, dtype=pack.dtype,
        family="quantize", phase=mode, recipe="dynamic_quantize",
        shape=(rows_int, width),
        read_bytes_override=rows * width * dtype_bytes(pack.dtype),
        write_bytes_override=rows * width * transport.payload_bytes
                             + rows * dtype_bytes(transport.scale_dtype),
        returns=("activation_scales",),
        metadata=(("input_format", pack.dtype), ("output_format", transport.payload_dtype),
                  ("activation_group_size", width),
                  ("activation_scale_format", transport.scale_dtype),
                  ("activation_scale_shape", (rows_int,)),
                  ("activation_scale_bytes", rows * dtype_bytes(transport.scale_dtype)),
                  ("payload_dtype", transport.payload_dtype),
                  ("scale_dtype", transport.scale_dtype),
                  ("scale_granularity", transport.scale_granularity),
                  ("source_rows", rows), ("source_assignments", rows),
                  ("source_assignment_partitions", partitions),
                  ("expected_rows", rows),
                  ("payload_producer", pack.op_id),
                  ("quantization_scope", "moe_transport_expert_input"),
                  ("quantizer_count", 1),
                  ("formula", "dynamic_per_row_symmetric_int8_fp32")),
    )
    return op


def dispatch(*, prefix, plan, stage, requests, quantized, transport, rows, width, ep,
             partitions, mode, hist_gather=None):
    """Scale exchange followed by the INT8 payload exchange."""
    ops = []
    if ep <= 1:
        return ops, None, None
    common = (("group_axis", "ep"), ("steps", ep - 1),
              ("source_assignment_partitions", partitions),
              ("remote_fraction", (ep - 1) / ep),
              ("traffic_model", "uniform_peer_payload"),
              ("source_rows", rows),
              ("scale_dtype", transport.scale_dtype))
    scale = comm_op(
        op_id=prefix + ".moe_scale_dispatch",
        deps=tuple(dict.fromkeys((quantized.op_id,) + ((hist_gather,) if hist_gather else ()))),
        stage=stage, layer=plan.index, requests=requests,
        family="alltoall", phase=mode, recipe="expert_scale_dispatch",
        link_bytes=rows * dtype_bytes(transport.scale_dtype) * (ep - 1) / ep,
        shape=(_ceil_rows(rows),),
        metadata=common + (("collective_id", prefix + ".moe_scale_dispatch"),
                           ("payload_kind", "row_scale"),
                           ("payload_dtype", transport.scale_dtype),
                           ("row_bytes", dtype_bytes(transport.scale_dtype))),
    )
    ops.append(scale)
    payload = comm_op(
        op_id=prefix + ".moe_payload_dispatch",
        deps=tuple(dict.fromkeys((quantized.op_id, scale.op_id)
                                 + ((hist_gather,) if hist_gather else ()))),
        stage=stage, layer=plan.index, requests=requests,
        family="alltoall", phase=mode, recipe="expert_quantized_dispatch",
        link_bytes=rows * width * transport.payload_bytes * (ep - 1) / ep,
        shape=(_ceil_rows(rows), width),
        metadata=common + (("collective_id", prefix + ".moe_payload_dispatch"),
                           ("payload_kind", "quantized_rows"),
                           ("payload_dtype", transport.payload_dtype),
                           ("row_elements", width),
                           ("row_bytes", width * transport.payload_bytes),
                           (_EXPERT_ROW_FORMAT, {
                               "elements": width,
                               "forward_dtype": transport.payload_dtype,
                               "reverse_dtype": "bf16",
                           })),
    )
    ops.append(payload)
    return ops, scale.op_id, payload.op_id


def combine(*, prefix, plan, stage, requests, deps, transport, rows, width, ep,
            partitions, mode, payload_dispatch):
    """BF16 return with dispatch source rows transposed before dtype conversion."""
    if ep <= 1:
        return None
    return comm_op(
        op_id=prefix + ".expert_combine", deps=(deps,), stage=stage,
        layer=plan.index, requests=requests, family="alltoall", phase=mode,
        recipe="expert_combine",
        link_bytes=rows * width * transport.return_bytes * (ep - 1) / ep,
        shape=(_ceil_rows(rows), width),
        metadata=(("group_axis", "ep"), ("steps", ep - 1),
                  ("source_assignment_partitions", partitions),
                  ("traffic_model", "transpose_dispatch_peer_rows"),
                  ("collective_id", prefix + ".expert_combine"),
                  ("reverse_of_collective", payload_dispatch),
                  ("payload_kind", "bf16_return_rows"),
                  ("payload_dtype", "bf16"),
                  ("row_elements", width),
                  ("row_bytes", width * transport.return_bytes),
                  (_EXPERT_ROW_FORMAT, {
                      "elements": width,
                      "forward_dtype": transport.payload_dtype,
                      "reverse_dtype": "bf16",
                  })),
    )


__all__ = [
    "Transport", "select", "validate_contract", "source_partitions", "source_rows",
    "histogram", "quantize", "dispatch", "combine",
]
