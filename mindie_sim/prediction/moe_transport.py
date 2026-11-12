"""Explicitly selected MoE dispatch/combine transport formats and EPLB scaling.

[Role] Build the local expert histogram, dynamic producer, split/block scale and
  payload exchanges and the transposed BF16 return for the selected ALLTOALL
  transport branch. Also expose the declared INT8/FP8/MXFP8 row format and the
  optional EPLB peak-to-average load ratio.
[Invariants] The selector is an explicit physical branch, never a model name.
  Tokens are counted once per local assignment; an explicit TP caller already
  owns its local rows, otherwise ``min(tp, ep)`` partitions them once.
  Histogram entry bytes are never TP-shrunk. The return ledger transposes source
  row counts and then applies BF16 row width. MXFP8 is 32 E4M3 payload elements
  plus one E8M0 block-32 scale byte (33/32 B per element, including its scale).
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
_KIND = "separate_quantized_payload"
_MXFP8 = frozenset({"mxfp8", "mx_fp8", "mxfp8_e4m3"})
_FP8 = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2"})
# Packed FP4 payloads.  ``dtype_bytes('mxfp4')`` already folds the one E8M0
# block byte per 32 elements into the row width; a typed exchange must keep
# the 0.5 B payload and the separate scale byte as distinct ledgers.
_FP4 = frozenset({"fp4", "fp4_e2m1", "mxfp4", "mx_fp4", "mxfp4_e2m1"})
_DISPATCH_DTYPES = frozenset({"bf16", "fp16", "int8"}) | _FP8 | _MXFP8
_OPTION_KEYS = frozenset({
    "layer_scopes", "mode", "deepep_overlap_ratio", "eplb_peak_to_avg",
    "ep_ratio", "ranks_per_node",
})
_CONFIG = {
    "kind": _KIND,
    "payload_dtype": "int8",
    "scale_dtype": "fp32",
    "scale_granularity": "row",
    "histogram_dtype": "int64",
}
_EXPERT_ROW_FORMAT = "peer_traffic_row_format"


def _normalized_dtype(value, reason):
    text = str(value).strip().lower().replace("-", "_")
    if text not in _DISPATCH_DTYPES:
        raise ModelingError(f"{reason}:{value!r}")
    return text


def _payload_dtype(dispatch_dtype):
    return "fp8_e4m3" if (dispatch_dtype in _MXFP8 or dispatch_dtype == "fp8") else dispatch_dtype


def _eplb_ratio(raw, metadata):
    value = raw.get("eplb_peak_to_avg", raw.get("ep_ratio",
                  metadata.get("eplb_peak_to_avg", 1.0)))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError("moe_transport_eplb_peak_to_avg_invalid") from exc
    if not math.isfinite(result) or result < 1.0:
        raise ModelingError("moe_transport_eplb_peak_to_avg_must_be_at_least_one")
    return result


def _overlap_ratio(raw, metadata):
    value = raw.get("deepep_overlap_ratio", metadata.get("deepep_overlap_ratio", 0.0))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError("deepep_overlap_ratio_invalid") from exc
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ModelingError("deepep_overlap_ratio_must_be_in_0_1")
    return result


def _scopes(raw, layer_scope):
    scopes = raw.get("layer_scopes")
    if scopes is None:
        return True
    if (not isinstance(scopes, (list, tuple)) or not scopes
            or any(scope not in {"body", "draft"} for scope in scopes)):
        raise ModelingError("moe_transport_layer_scopes_require_declared_model_scopes")
    return layer_scope in scopes


def _mode(raw):
    mode = raw.get("mode", "alltoall")
    if mode not in {"alltoall", "deepep"}:
        raise ModelingError(f"moe_transport_mode_unsupported:{mode!r}")
    return mode


def _ranks_per_node(raw, metadata):
    value = raw.get("ranks_per_node", metadata.get("deepep_ranks_per_node", 0))
    if value in (None, 0, "0"):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError("moe_transport_ranks_per_node_invalid") from exc
    if result < 1:
        raise ModelingError("moe_transport_ranks_per_node_invalid")
    return result


@dataclass(frozen=True)
class Transport:
    """Validated orthogonal representation choices for one physical transport."""

    kind: str
    payload_dtype: str
    scale_dtype: str
    scale_granularity: str
    histogram_dtype: str
    dispatch_dtype: str = "int8"
    scale_group: int = 0
    eplb_peak_to_avg: float = 1.0
    deepep_mode: bool = False
    deepep_overlap_ratio: float = 0.0
    ranks_per_node: int = 0

    @property
    def payload_bytes(self):
        return dtype_bytes(self.payload_dtype)

    @property
    def return_bytes(self):
        return dtype_bytes("bf16")

    @property
    def quantized(self):
        return self.dispatch_dtype not in {"bf16", "fp16"}

    @property
    def mxfp8(self):
        return self.dispatch_dtype in _MXFP8

    @property
    def fp8(self):
        return self.dispatch_dtype in _FP8

    @property
    def inline_scales(self):
        """Block scales travel inside each payload row (MXFP8)."""
        return bool(self.scale_group and self.scale_granularity not in {"row", "per_row"})

    def scale_bytes(self, width):
        if not self.quantized:
            return 0.0
        if self.scale_group:
            return math.ceil(width / self.scale_group) * dtype_bytes(self.scale_dtype)
        return dtype_bytes(self.scale_dtype)

    def row_bytes(self, width):
        """Total wire bytes for one token row, including inline block scales."""
        return width * self.payload_bytes + (
            self.scale_bytes(width) if self.inline_scales else 0.0)

    def element_bytes(self, width):
        return self.row_bytes(width) / float(width) if width else 0.0

    def format(self, width, *, layer_index=None):
        """Width-specific declaration used by ``_transport_format``."""
        payload = width * self.payload_bytes
        scales = self.scale_bytes(width)
        return {
            "kind": self.kind,
            "dispatch_dtype": self.dispatch_dtype,
            "payload_dtype": self.payload_dtype,
            "op_dtype": self.payload_dtype,
            "quantized": self.quantized,
            "payload_bytes": float(payload),
            "scale_dtype": self.scale_dtype,
            "scale_group": int(self.scale_group),
            "scale_bytes": float(scales),
            "row_bytes": float(payload + scales),
            "element_bytes": float((payload + scales) / width) if width else 0.0,
            "combine_dtype": "bf16",
            "eplb_peak_to_avg": self.eplb_peak_to_avg,
            "deepep_mode": self.deepep_mode,
            "deepep_overlap_ratio": self.deepep_overlap_ratio,
            "width": int(width),
            "layer_index": (None if layer_index is None else int(layer_index)),
        }


def _parse(raw, metadata, layer_scope):
    """Return a validated transport or ``None`` for options-only declarations."""
    if not isinstance(raw, Mapping):
        raise ModelingError("moe_transport_requires_mapping")
    unknown = set(raw) - set(_CONFIG) - set(_OPTION_KEYS) - {"dispatch_dtype", "scale_group",
                                                             "scale_Group"}
    if unknown:
        raise ModelingError("moe_transport_unknown_fields:" + ",".join(sorted(unknown)))
    if not _scopes(raw, layer_scope):
        return None
    _mode(raw)
    _overlap_ratio(raw, metadata)
    _ranks_per_node(raw, metadata)
    _eplb_ratio(raw, metadata)
    representation = set(raw) & (set(_CONFIG) | {"dispatch_dtype", "scale_group", "scale_Group"})
    if not representation:
        # DeepEP/EPLB execution options can be declared without selecting a
        # quantized payload transport. Graph lowering consumes the options.
        return None
    kind = raw.get("kind", _KIND)
    if kind != _KIND:
        raise ModelingError(f"moe_transport_unsupported:kind={kind!r};expected={_KIND!r}")
    histogram = raw.get("histogram_dtype", "int64")
    if histogram not in {"int32", "int64"}:
        raise ModelingError(
            f"moe_transport_unsupported:histogram_dtype={histogram!r};expected='int64'")
    declared = raw.get("dispatch_dtype")
    payload = raw.get("payload_dtype")
    if declared is not None:
        dispatch = _normalized_dtype(declared, "moe_transport_unsupported:dispatch_dtype")
        if payload is not None:
            wire = _normalized_dtype(payload, "moe_transport_unsupported:payload_dtype")
            expected = _payload_dtype(dispatch)
            if wire != expected and not (wire in _FP8 and expected in _FP8):
                raise ModelingError(
                    "moe_transport_dispatch_payload_conflict:"
                    f"{dispatch}!={payload}")
    else:
        if payload is None:
            dispatch = "int8"
        else:
            normalized = str(payload).strip().lower().replace("-", "_")
            if normalized not in ({"int8"} | _FP8 | _MXFP8):
                raise ModelingError(
                    "moe_transport_unsupported:payload_dtype="
                    f"{payload!r};expected='int8'")
            dispatch = normalized
    quantized = dispatch not in {"bf16", "fp16"}
    wire_dtype = _payload_dtype(dispatch)
    group_raw = raw.get("scale_Group", raw.get("scale_group"))
    if group_raw is not None:
        try:
            group = int(group_raw)
        except (TypeError, ValueError) as exc:
            raise ModelingError("moe_transport_scale_group_invalid") from exc
        if group < 1:
            raise ModelingError("moe_transport_scale_group_invalid")
    elif dispatch in _MXFP8:
        group = 32
    else:
        group = 0
    if not quantized:
        if group:
            raise ModelingError("moe_transport_unquantized_payload_has_no_scales")
        scale_dtype, granularity = "", "none"
    elif dispatch in _MXFP8:
        scale_dtype = str(raw.get("scale_dtype", "fp8_e8m0")).strip().lower()
        granularity = str(raw.get("scale_granularity", "block")).strip().lower()
        if scale_dtype not in {"fp8_e8m0", "e8m0"}:
            raise ModelingError(
                f"moe_transport_unsupported:scale_dtype={scale_dtype!r};"
                "expected='fp8_e8m0'")
        if granularity not in {"block", "block32", "group"}:
            raise ModelingError("moe_transport_mxfp8_requires_block_scale_granularity")
    else:
        scale_dtype = str(raw.get("scale_dtype", "fp32")).strip().lower()
        granularity = str(raw.get("scale_granularity", "row")).strip().lower()
        if scale_dtype not in {"fp32", "float32"}:
            raise ModelingError(
                f"moe_transport_unsupported:scale_dtype={scale_dtype!r};"
                "expected='fp32'")
        if granularity not in {"row", "per_row"}:
            raise ModelingError("moe_transport_int8_requires_row_scale_granularity")
    return Transport(
        kind=kind, payload_dtype=wire_dtype, scale_dtype=scale_dtype,
        scale_granularity=granularity, histogram_dtype=histogram,
        dispatch_dtype=dispatch, scale_group=group,
        eplb_peak_to_avg=_eplb_ratio(raw, metadata),
        deepep_mode=_mode(raw) == "deepep",
        deepep_overlap_ratio=_overlap_ratio(raw, metadata),
        ranks_per_node=_ranks_per_node(raw, metadata),
    )


def default_dispatch_spec(model, plan=None, width=None):
    """Resolve the vllm-ascend dispatch wire format from the routed-expert quant spec."""
    from mindie_sim.catalog.ops import activation_scales, compute_dtype_for

    scope = getattr(plan, "scope", "body") if plan is not None else "body"
    latent = int(
        width
        or (getattr(plan, "expert_latent_size", 0) if plan is not None else 0)
        or model.hidden_size
    )
    combine_dtype = model.activation_dtype
    combine_row_bytes = float(latent * dtype_bytes(combine_dtype))
    if scope in model.quant.exclusions:
        return {
            "dispatch_dtype": combine_dtype,
            "payload_dtype": combine_dtype,
            "combine_dtype": combine_dtype,
            "quantized": False,
            "scale_dtype": "",
            "scale_group": 0,
            "payload_bytes_per_row": combine_row_bytes,
            "scale_bytes_per_row": 0.0,
            "dispatch_row_bytes": combine_row_bytes,
            "combine_row_bytes": combine_row_bytes,
            "element_bytes": float(dtype_bytes(combine_dtype)),
        }
    quant = model.quant.for_category("routed_expert")
    _, act = compute_dtype_for(quant, "routed_expert")
    if act in _FP4:
        # MXFP4 is a 0.5 B E2M1 payload plus one E8M0 byte per 32-element
        # block.  dtype_bytes('mxfp4') already includes that scale byte, so
        # the typed exchange must split it back out instead of publishing one
        # mixed payload row with no separate scale ledger.
        group = int(getattr(quant, "activation_group_size", 0)
                    or getattr(quant, "expert_block", 0) or 32)
        scale_format = str(getattr(quant, "activation_scale_format", "")
                           or getattr(quant, "scale_format", ""))
        pack = int(getattr(quant, "activation_scale_pack", 0) or 1)
        groups = (latent + group - 1) // group
        scale_bytes_per_row = (
            float(((groups + pack - 1) // pack) * pack * dtype_bytes(scale_format))
            if scale_format else 0.0)
        payload_bytes_per_row = float(latent * dtype_bytes("fp4_e2m1"))
    else:
        group, scale_format, _, scale_bytes_per_row = activation_scales(
            quant, act, 1, latent)
        payload_bytes_per_row = float(latent * dtype_bytes(act))
    quantized = act not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}
    if act in _FP8 and group == 32 and scale_format in {"fp8_e8m0", "e8m0"}:
        dispatch_dtype = "mxfp8"
    else:
        dispatch_dtype = act
    dispatch_row_bytes = payload_bytes_per_row + (float(scale_bytes_per_row) if quantized else 0.0)
    return {
        "dispatch_dtype": dispatch_dtype,
        "payload_dtype": act,
        "combine_dtype": combine_dtype,
        "quantized": quantized,
        "scale_dtype": scale_format if quantized else "",
        "scale_group": int(group) if quantized else 0,
        "payload_bytes_per_row": payload_bytes_per_row,
        "scale_bytes_per_row": float(scale_bytes_per_row) if quantized else 0.0,
        "dispatch_row_bytes": dispatch_row_bytes,
        "combine_row_bytes": combine_row_bytes,
        "element_bytes": dispatch_row_bytes / float(latent) if latent > 0 else float(dtype_bytes(act)),
    }


def default_exchange_contract(model, runtime, plan, transport, latent, *,
                              ep=1, prefix="", stage=0, requests=(), pack=None,
                              tokens=0, partitions=1, mode="decode"):
    """Return (dispatch_row_bytes, combine_row_bytes, dispatch_meta, combine_meta, pack_quant_op)."""
    default_row = float(latent * dtype_bytes(model.activation_dtype))
    follow = (
        transport is None and ep > 1
        and dict(getattr(runtime, "metadata", ()) or ()).get("moe_dispatch_quantization")
        == "routed_expert"
    )
    if not follow:
        return default_row, default_row, (), (), None
    spec = default_dispatch_spec(model, plan=plan, width=latent)
    d_row, c_row = float(spec["dispatch_row_bytes"]), float(spec["combine_row_bytes"])
    if not spec["quantized"]:
        return d_row, c_row, (), (), None
    row_fmt = (("peer_traffic_row_format", {"forward_bytes": int(round(d_row)), "reverse_bytes": int(round(c_row))}),)
    d_meta = (
        ("dispatch_dtype", spec["dispatch_dtype"]), ("payload_dtype", spec["payload_dtype"]),
        ("scale_dtype", spec["scale_dtype"]), ("scale_group", spec["scale_group"]),
        ("row_bytes", d_row),
    ) + row_fmt
    c_meta = (
        ("dispatch_dtype", spec["dispatch_dtype"]), ("combine_dtype", spec["combine_dtype"]),
        ("payload_dtype", spec["combine_dtype"]), ("row_bytes", c_row),
    ) + row_fmt
    pack_quant = None
    if pack is not None and tokens > 0:
        rows = source_rows(tokens, plan.top_k, partitions)
        rows_int = _ceil_rows(rows)
        group = spec["scale_group"] or latent
        groups = max(1, math.ceil(latent / group))
        scale_bytes = rows * spec["scale_bytes_per_row"]
        scale_shape = (rows_int, groups) if groups > 1 else (rows_int,)
        pack_quant = vector_op(
            op_id=prefix + ".pack.quantize", deps=(pack.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=rows_int * latent,
            ops_per_element=0, vector_ops_override=4 * rows * latent + 2 * rows * groups,
            dtype=pack.dtype, family="quantize", phase=mode, recipe="dynamic_quantize",
            shape=(rows_int, latent),
            read_bytes_override=rows * latent * dtype_bytes(pack.dtype),
            write_bytes_override=rows * d_row, returns=("activation_scales",),
            metadata=(
                ("input_format", pack.dtype), ("output_format", spec["payload_dtype"]),
                ("activation_group_size", group), ("activation_scale_format", spec["scale_dtype"]),
                ("activation_scale_shape", scale_shape), ("activation_scale_bytes", scale_bytes),
                ("source_rows", rows), ("quantization_scope", "moe_dispatch_expert_input"),
            ),
        )
    return d_row, c_row, d_meta, c_meta, pack_quant


def select(runtime, *, layer_scope="body"):
    """Return the explicit transport declaration or ``None`` when absent."""
    kernels = getattr(runtime, "kernels", None) or {}
    raw = kernels.get(_SELECTOR)
    if raw is None:
        return None
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    return _parse(raw, metadata, layer_scope)


def _transport_format(case, layer, width):
    """Resolve the declared dispatch format for one layer and hidden width.

    ``case`` may be a parsed :class:`Case` or its mapping form. MXFP8 defaults
    are materialized here so callers see the exact wire ledger: 32 E4M3 payload
    bytes plus one E8M0 block-scale byte per 32 elements.
    """
    if isinstance(case, Mapping):
        from mindie_sim.prediction.structure import parse_case
        parsed = parse_case(case)
    else:
        parsed = case
    runtime = parsed.runtime
    raw = dict((getattr(runtime, "kernels", None) or {}).get(_SELECTOR) or {})
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    layer_scope = getattr(layer, "scope", "body")
    eplb = _eplb_ratio(raw, metadata)
    dispatch = raw.get("dispatch_dtype")
    if dispatch is None:
        # An options-only block intentionally leaves the ordinary BF16 path.
        dispatch = "bf16" if not (set(raw) & set(_CONFIG)) else raw.get("payload_dtype", "int8")
    dispatch = _normalized_dtype(dispatch, "moe_transport_unsupported:dispatch_dtype")
    if not _scopes(raw, layer_scope):
        dispatch = "bf16"
    group = raw.get("scale_Group", raw.get("scale_group"))
    if dispatch in _MXFP8 and group is None:
        group = 32
    scale_dtype = raw.get("scale_dtype", "fp8_e8m0" if dispatch in _MXFP8 else "fp32")
    granularity = raw.get("scale_granularity",
                          "block" if dispatch in _MXFP8 else "row")
    if group is None:
        group = 0
    else:
        try:
            group_value = int(group)
        except (TypeError, ValueError) as exc:
            raise ModelingError("moe_transport_scale_group_invalid") from exc
        if float(group) != float(group_value) or group_value < 1:
            raise ModelingError("moe_transport_scale_group_invalid")
        group = group_value
    transport = Transport(
        kind=_KIND, payload_dtype=_payload_dtype(dispatch),
        scale_dtype=str(scale_dtype).strip().lower(),
        scale_granularity=str(granularity).strip().lower(),
        histogram_dtype=str(raw.get("histogram_dtype", "int64")),
        dispatch_dtype=dispatch, scale_group=int(group or 0),
        eplb_peak_to_avg=eplb,
        deepep_mode=_mode(raw) == "deepep",
        deepep_overlap_ratio=_overlap_ratio(raw, metadata),
        ranks_per_node=_ranks_per_node(raw, metadata),
    )
    return transport.format(width, layer_index=getattr(layer, "index", None))


def validate_contract(model, plan, transport=None):
    """Reject expert representations this selected transport cannot honor."""
    quant = model.quant.for_category("routed_expert")
    weight_format = quant.format_for("routed_expert")
    width = plan.expert_latent_size or model.hidden_size
    if transport is not None and transport.mxfp8:
        if weight_format not in {"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8",
                                 "mx_fp8", "mxfp8_e4m3", "mxfp4"}:
            raise ModelingError(
                "moe_transport_mxfp8_requires_fp8_expert_weights:"
                f"{weight_format}")
        group = quant.activation_group_size or transport.scale_group
        scale_format = quant.activation_scale_format or "fp8_e8m0"
        if group not in (0, transport.scale_group):
            raise ModelingError(
                "moe_transport_mxfp8_requires_block32_activation:"
                f"group={group}")
        if scale_format not in {"fp8_e8m0", "e8m0"}:
            raise ModelingError(
                "moe_transport_mxfp8_requires_e8m0_scales:"
                f"{scale_format}")
        return
    if transport is not None and transport.fp8:
        if weight_format not in {"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8",
                                 "mx_fp8", "mxfp8_e4m3", "mxfp4"}:
            raise ModelingError(
                "moe_transport_fp8_requires_fp8_expert_weights:"
                f"{weight_format}")
        return
    if transport is not None and not transport.quantized:
        return
    if model.activation_dtype != "bf16":
        raise ModelingError("moe_transport_initial_branch_requires_bf16_activations")
    # W8A8 and W4A8 both expose INT8 dynamic activations; the packed weight
    # differs but the communicated expert input does not.
    if weight_format not in {"int8", "w8a8", "int4", "w4a8"}:
        raise ModelingError(
            "moe_transport_requires_int8_activations:"
            f"{weight_format}")
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
    """One dynamic payload producer after pack: row INT8 or block MXFP8."""
    rows_int = _ceil_rows(rows)
    if transport.quantized:
        group = transport.scale_group or width
        groups = max(1, math.ceil(width / group))
        groups_total = rows * groups
        vector = 4 * rows * width - groups_total + 3 * groups_total
        scale_bytes = rows * transport.scale_bytes(width)
        scale_shape = (rows_int, groups) if transport.inline_scales else (rows_int,)
        scale_format = transport.scale_dtype
        group_size = group
        write = rows * width * transport.payload_bytes + scale_bytes
    else:
        vector = rows * width
        scale_bytes = 0.0
        scale_shape = ()
        scale_format = ""
        group_size = width
        groups = 0
        write = rows * width * transport.payload_bytes
    op = vector_op(
        op_id=prefix + ".moe_transport_quantize", deps=(pack.op_id,), stage=stage,
        layer=plan.index, requests=requests, elements=rows_int * width,
        ops_per_element=0, vector_ops_override=vector, dtype=pack.dtype,
        family="quantize", phase=mode, recipe="dynamic_quantize",
        shape=(rows_int, width),
        read_bytes_override=rows * width * dtype_bytes(pack.dtype),
        write_bytes_override=write,
        returns=("activation_scales",),
        metadata=(("input_format", pack.dtype), ("output_format", transport.payload_dtype),
                  ("activation_group_size", group_size),
                  ("activation_scale_format", scale_format),
                  ("activation_scale_shape", scale_shape),
                  ("activation_scale_bytes", scale_bytes),
                  ("payload_dtype", transport.payload_dtype),
                  ("dispatch_dtype", transport.dispatch_dtype),
                  ("scale_dtype", transport.scale_dtype),
                  ("scale_granularity", transport.scale_granularity),
                  ("scale_group", group_size if transport.quantized else 0),
                  ("source_rows", rows), ("source_assignments", rows),
                  ("source_assignment_partitions", partitions),
                  ("expected_rows", rows),
                  ("payload_producer", pack.op_id),
                  ("quantization_scope", "moe_transport_expert_input"),
                  ("quantizer_count", 1),
                  ("eplb_peak_to_avg", transport.eplb_peak_to_avg),
                  ("formula", ("dynamic_per_row_symmetric_int8_fp32"
                               if not transport.inline_scales
                               else "dynamic_block32_e4m3_e8m0"))),
    )
    return op


def dispatch(*, prefix, plan, stage, requests, quantized, transport, rows, width, ep,
             partitions, mode, hist_gather=None):
    """Block/row payload exchange, with row scales split out when declared."""
    ops = []
    if ep <= 1:
        return ops, None, None
    common = (("group_axis", "ep"), ("steps", ep - 1),
              ("source_assignment_partitions", partitions),
              ("remote_fraction", (ep - 1) / ep),
              ("traffic_model", "uniform_peer_payload"),
              ("source_rows", rows),
              ("scale_dtype", transport.scale_dtype),
              ("eplb_peak_to_avg", transport.eplb_peak_to_avg),
              ("dispatch_dtype", transport.dispatch_dtype))
    scale = None
    if transport.quantized and not transport.inline_scales:
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
    payload_row_bytes = transport.row_bytes(width)
    if transport.inline_scales:
        row_format = {"forward_bytes": int(payload_row_bytes),
                      "reverse_bytes": int(width * transport.return_bytes)}
    else:
        row_format = {"elements": width,
                      "forward_dtype": transport.payload_dtype,
                      "reverse_dtype": "bf16"}
    payload = comm_op(
        op_id=prefix + ".moe_payload_dispatch",
        deps=tuple(dict.fromkeys((quantized.op_id,)
                                 + ((scale.op_id,) if scale else ())
                                 + ((hist_gather,) if hist_gather else ()))),
        stage=stage, layer=plan.index, requests=requests,
        family="alltoall", phase=mode, recipe="expert_quantized_dispatch",
        link_bytes=rows * payload_row_bytes * (ep - 1) / ep,
        shape=(_ceil_rows(rows), width),
        metadata=common + (("collective_id", prefix + ".moe_payload_dispatch"),
                           ("payload_kind", "quantized_rows"),
                           ("payload_dtype", transport.payload_dtype),
                           ("row_elements", width),
                           ("row_bytes", payload_row_bytes),
                           ("payload_bytes", width * transport.payload_bytes),
                           ("inline_scale_bytes",
                            transport.scale_bytes(width) if transport.inline_scales else 0.0),
                           ("scale_group", transport.scale_group),
                           (_EXPERT_ROW_FORMAT, row_format)),
    )
    ops.append(payload)
    return ops, (scale.op_id if scale else None), payload.op_id


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
                  ("combine_dtype", "bf16"),
                  ("row_elements", width),
                  ("row_bytes", width * transport.return_bytes),
                  ("dispatch_dtype", transport.dispatch_dtype),
                  ("eplb_peak_to_avg", transport.eplb_peak_to_avg),
                  (_EXPERT_ROW_FORMAT, (
                      {"forward_bytes": int(transport.row_bytes(width)),
                       "reverse_bytes": int(width * transport.return_bytes)}
                      if transport.inline_scales else
                      {"elements": width,
                       "forward_dtype": transport.payload_dtype,
                       "reverse_dtype": "bf16"}))),
    )


__all__ = [
    "Transport", "default_dispatch_spec", "select", "_transport_format", "validate_contract",
    "source_partitions", "source_rows", "histogram", "quantize", "dispatch", "combine",
]
