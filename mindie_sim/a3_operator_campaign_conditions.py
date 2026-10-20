"""Canonical operator condition extraction and lineage for the A3 campaign.

[Role]
  Classify actual OpSpecs into primitive/fused/reference execution variants,
  preserve precision/layout/quantization/feature/state facts, normalize layer
  templates and deduplicate equivalent physical conditions while retaining all
  origin lineage.

[Invariants]
  * Primitive, fused and reference-decomposition variants stay separate.
  * Model/layer origin never enters physical condition identity.
  * Pending/unsupported recipes retain reason and required backend.
  * Layer templates are normalized with a single-escaped regex.
  * No source SHA gates admission.

[Timeouts & Locks]
  Pure bounded records/arithmetic; no I/O, locks or devices.

[Side-Effect Policy]
  No filesystem/network/device mutation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError, RuntimeSpec, align_up
from .a3_operator_campaign_primitive import _actual_grouped_geometry, _precision

COMM_FAMILIES = {
    "allreduce", "allgather", "ep_dispatch", "ep_combine", "pp_p2p",
    "cache_transfer", "engram_broadcast",
}
PLAIN_GEMM_FAMILIES = {
    "gemm", "qkv_projection", "o_projection", "gate_up_projection",
    "down_projection", "mla_q_a", "mla_q_b", "mla_kv_a", "mla_kv_b",
    "mla_q_absorb", "shared_expert_gate_up", "shared_expert_down", "logits",
    "linear_q_projection", "linear_k_projection", "linear_v_projection",
    "linear_o_projection",
}
ATTENTION_FAMILIES = {"attention_qk", "attention_pv"}
SOFTMAX_FAMILIES = {"attention_softmax", "router_softmax", "softmax"}
NORM_FAMILIES = {"rmsnorm", "qk_norm"}
SWIGLU_FAMILIES = {"swiglu"}
ELEMENTWISE_FAMILIES = {"elementwise"}
CACHE_FAMILIES = {"cache_store"}
MOE_GROUPED_FAMILIES = {"moe_gate_up", "moe_down"}
EXCLUDED_FAMILIES = COMM_FAMILIES | {"barrier"}
# Boundary candidates are only used when the lowering padding actually equals
# align_up(launch, value); they are not global hardware assertions.
TILE_CANDIDATES = (8, 16, 32, 64, 128, 256)
LADDER_FACTORS = (0.25, 0.5, 2.0, 4.0)
PENDING_BACKENDS = {
    "moe_gate_up": "native grouped expert GEMM (e.g. torch_npu.npu_grouped_matmul / fused MoE)",
    "moe_down": "native grouped expert GEMM (e.g. torch_npu.npu_grouped_matmul / fused MoE)",
    "recurrent_state": "native KDA/GDN recurrent-state kernel",
    "linear_conv": "native causal convolution kernel",
    "linear_gate": "native linear-attention gate kernel",
    "linear_output_gate": "native gated output kernel",
    "attention_output_gate": "native attention output gate kernel",
    "qk_gate": "native QK gate kernel",
    "rope": "native rotary embedding kernel (torch_npu.npu_rope or equivalent)",
    "quantize": "native dynamic quantize kernel",
    "compress_pool": "native compressor pooling kernel",
    "engram_apply": "native Engram table gather/apply kernel",
    "engram_gather": "native Engram device gather kernel",
    "engram_ids": "native Engram node exchange/metadata kernel",
    "engram_query": "native Engram query kernel",
    "candidate_blocks": "native candidate-block selection kernel",
    "hash_router": "native hash router kernel",
    "topk_selection": "native top-k selection kernel",
    "sampling": "native sampling kernel",
    "embedding": "embedding gather worker with a declared real vocabulary table",
    "mhc": "native mHC sinkhorn/mix kernel",
    "mla_kv_b": "native MLA absorbed-KV expansion kernel",
    "mla_q_absorb": "native MLA Q absorption kernel",
    "dspark": "native DSpark proposal/confidence kernel",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(prefix: str, value: Any, *, length: int = 20) -> str:
    return prefix + hashlib.sha1(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _as_int(value: Any, reason: str, minimum: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise A3InputError(reason) from exc
    if result < minimum or float(value) != float(result):
        raise A3InputError(reason)
    return result


def _as_positive(value: Any, reason: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise A3InputError(reason) from exc
    if not math.isfinite(result) or result <= 0:
        raise A3InputError(reason)
    return result


def _tuple_ints(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _bytes_for(dtype: str) -> float:
    try:
        return float(dtype_bytes(dtype))
    except Exception:
        return 2.0


REFERENCE_TOPOLOGIES: dict[str, tuple[dict[str, Any], ...]] = {
    "deepseek-v4-10t": ({"name": "reference", "tp": 16, "ep": 16, "pp": 1, "dp": 1, "devices": 16},),
    "deepseek-v4-flash-0731": ({"name": "reference", "tp": 8, "ep": 8, "pp": 1, "dp": 1, "devices": 8},),
    "deepseek-v4-pro-0813": ({"name": "reference", "tp": 16, "ep": 16, "pp": 1, "dp": 1, "devices": 16},),
    "deepseek-v4.1-flash": ({"name": "reference", "tp": 16, "ep": 16, "pp": 2, "dp": 1, "devices": 32},),
    "glm-5.3": ({"name": "reference", "tp": 16, "ep": 16, "pp": 2, "dp": 1, "devices": 32},),
    "kimi-k3": ({"name": "reference", "tp": 16, "ep": 16, "pp": 3, "dp": 1, "devices": 48},),
    "kimi-k3-dspark": ({"name": "reference", "tp": 1, "ep": 1, "pp": 1, "dp": 1, "devices": 8},),
    "qwen3.8-2.4t-a95b": ({"name": "reference", "tp": 32, "ep": 32, "pp": 4, "dp": 1, "devices": 128},),
    "qwen3.8-27b": ({"name": "reference", "tp": 8, "ep": 1, "pp": 1, "dp": 1, "devices": 8},),
}
ALTERNATE_TOPOLOGIES: dict[str, tuple[dict[str, Any], ...]] = {
    "deepseek-v4-10t": ({"name": "alt_tp8", "tp": 8, "ep": 8, "pp": 1, "dp": 1, "devices": 16},),
    "deepseek-v4-flash-0731": ({"name": "alt_tp4", "tp": 4, "ep": 4, "pp": 1, "dp": 1, "devices": 8},),
    "deepseek-v4-pro-0813": ({"name": "alt_tp8", "tp": 8, "ep": 8, "pp": 1, "dp": 1, "devices": 16},),
    "deepseek-v4.1-flash": ({"name": "alt_tp8", "tp": 8, "ep": 8, "pp": 2, "dp": 1, "devices": 32},),
    "glm-5.3": ({"name": "alt_tp8", "tp": 8, "ep": 8, "pp": 2, "dp": 1, "devices": 32},),
    "kimi-k3": ({"name": "alt_tp8", "tp": 8, "ep": 8, "pp": 3, "dp": 1, "devices": 48},),
    "kimi-k3-dspark": ({"name": "alt_dp2", "tp": 1, "ep": 1, "pp": 1, "dp": 2, "devices": 8},),
    "qwen3.8-2.4t-a95b": ({"name": "alt_tp16", "tp": 16, "ep": 16, "pp": 4, "dp": 1, "devices": 128},),
    "qwen3.8-27b": ({"name": "alt_tp4", "tp": 4, "ep": 1, "pp": 1, "dp": 1, "devices": 8},),
}
DEFAULT_PREFILL_TOKENS = (1, 32, 128, 512, 4096, 16384)
DEFAULT_DECODE_CASES = (
    {"batch": 1, "context": 1024},
    {"batch": 8, "context": 8192},
    {"batch": 32, "context": 32768},
    {"batch": 128, "context": 1024},
)
DEFAULT_PROTOCOL = {
    "timer": "device_event",
    "timing_scope": "device_event_warm_operator",
    "warmup": 10,
    "warmup_scope": "before_each_trial",
    "iterations": 30,
    "trials": 2,
    "preallocated": True,
    "cache_policy": "same_preallocated_buffers_reused_no_cache_flush",
    "seed": 20261021,
    "source_class": "measured",
    "requested_accumulation_dtype": "backend_default",
}
DEFAULT_MACHINE = {
    "name": "declared-a3-operator-campaign",
    "arch_family": "A3",
    "dies_per_package": 2,
    "num_devices": 128,
    "ranks_per_node": 8,
    "device": {
        "hbm_gib": 64,
        "cube_flops_per_s": 2.70336e14,
        "vector_ops_per_s": 1.6896e13,
        "hbm_bw_bytes_per_s": 1.2e12,
        "mte_bw_bytes_per_s": 2.4e12,
        "launch_us": 20.0,
    },
    "compute_peaks": {"bf16": 2.70336e14, "fp8_e4m3": 2.70336e14},
    "links": [{"name": "d2d", "channel": "d2d_intra_node", "capacity_gbps": 196,
               "latency_us": 2.5, "members": list(range(128))}],
}


def _shape_map(op: OpSpec, kind: str) -> dict[str, int]:
    launch = _tuple_ints(op.launch_shape)
    semantic = _tuple_ints(op.semantic_shape)
    if kind == "gemm" and len(launch) >= 3:
        return {"m": launch[-3], "n": launch[-2], "k": launch[-1]}
    if kind == "attention" and len(launch) >= 4:
        return {"q": launch[0], "kv": launch[1], "heads": launch[2], "dim": launch[3]}
    if kind == "attention_softmax" and len(semantic) >= 3:
        return {"q": semantic[0], "heads": semantic[1], "kv": semantic[2]}
    if kind == "router_softmax" and len(semantic) >= 2:
        return {"tokens": semantic[0], "experts": semantic[1]}
    if kind == "moe_grouped" and len(semantic) >= 3:
        routed = 0
        try:
            routed = int(dict(op.metadata).get("routed_tokens", 0) or 0)
        except (TypeError, ValueError):
            routed = 0
        return {"experts": semantic[0], "n": semantic[1], "k": semantic[2],
                "routed_tokens": max(1, routed)}
    if kind == "cache" and len(semantic) >= 2:
        return {"rows": semantic[0], "row_bytes": semantic[1]}
    if len(semantic) == 2:
        return {"tokens": semantic[0], "width": semantic[1]}
    elements = 1
    for value in semantic or launch:
        elements *= max(1, int(value))
    return {"elements": max(1, elements)}


def _shape_kind(op: OpSpec) -> str:
    launch = _tuple_ints(op.launch_shape)
    semantic = _tuple_ints(op.semantic_shape)
    if op.family in ATTENTION_FAMILIES and len(launch) >= 4:
        return "attention"
    if op.family == "attention_softmax" and len(semantic) >= 3:
        return "attention_softmax"
    if op.family == "router_softmax" and len(semantic) >= 2:
        return "router_softmax"
    if op.family == "cache_store":
        return "cache"
    if op.family in MOE_GROUPED_FAMILIES:
        return "moe_grouped"
    if op.family in PLAIN_GEMM_FAMILIES and len(_tuple_ints(op.launch_shape)) >= 3:
        return "gemm"
    if op.vector_ops > 0 or op.flops <= 0:
        return "vector"
    return "other"




def _observed_boundaries(op: OpSpec, runtime: RuntimeSpec, model: Any, kind: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    launch = _tuple_ints(op.launch_shape)
    padded = _tuple_ints(op.padded_shape)
    for dim, (raw, pad) in enumerate(zip(launch, padded)):
        if pad == raw:
            continue
        for candidate in TILE_CANDIDATES:
            if align_up(raw, candidate) == pad:
                result.append({
                    "dim": dim, "value": candidate,
                    "source": "lowering_padding_alignment",
                    "launch": raw, "padded": pad,
                })
                break
    quant = getattr(model, "quant", None)
    if quant is not None and kind == "gemm":
        br, bc = quant.weight_block
        if br > 0:
            result.append({"dim": 1, "value": int(br), "source": "declared_weight_block"})
        if bc > 0:
            result.append({"dim": 2, "value": int(bc), "source": "declared_weight_block"})
        if str(quant.expert_weight_format).startswith(("mxfp4", "fp4")) and quant.expert_block > 0:
            result.append({"dim": 2, "value": int(quant.expert_block), "source": "declared_expert_block"})
    if kind in {"attention", "attention_softmax", "cache"} and runtime.kv_block_size > 0:
        result.append({"dim": 1, "value": int(runtime.kv_block_size), "source": "declared_kv_block"})
    dedup: dict[tuple[int, int, str], dict[str, Any]] = {}
    for item in result:
        dedup[(item["dim"], item["value"], item["source"])] = item
    return [dedup[key] for key in sorted(dedup)]


def _v(kind: str, recipe: str | None, backend: str | None, target: str,
       status: str, reason: str, required: str | None) -> dict[str, Any]:
    return {"execution_kind": kind, "worker_recipe": recipe, "backend": backend,
            "target_match": target, "status": status, "reason": reason,
            "required_backend": required}


def _execution_variants(op: OpSpec, precision: Mapping[str, Any], kind: str,
                        shape: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    shape = dict(shape or {})
    family, recipe = op.family, op.recipe
    if family in EXCLUDED_FAMILIES:
        return [_v("excluded_communication", None, None, "not_this_campaign",
                   "excluded_communication",
                   "collective/transfer/barrier work is owned by the communication lane; "
                   "this campaign does not measure A5 or merge communication cost",
                   "communication lane (core-owned AI_CPU/AIV expansion; excluded here)")]
    if kind == "gemm":
        fmt = str(precision.get("weight_format") or "")
        if fmt in {"bf16", "bfloat16", "fp16", "float16", ""}:
            return [_v("primitive", "torch_mm_preallocated", "torch.mm", "exact_primitive",
                       "executable", "plain dense GEMM on the declared dtype", "torch")]
        if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            return [
                _v("reference_decomposition", "fp8_block_dequant_bf16_mm",
                   "torch (dequant + BF16 mm)", "not_target_reference", "executable",
                   "block-FP8 dequant followed by BF16 mm; explicitly not the target fused FP8 kernel",
                   "torch"),
                _v("fused_kernel", "npu_blockwise_quant_matmul",
                   "torch_npu.npu_quant_matmul (blockwise FP8)", "target_candidate",
                   "pending_backend",
                   "exact blockwise FP8 route is not verified for the declared packing/scale layout",
                   "torch_npu.npu_quant_matmul with blockwise FP8 scales"),
            ]
        if fmt in {"int8", "w8a8"}:
            return [_v("reference_decomposition", "int8_dequant_bf16_mm",
                       "torch (dequant + BF16 mm)", "not_target_reference", "executable",
                       "explicit int8 dequant + BF16 mm reference; not a fused int8 kernel",
                       "torch")]
        return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
                   f"weight format {fmt!r} has no verified plain GEMM worker recipe",
                   PENDING_BACKENDS.get(family, "native quantized GEMM backend"))]
    if family in ATTENTION_FAMILIES:
        return [
            _v("primitive", "torch_attention_matmul", "torch.matmul", "exact_primitive",
               "executable", "standalone QK or PV primitive with explicit q/kv/head/dim; "
               "not fused attention", "torch"),
            _v("fused_kernel", "npu_fusion_attention", "torch_npu.npu_fusion_attention",
               "target_candidate", "pending_backend",
               "fused attention is optional and only selected if a verified native route is "
               "probed; this campaign never relabels primitive QK/softmax/PV as fused",
               "verified torch_npu.npu_fusion_attention"),
        ]
    if family in SOFTMAX_FAMILIES:
        if family == "router_softmax":
            meta = dict(op.metadata)
            contract = " ".join(str(meta.get(key) or "") for key in (
                "router_contract", "scoring_func", "semantic_family")).lower()
            if "topk" in contract or "sigmoid" in contract or "sqrtsoftplus" in contract:
                return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
                           "actual router scoring/top-k composition (for example sigmoid_topk); "
                           "a plain softmax is not the same runtime recipe",
                           "native router scoring + top-k composition")]
        variants = [_v("primitive", "torch_softmax", "torch.softmax", "exact_primitive",
                       "executable", "softmax primitive over the declared normalized axis", "torch")]
        if family == "attention_softmax" and kind == "attention_softmax":
            variants.append(_v("fused_kernel", "npu_scaled_masked_softmax",
                               "torch_npu.npu_scaled_masked_softmax", "target_candidate",
                               "conditional_backend",
                               "native masked-softmax candidate; official NCHW H/W constraints; "
                               "worker probes and validates masked positions/row sums before timing",
                               "torch_npu.npu_scaled_masked_softmax"))
        return variants
    if family in NORM_FAMILIES:
        meta = dict(op.metadata)
        if (family == "qk_norm" and (meta.get("not_rmsnorm")
                                     or str(meta.get("norm_type") or "rmsnorm").lower() != "rmsnorm"
                                     or bool(meta.get("composite")))):
            return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
                       "actual composite " + str(meta.get("semantic_family") or "qk norm") +
                       " is not RMSNorm; record its real subprimitives instead of substituting RMS",
                       "native composite (" + ", ".join(str(v) for v in (
                           meta.get("subprimitives") or ("split", "qk_l2norm", "mrope"))) + ")")]
        if not (shape.get("tokens") and shape.get("width")):
            return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
                       "RMSNorm condition lacks explicit tokens,width; no executable shape contract",
                       "condition with explicit tokens,width")]
        return [
            _v("fused_kernel", "npu_rms_norm", "torch_npu.npu_rms_norm", "target_candidate",
               "conditional_backend",
               "RMSNorm target route; worker probes torch_npu.npu_rms_norm and verifies output, "
               "otherwise pending", "torch_npu.npu_rms_norm"),
            _v("reference_decomposition", "rmsnorm_reference", "torch (rsqrt/mean/mul)",
               "not_target_reference", "executable",
               "explicit multi-op RMSNorm reference, never labelled as the fused target", "torch"),
        ]
    if family in SWIGLU_FAMILIES:
        if not (shape.get("tokens") and shape.get("width")):
            return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
                       "SwiGLU condition lacks explicit tokens,width; no executable shape contract",
                       "condition with explicit tokens,width")]
        return [
            _v("fused_kernel", "npu_swiglu", "torch_npu.npu_swiglu", "target_candidate",
               "conditional_backend",
               "SwiGLU target route; worker probes and verifies torch_npu.npu_swiglu, "
               "otherwise pending", "torch_npu.npu_swiglu"),
            _v("reference_decomposition", "swiglu_reference", "torch (silu+mul)",
               "not_target_reference", "executable",
               "explicit SiLU+mul reference decomposition, never labelled as the fused target",
               "torch"),
        ]
    if family in ELEMENTWISE_FAMILIES:
        if recipe in {"catalog_residual_add", "catalog_moe_residual_join",
                      "catalog_engram_residual_add"}:
            return [_v("reference_decomposition", "elementwise_add_reference", "torch.add",
                       "not_target_reference", "executable",
                       "two-input actual residual-add reference; fused residual/norm kernels "
                       "are not claimed", "torch")]
        return [_v("reference_decomposition", None, None, "not_target_reference",
                   "pending_backend",
                   "elementwise/mHC/router/gating semantics are not mapped to a plain add; "
                   "only actual residual-add recipes use the add reference",
                   PENDING_BACKENDS.get(family, "native elementwise/mHC kernel"))]
    if family in CACHE_FAMILIES:
        return [_v("reference_decomposition", "copy_rows_reference",
                   "torch copy_ into preallocated rows", "not_target_reference", "executable",
                   "row copy reference for cache preparation; not a fused cache/index kernel",
                   "torch")]
    if family in MOE_GROUPED_FAMILIES:
        fmt = str(precision.get("weight_format") or "")
        if fmt in {"bf16", "bfloat16", "fp16", "float16", ""}:
            return [
                _v("fused_kernel", "npu_grouped_matmul_list_bf16",
                   "torch_npu.npu_grouped_matmul (list, group_list=None)",
                   "target_candidate", "conditional_backend",
                   "honest native list recipe: A3 group_type=-1 requires group_list=None; "
                   "uses the actual point routed-token/expert histogram and validates "
                   "per-expert samples.  Not a packed single-X group_type=0 claim",
                   "torch_npu.npu_grouped_matmul list API (group_list=None, group_type=-1)"),
                _v("reference_decomposition", "grouped_expert_mm_reference",
                   "torch list of per-expert mm", "not_target_reference", "executable",
                   "explicit per-expert mm reference; never labelled as the grouped target",
                   "torch"),
            ]
        return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
                   "grouped expert GEMM for this packed format requires actual routing/expert "
                   "layout; a plain GEMM or BF16 group must not be relabelled",
                   PENDING_BACKENDS[family])]
    if family == "quantize":
        meta = dict(op.metadata)
        if (shape.get("tokens") and shape.get("width")
                and str(meta.get("quant_axis") or meta.get("scale_axis") or "") == "per_row"
                and (str(meta.get("output_format") or "") == "int8"
                     or str(precision.get("output_dtype") or "") == "int8")):
            return [
                _v("reference_decomposition", "int8_per_row_quant_reference",
                   "torch (row absmax scale + round/clamp)",
                   "not_target_reference", "executable",
                   "explicit int8 per-row quantize reference with the declared scale semantics; "
                   "not a native index/quant kernel", "torch"),
                _v("fused_kernel", None, None, "target_candidate", "pending_backend",
                   "native index-K int8 per-row quantize route is not verified; do not substitute "
                   "an unrelated dynamic-quant kernel",
                   str(meta.get("semantic_family") or "native index-K int8 per-row quantize")),
            ]
        return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
                   "quantize condition lacks the actual row axis/dtype/scale semantics",
                   "explicit quantize shape/scale contract")]
    required = PENDING_BACKENDS.get(family, f"native fused/selector backend for {family}")
    return [_v("fused_kernel", None, None, "target_candidate", "pending_backend",
               f"no verified worker recipe for family={family!r} recipe={recipe!r}", required)]


def _feature_signature(op: OpSpec, work: Any, topology: Mapping[str, Any], runtime: RuntimeSpec) -> dict[str, Any]:
    meta = {str(k): v for k, v in dict(op.metadata).items()}
    return {
        "metadata": meta,
        "mode": work.mode,
        "graph_bucket": work.graph_bucket,
        "graph_mode": runtime.graph_mode,
        "kv_block_size": runtime.kv_block_size,
        "moe_recipe": work.recipe,
        "parallel": {
            "tp": int(topology["tp"]), "ep": int(topology["ep"]),
            "pp": int(topology["pp"]), "dp": int(topology.get("dp", 1)),
        },
        "routing": work.routing.to_dict() if work.routing is not None else None,
    }


def _state_signature(op: OpSpec, model: Any, work: Any) -> dict[str, Any]:
    plan = None
    if 0 <= int(op.layer) < len(model.layer_plans):
        plan = model.layer_plans[int(op.layer)]
    state: dict[str, Any] = {
        "stage": op.stage, "layer": op.layer, "requests": list(op.requests),
        "phase": op.phase, "mode": work.mode, "padded_tokens": work.padded_tokens,
        "valid_tokens": work.valid_tokens, "graph_bucket": work.graph_bucket,
    }
    if plan is not None:
        state.update({
            "architecture": plan.architecture, "attention": plan.attention, "ffn": plan.ffn,
            "compress_ratio": plan.compress_ratio, "window": plan.window,
            "indexer": plan.indexer, "num_heads": plan.num_heads,
            "num_kv_heads": plan.num_kv_heads, "head_dim": plan.head_dim,
            "num_experts": plan.num_experts, "top_k": plan.top_k,
            "moe_intermediate_size": plan.moe_intermediate_size,
            "recurrent_state_elements": plan.recurrent_state_elements,
            "kv_cache_elements": plan.kv_cache_elements,
            "router_mode": plan.router_mode,
            "scope": plan.scope,
        })
    return state


def _lineage_origin(op: OpSpec, *, model_key: str, topology: Mapping[str, Any],
                    case: Mapping[str, Any], work: Any) -> dict[str, Any]:
    op_id = str(op.op_id)
    match = re.search(r"\.l(\d+)\.", op_id)
    layer = int(match.group(1)) if match else op.layer
    template = re.sub(r"\.l\d+\.", ".l{layer}.", op_id)
    return {
        "model": model_key,
        "topology": str(topology.get("name", "topology")),
        "phase": str(case["phase"]),
        "demand": str(case["demand"]),
        "op_id_template": template,
        "layer": layer,
        "stage": op.stage,
        "family": op.family,
        "recipe": op.recipe,
        "requests": list(op.requests),
        "mode": work.mode,
    }


def _merge_origin(origins: dict[str, dict[str, Any]], origin: Mapping[str, Any]) -> None:
    key = _canonical({
        "model": origin["model"], "topology": origin["topology"],
        "phase": origin["phase"], "demand": origin["demand"],
        "op_id_template": origin["op_id_template"], "stage": origin["stage"],
        "family": origin["family"], "recipe": origin["recipe"],
        "requests": origin["requests"],
    })
    current = origins.get(key)
    if current is None:
        current = dict(origin)
        current["layers"] = [int(origin["layer"])]
        origins[key] = current
    else:
        if int(origin["layer"]) not in current["layers"]:
            current["layers"].append(int(origin["layer"]))
    current["count"] = len(current["layers"])


def _origin_summary(origin: Mapping[str, Any]) -> dict[str, Any]:
    layers = sorted(set(int(item) for item in origin.get("layers", ())))
    item = {key: value for key, value in origin.items() if key != "layer"}
    item["layers"] = layers
    if not layers:
        item["layer_range"] = []
    elif layers == list(range(min(layers), max(layers) + 1)):
        item["layer_range"] = [min(layers), max(layers)]
    else:
        item["layer_range"] = []
    item["count"] = len(layers) if layers else 1
    return item


def _expansion_fields(op: OpSpec) -> dict[str, Any]:
    """Carry core-owned AI_CPU/AIV/communication expansion facts without merging."""
    meta = {str(k): v for k, v in dict(op.metadata).items()}
    def pick(*keys: str) -> Any:
        for key in keys:
            if key in meta:
                return meta[key]
        return None
    raw = {key: value for key, value in meta.items()
           if any(token in key.lower() for token in ("ai_cpu", "aicpu", "aiv", "communication", "comm_"))}
    return {
        "ai_cpu": pick("ai_cpu", "aicpu", "ai_cpu_us"),
        "aiv": pick("aiv", "aiv_us", "vector_expansion"),
        "communication": pick("communication", "comm", "communication_backend"),
        "raw": raw,
        "note": "carried generically from core metadata; never merged into operator timing; A5 collection is root-owned",
    }


def _identity_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Physical condition identity: drop origin-only layer/owner fields."""
    metadata = {
        key: value for key, value in dict(features.get("metadata") or {}).items()
        if not any(token in str(key).lower() for token in ("layer", "owner"))
    }
    return {**features, "metadata": metadata}


def _identity_state(state: Mapping[str, Any], family: str, kind: str) -> dict[str, Any]:
    """Keep only state facts that can change this physical operator recipe."""
    common = {"phase", "mode", "padded_tokens", "valid_tokens", "graph_bucket"}
    if kind == "moe_grouped" or family in MOE_GROUPED_FAMILIES or family == "router_softmax":
        common |= {"num_experts", "top_k", "moe_intermediate_size", "router_mode"}
    if kind in {"attention", "attention_softmax"}:
        common |= {"architecture", "attention", "compress_ratio", "window", "indexer",
                   "num_heads", "num_kv_heads", "head_dim"}
    if family in {"recurrent_state", "linear_conv"}:
        common |= {"architecture", "attention", "recurrent_state_elements", "kv_cache_elements"}
    return {key: value for key, value in dict(state).items() if key in common}


def _condition_key(payload: Mapping[str, Any]) -> str:
    identity = {
        "family": payload["family"], "recipe": payload["recipe"],
        "phase": payload["phase"], "execution_kind": payload["execution_kind"],
        "worker_recipe": payload["worker_recipe"], "backend": payload["backend"],
        "shape_kind": payload["shape_kind"],
        "semantic_shape": payload["semantic_shape"], "launch_shape": payload["launch_shape"],
        "padded_shape": payload["padded_shape"],
        "precision": payload["precision"], "layout": payload["layout"],
        "features": _identity_features(payload["features"]),
        "state": _identity_state(payload["state"], payload["family"], payload["shape_kind"]),
        "boundaries": payload["boundaries"], "expansion": payload["expansion"],
    }
    return _hash("cond:", identity)




def _build_condition(op: OpSpec, *, model: Any, model_key: str, topology: Mapping[str, Any],
                     case: Mapping[str, Any], work: Any, runtime: RuntimeSpec,
                     variant: Mapping[str, Any], config: Any) -> dict[str, Any]:
    precision = _precision(op, model)
    kind = _shape_kind(op)
    shape = _shape_map(op, kind)
    state = _state_signature(op, model, work)
    grouped = None
    if kind == "moe_grouped":
        grouped = _actual_grouped_geometry(
            op, work, int(state.get("num_experts") or shape.get("experts", 1)))
        if grouped is not None:
            shape["global_experts"] = int(grouped["global_experts"])
            shape["rank_experts"] = int(grouped["rank_experts"])
            shape["rank_routed_tokens"] = int(grouped["rank_routed_tokens"])
            shape["experts"] = max(1, int(grouped["actual_experts"]))
            shape["routed_tokens"] = max(1, int(grouped["actual_routed_tokens"]))
    boundaries = _observed_boundaries(op, runtime, model, kind)
    features = _feature_signature(op, work, topology, runtime)
    if grouped is not None:
        features["grouped_geometry"] = {
            key: grouped[key] for key in (
                "actual_experts", "actual_routed_tokens", "actual_expert_tokens",
                "actual_local_expert_start", "global_experts", "rank_experts",
                "rank_routed_tokens", "rank_expert_tokens", "source")
        }
        routing = dict(features.get("routing") or {})
        routing.update({
            "routing_tokens": int(grouped["actual_routed_tokens"]),
            "local_routing_tokens": int(grouped["actual_routed_tokens"]),
            "local_expert_counts": list(grouped["actual_expert_tokens"]),
            "local_experts": int(grouped["actual_experts"]),
            "local_expert_start": int(grouped["actual_local_expert_start"]),
            "global_experts": int(grouped["global_experts"]),
            "rank_experts": int(grouped["rank_experts"]),
            "rank_routing_tokens": int(grouped["rank_routed_tokens"]),
            "rank_local_expert_counts": list(grouped["rank_expert_tokens"]),
            "geometry_source": grouped["source"],
        })
        features["routing"] = routing
    condition = {
        "family": op.family, "recipe": op.recipe, "phase": op.phase,
        "execution_kind": variant["execution_kind"],
        "worker_recipe": variant.get("worker_recipe"),
        "backend": variant.get("backend"),
        "target_match": variant.get("target_match"),
        "status": variant.get("status"),
        "reason": variant.get("reason", ""),
        "required_backend": variant.get("required_backend"),
        "shape_kind": kind,
        "shape": shape,
        "semantic_shape": list(_tuple_ints(op.semantic_shape)),
        "launch_shape": list(_tuple_ints(op.launch_shape)),
        "padded_shape": list(_tuple_ints(op.padded_shape)),
        "local_shape": list(_tuple_ints(op.local_shape)),
        "precision": precision,
        "layout": op.layout,
        "quantization": precision["quantization"],
        "features": features,
        "state": state,
        "boundaries": boundaries,
        "expansion": _expansion_fields(op),
        "work": {
            "flops": float(op.flops), "hbm_bytes": float(op.hbm_bytes),
            "mte2_bytes": float(op.mte2_bytes), "mte3_bytes": float(op.mte3_bytes),
            "vector_ops": float(op.vector_ops), "dtype": op.dtype,
            "semantic_shape": list(_tuple_ints(op.semantic_shape)),
            "launch_shape": list(_tuple_ints(op.launch_shape)),
        },
        "protocol": dict(config.protocol),
        "sources": {},
    }
    condition["condition_key"] = _condition_key(condition)
    _merge_origin(condition["sources"], _lineage_origin(
        op, model_key=model_key, topology=topology, case=case, work=work,
    ))
    return condition


def _source_summary(condition: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, lossless-at-aggregate-level lineage for a selected condition."""
    raw_sources = condition.get("sources") or {}
    origins = list(raw_sources.values()) if isinstance(raw_sources, Mapping) else list(raw_sources)
    layer_total = sum(len(origin.get("layers", ())) or 1 for origin in origins)
    templates: dict[str, set[int]] = {}
    for origin in origins:
        templates.setdefault(str(origin["op_id_template"]), set()).update(
            int(layer) for layer in origin.get("layers", ()))
    return {
        "origin_count": len(origins),
        "source_case_count": layer_total,
        "models": sorted({origin["model"] for origin in origins}),
        "topologies": sorted({origin["topology"] for origin in origins}),
        "phases": sorted({origin["phase"] for origin in origins}),
        "demands": sorted({origin["demand"] for origin in origins}),
        "op_templates": [
            {"template": template, "layers": sorted(layers)}
            for template, layers in sorted(templates.items())[:24]
        ],
        "op_template_count": len(templates),
        "examples": [_origin_summary(origin) for origin in origins[:12]],
        "note": "all origins are counted; examples are bounded and full conditions are deterministically regenerable",
    }


def _condition_record(condition: Mapping[str, Any], *, point_count: int,
                      rejections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Complete compact condition lineage persisted in conditions.json."""
    origins = [_origin_summary(origin) for origin in condition["sources"].values()]
    return {
        "condition_key": condition["condition_key"],
        "family": condition["family"], "recipe": condition["recipe"],
        "phase": condition["phase"], "execution_kind": condition["execution_kind"],
        "worker_recipe": condition["worker_recipe"], "backend": condition["backend"],
        "target_match": condition["target_match"], "status": condition["status"],
        "reason": condition["reason"], "required_backend": condition["required_backend"],
        "shape_kind": condition["shape_kind"], "shape": condition["shape"],
        "semantic_shape": condition["semantic_shape"], "launch_shape": condition["launch_shape"],
        "padded_shape": condition["padded_shape"], "local_shape": condition["local_shape"],
        "precision": condition["precision"], "layout": condition["layout"],
        "quantization": condition["quantization"], "boundaries": condition["boundaries"],
        "features": condition["features"], "state": condition["state"],
        "expansion": condition["expansion"], "protocol": condition["protocol"],
        "work": condition["work"], "sources": origins,
        "source_case_count": sum(int(origin.get("count", 1)) for origin in origins),
        "materialized_point_count": point_count,
        "axis_rejections": list(rejections),
    }


