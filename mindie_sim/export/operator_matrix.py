"""Multi-model operator matrix engine, physics cliffs, dual-peak HBM gate & BenchPlanV2 compiler.

[Role] Compile catalog models, local HuggingFace ``config.json`` files, or inline
  ``ModelSpec`` mappings across Ascend SKUs (A2/A3/A5), phases, physics-derived
  sequence cliffs, concurrencies, and structural parallel arms through the single
  ``PredictionSpine`` (``build_export_bundle``) into ``OperatorMatrixManifest`` and
  deduplicated cross-model ``mindie-bench-plan/2`` plans with lossless occurrence
  reverse indices and ``--compat-v1`` projection. Also hosts the canonical strict
  1-to-1 builder/ABI/topology/fabric/slot-plan/hardware-view resolvers formerly in
  unpackaged ``tools/bench_plan_*.py``.
[Invariants]
  * 100% AG014-clean: zero ``model_id`` branches and zero hardcoded model probes.
  * Un-derated ``HardwareNominalSpec`` (``utilization == 1.0, launch_us == 0.0``)
    supplies every Roofline lower bound and MFU/MBU/Link-BU denominator.
  * Dual-peak HBM + host DDR admissibility gate checks BOTH ``runtime_peak`` and
    ``load_peak`` on every die plus optional HiSparse host DDR caps before replay.
  * ``derive_physical_seq_cliffs`` enforces ``local_kv`` ABI divisibility on every
    non-``abi_rejection`` sequence point (``R2-PA-12b-07``).
  * Zero runtime imports of ``mindie_profiler`` or ``mindie_sim.onboarding``;
    zero ``predicted_*`` keys in ``BenchPlanV2``.
[Timeouts & Locks] Bounded in-memory compilation and single-spine replay; no locks
  or device contexts.
[Side-Effect Policy] Pure in-memory evaluation; read-only over local config/campaign
  files when paths are supplied.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import accumulate
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.backends.ascend import (
    materialize as materialize_ascend,
    profiles as ascend_profiles,
)
from mindie_sim.catalog import load_model as load_catalog_model
from mindie_sim.catalog.cache import DEFAULT_CACHE_BLOCK_SIZE, layer_cache_components
from mindie_sim.catalog.context_parallel import validate_vllm_ascend_dcp_codomain
from mindie_sim.catalog.expert_slots import expert_slot_plan
from mindie_sim.catalog.specs import build_model_spec
from mindie_sim.compose import compose_case
from mindie_sim.compose.inputs import resolve_hardware_sku
from mindie_sim.observations.campaign_reconcile import (
    enrich_operator_matrix,
    reconcile_campaign,
    verify_campaign_integrity,
)
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.operators.sparse_indexer.sparse_kv_offload import HOST_BUFFER_POOLS
from mindie_sim.prediction.results import LoweringRequest, StepRequest
from mindie_sim.prediction.spine import default_spine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.abi import HARDWARE_USABLE_ABI_BACKENDS
from mindie_sim.schema.bench_matrix import (
    BENCH_PLAN_V2_SCHEMA,
    BYTES_PER_GB,
    DEFAULT_ANCHOR_LADDER,
    DEFAULT_CLIFF_BUDGET_PER_ARM,
    DEFAULT_SEQ_MAX,
    DEFAULT_SEQ_MIN,
    PCT_SCALE,
    US_PER_SECOND,
    AcceleratorCore,
    BenchPlanCaseV2,
    BenchPlanV2,
    BoundType,
    CampaignManifest,
    CanonicalKernelKey,
    CaptureMode,
    CliffKind,
    HardwareNominalSpec,
    HbmAdmissibilityReceipt,
    KernelKeyKind,
    MatrixSpec,
    ModelScopeKind,
    OperatorMatrixManifest,
    OperatorMatrixRow,
    ParallelArmSpec,
    PhaseKind,
    PlanSplit,
    ReverseIndexEntry,
    SeqCliffPoint,
    StepSummary,
    canonical_json,
    canonical_sha256,
    resolve_fabric_collective_abi as _schema_resolve_fabric_collective_abi,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes, inclusion_probability
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec

from .bundle import build_export_bundle, resolve_hardware_alias
from .memory_ledger import _resolve_ledger_workspace_tokens, compute_memory_quant_ledger
from .models import ExportBundle, MemoryCalculatorLedger, OperatorExportRow

# ---------------------------------------------------------------------------
# Constants & Canonical Builder / Topology Tables
# ---------------------------------------------------------------------------

ONBOARDING_HINT_SCHEMA = "mindie-unmodeled-structure-hint/1"
SUBGRAPH_PLAN_V1_SCHEMA = "mindie-bench-subgraph-plan/1"
SUBGRAPH_NODE_V1_SCHEMA = "mindie-bench-subgraph-node/1"
CONTROL_FAMILIES: frozenset[str] = frozenset({"barrier", "event", "hccl_boundary"})
BOUNDARY_BUILDER = "graph_boundary"
SUPPORTED_FABRIC_ROUTES: frozenset[str] = frozenset({
    "intra_package_die", "intra_node_hccs", "inter_node_roce",
})

DEFAULT_TP = 8
ALT_TP_QUARTER = 4
ALT_TP_WIDE = 16
DENSE_EP = 1
LOCAL_MOE_EP = 8
TRILLION_MOE_EP = 64
TRILLION_PARAMETER_COUNT = 1_000_000_000_000
DEFAULT_CP_DEGREE = 2
WIDE_MLA_DCP_DEGREE = 4
ROUTED_EXPERT_MATRIX_FACTOR = 3
MIB_BYTES = 1024 * 1024
GIB_BYTES = 1024 ** 3
S0_REPLAY_CEILING_BYTES = 8 * GIB_BYTES
FABRIC_COLD_RING_FACTOR = 3
FABRIC_SLOT_BYTES = 8 * MIB_BYTES
FABRIC_COLLECTIVES: tuple[str, ...] = (
    "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "all_to_all_v",
)
FABRIC_ROUTES: tuple[str, ...] = ("intra_package_die", "intra_node_hccs", "inter_node_roce")
HELDOUT_FABRIC_BYTES: frozenset[int] = frozenset({1 << 15, 1 << 19, 1 << 23, 1 << 27})
FABRIC_LADDER_BYTES: tuple[int, ...] = tuple(sorted({*(1 << n for n in (12, 14, 18, 22, 26)), *HELDOUT_FABRIC_BYTES}))
FABRIC_PARTICIPANTS: Mapping[str, int] = {
    "intra_package_die": 2,
    "intra_node_hccs": 4,
    "inter_node_roce": 8,
}
FABRIC_ROUTE_LINK_TIERS: Mapping[str, str] = {
    "intra_package_die": "d2d_intra_package",
    "intra_node_hccs": "d2d_intra_node",
    "inter_node_roce": "d2d_inter_node",
}
RING_ALLREDUCE_PHASES = 2


@dataclass(frozen=True, slots=True)
class FabricOperatorCurvePoint:
    """Physics-grounded fabric collective scaling point aligned with ``BenchPlanV2`` fabric cases."""

    case_id: str
    op_type: str
    kernel_name: str
    sku: str
    soc_arch: str
    route: str
    participants: int
    payload_bytes: int
    algo_wire_bytes: int
    nominal_theo_us: float
    predicted_us: float
    measured_p50_us: float | None
    measured_p99_us: float | None
    bound_type: str
    achieved_bw_gb_s: float
    comm_bw_util_pct: float
DEFAULT_HISPARSE_DRAM_GB = 128.0
DEFAULT_HISPARSE_TOPK_BUFFER = 2048
DEFAULT_HISPARSE_MISS_RATE = 0.15
DEFAULT_PREFILL_CHUNK_CAP = 2048
DEFAULT_OUTPUT_TOKENS_BUDGET = 128
DEFAULT_WAVE_TILE_STRIDE = 128
DEFAULT_HEAD_DIM = 128
MLA_LATENT_DIM = 512
MLA_ROPE_DIM = 64
MIN_Q_VIEW_RANK_3D = 3
MIN_Q_VIEW_RANK_4D = 4
KV_HEAD_AXIS = -2
DOUBLE_BUFFER_FACTOR = 2
L2_CLIFF_LOW_RATIO = 0.5
L2_CLIFF_HIGH_RATIO = 1.5
UB_SCORE_ELEM_BYTES = 4
MIN_POSITIVE_DENOM = 1e-12
ROUND_PCT_DIGITS = 4
ROUND_US_DIGITS = 6
MAX_CAMPAIGN_JSON_BYTES = 50 * MIB_BYTES
HEX_DIGEST_SHORT = 16
FRACTAL_NZ_FORMAT_CODE = 29
DEFAULT_LEGACY_SKUS: tuple[str, ...] = ("Ascend910B4", "Ascend910_9362", "Ascend910_9382")
DEFAULT_LOCAL_KV_ABI_DIVISOR = 16
HOST_ALIGNMENT_SLAB_BYTES = 2 * MIB_BYTES
HELDOUT_BATCH = 16
HELDOUT_KV_SET: frozenset[int] = frozenset({16384, 262144, 409600})
HELDOUT_MATRIX_ROWS: frozenset[int] = frozenset({3, 12, 96, 384, 1536, 3072})
MISSING_LAYER = -1
CANONICAL_KEY_CACHE_MAX = 65536
_CANONICAL_KEYS_BY_CASE_ID: dict[str, CanonicalKernelKey] = {}

EXPERT_BYTES_PER_ELEMENT: Mapping[str, float] = {
    "bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "int8": 1.0, "mxfp8": 1.0, "mxfp4": 0.5, "fp4": 0.5, "fp4_e2m1": 0.5,
    "int4": 0.5, "w4a8": 0.5, "w4a16": 0.5, "compressed_tensors_w4a16": 0.5,
    "mxfp4_routed_only": 0.5,
}
ELEMENT_BYTES: Mapping[str, int] = {
    "bf16": 2, "fp16": 2, "fp32": 4, "int8": 1, "int4": 1,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "mxfp8": 1, "mxfp4": 1,
}


class BenchTopologyError(ValueError):
    """Structured error raised when collective topology resolution fails."""


class BenchGeometryError(ValueError):
    """Structured error raised when a fabric route cannot be physically realized."""


class PlanError(RuntimeError):
    """Structured error raised on invalid plan geometry, ledger, or routing."""


COLLECTIVE_BUILDERS: frozenset[str] = frozenset({
    "all_reduce", "quantized_all_reduce", "all_gather", "reduce_scatter",
    "all_to_all", "all_to_all_v", "mc2_dispatch", "mc2_combine",
    "mc2_dispatch_v3", "mc2_combine_v3", "megamoe_fused_mc2", "p2p_send_recv",
})

@lru_cache(maxsize=1)
def _unit_offset_rms_norm_binding() -> tuple[str, str]:
    """Resolve CANN unit-offset RMSNorm ``(builder, abi_symbol)`` from ``ProbedAbiRegistry``."""
    reg = default_abi_registry()
    for arch in sorted(reg.archs):
        for entry in reg.get_snapshot(arch).entries:
            if (
                entry.backend_kind == "cann_aclnn"
                and entry.graph_motif.endswith("_rms_norm")
                and "unit-offset" in entry.probe_domain_note.lower()
            ):
                return (str(entry.graph_motif), str(entry.symbol_or_kernel))
    return ("rms_norm", "aclnnRmsNorm")


BUILDERS: frozenset[str] = frozenset({
    *COLLECTIVE_BUILDERS, BOUNDARY_BUILDER,
    "all_gather_matmul", "matmul_reduce_scatter",
    "add_rms_norm_quant", "rms_norm_quant", "add_rms_norm", _unit_offset_rms_norm_binding()[0],
    "rms_norm", "layer_norm", "swiglu_oai", "swiglu", "geglu", "gelu",
    "situ_activation", "silu_mul", "sigmoid", "softmax",
    "dequant_swiglu_quant", "swiglu_dynamic_quant", "dequant_situ_quant",
    "dynamic_quant", "grouped_matmul_swiglu_quant", "grouped_matmul",
    "mla_prolog_v3", "mla_prolog_v2", "mla_kv_compressor",
    "dcp_lse_attention_merge", "dcp_query_pack", "dcp_output_pack",
    "mixed_quant_sparse_mla", "sparse_flash_attention", "absorbed_mla",
    "block_sparse_attention_c8", "incre_flash_attention", "fusion_attention",
    "fused_infer_attention", "pool_key_indexer", "lightning_indexer",
    "indexer_quant_cache", "key_pool", "kv_compress_epilog", "histogram_v2",
    "scatter_pa_kv_cache", "scatter_index_cache",
    "moe_gating_topk_hash", "moe_gating_topk", "indexer_topk",
    "moe_init_routing", "moe_token_permute", "strided_gather",
    "moe_token_unpermute", "moe_ffn", "split_qkv_rmsnorm_rope", "rotary_embedding",
    "causal_conv1d_fn", "causal_conv1d_update", "fused_gdn_gating",
    "chunk_gated_delta_rule", "chunk_gdn_fwd_prepare", "chunk_gdn_fwd_h",
    "chunk_gdn_fwd_o", "recurrent_gdn_recover_ssm",
    "recurrent_gated_delta_rule", "fused_recurrent_kda", "chunk_kda_fwd",
    "kda_chunk_products", "kda_wy_transform", "kda_chunk_output",
    "kda_triangular_solve", "ssm_state_gather", "ssm_state_scatter",
    "hc_pre", "hc_post", "hc_combine_norm", "mhc_stream_mix",
    "attn_res_prefix", "ple_lookup", "engram_ngram_hash", "engram_uva_h2d",
    "gated_mul_add", "vocab_embedding", "rejection_sampler",
    "categorical_sampler", "quant_matmul", "matmul", "l2_norm",
    "rms_norm_gated", "vector_add", "muls_add", "vector_mul_gate",
    "vector_pack_indices", "index_add", "tensor_pad", "dtype_cast",
    "tensor_copy", "tensor_fill",
})

_QUANT_DTYPES: frozenset[str] = frozenset({
    "int8", "int4", "w4a8", "w4a16", "w8a8", "fp8", "fp8_e4m3", "fp8_e5m2",
    "mxfp8", "mxfp4", "fp4", "fp4_e2m1",
})
_DENSE_GEMM_FAMILIES: frozenset[str] = frozenset({
    "gemm", "matmul", "linear", "qkv_projection", "output_projection",
    "gate_up_projection", "down_projection", "index_projection",
    "gate_projection", "projection", "router", "logits",
    "attention_qk", "attention_pv", "vision_merge",
})
_UNQUANTIZED_GEMM_ROLES: frozenset[str] = frozenset({
    "router", "logits", "attention_qk", "attention_pv", "vision_merge",
})
_ELEMENTWISE_ADD_RECIPES: frozenset[str] = frozenset({
    "residual_add", "expert_add", "draft_feature_add", "vector_add", "add",
    "catalog_engram_residual_add", "markov_bias", "attention_residual_mix",
    "catalog_aligner_bias", "catalog_vision_bias",
    "catalog_vision_residual", "router_correction_bias",
})
_ELEMENTWISE_MUL_GATE_RECIPES: frozenset[str] = frozenset({
    "attention_gate", "gated_output", "shared_gate_apply",
    "index_key_basis_scale", "index_query_basis_scale",
    "route_normalize", "router_group_mask",
})
_DIRECT_FAMILY_BUILDERS: dict[str, tuple[str, str]] = {
    "allgather": ("all_gather", "hcclAllGather"),
    "engram_broadcast": ("all_gather", "hcclAllGather"),
    "dcp_allgather": ("all_gather", "hcclAllGather"),
    "engram_ids": ("all_to_all_v", "hcclAlltoAllV"),
    "reduce_scatter": ("reduce_scatter", "hcclReduceScatter"),
    "pp_p2p": ("p2p_send_recv", "hcclSendRecv"),
    "cache_transfer": ("p2p_send_recv", "hcclSendRecv"),
    "state_publish": ("p2p_send_recv", "hcclSendRecv"),
    "state_transfer": ("p2p_send_recv", "hcclSendRecv"),
    "add_rmsnorm_quant": ("add_rms_norm_quant", "aclnnAddRmsNormDynamicQuantV2"),
    "rmsnorm_quant": ("rms_norm_quant", "aclnnRmsNormDynamicQuant"),
    "add_rmsnorm": ("add_rms_norm", "aclnnAddRmsNorm"),
    "layernorm": ("layer_norm", "aclnnLayerNorm"),
    "geglu": ("geglu", "aclnnGeGluV2"),
    "gelu": ("gelu", "aclnnGelu"),
    "situ": ("situ_activation", "aclnnSitu"),
    "sigmoid": ("sigmoid", "aclnnSigmoid"),
    "softmax": ("softmax", "aclnnSoftmaxV2"),
    "attention_softmax": ("softmax", "aclnnSoftmaxV2"),
    "dequant_swiglu_quant": ("dequant_swiglu_quant", "aclnnDequantSwigluQuant"),
    "attention_merge": ("dcp_lse_attention_merge", "aclnnRingAttentionUpdate"),
    "indexer_quant_cache": ("indexer_quant_cache", "aclnnIndexerQuantCache"),
    "key_pool": ("key_pool", "aclnnKeyPool"),
    "kv_compress_epilog": ("kv_compress_epilog", "aclnnKvCompressEpilog"),
    "histogram": ("histogram_v2", "aclnnHistogramV2"),
    "moe_init_routing": ("moe_init_routing", "aclnnMoeInitRoutingV3"),
    "causal_conv1d_fn": ("causal_conv1d_fn", "aclnnCausalConv1d"),
    "short_convolution": ("causal_conv1d_update", "aclnnCausalConv1dUpdate"),
    "conv1d": ("causal_conv1d_update", "aclnnCausalConv1dUpdate"),
    "gdn_gating": ("fused_gdn_gating", "fused_gdn_gating_kernel"),
    "gdn_prepare": ("chunk_gdn_fwd_prepare", "aclnnChunkGatedDeltaRuleFwdPrepare"),
    "gdn_state": ("chunk_gdn_fwd_h", "aclnnChunkFwdH"),
    "gdn_output": ("chunk_gdn_fwd_o", "aclnnChunkFwdO"),
    "triangular_solve": ("kda_triangular_solve", "solve_tril_16x16_kernel"),
    "state_gather": ("ssm_state_gather", "_gather_ssm_states_kernel"),
    "state_scatter": ("ssm_state_scatter", "_scatter_ssm_states_kernel"),
    "attn_res_prefix": ("attn_res_prefix", "aclnnAttnResFwd"),
    "engram_gather": ("ple_lookup", "aclnnEmbedding"),
    "engram_query": ("engram_ngram_hash", "engram_ngram_ids_kernel"),
    "engram_rh2d": ("engram_uva_h2d", "aclrtMemcpyAsync"),
    "engram_apply": ("gated_mul_add", "aclnnMul"),
    "embedding": ("vocab_embedding", "aclnnEmbedding"),
    "sampling": ("categorical_sampler", "_categorical_sample_kernel"),
    "confidence": ("categorical_sampler", "_categorical_sample_kernel"),
    "token_commit": ("categorical_sampler", "_categorical_sample_kernel"),
    "moe_ffn": ("megamoe_fused_mc2", "aclnnMegaMoe"),
    "index_add": ("index_add", "aclnnIndexAdd"),
    "scatter_add": ("index_add", "aclnnIndexAdd"),
    "gather": ("strided_gather", "aclnnIndexSelect"),
    "strided_gather": ("strided_gather", "aclnnIndexSelect"),
    "tensor_select": ("strided_gather", "aclnnIndexSelect"),
    "pad": ("tensor_pad", "aclnnPadV3"),
    "cast": ("dtype_cast", "aclnnCast"),
    "tensor_copy": ("tensor_copy", "aclnnInplaceCopy"),
    "tensor_materialization": ("tensor_copy", "aclnnInplaceCopy"),
    "tensor_view": ("tensor_copy", "aclnnInplaceCopy"),
    "tensor_fill": ("tensor_fill", "aclnnInplaceZero"),
    "zero_fill": ("tensor_fill", "aclnnInplaceZero"),
    "fill": ("tensor_fill", "aclnnInplaceZero"),
    "dcp_query_pack": ("dcp_query_pack", "_q_gather_prep_head_major_kernel"),
    "dcp_output_pack": ("dcp_output_pack", "_pack_dcp_output_lse_kernel"),
    "sparse_indexer": ("lightning_indexer", "aclnnQuantLightningIndexerV2"),
    "kv_cache_read": ("tensor_copy", "aclnnInplaceCopy"),
    "kv_prefetch": ("engram_uva_h2d", "aclrtMemcpyAsync"),
    "sparse_kv_offload": ("engram_uva_h2d", "aclrtMemcpyAsync"),
    "comm_overlap": ("all_gather", "hcclAllGather"),
}

LADDER_KV_FAMILIES: frozenset[str] = frozenset({
    "attention", "compressed_attention", "indexer", "indexer_quant_cache",
    "kv_compress_epilog", "cache_store", "key_pool", "attn_res_prefix",
    "recurrent_scan", "recurrent_vector", "recurrent_local", "recurrent_output",
    "kda_recurrent", "short_convolution", "rotary",
})
_ROCE_STRICT_MARKERS = ("roce", "rdma", "scaleout", "scale_out")
_CROSS_NODE_LINK_MARKERS = (*_ROCE_STRICT_MARKERS, "inter_node", "inter_pod")
_TRITON_ABI_MARKERS = ("_kernel", "triton", "recompute_w_u")
_TORCH_NPU_ABI_MARKERS = ("torch_npu.", "torch.")


def _short_digest(value: Any, prefix: str = "") -> str:
    return prefix + canonical_sha256(value)[:HEX_DIGEST_SHORT]


# ---------------------------------------------------------------------------
# Strict 1-to-1 Builder, ABI Backend, Topology, Fabric & Slot-Plan Resolvers
# ---------------------------------------------------------------------------

def resolve_builder_strict(
    family: str,
    recipe: str,
    impl: str,
    dtypes: Sequence[str],
    meta: Mapping[str, Any] | None = None,
    *,
    model_id: str = "",
    layer_idx: int = -1,
    local_shape: Sequence[Any] | None = None,
) -> tuple[str, str]:
    """Return canonical ``(builder_name, upstream_abi)`` or raise ``UnmodeledCapabilityError``."""
    meta_dict = dict(meta or {})
    fam = str(family or "").strip()
    rec = str(recipe or "").strip()
    imp = str(impl or "").strip()
    if fam in CONTROL_FAMILIES:
        return (BOUNDARY_BUILDER, fam)
    has_quant = (
        any(str(d).lower() in _QUANT_DTYPES for d in dtypes)
        or str(meta_dict.get("weight_format") or "").lower() in _QUANT_DTYPES
        or str(meta_dict.get("dequant") or "").lower() not in {"", "none"}
    )
    resolved = _resolve_builder_family_rules(fam, rec, imp, has_quant, meta_dict)
    if resolved is not None:
        return resolved
    if fam in _DIRECT_FAMILY_BUILDERS:
        return _DIRECT_FAMILY_BUILDERS[fam]
    if fam == "copy":
        return ("tensor_copy", "aclnnInplaceCopy")
    hint = {
        "schema": ONBOARDING_HINT_SCHEMA,
        "model_id": str(model_id),
        "layer_idx": int(layer_idx),
        "family": fam,
        "recipe": rec,
        "implementation": imp,
        "local_shape": [int(x) if isinstance(x, int) else x for x in (local_shape or ())],
        "dtypes": [str(d) for d in dtypes],
        "metadata": {
            str(k): v for k, v in meta_dict.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
        "agent_action": "inspect upstream kernel ABI in vllm-ascend and register explicit PhysicalRecipe + bench operator builder",
    }
    err = UnmodeledCapabilityError(
        capability=f"{fam}:{rec}:{imp}",
        entry="operator_matrix.resolve_builder_strict",
        code="unmodeled_bench_operator",
        message=json.dumps(hint, sort_keys=True),
        hints=(hint,),
        minimal_replay_input={
            "model_id": str(model_id), "layer_idx": int(layer_idx),
            "family": fam, "recipe": rec, "implementation": imp,
        },
    )
    err.payload = hint
    raise err


def _resolve_builder_family_rules(
    fam: str, rec: str, imp: str, has_quant: bool, meta_dict: Mapping[str, Any],
) -> tuple[str, str] | None:
    if fam == "sp_matmul_collective" or rec == "sp_matmul_collective_pipeline":
        direction = str(
            meta_dict.get("sp_collective_kind")
            or ("allgather" if meta_dict.get("pipeline_gather") and not meta_dict.get("pipeline_scatter")
                else "reduce_scatter")
        )
        return ("all_gather_matmul", "aclnnAllGatherMatmul") if (
            "all_gather" in direction or "allgather" in direction
        ) else ("matmul_reduce_scatter", "aclnnMatmulReduceScatter")
    if fam == "mc2_protocol":
        return _resolve_mc2_builder(rec, imp, meta_dict)
    if fam in {"alltoall", "alltoallv"}:
        return _resolve_alltoall_builder(fam, rec, imp, meta_dict)
    if fam in {"allreduce", "tp_allreduce"}:
        if (meta_dict.get("quantized_all_reduce")
                or meta_dict.get("quantized_all_reduce_applied")
                or rec == "quantized_all_reduce"):
            return ("quantized_all_reduce", "hcclAllReduceInt8")
        return ("all_reduce", "hcclAllReduce")
    if fam == "rmsnorm":
        unit_builder, unit_abi = _unit_offset_rms_norm_binding()
        unit_native = unit_abi.removeprefix("aclnn")
        if (
            float(meta_dict.get("norm_weight_offset") or 0.0) != 0.0
            or str(meta_dict.get("native_norm_kind") or "") == unit_native
            or rec == unit_builder
            or imp in {unit_abi, unit_native}
        ):
            return (unit_builder, unit_abi)
        return ("rms_norm", "aclnnRmsNorm")
    if fam == "swiglu":
        return ("swiglu_oai", "aclnnSwiGluOai") if "oai" in rec else ("swiglu", "aclnnSwiGlu")
    if fam == "activation":
        if "swiglu" in rec:
            return ("swiglu_oai", "aclnnSwiGluOai") if "oai" in rec else ("swiglu", "aclnnSwiGlu")
        if "gelu" in rec:
            return ("gelu", "aclnnGelu")
        return ("silu_mul", "aclnnSilu")
    if fam == "quantize":
        if rec == "swiglu_quantize":
            return ("swiglu_dynamic_quant", "aclnnSwigluQuant")
        if rec == "situ_quantize":
            return ("dequant_situ_quant", "aclnnDequantSituQuant")
        if rec in {"dynamic_quantize", "index_query_quantize", "quantize"} or "quant" in rec:
            return ("dynamic_quant", "aclnnDynamicQuantV2")
    if fam == "grouped_gemm" or (fam == "gemm" and rec in {"grouped_output_a", "grouped_output_b"}):
        if meta_dict.get("onchip_handoff_to"):
            return ("grouped_matmul_swiglu_quant", "aclnnGroupedMatmulSwigluQuantWeightNZ")
        return ("grouped_matmul", "aclnnGroupedMatmulV5")
    if fam == "mla" and rec == "draft_context_mla_prolog_v3":
        return ("mla_prolog_v3", "aclnnMlaPrologV3WeightNz")
    if fam == "mla_prolog" or rec in {"mla_prolog_v2", "mla_prolog", "mla_preprocess"}:
        return ("mla_prolog_v2", "aclnnMlaPrologV2WeightNz")
    if fam == "compressed_attention" and rec == "aclnn_compressor":
        return ("mla_kv_compressor", "aclnnCompressor")
    if fam == "attention":
        return _resolve_attention_builder(rec, imp, meta_dict)
    if fam in {"lightning_indexer", "indexer"}:
        return _resolve_indexer_builder(fam, rec, imp)
    if fam in {"kv_cache_write", "cache_store"}:
        if rec in {"scatter_pa_kv_cache", "kv_store", "draft_context_kv_scatter"} or "kv" in rec:
            return ("scatter_pa_kv_cache", "aclnnScatterPaKvCache")
        if rec in {"qsa_index_cache_write", "index_store", "cann_keypool_cache_write"} or "index" in rec:
            return ("scatter_index_cache", "aclnnScatterNdUpdate")
    if fam == "topk":
        if rec == "moe_gating_topk":
            return ("moe_gating_topk_hash", "aclnnMoeGatingTopKHash") if "Hash" in imp else (
                "moe_gating_topk", "aclnnMoeGatingTopKSoftmaxV2"
            )
        return ("indexer_topk", "aclnnTopkV2")
    if fam == "routing":
        return _resolve_routing_builder(rec, meta_dict)
    if fam == "rotary" or (fam == "elementwise" and rec == "rotary"):
        if rec in {"split_qkv_rmsnorm_mrope_fused", "split_qkv_rmsnorm_rope_fused", "split_qkv_rmsnorm_rope_vnorm_fused"}:
            return ("split_qkv_rmsnorm_rope", "split_qkv_rmsnorm_mrope_kernel")
        return ("rotary_embedding", "aclnnApplyRotaryPosEmbV2")
    if fam in {"gdn_prefill", "recurrent_vector", "recurrent_scan", "kda_recurrent", "recurrent_local", "recurrent_output"}:
        return _resolve_recurrent_builder(fam, rec, imp, meta_dict)
    if fam == "copy" and rec == "cann_keypool_state_gather":
        return ("ssm_state_gather", "_gather_ssm_states_kernel")
    if fam == "mhc":
        return _resolve_mhc_builder(rec)
    if fam == "acceptance" and rec == "speculative_acceptance":
        return ("rejection_sampler", "rejection_sample_kernel")
    if rec == "mc2_routed_expert_program":
        return ("megamoe_fused_mc2", "aclnnMegaMoe")
    if fam in _DENSE_GEMM_FAMILIES:
        if has_quant and fam not in _UNQUANTIZED_GEMM_ROLES:
            return ("quant_matmul", "aclnnQuantMatmulWeightNz")
        return ("matmul", "aclnnMatmul")
    if fam == "elementwise":
        return _resolve_elementwise_builder(rec, meta_dict)
    return None


def _resolve_mc2_builder(rec: str, imp: str, meta_dict: Mapping[str, Any]) -> tuple[str, str] | None:
    if int(meta_dict.get("mc2_mode") or 0) in {2, 3}:
        return ("megamoe_fused_mc2", "aclnnMegaMoe")
    native_kind = str(meta_dict.get("native_kind") or "")
    is_v2 = ("V2" in imp or "v2" in rec or native_kind.endswith("V2"))
    if "Dispatch" in imp or "dispatch" in rec or "Dispatch" in native_kind:
        return ("mc2_dispatch", "aclnnMoeDistributeDispatchV2") if is_v2 else (
            "mc2_dispatch_v3", "aclnnMoeDistributeDispatchV3"
        )
    if "Combine" in imp or "combine" in rec or "Combine" in native_kind:
        return ("mc2_combine", "aclnnMoeDistributeCombineV2") if is_v2 else (
            "mc2_combine_v3", "aclnnMoeDistributeCombineV3"
        )
    return None


def _resolve_alltoall_builder(fam: str, rec: str, imp: str, meta_dict: Mapping[str, Any]) -> tuple[str, str] | None:
    if int(meta_dict.get("mc2_mode") or 0) in {2, 3}:
        return ("megamoe_fused_mc2", "aclnnMegaMoe")
    if rec == "arch22_v2_declared_peer_channel":
        native_kind = str(meta_dict.get("native_kind") or "")
        if "Combine" in imp or "Combine" in native_kind:
            return ("mc2_combine", "aclnnMoeDistributeCombineV2")
        return ("mc2_dispatch", "aclnnMoeDistributeDispatchV2")
    if rec == "expert_dispatch":
        return ("mc2_dispatch", "aclnnMoeDistributeDispatchV2")
    if rec == "expert_combine":
        return ("mc2_combine", "aclnnMoeDistributeCombineV2")
    if rec in {"all_to_all", "dcp_output_alltoall"}:
        return ("all_to_all", "hcclAlltoAll")
    if rec in {"alltoall", "all_to_all_v", "expert_quantized_dispatch", "expert_scale_dispatch"} or fam == "alltoallv":
        return ("all_to_all_v", "hcclAlltoAllV")
    return ("all_to_all", "hcclAlltoAll")


def _resolve_attention_builder(rec: str, imp: str, meta_dict: Mapping[str, Any]) -> tuple[str, str] | None:
    if rec == "aclnn_mixed_quant_sparse_flash_mla" or imp == "aclnnMixedQuantSparseFlashMla":
        return ("mixed_quant_sparse_mla", "aclnnMixedQuantSparseFlashMla")
    if rec in {"sparse_flash_attention", "dsa_sparse_attention"} or imp == "SparseFlashAttention":
        return ("sparse_flash_attention", "aclnnSparseFlashAttention")
    if rec in {
        "ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35",
        "latent_attention0", "latent_attention_batch", "paired_mla_attention_analytic",
    } or "mla" in rec:
        return ("absorbed_mla", "aclnnFusedInferAttentionScoreV4")
    if str(meta_dict.get("cache_layout") or "") == "int8_fa3" or rec in {"gbsa_attention", "generic_block_sparse_attention"}:
        return ("block_sparse_attention_c8", "aclnnGenericBlockSparseAttention")
    if rec == "flash_attention_decode":
        return ("incre_flash_attention", "aclnnFusedInferAttentionScoreV3")
    if rec in {"flash_attention", "flash_attention_prefill"}:
        return ("fusion_attention", "aclnnFlashAttentionVarLenScoreV2")
    if rec == "fused_infer_attention" or "flash" in rec or "attention" in rec:
        return ("fused_infer_attention", "aclnnFusedInferAttentionScoreV3")
    return None


def _resolve_indexer_builder(fam: str, rec: str, imp: str) -> tuple[str, str] | None:
    if fam == "lightning_indexer" and rec == "bf16_lightning_indexer_v2":
        return ("lightning_indexer", "aclnnLightningIndexerV2")
    if rec == "cann_pool_key_indexer" or imp == "PoolKeyIndexer":
        return ("pool_key_indexer", "aclnnPoolKeyIndexer")
    if rec in {"quant_lightning_indexer", "index_scores", "index_online_topk", "lightning_indexer"} or "index" in rec:
        return ("lightning_indexer", "aclnnQuantLightningIndexerV2")
    return None


def _resolve_routing_builder(rec: str, meta_dict: Mapping[str, Any]) -> tuple[str, str] | None:
    if rec == "moe_init_routing_v3" or rec == "moe_init_routing":
        return ("moe_init_routing", "aclnnMoeInitRoutingV3")
    if rec == "route_pack":
        return ("moe_token_permute", "aclnnMoeTokenPermute")
    if rec == "router_weight_gather":
        return ("strided_gather", "aclnnGatherElements")
    if rec == "route_unpack":
        return ("megamoe_fused_mc2", "aclnnMegaMoe") if int(meta_dict.get("mc2_mode") or 0) == 3 else (
            "moe_token_unpermute", "aclnnMoeTokenUnpermute"
        )
    return None


def _resolve_recurrent_builder(fam: str, rec: str, imp: str, meta_dict: Mapping[str, Any]) -> tuple[str, str] | None:
    if fam == "gdn_prefill" and rec == "gdn_phase6_arch22":
        return ("chunk_gated_delta_rule", "aclnnChunkGatedDeltaRuleFwd")
    if fam == "recurrent_vector":
        if "recover_ssm" in rec or bool(meta_dict.get("recover_ssm")):
            return ("recurrent_gdn_recover_ssm", "aclnnRecurrentGatedDeltaRule")
        return ("recurrent_gated_delta_rule", "aclnnRecurrentGatedDeltaRule")
    if fam in {"recurrent_scan", "kda_recurrent"}:
        if rec in {"vllm_ascend_recurrent_kda_arch22", "vllm_ascend_recurrent_kda_arch35"} or imp == "RecurrentKda":
            return ("fused_recurrent_kda", "aclnnRecurrentKda")
        if "gdn_state_contract" in meta_dict or "pcp_segment" in meta_dict:
            return ("chunk_gated_delta_rule", "aclnnChunkGatedDeltaRuleFwd")
        if rec.startswith("recurrent_scan") or rec in {"chunk_kda_fwd", "chunk_kda_fwd_arch22"} or fam == "kda_recurrent":
            return ("chunk_kda_fwd", "aclnnChunkKdaFwd")
    if fam == "recurrent_local":
        if rec.startswith("chunk_products"):
            return ("kda_chunk_products", "aclnnChunkKdaFwdPrepare")
        if rec.startswith("wy_transform"):
            return ("kda_wy_transform", "recompute_w_u_fwd_kernel")
    if fam == "recurrent_output" and rec.startswith("recurrent_output"):
        return ("kda_chunk_output", "aclnnChunkKdaFwdFinalize")
    return None


def _resolve_mhc_builder(rec: str) -> tuple[str, str] | None:
    if rec in {"hc_pre_fused", "hc_pre", "hc_pre_mix"}:
        return ("hc_pre", "aclnnHcPre")
    if rec in {"hc_post_fused", "hc_post", "hc_post_mix"}:
        return ("hc_post", "aclnnHcPost")
    if rec in {"hc_combine_norm", "hc_combine_rmsnorm"}:
        return ("hc_combine_norm", "_hc_combine_norm_kernel")
    if rec == "mc2_routed_expert_program":
        return ("megamoe_fused_mc2", "aclnnMegaMoe")
    if rec in {"hc_inverse_rms", "hc_normalize_scale_bias_gate", "hc_sinkhorn", "hc_gate_mix", "hc_combine"}:
        return ("mhc_stream_mix", f"mhc_{rec}_kernel")
    return None


def _resolve_elementwise_builder(
    rec: str, meta_dict: Mapping[str, Any] | None = None,
) -> tuple[str, str] | None:
    if rec == "short_convolution":
        return ("causal_conv1d_update", "aclnnCausalConv1dUpdate")
    if rec == "delta_gates":
        return ("fused_gdn_gating", "fused_gdn_gating_kernel")
    if rec == "recover_ssm_commit":
        return ("ssm_state_scatter", "_scatter_ssm_states_kernel")
    if rec in {"decode_q_l2norm", "decode_k_l2norm", "phase6_q_l2norm", "phase6_k_l2norm", "qk_l2norm"} or "l2norm" in rec:
        return ("l2_norm", "l2norm_fwd_kernel2")
    if rec == "linear_output_norm_gate":
        return ("rms_norm_gated", "_layer_norm_fwd_1pass_kernel_npu")
    if (
        rec in _ELEMENTWISE_ADD_RECIPES
        or rec.endswith(("_merger_bias", "_gamma_shift"))
        or bool((meta_dict or {}).get("weight_offset"))
    ):
        return ("vector_add", "aclnnAdd")
    if rec in {"muls_add", "moe_muls_add"}:
        return ("muls_add", "muls_add_kernel")
    if rec in _ELEMENTWISE_MUL_GATE_RECIPES:
        return ("vector_mul_gate", "aclnnMul")
    if rec in {"gdn_packed_prepare", "candidate_blocks", "candidate_pack"}:
        return ("vector_pack_indices", "aclnnIndexSelect")
    if rec == "dcp_query_pack":
        return ("dcp_query_pack", "_q_gather_prep_head_major_kernel")
    if rec == "dcp_output_pack":
        return ("dcp_output_pack", "_pack_dcp_output_lse_kernel")
    return None


def resolve_abi_backend(builder: str, upstream_abi: str, kernel_backend: str = "") -> str:
    """Resolve one canonical ``HARDWARE_USABLE_ABI_BACKENDS`` tag."""
    kb = str(kernel_backend or "").strip().lower()
    if kb in HARDWARE_USABLE_ABI_BACKENDS:
        return kb
    if "triton" in kb:
        return "triton"
    if "catlass" in kb:
        return "catlass"
    if "hccl" in kb or "mc2" in kb:
        return "hccl_mc2"
    if "torch_npu" in kb:
        return "torch_npu"
    if "csrc" in kb or "vllm_ascend" in kb:
        return "vllm_ascend_csrc"
    if builder in COLLECTIVE_BUILDERS or str(upstream_abi).lower().startswith(("hccl", "hcom")):
        return "hccl_mc2"
    low_abi = str(upstream_abi or "").lower()
    if any(marker in low_abi for marker in _TRITON_ABI_MARKERS):
        return "triton"
    if any(low_abi.startswith(marker) for marker in _TORCH_NPU_ABI_MARKERS):
        return "torch_npu"
    return "cann_aclnn"


def resolve_fabric_collective_abi(collective_op: str) -> tuple[str, str]:
    """Resolve ``(builder_id, upstream_abi)`` for a fabric collective."""
    try:
        return _schema_resolve_fabric_collective_abi(collective_op)
    except ModelingError as exc:
        raise BenchTopologyError(str(exc)) from exc


_ONBOARDING_FAMILY_DEFAULTS: Mapping[str, tuple[str, str]] = {
    "attention": ("attention", "latent_attention0"),
    "indexer": ("indexer", "lightning_indexer"),
    "sparse_indexer": ("indexer", "lightning_indexer"),
    "lightning_indexer": ("indexer", "lightning_indexer"),
    "rmsnorm": ("rmsnorm", "rmsnorm"),
    "add_rmsnorm": ("add_rmsnorm", "add_rmsnorm"),
    "layernorm": ("layernorm", "layernorm"),
    "swiglu": ("swiglu", "swiglu"),
    "rotary": ("elementwise", "rotary"),
    "gemm": ("gemm", "linear"),
    "matmul": ("gemm", "linear"),
    "linear": ("gemm", "linear"),
    "grouped_gemm": ("grouped_gemm", "grouped_matmul"),
    "routing": ("moe_init_routing", "moe_init_routing_v3"),
    "moe_init_routing": ("moe_init_routing", "moe_init_routing_v3"),
    "kv_compress_epilog": ("kv_compress_epilog", "kv_store"),
    "attention_merge": ("attention_merge", "dcp_lse_merge"),
    "dcp_allgather": ("dcp_allgather", "dcp_allgather"),
    "allgather": ("allgather", "allgather"),
    "allreduce": ("allreduce", "allreduce"),
    "alltoall": ("alltoall", "alltoall"),
    "reduce_scatter": ("reduce_scatter", "reduce_scatter"),
    "dcp_query_pack": ("elementwise", "dcp_query_pack"),
    "dcp_output_pack": ("elementwise", "dcp_output_pack"),
}

_BUILDER_REGISTRY_ABI_IDS: Mapping[str, tuple[str, ...]] = {
    "absorbed_mla": ("aclnnFusedInferAttentionScore",),
    "lightning_indexer": ("aclnnLightningIndexer", "aclnnQuantLightningIndexerV2"),
    "rms_norm": ("torch_npu.npu_rms_norm",),
    "add_rms_norm": ("torch_npu.npu_add_rms_norm", "AddRmsNormBias"),
    "layer_norm": ("_layer_norm_fwd_1pass_kernel_npu",),
    "swiglu": ("aclnnSwiGlu",),
    "rotary_embedding": ("torch_npu.npu_rotary_mul",),
    "matmul": ("aclnnMatmul",),
    "grouped_matmul": ("aclnnGroupedMatmul",),
    "moe_init_routing": ("torch_npu.npu_moe_init_routing_v2",),
    "kv_compress_epilog": ("aclnnKvRmsNormRopeCacheV2",),
    "dcp_lse_attention_merge": ("aclnnRingAttentionUpdateV2",),
    "all_gather": ("hcom_allGather",),
    "all_reduce": ("hcom_allReduce",),
    "all_to_all": ("hcom_alltoall", "hcom_alltoallv"),
    "all_to_all_v": ("hcom_alltoallv", "hcom_alltoall"),
    "reduce_scatter": ("hcom_reduceScatter",),
    "dcp_query_pack": ("_q_gather_prep_head_major_kernel",),
    "dcp_output_pack": ("_pack_dcp_output_lse_kernel", "_pack_dcp_output_lse_batched_kernel"),
}


def is_onboarding_family(family: str) -> bool:
    """Return True when ``family`` has a canonical onboarding builder/ABI mapping."""
    return str(family or "").strip() in _ONBOARDING_FAMILY_DEFAULTS


def resolve_onboarding_builder_and_abi(
    arch: str,
    family: str,
    recipe: str = "",
    *,
    prefer_quant_indexer: bool = False,
) -> tuple[str, str, str, str]:
    """Resolve ``(pfam, prec, builder, probed_abi_symbol)`` via canonical builder + ProbedAbiRegistry."""
    fam_key = str(family or "").strip()
    pfam, default_rec = _ONBOARDING_FAMILY_DEFAULTS.get(fam_key, (fam_key, str(recipe or fam_key)))
    prec = str(recipe or "").strip() or default_rec
    resolve_rec = "all_to_all" if (fam_key == "alltoall" and prec == "alltoall") else prec
    builder, upstream_abi = resolve_builder_strict(pfam, resolve_rec, "", ())
    cands = _BUILDER_REGISTRY_ABI_IDS.get(builder, (upstream_abi,))
    if builder == "lightning_indexer" and prefer_quant_indexer:
        cands = ("aclnnQuantLightningIndexerV2", "aclnnLightningIndexer")
    reg = default_abi_registry()
    for abi_id in cands:
        hits = reg.lookup(abi_id, arch, available_only=True)
        if hits:
            return pfam, default_rec, builder, str(hits[0].symbol_or_kernel)
    raise UnmodeledCapabilityError(
        capability="profiler_abi",
        entry=f"{arch}:{fam_key}",
        code="profiler_abi_unavailable",
        message=f"profiler_abi_unavailable:arch={arch}:family={fam_key}:candidates={list(cands)}",
    )


def attention_shape_dims(
    raw_meta: Mapping[str, Any],
    dims: Sequence[int],
    *,
    sequences: int = 1,
    kv_length: int | None = None,
    phase: str = "decode",
) -> dict[str, int]:
    """Extract caller-faithful attention shape with explicit sequences, q_len, and kv_len."""
    seqs_val = max(1, int(sequences or 1))
    is_prefill = 1 if str(phase) == "prefill" else 0
    gqa = raw_meta.get("gqa_decode") or raw_meta.get("gqa_prefill")
    if isinstance(gqa, Mapping):
        reqs = gqa.get("requests") or ()
        s_cnt = len(reqs) if reqs else seqs_val
        tokens = sum(int(q) for q, _ in reqs) if reqs else (int(dims[0]) if dims else 0)
        seq_len = max((int(q) + int(c) for q, c in reqs), default=max(1, int(kv_length or tokens)))
        q_len = max((int(q) for q, _ in reqs), default=max(1, tokens // max(1, s_cnt)))
        heads = int(gqa.get("query_heads") or 1)
        return {
            "tokens": tokens, "sequences": s_cnt, "q_len": q_len, "kv_len": seq_len,
            "is_prefill": is_prefill, "heads": heads,
            "kv_heads": int(gqa.get("kv_heads") or heads),
            "seq_len": seq_len, "head_dim": int(gqa.get("head_dim") or DEFAULT_HEAD_DIM),
        }
    mla = raw_meta.get("paged_mla_geometry")
    if isinstance(mla, Mapping):
        kvs, ctxs = mla.get("kv_lengths") or (), mla.get("contexts") or ()
        s_cnt = len(kvs) or len(ctxs) or seqs_val
        tokens = int(dims[0]) if dims else s_cnt
        heads = int(mla.get("logical_heads") or mla.get("heads") or 1)
        seq_len = (
            max((int(v) for v in kvs), default=1) if kvs
            else max((int(c) + 1 for c in ctxs), default=max(1, int(kv_length or 1)))
        )
        abi = raw_meta.get("native_attention_abi")
        latent = int(abi.get("latent_width") or MLA_LATENT_DIM) if isinstance(abi, Mapping) else MLA_LATENT_DIM
        return {
            "tokens": tokens, "sequences": s_cnt, "q_len": max(1, tokens // max(1, s_cnt)),
            "kv_len": seq_len, "is_prefill": is_prefill, "heads": heads,
            "kv_heads": 1, "seq_len": seq_len, "head_dim": latent,
        }
    idx = raw_meta.get("indexed_attention_geometry")
    if isinstance(idx, Mapping):
        seqs = idx.get("sequences") or ()
        s_cnt = len(seqs) if seqs else seqs_val
        tokens = sum(int(q) for q, _ in seqs) if seqs else int(idx.get("query") or (dims[0] if dims else 0))
        seq_len = (
            max((int(q) + int(c) for q, c in seqs), default=1) if seqs
            else max(1, int(idx.get("query") or 0) + int(idx.get("context") or 0))
        )
        q_len = max((int(q) for q, _ in seqs), default=max(1, tokens // max(1, s_cnt)))
        return {
            "tokens": tokens, "sequences": s_cnt, "q_len": q_len, "kv_len": seq_len,
            "is_prefill": is_prefill, "heads": int(idx.get("heads") or 1),
            "kv_heads": 1, "seq_len": seq_len, "head_dim": int(idx.get("latent") or MLA_LATENT_DIM),
        }
    tv = raw_meta.get("attention_tensor_view")
    if isinstance(tv, Mapping):
        q_sh = tuple(int(x) for x in (tv.get("query_shape") or ()) if isinstance(x, int))
        k_sh = tuple(int(x) for x in (tv.get("key_shape") or tv.get("cache_head_shape") or ()) if isinstance(x, int))
        kvs = tv.get("kv_lengths") or ()
        s_cnt = len(kvs) if kvs else seqs_val
        tokens = q_sh[0] if q_sh else (int(dims[0]) if dims else 0)
        heads = q_sh[1] if len(q_sh) >= 2 else 1
        head_dim = q_sh[-1] if len(q_sh) >= MIN_Q_VIEW_RANK_3D else DEFAULT_HEAD_DIM
        kv_heads = k_sh[KV_HEAD_AXIS] if len(k_sh) >= 2 else 1
        seq_len = (
            max((int(v) for v in kvs), default=1) if kvs else (
                k_sh[0] if len(k_sh) == MIN_Q_VIEW_RANK_3D else (
                    int(kv_length) if kv_length is not None else (
                        int(dims[1]) if len(dims) >= MIN_Q_VIEW_RANK_4D else max(1, tokens)
                    )
                )
            )
        )
        return {
            "tokens": tokens, "sequences": s_cnt, "q_len": max(1, tokens // max(1, s_cnt)),
            "kv_len": seq_len, "is_prefill": is_prefill, "heads": heads,
            "kv_heads": kv_heads, "seq_len": seq_len, "head_dim": head_dim,
        }
    tokens = int(dims[0]) if dims else 0
    heads = int(raw_meta.get("heads") or (
        dims[2] if len(dims) >= MIN_Q_VIEW_RANK_4D else (dims[1] if len(dims) > 1 else 1)
    ))
    seq_len = (
        int(kv_length) if kv_length is not None else (
            int(dims[1]) if len(dims) >= MIN_Q_VIEW_RANK_4D
            else int(raw_meta.get("seq_len") or (dims[2] if len(dims) > 2 else max(1, tokens)))
        )
    )
    return {
        "tokens": tokens, "sequences": seqs_val, "q_len": max(1, tokens // seqs_val),
        "kv_len": seq_len, "is_prefill": is_prefill, "heads": heads,
        "kv_heads": int(raw_meta.get("key_heads") or heads), "seq_len": seq_len,
        "head_dim": int(dims[3]) if len(dims) > MIN_Q_VIEW_RANK_3D else int(raw_meta.get("head_dim") or DEFAULT_HEAD_DIM),
    }


def _inter_link_groups(links: Sequence[Any], rank_node: Mapping[int, int]) -> list[list[list[int]]]:
    strict_groups: list[list[list[int]]] = []
    generic_groups: list[list[list[int]]] = []
    for lnk in links:
        text = (
            f"{lnk.get('channel', '')} {lnk.get('name', '')}" if isinstance(lnk, Mapping)
            else f"{getattr(lnk, 'channel', '')} {getattr(lnk, 'name', '')}"
        ).lower()
        raw_members = lnk.get("members", ()) if isinstance(lnk, Mapping) else getattr(lnk, "members", ())
        members = sorted({int(r) for r in (raw_members or ()) if int(r) in rank_node})
        nids = sorted({rank_node[r] for r in members})
        if len(nids) < 2:
            continue
        grouped = [[r for r in members if rank_node[r] == nid] for nid in nids]
        if any(m in text for m in _ROCE_STRICT_MARKERS):
            strict_groups.append(grouped)
        elif any(m in text for m in _CROSS_NODE_LINK_MARKERS):
            generic_groups.append(grouped)
    return strict_groups + generic_groups


def _machine_topology(
    hardware: Mapping[str, Any], machine: Any | None,
) -> tuple[list[list[int]], list[list[int]], list[list[list[int]]], bool, set[int] | None]:
    if machine is not None:
        dies = list(getattr(machine, "dies", ()) or ())
        node_ids = sorted({int(getattr(d, "node", 0)) for d in dies})
        pkg_keys = sorted({(int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) for d in dies})
        by_node = [sorted(int(d.rank) for d in dies if int(getattr(d, "node", 0)) == n) for n in node_ids]
        by_pkg = [
            sorted(int(d.rank) for d in dies if (int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) == k)
            for k in pkg_keys
        ]
        rank_node = {int(d.rank): int(getattr(d, "node", 0)) for d in dies}
        inter = _inter_link_groups(list(getattr(machine, "links", ()) or ()), rank_node)
        return by_node, by_pkg, inter, bool(inter), {int(d.rank) for d in dies}
    raw_by_node = hardware.get("ranks_by_node")
    raw_by_pkg = hardware.get("package_ranks")
    raw_ranks = hardware.get("ranks")
    avail = {int(r) for r in raw_ranks} if raw_ranks is not None else None
    if raw_by_node is not None:
        by_node = [[int(r) for r in grp] for grp in raw_by_node]
        by_pkg = [[int(r) for r in grp] for grp in (raw_by_pkg or ())]
        rank_node = {r: idx for idx, grp in enumerate(by_node) for r in grp}
        declared_links = hardware.get("declared_links")
        inter = _inter_link_groups(declared_links, rank_node) if declared_links is not None else []
        has_inter = bool(inter) if declared_links is not None else bool(hardware.get("has_inter_node_link", len(by_node) >= 2))
        if avail is None:
            avail = {r for grp in by_node for r in grp}
        return by_node, by_pkg, inter, has_inter, avail
    return [], [], [], bool(hardware.get("has_inter_node_link", True)), avail


def fabric_route_ranks(
    route: str,
    participants: int,
    hardware: Mapping[str, Any],
    *,
    available_ranks: Sequence[int] | None = None,
    machine: Any | None = None,
) -> list[int]:
    """Construct verified physical rank IDs guaranteed to traverse ``route``."""
    route_key = str(route or "").strip()
    if route_key not in SUPPORTED_FABRIC_ROUTES:
        raise BenchGeometryError(f"fabric_route_unknown:{route_key}")
    if isinstance(participants, bool) or not isinstance(participants, int) or participants < 2:
        raise BenchGeometryError(f"fabric_participants_invalid:{participants}")
    raw_pkg = int(machine.dies_per_package) if machine is not None else hardware.get("dies_per_package")
    raw_node = int(machine.ranks_per_node) if machine is not None else hardware.get("dies_per_node")
    if isinstance(raw_pkg, bool) or not isinstance(raw_pkg, int) or raw_pkg < 1:
        raise BenchGeometryError(f"fabric_dies_per_package_invalid:{raw_pkg}")
    if isinstance(raw_node, bool) or not isinstance(raw_node, int) or raw_node < raw_pkg:
        raise BenchGeometryError(f"fabric_dies_per_node_invalid:{raw_node}:per_pkg={raw_pkg}")
    if raw_node % raw_pkg != 0:
        raise BenchGeometryError(f"fabric_node_not_divisible_by_package:dies_per_node={raw_node},dies_per_package={raw_pkg}")
    by_node, by_pkg, inter_groups, has_inter, declared_avail = _machine_topology(hardware, machine)
    ranks = _select_fabric_route_ranks(
        route_key, participants, raw_pkg, raw_node, by_node, by_pkg, inter_groups, has_inter, hardware,
    )
    effective_avail = {int(r) for r in available_ranks} if available_ranks is not None else declared_avail
    if effective_avail is not None:
        missing = [r for r in ranks if r not in effective_avail]
        if missing:
            raise BenchGeometryError(f"fabric_route_ranks_unavailable:route={route_key},missing={missing}")
    return ranks


def _select_fabric_route_ranks(
    route_key: str,
    participants: int,
    raw_pkg: int,
    raw_node: int,
    by_node: list[list[int]],
    by_pkg: list[list[int]],
    inter_groups: list[list[list[int]]],
    has_inter: bool,
    hardware: Mapping[str, Any],
) -> list[int]:
    if route_key == "intra_package_die":
        if raw_pkg < 2 or participants > raw_pkg:
            raise BenchGeometryError(f"fabric_intra_package_unrealizable:participants={participants},dies_per_package={raw_pkg}")
        if by_pkg:
            pkg = next((grp for grp in by_pkg if len(grp) >= participants), None)
            if pkg is None:
                raise BenchGeometryError(f"fabric_intra_package_unrealizable:participants={participants}")
            return list(pkg[:participants])
        return list(range(participants))
    if route_key == "intra_node_hccs":
        if participants > raw_node:
            raise BenchGeometryError(f"fabric_intra_node_exceeds_node:participants={participants},dies_per_node={raw_node}")
        packages_per_node = raw_node // raw_pkg
        if packages_per_node < 2:
            raise BenchGeometryError(f"fabric_intra_node_hccs_requires_multiple_packages:packages={packages_per_node}")
        if by_node:
            node0 = by_node[0]
            if len(node0) < participants:
                raise BenchGeometryError(f"fabric_intra_node_exceeds_node:participants={participants},node_dies={len(node0)}")
            node0_set = set(node0)
            node0_pkgs = [grp for grp in by_pkg if grp and grp[0] in node0_set]
            return [grp[0] for grp in node0_pkgs[:participants]] if len(node0_pkgs) >= participants else list(node0[:participants])
        return [idx * raw_pkg for idx in range(participants)] if participants <= packages_per_node else list(range(participants))
    declared_nodes = len(by_node) if by_node else int(hardware.get("nodes") or 1)
    if declared_nodes < 2 or not has_inter:
        raise BenchTopologyError(f"fabric_inter_node_requires_multi_node:nodes={declared_nodes}")
    half, rem = participants // 2, participants - (participants // 2)
    if inter_groups:
        chosen_pair = next(
            ((ga, gb) for grp in inter_groups for i, ga in enumerate(grp)
             for gb in grp[i + 1 :] if len(ga) >= half and len(gb) >= rem),
            None,
        )
        if chosen_pair is None:
            raise BenchGeometryError(f"fabric_inter_node_half_exceeds_node:half={rem}")
        return [*chosen_pair[0][:half], *chosen_pair[1][:rem]]
    if by_node:
        if len(by_node[0]) < half or len(by_node[1]) < rem:
            raise BenchGeometryError(f"fabric_inter_node_half_exceeds_node:half={rem},node_dies={len(by_node[0])}")
        return [*by_node[0][:half], *by_node[1][:rem]]
    if rem > raw_node:
        raise BenchGeometryError(f"fabric_inter_node_half_exceeds_node:half={rem},dies_per_node={raw_node}")
    return [*range(half), *range(raw_node, raw_node + rem)]


def is_ladder_kv_node(node: Mapping[str, Any]) -> bool:
    fam = str(node.get("family") or "")
    if fam in LADDER_KV_FAMILIES:
        return True
    meta = dict((node.get("operator") or {}).get("metadata") or {})
    return str(meta.get("group_axis") or "") in {"cp", "pcp"}


def resolve_collective_cosets(
    topology: Any, metadata: Mapping[str, Any], stage: int = 0,
) -> list[list[int]]:
    """Resolve all physical rank cosets across every internal and ``external_dp`` replica."""
    if topology is None:
        raise BenchTopologyError("collective_topology_missing")
    raw_axis = str(metadata.get("group_axis") or "tp").strip()
    axis = "cp" if raw_axis == "dcp" else raw_axis
    stage_idx = int(stage)
    parallel = getattr(topology, "parallel", None)
    dp = max(1, int(getattr(parallel, "dp", 1) or 1))
    ext_dp = max(1, int(getattr(parallel, "external_dp", 1) or 1))
    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    total_dp = ext_dp * dp
    if axis == "dp":
        return _resolve_dp_cosets(topology, ext_dp, dp, stage_idx)
    if axis == "pp":
        return _resolve_pp_cosets(topology, total_dp, pp)
    from mindie_sim.prediction.sharding.local_tp import expert_tp_groups

    seen: set[tuple[int, ...]] = set()
    cosets: list[list[int]] = []
    for replica in range(total_dp):
        try:
            if not topology.ranks(replica, stage_idx):
                raise BenchTopologyError(f"collective_topology_missing_ranks:axis={axis},stage={stage_idx},replica={replica}")
            groups = (
                expert_tp_groups(topology, replica=replica, stage=stage_idx)
                if axis == "etp"
                else topology.groups(axis, replica, stage_idx, metadata=metadata)
            )
        except BenchTopologyError:
            raise
        except Exception as exc:
            raise BenchTopologyError(
                f"collective_topology_resolution_failed:axis={axis},stage={stage_idx},replica={replica}:{exc}"
            ) from exc
        for grp in groups:
            key = tuple(int(r) for r in grp)
            if not key:
                raise BenchTopologyError(f"collective_topology_empty_group:axis={axis},stage={stage_idx},replica={replica}")
            if key not in seen:
                seen.add(key)
                cosets.append(list(key))
    if not cosets:
        raise BenchTopologyError(f"collective_topology_no_groups:axis={axis},stage={stage_idx}")
    return cosets


def _resolve_dp_cosets(topology: Any, ext_dp: int, dp: int, stage_idx: int) -> list[list[int]]:
    cosets: list[list[int]] = []
    for ext in range(ext_dp):
        try:
            per_rep = [tuple(int(r) for r in topology.ranks(ext * dp + r, stage_idx)) for r in range(dp)]
        except Exception as exc:
            raise BenchTopologyError(f"collective_topology_dp_failed:stage={stage_idx}:{exc}") from exc
        if not per_rep or not per_rep[0]:
            raise BenchTopologyError(f"collective_topology_dp_empty:stage={stage_idx},ext={ext}")
        width = len(per_rep[0])
        if any(len(row) != width for row in per_rep):
            raise BenchTopologyError(f"collective_topology_dp_ragged:stage={stage_idx},ext={ext}")
        cosets.extend([per_rep[r][off] for r in range(dp)] for off in range(width))
    return cosets


def _resolve_pp_cosets(topology: Any, total_dp: int, pp: int) -> list[list[int]]:
    cosets: list[list[int]] = []
    try:
        for replica in range(total_dp):
            per_stage = [tuple(int(r) for r in topology.ranks(replica, s)) for s in range(pp)]
            if not per_stage or not per_stage[0]:
                raise BenchTopologyError(f"collective_topology_pp_empty:replica={replica}")
            width = len(per_stage[0])
            if any(len(row) != width for row in per_stage):
                raise BenchTopologyError(f"collective_topology_pp_ragged:replica={replica}")
            cosets.extend([per_stage[s][off] for s in range(pp)] for off in range(width))
    except BenchTopologyError:
        raise
    except Exception as exc:
        raise BenchTopologyError(f"collective_topology_pp_failed:{exc}") from exc
    return cosets


def resolve_collective_group_ranks(
    topology: Any, metadata: Mapping[str, Any], stage: int = 0, *, rank: int | None = None,
) -> list[int] | None:
    """Return the coset matching declared ``group_ranks`` or physical ``rank``."""
    if topology is None:
        return None
    cosets = resolve_collective_cosets(topology, metadata, stage)
    declared = metadata.get("group_ranks")
    meta_rank = metadata.get("physical_rank")
    if meta_rank is None:
        meta_rank = metadata.get("rank")
    if declared is not None:
        members = tuple(int(r) for r in declared)
        matched = next((grp for grp in cosets if len(grp) == len(members) and set(grp) == set(members)), None)
        if matched is None:
            raise BenchTopologyError(f"collective_declared_group_unmatched:declared={list(members)},cosets={cosets}")
        if meta_rank is not None and int(meta_rank) not in matched:
            raise BenchTopologyError(f"collective_metadata_rank_not_in_declared_group:rank={meta_rank},declared={list(members)}")
        if rank is not None and int(rank) not in matched:
            raise BenchTopologyError(f"collective_rank_not_in_declared_group:rank={rank},declared={list(members)}")
        return list(matched)
    target = int(rank) if rank is not None else int(meta_rank if meta_rank is not None else 0)
    chosen = next((grp for grp in cosets if target in grp), None)
    if chosen is None:
        raise BenchTopologyError(f"collective_rank_not_in_cosets:rank={target},stage={stage},cosets={cosets}")
    return list(chosen)


def collapse_device_kernel_groups(
    nodes: Sequence[Mapping[str, Any]],
    subgraphs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse multi-phase ``device_kernel_group`` records into one physical invocation."""
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for node in nodes:
        if str(node.get("family") or "") in CONTROL_FAMILIES:
            continue
        dkg = dict((node.get("operator") or {}).get("metadata") or {}).get("device_kernel_group")
        if isinstance(dkg, Mapping) and dkg.get("id"):
            gkey = (int(node.get("replica", 0)), int(node.get("stage", 0)), int(node.get("layer", -1)), str(dkg["id"]))
            buckets.setdefault(gkey, []).append(node)
    if not buckets:
        return [dict(n) for n in nodes], [dict(s) for s in subgraphs]
    alias: dict[str, str] = {}
    collapsed_by_owner: dict[str, dict[str, Any]] = {}
    removed_ids: set[str] = set()
    for members in buckets.values():
        owner = next(
            (m for m in reversed(members)
             if dict(dict((m.get("operator") or {}).get("metadata") or {}).get("device_kernel_group") or {}).get("exit")),
            members[-1],
        )
        owner_id = str(owner["node_id"])
        member_ids = {str(m["node_id"]) for m in members}
        for mid in member_ids:
            alias[mid] = owner_id
            if mid != owner_id:
                removed_ids.add(mid)
        ext_deps = list(dict.fromkeys(
            str(d) for m in members for d in (m.get("deps") or ()) if str(d) not in member_ids
        ))
        work: dict[str, Any] = {
            k: sum(float((m.get("physical_work") or {}).get(k, 0.0)) for m in members)
            for k in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "link_bytes")
        }
        mflops: dict[str, float] = {}
        rwork: dict[str, float] = {}
        for m in members:
            for dt, val in dict((m.get("physical_work") or {}).get("matrix_flops") or {}).items():
                mflops[str(dt)] = mflops.get(str(dt), 0.0) + float(val)
            for rk, val in dict((m.get("physical_work") or {}).get("resource_work") or {}).items():
                rwork[str(rk)] = rwork.get(str(rk), 0.0) + float(val)
        work.update(matrix_flops=mflops, resource_work=rwork, launches=1)
        op_copy = dict(owner.get("operator") or {})
        meta_copy = dict(op_copy.get("metadata") or {})
        meta_copy.pop("device_kernel_group", None)
        op_copy["metadata"] = meta_copy
        sig = (
            owner["physical_signature"] if len(members) == 1
            else "dkg:" + "|".join(str(m["physical_signature"]) for m in members)
        )
        collapsed_by_owner[owner_id] = {
            **dict(owner), "deps": ext_deps, "operator": op_copy,
            "physical_work": work, "physical_signature": sig,
        }
    out_nodes: list[dict[str, Any]] = []
    for node in nodes:
        nid = str(node["node_id"])
        if nid in removed_ids:
            continue
        rec = dict(collapsed_by_owner.get(nid, node))
        rec["deps"] = list(dict.fromkeys(
            alias.get(str(d), str(d)) for d in (rec.get("deps") or ()) if alias.get(str(d), str(d)) != nid
        ))
        out_nodes.append(rec)
    out_subs = [
        {**dict(sub), "node_ids": list(dict.fromkeys(alias.get(str(nid), str(nid)) for nid in (sub.get("node_ids") or ())))}
        for sub in subgraphs
    ]
    return out_nodes, out_subs


def slot_plan_for(experts: int, ep: int, *, redundant: int = 0) -> Any:
    """Resolve an ``ExpertSlotPlan`` for ``experts`` across ``ep`` ranks."""
    if ep <= 0:
        raise PlanError(f"expert_slot_ep_invalid:{ep}")
    return expert_slot_plan(
        None,
        SimpleNamespace(num_experts=int(experts), index=0),
        SimpleNamespace(ep=int(ep), redundant_experts_per_rank=int(redundant)),
    )


def expert_bank_bytes(model: Any, parallel: Mapping[str, Any], *, slots: Any = None) -> dict[str, Any] | None:
    """Resident routed-expert bank aligned with ``expert_slot_plan`` padded/redundant slots."""
    experts = int(model.num_experts)
    if experts <= 0:
        return None
    ep = int(parallel["ep"])
    redundant = int(parallel.get("redundant_experts_per_rank") or 0)
    resolved = slots if slots is not None else slot_plan_for(experts, ep, redundant=redundant)
    local_base = int(resolved.base_slots_by_rank[0])
    local_extra = int(resolved.redundant_slots_by_rank[0])
    local_total = len(resolved.expert_ids_by_rank[0])
    padded_slots = ep * local_base - experts
    intermediate = int(model.moe_intermediate_size) or max(
        (int(plan.moe_intermediate_size) for plan in model.layer_plans), default=0,
    )
    fmt = str(model.quant.expert_weight_format)
    if fmt not in EXPERT_BYTES_PER_ELEMENT:
        raise PlanError(f"unknown_expert_weight_format:{fmt}")
    width = EXPERT_BYTES_PER_ELEMENT[fmt]
    elements = local_total * ROUTED_EXPERT_MATRIX_FACTOR * int(model.hidden_size) * intermediate
    return {
        "local_experts": local_total,
        "base_local_experts": local_base,
        "redundant_local_experts": local_extra,
        "local_weight_expert_ids": list(resolved.expert_ids_by_rank[0]),
        "padded_expert_slots": padded_slots,
        "expert_count": experts,
        "ep": ep,
        "expert_weight_format": fmt,
        "bytes_per_element": width,
        "matrices_per_expert": ROUTED_EXPERT_MATRIX_FACTOR,
        "resident_expert_bank_bytes": int(elements * width),
        "resident_expert_bank_mib": round(elements * width / MIB_BYTES, 3),
        "scope": "three_routed_matrices_before_scales_and_workspace",
        "basis": "expert_slot_plan_declared_format",
    }


def explicit_ledger(
    expert_rows: Sequence[int],
    topk_ids: Sequence[Sequence[int]],
    ep: int,
    *,
    redundant: int = 0,
    slots: Any = None,
) -> dict[str, Any]:
    """Build exact per-rank integer routing ledger aligned with ``expert_slot_plan``."""
    experts = len(expert_rows)
    if ep <= 0 or experts <= 0:
        raise PlanError(f"explicit_ledger_invalid_geometry:{experts}:{ep}")
    if any(type(v) is not int or v < 0 for v in expert_rows):
        raise PlanError("explicit_ledger_expert_rows_require_nonnegative_integers")
    declared = [int(v) for v in expert_rows]
    observed = [0] * experts
    for ids in topk_ids:
        for raw_id in ids:
            if type(raw_id) is not int or not (0 <= raw_id < experts):
                raise PlanError(f"explicit_ledger_topk_expert_out_of_range:{raw_id}:experts={experts}")
            observed[raw_id] += 1
    if observed != declared:
        raise PlanError(f"explicit_ledger_topk_histogram_mismatch:declared={declared},observed={observed}")
    resolved = slots if slots is not None else slot_plan_for(experts, ep, redundant=redundant)
    if int(resolved.logical_experts) != experts or int(resolved.ranks) != ep:
        raise PlanError(f"explicit_ledger_slot_plan_mismatch:{resolved.logical_experts}:{resolved.ranks}")
    base = int(resolved.base_slots_by_rank[0])
    rank_rows = [[0] * len(resolved.expert_ids_by_rank[r]) for r in range(ep)]
    copy_index: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rank_idx, ids in enumerate(resolved.expert_ids_by_rank):
        for pos, eid in enumerate(ids):
            copy_index[int(eid)].append((rank_idx, pos))
    for expert in range(experts):
        copies = copy_index.get(expert)
        if not copies:
            raise PlanError(f"explicit_ledger_expert_unplaced:{expert}")
        for offset in range(declared[expert]):
            owner_rank, slot_index = copies[offset % len(copies)]
            rank_rows[owner_rank][slot_index] += 1
    local_rows = list(rank_rows[0])
    return {
        "basis": "explicit_token_routing_integer",
        "expert_rows": declared,
        "rank_expert_rows": rank_rows,
        "rank_assignment_rows": [sum(row) for row in rank_rows],
        "local_expert_rows": local_rows,
        "local_weight_expert_ids": list(resolved.expert_ids_by_rank[0]),
        "redundant_local_experts": int(resolved.redundant_slots_by_rank[0]),
        "padded_expert_slots": ep * base - experts,
        "topk_ids": [list(row) for row in topk_ids],
        "group_list": local_rows,
        "group_list_cumulative": list(accumulate(local_rows)),
        "group_list_type": 1,
        "m": sum(local_rows),
    }


def collective_group_size(metadata: Mapping[str, Any], parallel: Mapping[str, Any]) -> int:
    """Resolve the physical group size of a collective from its ``group_axis``."""
    explicit = int(metadata.get("group_size") or metadata.get("participants") or 0)
    if explicit > 0:
        return explicit
    axis = str(metadata.get("group_axis") or "tp")
    tp = int(parallel.get("tp") or 1)
    pcp = int(parallel.get("pcp") or 1)
    dcp = int(parallel.get("dcp") or parallel.get("cp") or 1)
    ep = int(parallel.get("ep") or 1)
    dp = int(parallel.get("dp") or 1)
    pp = int(parallel.get("pp") or 1)
    raw_etp = parallel.get("expert_tp")
    etp = (
        int(raw_etp) if raw_etp is not None
        else (tp if ep <= 1 else (tp // ep if ep <= tp and tp % ep == 0 else 1))
    )
    axis_sizes = {"tp": tp, "ep": ep, "pcp": pcp, "cp": dcp, "dcp": dcp, "dp": dp, "pp": pp, "etp": etp}
    return max(1, int(axis_sizes.get(axis, tp)))


def declared_implementation(metadata: Mapping[str, Any]) -> str:
    """Return an explicitly declared implementation/ABI from metadata, or empty."""
    for key in ("native_kind", "native_implementation", "kernel", "kernel_name", "implementation", "caller_implementation"):
        value = metadata.get(key)
        if isinstance(value, str) and value and value not in {"static", "declared"}:
            return value
    binding = metadata.get("implementation_binding")
    if isinstance(binding, Mapping):
        for key in ("implementation", "selector", "program_recipe"):
            if binding.get(key):
                return str(binding[key])
    return ""


def display_name(family: Any, recipe: Any, fallback: str = "") -> str:
    """Human-readable display label (never certified as an upstream ABI)."""
    return f"{family or 'operator'}:{recipe or fallback}"


def _collective_nbytes(operator: Mapping[str, Any], metadata: Mapping[str, Any], dtype: Any) -> int | None:
    for key in ("nbytes", "payload_bytes", "peer_bytes"):
        value = metadata.get(key)
        if type(value) is int and value > 0:
            return value
    elements = 1
    for value in operator.get("launch_shape") or operator.get("local_shape") or ():
        if type(value) is not int:
            return None
        elements *= value
    return elements * ELEMENT_BYTES.get(str(dtype), 4)


def _v1_shape_dims(
    family: str,
    builder: str,
    operator: Mapping[str, Any],
    metadata: Mapping[str, Any],
    parallel: Mapping[str, Any],
    *,
    raw_meta: Mapping[str, Any] | None = None,
    batch: int = 1,
    kv_length: int | None = None,
    phase: str = "decode",
) -> dict[str, int]:
    local = [int(v) for v in (operator.get("local_shape") or ()) if type(v) is int]
    dims = local or [int(v) for v in (operator.get("launch_shape") or ()) if type(v) is int]
    groups = metadata.get("matrix_group_rows") or ()
    shape: dict[str, Any] = {}
    if builder in {"matmul", "quant_matmul", "all_gather_matmul", "matmul_reduce_scatter"} or (
        family in _DENSE_GEMM_FAMILIES and len(dims) >= 2
    ):
        shape = {"m": dims[0], "n": dims[1], "k": dims[2]} if len(dims) >= 3 else ({"m": dims[0], "k": dims[-1]} if dims else {})
        return {k: int(v) for k, v in shape.items() if v is not None}
    if builder in {"grouped_matmul", "grouped_matmul_swiglu_quant"}:
        shape = {
            "m": dims[0] if dims else 0,
            "n": dims[1] if len(dims) > 1 else 0,
            "k": dims[2] if len(dims) > 2 else 0,
            "e": int(metadata.get("num_experts") or len(groups) or metadata.get("matrix_groups") or 0),
        }
        return {k: int(v) for k, v in shape.items() if v is not None}
    if builder in {
        "fusion_attention", "fused_infer_attention", "incre_flash_attention",
        "absorbed_mla", "sparse_flash_attention", "mixed_quant_sparse_mla", "block_sparse_attention_c8",
    }:
        shape = attention_shape_dims(
            raw_meta if raw_meta is not None else metadata, dims,
            sequences=batch, kv_length=kv_length, phase=phase,
        )
        return {k: int(v) for k, v in shape.items() if v is not None}
    if builder in {"topk", "moe_gating_topk", "moe_gating_topk_hash", "moe_init_routing"}:
        shape = {"tokens": dims[0] if dims else 0, "e": int(metadata.get("num_experts") or 0), "top_k": int(metadata.get("top_k") or 0)}
        return {k: int(v) for k, v in shape.items() if v is not None}
    if builder in COLLECTIVE_BUILDERS:
        shape = {"bytes": _collective_nbytes(operator, metadata, operator.get("dtype")), "group_size": collective_group_size(metadata, parallel)}
        return {k: int(v) for k, v in shape.items() if v is not None}
    if dims:
        shape = {"tokens": dims[0], "width": dims[-1]}
    return {k: int(v) for k, v in shape.items() if v is not None}


def build_v1_node_fields(
    *,
    node_id: str,
    family: Any,
    recipe: Any,
    operator: Mapping[str, Any],
    metadata: Mapping[str, Any],
    deps: Sequence[Any],
    outputs: Sequence[Any],
    parallel: Mapping[str, Any],
    model_id: str = "",
    layer_idx: int = MISSING_LAYER,
    topology: Any = None,
    stage_idx: int = 0,
    replica_idx: int = 0,
    batch: int = 1,
    kv_length: int | None = None,
    phase: str = "decode",
) -> dict[str, Any]:
    """Build one ``mindie-bench-subgraph-node/1`` record preserving ABI separation & cosets."""
    raw_meta = dict(operator.get("metadata") or {})
    dtype = str(operator.get("dtype") or "")
    raw_layout = str(operator.get("layout") or "").strip()
    layout = "contiguous" if raw_layout in {"", "declared"} else raw_layout
    decl_impl = declared_implementation(raw_meta)
    disp_name = display_name(family, recipe, str(recipe or family or "operator"))
    dtypes = [d for d in (dtype, str(raw_meta.get("output_dtype") or ""), str(raw_meta.get("activation_format") or "")) if d]
    builder, upstream_abi = resolve_builder_strict(
        str(family or ""), str(recipe or ""), decl_impl, dtypes, raw_meta,
        model_id=model_id, layer_idx=int(layer_idx), local_shape=tuple(operator.get("local_shape") or ()),
    )
    kind = (
        "boundary" if str(family or "") in CONTROL_FAMILIES
        else ("collective" if builder in COLLECTIVE_BUILDERS else "operator")
    )
    collective = builder in COLLECTIVE_BUILDERS
    topo_meta = {**metadata, **{k: raw_meta[k] for k in ("group_ranks", "physical_rank", "rank") if k in raw_meta}}
    cosets = resolve_collective_cosets(topology, topo_meta, stage_idx) if (collective and topology is not None) else None
    granks: list[int] | None = None
    if cosets is not None:
        if any(k in topo_meta for k in ("group_ranks", "physical_rank", "rank")):
            granks = resolve_collective_group_ranks(topology, topo_meta, stage_idx, rank=None)
        else:
            rep_ranks = tuple(topology.ranks(int(replica_idx), stage_idx))
            if not rep_ranks:
                raise BenchTopologyError(f"collective_replica_has_no_ranks:replica={replica_idx},stage={stage_idx}")
            granks = resolve_collective_group_ranks(topology, topo_meta, stage_idx, rank=int(rep_ranks[0]))
    gsize = len(granks) if granks else (collective_group_size(metadata, parallel) if collective else None)
    shape = _v1_shape_dims(
        str(family or ""), builder, operator, metadata, parallel,
        raw_meta=raw_meta, batch=batch, kv_length=kv_length, phase=phase,
    )
    if collective and gsize is not None:
        shape["group_size"] = int(gsize)
    realization = metadata.get("routing_realization")
    group = (
        list(realization.get("local_expert_rows") or ())
        if isinstance(realization, Mapping) and realization.get("local_expert_rows")
        else metadata.get("matrix_group_rows")
    )
    return {
        "schema": SUBGRAPH_NODE_V1_SCHEMA,
        "node_id": str(node_id),
        "kind": kind,
        "op": str(recipe or family or "unknown"),
        "display_name": disp_name,
        "builder": builder,
        "upstream_abi": upstream_abi,
        "phase": str(phase),
        "dtype": dtype or "fp32",
        "layout": layout,
        "output_dtype": str(metadata.get("output_dtype") or dtype or "fp32"),
        "group_list_type": 1,
        "shape": shape,
        "inputs": [str(v) for v in deps],
        "outputs": [str(v) for v in outputs],
        "implementation": decl_impl or upstream_abi,
        "implementation_source": "metadata" if decl_impl else "builder_upstream_abi",
        "kernel": str(metadata.get("kernel") or metadata.get("native_kind") or upstream_abi),
        "group_list": [int(v) for v in group] if group else None,
        "per_token_scale": bool(metadata["per_token_scale"]) if "per_token_scale" in metadata else None,
        "format_code": FRACTAL_NZ_FORMAT_CODE if layout == "FRACTAL_NZ" else None,
        "nbytes": _collective_nbytes(operator, metadata, dtype) if collective else None,
        "group_size": gsize,
        "group": str(metadata.get("group_axis") or metadata.get("collective") or "tp") if collective else None,
        **({"group_ranks": granks, "group_cosets": cosets} if granks else {}),
    }


def _hardware_machine_direct(sku: str) -> tuple[Hardware, Device]:
    """Materialize a pure ``Hardware`` machine from Ascend profiles with zero model probes (AG014)."""
    resolved_sku = resolve_hardware_sku(sku)
    profile = ascend_profiles().get(resolved_sku)
    if profile is None:
        raise ModelingError(f"unknown_ascend_sku:{sku}")
    dies_per_pkg = max(1, int(profile.get("dies_per_package") or 1))
    n_devices = DEFAULT_TP * dies_per_pkg
    raw_machine = materialize_ascend({
        "sku": resolved_sku,
        "num_devices": n_devices,
        "ranks_per_node": n_devices,
    })
    machine = Hardware.from_mapping(raw_machine)
    return machine, machine.dies[0]


def _capability_tokens(die: Device) -> tuple[str, ...]:
    cap = die.capability
    return tuple(sorted({
        f"preset:{cap.preset}",
        *(f"representation:{i.name}" for i in cap.representations),
        *(f"mode:{i.name}" for i in cap.native_modes),
        *(f"path:{i.name}" for i in cap.paths),
        *(f"runtime:{i}" for i in cap.runtimes),
    }))


def build_hardware_view(sku: str, structure_sku: str | None = None) -> dict[str, Any]:
    """Return a neutral hardware topology summary without probing any catalog model (AG014)."""
    resolved_sku = resolve_hardware_sku(sku)
    machine, die = _hardware_machine_direct(resolved_sku)
    l2_bytes = int(dict(die.buffers).get("l2", 0))
    cold_ring = FABRIC_COLD_RING_FACTOR * l2_bytes
    node_ids = sorted({int(getattr(d, "node", 0)) for d in machine.dies})
    pkg_keys = sorted({(int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) for d in machine.dies})
    rank_node = {int(d.rank): int(getattr(d, "node", 0)) for d in machine.dies}
    has_inter = any(
        len({rank_node[int(r)] for r in lnk.members if int(r) in rank_node}) >= 2
        and any(m in f"{lnk.channel} {lnk.name}".lower() for m in _CROSS_NODE_LINK_MARKERS)
        for lnk in machine.links
    )
    return {
        "sku": resolved_sku,
        "architecture": machine.arch_family,
        "capability_preset": die.capability.preset,
        "dies_per_node": int(machine.ranks_per_node),
        "capability_digest": _short_digest(_capability_tokens(die)),
        "dies_per_package": int(machine.dies_per_package),
        "l2_bytes_per_die": l2_bytes,
        "hbm_bytes_per_die": int(die.memory_bytes),
        "fabric_cold_ring_bytes": cold_ring,
        "structure_identity_sku": resolve_hardware_sku(structure_sku or resolved_sku),
        "nodes": len(node_ids),
        "total_dies": len(machine.dies),
        "ranks": [int(d.rank) for d in machine.dies],
        "ranks_by_node": [sorted(int(d.rank) for d in machine.dies if int(getattr(d, "node", 0)) == n) for n in node_ids],
        "package_ranks": [
            sorted(int(d.rank) for d in machine.dies if (int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) == k)
            for k in pkg_keys
        ],
        "has_inter_node_link": has_inter,
        "declared_links": [
            {"name": lnk.name, "channel": lnk.channel, "members": sorted(int(r) for r in lnk.members)}
            for lnk in machine.links
        ],
        "fabric_cold_ring_8mib_slots": int(math.ceil(cold_ring / FABRIC_SLOT_BYTES)),
        "structure_identity_is_sku_independent": False,
    }


def build_fabric_cases(hardware: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generate physical interconnect probe cases matching realized routes on ``hardware``."""
    routes = [
        r for r in FABRIC_ROUTES
        if (r != "intra_package_die" or int(hardware["dies_per_package"]) >= 2)
        and (r != "intra_node_hccs" or int(hardware["dies_per_node"]) // max(1, int(hardware["dies_per_package"])) >= 2)
        and (r != "inter_node_roce" or (int(hardware.get("nodes") or 1) >= 2 and bool(hardware.get("has_inter_node_link", True))))
    ]
    cases: list[dict[str, Any]] = []
    for collective_op in FABRIC_COLLECTIVES:
        builder, upstream_abi = resolve_fabric_collective_abi(collective_op)
        for route in routes:
            participants = FABRIC_PARTICIPANTS[route]
            ranks = fabric_route_ranks(route, participants, hardware)
            for payload in FABRIC_LADDER_BYTES:
                split: PlanSplit = "heldout" if int(payload) in HELDOUT_FABRIC_BYTES else "train"
                node = {
                    "schema": SUBGRAPH_NODE_V1_SCHEMA,
                    "node_id": "fab0",
                    "kind": "collective",
                    "op": collective_op,
                    "display_name": f"collective:{collective_op}",
                    "builder": builder,
                    "upstream_abi": upstream_abi,
                    "dtype": "fp32",
                    "output_dtype": "fp32",
                    "layout": "contiguous",
                    "inputs": [],
                    "outputs": [],
                    "group_list": None,
                    "group_list_type": 1,
                    "per_token_scale": None,
                    "format_code": None,
                    "nbytes": int(payload),
                    "group_size": participants,
                    "group": route,
                    "group_ranks": ranks,
                    "group_cosets": [ranks],
                    "implementation": upstream_abi,
                    "implementation_source": "builder_upstream_abi",
                    "kernel": upstream_abi,
                    "shape": {"bytes": int(payload), "group_size": participants},
                }
                cases.append({
                    "case_id": _short_digest(("fabric", collective_op, route, payload), "fabric-"),
                    "kind": "fabric",
                    "family": "collective",
                    "recipe": collective_op,
                    "phase": "fabric",
                    "scope": "interconnect",
                    "sku": str(hardware["sku"]),
                    "route": route,
                    "participants": participants,
                    "group_ranks": ranks,
                    "group_cosets": [ranks],
                    "fabric_bytes": int(payload),
                    "work_bytes": int(payload),
                    "stage": "s1_joint",
                    "nodes": [node],
                    "edges": [],
                    "timed_boundary": {"include": ["fab0"]},
                    "occurrence_count": 0,
                    "split": split,
                    "split_reasons": [f"fabric_bytes:{payload}"] if split == "heldout" else [],
                })
    return cases


# ---------------------------------------------------------------------------
# Physics-Derived Sequence Cliffs (100% AG014-Clean + R2-PA-12b-07 Safe)
# ---------------------------------------------------------------------------

def _snap_executable_seq(
    raw_seq: int, *, seq_min: int, seq_max: int, align: int,
) -> int | None:
    if seq_max < seq_min:
        return None
    clamped = max(seq_min, min(seq_max, int(raw_seq)))
    if align <= 1:
        return clamped
    down = (clamped // align) * align
    up = ((clamped + align - 1) // align) * align
    cands = [v for v in (down, up) if seq_min <= v <= seq_max and v > 0]
    if not cands:
        return None
    return min(cands, key=lambda v: (abs(v - clamped), v))


def _neighbor_executable_seqs(
    knee: int, *, seq_min: int, seq_max: int, align: int, delta_override: int | None = None,
) -> tuple[int, ...]:
    step = max(align, delta_override if delta_override is not None else align)
    if step % align != 0:
        step = ((step + align - 1) // align) * align
    snapped_knee = _snap_executable_seq(knee, seq_min=seq_min, seq_max=seq_max, align=align)
    if snapped_knee is None:
        return ()
    out: list[int] = []
    for raw in (snapped_knee - step, snapped_knee, snapped_knee + step):
        val = _snap_executable_seq(raw, seq_min=seq_min, seq_max=seq_max, align=align)
        if val is not None and val not in out:
            out.append(val)
    return tuple(out)


def _estimate_layer_kv_and_ub_bytes(
    model: ModelSpec, parallel: ParallelSpec | ParallelArmSpec,
) -> tuple[float, float, int, int | None]:
    """Derive per-token KV bytes, per-position UB bytes, local KV heads, and sparse top-k from ``ModelSpec``."""
    plans = tuple(model.layer_plans)
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    dcp = max(1, int(getattr(parallel, "dcp", getattr(parallel, "cp", 1)) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    cp_total = max(1, dcp * pcp)
    kv_b_per_tok = 0.0
    ub_b_per_pos = 0.0
    local_kv_heads = 1
    sparse_topk: int | None = None
    kv_dt = str(getattr(getattr(model, "quant", None), "kv_dtype", "") or "bf16")
    kv_elem_b = max(1.0, float(dtype_bytes(kv_dt)))
    pspec = ParallelSpec(
        dp=max(1, int(getattr(parallel, "dp", 1) or 1)),
        tp=tp,
        ep=max(1, int(getattr(parallel, "ep", 1) or 1)),
        pp=max(1, int(getattr(parallel, "pp", 1) or 1)),
        dcp=dcp,
        pcp=pcp,
    )
    probe_blk = DEFAULT_CACHE_BLOCK_SIZE

    for lp in plans:
        attn = str(getattr(lp, "attention", "gqa") or "gqa")
        if attn == "mla":
            latent = max(1, int(getattr(lp, "kv_lora_rank", MLA_LATENT_DIM) or MLA_LATENT_DIM))
            rope = max(0, int(getattr(lp, "qk_rope_head_dim", MLA_ROPE_DIM) or MLA_ROPE_DIM))
            comp_bytes = (latent + rope) * kv_elem_b
            kv_b_per_tok = max(kv_b_per_tok, comp_bytes / float(cp_total))
            heads_loc = max(1, int(getattr(lp, "num_heads", 16) or 16) // tp)
            ub_b_per_pos = max(ub_b_per_pos, float(heads_loc * UB_SCORE_ELEM_BYTES + comp_bytes))
            local_kv_heads = max(local_kv_heads, 1)
        elif attn in {"gqa", "mha", "swa"}:
            n_kv = max(1, int(getattr(lp, "num_kv_heads", 1) or 1))
            h_dim = max(1, int(getattr(lp, "head_dim", DEFAULT_HEAD_DIM) or DEFAULT_HEAD_DIM))
            loc_kv = max(1, n_kv // min(tp, n_kv))
            local_kv_heads = max(local_kv_heads, loc_kv)
            tok_bytes = 2.0 * float(loc_kv * h_dim) * kv_elem_b / float(cp_total)
            kv_b_per_tok = max(kv_b_per_tok, tok_bytes)
            n_q = max(1, int(getattr(lp, "num_heads", loc_kv) or loc_kv))
            heads_loc = max(1, n_q // tp)
            ub_b_per_pos = max(ub_b_per_pos, float(heads_loc * UB_SCORE_ELEM_BYTES) + tok_bytes)
        try:
            c1 = layer_cache_components(model, lp, pspec, probe_blk, capacity_padded=False)
            c2 = layer_cache_components(model, lp, pspec, 2 * probe_blk, capacity_padded=False)
            s1 = sum(float(v) for v in c1.values() if isinstance(v, (int, float)) and v > 0)
            s2 = sum(float(v) for v in c2.values() if isinstance(v, (int, float)) and v > 0)
            delta_per_tok = max(0.0, (s2 - s1) / float(probe_blk))
            if delta_per_tok > 0.0:
                kv_b_per_tok = max(kv_b_per_tok, delta_per_tok)
        except Exception:
            pass
        idx_topk = int(getattr(lp, "index_topk", 0) or 0)
        if bool(getattr(lp, "indexer", False)) or idx_topk > 0:
            sparse_topk = max(sparse_topk or 0, idx_topk or DEFAULT_HISPARSE_TOPK_BUFFER)

    if ub_b_per_pos <= 0.0:
        ub_b_per_pos = float(DEFAULT_HEAD_DIM * UB_SCORE_ELEM_BYTES)
    return kv_b_per_tok, ub_b_per_pos, local_kv_heads, sparse_topk


def _resolve_local_kv_abi_divisor(model: ModelSpec, parallel: ParallelSpec | ParallelArmSpec) -> int:
    """Resolve executable local-KV ABI alignment divisor structurally (R2-PA-12b-07)."""
    dcp = max(1, int(getattr(parallel, "dcp", getattr(parallel, "cp", 1)) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    if dcp * pcp <= 1:
        return 1
    for lp in (model.layer_plans or ()):
        attn = str(getattr(lp, "attention", "") or "").lower()
        if attn in {"mla", "gqa", "mha", "swa"} or bool(getattr(lp, "indexer", False)):
            return DEFAULT_LOCAL_KV_ABI_DIVISOR
    return 1


def derive_physical_seq_cliffs(
    model: ModelSpec,
    device: Device,
    parallel: ParallelSpec | ParallelArmSpec,
    *,
    seq_min: int = DEFAULT_SEQ_MIN,
    seq_max: int = DEFAULT_SEQ_MAX,
    anchor_ladder: Sequence[int] = DEFAULT_ANCHOR_LADDER,
    concurrency: int = 1,
    kv_block_size: int = DEFAULT_CACHE_BLOCK_SIZE,
    cliff_budget_per_arm: int = DEFAULT_CLIFF_BUDGET_PER_ARM,
    auto_physical_cliffs: bool = True,
    sparse_topk_buffer_size: int | None = None,
    local_kv_abi_divisor: int = 1,
    include_abi_rejection: bool = False,
) -> tuple[SeqCliffPoint, ...]:
    """[Cohesive-Pipeline] Derive 100% AG014-clean physical sequence cliffs with ``R2-PA-12b-07`` ABI divisibility."""
    dcp = max(1, int(getattr(parallel, "dcp", getattr(parallel, "cp", 1)) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    cp_stride = max(1, dcp * pcp)
    abi_div = max(1, int(local_kv_abi_divisor))
    exec_align = cp_stride * abi_div
    blk = max(1, int(kv_block_size))
    batch = max(1, int(concurrency))

    aic = max(1, int(device.units("cube")))
    aiv = max(1, int(device.units("vector")))
    ub_cap = max(1, int(device.pool_capacity("ub")))
    l2_cap = max(1, int(device.pool_capacity("l2")))
    ub_carve = max(0, int(getattr(device, "simt_ub_carveout_bytes", 0) or 0))
    ub_eff = max(1, ub_cap - ub_carve)
    kv_b_per_tok, ub_b_per_pos, local_kv_heads, sparse_topk = _estimate_layer_kv_and_ub_bytes(model, parallel)

    by_seq: dict[int, SeqCliffPoint] = {}
    anchors: list[int] = []
    for raw_a in anchor_ladder:
        if seq_min <= int(raw_a) <= seq_max:
            snapped = _snap_executable_seq(int(raw_a), seq_min=seq_min, seq_max=seq_max, align=exec_align)
            if snapped is not None and snapped not in by_seq:
                anchors.append(snapped)
                by_seq[snapped] = SeqCliffPoint(
                    seq_len=snapped,
                    cliff_kind="anchor",
                    split="train",
                    priority=0,
                    formula_expr=f"anchor_ladder_clamped(align={exec_align})",
                    hw_inputs={"cp_stride": cp_stride, "local_kv_abi_divisor": abi_div},
                    rationale="Standard production context ladder anchor.",
                )
    if not anchors:
        fallback_a = _snap_executable_seq(seq_min, seq_min=seq_min, seq_max=seq_max, align=exec_align)
        if fallback_a is not None:
            by_seq[fallback_a] = SeqCliffPoint(
                seq_len=fallback_a,
                cliff_kind="anchor",
                split="train",
                priority=0,
                formula_expr=f"seq_min_anchor(align={exec_align})",
                hw_inputs={"cp_stride": cp_stride, "local_kv_abi_divisor": abi_div},
                rationale="Range lower-bound executable anchor.",
            )
    if not auto_physical_cliffs or cliff_budget_per_arm <= 0:
        return tuple(sorted(by_seq.values(), key=lambda pt: (pt.seq_len, pt.priority)))

    candidates: list[SeqCliffPoint] = []
    hw_base = {
        "aic_count": aic, "aiv_count": aiv, "ub_eff_bytes": ub_eff,
        "l2_bytes": l2_cap, "cp_stride": cp_stride, "local_kv_heads": local_kv_heads,
        "kv_block_size": blk, "concurrency": batch, "local_kv_abi_divisor": abi_div,
    }

    # 1. Wave quantization cliff (AIC/AIV core waves)
    wave_seq = cp_stride * math.ceil(aic / max(1, batch * local_kv_heads)) * DEFAULT_WAVE_TILE_STRIDE
    for mult in (1, 2):
        for s_val in _neighbor_executable_seqs(mult * wave_seq, seq_min=seq_min, seq_max=seq_max, align=exec_align):
            candidates.append(SeqCliffPoint(
                seq_len=s_val, cliff_kind="wave_cliff", split="heldout", priority=1,
                formula_expr=f"{mult} * dcp*pcp * ceil(AIC / (B*H_kv_loc)) * {DEFAULT_WAVE_TILE_STRIDE}",
                hw_inputs={**hw_base, "wave_seq": wave_seq, "wave_mult": mult},
                rationale="AIC/AIV core-wave tail quantization boundary.",
            ))

    # 2. UB double-buffer reload cliff
    s_ub = int(ub_eff // max(1.0, float(DOUBLE_BUFFER_FACTOR) * ub_b_per_pos)) * cp_stride
    for s_val in _neighbor_executable_seqs(s_ub, seq_min=seq_min, seq_max=seq_max, align=exec_align, delta_override=blk):
        candidates.append(SeqCliffPoint(
            seq_len=s_val, cliff_kind="ub_cliff", split="heldout", priority=2,
            formula_expr="floor(U_eff / (2 * b_ub_per_pos)) * dcp * pcp",
            hw_inputs={**hw_base, "s_ub_raw": s_ub, "ub_bytes_per_pos": round(ub_b_per_pos, 2)},
            rationale="Unified Buffer double-buffered single-pass reload knee.",
        ))

    # 3. Shared L2 cache residency cliff (buffer_units.l2 == 1; skipped for fixed-size recurrent-only models)
    s_l2 = int(l2_cap // max(1.0, float(batch) * kv_b_per_tok)) if kv_b_per_tok > 0.0 else 0
    if kv_b_per_tok > 0.0 and s_l2 > 0:
        for mult_tag, raw_s in (
            ("0.5", int(s_l2 * L2_CLIFF_LOW_RATIO)),
            ("1.0", s_l2),
            ("1.5", int(math.ceil(s_l2 * L2_CLIFF_HIGH_RATIO))),
        ):
            s_val = _snap_executable_seq(raw_s, seq_min=seq_min, seq_max=seq_max, align=exec_align)
            if s_val is not None:
                candidates.append(SeqCliffPoint(
                    seq_len=s_val, cliff_kind="l2_cliff", split="heldout", priority=1,
                    formula_expr=f"{mult_tag} * floor(L2 / (B * b_kv_per_tok))",
                    hw_inputs={**hw_base, "s_l2_raw": s_l2, "kv_bytes_per_tok": round(kv_b_per_tok, 2)},
                    rationale="Shared L2 warm-to-HBM-streaming transition knee.",
                ))

    # 4. Paged KV block-table boundary cliff nearest L2/UB knee
    pivot = max(blk, min(seq_max, s_l2 if (s_l2 > 0 and seq_min <= s_l2 <= seq_max) else s_ub))
    page_edge = max(blk, (pivot // blk) * blk)
    for raw_s in (page_edge - exec_align, page_edge + exec_align):
        s_val = _snap_executable_seq(raw_s, seq_min=seq_min, seq_max=seq_max, align=exec_align)
        if s_val is not None:
            candidates.append(SeqCliffPoint(
                seq_len=s_val, cliff_kind="kv_block_cliff", split="heldout", priority=3,
                formula_expr=f"m * block_size({blk}) +/- align({exec_align})",
                hw_inputs={**hw_base, "page_edge": page_edge},
                rationale="Paged KV cache tail-block allocation transition.",
            ))

    # 5. Sparse indexer top-k & HiSparse host buffer transition cliff
    for label, threshold in (
        ("index_topk", sparse_topk),
        ("hisparse_buffer", sparse_topk_buffer_size),
    ):
        if threshold is None or threshold <= 0:
            continue
        for s_val in _neighbor_executable_seqs(int(threshold), seq_min=seq_min, seq_max=seq_max, align=exec_align):
            candidates.append(SeqCliffPoint(
                seq_len=s_val, cliff_kind="index_pool_cliff", split="heldout", priority=1,
                formula_expr=f"{label}({threshold}) +/- align({exec_align})",
                hw_inputs={**hw_base, label: int(threshold)},
                rationale=f"Sparse {label} dense-to-sparse / host-offload threshold.",
            ))

    # Deduplicate and respect cliff_budget_per_arm while ensuring cliff-kind diversity
    added = 0
    seen_kinds: set[str] = set()
    ordered = sorted(candidates, key=lambda c: (c.priority, c.cliff_kind, c.seq_len))
    # First pass: at least one point per cliff kind
    for cand in ordered:
        if added >= cliff_budget_per_arm:
            break
        if cand.seq_len in by_seq or cand.cliff_kind in seen_kinds:
            continue
        by_seq[cand.seq_len] = cand
        seen_kinds.add(cand.cliff_kind)
        added += 1
    # Second pass: fill remaining budget by priority
    for cand in ordered:
        if added >= cliff_budget_per_arm:
            break
        if cand.seq_len in by_seq:
            continue
        by_seq[cand.seq_len] = cand
        added += 1

    # Optional explicit negative ABI rejection probe when exec_align > 1
    if include_abi_rejection and exec_align > 1:
        rej_seq = min(seq_max, max(seq_min, ((seq_min + exec_align - 1) // exec_align) * exec_align + 1))
        if rej_seq % exec_align != 0 and rej_seq not in by_seq:
            by_seq[rej_seq] = SeqCliffPoint(
                seq_len=rej_seq,
                cliff_kind="kv_block_cliff",
                split="abi_rejection",
                priority=9,
                formula_expr=f"indivisible_probe(align={exec_align})",
                hw_inputs=hw_base,
                rationale="Intentional ABI-indivisible negative probe point.",
            )

    return tuple(sorted(by_seq.values(), key=lambda pt: (pt.seq_len, pt.priority)))


# ---------------------------------------------------------------------------
# Structural Parallel-Arm Enumerator (First-Principles General + Shipping Rule)
# ---------------------------------------------------------------------------

def _model_heads_divisible(model: ModelSpec, tp: int, pcp: int = 1) -> bool:
    for lp in model.layer_plans:
        heads = int(getattr(lp, "num_heads", 0) or 0)
        if heads > 0 and heads % tp != 0:
            return False
        if str(getattr(lp, "attention", "")) == "kda" and heads > 0 and heads % (tp * pcp) != 0:
            return False
    return True


def _model_has_sparse_indexer(model: ModelSpec) -> bool:
    return any(
        bool(getattr(lp, "indexer", False)) or int(getattr(lp, "index_topk", 0) or 0) > 0
        for lp in model.layer_plans
    )


def _validate_arm_for_model(
    model: ModelSpec,
    arm: ParallelArmSpec,
    *,
    feature_track: str = "shipping",
) -> tuple[bool, str]:
    if not _model_heads_divisible(model, arm.tp, arm.pcp):
        return False, f"heads_not_divisible:tp={arm.tp},pcp={arm.pcp}"
    experts = int(getattr(model, "num_experts", 0) or 0)
    if arm.ep > 1 and experts <= 0:
        return False, f"ep_requested_for_dense_model:ep={arm.ep}"
    if experts > 0 and arm.ep > experts:
        return False, f"ep_exceeds_experts:ep={arm.ep},experts={experts}"
    if arm.routed_expert_tp > 1:
        if experts <= 0:
            return False, "routed_expert_tp_requires_moe"
        if str(feature_track).strip().lower() == "shipping":
            return False, "shipping_vllm_ascend_requires_routed_expert_tp_1"
    if arm.sparse_kv_offload and not _model_has_sparse_indexer(model):
        return False, "hisparse_requires_sparse_indexer_model"
    eff_etp = int(arm.routed_expert_tp) if experts > 0 else 1
    try:
        pspec = ParallelSpec(
            dp=arm.dp, tp=arm.tp, ep=arm.ep, pp=arm.pp,
            dcp=arm.dcp, pcp=arm.pcp,
            expert_tp=(eff_etp if experts > 0 else None),
            q_proj_tp=arm.q_proj_tp or 1,
            o_proj_tp=arm.o_proj_tp or 1,
            dcp_linear_tp=arm.dcp_linear_tp or 1,
            shared_expert_tp=arm.shared_expert_tp or 1,
        )
        pspec.validate(model)
    except Exception as exc:
        return False, f"parallel_spec_invalid:{exc}"
    if experts > 0 and str(feature_track).strip().lower() == "shipping" and int(pspec.etp) != 1:
        return False, "shipping_vllm_ascend_requires_routed_expert_tp_1"
    if arm.dcp > 1:
        ok, reason = validate_vllm_ascend_dcp_codomain(model, pspec)
        if not ok:
            return False, f"dcp_codomain_invalid:{reason}"
    return True, "ok"


def _default_production_arms_for_model(
    model: ModelSpec, *, feature_track: str = "shipping", deployment_mode: str = "colocated",
) -> list[ParallelArmSpec]:
    experts = int(getattr(model, "num_experts", 0) or 0)
    candidates: list[ParallelArmSpec] = []
    if experts > 0:
        base_tp = DEFAULT_TP if _model_heads_divisible(model, DEFAULT_TP) else (
            ALT_TP_QUARTER if _model_heads_divisible(model, ALT_TP_QUARTER) else 1
        )
        base_ep = min(experts, max(base_tp, LOCAL_MOE_EP))
        if base_ep % base_tp != 0:
            base_ep = base_tp if base_tp <= experts else 1
        dp_base = max(1, base_ep // max(1, base_tp))
        candidates.append(ParallelArmSpec(
            arm_id=f"tp{base_tp}_ep{base_ep}", dp=dp_base, tp=base_tp, ep=base_ep,
            num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
        ))
        if base_ep > 1:
            candidates.append(ParallelArmSpec(
                arm_id=f"dp{base_ep}_ep{base_ep}", dp=base_ep, tp=1, ep=base_ep,
                num_devices=base_ep, deployment_mode=deployment_mode,
            ))
        if base_tp >= 2:
            dcp_deg = base_tp if _model_has_sparse_indexer(model) else DEFAULT_CP_DEGREE
            candidates.append(ParallelArmSpec(
                arm_id=f"dp_dcp{dcp_deg}_tp{base_tp}_ep{base_ep}",
                dp=dp_base, tp=base_tp, ep=base_ep, dcp=dcp_deg,
                num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
            ))
            candidates.append(ParallelArmSpec(
                arm_id=f"local_tp{base_tp}_sh{base_tp}",
                dp=dp_base, tp=base_tp, ep=base_ep,
                q_proj_tp=base_tp, o_proj_tp=base_tp,
                dcp_linear_tp=base_tp, shared_expert_tp=base_tp,
                num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
            ))
        if base_tp >= ALT_TP_QUARTER and _model_heads_divisible(model, ALT_TP_QUARTER, DEFAULT_CP_DEGREE):
            pcp_ep = min(experts, max(ALT_TP_QUARTER * DEFAULT_CP_DEGREE, base_ep))
            pcp_dp = max(1, pcp_ep // (ALT_TP_QUARTER * DEFAULT_CP_DEGREE))
            candidates.append(ParallelArmSpec(
                arm_id=f"pcp{DEFAULT_CP_DEGREE}_dcp{DEFAULT_CP_DEGREE}_tp{ALT_TP_QUARTER}",
                dp=pcp_dp, tp=ALT_TP_QUARTER, ep=pcp_ep,
                pcp=DEFAULT_CP_DEGREE, dcp=DEFAULT_CP_DEGREE,
                num_devices=pcp_dp * ALT_TP_QUARTER * DEFAULT_CP_DEGREE,
                deployment_mode=deployment_mode,
            ))
        if _model_has_sparse_indexer(model):
            candidates.append(ParallelArmSpec(
                arm_id=f"hisparse_tp{base_tp}_ep{base_ep}",
                dp=dp_base, tp=base_tp, ep=base_ep,
                sparse_kv_offload=True, num_devices=dp_base * base_tp,
                deployment_mode=deployment_mode,
            ))
        if str(feature_track).strip().lower() != "shipping" and base_tp >= 2:
            etp_ep = max(1, base_ep // 2)
            candidates.append(ParallelArmSpec(
                arm_id=f"physics_etp2_ep{etp_ep}",
                dp=max(1, (etp_ep * 2) // base_tp), tp=base_tp, ep=etp_ep,
                routed_expert_tp=2, shared_expert_tp=base_tp,
                num_devices=max(base_tp, etp_ep * 2),
                deployment_mode=deployment_mode,
            ))
    else:
        for tp_cand in (DEFAULT_TP, ALT_TP_QUARTER, 1):
            if _model_heads_divisible(model, tp_cand):
                candidates.append(ParallelArmSpec(
                    arm_id=f"tp{tp_cand}", dp=1, tp=tp_cand, ep=1, num_devices=tp_cand,
                    deployment_mode=deployment_mode,
                ))
                break
        if _model_heads_divisible(model, ALT_TP_QUARTER, DEFAULT_CP_DEGREE):
            candidates.append(ParallelArmSpec(
                arm_id=f"pcp{DEFAULT_CP_DEGREE}_dcp{DEFAULT_CP_DEGREE}_tp{ALT_TP_QUARTER}",
                dp=1, tp=ALT_TP_QUARTER, ep=1, pcp=DEFAULT_CP_DEGREE, dcp=DEFAULT_CP_DEGREE,
                num_devices=ALT_TP_QUARTER * DEFAULT_CP_DEGREE,
                deployment_mode=deployment_mode,
            ))
    return candidates


def _extract_sla_parallel(best: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("decode_parallel", "parallel", "prefill_parallel"):
        cand = best.get(key)
        if isinstance(cand, Mapping) and cand:
            return cand
    if any(k in best for k in ("tp", "dp", "ep", "pp")):
        return best
    return {}


def _solve_auto_sla_arm(
    model_ref: str | Mapping[str, Any],
    sku: str,
    *,
    deployment_mode: str = "colocated",
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    seq_len: int = DEFAULT_SEQ_MIN,
    layers: int | None = None,
    num_devices: int | None = None,
) -> tuple[ParallelArmSpec | None, str]:
    from mindie_sim.sweep.pd import solve_optimal_deployment

    r_sku, topo_preset = resolve_hardware_alias(sku)
    prof = ascend_profiles().get(r_sku) or {}
    default_nd = int(
        num_devices
        or (prof.get("topology") or {}).get("ranks_per_node")
        or (prof.get("device") or {}).get("ranks_per_node")
        or DEFAULT_TP
    )
    solved = solve_optimal_deployment(
        model_ref, r_sku,
        num_devices=max(1, default_nd),
        topology_preset=topo_preset,
        prompt_tokens=max(1, int(seq_len)),
        output_tokens=DEFAULT_OUTPUT_TOKENS_BUDGET,
        ttft_sla_ms=ttft_slo_ms,
        tpot_sla_ms=tpot_slo_ms,
        deployment_mode=deployment_mode,
        layers=layers,
        allow_unconstrained_ttft=(ttft_slo_ms is None),
    )
    best = (
        solved.get("best_configuration")
        or solved.get("best_colocated")
        or solved.get("best_pd")
        or solved.get("best")
        or solved.get("recommended")
        or ((solved.get("pareto_frontier") or [None])[0])
    )
    if not isinstance(best, Mapping):
        reason = str(solved.get("limiting_constraint") or solved.get("status") or "auto_sla_infeasible")
        return None, reason
    par = _extract_sla_parallel(best)
    if not par:
        return None, "auto_sla_missing_parallel_axes"
    dp = max(1, int(par.get("dp", 1) or 1))
    tp = max(1, int(par.get("tp", 1) or 1))
    ep = max(1, int(par.get("ep", 1) or 1))
    pp = max(1, int(par.get("pp", 1) or 1))
    dcp = max(1, int(par.get("dcp", par.get("cp", 1)) or 1))
    pcp = max(1, int(par.get("pcp", 1) or 1))
    nd = max(dp * tp * pp * pcp, int(best.get("used_dies") or default_nd))
    return ParallelArmSpec(
        arm_id=f"auto_sla_dp{dp}_tp{tp}_ep{ep}_pp{pp}",
        dp=dp, tp=tp, ep=ep, pp=pp, dcp=dcp, pcp=pcp, num_devices=nd,
        deployment_mode=deployment_mode, auto_sla=True,
    ), "ok"


def enumerate_parallel_arms(
    model: ModelSpec,
    sku: str,
    requested_arms: Sequence[ParallelArmSpec | str] = (),
    *,
    model_ref: str | Mapping[str, Any] | None = None,
    deployment_mode: str = "colocated",
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    seq_len: int = DEFAULT_SEQ_MIN,
    layers: int | None = None,
    feature_track: str = "shipping",
) -> tuple[tuple[ParallelArmSpec, ...], tuple[dict[str, Any], ...]]:
    """Expand and validate structural parallel arms for ``model`` without AG014 model checks."""
    raw_list: list[ParallelArmSpec] = []
    skipped: list[dict[str, Any]] = []
    explicit_request = bool(requested_arms)
    if not requested_arms:
        raw_list.extend(_default_production_arms_for_model(
            model, feature_track=feature_track, deployment_mode=deployment_mode,
        ))
    else:
        for item in requested_arms:
            if isinstance(item, ParallelArmSpec):
                if item.auto_sla:
                    solved, reason = _solve_auto_sla_arm(
                        model_ref if model_ref is not None else model.to_dict(), sku,
                        deployment_mode=item.deployment_mode or deployment_mode,
                        ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
                        seq_len=seq_len, layers=layers, num_devices=item.num_devices,
                    )
                    if solved is not None:
                        raw_list.append(solved)
                    else:
                        skipped.append({"arm_id": item.arm_id, "reason": reason})
                else:
                    raw_list.append(item)
            else:
                tok = str(item).strip().lower()
                expanded, exp_skip = _expand_named_arm_preset(
                    tok, model, sku,
                    model_ref=model_ref, deployment_mode=deployment_mode,
                    ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
                    seq_len=seq_len, layers=layers, feature_track=feature_track,
                )
                raw_list.extend(expanded)
                skipped.extend(exp_skip)
    valid: list[ParallelArmSpec] = []
    seen_ids: set[str] = set()
    for arm in raw_list:
        ok, reason = _validate_arm_for_model(model, arm, feature_track=feature_track)
        if not ok:
            skipped.append({"arm_id": arm.arm_id, "reason": reason})
            continue
        if arm.arm_id not in seen_ids:
            seen_ids.add(arm.arm_id)
            valid.append(arm)
    if not valid and not explicit_request:
        fallback = ParallelArmSpec(
            arm_id="tp1", dp=1, tp=1, ep=1, pp=1, num_devices=1, deployment_mode=deployment_mode,
        )
        ok, _ = _validate_arm_for_model(model, fallback, feature_track=feature_track)
        if ok:
            valid.append(fallback)
    return tuple(valid), tuple(skipped)


def _expand_named_arm_preset(
    tok: str,
    model: ModelSpec,
    sku: str,
    *,
    model_ref: str | Mapping[str, Any] | None,
    deployment_mode: str,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    seq_len: int,
    layers: int | None,
    feature_track: str,
) -> tuple[list[ParallelArmSpec], list[dict[str, Any]]]:
    if tok == "auto_production":
        return _default_production_arms_for_model(
            model, feature_track=feature_track, deployment_mode=deployment_mode,
        ), []
    if tok == "auto_sla":
        solved, reason = _solve_auto_sla_arm(
            model_ref if model_ref is not None else model.to_dict(), sku,
            deployment_mode=deployment_mode, ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms, seq_len=seq_len, layers=layers,
        )
        return ([solved], []) if solved is not None else ([], [{"arm_id": "auto_sla", "reason": reason}])
    experts = int(getattr(model, "num_experts", 0) or 0)
    base_tp = DEFAULT_TP if _model_heads_divisible(model, DEFAULT_TP) else (
        ALT_TP_QUARTER if _model_heads_divisible(model, ALT_TP_QUARTER) else 1
    )
    base_ep = min(experts, max(base_tp, LOCAL_MOE_EP)) if experts > 0 else 1
    if experts > 0 and base_ep % base_tp != 0:
        base_ep = base_tp if base_tp <= experts else 1
    dp_base = max(1, base_ep // max(1, base_tp)) if experts > 0 else 1
    presets: dict[str, ParallelArmSpec] = {
        "tp_x_ep": ParallelArmSpec(
            arm_id=f"tp{base_tp}_ep{base_ep}", dp=dp_base, tp=base_tp, ep=base_ep,
            num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
        ),
        "dp_x_ep": ParallelArmSpec(
            arm_id=f"dp{max(1, base_ep)}_ep{max(1, base_ep)}", dp=max(1, base_ep), tp=1, ep=max(1, base_ep),
            num_devices=max(1, base_ep), deployment_mode=deployment_mode,
        ),
        "dp_x_dcp": ParallelArmSpec(
            arm_id=f"dp_dcp{base_tp if _model_has_sparse_indexer(model) else DEFAULT_CP_DEGREE}",
            dp=dp_base, tp=base_tp, ep=base_ep,
            dcp=base_tp if _model_has_sparse_indexer(model) else DEFAULT_CP_DEGREE,
            num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
        ),
        "dp_dcp": ParallelArmSpec(
            arm_id=f"dp_dcp{base_tp if _model_has_sparse_indexer(model) else DEFAULT_CP_DEGREE}",
            dp=dp_base, tp=base_tp, ep=base_ep,
            dcp=base_tp if _model_has_sparse_indexer(model) else DEFAULT_CP_DEGREE,
            num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
        ),
        "local_tp": ParallelArmSpec(
            arm_id=f"local_tp{base_tp}_sh{base_tp}",
            dp=dp_base, tp=base_tp, ep=base_ep,
            q_proj_tp=base_tp, o_proj_tp=base_tp, dcp_linear_tp=base_tp, shared_expert_tp=base_tp,
            num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
        ),
        "pcp_x_dcp": ParallelArmSpec(
            arm_id=f"pcp{DEFAULT_CP_DEGREE}_dcp{DEFAULT_CP_DEGREE}_tp{ALT_TP_QUARTER}",
            dp=1, tp=ALT_TP_QUARTER, ep=min(experts, ALT_TP_QUARTER * DEFAULT_CP_DEGREE) if experts > 0 else 1,
            pcp=DEFAULT_CP_DEGREE, dcp=DEFAULT_CP_DEGREE,
            num_devices=ALT_TP_QUARTER * DEFAULT_CP_DEGREE, deployment_mode=deployment_mode,
        ),
        "hisparse": ParallelArmSpec(
            arm_id=f"hisparse_tp{base_tp}_ep{base_ep}",
            dp=dp_base, tp=base_tp, ep=base_ep, sparse_kv_offload=True,
            num_devices=dp_base * base_tp, deployment_mode=deployment_mode,
        ),
    }
    if tok in presets:
        return [presets[tok]], []
    if tok.startswith("tp") and tok[2:].isdigit():
        tval = max(1, int(tok[2:]))
        return [ParallelArmSpec(
            arm_id=f"tp{tval}", dp=1, tp=tval, ep=1, num_devices=tval, deployment_mode=deployment_mode,
        )], []
    raise ModelingError(f"unknown_parallel_arm_preset:{tok!r}")


# ---------------------------------------------------------------------------
# Dual-Peak HBM + HiSparse Host DDR Admissibility Gate (Astra #4)
# ---------------------------------------------------------------------------

def evaluate_hbm_admissibility(
    parsed_case: Case,
    *,
    phase: PhaseKind,
    seq_len: int,
    concurrency: int,
    cell_id: str = "",
    usable_host_ddr_bytes: int | None = None,
) -> tuple[HbmAdmissibilityReceipt, MemoryCalculatorLedger]:
    """Check BOTH ``runtime_peak_hbm`` AND ``load_peak_hbm`` per die plus HiSparse usable host DDR."""
    base_seq = max(1, int(seq_len))
    declared_out = (
        max((int(getattr(req, "output_tokens", 0) or 0) for req in parsed_case.workload.requests), default=0)
        if parsed_case.workload.requests else 0
    )
    resolved_seq = max(base_seq + declared_out, base_seq) if str(phase) == "decode" else base_seq
    resolved_conc = max(1, int(concurrency))
    ws_tok = _resolve_ledger_workspace_tokens(
        parsed_case, phase=str(phase), resolved_seq=resolved_seq, resolved_conc=resolved_conc,
    )
    mem = Memory(parsed_case, workspace_tokens=ws_tok)
    proj = mem.cache_ledger_projection(
        scenario=str(parsed_case.model.name),
        phase=str(phase),
        sequence_tokens=resolved_seq,
        batch_per_dp=resolved_conc,
        vision_accounting=mem.vision_accounting,
    )
    ledger = compute_memory_quant_ledger(
        parsed_case,
        phase=str(phase),
        seq_len=resolved_seq,
        concurrency=resolved_conc,
        memory_receipt=proj.receipt,
    )
    dies = proj.receipt.dies
    if dies:
        usable_by_die = tuple(max(1, int(d.usable_hbm_bytes)) for d in dies)
        rt_peaks = tuple(int(d.simultaneous_peak_hbm_bytes) for d in dies)
        ld_peaks = tuple(int(d.load_peak_hbm_bytes) for d in dies)
        host_ddr_bytes = max(int(d.host_dram_bytes) for d in dies)
    else:
        usable_by_die = (max(1, int(ledger.usable_hbm_bytes)),)
        rt_peaks = (int(ledger.max_stage_hbm_bytes),)
        ld_peaks = (int(ledger.max_stage_hbm_bytes),)
        host_ddr_bytes = int(ledger.max_stage_host_offload_bytes)

    eff_host_cap = usable_host_ddr_bytes
    if eff_host_cap is None:
        sparse_cfg = (parsed_case.runtime.kernels.get("lowerings") or {}).get("sparse_kv_offload")
        if isinstance(sparse_cfg, Mapping):
            opts = sparse_cfg.get("options") if isinstance(sparse_cfg.get("options"), Mapping) else {}
            dram_gb = opts.get("dram_size_per_dp_gb")
            if isinstance(dram_gb, (int, float)) and not isinstance(dram_gb, bool) and dram_gb > 0:
                offload_layers = [
                    lp for lp in (parsed_case.model.layer_plans or ())
                    if bool(getattr(lp, "indexer", False)) or int(getattr(lp, "index_topk", 0) or 0) > 0
                ]
                n_off = len(offload_layers) or max(1, int(parsed_case.model.layers or 1))
                reserve_b = n_off * HOST_BUFFER_POOLS * HOST_ALIGNMENT_SLAB_BYTES
                eff_host_cap = max(0, int(float(dram_gb) * float(GIB_BYTES)) - reserve_b)

    rt_fail_caps = [u for p, u in zip(rt_peaks, usable_by_die) if p > u]
    ld_fail_caps = [u for p, u in zip(ld_peaks, usable_by_die) if p > u]
    rt_exceed = bool(rt_fail_caps)
    ld_exceed = bool(ld_fail_caps)
    host_exceed = eff_host_cap is not None and host_ddr_bytes > eff_host_cap

    if rt_exceed and ld_exceed:
        admissible, reason = False, "HBM_RUNTIME_AND_LOAD_PEAK_EXCEEDED"
    elif ld_exceed:
        admissible, reason = False, "HBM_LOAD_PEAK_EXCEEDED"
    elif rt_exceed:
        admissible, reason = False, "HBM_RUNTIME_PEAK_EXCEEDED"
    elif host_exceed:
        admissible, reason = False, "HOST_DDR_CAPACITY_EXCEEDED"
    else:
        admissible, reason = True, "ADMISSIBLE"

    usable_scalar = (
        max(usable_by_die) if admissible
        else min(rt_fail_caps + ld_fail_caps) if (rt_fail_caps or ld_fail_caps)
        else min(usable_by_die)
    )
    max_conc = 0 if ld_exceed or host_exceed else max(0, int(ledger.max_hbm_concurrency_per_dp))
    receipt = HbmAdmissibilityReceipt(
        admissible=admissible,
        reason=reason,
        usable_hbm_bytes_per_die=usable_scalar,
        runtime_peak_hbm_bytes_by_die=rt_peaks,
        load_peak_hbm_bytes_by_die=ld_peaks,
        host_ddr_bytes=host_ddr_bytes,
        usable_host_ddr_bytes=eff_host_cap,
        max_admissible_concurrency=max_conc,
        evaluated_concurrency=resolved_conc,
        cell_id=cell_id,
    )
    return receipt, ledger


# ---------------------------------------------------------------------------
# White-Box Formulas, msprof Mapping & 31-Field CanonicalKernelKey
# ---------------------------------------------------------------------------

def _msprof_accelerator_core(op: OperatorExportRow, builder: str) -> AcceleratorCore:
    if builder in COLLECTIVE_BUILDERS:
        return "HCCL"
    tt = str(op.task_type or "").strip().upper()
    if tt in {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "HCCL", "DSA"}:
        return tt  # type: ignore[return-value]
    if "HCCL" in tt or "COMM" in tt:
        return "HCCL"
    if "MIX_AIC" in tt or "MIX" in tt:
        return "MIX_AIC"
    if "VECTOR" in tt or "AIV" in tt:
        return "AI_VECTOR_CORE"
    if op.flops > 0 and op.vector_ops > 0:
        return "MIX_AIC"
    if op.flops > 0:
        return "AI_CORE"
    return "AI_VECTOR_CORE"


def _msprof_bound_type(
    op: OperatorExportRow,
    nominal_spec: HardwareNominalSpec,
    *,
    cube_dtype: str,
    link_tier: str,
) -> tuple[BoundType, str]:
    t_cube = (op.flops / nominal_spec.cube_peak_for_dtype(cube_dtype)) if op.flops > 0.0 else 0.0
    t_vec = (op.vector_ops / nominal_spec.vector_peak_ops_per_s) if op.vector_ops > 0.0 else 0.0
    t_hbm = (op.hbm_total_bytes / nominal_spec.hbm_peak_bytes_per_s) if op.hbm_total_bytes > 0.0 else 0.0
    try:
        link_peak = nominal_spec.link_peak_for_tier(link_tier)
    except ModelingError:
        link_peak = next(iter(nominal_spec.raw_link_peaks_bytes_per_s.values()), 1.0)
    t_link = (op.link_bytes / link_peak) if op.link_bytes > 0.0 else 0.0
    best = max(t_cube, t_vec, t_hbm, t_link)
    if best <= 0.0:
        return "Latency Bound", "launch"
    if best == t_link and t_link > 0.0:
        return "Communication Bound", "link"
    if best == t_cube and t_cube > 0.0:
        return "Compute Bound (Cube)", "cube"
    if best == t_vec and t_vec > 0.0:
        return "Compute Bound (Vector)", "vector"
    return "Memory Bound (HBM)", "hbm"


def _clean_cube_dtype(raw_dtype: str, nominal_spec: HardwareNominalSpec) -> str:
    low = str(raw_dtype or "bf16").strip().lower()
    for token in (low, low.split("/")[0], low.split("_")[0], "bf16", "fp16"):
        try:
            nominal_spec.cube_peak_for_dtype(token)
            return token
        except ModelingError:
            continue
    return next(iter(nominal_spec.cube_peaks_flops_per_s))


def _format_tensor_triplet(shapes: Sequence[Sequence[int]], dtypes: Sequence[str], layouts: Sequence[str]) -> tuple[str, str, str]:
    sh_str = ";".join(",".join(str(int(d)) for d in s) for s in shapes if s)
    dt_str = ";".join(str(d).upper() for d in dtypes if d)
    ly_str = ";".join(str(ly).upper() for ly in layouts if ly)
    return sh_str, dt_str, ly_str


def build_operator_formula(
    op: OperatorExportRow,
    builder: str,
    *,
    shape_dict: Mapping[str, int],
    cube_dtype: str,
    seq_len: int,
    concurrency: int,
) -> tuple[str, dict[str, Any]]:
    """Produce a human-readable white-box analytical formula and substituted parameters (pure English)."""
    b_el = max(1, ELEMENT_BYTES.get(cube_dtype.lower(), 2))
    if builder in {"matmul", "quant_matmul", "all_gather_matmul", "matmul_reduce_scatter"}:
        m = int(shape_dict.get("m", op.shape[0] if op.shape else concurrency))
        n = int(shape_dict.get("n", op.shape[1] if len(op.shape) > 1 else 1))
        k = int(shape_dict.get("k", op.shape[2] if len(op.shape) > 2 else 1))
        return (
            "FLOPs = 2*M*N*K; HBM = M*K*b_in + K*N*b_w + M*N*b_out; T_theo = max(FLOPs/P_cube_nom, HBM/B_hbm_nom)",
            {"M": m, "N": n, "K": k, "b_in": b_el, "b_w": b_el, "b_out": b_el},
        )
    if builder in {"grouped_matmul", "grouped_matmul_swiglu_quant"}:
        m = int(shape_dict.get("m", op.shape[0] if op.shape else concurrency))
        n = int(shape_dict.get("n", op.shape[1] if len(op.shape) > 1 else 1))
        k = int(shape_dict.get("k", op.shape[2] if len(op.shape) > 2 else 1))
        e = int(shape_dict.get("e", 1))
        return (
            "FLOPs = 2*M_sum*N*K; HBM = M_sum*K*b_in + E_act*K*N*b_w + M_sum*N*b_out; T_theo = max(FLOPs/P_cube_nom, HBM/B_hbm_nom)",
            {"M_sum": m, "N": n, "K": k, "E_act": e, "b_w": b_el},
        )
    if builder in {
        "absorbed_mla", "fused_infer_attention", "incre_flash_attention",
        "fusion_attention", "sparse_flash_attention", "mixed_quant_sparse_mla", "block_sparse_attention_c8",
    }:
        return (
            "FLOPs = 2*B*H_q*Q*S_kv*(D_qk + D_v); HBM = Q_bytes + KV_read_bytes + O_bytes; T_theo = max(FLOPs/P_cube_nom, HBM/B_hbm_nom)",
            {
                "B": int(shape_dict.get("sequences", concurrency)),
                "H_q": int(shape_dict.get("heads", 1)),
                "Q": int(shape_dict.get("q_len", 1)),
                "S_kv": int(shape_dict.get("kv_len", seq_len)),
                "D": int(shape_dict.get("head_dim", DEFAULT_HEAD_DIM)),
            },
        )
    if builder in COLLECTIVE_BUILDERS:
        nbytes = int(shape_dict.get("bytes", int(round(op.link_bytes))))
        gsize = max(1, int(shape_dict.get("group_size", 1)))
        return (
            "CommBytes = V_rank * factor(N_ranks); T_theo = CommBytes / B_link_raw_nom",
            {"CommBytes": nbytes, "N_ranks": gsize},
        )
    if builder in {"lightning_indexer", "pool_key_indexer"}:
        return (
            "FLOPs = 2*B*H_idx*Q*S_kv*D_idx; HBM = Q_idx + K_idx + Scores; T_theo = max(FLOPs/P_cube_nom, HBM/B_hbm_nom)",
            {"B": concurrency, "S_kv": seq_len, "HBM_bytes": int(round(op.hbm_total_bytes))},
        )
    return (
        "T_theo = max(FLOPs/P_cube_nom, VecOps/P_vec_nom, (HBM_read + HBM_write)/B_hbm_nom)",
        {
            "FLOPs": int(round(op.flops)),
            "VecOps": int(round(op.vector_ops)),
            "HBM_read": int(round(op.hbm_read_bytes)),
            "HBM_write": int(round(op.hbm_write_bytes)),
        },
    )


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    if not shape:
        return ()
    out = [1] * len(shape)
    acc = 1
    for idx in range(len(shape) - 1, -1, -1):
        out[idx] = acc
        acc *= max(1, int(shape[idx]))
    return tuple(out)


def _cache_canonical_key(cid: str, key: CanonicalKernelKey) -> None:
    if len(_CANONICAL_KEYS_BY_CASE_ID) >= CANONICAL_KEY_CACHE_MAX:
        _CANONICAL_KEYS_BY_CASE_ID.clear()
    _CANONICAL_KEYS_BY_CASE_ID[cid] = key


def build_canonical_kernel_key(
    op: OperatorExportRow,
    *,
    sku: str,
    soc_arch: str,
    builder: str,
    upstream_abi: str,
    abi_backend: str,
    phase: PhaseKind,
    seq_len: int,
    concurrency: int,
    capture_mode: CaptureMode,
    parallel: ParallelArmSpec,
    kv_block_size: int,
    shape_dict: Mapping[str, int],
    cube_dtype: str,
    link_tier: str,
    per_expert_token_counts: tuple[int, ...] | None = None,
    per_rank_token_counts: tuple[int, ...] | None = None,
    group_ranks: tuple[int, ...] | None = None,
    precision_abi: Any = None,
) -> CanonicalKernelKey:
    """Build exact 31-field model-agnostic ``CanonicalKernelKey`` (matching golden fixtures)."""
    kind: KernelKeyKind = "collective" if builder in COLLECTIVE_BUILDERS else "operator"
    dims = tuple(max(1, int(d)) for d in (op.shape or ()))
    is_kv_op = builder in {
        "absorbed_mla", "fused_infer_attention", "incre_flash_attention",
        "fusion_attention", "sparse_flash_attention", "mixed_quant_sparse_mla",
        "block_sparse_attention_c8", "scatter_pa_kv_cache", "scatter_index_cache",
        "lightning_indexer", "pool_key_indexer", "kv_compress_epilog",
    }
    in_layout = "TND" if is_kv_op else "ND"
    in_dt = str(getattr(precision_abi, "input_dtype", "") or cube_dtype).lower()
    out_dt = str(getattr(precision_abi, "output_dtype", "") or cube_dtype).lower()
    w_dt = str(getattr(precision_abi, "weight_dtype", "") or cube_dtype).lower()
    w_layout = str(
        getattr(precision_abi, "weight_layout", "")
        or ("FRACTAL_NZ" if builder in {"quant_matmul", "grouped_matmul", "grouped_matmul_swiglu_quant"} else "ND")
    )
    dtypes: dict[str, str] = {"input": in_dt, "output": out_dt}
    layouts: dict[str, str] = {"input": in_layout, "output": in_layout}
    if builder in {"matmul", "quant_matmul", "grouped_matmul", "grouped_matmul_swiglu_quant", "all_gather_matmul", "matmul_reduce_scatter"}:
        dtypes["weight"] = w_dt
        layouts["weight"] = w_layout
        w_sc = str(getattr(precision_abi, "weight_scale_dtype", "") or "").lower()
        a_sc = str(getattr(precision_abi, "act_scale_dtype", "") or "").lower()
        if w_sc:
            dtypes["weight_scale"] = w_sc
            layouts["weight_scale"] = "ND"
        if a_sc:
            dtypes["act_scale"] = a_sc
            layouts["act_scale"] = "ND"
    if is_kv_op:
        dtypes["cache"] = in_dt
        layouts["cache"] = "PA_BSND"
    strides: dict[str, tuple[int, ...]] = {}
    offsets: dict[str, int] = {}
    if dims:
        strides["input"] = _contiguous_strides(dims)
        offsets["input"] = 0
    cp_stride = max(1, parallel.dcp * parallel.pcp)
    local_kv = int(shape_dict.get("kv_len", max(1, seq_len // cp_stride)))
    page_geo: dict[str, Any] | None = None
    kv_alloc_blk: int | None = None
    kv_kern_blk: int | None = None
    cp_interleave: int | None = None
    if is_kv_op:
        kv_alloc_blk = max(1, int(kv_block_size))
        kv_kern_blk = max(1, int(kv_block_size))
        cp_interleave = kv_alloc_blk if local_kv < seq_len else None
        num_pages = max(1, (local_kv + kv_alloc_blk - 1) // kv_alloc_blk) * max(1, concurrency)
        slot_stride = int(shape_dict.get("head_dim", DEFAULT_HEAD_DIM))
        page_geo = {
            "num_pages": num_pages,
            "page_bytes": kv_alloc_blk * slot_stride * max(1, ELEMENT_BYTES.get(in_dt, 2)),
            "slot_stride": slot_stride,
        }
    ordered_ranks: tuple[int, ...] | None = None
    link_tiers: tuple[str, ...] | None = None
    coll_algo: str | None = None
    if kind == "collective":
        if group_ranks:
            ordered_ranks = tuple(int(r) for r in group_ranks)
        else:
            gsize = max(1, int(shape_dict.get("group_size", 1)))
            ordered_ranks = tuple(range(gsize))
        link_tiers = (link_tier,)
        coll_algo = "ring"
    q_meta: dict[str, Any] = {}
    if w_layout == "FRACTAL_NZ":
        q_meta["format_code"] = FRACTAL_NZ_FORMAT_CODE
    if precision_abi is not None and str(getattr(precision_abi, "activation_quant_mode", "none")) != "none":
        q_meta["activation_quant_mode"] = str(precision_abi.activation_quant_mode)
    ck = CanonicalKernelKey(
        kind=kind,
        sku=sku,
        soc_arch=soc_arch,  # type: ignore[arg-type]
        abi_backend=abi_backend,
        upstream_abi=upstream_abi,
        bench_builder=builder,
        canonical_shapes=dict(shape_dict),
        strides=strides,
        storage_offsets=offsets,
        dtypes=dtypes,
        layouts=layouts,
        kv_allocator_block_size=kv_alloc_blk,
        kv_kernel_block_size=kv_kern_blk,
        cp_kv_cache_interleave_size=cp_interleave,
        page_geometry=page_geo,
        per_expert_token_counts=per_expert_token_counts,
        per_rank_token_counts=per_rank_token_counts,
        capture_mode=capture_mode,
        aclgraph_padded_bucket=(max(1, concurrency) if capture_mode == "aclgraph" else None),
        workspace_policy="default",
        stream_dependencies=(),
        residency="cold_hbm",
        quant_meta=q_meta,
        op_attrs={"phase": str(phase)},
        ordered_rank_placement=ordered_ranks,
        link_tiers=link_tiers,
        collective_algorithm=coll_algo,
        split_sizes=None,
        timed_nodes=None,
        edges=None,
    )
    _cache_canonical_key(ck.case_id(), ck)
    return ck


# ---------------------------------------------------------------------------
# Single-Spine Operator Matrix Evaluation
# ---------------------------------------------------------------------------

def resolve_model_spec_public(model_ref: str | Path | Mapping[str, Any] | ModelSpec) -> ModelSpec:
    """Resolve a catalog ID, local HuggingFace ``config.json``, or inline mapping via public catalog APIs."""
    if isinstance(model_ref, ModelSpec):
        return model_ref
    if isinstance(model_ref, Mapping):
        raw = dict(model_ref)
        if "layers" in raw and "hidden_size" in raw and "model_type" not in raw:
            return ModelSpec.from_mapping(raw)
        return build_model_spec(str(raw.get("name") or raw.get("model_type") or "inline-model"), raw)
    path = Path(model_ref)
    if path.is_file():
        cfg_file = path / "config.json" if path.is_dir() else path
        raw_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if isinstance(raw_cfg, Mapping) and "layers" in raw_cfg and "hidden_size" in raw_cfg and "model_type" not in raw_cfg:
            return ModelSpec.from_mapping(raw_cfg)
        return build_model_spec(cfg_file.stem, raw_cfg)
    if path.is_dir() and (path / "config.json").is_file():
        raw_cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        return build_model_spec(path.name, raw_cfg)
    return load_catalog_model(str(model_ref))


def _resolve_model_label(model_input: str | Path | Mapping[str, Any], model_spec: ModelSpec) -> str:
    if isinstance(model_input, str):
        p = Path(model_input)
        if p.suffix.lower() == ".json":
            return str(model_spec.name or p.stem)
        return model_input.strip()
    if isinstance(model_input, Path):
        return str(model_spec.name or model_input.stem)
    return str(model_spec.name or "custom-model")


def _compose_cell_case(
    model_input: str | Path | Mapping[str, Any],
    sku: str,
    *,
    arm: ParallelArmSpec,
    phase: PhaseKind,
    seq_len: int,
    concurrency: int,
    spec: MatrixSpec,
    layers: int | None,
    feature_track: str,
    topology_preset_override: str | None = None,
) -> tuple[Case, StepRequest]:
    resolved_sku, alias_preset = resolve_hardware_alias(sku)
    topo_preset = topology_preset_override if topology_preset_override is not None else alias_preset
    eff_layers = layers
    eff_tp = arm.tp
    eff_dp = arm.dp
    eff_ep = arm.ep
    eff_pp = arm.pp
    eff_dcp = arm.dcp
    eff_pcp = arm.pcp
    eff_nd = arm.num_devices or (eff_dp * eff_tp * eff_pp * eff_pcp)
    quant_arg: str | None = None
    va_rec: bool | None = None
    if spec.quant_policy == "auto_va":
        va_rec = True
    elif spec.quant_policy not in {"", "checkpoint"}:
        quant_arg = spec.quant_policy

    opts: tuple[str, ...] = ("sparse_kv_offload",) if arm.sparse_kv_offload else ()
    prompt_tok = max(1, int(seq_len))
    q_tok = 1 + max(0, int(spec.mtp_proposal_tokens))
    out_tok = q_tok if phase == "decode" else DEFAULT_OUTPUT_TOKENS_BUDGET
    raw_case = compose_case(
        model=model_input,
        hardware=resolved_sku,
        topology_preset=topo_preset,
        dp=eff_dp,
        tp=eff_tp,
        ep=eff_ep,
        pp=eff_pp,
        dcp=eff_dcp,
        pcp=eff_pcp,
        sp=arm.sp,
        num_devices=eff_nd,
        layers=eff_layers,
        quant=quant_arg,
        va_recommended=va_rec,
        prompt_tokens=prompt_tok,
        output_tokens=out_tok,
        concurrency=max(1, int(concurrency)),
        max_model_len=max(prompt_tok + out_tok + DEFAULT_CACHE_BLOCK_SIZE, DEFAULT_SEQ_MIN * 2),
        mtp_proposal_tokens=int(spec.mtp_proposal_tokens) if phase == "decode" else 0,
        mtp_acceptance_length=spec.mtp_acceptance_length,
        ttft_slo_ms=spec.ttft_slo_ms,
        tpot_slo_ms=spec.tpot_slo_ms,
        q_proj_tp=arm.q_proj_tp or 1,
        o_proj_tp=arm.o_proj_tp or 1,
        dcp_linear_tp=arm.dcp_linear_tp or 1,
        shared_expert_tp=arm.shared_expert_tp or 1,
        graph_capture_mode=("full" if phase == "decode" else "none"),
        feature_track=feature_track,
        optimizations=opts,
        deployment_mode=arm.deployment_mode or spec.deployment_mode,
        _stack_phase=str(phase),
    )
    model_experts = int((raw_case.get("model") or {}).get("num_experts") or 0)
    if model_experts > 0:
        rt_par = dict((raw_case.get("runtime") or {}).get("parallel") or {})
        rt_par["expert_tp"] = int(arm.routed_expert_tp)
        raw_case["runtime"]["parallel"] = rt_par
    if arm.sparse_kv_offload:
        rt = dict(raw_case.get("runtime") or {})
        kernels = dict(rt.get("kernels") or {})
        lowerings = dict(kernels.get("lowerings") or {})
        if "sparse_kv_offload" not in lowerings:
            lowerings["sparse_kv_offload"] = {
                "implementation": "vllm_ascend_sparse_kv_offload",
                "options": {
                    "dram_size_per_dp_gb": DEFAULT_HISPARSE_DRAM_GB,
                    "topk_buffer_size": DEFAULT_HISPARSE_TOPK_BUFFER,
                    "miss_rate": DEFAULT_HISPARSE_MISS_RATE,
                },
            }
            kernels["lowerings"] = lowerings
            rt["kernels"] = kernels
            raw_case["runtime"] = rt
    parsed = parse_case(raw_case)
    if phase == "prefill":
        chunk = min(prompt_tok, DEFAULT_PREFILL_CHUNK_CAP)
        ctx = max(0, prompt_tok - chunk)
        step_req = StepRequest(phase="prefill", tokens=chunk, context=ctx, sequences=max(1, int(concurrency)), scope="model")
    else:
        step_req = StepRequest(phase="decode", tokens=q_tok, context=prompt_tok, sequences=max(1, int(concurrency)), scope="model")
    return parsed, step_req


def _build_cell_dag(operators: Sequence[OperatorExportRow]) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Build acyclic unfolded DAG ``(nodes, edges)`` preserving true stream/predecessor concurrency."""
    nodes: list[str] = []
    node_set: set[str] = set()
    edges: list[tuple[str, str]] = []
    by_op_id = {op.op_id: op for op in operators}
    last_on_stream: dict[str, str] = {}
    for op in operators:
        rep = max(1, int(op.repeat_count))
        stream_key = str(op.stream or f"stream_{int(op.stream_index or 0)}")
        explicit_preds: list[str] = []
        for pred in op.predecessors:
            p_op = by_op_id.get(pred.op_id)
            if p_op is not None and p_op.op_id != op.op_id:
                src = p_op.op_id if int(p_op.repeat_count) <= 1 else f"{p_op.op_id}#{int(p_op.repeat_count)}"
                if src in node_set:
                    explicit_preds.append(src)
        if rep == 1:
            nid = op.op_id
            nodes.append(nid)
            node_set.add(nid)
            for src in explicit_preds:
                edges.append((src, nid))
            if not explicit_preds and stream_key in last_on_stream and last_on_stream[stream_key] != nid:
                edges.append((last_on_stream[stream_key], nid))
            last_on_stream[stream_key] = nid
        else:
            chain_prev = last_on_stream.get(stream_key) if not explicit_preds else None
            for occ_idx in range(1, rep + 1):
                nid = f"{op.op_id}#{occ_idx}"
                nodes.append(nid)
                node_set.add(nid)
                if occ_idx == 1 and explicit_preds:
                    for src in explicit_preds:
                        edges.append((src, nid))
                elif chain_prev is not None and chain_prev != nid:
                    edges.append((chain_prev, nid))
                chain_prev = nid
            if chain_prev is not None:
                last_on_stream[stream_key] = chain_prev
    return tuple(nodes), tuple(edges)


def _extract_int_tuple(raw: Any) -> tuple[int, ...] | None:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
        vals: list[int] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            vals.append(max(0, int(round(item))))
        return tuple(vals)
    return None


def _resolve_op_link_tier(
    op: OperatorExportRow,
    op_spec: OpSpec | None,
    meta_dict: Mapping[str, Any],
    nominal_spec: HardwareNominalSpec,
    *,
    group_ranks: tuple[int, ...],
    ranks_per_node: int,
    dies_per_pkg: int,
) -> str:
    raw_hint = str(
        meta_dict.get("link_tier")
        or meta_dict.get("channel")
        or meta_dict.get("route")
        or (op_spec.link_scope if op_spec is not None else "")
        or op.comm_domain_tag
        or ""
    ).strip().lower()
    peaks = nominal_spec.raw_link_peaks_bytes_per_s
    if raw_hint in peaks:
        return raw_hint
    if "inter_node" in raw_hint or "roce" in raw_hint:
        for cand in ("roce_inter_node", "d2d_inter_node"):
            if cand in peaks:
                return cand
    if "intra_pkg" in raw_hint or "sio" in raw_hint or "intra_package" in raw_hint:
        for cand in ("d2d_intra_pkg", "sio"):
            if cand in peaks:
                return cand
    if group_ranks:
        span_nodes = len({r // max(1, ranks_per_node) for r in group_ranks})
        span_pkgs = len({r // max(1, dies_per_pkg) for r in group_ranks})
        if span_nodes > 1:
            for cand in ("roce_inter_node", "d2d_inter_node"):
                if cand in peaks:
                    return cand
        if dies_per_pkg > 1 and span_pkgs == 1 and len(group_ranks) > 1:
            for cand in ("d2d_intra_pkg", "sio"):
                if cand in peaks:
                    return cand
    if "d2d_intra_node" in peaks:
        return "d2d_intra_node"
    return next(iter(peaks), "d2d_intra_node")


def _resolve_collective_participants(
    op: OperatorExportRow,
    meta_dict: Mapping[str, Any],
    parallel: ParallelArmSpec,
) -> tuple[int, ...]:
    raw_ranks = _extract_int_tuple(meta_dict.get("group_ranks") or meta_dict.get("participating_ranks") or meta_dict.get("ranks"))
    if raw_ranks:
        return raw_ranks
    domain = str(meta_dict.get("comm_domain") or meta_dict.get("collective_domain") or op.comm_domain_tag or "").lower()
    fam = str(op.family or "").lower()
    if "ep" in domain or fam in {"alltoall", "alltoallv", "mc2_protocol"} or "expert" in str(op.recipe or "").lower():
        gsize = max(1, int(parallel.ep))
    elif "dcp" in domain or "dcp" in fam:
        gsize = max(1, int(parallel.dcp))
    elif "pcp" in domain or "pcp" in fam:
        gsize = max(1, int(parallel.pcp))
    elif "pp" in domain or fam == "pp_p2p":
        gsize = max(1, int(parallel.pp))
    elif "dp" in domain:
        gsize = max(1, int(parallel.dp))
    else:
        gsize = max(1, int(parallel.tp))
    return tuple(range(gsize))


def _op_shape_dict(
    op: OperatorExportRow,
    builder: str,
    *,
    seq_len: int,
    concurrency: int,
    parallel: ParallelArmSpec,
    phase: PhaseKind,
    q_len_step: int,
    meta_dict: Mapping[str, Any],
    model_spec: ModelSpec,
    group_ranks: tuple[int, ...],
) -> dict[str, int]:
    dims = [int(d) for d in (op.shape or ()) if isinstance(d, int) and d > 0]
    if builder in {"matmul", "quant_matmul", "all_gather_matmul", "matmul_reduce_scatter"}:
        if len(dims) >= 3:
            return {"m": dims[0], "n": dims[1], "k": dims[2]}
        if len(dims) == 2:
            return {"m": dims[0], "n": dims[1], "k": dims[1]}
        return {"m": max(1, concurrency * q_len_step), "n": dims[0] if dims else 1, "k": dims[0] if dims else 1}
    if builder in {"grouped_matmul", "grouped_matmul_swiglu_quant"}:
        exp_counts = _extract_int_tuple(
            meta_dict.get("per_expert_token_counts")
            or meta_dict.get("expert_token_counts")
            or meta_dict.get("routing_counts")
        )
        local_e = (
            len(exp_counts) if exp_counts
            else int(meta_dict.get("local_experts") or 0)
            or (dims[3] if len(dims) > 3 else max(1, int(getattr(model_spec, "num_experts", 1) or 1) // max(1, parallel.ep)))
        )
        return {
            "m": dims[0] if dims else max(1, concurrency * q_len_step),
            "n": dims[1] if len(dims) > 1 else 1,
            "k": dims[2] if len(dims) > 2 else 1,
            "e": max(1, local_e),
        }
    if builder in {
        "absorbed_mla", "fused_infer_attention", "incre_flash_attention",
        "fusion_attention", "sparse_flash_attention", "mixed_quant_sparse_mla", "block_sparse_attention_c8",
    }:
        local_kv = max(1, seq_len // max(1, parallel.dcp * parallel.pcp))
        if meta_dict.get("gqa_decode") or meta_dict.get("gqa_prefill") or meta_dict.get("mla_decode") or meta_dict.get("mla_prefill"):
            return attention_shape_dims(meta_dict, dims, sequences=concurrency, kv_length=local_kv, phase=phase)
        lp0 = model_spec.layer_plans[0] if model_spec.layer_plans else None
        q_len = max(1, int(q_len_step))
        heads = dims[2] if len(dims) >= 4 else (dims[1] if len(dims) >= 2 else max(1, int(getattr(lp0, "num_heads", 1) or 1) // max(1, parallel.tp)))
        raw_kv_h = int(getattr(lp0, "num_kv_heads", 1) or 1)
        kv_h = 1 if str(getattr(lp0, "attention", "")) == "mla" else max(1, raw_kv_h // min(max(1, parallel.tp), max(1, raw_kv_h)))
        h_dim = dims[-1] if dims else int(getattr(lp0, "head_dim", DEFAULT_HEAD_DIM) or DEFAULT_HEAD_DIM)
        kv_actual = dims[1] if len(dims) >= 4 else local_kv
        return {
            "tokens": dims[0] if dims else max(1, concurrency * q_len),
            "sequences": max(1, concurrency),
            "q_len": q_len,
            "kv_len": max(1, kv_actual),
            "heads": max(1, heads),
            "kv_heads": max(1, kv_h),
            "head_dim": max(1, h_dim),
            "is_prefill": 1 if phase == "prefill" else 0,
        }
    if builder in COLLECTIVE_BUILDERS:
        return {"bytes": max(1, int(round(op.link_bytes))), "group_size": max(1, len(group_ranks))}
    if len(dims) >= 2:
        return {"tokens": dims[0], "width": dims[-1]}
    if dims:
        return {"tokens": dims[0], "width": 1}
    return {"tokens": max(1, concurrency * q_len_step), "width": 1}


def _resolve_op_port_abi(
    op: OperatorExportRow,
    builder: str,
    shape_dict: Mapping[str, int],
    ck: CanonicalKernelKey,
    precision_abi: Any,
) -> tuple[str, str, str, str, str, str]:
    """Emit per-port CANN msprof Input/Output Shapes, Data Types, and Formats."""
    in_dt = ck.dtypes.get("input", "bf16")
    out_dt = ck.dtypes.get("output", in_dt)
    in_fm = ck.layouts.get("input", "ND")
    out_fm = ck.layouts.get("output", in_fm)
    if builder in {"matmul", "quant_matmul", "all_gather_matmul", "matmul_reduce_scatter"}:
        m, n, k = int(shape_dict.get("m", 1)), int(shape_dict.get("n", 1)), int(shape_dict.get("k", 1))
        in_shapes: list[tuple[int, ...]] = [(m, k), (k, n)]
        in_dts: list[str] = [in_dt, ck.dtypes.get("weight", in_dt)]
        in_fms: list[str] = [in_fm, ck.layouts.get("weight", "ND")]
        if "weight_scale" in ck.dtypes:
            in_shapes.append((n,))
            in_dts.append(ck.dtypes["weight_scale"])
            in_fms.append("ND")
        if "act_scale" in ck.dtypes:
            in_shapes.append((m,))
            in_dts.append(ck.dtypes["act_scale"])
            in_fms.append("ND")
        return (*_format_tensor_triplet(in_shapes, in_dts, in_fms), *_format_tensor_triplet(((m, n),), (out_dt,), (out_fm,)))
    if builder in {"grouped_matmul", "grouped_matmul_swiglu_quant"}:
        m, n, k, e = int(shape_dict.get("m", 1)), int(shape_dict.get("n", 1)), int(shape_dict.get("k", 1)), int(shape_dict.get("e", 1))
        in_shapes = [(m, k), (e, k, n), (e,)]
        in_dts = [in_dt, ck.dtypes.get("weight", in_dt), "int64"]
        in_fms = [in_fm, ck.layouts.get("weight", "FRACTAL_NZ"), "ND"]
        if "weight_scale" in ck.dtypes:
            in_shapes.append((e, n))
            in_dts.append(ck.dtypes["weight_scale"])
            in_fms.append("ND")
        if "act_scale" in ck.dtypes:
            in_shapes.append((m,))
            in_dts.append(ck.dtypes["act_scale"])
            in_fms.append("ND")
        return (*_format_tensor_triplet(in_shapes, in_dts, in_fms), *_format_tensor_triplet(((m, n),), (out_dt,), (out_fm,)))
    if "kv_len" in shape_dict and "heads" in shape_dict:
        tok, kv_l, h, kvh, hd = (
            int(shape_dict["tokens"]), int(shape_dict["kv_len"]),
            int(shape_dict["heads"]), int(shape_dict.get("kv_heads", 1)), int(shape_dict.get("head_dim", DEFAULT_HEAD_DIM)),
        )
        c_dt, c_fm = ck.dtypes.get("cache", in_dt), ck.layouts.get("cache", "PA_BSND")
        return (
            *_format_tensor_triplet(((tok, h, hd), (kv_l, kvh, hd), (kv_l, kvh, hd)), (in_dt, c_dt, c_dt), (in_fm, c_fm, c_fm)),
            *_format_tensor_triplet(((tok, h, hd),), (out_dt,), (out_fm,)),
        )
    dims = tuple(int(d) for d in (op.shape or ()) if isinstance(d, int) and d > 0)
    return (
        *_format_tensor_triplet((dims,) if dims else (), (in_dt,), (in_fm,)),
        *_format_tensor_triplet((dims,) if dims else (), (out_dt,), (out_fm,)),
    )


def _extract_bundle_rows(
    bundle: ExportBundle,
    *,
    parsed_case: Case,
    step_req: StepRequest,
    spec_by_op_id: Mapping[str, OpSpec],
    cell_id: str,
    model_id: str,
    sku: str,
    nominal_spec: HardwareNominalSpec,
    phase: PhaseKind,
    seq_len: int,
    cliff_kind: CliffKind,
    concurrency: int,
    arm: ParallelArmSpec,
    arm_display_id: str,
    stage: str,
    kv_block_size: int,
) -> tuple[list[OperatorMatrixRow], dict[str, list[ReverseIndexEntry]], dict[str, CanonicalKernelKey], StepSummary]:
    """[Cohesive-Pipeline] Project single-spine ``ExportBundle`` + lowered ``OpSpec`` rows into nominal Roofline rows."""
    capture_mode: CaptureMode = "aclgraph" if str(bundle.graph_capture_mode or "").lower() != "none" and phase == "decode" else "eager"
    step_span_us = max(float(bundle.step_span_us), MIN_POSITIVE_DENOM)
    num_dies = max(1, int(arm.num_devices or (arm.dp * arm.tp * arm.pp * arm.pcp)))
    ranks_per_node = max(1, int(getattr(parsed_case.machine, "ranks_per_node", num_dies) or num_dies))
    dies_per_pkg = max(1, int(getattr(parsed_case.machine, "dies_per_package", 1) or 1))
    arch_by_id = {a.archetype_id: a for a in (bundle.archetypes or ())}

    rows: list[OperatorMatrixRow] = []
    rev_entries: dict[str, list[ReverseIndexEntry]] = defaultdict(list)
    key_by_cid: dict[str, CanonicalKernelKey] = {}
    total_hbm_rw_pipeline = 0
    flops_by_dtype: dict[str, int] = defaultdict(int)
    link_by_tier: dict[str, int] = defaultdict(int)

    for op in bundle.operators:
        fam = str(op.family or "elementwise")
        rec = str(op.recipe or fam)
        if fam in CONTROL_FAMILIES:
            continue
        op_spec = spec_by_op_id.get(op.op_id)
        meta_dict = dict(op_spec.metadata) if (op_spec is not None and op_spec.metadata) else {}
        prec_abi = op_spec.precision_abi if op_spec is not None else None
        raw_cube_dt = str(getattr(prec_abi, "compute_dtype", "") or op.dtype or "bf16")
        cube_dtype = _clean_cube_dtype(raw_cube_dt, nominal_spec)
        try:
            builder, upstream_abi = resolve_builder_strict(
                fam, rec, str(op.ascend_kernel or op.implementation or ""), (cube_dtype,), meta_dict,
                model_id=model_id, layer_idx=int(op.layer if op.layer is not None else -1), local_shape=op.shape,
            )
        except UnmodeledCapabilityError:
            builder = "tensor_copy" if op.flops <= 0 else "matmul"
            upstream_abi = str(op.ascend_kernel or op.implementation or ("aclnnInplaceCopy" if builder == "tensor_copy" else "aclnnMatmul"))
        abi_backend = resolve_abi_backend(builder, upstream_abi, op.kernel_backend)
        group_ranks = _resolve_collective_participants(op, meta_dict, arm) if builder in COLLECTIVE_BUILDERS else ()
        op_link_tier = _resolve_op_link_tier(
            op, op_spec, meta_dict, nominal_spec,
            group_ranks=group_ranks, ranks_per_node=ranks_per_node, dies_per_pkg=dies_per_pkg,
        )
        shape_dict = _op_shape_dict(
            op, builder, seq_len=seq_len, concurrency=concurrency, parallel=arm, phase=phase,
            q_len_step=int(step_req.tokens), meta_dict=meta_dict, model_spec=parsed_case.model,
            group_ranks=group_ranks,
        )
        per_exp = _extract_int_tuple(
            meta_dict.get("per_expert_token_counts") or meta_dict.get("expert_token_counts") or meta_dict.get("routing_counts")
        )
        per_rk = _extract_int_tuple(meta_dict.get("per_rank_token_counts") or meta_dict.get("rank_token_counts"))
        ck = build_canonical_kernel_key(
            op, sku=sku, soc_arch=nominal_spec.soc_arch, builder=builder, upstream_abi=upstream_abi,
            abi_backend=abi_backend, phase=phase, seq_len=seq_len, concurrency=concurrency,
            capture_mode=capture_mode, parallel=arm, kv_block_size=kv_block_size,
            shape_dict=shape_dict, cube_dtype=cube_dtype, link_tier=op_link_tier,
            per_expert_token_counts=per_exp, per_rank_token_counts=per_rk,
            group_ranks=group_ranks or None, precision_abi=prec_abi,
        )
        cid = ck.case_id()
        key_by_cid[cid] = ck

        hbm_rd = max(0, int(round(float(op.hbm_read_bytes or 0.0))))
        hbm_wr = max(0, int(round(float(op.hbm_write_bytes or 0.0))))
        if hbm_rd == 0 and hbm_wr == 0 and float(op.hbm_total_bytes or 0.0) > 0.0:
            hbm_rd = int(round(float(op.hbm_total_bytes)))
        flops_i = max(0, int(round(float(op.flops or 0.0))))
        vec_i = max(0, int(round(float(op.vector_ops or 0.0))))
        link_i = max(0, int(round(float(op.link_bytes or 0.0))))
        hit_frac = max(0.0, min(99.9, float(op.l2_hit_pct or 0.0))) / PCT_SCALE
        l2_from_meta = meta_dict.get("l2_traffic_bytes") or meta_dict.get("l2_bytes")
        l2_i = (
            max(0, int(round(float(l2_from_meta))))
            if isinstance(l2_from_meta, (int, float)) and not isinstance(l2_from_meta, bool)
            else (int(round(hbm_rd * hit_frac / max(1e-6, 1.0 - hit_frac))) if (hbm_rd > 0 and hit_frac > 0.0) else 0)
        )
        ub_from_meta = meta_dict.get("ub_working_set_bytes") or meta_dict.get("ub_bytes") or meta_dict.get("sram_working_set_bytes")
        ub_i = (
            max(0, int(round(float(ub_from_meta))))
            if isinstance(ub_from_meta, (int, float)) and not isinstance(ub_from_meta, bool)
            else (min(int(nominal_spec.ub_capacity_bytes), hbm_rd + hbm_wr) if (vec_i > 0 and (hbm_rd + hbm_wr) > 0) else 0)
        )
        nom_theo_us = nominal_spec.nominal_theo_us(
            cube_flops=float(flops_i), cube_dtype=cube_dtype, vector_ops=float(vec_i),
            hbm_read_bytes=float(hbm_rd), hbm_write_bytes=float(hbm_wr),
            link_bytes=float(link_i), link_tier=op_link_tier,
        )
        # Preserve raw simulator prediction faithfully (NEVER clamp to nominal_theo_us!).
        per_call_us = max(float(op.dur_us or 0.0), 1e-6)
        rep = max(1, int(op.repeat_count or 1))
        exposed_us = max(0.0, float(op.exposed_dur_us or 0.0) * float(rep))
        share_pct = (exposed_us / step_span_us) * PCT_SCALE
        mfu, mbu_rw, mbu_rd, link_bu = nominal_spec.utilization_pcts(
            per_call_us,
            cube_flops=float(flops_i), cube_dtype=cube_dtype,
            hbm_read_bytes=float(hbm_rd), hbm_write_bytes=float(hbm_wr),
            link_bytes=float(link_i), link_tier=op_link_tier,
        )
        bound_type, bottleneck_res = _msprof_bound_type(op, nominal_spec, cube_dtype=cube_dtype, link_tier=op_link_tier)
        accel_core = _msprof_accelerator_core(op, builder)
        blk_dim = nominal_spec.aic_count if accel_core in {"AI_CORE", "MIX_AIC"} else (
            nominal_spec.aiv_count if accel_core in {"AI_VECTOR_CORE", "MIX_AIV"} else 1
        )
        mix_blk = nominal_spec.aiv_count if accel_core in {"MIX_AIC", "MIX_AIV"} else 0
        formula_sym, formula_terms = build_operator_formula(
            op, builder, shape_dict=shape_dict, cube_dtype=cube_dtype, seq_len=seq_len, concurrency=concurrency,
        )
        in_sh, in_dt, in_fm, out_sh, out_dt, out_fm = _resolve_op_port_abi(op, builder, shape_dict, ck, prec_abi)
        ai = float(flops_i) / float(max(1, hbm_rd + hbm_wr)) if flops_i > 0 else 0.0
        native_layer = max(0, int(op.layer if op.layer is not None and int(op.layer) >= 0 else 0))
        model_scope: ModelScopeKind = "draft" if str(op.model_scope or "").lower() == "draft" else "target"
        occ_rank = int(op.occurrence.rank) if (op.occurrence is not None and getattr(op.occurrence, "rank", None) is not None) else 0
        clean_op_name = str(op.logical_role or op.op_id or fam).replace("Stage ", "Step ")
        l_star_us = max(0.0, float(op.l_star_lean_us or 0.0))

        row = OperatorMatrixRow(
            cell_id=cell_id,
            model_id=model_id,
            sku=sku,
            soc_arch=nominal_spec.soc_arch,
            phase=phase,
            seq_len=seq_len,
            seq_point_kind=cliff_kind,
            concurrency=concurrency,
            parallel_arm=arm_display_id,
            layer_archetype_id=str(op.archetype_id or "body"),
            native_layer=native_layer,
            model_scope=model_scope,
            replica=int(op.occurrence.replica) if op.occurrence is not None else 0,
            rank=occ_rank,
            op_id=str(op.op_id),
            occurrence_count=rep,
            family=fam,
            recipe=rec,
            abi_backend=abi_backend,
            upstream_abi=upstream_abi,
            bench_builder=builder,
            bench_case_id=cid,
            capture_mode=capture_mode,
            formula_symbolic=formula_sym,
            formula_terms=formula_terms,
            nominal_theo_us=round(nom_theo_us, ROUND_US_DIGITS),
            per_call_stage0_us=round(per_call_us, ROUND_US_DIGITS),
            model_exposed_stage0_us=round(exposed_us, ROUND_US_DIGITS),
            step_share_pct=round(share_pct, ROUND_PCT_DIGITS),
            l_star_us=round(l_star_us, ROUND_US_DIGITS),
            hbm_read_bytes=hbm_rd,
            hbm_write_bytes=hbm_wr,
            sim_mfu_pct=round(mfu, ROUND_PCT_DIGITS),
            sim_mbu_rw_pct=round(mbu_rw, ROUND_PCT_DIGITS),
            sim_mbu_read_pct=round(mbu_rd, ROUND_PCT_DIGITS),
            sim_link_bu_pct=round(link_bu, ROUND_PCT_DIGITS),
            op_name=clean_op_name,
            op_type=fam,
            kernel_name=upstream_abi,
            accelerator_core=accel_core,
            block_dim=blk_dim,
            mix_block_dim=mix_blk,
            stream_id=max(0, int(op.stream_index or 0)),
            input_shapes=in_sh,
            input_data_types=in_dt,
            input_formats=in_fm,
            output_shapes=out_sh,
            output_data_types=out_dt,
            output_formats=out_fm,
            arithmetic_intensity=round(ai, ROUND_PCT_DIGITS),
            bound_type=bound_type,
            cube_flops=flops_i,
            cube_dtype=cube_dtype,
            vector_ops=vec_i,
            l2_bytes=l2_i,
            ub_bytes=ub_i,
            link_bytes=link_i,
            link_tier=op_link_tier,
            bottleneck_resource=bottleneck_res,
            tensor_shapes=shape_dict,
            strides=ck.strides,
            dtypes=ck.dtypes,
            layouts=ck.layouts,
            stage=stage,
            clock_operator="elapsed_span",
            normalized_clock_mhz=nominal_spec.nominal_clock_mhz,
            participating_ranks=tuple(ck.ordered_rank_placement or (occ_rank,)),
            timed_nodes=("n0",),
        )
        rows.append(row)
        arch = arch_by_id.get(str(op.archetype_id or ""))
        expanded_layers = (
            tuple(int(li) for li in arch.layer_indices)
            if (arch is not None and arch.layer_indices and rep == len(arch.layer_indices))
            else ()
        )
        if expanded_layers:
            per_layer_share = round(row.step_share_pct / float(len(expanded_layers)), ROUND_PCT_DIGITS)
            for li in expanded_layers:
                rev_entries[cid].append(ReverseIndexEntry(
                    cell_id=cell_id, stage=stage, scope=model_scope, native_layer=li,
                    replica=row.replica, rank=occ_rank, op_id=row.op_id, model_id=model_id,
                    sku=sku, phase=phase, seq_len=seq_len, concurrency=concurrency,
                    parallel_arm=arm_display_id, layer_archetype_id=row.layer_archetype_id,
                    occurrence_count=1, model_step_share_pct=per_layer_share,
                ))
        else:
            rev_entries[cid].append(ReverseIndexEntry(
                cell_id=cell_id, stage=stage, scope=model_scope, native_layer=native_layer,
                replica=row.replica, rank=occ_rank, op_id=row.op_id, model_id=model_id,
                sku=sku, phase=phase, seq_len=seq_len, concurrency=concurrency,
                parallel_arm=arm_display_id, layer_archetype_id=row.layer_archetype_id,
                occurrence_count=rep, model_step_share_pct=row.step_share_pct,
            ))
        total_hbm_rw_pipeline += (hbm_rd + hbm_wr) * rep
        if flops_i > 0:
            flops_by_dtype[cube_dtype] += flops_i * rep
        if link_i > 0:
            link_by_tier[op_link_tier] += link_i * rep

    step_summary = _build_step_summary(
        bundle=bundle,
        parsed_case=parsed_case,
        step_req=step_req,
        cell_id=cell_id,
        model_id=model_id,
        sku=sku,
        nominal_spec=nominal_spec,
        phase=phase,
        seq_len=seq_len,
        concurrency=concurrency,
        arm=arm,
        arm_display_id=arm_display_id,
        num_dies=num_dies,
        total_hbm_rw_pipeline=total_hbm_rw_pipeline,
        flops_by_dtype=flops_by_dtype,
        link_by_tier=link_by_tier,
    )
    return rows, rev_entries, key_by_cid, step_summary


def _step_active_weights_and_kv(
    bundle: ExportBundle,
    parsed_case: Case,
    step_req: StepRequest,
    *,
    concurrency: int,
    stage_replicas: int,
) -> tuple[int, int, int]:
    """Compute true cluster active weights, prefix KV reads, and new KV writes for Model MBU."""
    ledger = bundle.memory_ledger
    model = parsed_case.model
    experts = int(getattr(model, "num_experts", 0) or 0)
    top_k = max(1, int(getattr(model, "num_experts_per_tok", 1) or 1))
    step_tokens_per_dp = max(1, int(concurrency) * max(1, int(step_req.tokens)))
    routed_active_frac = (
        inclusion_probability(float(step_tokens_per_dp), min(1.0, float(top_k) / float(experts)))
        if experts > 0 else 1.0
    )
    active_w_pipeline = 0.0
    if ledger.weight_rows:
        for wr in ledger.weight_rows:
            cat_low = f"{wr.group}:{wr.category}:{wr.name}".lower()
            is_routed = (
                experts > 0
                and ("routed" in cat_low or ("expert" in cat_low and "shared" not in cat_low))
            )
            stage_sum = float(sum(int(r) for r in wr.stage_repeats)) if wr.stage_repeats else float(max(1, int(wr.max_stage_repeats)))
            per_layer_b = float(wr.per_die_bytes) / float(max(1, int(wr.max_stage_repeats or 1)))
            pipeline_b = per_layer_b * stage_sum
            active_w_pipeline += pipeline_b * (routed_active_frac if is_routed else 1.0)
    else:
        active_w_pipeline = float(sum(float(v) for v in (ledger.sim_memory_weight_bytes_by_stage or ())))
        if active_w_pipeline <= 0.0:
            active_w_pipeline = float(getattr(ledger, "weight_footprint_per_die_bytes", 0.0) or 0.0)

    kv_b_per_tok_pipeline = 0.0
    fixed_state_pipeline = 0.0
    for cr in (ledger.cache_rows or ()):
        total_layers = sum(int(v) for v in cr.active_layers_per_stage) if cr.active_layers_per_stage else int(cr.max_stage_layers or 0)
        if cr.scales_with_tokens:
            kv_b_per_tok_pipeline += float(cr.bytes_per_token_per_layer) * float(max(0, total_layers))
        else:
            fixed_state_pipeline += float(sum(float(v) for v in cr.per_stage_bytes)) if cr.per_stage_bytes else float(cr.batch_total_bytes)

    ctx_tok = max(0, int(step_req.context))
    new_tok = max(0, int(step_req.tokens))
    kv_read_pipeline = (kv_b_per_tok_pipeline * float(ctx_tok) * float(concurrency)) + (
        fixed_state_pipeline if ctx_tok > 0 else 0.0
    )
    new_kv_pipeline = (kv_b_per_tok_pipeline * float(new_tok) * float(concurrency)) + fixed_state_pipeline
    return (
        max(0, int(round(active_w_pipeline * float(stage_replicas)))),
        max(0, int(round(kv_read_pipeline * float(stage_replicas)))),
        max(0, int(round(new_kv_pipeline * float(stage_replicas)))),
    )


def _build_step_summary(
    *,
    bundle: ExportBundle,
    parsed_case: Case,
    step_req: StepRequest,
    cell_id: str,
    model_id: str,
    sku: str,
    nominal_spec: HardwareNominalSpec,
    phase: PhaseKind,
    seq_len: int,
    concurrency: int,
    arm: ParallelArmSpec,
    arm_display_id: str,
    num_dies: int,
    total_hbm_rw_pipeline: int,
    flops_by_dtype: Mapping[str, int],
    link_by_tier: Mapping[str, int],
) -> StepSummary:
    # Pipeline operators span all arm.pp stages; replicate across stage_replicas = num_dies // arm.pp (zero PP double-count!).
    stage_replicas = max(1, int(num_dies) // max(1, int(arm.pp)))
    active_w_cluster, kv_read_cluster, new_kv_cluster = _step_active_weights_and_kv(
        bundle, parsed_case, step_req, concurrency=concurrency, stage_replicas=stage_replicas,
    )
    total_rw_cluster = total_hbm_rw_pipeline * stage_replicas
    total_flops_cluster = sum(flops_by_dtype.values()) * stage_replicas
    total_link_cluster = sum(link_by_tier.values()) * stage_replicas

    dur_s = max(float(bundle.step_span_us) / US_PER_SECOND, MIN_POSITIVE_DENOM)
    hbm_denom = float(num_dies) * dur_s * nominal_spec.hbm_peak_bytes_per_s
    irreducible = active_w_cluster + kv_read_cluster + new_kv_cluster
    model_mbu = (float(irreducible) / hbm_denom) * PCT_SCALE if hbm_denom > 0.0 else 0.0
    bus_util = (float(total_rw_cluster) / hbm_denom) * PCT_SCALE if hbm_denom > 0.0 else 0.0

    # Order-independent mixed-precision MFU: sum (flops_d / P_peak_d) across dtypes.
    ideal_compute_die_s = sum(
        float(fl * stage_replicas) / nominal_spec.cube_peak_for_dtype(dt)
        for dt, fl in sorted(flops_by_dtype.items()) if fl > 0
    )
    step_mfu = (ideal_compute_die_s / (float(num_dies) * dur_s)) * PCT_SCALE if (dur_s > 0.0 and ideal_compute_die_s > 0.0) else 0.0
    dom_cube_dtype = (
        max(sorted(flops_by_dtype.items()), key=lambda kv: (kv[1], kv[0]))[0]
        if flops_by_dtype else _clean_cube_dtype("bf16", nominal_spec)
    )
    ideal_link_die_s = 0.0
    for tier, nbytes in sorted(link_by_tier.items()):
        if nbytes <= 0:
            continue
        try:
            pk = nominal_spec.link_peak_for_tier(tier)
        except ModelingError:
            pk = 0.0
        if pk > 0.0:
            ideal_link_die_s += float(nbytes * stage_replicas) / pk
    step_link_bu = (ideal_link_die_s / (float(num_dies) * dur_s)) * PCT_SCALE if (dur_s > 0.0 and ideal_link_die_s > 0.0) else 0.0
    dom_link_tier = (
        max(sorted(link_by_tier.items()), key=lambda kv: (kv[1], kv[0]))[0]
        if link_by_tier else next(iter(nominal_spec.raw_link_peaks_bytes_per_s), "d2d_intra_node")
    )
    return StepSummary(
        cell_id=cell_id,
        model_id=model_id,
        sku=sku,
        soc_arch=nominal_spec.soc_arch,
        phase=phase,
        seq_len=seq_len,
        concurrency=concurrency,
        parallel_arm=arm_display_id,
        num_dies=num_dies,
        admissible=True,
        sim_step_us=round(float(bundle.step_span_us), ROUND_US_DIGITS),
        l_star_step_us=round(float(bundle.step_l_star_lean_us), ROUND_US_DIGITS),
        active_weight_bytes=active_w_cluster,
        kv_read_bytes=kv_read_cluster,
        new_kv_write_bytes=new_kv_cluster,
        total_step_hbm_rw_bytes=total_rw_cluster,
        total_step_cube_flops=total_flops_cluster,
        model_mbu_pct=round(model_mbu, ROUND_PCT_DIGITS),
        hbm_bus_util_pct=round(bus_util, ROUND_PCT_DIGITS),
        step_sim_mfu_pct=round(step_mfu, ROUND_PCT_DIGITS),
        total_step_link_bytes=total_link_cluster,
        step_sim_link_bu_pct=round(step_link_bu, ROUND_PCT_DIGITS),
        cube_dtype=dom_cube_dtype,
        link_tier=dom_link_tier,
    )


def load_campaign_manifest(campaign: CampaignManifest | Mapping[str, Any] | str | Path) -> CampaignManifest:
    """Load and cryptographically verify a ``mindie-bench-campaign/1`` manifest."""
    if isinstance(campaign, CampaignManifest):
        return verify_campaign_integrity(campaign.to_dict(include_seal=True), verify_case_raw_sha256=True)
    if isinstance(campaign, (str, Path)):
        p = Path(campaign)
        try:
            raw_bytes = p.read_bytes()
        except TimeoutError:
            raise
        except OSError as exc:
            raise ModelingError(f"campaign_read_failed:{p.name}:{exc}") from exc
        if len(raw_bytes) > MAX_CAMPAIGN_JSON_BYTES:
            raise ModelingError(f"campaign_json_too_large:{p.name}")
        raw_map = json.loads(raw_bytes.decode("utf-8"))
        return verify_campaign_integrity(raw_map, base_dir=p.parent, verify_case_raw_sha256=True)
    return verify_campaign_integrity(campaign, verify_case_raw_sha256=True)


_load_campaign_manifest = load_campaign_manifest


def _match_campaign_plan(
    manifest: OperatorMatrixManifest,
    camp_manifest: CampaignManifest,
) -> BenchPlanV2:
    sku = camp_manifest.version_fingerprint.sku
    for include_fab in (False, True):
        plans = build_bench_plans_by_sku(manifest, include_fabric=include_fab)
        sku_plan = plans.get(sku)
        if sku_plan is not None and camp_manifest.plan_sha256 == sku_plan.plan_sha256:
            return sku_plan
    raise ModelingError(
        f"campaign_plan_mismatch:sku={sku}:plan_sha256={camp_manifest.plan_sha256}"
    )


def _evaluate_matrix_cell(
    *,
    model_ref: str | Path | Mapping[str, Any],
    model_id: str,
    sku: str,
    topo_preset: str | None,
    nom_spec: HardwareNominalSpec,
    arm: ParallelArmSpec,
    arm_display_id: str,
    phase: PhaseKind,
    cliff: SeqCliffPoint,
    conc: int,
    stage: str,
    spec: MatrixSpec,
    layers: int | None,
    feature_track: str,
    usable_host_ddr_bytes: int | None,
) -> tuple[
    HbmAdmissibilityReceipt,
    StepSummary,
    list[OperatorMatrixRow],
    dict[str, list[ReverseIndexEntry]],
    tuple[tuple[str, ...], tuple[tuple[str, str], ...]] | None,
]:
    cell_id = _short_digest((model_id, sku, topo_preset or "", phase, cliff.seq_len, conc, arm_display_id, stage), "c-")
    parsed_case, step_req = _compose_cell_case(
        model_ref, sku, arm=arm, phase=phase, seq_len=cliff.seq_len,
        concurrency=conc, spec=spec, layers=layers, feature_track=feature_track,
        topology_preset_override=topo_preset,
    )
    receipt, _ledger = evaluate_hbm_admissibility(
        parsed_case, phase=phase, seq_len=cliff.seq_len, concurrency=conc,
        cell_id=cell_id, usable_host_ddr_bytes=usable_host_ddr_bytes,
    )
    if not receipt.admissible:
        num_dies = max(1, int(arm.num_devices or (arm.dp * arm.tp * arm.pp * arm.pcp)))
        infeasible_step = StepSummary(
            cell_id=cell_id, model_id=model_id, sku=sku, soc_arch=nom_spec.soc_arch,
            phase=phase, seq_len=cliff.seq_len, concurrency=conc, parallel_arm=arm_display_id,
            num_dies=num_dies, admissible=False, sim_step_us=0.0, l_star_step_us=0.0,
            active_weight_bytes=0, kv_read_bytes=0, new_kv_write_bytes=0,
            total_step_hbm_rw_bytes=0, total_step_cube_flops=0,
            model_mbu_pct=0.0, hbm_bus_util_pct=0.0, step_sim_mfu_pct=0.0,
        )
        return receipt, infeasible_step, [], {}, None
    bundle = build_export_bundle(
        parsed_case, request=step_req, apply_target_envelope=(stage == "after"),
    )
    spec_by_op_id: dict[str, OpSpec] = {}
    try:
        prepared = default_spine().lower(parsed_case, LoweringRequest(step=step_req))
        for block in prepared.graph.executable:
            for item in getattr(block, "ops", ()):
                spec_by_op_id[str(item.op_id)] = item
    except Exception:
        spec_by_op_id = {}
    cell_rows, cell_rev, _c_keys, step_sum = _extract_bundle_rows(
        bundle, parsed_case=parsed_case, step_req=step_req, spec_by_op_id=spec_by_op_id,
        cell_id=cell_id, model_id=model_id, sku=sku,
        nominal_spec=nom_spec, phase=phase, seq_len=cliff.seq_len,
        cliff_kind=cliff.cliff_kind, concurrency=conc, arm=arm, arm_display_id=arm_display_id,
        stage=stage, kv_block_size=int(parsed_case.runtime.kv_block_size or DEFAULT_CACHE_BLOCK_SIZE),
    )
    return receipt, step_sum, cell_rows, cell_rev, _build_cell_dag(bundle.operators)


def _sku_scoped_manifest(manifest: OperatorMatrixManifest, sku: str) -> OperatorMatrixManifest:
    if len(manifest.nominal_specs) <= 1 and all(r.sku == sku for r in manifest.rows):
        return manifest
    sku_rows = tuple(r for r in manifest.rows if r.sku == sku)
    sku_steps = tuple(s for s in manifest.step_summaries if s.sku == sku)
    sku_rev = {
        cid: tuple(e for e in entries if e.sku == sku)
        for cid, entries in manifest.reverse_index.items()
        if any(e.sku == sku for e in entries)
    }
    sku_nom = {sku: manifest.nominal_specs[sku]} if sku in manifest.nominal_specs else manifest.nominal_specs
    return replace(
        manifest,
        nominal_specs=sku_nom,
        rows=sku_rows,
        step_summaries=sku_steps,
        reverse_index=sku_rev,
    )


def _apply_verified_campaign(
    manifest: OperatorMatrixManifest,
    campaign: CampaignManifest | Mapping[str, Any] | str | Path,
    step_dags: Mapping[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]],
) -> OperatorMatrixManifest:
    """Reconcile campaign against per-SKU plan before enriching; require 100% cell coverage for step replay."""
    camp_manifest = load_campaign_manifest(campaign)
    sku_plan = _match_campaign_plan(manifest, camp_manifest)
    sku_matrix = _sku_scoped_manifest(manifest, camp_manifest.version_fingerprint.sku)
    report = reconcile_campaign(camp_manifest, plan=sku_plan, matrix=sku_matrix, strict=False)
    qual_cids = {c.case_id for c in camp_manifest.cases if c.status == "ok" and c.qualified}
    rows_by_cell: dict[str, list[OperatorMatrixRow]] = defaultdict(list)
    for row in manifest.rows:
        rows_by_cell[row.cell_id].append(row)
    covered_dags = {
        cid: dag for cid, dag in step_dags.items()
        if rows_by_cell.get(cid) and all(r.bench_case_id in qual_cids for r in rows_by_cell[cid])
    }
    return enrich_operator_matrix(manifest, camp_manifest, report, step_dags=covered_dags)


def enrich_manifest_with_campaign(
    manifest: OperatorMatrixManifest,
    campaign: CampaignManifest | Mapping[str, Any] | str | Path,
) -> tuple[OperatorMatrixManifest, CampaignManifest]:
    """Verify and attach a ``mindie-bench-campaign/1`` manifest onto a pre-built ``OperatorMatrixManifest``."""
    camp_manifest = load_campaign_manifest(campaign)
    if manifest.campaign_id == camp_manifest.campaign_id:
        return manifest, camp_manifest
    sku_plan = _match_campaign_plan(manifest, camp_manifest)
    sku_matrix = _sku_scoped_manifest(manifest, camp_manifest.version_fingerprint.sku)
    report = reconcile_campaign(camp_manifest, plan=sku_plan, matrix=sku_matrix, strict=False)
    return enrich_operator_matrix(manifest, camp_manifest, report, step_dags=None), camp_manifest


def _evaluate_arm_grid(
    *,
    model_ref: str | Path | Mapping[str, Any],
    model_spec: ModelSpec,
    model_id: str,
    r_sku: str,
    topo_preset: str | None,
    nom_spec: HardwareNominalSpec,
    device: Device,
    arm: ParallelArmSpec,
    arm_disp: str,
    spec: MatrixSpec,
    layers: int | None,
    feature_track: str,
    include_abi_rejection: bool,
    usable_host_ddr_bytes: int | None,
    all_cliffs: list[SeqCliffPoint],
    all_receipts: list[HbmAdmissibilityReceipt],
    all_rows: list[OperatorMatrixRow],
    all_steps: list[StepSummary],
    rev_index: dict[str, list[ReverseIndexEntry]],
    step_dags: dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]],
) -> None:
    abi_div = _resolve_local_kv_abi_divisor(model_spec, arm)
    for conc in spec.concurrencies:
        cliffs = derive_physical_seq_cliffs(
            model_spec, device, arm,
            seq_min=spec.seq_min, seq_max=spec.seq_max,
            anchor_ladder=spec.anchor_ladder,
            concurrency=conc,
            cliff_budget_per_arm=spec.cliff_budget_per_arm,
            auto_physical_cliffs=spec.auto_physical_cliffs,
            sparse_topk_buffer_size=(DEFAULT_HISPARSE_TOPK_BUFFER if arm.sparse_kv_offload else None),
            local_kv_abi_divisor=abi_div,
            include_abi_rejection=include_abi_rejection,
        )
        all_cliffs.extend(cliffs)
        exec_cliffs = [c for c in cliffs if c.split != "abi_rejection"]
        cells = [(phase, cliff, stage) for phase in spec.phases for cliff in exec_cliffs for stage in spec.stages]
        for phase, cliff, stage in cells:
            receipt, step_sum, cell_rows, cell_rev, dag = _evaluate_matrix_cell(
                model_ref=model_ref, model_id=model_id, sku=r_sku,
                topo_preset=topo_preset, nom_spec=nom_spec,
                arm=arm, arm_display_id=arm_disp, phase=phase, cliff=cliff,
                conc=conc, stage=stage, spec=spec, layers=layers,
                feature_track=feature_track, usable_host_ddr_bytes=usable_host_ddr_bytes,
            )
            all_receipts.append(receipt)
            all_steps.append(step_sum)
            all_rows.extend(cell_rows)
            if dag is not None:
                step_dags[step_sum.cell_id] = dag
            for cid, entries in cell_rev.items():
                rev_index[cid].extend(entries)


def build_operator_matrix(
    spec: MatrixSpec,
    *,
    campaign: CampaignManifest | Mapping[str, Any] | str | Path | None = None,
    layers: int | None = None,
    feature_track: str = "shipping",
    include_abi_rejection: bool = False,
    usable_host_ddr_bytes: int | None = None,
) -> OperatorMatrixManifest:
    """[Cohesive-Pipeline] Evaluate a multi-model operator matrix via the unified single spine."""
    profiles = ascend_profiles()
    nominal_specs: dict[str, HardwareNominalSpec] = {}
    devices_by_sku: dict[str, Device] = {}
    sku_targets: list[tuple[str, str, str | None]] = []
    seen_targets: set[tuple[str, str | None]] = set()
    presets_per_sku: dict[str, set[str | None]] = defaultdict(set)
    for raw_sku in spec.skus:
        r_sku, preset = resolve_hardware_alias(raw_sku)
        if r_sku not in profiles:
            raise ModelingError(f"matrix_unknown_sku:{raw_sku}")
        if (r_sku, preset) not in seen_targets:
            seen_targets.add((r_sku, preset))
            sku_targets.append((raw_sku, r_sku, preset))
            presets_per_sku[r_sku].add(preset)
        if r_sku not in nominal_specs:
            nominal_specs[r_sku] = HardwareNominalSpec.from_ascend_profile(r_sku, profiles[r_sku])
            _m, dev = _hardware_machine_direct(r_sku)
            devices_by_sku[r_sku] = dev

    all_cliffs: list[SeqCliffPoint] = []
    all_receipts: list[HbmAdmissibilityReceipt] = []
    all_rows: list[OperatorMatrixRow] = []
    all_steps: list[StepSummary] = []
    rev_index: dict[str, list[ReverseIndexEntry]] = defaultdict(list)
    step_dags: dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {}

    for model_ref in spec.models:
        model_spec = resolve_model_spec_public(model_ref)
        model_id = _resolve_model_label(model_ref, model_spec)
        for raw_sku, r_sku, topo_preset in sku_targets:
            nom_spec = nominal_specs[r_sku]
            device = devices_by_sku[r_sku]
            arms, skipped_arms = enumerate_parallel_arms(
                model_spec, raw_sku, spec.parallel_arms,
                model_ref=model_ref, deployment_mode=spec.deployment_mode,
                ttft_slo_ms=spec.ttft_slo_ms, tpot_slo_ms=spec.tpot_slo_ms,
                seq_len=spec.seq_min, layers=layers, feature_track=feature_track,
            )
            if spec.parallel_arms and not arms:
                raise ModelingError(f"no_feasible_parallel_arms:{model_id}:{raw_sku}:{skipped_arms}")
            disambig_preset = topo_preset if (topo_preset and len(presets_per_sku[r_sku]) > 1) else None
            for arm in arms:
                arm_disp = f"{arm.arm_id}@{disambig_preset}" if disambig_preset else arm.arm_id
                _evaluate_arm_grid(
                    model_ref=model_ref, model_spec=model_spec, model_id=model_id,
                    r_sku=r_sku, topo_preset=topo_preset, nom_spec=nom_spec,
                    device=device, arm=arm, arm_disp=arm_disp, spec=spec,
                    layers=layers, feature_track=feature_track,
                    include_abi_rejection=include_abi_rejection,
                    usable_host_ddr_bytes=usable_host_ddr_bytes,
                    all_cliffs=all_cliffs, all_receipts=all_receipts,
                    all_rows=all_rows, all_steps=all_steps,
                    rev_index=rev_index, step_dags=step_dags,
                )

    manifest = OperatorMatrixManifest(
        spec=spec,
        nominal_specs=nominal_specs,
        cliffs=tuple(all_cliffs),
        hbm_receipts=tuple(all_receipts),
        rows=tuple(all_rows),
        step_summaries=tuple(all_steps),
        reverse_index={cid: tuple(entries) for cid, entries in sorted(rev_index.items())},
    )
    if campaign is not None:
        return _apply_verified_campaign(manifest, campaign, step_dags)
    return manifest


# ---------------------------------------------------------------------------
# Deduplicated Cross-Model BenchPlanV2 Compiler & V1 Compat Projector
# ---------------------------------------------------------------------------

def _row_to_canonical_key(row: OperatorMatrixRow) -> CanonicalKernelKey:
    kind: KernelKeyKind = "collective" if row.bench_builder in COLLECTIVE_BUILDERS else "operator"
    is_kv_op = row.bench_builder in {
        "absorbed_mla", "fused_infer_attention", "incre_flash_attention",
        "fusion_attention", "sparse_flash_attention", "mixed_quant_sparse_mla",
        "block_sparse_attention_c8", "scatter_pa_kv_cache", "scatter_index_cache",
        "lightning_indexer", "pool_key_indexer", "kv_compress_epilog",
    }
    kv_blk = DEFAULT_CACHE_BLOCK_SIZE if is_kv_op else None
    local_kv = int(row.tensor_shapes.get("kv_len", row.seq_len)) if is_kv_op else None
    cp_interleave = kv_blk if (is_kv_op and local_kv is not None and local_kv < row.seq_len) else None
    page_geo = None
    if is_kv_op and kv_blk is not None and local_kv is not None:
        slot_stride = int(row.tensor_shapes.get("head_dim", DEFAULT_HEAD_DIM))
        num_pages = max(1, (local_kv + kv_blk - 1) // kv_blk) * max(1, row.concurrency)
        page_geo = {
            "num_pages": num_pages,
            "page_bytes": kv_blk * slot_stride * max(1, ELEMENT_BYTES.get(row.cube_dtype, 2)),
            "slot_stride": slot_stride,
        }
    offsets = {k: 0 for k in row.strides}
    w_nz = row.layouts.get("weight") == "FRACTAL_NZ"
    return CanonicalKernelKey(
        kind=kind,
        sku=row.sku,
        soc_arch=row.soc_arch,
        abi_backend=row.abi_backend,
        upstream_abi=row.upstream_abi,
        bench_builder=row.bench_builder,
        canonical_shapes=dict(row.tensor_shapes),
        strides=dict(row.strides),
        storage_offsets=offsets,
        dtypes=dict(row.dtypes),
        layouts=dict(row.layouts),
        kv_allocator_block_size=kv_blk,
        kv_kernel_block_size=kv_blk,
        cp_kv_cache_interleave_size=cp_interleave,
        page_geometry=page_geo,
        capture_mode=row.capture_mode,
        aclgraph_padded_bucket=(max(1, row.concurrency) if row.capture_mode == "aclgraph" else None),
        workspace_policy="default",
        stream_dependencies=(),
        residency="cold_hbm",
        quant_meta={"format_code": FRACTAL_NZ_FORMAT_CODE} if w_nz else {},
        op_attrs={"phase": str(row.phase)},
        ordered_rank_placement=(row.participating_ranks if kind == "collective" else None),
        link_tiers=((row.link_tier,) if kind == "collective" else None),
        collective_algorithm=("ring" if kind == "collective" else None),
    )


def _split_for_row(row: OperatorMatrixRow) -> PlanSplit:
    if row.seq_point_kind != "anchor" or row.seq_len in HELDOUT_KV_SET or row.concurrency == HELDOUT_BATCH:
        return "heldout"
    m_rows = row.tensor_shapes.get("m")
    if isinstance(m_rows, int) and m_rows in HELDOUT_MATRIX_ROWS:
        return "heldout"
    return "train"


def _fabric_v2_cases_for_sku(sku: str, soc_arch: str) -> list[BenchPlanCaseV2]:
    hw_view = build_hardware_view(sku)
    raw_fab = build_fabric_cases(hw_view)
    out: list[BenchPlanCaseV2] = []
    for idx, fab in enumerate(raw_fab, start=1):
        nd = fab["nodes"][0]
        ck = CanonicalKernelKey(
            kind="collective",
            sku=sku,
            soc_arch=soc_arch,  # type: ignore[arg-type]
            abi_backend="hccl_mc2",
            upstream_abi=str(nd["upstream_abi"]),
            bench_builder=str(nd["builder"]),
            canonical_shapes={"bytes": int(fab["fabric_bytes"]), "group_size": int(fab["participants"])},
            dtypes={"input": "fp32", "output": "fp32"},
            layouts={"input": "ND", "output": "ND"},
            capture_mode="eager",
            residency="cold_hbm",
            ordered_rank_placement=tuple(int(r) for r in fab["group_ranks"]),
            link_tiers=(str(fab["route"]),),
            collective_algorithm="ring",
        )
        out.append(BenchPlanCaseV2(
            acquisition_uid=f"acq-fab-{idx:04d}",
            case_id=ck.case_id(),
            canonical_key_digest=ck.digest(),
            canonical_key=ck,
            kind="collective",
            level="L1",
            boundary="device_kernel",
            split=fab["split"],
            phase="fabric",
            stage="s1_joint",
            capture_mode="eager",
            residency="cold_hbm",
            clock_operator="elapsed_span",
            work_flops=0,
            work_bytes=int(fab["fabric_bytes"]),
            nodes=({
                "node_id": "fab0",
                "kind": "collective",
                "builder": str(nd["builder"]),
                "upstream_abi": str(nd["upstream_abi"]),
                "abi_backend": "hccl_mc2",
                "group_ranks": list(fab["group_ranks"]),
            },),
            edges=(),
            timed_boundary={"include": ["fab0"]},
            identities={},
        ))
    return out


def build_fabric_operator_curve_rows(
    nominal_specs: Mapping[str, HardwareNominalSpec],
    *,
    measured_by_case_id: Mapping[str, tuple[float, float]] | None = None,
) -> list[FabricOperatorCurvePoint]:
    """Compute physics-grounded collective scaling curves across realized interconnect routes per SKU."""
    meas_map = measured_by_case_id or {}
    points: list[FabricOperatorCurvePoint] = []
    for sku, nom_spec in sorted(nominal_specs.items()):
        machine, _die = _hardware_machine_direct(sku)
        link_by_channel = {str(lnk.channel): lnk for lnk in machine.links}
        hw_view = build_hardware_view(sku)
        raw_cases = build_fabric_cases(hw_view)
        v2_cases = _fabric_v2_cases_for_sku(sku, nom_spec.soc_arch)
        for fab, v2_case in zip(raw_cases, v2_cases):
            op_type = str(fab["recipe"])
            route = str(fab["route"])
            participants = max(2, int(fab["participants"]))
            payload_b = max(1, int(fab["fabric_bytes"]))
            channel = FABRIC_ROUTE_LINK_TIERS.get(route, "d2d_intra_node")
            ring_mult = RING_ALLREDUCE_PHASES if op_type == "all_reduce" else 1
            ring_steps = ring_mult * (participants - 1)
            algo_wire_f = float(payload_b) * float(ring_steps) / float(participants)
            algo_wire_b = max(1, int(round(algo_wire_f)))
            theo_us = nom_spec.nominal_theo_us(link_bytes=algo_wire_f, link_tier=channel)
            link_obj = link_by_channel.get(channel)
            peak_b_s = nom_spec.link_peak_for_tier(channel)
            sustained_b_s = (
                float(link_obj.capacity_bytes_per_ms) * 1000.0
                if (link_obj is not None and float(link_obj.capacity_bytes_per_ms) > 0.0)
                else peak_b_s
            )
            hop_lat_us = float(link_obj.latency_us) if link_obj is not None else 0.0
            lat_part_us = float(ring_steps) * hop_lat_us
            bw_part_us = (algo_wire_f / max(sustained_b_s, MIN_POSITIVE_DENOM)) * US_PER_SECOND
            pred_us = max(theo_us, lat_part_us + bw_part_us, MIN_POSITIVE_DENOM)
            dur_s = pred_us / US_PER_SECOND
            achieved_bw = (algo_wire_f / dur_s) / BYTES_PER_GB
            comm_bu = (algo_wire_f / (dur_s * max(peak_b_s, MIN_POSITIVE_DENOM))) * PCT_SCALE
            bound = "Latency Bound" if lat_part_us > bw_part_us else "Communication Bound"
            meas_pair = meas_map.get(v2_case.case_id)
            points.append(FabricOperatorCurvePoint(
                case_id=v2_case.case_id,
                op_type=op_type,
                kernel_name=str(v2_case.canonical_key.upstream_abi),
                sku=sku,
                soc_arch=nom_spec.soc_arch,
                route=route,
                participants=participants,
                payload_bytes=payload_b,
                algo_wire_bytes=algo_wire_b,
                nominal_theo_us=round(theo_us, ROUND_US_DIGITS),
                predicted_us=round(pred_us, ROUND_US_DIGITS),
                measured_p50_us=round(float(meas_pair[0]), ROUND_US_DIGITS) if meas_pair is not None else None,
                measured_p99_us=round(float(meas_pair[1]), ROUND_US_DIGITS) if meas_pair is not None else None,
                bound_type=bound,
                achieved_bw_gb_s=round(achieved_bw, ROUND_PCT_DIGITS),
                comm_bw_util_pct=round(comm_bu, ROUND_PCT_DIGITS),
            ))
    return points


def build_bench_plans_by_sku(
    spec_or_manifest: MatrixSpec | OperatorMatrixManifest,
    *,
    include_fabric: bool = True,
    layers: int | None = None,
    feature_track: str = "shipping",
) -> dict[str, BenchPlanV2]:
    """Compile deduplicated, SHA-256-sealed ``mindie-bench-plan/2`` plans keyed by SKU."""
    manifest = (
        spec_or_manifest if isinstance(spec_or_manifest, OperatorMatrixManifest)
        else build_operator_matrix(spec_or_manifest, layers=layers, feature_track=feature_track)
    )
    rows_by_sku: dict[str, list[OperatorMatrixRow]] = defaultdict(list)
    for row in manifest.rows:
        rows_by_sku[row.sku].append(row)
    for sku in manifest.nominal_specs:
        rows_by_sku.setdefault(sku, [])

    plans: dict[str, BenchPlanV2] = {}
    for sku, nom_spec in sorted(manifest.nominal_specs.items()):
        sku_rows = rows_by_sku.get(sku, [])
        cases_by_cid: dict[str, BenchPlanCaseV2] = {}
        rev_by_cid: dict[str, list[ReverseIndexEntry]] = defaultdict(list)
        max_world = 1
        seq_idx = 0
        # Preserve exact expanded reverse_index entries from manifest when present!
        for cid_m, entries_m in manifest.reverse_index.items():
            sku_entries = [e for e in entries_m if e.sku == sku]
            if sku_entries:
                rev_by_cid[cid_m].extend(sku_entries)
        for row in sku_rows:
            cid = row.bench_case_id
            if cid not in manifest.reverse_index:
                rev_by_cid[cid].append(ReverseIndexEntry(
                    cell_id=row.cell_id, stage=row.stage, scope=row.model_scope,
                    native_layer=row.native_layer, replica=row.replica, rank=row.rank,
                    op_id=row.op_id, model_id=row.model_id, sku=row.sku, phase=row.phase,
                    seq_len=row.seq_len, concurrency=row.concurrency, parallel_arm=row.parallel_arm,
                    layer_archetype_id=row.layer_archetype_id, occurrence_count=row.occurrence_count,
                    model_step_share_pct=row.step_share_pct,
                ))
            split = _split_for_row(row)
            if cid in cases_by_cid:
                existing = cases_by_cid[cid]
                if existing.split != split and split == "heldout":
                    cases_by_cid[cid] = replace(existing, split="heldout")
                continue
            seq_idx += 1
            ck = _CANONICAL_KEYS_BY_CASE_ID.get(cid)
            if ck is None:
                ck = _row_to_canonical_key(row)
                cid_eff = ck.case_id()
                if cid_eff != cid:
                    rev_by_cid[cid_eff].extend(rev_by_cid.pop(cid, []))
                    cid = cid_eff
            if ck.ordered_rank_placement:
                max_world = max(max_world, max(ck.ordered_rank_placement) + 1, len(ck.ordered_rank_placement))
            stage_tag = "s1_joint" if ck.kind == "collective" else "s0_local"
            node_obj: dict[str, Any] = {
                "node_id": "n0",
                "kind": ck.kind,
                "builder": ck.bench_builder,
                "upstream_abi": ck.upstream_abi,
                "abi_backend": ck.abi_backend,
            }
            if ck.ordered_rank_placement:
                node_obj["group_ranks"] = list(ck.ordered_rank_placement)
            if ck.per_expert_token_counts:
                node_obj["per_expert_token_counts"] = list(ck.per_expert_token_counts)
            cases_by_cid[cid] = BenchPlanCaseV2(
                acquisition_uid=f"acq-{seq_idx:04d}",
                case_id=cid,
                canonical_key_digest=ck.digest(),
                canonical_key=ck,
                kind=ck.kind,
                level="L1",
                boundary="device_kernel",
                split=split,
                phase=str(row.phase),
                stage=stage_tag,
                capture_mode=row.capture_mode,
                residency="cold_hbm",
                clock_operator=row.clock_operator,
                work_flops=row.cube_flops,
                work_bytes=row.hbm_total_bytes + row.link_bytes,
                nodes=(node_obj,),
                edges=(),
                timed_boundary={"include": ["n0"]},
                identities={},
            )
        if include_fabric:
            for fab_case in _fabric_v2_cases_for_sku(sku, nom_spec.soc_arch):
                if fab_case.case_id not in cases_by_cid:
                    cases_by_cid[fab_case.case_id] = fab_case
                    rev_by_cid.setdefault(fab_case.case_id, [])
                    ranks_f = fab_case.canonical_key.ordered_rank_placement
                    if ranks_f:
                        max_world = max(max_world, max(ranks_f) + 1, len(ranks_f))
        ordered_cases = tuple(cases_by_cid[cid] for cid in sorted(cases_by_cid))
        ordered_rev = {cid: tuple(rev_by_cid.get(cid, ())) for cid in sorted(cases_by_cid)}
        digest = _short_digest([c.case_id for c in ordered_cases])
        has_joint = any(c.stage == "s1_joint" for c in ordered_cases)
        plan = BenchPlanV2(
            plan_id=f"plan2-{sku}-{digest}",
            sku=sku,
            soc_arch=nom_spec.soc_arch,
            cases=ordered_cases,
            reverse_index=ordered_rev,
            plan_kind="model_matrix",
            stage="s1_joint" if has_joint else "s0_local",
            cache_partition=manifest.spec.cache_partition,
            world_size=max_world,
        ).with_computed_seal()
        plans[sku] = plan
    return plans


def build_bench_plan(
    spec_or_manifest: MatrixSpec | OperatorMatrixManifest,
    *,
    sku: str | None = None,
    include_fabric: bool = True,
    compat_v1: bool = False,
    layers: int | None = None,
    feature_track: str = "shipping",
) -> BenchPlanV2 | dict[str, Any]:
    """Build one sealed ``BenchPlanV2`` (or v1-compat dict when ``compat_v1=True``)."""
    plans = build_bench_plans_by_sku(
        spec_or_manifest, include_fabric=include_fabric, layers=layers, feature_track=feature_track,
    )
    if sku is not None:
        r_sku, _ = resolve_hardware_alias(sku)
        if r_sku not in plans:
            raise ModelingError(f"bench_plan_sku_not_found:{sku}")
        chosen = plans[r_sku]
    else:
        if len(plans) != 1:
            raise ModelingError(f"build_bench_plan_requires_sku_for_multi_sku:{sorted(plans)}")
        chosen = next(iter(plans.values()))
    return render_subgraph_plan_v1_compat(chosen) if compat_v1 else chosen


def render_subgraph_plan_v1_compat(plan_v2: BenchPlanV2) -> dict[str, Any]:
    """Project a sealed ``BenchPlanV2`` losslessly to ``mindie-bench-subgraph-plan/1``."""
    v1_cases: list[dict[str, Any]] = []
    v1_rev: dict[str, list[dict[str, Any]]] = {}
    exec_modes: set[str] = set()
    for case in plan_v2.cases:
        ck = case.canonical_key
        exec_modes.add(case.capture_mode)
        v1_nodes: list[dict[str, Any]] = []
        for nd in case.nodes:
            v1_nodes.append({
                "schema": SUBGRAPH_NODE_V1_SCHEMA,
                "node_id": str(nd.get("node_id") or "n0"),
                "kind": str(nd.get("kind") or case.kind),
                "op": ck.bench_builder,
                "display_name": f"{case.kind}:{ck.bench_builder}",
                "builder": ck.bench_builder,
                "upstream_abi": ck.upstream_abi,
                "abi_backend": ck.abi_backend,
                "phase": case.phase,
                "dtype": ck.dtypes.get("input", "bf16"),
                "output_dtype": ck.dtypes.get("output", ck.dtypes.get("input", "bf16")),
                "layout": ck.layouts.get("input", "ND"),
                "shape": dict(ck.canonical_shapes),
                "inputs": [],
                "outputs": [],
                "implementation": ck.upstream_abi,
                "implementation_source": "builder_upstream_abi",
                "kernel": ck.upstream_abi,
                **({"group_ranks": list(ck.ordered_rank_placement)} if ck.ordered_rank_placement else {}),
            })
        entries = plan_v2.reverse_index.get(case.case_id, ())
        occ_total = sum(e.occurrence_count for e in entries)
        v1_cases.append({
            "case_id": case.case_id,
            "acquisition_uid": case.acquisition_uid,
            "canonical_key_digest": case.canonical_key_digest,
            "kind": case.kind,
            "level": case.level,
            "boundary": case.boundary,
            "split": case.split,
            "phase": case.phase,
            "stage": case.stage,
            "sku": plan_v2.sku,
            "execution_mode": case.capture_mode,
            "execution_modes": [case.capture_mode],
            "clock_operator": case.clock_operator,
            "occurrence_count": occ_total,
            "work_flops": case.work_flops,
            "work_bytes": case.work_bytes,
            "nodes": v1_nodes,
            "edges": [
                {"from": str(e[0]), "to": str(e[1]), "kind": str(e[2]) if len(e) > 2 else "data"}
                for e in case.edges if len(e) >= 2
            ],
            "timed_boundary": dict(case.timed_boundary),
            "timing": {"boundary": case.boundary, "protocol": "v2_compat_projection"},
            **({"group_ranks": list(ck.ordered_rank_placement)} if ck.ordered_rank_placement else {}),
        })
        v1_rev[case.case_id] = [e.to_dict() for e in entries]
    return {
        "schema": SUBGRAPH_PLAN_V1_SCHEMA,
        "source_schema": BENCH_PLAN_V2_SCHEMA,
        "plan_id": plan_v2.plan_id,
        "plan_sha256": plan_v2.plan_sha256,
        "sku": plan_v2.sku,
        "soc_arch": plan_v2.soc_arch,
        "stage": plan_v2.stage,
        "world_size": plan_v2.world_size,
        "execution_modes": sorted(exec_modes or {"eager"}),
        "cases": v1_cases,
        "reverse_index": v1_rev,
        "summary": {
            "cases": len(v1_cases),
            "splits": dict(sorted(Counter(c["split"] for c in v1_cases).items())),
            "by_stage": dict(sorted(Counter(c["stage"] for c in v1_cases).items())),
            "occurrences": sum(c["occurrence_count"] for c in v1_cases),
        },
    }


# ---------------------------------------------------------------------------
# CLI Parsing Helpers
# ---------------------------------------------------------------------------

def _parse_int_suffix(token: str) -> int:
    s = str(token or "").strip().lower().replace("_", "")
    if not s:
        raise ModelingError("empty_integer_token")
    if s.endswith("k"):
        return int(round(float(s[:-1]) * 1024))
    if s.endswith("m"):
        return int(round(float(s[:-1]) * 1024 * 1024))
    return int(s)


def parse_seq_range(text: str) -> tuple[int, int]:
    """Parse ``'512..400k'`` or ``'4096'`` into ``(seq_min, seq_max)``."""
    raw = str(text or "").strip()
    if ".." in raw:
        left, right = raw.split("..", 1)
        lo, hi = _parse_int_suffix(left), _parse_int_suffix(right)
    elif ":" in raw:
        left, right = raw.split(":", 1)
        lo, hi = _parse_int_suffix(left), _parse_int_suffix(right)
    else:
        lo = hi = _parse_int_suffix(raw)
    if lo < 1 or hi < lo:
        raise ModelingError(f"invalid_seq_range:{text!r}")
    return lo, hi


def parse_parallel_arm_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    """Normalize comma/whitespace-separated parallel-arm tokens for ``enumerate_parallel_arms``."""
    out: list[str] = []
    for tok in tokens:
        for part in str(tok or "").split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return tuple(out)


__all__ = [
    "BOUNDARY_BUILDER",
    "BUILDERS",
    "COLLECTIVE_BUILDERS",
    "CONTROL_FAMILIES",
    "DEFAULT_LEGACY_SKUS",
    "ONBOARDING_HINT_SCHEMA",
    "SUBGRAPH_NODE_V1_SCHEMA",
    "SUBGRAPH_PLAN_V1_SCHEMA",
    "SUPPORTED_FABRIC_ROUTES",
    "BenchGeometryError",
    "BenchTopologyError",
    "FABRIC_ROUTE_LINK_TIERS",
    "FabricOperatorCurvePoint",
    "PlanError",
    "UnmodeledCapabilityError",
    "attention_shape_dims",
    "build_bench_plan",
    "build_bench_plans_by_sku",
    "build_canonical_kernel_key",
    "build_fabric_cases",
    "build_fabric_operator_curve_rows",
    "build_hardware_view",
    "build_operator_formula",
    "build_operator_matrix",
    "build_v1_node_fields",
    "collapse_device_kernel_groups",
    "collective_group_size",
    "declared_implementation",
    "derive_physical_seq_cliffs",
    "display_name",
    "enrich_manifest_with_campaign",
    "enumerate_parallel_arms",
    "evaluate_hbm_admissibility",
    "expert_bank_bytes",
    "explicit_ledger",
    "fabric_route_ranks",
    "is_ladder_kv_node",
    "is_onboarding_family",
    "load_campaign_manifest",
    "parse_parallel_arm_tokens",
    "parse_seq_range",
    "render_subgraph_plan_v1_compat",
    "resolve_abi_backend",
    "resolve_builder_strict",
    "resolve_collective_cosets",
    "resolve_collective_group_ranks",
    "resolve_fabric_collective_abi",
    "resolve_onboarding_builder_and_abi",
    "slot_plan_for",
]
