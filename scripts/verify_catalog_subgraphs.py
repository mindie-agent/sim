#!/usr/bin/env python3
"""39+ catalog model 1-layer subgraph verification harness (Gate A + Gate B).

[Role]
  Provide two-gate physical verification across all 39+ catalog models (39 core
  plus 4 architecture presets = 43 models):
  * Gate A (--gate-a): 100% graph-structure and cross-rank physical conservation
    across Pure-DP+EP, TP+EP, DP+DCP(+Local-TP), PCP, PCP+DCP, feature overlays
    (Quant BF16/W8A8/W4A8/C8/MXFP, MTP q_len=1 vs K+1, MC2/MSO, aclgraph padded
    RequestView batches), official vllm-ascend tutorial deployments (Colocated
    and PD P/D nodes), hardware perturbation continuity (AIC wave steps and HBM
    bandwidth regime transitions), and PD Decode DCP vs Local-TP crossover.
  * Gate B (--gate-b): Offline deterministic evaluator comparing live NPU
    1-layer ``kernel_details.csv`` captures (A2, A3, A5-DT) against
    ``default_spine()`` via 1:1 topology isomorphism diff, strict L2
    ``|T_pred - T_obs| <= max(0.10 * T_obs, 2.0 us)``, and Two-Tier L1-in-L2
    checks (Median <= 8%, P90 <= 12%, cliff <= max(15%, 3.0 us)).

[Invariants]
  * Reuses canonical ``mindie_sim`` structs (``IterationPlan``, ``RequestView``,
    ``ParallelSpec``, ``default_spine``, ``compose_case``,
    ``OperatorPlanningContext``, ``derive_hardware_sku``) with zero parallel IR.
  * Zero device context or ``torch_npu`` imports; read-only over captures.
  * Never filters/skips small shapes or applies empirical tuning multipliers;
    explicitly reports per-case live coverage tiers without silent skips.

[Timeouts & Locks]
  Bounded CPU evaluation and local JSON reads (< 50 MiB); zero locks or NPU use.

[Side-Effect Policy]
  Read-only except when ``--output`` or ``--export-specs`` is requested.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
import gc
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from mindie_sim import target_graphs  # noqa: E402
from mindie_sim.catalog.cache import layer_cache_components  # noqa: E402
from mindie_sim.catalog.context_parallel import is_v41_replicated_cache_model  # noqa: E402
from mindie_sim.catalog.fleet import ARCHITECTURE_PRESET_CATALOG, CORE_CATALOG_ENTRIES  # noqa: E402
from mindie_sim.catalog.paired_sfa import execution as paired_sfa_execution  # noqa: E402
from mindie_sim.compose import compose_case  # noqa: E402
from mindie_sim.hardware_derivation import derive_hardware_sku  # noqa: E402
from mindie_sim.model_catalog import load_model  # noqa: E402
from mindie_sim.observations.multilevel_gate import (  # noqa: E402
    exceeds_physical_tolerance as _canonical_exceeds_physical_tolerance,
)
from mindie_sim.prediction.mechanism import Geometry  # noqa: E402
from mindie_sim.prediction.program import RecipeBook  # noqa: E402
from mindie_sim.prediction.results import ReplayOptions, StepRequest  # noqa: E402
from mindie_sim.prediction.spine import PredictionSpine, PreparedStep, default_spine  # noqa: E402
from mindie_sim.prediction.structure import parse_case  # noqa: E402
from mindie_sim.primitives import load_primitives  # noqa: E402
from mindie_sim.schema.common import ModelingError  # noqa: E402
from mindie_sim.schema.parallel import ParallelSpec  # noqa: E402
from mindie_sim.schema.workload import IterationPlan, RequestView  # noqa: E402
from mindie_sim.stack_registry import LoweringRequest  # noqa: E402
from tools._fleet_400k_reconcile import nearest_rank_p90  # noqa: E402

PHYSICAL_TOL_PCT = 10.0
PHYSICAL_ATOL_NS = 2000.0
L1_MEDIAN_APE_LIMIT_PCT = 8.0
L1_P90_APE_LIMIT_PCT = 12.0
L1_CLIFF_TOL_PCT = 15.0
L1_CLIFF_ATOL_US = 3.0
BUBBLE_REL_LIMIT = 0.10
BUBBLE_ATOL_US = 20.0
PERCENT_SCALE = 100.0
US_TO_NS = 1000.0
MS_TO_US = 1000.0
FLOP_CONSERVATION_REL_TOL = 0.015
GIB = 1024 ** 3
SYNTH_WEIGHT_CAP_BYTES = int(4.5 * GIB)
DEFAULT_AUDIT_LAYERS = 4
DECODE_AUDIT_CONTEXT = 4096
DECODE_AUDIT_SEQS = 4
PREFILL_AUDIT_TOKENS = 2048
PREFILL_LIVE_TOKENS = 1024
ACLGRAPH_ACTIVE_SEQS = 48
ACLGRAPH_PAD_SEQS = 16
MTP_PROPOSAL_K3 = 3
MTP_PROPOSAL_K1 = 1
MTP_QUERY_LEN_K3 = 4
LIVE_DECODE_CONTEXT = 2048
LIVE_DECODE_SEQS = 4
LIVE_LONG_CONTEXT = 16384
LIVE_HIGH_CONC_SEQS = 32
LIVE_LOW_CONC_SEQS = 2
WORLD_SINGLE = 1
WORLD_QUAD = 4
WORLD_OCT = 8
WORLD_HEX = 16
WORLD_MOE_LARGE = 32
PERTURB_AIC_COUNTS = (28, 32, 36)
PERTURB_HBM_BW_TBPS = (4.0, 2.4, 1.6, 1.4)
TBPS_TO_BPS = 1e12
PERTURB_REL_MONOTONIC_TOL = 1e-4
CROSSOVER_LOW_SEQS = 2
CROSSOVER_LOW_CTX = 2048
CROSSOVER_HIGH_SEQS = 64
CROSSOVER_HIGH_CTX = 32768
DEFAULT_GATE_A_SKU = "Ascend910_9362"
DEFAULT_A5_SKU = "Ascend950DT_9582"
DEFAULT_A5_PR_SKU = "Ascend950PR_9579"
DEFAULT_A2_SKU = "Ascend910B4"
DEFAULT_KV_BLOCK_SIZE = 128
LOCAL_TP_WIDTH_FOUR = 4

FEATURE_TRACKS = ("raw_eager", "merged", "unmerged_pr", "expected_ideal")
TRACK_MONOTONIC_TOL_MS = 1e-4

TIER_VLLM_1LAYER = "live_vllm_1layer_verified"
TIER_TORCHNPU_COMPOSITE = "live_torchnpu_composite_verified"
TIER_UNMERGED_ABSENT = "unmerged_op_container_absent"
TIER_RUNNER_UNSUPPORTED = "runner_overlay_unsupported"
VALID_COVERAGE_TIERS = frozenset({
    TIER_VLLM_1LAYER, TIER_TORCHNPU_COMPOSITE, TIER_UNMERGED_ABSENT, TIER_RUNNER_UNSUPPORTED,
})

ALL_CATALOG_MODELS: tuple[str, ...] = tuple(
    list(CORE_CATALOG_ENTRIES.keys())
    + [k for k in ARCHITECTURE_PRESET_CATALOG if k not in CORE_CATALOG_ENTRIES]
)
ALL_CATALOG_MODEL_SET: frozenset[str] = frozenset(ALL_CATALOG_MODELS)
AUX_NON_ATTN_KERNEL_TYPES = frozenset({
    "slice", "stridedslice", "transpose", "transdata", "cast", "fill", "fills",
    "concatd", "splitd", "reshape", "zeroslike", "broadcastto", "copy", "padv3",
})
CONTROL_HELPER_KERNEL_TYPES = frozenset({
    "hccllaunchaicpukernel", "memset", "memcpyasync",
    "notifywait", "notifyrecord", "eventwait", "eventrecord",
})
STREAM_BLOCKER_ITERS_EXPECTED = 4
EXPECTED_CAPTURE_SCHEMA_VERSION = "mindie-sim-1layer-live-capture/2"
KV_LEDGER_REL_TOL = 0.02
KV_LAYER_ABS_TOL = 1e-2
ALGEBRAIC_FLOP_LOW_FACTOR = 0.5
ALGEBRAIC_FLOP_HIGH_FACTOR = 2.0
DENSE_FFN_FLOP_LOW = 0.92
DENSE_FFN_FLOP_HIGH = 1.08
MOE_FFN_FLOP_LOW = 0.80
MOE_FFN_FLOP_HIGH = 1.25
COLLECTIVE_ANALYTICAL_MIN_REL = 0.45
MTP_Q4_MIN_FLOP_RATIO = 2.5
MIN_COLLECTIVE_LINK_BYTES = 1024.0
COLLECTIVE_PAYLOAD_MIN_REL = 0.25
COLLECTIVE_PAYLOAD_MAX_REL = 4.0
MIN_OBSERVED_COLLECTIVE_BYTES = 64
MIN_GEMM_INNER_DIM = 64
FRACTAL_NZ_TILE = 16
LAYER_WINDOW_MIN_CONSISTENCY = 0.75
LAYER_WINDOW_MAX_CONSISTENCY = 1.35
LAYER_CORE_COMPUTE_MIN_RATIO = 0.70
MIN_PLAUSIBLE_AICORE_FREQ_MHZ = 50.0
CLOCK_CONSISTENCY_MIN_RATIO = 0.80
MIN_QUANT_OP_COVERAGE_RATIO = 0.5
MIN_DENSE_QUANT_OPS = 4
MIN_EXPERT_QUANT_OPS = 2
STREAM_JSON_THRESHOLD_BYTES = 16 * 1024 * 1024
JSON_STREAM_CHUNK_SIZE = 65536
MAX_SINGLE_JSON_VALUE_BYTES = 8 * 1024 * 1024
STREAM_CASE_BATCH_SIZE = 4
RSS_ISOLATE_DP_THRESHOLD = 16
DEFAULT_EXPECTED_LIVE_SOCS = (DEFAULT_A2_SKU, DEFAULT_GATE_A_SKU, DEFAULT_A5_SKU)
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{16}$")
DTYPE_BYTES_TABLE: dict[str, float] = {
    "fp32": 4.0, "float32": 4.0, "bf16": 2.0, "fp16": 2.0, "float16": 2.0,
    "int8": 1.0, "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "mxfp8": 1.0,
    "int4": 0.5, "mxfp4": 0.5, "none": 0.0, "": 0.0,
}


def _sanitize_diagnostic(text: str) -> str:
    """Redact filesystem paths (with spaces), emails, IPv4/IPv6 (all-letter & zoned), and hex digests."""
    out = re.sub(r"/(?:Users|home|root)/[^\r\n'\"\]\[;,{}()<>]+", "/<REDACTED_HOME>", str(text))
    out = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "<REDACTED_EMAIL>", out)
    out = re.sub(
        r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        "<REDACTED_IPV6>", out,
    )
    out = re.sub(
        r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{1,4}:){2,6}:(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*)?(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        "<REDACTED_IPV6>", out,
    )
    out = re.sub(
        r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{1,4}:)?:(?:[0-9a-fA-F]{1,4}:){1,6}[0-9a-fA-F]{1,4}(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        "<REDACTED_IPV6>", out,
    )
    out = re.sub(
        r"(?<![0-9a-fA-F:])(?:[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4})(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        lambda m: "<REDACTED_IPV6>" if (any(c.isdigit() for c in m.group(0)) or len(m.group(0).split("::")[0]) == 4 or len(m.group(0).split("::")[-1].split("%")[0]) == 4) else m.group(0),
        out,
    )
    out = re.sub(r"\b(?!127\.0\.0\.1\b)\d{1,3}(?:\.\d{1,3}){3}\b", "<REDACTED_IP>", out)
    digest_pat = r"@?" + "sha" + r"256:[0-9a-fA-F]{16,}"
    out = re.sub(digest_pat, "<REDACTED_DIGEST>", out)
    return re.sub(r"\b[0-9a-fA-F]{64}\b", "<REDACTED_HEX64>", out)

ATTN_FAMILIES = frozenset({"attention", "compressed_attention"})
LINEAR_ATTN_FAMILIES = frozenset({
    "gdn", "kda", "linear_attention", "chunk_linear_attention",
    "recurrent_linear_attention", "causal_conv1d", "causal_conv",
    "recurrent_scan", "short_convolution", "chunk_scan",
    "gated_delta_net", "kda_recurrent", "kda_chunk",
})
INDEXER_FAMILIES = frozenset({"indexer", "indexer_quant_cache", "key_pool"})
MOE_FAMILIES = frozenset({
    "moe", "grouped_matmul", "moe_routing", "moe_gating", "moe_dispatch",
    "moe_combine", "mc2", "dispatch", "combine", "swiglu", "moe_ffn",
})
COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "allgather", "reducescatter", "reduce_scatter", "alltoall",
    "all_to_all", "all_to_all_v", "dcp_allgather", "dcp_alltoall", "mc2",
    "dispatch", "combine", "collective", "state_transfer",
})
FORBIDDEN_IMPL_MARKERS = ("resourceenvelope", "unmodeled_operator_mechanism", "fallback")
C8_QUANT_SPEC: dict[str, Any] = {
    "name": "w8a8c8",
    "weight_format": "int8",
    "activation_format": "int8",
    "expert_weight_format": "int8",
    "kv_dtype": "int8",
    "index_dtype": "bf16",
    "index_scale_dtype": "none",
    "native_format": "bf16",
}
W8A8_FP8KV_QUANT_SPEC: dict[str, Any] = {
    "name": "w8a8_fp8kv",
    "weight_format": "int8",
    "activation_format": "int8",
    "expert_weight_format": "int8",
    "kv_dtype": "fp8_e4m3",
    "index_dtype": "fp8_e4m3",
    "native_format": "bf16",
}
MXFP4_FP8KV_QUANT_SPEC: dict[str, Any] = {
    "name": "mxfp4_fp8kv",
    "weight_format": "mxfp4",
    "activation_format": "mxfp4",
    "expert_weight_format": "mxfp4",
    "kv_dtype": "fp8_e4m3",
    "native_format": "bf16",
}

# Canonical documented vllm-ascend deployment matrix (Colocated, PD P-node, PD D-node)
# extracted from vllm-ascend/docs/source/tutorials/models/*.md across A2, A3, and 950DT.
VA_TUTORIAL_DEPLOYMENTS: tuple[dict[str, Any], ...] = (
    # GLM-5.3-Flash (A5-950DT / A3 / A2: Colocated + PD P/D)
    {"id": "glm53f_a5_coloc", "model": "glm-5.3-flash", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 8, "ep": 8, "quant": "mxfp8", "mtp": 5},
    {"id": "glm53f_a5_pd_p", "model": "glm-5.3-flash", "hw": DEFAULT_A5_SKU, "role": "pd_prefill", "phase": "prefill", "dp": 1, "tp": 1, "pcp": 8, "ep": 8, "quant": "mxfp8"},
    {"id": "glm53f_a5_pd_d", "model": "glm-5.3-flash", "hw": DEFAULT_A5_SKU, "role": "pd_decode", "phase": "decode", "dp": 8, "tp": 1, "ep": 8, "quant": "mxfp8", "mso": True},
    {"id": "glm53f_a3_coloc", "model": "glm-5.3-flash", "hw": DEFAULT_GATE_A_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 16, "ep": 16, "quant": "w8a8", "mtp": 5, "mso": True},
    {"id": "glm53f_a3_pd_p", "model": "glm-5.3-flash", "hw": DEFAULT_GATE_A_SKU, "role": "pd_prefill", "phase": "prefill", "dp": 2, "tp": 1, "pcp": 8, "ep": 16, "quant": "w8a8"},
    {"id": "glm53f_a3_pd_d", "model": "glm-5.3-flash", "hw": DEFAULT_GATE_A_SKU, "role": "pd_decode", "phase": "decode", "dp": 8, "tp": 2, "ep": 16, "quant": "w8a8", "mtp": 3, "mso": True},
    {"id": "glm53f_a2_coloc", "model": "glm-5.3-flash", "hw": DEFAULT_A2_SKU, "role": "colocated", "phase": "decode", "dp": 2, "tp": 8, "ep": 16, "quant": "w8a8"},
    {"id": "glm53f_a2_pd_d", "model": "glm-5.3-flash", "hw": DEFAULT_A2_SKU, "role": "pd_decode", "phase": "decode", "dp": 8, "tp": 2, "ep": 16, "quant": "w8a8"},
    # DeepSeek-V4.1-Flash (950DT / A3 / A2: Colocated + PD P/D)
    {"id": "dsv41f_a5_coloc", "model": "deepseek-v4.1-flash", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 8, "tp": 1, "ep": 8, "mtp": 5},
    {"id": "dsv41f_a3_coloc_w4a8", "model": "deepseek-v4.1-flash", "hw": DEFAULT_GATE_A_SKU, "role": "colocated", "phase": "decode", "dp": 4, "tp": 4, "ep": 16, "quant": "w4a8", "mtp": 5},
    {"id": "dsv41f_a3_pd_d", "model": "deepseek-v4.1-flash", "hw": DEFAULT_GATE_A_SKU, "role": "pd_decode", "phase": "decode", "dp": 16, "tp": 1, "ep": 16, "quant": "w4a8"},
    {"id": "dsv41f_a2_coloc_w4a8", "model": "deepseek-v4.1-flash", "hw": DEFAULT_A2_SKU, "role": "colocated", "phase": "decode", "dp": 2, "tp": 8, "ep": 16, "quant": "w4a8"},
    # Kimi-K3.1 (A3 / 950DT PD P/D)
    {"id": "k31_a3_pd_d_dcp", "model": "kimi-k3.1", "hw": DEFAULT_GATE_A_SKU, "role": "pd_decode", "phase": "decode", "dp": 1, "tp": 16, "dcp": 16, "ep": 16, "quant": "w4a8", "mso": True},
    {"id": "k31_a5_pd_p", "model": "kimi-k3.1", "hw": DEFAULT_A5_SKU, "role": "pd_prefill", "phase": "prefill", "dp": 1, "tp": 1, "pcp": 8, "ep": 8},
    # DeepSeek-V3.2 / R1 / V4-Pro / V4-Flash
    {"id": "dsv32_a3_dp_ep", "model": "deepseek-v3.2", "hw": DEFAULT_GATE_A_SKU, "role": "pd_decode", "phase": "decode", "dp": 16, "tp": 1, "ep": 16, "quant": "w8a8"},
    {"id": "dsr1_a3_coloc", "model": "deepseek-r1", "hw": DEFAULT_GATE_A_SKU, "role": "colocated", "phase": "decode", "dp": 2, "tp": 8, "ep": 16, "quant": "w8a8"},
    {"id": "dsv4pro_a5_pd_p", "model": "deepseek-v4-pro-0813", "hw": DEFAULT_A5_SKU, "role": "pd_prefill", "phase": "prefill", "dp": 1, "tp": 1, "pcp": 8, "ep": 8},
    {"id": "dsv4f_a5_coloc", "model": "deepseek-v4-flash-0731", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 8, "tp": 1, "ep": 8},
    # GLM-5.2 / GLM-5.3
    {"id": "glm52_a5_coloc", "model": "glm-5.2", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 8, "ep": 8},
    {"id": "glm53_a3_pd_p", "model": "glm-5.3", "hw": DEFAULT_GATE_A_SKU, "role": "pd_prefill", "phase": "prefill", "dp": 1, "tp": 1, "pcp": 8, "ep": 8, "quant": "w8a8"},
    # Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8 / MiniMax-M3 / Step-5 / Gemma-4
    {"id": "q3_32b_a3_coloc", "model": "qwen3-32b", "hw": DEFAULT_GATE_A_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 4, "ep": 1, "quant": "w8a8"},
    {"id": "q3_30b_a3_coloc", "model": "qwen3-30b-a3b", "hw": DEFAULT_GATE_A_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 4, "ep": 4, "quant": "w8a8"},
    {"id": "q35_397b_a5_coloc", "model": "qwen3.5-397b-a17b", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 8, "ep": 8, "quant": "mxfp8", "mtp": 3},
    {"id": "q36_35b_a3_coloc", "model": "qwen3.6-35b-a3b", "hw": DEFAULT_GATE_A_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 2, "ep": 2, "quant": "w8a8", "mtp": 3},
    {"id": "q38_27b_a5_coloc", "model": "qwen3.8-27b", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 2, "ep": 1, "quant": "mxfp8", "mtp": 3},
    {"id": "q38_fn_a5_coloc", "model": "qwen3.8-flash-next", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 4, "ep": 4, "mtp": 3},
    {"id": "q38_24t_a5_coloc", "model": "qwen3.8-2.4t-a95b", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 2, "tp": 8, "ep": 16, "quant": "mxfp8"},
    {"id": "mm_m3_a3_coloc", "model": "minimax-m3", "hw": DEFAULT_GATE_A_SKU, "role": "colocated", "phase": "decode", "dp": 4, "tp": 4, "ep": 16, "quant": "w8a8", "mtp": 3},
    {"id": "mm_m31f_a5_coloc", "model": "minimax-m3.1-flash", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 2, "tp": 4, "ep": 8, "quant": "mxfp8"},
    {"id": "step5_a5_coloc", "model": "step-5-preview", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 4, "tp": 4, "ep": 16, "quant": "mxfp8", "mtp": 3},
    {"id": "gemma4_26b_a5_coloc", "model": "gemma-4-26b-a4b", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 2, "tp": 4, "ep": 8},
    {"id": "gemma4_31b_a5_coloc", "model": "gemma-4-31b", "hw": DEFAULT_A5_SKU, "role": "colocated", "phase": "decode", "dp": 1, "tp": 4, "ep": 1},
)


def exceeds_physical_tolerance(
    predicted_ns: float,
    measured_ns: float,
    *,
    threshold_pct: float = PHYSICAL_TOL_PCT,
    atol_ns: float = PHYSICAL_ATOL_NS,
) -> bool:
    """Unified physical tolerance check ``|T_pred - T_obs| > max(10% * T_obs, 2.0 us)``."""
    return bool(_canonical_exceeds_physical_tolerance(
        int(round(predicted_ns)), int(round(measured_ns)),
        threshold_pct=threshold_pct, atol_ns=atol_ns,
    ))


def _physical_limit_us(measured_us: float) -> float:
    return max(PHYSICAL_TOL_PCT / PERCENT_SCALE * abs(measured_us), PHYSICAL_ATOL_NS / US_TO_NS)


def _op_work_flops(
    op: Any,
    resolved_map: Mapping[tuple[int, str], Any],
    replica: int,
    book: RecipeBook,
    geometry: Geometry,
    prepare_errors: list[str],
) -> float:
    """Extract authoritative executed cube FLOPs from governed plan or RecipeBook."""
    res = resolved_map.get((replica, op.op_id))
    target_op = res.op if res is not None else op
    if target_op.matrix_flops:
        return float(sum(float(v) for _, v in target_op.matrix_flops))
    if target_op.flops:
        return float(target_op.flops)
    try:
        prepared_op, plan = book.prepare(target_op, geometry)
    except Exception as exc:  # noqa: BLE001
        prepare_errors.append(f"{op.op_id}:prepare_error:{type(exc).__name__}:{exc}")
        return 0.0
    work = getattr(plan, "work", None)
    matrix_work = getattr(work, "matrix", None) if work is not None else None
    if matrix_work:
        return float(sum(float(v) for v in matrix_work.values()))
    if prepared_op.flops:
        return float(prepared_op.flops)
    return 0.0


def _op_attn_logical_and_cube_flops(
    op: Any,
    resolved_map: Mapping[tuple[int, str], Any],
    replica: int,
    book: RecipeBook,
    geometry: Geometry,
    prepare_errors: list[str],
    base_flops: float,
) -> tuple[float, float]:
    """Separate logical attention FLOPs from Cube FRACTAL_NZ head-padded work."""
    res = resolved_map.get((replica, op.op_id))
    target_op = res.op if res is not None else op
    cube_flops = base_flops
    if target_op.flops and not target_op.matrix_flops:
        try:
            _, plan = book.prepare(target_op, geometry)
            m_work = getattr(getattr(plan, "work", None), "matrix", None)
            if m_work:
                cube_flops = float(sum(float(v) for v in m_work.values()))
        except Exception as exc:  # noqa: BLE001
            prepare_errors.append(f"{op.op_id}:prepare_error:{type(exc).__name__}:{exc}")
        return float(target_op.flops), max(cube_flops, float(target_op.flops))
    if cube_flops <= 0.0:
        return 0.0, 0.0
    meta = dict(op.metadata) if op.metadata else {}
    pgeo = meta.get("paged_mla_geometry")
    if isinstance(pgeo, Mapping):
        l_heads = int(pgeo.get("logical_heads") or 0)
        if l_heads > 0:
            padded_heads = ((l_heads + FRACTAL_NZ_TILE - 1) // FRACTAL_NZ_TILE) * FRACTAL_NZ_TILE
            if padded_heads > l_heads:
                return cube_flops * (float(l_heads) / float(padded_heads)), cube_flops
    return cube_flops, cube_flops


def _op_dep_sig(op: Any) -> tuple[str, ...]:
    """Return normalized dependency op-id suffixes (stripping replica/step prefix)."""
    out: list[str] = []
    for d in getattr(op, "deps", ()):
        ds = str(d)
        parts = ds.split(".", 1)
        out.append(parts[1] if len(parts) > 1 else ds)
    return tuple(sorted(out))


def _sum_replica_flops(
    entry: Any,
    resolved_map: Mapping[tuple[int, str], Any],
    book: RecipeBook,
    geometry: Geometry,
    prepare_errors: list[str],
    fallbacks: list[str],
) -> tuple[float, list[tuple[str, int, Any, tuple[str, ...]]], bool]:
    """Sum total FLOPs, ordered op+dep structure, self-deps, and fallback recipes for one replica."""
    total = 0.0
    ordered_sig: list[tuple[str, int, Any, tuple[str, ...]]] = []
    has_self_dep = False
    for op in entry.ops:
        deps = tuple(str(d) for d in getattr(op, "deps", ()))
        if str(op.op_id) in deps:
            has_self_dep = True
        ordered_sig.append((str(op.family), int(getattr(op, "layer", -1)), getattr(op, "stage", None), _op_dep_sig(op)))
        res = resolved_map.get((entry.replica, op.op_id))
        impl_name = str(res.implementation if res is not None else (op.recipe or ""))
        if any(m in impl_name.lower() for m in FORBIDDEN_IMPL_MARKERS):
            fallbacks.append(f"r{entry.replica}:{op.op_id}:{impl_name}")
        total += _op_work_flops(op, resolved_map, entry.replica, book, geometry, prepare_errors)
    return total, ordered_sig, has_self_dep


def _mc2_protocol_link_bytes(meta: Mapping[str, Any]) -> float:
    """Extract rank-0 off-diagonal peer transfer bytes from ``mc2_protocol`` metadata."""
    phase_m = meta.get("mc2_phase")
    if not isinstance(phase_m, Mapping):
        return 0.0
    rmat = phase_m.get("route_matrix_bytes")
    if isinstance(rmat, Sequence) and rmat and isinstance(rmat[0], Sequence):
        return float(sum(float(v or 0.0) for j, v in enumerate(rmat[0]) if j != 0))
    return 0.0


def _sharded_proj_fraction(meta: Mapping[str, Any]) -> float:
    """Return exact fraction of packed projection output columns that are TP-sharded (``replicated=False``)."""
    slices = meta.get("packed_projection_slices")
    if not isinstance(slices, Sequence) or not slices:
        return 1.0
    tot = 0
    sharded = 0
    for sl in slices:
        if not isinstance(sl, Mapping):
            continue
        sh = sl.get("shape")
        width = int(sh[-1]) if isinstance(sh, Sequence) and sh else 0
        tot += width
        if not bool(sl.get("replicated", False)):
            sharded += width
    return (float(sharded) / float(tot)) if tot > 0 else 1.0


def _step_conservation_metrics(step: PreparedStep, spine: PredictionSpine) -> dict[str, Any]:
    """Compute per-rank FLOPs, per-layer attn/ffn FLOPs, full-replica parity, collectives, and fallbacks."""
    case = step.case
    book = RecipeBook.from_mapping(case.runtime.kernels, family_registry=spine.families)
    geometry = Geometry.from_devices(
        case.machine.dies[:1], links=case.machine.links,
        transfer_channels=case.machine.transfer_channels,
    )
    resolved_map = {(r.occurrence.replica, r.occurrence.op_id): r for r in step.operators}

    if not step.graph.executable:
        return {
            "attn_flops": 0.0, "attn_cube_flops": 0.0, "compress_flops": 0.0, "attn_proj_flops": 0.0,
            "sharded_attn_proj_flops": 0.0, "ffn_flops": 0.0, "proj_flops": 0.0, "moe_flops": 0.0,
            "layer_flops": {}, "layer_attn_flops": {}, "layer_ffn_flops": {}, "layer_ffn_ops": {},
            "collectives": {}, "fallbacks": ["empty_executable_graph"], "prepare_errors": [],
            "op_count": 0, "replica_count": 0, "replica_issues": ["empty_executable_graph"],
            "stream_count": 0,
        }

    attn_flops = attn_cube_flops = compress_flops = attn_proj_flops = sharded_attn_proj_flops = ffn_flops = moe_flops = r0_raw_total = 0.0
    collectives: dict[str, dict[str, float]] = {}
    fallbacks: list[str] = []
    prepare_errors: list[str] = []
    replica_issues: list[str] = []

    entry = step.graph.executable[0]
    r0_sig: list[tuple[str, int, Any, tuple[str, ...]]] = []
    layer_flops: dict[int, float] = {}
    layer_attn_flops: dict[int, float] = {}
    layer_ffn_flops: dict[int, float] = {}
    layer_ffn_ops: dict[int, set[str]] = {}
    replicated_proj_tokens = (
        ".q_a", ".kv_a", ".qkv_a", ".hc_project", ".hc_down_inject", ".engram_", "index_k", "index_q",
        "index_wk", "index_wq", "index_weights", "index_key_", "index_query_",
        "linear_beta_alpha", "linear_decay_a", "linear_gate_a",
    )
    for op in entry.ops:
        fam = str(op.family)
        layer_id = int(getattr(op, "layer", -1))
        deps = tuple(str(d) for d in getattr(op, "deps", ()))
        if str(op.op_id) in deps:
            replica_issues.append(f"r0_self_dependency:{op.op_id}")
        r0_sig.append((fam, layer_id, getattr(op, "stage", None), _op_dep_sig(op)))
        res = resolved_map.get((entry.replica, op.op_id))
        impl_name = str(res.implementation if res is not None else (op.recipe or ""))
        if any(m in impl_name.lower() for m in FORBIDDEN_IMPL_MARKERS):
            fallbacks.append(f"{op.op_id}:{impl_name}")
        flops = _op_work_flops(op, resolved_map, entry.replica, book, geometry, prepare_errors)
        r0_raw_total += flops
        oid_low = op.op_id.lower()
        meta = dict(op.metadata) if op.metadata else {}
        blk = str(meta.get("structure_block", "")).lower()
        is_compressor = "compress" in oid_low or "compressor" in impl_name.lower()
        if fam in ATTN_FAMILIES and not is_compressor:
            logical_flops, cube_flops = _op_attn_logical_and_cube_flops(
                op, resolved_map, entry.replica, book, geometry, prepare_errors, flops,
            )
        else:
            logical_flops, cube_flops = flops, flops
        if layer_id >= 0:
            layer_flops[layer_id] = layer_flops.get(layer_id, 0.0) + logical_flops
        if fam in ATTN_FAMILIES and is_compressor:
            compress_flops += flops
        elif fam in ATTN_FAMILIES:
            attn_flops += logical_flops
            attn_cube_flops += cube_flops
            if layer_id >= 0:
                layer_attn_flops[layer_id] = layer_attn_flops.get(layer_id, 0.0) + logical_flops
        elif ".hc_" in oid_low:
            attn_proj_flops += flops
        elif (
            fam in MOE_FAMILIES or fam == "grouped_gemm" or isinstance(meta.get("megamoe_geometry"), Mapping)
            or "expert" in oid_low or "gmm" in oid_low or "latent_down" in oid_low or "latent_up" in oid_low
        ):
            moe_flops += flops
            ffn_flops += flops
            if layer_id >= 0:
                layer_ffn_flops[layer_id] = layer_ffn_flops.get(layer_id, 0.0) + flops
                fset = layer_ffn_ops.setdefault(layer_id, set())
                if isinstance(meta.get("megamoe_geometry"), Mapping) or (fam == "grouped_gemm" and "expert" not in oid_low):
                    fset.add("megamoe")
                elif "expert_up" in oid_low or ("gmm" in oid_low and "down" not in oid_low):
                    fset.add("expert_up")
                elif "expert_down" in oid_low or ("gmm" in oid_low and "down" in oid_low):
                    fset.add("expert_down")
        elif blk == "ffn" or any(k in oid_low for k in ("gate_up", ".down", "shared.", "router", "ffn", "mlp")):
            ffn_flops += flops
            if layer_id >= 0:
                layer_ffn_flops[layer_id] = layer_ffn_flops.get(layer_id, 0.0) + flops
                fset = layer_ffn_ops.setdefault(layer_id, set())
                if "shared." not in oid_low:
                    if fam == "gate_up_projection" or "gate_up" in oid_low or "up_proj" in oid_low or "fc1" in oid_low or "mlp_in" in oid_low:
                        fset.add("gate_up")
                    elif fam == "down_projection" or oid_low.endswith(".down") or ".down_proj" in oid_low or "fc2" in oid_low or "mlp_out" in oid_low:
                        fset.add("down")
        else:
            attn_proj_flops += flops
            if fam not in ("index_projection", "indexer", "key_pool") and not any(tok in oid_low for tok in replicated_proj_tokens):
                sharded_attn_proj_flops += flops * _sharded_proj_fraction(meta)

        if float(op.link_bytes or 0.0) > 0.0 or fam in COLLECTIVE_FAMILIES:
            kind = str(meta.get("collective") or meta.get("collective_op") or fam)
            bucket = collectives.setdefault(kind, {"count": 0.0, "link_bytes": 0.0})
            bucket["count"] += 1.0
            bucket["link_bytes"] += float(op.link_bytes or 0.0)
        elif fam == "mc2_protocol":
            mc2_lb = _mc2_protocol_link_bytes(meta)
            if mc2_lb > 0.0:
                kind = f"mc2_{str((meta.get('mc2_phase') or {}).get('kind', 'protocol'))}"
                bucket = collectives.setdefault(kind, {"count": 0.0, "link_bytes": 0.0})
                bucket["count"] += 1.0
                bucket["link_bytes"] += mc2_lb
        elif isinstance(meta.get("megamoe_geometry"), Mapping):
            mg = meta["megamoe_geometry"]
            fused_lb = float(mg.get("dispatch_link_bytes") or 0.0) + float(mg.get("combine_link_bytes") or 0.0)
            if fused_lb > 0.0:
                bucket = collectives.setdefault("mc2_megamoe", {"count": 0.0, "link_bytes": 0.0})
                bucket["count"] += 1.0
                bucket["link_bytes"] += fused_lb

    r0_total = r0_raw_total
    for rep_entry in step.graph.executable[1:]:
        rep_fb_before = len(fallbacks)
        rep_total, rep_sig, rep_self_dep = _sum_replica_flops(
            rep_entry, resolved_map, book, geometry, prepare_errors, fallbacks,
        )
        if rep_self_dep:
            replica_issues.append(f"replica_{rep_entry.replica}_self_dependency")
        if len(fallbacks) > rep_fb_before:
            replica_issues.append(f"replica_{rep_entry.replica}_fallback_recipes")
        if rep_sig != r0_sig:
            replica_issues.append(f"replica_{rep_entry.replica}_ordered_graph_mismatch")
        elif r0_total > 0.0 and abs(rep_total - r0_total) / r0_total > FLOP_CONSERVATION_REL_TOL:
            replica_issues.append(f"replica_{rep_entry.replica}_flops_mismatch:{rep_total}!={r0_total}")

    streams = {str(getattr(op, "stream", "main")) for op in entry.ops}
    return {
        "attn_flops": attn_flops, "attn_cube_flops": attn_cube_flops, "compress_flops": compress_flops,
        "attn_proj_flops": attn_proj_flops, "sharded_attn_proj_flops": sharded_attn_proj_flops,
        "ffn_flops": ffn_flops,
        "proj_flops": attn_proj_flops + (ffn_flops - moe_flops), "moe_flops": moe_flops,
        "layer_flops": layer_flops, "layer_attn_flops": layer_attn_flops, "layer_ffn_flops": layer_ffn_flops,
        "layer_ffn_ops": {k: sorted(v) for k, v in layer_ffn_ops.items()},
        "collectives": collectives, "fallbacks": fallbacks, "prepare_errors": prepare_errors,
        "op_count": len(entry.ops), "replica_count": len(step.graph.executable),
        "replica_issues": replica_issues, "stream_count": len(streams),
    }


def _independent_expected_attn_flops(
    spec: Any,
    layers: int,
    *,
    phase: str,
    tokens: int,
    context: int,
    sequences: int,
    noncausal: bool = False,
) -> float:
    """Derive independent algebraic Attention FLOPs strictly from LayerPlan dims."""
    total = 0.0
    for plan in spec.layer_plans[:layers]:
        if plan.attention not in ("gqa", "mla"):
            continue
        heads = int(plan.num_heads or spec.num_heads)
        if plan.attention == "mla":
            d_qk = int(plan.kv_lora_rank or 512) + int(plan.qk_rope_head_dim or 0)
            d_v = int(plan.kv_lora_rank or 512)
        else:
            d_qk = int(plan.head_dim or spec.head_dim or DEFAULT_KV_BLOCK_SIZE)
            d_v = int(plan.v_head_dim or d_qk)
        if phase == "prefill":
            if str(plan.cache_kind) == "window_compressed":
                s_win = float(min(tokens, plan.window) if plan.window > 0 else 0)
                s_long = (float(tokens) / float(int(plan.compress_ratio))) if int(plan.compress_ratio) > 0 else 0.0
                s_eff = s_win + s_long
            else:
                s_eff = float(min(tokens, plan.window) if plan.window > 0 else tokens)
            causal_factor = 1.0 if noncausal else 0.5
            pairs = float(sequences) * (float(tokens) * s_eff * causal_factor)
            idx_pairs = float(sequences) * (float(tokens) * (float(tokens) / float(max(1, int(plan.compress_ratio)))) * causal_factor)
        else:
            if str(plan.cache_kind) == "window_compressed":
                s_win = float(min(context, plan.window) if plan.window > 0 else 0)
                s_long = (float(context) / float(int(plan.compress_ratio))) if int(plan.compress_ratio) > 0 else 0.0
                if int(plan.index_topk) > 0:
                    s_long = min(s_long, float(plan.index_topk))
                s_eff = s_win + s_long
            else:
                s_eff = float(min(context, plan.window) if plan.window > 0 else context)
                if int(plan.index_topk) > 0:
                    s_eff = min(s_eff, float(plan.index_topk))
                if int(plan.compress_ratio) > 1:
                    s_eff = s_eff / float(plan.compress_ratio)
            pairs = float(sequences * tokens) * max(1.0, s_eff)
            idx_pairs = float(sequences * tokens) * max(1.0, float(context) / float(max(1, int(plan.compress_ratio))))
        total += 2.0 * float(heads) * pairs * float(d_qk + d_v)
        if plan.indexer and int(plan.index_heads) > 0 and int(plan.index_dim) > 0:
            total += 2.0 * float(plan.index_heads) * idx_pairs * float(plan.index_dim)
    return total


def _independent_expected_ffn_and_proj_flops(
    spec: Any,
    layers: int,
    *,
    tokens: int,
    sequences: int,
    tp: int = 1,
    has_sp: bool = False,
) -> tuple[float, float]:
    """Derive independent algebraic FFN (incl. Latent MoE, SP padding & total shared width) and attn-proj FLOPs."""
    total_ffn = 0.0
    total_proj = 0.0
    h = float(spec.hidden_size)
    m = float(max(1, sequences * tokens))
    m_sp = max(m, float(tp)) if (tp > 1 and has_sp) else m
    for plan in spec.layer_plans[:layers]:
        if plan.ffn == "moe" and int(plan.num_experts) > 0:
            lat = float(getattr(plan, "expert_latent_size", 0) or 0)
            if lat > 0.0:
                routed = 6.0 * m_sp * float(max(1, plan.top_k)) * lat * float(plan.moe_intermediate_size) + 4.0 * m_sp * h * lat
            else:
                routed = 6.0 * m_sp * float(max(1, plan.top_k)) * h * float(plan.moe_intermediate_size)
            sh_total_i = (
                float(plan.shared_intermediate_size)
                if int(plan.shared_intermediate_size or 0) > 0
                else float(plan.num_shared_experts) * float(plan.moe_intermediate_size)
            )
            shared = 6.0 * m_sp * h * sh_total_i if int(plan.num_shared_experts) > 0 else 0.0
            router = 2.0 * m_sp * h * float(plan.num_experts)
            total_ffn += routed + shared + router
        else:
            inter = float(plan.intermediate_size or spec.intermediate_size)
            total_ffn += 6.0 * m_sp * h * inter
        heads = float(plan.num_heads or spec.num_heads)
        if plan.attention == "mla":
            ql = float(plan.q_lora_rank or h)
            kvl = float(plan.kv_lora_rank or 512)
            qd = float(plan.qk_nope_head_dim + plan.qk_rope_head_dim)
            vd = float(plan.v_head_dim or plan.qk_nope_head_dim)
            total_proj += 2.0 * m * (h * ql + ql * heads * qd + h * (kvl + float(plan.qk_rope_head_dim)) + heads * vd * h)
        elif plan.attention in ("gdn", "kda"):
            nk = float(plan.linear_key_heads or plan.linear_heads or 1)
            nv = float(plan.linear_value_heads or plan.linear_heads or 1)
            dk = float(plan.linear_key_head_dim or plan.linear_head_dim or DEFAULT_KV_BLOCK_SIZE)
            dv = float(plan.linear_head_dim or DEFAULT_KV_BLOCK_SIZE)
            total_proj += 2.0 * m * h * (2.0 * nk * dk + 2.0 * nv * dv)
        else:
            kvh = float(plan.num_kv_heads or spec.num_kv_heads or 1)
            hd = float(plan.head_dim or spec.head_dim or DEFAULT_KV_BLOCK_SIZE)
            total_proj += 2.0 * m * h * (2.0 * heads * hd + 2.0 * kvh * hd)
    return total_ffn, total_proj


def _independent_index_row_bytes(model_spec: Any, index_dim: int) -> float:
    """Derive index-K physical row bytes strictly from ModelSpec primitive fields (no cache_index call)."""
    raw_cr = dict(getattr(model_spec, "metadata", ())).get("cache_representations", {})
    if isinstance(raw_cr, Mapping) and isinstance(raw_cr.get("index_k"), Mapping):
        spec_ik = raw_cr["index_k"]
        e_b = DTYPE_BYTES_TABLE.get(str(spec_ik.get("dtype", "bf16")).lower(), 2.0)
        suffix = int(spec_ik.get("suffix_elements", 0))
        quant_elems = max(0, int(index_dim) - suffix)
        grp = int(spec_ik.get("group_size", 0))
        s_b = DTYPE_BYTES_TABLE.get(str(spec_ik.get("scale_dtype", "none")).lower(), 0.0)
        scales = (math.ceil(quant_elems / grp) * s_b) if (grp > 0 and s_b > 0.0) else 0.0
        return float(quant_elems) * e_b + float(suffix) * 2.0 + scales
    quant = getattr(model_spec, "quant", None)
    idx_dt = str(getattr(quant, "index_dtype", "") if quant is not None else "").lower() or "bf16"
    sc_dt = str(getattr(quant, "index_scale_dtype", "") if quant is not None else "").lower() or "none"
    return float(index_dim) * DTYPE_BYTES_TABLE.get(idx_dt, 2.0) + DTYPE_BYTES_TABLE.get(sc_dt, 0.0)


def _independent_primary_kv_row_bytes(model_spec: Any, plan: Any) -> float:
    """Derive primary KV row bytes directly from primitive ModelSpec/LayerPlan fields (zero CacheFormat call)."""
    ck = str(getattr(plan, "cache_kind", "")).lower()
    if ck == "none":
        return 0.0
    raw_cr = dict(getattr(model_spec, "metadata", ())).get("cache_representations", {})
    quant = getattr(model_spec, "quant", None)
    kv_dt = str(getattr(quant, "kv_dtype", "") if quant is not None else "").lower() or "bf16"
    kv_elem_b = DTYPE_BYTES_TABLE.get(kv_dt, 2.0)
    if plan.attention == "mla":
        kvl = int(plan.kv_lora_rank or 0)
        qkr = 0 if ck == "window_compressed" else int(plan.qk_rope_head_dim or 0)
        total_elems = kvl + qkr
        if isinstance(raw_cr, Mapping):
            cr_key = "compressed_kv" if ck == "window_compressed" and "compressed_kv" in raw_cr else "mla_kv"
            spec_kv = raw_cr.get(cr_key)
            if isinstance(spec_kv, Mapping):
                e_b = DTYPE_BYTES_TABLE.get(str(spec_kv.get("dtype", kv_dt)).lower(), kv_elem_b)
                suffix = int(spec_kv.get("suffix_elements", 0))
                q_elems = max(0, total_elems - suffix)
                grp = int(spec_kv.get("group_size", 0))
                s_b = DTYPE_BYTES_TABLE.get(str(spec_kv.get("scale_dtype", "none")).lower(), 0.0)
                scales = (math.ceil(q_elems / grp) * s_b) if (grp > 0 and s_b > 0.0) else 0.0
                return float(q_elems) * e_b + float(suffix) * 2.0 + scales
        return float(kvl) * kv_elem_b + float(qkr) * 2.0
    if plan.attention == "gqa":
        kvh = int(plan.num_kv_heads or getattr(model_spec, "num_kv_heads", 0) or 0)
        hd = int(plan.head_dim or getattr(model_spec, "head_dim", 0) or 0)
        v_hd = int(plan.v_head_dim or hd)
        return float(kvh * (hd + v_hd)) * kv_elem_b
    return 0.0


def _kv_and_indexer_memory_audit(model_spec: Any, parallel: ParallelSpec) -> dict[str, Any]:
    """Cross-check independent primary-KV & Indexer byte ledger against ``layer_cache_components``."""
    kv_bytes_per_rank = 0.0
    indexer_bytes_per_rank = 0.0
    runtime_kv_bytes_per_rank = 0.0
    runtime_idx_bytes_per_rank = 0.0
    replicated_layers: list[int] = []
    audit_seq = DECODE_AUDIT_CONTEXT
    cp_factor = float(max(1, int(parallel.cp)))
    idx_sharded_dcp = (
        (bool(paired_sfa_execution(model_spec).get("dcp_indexer_sharded")) and int(parallel.cp) > 1)
        or str(getattr(parallel, "cp_plan", "")) == "dsa_cp"
    )

    layer_mismatches = 0
    for plan in model_spec.layer_plans:
        full_row = _independent_primary_kv_row_bytes(model_spec, plan)
        exp_layer_kv = 0.0
        exp_layer_idx = 0.0
        if plan.attention == "mla":
            if parallel.tp > 1 and parallel.dcp == 1:
                replicated_layers.append(plan.index)
            if str(plan.cache_kind) == "window_compressed":
                win_frac = (min(audit_seq, int(plan.window)) / float(audit_seq)) if int(plan.window) > 0 else 0.0
                long_frac = (1.0 / float(max(1, int(plan.compress_ratio)))) if plan.owns_long_kv else 0.0
                exp_layer_kv = full_row * (win_frac + long_frac) / cp_factor
            else:
                win_frac = (min(audit_seq, int(plan.window)) / float(audit_seq)) if int(plan.window) > 0 else 1.0
                exp_layer_kv = full_row * win_frac / cp_factor
        elif plan.attention == "gqa":
            if parallel.tp > plan.num_kv_heads and parallel.dcp == 1:
                replicated_layers.append(plan.index)
            if not (0 <= int(plan.kv_owner_layer_id) < int(plan.index)):
                head_shard = min(parallel.tp, max(1, int(plan.num_kv_heads)))
                win_frac = (min(audit_seq, int(plan.window)) / float(audit_seq)) if int(plan.window) > 0 else 1.0
                exp_layer_kv = full_row * win_frac / float(head_shard * cp_factor)
        kv_bytes_per_rank += exp_layer_kv

        if plan.owns_index_k:
            idx_ratio = max(1, int(plan.compress_ratio), int(plan.index_kpool))
            row_b = _independent_index_row_bytes(model_spec, int(plan.index_dim))
            idx_div = cp_factor if idx_sharded_dcp else 1.0
            exp_layer_idx = row_b / float(idx_ratio * idx_div)
            indexer_bytes_per_rank += exp_layer_idx

        comps = layer_cache_components(
            model_spec, plan, parallel, audit_seq, capacity_padded=False,
        )
        obs_layer_kv = sum(
            float(v) for k, v in comps.items() if k in ("gqa_kv", "mla_kv", "window", "compressed_kv")
        ) / float(audit_seq)
        obs_layer_idx = float(comps.get("index_k", 0.0)) / float(audit_seq)
        runtime_kv_bytes_per_rank += obs_layer_kv
        runtime_idx_bytes_per_rank += obs_layer_idx
        if abs(exp_layer_kv - obs_layer_kv) > KV_LAYER_ABS_TOL or abs(exp_layer_idx - obs_layer_idx) > KV_LAYER_ABS_TOL:
            layer_mismatches += 1

    kv_ok = layer_mismatches == 0 and abs(kv_bytes_per_rank - runtime_kv_bytes_per_rank) <= KV_LAYER_ABS_TOL
    idx_ok = layer_mismatches == 0 and abs(indexer_bytes_per_rank - runtime_idx_bytes_per_rank) <= KV_LAYER_ABS_TOL
    return {
        "kv_bytes_per_token_per_rank": round(kv_bytes_per_rank, 2),
        "runtime_kv_bytes_per_token_per_rank": round(runtime_kv_bytes_per_rank, 2),
        "indexer_bytes_per_token_per_rank": round(indexer_bytes_per_rank, 2),
        "runtime_indexer_bytes_per_token_per_rank": round(runtime_idx_bytes_per_rank, 2),
        "kv_cache_replicated_tp": bool(replicated_layers),
        "replicated_layer_count": len(replicated_layers),
        "layer_kv_mismatches": layer_mismatches,
        "ledger_matches_runtime": bool(kv_ok and idx_ok),
    }


def _select_dcp_geometry(model_spec: Any) -> tuple[tuple[int, int] | None, str]:
    """Choose a valid algebraic ``(tp, dcp)`` pair up to TP16, or return explicit exclusion reason."""
    is_mla = any(p.attention == "mla" for p in model_spec.layer_plans)
    if is_mla:
        for width in (WORLD_QUAD, 2):
            if model_spec.num_heads % width == 0:
                return (width, width), ""
        return None, f"mla_heads_indivisible:{model_spec.num_heads}"
    kv_vals = [p.num_kv_heads for p in model_spec.layer_plans if p.attention == "gqa" and p.num_kv_heads > 0]
    if not kv_vals:
        return None, "no_gqa_or_mla_layers"
    base_kv = int(model_spec.num_kv_heads)
    if base_kv >= WORLD_HEX:
        return None, f"gqa_kv_heads_exceeds_tp16_domain:kv_heads={base_kv}"
    tp = min(WORLD_HEX, max(2, base_kv * 2))
    if model_spec.num_heads % tp != 0 or tp % base_kv != 0:
        return None, f"gqa_heads_indivisible_for_dcp:heads={model_spec.num_heads},kv={base_kv},tp={tp}"
    dcp = tp // base_kv
    return ((tp, dcp), "") if dcp > 1 else (None, f"dcp_equals_1:tp={tp},kv={base_kv}")


def _check_ratio_non_tautological(
    label: str,
    scaled_val: float,
    ref_val: float,
    expected_positive: float,
    findings: list[str],
) -> float:
    """Verify conservation against single-rank reference without substituting 1.0 on zero work."""
    if expected_positive > 0.0:
        if ref_val <= 0.0 or scaled_val <= 0.0:
            findings.append(f"{label}_zero_work_with_positive_expected:scaled={scaled_val},ref={ref_val}")
            return 0.0
        ratio = scaled_val / ref_val
        if abs(ratio - 1.0) > FLOP_CONSERVATION_REL_TOL:
            findings.append(f"{label}_flops_drift:{ratio:.4f}")
        return ratio
    return 1.0 if (ref_val == 0.0 and scaled_val == 0.0) else 0.0


def _sum_pcp_physical_rank_attn_flops(parsed_case: Any, pcp: int, tokens: int) -> float:
    """Sum logical prefill attention FLOPs across all ``pcp`` physical ranks (handles SWA/DSA/CSA2 asymmetry)."""
    total = 0.0
    for rk in range(pcp):
        graphs = target_graphs(
            parsed_case, scope="body", phase="prefill", tokens=tokens, context=0, sequences=1, physical_rank=rk,
        )
        for op in graphs[0]:
            if str(op.family) in ATTN_FAMILIES and "compress" not in str(op.op_id).lower():
                total += float(op.flops or 0.0)
    return total


def _select_canonical_tp_width(spec: Any, *, is_mla: bool) -> int:
    """Choose canonical non-KV-replicated TP width for TP+EP conservation checks."""
    if is_mla:
        # Prefer Cube-tile-aligned (>= 16) local MLA heads first, then smaller divisible TP widths
        # (e.g. deepseek-v2-lite with 16 heads -> TP=2) so TP/EP checks never degenerate to TP=1.
        for cand in (WORLD_QUAD, 2):
            if spec.num_heads % cand == 0 and (spec.num_heads // cand) >= FRACTAL_NZ_TILE:
                return cand
        for cand in (2, WORLD_QUAD):
            if spec.num_heads % cand == 0:
                return cand
        return 1
    gqa_kv = [int(p.num_kv_heads) for p in spec.layer_plans if p.attention == "gqa" and int(p.num_kv_heads) > 0]
    min_kv = min(gqa_kv) if gqa_kv else int(spec.num_kv_heads or 1)
    for cand in (WORLD_QUAD, 2):
        if spec.num_heads % cand == 0 and (min_kv == 1 or min_kv % cand == 0):
            return cand
    return 1


def _check_attn_cube_padding(
    spec: Any, *, is_mla: bool, tp_w: int, m_tp: Mapping[str, Any], ref_dec: Mapping[str, Any], findings: list[str],
) -> float:
    """Verify FRACTAL_NZ padded Cube work separately from logical attention FLOP conservation."""
    cube_tp, cube_ref = float(m_tp.get("attn_cube_flops", 0.0)), float(ref_dec.get("attn_cube_flops", 0.0))
    if cube_ref <= 0.0:
        return 1.0
    cube_ratio = (cube_tp * float(tp_w)) / cube_ref
    if cube_tp + KV_LAYER_ABS_TOL < float(m_tp.get("attn_flops", 0.0)):
        findings.append(f"tp_ep_cube_below_logical:{cube_tp:.3e}<{float(m_tp.get('attn_flops', 0.0)):.3e}")
    if is_mla and tp_w > 1 and spec.num_heads % tp_w == 0:
        local_h = spec.num_heads // tp_w
        padded_h = ((local_h + FRACTAL_NZ_TILE - 1) // FRACTAL_NZ_TILE) * FRACTAL_NZ_TILE
        exp_ratio = float(padded_h) / float(local_h)
        if abs(cube_ratio - exp_ratio) > FLOP_CONSERVATION_REL_TOL:
            findings.append(f"tp_ep_fractal_nz_cube_drift:{cube_ratio:.4f}!=exp:{exp_ratio:.4f}")
    return cube_ratio


def _audit_model_cp_regimes(
    model_key: str,
    spec: Any,
    spine: PredictionSpine,
    sku: str,
    layers: int,
    ref_dec: Mapping[str, Any],
    ref_pre: Mapping[str, Any],
    exp_dec_attn: float,
    exp_pre_attn: float,
    regimes: dict[str, Any],
    findings: list[str],
) -> None:
    """Audit DCP, PCP (summed across physical ranks), and Dual-Phase PCP+DCP FLOP conservation."""
    is_moe = spec.num_experts > 0
    is_mla = any(p.attention == "mla" for p in spec.layer_plans)
    replicated_v41 = bool(is_v41_replicated_cache_model(spec))

    # 3) DCP (long-context decode KV sharding; V4.1 replicated-cache enforces dcp==1 guard)
    dcp_pair, dcp_excl = _select_dcp_geometry(spec)
    if dcp_pair is None:
        regimes["dcp_excluded"] = {"reason": dcp_excl}
    elif replicated_v41:
        tp_d, dcp_d = dcp_pair
        try:
            parse_case(compose_case(
                model_key, hardware=sku, dp=1, tp=tp_d, dcp=dcp_d,
                ep=tp_d if is_moe else 1, num_devices=tp_d, layers=layers,
            ))
            findings.append("v41_replicated_cache_failed_to_reject_dcp_gt_1")
        except Exception as exc:  # noqa: BLE001
            if "v41_replicated_cache_requires_dcp_1" in str(exc):
                regimes["dcp_excluded"] = {"reason": "v41_replicated_cache_requires_dcp_1_enforced"}
            else:
                findings.append(f"dcp_unexpected_error:{type(exc).__name__}:{exc}")
    else:
        tp_d, dcp_d = dcp_pair
        try:
            c_dcp = parse_case(compose_case(
                model_key, hardware=sku, dp=1, tp=tp_d, dcp=dcp_d,
                ep=tp_d if is_moe else 1, num_devices=tp_d, layers=layers,
            ))
            s_dcp = spine.lower(
                c_dcp,
                LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")),
            )
            m_dcp = _step_conservation_metrics(s_dcp, spine)
            kv_dcp = _kv_and_indexer_memory_audit(spec, c_dcp.runtime.parallel)
            attn_ratio = _check_ratio_non_tautological("dcp_attn", m_dcp["attn_flops"] * tp_d, ref_dec["attn_flops"], exp_dec_attn, findings)
            if not kv_dcp["ledger_matches_runtime"]:
                findings.append(f"dcp_kv_ledger_mismatch:{kv_dcp}")
            if m_dcp["fallbacks"] or m_dcp["prepare_errors"] or m_dcp["replica_issues"]:
                findings.append(f"dcp_issues:fb={m_dcp['fallbacks']},pe={m_dcp['prepare_errors']},rep={m_dcp['replica_issues']}")
            regimes["dcp"] = {
                "tp": tp_d, "dcp": dcp_d, "attn_flop_ratio": round(attn_ratio, 4),
                "kv_audit": kv_dcp, "collectives": m_dcp["collectives"],
            }
        except Exception as exc:  # noqa: BLE001
            findings.append(f"dcp_error:tp={tp_d},dcp={dcp_d}:{type(exc).__name__}:{exc}")

    # 4) PCP (long-context prefill — sum across all physical ranks for exact conservation on SWA/DSA/CSA2)
    try:
        c_pcp = parse_case(compose_case(
            model_key, hardware=sku, dp=1, tp=1, pcp=2,
            ep=2 if is_moe else 1, num_devices=2, layers=layers,
        ))
        s_pcp = spine.lower(
            c_pcp,
            LoweringRequest(step=StepRequest(phase="prefill", tokens=PREFILL_AUDIT_TOKENS, context=0, sequences=1, scope="body")),
        )
        m_pcp = _step_conservation_metrics(s_pcp, spine)
        pcp_total_attn = _sum_pcp_physical_rank_attn_flops(c_pcp, 2, PREFILL_AUDIT_TOKENS)
        if pcp_total_attn <= 0.0:
            pcp_total_attn = m_pcp["attn_flops"] * 2.0
        attn_ratio = _check_ratio_non_tautological("pcp_attn", pcp_total_attn, ref_pre["attn_flops"], exp_pre_attn, findings)
        if m_pcp["fallbacks"] or m_pcp["prepare_errors"] or m_pcp["replica_issues"]:
            findings.append(f"pcp_issues:fb={m_pcp['fallbacks']},pe={m_pcp['prepare_errors']},rep={m_pcp['replica_issues']}")
        regimes["pcp"] = {
            "pcp": 2, "attn_flop_ratio": round(attn_ratio, 4), "collectives": m_pcp["collectives"],
        }
    except Exception as exc:  # noqa: BLE001
        findings.append(f"pcp_error:{type(exc).__name__}:{exc}")

    # 5) Dual-Phase PCP + DCP (where DCP is valid and not V4.1 replicated cache)
    if dcp_pair is not None and is_mla and not replicated_v41:
        try:
            c_pdcp = parse_case(compose_case(
                model_key, hardware=sku, dp=1, tp=2, pcp=2, dcp=WORLD_QUAD,
                ep=WORLD_QUAD if is_moe else 1, num_devices=WORLD_QUAD, layers=layers,
            ))
            s_pdcp = spine.lower(
                c_pdcp,
                LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")),
            )
            m_pdcp = _step_conservation_metrics(s_pdcp, spine)
            kv_pdcp = _kv_and_indexer_memory_audit(spec, c_pdcp.runtime.parallel)
            pdcp_ratio = _check_ratio_non_tautological("pcp_dcp_attn", m_pdcp["attn_flops"] * float(WORLD_QUAD), ref_dec["attn_flops"], exp_dec_attn, findings)
            if not kv_pdcp["ledger_matches_runtime"]:
                findings.append(f"pcp_dcp_kv_ledger_mismatch:{kv_pdcp}")
            if m_pdcp["fallbacks"] or m_pdcp["prepare_errors"] or m_pdcp["replica_issues"]:
                findings.append(f"pcp_dcp_issues:fb={m_pdcp['fallbacks']},pe={m_pdcp['prepare_errors']}")
            regimes["pcp_dcp"] = {
                "pcp": 2, "tp": 2, "dcp": WORLD_QUAD, "attn_flop_ratio": round(pdcp_ratio, 4),
                "kv_audit": kv_pdcp, "collectives": m_pdcp["collectives"],
            }
        except Exception as exc:  # noqa: BLE001
            findings.append(f"pcp_dcp_error:{type(exc).__name__}:{exc}")


def _check_algebraic_magnitude(
    label: str,
    measured: float,
    expected: float,
    findings: list[str],
) -> None:
    """Reject measured FLOPs far outside independent tensor-algebra bounds (non-self-referential)."""
    if expected <= 0.0:
        return
    if measured < ALGEBRAIC_FLOP_LOW_FACTOR * expected or measured > ALGEBRAIC_FLOP_HIGH_FACTOR * expected:
        findings.append(f"{label}_outside_algebraic_bounds:measured={measured:.3e},expected={expected:.3e}")


def _check_per_layer_conservation(
    spec: Any,
    layers: int,
    metrics: Mapping[str, Any],
    *,
    phase: str,
    tokens: int,
    context: int,
    sequences: int,
    noncausal: bool = False,
    scale: float = 1.0,
    tp: int = 1,
    label: str,
    findings: list[str],
) -> None:
    """Verify every audited layer preserves tight algebraic attn, FFN (with gate_up+down), and total FLOPs."""
    l_tot = metrics.get("layer_flops", {})
    l_attn = metrics.get("layer_attn_flops", {})
    l_ffn = metrics.get("layer_ffn_flops", {})
    l_fops = metrics.get("layer_ffn_ops", {})
    colls = metrics.get("collectives", {})
    has_sp = any(k in colls for k in ("reduce_scatter", "reducescatter", "allgather"))
    for li in range(layers):
        plan_i = spec.layer_plans[li]
        single = replace(spec, layers=1, body_layers=1, layer_plans=(plan_i,))
        exp_a = _independent_expected_attn_flops(
            single, 1, phase=phase, tokens=tokens, context=context, sequences=sequences, noncausal=noncausal,
        )
        exp_f, exp_p = _independent_expected_ffn_and_proj_flops(
            single, 1, tokens=tokens, sequences=sequences, tp=tp, has_sp=has_sp,
        )
        exp_layer = exp_a + exp_f + exp_p
        obs_layer = float(l_tot.get(li, 0.0)) * scale
        obs_ffn = float(l_ffn.get(li, 0.0)) * scale
        if exp_layer > 0.0:
            if obs_layer < ALGEBRAIC_FLOP_LOW_FACTOR * exp_layer or obs_layer > ALGEBRAIC_FLOP_HIGH_FACTOR * exp_layer:
                findings.append(f"{label}_layer{li}_flops_out_of_bounds:obs={obs_layer:.3e},exp={exp_layer:.3e}")
        if exp_a > 0.0 and float(l_attn.get(li, 0.0)) <= 0.0:
            findings.append(f"{label}_layer{li}_missing_attn_flops")
        if exp_f > 0.0:
            is_moe_l = plan_i.ffn == "moe" and int(plan_i.num_experts) > 0
            lo_f = MOE_FFN_FLOP_LOW if is_moe_l else DENSE_FFN_FLOP_LOW
            hi_f = MOE_FFN_FLOP_HIGH if is_moe_l else DENSE_FFN_FLOP_HIGH
            if obs_ffn < lo_f * exp_f or obs_ffn > hi_f * exp_f:
                findings.append(f"{label}_layer{li}_ffn_flops_out_of_bounds:obs={obs_ffn:.3e},exp={exp_f:.3e}")
            fops = set(l_fops.get(li, ()))
            if is_moe_l:
                if "megamoe" not in fops and not ({"expert_up", "expert_down"} <= fops):
                    findings.append(f"{label}_layer{li}_missing_moe_expert_pair:ops={sorted(fops)}")
            elif not ({"gate_up", "down"} <= fops):
                findings.append(f"{label}_layer{li}_missing_dense_ffn_pair:ops={sorted(fops)}")


def _total_collective_link_bytes(collectives: Mapping[str, Mapping[str, float]]) -> float:
    return float(sum(float(v.get("link_bytes", 0.0)) for v in collectives.values()))


def _total_collective_count(collectives: Mapping[str, Mapping[str, float]]) -> int:
    return int(round(sum(float(v.get("count", 0.0)) for v in collectives.values())))


def _check_collective_conservation(
    spec: Any,
    layers: int,
    collectives: Mapping[str, Mapping[str, float]],
    *,
    tp: int,
    ep: int,
    dcp: int = 1,
    pcp: int = 1,
    tokens: int,
    sequences: int,
    label: str,
    findings: list[str],
) -> None:
    """Enforce analytical minimum collective counts and ring/dispatch link bytes across parallel dims."""
    if tp <= 1 and ep <= 1 and dcp <= 1 and pcp <= 1:
        return
    obs_bytes = _total_collective_link_bytes(collectives)
    obs_count = _total_collective_count(collectives)
    if obs_bytes < MIN_COLLECTIVE_LINK_BYTES or obs_count <= 0:
        findings.append(f"{label}_missing_collectives:count={obs_count},bytes={obs_bytes:.1f}")
        return
    m_tok = float(max(1, (sequences * tokens) // max(1, pcp))) if pcp > 1 else float(max(1, sequences * tokens))
    h = float(spec.hidden_size)
    min_expected_count = 0
    min_expected_bytes = 0.0
    if tp > 1:
        # Each TP layer executes at least 2 reductions (post-attn + post-FFN) across hidden activations
        min_expected_count += layers * 2
        # Conservative 1 B/elem lower bound (covers FP8/INT8 comms while catching 1024 B stubs!)
        ring_per_reduce = m_tok * h * 2.0 * float(tp - 1) / float(tp)
        min_expected_bytes += float(layers) * 2.0 * ring_per_reduce
    moe_layers = sum(1 for p in spec.layer_plans[:layers] if p.ffn == "moe" and int(p.num_experts) > 0)
    if ep > 1 and moe_layers > 0:
        min_expected_count += moe_layers
        min_expected_bytes += float(moe_layers) * m_tok * h * float(ep - 1) / float(ep)
    if pcp > 1 or dcp > 1:
        min_expected_count = max(min_expected_count, layers)
    if min_expected_count > 0 and obs_count < min_expected_count:
        findings.append(f"{label}_collective_count_insufficient:obs={obs_count}<min={min_expected_count}")
    if min_expected_bytes > 0.0 and obs_bytes < COLLECTIVE_ANALYTICAL_MIN_REL * min_expected_bytes:
        findings.append(
            f"{label}_collective_bytes_under_conservation:obs={obs_bytes:.1f}<min={COLLECTIVE_ANALYTICAL_MIN_REL * min_expected_bytes:.1f}"
        )


def _audit_model_gate_a(model_key: str, spine: PredictionSpine, sku: str) -> dict[str, Any]:
    """Run Gate A parallelism, FLOP conservation, KV/Indexer, and collective checks for 1 model."""
    spec = load_model(model_key)
    is_dit = model_key == "hunyuan-video-dit"
    is_moe = spec.num_experts > 0
    is_mla = any(p.attention == "mla" for p in spec.layer_plans)
    layers = min(DEFAULT_AUDIT_LAYERS, spec.layers)
    findings: list[str] = []
    regimes: dict[str, Any] = {}

    if is_dit:
        raw = compose_case(model_key, hardware=sku, tp=WORLD_QUAD, ep=1, num_devices=WORLD_QUAD, layers=layers)
        step = spine.lower(
            parse_case(raw),
            LoweringRequest(step=StepRequest(phase="prefill", tokens=PREFILL_LIVE_TOKENS, context=0, sequences=1, scope="body")),
        )
        metrics = _step_conservation_metrics(step, spine)
        ev = spine.run_step(step, ReplayOptions(include_submission=False))
        exp_dit_attn = _independent_expected_attn_flops(
            spec, layers, phase="prefill", tokens=PREFILL_LIVE_TOKENS, context=0, sequences=1, noncausal=True,
        )
        exp_dit_ffn, exp_dit_proj = _independent_expected_ffn_and_proj_flops(
            spec, layers, tokens=PREFILL_LIVE_TOKENS, sequences=1,
        )
        _check_algebraic_magnitude("dit_attn", metrics["attn_flops"] * WORLD_QUAD, exp_dit_attn, findings)
        _check_algebraic_magnitude("dit_ffn", metrics["ffn_flops"] * WORLD_QUAD, exp_dit_ffn, findings)
        _check_algebraic_magnitude("dit_proj", metrics["attn_proj_flops"] * WORLD_QUAD, exp_dit_proj, findings)
        _check_per_layer_conservation(
            spec, layers, metrics, phase="prefill", tokens=PREFILL_LIVE_TOKENS, context=0,
            sequences=1, noncausal=True, scale=float(WORLD_QUAD), tp=WORLD_QUAD, label="dit", findings=findings,
        )
        _check_collective_conservation(
            spec, layers, metrics["collectives"], tp=WORLD_QUAD, ep=1,
            tokens=PREFILL_LIVE_TOKENS, sequences=1, label="dit", findings=findings,
        )
        if metrics["fallbacks"] or metrics["prepare_errors"] or metrics["replica_issues"] or ev.span_ms <= 0.0:
            findings.append(f"dit_invalid:fb={metrics['fallbacks']},pe={metrics['prepare_errors']},rep={metrics['replica_issues']},span={ev.span_ms}")
        if hasattr(parse_case, "cache_clear"):
            parse_case.cache_clear()
        return {"model": model_key, "is_mla": False, "is_moe": False, "regimes": {"dit_tp4": {**metrics, "span_ms": ev.span_ms}}, "findings": findings}

    exp_dec_attn = _independent_expected_attn_flops(
        spec, layers, phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS,
    )
    exp_pre_attn = _independent_expected_attn_flops(
        spec, layers, phase="prefill", tokens=PREFILL_AUDIT_TOKENS, context=0, sequences=1,
    )
    exp_dec_ffn, exp_dec_proj = _independent_expected_ffn_and_proj_flops(
        spec, layers, tokens=1, sequences=DECODE_AUDIT_SEQS,
    )
    ref_dec_step = spine.lower(
        parse_case(compose_case(model_key, hardware=sku, tp=1, dp=1, ep=1, num_devices=1, layers=layers)),
        LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")),
    )
    ref_dec = _step_conservation_metrics(ref_dec_step, spine)
    _check_algebraic_magnitude("ref_dec_attn", ref_dec["attn_flops"], exp_dec_attn, findings)
    _check_algebraic_magnitude("ref_dec_ffn", ref_dec["ffn_flops"], exp_dec_ffn, findings)
    _check_algebraic_magnitude("ref_dec_proj", ref_dec["attn_proj_flops"], exp_dec_proj, findings)
    if ref_dec["prepare_errors"]:
        findings.append(f"ref_dec_prepare_errors:{ref_dec['prepare_errors']}")

    ref_pre_step = spine.lower(
        parse_case(compose_case(model_key, hardware=sku, tp=1, dp=1, ep=1, num_devices=1, layers=layers)),
        LoweringRequest(step=StepRequest(phase="prefill", tokens=PREFILL_AUDIT_TOKENS, context=0, sequences=1, scope="body")),
    )
    ref_pre = _step_conservation_metrics(ref_pre_step, spine)
    _check_algebraic_magnitude("ref_pre_attn", ref_pre["attn_flops"], exp_pre_attn, findings)
    if ref_pre["prepare_errors"]:
        findings.append(f"ref_pre_prepare_errors:{ref_pre['prepare_errors']}")

    dp_ep = WORLD_OCT if is_moe else 1
    try:
        c_dp = parse_case(compose_case(model_key, hardware=sku, dp=WORLD_OCT, tp=1, ep=dp_ep, num_devices=WORLD_OCT, layers=layers))
        s_dp = spine.lower(c_dp, LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")))
        m_dp = _step_conservation_metrics(s_dp, spine)
        ev_dp = spine.run_step(s_dp, ReplayOptions(include_submission=False))
        kv_dp = _kv_and_indexer_memory_audit(spec, c_dp.runtime.parallel)
        attn_ratio = _check_ratio_non_tautological("pure_dp_attn", m_dp["attn_flops"], ref_dec["attn_flops"], exp_dec_attn, findings)
        ffn_ratio = _check_ratio_non_tautological("pure_dp_ffn", m_dp["ffn_flops"], ref_dec["ffn_flops"], exp_dec_ffn, findings)
        proj_ratio = _check_ratio_non_tautological("pure_dp_proj", m_dp["attn_proj_flops"], ref_dec["attn_proj_flops"], exp_dec_proj, findings)
        if not kv_dp["ledger_matches_runtime"]:
            findings.append(f"pure_dp_kv_ledger_mismatch:{kv_dp}")
        if kv_dp["kv_cache_replicated_tp"]:
            findings.append("pure_dp_unexpected_kv_replication")
        if "allreduce" in m_dp["collectives"]:
            findings.append("pure_dp_unexpected_allreduce")
        if m_dp["fallbacks"] or m_dp["prepare_errors"] or m_dp["replica_issues"] or ev_dp.span_ms <= 0.0:
            findings.append(f"pure_dp_issues:fb={m_dp['fallbacks']},pe={m_dp['prepare_errors']},rep={m_dp['replica_issues']}")
        regimes["pure_dp_ep"] = {
            "attn_flop_ratio": round(attn_ratio, 4), "ffn_flop_ratio": round(ffn_ratio, 4), "proj_flop_ratio": round(proj_ratio, 4),
            "span_ms": round(ev_dp.span_ms, 4), "kv_audit": kv_dp, "collectives": m_dp["collectives"],
        }
    except Exception as exc:  # noqa: BLE001
        findings.append(_sanitize_diagnostic(f"pure_dp_ep_error:{type(exc).__name__}:{exc}"))

    tp_w = _select_canonical_tp_width(spec, is_mla=is_mla)
    tp_ep = tp_w if is_moe else 1
    try:
        c_tp = parse_case(compose_case(model_key, hardware=sku, dp=1, tp=tp_w, ep=tp_ep, num_devices=tp_w, layers=layers))
        s_tp = spine.lower(c_tp, LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")))
        m_tp = _step_conservation_metrics(s_tp, spine)
        ev_tp = spine.run_step(s_tp, ReplayOptions(include_submission=False))
        kv_tp = _kv_and_indexer_memory_audit(spec, c_tp.runtime.parallel)
        attn_ratio = _check_ratio_non_tautological("tp_ep_attn", m_tp["attn_flops"] * tp_w, ref_dec["attn_flops"], exp_dec_attn, findings)
        attn_cube_ratio = _check_attn_cube_padding(spec, is_mla=is_mla, tp_w=tp_w, m_tp=m_tp, ref_dec=ref_dec, findings=findings)
        ffn_ratio_tp = _check_ratio_non_tautological("tp_ep_ffn", m_tp["ffn_flops"] * tp_w, ref_dec["ffn_flops"], exp_dec_ffn, findings)
        proj_ratio_tp = _check_ratio_non_tautological(
            "tp_ep_proj", m_tp["sharded_attn_proj_flops"] * tp_w,
            ref_dec["sharded_attn_proj_flops"], exp_dec_proj if ref_dec["sharded_attn_proj_flops"] > 0.0 else 0.0,
            findings,
        )
        if ref_dec["attn_proj_flops"] > 0.0:
            tot_proj_ratio = (m_tp["attn_proj_flops"] * tp_w) / ref_dec["attn_proj_flops"]
            if tot_proj_ratio < 1.0 - FLOP_CONSERVATION_REL_TOL or tot_proj_ratio > float(tp_w) * (1.0 + FLOP_CONSERVATION_REL_TOL):
                findings.append(f"tp_ep_total_proj_out_of_bounds:{tot_proj_ratio:.4f}")
        _check_per_layer_conservation(
            spec, layers, m_tp, phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT,
            sequences=DECODE_AUDIT_SEQS, scale=float(tp_w), tp=tp_w, label="tp_ep", findings=findings,
        )
        _check_collective_conservation(
            spec, layers, m_tp["collectives"], tp=tp_w, ep=tp_ep,
            tokens=1, sequences=DECODE_AUDIT_SEQS, label="tp_ep", findings=findings,
        )
        if not kv_tp["ledger_matches_runtime"]:
            findings.append(f"tp_ep_kv_ledger_mismatch:{kv_tp}")
        if is_mla and tp_w > 1 and not kv_tp["kv_cache_replicated_tp"]:
            findings.append("mla_tp_missing_kv_cache_replicated_tp_flag")
        if m_tp["fallbacks"] or m_tp["prepare_errors"] or m_tp["replica_issues"] or ev_tp.span_ms <= 0.0:
            findings.append(f"tp_ep_issues:fb={m_tp['fallbacks']},pe={m_tp['prepare_errors']},rep={m_tp['replica_issues']}")
        regimes["tp_ep"] = {
            "tp": tp_w, "ep": tp_ep, "attn_flop_ratio": round(attn_ratio, 4),
            "attn_cube_padded_ratio": round(attn_cube_ratio, 4),
            "ffn_flop_ratio": round(ffn_ratio_tp, 4), "proj_flop_ratio": round(proj_ratio_tp, 4),
            "span_ms": round(ev_tp.span_ms, 4), "kv_audit": kv_tp, "collectives": m_tp["collectives"],
        }
    except Exception as exc:  # noqa: BLE001
        findings.append(_sanitize_diagnostic(f"tp_ep_error:{type(exc).__name__}:{exc}"))

    _audit_model_cp_regimes(
        model_key, spec, spine, sku, layers, ref_dec, ref_pre, exp_dec_attn, exp_pre_attn, regimes, findings,
    )
    if hasattr(parse_case, "cache_clear"):
        parse_case.cache_clear()
    return {
        "model": model_key, "is_mla": is_mla, "is_moe": is_moe,
        "layers_audited": layers, "regimes": regimes, "findings": findings,
    }


def _inspect_lowered_quant_evidence(step: PreparedStep) -> tuple[set[str], set[str], list[str], list[str]]:
    """Inspect lowered dense vs MoE-expert GEMM weight formats and attention KV dtypes."""
    wfmts: set[str] = set()
    kv_dtypes: set[str] = set()
    dense_wfmts: list[str] = []
    expert_wfmts: list[str] = []
    if not step.graph.executable:
        return wfmts, kv_dtypes, dense_wfmts, expert_wfmts
    skip_tokens = ("router", "shared_gate", "hc_project")
    for op in step.graph.executable[0].ops:
        meta = dict(op.metadata) if op.metadata else {}
        oid_low = op.op_id.lower()
        fam_low = str(op.family).lower()
        wf = str(
            meta.get("weight_format")
            or meta.get("weight_dtype")
            or (meta.get("compute_format") if fam_low == "grouped_gemm" else "")
            or ""
        ).lower()
        cat = _classify_sim_op(op)
        is_expert = fam_low == "grouped_gemm" or ("expert_" in oid_low and cat in ("matmul", "moe"))
        is_weighted = bool(wf) and not any(tok in oid_low for tok in skip_tokens)
        if is_weighted and (cat == "matmul" or is_expert):
            wfmts.add(wf)
            if is_expert:
                expert_wfmts.append(wf)
            else:
                dense_wfmts.append(wf)
        atv = meta.get("attention_tensor_view")
        if isinstance(atv, Mapping) and atv.get("cache_dtype"):
            kv_dtypes.add(str(atv["cache_dtype"]).lower())
        gqa_d = meta.get("gqa_decode")
        if isinstance(gqa_d, Mapping) and gqa_d.get("kv_dtype"):
            kv_dtypes.add(str(gqa_d["kv_dtype"]).lower())
    return wfmts, kv_dtypes, dense_wfmts, expert_wfmts


def _audit_quant_overlays(spine: PredictionSpine) -> tuple[dict[str, Any], list[str]]:
    """Verify actual lowered dense & expert op weight/KV formats across BF16/W8A8/W4A8/C8/MXFP8/MXFP4."""
    checks: dict[str, Any] = {}
    findings: list[str] = []
    quant_matrix: tuple[tuple[str, str, str, Any, str, str, str, dict[str, Any]], ...] = (
        ("bf16_a3", "deepseek-v3", DEFAULT_GATE_A_SKU, "bf16", "bf16", "bf16", "bf16", dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("w8a8_a3", "deepseek-v3", DEFAULT_GATE_A_SKU, "w8a8", "int8", "int8", "bf16", dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("w4a8_a3", "deepseek-v4.1-flash", DEFAULT_GATE_A_SKU, "w4a8", "int8", "int4", "bf16", dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("c8_a3", "glm-5.3", DEFAULT_GATE_A_SKU, C8_QUANT_SPEC, "int8", "int8", "int8", dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("mxfp8_a5", "qwen3.5-397b-a17b", DEFAULT_A5_SKU, "mxfp8", "fp8", "fp8", "bf16", dict(dp=1, tp=WORLD_OCT, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("mxfp4_a5", "glm-5.2", DEFAULT_A5_SKU, "mxfp4", "mxfp4", "mxfp4", "fp8", dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("mxfp4_bf16kv_a5", "deepseek-v3", DEFAULT_A5_SKU, "mxfp4", "mxfp4", "mxfp4", "bf16", dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
    )
    for tag, model_key, sku, quant_name, exp_dense, exp_expert, exp_kv, kw in quant_matrix:
        try:
            is_moe_model = load_model(model_key).num_experts > 0
            raw = compose_case(model_key, hardware=sku, quant=quant_name, layers=DEFAULT_AUDIT_LAYERS, **kw)
            step = spine.lower(
                parse_case(raw),
                LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")),
            )
            m = _step_conservation_metrics(step, spine)
            ev = spine.run_step(step, ReplayOptions(include_submission=False))
            obs_wfmts, obs_kvs, dense_wfmts, expert_wfmts = _inspect_lowered_quant_evidence(step)
            dense_ratio = sum(1 for wf in dense_wfmts if exp_dense in wf) / float(max(1, len(dense_wfmts)))
            expert_ratio = sum(1 for wf in expert_wfmts if exp_expert in wf) / float(max(1, len(expert_wfmts)))
            checks[f"quant_{tag}"] = {
                "span_ms": round(ev.span_ms, 4), "fallbacks": len(m["fallbacks"]),
                "lowered_weight_formats": sorted(obs_wfmts), "lowered_kv_dtypes": sorted(obs_kvs),
                "dense_op_count": len(dense_wfmts), "expert_op_count": len(expert_wfmts),
                "dense_match_ratio": round(dense_ratio, 3), "expert_match_ratio": round(expert_ratio, 3),
            }
            if m["fallbacks"] or m["prepare_errors"] or ev.span_ms <= 0.0:
                findings.append(f"quant_fail:{tag}")
            if len(dense_wfmts) < MIN_DENSE_QUANT_OPS or dense_ratio < MIN_QUANT_OP_COVERAGE_RATIO:
                findings.append(f"quant_dense_weight_mismatch:{tag}:expected={exp_dense},n={len(dense_wfmts)},ratio={dense_ratio:.2f},got={sorted(obs_wfmts)}")
            if is_moe_model and (len(expert_wfmts) < MIN_EXPERT_QUANT_OPS or expert_ratio < 1.0 - 1e-9):
                findings.append(f"quant_expert_weight_mismatch:{tag}:expected={exp_expert},n={len(expert_wfmts)},ratio={expert_ratio:.2f},got={sorted(obs_wfmts)}")
            if exp_dense == "bf16" and any(wf not in ("bf16", "fp16", "fp32") for wf in obs_wfmts):
                findings.append(f"quant_bf16_unexpected_quant_op:{tag}:got={sorted(obs_wfmts)}")
            if exp_kv not in obs_kvs:
                findings.append(f"quant_lowered_kv_dtype_mismatch:{tag}:expected={exp_kv},got={sorted(obs_kvs)}")
        except Exception as exc:  # noqa: BLE001
            findings.append(_sanitize_diagnostic(f"quant_err:{tag}:{exc}"))
    return checks, findings


def _audit_feature_overlays(spine: PredictionSpine) -> dict[str, Any]:
    """Verify Quant (dense+expert), MTP (strict q1<q2<q4 + nonzero draft FLOPs), MC2/MSO (same-layer overlap), and aclgraph."""
    checks, findings = _audit_quant_overlays(spine)

    # 2. MTP q_len=1 < q_len=2 < q_len=4 strict monotonicity, proposal FLOPs, and nonzero draft-op FLOPs
    try:
        spans: dict[str, float] = {}
        flops_by_q: dict[str, float] = {}
        draft_ops_by_q: dict[str, int] = {}
        draft_flops_by_q: dict[str, float] = {}
        for q_len in (1, 2, MTP_QUERY_LEN_K3):
            raw = compose_case(
                "deepseek-v3", hardware=DEFAULT_GATE_A_SKU, dp=WORLD_OCT, tp=1, ep=WORLD_OCT,
                num_devices=WORLD_OCT, layers=DEFAULT_AUDIT_LAYERS, mtp_proposal_tokens=q_len - 1,
            )
            step = spine.lower(
                parse_case(raw),
                LoweringRequest(step=StepRequest(phase="decode", tokens=q_len, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="model")),
            )
            m_q = _step_conservation_metrics(step, spine)
            ev = spine.run_step(step, ReplayOptions(include_submission=False))
            ops_q = list(step.graph.executable[0].ops) if step.graph.executable else []
            resolved_q = {(r.occurrence.replica, r.occurrence.op_id): r for r in step.operators}
            book_q = RecipeBook.from_mapping(step.case.runtime.kernels, family_registry=spine.families)
            geom_q = Geometry.from_devices(
                step.case.machine.dies[:1], links=step.case.machine.links,
                transfer_channels=step.case.machine.transfer_channels,
            )
            draft_ops = [op for op in ops_q if ("draft" in op.op_id.lower() or "mtp" in op.op_id.lower())]
            draft_errs: list[str] = []
            draft_flops = sum(_op_work_flops(op, resolved_q, 0, book_q, geom_q, draft_errs) for op in draft_ops)
            spans[f"q{q_len}"] = round(ev.span_ms, 4)
            flops_by_q[f"q{q_len}"] = m_q["attn_proj_flops"] + m_q["ffn_flops"]
            draft_ops_by_q[f"q{q_len}"] = len(draft_ops)
            draft_flops_by_q[f"q{q_len}"] = draft_flops
        checks["mtp_q_scaling"] = {
            "spans_ms": spans, "flops": flops_by_q,
            "draft_ops": draft_ops_by_q, "draft_flops": draft_flops_by_q,
        }
        if not (spans["q1"] < spans["q2"] < spans["q4"]):
            findings.append(f"mtp_spans_not_strictly_monotonic:{spans}")
        if not (flops_by_q["q1"] < flops_by_q["q2"] < flops_by_q["q4"]) or (
            flops_by_q["q1"] <= 0.0 or flops_by_q["q4"] < MTP_Q4_MIN_FLOP_RATIO * flops_by_q["q1"]
        ):
            findings.append(f"mtp_flops_scaling_invalid:{flops_by_q}")
        if draft_ops_by_q["q2"] <= 0 or draft_ops_by_q["q4"] <= 0 or not (0.0 < draft_flops_by_q["q2"] < draft_flops_by_q["q4"]):
            findings.append(f"mtp_draft_ops_or_flops_missing:ops={draft_ops_by_q},flops={draft_flops_by_q}")
    except Exception as exc:  # noqa: BLE001
        findings.append(_sanitize_diagnostic(f"mtp_err:{exc}"))

    # 3. MC2 / MSO shared-expert overlap (dispatch>=1 KiB + combine>=1 KiB + SAME-LAYER shared-expert overlap)
    try:
        raw_mso = compose_case(
            "deepseek-v3", hardware=DEFAULT_GATE_A_SKU, dp=WORLD_OCT, tp=1, ep=WORLD_OCT,
            num_devices=WORLD_OCT, layers=DEFAULT_AUDIT_LAYERS, multistream_overlap_shared_expert=True,
        )
        step_mso = spine.lower(
            parse_case(raw_mso),
            LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")),
        )
        m_mso = _step_conservation_metrics(step_mso, spine)
        ev_mso = spine.run_step(step_mso, ReplayOptions(include_submission=False))
        ops_mso = list(step_mso.graph.executable[0].ops) if step_mso.graph.executable else []
        op_by_id = {op.op_id: op for op in ops_mso}
        disp_cnt = comb_cnt = 0
        disp_bytes = comb_bytes = 0.0
        for op in ops_mso:
            ck = _classify_collective_kind_sim(op)
            lb = float(op.link_bytes or 0.0)
            if ck in ("mc2_dispatch", "all_to_all_dispatch"):
                disp_cnt += 1
                disp_bytes += lb
            elif ck in ("mc2_combine", "all_to_all_combine"):
                comb_cnt += 1
                comb_bytes += lb
        comm_ivs: list[tuple[int, str, float, float]] = []
        shared_ivs: list[tuple[int, str, float, float]] = []
        for row in ev_mso.operators:
            if row.occurrence.replica != 0 or row.start_ms is None or row.end_ms is None or float(row.end_ms) <= float(row.start_ms):
                continue
            op = op_by_id.get(row.occurrence.op_id)
            if op is None:
                continue
            ly = int(getattr(op, "layer", -1))
            iv = (ly, str(row.stream), float(row.start_ms), float(row.end_ms))
            ck = _classify_collective_kind_sim(op)
            if ck in ("mc2_dispatch", "mc2_combine", "all_to_all_dispatch", "all_to_all_combine"):
                comm_ivs.append(iv)
            elif "shared" in op.op_id.lower() and _classify_sim_op(op) in ("matmul", "moe"):
                shared_ivs.append(iv)
        has_temporal_overlap = any(
            cl == kl and cs != ks and max(c0, k0) < min(c1, k1)
            for cl, cs, c0, c1 in comm_ivs for kl, ks, k0, k1 in shared_ivs
        )
        checks["mc2_mso_overlap"] = {
            "span_ms": round(ev_mso.span_ms, 4), "collectives": m_mso["collectives"],
            "stream_count": m_mso["stream_count"], "dispatch_count": disp_cnt, "combine_count": comb_cnt,
            "dispatch_bytes": disp_bytes, "combine_bytes": comb_bytes, "has_temporal_overlap": has_temporal_overlap,
        }
        if (
            disp_cnt < 1 or comb_cnt < 1 or disp_bytes < MIN_COLLECTIVE_LINK_BYTES or comb_bytes < MIN_COLLECTIVE_LINK_BYTES
            or m_mso["stream_count"] < 2 or not has_temporal_overlap
            or m_mso["fallbacks"] or m_mso["prepare_errors"]
        ):
            findings.append(
                f"mc2_mso_invalid:disp={disp_cnt}/{disp_bytes},comb={comb_cnt}/{comb_bytes},streams={m_mso['stream_count']},overlap={has_temporal_overlap}"
            )
    except Exception as exc:  # noqa: BLE001
        findings.append(_sanitize_diagnostic(f"mc2_mso_err:{exc}"))

    acl_check, acl_findings = _audit_aclgraph_padded_overlay(spine)
    if acl_check:
        checks["aclgraph_padded_batch"] = acl_check
    findings.extend(acl_findings)
    return {"checks": checks, "findings": findings}


def _audit_aclgraph_padded_overlay(spine: PredictionSpine) -> tuple[dict[str, Any], list[str]]:
    """Verify aclgraph padded ``RequestView(padding=True)`` keeps 0 logical KV & full projection rows."""
    findings: list[str] = []
    try:
        active_views = tuple(
            RequestView(request_id=f"act-{i}", q_len=1, context=DECODE_AUDIT_CONTEXT, samples=True, phase="decode", padding=False)
            for i in range(ACLGRAPH_ACTIVE_SEQS)
        )
        pad_views = tuple(
            RequestView(request_id=f"pad-{i}", q_len=1, context=DECODE_AUDIT_CONTEXT, samples=False, phase="decode", padding=True)
            for i in range(ACLGRAPH_PAD_SEQS)
        )
        full_active_views = tuple(
            RequestView(request_id=f"full-{i}", q_len=1, context=DECODE_AUDIT_CONTEXT, samples=True, phase="decode", padding=False)
            for i in range(ACLGRAPH_ACTIVE_SEQS + ACLGRAPH_PAD_SEQS)
        )
        plan_padded = IterationPlan(requests=active_views + pad_views, iteration=0)
        plan_active = IterationPlan(requests=active_views, iteration=0)
        plan_full_64 = IterationPlan(requests=full_active_views, iteration=0)
        if plan_padded.valid_tokens != ACLGRAPH_ACTIVE_SEQS:
            findings.append(f"aclgraph_valid_tokens_wrong:{plan_padded.valid_tokens}")
        if any(v.kv_len != 0 for v in pad_views):
            findings.append("aclgraph_padding_view_nonzero_kv_len")
        raw_eager_48 = compose_case(
            "qwen3.5-397b-a17b", hardware=DEFAULT_A5_SKU, tp=WORLD_OCT, ep=WORLD_OCT,
            num_devices=WORLD_OCT, layers=DEFAULT_AUDIT_LAYERS, graph_capture_mode="none",
        )
        raw_pad = compose_case(
            "qwen3.5-397b-a17b", hardware=DEFAULT_A5_SKU, tp=WORLD_OCT, ep=WORLD_OCT,
            num_devices=WORLD_OCT, layers=DEFAULT_AUDIT_LAYERS, graph_capture_mode="full",
        )
        case_pad = parse_case(raw_pad)
        step_unpadded_48 = spine.lower(parse_case(raw_eager_48), LoweringRequest(iterations=((0, plan_active),)))
        step_pad = spine.lower(case_pad, LoweringRequest(iterations=((0, plan_padded),)))
        step_full_64 = spine.lower(case_pad, LoweringRequest(iterations=((0, plan_full_64),)))
        m_act = _step_conservation_metrics(step_unpadded_48, spine)
        m_pad = _step_conservation_metrics(step_pad, spine)
        m_full = _step_conservation_metrics(step_full_64, spine)
        ev_act = spine.run_step(step_unpadded_48, ReplayOptions(include_submission=False))
        ev_pad = spine.run_step(step_pad, ReplayOptions(include_submission=False))
        ev_full = spine.run_step(step_full_64, ReplayOptions(include_submission=False))
        check = {
            "active_seqs": ACLGRAPH_ACTIVE_SEQS, "padded_seqs": ACLGRAPH_PAD_SEQS,
            "valid_tokens": plan_padded.valid_tokens,
            "attn_flops_active_48": m_act["attn_flops"], "attn_flops_padded_48_16": m_pad["attn_flops"],
            "attn_flops_full_64": m_full["attn_flops"], "proj_flops_unpadded_48": m_act["proj_flops"],
            "proj_flops_padded_48_16": m_pad["proj_flops"], "proj_flops_full_64": m_full["proj_flops"],
            "span_active_48_ms": round(ev_act.span_ms, 4), "span_padded_48_16_ms": round(ev_pad.span_ms, 4),
            "span_full_64_ms": round(ev_full.span_ms, 4),
        }
        if m_act["attn_flops"] > 0 and abs(m_pad["attn_flops"] - m_act["attn_flops"]) / m_act["attn_flops"] > FLOP_CONSERVATION_REL_TOL:
            findings.append(f"aclgraph_pad_inflated_attn_flops:{m_pad['attn_flops']}>{m_act['attn_flops']}")
        if m_full["attn_flops"] <= m_pad["attn_flops"]:
            findings.append("aclgraph_pad_attn_not_less_than_full_64_active")
        if m_pad["proj_flops"] <= m_act["proj_flops"]:
            findings.append("aclgraph_pad_did_not_execute_physical_projection_rows")
        return check, findings
    except Exception as exc:  # noqa: BLE001
        findings.append(_sanitize_diagnostic(f"aclgraph_padded_err:{type(exc).__name__}:{exc}"))
        return {}, findings


def _audit_single_tutorial(cfg: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Lower and execute one documented vllm-ascend deployment config on default_spine()."""
    spine = default_spine()
    dp = int(cfg.get("dp", 1))
    tp = int(cfg.get("tp", 1))
    pp = int(cfg.get("pp", 1))
    pcp = int(cfg.get("pcp", 1))
    dcp = int(cfg.get("dcp", 1))
    ep = int(cfg.get("ep", 1))
    num_devices = dp * tp * pcp * pp
    layers = DEFAULT_AUDIT_LAYERS * pp
    kw: dict[str, Any] = {
        "hardware": str(cfg["hw"]),
        "dp": dp, "tp": tp, "pp": pp, "pcp": pcp, "dcp": dcp, "ep": ep,
        "num_devices": num_devices,
        "layers": layers,
    }
    if cfg.get("quant"):
        kw["quant"] = dict(cfg["quant"]) if isinstance(cfg["quant"], Mapping) else str(cfg["quant"])
    if cfg.get("mtp"):
        kw["mtp_proposal_tokens"] = int(cfg["mtp"])
    if cfg.get("mso"):
        kw["multistream_overlap_shared_expert"] = True
    phase = str(cfg["phase"])
    toks = PREFILL_LIVE_TOKENS if phase == "prefill" else 1
    ctx = 0 if phase == "prefill" else DECODE_AUDIT_CONTEXT
    seqs = 1 if phase == "prefill" else DECODE_AUDIT_SEQS
    step_req = StepRequest(phase=phase, tokens=toks, context=ctx, sequences=seqs, scope="body")
    try:
        spec_m = load_model(str(cfg["model"]))
        eff_layers = min(layers, spec_m.layers)
        kw["layers"] = eff_layers
        raw = compose_case(str(cfg["model"]), **kw)
        parsed = parse_case(raw)
        step = spine.lower(parsed, LoweringRequest(step=step_req))
        m = _step_conservation_metrics(step, spine)
        ev = spine.run_step(step, ReplayOptions(include_submission=False))
        kv_audit = _kv_and_indexer_memory_audit(spec_m, parsed.runtime.parallel)
        total_flops = m["attn_flops"] + m["compress_flops"] + m["proj_flops"] + m["moe_flops"]
        needs_comm = (tp > 1) or (pcp > 1) or (dcp > 1) or (ep > 1)
        structural_issues: list[str] = []
        if m["fallbacks"]:
            structural_issues.append(f"fallbacks={m['fallbacks']}")
        if m["prepare_errors"]:
            structural_issues.append(f"prepare_errors={m['prepare_errors']}")
        if m["replica_issues"]:
            structural_issues.append(f"replica_issues={m['replica_issues']}")
        if ev.span_ms <= 0.0 or m["op_count"] <= 0 or total_flops <= 0.0:
            structural_issues.append(f"empty_work:span={ev.span_ms},ops={m['op_count']},flops={total_flops}")
        if needs_comm:
            _check_collective_conservation(
                spec_m, eff_layers, m["collectives"], tp=tp, ep=ep if spec_m.num_experts > 0 else 1,
                dcp=dcp, pcp=pcp, tokens=toks, sequences=seqs, label="tutorial", findings=structural_issues,
            )
        _check_per_layer_conservation(
            spec_m, eff_layers, m, phase=phase, tokens=toks, context=ctx,
            sequences=seqs, scale=float(tp * pcp), tp=tp, label="tutorial", findings=structural_issues,
        )
        if not kv_audit["ledger_matches_runtime"]:
            structural_issues.append(f"kv_ledger_mismatch:kv={kv_audit['kv_bytes_per_token_per_rank']}!={kv_audit['runtime_kv_bytes_per_token_per_rank']}")
        ok = not structural_issues
        row = {
            "id": cfg["id"], "model": cfg["model"], "hw": cfg["hw"], "role": cfg["role"],
            "status": "pass" if ok else "fail", "span_ms": round(ev.span_ms, 4),
            "op_count": m["op_count"], "total_flops": total_flops,
            "collectives": m["collectives"], "kv_audit": kv_audit,
            "fallbacks": m["fallbacks"], "issues": structural_issues,
        }
        return row, (None if ok else _sanitize_diagnostic(f"tutorial_fail:{cfg['id']}:{structural_issues}"))
    except Exception as exc:  # noqa: BLE001
        err_s = _sanitize_diagnostic(f"{type(exc).__name__}:{exc}")
        row = {
            "id": cfg["id"], "model": cfg["model"], "hw": cfg["hw"], "role": cfg["role"],
            "status": "error", "error": err_s,
        }
        return row, _sanitize_diagnostic(f"tutorial_err:{cfg['id']}:{err_s}")


def _match_catalog_model_from_text(text: str) -> str | None:
    """Resolve a tutorial filename or vllm-serve model path to a canonical catalog model."""
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not compact:
        return None
    for key in sorted(ALL_CATALOG_MODELS, key=lambda k: len(re.sub(r"[^a-z0-9]+", "", k)), reverse=True):
        key_compact = re.sub(r"[^a-z0-9]+", "", key.lower())
        if key_compact and key_compact in compact:
            return key
    return None


def _parse_int_flag(cmd: str, *flags: str, default: int = 1) -> int:
    for flag in flags:
        m = re.search(rf"(?:^|\s){re.escape(flag)}(?:=|\s+)['\"]?(\d+)['\"]?(?=\s|$)", cmd)
        if m:
            return max(1, int(m.group(1)))
    return default


def _has_unresolved_parallel_var(cmd: str) -> bool:
    """Return True if any parallel flag passes a shell variable like ``$TP`` instead of an integer."""
    pflags = (
        "--data-parallel-size", "-dp", "--tensor-parallel-size", "-tp",
        "--pipeline-parallel-size", "-pp", "--prefill-context-parallel-size", "-pcp",
        "--decode-context-parallel-size", "-dcp",
    )
    for flag in pflags:
        if re.search(rf"(?:^|\s){re.escape(flag)}(?:=|\s+)['\"]?\$", cmd):
            return True
    return False


def _parse_mtp_spec_tokens(cmd: str) -> int:
    """Extract native MTP ``num_speculative_tokens`` (ignoring external ``eagle3``/``ngram`` draft configs)."""
    m_meth = re.search(r"method[\"']?\s*[:=]\s*[\"']?([a-z0-9_-]+)", cmd.lower())
    if m_meth and m_meth.group(1) != "mtp" and not m_meth.group(1).endswith("_mtp"):
        return 0
    m = re.search(r"num_speculative_tokens[\"']?\s*[:=]\s*(\d+)", cmd)
    return int(m.group(1)) if m else 0


def audit_va_tutorial_discovery(tutorials_dir: Path) -> dict[str, Any]:
    """Parse ``vllm serve`` deployments with zero-drop accounting across tutorial docs."""
    if not tutorials_dir.is_dir():
        raise FileNotFoundError(_sanitize_diagnostic(f"va_tutorials_dir_not_found:{tutorials_dir}"))
    discovered: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    total_commands = 0
    duplicate_commands = 0
    for md_path in sorted(tutorials_dir.glob("*.md")):
        raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
        fallback_model = _match_catalog_model_from_text(md_path.stem)
        current_hw = DEFAULT_GATE_A_SKU
        lines = raw_text.splitlines()
        idx = 0
        while idx < len(lines):
            stripped = lines[idx].lstrip()
            low = stripped.lower()
            if "tab-item" in low or stripped.startswith("#") or stripped.startswith("==="):
                if "950pr" in low:
                    current_hw = DEFAULT_A5_PR_SKU
                elif "950dt" in low or "a5" in low:
                    current_hw = DEFAULT_A5_SKU
                elif "a2" in low or "910b" in low:
                    current_hw = DEFAULT_A2_SKU
                elif "a3" in low or "910c" in low or "9362" in low:
                    current_hw = DEFAULT_GATE_A_SKU
            if re.match(r"^(?:exec\s+)?vllm\s+serve\s+", stripped):
                total_commands += 1
                cmd_parts = [stripped]
                while cmd_parts[-1].endswith("\\") and idx + 1 < len(lines):
                    idx += 1
                    cmd_parts[-1] = cmd_parts[-1][:-1].strip()
                    cmd_parts.append(lines[idx].strip())
                cmd = " ".join(cmd_parts)
                m_arg = re.search(r"vllm\s+serve\s+([^\s\\]+)", cmd)
                model_key = (
                    _match_catalog_model_from_text(m_arg.group(1)) if m_arg else None
                ) or fallback_model
                if model_key is None or model_key == "hunyuan-video-dit":
                    skipped.append({"doc": md_path.name, "reason": f"non_catalog_or_dit:{model_key}"})
                    idx += 1
                    continue
                if _has_unresolved_parallel_var(cmd):
                    skipped.append({"doc": md_path.name, "model": model_key, "reason": "unresolved_shell_var_parallel_flag"})
                    idx += 1
                    continue
                dp = _parse_int_flag(cmd, "--data-parallel-size", "-dp", default=1)
                tp = _parse_int_flag(cmd, "--tensor-parallel-size", "-tp", default=1)
                pp = _parse_int_flag(cmd, "--pipeline-parallel-size", "-pp", default=1)
                pcp = _parse_int_flag(cmd, "--prefill-context-parallel-size", "-pcp", default=1)
                dcp = _parse_int_flag(cmd, "--decode-context-parallel-size", "-dcp", default=1)
                ep_on = "--enable-expert-parallel" in cmd
                spec_m = load_model(model_key)
                ep = (dp * tp * pcp) if (ep_on and spec_m.num_experts > 0) else 1
                if tp > 1 and spec_m.num_heads % tp != 0:
                    skipped.append({"doc": md_path.name, "model": model_key, "reason": f"tp_indivisible_heads:tp={tp},heads={spec_m.num_heads}"})
                    idx += 1
                    continue
                cmd_low = cmd.lower()
                if "kv_producer" in cmd_low or pcp > 1:
                    role, phase = "pd_prefill", "prefill"
                elif "kv_consumer" in cmd_low or dcp > 1:
                    role, phase = "pd_decode", "decode"
                else:
                    role, phase = "colocated", "decode"
                quant: Any = None
                has_fp8_kv = bool(re.search(r"--kv-cache-dtype(?:=|\s+)['\"]?fp8", cmd_low))
                has_int8_kv = bool(re.search(r"--kv-cache-dtype(?:=|\s+)['\"]?int8", cmd_low))
                cmd_hw = current_hw
                a5_skus = (DEFAULT_A5_SKU, DEFAULT_A5_PR_SKU)
                if "mxfp4" in cmd_low or "w4a4" in cmd_low:
                    cmd_hw = cmd_hw if cmd_hw in a5_skus else DEFAULT_A5_SKU
                    quant = MXFP4_FP8KV_QUANT_SPEC if (has_fp8_kv or "c8" in cmd_low) else "mxfp4"
                elif "mxfp8" in cmd_low:
                    cmd_hw = cmd_hw if cmd_hw in a5_skus else DEFAULT_A5_SKU
                    quant = "mxfp8"
                elif "w4a8" in cmd_low:
                    quant = "w4a8"
                elif has_fp8_kv:
                    cmd_hw = cmd_hw if cmd_hw in a5_skus else DEFAULT_A5_SKU
                    quant = W8A8_FP8KV_QUANT_SPEC
                elif has_int8_kv or "w8a8c8" in cmd_low:
                    quant = C8_QUANT_SPEC
                elif "w8a8" in cmd_low or ("ascend" in cmd_low and "--quantization" in cmd_low):
                    quant = "w8a8"
                mtp = _parse_mtp_spec_tokens(cmd)
                mso = bool(re.search(r"multistream_overlap_shared_expert[\"']?\s*[:=]\s*true\b", cmd_low))
                q_sig = quant["name"] if isinstance(quant, dict) else quant
                sig = (model_key, cmd_hw, role, phase, dp, tp, pp, pcp, dcp, ep, q_sig, mtp, mso)
                if sig in seen_keys:
                    duplicate_commands += 1
                else:
                    seen_keys.add(sig)
                    rec: dict[str, Any] = {
                        "id": f"doc_{md_path.stem.lower()}_{len(discovered)}",
                        "model": model_key, "hw": cmd_hw, "role": role, "phase": phase,
                        "dp": dp, "tp": tp, "pp": pp, "pcp": pcp, "dcp": dcp, "ep": ep,
                        "quant": quant, "mso": mso,
                    }
                    if mtp > 0:
                        rec["mtp"] = mtp
                    discovered.append(rec)
            idx += 1
    return {
        "total_commands": total_commands,
        "discovered": discovered,
        "explicitly_skipped": skipped,
        "duplicate_commands": duplicate_commands,
        "zero_drop_conserved": total_commands == len(discovered) + len(skipped) + duplicate_commands,
    }


def discover_va_tutorial_deployments(tutorials_dir: Path) -> list[dict[str, Any]]:
    """Return deduplicated real ``vllm serve`` deployments from tutorial markdown docs."""
    return list(audit_va_tutorial_discovery(tutorials_dir)["discovered"])


def _spawn_supported() -> bool:
    """Return True when multiprocessing 'spawn' can safely re-import __main__."""
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    return main_file is None or Path(str(main_file)).is_file()


def _spawn_map_chunked(
    fn: Any,
    items: Sequence[Any],
    *,
    workers: int = 1,
    chunk_size: int = 4,
) -> list[Any]:
    """Map ``fn`` across chunked spawned worker pools without Python 3.11 max_tasks_per_child deadlock."""
    if not items:
        return []
    if not _spawn_supported():
        return [fn(x) for x in items]
    out: list[Any] = []
    ctx = mp.get_context("spawn")
    w = max(1, int(workers))
    stride = max(1, w * max(1, int(chunk_size)))
    for start in range(0, len(items), stride):
        chunk = list(items[start:start + stride])
        with ProcessPoolExecutor(max_workers=min(w, len(chunk)), mp_context=ctx) as pool:
            out.extend(pool.map(fn, chunk))
    return out


def _audit_va_tutorial_deployments(
    spine: PredictionSpine,
    *,
    workers: int = 1,
    tutorials_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify official vllm-ascend tutorial Colocated + PD P/D deployment configs in parallel."""
    del spine
    configs = list(VA_TUTORIAL_DEPLOYMENTS)
    discovery_audit: dict[str, Any] = {
        "total_commands": 0, "discovered_count": 0, "skipped_count": 0, "zero_drop_conserved": True,
    }
    if tutorials_dir is not None:
        disc = audit_va_tutorial_discovery(tutorials_dir)
        configs.extend(disc["discovered"])
        discovery_audit = {
            "total_commands": disc["total_commands"],
            "discovered_count": len(disc["discovered"]),
            "skipped_count": len(disc["explicitly_skipped"]),
            "duplicate_commands": disc["duplicate_commands"],
            "zero_drop_conserved": disc["zero_drop_conserved"],
        }
    if len(configs) > 4 or (workers > 1 and len(configs) > 1):
        pairs = _spawn_map_chunked(_audit_single_tutorial, configs, workers=workers, chunk_size=4)
    else:
        pairs = [_audit_single_tutorial(cfg) for cfg in configs]
    rows = [p[0] for p in pairs]
    findings = [p[1] for p in pairs if p[1] is not None]
    passed = sum(1 for r in rows if r["status"] == "pass")
    failed = len(rows) - passed
    return {
        "total": len(rows),
        "builtin_count": len(VA_TUTORIAL_DEPLOYMENTS),
        "discovered_count": discovery_audit["discovered_count"],
        "discovery_audit": discovery_audit,
        "passed": passed,
        "failed": failed,
        "zero_drop_conserved": (len(rows) == passed + failed) and discovery_audit["zero_drop_conserved"],
        "rows": rows,
        "findings": findings,
    }


def _audit_hardware_perturbation_continuity(spine: PredictionSpine) -> dict[str, Any]:
    """Verify smooth causal response to AIC core-count and HBM-bandwidth perturbations (zero SKU hardcoding)."""
    findings: list[str] = []
    # 1. Core-count sweep (28 -> 32 -> 36 AIC) on compute-bound prefill
    base_aic_case = parse_case(compose_case(
        "qwen3-32b", hardware=DEFAULT_A5_SKU, tp=WORLD_QUAD, ep=1, num_devices=WORLD_QUAD, layers=2,
    ))
    aic_spans: dict[int, float] = {}
    for aic in PERTURB_AIC_COUNTS:
        derived_hw = derive_hardware_sku(base_aic_case.machine, sku=f"synth-aic-{aic}", aic_count=aic)
        perturbed_case = replace(base_aic_case, machine=derived_hw)
        step = spine.lower(
            perturbed_case,
            LoweringRequest(step=StepRequest(phase="prefill", tokens=PREFILL_AUDIT_TOKENS, context=0, sequences=1, scope="body")),
        )
        ev = spine.run_step(step, ReplayOptions(include_submission=False))
        aic_spans[aic] = round(ev.span_ms, 6)
    vals_aic = [aic_spans[c] for c in PERTURB_AIC_COUNTS]
    if any(vals_aic[i + 1] > vals_aic[i] * (1.0 + PERTURB_REL_MONOTONIC_TOL) for i in range(len(vals_aic) - 1)):
        findings.append(f"aic_perturbation_non_monotonic:{aic_spans}")
    if vals_aic[0] <= vals_aic[-1]:
        findings.append(f"aic_perturbation_insensitive:{aic_spans}")

    # 2. HBM bandwidth sweep (4.0 -> 2.4 -> 1.6 -> 1.4 TB/s) comparing decode vs prefill
    base_bw_case = parse_case(compose_case(
        "deepseek-v3", hardware=DEFAULT_A5_SKU, dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT, layers=DEFAULT_AUDIT_LAYERS,
    ))
    bw_decode: dict[float, float] = {}
    bw_prefill: dict[float, float] = {}
    for bw_tbps in PERTURB_HBM_BW_TBPS:
        derived_hw = derive_hardware_sku(
            base_bw_case.machine, sku=f"synth-bw-{bw_tbps}", hbm_peak_bytes_per_s=bw_tbps * TBPS_TO_BPS,
        )
        perturbed_case = replace(base_bw_case, machine=derived_hw)
        ev_dec = spine.run_step(
            spine.lower(perturbed_case, LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body"))),
            ReplayOptions(include_submission=False),
        )
        ev_pre = spine.run_step(
            spine.lower(perturbed_case, LoweringRequest(step=StepRequest(phase="prefill", tokens=PREFILL_AUDIT_TOKENS, context=0, sequences=1, scope="body"))),
            ReplayOptions(include_submission=False),
        )
        bw_decode[bw_tbps] = round(ev_dec.span_ms, 6)
        bw_prefill[bw_tbps] = round(ev_pre.span_ms, 6)

    dec_vals = [bw_decode[b] for b in PERTURB_HBM_BW_TBPS]
    pre_vals = [bw_prefill[b] for b in PERTURB_HBM_BW_TBPS]
    if any(dec_vals[i + 1] < dec_vals[i] * (1.0 - PERTURB_REL_MONOTONIC_TOL) for i in range(len(dec_vals) - 1)):
        findings.append(f"hbm_bw_decode_non_monotonic:{bw_decode}")
    if dec_vals[-1] <= dec_vals[0]:
        findings.append(f"hbm_bw_decode_insensitive:{bw_decode}")
    decode_slowdown = dec_vals[-1] / dec_vals[0] if dec_vals[0] > 0 else 0.0
    prefill_slowdown = pre_vals[-1] / pre_vals[0] if pre_vals[0] > 0 else 0.0
    if decode_slowdown <= prefill_slowdown:
        findings.append(f"hbm_regime_transition_missing:dec={decode_slowdown:.3f}<=pre={prefill_slowdown:.3f}")

    return {
        "aic_core_sweep_ms": aic_spans,
        "hbm_bw_decode_ms": bw_decode,
        "hbm_bw_prefill_ms": bw_prefill,
        "decode_bw_slowdown": round(decode_slowdown, 4),
        "prefill_bw_slowdown": round(prefill_slowdown, 4),
        "findings": findings,
    }


def _audit_pd_decode_crossover(spine: PredictionSpine) -> dict[str, Any]:
    """Verify GLM-5.3-Flash PD Decode crossover: DCP wins latency-bound low-conc/long-ctx vs Local-TP+DP at high-conc."""
    # Points hold total 8-die node batch fixed: (label, seqs_per_dp_on_dp8, ctx)
    points = (
        ("low_conc_long_ctx", 1, 131072),
        ("crossover_low", 2, 131072),
        ("crossover_high", 4, 131072),
        ("high_conc_mid_ctx", 32, 8192),
    )
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    layers = min(DEFAULT_AUDIT_LAYERS * 2, load_model("glm-5.3-flash").layers)

    for label, seqs_per_dp, ctx in points:
        try:
            # Topology A: KDA Local-TP + MLA Pure-DP + o_proj_tp (dp=8, tp=1, ep=8, local_tp=4)
            raw_local = compose_case(
                "glm-5.3-flash", hardware=DEFAULT_GATE_A_SKU, dp=WORLD_OCT, tp=1, ep=WORLD_OCT,
                num_devices=WORLD_OCT, layers=layers, quant="w8a8",
                o_proj_tp=LOCAL_TP_WIDTH_FOUR, linear_proj_tp=LOCAL_TP_WIDTH_FOUR,
            )
            ev_local = spine.run_step(
                spine.lower(
                    parse_case(raw_local),
                    LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=ctx, sequences=seqs_per_dp, scope="body")),
                ),
                ReplayOptions(include_submission=False),
            )
            # Topology B: DCP (dp=2, tp=4, dcp=4, ep=8) serving the SAME total node batch (seqs_per_dp * 4 per DP replica)
            raw_dcp = compose_case(
                "glm-5.3-flash", hardware=DEFAULT_GATE_A_SKU, dp=2, tp=WORLD_QUAD, dcp=WORLD_QUAD, ep=WORLD_OCT,
                num_devices=WORLD_OCT, layers=layers, quant="w8a8",
            )
            ev_dcp = spine.run_step(
                spine.lower(
                    parse_case(raw_dcp),
                    LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=ctx, sequences=seqs_per_dp * WORLD_QUAD, scope="body")),
                ),
                ReplayOptions(include_submission=False),
            )
            winner = "local_tp_dp" if ev_local.span_ms <= ev_dcp.span_ms else "dcp"
            rows.append({
                "point": label,
                "total_node_seqs": seqs_per_dp * WORLD_OCT,
                "seqs_per_dp_local": seqs_per_dp,
                "seqs_per_dp_dcp": seqs_per_dp * WORLD_QUAD,
                "context": ctx,
                "local_tp_dp_ms": round(ev_local.span_ms, 4),
                "dcp_ms": round(ev_dcp.span_ms, 4),
                "winner": winner,
            })
        except Exception as exc:  # noqa: BLE001
            findings.append(_sanitize_diagnostic(f"crossover_err:{label}:{type(exc).__name__}:{exc}"))

    if len(rows) >= 2:
        if rows[0]["winner"] != "dcp":
            findings.append(f"crossover_low_conc_long_ctx_expected_dcp_got:{rows[0]['winner']}")
        if rows[-1]["winner"] != "local_tp_dp":
            findings.append(f"crossover_high_conc_expected_local_tp_dp_got:{rows[-1]['winner']}")
        if len({r["winner"] for r in rows}) < 2:
            findings.append("crossover_missing_regime_flip")
    return {"model": "glm-5.3-flash", "points": rows, "findings": findings}


def _audit_single_track_model(item: tuple[str, str, dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    model_key, sku, kw = item
    spine = default_spine()
    spec_m = load_model(model_key)
    tp_scale = float(max(1, int(kw.get("tp", 1))))
    needs_comm = int(kw.get("tp", 1)) > 1 or int(kw.get("ep", 1)) > 1 or int(kw.get("dcp", 1)) > 1
    track_ms: dict[str, float] = {}
    findings: list[str] = []
    try:
        for trk in FEATURE_TRACKS:
            raw = compose_case(
                model_key, hardware=sku, feature_track=trk, _stack_phase="decode",
                layers=DEFAULT_AUDIT_LAYERS, **kw,
            )
            step = spine.lower(
                parse_case(raw),
                LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS, scope="body")),
            )
            m = _step_conservation_metrics(step, spine)
            ev = spine.run_step(step, ReplayOptions(include_submission=False))
            if m["fallbacks"] or m["prepare_errors"] or m["replica_issues"] or ev.span_ms <= 0.0:
                findings.append(f"track_fallback_or_zero:{model_key}:{trk}:{m['fallbacks']}:{m['prepare_errors']}")
            if needs_comm:
                _check_collective_conservation(
                    spec_m, min(DEFAULT_AUDIT_LAYERS, spec_m.layers), m["collectives"],
                    tp=int(kw.get("tp", 1)), ep=int(kw.get("ep", 1)) if spec_m.num_experts > 0 else 1,
                    dcp=int(kw.get("dcp", 1)), tokens=1, sequences=DECODE_AUDIT_SEQS,
                    label=f"track_{model_key}_{trk}", findings=findings,
                )
            _check_per_layer_conservation(
                spec_m, min(DEFAULT_AUDIT_LAYERS, spec_m.layers), m,
                phase="decode", tokens=1, context=DECODE_AUDIT_CONTEXT, sequences=DECODE_AUDIT_SEQS,
                scale=tp_scale, tp=int(kw.get("tp", 1)), label=f"track_{model_key}_{trk}", findings=findings,
            )
            track_ms[trk] = round(ev.span_ms, 5)
        for hi, lo in zip(FEATURE_TRACKS[1:-1], FEATURE_TRACKS[2:]):
            if track_ms[lo] > track_ms[hi] + TRACK_MONOTONIC_TOL_MS:
                findings.append(f"track_non_monotonic:{model_key}:{hi}={track_ms[hi]}<{lo}={track_ms[lo]}")
        return {
            "model": model_key, "tracks_ms": track_ms,
            "status": "pass" if not findings else "fail",
        }, findings
    except Exception as exc:  # noqa: BLE001
        err_s = _sanitize_diagnostic(f"{type(exc).__name__}:{exc}")
        findings.append(_sanitize_diagnostic(f"track_err:{model_key}:{err_s}"))
        return {"model": model_key, "status": "error", "error": err_s}, findings


def _audit_feature_tracks(spine: PredictionSpine, *, workers: int = 1) -> dict[str, Any]:
    """Verify 4-track recipe waterfall (raw_eager/merged/unmerged_pr/expected_ideal) in parallel."""
    del spine
    sample = (
        ("deepseek-v3", DEFAULT_A5_SKU, dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("glm-5.3-flash", DEFAULT_A5_SKU, dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("kimi-k3.1", DEFAULT_A5_SKU, dict(dp=2, tp=WORLD_QUAD, dcp=WORLD_QUAD, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("qwen3.5-397b-a17b", DEFAULT_A5_SKU, dict(dp=1, tp=WORLD_OCT, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("qwen3-32b", DEFAULT_A5_SKU, dict(dp=1, tp=WORLD_QUAD, ep=1, num_devices=WORLD_QUAD)),
        ("minimax-m3", DEFAULT_A5_SKU, dict(dp=2, tp=WORLD_QUAD, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("step-5-preview", DEFAULT_A5_SKU, dict(dp=2, tp=WORLD_QUAD, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("gemma-4-26b-a4b", DEFAULT_A5_SKU, dict(dp=2, tp=WORLD_QUAD, ep=WORLD_OCT, num_devices=WORLD_OCT)),
        ("deepseek-v4.1-flash", DEFAULT_A5_SKU, dict(dp=WORLD_OCT, tp=1, ep=WORLD_OCT, num_devices=WORLD_OCT)),
    )
    if len(sample) > 4 or workers > 1:
        pairs = _spawn_map_chunked(_audit_single_track_model, sample, workers=workers, chunk_size=4)
    else:
        pairs = [_audit_single_track_model(item) for item in sample]
    rows = [p[0] for p in pairs]
    findings = [f for p in pairs for f in p[1]]
    return {"tracks": list(FEATURE_TRACKS), "rows": rows, "findings": findings}


def _audit_model_worker(pair: tuple[str, str]) -> dict[str, Any]:
    model_key, sku = pair
    res = _audit_model_gate_a(model_key, default_spine(), sku)
    gc.collect()
    return res


def _audit_phase_worker(phase_name: str) -> Any:
    """Execute one heavy Gate A audit phase inside a fresh spawned child to cap RSS <= 512 MiB."""
    spine = default_spine()
    if phase_name == "overlays":
        return _audit_feature_overlays(spine)
    if phase_name == "aclgraph":
        return _audit_aclgraph_padded_overlay(spine)
    if phase_name == "perturb":
        return _audit_hardware_perturbation_continuity(spine)
    if phase_name == "crossover":
        return _audit_pd_decode_crossover(spine)
    raise ValueError(f"unknown_gate_a_phase:{phase_name}")


def run_gate_a(
    models: Sequence[str] = ALL_CATALOG_MODELS,
    *,
    sku: str = DEFAULT_GATE_A_SKU,
    workers: int = 1,
    include_tutorials: bool = True,
    tutorials_dir: Path | None = None,
    include_perturb: bool = True,
    include_crossover: bool = True,
    include_tracks: bool = True,
) -> dict[str, Any]:
    """Execute full Gate A structure, conservation, 4-track, tutorial, perturbation, and crossover suite."""
    if len(set(models)) != len(models):
        raise ValueError("duplicate_models_requested_in_gate_a")
    t0 = time.perf_counter()
    use_spawn = _spawn_supported() and len(models) > 4
    spine = None if use_spawn else default_spine()
    if use_spawn or (workers > 1 and len(models) > 1):
        model_rows = _spawn_map_chunked(_audit_model_worker, [(m, sku) for m in models], workers=workers, chunk_size=2)
    else:
        assert spine is not None
        model_rows = [_audit_model_gate_a(m, spine, sku) for m in models]
    gc.collect()
    if use_spawn:
        overlays = _spawn_map_chunked(_audit_phase_worker, ["overlays"], workers=1, chunk_size=1)[0]
        tracks = _audit_feature_tracks(None, workers=workers) if include_tracks else {"tracks": [], "rows": [], "findings": []}
        tutorials = (
            _audit_va_tutorial_deployments(None, workers=workers, tutorials_dir=tutorials_dir)
            if include_tutorials
            else {"total": 0, "passed": 0, "rows": [], "findings": []}
        )
        perturb = _spawn_map_chunked(_audit_phase_worker, ["perturb"], workers=1, chunk_size=1)[0] if include_perturb else {"findings": []}
        crossover = _spawn_map_chunked(_audit_phase_worker, ["crossover"], workers=1, chunk_size=1)[0] if include_crossover else {"points": [], "findings": []}
    else:
        assert spine is not None
        overlays = _audit_feature_overlays(spine)
        tracks = _audit_feature_tracks(spine, workers=workers) if include_tracks else {"tracks": [], "rows": [], "findings": []}
        tutorials = (
            _audit_va_tutorial_deployments(spine, workers=workers, tutorials_dir=tutorials_dir)
            if include_tutorials
            else {"total": 0, "passed": 0, "rows": [], "findings": []}
        )
        perturb = _audit_hardware_perturbation_continuity(spine) if include_perturb else {"findings": []}
        crossover = _audit_pd_decode_crossover(spine) if include_crossover else {"points": [], "findings": []}

    all_findings = (
        [f"{r['model']}:{item}" for r in model_rows for item in r["findings"]]
        + list(overlays["findings"])
        + list(tracks["findings"])
        + list(tutorials["findings"])
        + list(perturb["findings"])
        + list(crossover["findings"])
    )
    models_passed = sum(1 for r in model_rows if not r["findings"])
    models_failed = len(model_rows) - models_passed
    verifier_sha = _stream_file_sha256(Path(__file__).resolve())
    cat_sig = json.dumps(
        {
            "sku": sku,
            "models": sorted(models),
            "tutorials_dir": str(tutorials_dir.name) if tutorials_dir is not None else "",
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    catalog_digest = hashlib.sha256(cat_sig).hexdigest()[:16]
    return {
        "gate": "A",
        "status": "pass" if not all_findings else "fail",
        "verifier_sha256": verifier_sha,
        "catalog_digest": catalog_digest,
        "models_audited": len(model_rows),
        "models_passed": models_passed,
        "models_failed": models_failed,
        "zero_drop_conserved": (
            len(model_rows) == len(models) == models_passed + models_failed
            and bool(tutorials.get("zero_drop_conserved", True))
        ),
        "finding_count": len(all_findings),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "models": model_rows,
        "overlays": overlays,
        "feature_tracks": tracks,
        "tutorials": tutorials,
        "hardware_perturbation": perturb,
        "pd_decode_crossover": crossover,
        "findings": all_findings,
    }


def _choose_1layer_ep(hidden: int, num_experts: int, moe_i: int, quant_mode: str) -> int:
    """Pick smallest power-of-2 EP keeping 1-layer synthetic MoE weights within budget."""
    if num_experts <= 0:
        return 1
    elem_b = 1 if quant_mode == "w8a8" else 2
    for ep in (WORLD_OCT, WORLD_HEX, WORLD_MOE_LARGE, WORLD_MOE_LARGE * 2):
        local_e = max(1, math.ceil(num_experts / ep))
        if local_e * 3 * hidden * moe_i * elem_b <= SYNTH_WEIGHT_CAP_BYTES:
            return ep
    return WORLD_MOE_LARGE * 2


def _layer_structural_signature(spec: Any, plan: Any) -> tuple[Any, ...]:
    """Exhaustive structural signature over ALL LayerPlan fields & owner-role relations."""
    owner_fields = frozenset({
        "kv_owner_layer_id", "index_k_owner_layer_id", "topk_owner_layer_id",
        "candidate_source_layer_id", "index_source_layer_id",
    })
    plans = getattr(spec, "layer_plans", ())
    kv_o = int(getattr(plan, "kv_owner_layer_id", -1))
    tk_o = int(getattr(plan, "topk_owner_layer_id", -1))
    parts: list[Any] = []
    for fld in fields(plan):
        if fld.name in ("index", "name", "native_layer_id"):
            continue
        val = getattr(plan, fld.name)
        if fld.name in owner_fields:
            iv = int(val)
            is_ext = iv >= 0 and iv != int(plan.index)
            if is_ext and 0 <= iv < len(plans):
                op = plans[iv]
                val = (
                    True, int(op.compress_ratio), bool(op.indexer), bool(op.owns_long_kv),
                    iv == kv_o, iv == tk_o,
                )
            else:
                val = (False, -1, False, False, False, False)
        elif isinstance(val, (list, dict, set)):
            val = json.dumps(val, sort_keys=True)
        parts.append((fld.name, val))
    return tuple(parts)


def export_live_runner_specs(
    models: Sequence[str] = ALL_CATALOG_MODELS,
    *,
    quant_mode: str = "bf16",
    include_multirank: bool = True,
) -> dict[str, Any]:
    """Export 43-model 1-layer synthetic NPU runner specs directly from ``load_model``."""
    if len(set(models)) != len(models):
        raise ValueError("duplicate_models_requested_in_export_specs")
    cases: list[dict[str, Any]] = []
    for model_key in models:
        spec = load_model(model_key)
        seen_sigs: set[tuple[Any, ...]] = set()
        for plan in spec.layer_plans:
            sig = _layer_structural_signature(spec, plan)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            is_mla = plan.attention == "mla"
            is_dit = model_key == "hunyuan-video-dit"
            ep = _choose_1layer_ep(spec.hidden_size, plan.num_experts, plan.moe_intermediate_size, quant_mode)
            tp = 1 if is_mla else (
                min(WORLD_QUAD, max(1, plan.num_kv_heads))
                if spec.num_heads % min(WORLD_QUAD, max(1, plan.num_kv_heads)) == 0
                else 1
            )
            phase_str = "prefill" if is_dit else "decode"
            phase_suffix = "pre" if is_dit else "dec"
            base_case: dict[str, Any] = {
                "case_id": f"{model_key}__L{plan.index}_{plan.architecture}_{plan.ffn}_{quant_mode}_{phase_suffix}",
                "model": model_key,
                "layer_index": int(plan.index),
                "world_size": WORLD_SINGLE,
                "attention": str(plan.attention),
                "architecture": str(plan.architecture),
                "ffn": str(plan.ffn),
                "overlay": f"1layer_{quant_mode}_{phase_str}",
                "phase": phase_str,
                "hidden_size": int(spec.hidden_size),
                "num_heads": int(plan.num_heads or spec.num_heads),
                "num_kv_heads": int(plan.num_kv_heads or spec.num_kv_heads or 1),
                "head_dim": int(plan.head_dim or spec.head_dim or DEFAULT_KV_BLOCK_SIZE),
                "q_lora_rank": int(plan.q_lora_rank),
                "kv_lora_rank": int(plan.kv_lora_rank),
                "qk_nope_head_dim": int(plan.qk_nope_head_dim),
                "qk_rope_head_dim": int(plan.qk_rope_head_dim),
                "v_head_dim": int(plan.v_head_dim or plan.head_dim or DEFAULT_KV_BLOCK_SIZE),
                "intermediate_size": int(plan.intermediate_size or spec.intermediate_size),
                "num_experts": int(plan.num_experts),
                "top_k": int(plan.top_k),
                "moe_intermediate_size": int(plan.moe_intermediate_size),
                "num_shared_experts": int(plan.num_shared_experts),
                "shared_intermediate_size": int(plan.shared_intermediate_size or plan.moe_intermediate_size),
                "indexer": bool(plan.indexer),
                "indexer_type": str(plan.indexer_type),
                "uses_external_topk": bool(
                    (not plan.indexer)
                    and int(getattr(plan, "topk_owner_layer_id", -1)) >= 0
                    and int(plan.index_topk) > 0
                ),
                "engram": bool(getattr(plan, "engram", False)),
                "engram_width": int(getattr(getattr(spec, "engram", None), "width", 256)) if getattr(plan, "engram", False) else 0,
                "cache_kind": str(plan.cache_kind),
                "window": int(plan.window),
                "compress_ratio": int(plan.compress_ratio),
                "owns_long_kv": bool(plan.owns_long_kv),
                "owns_candidate_blocks": bool(getattr(plan, "owns_candidate_blocks", False)),
                "candidate_filter": bool(getattr(plan, "candidate_filter", False)),
                "candidate_block_size": int(getattr(plan, "candidate_block_size", 1)),
                "candidate_count": int(getattr(plan, "candidate_topk_blocks", 0)) * int(getattr(plan, "candidate_block_size", 1)),
                "index_heads": int(plan.index_heads),
                "index_dim": int(plan.index_dim),
                "index_topk": int(plan.index_topk),
                "qk_norm": bool(plan.qk_norm),
                "q_output_gate": bool(plan.q_output_gate),
                "linear_num_k_heads": int(plan.linear_key_heads or plan.linear_heads),
                "linear_num_v_heads": int(plan.linear_value_heads or plan.linear_heads),
                "linear_k_head_dim": int(plan.linear_key_head_dim or plan.linear_head_dim or DEFAULT_KV_BLOCK_SIZE),
                "linear_v_head_dim": int(plan.linear_head_dim or DEFAULT_KV_BLOCK_SIZE),
                "dp": ep if is_mla else 1,
                "tp": tp,
                "ep": ep,
                "dcp": 1,
                "pcp": 1,
                "sequences": 1 if is_dit else LIVE_DECODE_SEQS,
                "active_sequences": 1 if is_dit else LIVE_DECODE_SEQS,
                "q_len": PREFILL_LIVE_TOKENS if is_dit else 1,
                "context": 0 if is_dit else (
                    max(LIVE_DECODE_CONTEXT, min(32768, int(getattr(plan, "candidate_topk_blocks", 0)) * int(getattr(plan, "candidate_block_size", 1)) * 2))
                    if (plan.indexer and (bool(getattr(plan, "candidate_filter", False)) or bool(getattr(plan, "owns_candidate_blocks", False))))
                    else LIVE_DECODE_CONTEXT
                ),
                "quant_mode": quant_mode,
            }
            cases.append(base_case)

    overlay_targets = (
        ("deepseek-v3", 3, "prefill", dict(phase="prefill", sequences=1, active_sequences=1, q_len=PREFILL_LIVE_TOKENS, context=0)),
        ("deepseek-v3", 3, "mtp_q4", dict(phase="decode", sequences=LIVE_DECODE_SEQS, active_sequences=LIVE_DECODE_SEQS, q_len=MTP_QUERY_LEN_K3, context=LIVE_DECODE_CONTEXT)),
        ("qwen3.5-397b-a17b", 3, "eager_padded_b8_a6", dict(phase="decode", sequences=WORLD_OCT, active_sequences=6, q_len=1, context=LIVE_DECODE_CONTEXT)),
        ("qwen3.5-397b-a17b", 0, "gdn_prefill", dict(phase="prefill", sequences=1, active_sequences=1, q_len=PREFILL_LIVE_TOKENS, context=0)),
        ("glm-5.3-flash", 3, "w8a8_dsa_decode", dict(phase="decode", quant_mode="w8a8", sequences=LIVE_DECODE_SEQS, active_sequences=LIVE_DECODE_SEQS, q_len=1, context=LIVE_DECODE_CONTEXT)),
        ("glm-5.3-flash", 4, "kda_prefill", dict(phase="prefill", sequences=1, active_sequences=1, q_len=PREFILL_LIVE_TOKENS, context=0)),
        ("deepseek-v3", 3, "mxfp8_decode", dict(phase="decode", quant_mode="mxfp8", sequences=LIVE_DECODE_SEQS, active_sequences=LIVE_DECODE_SEQS, q_len=1, context=LIVE_DECODE_CONTEXT)),
        ("qwen3-32b", 0, "c8_decode", dict(phase="decode", quant_mode="c8", sequences=LIVE_DECODE_SEQS, active_sequences=LIVE_DECODE_SEQS, q_len=1, context=LIVE_DECODE_CONTEXT)),
        ("qwen3-32b", 0, "w4a8_decode", dict(phase="decode", quant_mode="w4a8", sequences=LIVE_DECODE_SEQS, active_sequences=LIVE_DECODE_SEQS, q_len=1, context=LIVE_DECODE_CONTEXT)),
        ("qwen3-32b", 0, "mxfp4_decode", dict(phase="decode", quant_mode="mxfp4", sequences=LIVE_DECODE_SEQS, active_sequences=LIVE_DECODE_SEQS, q_len=1, context=LIVE_DECODE_CONTEXT)),
    )
    by_model_layer = {(c["model"], c["layer_index"]): c for c in cases}
    for m_key, l_idx, tag, overrides in overlay_targets:
        template = by_model_layer.get((m_key, l_idx))
        if template is not None:
            cases.append({**template, **overrides, "case_id": f"{m_key}__L{l_idx}_{tag}", "overlay": tag})

    if include_multirank:
        mr_templates = (
            ("qwen3-235b-a22b", 0, "mr4_tp4_ep4_dec", dict(world_size=WORLD_QUAD, dp=1, tp=WORLD_QUAD, ep=WORLD_QUAD, dcp=1, pcp=1)),
            ("deepseek-v3", 3, "mr4_puredp4_ep4_dec", dict(world_size=WORLD_QUAD, dp=WORLD_QUAD, tp=1, ep=WORLD_QUAD, dcp=1, pcp=1, quant_mode="w8a8")),
            ("deepseek-v3", 3, "mr4_dcp4_dec", dict(world_size=WORLD_QUAD, dp=1, tp=WORLD_QUAD, ep=WORLD_QUAD, dcp=WORLD_QUAD, pcp=1, context=LIVE_LONG_CONTEXT, quant_mode="w8a8")),
            ("deepseek-v3", 3, "mr4_pcp2_tp2_dcp4_dec", dict(world_size=WORLD_QUAD, dp=1, tp=2, ep=WORLD_QUAD, dcp=WORLD_QUAD, pcp=2, context=LIVE_LONG_CONTEXT, quant_mode="w8a8")),
            ("deepseek-v3", 3, "mr4_pcp4_pre", dict(world_size=WORLD_QUAD, phase="prefill", dp=1, tp=1, ep=WORLD_QUAD, dcp=1, pcp=WORLD_QUAD, sequences=1, active_sequences=1, q_len=PREFILL_LIVE_TOKENS, context=0, quant_mode="w8a8")),
            ("qwen3-32b", 0, "mr4_pcp4_pre", dict(world_size=WORLD_QUAD, phase="prefill", dp=1, tp=1, ep=1, dcp=1, pcp=WORLD_QUAD, sequences=1, active_sequences=1, q_len=PREFILL_LIVE_TOKENS, context=0)),
            ("glm-5.3-flash", 4, "mr4_xover_kda_localtp_low", dict(world_size=WORLD_QUAD, dp=WORLD_QUAD, tp=1, ep=WORLD_QUAD, linear_proj_tp=WORLD_QUAD, sequences=LIVE_LOW_CONC_SEQS, active_sequences=LIVE_LOW_CONC_SEQS, context=LIVE_DECODE_CONTEXT)),
            ("glm-5.3-flash", 3, "mr4_xover_mla_dp_oprojtp_low", dict(world_size=WORLD_QUAD, dp=WORLD_QUAD, tp=1, ep=WORLD_QUAD, dcp=1, o_proj_tp=WORLD_QUAD, sequences=LIVE_LOW_CONC_SEQS, active_sequences=LIVE_LOW_CONC_SEQS, context=LIVE_DECODE_CONTEXT)),
            ("glm-5.3-flash", 3, "mr4_xover_mla_dcp4_low", dict(world_size=WORLD_QUAD, dp=1, tp=WORLD_QUAD, ep=WORLD_QUAD, dcp=WORLD_QUAD, sequences=LIVE_LOW_CONC_SEQS * WORLD_QUAD, active_sequences=LIVE_LOW_CONC_SEQS * WORLD_QUAD, context=LIVE_DECODE_CONTEXT)),
            ("glm-5.3-flash", 4, "mr4_xover_kda_localtp_high", dict(world_size=WORLD_QUAD, dp=WORLD_QUAD, tp=1, ep=WORLD_QUAD, linear_proj_tp=WORLD_QUAD, sequences=LIVE_HIGH_CONC_SEQS, active_sequences=LIVE_HIGH_CONC_SEQS, context=LIVE_LONG_CONTEXT)),
            ("glm-5.3-flash", 3, "mr4_xover_mla_dp_oprojtp_high", dict(world_size=WORLD_QUAD, dp=WORLD_QUAD, tp=1, ep=WORLD_QUAD, dcp=1, o_proj_tp=WORLD_QUAD, sequences=LIVE_HIGH_CONC_SEQS, active_sequences=LIVE_HIGH_CONC_SEQS, context=LIVE_LONG_CONTEXT)),
            ("glm-5.3-flash", 3, "mr4_xover_mla_dcp4_high", dict(world_size=WORLD_QUAD, dp=1, tp=WORLD_QUAD, ep=WORLD_QUAD, dcp=WORLD_QUAD, sequences=LIVE_HIGH_CONC_SEQS * WORLD_QUAD, active_sequences=LIVE_HIGH_CONC_SEQS * WORLD_QUAD, context=LIVE_LONG_CONTEXT)),
        )
        for m_key, l_idx, tag, overrides in mr_templates:
            template = by_model_layer.get((m_key, l_idx))
            if template is not None:
                cases.append({**template, **overrides, "case_id": f"{m_key}__L{l_idx}_{tag}", "overlay": tag})

    spec_digest = _compute_manifest_cases_digest(cases)
    return {
        "schema_version": "mindie-sim-1layer-specs/2",
        "model_count": len(models),
        "spec_sha256": spec_digest,
        "cases": cases,
    }


def _compute_manifest_cases_digest(cases: Sequence[Mapping[str, Any]]) -> str:
    """Compute canonical 16-hex SHA-256 digest over predeclared manifest case definitions."""
    return hashlib.sha256(
        json.dumps(list(cases), sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()[:16]


def _safe_float(val: Any, default: float = float("nan")) -> float:
    if isinstance(val, bool):
        return default
    try:
        out = float(val)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = -1) -> int:
    if isinstance(val, bool) or val is None:
        return default
    try:
        if isinstance(val, float) and (not math.isfinite(val) or not val.is_integer()):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _is_control_helper_row(kernel: Mapping[str, Any]) -> bool:
    """Return True for non-compute control/launch helpers, NEVER masking A5/A3 ``COMMUNICATION``/``hcom_*`` comms."""
    op_type = str(kernel.get("type", "")).strip().lower()
    task_type = str(kernel.get("task_type", "")).strip().upper()
    core_type = str(kernel.get("core", "")).strip().upper()
    if task_type == "COMMUNICATION" or core_type == "COMMUNICATION" or op_type.startswith("hcom_"):
        return False
    return op_type in CONTROL_HELPER_KERNEL_TYPES or op_type.endswith("aicpukernel")


def _classify_csv_kernel(kernel: Mapping[str, Any]) -> str:
    """Classify one CANN ``kernel_details.csv`` row strictly by ``type`` and ``COMMUNICATION`` metadata."""
    op_type = str(kernel.get("type", "")).strip().lower()
    task_type = str(kernel.get("task_type", "")).strip().upper()
    core_type = str(kernel.get("core", "")).strip().upper()
    if _is_control_helper_row(kernel):
        return "control_helper"
    if task_type == "COMMUNICATION" or core_type == "COMMUNICATION":
        return "collective"
    if op_type in AUX_NON_ATTN_KERNEL_TYPES:
        return "other"
    primary = op_type if (op_type and op_type != "unknown") else str(kernel.get("name", "")).strip().lower()
    if any(t in primary for t in ("hcom", "hccl", "allreduce", "allgather", "alltoall", "reducescatter", "distribute_dispatch", "distribute_combine")):
        return "collective"
    if any(t in primary for t in ("lightningindexer", "lightning_indexer")):
        return "indexer"
    if any(t in primary for t in ("fusedinferattention", "sparseflashattention", "promptflashattention", "increflashattention", "attentionupdate", "fia")):
        return "attention"
    if any(t in primary for t in ("gateddeltarule", "kda", "causalconv", "recurrent")):
        return "linear_attention"
    if any(t in primary for t in ("groupedmatmul", "moeinitrouting", "moegating", "moefinalize")):
        return "moe"
    if any(t in primary for t in ("rmsnorm", "layernorm", "rotary", "dynamicquant", "dynamicmxquant", "swiglu")):
        return "norm_quant_rope"
    if any(t in primary for t in ("matmul", "batchmatmul", "gemm", "quantbatchmatmul", "quantmatmul")):
        return "matmul"
    return "other"


def _classify_sim_op(op: Any) -> str:
    """Map one lowered ``OpSpec`` into the same canonical category for 1:1 diff."""
    fam = str(op.family).lower()
    oid = str(op.op_id).lower()
    blk = str(dict(op.metadata).get("structure_block", "")).lower() if op.metadata else ""
    if float(op.link_bytes or 0.0) > 0.0 or fam in COLLECTIVE_FAMILIES:
        return "collective"
    if fam in INDEXER_FAMILIES or "indexer" in oid or (fam == "topk" and blk == "attention") or "index_topk" in oid:
        return "indexer"
    if fam in ATTN_FAMILIES:
        return "other" if "compress" in oid else "attention"
    if fam in LINEAR_ATTN_FAMILIES or any(
        t in fam or t in oid for t in ("gdn", "kda", "recurrent_scan", "short_conv", "linear_attn", "delta_rule")
    ):
        return "linear_attention"
    if (
        fam in (MOE_FAMILIES - {"swiglu"})
        or "expert" in oid
        or "gmm" in oid
        or "router" in oid
        or fam == "routing"
        or (fam == "topk" and blk != "attention")
    ):
        return "moe"
    if any(t in fam or t in oid for t in ("norm", "rotary", "quant", "swiglu")):
        return "norm_quant_rope"
    if any(t in fam for t in ("gemm", "matmul", "projection", "bmm")):
        return "matmul"
    return "other"


def _build_layer_slice_spec(spec: Any, layer_idx: int) -> tuple[Any, int]:
    """Slice ``spec`` to include external cache/index owner layers without mutating ownership."""
    total_layers = len(spec.layer_plans)
    if layer_idx < 0 or layer_idx >= total_layers:
        raise ValueError(f"layer_index_out_of_range:{layer_idx}:total_layers={total_layers}")

    needed: set[int] = {layer_idx}
    frontier = [layer_idx]
    owner_attrs = (
        "kv_owner_layer_id", "index_k_owner_layer_id", "topk_owner_layer_id",
        "candidate_source_layer_id", "index_source_layer_id",
    )
    while frontier:
        cur = frontier.pop()
        plan = spec.layer_plans[cur]
        for attr in owner_attrs:
            ref = int(getattr(plan, attr, -1))
            if 0 <= ref < total_layers and ref not in needed:
                needed.add(ref)
                frontier.append(ref)

    ordered = sorted(needed)
    remap = {old_i: new_i for new_i, old_i in enumerate(ordered)}
    sliced_plans = []
    for old_i in ordered:
        p = spec.layer_plans[old_i]
        new_i = remap[old_i]
        sliced_plans.append(replace(
            p,
            index=new_i,
            native_layer_id=new_i if int(getattr(p, "native_layer_id", -1)) >= 0 else -1,
            kv_owner_layer_id=remap.get(int(getattr(p, "kv_owner_layer_id", -1)), -1),
            index_k_owner_layer_id=remap.get(int(getattr(p, "index_k_owner_layer_id", -1)), -1),
            topk_owner_layer_id=remap.get(int(getattr(p, "topk_owner_layer_id", -1)), -1),
            candidate_source_layer_id=remap.get(int(getattr(p, "candidate_source_layer_id", -1)), -1),
            index_source_layer_id=remap.get(int(getattr(p, "index_source_layer_id", -1)), -1),
        ))
    engram_val = getattr(spec, "engram", None)
    if engram_val is not None and getattr(engram_val, "tables", None):
        kept_tables = tuple(
            replace(t, layer_id=remap[int(t.layer_id)])
            for t in engram_val.tables
            if int(t.layer_id) in remap
        )
        engram_val = replace(engram_val, tables=kept_tables) if kept_tables else None
    sliced_spec = replace(
        spec, layers=len(sliced_plans), body_layers=len(sliced_plans),
        layer_plans=tuple(sliced_plans), engram=engram_val,
    )
    return sliced_spec, remap[layer_idx]


def _interval_union_us(kernels: Sequence[Mapping[str, Any]]) -> float:
    """Compute exact merged timeline union of finite positive device-busy intervals."""
    intervals: list[tuple[float, float]] = []
    for k in kernels:
        d_u = _safe_float(k.get("duration_us", 0.0))
        s_u = _safe_float(k.get("start_us", 0.0))
        if math.isfinite(d_u) and math.isfinite(s_u) and d_u > 0.0 and s_u > 0.0:
            intervals.append((s_u, s_u + d_u))
    intervals.sort()
    if not intervals:
        return 0.0
    lo, hi = intervals[0]
    busy = 0.0
    for s_u, e_u in intervals[1:]:
        if s_u <= hi:
            hi = max(hi, e_u)
        else:
            busy += hi - lo
            lo, hi = s_u, e_u
    busy += hi - lo
    return float(busy)


def _recompute_window_timing(cap_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute spans, interval-union gap, compute sum, waits, and blocker evidence from raw rows."""
    raw_rows = list(cap_dict.get("kernels") or [])
    valid: list[Mapping[str, Any]] = []
    for k in raw_rows:
        d_u = _safe_float(k.get("duration_us", 0.0))
        if not math.isfinite(d_u):
            return {
                "valid": False, "elapsed_span_us": 0.0, "union_busy_us": 0.0,
                "uncovered_gap_us": 0.0, "sum_kernel_us": 0.0, "compute_sum_us": 0.0,
                "core_compute_sum_us": 0.0, "wait_us_sum": 0.0, "kernel_count": 0, "stream_count": 0,
                "stream_blocked": False, "gap_contaminated": True, "reason": "non_numeric_kernel_duration",
            }
        if d_u > 0.0:
            valid.append(k)
    if not valid:
        return {
            "valid": False, "elapsed_span_us": 0.0, "union_busy_us": 0.0,
            "uncovered_gap_us": 0.0, "sum_kernel_us": 0.0, "compute_sum_us": 0.0,
            "core_compute_sum_us": 0.0, "wait_us_sum": 0.0, "kernel_count": 0, "stream_count": 0,
            "stream_blocked": False, "gap_contaminated": True, "reason": "empty_window_kernels",
        }
    starts = [_safe_float(k.get("start_us", 0.0)) for k in valid]
    if any((not math.isfinite(s)) or s <= 0.0 for s in starts):
        return {
            "valid": False, "elapsed_span_us": 0.0, "union_busy_us": 0.0,
            "uncovered_gap_us": 0.0, "sum_kernel_us": 0.0, "compute_sum_us": 0.0,
            "core_compute_sum_us": 0.0, "wait_us_sum": 0.0, "kernel_count": len(valid), "stream_count": 0,
            "stream_blocked": False, "gap_contaminated": True, "reason": "missing_kernel_start_timestamps",
        }
    ends = [s + float(k["duration_us"]) for s, k in zip(starts, valid)]
    sum_us = float(sum(float(k["duration_us"]) for k in valid))
    compute_rows = [k for k in valid if _classify_csv_kernel(k) != "control_helper"]
    compute_sum_us = float(sum(float(k["duration_us"]) for k in compute_rows))
    core_compute_sum_us = float(sum(
        float(k["duration_us"]) for k in compute_rows if _classify_csv_kernel(k) != "other"
    ))
    elapsed = max(ends) - min(starts)
    union_busy = _interval_union_us(valid)
    gap_us = max(0.0, elapsed - union_busy)
    wait_sum = float(sum(max(0.0, _safe_float(k.get("wait_us", 0.0), 0.0)) for k in valid))
    stream_blocked = bool(cap_dict.get("stream_blocked", False))
    exp_blk = _safe_int(cap_dict.get("blocker_expected_count", 0), -1)
    got_blk = _safe_int(cap_dict.get("blocker_matched_count", 0), -1)
    blocker_ok = bool(stream_blocked and exp_blk >= STREAM_BLOCKER_ITERS_EXPECTED and got_blk == exp_blk)
    gap_limit = max(BUBBLE_REL_LIMIT * union_busy, BUBBLE_ATOL_US)
    gap_bad = (not blocker_ok) or (elapsed <= 0.0) or (compute_sum_us <= 0.0) or (gap_us > gap_limit)
    compute_stream_ids = {_safe_int(k.get("stream_id", 0), -1) for k in compute_rows}
    all_stream_ids = {_safe_int(k.get("stream_id", 0), -1) for k in valid}
    bad_stream_id = (-1 in all_stream_ids) or any(_safe_int(k.get("stream_id", 0), -1) < 0 for k in valid)
    if bad_stream_id or exp_blk < 0 or got_blk < 0:
        return {
            "valid": False, "elapsed_span_us": 0.0, "union_busy_us": 0.0,
            "uncovered_gap_us": 0.0, "sum_kernel_us": 0.0, "compute_sum_us": 0.0,
            "core_compute_sum_us": 0.0, "wait_us_sum": 0.0, "kernel_count": len(valid),
            "stream_count": 0, "all_stream_count": 0, "stream_blocked": False,
            "gap_contaminated": True, "reason": "non_numeric_stream_or_blocker_field",
        }
    return {
        "valid": elapsed > 0.0 and compute_sum_us > 0.0,
        "elapsed_span_us": round(elapsed, 4),
        "union_busy_us": round(union_busy, 4),
        "uncovered_gap_us": round(gap_us, 4),
        "sum_kernel_us": round(sum_us, 4),
        "compute_sum_us": round(compute_sum_us, 4),
        "core_compute_sum_us": round(core_compute_sum_us, 4),
        "wait_us_sum": round(wait_sum, 4),
        "kernel_count": len(valid),
        "stream_count": len(compute_stream_ids),
        "all_stream_count": len(all_stream_ids),
        "stream_blocked": blocker_ok,
        "gap_contaminated": gap_bad,
        "reason": "" if not gap_bad else (
            "no_compute_kernels" if compute_sum_us <= 0.0
            else ("unblocked_window" if not blocker_ok else "uncovered_gap_exceeded")
        ),
    }


def _score_latency_pair(pred_u: float, obs_u: float) -> dict[str, Any]:
    """Unfiltered ``|T_pred - T_obs| <= max(10% * T_obs, 2.0 us)`` requiring positive finite values."""
    if (not math.isfinite(pred_u)) or (not math.isfinite(obs_u)) or pred_u <= 0.0 or obs_u <= 0.0:
        return {
            "pred_us": round(pred_u, 3) if math.isfinite(pred_u) else None,
            "obs_us": round(obs_u, 3) if math.isfinite(obs_u) else None,
            "signed_err_pct": None,
            "limit_us": round(PHYSICAL_ATOL_NS / US_TO_NS, 3),
            "pass_physical_tol": False,
            "invalid_measurement": True,
        }
    err_u = pred_u - obs_u
    pct = err_u / obs_u * PERCENT_SCALE
    exceeded = exceeds_physical_tolerance(pred_u * US_TO_NS, obs_u * US_TO_NS)
    return {
        "pred_us": round(pred_u, 3),
        "obs_us": round(obs_u, 3),
        "signed_err_pct": pct,
        "limit_us": round(_physical_limit_us(obs_u), 3),
        "pass_physical_tol": not exceeded,
    }


def _classify_collective_kind_csv(kernel: Mapping[str, Any], *, after_gmm: bool = False) -> str | None:
    """Classify an HCCL/MC2 CSV kernel into its exact directed collective primitive."""
    if _is_control_helper_row(kernel):
        return None
    op_type = str(kernel.get("type", "")).strip().lower()
    name_low = str(kernel.get("name", "")).strip().lower()
    text = op_type if (op_type and op_type != "unknown") else name_low
    full = f"{text} {name_low}"
    if any(t in full for t in ("distribute_dispatch", "distribute_combine", "moe_distribute", "mc2")):
        if "combine" in full:
            return "mc2_combine"
        if "dispatch" in full:
            return "mc2_dispatch"
        return "mc2_combine" if after_gmm else "mc2_dispatch"
    if "alltoall" in text or "all_to_all" in text:
        return "all_to_all_combine" if after_gmm else "all_to_all_dispatch"
    if "allgather" in text or "all_gather" in text:
        return "all_gather"
    if "reducescatter" in text or "reduce_scatter" in text:
        return "reduce_scatter"
    if "allreduce" in text or "all_reduce" in text:
        return "all_reduce"
    task_type = str(kernel.get("task_type", "")).strip().upper()
    core_type = str(kernel.get("core", "")).strip().upper()
    if "hccl" in text or "hcom" in text or task_type == "COMMUNICATION" or core_type == "COMMUNICATION":
        return "other_collective"
    return None


def _classify_collective_kind_sim(op: Any) -> str | None:
    """Classify a lowered ``OpSpec`` into its exact directed collective primitive."""
    fam = str(op.family).lower()
    oid = str(op.op_id).lower()
    meta = dict(op.metadata) if op.metadata else {}
    kind_text = f"{fam} {oid} {meta.get('collective', '')} {meta.get('collective_op', '')}".lower()
    if float(op.link_bytes or 0.0) <= 0.0 and fam not in COLLECTIVE_FAMILIES:
        return None
    if any(t in kind_text for t in ("mc2", "moe_distribute")):
        return "mc2_combine" if "combine" in kind_text else "mc2_dispatch"
    if "alltoall" in kind_text or "all_to_all" in kind_text or fam in ("dispatch", "combine"):
        return "all_to_all_combine" if "combine" in kind_text else "all_to_all_dispatch"
    if "allgather" in kind_text or "all_gather" in kind_text:
        return "all_gather"
    if "reducescatter" in kind_text or "reduce_scatter" in kind_text:
        return "reduce_scatter"
    if "allreduce" in kind_text or "all_reduce" in kind_text or "output_reduce" in oid:
        return "all_reduce"
    return "other_collective"


def _extract_observed_gemm_shapes(kernels: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, int]]:
    """Parse real positive-duration ``(M, K, N)`` matrix shapes (ND 2D, transposed QuantMatmul, or FRACTAL_NZ 4D)."""
    shapes: list[tuple[int, int, int]] = []
    for k in kernels:
        if _classify_csv_kernel(k) != "matmul":
            continue
        raw = str(k.get("input_shapes", "")).strip().strip('"')
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        if len(parts) < 2:
            continue
        a_dims = [int(x) for x in parts[0].split(",") if x.strip().isdigit()]
        b_dims = [int(x) for x in parts[1].split(",") if x.strip().isdigit()]
        out_raw = str(k.get("output_shapes", "")).strip().strip('"')
        out_parts = [p.strip() for p in out_raw.split(";") if p.strip()]
        out_dims = [int(x) for x in out_parts[0].split(",") if x.strip().isdigit()] if out_parts else []
        if len(a_dims) == 2 and len(b_dims) == 2 and a_dims[1] == b_dims[0]:
            shapes.append((a_dims[0], a_dims[1], b_dims[1]))
        elif (
            len(a_dims) == 2 and len(b_dims) == 2 and len(out_dims) == 2
            and a_dims[0] == out_dims[0] and b_dims[0] == out_dims[1]
            and a_dims[1] in (b_dims[1], b_dims[1] * 2)
        ):
            shapes.append((a_dims[0], a_dims[1], b_dims[0]))
        elif (
            len(a_dims) == 4 and len(b_dims) == 4
            and a_dims[2] == a_dims[3] == b_dims[2] == b_dims[3] == FRACTAL_NZ_TILE
            and a_dims[0] == b_dims[1]
        ):
            shapes.append((a_dims[1] * FRACTAL_NZ_TILE, a_dims[0] * FRACTAL_NZ_TILE, b_dims[0] * FRACTAL_NZ_TILE))
    return shapes


def _extract_observed_attn_tensors(kernels: Sequence[Mapping[str, Any]]) -> list[list[tuple[int, ...]]]:
    """Extract per-kernel executed tensor shape tuples from positive attention / linear-attn kernels."""
    per_kernel: list[list[tuple[int, ...]]] = []
    for k in kernels:
        if _classify_csv_kernel(k) not in ("attention", "linear_attention"):
            continue
        raw = str(k.get("input_shapes", "")).strip().strip('"')
        tensors: list[tuple[int, ...]] = []
        for part in raw.split(";"):
            dims = tuple(int(tok.strip()) for tok in part.split(",") if tok.strip().isdigit())
            if dims:
                tensors.append(dims)
        per_kernel.append(tensors)
    return per_kernel


def _collapse_phase_order(categories: Sequence[str]) -> list[str]:
    """Extract deduplicated sequential order of core semantic layer phases."""
    core = {"indexer", "attention", "linear_attention", "moe"}
    seq: list[str] = []
    for c in categories:
        if c in core and (not seq or seq[-1] != c):
            seq.append(c)
    return seq


def _compute_layer_shape_digest(kernels: Sequence[Mapping[str, Any]]) -> str:
    """Recompute per-stream execution-ordered 16-hex compute-shape digest matching ``run_1layer_npu_subgraphs``."""
    by_stream: dict[int, list[tuple[float, int, Mapping[str, Any]]]] = {}
    for orig_idx, k in enumerate(kernels):
        if _is_control_helper_row(k):
            continue
        by_stream.setdefault(_safe_int(k.get("stream_id", 0), 0), []).append(
            (_safe_float(k.get("start_us", 0.0), 0.0), orig_idx, k)
        )
    stream_sigs: list[tuple[str, ...]] = []
    for rows in by_stream.values():
        rows.sort(key=lambda t: (t[0], t[1]))
        stream_sigs.append(tuple(
            f"{k.get('type', '')}|{k.get('input_shapes', '')}|{k.get('input_dtypes', '')}" for _, _, k in rows
        ))
    stream_sigs.sort()
    det_sig = [line for seq in stream_sigs for line in (*seq, "--")]
    return hashlib.sha256("\n".join(det_sig).encode("utf-8")).hexdigest()[:16]


def _verify_rank_receipts(
    capture_case: Mapping[str, Any],
    world_size: int,
    layer_win: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit unique full-rank receipts, physical telemetry bounds, per-stream digest, and clocks."""
    receipts = list(capture_case.get("rank_receipts") or [])
    diffs: list[str] = []
    if not receipts:
        return {
            "rank_parity_ok": False, "rank_parity_diffs": ["missing_rank_receipts"],
            "ranks_verified": 0, "worst_rank_span_us": 0.0, "rank_freq_mhz_min": 0.0,
        }
    rank_list = [_safe_int(r.get("rank", -1), -1) for r in receipts]
    expected_ranks = set(range(world_size))
    if len(receipts) != world_size or len(rank_list) != len(set(rank_list)) or set(rank_list) != expected_ranks:
        diffs.append(f"rank_receipts_invalid_set:seen={rank_list},expected={sorted(expected_ranks)}")

    layer_kernels = list((capture_case.get("layer_capture") or {}).get("kernels") or [])
    expected_r0_digest = _compute_layer_shape_digest(layer_kernels) if layer_kernels else ""
    valid_spans: list[float] = []
    valid_freqs: list[float] = []
    for rec in receipts:
        rk = _safe_int(rec.get("rank", -1), -1)
        if rec.get("status") != "ok":
            diffs.append(f"rank_{rk}_status_non_ok:{rec.get('status')}")
        if "sum_kernel_us" not in rec:
            diffs.append(f"rank_{rk}_missing_sum_kernel_us")
            continue
        span = _safe_float(rec.get("elapsed_span_us"))
        busy = _safe_float(rec.get("union_busy_us"))
        gap = _safe_float(rec.get("uncovered_gap_us"))
        sum_k = _safe_float(rec.get("sum_kernel_us"))
        freq = _safe_float(rec.get("observed_aicore_freq_mhz"))
        k_cnt = _safe_int(rec.get("kernel_count", -1), -1)
        s_cnt = _safe_int(rec.get("stream_count", -1), -1)
        b_cnt = _safe_int(rec.get("blocker_matched_count", -1), -1)
        if any(not math.isfinite(x) for x in (span, busy, gap, sum_k, freq)) or min(k_cnt, s_cnt, b_cnt) < 0:
            diffs.append(f"rank_{rk}_non_numeric_telemetry")
            continue
        if k_cnt <= 0 or span <= 0.0 or s_cnt < 1 or s_cnt > k_cnt or b_cnt < STREAM_BLOCKER_ITERS_EXPECTED:
            diffs.append(f"rank_{rk}_invalid_counts_or_span:k={k_cnt},s={s_cnt},blk={b_cnt},span={span}")
        else:
            valid_spans.append(span)
        if (
            busy <= 0.0 or gap < 0.0 or busy > span + 1.0
            or sum_k <= 0.0 or sum_k < busy - 1.0
            or abs((span - busy) - gap) > 1.0
        ):
            diffs.append(f"rank_{rk}_impossible_busy_span:busy={busy},span={span},gap={gap},sum={sum_k}")
        elif (not bool(rec.get("stream_blocked", False))) or gap > max(BUBBLE_REL_LIMIT * busy, BUBBLE_ATOL_US):
            diffs.append(f"rank_{rk}_bubble_contaminated:gap={gap}")
        if freq < MIN_PLAUSIBLE_AICORE_FREQ_MHZ:
            diffs.append(f"rank_{rk}_implausible_or_missing_freq:{freq}")
        else:
            valid_freqs.append(freq)
        digest = str(rec.get("shape_digest", "")).strip()
        if not HEX_DIGEST_RE.match(digest):
            diffs.append(f"rank_{rk}_invalid_shape_digest_format:{digest}")
        elif rk == 0 and expected_r0_digest and digest != expected_r0_digest:
            diffs.append(f"rank_0_shape_digest_mismatch:{digest}!={expected_r0_digest}")
        if rk == 0 and layer_win is not None and layer_win.get("valid"):
            exp_r0_streams = int(layer_win.get("all_stream_count", layer_win["stream_count"]))
            if (
                abs(span - float(layer_win["elapsed_span_us"])) > 1.0
                or abs(busy - float(layer_win["union_busy_us"])) > 1.0
                or abs(gap - float(layer_win["uncovered_gap_us"])) > 1.0
                or abs(sum_k - float(layer_win["sum_kernel_us"])) > 1.0
                or k_cnt != int(layer_win["kernel_count"])
                or s_cnt != exp_r0_streams
            ):
                diffs.append(f"rank_0_layer_receipt_mismatch:rec=({span},{busy},{gap},{sum_k},{k_cnt},{s_cnt})!=win_s={exp_r0_streams}")
        if "compute_kernel_count" in rec:
            ck_cnt = _safe_int(rec.get("compute_kernel_count", -1), -1)
            if ck_cnt <= 0 or ck_cnt > k_cnt:
                diffs.append(f"rank_{rk}_invalid_compute_kernel_count:ck={ck_cnt},k={k_cnt}")
            elif rk == 0 and layer_kernels and ck_cnt != sum(1 for k in layer_kernels if not _is_control_helper_row(k)):
                diffs.append(f"rank_0_compute_kernel_count_mismatch:{ck_cnt}")

    if world_size > 1:
        ok_recs = [r for r in receipts if r.get("status") == "ok"]
        if ok_recs and all("compute_kernel_count" in r for r in ok_recs):
            k_counts = {_safe_int(r.get("compute_kernel_count", -1), -1) for r in ok_recs}
        else:
            k_counts = {_safe_int(r.get("kernel_count", -1), -1) for r in ok_recs}
        if len(k_counts) > 1:
            diffs.append(f"rank_kernel_count_divergence:{sorted(k_counts)}")
        par_map = dict(capture_case.get("parallel") or {})
        is_dynamic_ep = _safe_int(par_map.get("ep", 1), 1) > 1 and str(capture_case.get("ffn", "dense")) == "moe"
        if is_dynamic_ep and all(HEX_DIGEST_RE.match(str(r.get("op_type_digest", ""))) for r in receipts if r.get("status") == "ok"):
            op_digs = {str(r.get("op_type_digest", "")) for r in receipts if r.get("status") == "ok"}
            if len(op_digs) > 1:
                diffs.append(f"rank_op_type_digest_divergence:{sorted(op_digs)}")
        else:
            digests = {str(r.get("shape_digest", "")) for r in receipts if r.get("status") == "ok" and HEX_DIGEST_RE.match(str(r.get("shape_digest", "")))}
            if len(digests) > 1:
                diffs.append(f"rank_shape_digest_divergence:{sorted(digests)}")

    all_ranks_have_freq = len(valid_freqs) == world_size
    return {
        "rank_parity_ok": not diffs,
        "rank_parity_diffs": diffs,
        "ranks_verified": len(set(rank_list) & expected_ranks) if not diffs else 0,
        "worst_rank_span_us": round(max(valid_spans, default=0.0), 4),
        "rank_freq_mhz_min": round(min(valid_freqs), 2) if all_ranks_have_freq else 0.0,
    }


def _check_collective_payloads(
    sim_ops: Sequence[Any],
    obs_positive: Sequence[Mapping[str, Any]],
    world_size: int,
    topo_diffs: list[str],
) -> bool:
    """Validate observed collective group sizes and payload bytes; return True iff every comm row carried verified telemetry."""
    sim_links = [float(op.link_bytes or 0.0) for op in sim_ops if float(op.link_bytes or 0.0) >= MIN_COLLECTIVE_LINK_BYTES]
    if not sim_links:
        return False
    min_required_bytes = max(float(MIN_OBSERVED_COLLECTIVE_BYTES), COLLECTIVE_PAYLOAD_MIN_REL * min(sim_links))
    max_allowed_bytes = COLLECTIVE_PAYLOAD_MAX_REL * max(sim_links)
    comm_rows = [row for row in obs_positive if _classify_collective_kind_csv(row) is not None]
    if not comm_rows:
        return False
    all_have_payload_and_group = True
    for row in comm_rows:
        grp = row.get("comm_group_size", row.get("group_size"))
        p_bytes = row.get("payload_bytes")
        if grp is None or p_bytes is None:
            all_have_payload_and_group = False
        if grp is not None:
            g_int = _safe_int(grp, -1)
            if g_int < 2 or g_int > max(2, world_size):
                topo_diffs.append(f"collective_group_size_out_of_bounds:obs={grp},world={world_size}")
                return False
        if p_bytes is not None:
            val = _safe_float(p_bytes)
            if (not math.isfinite(val)) or val < min_required_bytes or val > max_allowed_bytes:
                topo_diffs.append(
                    f"collective_payload_out_of_bounds:obs={p_bytes},expected=[{min_required_bytes:.1f},{max_allowed_bytes:.1f}]"
                )
                return False
    return all_have_payload_and_group


def _has_cross_stream_overlap(obs_positive: Sequence[Mapping[str, Any]]) -> bool:
    """Return True iff two positive non-helper compute kernels on distinct streams overlap in time."""
    by_stream: dict[int, list[tuple[float, float]]] = {}
    for k in obs_positive:
        s_u = _safe_float(k.get("start_us", 0.0))
        d_u = _safe_float(k.get("duration_us", 0.0))
        if math.isfinite(s_u) and math.isfinite(d_u) and s_u > 0.0 and d_u > 0.0:
            by_stream.setdefault(_safe_int(k.get("stream_id", 0), 0), []).append((s_u, s_u + d_u))
    s_ids = list(by_stream.keys())
    return any(
        max(a0, b0) < min(a1, b1)
        for i, s1 in enumerate(s_ids) for s2 in s_ids[i + 1:]
        for a0, a1 in by_stream[s1] for b0, b1 in by_stream[s2]
    )


def _token_dim_matches(m: int, expected_tokens: int) -> bool:
    return (
        m == expected_tokens
        or (m % FRACTAL_NZ_TILE == 0 and m >= expected_tokens and m - expected_tokens < FRACTAL_NZ_TILE)
    )


def _check_attention_qkv_shapes(
    obs_positive: Sequence[Mapping[str, Any]],
    obs_cats: Counter[str],
    *,
    expected_tokens: int,
    expected_head_dims: frozenset[int],
    expected_kv_dims: frozenset[int],
    expected_local_heads: int,
    topo_diffs: list[str],
) -> None:
    """Validate Q, K, and V tensor shapes on executed attention / linear-attention kernels."""
    attn_tensors = _extract_observed_attn_tensors(obs_positive)
    if obs_cats.get("attention", 0) <= 0 and obs_cats.get("linear_attention", 0) <= 0:
        return
    if not attn_tensors or all(not ts for ts in attn_tensors):
        topo_diffs.append("missing_executed_attention_shapes")
        return
    if not expected_head_dims:
        return

    def _q_matches(t: tuple[int, ...]) -> bool:
        if len(t) < 3 or t[-1] not in expected_head_dims:
            return False
        if expected_local_heads > 1 and expected_local_heads not in t[:-1] and (obs_cats.get("attention", 0) > 0):
            if not any(d >= expected_local_heads and d % expected_local_heads == 0 for d in t[:-1]):
                return False
        if not expected_tokens:
            return True
        if len(t) == 3:
            return t[0] == expected_tokens or t[1] == expected_tokens
        return t[0] * t[1] == expected_tokens or t[0] * t[2] == expected_tokens or t[1] * t[2] == expected_tokens

    kv_allowed = expected_kv_dims or expected_head_dims

    def _kv_matches(t: tuple[int, ...]) -> bool:
        if len(t) < 3 or t[-1] not in kv_allowed:
            return False
        leading_elems = 1
        for d in t[:-1]:
            leading_elems *= d
        return leading_elems >= max(1, expected_tokens)

    is_full_attn = obs_cats.get("attention", 0) > 0
    if is_full_attn:
        attn_ok = any(
            len(ts) >= 3 and _q_matches(ts[0]) and _kv_matches(ts[1]) and _kv_matches(ts[2])
            for ts in attn_tensors
        )
    else:
        attn_ok = any(any(_q_matches(t) for t in ts) for ts in attn_tensors)
    if not attn_ok:
        topo_diffs.append(
            f"observed_attention_shape_mismatch:expected_tok={expected_tokens},heads={expected_local_heads},"
            f"head_dims={sorted(expected_head_dims)},kv_dims={sorted(kv_allowed)}"
        )


def _check_gemm_and_attn_shapes(
    obs_positive: Sequence[Mapping[str, Any]],
    sim_cats: Counter[str],
    obs_cats: Counter[str],
    *,
    expected_tokens: int,
    expected_hidden: int,
    expected_head_dims: frozenset[int],
    expected_kv_dims: frozenset[int],
    expected_gemm_widths: frozenset[int],
    expected_attn_gemm_widths: frozenset[int],
    expected_attn_in_widths: frozenset[int],
    expected_attn_o_widths: frozenset[int],
    expected_ffn_gemm_widths: frozenset[int],
    expected_ffn_gate_up_widths: frozenset[int],
    expected_ffn_down_widths: frozenset[int],
    is_moe: bool,
    expected_min_gemms: int,
    expected_local_heads: int,
    topo_diffs: list[str],
) -> bool:
    """Validate directed attn & FFN GEMM shapes/roles, FFN activation, and Q/K/V Attention shapes."""
    obs_gemms = _extract_observed_gemm_shapes(obs_positive)
    needs_gemm = (sim_cats.get("matmul", 0) > 0) or (sim_cats.get("moe", 0) > 0)
    joint_gemm_ok = False
    if needs_gemm:
        if not obs_gemms:
            topo_diffs.append("missing_executed_gemm_shapes")
        else:
            def _matches_widths(widths: frozenset[int]) -> list[tuple[int, int, int]]:
                return [
                    (m, k_d, n_d) for m, k_d, n_d in obs_gemms
                    if _token_dim_matches(m, expected_tokens)
                    and (
                        (k_d == expected_hidden and (n_d in widths if widths else n_d >= MIN_GEMM_INNER_DIM))
                        or (n_d == expected_hidden and (k_d in widths if widths else k_d >= MIN_GEMM_INNER_DIM))
                    )
                ]

            def _has_dir_in(widths: frozenset[int]) -> bool:
                return any(
                    _token_dim_matches(m, expected_tokens) and k_d == expected_hidden and n_d in widths
                    for m, k_d, n_d in obs_gemms
                )

            def _has_dir_out(widths: frozenset[int]) -> bool:
                return any(
                    _token_dim_matches(m, expected_tokens) and k_d in widths and n_d == expected_hidden
                    for m, k_d, n_d in obs_gemms
                )

            matched_gemms = _matches_widths(expected_gemm_widths)
            joint_gemm_ok = bool(matched_gemms)
            if expected_tokens > 0 and expected_hidden > 0 and not joint_gemm_ok:
                topo_diffs.append(
                    f"observed_gemm_joint_shape_mismatch:expected=(M={expected_tokens},H={expected_hidden}),obs={obs_gemms[:6]}"
                )
            if expected_tokens > 0 and expected_hidden > 0 and expected_attn_gemm_widths and not _matches_widths(expected_attn_gemm_widths):
                topo_diffs.append("missing_executed_attention_projection_gemms")
            if expected_tokens > 0 and expected_hidden > 0 and expected_attn_in_widths and not _has_dir_in(expected_attn_in_widths):
                topo_diffs.append("missing_executed_attention_in_projection_gemm")
            if expected_tokens > 0 and expected_hidden > 0 and expected_attn_o_widths and not _has_dir_out(expected_attn_o_widths):
                topo_diffs.append("missing_executed_attention_out_projection_gemm")
            if expected_tokens > 0 and expected_hidden > 0 and (not is_moe) and expected_ffn_gemm_widths:
                if not _matches_widths(expected_ffn_gemm_widths):
                    topo_diffs.append("missing_executed_ffn_projection_gemms")
                if expected_ffn_gate_up_widths and not _has_dir_in(expected_ffn_gate_up_widths):
                    topo_diffs.append("missing_executed_ffn_gate_up_gemm")
                if expected_ffn_down_widths and not _has_dir_out(expected_ffn_down_widths):
                    topo_diffs.append("missing_executed_ffn_down_gemm")
                has_ffn_act = any(
                    any(tok in str(k.get("type", "")).lower() for tok in ("swiglu", "silu", "gelu", "fastgelu"))
                    for k in obs_positive
                )
                if not has_ffn_act:
                    topo_diffs.append("missing_executed_ffn_activation_kernel")
            if expected_min_gemms > 0 and len(obs_gemms) < expected_min_gemms:
                topo_diffs.append(f"observed_gemm_count_insufficient:obs={len(obs_gemms)}<min={expected_min_gemms}")

    _check_attention_qkv_shapes(
        obs_positive, obs_cats,
        expected_tokens=expected_tokens, expected_head_dims=expected_head_dims,
        expected_kv_dims=expected_kv_dims, expected_local_heads=expected_local_heads,
        topo_diffs=topo_diffs,
    )
    return joint_gemm_ok


def _check_causal_stage_order(
    obs_positive: Sequence[Mapping[str, Any]],
    *,
    expected_tokens: int,
    expected_hidden: int,
    expected_ffn_gate_up_widths: frozenset[int],
    expected_ffn_down_widths: frozenset[int],
    is_moe: bool,
    topo_diffs: list[str],
    expected_attn_in_widths: frozenset[int] = frozenset(),
) -> None:
    """Enforce forward causal order inside the whole-layer trace (reject reversed layers)."""
    idx_pos: list[int] = []
    attn_pos: list[int] = []
    ffn_act_pos: list[int] = []
    ffn_gu_pos: list[int] = []
    ffn_gu_shared_pos: list[int] = []
    ffn_dn_pos: list[int] = []
    moe_pos: list[int] = []
    for i, k in enumerate(obs_positive):
        cat = _classify_csv_kernel(k)
        op_t = str(k.get("type", "")).lower()
        if cat == "indexer":
            idx_pos.append(i)
        elif cat in ("attention", "linear_attention"):
            attn_pos.append(i)
        elif cat == "moe":
            moe_pos.append(i)
        if any(tok in op_t for tok in ("swiglu", "silu", "gelu", "fastgelu")):
            ffn_act_pos.append(i)
        if (not is_moe) and cat == "matmul" and expected_hidden > 0:
            for m, k_d, n_d in _extract_observed_gemm_shapes([k]):
                if _token_dim_matches(m, expected_tokens):
                    if k_d == expected_hidden and n_d in expected_ffn_gate_up_widths:
                        if n_d in expected_attn_in_widths:
                            ffn_gu_shared_pos.append(i)
                        else:
                            ffn_gu_pos.append(i)
                    if k_d in expected_ffn_down_widths and n_d == expected_hidden:
                        ffn_dn_pos.append(i)
    if len(ffn_gu_shared_pos) >= 2:
        ffn_gu_pos.extend(ffn_gu_shared_pos[1:])
    if idx_pos and attn_pos and min(idx_pos) > min(attn_pos):
        topo_diffs.append("causal_order_violation:indexer_after_attention")
    ffn_markers = moe_pos if is_moe else (ffn_gu_pos + ffn_act_pos + ffn_dn_pos)
    if attn_pos and ffn_markers and max(attn_pos) > min(ffn_markers):
        topo_diffs.append("causal_order_violation:attention_after_ffn")
    if (not is_moe) and ffn_gu_pos and ffn_dn_pos and min(ffn_gu_pos) > max(ffn_dn_pos):
        topo_diffs.append("causal_order_violation:ffn_down_before_gate_up")


def _verify_topology_isomorphism(
    sim_ops: Sequence[Any],
    obs_kernels: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    expected_tokens: int,
    expected_hidden: int,
    expected_head_dims: frozenset[int] = frozenset(),
    expected_kv_dims: frozenset[int] = frozenset(),
    expected_gemm_widths: frozenset[int] = frozenset(),
    expected_attn_gemm_widths: frozenset[int] = frozenset(),
    expected_attn_in_widths: frozenset[int] = frozenset(),
    expected_attn_o_widths: frozenset[int] = frozenset(),
    expected_ffn_gemm_widths: frozenset[int] = frozenset(),
    expected_ffn_gate_up_widths: frozenset[int] = frozenset(),
    expected_ffn_down_widths: frozenset[int] = frozenset(),
    is_moe: bool = False,
    expected_min_gemms: int = 0,
    expected_local_heads: int = 0,
    quant_mode: str = "bf16",
) -> dict[str, Any]:
    """Strict 1:1 topology check over positive non-helper kernels, directed comms, shapes, order, and overlap."""
    sim_cat_seq = [_classify_sim_op(op) for op in sim_ops]
    obs_positive = sorted(
        (
            k for k in obs_kernels
            if math.isfinite(_safe_float(k.get("duration_us", 0.0)))
            and _safe_float(k.get("duration_us", 0.0)) > 0.0
            and _classify_csv_kernel(k) != "control_helper"
        ),
        key=lambda k: _safe_float(k.get("start_us", 0.0), 0.0),
    )
    obs_cat_seq = [_classify_csv_kernel(k) for k in obs_positive]
    sim_cats: Counter[str] = Counter(sim_cat_seq)
    obs_cats: Counter[str] = Counter(obs_cat_seq)

    core_cats = (
        ("attention", "indexer", "linear_attention", "moe", "collective")
        if world_size > 1
        else ("attention", "indexer", "linear_attention", "moe")
    )
    topo_diffs: list[str] = []
    if not obs_positive:
        topo_diffs.append("empty_positive_compute_kernels")
    for cat in core_cats:
        if (sim_cats.get(cat, 0) > 0) != (obs_cats.get(cat, 0) > 0):
            topo_diffs.append(f"category_mismatch:{cat}:sim={sim_cats.get(cat, 0)},obs={obs_cats.get(cat, 0)}")

    sim_phase_seq = _collapse_phase_order(sim_cat_seq)
    obs_phase_seq = _collapse_phase_order(obs_cat_seq)
    if sim_phase_seq and obs_phase_seq:
        for earlier, later in (("indexer", "attention"), ("attention", "moe"), ("linear_attention", "moe")):
            if earlier in sim_phase_seq and later in sim_phase_seq and earlier in obs_phase_seq and later in obs_phase_seq:
                if (sim_phase_seq.index(earlier) < sim_phase_seq.index(later)) != (
                    obs_phase_seq.index(earlier) < obs_phase_seq.index(later)
                ):
                    topo_diffs.append(f"phase_order_mismatch:{earlier}_vs_{later}:sim={sim_phase_seq},obs={obs_phase_seq}")

    _check_causal_stage_order(
        obs_positive,
        expected_tokens=expected_tokens, expected_hidden=expected_hidden,
        expected_ffn_gate_up_widths=expected_ffn_gate_up_widths,
        expected_ffn_down_widths=expected_ffn_down_widths,
        is_moe=is_moe, topo_diffs=topo_diffs,
        expected_attn_in_widths=expected_attn_in_widths,
    )

    sim_comm_kinds: Counter[str] = Counter(
        k for op in sim_ops if (k := _classify_collective_kind_sim(op)) is not None
    )
    seen_gmm = False
    obs_comm_list: list[str] = []
    for row in obs_positive:
        if "groupedmatmul" in str(row.get("type", "")).lower():
            seen_gmm = True
        ck = _classify_collective_kind_csv(row, after_gmm=seen_gmm)
        if ck is not None:
            obs_comm_list.append(ck)
    obs_comm_kinds: Counter[str] = Counter(obs_comm_list)
    comm_payloads_verified = False
    if world_size > 1:
        if sim_comm_kinds != obs_comm_kinds:
            topo_diffs.append(f"collective_primitive_mismatch:sim={dict(sim_comm_kinds)},obs={dict(obs_comm_kinds)}")
        comm_payloads_verified = _check_collective_payloads(sim_ops, obs_positive, world_size, topo_diffs)

    joint_gemm_ok = _check_gemm_and_attn_shapes(
        obs_positive, sim_cats, obs_cats,
        expected_tokens=expected_tokens, expected_hidden=expected_hidden,
        expected_head_dims=expected_head_dims, expected_kv_dims=expected_kv_dims,
        expected_gemm_widths=expected_gemm_widths,
        expected_attn_gemm_widths=expected_attn_gemm_widths,
        expected_attn_in_widths=expected_attn_in_widths,
        expected_attn_o_widths=expected_attn_o_widths,
        expected_ffn_gemm_widths=expected_ffn_gemm_widths,
        expected_ffn_gate_up_widths=expected_ffn_gate_up_widths,
        expected_ffn_down_widths=expected_ffn_down_widths,
        is_moe=is_moe,
        expected_min_gemms=expected_min_gemms, expected_local_heads=expected_local_heads,
        topo_diffs=topo_diffs,
    )

    if quant_mode in ("w8a8", "c8", "mxfp8", "mxfp4"):
        target_toks = ("int8",) if quant_mode in ("w8a8", "c8") else (("float4", "fp4") if quant_mode == "mxfp4" else ("fp8", "float8"))
        quant_rows = [
            str(k.get("input_dtypes", "")).lower()
            for k in obs_positive
            if _classify_csv_kernel(k) in ("matmul", "moe")
        ]
        if not any(any(t in dt for t in target_toks) for dt in quant_rows):
            topo_diffs.append(f"quant_dtype_evidence_missing:mode={quant_mode}")

    obs_streams = len({_safe_int(k.get("stream_id", 0), 0) for k in obs_positive})
    sim_streams = len({str(getattr(op, "stream", "main")) for op in sim_ops})
    if world_size > 1 and sim_streams >= 2:
        if obs_streams < 2 or not _has_cross_stream_overlap(obs_positive):
            topo_diffs.append(f"stream_concurrency_mismatch:sim={sim_streams},obs={obs_streams}")

    return {
        "topology_isomorphic": not topo_diffs,
        "topology_diffs": topo_diffs,
        "sim_categories": dict(sim_cats),
        "obs_categories": dict(obs_cats),
        "sim_phase_sequence": sim_phase_seq,
        "obs_phase_sequence": obs_phase_seq,
        "sim_collective_kinds": dict(sim_comm_kinds),
        "obs_collective_kinds": dict(obs_comm_kinds),
        "collectives_verified": bool(
            world_size > 1 and sim_comm_kinds and sim_comm_kinds == obs_comm_kinds
            and comm_payloads_verified and not any(d.startswith("collective_") for d in topo_diffs)
        ),
        "observed_gemm_shapes_verified": joint_gemm_ok,
        "sim_stream_count": sim_streams,
        "obs_stream_count": obs_streams,
    }


def _build_capture_geom_meta(
    spec: Any,
    target_plan: Any,
    *,
    local_tokens: int,
    tp: int,
    o_proj_tp: int,
    lin_tp: int,
    quant_mode: str,
    target_local_idx: int,
) -> dict[str, Any]:
    """Derive directed GEMM width sets and attention Q/K/V dimension invariants for one capture layer."""
    head_dims = frozenset(
        d for d in (
            int(target_plan.head_dim or spec.head_dim or 0),
            int(target_plan.kv_lora_rank or 0),
            int(target_plan.qk_nope_head_dim or 0) + int(target_plan.qk_rope_head_dim or 0),
            int(target_plan.v_head_dim or 0),
            int(target_plan.linear_head_dim or 0),
            int(target_plan.linear_key_head_dim or 0),
        )
        if d > 0
    )
    local_heads = max(1, int(target_plan.num_heads or spec.num_heads) // tp)
    raw_kv = max(1, int(target_plan.num_kv_heads or spec.num_kv_heads or 1))
    local_kv = max(1, raw_kv // tp) if raw_kv >= tp else 1
    hd = int(target_plan.head_dim or spec.head_dim or DEFAULT_KV_BLOCK_SIZE)
    inter_loc = max(1, int(target_plan.intermediate_size or spec.intermediate_size) // tp)
    q_mult = 2 if bool(target_plan.q_output_gate) else 1
    kv_lora = int(target_plan.kv_lora_rank or 0)
    qk_rope = int(target_plan.qk_rope_head_dim or 0)
    qk_nope = int(target_plan.qk_nope_head_dim or 0)
    v_dim = int(target_plan.v_head_dim or hd)
    o_div, lin_div = max(tp, o_proj_tp), max(tp, lin_tp)
    nk = max(1, int(target_plan.linear_key_heads or target_plan.linear_heads or 1) // lin_div)
    nv = max(1, int(target_plan.linear_value_heads or target_plan.linear_heads or 1) // lin_div)
    dk = int(target_plan.linear_key_head_dim or target_plan.linear_head_dim or DEFAULT_KV_BLOCK_SIZE)
    dv = int(target_plan.linear_head_dim or DEFAULT_KV_BLOCK_SIZE)
    kv_dims = frozenset(d for d in (hd, v_dim, local_kv * hd, local_kv * v_dim, kv_lora, kv_lora + qk_rope, qk_nope) if d > 0)
    expected_attn_in_widths = frozenset(
        w for w in (
            q_mult * local_heads * hd + 2 * local_kv * hd, local_heads * hd,
            int(target_plan.q_lora_rank or 0), kv_lora + qk_rope,
            local_heads * (qk_nope + qk_rope), 2 * nk * dk + nv * dv,
        )
        if w > 0
    )
    expected_attn_o_widths = frozenset(
        w for w in (local_heads * hd, max(1, (int(target_plan.num_heads or spec.num_heads) * v_dim) // o_div), nv * dv)
        if w > 0
    )
    expected_attn_widths = expected_attn_in_widths | expected_attn_o_widths | frozenset(
        w for w in (local_heads * (qk_nope + v_dim), int(target_plan.index_heads or 0) * int(target_plan.index_dim or 0))
        if w > 0
    )
    expected_ffn_gate_up_widths = frozenset(w for w in (2 * inter_loc,) if w > 0) if target_plan.ffn != "moe" else frozenset()
    expected_ffn_down_widths = frozenset(w for w in (inter_loc,) if w > 0) if target_plan.ffn != "moe" else frozenset()
    sh_dim = int(target_plan.shared_intermediate_size or target_plan.moe_intermediate_size or 0)
    sh_n = int(target_plan.num_shared_experts or 0)
    expected_ffn_widths = frozenset(w for w in (2 * inter_loc, inter_loc, int(target_plan.num_experts or 0), 2 * sh_n * sh_dim, sh_n * sh_dim) if w > 0)
    return {
        "expected_tokens": local_tokens, "expected_hidden": int(spec.hidden_size),
        "expected_head_dims": head_dims, "expected_kv_dims": kv_dims,
        "expected_gemm_widths": expected_attn_widths | expected_ffn_widths,
        "expected_attn_gemm_widths": expected_attn_widths,
        "expected_attn_in_widths": expected_attn_in_widths,
        "expected_attn_o_widths": expected_attn_o_widths,
        "expected_ffn_gemm_widths": expected_ffn_widths,
        "expected_ffn_gate_up_widths": expected_ffn_gate_up_widths,
        "expected_ffn_down_widths": expected_ffn_down_widths,
        "expected_min_gemms": 4 if target_plan.ffn == "dense" else 2,
        "expected_local_heads": local_heads if target_plan.attention in ("gqa", "mla") else 0,
        "kv_a_out_dim": kv_lora + qk_rope if target_plan.attention == "mla" else 0,
        "q_lora_rank": int(target_plan.q_lora_rank or 0) if target_plan.attention == "mla" else 0,
        "model_plan_attention": str(target_plan.attention),
        "model_plan_architecture": str(target_plan.architecture),
        "model_plan_ffn": str(target_plan.ffn),
        "quant_mode": quant_mode, "is_moe": target_plan.ffn == "moe",
        "target_layer_prefix": f"s0.l{target_local_idx}.",
    }


_PRIMITIVE_TABLE_CACHE: dict[str, Any] = {}


def _load_calibrated_primitive_table(hardware_soc: str) -> Any:
    """Load calibrated L0 ``PrimitiveTable`` for ``hardware_soc`` if registered (cached)."""
    key = str(hardware_soc or "").strip()
    if not key:
        return None
    if key in _PRIMITIVE_TABLE_CACHE:
        return _PRIMITIVE_TABLE_CACHE[key]
    try:
        table = load_primitives(key)
    except (ModelingError, FileNotFoundError, KeyError, ValueError):
        table = None
    _PRIMITIVE_TABLE_CACHE[key] = table
    return table


def _lower_capture_sim_step(
    capture_case: Mapping[str, Any],
    hardware_soc: str,
    spine: PredictionSpine,
    world_size: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Compose, lower, and replay the exact body-layer ``mindie_sim`` graph matching a capture."""
    del world_size
    model_key = str(capture_case["model"])
    par = dict(capture_case.get("parallel", {}))
    wl = dict(capture_case.get("workload", {}))
    phase = str(capture_case.get("phase", "decode"))
    dp = _safe_int(par.get("dp", 1), -1)
    tp = _safe_int(par.get("tp", 1), -1)
    ep = _safe_int(par.get("ep", 1), -1)
    dcp = _safe_int(par.get("dcp", 1), -1)
    pcp = _safe_int(par.get("pcp", 1), -1)
    o_proj_tp = _safe_int(par.get("o_proj_tp", 1), -1)
    lin_tp = _safe_int(par.get("linear_proj_tp", 1), -1)
    layer_idx = _safe_int(capture_case.get("layer_index", 0), -1)
    quant_mode = str(wl.get("quant_mode", "bf16"))
    seqs = _safe_int(wl.get("sequences", LIVE_DECODE_SEQS), -1)
    active_seqs = _safe_int(wl.get("active_sequences", seqs), -1)
    q_len = _safe_int(wl.get("q_len", 1), -1)
    context = _safe_int(wl.get("context", LIVE_DECODE_CONTEXT), -1)
    if min(dp, tp, ep, dcp, pcp, o_proj_tp, lin_tp, seqs, q_len) < 1 or min(layer_idx, active_seqs, context) < 0:
        raise ValueError("invalid_non_integer_capture_parallel_or_workload_field")
    is_padded_batch = active_seqs < seqs

    spec = load_model(model_key)
    sliced_spec, target_local_idx = _build_layer_slice_spec(spec, layer_idx)
    target_plan = spec.layer_plans[layer_idx]
    eff_ep = ep if target_plan.ffn == "moe" else 1
    dp_eff = max(dp, math.ceil(eff_ep / max(1, tp * pcp))) if eff_ep > 1 else dp
    quant_arg: Any = C8_QUANT_SPEC if quant_mode == "c8" else quant_mode
    compose_kw: dict[str, Any] = dict(
        hardware=hardware_soc, dp=dp_eff, tp=tp, ep=eff_ep,
        dcp=dcp, pcp=pcp, num_devices=dp_eff * tp * pcp, quant=quant_arg,
    )
    overlay_low = str(capture_case.get("overlay", "")).lower()
    if is_padded_batch:
        compose_kw["graph_capture_mode"] = "none" if "eager" in overlay_low else "full"
        if dp_eff > 1 and eff_ep > tp:
            compose_kw["sp"] = False
    if o_proj_tp > 1 and tp == 1 and dcp == 1:
        compose_kw["o_proj_tp"] = o_proj_tp
    if lin_tp > 1 and tp == 1 and dcp == 1:
        compose_kw["linear_proj_tp"] = lin_tp

    raw = compose_case(sliced_spec.to_dict(), **compose_kw)
    parsed = parse_case(raw)
    if is_padded_batch:
        active_ctx = max(0, context - q_len)
        views = tuple(
            RequestView(
                request_id=f"r{i}", q_len=q_len, context=active_ctx if i < active_seqs else 0,
                samples=i < active_seqs, phase=phase, padding=i >= active_seqs,
            )
            for i in range(seqs)
        )
        iter_plan = IterationPlan(requests=views, iteration=0)
        req = LoweringRequest(iterations=tuple((r, iter_plan) for r in range(dp_eff)))
    else:
        # RequestView.kv_len = context + q_len; subtract q_len on decode so sim kv_len == live context exactly
        sim_ctx = max(0, context - q_len) if phase == "decode" else context
        req = LoweringRequest(step=StepRequest(phase=phase, tokens=q_len, context=sim_ctx, sequences=seqs, scope="body"))
    step = spine.lower(parsed, req)
    pt = _load_calibrated_primitive_table(hardware_soc)
    ev = spine.run_step(step, ReplayOptions(include_submission=False), primitive_table=pt)
    local_tokens = max(1, (seqs * q_len) // pcp) if phase == "prefill" else (seqs * q_len)
    geom_meta = _build_capture_geom_meta(
        spec, target_plan, local_tokens=local_tokens, tp=tp, o_proj_tp=o_proj_tp,
        lin_tp=lin_tp, quant_mode=quant_mode, target_local_idx=target_local_idx,
    )
    return step, ev, geom_meta


def _classify_sim_fine_bucket(op: Any, block: str) -> str:
    """Assign one lowered body-layer ``OpSpec`` to a fine-grained non-cancelling L1 bucket."""
    cat = _classify_sim_op(op)
    oid = str(op.op_id).lower()
    fam = str(op.family).lower()
    if block == "attention":
        if cat == "collective":
            return "attn_collective"
        if cat in ("attention", "indexer", "linear_attention"):
            return "attn_core" if cat == "attention" else cat
        if cat == "norm_quant_rope":
            if "rotary" in fam or "rope" in oid or "rotary" in oid:
                return "attn_rope"
            if "quant" in fam or "quant" in oid:
                return "attn_quant"
            return "attn_norm"
        if cat == "matmul":
            if any(t in oid for t in (".output", "o_proj", "v_expand", "linear_output", "grouped_output")) or fam == "output_projection":
                return "attn_o_proj"
            if any(t in oid for t in ("kv_a", "kv_b", "kv_proj", "kv_compress", "k_proj", "v_proj")):
                return "attn_kv_proj"
            if any(t in oid for t in ("absorb", "w_uk", "w_uv", "bmm")):
                return "attn_bmm_absorb"
            if "q_a" in oid:
                return "attn_q_a_proj"
            if "q_b" in oid:
                return "attn_q_b_proj"
            if any(t in oid for t in (".hc_", "engram_")):
                return "attn_other"
            return "attn_q_proj"
        return "attn_other"
    if cat == "collective":
        return "ffn_collective"
    if any(t in oid for t in ("shared.", "latent_down", "latent_up")) and cat == "matmul":
        return "ffn_shared"
    if "router" in oid or fam == "router":
        return "ffn_router"
    if fam == "grouped_gemm" or "expert_up" in oid or "expert_down" in oid:
        return "moe_gmm_down" if ("down" in oid or "w2" in oid) else "moe_gmm_gate_up"
    if cat == "moe":
        return "moe_routing"
    if cat == "norm_quant_rope":
        if "swiglu" in fam or "swiglu" in oid or "act" in oid:
            return "ffn_swiglu"
        if "quant" in fam or "quant" in oid:
            return "ffn_quant"
        return "ffn_norm"
    if cat == "matmul":
        if any(t in oid for t in (".hc_", "engram_")):
            return "ffn_other"
        if "down" in oid or fam == "down_projection":
            return "ffn_down"
        return "ffn_gate_up"
    return "ffn_other"


def _bucket_obs_fine_us(
    attn_kernels: Sequence[Mapping[str, Any]],
    ffn_kernels: Sequence[Mapping[str, Any]],
    *,
    is_moe: bool,
    kv_a_out_dim: int = 0,
    q_lora_rank: int = 0,
) -> dict[str, float]:
    """Bucket observed hardware kernels into fine-grained non-cancelling L1 sub-blocks."""
    out: dict[str, float] = {}
    a_rows = sorted(attn_kernels, key=lambda k: _safe_float(k.get("start_us", 0.0), 0.0))
    core_indices = [i for i, k in enumerate(a_rows) if _classify_csv_kernel(k) in ("attention", "linear_attention")]
    core_idx = core_indices[0] if core_indices else len(a_rows)
    for idx, k in enumerate(a_rows):
        dur = _safe_float(k.get("duration_us", 0.0), 0.0)
        if dur <= 0.0:
            continue
        cat = _classify_csv_kernel(k)
        if cat == "control_helper":
            continue
        op_t = str(k.get("type", "")).lower()
        op_n = str(k.get("name", "")).lower()
        if cat == "collective":
            b = "attn_collective"
        elif cat == "attention":
            b = "attn_core"
        elif cat in ("indexer", "linear_attention"):
            b = cat
        elif cat == "norm_quant_rope":
            if "rotary" in op_t or "rope" in op_t:
                b = "attn_rope"
            elif "quant" in op_t:
                b = "attn_quant"
            else:
                b = "attn_norm"
        elif cat == "matmul":
            if idx > core_idx:
                b = "attn_bmm_absorb" if "batchmatmul" in op_t else "attn_o_proj"
            elif "batchmatmul" in op_t:
                b = "attn_bmm_absorb"
            else:
                shapes = _extract_observed_gemm_shapes([k])
                is_kv_gemm = any(tok in op_n for tok in ("kv_a", "kv_b", "kv_proj", "k_proj", "v_proj")) or (
                    kv_a_out_dim > 0 and any(n_d == kv_a_out_dim for _, _, n_d in shapes)
                )
                if is_kv_gemm:
                    b = "attn_kv_proj"
                elif q_lora_rank > 0:
                    if "q_a" in op_n or any(n_d == q_lora_rank for _, _, n_d in shapes):
                        b = "attn_q_a_proj"
                    elif "q_b" in op_n or any(k_d == q_lora_rank for _, k_d, _ in shapes):
                        b = "attn_q_b_proj"
                    else:
                        b = "attn_q_proj"
                else:
                    b = "attn_q_proj"
        else:
            b = "attn_other"
        out[b] = out.get(b, 0.0) + dur

    f_rows = sorted(ffn_kernels, key=lambda k: _safe_float(k.get("start_us", 0.0), 0.0))
    swiglu_indices = [i for i, k in enumerate(f_rows) if "swiglu" in str(k.get("type", "")).lower()]
    first_swiglu = swiglu_indices[0] if swiglu_indices else len(f_rows)
    gmm_indices = [i for i, k in enumerate(f_rows) if "groupedmatmul" in str(k.get("type", "")).lower()]
    gmm_last = gmm_indices[-1] if gmm_indices else -1
    for idx, k in enumerate(f_rows):
        dur = _safe_float(k.get("duration_us", 0.0), 0.0)
        if dur <= 0.0:
            continue
        cat = _classify_csv_kernel(k)
        if cat == "control_helper":
            continue
        op_t = str(k.get("type", "")).lower()
        if cat == "collective":
            b = "ffn_collective"
        elif is_moe and "groupedmatmul" in op_t:
            b = "moe_gmm_down" if idx > first_swiglu else "moe_gmm_gate_up"
        elif cat == "moe":
            b = "moe_routing"
        elif cat == "norm_quant_rope":
            if "swiglu" in op_t:
                b = "ffn_swiglu"
            elif "quant" in op_t:
                b = "ffn_quant"
            else:
                b = "ffn_norm"
        elif cat == "matmul":
            if is_moe:
                b = "ffn_shared" if (gmm_last >= 0 and idx > gmm_last) else "ffn_router"
            else:
                b = "ffn_down" if idx > first_swiglu else "ffn_gate_up"
        else:
            b = "ffn_other"
        out[b] = out.get(b, 0.0) + dur
    return out


def _merge_sim_interval_union(ivs: Sequence[tuple[float, float]]) -> float:
    """Compute exact union duration over simulated ``(start_us, end_us)`` intervals."""
    if not ivs:
        return 0.0
    ordered = sorted(ivs)
    lo, hi = ordered[0]
    total = 0.0
    for s_u, e_u in ordered[1:]:
        if s_u <= hi:
            hi = max(hi, e_u)
        else:
            total += hi - lo
            lo, hi = s_u, e_u
    total += hi - lo
    return float(total)


def _extract_sim_block_spans_and_buckets(
    sim_ops: Sequence[Any],
    ev: Any,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Compute critical-path elapsed spans and fine-grained L1 buckets for body-layer ops."""
    op_by_id = {op.op_id: op for op in sim_ops}
    sim_fine_us: dict[str, float] = {}
    block_intervals: dict[str, list[tuple[float, float]]] = {"attention": [], "ffn": [], "layer": []}
    block_iso_sum: dict[str, float] = {"attention": 0.0, "ffn": 0.0, "layer": 0.0}
    sim_excluded_comm_us = 0.0

    for row in ev.operators:
        if row.occurrence.replica != 0:
            continue
        op = op_by_id.get(row.occurrence.op_id)
        if op is None:
            continue
        dur_us = float(row.isolated_ms or 0.0) * MS_TO_US
        if dur_us <= 0.0 and row.start_ms is not None and row.end_ms is not None:
            dur_us = max(0.0, float(row.end_ms - row.start_ms) * MS_TO_US)
        cat = _classify_sim_op(op)
        if world_size == 1 and cat == "collective":
            sim_excluded_comm_us += dur_us
            continue
        blk_meta = str(dict(op.metadata).get("structure_block", "")).lower() if op.metadata else ""
        if blk_meta == "attention" or (blk_meta != "ffn" and (cat in ("attention", "indexer", "linear_attention") or any(
            k in op.op_id.lower() for k in ("q_", "kv_", "o_", "rotary", "mla", "gdn", "kda", "attn")
        ))):
            block = "attention"
        else:
            block = "ffn"
        bucket = _classify_sim_fine_bucket(op, block)
        sim_fine_us[bucket] = sim_fine_us.get(bucket, 0.0) + dur_us
        block_iso_sum[block] += dur_us
        block_iso_sum["layer"] += dur_us
        if row.start_ms is not None and row.end_ms is not None and float(row.end_ms) > float(row.start_ms):
            iv = (float(row.start_ms) * MS_TO_US, float(row.end_ms) * MS_TO_US)
            block_intervals[block].append(iv)
            block_intervals["layer"].append(iv)

    def _span(name: str) -> float:
        ivs = block_intervals[name]
        if ivs and world_size > 1:
            return max(e for _, e in ivs) - min(s for s, _ in ivs)
        if ivs:
            return _merge_sim_interval_union(ivs)
        return block_iso_sum[name]

    return {
        "sim_attn_span_us": _span("attention"),
        "sim_ffn_span_us": _span("ffn"),
        "sim_layer_span_us": _span("layer"),
        "sim_attn_iso_us": block_iso_sum["attention"],
        "sim_ffn_iso_us": block_iso_sum["ffn"],
        "sim_layer_iso_us": block_iso_sum["layer"],
        "sim_fine_us": sim_fine_us,
        "sim_excluded_comm_us": sim_excluded_comm_us,
    }


def _l1_cliff_limit_us(obs_us: float) -> float:
    return max((L1_CLIFF_TOL_PCT / PERCENT_SCALE) * max(obs_us, 0.0), L1_CLIFF_ATOL_US)


def _quantile_p90(values: Sequence[float]) -> float:
    return nearest_rank_p90(values) if values else 0.0


def _summarize_l1_two_tier(l1_in_l2: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize L1-in-L2 under Two-Tier Law: Median <= 8%, P90 <= 12%, cliff <= max(15%, 3.0 us)."""
    total = len(l1_in_l2)
    if total == 0:
        return {
            "l1_cliff_pass_count": 0, "l1_cliff_pass_rate": 0.0,
            "l1_median_ape_pct": None, "l1_p90_ape_pct": None, "l1_two_tier_pass": False,
        }
    cliff_pass = sum(1 for sc in l1_in_l2.values() if sc.get("pass_l1_cliff_tol"))
    apes = [abs(float(sc.get("signed_err_pct") if sc.get("signed_err_pct") is not None else PERCENT_SCALE)) for sc in l1_in_l2.values()]
    raw_med = float(statistics.median(apes))
    raw_p90 = _quantile_p90(apes)
    return {
        "l1_cliff_pass_count": cliff_pass,
        "l1_cliff_pass_rate": round(cliff_pass / total, 4),
        "l1_median_ape_pct": round(raw_med, 2),
        "l1_p90_ape_pct": round(raw_p90, 2),
        "l1_two_tier_pass": bool(
            cliff_pass == total and raw_med <= L1_MEDIAN_APE_LIMIT_PCT and raw_p90 <= L1_P90_APE_LIMIT_PCT
        ),
    }


def _score_l1_in_l2_buckets(
    sim_fine_us: Mapping[str, float],
    obs_fine_us: Mapping[str, float],
) -> tuple[dict[str, Any], int, int, dict[str, Any] | None, float]:
    """Score every fine-grained L1-in-L2 bucket under both 10%/2us diagnostic & Two-Tier 15%/3us cliff."""
    l1_in_l2: dict[str, Any] = {}
    abs_err_sum = 0.0
    for bucket in sorted(set(sim_fine_us) | set(obs_fine_us)):
        p_u, o_u = sim_fine_us.get(bucket, 0.0), obs_fine_us.get(bucket, 0.0)
        if p_u <= 0.0 and o_u <= 0.0:
            continue
        raw_cliff_lim = _l1_cliff_limit_us(max(o_u, 0.0))
        cliff_lim = round(raw_cliff_lim, 3)
        if o_u > 0.0 and p_u > 0.0:
            sc = _score_latency_pair(p_u, o_u)
            sc["l1_cliff_limit_us"] = cliff_lim
            sc["pass_l1_cliff_tol"] = abs(p_u - o_u) <= raw_cliff_lim
        else:
            sc = {
                "pred_us": round(p_u, 3), "obs_us": round(o_u, 3),
                "signed_err_pct": PERCENT_SCALE if p_u > o_u else -PERCENT_SCALE,
                "limit_us": round(_physical_limit_us(max(o_u, 0.0)), 3),
                "l1_cliff_limit_us": cliff_lim,
                "pass_physical_tol": False,
                "pass_l1_cliff_tol": False,
                "unmatched_side": True,
            }
        l1_in_l2[bucket] = sc
        abs_err_sum += abs(p_u - o_u)
    pass_cnt = sum(1 for sc in l1_in_l2.values() if sc["pass_physical_tol"])
    total_cnt = len(l1_in_l2)
    def _outlier_rank(item: dict[str, Any]) -> tuple[int, float, float]:
        failed = 0 if item.get("pass_l1_cliff_tol") else 1
        p_val = float(item.get("pred_us") or 0.0)
        o_val = float(item.get("obs_us") or 0.0)
        lim = max(float(item.get("l1_cliff_limit_us") or L1_CLIFF_ATOL_US), 1e-6)
        return (failed, abs(p_val - o_val) / lim, abs(float(item.get("signed_err_pct") or 0.0)))

    worst = max(({"bucket": k, **v} for k, v in l1_in_l2.items()), key=_outlier_rank, default=None)
    return l1_in_l2, pass_cnt, total_cnt, worst, abs_err_sum


def _reconcile_whole_layer_windows(
    attn_cap: Mapping[str, Any],
    ffn_cap: Mapping[str, Any],
    layer_cap: Mapping[str, Any],
    attn_win: Mapping[str, Any],
    ffn_win: Mapping[str, Any],
    layer_win: Mapping[str, Any],
    geom_meta: Mapping[str, Any] | None = None,
) -> bool:
    """Verify whole-layer window structurally contains BOTH Attention & FFN compute in forward order."""
    sub_compute = float(attn_win["compute_sum_us"]) + float(ffn_win["compute_sum_us"])
    layer_compute = float(layer_win["compute_sum_us"])
    if sub_compute <= 0.0 or layer_compute <= 0.0:
        return False
    if not (LAYER_WINDOW_MIN_CONSISTENCY * sub_compute <= layer_compute <= LAYER_WINDOW_MAX_CONSISTENCY * sub_compute):
        return False
    if float(layer_win.get("core_compute_sum_us", 0.0)) < LAYER_CORE_COMPUTE_MIN_RATIO * layer_compute:
        return False

    def _mm_moe_count(cap: Mapping[str, Any]) -> int:
        return sum(
            1 for k in (cap.get("kernels") or [])
            if _safe_float(k.get("duration_us", 0.0), 0.0) > 0.0
            and _classify_csv_kernel(k) in ("matmul", "moe")
        )

    def _semantic_tags(cap: Mapping[str, Any]) -> set[str]:
        tags: set[str] = set()
        for k in (cap.get("kernels") or []):
            if _safe_float(k.get("duration_us", 0.0), 0.0) <= 0.0:
                continue
            cat = _classify_csv_kernel(k)
            if cat in ("attention", "linear_attention", "indexer", "moe"):
                tags.add(cat)
            if any(tok in str(k.get("type", "")).lower() for tok in ("swiglu", "silu", "gelu", "fastgelu")):
                tags.add("ffn_activation")
        return tags

    sub_tags = _semantic_tags(attn_cap) | _semantic_tags(ffn_cap)
    layer_tags = _semantic_tags(layer_cap)
    if not sub_tags.issubset(layer_tags):
        return False
    if geom_meta is not None and not bool(geom_meta.get("is_moe", False)):
        exp_h = int(geom_meta.get("expected_hidden", 0))
        exp_t = int(geom_meta.get("expected_tokens", 0))
        dn_widths = frozenset(geom_meta.get("expected_ffn_down_widths", ()))
        if exp_h > 0 and exp_t > 0 and dn_widths:
            ffn_gemms = _extract_observed_gemm_shapes(list(ffn_cap.get("kernels") or []))
            if not any(_token_dim_matches(m, exp_t) and k_d in dn_widths and n_d == exp_h for m, k_d, n_d in ffn_gemms):
                return False
    return _mm_moe_count(layer_cap) >= (_mm_moe_count(attn_cap) + _mm_moe_count(ffn_cap))


def _check_window_clocks_consistent(
    attn_cap: Mapping[str, Any],
    ffn_cap: Mapping[str, Any],
    layer_cap: Mapping[str, Any],
    rank_freq_min: float,
) -> tuple[bool, str]:
    """Require valid clocks >= 50 MHz on ALL 3 windows and within 20% across windows & rank receipts."""
    recorded: list[float] = []
    for label, cap in (("attention", attn_cap), ("ffn", ffn_cap), ("layer", layer_cap)):
        if "observed_aicore_freq_mhz" not in cap or cap["observed_aicore_freq_mhz"] is None:
            return False, f"window_{label}_missing_clock"
        val = _safe_float(cap["observed_aicore_freq_mhz"])
        if (not math.isfinite(val)) or val < MIN_PLAUSIBLE_AICORE_FREQ_MHZ:
            return False, f"window_{label}_invalid_clock:{cap['observed_aicore_freq_mhz']}"
        recorded.append(val)
    if rank_freq_min < MIN_PLAUSIBLE_AICORE_FREQ_MHZ:
        return False, f"rank_receipt_invalid_clock:{rank_freq_min}"
    all_clocks = [*recorded, rank_freq_min]
    if min(all_clocks) / max(all_clocks) < CLOCK_CONSISTENCY_MIN_RATIO:
        return False, f"window_clock_dvfs_drift:min={min(all_clocks):.1f},max={max(all_clocks):.1f}"
    return True, ""


def _check_capture_model_metadata(
    capture_case: Mapping[str, Any],
    geom_meta: Mapping[str, Any],
) -> str | None:
    """Verify capture architecture/attention/ffn labels match the catalog LayerPlan."""
    for field_k, meta_k in (
        ("attention", "model_plan_attention"),
        ("architecture", "model_plan_architecture"),
        ("ffn", "model_plan_ffn"),
    ):
        decl = str(capture_case.get(field_k, "")).strip()
        if decl and decl != str(geom_meta[meta_k]):
            return _sanitize_diagnostic(f"capture_{field_k}_mismatch:{decl}!={geom_meta[meta_k]}")
    return None


def _check_capture_tier_and_status(
    capture_case: Mapping[str, Any],
    hardware_soc: str,
    world_size: int,
    bundle_runner_mode: str = "",
) -> dict[str, Any] | None:
    """Validate model identity, strict coverage tier provenance, and peer error escalation before replay."""
    tier = str(capture_case.get("coverage_tier", TIER_UNMERGED_ABSENT))
    status = str(capture_case.get("status", "absent_custom_op"))
    model_key = str(capture_case.get("model", ""))
    runner_mode = str(capture_case.get("runner_mode") or bundle_runner_mode or "").strip()
    base = {
        "case_id": capture_case.get("case_id"), "model": model_key,
        "hardware_soc": hardware_soc, "world_size": world_size,
        "coverage_tier": tier, "pass_full_case": False,
    }
    if model_key not in ALL_CATALOG_MODEL_SET:
        return {**base, "status": "invalid_capture", "error": _sanitize_diagnostic(f"unknown_catalog_model:{model_key}")}
    if tier not in VALID_COVERAGE_TIERS:
        return {**base, "status": "invalid_capture", "error": _sanitize_diagnostic(f"unknown_coverage_tier:{tier}")}
    if tier == TIER_VLLM_1LAYER:
        vllm_cls = str(capture_case.get("native_vllm_module_class", "")).strip()
        vllm_ver = str(capture_case.get("vllm_ascend_version", "")).strip()
        if (
            capture_case.get("native_vllm_layer_executed") is not True
            or runner_mode != "vllm_native"
            or (not vllm_cls.startswith("vllm"))
            or (not vllm_ver)
        ):
            return {**base, "status": "invalid_capture", "error": "unsubstantiated_native_vllm_tier_claim"}
    if tier == TIER_TORCHNPU_COMPOSITE and capture_case.get("native_vllm_layer_executed") is True:
        return {**base, "status": "invalid_capture", "error": "inconsistent_composite_native_vllm_flag"}
    if status == "ok" and tier in (TIER_UNMERGED_ABSENT, TIER_RUNNER_UNSUPPORTED):
        return {**base, "status": "invalid_capture", "error": f"ok_status_with_non_executed_tier:{tier}"}
    if status != "ok":
        peer_recs = list(capture_case.get("rank_receipts") or [])
        if status == "absent_custom_op" and any(str(r.get("status", "")) not in ("", "absent_custom_op") for r in peer_recs if isinstance(r, Mapping)):
            return {**base, "status": "peer_runtime_error", "error": "peer_rank_failed_while_rank0_absent_custom_op"}
        return {**base, "status": status, "absent_reason": _sanitize_diagnostic(str(capture_case.get("absent_reason", "")))}
    return None


def _evaluate_capture_case(
    capture_case: Mapping[str, Any],
    hardware_soc: str,
    spine: PredictionSpine,
    *,
    world_size: int = 1,
    observed_aicore_freq_mhz: float = 0.0,
    bundle_runner_mode: str = "",
) -> dict[str, Any]:
    """Compare one captured live 1-layer subgraph against ``default_spine()``."""
    del observed_aicore_freq_mhz
    early = _check_capture_tier_and_status(capture_case, hardware_soc, world_size, bundle_runner_mode=bundle_runner_mode)
    if early is not None:
        return early
    tier = str(capture_case.get("coverage_tier", TIER_UNMERGED_ABSENT))
    model_key = str(capture_case.get("model", ""))

    try:
        step, ev, geom_meta = _lower_capture_sim_step(capture_case, hardware_soc, spine, world_size)
    except Exception as exc:  # noqa: BLE001
        return {
            "case_id": capture_case.get("case_id"), "model": model_key,
            "hardware_soc": hardware_soc, "world_size": world_size,
            "coverage_tier": tier, "status": "sim_lowering_error",
            "error": _sanitize_diagnostic(f"{type(exc).__name__}:{exc}"), "pass_full_case": False,
        }

    meta_err = _check_capture_model_metadata(capture_case, geom_meta)
    if meta_err is not None:
        return {
            "case_id": capture_case.get("case_id"), "model": model_key,
            "hardware_soc": hardware_soc, "world_size": world_size,
            "coverage_tier": tier, "status": "invalid_capture", "error": meta_err, "pass_full_case": False,
        }

    layer_prefix = str(geom_meta["target_layer_prefix"])
    all_ops = list(step.graph.executable[0].ops) if step.graph.executable else []
    sim_ops = [op for op in all_ops if str(op.op_id).startswith(layer_prefix)]
    sim_stats = _extract_sim_block_spans_and_buckets(sim_ops, ev, world_size=world_size)
    del step, ev, all_ops

    attn_cap = dict(capture_case.get("attention_capture", {}))
    ffn_cap = dict(capture_case.get("ffn_capture", {}))
    layer_cap = dict(capture_case.get("layer_capture", {}))
    attn_win = _recompute_window_timing(attn_cap)
    ffn_win = _recompute_window_timing(ffn_cap)
    layer_win = _recompute_window_timing(layer_cap)
    sub_compute = float(attn_win["compute_sum_us"]) + float(ffn_win["compute_sum_us"])
    layer_consistent = _reconcile_whole_layer_windows(
        attn_cap, ffn_cap, layer_cap, attn_win, ffn_win, layer_win, geom_meta=geom_meta,
    )
    windows_valid = bool(attn_win["valid"] and ffn_win["valid"] and layer_win["valid"] and layer_consistent)
    bubble_contaminated = bool(
        attn_win["gap_contaminated"] or ffn_win["gap_contaminated"] or layer_win["gap_contaminated"]
    )

    rank_audit = _verify_rank_receipts(capture_case, world_size, layer_win=layer_win)
    eff_freq_mhz = rank_audit["rank_freq_mhz_min"]
    freq_telemetry_ok, clock_err = _check_window_clocks_consistent(attn_cap, ffn_cap, layer_cap, eff_freq_mhz)
    if clock_err:
        rank_audit["rank_parity_diffs"].append(clock_err)
        rank_audit["rank_parity_ok"] = False
    obs_layer_boundary_us = (
        max(layer_win["elapsed_span_us"], rank_audit["worst_rank_span_us"])
        if world_size > 1
        else layer_win["elapsed_span_us"]
    )
    topo_audit = _verify_topology_isomorphism(
        sim_ops, list(layer_cap.get("kernels") or []), world_size=world_size,
        expected_tokens=geom_meta["expected_tokens"], expected_hidden=geom_meta["expected_hidden"],
        expected_head_dims=geom_meta["expected_head_dims"],
        expected_kv_dims=geom_meta["expected_kv_dims"],
        expected_gemm_widths=geom_meta["expected_gemm_widths"],
        expected_attn_gemm_widths=geom_meta["expected_attn_gemm_widths"],
        expected_attn_in_widths=geom_meta["expected_attn_in_widths"],
        expected_attn_o_widths=geom_meta["expected_attn_o_widths"],
        expected_ffn_gemm_widths=geom_meta["expected_ffn_gemm_widths"],
        expected_ffn_gate_up_widths=geom_meta["expected_ffn_gate_up_widths"],
        expected_ffn_down_widths=geom_meta["expected_ffn_down_widths"],
        is_moe=bool(geom_meta["is_moe"]),
        expected_min_gemms=geom_meta["expected_min_gemms"],
        expected_local_heads=geom_meta["expected_local_heads"],
        quant_mode=geom_meta["quant_mode"],
    )
    l2_scores = {
        "attention": _score_latency_pair(sim_stats["sim_attn_span_us"], attn_win["elapsed_span_us"]),
        "ffn": _score_latency_pair(sim_stats["sim_ffn_span_us"], ffn_win["elapsed_span_us"]),
        "layer_total": _score_latency_pair(sim_stats["sim_layer_span_us"], obs_layer_boundary_us),
    }
    l2_device_kernel_scores = {
        "attention": _score_latency_pair(sim_stats["sim_attn_iso_us"], attn_win["sum_kernel_us"]),
        "ffn": _score_latency_pair(sim_stats["sim_ffn_iso_us"], ffn_win["sum_kernel_us"]),
        "layer_total": _score_latency_pair(sim_stats["sim_layer_iso_us"], layer_win["sum_kernel_us"]),
    }
    obs_fine_us = _bucket_obs_fine_us(
        list(attn_cap.get("kernels") or []), list(ffn_cap.get("kernels") or []),
        is_moe=bool(geom_meta["is_moe"]), kv_a_out_dim=int(geom_meta.get("kv_a_out_dim", 0)),
        q_lora_rank=int(geom_meta.get("q_lora_rank", 0)),
    )
    l1_in_l2, l1_pass_cnt, l1_total_cnt, worst_l1, abs_err_sum = _score_l1_in_l2_buckets(
        sim_stats["sim_fine_us"], obs_fine_us,
    )
    l1_tier = _summarize_l1_two_tier(l1_in_l2)
    net_err = abs(sim_stats["sim_layer_iso_us"] - sub_compute)
    cancel_ratio = round(abs_err_sum / max(net_err, 1.0), 3) if windows_valid else 0.0
    comm_gate_ok = (world_size <= 1) or (not topo_audit["sim_collective_kinds"]) or bool(topo_audit["collectives_verified"])
    pass_full_case = bool(
        windows_valid and topo_audit["topology_isomorphic"] and comm_gate_ok
        and rank_audit["rank_parity_ok"] and freq_telemetry_ok and (not bubble_contaminated)
        and all(sc["pass_physical_tol"] for sc in l2_scores.values())
        and all(sc["pass_physical_tol"] for sc in l2_device_kernel_scores.values())
        and bool(l1_tier["l1_two_tier_pass"])
    )
    return {
        "case_id": capture_case.get("case_id"), "model": model_key,
        "hardware_soc": hardware_soc, "world_size": world_size,
        "evaluation_scope": "multirank_end_to_end_with_hccl" if world_size > 1 else "single_card_local_compute",
        "observed_aicore_freq_mhz": eff_freq_mhz, "freq_telemetry_ok": freq_telemetry_ok,
        "coverage_tier": tier,
        "stream_blocked": bool(attn_win["stream_blocked"] and ffn_win["stream_blocked"] and layer_win["stream_blocked"]),
        "windows_valid": windows_valid,
        "window_gaps_us": {
            "attention": attn_win["uncovered_gap_us"],
            "ffn": ffn_win["uncovered_gap_us"],
            "layer": layer_win["uncovered_gap_us"],
        },
        "host_dispatch_bubble_contaminated": bubble_contaminated,
        "hardware_wait_us_sum": layer_win["wait_us_sum"],
        "sim_excluded_single_card_comm_us": round(sim_stats["sim_excluded_comm_us"], 3),
        "cancellation_ratio": cancel_ratio, "status": "evaluated",
        **rank_audit, **topo_audit,
        "l2_scores": l2_scores, "l2_device_kernel_scores": l2_device_kernel_scores,
        "l1_in_l2_scores": l1_in_l2, "l1_in_l2_pass_count": l1_pass_cnt,
        "l1_in_l2_total_count": l1_total_cnt,
        "l1_in_l2_pass_rate": round(l1_pass_cnt / l1_total_cnt, 4) if l1_total_cnt else 0.0,
        **l1_tier,
        "worst_l1_outlier": worst_l1,
        "pass_l2_layer": bool(l2_scores["layer_total"]["pass_physical_tol"]),
        "pass_l2_device_kernel": bool(l2_device_kernel_scores["layer_total"]["pass_physical_tol"]),
        "pass_full_case": pass_full_case,
    }


def _validate_case_against_manifest(
    cap_case: Mapping[str, Any],
    manifest_map: Mapping[tuple[str, int], Mapping[str, Any]],
    world_size: int,
) -> str | None:
    """Verify capture metadata matches the predeclared spec manifest across ALL structural & parallel fields."""
    cid = str(cap_case.get("case_id", ""))
    expected = manifest_map.get((cid, world_size))
    if expected is None:
        return _sanitize_diagnostic(f"undeclared_case_in_manifest:{cid}:world={world_size}")
    for k in ("model", "phase", "attention", "architecture", "ffn", "overlay"):
        exp_val = str(expected.get(k, "")).strip()
        if exp_val and str(cap_case.get(k, "")).strip() != exp_val:
            return _sanitize_diagnostic(f"{k}_mismatch:{cap_case.get(k)}!={exp_val}")
    cap_li = _safe_int(cap_case.get("layer_index", -1), -999)
    exp_li = _safe_int(expected.get("layer_index", -2), -998)
    if cap_li < 0 or cap_li != exp_li:
        return _sanitize_diagnostic(f"layer_index_mismatch:{cap_case.get('layer_index')}!={expected.get('layer_index')}")
    par = cap_case.get("parallel")
    if not isinstance(par, Mapping):
        return "invalid_parallel_mapping"
    for k in ("dp", "tp", "ep", "dcp", "pcp", "o_proj_tp", "linear_proj_tp"):
        cv = _safe_int(par.get(k, 1), -1)
        ev = _safe_int(expected.get(k, 1), -2)
        if cv < 1 or cv != ev:
            return _sanitize_diagnostic(f"parallel_{k}_mismatch:{par.get(k)}!={expected.get(k)}")
    wl = cap_case.get("workload")
    if not isinstance(wl, Mapping):
        return "invalid_workload_mapping"
    for k in ("sequences", "active_sequences", "q_len", "context", "quant_mode"):
        if str(wl.get(k, "")) != str(expected.get(k, "")):
            return _sanitize_diagnostic(f"workload_{k}_mismatch:{wl.get(k)}!={expected.get(k)}")
    return None


def _eval_capture_worker(item: tuple[dict[str, Any], str, int, float, str]) -> dict[str, Any]:
    cap_case, soc, world, freq_mhz, runner_mode = item
    res = _evaluate_capture_case(
        cap_case, soc, default_spine(), world_size=world,
        observed_aicore_freq_mhz=freq_mhz, bundle_runner_mode=runner_mode,
    )
    gc.collect()
    return res


def _needs_isolated_rss_worker(cap_case: Mapping[str, Any]) -> bool:
    """Return True for high-DP or linear-attention prefill cases whose spine replay needs a fresh child."""
    par = cap_case.get("parallel") if isinstance(cap_case.get("parallel"), Mapping) else {}
    eff_dp = max(_safe_int(par.get("dp", 1), 1), _safe_int(par.get("ep", 1), 1))
    attn = str(cap_case.get("attention", "")).lower()
    phase = str(cap_case.get("phase", "")).lower()
    return eff_dp >= RSS_ISOLATE_DP_THRESHOLD or (phase == "prefill" and attn in ("gdn", "kda"))


def _stream_file_sha256(path: Path) -> str:
    """Compute file SHA-256[:16] in 64 KiB chunks without materializing whole-file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def _reconcile_manifest_gaps(
    manifest_map: Mapping[tuple[str, int], Mapping[str, Any]],
    seen_keys: set[tuple[str, str, int]],
    hw_socs: set[str],
    excluded_soc_worlds: frozenset[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Emit ``missing_capture`` rows for EVERY manifest entry absent on each arrived SOC."""
    missing: list[dict[str, Any]] = []
    for soc in sorted(hw_socs):
        for (cid, m_world), spec_row in sorted(manifest_map.items()):
            if (soc, m_world) in excluded_soc_worlds:
                continue
            if (cid, soc, m_world) not in seen_keys:
                missing.append({
                    "case_id": cid, "model": str(spec_row.get("model", "")),
                    "hardware_soc": soc, "world_size": m_world,
                    "coverage_tier": TIER_UNMERGED_ABSENT,
                    "status": "missing_capture",
                    "error": f"expected_by_manifest_missing_on_{soc}_w{m_world}",
                    "pass_full_case": False,
                })
    return missing


def _iter_json_bundle_events(path: Path) -> Any:
    """Strictly stream a capture JSON in 64 KiB chunks, yielding ``('case', item)`` and final ``('header', hdr)``.

    Rejects trailing commas, truncated containers, trailing garbage, and oversized/syntax-broken values
    without retaining the rest of the file, and refills numbers/literals split across chunk boundaries.
    """
    dec = json.JSONDecoder()
    header: dict[str, Any] = {}
    buf = ""
    eof = False

    with path.open("r", encoding="utf-8") as handle:
        def _refill() -> bool:
            nonlocal buf, eof
            if eof:
                return False
            data = handle.read(JSON_STREAM_CHUNK_SIZE)
            if not data:
                eof = True
                return False
            buf += data
            return True

        def _skip_ws(pos: int) -> int:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n":
                    pos += 1
                if pos < len(buf):
                    return pos
                if not _refill():
                    return pos

        def _decode_at(pos: int) -> tuple[Any, int]:
            while True:
                pos = _skip_ws(pos)
                if pos >= len(buf):
                    raise ValueError(f"bundle_json_unexpected_eof:{path.name}")
                try:
                    val, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError as exc:
                    is_trunc = exc.pos >= len(buf) - 1 or ("Unterminated" in exc.msg)
                    if (not is_trunc) or (len(buf) - pos >= MAX_SINGLE_JSON_VALUE_BYTES) or (not _refill()):
                        raise
                    continue
                if end == len(buf) and (not eof) and not isinstance(val, (dict, list, str)):
                    if _refill():
                        continue
                return val, end

        pos = _skip_ws(0)
        if pos >= len(buf) or buf[pos] != "{":
            raise ValueError(f"bundle_json_expected_object:{path.name}")
        pos += 1
        closed_root = False
        expect_root_key = False
        while True:
            pos = _skip_ws(pos)
            if pos >= len(buf):
                raise ValueError(f"bundle_json_truncated_root:{path.name}")
            if buf[pos] == "}":
                if expect_root_key:
                    raise ValueError(f"bundle_json_trailing_comma_root:{path.name}")
                pos += 1
                closed_root = True
                break
            expect_root_key = False
            key, pos = _decode_at(pos)
            if not isinstance(key, str):
                raise ValueError(f"bundle_json_expected_string_key:{path.name}")
            pos = _skip_ws(pos)
            if pos >= len(buf) or buf[pos] != ":":
                raise ValueError(f"bundle_json_expected_colon:{path.name}")
            pos = _skip_ws(pos + 1)
            if key == "cases" and pos < len(buf) and buf[pos] == "[":
                pos += 1
                closed_arr = False
                expect_item = False
                while True:
                    pos = _skip_ws(pos)
                    if pos >= len(buf):
                        raise ValueError(f"bundle_json_truncated_cases:{path.name}")
                    if buf[pos] == "]":
                        if expect_item:
                            raise ValueError(f"bundle_json_trailing_comma_cases:{path.name}")
                        pos += 1
                        closed_arr = True
                        break
                    expect_item = False
                    item, pos = _decode_at(pos)
                    yield ("case", item)
                    buf = buf[pos:]
                    pos = 0
                    pos = _skip_ws(pos)
                    if pos >= len(buf):
                        raise ValueError(f"bundle_json_truncated_cases:{path.name}")
                    if buf[pos] == ",":
                        pos += 1
                        expect_item = True
                    elif buf[pos] != "]":
                        raise ValueError(f"bundle_json_expected_comma_or_end:{path.name}")
                if not closed_arr:
                    raise ValueError(f"bundle_json_truncated_cases:{path.name}")
            else:
                val, pos = _decode_at(pos)
                header[key] = val
            buf = buf[pos:]
            pos = 0
            pos = _skip_ws(pos)
            if pos >= len(buf):
                raise ValueError(f"bundle_json_truncated_root:{path.name}")
            if buf[pos] == ",":
                pos += 1
                expect_root_key = True
            elif buf[pos] != "}":
                raise ValueError(f"bundle_json_expected_comma_or_brace:{path.name}")
        if not closed_root:
            raise ValueError(f"bundle_json_truncated_root:{path.name}")
        pos = _skip_ws(pos)
        if pos < len(buf) or _refill():
            rem = buf[pos:].strip()
            if rem:
                raise ValueError(f"bundle_json_trailing_garbage:{path.name}")
    yield ("header", header)


def _stream_json_bundle_header(path: Path) -> tuple[dict[str, Any], int]:
    """Validate strict JSON completeness and extract header + case count in O(1) memory."""
    header: dict[str, Any] = {}
    count = 0
    for kind, payload in _iter_json_bundle_events(path):
        if kind == "case":
            count += 1
        else:
            header = dict(payload)
    return header, count


def _iter_json_bundle_cases(path: Path) -> Any:
    """Lazily yield one decoded case dict at a time in 64 KiB chunks."""
    for kind, payload in _iter_json_bundle_events(path):
        if kind == "case":
            yield payload


def _stream_json_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a small capture bundle strictly via the bounded 64 KiB streaming parser."""
    header: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    for kind, payload in _iter_json_bundle_events(path):
        if kind == "case":
            cases.append(payload)
        else:
            header = dict(payload)
    return header, cases


def _strict_pos_int(val: Any) -> int | None:
    if isinstance(val, int) and not isinstance(val, bool) and val >= 1:
        return int(val)
    return None


def _check_bundle_provenance(
    path: Path,
    bundle: Mapping[str, Any],
    cases_len: int,
    *,
    current_runner_sha: str,
    manifest_file_sha: str,
    manifest_inner_sha: str,
    allow_stale_runner: bool,
) -> list[str]:
    """Enforce capture bundle schema, non-emptiness, strict int world_size, completion, and digests."""
    errs: list[str] = []
    if str(bundle.get("schema_version", "")) != EXPECTED_CAPTURE_SCHEMA_VERSION:
        errs.append(_sanitize_diagnostic(f"invalid_schema_version:{path.name}:{bundle.get('schema_version')}"))
    if bundle.get("shard_complete") is not True:
        errs.append(_sanitize_diagnostic(f"incomplete_shard:{path.name}:{bundle.get('shard_complete')}"))
    if cases_len <= 0:
        errs.append(_sanitize_diagnostic(f"empty_capture_bundle:{path.name}"))
    if "world_size" in bundle and _strict_pos_int(bundle.get("world_size")) is None:
        errs.append(_sanitize_diagnostic(f"invalid_bundle_world_size:{path.name}:{bundle.get('world_size')}"))
    if "expected_cases" in bundle:
        exp_n = _strict_pos_int(bundle["expected_cases"])
        if exp_n is None or cases_len != exp_n:
            errs.append(_sanitize_diagnostic(f"shard_case_count_mismatch:{path.name}:{cases_len}!={bundle['expected_cases']}"))
    r_sha = str(bundle.get("runner_sha256", "")).strip()
    if (not allow_stale_runner) and current_runner_sha and r_sha != current_runner_sha:
        errs.append(_sanitize_diagnostic(f"runner_sha256_mismatch:{path.name}:{r_sha}!={current_runner_sha}"))
    if manifest_file_sha:
        s_sha = str(bundle.get("spec_sha256", "")).strip()
        if s_sha not in (manifest_file_sha, manifest_inner_sha):
            errs.append(_sanitize_diagnostic(f"spec_sha256_mismatch:{path.name}:{s_sha}!={manifest_file_sha}"))
    return errs


def _load_and_validate_manifest(
    spec_manifest: Path,
    shard_integrity_errors: list[str],
) -> tuple[dict[tuple[str, int], dict[str, Any]], str, str]:
    """Parse and strictly validate a predeclared spec manifest (rejecting duplicates/malformed entries)."""
    manifest_map: dict[tuple[str, int], dict[str, Any]] = {}
    manifest_file_sha = _stream_file_sha256(spec_manifest)
    try:
        m_doc = json.loads(spec_manifest.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        shard_integrity_errors.append(_sanitize_diagnostic(f"malformed_spec_manifest_json:{type(exc).__name__}:{exc}"))
        return manifest_map, manifest_file_sha, ""
    if not isinstance(m_doc, dict) or not isinstance(m_doc.get("cases"), list):
        shard_integrity_errors.append("malformed_spec_manifest_schema")
        return manifest_map, manifest_file_sha, ""
    m_cases = list(m_doc["cases"])
    if not m_cases:
        shard_integrity_errors.append("empty_spec_manifest")
    if any(not isinstance(r, dict) for r in m_cases):
        shard_integrity_errors.append("malformed_spec_manifest_case_entry")
        return manifest_map, manifest_file_sha, ""
    recomputed_inner = _compute_manifest_cases_digest(m_cases)
    decl_inner = str(m_doc.get("spec_sha256", "")).strip()
    if decl_inner and decl_inner != recomputed_inner:
        shard_integrity_errors.append(
            _sanitize_diagnostic(f"manifest_inner_spec_sha256_mismatch:declared={decl_inner},recomputed={recomputed_inner}")
        )
    for row in m_cases:
        cid = str(row.get("case_id", "")).strip()
        w_val = _strict_pos_int(row.get("world_size", 1))
        if not cid or w_val is None:
            shard_integrity_errors.append(_sanitize_diagnostic(f"invalid_manifest_entry:{cid}:world={row.get('world_size')}"))
            continue
        key = (cid, w_val)
        if key in manifest_map:
            shard_integrity_errors.append(_sanitize_diagnostic(f"duplicate_manifest_case_id:{cid}:w{w_val}"))
            continue
        manifest_map[key] = dict(row)
    return manifest_map, manifest_file_sha, recomputed_inner


def _flush_capture_batch(
    batch_items: list[tuple[dict[str, Any], str, int, float, str]],
    evaluated_rows: list[dict[str, Any]],
    *,
    workers: int,
    spine_box: list[PredictionSpine | None],
) -> None:
    """Evaluate a bounded chunk of streamed cases immediately and release raw payloads."""
    if not batch_items:
        return
    heavy_items = [it for it in batch_items if _needs_isolated_rss_worker(it[0])]
    light_items = [it for it in batch_items if not _needs_isolated_rss_worker(it[0])]
    if heavy_items:
        evaluated_rows.extend(_spawn_map_chunked(_eval_capture_worker, heavy_items, workers=1, chunk_size=1))
    if workers > 1 and len(light_items) > 1:
        evaluated_rows.extend(_spawn_map_chunked(_eval_capture_worker, light_items, workers=workers, chunk_size=STREAM_CASE_BATCH_SIZE))
    else:
        for item in light_items:
            if spine_box[0] is None:
                spine_box[0] = default_spine()
            evaluated_rows.append(
                _evaluate_capture_case(
                    item[0], item[1], spine_box[0],
                    world_size=item[2], observed_aicore_freq_mhz=item[3], bundle_runner_mode=item[4],
                )
            )
    batch_items.clear()
    gc.collect()


def _aggregate_gate_b_two_tier_stats(ok_eval: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate Gate B L2 (10%/2us) & L1-in-L2 (Med<=8%, P90<=12%, cliff<=15%/3us) distribution metrics."""
    l2_apes = [
        abs(float(r["l2_scores"]["layer_total"]["signed_err_pct"]))
        for r in ok_eval
        if isinstance((r.get("l2_scores") or {}).get("layer_total"), Mapping)
        and r["l2_scores"]["layer_total"].get("signed_err_pct") is not None
    ]
    by_bucket: dict[str, list[tuple[float, bool]]] = {}
    all_l1_apes: list[float] = []
    all_l1_cliff = 0
    for r in ok_eval:
        for bkt, sc in dict(r.get("l1_in_l2_scores") or {}).items():
            ape = abs(float(sc.get("signed_err_pct") if sc.get("signed_err_pct") is not None else PERCENT_SCALE))
            c_ok = bool(sc.get("pass_l1_cliff_tol"))
            by_bucket.setdefault(str(bkt), []).append((ape, c_ok))
            all_l1_apes.append(ape)
            all_l1_cliff += int(c_ok)
    bucket_summary: dict[str, Any] = {}
    for bkt in sorted(by_bucket):
        pairs = by_bucket[bkt]
        b_apes = [p[0] for p in pairs]
        raw_b_med = float(statistics.median(b_apes))
        raw_b_p90 = _quantile_p90(b_apes)
        b_cliff = sum(1 for _, ok in pairs if ok)
        bucket_summary[bkt] = {
            "count": len(pairs),
            "median_ape_pct": round(raw_b_med, 2),
            "p90_ape_pct": round(raw_b_p90, 2),
            "cliff_pass_rate": round(b_cliff / len(pairs), 4),
            "two_tier_pass": bool(b_cliff == len(pairs) and raw_b_med <= L1_MEDIAN_APE_LIMIT_PCT and raw_b_p90 <= L1_P90_APE_LIMIT_PCT),
        }
    return {
        "l2_median_ape_pct": round(float(statistics.median(l2_apes)), 2) if l2_apes else None,
        "l2_p90_ape_pct": round(_quantile_p90(l2_apes), 2) if l2_apes else None,
        "l1_in_l2_median_ape_pct": round(float(statistics.median(all_l1_apes)), 2) if all_l1_apes else None,
        "l1_in_l2_p90_ape_pct": round(_quantile_p90(all_l1_apes), 2) if all_l1_apes else None,
        "l1_in_l2_cliff_pass_rate": round(all_l1_cliff / len(all_l1_apes), 4) if all_l1_apes else 0.0,
        "l1_in_l2_two_tier_pass_cases": sum(1 for r in ok_eval if r.get("l1_two_tier_pass")),
        "l1_in_l2_all_buckets_two_tier_pass": bool(
            bucket_summary and all(bool(b.get("two_tier_pass")) for b in bucket_summary.values())
        ),
        "l1_in_l2_by_bucket": bucket_summary,
    }


def run_gate_b(
    captures_dir: Path,
    *,
    workers: int = 1,
    spec_manifest: Path | None = None,
    exclude_soc_worlds: Sequence[tuple[str, int]] = (),
    expected_socs: Sequence[str] | None = None,
    allow_stale_runner: bool = False,
) -> dict[str, Any]:
    """Evaluate live 1-layer capture bundles with strict provenance, O(1)-case streaming I/O, and RSS bounds."""
    t0 = time.perf_counter()
    bundle_paths = sorted(captures_dir.glob("*.json")) if captures_dir.is_dir() else []
    if spec_manifest is not None:
        manifest_resolved = spec_manifest.resolve()
        bundle_paths = [p for p in bundle_paths if p.resolve() != manifest_resolved]
    if not bundle_paths:
        return {"gate": "B", "status": "fail", "reason": _sanitize_diagnostic(f"no_capture_bundles_found:{captures_dir}")}

    manifest_map: dict[tuple[str, int], dict[str, Any]] = {}
    manifest_file_sha = manifest_inner_sha = ""
    shard_integrity_errors: list[str] = []
    has_manifest = spec_manifest is not None
    if spec_manifest is not None:
        if not spec_manifest.is_file():
            return {"gate": "B", "status": "fail", "reason": _sanitize_diagnostic(f"spec_manifest_not_found:{spec_manifest}")}
        manifest_map, manifest_file_sha, manifest_inner_sha = _load_and_validate_manifest(spec_manifest, shard_integrity_errors)

    verifier_sha = _stream_file_sha256(Path(__file__).resolve())
    runner_path = SIM_ROOT / "scripts" / "run_1layer_npu_subgraphs.py"
    current_runner_sha = _stream_file_sha256(runner_path) if runner_path.is_file() else ""
    excluded_set = frozenset((str(s), int(w)) for s, w in exclude_soc_worlds)
    bundle_digests: dict[str, str] = {}
    runner_shas: set[str] = set()
    spec_shas: set[str] = set()
    hw_socs: set[str] = set()
    seen_identities: set[tuple[str, str, int]] = set()
    evaluated_rows: list[dict[str, Any]] = []
    spine_box: list[PredictionSpine | None] = [None]

    for path in bundle_paths:
        bundle_digests[path.name] = _stream_file_sha256(path)
        try:
            bundle, cases_len = _stream_json_bundle_header(path)
        except Exception as exc:  # noqa: BLE001
            shard_integrity_errors.append(_sanitize_diagnostic(f"malformed_bundle_json:{path.name}:{type(exc).__name__}:{exc}"))
            continue
        soc = str(bundle.get("hardware_soc", "unknown"))
        raw_world = _strict_pos_int(bundle.get("world_size", 1))
        world = raw_world if raw_world is not None else 1
        freq_mhz = _safe_float(bundle.get("observed_aicore_freq_mhz", 0.0), 0.0)
        bundle_runner_mode = str(bundle.get("runner_mode", "")).strip()
        hw_socs.add(soc)
        raw_r_sha = str(bundle.get("runner_sha256", "")).strip()
        if raw_r_sha:
            runner_shas.add(raw_r_sha if HEX_DIGEST_RE.match(raw_r_sha) else "<invalid_digest>")
        raw_s_sha = str(bundle.get("spec_sha256", "")).strip()
        if raw_s_sha:
            spec_shas.add(raw_s_sha if HEX_DIGEST_RE.match(raw_s_sha) else "<invalid_digest>")
        shard_integrity_errors.extend(_check_bundle_provenance(
            path, bundle, cases_len,
            current_runner_sha=current_runner_sha, manifest_file_sha=manifest_file_sha,
            manifest_inner_sha=manifest_inner_sha, allow_stale_runner=allow_stale_runner,
        ))
        del bundle

        batch_items: list[tuple[dict[str, Any], str, int, float, str]] = []
        for cap_case in _iter_json_bundle_cases(path):
            if not isinstance(cap_case, dict):
                evaluated_rows.append({
                    "case_id": "<non_object_case>", "model": "", "hardware_soc": soc, "world_size": world,
                    "coverage_tier": TIER_UNMERGED_ABSENT, "status": "invalid_capture",
                    "error": "non_object_case_entry", "pass_full_case": False,
                })
                continue
            cid = str(cap_case.get("case_id", ""))
            ident = (cid, soc, world)
            if ident in seen_identities:
                evaluated_rows.append({
                    "case_id": cid, "model": str(cap_case.get("model", "")),
                    "hardware_soc": soc, "world_size": world,
                    "coverage_tier": str(cap_case.get("coverage_tier", TIER_UNMERGED_ABSENT)),
                    "status": "duplicate_capture", "pass_full_case": False,
                })
                continue
            seen_identities.add(ident)
            if has_manifest:
                mismatch = _validate_case_against_manifest(cap_case, manifest_map, world)
                if mismatch is not None:
                    evaluated_rows.append({
                        "case_id": cid, "model": str(cap_case.get("model", "")),
                        "hardware_soc": soc, "world_size": world,
                        "coverage_tier": str(cap_case.get("coverage_tier", TIER_UNMERGED_ABSENT)),
                        "status": "spec_mismatch", "error": mismatch, "pass_full_case": False,
                    })
                    continue
            batch_items.append((cap_case, soc, world, freq_mhz, bundle_runner_mode))
            if len(batch_items) >= STREAM_CASE_BATCH_SIZE:
                _flush_capture_batch(batch_items, evaluated_rows, workers=workers, spine_box=spine_box)
        _flush_capture_batch(batch_items, evaluated_rows, workers=workers, spine_box=spine_box)

    if has_manifest:
        reconcile_socs = set(expected_socs) if expected_socs is not None else (hw_socs | set(DEFAULT_EXPECTED_LIVE_SOCS))
        evaluated_rows.extend(_reconcile_manifest_gaps(manifest_map, seen_identities, reconcile_socs, excluded_set))

    tier_counts: Counter[str] = Counter(str(r.get("coverage_tier", TIER_UNMERGED_ABSENT)) for r in evaluated_rows)
    ok_eval = [r for r in evaluated_rows if r.get("status") == "evaluated"]
    blocked_cases = [r for r in evaluated_rows if r.get("status") in ("absent_custom_op", "runner_unsupported")]
    passed_cases = [r for r in ok_eval if r.get("pass_full_case")]
    failed_cases = [
        r for r in evaluated_rows
        if r.get("status") not in ("absent_custom_op", "runner_unsupported") and not r.get("pass_full_case")
    ]
    models_received = sorted({str(r["model"]) for r in evaluated_rows if str(r.get("model", "")) in ALL_CATALOG_MODEL_SET})
    models_verified_pass = sorted({str(r["model"]) for r in passed_cases if str(r.get("model", "")) in ALL_CATALOG_MODEL_SET})
    total_cases = len(evaluated_rows)
    zero_drop_ok = total_cases == len(passed_cases) + len(failed_cases) + len(blocked_cases)
    two_tier_stats = _aggregate_gate_b_two_tier_stats(ok_eval)
    gate_pass = bool(
        passed_cases and not failed_cases and not blocked_cases
        and not shard_integrity_errors and zero_drop_ok
        and two_tier_stats["l1_in_l2_all_buckets_two_tier_pass"]
    )

    return {
        "gate": "B", "status": "pass" if gate_pass else "fail",
        "verifier_sha256": verifier_sha, "manifest_sha256": manifest_file_sha,
        "runner_sha256s": sorted(runner_shas), "spec_sha256s": sorted(spec_shas),
        "bundle_digests": bundle_digests, "shard_integrity_errors": shard_integrity_errors,
        "explicit_scope_exclusions": [f"{s}:w{w}" for s, w in sorted(excluded_set)],
        "hardware_socs": sorted(hw_socs), "models_received": len(models_received),
        "models_verified_pass": len(models_verified_pass), "total_cases": total_cases,
        "evaluated_cases": len(ok_eval), "passed_cases": len(passed_cases),
        "failed_cases": len(failed_cases), "blocked_cases": len(blocked_cases),
        "zero_drop_conserved": zero_drop_ok,
        "topology_isomorphic_pass": sum(1 for r in ok_eval if r.get("topology_isomorphic")),
        "rank_parity_pass": sum(1 for r in ok_eval if r.get("rank_parity_ok")),
        "l2_physical_tol_pass": sum(1 for r in ok_eval if r.get("pass_l2_layer")),
        "l2_device_kernel_tol_pass": sum(1 for r in ok_eval if r.get("pass_l2_device_kernel")),
        **two_tier_stats,
        "coverage_tiers": dict(tier_counts), "wall_seconds": round(time.perf_counter() - t0, 3),
        "rows": evaluated_rows,
    }


def _parse_soc_world_exclusions(raw_items: Sequence[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in raw_items:
        if ":" not in item:
            raise SystemExit(f"invalid --exclude-soc-world value '{item}': expected SOC:WORLD")
        soc, w_str = item.rsplit(":", 1)
        out.append((soc.strip(), max(1, int(w_str.strip()))))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-a", action="store_true", help="Run Gate A 43-model structure & conservation checks")
    parser.add_argument("--gate-b", action="store_true", help="Run Gate B live 1-layer capture isomorphism & latency check")
    parser.add_argument("--export-specs", type=Path, default=None, help="Export 43-model live 1-layer runner spec JSON")
    parser.add_argument("--quant-mode", default="bf16", choices=("bf16", "w8a8", "w4a8", "c8", "mxfp8", "mxfp4"))
    parser.add_argument("--captures-dir", type=Path, default=None, help="Directory of live NPU JSON capture bundles")
    parser.add_argument("--spec-manifest", type=Path, default=None, help="Predeclared spec manifest JSON for Gate B reconciliation")
    parser.add_argument("--exclude-soc-world", action="append", default=[], help="Explicit scope exclusion e.g. Ascend910B4:4")
    parser.add_argument("--expected-soc", action="append", default=[], help="Explicit expected SOC list for Gate B manifest reconciliation")
    parser.add_argument("--allow-stale-runner", action="store_true", help="Diagnostic flag permitting runner_sha256 drift")
    parser.add_argument("--va-tutorials-dir", type=Path, default=None, help="Optional path to vllm-ascend/docs/source/tutorials/models")
    parser.add_argument("--sku", default=DEFAULT_GATE_A_SKU, help="Primary Ascend SKU for Gate A")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes for Gate A/B")
    parser.add_argument("--models", default="", help="Optional comma-separated subset of catalog models")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--diagnostic-only", action="store_true", help="Exit 0 even when requested gate status is fail")
    args = parser.parse_args(argv)

    models = tuple(m.strip() for m in args.models.split(",") if m.strip()) if args.models else ALL_CATALOG_MODELS
    report: dict[str, Any] = {}

    if args.export_specs is not None:
        specs = export_live_runner_specs(models, quant_mode=args.quant_mode)
        args.export_specs.parent.mkdir(parents=True, exist_ok=True)
        args.export_specs.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"exported_cases": len(specs["cases"]), "models": specs["model_count"], "spec_sha256": specs["spec_sha256"], "path": str(args.export_specs)}))
        if not args.gate_a and not args.gate_b:
            return 0

    if args.gate_a:
        report["gate_a"] = run_gate_a(
            models, sku=args.sku, workers=max(1, int(args.workers)), tutorials_dir=args.va_tutorials_dir,
        )
    if args.gate_b:
        if args.captures_dir is None:
            raise SystemExit("--gate-b requires --captures-dir")
        report["gate_b"] = run_gate_b(
            args.captures_dir,
            workers=max(1, int(args.workers)),
            spec_manifest=args.spec_manifest,
            exclude_soc_worlds=_parse_soc_world_exclusions(args.exclude_soc_world),
            expected_socs=list(args.expected_soc) if args.expected_soc else None,
            allow_stale_runner=bool(args.allow_stale_runner),
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {}
    all_passed = bool(report)
    if "gate_a" in report:
        ga = report["gate_a"]
        summary["gate_a"] = {k: v for k, v in ga.items() if k not in ("models", "tutorials", "findings")}
        summary["gate_a"]["tutorial_pass"] = f"{ga['tutorials']['passed']}/{ga['tutorials']['total']}"
        summary["gate_a"]["sample_findings"] = ga["findings"][:12]
        all_passed = all_passed and (ga.get("status") == "pass")
    if "gate_b" in report:
        gb = report["gate_b"]
        summary["gate_b"] = {k: v for k, v in gb.items() if k != "rows"}
        all_passed = all_passed and (gb.get("status") == "pass")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (args.diagnostic_only or all_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
