"""Fitted physical (roofline) calibration for A2/A3 operator durations.

[Role]
  Fit and serve a versioned, provenance-bearing physical cost model per
  ``(hardware_key, family, recipe, protocol_signature)``: a wave-quantized
  roofline with fitted tile geometry, launch intercept, cube / memory
  efficiency and an optional fitted L2-residency split.  This is the physical
  layer that ``PredictiveCostModel`` consumes above the analytic formula; it
  replaces the fixed 64x128 tiling, the constant AIC count, the multiplicative
  calibration factor and the invented launch floor of ``a3_cost.py``
  (see ``.workstreams/code-audit/formula-weaknesses.md`` W1/W2/W8/W9).

[Invariants]
  * Fitting reads train observations only.  Any observation whose ``split`` is
    not ``train`` is rejected before a parameter is estimated, and a declared
    heldout guard (condition keys plus physical identities) is asserted
    disjoint from the training set.
  * Observations are never pooled across ``protocol_signature`` or
    ``hardware_key``.  An ambiguous identity is a rejection with a reason, never
    a silent merge; a family below the minimum sample count is fitted at a
    coarser level that stays inside one protocol, or is declared unmodeled.
  * Every fitted value lives inside a declared physical bound (``eta_*`` in
    ``(0, 1]``, launch in ``LAUNCH_BOUNDS_MS``, ``block_*`` from the declared
    grid, ``l2_usable`` at most the declared L2 capacity) and is stored next to
    its sample count, train metrics and train-only CV metrics.
  * The L2 bandwidth is the only fitted rate allowed above the HBM peak, and
    only when the fitted blend beats the HBM-only model on train-only CV by
    ``MIN_CV_IMPROVEMENT_PCT``; otherwise ``memory_model`` is ``"hbm"``.
  * A blend is a claim about *residency*, and prediction tests it before
    crediting anything: a weight-dominated GEMM whose weight cannot survive the
    per-token stream (``weight_stream_basis``) and a whole working set at least
    ``STREAMING_L2_RATIO`` x the declared L2 are priced with **no** L2 credit at
    an HBM rate.  That rate is the family's own fitted streaming efficiency when
    its corpus measured the regime, else the hardware key's declared measured
    bound, else the declared peak -- recorded in ``streaming_rate_source`` and
    never borrowed across parts.  The rule reads shape, bytes and declared
    constants only; it is prediction-side, so no training row or CV moves.
  * Prediction never returns a number for an unmodeled family / recipe / dtype,
    or for a machine whose hardware key contradicts the artifact: it returns
    ``None`` and an explicit reason (``predict_op_explained``).
  * Model terms are shape-derived, never per-shape corrections: the only
    per-family degrees of freedom are tile geometry, launch, the efficiencies
    and the L2 split.

[Timeouts & Locks]
  Pure in-memory arithmetic with bounded loops: no device, subprocess, network,
  lock, sleep or filesystem access.  Fitting is O(tile grid x memory candidates
  x CV folds x observations) with every loop count fixed by this module's
  declared constants.

[Side-Effect Policy]
  Importing and fitting write nothing and mutate no environment.  The
  ``__main__`` CLI reads the paths it is given and writes only ``--output``.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SCHEMA_VERSION", "OBSERVATION_SCHEMA", "MODEL_KINDS",
    "ATTENTION_KINDS", "GROUPED_KINDS", "ATTENTION_Q_TILES", "ATTENTION_KV_TILES",
    "ATTENTION_DIM_TILE", "ATTENTION_DEVICE_SCOPE", "ATTENTION_BLOCK_SCOPE",
    "attention_visible_kv", "read_attention_results",
    "BLOCK_M_CHOICES", "BLOCK_N_CHOICES", "LAUNCH_BOUNDS_MS", "ETA_BOUNDS",
    "MODEL_FORMS", "FAMILY_ALIASES", "ACCEPTED_TIMING_SCOPES", "M_BAND_LIMITS",
    "MIN_SAMPLES_PER_FIT", "PhysicalCalibrationError", "MachineLimits",
    "Observation", "FamilyParameters", "OpPrediction", "HeldoutGuard",
    "PhysicalCalibration", "fit_calibration", "read_observations_json", "read_campaign_results",
    "read_sqlite_observations", "observation_identity", "canonical_shape",
    "streaming_regime", "weight_stream_basis", "STREAMING_BW_BYTES_PER_S",
    "STREAMING_WEIGHT_RESIDENT_SHARE", "STREAMING_WEIGHT_MAX_M",
    "STREAMING_WEIGHT_DOMINANCE",
    "MC2_GROUPED_RECIPE", "MC2_GROUPED_SCOPE", "MC2_GROUPED_PROTOCOL",
    "MC2_GROUPED_LEGS", "MC2_ACTIVATION_LEG",
    "read_mc2_grouped_results",
]

SCHEMA_VERSION = "mindie-sim.physical-calibration/1"
OBSERVATION_SCHEMA = "mindie-sim.observations/1"

# Tile geometry grid searched per fit (W1: fixed 64x128x64 is structural).
BLOCK_M_CHOICES = (32, 64, 128, 256)
BLOCK_N_CHOICES = (64, 128, 256, 512)
# Launch intercept: physically 0..60 us (W8: 20 us was fabricated for every SKU).
LAUNCH_BOUNDS_MS = (0.0, 0.060)
# Efficiency bounds; eta = 1 is the declared peak, never exceeded on HBM.
ETA_BOUNDS = (0.05, 1.0)
# L2 usable-capacity candidates as a fraction of the declared L2 capacity.
# Capped at 62.5 %: the resident prefix is not identifiable above it.  On the A3
# per-call GEMM corpus a 144 MiB claim (75 %) needs a 4.8 TB/s L2 rate -- the
# declared 4x ceiling, which no observation in that corpus reaches (the measured
# memory-bound ladder tops out at 1.48 TB/s) -- and the training-WAPE plateau is
# flat from 120 to 192 MiB, so every candidate in the tolerance band carries an
# L2 rate sitting on the cap.  The cap costs 0.28 pp of train-only CV on that row
# (9.02 / 9.12 % vs 8.72 / 8.84 %) and removes a claim the data does not support.
L2_USABLE_FRACTIONS = (0.0625, 0.125, 0.25, 0.375, 0.5, 0.625)
# Fitted L2 read bandwidth, as a multiple of the declared HBM peak.
L2_BW_MULTIPLE_BOUNDS = (1.0, 4.0)
# `l2_usable` and `eta_memory` trade off on a flat plateau once every binding row
# is fully resident; prefer the smallest resident prefix whose train WAPE is
# within this relative tolerance of the best, so the split stays identifiable.
L2_USABLE_SELECTION_TOLERANCE = 0.05

# Additive pipeline fill for short-k GEMMs (W3): declared k threshold and bounds.
K_FILL_MAX_K = 256.0
K_FILL_BOUNDS_MS = (0.0, 0.2)
# Memory-side n-alignment penalty: extra full-bandwidth passes paid when
# n % block_n != 0 (misaligned column tiles stream at a fraction of peak).
ALIGN_PENALTY_BOUNDS = (0.0, 4.0)
# A tail term must buy this much train-only CV before it is kept.
MIN_TAIL_IMPROVEMENT_PCT = 0.5
# A cross-scope shared fit must buy this much pooled train-only CV over the
# per-scope independent fits before it is adopted.
MIN_SHARED_SCOPE_IMPROVEMENT_PCT = 1.0

HUBER_K = 1.345
IRLS_ITERATIONS = 12
MIN_SAMPLES_PER_FIT = 6
CV_FOLDS = 5
# Declared issue ceiling per AIV (4 issue slots x 128 lanes): the physical upper
# bound for the fitted effective vector throughput.
VEC_ISSUE_MAX_PER_AIV = 512.0
# Slowest vector throughput the fit may claim, as a fraction of that ceiling.
VEC_THROUGHPUT_MIN_FRACTION = 1.0 / 64.0
# Profile interval: the parameter range whose train WAPE stays within this
# relative tolerance of the best value, all other parameters refitted.  A 25%
# band is a practical-equivalence width for a corpus whose own repeat spread is
# 10-40%, not a sampling confidence interval.
CI_TOLERANCE = 0.25
CI_SCAN_POINTS = 9
# The L2 blend must earn its two extra parameters on the regime where it can act.
MIN_CV_IMPROVEMENT_PCT = 1.0
MAX_FULL_CV_DEGRADATION_PCT = 1.0
# Timing scopes are physical-protocol identities: a batched interval and a
# per-call interval never share a launch intercept, so they never share a fit.
# ``block_interval_warm_operator`` is the attention worker's completion-clock
# scope: one API call per sample, elapsed wall time *including* host submit and
# output-allocation gaps.  It is a block interval, not a device-event interval,
# so it never shares an intercept (or a fit) with ``device_event_warm_operator``.
ACCEPTED_TIMING_SCOPES = (
    "device_event_warm",
    "device_event_warm_operator",
    "device_event_warm_per_call",
    "device_event_warm_batched",
    "device_event_warm_operator_batched_amortized",
    "device_event_warm_operator_per_call",
    "block_interval_warm_operator",
)
MIN_CACHE_PROBE_SAMPLES = 8
# Cache-probe subset: outcome-free, shape-only (working set fits the declared L2
# and the op is not compute-bound, so a residency split can change the answer).
CACHE_PROBE_MAX_INTENSITY = 32.0
# A rate is identified only when at least this many binding rows constrain it.
MIN_IDENTIFYING_ROWS = 3
MIN_IDENTIFYING_SHARE = 0.10
# Above this train-only CV WAPE a family fit is reported but flagged low_confidence.
MAX_TRUSTED_CV_WAPE_PCT = 25.0
MAX_TRUSTED_P90_APE_PCT = 60.0
ACCEPTED_SOURCE_CLASSES = ("measured",)
# family -> model kind.  Only families with measured train support are fitted;
# a family absent from the artifact is unmodeled, never extrapolated.
# Declared alias from an acceptance family name to a fitted family.  The A2 split
# declaration names its softmax family ``attention_softmax`` while the measured
# rows are ``torch_softmax`` at the same ``tokens x width`` geometry; the alias is
# explicit, recorded in the artifact and reported in every prediction reason.
FAMILY_ALIASES = {"attention_softmax": "softmax"}

# Family -> physics.  ``attention`` and ``grouped_gemm`` are their own kinds:
# an attention kernel is two batched GEMMs over a score matrix and a grouped
# matmul is one GEMM per expert, so neither is ever fitted or predicted as a
# dense GEMM (see ``ATTENTION_KINDS`` / ``GROUPED_KINDS`` and the tests).
MODEL_KINDS = {
    "gemm": "gemm",
    "copy": "vector",
    "add": "vector",
    "rmsnorm": "vector",
    "swiglu": "vector",
    "rope": "vector",
    "softmax": "vector",
    "masked_softmax": "vector",
    "quantize": "vector",
    "cast": "vector",
    "attention_fused": "attention",
    "attention_qk": "attention",
    "attention_pv": "attention",
    "moe_gate_up": "grouped_gemm",
    "moe_down": "grouped_gemm",
    "grouped_matmul": "grouped_gemm",
}
ATTENTION_KINDS = ("attention_fused", "attention_qk", "attention_pv")
GROUPED_KINDS = ("moe_gate_up", "moe_down", "grouped_matmul")
# Declared ingestion mapping: a campaign may label the same grouped kernel by the
# model component it stands in for.  ``dense_ffn_gate_up``/``dense_ffn_down`` are
# grouped matmuls over an experts x tokens_per_expert geometry, i.e. the same
# physics as ``moe_gate_up``/``moe_down``; the mapping is recorded in the
# artifact's provenance and never applied at prediction time.
CAMPAIGN_FAMILY_ALIASES = {
    "dense_ffn_gate_up": "moe_gate_up",
    "dense_ffn_down": "moe_down",
}
CUBE_KINDS = ("gemm", "attention", "grouped_gemm")

# Attention tile grid: q_tile x kv_tile.  ``dim`` (the PV contraction-free axis,
# i.e. the value width) is a property of the kernel, not of the shape, so it is
# a declared constant tile rather than a fitted one -- a 1024-wide dim tile
# would invent work that no fused-attention kernel executes.
ATTENTION_Q_TILES = (32, 64, 128, 256)
ATTENTION_KV_TILES = (64, 128, 256, 512, 1024)
ATTENTION_DIM_TILE = 128
# Per-ragged-tile tail cost in ms (a partial q or kv tile is rounded up to a
# full tile of extra cube work).  Bounded like the launch intercept.
ATTENTION_TAIL_BOUNDS_MS = (0.0, 0.060)
# Per-KV-token latency: a decode kernel walks the cache sequentially and pays a
# fixed cost per visited KV position, which a byte-rate model cannot see when the
# query row is a single token (measured: kv=32769 with 3 query heads takes 0.93 ms
# against 33.6 MB of KV bytes, i.e. 28 ns per visited position).  The coefficient is
# ``visible_kv``, so it stays well defined with or without a fitted tile grid.
# Forward-selected like every other tail.
ATTENTION_TOKEN_LATENCY_BOUNDS_MS = (0.0, 0.001)
# Workspace bytes per (query token x query head): the fused kernel keeps two
# fp32 row-statistic buffers of 8 elements (softmax max / sum), which is what the
# attention worker's ``allocation.auxiliaries`` reports (96 bytes each at q=1).
ATTENTION_WORKSPACE_BYTES_PER_QUERY_HEAD = 64.0
# KV-cache broadcast limit: one KV read can feed at most this many query heads.
# Beyond it the kernel re-reads the cache per query head, so the effective KV
# traffic is ``kv * max(kv_heads, ceil(q_heads / B)) * (qk_dim + v_dim)``.
# ``None`` is the declared base formula (one read per KV head); the alternatives
# are CV variants.  The A3 MLA decode case (64 query heads, 1 KV head) measures
# 9.84 ms against 142 MB of base KV bytes, which only the re-read explains.
ATTENTION_KV_BROADCAST_CHOICES = (None, 16, 8, 4)
# Grouped-matmul per-expert fixed cost (dispatch / padding / histogram walk),
# in ms per fitted expert-count scaling.  One list-API call launches one kernel
# (``single_list_call``), so an expert costs a few microseconds of serialization
# at most -- not a launch intercept, and not 60 us scaled over 384 experts.
GROUPED_PER_EXPERT_BOUNDS_MS = (0.0, 0.020)
# ``npu_grouped_matmul``'s tensorsList holds at most 128 entries, so a call with
# more experts is split into ceil(experts/128) sequential launches.  The model
# charges one fitted fixed cost per extra launch; the 128 cap is a declared API
# limit of the measured backend, not a tuned threshold.
GROUPED_LAUNCH_CAPACITY = 128
# Declared grouped weight-width classes.  The measured A3 corpus pools two model
# configurations of the same projection family: ``moe_intermediate_size=2048``
# (gate_up n=4096) and ``7168`` (n=14336).  They overlap compute and weight
# streaming differently -- an additive fit is exact on the wide class and needs a
# negative intercept on the narrow one -- so the split is offered to train-only
# CV as a model-form decision (adopted only above the variant margin).
GROUPED_NARROW_N = 8192
GROUPED_PER_LAUNCH_BOUNDS_MS = (0.0, 0.060)
# Measured fresh/hot per-call device dispatch on the A3 die
# (``.workstreams/bigops-a3/results/dispatch-overhead.json``:
# 19.8-20.2 us device gap around a 10-kernel batched region 11.7-11.9 us per
# kernel).  The 24.5 us figure this round replaced was a round-1 constant.
ATTENTION_DISPATCH_OVERHEAD_MS = 0.020
# Device-event scopes of the attention/grouped workers dispatch one native call
# per sample, so their intercept is bounded by the measured dispatch plus a
# margin.  The completion-clock block interval also carries the host/allocation
# gap (measured 119 us wall), so it keeps the wider declared launch bounds.
DISPATCH_LAUNCH_BOUNDS_MS = (0.0, 0.025)
# The 20 us dispatch prior is an A3 (Ascend910_9362) measurement.  A2's own
# measured per-call dispatch is ~52-75 us (``model-physical/report.md``), so the
# narrow bound is declared per hardware key and a part without an entry keeps the
# generic launch bounds.
DISPATCH_LAUNCH_BOUNDS_BY_HARDWARE = {"Ascend910_9362": DISPATCH_LAUNCH_BOUNDS_MS}
# The 20 us dispatch prior is per hardware key, exactly like the launch bound: a
# pin that borrows another SKU's measurement would import an A3 number into, say,
# an A5 artifact whose grouped corpus has fewer than MIN_IDENTIFYING_ROWS rows
# above the 128-expert cap.  A part without a measured dispatch does not pin.
DISPATCH_OVERHEAD_MS_BY_HARDWARE = {"Ascend910_9362": ATTENTION_DISPATCH_OVERHEAD_MS}
# A working set this many times the declared L2 capacity cannot be resident, so
# the L2 blend must not credit it.  The largest A3 training working set is
# 3.5x the declared L2, so the multi-GB weight-streaming regime of a decode
# GEMV is *unmeasured* and cannot be extrapolated from a warm fitted rate.
STREAMING_L2_RATIO = 8.0
# Measured sustained HBM rate to use when a family never measured the streaming
# regime: A3 512 MiB copy 1.249 TB/s (read+write) and graph-mode decode
# 1.01 TB/s effective (Qwen2.5-7B TP1, 14.21 GiB weights, 15.10 ms TPOT) ->
# 1.01e12 B/s.  A2's own corpus has no streaming row either, so it is declared
# from its campaign's measured decode band (`.workstreams/model-bigops/report.md`
# section 3.2: 471-955 GB/s, i.e. 0.59-1.19x its declared 800 GB/s peak); the low
# end 471e9 is chosen as the bound, the same conservative choice as A3's 1.01 TB/s
# (below the measured 1.249 TB/s copy).  Declared per hardware key: a part without
# a measurement keeps its fitted efficiency (priced at peak x eta, never at a
# borrowed number) instead of taking another part's rate.
STREAMING_BW_BYTES_PER_S = {"Ascend910_9362": 1.01e12, "Ascend910B4": 471e9}
# A weight tensor survives from one use to the next only if it is small relative
# to the stream that runs between the two uses.  A decode step streams at least
# ``STREAMING_L2_RATIO`` resident prefixes (that is the declared meaning of the
# constant above), so a tensor larger than one ratio-th of the resident prefix is
# evicted before it is read again: the L2 blend must not credit it.
STREAMING_WEIGHT_RESIDENT_SHARE = 1.0 / STREAMING_L2_RATIO
# Largest activation row count still counted as a single weight fetch.  A GEMM
# walks its activation in row tiles of the fitted ``block_m`` (32/64/128 in this
# artifact); at most a couple of tiles means every weight byte is fetched exactly
# once per call.  A wider m re-reads the weight out of L2 for the next tile, which
# is the reuse the blend legitimately models.
STREAMING_WEIGHT_MAX_M = 128.0
# The op's traffic must *be* the weight fetch: at least this many times more
# weight bytes than activation bytes.  A short-wide GEMM (k >> n) reads an
# activation as large as its weights, so its working set is not a bare stream.
STREAMING_WEIGHT_DOMINANCE = 4.0
# A grouped matmul streams its weight matrices from HBM and the declared
# ``peak_bw_bytes_per_s`` is a conservative SKU constant (1.2 TB/s on A3) while
# the big-operator block measures >= 1.36 TB/s sustained streaming.  The memory
# efficiency of a grouped family is therefore allowed above 1.0 -- an *effective
# bandwidth* parameter, reported as such, never silently absorbed.
GROUPED_ETA_MEMORY_MAX = 1.5
# Timing-scope families.  The same physical protocol is spelled differently by the
# different workers (the operator campaign, the attention worker, the big-operator
# worker), so a prediction that names one spelling may be answered by a fitted row
# of the same family -- never by the other family.  A batched interval amortises
# dispatch over ``repeats_per_sample`` calls and a per-call interval includes it,
# so crossing the two would be a different physical protocol, not a fallback.
SCOPE_FAMILY_PER_CALL = (
    "device_event_warm",
    "device_event_warm_operator",
    "device_event_warm_operator_per_call",
    "device_event_warm_per_call",
    "block_interval_warm_operator",
)
SCOPE_FAMILY_BATCHED = (
    "device_event_warm_batched",
    "device_event_warm_operator_batched_amortized",
)
# Declared attention topologies, classified from (q_heads, kv_heads) only:
#   mha       -- one KV head per query head (q_heads == kv_heads);
#   shared_kv -- a single shared KV head (absorbed MLA / MQA);
#   gqa       -- anything in between.
ATTENTION_TOPOLOGIES = ("mha", "shared_kv", "gqa")
ATTENTION_PHASES = ("prefill", "decode")
# Expert-count scaling exponents tried for the per-expert fixed cost.  A value
# of 1.0 is a linear per-expert cost; the grid lets the data select sub/super
# linear dispatch.  Selected on train-only CV with the same 1 pp margin.
GROUPED_EXPERT_SCALING_CHOICES = (0.5, 0.75, 1.0, 1.25)
# A byte-convention variant must buy this much train-only CV to replace the
# declared default convention.
MIN_VARIANT_IMPROVEMENT_PCT = 1.0
# Declared default: do the grouped-matmul bytes include the per-expert
# activation read?  Physically yes, but an implementation that walks the
# activation once (or keeps it resident) can hide it; the data decides.
GROUPED_DEFAULT_ACTIVATION_READ = True
# The MC2 campaign measures the fused decode chunk's grouped gate/up
# (``split_item`` 3) and down (``split_item`` 2) calls directly, on the *real*
# routing ``group_list`` of a 512..16384-token layer at ep 2/4/8.  That is its own
# measurement recipe: the prefill grouped corpus feeds one list call per sample
# over a *uniform* expert count (``split_item`` 0) and its fitted row answers the
# wide-token/wide-expert regime.  Fitting the two populations into one row makes
# neither regime right (the joint fit's out-of-fold WAPE is worse than either
# population alone), so the MC2 legs are ingested under this recipe and the cost
# model resolves an MC2 chunk's compute members to it (``a3_cost``'s declared
# ``PHYSICAL_MC2_MEASURED_RECIPES``).  The same kernel is still one physical
# family: nothing here turns the chain into a second model.
MC2_GROUPED_RECIPE = "npu_grouped_matmul_list_bf16_mc2"
# The MC2 legs are device-event medians around one native call per sample, i.e.
# the operator scope (never a batched/amortised or block-interval scope).
MC2_GROUPED_SCOPE = "device_event_warm_operator"
# Declared protocol of the MC2 grouped legs: one native ``npu_grouped_matmul``
# call inside a device-event window, warm, on preallocated buffers, bf16, ND
# weights, 2 trials of N samples.  It is written as the measurement declares it
# (``mc2_worker.py`` kind_gmm): the split-item and the group-list convention are
# part of the *recipe*, not of the timing protocol.
MC2_GROUPED_PROTOCOL = {
    "timer": "device_event",
    "preallocated": True,
    "compute_dtype": "bf16",
    "weight_dtype": "bf16",
    "activation_dtype": "bf16",
    "quantization": "none",
    "layout": "contiguous",
    "trials": 2,
}
#: A fused-chunk *member* is not a native call and not an expert group.  The
#: chunk's single launch is charged once by the composer
#: (``a3_cost.mc2_chunk_estimate``), and the deployed MC2 decomposition is one
#: member per local expert, so the composer sums per-member prices to get the
#: chunk's compute.  That sum is only the measured leg's cost when every member
#: term is *additive*: an intercept would be charged once per expert (32 x 25 us
#: on a 0.9 ms leg) and a super-linear ``experts ** gamma`` tax would be summed as
#: ``experts ** 1``.  A fused-chunk member row therefore declares a zero intercept
#: and no per-expert term; what it carries is the per-work physics (cube and
#: memory efficiency) that the deployed decomposition reproduces exactly.
#: The L2-residency blend is excluded for the same reason: a residency credit is
#: ``min(bytes, l2_usable)`` per call, so summing it over 32 members credits far
#: more than the call's own working set (measured on the published A2 row: the
#: per-member sum priced a 1.51 ms leg at 0.53 ms).  A fused-chunk member row is
#: therefore ``hbm``-only, and the whole-call residency question stays with the
#: families that measure a whole call.
FUSED_CHUNK_MEMBER_TERMS: dict[str, dict[str, Any]] = {
    MC2_GROUPED_RECIPE: {
        "launch_bounds_ms": (0.0, 0.0),
        "include_per_expert": False,
        "include_per_launch": False,
        "memory_models": ("hbm",),
    },
}
# The two MC2 compute legs, as the campaign times them: ``(timing_key, family,
# shape function)``.  ``gate_up`` produces ``2 x intermediate`` columns from
# ``hidden``; ``down`` produces ``hidden`` from ``intermediate``.
MC2_GROUPED_LEGS = (
    ("gmm_gate_up", "moe_gate_up", "gate_up"),
    ("gmm_down", "moe_down", "down"),
)
# A leg the campaign times but that is not part of the MC2 chunk's grouped
# compute identity: the separately measured activation.  It is extracted (so the
# report can state the chain's non-compute leg) and deliberately *not* ingested,
# because it is a vector kernel of the ``swiglu`` family and pricing it here
# would silently re-fit another family from a single campaign.
MC2_ACTIVATION_LEG = "swiglu"
# Declared byte width per compute dtype.  This module is stdlib-only by design,
# so the table is mirrored from ``a3_types._DTYPE_BYTES`` (the package's single
# source of truth) and a unit test asserts the two stay identical.  A dtype that
# is not declared here is a rejection with a reason -- never a silent bf16 width:
# an A5 fp8 GEMM priced at 2 bytes doubles ``hbm_bytes`` and the fitted
# ``eta_memory`` absorbs the error invisibly.
DTYPE_BYTES = {
    "fp32": 4.0, "float32": 4.0, "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0,
    "float16": 2.0, "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "int8": 1.0, "int4": 0.5, "int32": 4.0, "bool": 1.0,
}
# Quantized dtypes whose "bytes" are a packed fraction of a byte are not
# supported by the readers: an element that is not byte-aligned needs the
# caller's packing convention, so they are declared here for prediction only.
BYTE_ALIGNED_DTYPES = frozenset(
    name for name, width in DTYPE_BYTES.items() if width >= 1.0
)


def dtype_itemsize(dtype: Any, *, default: str = "") -> float:
    """Declared byte width of one dtype; raises instead of guessing.

    ``default`` is only for a caller that already knows the value is absent and
    wants the declared compute dtype substituted (the CLI predict path does).
    """
    text = _normalise_dtype(dtype) or _normalise_dtype(default)
    if not text:
        raise PhysicalCalibrationError("dtype_undeclared")
    width = DTYPE_BYTES.get(text)
    if width is None:
        raise PhysicalCalibrationError("dtype_byte_width_undeclared:%s" % (text,))
    return float(width)


class PhysicalCalibrationError(ValueError):
    """Raised for an inconsistent artifact, observation document or fit request."""


# ---------------------------------------------------------------------------
# Canonical identity helpers
# ---------------------------------------------------------------------------
def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return ("%.9g" % float(value))


def canonical_shape(shape: Mapping[str, Any]) -> str:
    """Stable text identity of a shape mapping; ``None`` entries are dropped.

    The attention axes (``q``/``kv``/``heads``/``dim``) and the grouped axes
    (``tokens``/``experts``) are part of the identity: two attention calls with
    the same q and kv but different head counts are different physical work, and
    so are a causal and a full-visible mask at the same geometry.
    """
    parts = []
    for key in ("m", "n", "k", "elements", "q", "kv", "heads", "kv_heads", "dim",
                "tokens", "experts"):
        if key in shape and shape[key] is not None:
            parts.append("%s=%s" % (key, _number(float(shape[key]))))
    if shape.get("causal") is not None:
        parts.append("causal=%d" % (1 if shape.get("causal") else 0))
    return ",".join(parts)


def observation_identity(family: str, shape: Mapping[str, Any]) -> str:
    """Physical identity used for heldout exclusion: ``family|canonical_shape``."""
    return "%s|%s" % (str(family), canonical_shape(shape))


def _stable_digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_float(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalCalibrationError("not_a_number:%s=%r" % (name, value))
    out = float(value)
    if not math.isfinite(out):
        raise PhysicalCalibrationError("not_finite:%s" % name)
    return out


def _ceil_div(total: float, width: float) -> int:
    if width <= 0:
        raise PhysicalCalibrationError("non_positive_tile")
    return max(1, int(math.ceil(max(0.0, total) / width)))


# ---------------------------------------------------------------------------
# Machine limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MachineLimits:
    """Per-die peak rates the model divides by; every field names its unit."""

    hardware_key: str
    arch_family: str
    aic_count: int
    aiv_count: int
    clock_mhz: float
    sustained_clock_mhz: float
    clock_source: str
    ops_per_cycle_per_aic: float
    peak_flops: float
    peak_bw_bytes_per_s: float
    l2_bytes: int
    vec_elements_per_cycle_per_aiv: float
    vec_peak_ops_per_s: float
    vec_max_ops_per_s: float
    evidence: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MachineLimits":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("machine_must_be_a_mapping")
        key = str(value.get("hardware_key") or "")
        arch = str(value.get("arch_family") or "")
        if value.get("clock_mhz") is not None:
            clock_mhz = _as_float(value.get("clock_mhz"), "clock_mhz")
        elif value.get("clock_hz") is not None:
            clock_mhz = _as_float(value.get("clock_hz"), "clock_hz") / 1e6
        else:
            raise PhysicalCalibrationError("machine_missing_clock")
        # The sustained (under-load) clock governs a warm repeated-call interval.
        sustained = _as_float(
            value.get("sustained_clock_mhz"), "sustained_clock_mhz", allow_none=True
        )
        if sustained is None:
            sustained = clock_mhz
            clock_source = "rated"
        else:
            clock_source = "sustained"
        declared_source = str(value.get("clock_source") or "")
        if declared_source:
            clock_source = declared_source
        if sustained > clock_mhz * 1.001:
            raise PhysicalCalibrationError("sustained_clock_above_rated")
        aic = int(_as_float(value.get("aic_count"), "aic_count"))
        aiv = int(_as_float(value.get("aiv_count", 0), "aiv_count", allow_none=True) or 0)
        ops_per_cycle = _as_float(value.get("ops_per_cycle"), "ops_per_cycle")
        declared_peak = _as_float(value.get("peak_flops"), "peak_flops", allow_none=True)
        peak = aic * ops_per_cycle * sustained * 1e6
        if declared_peak is not None and abs(declared_peak - peak) > 1e-3 * peak:
            raise PhysicalCalibrationError(
                "peak_flops_contradicts_components: declared=%r derivable=%r" % (declared_peak, peak)
            )
        bw = _as_float(value.get("peak_bw_bytes_per_s"), "peak_bw_bytes_per_s")
        if bw <= 0:
            raise PhysicalCalibrationError("non_positive_peak_bw")
        l2 = int(_as_float(value.get("l2_bytes"), "l2_bytes"))
        lanes = _as_float(
            value.get("vec_elements_per_cycle_per_aiv"), "vec_elements_per_cycle_per_aiv", allow_none=True
        )
        vec_peak = _as_float(value.get("vec_peak_ops_per_s"), "vec_peak_ops_per_s", allow_none=True)
        if lanes is None and vec_peak is not None and aiv > 0:
            lanes = vec_peak / (float(aiv) * sustained * 1e6)
        if lanes is None or aiv <= 0:
            raise PhysicalCalibrationError("vec_peak_underivable")
        vec_peak = float(aiv) * lanes * sustained * 1e6
        vec_max = float(aiv) * VEC_ISSUE_MAX_PER_AIV * sustained * 1e6
        declared_vec_max = _as_float(value.get("vec_max_ops_per_s"), "vec_max_ops_per_s", allow_none=True)
        if declared_vec_max is not None and declared_vec_max < vec_peak:
            raise PhysicalCalibrationError("vec_max_below_nominal_peak")
        evidence = value.get("evidence") or ()
        rows: list[tuple[str, str]] = []
        if isinstance(evidence, Mapping):
            rows = [(str(k), str(v)) for k, v in sorted(evidence.items())]
        elif isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            for item in evidence:
                if isinstance(item, Mapping):
                    rows.append((str(item.get("field")), str(item.get("source"))))
                elif isinstance(item, Sequence) and len(item) == 2:
                    rows.append((str(item[0]), str(item[1])))
        return cls(
            hardware_key=key,
            arch_family=arch,
            aic_count=aic,
            aiv_count=aiv,
            clock_mhz=clock_mhz,
            sustained_clock_mhz=float(sustained),
            clock_source=clock_source,
            ops_per_cycle_per_aic=ops_per_cycle,
            peak_flops=peak,
            peak_bw_bytes_per_s=bw,
            l2_bytes=l2,
            vec_elements_per_cycle_per_aiv=float(lanes),
            vec_peak_ops_per_s=float(vec_peak),
            vec_max_ops_per_s=float(max(vec_max, declared_vec_max or 0.0)),
            evidence=tuple(rows),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hardware_key": self.hardware_key,
            "arch_family": self.arch_family,
            "aic_count": self.aic_count,
            "aiv_count": self.aiv_count,
            "clock_mhz": self.clock_mhz,
            "sustained_clock_mhz": self.sustained_clock_mhz,
            "clock_source": self.clock_source,
            "ops_per_cycle": self.ops_per_cycle_per_aic,
            "peak_flops": self.peak_flops,
            "peak_bw_bytes_per_s": self.peak_bw_bytes_per_s,
            "l2_bytes": self.l2_bytes,
            "vec_elements_per_cycle_per_aiv": self.vec_elements_per_cycle_per_aiv,
            "vec_peak_ops_per_s": self.vec_peak_ops_per_s,
            "vec_max_ops_per_s": self.vec_max_ops_per_s,
        }
        if self.evidence:
            out["evidence"] = [{"field": f, "source": s} for f, s in self.evidence]
        return out


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Observation:
    """One measured physical condition, already reduced to a target duration."""

    condition_key: str
    hardware_key: str
    arch_family: str
    family: str
    recipe: str
    protocol_signature: str
    split: str
    duration_ms: float
    flops: float
    hbm_bytes: float
    elements: float
    shape: Mapping[str, Any]
    dtype: str
    timing_scope: str
    source_class: str
    die: str
    run_id: str
    raw_event_count: int
    model_kind: str
    target_estimator: str = ""
    protocol_core_signature: str = ""
    package_tree: str = ""

    @property
    def identity(self) -> str:
        return observation_identity(self.family, self.shape)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, model_kind: str | None = None) -> "Observation":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("observation_must_be_a_mapping")
        family = str(value.get("family") or "").strip()
        if not family:
            raise PhysicalCalibrationError("observation_missing_family")
        kind = model_kind or MODEL_KINDS.get(family, "")
        shape = value.get("shape") or {}
        if not isinstance(shape, Mapping):
            raise PhysicalCalibrationError("observation_shape_must_be_a_mapping")
        shape = {k: v for k, v in shape.items() if v is not None}
        raw = value.get("raw_samples_ms") or ()
        raw_count = value.get("raw_event_count")
        if raw_count is None:
            raw_count = len(raw) if isinstance(raw, Sequence) else 0
        duration = _as_float(value.get("duration_ms"), "duration_ms")
        estimator = str(value.get("target_estimator") or "")
        trials = value.get("trial_medians_ms")
        if isinstance(trials, Sequence) and not isinstance(trials, (str, bytes)) and len(trials):
            medians = sorted(float(item) for item in trials if item is not None)
            if medians:
                # Acceptance target is the median of per-trial event medians; the
                # pooled repeat median is a weaker estimator of the same quantity.
                duration = medians[len(medians) // 2] if len(medians) % 2 else (
                    0.5 * (medians[len(medians) // 2 - 1] + medians[len(medians) // 2])
                )
                estimator = "median_of_trial_medians"
        if not estimator:
            estimator = "declared_duration_ms"
        if duration is None or duration <= 0:
            raise PhysicalCalibrationError("observation_duration_not_positive")
        flops = _as_float(value.get("flops"), "flops", allow_none=True) or 0.0
        hbm_bytes = _as_float(value.get("hbm_bytes"), "hbm_bytes", allow_none=True) or 0.0
        elements = _as_float(value.get("elements"), "elements", allow_none=True) or 0.0
        if kind == "gemm" and not (flops > 0 and hbm_bytes > 0):
            raise PhysicalCalibrationError("gemm_observation_missing_flops_or_bytes")
        if kind == "attention":
            if not (flops > 0 and hbm_bytes > 0):
                raise PhysicalCalibrationError("attention_observation_missing_flops_or_bytes")
            if not (float(shape.get("q") or 0) > 0 and float(shape.get("kv") or 0) > 0):
                raise PhysicalCalibrationError("attention_observation_missing_q_kv")
        if kind == "grouped_gemm":
            if not (flops > 0 and hbm_bytes > 0):
                raise PhysicalCalibrationError("grouped_observation_missing_flops_or_bytes")
            if not (
                float(shape.get("tokens") or 0) > 0
                and float(shape.get("experts") or 0) > 0
                and float(shape.get("n") or 0) > 0
                and float(shape.get("k") or 0) > 0
            ):
                raise PhysicalCalibrationError("grouped_observation_missing_tokens_experts_n_k")
        if kind == "vector" and not (elements > 0 or hbm_bytes > 0):
            raise PhysicalCalibrationError("vector_observation_missing_elements_and_bytes")
        declared_dtype = str(value.get("dtype") or value.get("compute_dtype") or "")
        if declared_dtype and _normalise_dtype(declared_dtype) not in DTYPE_BYTES:
            # The byte width of the shape is part of the observation, so an
            # undeclared dtype is rejected here instead of being priced at bf16.
            raise PhysicalCalibrationError(
                "dtype_byte_width_undeclared:%s" % (_normalise_dtype(declared_dtype),)
            )
        return cls(
            condition_key=str(value.get("condition_key") or ""),
            hardware_key=str(value.get("hardware_key") or ""),
            arch_family=str(value.get("arch_family") or ""),
            family=family,
            recipe=str(value.get("recipe") or ""),
            protocol_signature=str(value.get("protocol_signature") or ""),
            split=str(value.get("split") or ""),
            duration_ms=float(duration),
            flops=flops,
            hbm_bytes=hbm_bytes,
            elements=elements,
            shape=dict(shape),
            dtype=str(value.get("dtype") or value.get("compute_dtype") or ""),
            timing_scope=str(value.get("timing_scope") or ""),
            source_class=str(value.get("source_class") or ""),
            die=str(value.get("die") or ""),
            run_id=str(value.get("run_id") or ""),
            raw_event_count=int(raw_count or 0),
            model_kind=kind,
            target_estimator=estimator,
            protocol_core_signature=str(
                value.get("protocol_core_signature")
                or protocol_core_signature(value.get("protocol") or {})
                or str(value.get("protocol_signature") or "")
            ),
            package_tree=str(value.get("package_tree") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_key": self.condition_key,
            "hardware_key": self.hardware_key,
            "die": self.die,
            "arch_family": self.arch_family,
            "family": self.family,
            "recipe": self.recipe,
            "protocol_signature": self.protocol_signature,
            "run_id": self.run_id,
            "split": self.split,
            "shape": dict(self.shape),
            "elements": self.elements or None,
            "flops": self.flops,
            "hbm_bytes": self.hbm_bytes,
            "duration_ms": self.duration_ms,
            "dtype": self.dtype,
            "source_class": self.source_class,
            "timing_scope": self.timing_scope,
            "raw_event_count": self.raw_event_count,
            "target_estimator": self.target_estimator,
            "protocol_core_signature": self.protocol_core_signature,
            "package_tree": self.package_tree,
        }


# ---------------------------------------------------------------------------
# Heldout guard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeldoutGuard:
    """Declared heldout conditions that must never enter a fit."""

    condition_keys: frozenset[str]
    identities: frozenset[str]
    sources: tuple[str, ...]
    declared_conditions: int

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> "HeldoutGuard":
        keys: set[str] = set()
        identities: set[str] = set()
        used: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            value = json.loads(path.read_text())
            found_keys, found_ids = _extract_heldout(value)
            if not found_ids and not found_keys:
                continue
            keys |= found_keys
            identities |= found_ids
            used.append(str(path))
        return cls(
            condition_keys=frozenset(keys),
            identities=frozenset(identities),
            sources=tuple(used),
            declared_conditions=len(identities | {k for k in keys}),
        )

    def overlap(self, observations: Sequence[Observation]) -> list[str]:
        hits = []
        for obs in observations:
            if obs.identity in self.identities or (obs.condition_key and obs.condition_key in self.condition_keys):
                hits.append("%s|%s" % (obs.family, canonical_shape(obs.shape)))
        return hits

    def to_dict(self, *, status: str, overlap: Sequence[str] = ()) -> dict[str, Any]:
        return {
            "status": status,
            "declared_conditions": self.declared_conditions,
            "identity_count": len(self.identities),
            "condition_key_count": len(self.condition_keys),
            "sources": list(self.sources),
            "overlap": list(overlap),
        }


def _extract_heldout(value: Any) -> tuple[set[str], set[str]]:
    """Collect heldout condition keys and physical identities from known schemas."""
    keys: set[str] = set()
    identities: set[str] = set()

    def take(family: Any, shape: Any, *key_values: Any) -> None:
        for item in key_values:
            if isinstance(item, str) and item:
                keys.add(item)
        if family and isinstance(shape, Mapping):
            identities.add(observation_identity(str(family), shape))

    if not isinstance(value, Mapping):
        return keys, identities
    # Frozen score rows: one row per (condition, variant).
    for row in value.get("rows") or ():
        if isinstance(row, Mapping):
            take(row.get("family"), row.get("physical_shape"), row.get("condition_id"))
    # Split declarations: families[].heldout[], new_heldout.points[], cross_sku_matched[].
    for family_block in value.get("families") or ():
        if isinstance(family_block, Mapping):
            name = family_block.get("family")
            if isinstance(name, str):
                take(name, None)
            for point in family_block.get("heldout") or ():
                if isinstance(point, Mapping):
                    take(point.get("family") or family_block.get("family"), point.get("shape"),
                         point.get("point_id"), point.get("geometry_key"))
    for section in ("new_heldout", "gap_fill_train"):
        block = value.get(section)
        if isinstance(block, Mapping):
            for point in block.get("points") or ():
                if isinstance(point, Mapping) and str(point.get("split") or "heldout") == "heldout":
                    take(point.get("family"), point.get("shape"), point.get("point_id"),
                         point.get("geometry_key"))
    for point in value.get("cross_sku_matched") or ():
        if isinstance(point, Mapping):
            take(point.get("family"), point.get("shape"), point.get("a3_geometry_key"),
                 point.get("a2_geometry_key"))
    # Campaign result files: cases[] / results[] with a request block.
    for case in value.get("cases") or ():
        if isinstance(case, Mapping):
            request = case.get("request") if isinstance(case.get("request"), Mapping) else case
            take(request.get("op_family") or request.get("family"), request.get("shape"),
                 request.get("case_id"), case.get("case_id"))
    for result in value.get("results") or ():
        if isinstance(result, Mapping):
            request = result.get("request") if isinstance(result.get("request"), Mapping) else result
            take(request.get("op_family") or request.get("family"),
                 request.get("shape") or result.get("shape"),
                 request.get("case_id"), result.get("case_id"))
    return keys, identities


# ---------------------------------------------------------------------------
# Family parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilyParameters:
    """Fitted parameters of one ``(family, recipe, protocol)`` model."""

    family: str
    recipe: str
    protocol_signature: str
    timing_scope: str
    model_kind: str
    model_form: str
    fit_level: str
    block_m: int | None
    block_n: int | None
    launch_ms: float
    eta_compute: float | None
    eta_memory: float | None
    vec_ops_per_s: float | None
    memory_model: str
    l2_usable_bytes: int
    l2_bw_bytes_per_s: float | None
    compute_dtype: str
    samples: int
    raw_event_count: int
    train_wape_pct: float
    cv_wape_pct: float | None
    median_ape_pct: float
    p90_ape_pct: float
    max_ape_pct: float
    protocol_core_signature: str = ""
    launch_ms_by_scope: Mapping[str, float] = field(default_factory=dict)
    scope_samples: Mapping[str, int] = field(default_factory=dict)
    tail: Mapping[str, bool] = field(default_factory=dict)
    k_fill: bool = False
    k_fill_ms: float | None = None
    align_penalty: float | None = None
    # Attention: per-ragged-tile tail cost (q/kv alignment) in ms.
    tail_tile_ms: float | None = None
    # Attention: per-KV-token latency (sequential cache walk) in ms.
    kv_token_ms: float | None = None
    # Grouped matmul: per-expert fixed cost in ms and the fitted expert-count
    # scaling exponent it is charged with (``experts ** expert_scaling``).
    per_expert_ms: float | None = None
    expert_scaling: float | None = None
    # Grouped matmul byte convention actually used by the fit.
    activation_read: bool = False
    # Grouped matmul: fixed cost per extra sequential launch (experts/128 cap).
    per_launch_ms: float | None = None
    # Attention: which declared topology this row answers for ('' when pooled).
    topology: str = ""
    # Grouped matmul: which declared weight-width class this row answers for.
    corpus_class: str = ""
    # Set on a returned row when the requested timing scope was answered by a
    # different spelling inside the same scope family.
    scope_alias_from: str = ""
    scope_alias_to: str = ""
    # Attention: KV-cache broadcast limit (None = one read per KV head).
    kv_broadcast: int | None = None
    # Whether the fitted corpus contains a row in the multi-GB streaming regime.
    # When it does not, a streaming prediction uses the declared measured rate
    # instead of extrapolating the (warm) fitted efficiency.
    streaming_measured: bool = True
    binding_share: Mapping[str, float] = field(default_factory=dict)
    binding_counts: Mapping[str, int] = field(default_factory=dict)
    identified: Mapping[str, bool] = field(default_factory=dict)
    pinned_parameters: tuple[str, ...] = ()
    intervals: Mapping[str, Any] = field(default_factory=dict)
    breakdown: Mapping[str, Any] = field(default_factory=dict)
    model_selection: Mapping[str, Any] = field(default_factory=dict)
    memory_model_selection: Mapping[str, Any] = field(default_factory=dict)
    quality: str = "trusted"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family,
            "recipe": self.recipe,
            "protocol_signature": self.protocol_signature,
            "protocol_core_signature": self.protocol_core_signature,
            "timing_scope": self.timing_scope,
            "launch_ms_by_scope": dict(self.launch_ms_by_scope),
            "scope_samples": dict(self.scope_samples),
            "model_kind": self.model_kind,
            "model_form": self.model_form,
            "tail": dict(self.tail),
            "k_fill": self.k_fill,
            "k_fill_ms": self.k_fill_ms,
            "align_penalty": self.align_penalty,
            "tail_tile_ms": self.tail_tile_ms,
            "kv_token_ms": self.kv_token_ms,
            "per_expert_ms": self.per_expert_ms,
            "expert_scaling": self.expert_scaling,
            "activation_read": self.activation_read,
            "per_launch_ms": self.per_launch_ms,
            "topology": self.topology,
            "corpus_class": self.corpus_class,
            "scope_alias_from": self.scope_alias_from,
            "scope_alias_to": self.scope_alias_to,
            "streaming_measured": self.streaming_measured,
            "kv_broadcast": self.kv_broadcast,
            "fit_level": self.fit_level,
            "block_m": self.block_m,
            "block_n": self.block_n,
            "launch_ms": self.launch_ms,
            "eta_compute": self.eta_compute,
            "eta_memory": self.eta_memory,
            "vec_ops_per_s": self.vec_ops_per_s,
            "memory_model": self.memory_model,
            "l2_usable_bytes": self.l2_usable_bytes,
            "l2_bw_bytes_per_s": self.l2_bw_bytes_per_s,
            "compute_dtype": self.compute_dtype,
            "samples": self.samples,
            "raw_event_count": self.raw_event_count,
            "train_wape_pct": self.train_wape_pct,
            "cv_wape_pct": self.cv_wape_pct,
            "median_ape_pct": self.median_ape_pct,
            "p90_ape_pct": self.p90_ape_pct,
            "max_ape_pct": self.max_ape_pct,
            "binding_share": dict(self.binding_share),
            "binding_counts": dict(self.binding_counts),
            "identified": dict(self.identified),
            "pinned_parameters": list(self.pinned_parameters),
            "intervals": {key: list(value) for key, value in self.intervals.items()},
            "breakdown": {
                key: (dict(value) if isinstance(value, Mapping) else value)
                for key, value in self.breakdown.items()
            },
            "model_selection": dict(self.model_selection),
            "memory_model_selection": dict(self.memory_model_selection),
            "quality": self.quality,
            "warnings": list(self.warnings),
        }
        return out

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FamilyParameters":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("family_parameters_must_be_a_mapping")
        try:
            return cls(
                family=str(value["family"]),
                recipe=str(value.get("recipe") or ""),
                protocol_signature=str(value.get("protocol_signature") or ""),
                protocol_core_signature=str(value.get("protocol_core_signature") or ""),
                timing_scope=str(value.get("timing_scope") or ""),
                launch_ms_by_scope={
                    str(k): float(v) for k, v in (value.get("launch_ms_by_scope") or {}).items()
                },
                scope_samples={
                    str(k): int(v) for k, v in (value.get("scope_samples") or {}).items()
                },
                model_kind=str(value["model_kind"]),
                model_form=str(value.get("model_form") or "wave_max"),
                tail={str(k): bool(v) for k, v in (value.get("tail") or {}).items()},
                k_fill=bool(value.get("k_fill") or False),
                k_fill_ms=None if value.get("k_fill_ms") is None else float(value["k_fill_ms"]),
                align_penalty=(
                    None if value.get("align_penalty") is None else float(value["align_penalty"])
                ),
                tail_tile_ms=(
                    None if value.get("tail_tile_ms") is None else float(value["tail_tile_ms"])
                ),
                kv_token_ms=(
                    None if value.get("kv_token_ms") is None else float(value["kv_token_ms"])
                ),
                per_expert_ms=(
                    None if value.get("per_expert_ms") is None else float(value["per_expert_ms"])
                ),
                expert_scaling=(
                    None if value.get("expert_scaling") is None else float(value["expert_scaling"])
                ),
                activation_read=bool(value.get("activation_read") or False),
                per_launch_ms=(
                    None if value.get("per_launch_ms") is None else float(value["per_launch_ms"])
                ),
                topology=str(value.get("topology") or ""),
                corpus_class=str(value.get("corpus_class") or ""),
                scope_alias_from=str(value.get("scope_alias_from") or ""),
                scope_alias_to=str(value.get("scope_alias_to") or ""),
                streaming_measured=bool(value.get("streaming_measured", True)),
                kv_broadcast=(
                    None if value.get("kv_broadcast") is None else int(value["kv_broadcast"])
                ),
                fit_level=str(value["fit_level"]),
                block_m=None if value.get("block_m") is None else int(value["block_m"]),
                block_n=None if value.get("block_n") is None else int(value["block_n"]),
                launch_ms=float(value["launch_ms"]),
                eta_compute=None if value.get("eta_compute") is None else float(value["eta_compute"]),
                eta_memory=None if value.get("eta_memory") is None else float(value["eta_memory"]),
                vec_ops_per_s=(
                    None if value.get("vec_ops_per_s") is None else float(value["vec_ops_per_s"])
                ),
                memory_model=str(value["memory_model"]),
                l2_usable_bytes=int(value.get("l2_usable_bytes") or 0),
                l2_bw_bytes_per_s=(
                    None if value.get("l2_bw_bytes_per_s") is None else float(value["l2_bw_bytes_per_s"])
                ),
                compute_dtype=str(value.get("compute_dtype") or "bf16"),
                samples=int(value.get("samples") or 0),
                raw_event_count=int(value.get("raw_event_count") or 0),
                train_wape_pct=float(value.get("train_wape_pct") or 0.0),
                cv_wape_pct=None if value.get("cv_wape_pct") is None else float(value["cv_wape_pct"]),
                median_ape_pct=float(value.get("median_ape_pct") or 0.0),
                p90_ape_pct=float(value.get("p90_ape_pct") or 0.0),
                max_ape_pct=float(value.get("max_ape_pct") or 0.0),
                binding_share=dict(value.get("binding_share") or {}),
                binding_counts={str(k): int(v) for k, v in (value.get("binding_counts") or {}).items()},
                identified=dict(value.get("identified") or {}),
                pinned_parameters=tuple(value.get("pinned_parameters") or ()),
                intervals=dict(value.get("intervals") or {}),
                breakdown=dict(value.get("breakdown") or {}),
                model_selection=dict(value.get("model_selection") or {}),
                memory_model_selection=dict(value.get("memory_model_selection") or {}),
                quality=str(value.get("quality") or "trusted"),
                warnings=tuple(value.get("warnings") or ()),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PhysicalCalibrationError("malformed_family_parameters:%s" % (exc,)) from exc


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpPrediction:
    """One predicted operator duration plus the reason it exists (or does not)."""

    duration_ms: float | None
    status: str
    reason: str
    family: str
    recipe: str
    protocol_signature: str = ""
    timing_scope: str = ""
    model_form: str = ""
    fit_level: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    breakdown: Mapping[str, float] = field(default_factory=dict)
    machine_source: str = ""
    machine_fields_from_artifact: tuple[str, ...] = ()

    @property
    def modeled(self) -> bool:
        return self.duration_ms is not None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "family": self.family,
            "recipe": self.recipe,
            "protocol_signature": self.protocol_signature,
            "timing_scope": self.timing_scope,
            "model_form": self.model_form,
            "fit_level": self.fit_level,
            "machine_source": self.machine_source,
        }
        if self.parameters:
            out["parameters"] = dict(self.parameters)
        if self.breakdown:
            out["breakdown"] = dict(self.breakdown)
        if self.machine_fields_from_artifact:
            out["machine_fields_from_artifact"] = list(self.machine_fields_from_artifact)
        return out


# ---------------------------------------------------------------------------
# Fitting machinery
# ---------------------------------------------------------------------------
def _gemm_geometry(obs: Observation, block_m: int, block_n: int, aic: int) -> tuple[int, int, float]:
    m = float(obs.shape.get("m") or 0.0)
    n = float(obs.shape.get("n") or 0.0)
    blocks = _ceil_div(m, block_m) * _ceil_div(n, block_n)
    waves = max(1, _ceil_div(blocks, aic))
    return blocks, waves, blocks / float(waves * max(1, aic))


def _memory_split(total_bytes: float, l2_usable: int, peak_bw: float) -> tuple[float, float]:
    """(hbm_part, l2_part) in bytes for the declared residency split."""
    if l2_usable <= 0:
        return total_bytes, 0.0
    l2 = min(total_bytes, float(l2_usable))
    return max(0.0, total_bytes - l2), l2


def streaming_regime(hbm_bytes: float, l2_bytes: int) -> bool:
    """True when a working set is far too large to be L2-resident."""
    return l2_bytes > 0 and hbm_bytes >= STREAMING_L2_RATIO * float(l2_bytes)


def weight_stream_basis(
    *,
    model_kind: str,
    m: float,
    k: float,
    n: float,
    itemsize: float,
    l2_usable_bytes: int,
    l2_bytes: int = 0,
) -> dict[str, Any]:
    """Declared test: does this op fetch its weight from HBM once and lose it?

    A dense GEMM that streams its weight has three shape-visible properties, and
    ``basis`` names the one that decided the answer:

    * ``m <= STREAMING_WEIGHT_MAX_M`` -- the activation is at most a couple of row
      tiles, so each weight byte is fetched exactly once per call;
    * ``weight_bytes >= STREAMING_WEIGHT_DOMINANCE x activation_bytes`` -- the
      call's traffic *is* the weight fetch, not an activation walk that could
      legitimately stay hot;
    * ``weight_bytes >= resident_prefix x STREAMING_WEIGHT_RESIDENT_SHARE`` -- the
      tensor cannot survive the per-token reuse distance, so no part of it is
      resident when the call runs (see the constant).

    The inputs are the shape, the byte counts and this module's declared
    constants -- never a model name, a family name, a recipe or a measured E2E
    number.  ``resident_prefix`` is the family's fitted ``l2_usable_bytes`` when
    it has one (that is the capacity the blend believes in) and the declared L2
    otherwise.

    Returns ``streamed``, the two byte counts, the residency threshold and the
    basis string.  A non-GEMM op is never a weight stream: its family either has
    no L2 blend (``l2_usable_bytes == 0`` in both fitted artifacts) or its memory
    is a KV/activation walk, which the blend models.
    """
    weight_bytes = max(0.0, float(k)) * max(0.0, float(n)) * float(itemsize)
    activation_bytes = max(0.0, float(m)) * max(0.0, float(k)) * float(itemsize)
    prefix = float(l2_usable_bytes or 0) or float(l2_bytes or 0)
    threshold = prefix * STREAMING_WEIGHT_RESIDENT_SHARE
    basis = {
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "resident_threshold_bytes": threshold,
        "streamed": False,
        "basis": "not_a_weight_stream",
    }
    if str(model_kind) != "gemm" or weight_bytes <= 0.0 or threshold <= 0.0:
        return basis
    if float(m) > STREAMING_WEIGHT_MAX_M:
        basis["basis"] = "activation_reuse:m=%.0f>%.0f" % (m, STREAMING_WEIGHT_MAX_M)
        return basis
    if weight_bytes < STREAMING_WEIGHT_DOMINANCE * activation_bytes:
        basis["basis"] = "activation_dominated:weights=%.1fMiB<%.0fxact=%.1fMiB" % (
            weight_bytes / 2**20, STREAMING_WEIGHT_DOMINANCE, activation_bytes / 2**20,
        )
        return basis
    if weight_bytes < threshold:
        basis["basis"] = "resident:weights=%.1fMiB<threshold=%.1fMiB" % (
            weight_bytes / 2**20, threshold / 2**20,
        )
        return basis
    basis["streamed"] = True
    basis["basis"] = "weight_stream:weights=%.1fMiB>=threshold=%.1fMiB,m=%.0f<=%.0f,act=%.2fMiB" % (
        weight_bytes / 2**20, threshold / 2**20, m, STREAMING_WEIGHT_MAX_M,
        activation_bytes / 2**20,
    )
    return basis



def grouped_shape_class(shape: Mapping[str, Any]) -> str:
    """Declared weight-width class of one grouped matmul (shape-only)."""
    n = float(shape.get("n") or 0.0)
    return "narrow" if n <= GROUPED_NARROW_N else "wide"


def _short_k(obs: Observation, threshold: float = K_FILL_MAX_K) -> bool:
    """True when a GEMM's k is short enough to pay a pipeline fill (W3)."""
    if obs.model_kind != "gemm":
        return False
    k = float(obs.shape.get("k") or 0.0)
    return 0.0 < k <= threshold


def _padded_flops(obs: Observation, block_m: int | None, tail: Mapping[str, bool]) -> float:
    """FLOPs the cube executes once a partial last row tile is padded.

    Dense GEMM pads the m axis; a grouped matmul pads *each expert's* token
    block, so the padded row count is ``experts x ceil(tokens_per_expert/m_tile)``.
    """
    n = float(obs.shape.get("n") or 0.0)
    k = float(obs.shape.get("k") or 0.0)
    if obs.model_kind == "grouped_gemm":
        tokens = float(obs.shape.get("tokens") or 0.0)
        experts = float(obs.shape.get("experts") or 0.0)
        if tokens <= 0 or experts <= 0 or n <= 0 or k <= 0:
            return obs.flops
        tokens_per_expert = tokens / experts
        if tail.get("m") and block_m:
            rows = experts * _ceil_div(tokens_per_expert, block_m) * block_m
        else:
            rows = tokens
        return 2.0 * rows * n * k
    m = float(obs.shape.get("m") or 0.0)
    if m <= 0 or n <= 0 or k <= 0:
        return obs.flops
    rows = _ceil_div(m, block_m) * block_m if (tail.get("m") and block_m) else m
    return 2.0 * rows * n * k


def _misaligned(obs: Observation, block_n: int | None) -> bool:
    """True when the column tile does not divide n (memory-side tail)."""
    if not block_n:
        return False
    n = float(obs.shape.get("n") or 0.0)
    if n <= 0:
        return False
    return int(n) % int(block_n) != 0


def attention_visible_kv(q: float, kv: float, causal: bool) -> float:
    """Mean number of keys one query row can see under the declared mask.

    Causal (bottom-right) attention with ``kv >= q`` leaves the prior context
    fully visible and only the in-chunk part triangular:
    ``sum_i (kv - q + i + 1) = q * (kv - (q-1)/2)``, so the per-query mean is
    ``kv - (q-1)/2``.  Decode (``q == 1``) sees the whole cache, and a
    full-visible mask sees the whole cache, so both reduce to ``kv``.
    """
    if kv <= 0:
        return 0.0
    if not causal or q <= 1.0 or kv < q:
        return float(kv)
    return float(max(1.0, kv - (q - 1.0) / 2.0))


def scope_family(scope: str) -> str:
    """Declared timing-scope family of one scope spelling ('' when unknown)."""
    name = str(scope or "")
    if name in SCOPE_FAMILY_PER_CALL:
        return "per_call"
    if name in SCOPE_FAMILY_BATCHED or "batched" in name or "amortized" in name:
        return "batched"
    return ""


def scope_fallback_order(scope: str) -> tuple[str, ...]:
    """Within-family preference when the exact scope has no fitted row.

    A device-event interval is the kernel's own time; the completion-clock block
    interval also carries the host submit and output-allocation gap, so it is the
    last resort inside the per-call family.
    """
    family = scope_family(scope)
    if family == "per_call":
        return tuple(
            item for item in SCOPE_FAMILY_PER_CALL if "block_interval" not in item
        ) + ("block_interval_warm_operator",)
    if family == "batched":
        return SCOPE_FAMILY_BATCHED
    return ()


def attention_kv_heads_effective(q_heads: float, kv_heads: float, broadcast: int | None) -> float:
    """KV re-read multiplicity for one declared broadcast limit (None = base)."""
    heads = max(1.0, float(q_heads or 0.0))
    kv_heads = max(1.0, float(kv_heads or 0.0))
    if not broadcast:
        return kv_heads
    return float(max(kv_heads, math.ceil(heads / float(broadcast))))


def attention_kv_extra_bytes(
    shape: Mapping[str, Any], broadcast: int | None, itemsize: float
) -> float:
    """Extra KV-cache bytes the kernel re-reads beyond the declared base."""
    if not broadcast:
        return 0.0
    q_heads = float(shape.get("heads") or 0.0)
    kv_heads = float(shape.get("kv_heads") or q_heads or 0.0)
    effective = attention_kv_heads_effective(q_heads, kv_heads, broadcast)
    extra_heads = effective - max(1.0, kv_heads)
    if extra_heads <= 0:
        return 0.0
    kv = float(shape.get("kv") or 0.0)
    dim = float(shape.get("dim") or 0.0)
    v_dim = float(shape.get("v_dim") or dim)
    return extra_heads * kv * (dim + v_dim) * itemsize


def attention_topology(q_heads: float, kv_heads: float) -> str:
    """Declared attention topology from the head geometry alone.

    An absorbed-MLA / MQA call shares one KV head across every query head
    (``kv_heads == 1``); a multi-head call reads one KV head per query head
    (``q_heads == kv_heads``); everything in between is grouped-query attention.
    The classification depends only on the shape, never on an outcome.
    """
    q = max(1.0, float(q_heads or 0.0))
    kv = max(1.0, float(kv_heads or 0.0))
    if kv >= q:
        return "mha"
    if kv <= 1.0:
        return "shared_kv"
    return "gqa"


def attention_phase(q_tokens: float) -> str:
    return "prefill" if float(q_tokens or 0.0) > 1.0 else "decode"


def _attention_geometry(
    shape: Mapping[str, Any],
    family: str,
    q_tile: int,
    kv_tile: int,
    aic: int,
) -> dict[str, Any]:
    """Tile/wave geometry of one fused-attention launch.

    The score axis is ``q x visible_kv`` and the value axis is ``q x dim``.  A
    fused kernel executes both matmuls, ``attention_qk`` only the score matmul
    and ``attention_pv`` only the value matmul, so the three families share one
    geometry with a different tile count -- they are never reduced to a GEMM.
    """
    q = float(shape.get("q") or 0.0)
    kv = float(shape.get("kv") or 0.0)
    heads = float(shape.get("heads") or 1.0)
    dim = float(shape.get("dim") or 0.0)
    causal = bool(shape.get("causal"))
    visible = attention_visible_kv(q, kv, causal)
    q_blocks = _ceil_div(q, q_tile)
    kv_blocks = _ceil_div(visible, kv_tile)
    dim_blocks = _ceil_div(dim, ATTENTION_DIM_TILE)
    qk_tiles = q_blocks * kv_blocks
    pv_tiles = q_blocks * dim_blocks
    if family == "attention_qk":
        tiles = qk_tiles
    elif family == "attention_pv":
        tiles = pv_tiles
    else:
        tiles = qk_tiles + pv_tiles
    blocks = tiles * max(1.0, heads)
    waves = max(1, _ceil_div(blocks, aic))
    # A ragged edge costs at most ONE extra tile per (head, dimension): the
    # partial tile the kernel still has to execute.  Charging one per kv tile
    # instead would make the tail term proportional to the KV length and
    # therefore indistinguishable from the memory rate -- an invented slope.
    q_rem = int(round(q)) % int(q_tile) != 0
    kv_rem = int(math.ceil(visible)) % int(kv_tile) != 0
    tail_tiles = ((1 if q_rem else 0) + (1 if kv_rem else 0)) * max(1.0, heads)
    return {
        "q_blocks": float(q_blocks),
        "kv_blocks": float(kv_blocks),
        "dim_blocks": float(dim_blocks),
        "qk_tiles": float(qk_tiles),
        "pv_tiles": float(pv_tiles),
        "tiles": float(tiles),
        "blocks": float(blocks),
        "waves": float(waves),
        "wave_eff": blocks / float(waves * max(1, aic)),
        "visible_kv": float(visible),
        "causal_factor": (visible / kv) if kv > 0 else 1.0,
        "tail_tiles": float(tail_tiles),
    }


def _grouped_geometry(
    shape: Mapping[str, Any], block_m: int | None, block_n: int | None, aic: int
) -> dict[str, Any]:
    """Tile/wave geometry of one grouped matmul: ``experts`` small GEMMs.

    Each expert contracts ``tokens_per_expert x k`` against a ``k x n`` weight
    matrix, so the cube sees ``experts * ceil(tpe/m_tile) * ceil(n/n_tile)``
    tiles.  That product is the expert-count scaling that a dense-GEMM model
    cannot express: ``tokens`` can stay fixed while ``experts`` grows and the
    tile count grows with it.
    """
    tokens = float(shape.get("tokens") or 0.0)
    experts = float(shape.get("experts") or 0.0)
    n = float(shape.get("n") or 0.0)
    tokens_per_expert = tokens / experts if experts > 0 else tokens
    rows = _ceil_div(tokens_per_expert, block_m) if block_m else 1
    cols = _ceil_div(n, block_n) if block_n else 1
    tiles = max(1.0, experts) * rows * cols
    waves = max(1, _ceil_div(tiles, aic))
    return {
        "tokens_per_expert": float(tokens_per_expert),
        "rows_per_expert": float(rows),
        "cols": float(cols),
        "tiles": float(tiles),
        "blocks": float(tiles),
        "waves": float(waves),
        "wave_eff": tiles / float(waves * max(1, aic)),
    }


def _term_specs(
    kind: str,
    tail: Mapping[str, bool],
    k_fill: bool,
    model_terms: Mapping[str, Any] | None = None,
) -> tuple[bool, list[tuple[str, tuple[float, float]]]]:
    """Declared additive-term slots, in parameter order.

    The memory-side ``n``-alignment slot (``align_penalty``) is separate and
    always first; these are the purely additive slots that follow it.
    """
    terms = dict(model_terms or {})
    specs: list[tuple[str, tuple[float, float]]] = []
    if kind == "attention" and tail.get("tile"):
        specs.append(("tail_tile_ms", ATTENTION_TAIL_BOUNDS_MS))
    if kind == "attention" and tail.get("token_latency"):
        specs.append(("kv_token_ms", ATTENTION_TOKEN_LATENCY_BOUNDS_MS))
    if kind == "grouped_gemm" and terms.get("include_per_expert", True):
        specs.append(("per_expert_ms", GROUPED_PER_EXPERT_BOUNDS_MS))
    if kind == "grouped_gemm" and terms.get("include_per_launch", True):
        specs.append(("per_launch_ms", GROUPED_PER_LAUNCH_BOUNDS_MS))
    if k_fill:
        specs.append(("k_fill_ms", K_FILL_BOUNDS_MS))
    return bool(tail.get("n")), specs


def _grouped_activation_bytes(obs: Observation) -> float:
    """Bytes the per-expert activation read moves (``experts x tpe x k``)."""
    tokens = float(obs.shape.get("tokens") or 0.0)
    k = float(obs.shape.get("k") or 0.0)
    itemsize = dtype_itemsize(obs.dtype)
    return tokens * k * itemsize


def _design_row(
    obs: Observation,
    machine: MachineLimits,
    form: str,
    memory_model: str,
    l2_usable: int,
    block_m: int | None,
    block_n: int | None,
    *,
    tail: Mapping[str, bool],
    launch_index: int,
    n_launch: int,
    k_fill: bool,
    model_terms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One coefficient row for ``t = launch[scope] + additive + combine(compute, memory)``.

    The parameter vector is ``[launch_0 .. launch_{S-1}, rate_compute, rate_memory,
    (rate_l2), (align_penalty), (additive terms...)]``; the row stores its own
    offsets so scoring and fitting stay linear in the parameters for a fixed
    form.
    """
    terms = dict(model_terms or {})
    to_ms = 1000.0
    n_rate = 3 if memory_model == "l2_blend" else 2
    i_rate = n_launch
    i_add = i_rate + n_rate
    compute_coef = 0.0
    geometry: dict[str, float] = {"wave_eff": 1.0, "blocks": 0.0, "waves": 0.0}
    memory_bytes = obs.hbm_bytes
    if obs.model_kind == "gemm":
        wave_eff = 1.0
        if form == "wave_max":
            blocks, waves, wave_eff = _gemm_geometry(
                obs, int(block_m or 1), int(block_n or 1), machine.aic_count
            )
            geometry = {"blocks": float(blocks), "waves": float(waves), "wave_eff": wave_eff}
        flops = _padded_flops(obs, block_m, tail)
        geometry["flops_effective"] = flops
        if flops > 0:
            compute_coef = flops / (machine.peak_flops * max(wave_eff, 1e-9)) * to_ms
    elif obs.model_kind == "attention":
        geo = _attention_geometry(
            obs.shape, obs.family, int(block_m or 1), int(block_n or 1), machine.aic_count
        )
        if form != "wave_max":
            geo["wave_eff"] = 1.0
        flops = obs.flops * geo["causal_factor"]
        geometry = {key: float(value) for key, value in geo.items()}
        broadcast = terms.get("kv_broadcast")
        if broadcast:
            itemsize = dtype_itemsize(obs.dtype)
            extra_bytes = attention_kv_extra_bytes(obs.shape, int(broadcast), itemsize)
            memory_bytes = obs.hbm_bytes + extra_bytes
            geometry["kv_broadcast"] = float(broadcast)
            geometry["kv_heads_effective"] = attention_kv_heads_effective(
                float(obs.shape.get("heads") or 0.0),
                float(obs.shape.get("kv_heads") or obs.shape.get("heads") or 0.0),
                int(broadcast),
            )
            geometry["kv_extra_bytes"] = extra_bytes
        geometry["flops_effective"] = flops
        if flops > 0:
            compute_coef = (
                flops / (machine.peak_flops * max(geometry["wave_eff"], 1e-9)) * to_ms
            )
    elif obs.model_kind == "grouped_gemm":
        geo = _grouped_geometry(obs.shape, block_m, block_n, machine.aic_count)
        if form != "wave_max":
            geo["wave_eff"] = 1.0
        flops = _padded_flops(obs, block_m, tail)
        geometry = {key: float(value) for key, value in geo.items()}
        geometry["flops_effective"] = flops
        if flops > 0:
            compute_coef = (
                flops / (machine.peak_flops * max(geometry["wave_eff"], 1e-9)) * to_ms
            )
        if not terms.get("activation_read", GROUPED_DEFAULT_ACTIVATION_READ):
            # The declared byte convention minus the per-expert activation read.
            memory_bytes = max(
                0.0, obs.hbm_bytes - _grouped_activation_bytes(obs)
            )
    else:
        if obs.elements > 0:
            # Normalised by the declared issue ceiling, so the fitted coordinate
            # is ceiling/throughput and the rate keeps a physical upper bound.
            compute_coef = obs.elements / machine.vec_max_ops_per_s * to_ms
    hbm_part, l2_part = _memory_split(memory_bytes, l2_usable, machine.peak_bw_bytes_per_s)
    memory_coef = [hbm_part / machine.peak_bw_bytes_per_s * to_ms]
    if memory_model == "l2_blend":
        memory_coef.append(l2_part / machine.peak_bw_bytes_per_s * to_ms)
    n_align, specs = _term_specs(obs.model_kind, tail, k_fill, terms)
    additive = []
    additive_names = []
    for name, _bounds in specs:
        additive_names.append(name)
        if name == "tail_tile_ms":
            additive.append(geometry.get("tail_tiles", 0.0))
        elif name == "kv_token_ms":
            additive.append(geometry.get("visible_kv", 0.0))
        elif name == "per_expert_ms":
            gamma = float(terms.get("expert_scaling", 1.0))
            additive.append(float(obs.shape.get("experts") or 0.0) ** gamma)
        elif name == "per_launch_ms":
            experts = float(obs.shape.get("experts") or 0.0)
            additive.append(max(0.0, _ceil_div(experts, GROUPED_LAUNCH_CAPACITY) - 1.0))
        elif name == "k_fill_ms":
            additive.append(1.0 if _short_k(obs) else 0.0)
        else:
            additive.append(0.0)
    i_align = i_add
    align_coefs = [0.0] * len(memory_coef)
    if n_align and _misaligned(obs, block_n):
        # One extra full-bandwidth pass over the whole working set per unit of
        # the fitted penalty: memory = coef/eta_m + coef*p_align.
        align_coefs = list(memory_coef)
    return {
        "form": form,
        "launch_index": launch_index,
        "i_rate": i_rate,
        "i_add": i_add + (1 if n_align else 0),
        "i_align": i_align,
        "n_align": 1 if n_align else 0,
        "compute_coef": compute_coef,
        "memory_coef": memory_coef,
        "align_coefs": align_coefs,
        "additive": additive,
        "additive_names": additive_names,
        "geometry": geometry,
        "target": obs.duration_ms,
    }


def _term_values(row: Mapping[str, Any], theta: Sequence[float]) -> tuple[float, float]:
    """(compute, memory) terms of one design row for the current parameters."""
    i = row["i_rate"]
    compute = row["compute_coef"] * theta[i]
    memory = sum(coef * theta[i + 1 + index] for index, coef in enumerate(row["memory_coef"]))
    if row.get("n_align"):
        memory += theta[row["i_align"]] * sum(row["align_coefs"])
    return compute, memory


def _evaluate(row: Mapping[str, Any], theta: Sequence[float]) -> float:
    """Predicted duration of one design row under the row's declared form."""
    compute, memory = _term_values(row, theta)
    base = theta[row["launch_index"]]
    for index, coefficient in enumerate(row["additive"]):
        base += theta[row["i_add"] + index] * coefficient
    if row.get("form", "wave_max") == "wave_max":
        return base + max(compute, memory)
    return base + compute + memory


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    work = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-18:
            return None
        work[col], work[pivot] = work[pivot], work[col]
        inverse = 1.0 / work[col][col]
        for r in range(col + 1, n):
            factor = work[r][col] * inverse
            if factor:
                for c in range(col, n + 1):
                    work[r][c] -= factor * work[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = work[i][n] - sum(work[i][j] * out[j] for j in range(i + 1, n))
        out[i] = acc / work[i][i]
    return out


def _launch_bounds_for(
    kind: str, scope: str, machine: "MachineLimits | None" = None, recipe: str = ""
) -> tuple[float, float]:
    """Declared launch bounds of one (kind, timing scope, recipe) column.

    The attention and grouped workers dispatch one native call per sample, so on
    a part whose per-call dispatch was measured their device-event intercept is
    bounded by that measurement plus a margin.  The completion-clock block
    interval also carries the host/allocation gap, so it keeps the wider generic
    launch bounds, and a part without a dispatch measurement does too.

    A fused-chunk member recipe declares a zero intercept
    (:data:`FUSED_CHUNK_MEMBER_TERMS`): the member is one term of a kernel whose
    launch the composer charges once.
    """
    declared = FUSED_CHUNK_MEMBER_TERMS.get(str(recipe or ""))
    if declared is not None:
        return tuple(declared["launch_bounds_ms"])
    if kind in ("attention", "grouped_gemm") and scope != ATTENTION_BLOCK_SCOPE:
        key = getattr(machine, "hardware_key", "") if machine is not None else ""
        return DISPATCH_LAUNCH_BOUNDS_BY_HARDWARE.get(key, LAUNCH_BOUNDS_MS)
    return LAUNCH_BOUNDS_MS


def _parameter_bounds(
    n_launch: int,
    kind: str,
    memory_model: str,
    k_fill: bool,
    n_align: bool = False,
    l2_multiple_bounds: Sequence[float] = L2_BW_MULTIPLE_BOUNDS,
    model_terms: Mapping[str, Any] | None = None,
    tail: Mapping[str, bool] | None = None,
    launch_bounds: Sequence[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    tail_map = dict(tail or {})
    n_align = bool(n_align or tail_map.get("n"))
    low, high = ETA_BOUNDS
    mem_high = GROUPED_ETA_MEMORY_MAX if kind == "grouped_gemm" else high
    if launch_bounds is not None:
        bounds = [tuple(item) for item in launch_bounds]
        if not bounds:
            bounds = [LAUNCH_BOUNDS_MS]
    else:
        bounds = [LAUNCH_BOUNDS_MS] * max(1, n_launch)
    if kind in CUBE_KINDS:
        bounds.append((1.0 / high, 1.0 / low))
    else:
        bounds.append((1.0, 1.0 / VEC_THROUGHPUT_MIN_FRACTION))
    bounds.append((1.0 / mem_high, 1.0 / low))
    if memory_model == "l2_blend":
        bounds.append((1.0 / l2_multiple_bounds[1], 1.0 / l2_multiple_bounds[0]))
    if n_align:
        bounds.append(ALIGN_PENALTY_BOUNDS)
    specs = _term_specs(
        kind, dict(tail_map, n=n_align, tile=bool(tail_map.get("tile"))), k_fill, model_terms
    )[1]
    for _name, spec_bounds in specs:
        bounds.append(spec_bounds)
    return bounds


def _clamp(theta: Sequence[float], bounds: Sequence[tuple[float, float]]) -> list[float]:
    return [min(max(value, lo), hi) for value, (lo, hi) in zip(theta, bounds)]


def _initial_theta(bounds: Sequence[tuple[float, float]]) -> list[float]:
    # Start every launch at 0, every efficiency branch at eta = 0.5 (or the
    # vector ceiling fraction 2), the L2 rate at 2x peak and the fill at 0.
    start = []
    for lo, hi in bounds:
        if lo == 0.0:
            start.append(0.0)
        elif hi <= 1.0:
            start.append(0.5 * (lo + hi) if hi < 1.0 else 2.0)
        else:
            start.append(min(max(2.0, lo), hi))
    return _clamp(start, bounds)


def _robust_scale(residuals: Sequence[float]) -> float:
    if not residuals:
        return 1e-9
    ordered = sorted(residuals)
    mid = ordered[len(ordered) // 2]
    deviations = sorted(abs(r - mid) for r in residuals)
    mad = deviations[len(deviations) // 2]
    return max(1.4826 * mad, 1e-9)


def _irls_fit(
    rows: Sequence[Mapping[str, Any]],
    bounds: Sequence[tuple[float, float]],
    iterations: int = IRLS_ITERATIONS,
    pinned: Mapping[int, float] | None = None,
) -> list[float]:
    """Projected IRLS with Huber weights over the active (binding) branch.

    ``pinned`` fixes coordinates at declared values (used by the profile
    interval, which refits every other parameter at a fixed candidate rate).
    """
    theta = _initial_theta(bounds)
    n_params = len(bounds)
    if not rows:
        return theta
    pinned = dict(pinned or {})
    for index, value in pinned.items():
        theta[index] = min(max(value, bounds[index][0]), bounds[index][1])
    if len(pinned) >= n_params:
        return theta
    scale = 1.0
    for _ in range(iterations):
        predictions = [_evaluate(row, theta) for row in rows]
        residuals = [row["target"] - pred for row, pred in zip(rows, predictions)]
        scale = _robust_scale(residuals)
        weights = []
        for residual in residuals:
            magnitude = abs(residual)
            weights.append(1.0 if magnitude <= HUBER_K * scale else (HUBER_K * scale) / magnitude)
        matrix = [[0.0] * n_params for _ in range(n_params)]
        vector = [0.0] * n_params
        for row, weight in zip(rows, weights):
            compute, memory = _term_values(row, theta)
            active_memory = memory > compute
            design = [0.0] * n_params
            design[row["launch_index"]] = 1.0
            for index, coefficient in enumerate(row["additive"]):
                design[row["i_add"] + index] += coefficient
            # A max-composed row is linearized on its active branch; an additive
            # form carries BOTH terms in every row, so both enter the design.
            additive_form = row.get("form") == "flat_additive"
            if active_memory or additive_form:
                for index, coefficient in enumerate(row["memory_coef"]):
                    design[row["i_rate"] + 1 + index] += coefficient
                if row.get("n_align"):
                    design[row["i_align"]] += sum(row["align_coefs"])
            if not active_memory or additive_form:
                design[row["i_rate"]] += row["compute_coef"]
            for i in range(n_params):
                if design[i] == 0.0:
                    continue
                vector[i] += weight * design[i] * row["target"]
                for j in range(i, n_params):
                    matrix[i][j] += weight * design[i] * design[j]
        for i in range(n_params):
            for j in range(i):
                matrix[i][j] = matrix[j][i]
        trace = sum(matrix[i][i] for i in range(n_params))
        ridge = max(trace, 1.0) * 1e-12
        for i in range(n_params):
            matrix[i][i] += ridge
        solved = _solve(matrix, vector)
        if solved is None:
            break
        theta = _clamp(solved, bounds)
        for index, value in pinned.items():
            theta[index] = value
    return theta


def _score(rows: Sequence[Mapping[str, Any]], theta: Sequence[float]) -> dict[str, Any]:
    targets = [row["target"] for row in rows]
    predictions = [_evaluate(row, theta) for row in rows]
    residuals = [pred - target for pred, target in zip(predictions, targets)]
    total = sum(targets)
    absolute = sum(abs(r) for r in residuals)
    apes = sorted(100.0 * abs(r) / target for r, target in zip(residuals, targets) if target > 0)
    return {
        "wape_pct": 100.0 * absolute / total if total > 0 else float("nan"),
        "median_ape_pct": _quantile(apes, 0.5),
        "p90_ape_pct": _quantile(apes, 0.9),
        "max_ape_pct": apes[-1] if apes else float("nan"),
        "signed_bias_pct": 100.0 * sum(residuals) / total if total > 0 else float("nan"),
    }


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
    return float(ordered[index])


def _binding_detail(rows: Sequence[Mapping[str, Any]], theta: Sequence[float]) -> dict[str, Any]:
    """Which term dominates per row, and whether each rate entered that term."""
    counts = {"compute": 0, "memory": 0, "memory_hbm_part": 0, "memory_l2_part": 0}
    for row in rows:
        compute, memory = _term_values(row, theta)
        if row.get("form") == "flat_additive":
            # An additive row carries both terms: every row constrains both
            # rates, so neither is "unidentified" merely because the other is
            # larger at the current parameters.
            counts["compute"] += 1
            counts["memory"] += 1
            if row["memory_coef"][0] > 0:
                counts["memory_hbm_part"] += 1
            if len(row["memory_coef"]) > 1 and row["memory_coef"][1] > 0:
                counts["memory_l2_part"] += 1
            continue
        if compute >= memory:
            counts["compute"] += 1
            continue
        counts["memory"] += 1
        if row["memory_coef"][0] > 0:
            counts["memory_hbm_part"] += 1
        if len(row["memory_coef"]) > 1 and row["memory_coef"][1] > 0:
            counts["memory_l2_part"] += 1
    total = max(1, len(rows))
    return {
        "share": {"compute": counts["compute"] / total, "memory": counts["memory"] / total},
        "counts": counts,
        "samples": len(rows),
    }


def _theta_to_parameters(
    theta: Sequence[float],
    kind: str,
    machine: MachineLimits,
    n_launch: int,
    memory_model: str,
    k_fill: bool,
    n_align: bool = False,
    model_terms: Mapping[str, Any] | None = None,
    tail: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Convert the fitted vector to physical rates (efficiencies / throughput)."""
    tail_map = dict(tail or {})
    n_align = bool(n_align or tail_map.get("n"))
    i_rate = n_launch
    i_mem = i_rate + 1
    launches = list(theta[:n_launch])
    eta_memory = 1.0 / theta[i_mem] if theta[i_mem] > 0 else ETA_BOUNDS[0]
    out: dict[str, Any] = {
        "launches_ms": launches,
        "launch_ms": launches[0] if launches else 0.0,
        "eta_compute": None,
        "eta_memory": eta_memory,
        "vec_ops_per_s": None,
        "l2_bw_bytes_per_s": None,
        "k_fill_ms": None,
        "align_penalty": None,
        "tail_tile_ms": None,
        "kv_token_ms": None,
        "per_expert_ms": None,
        "per_launch_ms": None,
    }
    if kind in CUBE_KINDS:
        out["eta_compute"] = 1.0 / theta[i_rate] if theta[i_rate] > 0 else ETA_BOUNDS[0]
    else:
        out["vec_ops_per_s"] = (
            machine.vec_max_ops_per_s / theta[i_rate] if theta[i_rate] > 0 else machine.vec_max_ops_per_s
        )
    if memory_model == "l2_blend":
        i_l2 = i_mem + 1
        if theta[i_l2] > 0:
            out["l2_bw_bytes_per_s"] = machine.peak_bw_bytes_per_s / theta[i_l2]
    n_mem = 3 if memory_model == "l2_blend" else 2
    i_slot = i_mem + n_mem - 1
    if n_align:
        out["align_penalty"] = float(theta[i_slot])
        i_slot += 1
    for name, _bounds in _term_specs(
        kind, dict(tail_map, n=n_align, tile=bool(tail_map.get("tile"))), k_fill, model_terms
    )[1]:
        if name == "k_fill_ms":
            out["k_fill_ms"] = float(theta[i_slot])
        elif name == "tail_tile_ms":
            out["tail_tile_ms"] = float(theta[i_slot])
        elif name == "kv_token_ms":
            out["kv_token_ms"] = float(theta[i_slot])
        elif name == "per_expert_ms":
            out["per_expert_ms"] = float(theta[i_slot])
        elif name == "per_launch_ms":
            out["per_launch_ms"] = float(theta[i_slot])
        i_slot += 1
    return out


def _fold_index(observations: Sequence[Observation], folds: int) -> list[int]:
    ordered = sorted(
        range(len(observations)),
        key=lambda i: (observations[i].condition_key, observations[i].identity, i),
    )
    assignment = [0] * len(observations)
    for slot, index in enumerate(ordered):
        assignment[index] = slot % folds
    return assignment


def _launch_indices(observations: Sequence[Observation], launch_scopes: Sequence[str]) -> list[int]:
    order = {scope: index for index, scope in enumerate(launch_scopes)}
    return [order.get(obs.timing_scope, 0) for obs in observations]


def _build_rows(
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    form: str,
    memory_model: str,
    l2_usable: int,
    block_m: int | None,
    block_n: int | None,
    tail: Mapping[str, bool],
    launch_scopes: Sequence[str],
    k_fill: bool,
    model_terms: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    indices = _launch_indices(observations, launch_scopes)
    return [
        _design_row(
            obs,
            machine,
            form,
            memory_model,
            l2_usable,
            block_m,
            block_n,
            tail=tail,
            launch_index=index,
            n_launch=max(1, len(launch_scopes)),
            k_fill=k_fill,
            model_terms=model_terms,
        )
        for obs, index in zip(observations, indices)
    ]


def _cross_validate(
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    form: str,
    memory_model: str,
    l2_usable: int,
    block_m: int | None,
    block_n: int | None,
    tail: Mapping[str, bool],
    launch_scopes: Sequence[str],
    k_fill: bool,
    kind: str,
    folds: int,
    model_terms: Mapping[str, Any] | None = None,
    fold_assignment: Sequence[int] | None = None,
    launch_bounds: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Train-only k-fold CV: refit on each complement, score the held-out fold."""
    if len(observations) < folds or folds < 2:
        return {"wape_pct": None, "folds": 0, "method": "insufficient_samples"}
    assignment = (
        list(fold_assignment) if fold_assignment is not None else _fold_index(observations, folds)
    )
    rows = _build_rows(
        observations,
        machine,
        form=form,
        memory_model=memory_model,
        l2_usable=l2_usable,
        block_m=block_m,
        block_n=block_n,
        tail=tail,
        launch_scopes=launch_scopes,
        k_fill=k_fill,
        model_terms=model_terms,
    )
    bounds = _parameter_bounds(
        max(1, len(launch_scopes)), kind, memory_model, k_fill, bool(tail.get("n")),
        model_terms=model_terms, tail=tail,
        launch_bounds=(
            list(launch_bounds) if launch_bounds is not None
            else [_launch_bounds_for(kind, scope, machine) for scope in launch_scopes]
        ),
    )
    absolute = 0.0
    total = 0.0
    per_fold = []
    out_of_fold: list[tuple[int, float, float]] = []
    for fold in range(folds):
        train_index = [i for i, held in enumerate(assignment) if held != fold]
        test_index = [i for i, held in enumerate(assignment) if held == fold]
        if not train_index or not test_index:
            continue
        theta = _irls_fit([rows[i] for i in train_index], bounds)
        fold_abs = 0.0
        fold_total = 0.0
        for i in test_index:
            row = rows[i]
            prediction = _evaluate(row, theta)
            fold_abs += abs(prediction - row["target"])
            fold_total += row["target"]
            out_of_fold.append((i, prediction, row["target"]))
        absolute += fold_abs
        total += fold_total
        per_fold.append(
            {"fold": fold, "train": len(train_index), "test": len(test_index),
             "wape_pct": 100.0 * fold_abs / fold_total if fold_total > 0 else None}
        )
    return {
        "wape_pct": 100.0 * absolute / total if total > 0 else None,
        "folds": len(per_fold),
        "per_fold": per_fold,
        "method": "train_only_kfold",
        "grid_selected_on": "full_train",
        "out_of_fold": out_of_fold,
    }


def _memory_candidates(machine: MachineLimits) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = [("hbm", 0)]
    for fraction in L2_USABLE_FRACTIONS:
        usable = int(round(machine.l2_bytes * fraction))
        if usable > 0:
            candidates.append(("l2_blend", usable))
    return candidates


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, int, int]:
    # Strict train-WAPE order; ties prefer the coarser tile geometry.
    return (candidate["metrics"]["wape_pct"], candidate["block_m"] or 0, candidate["block_n"] or 0)


def _l2_bw_at_bound(theta: Sequence[float], n_launch: int, memory_model: str) -> bool:
    """True when the fitted L2 rate sits on a declared bound (not identified)."""
    if memory_model != "l2_blend":
        return False
    ratio = theta[n_launch + 2]
    lower = 1.0 / L2_BW_MULTIPLE_BOUNDS[1]
    upper = 1.0 / L2_BW_MULTIPLE_BOUNDS[0]
    return ratio <= lower * 1.01 or ratio >= upper * 0.99


def _select_best(
    candidates: Sequence[Mapping[str, Any]], identifying_threshold: int, n_launch: int
) -> Mapping[str, Any]:
    """Best candidate of one memory model for one form.

    A candidate whose L2 rate sits on a declared bound is a clipped parameter,
    not an estimate, so it is only kept when no interior solution fits within
    ``L2_USABLE_SELECTION_TOLERANCE`` of it.  Among equally good interior
    solutions, ``l2_usable`` and ``eta_memory`` trade off on a flat plateau once
    no binding row has an HBM part; there the smallest resident prefix is the
    conservative, identifiable claim.
    """
    def degenerate(candidate: Mapping[str, Any]) -> bool:
        # A blend is degenerate either when its L2 rate sits on a declared bound
        # or when the resident prefix is so large that the HBM rate is left with
        # too few binding rows to be identified.
        if _l2_bw_at_bound(candidate["theta"], n_launch, candidate["memory_model"]):
            return True
        return (
            candidate["memory_model"] == "l2_blend"
            and candidate["binding"]["counts"]["memory_hbm_part"] < identifying_threshold
        )

    best = min(candidates, key=_candidate_sort_key)
    if best["memory_model"] != "l2_blend" or not degenerate(best):
        return best
    limit = best["metrics"]["wape_pct"] * (1.0 + L2_USABLE_SELECTION_TOLERANCE)
    interior = [
        item
        for item in candidates
        if not degenerate(item)
        and item["metrics"]["wape_pct"] <= limit
    ]
    if not interior:
        return best
    identified = [
        item for item in interior if item["binding"]["counts"]["memory_hbm_part"] >= identifying_threshold
    ]
    if identified:
        return min(identified, key=_candidate_sort_key)
    return min(interior, key=lambda item: (item["l2_usable"],) + _candidate_sort_key(item))


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _wape_from_triples(triples: Sequence[tuple[int, float, float]]) -> float | None:
    total = sum(target for _index, _prediction, target in triples)
    if total <= 0:
        return None
    return 100.0 * sum(abs(prediction - target) for _index, prediction, target in triples) / total


def _quality(metrics: Mapping[str, Any], cv_wape_pct: float | None) -> str:
    """Declared trust gate: reported either way, flagged when outside the bound."""
    if cv_wape_pct is None or cv_wape_pct > MAX_TRUSTED_CV_WAPE_PCT:
        return "low_confidence"
    if metrics["p90_ape_pct"] > MAX_TRUSTED_P90_APE_PCT:
        return "low_confidence"
    return "trusted"


def _cache_probe_flags(observations: Sequence[Observation], machine: MachineLimits) -> list[bool]:
    """Outcome-free subset where an L2 residency split can change the prediction."""
    flags = []
    for observation in observations:
        if observation.model_kind not in ("gemm", "grouped_gemm"):
            flags.append(observation.hbm_bytes <= machine.l2_bytes)
            continue
        intensity = observation.flops / max(1.0, observation.hbm_bytes)
        flags.append(
            observation.hbm_bytes <= machine.l2_bytes and intensity <= CACHE_PROBE_MAX_INTENSITY
        )
    return flags


# ---------------------------------------------------------------------------
# Reported breakdowns: M bands and dominant term
# ---------------------------------------------------------------------------
M_BAND_LIMITS = ((2, "m<=2"), (32, "m3-32"), (127, "m33-127"), (129, "m128-129"),
                 (512, "m130-512"), (None, "m>512"))
# Attention is banded by KV length (the axis that decides launch- vs data-bound)
# and a grouped matmul by routed tokens (the axis that decides whether each
# expert's GEMM is a single row or a full tile).
ATTENTION_KV_BAND_LIMITS = ((512, "kv<=512"), (4096, "kv513-4096"),
                            (16384, "kv4k-16k"), (None, "kv>16k"))
GROUPED_TOKEN_BAND_LIMITS = ((64, "tokens<=64"), (512, "tokens65-512"),
                             (2048, "tokens513-2048"), (None, "tokens>2048"))
GROUPED_EXPERT_BAND_LIMITS = ((32, "experts=32"), (64, "experts=64"),
                              (128, "experts=128"), (256, "experts=256"),
                              (None, "experts>256"))


def _band_of(observation: Observation) -> str:
    """Shape band of one observation, on the axis that governs its physics."""
    if observation.model_kind == "attention":
        kv = float(observation.shape.get("kv") or 0.0)
        for limit, label in ATTENTION_KV_BAND_LIMITS:
            if limit is None or kv <= limit:
                return label
        return "kv>16k"
    if observation.model_kind == "grouped_gemm":
        tokens = float(observation.shape.get("tokens") or 0.0)
        for limit, label in GROUPED_TOKEN_BAND_LIMITS:
            if limit is None or tokens <= limit:
                return label
        return "tokens>2048"
    if observation.model_kind != "gemm":
        return "vector"
    m = float(observation.shape.get("m") or 0.0)
    for limit, label in M_BAND_LIMITS:
        if limit is None or m <= limit:
            return label
    return "m>512"


def _band_axis(kind: str) -> str:
    return {
        "attention": "kv_tokens",
        "grouped_gemm": "routed_tokens",
        "gemm": "m",
    }.get(kind, "elements")


def _breakdown(
    observations: Sequence[Observation],
    rows: Sequence[Mapping[str, Any]],
    theta: Sequence[float],
    out_of_fold: Sequence[tuple[int, float, float]],
) -> dict[str, Any]:
    """Train and out-of-fold WAPE by M band and by dominant term."""
    by_band: dict[str, dict[str, Any]] = {}
    by_term: dict[str, dict[str, Any]] = {}
    by_topology: dict[str, dict[str, Any]] = {}
    by_experts: dict[str, dict[str, Any]] = {}
    oof = {index: (prediction, target) for index, prediction, target in out_of_fold}
    for index, (observation, row) in enumerate(zip(observations, rows)):
        prediction = _evaluate(row, theta)
        target = row["target"]
        compute, memory = _term_values(row, theta)
        term = "compute" if compute >= memory else "memory"
        tables = [(by_band, _band_of(observation)), (by_term, term)]
        if observation.model_kind == "attention":
            tables.append((by_topology, _observation_topology(observation)))
        elif observation.model_kind == "grouped_gemm":
            tables.append((by_experts, _expert_band_of(observation)))
        for table, key in tables:
            entry = table.setdefault(
                key, {"samples": 0, "train_abs": 0.0, "train_total": 0.0,
                      "cv_abs": 0.0, "cv_total": 0.0, "cv_samples": 0}
            )
            entry["samples"] += 1
            entry["train_abs"] += abs(prediction - target)
            entry["train_total"] += target
            if index in oof:
                held_out, held_target = oof[index]
                entry["cv_samples"] += 1
                entry["cv_abs"] += abs(held_out - held_target)
                entry["cv_total"] += held_target
    def finish(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for key, entry in sorted(table.items()):
            out[key] = {
                "samples": entry["samples"],
                "train_wape_pct": (
                    100.0 * entry["train_abs"] / entry["train_total"] if entry["train_total"] else None
                ),
                "cv_samples": entry["cv_samples"],
                "cv_wape_pct": (
                    100.0 * entry["cv_abs"] / entry["cv_total"] if entry["cv_total"] else None
                ),
            }
        return out
    axes = sorted({_band_axis(observation.model_kind) for observation in observations})
    out: dict[str, Any] = {
        "band_axis": axes[0] if len(axes) == 1 else axes,
        "by_m_band": finish(by_band),
        "by_dominant_term": finish(by_term),
    }
    if by_topology:
        out["by_topology"] = finish(by_topology)
    if by_experts:
        out["by_expert_band"] = finish(by_experts)
    return out


def _expert_band_of(observation: Observation) -> str:
    experts = float(observation.shape.get("experts") or 0.0)
    for limit, label in GROUPED_EXPERT_BAND_LIMITS:
        if limit is None or experts <= limit:
            return label
    return "experts>256"


def _profile_interval(
    rows: Sequence[Mapping[str, Any]],
    bounds: Sequence[tuple[float, float]],
    theta: Sequence[float],
    index: int,
    best_wape: float,
    to_physical,
) -> list[float] | None:
    """Profile interval: the range whose refitted train WAPE stays in tolerance.

    Deterministic and outcome-free beyond the fitted data: the parameter is
    pinned on a log grid inside its declared bounds and everything else is
    refitted, so the reported interval is a plateau width, not a sampling CI.
    """
    low, high = bounds[index]
    if not (high > low):
        return None
    steps = CI_SCAN_POINTS
    accepted = []
    grid = [theta[index]]
    for step in range(steps + 1):
        fraction = step / float(steps)
        # Log grid for multiplicative rates, linear for additive intercepts.
        grid.append(low * (high / low) ** fraction if low > 0 else low + (high - low) * fraction)
    for value in grid:
        pinned = {index: value}
        candidate = _irls_fit(rows, bounds, iterations=max(3, IRLS_ITERATIONS // 4), pinned=pinned)
        wape = _score(rows, candidate)["wape_pct"]
        if wape <= best_wape * (1.0 + CI_TOLERANCE):
            accepted.append(value)
    if not accepted:
        return None
    endpoints = sorted((to_physical(min(accepted)), to_physical(max(accepted))))
    return [endpoints[0], endpoints[1]]


MODEL_FORMS = ("flat_additive", "wave_max")
FORM_DESCRIPTIONS = {
    "wave_max": "t = launch + max(flops_eff/(peak*eta*wave_eff), memory)",
    "flat_additive": "t = launch + flops_eff/(peak*eta) + memory",
}
TAIL_ORDER = ("m", "n", "k")
# Attention tail: the ragged q / kv edge tile is rounded up to a whole tile.
ATTENTION_TAIL_ORDER = ("tile", "token_latency")


def _tail_order(kind: str) -> tuple[str, ...]:
    if kind == "attention":
        return ATTENTION_TAIL_ORDER
    if kind == "grouped_gemm":
        return ("m", "n")
    if kind == "gemm":
        return TAIL_ORDER
    return ()


def _tile_choices(
    form: str, tail: Mapping[str, bool], kind: str = "gemm"
) -> list[tuple[int | None, int | None]]:
    """Tile grid for one form.

    ``wave_max`` always carries a tile geometry (it is what quantizes waves).
    An attention grid is q_tile x kv_tile; a dense GEMM or a per-expert grouped
    GEMM grid is m_tile x n_tile.
    """
    if kind == "attention":
        grid = [(q, kv) for q in ATTENTION_Q_TILES for kv in ATTENTION_KV_TILES]
        if form == "wave_max":
            return grid
        return grid if tail.get("tile") else [(None, None)]
    if form == "wave_max":
        return [(bm, bn) for bm in BLOCK_M_CHOICES for bn in BLOCK_N_CHOICES]
    block_m = None if not tail.get("m") else BLOCK_M_CHOICES
    block_n = None if not tail.get("n") else BLOCK_N_CHOICES
    if block_m is None and block_n is None:
        return [(None, None)]
    rows_m = block_m or (None,)
    rows_n = block_n or (None,)
    return [(bm, bn) for bm in rows_m for bn in rows_n]


def _fit_form(
    observations: Sequence[Observation],
    machine: MachineLimits,
    form: str,
    kind: str,
    folds: int,
    launch_scopes: Sequence[str],
    tail: Mapping[str, bool],
    variants: Sequence[Mapping[str, Any]] = (),
    fold_assignment: Sequence[int] | None = None,
    launch_bounds: Sequence[tuple[float, float]] | None = None,
    memory_models: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Fit one model form: grid the tiles / resident prefix, then robust IRLS.

    A *variant* is a declared physics convention (a grouped matmul's activation
    read, its expert-count scaling exponent).  Variants are part of the model
    grid but are decided on train-only CV, not on train WAPE, and only replace
    the declared default convention when they buy ``MIN_VARIANT_IMPROVEMENT_PCT``.
    """
    n_launch = max(1, len(launch_scopes))
    tile_choices = _tile_choices(form, tail, kind)
    variant_list = [dict(item) for item in (variants or ({} ,))]
    candidates: list[dict[str, Any]] = []
    allowed_models = (
        {str(item) for item in memory_models} if memory_models is not None else None
    )
    for variant_index, variant in enumerate(variant_list):
        for memory_model, l2_usable in _memory_candidates(machine):
            if allowed_models is not None and memory_model not in allowed_models:
                continue
            for k_fill in (False, True) if tail.get("k") else (False,):
                if k_fill and not any(_short_k(obs) for obs in observations):
                    continue
                bounds = _parameter_bounds(
                    n_launch, kind, memory_model, k_fill, bool(tail.get("n")),
                    model_terms=variant, tail=tail,
                    launch_bounds=(
                        list(launch_bounds) if launch_bounds is not None
                        else [_launch_bounds_for(kind, scope, machine) for scope in launch_scopes]
                    ),
                )
                for block_m, block_n in tile_choices:
                    rows = _build_rows(
                        observations,
                        machine,
                        form=form,
                        memory_model=memory_model,
                        l2_usable=l2_usable,
                        block_m=block_m,
                        block_n=block_n,
                        tail=tail,
                        launch_scopes=launch_scopes,
                        k_fill=k_fill,
                        model_terms=variant,
                    )
                    theta = _irls_fit(rows, bounds)
                    metrics = _score(rows, theta)
                    candidates.append(
                        {
                            "form": form,
                            "tail": dict(tail),
                            "k_fill": k_fill,
                            "memory_model": memory_model,
                            "l2_usable": l2_usable,
                            "block_m": block_m,
                            "block_n": block_n,
                            "theta": theta,
                            "metrics": metrics,
                            "binding": _binding_detail(rows, theta),
                            "variant": dict(variant),
                            "variant_index": variant_index,
                        }
                    )
    if not candidates:
        return None
    identifying_threshold = max(
        MIN_IDENTIFYING_ROWS, int(math.ceil(MIN_IDENTIFYING_SHARE * len(observations)))
    )
    # Best grid point of every (variant, memory model), then every one of those
    # is scored out-of-fold: the variant is a CV decision, not a train-WAPE one.
    best_by_variant: dict[tuple[int, str], Mapping[str, Any]] = {}
    for key in sorted(
        {(candidate["variant_index"], candidate["memory_model"]) for candidate in candidates}
    ):
        best_by_variant[key] = _select_best(
            [
                candidate
                for candidate in candidates
                if (candidate["variant_index"], candidate["memory_model"]) == key
            ],
            identifying_threshold,
            n_launch,
        )
    probe_flags = _cache_probe_flags(observations, machine)
    probe_index = {i for i, flag in enumerate(probe_flags) if flag}
    selection: dict[str, Any] = {
        "chosen_on": "train_only_cv",
        "margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "max_full_cv_degradation_pct": MAX_FULL_CV_DEGRADATION_PCT,
        "cache_probe_samples": len(probe_index),
        "cache_probe_rule": (
            "hbm_bytes <= declared l2_bytes and flops/byte <= %.1f" % CACHE_PROBE_MAX_INTENSITY
        ),
    }
    for candidate in best_by_variant.values():
        candidate["cv"] = _cross_validate(
            observations,
            machine,
            form=form,
            memory_model=candidate["memory_model"],
            l2_usable=candidate["l2_usable"],
            block_m=candidate["block_m"],
            block_n=candidate["block_n"],
            tail=tail,
            launch_scopes=launch_scopes,
            k_fill=candidate["k_fill"],
            kind=kind,
            folds=folds,
            model_terms=candidate["variant"],
            fold_assignment=fold_assignment,
            launch_bounds=launch_bounds,
        )
        out_of_fold = candidate["cv"].pop("out_of_fold", [])
        candidate["out_of_fold"] = out_of_fold
        candidate["probe_cv_wape_pct"] = _wape_from_triples(
            [triple for triple in out_of_fold if triple[0] in probe_index]
        )
    # Variant decision: the declared default is variant 0; a later variant must
    # beat it by MIN_VARIANT_IMPROVEMENT_PCT of train-only CV to replace it.
    variant_trials: list[dict[str, Any]] = []
    best_variant_index = 0
    best_variant_cv: float | None = None
    for variant_index, variant in enumerate(variant_list):
        scores = [
            candidate["cv"].get("wape_pct")
            for key, candidate in best_by_variant.items()
            if key[0] == variant_index and candidate["cv"].get("wape_pct") is not None
        ]
        cv_wape = min(scores) if scores else None
        variant_trials.append({"variant": dict(variant), "cv_wape_pct": cv_wape})
        if cv_wape is not None and (best_variant_cv is None or cv_wape < best_variant_cv):
            best_variant_cv = cv_wape
            best_variant_index = variant_index
    default_cv = variant_trials[0]["cv_wape_pct"] if variant_trials else None
    if (
        best_variant_index != 0
        and default_cv is not None
        and best_variant_cv is not None
        and (default_cv - best_variant_cv) >= MIN_VARIANT_IMPROVEMENT_PCT
    ):
        chosen_variant_index = best_variant_index
        variant_reason = "variant_supported_by_train_only_cv"
    else:
        chosen_variant_index = 0
        variant_reason = "declared_default_variant_kept"
    for trial in variant_trials:
        trial["adopted"] = trial["variant"] == variant_list[chosen_variant_index]
    selection["variants"] = {
        "chosen": dict(variant_list[chosen_variant_index]),
        "reason": variant_reason,
        "margin_pct": MIN_VARIANT_IMPROVEMENT_PCT,
        "trials": variant_trials,
    }
    best_by_memory: dict[str, Mapping[str, Any]] = {}
    for (variant_index, memory_model), candidate in sorted(best_by_variant.items()):
        if variant_index != chosen_variant_index:
            continue
        best_by_memory[memory_model] = candidate
        selection[memory_model] = {
            "train_wape_pct": candidate["metrics"]["wape_pct"],
            "cv_wape_pct": candidate["cv"].get("wape_pct"),
            "cache_probe_cv_wape_pct": candidate["probe_cv_wape_pct"],
            "block_m": candidate["block_m"],
            "block_n": candidate["block_n"],
            "l2_usable_bytes": candidate["l2_usable"] or None,
        }
    chosen = best_by_memory["hbm"]
    if "l2_blend" in best_by_memory:
        blend = best_by_memory["l2_blend"]
        hbm = best_by_memory["hbm"]
        full_gain = _difference(hbm["cv"].get("wape_pct"), blend["cv"].get("wape_pct"))
        probe_gain = _difference(hbm["probe_cv_wape_pct"], blend["probe_cv_wape_pct"])
        degrade = _difference(blend["cv"].get("wape_pct"), hbm["cv"].get("wape_pct"))
        selection["compared"] = True
        selection["cv_improvement_pct"] = full_gain
        selection["cache_probe_cv_improvement_pct"] = probe_gain
        selection["full_cv_degradation_pct"] = degrade
        if len(probe_index) >= MIN_CACHE_PROBE_SAMPLES:
            supported = (
                probe_gain is not None
                and probe_gain >= MIN_CV_IMPROVEMENT_PCT
                and (degrade is None or degrade <= MAX_FULL_CV_DEGRADATION_PCT)
            )
            rule = "cache_probe_cv_with_full_cv_guard"
        else:
            supported = full_gain is not None and full_gain >= MIN_CV_IMPROVEMENT_PCT
            rule = "full_cv_only_because_probe_subset_too_small"
        selection["decision_rule"] = rule
        if supported:
            chosen = blend
            selection["reason"] = "blend_supported_by_train_only_cv"
        else:
            selection["reason"] = "blend_not_supported_by_train_only_cv"
    else:
        selection["compared"] = False
        selection["reason"] = "blend_disabled_by_configuration"
    selection["chosen"] = chosen["memory_model"]
    return {
        "form": form,
        "chosen": chosen,
        "selection": selection,
        "cv_wape_pct": chosen["cv"].get("wape_pct"),
    }


def _multi_launch_rows(observations: Sequence[Observation]) -> int:
    """Rows whose expert count exceeds one ``npu_grouped_matmul`` call."""
    return sum(
        1
        for observation in observations
        if observation.model_kind == "grouped_gemm"
        and float(observation.shape.get("experts") or 0.0) > GROUPED_LAUNCH_CAPACITY
    )


def _declared_variants(
    observations: Sequence[Observation], kind: str,
    member_terms: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Physics-convention variants tried for one family, default convention first.

    * grouped matmul: the per-expert activation read (declared default: read)
      crossed with the expert-count scaling exponent of the per-expert fixed
      cost.  A constant expert count cannot identify the exponent, so that axis
      collapses to a single value and the per-expert term is dropped by
      ``include_per_expert``.
    * attention and dense GEMM: one declared convention.
    """
    if kind == "attention":
        return [{}] + [
            {"kv_broadcast": int(broadcast)}
            for broadcast in ATTENTION_KV_BROADCAST_CHOICES
            if broadcast
        ]
    if kind != "grouped_gemm":
        return [{}]
    forced = dict(member_terms or {})
    expert_counts = {round(float(obs.shape.get("experts") or 0.0), 6) for obs in observations}
    include_per_expert = len(expert_counts) >= 2 and bool(
        forced.get("include_per_expert", True)
    )
    if include_per_expert:
        scalings = [1.0] + [
            value for value in GROUPED_EXPERT_SCALING_CHOICES if value != 1.0
        ]
    else:
        scalings = [1.0]
    # The declared default convention (activation read, linear per-expert cost)
    # is variant 0: a different convention must beat it on train-only CV.
    activations = [GROUPED_DEFAULT_ACTIVATION_READ, not GROUPED_DEFAULT_ACTIVATION_READ]
    # A per-launch cost is fitted only when enough rows actually cross the
    # tensorsList cap to constrain it; otherwise it is pinned to the measured
    # dispatch (see ``_fit_scope_group``).
    include_per_launch = (
        _multi_launch_rows(observations) >= MIN_IDENTIFYING_ROWS
        and bool(forced.get("include_per_launch", True))
    )
    variants: list[dict[str, Any]] = []
    for activation in activations:
        for scaling in scalings:
            variants.append(
                {
                    "activation_read": bool(activation),
                    "expert_scaling": float(scaling),
                    "include_per_expert": bool(include_per_expert),
                    "include_per_launch": bool(include_per_launch),
                }
            )
    if include_per_expert:
        # A per-expert cost that does not pay for itself is dropped: the term has
        # to earn MIN_VARIANT_IMPROVEMENT_PCT of train-only CV like every other
        # optional term, otherwise it absorbs bandwidth error at the turn of the
        # expert-count axis.
        variants.append(
            {
                "activation_read": GROUPED_DEFAULT_ACTIVATION_READ,
                "expert_scaling": 1.0,
                "include_per_expert": False,
            }
        )
    return variants


def _dense_transfer_observations(observations: Sequence[Observation]) -> list[Observation]:
    """The same grouped observations seen as one dense GEMM each.

    This is the *rejected alternative* the artifact must report against: a dense
    GEMM over ``tokens x n x k`` reads the weight matrix once and has no
    per-expert dispatch, so it cannot express the grouped cost.
    """
    proxies: list[Observation] = []
    for observation in observations:
        tokens = float(observation.shape.get("tokens") or 0.0)
        n = float(observation.shape.get("n") or 0.0)
        k = float(observation.shape.get("k") or 0.0)
        itemsize = dtype_itemsize(observation.dtype)
        proxies.append(
            dataclasses.replace(
                observation,
                flops=2.0 * tokens * n * k,
                hbm_bytes=(tokens * k + k * n + tokens * n) * itemsize,
                elements=0.0,
                shape={"m": tokens, "n": n, "k": k},
                model_kind="gemm",
            )
        )
    return proxies


def _dense_transfer_baseline(
    observations: Sequence[Observation],
    machine: MachineLimits,
    launch_scopes: Sequence[str],
    folds: int,
) -> dict[str, Any] | None:
    """Best dense-GEMM fit on the grouped corpus, scored out-of-fold.

    The fold assignment is the one ``_cross_validate`` uses (sorted by condition
    key), and the proxy rows keep the condition keys, so the baseline CV and the
    grouped CV are computed on the same held-out partitions.
    """
    proxies = _dense_transfer_observations(observations)
    fitted: dict[str, dict[str, Any]] = {}
    for form in MODEL_FORMS:
        result = _fit_form(
            proxies, machine, form, "gemm", folds, launch_scopes,
            {"m": False, "n": False, "k": False, "tile": False},
        )
        if result is not None:
            fitted[form] = result
    if not fitted:
        return None

    def sort_key(item):
        cv = item[1]["cv_wape_pct"]
        return (cv is None, cv if cv is not None else float("inf"))

    form, result = min(fitted.items(), key=sort_key)
    chosen = result["chosen"]
    parameters = _theta_to_parameters(
        chosen["theta"], "gemm", machine, max(1, len(launch_scopes)),
        chosen["memory_model"], chosen["k_fill"], False,
        model_terms=chosen.get("variant") or {}, tail={"n": False, "tile": False},
    )
    return {
        "form": form,
        "train_wape_pct": chosen["metrics"]["wape_pct"],
        "cv_wape_pct": result["cv_wape_pct"],
        "block_m": chosen["block_m"],
        "block_n": chosen["block_n"],
        "memory_model": chosen["memory_model"],
        "eta_compute": parameters["eta_compute"],
        "eta_memory": parameters["eta_memory"],
        "launch_ms": parameters["launch_ms"],
        "model": "dense_gemm_transfer: t = launch + max(2*tokens*n*k/(peak*eta*wave_eff), bytes/bw)",
    }


def _fit_scope_group(
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    family: str,
    recipe: str,
    protocol_signature: str,
    protocol_core_signature: str,
    timing_scope: str,
    launch_scopes: Sequence[str],
    fit_level: str,
    compute_dtype: str,
    folds: int,
    topology: str = "",
    corpus_class: str = "",
    fold_assignment: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Select base form, tail terms and memory model by train-only CV."""
    kind = observations[0].model_kind
    # Declared per-(kind, scope, recipe) intercept bounds: a fused-chunk member
    # recipe gets no intercept of its own (its kernel's launch is charged once).
    member_terms = FUSED_CHUNK_MEMBER_TERMS.get(recipe)
    launch_bounds = [
        _launch_bounds_for(kind, scope, machine, recipe) for scope in launch_scopes
    ]
    memory_models = (member_terms or {}).get("memory_models")
    variants = _declared_variants(observations, kind, member_terms)
    # No tail term is assumed: m, n and k tails are forward-selected only if they
    # buy MIN_TAIL_IMPROVEMENT_PCT of train-only CV (padding a tile is physical,
    # but it trades off against eta, so it must be identified by the data).
    base_tail = {"m": False, "n": False, "k": False, "tile": False}
    fitted: dict[str, dict[str, Any]] = {}
    for form in MODEL_FORMS:
        result = _fit_form(
            observations, machine, form, kind, folds, launch_scopes, base_tail, variants,
            fold_assignment=fold_assignment, launch_bounds=launch_bounds,
            memory_models=memory_models,
        )
        if result is not None:
            fitted[form] = result
    form_selection: dict[str, Any] = {
        "chosen_on": "train_only_cv",
        "margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "candidates": {
            form: {
                "description": FORM_DESCRIPTIONS[form],
                "train_wape_pct": result["chosen"]["metrics"]["wape_pct"],
                "cv_wape_pct": result["cv_wape_pct"],
                "memory_model": result["selection"]["chosen"],
                "block_m": result["chosen"]["block_m"],
                "block_n": result["chosen"]["block_n"],
            }
            for form, result in fitted.items()
        },
    }
    chosen_form = None
    for form in MODEL_FORMS:
        result = fitted.get(form)
        if result is None:
            continue
        if chosen_form is None:
            chosen_form = form
            continue
        incumbent = fitted[chosen_form]["cv_wape_pct"]
        challenger = result["cv_wape_pct"]
        if (
            incumbent is not None
            and challenger is not None
            and (incumbent - challenger) >= MIN_CV_IMPROVEMENT_PCT
        ):
            chosen_form = form
    assert chosen_form is not None
    form_selection["chosen"] = chosen_form
    form_selection["chosen_form_wape_pct"] = fitted[chosen_form]["cv_wape_pct"]
    if len(fitted) > 1:
        best_other = min(
            (result["cv_wape_pct"] for form, result in fitted.items()
             if form != chosen_form and result["cv_wape_pct"] is not None),
            default=None,
        )
        form_selection["best_rejected_cv_wape_pct"] = best_other
        form_selection["reason"] = (
            "chosen_form_declared_order"
            if best_other is None
            or (best_other - (fitted[chosen_form]["cv_wape_pct"] or 0.0)) < MIN_CV_IMPROVEMENT_PCT
            else "chosen_form_better_cv"
        )

    result = fitted[chosen_form]
    tail_selection: dict[str, Any] = {
        "margin_pct": MIN_TAIL_IMPROVEMENT_PCT,
        "trials": [
            {
                "tail": dict(base_tail),
                "cv_wape_pct": result["cv_wape_pct"],
                "train_wape_pct": result["chosen"]["metrics"]["wape_pct"],
                "adopted": True,
            }
        ],
        "chosen": dict(base_tail),
    }
    # Forward selection: each extra tail term must buy MIN_TAIL_IMPROVEMENT_PCT
    # of train-only CV.  The incumbent is always refitted with the tested tail.
    incumbent = result
    for tail_name in _tail_order(kind):
        trial_tail = dict(tail_selection["chosen"], **{tail_name: True})
        if trial_tail == tail_selection["chosen"]:
            continue
        if tail_name == "k" and not any(_short_k(obs) for obs in observations):
            tail_selection["trials"].append(
                {"tail": trial_tail, "cv_wape_pct": None, "adopted": False,
                 "reason": "no_short_k_observations"}
            )
            continue
        trial = _fit_form(
            observations, machine, chosen_form, kind, folds, launch_scopes, trial_tail, variants,
            fold_assignment=fold_assignment, launch_bounds=launch_bounds,
            memory_models=memory_models,
        )
        if trial is None:
            continue
        improved = (
            incumbent["cv_wape_pct"] is not None
            and trial["cv_wape_pct"] is not None
            and (incumbent["cv_wape_pct"] - trial["cv_wape_pct"]) >= MIN_TAIL_IMPROVEMENT_PCT
        )
        tail_selection["trials"].append(
            {
                "tail": dict(trial_tail),
                "cv_wape_pct": trial["cv_wape_pct"],
                "train_wape_pct": trial["chosen"]["metrics"]["wape_pct"],
                "adopted": bool(improved),
            }
        )
        if improved:
            incumbent = trial
            tail_selection["chosen"] = trial_tail
    result = incumbent

    chosen = result["chosen"]
    selection = dict(result["selection"])
    selection["tails"] = tail_selection["chosen"]
    theta = chosen["theta"]
    metrics = chosen["metrics"]
    model_terms = dict(chosen.get("variant") or {})
    tail_chosen = tail_selection["chosen"]
    selection["terms"] = {
        "activation_read": bool(model_terms.get("activation_read", False)),
        "expert_scaling": model_terms.get("expert_scaling"),
        "include_per_expert": bool(model_terms.get("include_per_expert", kind != "grouped_gemm")),
        "include_per_launch": bool(model_terms.get("include_per_launch", kind != "grouped_gemm")),
        "kv_broadcast": model_terms.get("kv_broadcast"),
        "attention_dim_tile": ATTENTION_DIM_TILE if kind == "attention" else None,
        "causal_model": (
            "visible_kv = kv - (q-1)/2 for causal masks, kv otherwise"
            if kind == "attention" else None
        ),
    }
    n_launch = max(1, len(launch_scopes))
    parameters = _theta_to_parameters(
        theta, kind, machine, n_launch, chosen["memory_model"], chosen["k_fill"],
        bool(tail_chosen.get("n")), model_terms=model_terms, tail=tail_chosen,
    )
    counts = chosen["binding"]["counts"]
    n_align = bool(tail_chosen.get("n"))
    threshold = max(MIN_IDENTIFYING_ROWS, int(math.ceil(MIN_IDENTIFYING_SHARE * len(observations))))
    compute_name = "eta_compute" if kind in CUBE_KINDS else "vec_ops_per_s"
    identified = {
        "launch": True,
        compute_name: bool(counts["compute"] >= threshold),
        "eta_memory": bool(counts["memory_hbm_part"] >= threshold),
        "tile_geometry": chosen["block_m"] is not None or chosen["block_n"] is not None,
        "l2_split": bool(
            chosen["memory_model"] == "l2_blend" and counts["memory_l2_part"] >= threshold
        ),
        "k_fill": bool(chosen["k_fill"]),
        "attention_tail": bool(tail_chosen.get("tile")),
        "attention_token_latency": bool(tail_chosen.get("token_latency")),
        "per_expert": bool(parameters.get("per_expert_ms") is not None),
        "per_launch": bool(parameters.get("per_launch_ms") is not None),
        "expert_scaling": bool(
            parameters.get("per_expert_ms") is not None
            and len({round(float(obs.shape.get("experts") or 0.0), 6) for obs in observations}) >= 2
        ),
    }
    warnings: list[str] = []
    # A rate that too few binding observations constrain is reported at the
    # declared ceiling instead of at an arbitrary solved value, and flagged.
    pinned: list[str] = []
    compute_rate = (
        parameters["eta_compute"] if kind in CUBE_KINDS else parameters["vec_ops_per_s"]
    )
    eta_memory = parameters["eta_memory"]
    if not identified[compute_name]:
        pinned.append(compute_name)
        warnings.append("%s_unidentified_pinned_to_declared_peak" % compute_name)
        compute_rate = 1.0 if kind in CUBE_KINDS else machine.vec_max_ops_per_s
    if not identified["eta_memory"]:
        pinned.append("eta_memory")
        warnings.append("eta_memory_unidentified_pinned_to_declared_peak")
        eta_memory = 1.0
    eta_compute = compute_rate if kind in CUBE_KINDS else None
    vec_ops_per_s = None if kind in CUBE_KINDS else compute_rate
    l2_bw = parameters["l2_bw_bytes_per_s"]
    if l2_bw is not None and l2_bw >= L2_BW_MULTIPLE_BOUNDS[1] * machine.peak_bw_bytes_per_s * 0.99:
        warnings.append("l2_bw_at_declared_cap")
    if l2_bw is not None and l2_bw <= L2_BW_MULTIPLE_BOUNDS[0] * machine.peak_bw_bytes_per_s * 1.01:
        warnings.append("l2_bw_at_declared_peak_blend_degenerates_to_hbm")
    for name, eta in (("eta_compute", eta_compute), ("eta_memory", eta_memory)):
        if eta is None or name in pinned:
            continue
        if eta <= ETA_BOUNDS[0] * 1.0000001:
            warnings.append("%s_at_lower_bound" % name)
        if eta > 1.0 + 1e-9:
            # Only a grouped family may exceed the declared HBM peak: its memory
            # efficiency is an *effective bandwidth* fitted from weight
            # streaming, and the declared SKU peak is a conservative constant.
            warnings.append("eta_memory_above_declared_peak_effective_bandwidth")
        elif eta >= 1.0 - 1e-9:
            warnings.append("%s_at_declared_peak" % name)
    launch_map = {
        scope: value for scope, value in zip(launch_scopes, parameters["launches_ms"])
    }
    scope_counts: dict[str, int] = {}
    for observation in observations:
        scope_counts[observation.timing_scope] = scope_counts.get(observation.timing_scope, 0) + 1
    for scope, count in sorted(scope_counts.items()):
        if count < MIN_IDENTIFYING_ROWS:
            warnings.append("scope_launch_weakly_identified:%s" % scope)
    for scope, value in launch_map.items():
        if value <= LAUNCH_BOUNDS_MS[0] + 1e-9:
            warnings.append("launch_at_lower_bound:%s" % scope)
            identified["launch"] = False
        elif value >= LAUNCH_BOUNDS_MS[1] - 1e-9:
            warnings.append("launch_at_upper_bound:%s" % scope)
            identified["launch"] = False
    if chosen["memory_model"] == "l2_blend" and chosen["l2_usable"] >= int(round(machine.l2_bytes)):
        # The declared capacity itself, not the largest grid fraction: with the
        # grid capped at 62.5 % the old test would flag every allowed prefix.
        warnings.append("l2_usable_at_declared_capacity")
    if n_align and parameters["align_penalty"] is not None:
        if parameters["align_penalty"] <= ALIGN_PENALTY_BOUNDS[0] + 1e-9:
            warnings.append("align_penalty_at_lower_bound")
        if parameters["align_penalty"] >= ALIGN_PENALTY_BOUNDS[1] - 1e-9:
            warnings.append("align_penalty_at_upper_bound")
    if chosen["k_fill"] and parameters["k_fill_ms"] is not None:
        if parameters["k_fill_ms"] <= K_FILL_BOUNDS_MS[0] + 1e-9:
            warnings.append("k_fill_at_lower_bound")
            identified["k_fill"] = False
        if parameters["k_fill_ms"] >= K_FILL_BOUNDS_MS[1] - 1e-9:
            warnings.append("k_fill_at_upper_bound")
    if chosen["cv"].get("wape_pct") is not None and chosen["cv"]["wape_pct"] > MAX_TRUSTED_CV_WAPE_PCT:
        warnings.append("cv_wape_above_%.0f_pct" % MAX_TRUSTED_CV_WAPE_PCT)
    if metrics["p90_ape_pct"] > MAX_TRUSTED_P90_APE_PCT:
        warnings.append("p90_ape_above_%.0f_pct" % MAX_TRUSTED_P90_APE_PCT)
    if kind in CUBE_KINDS and not identified["eta_compute"]:
        warnings.append("compute_branch_weakly_identified")
    if kind == "attention" and tail_chosen.get("token_latency") and parameters["kv_token_ms"] is not None:
        if parameters["kv_token_ms"] <= ATTENTION_TOKEN_LATENCY_BOUNDS_MS[0] + 1e-12:
            warnings.append("kv_token_latency_at_lower_bound")
        elif parameters["kv_token_ms"] >= ATTENTION_TOKEN_LATENCY_BOUNDS_MS[1] - 1e-12:
            warnings.append("kv_token_latency_at_upper_bound")
    if kind == "attention" and tail_chosen.get("tile") and parameters["tail_tile_ms"] is not None:
        if parameters["tail_tile_ms"] <= ATTENTION_TAIL_BOUNDS_MS[0] + 1e-9:
            warnings.append("attention_tail_at_lower_bound")
        if parameters["tail_tile_ms"] >= ATTENTION_TAIL_BOUNDS_MS[1] - 1e-9:
            warnings.append("attention_tail_at_upper_bound")
    multi_launch_rows = _multi_launch_rows(observations)
    if kind == "grouped_gemm" and multi_launch_rows and parameters["per_launch_ms"] is None:
        # Fewer than MIN_IDENTIFYING_ROWS rows cross the 128-entry tensorsList
        # cap, so the per-launch cost cannot be fitted.  Every extra launch is one
        # measured device dispatch (20 us), so that measured prior is pinned and
        # reported instead of the parameter drifting to a bound.
        measured_dispatch = DISPATCH_OVERHEAD_MS_BY_HARDWARE.get(machine.hardware_key)
        if measured_dispatch is not None:
            parameters["per_launch_ms"] = measured_dispatch
            pinned.append("per_launch_ms")
            identified["per_launch"] = False
            warnings.append("per_launch_ms_pinned_to_measured_dispatch")
        else:
            # No dispatch measurement on this part: leave the term unset rather
            # than importing another SKU's fixed cost.
            identified["per_launch"] = False
            warnings.append("per_launch_ms_unidentified_no_dispatch_measurement")
    quality = _quality(metrics, chosen["cv"].get("wape_pct"))
    if kind == "grouped_gemm":
        expert_counts = {
            round(float(obs.shape.get("experts") or 0.0), 6) for obs in observations
        }
        if parameters["per_expert_ms"] is None:
            if len(expert_counts) < 2:
                warnings.append("per_expert_cost_unidentified_constant_expert_count")
                # The defining axis of a grouped matmul is the expert count; a
                # corpus with one expert count cannot identify it, so the family
                # is reported but never trusted as a large-expert model.
                warnings.append("grouped_expert_axis_unmodeled_low_confidence")
                quality = "low_confidence"
            else:
                # The term was offered and train-only CV preferred the model
                # without it: a fitted zero, not a missing axis.
                warnings.append("per_expert_term_dropped_by_cv")
        elif parameters["per_expert_ms"] <= GROUPED_PER_EXPERT_BOUNDS_MS[0] + 1e-9:
            warnings.append("per_expert_at_lower_bound")
        elif parameters["per_expert_ms"] >= GROUPED_PER_EXPERT_BOUNDS_MS[1] - 1e-9:
            warnings.append("per_expert_at_upper_bound")
        if not identified["expert_scaling"]:
            warnings.append("expert_scaling_unidentified_single_expert_count")
        if not bool(model_terms.get("activation_read", GROUPED_DEFAULT_ACTIVATION_READ)):
            warnings.append("grouped_activation_read_convention_dropped_by_cv")
        if parameters["per_launch_ms"] is not None and "per_launch_ms" not in pinned:
            if parameters["per_launch_ms"] <= GROUPED_PER_LAUNCH_BOUNDS_MS[0] + 1e-9:
                warnings.append("per_launch_at_lower_bound")
            elif parameters["per_launch_ms"] >= GROUPED_PER_LAUNCH_BOUNDS_MS[1] - 1e-9:
                warnings.append("per_launch_at_upper_bound")

    rows = _build_rows(
        observations,
        machine,
        form=chosen["form"],
        memory_model=chosen["memory_model"],
        l2_usable=chosen["l2_usable"],
        block_m=chosen["block_m"],
        block_n=chosen["block_n"],
        tail=tail_chosen,
        launch_scopes=launch_scopes,
        k_fill=chosen["k_fill"],
        model_terms=model_terms,
    )
    breakdown = _breakdown(observations, rows, theta, chosen["out_of_fold"])
    streaming_rows = sum(
        1 for obs in observations if streaming_regime(obs.hbm_bytes, machine.l2_bytes)
    )
    corpus_contract: dict[str, Any] | None = None
    if kind == "attention":
        qk_v = sorted(
            {
                (
                    int(round(float(obs.shape.get("dim") or 0.0))),
                    int(round(float(obs.shape.get("v_dim") or obs.shape.get("dim") or 0.0))),
                )
                for obs in observations
            }
        )
        mask_forms: dict[str, int] = {}
        mask_contracts: dict[str, int] = {}
        for obs in observations:
            form = str(obs.shape.get("mask") or "undeclared")
            mask_forms[form] = mask_forms.get(form, 0) + 1
            contract_name = str(obs.shape.get("mask_contract") or "undeclared")
            mask_contracts[contract_name] = mask_contracts.get(contract_name, 0) + 1
        absorbed = sum(
            1
            for obs in observations
            if float(obs.shape.get("v_dim") or obs.shape.get("dim") or 0.0)
            != float(obs.shape.get("dim") or 0.0)
        )
        corpus_contract = {
            "qk_v_dims": [list(pair) for pair in qk_v],
            "absorbed_mla_rows": absorbed,
            "absorbed_mla": absorbed > 0,
            "mask_forms": dict(sorted(mask_forms.items())),
            "mask_contracts": dict(sorted(mask_contracts.items())),
            "note": (
                "qk_dim and v_dim are the measured geometry; absorbed_mla is true only when "
                "the corpus itself carries v_dim != qk_dim, and the mask contract is what the "
                "runner used (a kernel mask the API ignores is recorded as such, never "
                "assumed)"
            ),
        }
    memory_regime = {
        "streaming_threshold_bytes": int(round(STREAMING_L2_RATIO * machine.l2_bytes)),
        "streaming_rows": streaming_rows,
        "streaming_measured": streaming_rows > 0,
        "policy": (
            "a family that never measured a working set >= %.0fx the declared L2 keeps a "
            "declared measured streaming rate (per hardware key) for such predictions; "
            "the fitted efficiency is not extrapolated there and no L2 residency is "
            "credited" % STREAMING_L2_RATIO
        ),
    }
    dense_transfer: dict[str, Any] | None = None
    if kind == "grouped_gemm":
        dense_transfer = _dense_transfer_baseline(observations, machine, launch_scopes, folds)
        if dense_transfer is not None and chosen["cv"].get("wape_pct") is not None:
            baseline_cv = dense_transfer["cv_wape_pct"]
            gain = _difference(baseline_cv, chosen["cv"].get("wape_pct"))
            dense_transfer["grouped_cv_wape_pct"] = chosen["cv"].get("wape_pct")
            dense_transfer["cv_improvement_pct"] = gain
            dense_transfer["beats_dense_transfer"] = bool(
                gain is not None and gain > 0.0
            )
            if not dense_transfer["beats_dense_transfer"]:
                warnings.append("grouped_fit_does_not_beat_dense_gemm_transfer_on_cv")
    bounds = _parameter_bounds(
        n_launch, kind, chosen["memory_model"], chosen["k_fill"], n_align,
        model_terms=model_terms, tail=tail_chosen,
        launch_bounds=[_launch_bounds_for(kind, scope, machine) for scope in launch_scopes],
    )
    i_rate = n_launch
    i_mem = i_rate + 1
    i_l2 = i_mem + 1
    n_mem = 3 if chosen["memory_model"] == "l2_blend" else 2
    i_align = i_mem + n_mem - 1
    intervals: dict[str, Any] = {}
    for scope_index, scope in enumerate(launch_scopes):
        interval = _profile_interval(
            rows, bounds, theta, scope_index, metrics["wape_pct"], lambda value: value
        )
        if interval is not None:
            intervals["launch_ms:%s" % scope] = interval
    rate_to_physical = (
        (lambda value: 1.0 / value) if kind in CUBE_KINDS
        else (lambda value: machine.vec_max_ops_per_s / value)
    )
    interval = _profile_interval(rows, bounds, theta, i_rate, metrics["wape_pct"], rate_to_physical)
    if interval is not None:
        intervals[compute_name] = interval
    interval = _profile_interval(rows, bounds, theta, i_mem, metrics["wape_pct"],
                                 lambda value: 1.0 / value)
    if interval is not None:
        intervals["eta_memory"] = interval
    if chosen["memory_model"] == "l2_blend":
        interval = _profile_interval(
            rows, bounds, theta, i_l2, metrics["wape_pct"],
            lambda value: machine.peak_bw_bytes_per_s / value,
        )
        if interval is not None:
            intervals["l2_bw_bytes_per_s"] = interval
    i_slot = i_align
    if n_align:
        interval = _profile_interval(rows, bounds, theta, i_slot,
                                     metrics["wape_pct"], lambda value: value)
        if interval is not None:
            intervals["align_penalty"] = interval
        i_slot += 1
    for name, _spec_bounds in _term_specs(kind, tail_chosen, chosen["k_fill"], model_terms)[1]:
        interval = _profile_interval(rows, bounds, theta, i_slot,
                                     metrics["wape_pct"], lambda value: value)
        if interval is not None:
            intervals[name] = interval
        i_slot += 1
    return {
        "family": family,
        "recipe": recipe,
        "protocol_signature": protocol_signature,
        "protocol_core_signature": protocol_core_signature,
        "timing_scope": timing_scope,
        "launch_ms_by_scope": launch_map,
        "scope_samples": scope_counts,
        "model_kind": kind,
        "model_form": chosen["form"],
        "tail": tail_chosen,
        "k_fill": bool(chosen["k_fill"]),
        "align_penalty": parameters["align_penalty"],
        "tail_tile_ms": parameters["tail_tile_ms"],
        "kv_token_ms": parameters["kv_token_ms"],
        "per_expert_ms": parameters["per_expert_ms"],
        "per_launch_ms": parameters["per_launch_ms"],
        "topology": topology,
        "corpus_class": corpus_class,
        "expert_scaling": (
            float(model_terms.get("expert_scaling"))
            if kind == "grouped_gemm" and parameters["per_expert_ms"] is not None
            else None
        ),
        "activation_read": bool(
            kind == "grouped_gemm"
            and model_terms.get("activation_read", GROUPED_DEFAULT_ACTIVATION_READ)
        ),
        "streaming_measured": streaming_rows > 0,
        "kv_broadcast": (
            int(model_terms["kv_broadcast"]) if model_terms.get("kv_broadcast") else None
        ),
        "fit_level": fit_level,
        "block_m": chosen["block_m"],
        "block_n": chosen["block_n"],
        "launch_ms": parameters["launch_ms"],
        "eta_compute": eta_compute,
        "eta_memory": eta_memory,
        "vec_ops_per_s": vec_ops_per_s,
        "k_fill_ms": parameters["k_fill_ms"],
        "memory_model": chosen["memory_model"],
        "l2_usable_bytes": chosen["l2_usable"] if chosen["memory_model"] == "l2_blend" else 0,
        "l2_bw_bytes_per_s": l2_bw,
        "compute_dtype": compute_dtype,
        "samples": len(observations),
        "raw_event_count": sum(obs.raw_event_count for obs in observations),
        "train_wape_pct": metrics["wape_pct"],
        "cv_wape_pct": chosen["cv"].get("wape_pct"),
        "median_ape_pct": metrics["median_ape_pct"],
        "p90_ape_pct": metrics["p90_ape_pct"],
        "max_ape_pct": metrics["max_ape_pct"],
        "binding_share": chosen["binding"]["share"],
        "binding_counts": chosen["binding"]["counts"],
        "identified": identified,
        "pinned_parameters": tuple(pinned),
        "intervals": intervals,
        "breakdown": breakdown,
        "memory_model_selection": selection,
        "model_selection": dict(
            form_selection,
            tails=tail_selection,
            dense_gemm_transfer=dense_transfer,
            memory_regime=memory_regime,
            **({"corpus_contract": corpus_contract} if corpus_contract else {}),
        ),
        "model_terms": dict(model_terms),
        "quality": quality,
        "warnings": tuple(warnings),
        "cv_detail": {"folds": chosen["cv"], "training_sample_count": len(observations)},
        "_oof": list(chosen.get("out_of_fold") or ()),
        "_observations": list(observations),
    }


# ---------------------------------------------------------------------------
# Public calibration object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicalCalibration:
    """Versioned physical calibration artifact for one hardware key."""

    hardware_key: str
    arch_family: str
    machine: MachineLimits
    families: Mapping[str, FamilyParameters]
    fit_config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    unmodeled: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = SCHEMA_VERSION

    # -- construction -----------------------------------------------------
    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhysicalCalibration":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("artifact_must_be_a_mapping")
        version = str(value.get("schema_version") or "")
        if version != SCHEMA_VERSION:
            raise PhysicalCalibrationError("unsupported_schema_version:%r" % (version,))
        hardware_key = str(value.get("hardware_key") or "")
        if not hardware_key:
            raise PhysicalCalibrationError("artifact_missing_hardware_key")
        machine_value = dict(value.get("machine") or {})
        machine_key = str(machine_value.get("hardware_key") or "")
        if machine_key and machine_key != hardware_key:
            raise PhysicalCalibrationError(
                "artifact_hardware_key_mismatch:%s!=%s" % (hardware_key, machine_key)
            )
        machine_value.setdefault("hardware_key", hardware_key)
        machine_value.setdefault("arch_family", value.get("arch_family") or "")
        machine = MachineLimits.from_mapping(machine_value)
        families_value = value.get("families") or {}
        if not isinstance(families_value, Mapping):
            raise PhysicalCalibrationError("artifact_families_must_be_a_mapping")
        families = {}
        for key, item in families_value.items():
            parameters = FamilyParameters.from_mapping(item)
            if parameters.model_kind not in ("gemm", "vector", "attention", "grouped_gemm"):
                raise PhysicalCalibrationError("artifact_unknown_model_kind:%r" % (parameters.model_kind,))
            families[str(key)] = parameters
        return cls(
            hardware_key=hardware_key,
            arch_family=str(value.get("arch_family") or machine.arch_family),
            machine=machine,
            families=families,
            fit_config=dict(value.get("fit_config") or {}),
            provenance=dict(value.get("provenance") or {}),
            unmodeled=tuple(value.get("unmodeled") or ()),
            schema_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PhysicalCalibration":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hardware_key": self.hardware_key,
            "arch_family": self.arch_family,
            "machine": self.machine.to_dict(),
            "fit_config": dict(self.fit_config),
            "families": {key: value.to_dict() for key, value in sorted(self.families.items())},
            "provenance": dict(self.provenance),
            "unmodeled": [dict(item) for item in self.unmodeled],
        }

    # -- lookup -----------------------------------------------------------
    def parameters_for(
        self,
        family: str,
        recipe: str,
        *,
        protocol_signature: str = "",
        timing_scope: str = "",
        topology: str = "",
        corpus_class: str = "",
    ) -> FamilyParameters | None:
        """Exact ``(family, recipe)`` match, else the pooled ``family`` fit.

        ``protocol_signature``, ``timing_scope`` and ``topology`` are hard
        filters: a query that cannot name its protocol (or its attention
        topology, when the artifact was fitted per topology) is only answered
        when the artifact has exactly one candidate, never by borrowing another
        protocol's parameters.
        """
        exact = [
            item
            for item in self.families.values()
            if item.family == family and item.recipe == recipe
        ]
        pooled = [
            item
            for item in self.families.values()
            if item.family == family and item.recipe == "*"
        ]
        for group in (exact, pooled):
            if not group:
                continue
            candidates = list(group)
            if protocol_signature:
                candidates = [
                    item
                    for item in candidates
                    if item.protocol_signature == protocol_signature
                    or (item.fit_level == "family_scope_shared" and not item.protocol_signature)
                ]
            candidates = [item for item in candidates if item.topology == topology]
            candidates = [item for item in candidates if item.corpus_class == corpus_class]
            # Ambiguity is resolved by the declared scope family (or refused);
            # an empty signature is not itself a reason to refuse when the
            # caller named a scope.
            selected = self._select_by_scope(candidates, timing_scope)
            if selected is not None:
                return selected
        return None

    @staticmethod
    def _select_by_scope(
        candidates: Sequence[FamilyParameters], timing_scope: str
    ) -> FamilyParameters | None:
        """Exact scope first, then the declared scope family (never across it)."""
        if not candidates:
            return None
        if not timing_scope:
            return candidates[0] if len(candidates) == 1 else None
        exact = [
            item
            for item in candidates
            if item.timing_scope == timing_scope
            or (
                item.fit_level == "family_scope_shared"
                and timing_scope in item.launch_ms_by_scope
            )
        ]
        if exact:
            return exact[0] if len(exact) == 1 else None
        family = scope_family(timing_scope)
        if not family:
            return None
        order = scope_fallback_order(timing_scope)
        for scope in order:
            if scope == timing_scope:
                continue
            matches = [
                item
                for item in candidates
                if item.timing_scope == scope and scope_family(item.timing_scope) == family
            ]
            if len(matches) == 1:
                return dataclasses.replace(
                    matches[0], scope_alias_from=timing_scope, scope_alias_to=scope
                )
            if len(matches) > 1:
                # Two fitted rows share one spelling inside the family: refuse
                # rather than pick one.
                return None
        return None

    @staticmethod
    def _filter_protocol(
        group: Sequence[FamilyParameters], protocol_signature: str, timing_scope: str
    ) -> list[FamilyParameters]:
        candidates = list(group)
        if protocol_signature:
            candidates = [
                item
                for item in candidates
                if item.protocol_signature == protocol_signature
                or (item.fit_level == "family_scope_shared" and not item.protocol_signature)
            ]
        if timing_scope:
            # A shared-efficiency record answers only for the scopes it fitted a
            # launch intercept for; anything else is unmodeled, not borrowed.
            candidates = [
                item
                for item in candidates
                if item.timing_scope == timing_scope
                or (item.fit_level == "family_scope_shared" and timing_scope in item.launch_ms_by_scope)
            ]
        if not protocol_signature and not timing_scope and len(candidates) > 1:
            # Ambiguous protocol identity: refuse rather than guess.
            return []
        return candidates

    # -- prediction -------------------------------------------------------
    def predict_op(
        self,
        op: Mapping[str, Any],
        machine: Mapping[str, Any] | None = None,
        *,
        compute_dtype: str = "bf16",
    ) -> float | None:
        """Predicted duration in ms, or ``None`` when the op is unmodeled.

        The explicit reason for ``None`` (and the parameter provenance for a
        number) is available from :meth:`predict_op_explained`.
        """
        return self.predict_op_explained(op, machine, compute_dtype=compute_dtype).duration_ms

    def predict_op_explained(
        self,
        op: Mapping[str, Any],
        machine: Mapping[str, Any] | None = None,
        *,
        compute_dtype: str = "bf16",
    ) -> OpPrediction:
        if not isinstance(op, Mapping):
            raise PhysicalCalibrationError("op_must_be_a_mapping")
        family = str(op.get("family") or "")
        recipe = str(op.get("recipe") or "")
        signature = str(op.get("protocol_signature") or "")
        scope = str(op.get("timing_scope") or "")
        limits, source, from_artifact = self._resolve_machine(machine)
        mismatch = str((machine or {}).get("hardware_key") or "")
        if mismatch and mismatch != self.hardware_key:
            return self._unmodeled(op, "hardware_key_mismatch", source, from_artifact)
        topology_candidates: tuple[str, ...] = ("",)
        if str(op.get("topology") or ""):
            topology_candidates = (str(op["topology"]), "")
        else:
            raw_shape = op.get("shape") if isinstance(op.get("shape"), Mapping) else {}
            heads = raw_shape.get("heads") or op.get("heads")
            kv_heads = raw_shape.get("kv_heads") or op.get("kv_heads") or heads
            q_tokens = raw_shape.get("q") or op.get("q")
            if heads and kv_heads:
                derived = attention_topology(float(heads), float(kv_heads))
                derived_class = attention_topology_class(derived)
                phase = attention_phase(float(q_tokens or 0.0))
                topology_candidates = (
                    "%s/%s" % (derived_class, phase),
                    derived_class,
                    "%s/%s" % (derived, phase),
                    derived,
                    "",
                )
        class_candidates: tuple[str, ...] = ("",)
        if family in GROUPED_KINDS:
            raw_shape = op.get("shape") if isinstance(op.get("shape"), Mapping) else {}
            n_value = raw_shape.get("n") if raw_shape.get("n") is not None else op.get("n")
            if n_value is not None:
                class_candidates = (grouped_shape_class({"n": n_value}), "")
        parameters = None
        for topology_key in topology_candidates:
            parameters = self.parameters_for(
                family, recipe, protocol_signature=signature, timing_scope=scope,
                topology=topology_key, corpus_class=class_candidates[0],
            )
            if parameters is not None:
                break
            if class_candidates[0]:
                parameters = self.parameters_for(
                    family, recipe, protocol_signature=signature, timing_scope=scope,
                    topology=topology_key, corpus_class="",
                )
                if parameters is not None:
                    class_candidates = ("",)
                    break
        alias = ""
        if parameters is None and family in FAMILY_ALIASES:
            aliased = FAMILY_ALIASES[family]
            for topology_key in topology_candidates:
                parameters = self.parameters_for(
                    aliased, recipe, protocol_signature=signature, timing_scope=scope,
                    topology=topology_key,
                )
                if parameters is not None:
                    break
            if parameters is not None:
                alias = aliased
        if parameters is None:
            known = {item.family for item in self.families.values()}
            if family in known:
                topology_rows = [
                    item for item in self.families.values()
                    if item.family == family and item.topology
                ]
                if topology_rows:
                    return self._unmodeled(
                        op,
                        "unmodeled_topology:%s" % (topology_candidates[0] or "unspecified"),
                        source, from_artifact,
                    )
                reason = "unmodeled_recipe_or_protocol"
            else:
                reason = "unmodeled_family"
            return self._unmodeled(op, reason, source, from_artifact)
        dtype = str(op.get("dtype") or compute_dtype or parameters.compute_dtype)
        if dtype != parameters.compute_dtype:
            return self._unmodeled(
                op,
                "dtype_outside_fit:%s!=%s" % (dtype, parameters.compute_dtype),
                source,
                from_artifact,
                parameters,
            )
        try:
            itemsize = dtype_itemsize(parameters.compute_dtype)
        except PhysicalCalibrationError as exc:
            # A hand-built artifact could declare a dtype this module has no byte
            # width for; refuse rather than price it at a guessed width.
            return self._unmodeled(op, str(exc), source, from_artifact, parameters)
        shape = op.get("shape")
        shape = dict(shape) if isinstance(shape, Mapping) else {}
        for key in ("m", "n", "k", "elements", "q", "kv", "heads", "dim",
                    "tokens", "experts", "batch", "causal", "mask"):
            if op.get(key) is not None and key not in shape:
                shape[key] = op[key]
        flops = _as_float(op.get("flops"), "flops", allow_none=True)
        hbm_bytes = _as_float(op.get("hbm_bytes"), "hbm_bytes", allow_none=True)
        elements = _as_float(op.get("elements"), "elements", allow_none=True)
        breakdown: dict[str, float] = {}
        additive_ms = 0.0
        stream: dict[str, Any] | None = None
        if parameters.model_kind == "attention":
            q = _as_float(shape.get("q"), "q", allow_none=True)
            kv = _as_float(shape.get("kv"), "kv", allow_none=True)
            if q is None or kv is None:
                return self._unmodeled(
                    op, "missing_shape_q_kv", source, from_artifact, parameters
                )
            heads = _as_float(shape.get("heads"), "heads", allow_none=True) or 1.0
            dim = _as_float(shape.get("dim"), "dim", allow_none=True)
            if dim is None:
                return self._unmodeled(
                    op, "missing_shape_dim", source, from_artifact, parameters
                )
            causal = shape.get("causal")
            if causal is None:
                mask = str(shape.get("mask") or "")
                causal = "causal" in mask.lower() if mask else True
            breakdown["causal"] = 1.0 if causal else 0.0
            attention_shape = {
                "q": q, "kv": kv, "heads": heads, "dim": dim, "causal": bool(causal),
            }
            geo = _attention_geometry(
                attention_shape, family, int(parameters.block_m or 1),
                int(parameters.block_n or 1), limits.aic_count,
            )
            if parameters.model_form != "wave_max":
                geo["wave_eff"] = 1.0
            v_dim = _as_float(shape.get("v_dim"), "v_dim", allow_none=True) or dim
            if flops is None:
                if parameters.family == "attention_qk":
                    flops = 2.0 * q * kv * heads * dim
                elif parameters.family == "attention_pv":
                    flops = 2.0 * q * kv * heads * v_dim
                else:
                    # QK contracts qk_dim, PV contracts v_dim: an absorbed-MLA
                    # call has qk_dim 576 and v_dim 512, so one shared "dim"
                    # would be wrong for it.
                    flops = 2.0 * q * kv * heads * (dim + v_dim)
            effective_flops = flops * geo["causal_factor"]
            for key in ("q_blocks", "kv_blocks", "dim_blocks", "qk_tiles", "pv_tiles",
                        "tiles", "blocks", "waves", "wave_eff", "visible_kv",
                        "causal_factor", "tail_tiles"):
                breakdown[key] = float(geo[key])
            breakdown["flops_effective"] = effective_flops
            compute_ms = (
                effective_flops
                / (limits.peak_flops * (parameters.eta_compute or 0.0) * max(geo["wave_eff"], 1e-9))
                * 1000.0
                if parameters.eta_compute
                else 0.0
            )
            breakdown["compute_ms"] = compute_ms
            itemsize = dtype_itemsize(dtype, default=parameters.compute_dtype)
            qkv_heads = _as_float(shape.get("kv_heads"), "kv_heads", allow_none=True) or heads
            if hbm_bytes is None:
                hbm_bytes = (
                    q * heads * dim
                    + kv * qkv_heads * (dim + v_dim)
                    + q * heads * v_dim
                ) * itemsize + ATTENTION_WORKSPACE_BYTES_PER_QUERY_HEAD * q * heads
            if parameters.kv_broadcast:
                kv_shape = {
                    "heads": heads, "kv_heads": qkv_heads, "kv": kv,
                    "dim": dim, "v_dim": v_dim,
                }
                extra_bytes = attention_kv_extra_bytes(
                    kv_shape, parameters.kv_broadcast, itemsize
                )
                hbm_bytes += extra_bytes
                breakdown["kv_broadcast"] = float(parameters.kv_broadcast)
                breakdown["kv_heads_effective"] = attention_kv_heads_effective(
                    heads, qkv_heads, parameters.kv_broadcast
                )
                breakdown["kv_extra_mb"] = extra_bytes / 1e6
            if parameters.tail.get("tile") and parameters.tail_tile_ms:
                additive_ms += parameters.tail_tile_ms * geo["tail_tiles"]
                breakdown["tail_tile_ms"] = additive_ms
            if parameters.tail.get("token_latency") and parameters.kv_token_ms:
                walk_ms = parameters.kv_token_ms * geo["visible_kv"]
                additive_ms += walk_ms
                breakdown["kv_token_ms"] = walk_ms
        elif parameters.model_kind == "grouped_gemm":
            tokens = _as_float(shape.get("tokens"), "tokens", allow_none=True)
            experts = _as_float(shape.get("experts"), "experts", allow_none=True)
            n = _as_float(shape.get("n"), "n", allow_none=True)
            k = _as_float(shape.get("k"), "k", allow_none=True)
            if tokens is None or experts is None or n is None or k is None:
                return self._unmodeled(
                    op, "missing_shape_tokens_experts_n_k", source, from_artifact, parameters
                )
            grouped_shape = {"tokens": tokens, "experts": experts, "n": n, "k": k}
            itemsize = dtype_itemsize(dtype, default=parameters.compute_dtype)
            if flops is None:
                flops = 2.0 * tokens * n * k
            if hbm_bytes is None:
                hbm_bytes = (
                    experts * n * k + tokens * k + tokens * n
                ) * itemsize
            elif not parameters.activation_read:
                hbm_bytes = max(0.0, hbm_bytes - tokens * k * itemsize)
            geo = _grouped_geometry(
                grouped_shape, parameters.block_m, parameters.block_n, limits.aic_count
            )
            if parameters.model_form != "wave_max":
                geo["wave_eff"] = 1.0
            shape_only = _ShapeOnly(family, grouped_shape, 0.0, flops, hbm_bytes)
            effective_flops = _padded_flops(shape_only, parameters.block_m, parameters.tail or {})
            for key in ("tokens_per_expert", "rows_per_expert", "cols", "tiles",
                        "waves", "wave_eff"):
                breakdown[key] = float(geo[key])
            breakdown["flops_effective"] = effective_flops
            compute_ms = (
                effective_flops
                / (limits.peak_flops * (parameters.eta_compute or 0.0) * max(geo["wave_eff"], 1e-9))
                * 1000.0
                if parameters.eta_compute
                else 0.0
            )
            breakdown["compute_ms"] = compute_ms
            if parameters.per_expert_ms is not None:
                gamma = parameters.expert_scaling if parameters.expert_scaling is not None else 1.0
                additive_ms += parameters.per_expert_ms * (experts ** gamma)
                breakdown["per_expert_ms"] = additive_ms
            if parameters.per_launch_ms is not None:
                extra_launches = max(
                    0.0, _ceil_div(experts, GROUPED_LAUNCH_CAPACITY) - 1.0
                )
                additive_ms += parameters.per_launch_ms * extra_launches
                breakdown["per_launch_ms"] = parameters.per_launch_ms * extra_launches
                breakdown["launches"] = 1.0 + extra_launches
        elif parameters.model_kind == "gemm":
            m = _as_float(shape.get("m"), "m", allow_none=True)
            n = _as_float(shape.get("n"), "n", allow_none=True)
            k = _as_float(shape.get("k"), "k", allow_none=True)
            if m is None or n is None or k is None:
                return self._unmodeled(op, "missing_shape_m_n_k", source, from_artifact, parameters)
            if flops is None:
                flops = 2.0 * m * n * k
            if hbm_bytes is None:
                hbm_bytes = (m * k + k * n + m * n) * dtype_itemsize(dtype, default=parameters.compute_dtype)
            wave_eff = 1.0
            shape_only = _ShapeOnly(family, shape, elements or 0.0, flops, hbm_bytes)
            effective_flops = _padded_flops(shape_only, parameters.block_m, parameters.tail or {})
            if parameters.model_form == "wave_max":
                blocks, waves, wave_eff = _gemm_geometry(
                    shape_only,
                    int(parameters.block_m or 1),
                    int(parameters.block_n or 1),
                    limits.aic_count,
                )
                breakdown["blocks"] = float(blocks)
                breakdown["waves"] = float(waves)
                breakdown["wave_eff"] = wave_eff
            breakdown["flops_effective"] = effective_flops
            compute_ms = (
                effective_flops
                / (limits.peak_flops * (parameters.eta_compute or 0.0) * max(wave_eff, 1e-9))
                * 1000.0
                if parameters.eta_compute
                else 0.0
            )
            breakdown["compute_ms"] = compute_ms
            # A blend is only a claim about residency; test it on the shape before
            # crediting anything (see ``weight_stream_basis``).  A family without a
            # blend has no residency credit to withdraw, so its fitted efficiency
            # stands untouched.
            if parameters.memory_model == "l2_blend" and parameters.l2_usable_bytes > 0:
                stream = weight_stream_basis(
                    model_kind=parameters.model_kind, m=m, k=k, n=n,
                    itemsize=dtype_itemsize(dtype, default=parameters.compute_dtype),
                    l2_usable_bytes=parameters.l2_usable_bytes,
                    l2_bytes=limits.l2_bytes,
                )
        else:
            if elements is None:
                # Accept the measured geometry directly (tokens x width, rows x
                # cols, heads x q x kv) instead of forcing the caller to reduce it.
                elements = _campaign_elements(shape)
            if elements is None:
                return self._unmodeled(op, "missing_elements", source, from_artifact, parameters)
            if hbm_bytes is None:
                hbm_bytes = elements * 2.0 * dtype_itemsize(dtype, default=parameters.compute_dtype)
            compute_ms = (
                elements / parameters.vec_ops_per_s * 1000.0
                if parameters.vec_ops_per_s
                else 0.0
            )
            breakdown["vector_ms"] = compute_ms
        memory = self._memory_terms(limits, parameters, hbm_bytes, stream)
        memory_ms = memory.pop("memory_ms")
        breakdown.update(memory)
        if (stream is not None and stream.get("streamed")):
            breakdown["weight_bytes"] = stream["weight_bytes"]
            breakdown["resident_threshold_bytes"] = stream["resident_threshold_bytes"]
        if (
            parameters.tail.get("n")
            and parameters.align_penalty
            and parameters.block_n
            and shape.get("n") is not None
            and int(float(shape["n"])) % int(parameters.block_n) != 0
        ):
            # Extra full-bandwidth passes paid when the column tile does not divide n.
            align_ms = parameters.align_penalty * hbm_bytes / limits.peak_bw_bytes_per_s * 1000.0
            memory_ms += align_ms
            breakdown["align_penalty_ms"] = align_ms
        breakdown["memory_ms"] = memory_ms
        launch_ms = parameters.launch_ms
        if parameters.fit_level == "family_scope_shared":
            if not scope or scope not in parameters.launch_ms_by_scope:
                return self._unmodeled(
                    op, "unmodeled_timing_scope:%s" % (scope or "unspecified"),
                    source, from_artifact, parameters,
                )
            launch_ms = parameters.launch_ms_by_scope[scope]
        breakdown["launch_ms"] = launch_ms
        k_fill_ms = 0.0
        if parameters.k_fill and parameters.k_fill_ms:
            if float(shape.get("k") or 0.0) <= K_FILL_MAX_K:
                k_fill_ms = parameters.k_fill_ms
                breakdown["k_fill_ms"] = k_fill_ms
        if parameters.model_form == "wave_max":
            work_ms = max(compute_ms, memory_ms)
        else:
            work_ms = compute_ms + memory_ms
        breakdown["work_ms"] = work_ms
        breakdown["additive_ms"] = additive_ms
        return OpPrediction(
            duration_ms=launch_ms + k_fill_ms + additive_ms + work_ms,
            status="predicted",
            reason=(
                (
                    ("physical_calibration_via_family_alias:%s" % alias)
                    if alias
                    else (
                        "physical_calibration"
                        if parameters.quality == "trusted"
                        else "physical_calibration_low_confidence"
                    )
                )
                + (
                    ";scope_alias_from=%s;scope_alias_to=%s"
                    % (parameters.scope_alias_from, parameters.scope_alias_to)
                    if parameters.scope_alias_from
                    else ""
                )
            ),
            family=family,
            recipe=recipe,
            protocol_signature=parameters.protocol_signature,
            timing_scope=parameters.timing_scope,
            model_form=parameters.model_form,
            fit_level=parameters.fit_level,
            parameters={
                "block_m": parameters.block_m,
                "block_n": parameters.block_n,
                "launch_ms": parameters.launch_ms,
                "eta_compute": parameters.eta_compute,
                "eta_memory": parameters.eta_memory,
                "vec_ops_per_s": parameters.vec_ops_per_s,
                "memory_model": parameters.memory_model,
                "l2_usable_bytes": parameters.l2_usable_bytes,
                "l2_bw_bytes_per_s": parameters.l2_bw_bytes_per_s,
                "launch_ms_by_scope": dict(parameters.launch_ms_by_scope),
                "k_fill_ms": parameters.k_fill_ms,
                "align_penalty": parameters.align_penalty,
                "tail_tile_ms": parameters.tail_tile_ms,
                "per_expert_ms": parameters.per_expert_ms,
                "expert_scaling": parameters.expert_scaling,
                "per_launch_ms": parameters.per_launch_ms,
                "activation_read": parameters.activation_read,
                "topology": parameters.topology,
                "corpus_class": parameters.corpus_class,
                "kv_broadcast": parameters.kv_broadcast,
                "kv_token_ms": parameters.kv_token_ms,
                "scope_alias_from": parameters.scope_alias_from or None,
                "scope_alias_to": parameters.scope_alias_to or None,
                "tail": dict(parameters.tail),
                "samples": parameters.samples,
                "cv_wape_pct": parameters.cv_wape_pct,
                "quality": parameters.quality,
                "family_alias_from": alias or None,
            },
            breakdown=breakdown,
            machine_source=source,
            machine_fields_from_artifact=from_artifact,
        )

    def _unmodeled(
        self,
        op: Mapping[str, Any],
        reason: str,
        source: str,
        from_artifact: Sequence[str],
        parameters: FamilyParameters | None = None,
    ) -> OpPrediction:
        return OpPrediction(
            duration_ms=None,
            status="unmodeled",
            reason=reason,
            family=str(op.get("family") or ""),
            recipe=str(op.get("recipe") or ""),
            protocol_signature=str(op.get("protocol_signature") or ""),
            timing_scope=str(op.get("timing_scope") or ""),
            model_form=parameters.model_form if parameters else "",
            fit_level=parameters.fit_level if parameters else "",
            machine_source=source,
            machine_fields_from_artifact=tuple(from_artifact),
        )

    # -- helpers ----------------------------------------------------------
    def _streaming_rate(
        self, limits: MachineLimits, parameters: FamilyParameters,
    ) -> tuple[float, str]:
        """Effective HBM rate for a streamed working set, and where it came from.

        A family that measured the streaming regime keeps its own fitted
        efficiency (``peak x eta_memory``): that number *is* a measurement of the
        regime.  A family that never did is priced at its hardware key's declared
        measured streaming bound, and only a part with no declared bound at all
        falls back to the declared SKU peak -- always without L2 credit, and
        ``streaming_rate_source`` says which of the three was used.
        """
        declared = STREAMING_BW_BYTES_PER_S.get(
            self.hardware_key or limits.hardware_key
        )
        if parameters.streaming_measured:
            return (
                limits.peak_bw_bytes_per_s * (parameters.eta_memory or ETA_BOUNDS[0]),
                "fitted_streaming_efficiency",
            )
        if declared is not None and float(declared) > 0.0:
            return float(declared), "declared_measured_bound"
        return limits.peak_bw_bytes_per_s, "declared_peak_bound"

    def _memory_terms(
        self,
        limits: MachineLimits,
        parameters: FamilyParameters,
        hbm_bytes: float,
        stream: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Memory time, residency credit and the declared regime behind them.

        Two regimes withdraw the L2 credit and price every byte at an HBM rate:

        * ``weight_stream`` -- the shape test says the op's weight is fetched once
          and cannot survive to its next use (:func:`weight_stream_basis`);
        * ``whole_working_set_stream`` -- the op's whole working set is at least
          ``STREAMING_L2_RATIO`` x the declared L2, so nothing of it is resident.

        Everything else keeps the fitted blend exactly as fitted: the credit is
        ``min(bytes, l2_usable)`` and the L2 part is priced at the fitted L2
        bandwidth.  ``l2_credit_bytes``, ``eta_memory_effective``,
        ``memory_regime``, ``streaming_regime``, ``streaming_rate_source`` and
        ``streaming_basis`` are reported for every predicted op, so a reader can
        see which regime priced it and why.
        """
        weighted = bool(stream and stream.get("streamed"))
        whole_set = not weighted and streaming_regime(hbm_bytes, limits.l2_bytes)
        if hbm_bytes <= 0:
            return {
                "memory_ms": 0.0, "l2_credit_bytes": 0.0,
                "eta_memory_effective": parameters.eta_memory or ETA_BOUNDS[0],
                "memory_regime": "hbm", "streaming_regime": 0.0,
                "streaming_rate_source": None, "streaming_basis": "no_memory_traffic",
            }
        if weighted or whole_set:
            rate, source = self._streaming_rate(limits, parameters)
            return {
                "memory_ms": hbm_bytes / rate * 1000.0,
                "l2_credit_bytes": 0.0,
                "eta_memory_effective": rate / limits.peak_bw_bytes_per_s,
                "memory_regime": "weight_stream" if weighted else "whole_working_set_stream",
                "streaming_regime": 1.0,
                "streaming_rate_source": source,
                "streaming_basis": str(
                    (stream or {}).get("basis")
                    or "whole_working_set_stream:hbm_bytes=%.1fMiB>=%.0fxL2"
                    % (hbm_bytes / 2**20, STREAMING_L2_RATIO)
                ),
            }
        hbm_part, l2_part = _memory_split(
            hbm_bytes, parameters.l2_usable_bytes, limits.peak_bw_bytes_per_s
        )
        eta = parameters.eta_memory or ETA_BOUNDS[0]
        total = hbm_part / (limits.peak_bw_bytes_per_s * eta) * 1000.0
        if parameters.memory_model == "l2_blend" and l2_part > 0 and parameters.l2_bw_bytes_per_s:
            total += l2_part / parameters.l2_bw_bytes_per_s * 1000.0
        elif l2_part > 0:
            total += l2_part / (limits.peak_bw_bytes_per_s * eta) * 1000.0
        return {
            "memory_ms": total,
            "l2_credit_bytes": l2_part,
            "eta_memory_effective": eta,
            "memory_regime": "l2_blend" if l2_part > 0 else "hbm",
            "streaming_regime": 0.0,
            "streaming_rate_source": None,
            "streaming_basis": str(
                (stream or {}).get("basis") or "family_has_no_l2_blend"
            ),
        }

    def _resolve_machine(
        self, machine: Mapping[str, Any] | None
    ) -> tuple[MachineLimits, str, tuple[str, ...]]:
        if not machine:
            return self.machine, "artifact", ()
        merged = self.machine.to_dict()
        fields_from_artifact = []
        key_map = {
            "peak_flops": "peak_flops",
            "peak_flops_per_s": "peak_flops",
            "peak_bw_bytes_per_s": "peak_bw_bytes_per_s",
            "hbm_bw_bytes_per_s": "peak_bw_bytes_per_s",
            "aic_count": "aic_count",
            "aiv_count": "aiv_count",
            "clock_mhz": "clock_mhz",
            "l2_bytes": "l2_bytes",
            "vec_peak_ops_per_s": "vec_peak_ops_per_s",
            "ops_per_cycle": "ops_per_cycle",
        }
        for source_key, artifact_key in key_map.items():
            if machine.get(source_key) is not None:
                merged[artifact_key] = machine[source_key]
            else:
                fields_from_artifact.append(artifact_key)
        if machine.get("hardware_key"):
            merged["hardware_key"] = str(machine["hardware_key"])
        try:
            limits = MachineLimits.from_mapping(merged)
        except PhysicalCalibrationError:
            return self.machine, "artifact", tuple(sorted(set(fields_from_artifact)))
        return limits, "call", tuple(sorted(set(fields_from_artifact)))

    def transfer_to(
        self,
        machine: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> "PhysicalCalibration":
        """Re-target the fitted efficiencies onto another machine (cross-SKU).

        Only *efficiencies* and the fitted L2 bandwidth cross the SKU boundary.
        The peaks come from the target machine, the resident prefix is capped by
        the target L2 capacity, and the launch intercept must be supplied per
        target from that target's own measurement -- a dispatch cost is a
        property of the protocol and the part, never of the source artifact.
        """
        limits = MachineLimits.from_mapping(machine)
        if limits.hardware_key == self.hardware_key:
            raise PhysicalCalibrationError("transfer_target_is_the_source_hardware_key")
        transferred: dict[str, FamilyParameters] = {}
        launch_sources = {}
        for target in targets:
            launch = _as_float(target.get("launch_ms"), "launch_ms")
            if launch is None or launch < 0:
                raise PhysicalCalibrationError("transfer_target_missing_launch_ms")
            source_key = str(target.get("source_family_key") or "")
            source = self.families.get(source_key)
            if source is None:
                raise PhysicalCalibrationError("transfer_source_family_unknown:%s" % source_key)
            key = str(target.get("family_key") or source_key)
            transferred[key] = dataclasses.replace(
                source,
                recipe=str(target.get("recipe") or source.recipe),
                protocol_signature=str(target.get("protocol_signature") or ""),
                timing_scope=str(target.get("timing_scope") or ""),
                fit_level="cross_sku_transfer",
                launch_ms=float(launch),
                l2_usable_bytes=int(min(source.l2_usable_bytes, limits.l2_bytes)),
                samples=0,
                raw_event_count=0,
                train_wape_pct=0.0,
                cv_wape_pct=None,
                median_ape_pct=0.0,
                p90_ape_pct=0.0,
                max_ape_pct=0.0,
                quality="transferred",
                warnings=tuple(
                    list(source.warnings)
                    + ["efficiencies_transferred_from:%s" % self.hardware_key,
                       "launch_from:%s" % str(target.get("launch_source") or "declared"),
                       "source_train_metrics_not_applicable"]
                ),
            )
            launch_sources[key] = str(target.get("launch_source") or "declared")
        if not transferred:
            raise PhysicalCalibrationError("transfer_without_targets")
        provenance = dict(self.provenance)
        provenance["transfer"] = {
            "source_hardware_key": self.hardware_key,
            "source_artifact_schema": self.schema_version,
            "target_hardware_key": limits.hardware_key,
            "launch_sources": launch_sources,
            "policy": (
                "efficiencies, optional L2 split and vector throughput transferred; peaks from "
                "the target machine; launch never transferred; resident prefix capped by the "
                "target L2 capacity"
            ),
            "fitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        provenance["source_runs"] = []
        provenance["condition_count"] = 0
        provenance["raw_event_count"] = 0
        provenance["training_hash"] = ""
        provenance["heldout_guard"] = {
            "status": "not_applicable",
            "note": "transfer artifact carries no target observations of its own",
        }
        return PhysicalCalibration(
            hardware_key=limits.hardware_key,
            arch_family=limits.arch_family or self.arch_family,
            machine=limits,
            families=transferred,
            fit_config=dict(self.fit_config, transfer=True),
            provenance=provenance,
            unmodeled=tuple(
                dict(item, reason="transferred_model_unmodeled") for item in self.unmodeled
            ),
            schema_version=self.schema_version,
        )

    def report(self) -> dict[str, Any]:
        rows = []
        for key, parameters in sorted(self.families.items()):
            rows.append(
                {
                    "key": key,
                    "family": parameters.family,
                    "recipe": parameters.recipe,
                    "timing_scope": parameters.timing_scope,
                    "fit_level": parameters.fit_level,
                    "model_kind": parameters.model_kind,
                    "model_form": parameters.model_form,
                    "tail": dict(parameters.tail),
                    "k_fill_ms": parameters.k_fill_ms,
                    "align_penalty": parameters.align_penalty,
                    "launch_ms_by_scope": dict(parameters.launch_ms_by_scope),
                    "scope_samples": dict(parameters.scope_samples),
                    "samples": parameters.samples,
                    "raw_event_count": parameters.raw_event_count,
                    "train_wape_pct": parameters.train_wape_pct,
                    "cv_wape_pct": parameters.cv_wape_pct,
                    "median_ape_pct": parameters.median_ape_pct,
                    "p90_ape_pct": parameters.p90_ape_pct,
                    "max_ape_pct": parameters.max_ape_pct,
                    "memory_model": parameters.memory_model,
                    "block_m": parameters.block_m,
                    "block_n": parameters.block_n,
                    "launch_ms": parameters.launch_ms,
                    "eta_compute": parameters.eta_compute,
                    "eta_memory": parameters.eta_memory,
                    "vec_ops_per_s": parameters.vec_ops_per_s,
                    "l2_usable_bytes": parameters.l2_usable_bytes,
                    "l2_bw_bytes_per_s": parameters.l2_bw_bytes_per_s,
                    "intervals": {key: list(value) for key, value in parameters.intervals.items()},
                    "breakdown": {
                        key: (dict(value) if isinstance(value, Mapping) else value)
                        for key, value in parameters.breakdown.items()
                    },
                    "model_selection": dict(parameters.model_selection),
                    "quality": parameters.quality,
                    "warnings": list(parameters.warnings),
                }
            )
        return {
            "schema_version": self.schema_version,
            "hardware_key": self.hardware_key,
            "arch_family": self.arch_family,
            "machine": self.machine.to_dict(),
            "families": rows,
            "unmodeled": [dict(item) for item in self.unmodeled],
            "provenance": dict(self.provenance),
            "fit_config": dict(self.fit_config),
        }


@dataclass(frozen=True)
class _ShapeOnly:
    """Minimal stand-in carrying the fields ``_gemm_geometry`` reads."""

    family: str
    shape: Mapping[str, Any]
    elements: float
    flops: float
    hbm_bytes: float
    model_kind: str = "gemm"


# ---------------------------------------------------------------------------
# Fit entry point
# ---------------------------------------------------------------------------
def _observation_topology(observation: Observation) -> str:
    """Declared topology of one attention observation (shape-only)."""
    shape = observation.shape
    label = str(shape.get("topology") or "")
    if label:
        return label
    heads = float(shape.get("heads") or 0.0)
    kv_heads = float(shape.get("kv_heads") or heads or 0.0)
    return attention_topology(heads, kv_heads)


def attention_topology_class(topology: str) -> str:
    """Two-class attention topology used by the adoptable split.

    ``gqa`` keeps its own class; a single shared KV head (absorbed MLA / MQA) and
    an MHA-shaped MLA call (``q_heads == kv_heads``) share the
    ``shared_kv_mla`` class, because a 2-sample MHA class cannot be fitted on its
    own and both keep the whole KV axis in one tensor rather than a group of
    query heads per KV head.
    """
    return "gqa" if topology == "gqa" else "shared_kv_mla"


def _observation_topology_class(observation: Observation) -> str:
    return attention_topology_class(_observation_topology(observation))


def _observation_phase(observation: Observation) -> str:
    return str(
        observation.shape.get("phase")
        or attention_phase(observation.shape.get("q") or 0.0)
    )


def _pooled_cv(records: Sequence[Mapping[str, Any]]) -> tuple[float | None, int]:
    """Duration-weighted out-of-fold WAPE of a set of independently fitted rows."""
    absolute = 0.0
    total = 0.0
    samples = 0
    for record in records:
        triples = record.get("_oof") or ()
        if not triples:
            return None, samples
        for _index, prediction, target in triples:
            absolute += abs(prediction - target)
            total += target
            samples += 1
    if total <= 0:
        return None, samples
    return 100.0 * absolute / total, samples


def _attention_topology_split(
    record: Mapping[str, Any],
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    recipe: str,
    protocol_signature: str,
    protocol_core_signature: str,
    timing_scope: str,
    fit_level: str,
    compute_dtype: str,
    folds: int,
    min_samples: int,
) -> dict[str, Any] | None:
    """Topology-aware attention alternative, scored on the same folds.

    The pooled row is compared against one row per declared topology (and, if
    every resulting group still has ``min_samples``, per topology x phase).  The
    split is adopted only when its duration-weighted out-of-fold WAPE improves by
    ``MIN_CV_IMPROVEMENT_PCT``.  Efficiencies, tiles and the memory model stay
    per-row, so this is a real model-form comparison, not a per-shape correction.
    """
    assignment = _fold_index(observations, folds)
    position = {id(obs): index for index, obs in enumerate(observations)}

    def grouped(key_fn) -> dict[str, list[Observation]] | None:
        groups: dict[str, list[Observation]] = {}
        for observation in observations:
            groups.setdefault(key_fn(observation), []).append(observation)
        if len(groups) < 2:
            return None
        if any(len(items) < min_samples for items in groups.values()):
            return None
        return groups

    def phase_key(observation: Observation) -> str:
        return "%s/%s" % (
            _observation_topology(observation),
            str(observation.shape.get("phase") or attention_phase(observation.shape.get("q") or 0.0)),
        )

    def class_phase_key(observation: Observation) -> str:
        return "%s/%s" % (
            _observation_topology_class(observation), _observation_phase(observation)
        )

    candidates: list[dict[str, Any]] = []
    for axis, key_fn in (
        ("topology_class", _observation_topology_class),
        ("topology_class_phase", class_phase_key),
        ("topology", _observation_topology),
        ("topology_phase", phase_key),
    ):
        groups = grouped(key_fn)
        if groups is None:
            candidates.append(
                {
                    "axis": axis,
                    "adopted": False,
                    "reason": "insufficient_samples_per_group",
                }
            )
            continue
        records: list[dict[str, Any]] = []
        for label in sorted(groups):
            subset = groups[label]
            subset_folds = [assignment[position[id(item)]] for item in subset]
            fitted = _fit_scope_group(
                sorted(subset, key=lambda item: (item.identity, item.condition_key)),
                machine,
                family=record["family"],
                recipe=recipe,
                protocol_signature=protocol_signature,
                protocol_core_signature=protocol_core_signature,
                timing_scope=timing_scope,
                launch_scopes=(timing_scope,),
                fit_level="%s_%s" % (fit_level, axis),
                compute_dtype=compute_dtype,
                folds=folds,
                topology=label,
                fold_assignment=subset_folds,
            )
            records.append(fitted)
        pooled, samples = _pooled_cv(records)
        candidates.append(
            {
                "axis": axis,
                "groups": {
                    label: {
                        "samples": len(items),
                        "cv_wape_pct": next(
                            item["cv_wape_pct"] for item in records
                            if item["topology"] == label
                        ),
                    }
                    for label, items in sorted(groups.items())
                },
                "pooled_cv_wape_pct": pooled,
                "cv_samples": samples,
                "records": records,
            }
        )
    comparable = [
        item for item in candidates
        if item.get("pooled_cv_wape_pct") is not None and item.get("records")
    ]
    single_cv = record.get("cv_wape_pct")
    comparison: dict[str, Any] = {
        "chosen_on": "train_only_cv",
        "margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "single_cv_wape_pct": single_cv,
        "candidates": [
            {key: value for key, value in item.items() if key != "records"}
            for item in candidates
        ],
    }
    if not comparable:
        comparison["adopted"] = False
        comparison["reason"] = "no_adoptable_split"
        return comparison
    best = min(comparable, key=lambda item: item["pooled_cv_wape_pct"])
    gain = _difference(single_cv, best["pooled_cv_wape_pct"])
    comparison["best_axis"] = best["axis"]
    comparison["best_pooled_cv_wape_pct"] = best["pooled_cv_wape_pct"]
    comparison["cv_improvement_pct"] = gain
    if single_cv is not None and gain is not None and gain >= MIN_CV_IMPROVEMENT_PCT:
        comparison["adopted"] = True
        comparison["reason"] = "topology_split_supported_by_train_only_cv"
        comparison["records"] = best["records"]
    else:
        comparison["adopted"] = False
        comparison["reason"] = "pooled_attention_row_kept"
    return comparison


def _grouped_shape_split(
    record: Mapping[str, Any],
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    recipe: str,
    protocol_signature: str,
    protocol_core_signature: str,
    timing_scope: str,
    fit_level: str,
    compute_dtype: str,
    folds: int,
    min_samples: int,
) -> dict[str, Any] | None:
    """Weight-width class alternative for a grouped family, scored on the same folds.

    The narrow and wide classes overlap compute and weight streaming differently,
    and the pooled additive form can only satisfy both by sitting at a bound.  The
    split is a model-form decision: one row per declared class, same folds, pooled
    out-of-fold WAPE compared with the single row, adopted only when it improves by
    ``MIN_CV_IMPROVEMENT_PCT``.
    """
    groups: dict[str, list[Observation]] = {}
    for observation in observations:
        groups.setdefault(grouped_shape_class(observation.shape), []).append(observation)
    comparison: dict[str, Any] = {
        "chosen_on": "train_only_cv",
        "margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "axis": "weight_width_class",
        "threshold_n": GROUPED_NARROW_N,
        "single_cv_wape_pct": record.get("cv_wape_pct"),
        "groups": {
            label: {"samples": len(items)} for label, items in sorted(groups.items())
        },
    }
    if len(groups) < 2:
        comparison.update(adopted=False, reason="single_weight_width_class")
        return comparison
    if any(len(items) < min_samples for items in groups.values()):
        comparison.update(adopted=False, reason="insufficient_samples_per_group")
        return comparison
    assignment = _fold_index(observations, folds)
    position = {id(item): index for index, item in enumerate(observations)}
    records: list[dict[str, Any]] = []
    for label in sorted(groups):
        subset = groups[label]
        subset_folds = [assignment[position[id(item)]] for item in subset]
        records.append(
            _fit_scope_group(
                sorted(subset, key=lambda item: (item.identity, item.condition_key)),
                machine,
                family=record["family"],
                recipe=recipe,
                protocol_signature=protocol_signature,
                protocol_core_signature=protocol_core_signature,
                timing_scope=timing_scope,
                launch_scopes=(timing_scope,),
                fit_level="%s_weight_width" % fit_level,
                compute_dtype=compute_dtype,
                folds=folds,
                corpus_class=label,
                fold_assignment=subset_folds,
            )
        )
    pooled, samples = _pooled_cv(records)
    gain = _difference(record.get("cv_wape_pct"), pooled)
    comparison.update(
        pooled_cv_wape_pct=pooled,
        cv_samples=samples,
        cv_improvement_pct=gain,
        groups={
            item["corpus_class"]: {
                "samples": len(groups[item["corpus_class"]]),
                "cv_wape_pct": item["cv_wape_pct"],
                "model_form": item["model_form"],
                "block_m": item["block_m"],
                "block_n": item["block_n"],
                "eta_compute": item["eta_compute"],
                "eta_memory": item["eta_memory"],
            }
            for item in records
        },
    )
    if gain is not None and gain >= MIN_CV_IMPROVEMENT_PCT:
        comparison.update(adopted=True, reason="weight_width_split_supported_by_train_only_cv",
                          records=records)
    else:
        comparison.update(adopted=False, reason="pooled_grouped_row_kept")
    return comparison


def _merge_repeat_conditions(
    observations: Sequence[Observation],
) -> tuple[list[Observation], dict[str, int]]:
    """One row per measured physical condition.

    Campaign directories can be mirrored (the same run appears under two names)
    and a condition can be re-measured.  Either way the target for a condition is
    a single number, so repeats are merged by median instead of being fed to the
    fit as independent rows (which would double-weight them).  Runs are told
    apart by ``condition_key`` when the runner provides one.
    """
    groups: dict[tuple[Any, ...], list[Observation]] = {}
    for observation in observations:
        if observation.condition_key:
            key = (observation.condition_key, observation.protocol_signature, observation.timing_scope)
        else:
            key = (
                observation.identity,
                observation.protocol_signature,
                observation.timing_scope,
                observation.package_tree,
                round(observation.duration_ms, 9),
            )
        groups.setdefault(key, []).append(observation)
    merged: list[Observation] = []
    duplicates = 0
    repeats = 0
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        group = groups[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        duplicates += len(group) - 1
        durations = sorted(item.duration_ms for item in group)
        middle = len(durations) // 2
        median = (
            durations[middle]
            if len(durations) % 2
            else 0.5 * (durations[middle - 1] + durations[middle])
        )
        if len({round(item.duration_ms, 9) for item in group}) == 1:
            # Byte-identical mirror of the same measurement, not a repeat.
            merged.append(group[0])
            continue
        repeats += 1
        merged.append(
            dataclasses.replace(
                group[0],
                duration_ms=median,
                raw_event_count=max(item.raw_event_count for item in group),
                target_estimator="repeat_condition_median",
            )
        )
    return merged, {
        "observations_deduplicated": duplicates,
        "repeat_conditions_merged": repeats,
        "conditions_after_merge": len(merged),
    }


def _count_by(observations: Sequence[Observation], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for observation in observations:
        value = key(observation)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _reject(rejections: dict[str, Any], reason: str, detail: str) -> None:
    entry = rejections.setdefault(reason, {"count": 0, "examples": []})
    entry["count"] += 1
    if len(entry["examples"]) < 5:
        entry["examples"].append(detail)


def fit_calibration(
    observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    machine: Mapping[str, Any],
    *,
    hardware_key: str = "",
    arch_family: str = "",
    compute_dtype: str = "bf16",
    family_kinds: Mapping[str, str] | None = None,
    min_samples: int = MIN_SAMPLES_PER_FIT,
    cv_folds: int = CV_FOLDS,
    accepted_package_trees: Sequence[str] = (),
    heldout_guard: HeldoutGuard | None = None,
    git_head: str | None = None,
    source_artifacts: Sequence[str] = (),
) -> PhysicalCalibration:
    """Fit the physical model on train observations only.

    ``observations`` is either the observation document
    (``{"schema_version": OBSERVATION_SCHEMA, "observations": [...]}``) or the
    bare list.  Non-train split, non-measured source, foreign hardware key,
    unsupported timing scope, empty protocol signature and missing shape are
    rejected with a counted reason before any parameter is estimated.
    """
    kind_table = dict(MODEL_KINDS)
    if family_kinds:
        kind_table.update({str(k): str(v) for k, v in family_kinds.items()})
    if isinstance(observations, Mapping):
        document = observations
        declared = str(document.get("schema_version") or "")
        if declared and declared != OBSERVATION_SCHEMA:
            raise PhysicalCalibrationError("unsupported_observation_schema:%r" % (declared,))
        rows = list(document.get("observations") or ())
    else:
        rows = list(observations)
    limits = MachineLimits.from_mapping(machine)
    target_key = hardware_key or limits.hardware_key
    target_arch = arch_family or limits.arch_family
    if not target_key:
        raise PhysicalCalibrationError("missing_hardware_key")

    rejections: dict[str, Any] = {}
    accepted: list[Observation] = []
    for row in rows:
        detail = ""
        try:
            family = str((row or {}).get("family") or "")
            if not family:
                raise PhysicalCalibrationError("missing_family")
            if family not in kind_table:
                _reject(rejections, "unsupported_family", family)
                continue
            observation = Observation.from_mapping(row, model_kind=kind_table[family])
            detail = observation.condition_key or observation.identity
        except PhysicalCalibrationError as exc:
            _reject(rejections, str(exc), str((row or {}).get("condition_key") or "?"))
            continue
        if observation.split != "train":
            _reject(rejections, "non_train_split:%s" % observation.split, detail)
            continue
        if observation.hardware_key and observation.hardware_key != target_key:
            _reject(rejections, "hardware_key_mismatch:%s" % observation.hardware_key, detail)
            continue
        if observation.source_class not in ACCEPTED_SOURCE_CLASSES:
            _reject(rejections, "non_measured_source_class:%s" % (observation.source_class or "unknown"), detail)
            continue
        if observation.timing_scope not in ACCEPTED_TIMING_SCOPES:
            _reject(rejections, "unsupported_timing_scope:%s" % (observation.timing_scope or "unknown"), detail)
            continue
        if not observation.protocol_signature:
            _reject(rejections, "ambiguous_protocol_signature", detail)
            continue
        if (
            accepted_package_trees
            and observation.package_tree
            and observation.package_tree not in accepted_package_trees
        ):
            # A run from a divergent mindie_sim package is not comparable data.
            _reject(
                rejections,
                "divergent_package_tree:%s" % observation.package_tree[:12],
                detail,
            )
            continue
        if observation.dtype and observation.dtype != compute_dtype:
            _reject(rejections, "dtype_outside_fit:%s" % observation.dtype, detail)
            continue
        accepted.append(observation)
    if not accepted:
        raise PhysicalCalibrationError("no_train_observations_after_rejection")
    accepted, merge_stats = _merge_repeat_conditions(accepted)
    guard_status = "absent"
    overlap: list[str] = []
    if heldout_guard is not None:
        overlap = heldout_guard.overlap(accepted)
        if overlap:
            raise PhysicalCalibrationError(
                "heldout_identity_in_training_set:%s" % (sorted(overlap)[:5],)
            )
        guard_status = "clean"

    # Per-scope fit identity: (family, recipe, protocol signature, timing scope).
    # Two timing scopes are different physical protocols (a batched interval
    # amortises dispatch), so the conservative baseline never shares them.
    groups: dict[tuple[str, str, str, str], list[Observation]] = {}
    for observation in accepted:
        groups.setdefault(
            (
                observation.family,
                observation.recipe,
                observation.protocol_signature,
                observation.timing_scope,
            ),
            [],
        ).append(observation)
    # Cross-scope identity: (family, protocol core signature).  The core excludes
    # the timing scope and the repeat settings, so a batched and a per-call
    # interval of the same physical protocol can share efficiencies while keeping
    # their own launch intercept.
    cores: dict[tuple[str, str], list[Observation]] = {}
    for observation in accepted:
        cores.setdefault(
            (observation.family, observation.protocol_core_signature), []
        ).append(observation)
    protocol_count = len({key[2] for key in groups})
    scope_count = len({key[3] for key in groups})
    protocol_of_family: dict[str, set[str]] = {}
    for family, _recipe, signature, scope in groups:
        protocol_of_family.setdefault(family, set()).add("%s|%s" % (signature, scope))

    families: dict[str, FamilyParameters] = {}
    unmodeled: list[dict[str, Any]] = []
    cv_details: dict[str, Any] = {}
    family_fits: dict[str, dict[str, Any]] = {}
    for (family, recipe, signature, scope), group in sorted(groups.items()):
        group_sorted = sorted(group, key=lambda item: (item.identity, item.condition_key))
        record: dict[str, Any] | None = None
        if len(group_sorted) >= min_samples:
            record = _fit_scope_group(
                group_sorted,
                limits,
                family=family,
                recipe=recipe,
                protocol_signature=signature,
                protocol_core_signature=group_sorted[0].protocol_core_signature,
                timing_scope=scope,
                launch_scopes=(scope,),
                fit_level="family_recipe_protocol",
                compute_dtype=compute_dtype,
                folds=cv_folds,
            )
        else:
            pooled = [
                item
                for item in accepted
                if item.family == family
                and item.protocol_signature == signature
                and item.timing_scope == scope
            ]
            if len(pooled) >= min_samples and len({item.recipe for item in pooled}) > 1:
                record = _fit_scope_group(
                    sorted(pooled, key=lambda item: (item.identity, item.condition_key)),
                    limits,
                    family=family,
                    recipe="*",
                    protocol_signature=signature,
                    protocol_core_signature=pooled[0].protocol_core_signature,
                    timing_scope=scope,
                    launch_scopes=(scope,),
                    fit_level="family_protocol",
                    compute_dtype=compute_dtype,
                    folds=cv_folds,
                )
            else:
                unmodeled.append(
                    {
                        "family": family,
                        "recipe": recipe,
                        "protocol_signature": signature,
                        "timing_scope": scope,
                        "reason": "insufficient_samples",
                        "samples": len(group_sorted),
                        "min_samples": min_samples,
                    }
                )
        if record is None:
            continue
        family_fits.setdefault(family, {"per_scope": [], "core": None})["per_scope"].append(record)
        family_fits[family]["core"] = record["protocol_core_signature"]

    # Attention topology: a topology-aware alternative (and topology x phase, when
    # every group still clears ``min_samples``) is scored out-of-fold on the same
    # folds as the pooled row, and adopted only when it wins by the margin.
    for family, bundle in sorted(family_fits.items()):
        records = bundle["per_scope"]
        if not records or all(item["model_kind"] != "attention" for item in records):
            continue
        resolved: list[dict[str, Any]] = []
        for record in records:
            if record["model_kind"] != "attention":
                resolved.append(record)
                continue
            comparison = _attention_topology_split(
                record,
                record.get("_observations") or (),
                limits,
                recipe=record["recipe"],
                protocol_signature=record["protocol_signature"],
                protocol_core_signature=record["protocol_core_signature"],
                timing_scope=record["timing_scope"],
                fit_level=record["fit_level"],
                compute_dtype=compute_dtype,
                folds=cv_folds,
                min_samples=min_samples,
            )
            if comparison is None:
                resolved.append(record)
                continue
            record["model_selection"] = dict(
                record["model_selection"], topology_split=comparison
            )
            if comparison.get("adopted"):
                split_records = comparison.pop("records")
                for item in split_records:
                    # The comparison is the provenance of every split row.
                    item["model_selection"] = dict(
                        item["model_selection"], topology_split=comparison
                    )
                resolved.extend(split_records)
            else:
                resolved.append(record)
        bundle["per_scope"] = resolved

    # Grouped matmul: a weight-width split is scored out-of-fold on the same folds
    # and adopted only when it beats the pooled row by the margin.
    for family, bundle in sorted(family_fits.items()):
        records = bundle["per_scope"]
        if not records or all(item["model_kind"] != "grouped_gemm" for item in records):
            continue
        resolved: list[dict[str, Any]] = []
        for record in records:
            if record["model_kind"] != "grouped_gemm":
                resolved.append(record)
                continue
            comparison = _grouped_shape_split(
                record,
                record.get("_observations") or (),
                limits,
                recipe=record["recipe"],
                protocol_signature=record["protocol_signature"],
                protocol_core_signature=record["protocol_core_signature"],
                timing_scope=record["timing_scope"],
                fit_level=record["fit_level"],
                compute_dtype=compute_dtype,
                folds=cv_folds,
                min_samples=min_samples,
            )
            if comparison is None:
                resolved.append(record)
                continue
            record["model_selection"] = dict(
                record["model_selection"], shape_split=comparison
            )
            if comparison.get("adopted"):
                split_records = comparison.pop("records")
                for item in split_records:
                    item["model_selection"] = dict(
                        item["model_selection"], shape_split=comparison
                    )
                resolved.extend(split_records)
            else:
                resolved.append(record)
        bundle["per_scope"] = resolved

    # Cross-scope shared variant: one efficiency / tile / L2 set, one launch per
    # scope.  Adopted only when the pooled train-only CV beats the per-scope fits.
    for family, bundle in sorted(family_fits.items()):
        per_scope = bundle["per_scope"]
        core = bundle["core"]
        if any(record.get("topology") for record in per_scope):
            # A topology split keeps its rows: pooling them again through a
            # shared-efficiency variant would silently undo the comparison.
            bundle["decision"] = {
                "compared": False,
                "reason": "topology_rows_kept_separate",
                "chosen": "per_scope_independent",
                "scopes": sorted({record["timing_scope"] for record in per_scope}),
            }
            continue
        scopes = sorted({record["timing_scope"] for record in per_scope})
        if len(scopes) < 2:
            family_fits[family]["decision"] = {
                "compared": False,
                "reason": "single_timing_scope",
                "chosen": "per_scope_independent",
                "scopes": scopes,
            }
            continue
        union = sorted(
            [obs for obs in accepted if obs.family == family
             and obs.protocol_core_signature == core],
            key=lambda item: (item.timing_scope, item.identity, item.condition_key),
        )
        if len(union) < 2 * min_samples:
            family_fits[family]["decision"] = {
                "compared": False,
                "reason": "insufficient_samples_for_shared_fit",
                "chosen": "per_scope_independent",
                "scopes": scopes,
            }
            continue
        union_scopes = sorted({obs.timing_scope for obs in union})
        shared = _fit_scope_group(
            union,
            limits,
            family=family,
            recipe="*",
            protocol_signature="",
            protocol_core_signature=core,
            timing_scope="*",
            launch_scopes=union_scopes,
            fit_level="family_scope_shared",
            compute_dtype=compute_dtype,
            folds=cv_folds,
        )
        # The per-scope baseline is scored out-of-fold with the SAME fold
        # assignment as the shared model, so the two numbers are comparable.
        assignment = _fold_index(union, cv_folds)
        pooled_abs = 0.0
        pooled_total = 0.0
        for record in per_scope:
            scope_index = [
                i for i, obs in enumerate(union)
                if obs.timing_scope == record["timing_scope"]
            ]
            observations_scope = [union[i] for i in scope_index]
            folds_local = [assignment[i] for i in scope_index]
            rows = _build_rows(
                observations_scope,
                limits,
                form=record["model_form"],
                memory_model=record["memory_model"],
                l2_usable=record["l2_usable_bytes"],
                block_m=record["block_m"],
                block_n=record["block_n"],
                tail=record["tail"],
                launch_scopes=(record["timing_scope"],),
                k_fill=record["k_fill"],
                model_terms=record.get("model_terms"),
            )
            bounds = _parameter_bounds(
                1, record["model_kind"], record["memory_model"], record["k_fill"],
                bool((record["tail"] or {}).get("n")),
                model_terms=record.get("model_terms"), tail=record["tail"],
                launch_bounds=[
                    _launch_bounds_for(record["model_kind"], record["timing_scope"], limits)
                ],
            )
            for fold in range(cv_folds):
                train_rows = [row for row, held in zip(rows, folds_local) if held != fold]
                test_rows = [row for row, held in zip(rows, folds_local) if held == fold]
                if not train_rows or not test_rows:
                    continue
                theta = _irls_fit(train_rows, bounds)
                for row in test_rows:
                    prediction = _evaluate(row, theta)
                    pooled_abs += abs(prediction - row["target"])
                    pooled_total += row["target"]
        baseline_cv = 100.0 * pooled_abs / pooled_total if pooled_total else None
        gain = _difference(baseline_cv, shared["cv_wape_pct"])
        adopted = gain is not None and gain >= MIN_SHARED_SCOPE_IMPROVEMENT_PCT
        family_fits[family]["shared"] = shared
        family_fits[family]["decision"] = {
            "compared": True,
            "chosen": "shared_efficiencies_per_scope_launch" if adopted else "per_scope_independent",
            "scopes": union_scopes,
            "per_scope_pooled_cv_wape_pct": baseline_cv,
            "shared_cv_wape_pct": shared["cv_wape_pct"],
            "cv_improvement_pct": gain,
            "margin_pct": MIN_SHARED_SCOPE_IMPROVEMENT_PCT,
            "baseline_estimator": "per-scope model refit on 4/5 folds, scored on fold 0",
            "shared_scope_samples": shared["scope_samples"],
        }
        if adopted:
            family_fits[family]["per_scope"] = []

    for family, bundle in sorted(family_fits.items()):
        records = list(bundle["per_scope"])
        if bundle.get("shared") is not None and not records:
            records = [bundle["shared"]]
        for record in records:
            decisive = bundle.get("decision") or {}
            if decisive.get("compared"):
                record["memory_model_selection"] = dict(
                    record["memory_model_selection"], scope_selection=decisive
                )
            # The key and the stored recipe must agree: a pooled row keyed with
            # "*" has to answer for "*", otherwise it is unreachable.
            record["recipe"] = record["recipe"] or "*"
            key = "%s/%s" % (record["family"], record["recipe"])
            if record.get("topology"):
                key = "%s@%s" % (key, record["topology"])
            if record.get("corpus_class"):
                key = "%s@%s" % (key, record["corpus_class"])
            if key in families:
                suffix = (record["protocol_signature"] or record["protocol_core_signature"]).replace(
                    "sha256:", ""
                )[:8]
                key = "%s#%s" % (key, suffix)
            cv_details[key] = record.pop("cv_detail", {})
            record.pop("_theta", None)
            record.pop("_oof", None)
            record.pop("_observations", None)
            families[key] = FamilyParameters.from_mapping(record)
    for family, signatures in sorted(protocol_of_family.items()):
        if len(signatures) > 1:
            unmodeled.append(
                {
                    "family": family,
                    "reason": "protocol_groups_kept_separate",
                    "protocol_signatures": sorted(signatures),
                    "note": (
                        "per-scope fits stay separate unless the shared-efficiency variant wins "
                        "on pooled train-only CV; a launch intercept is never transferred"
                    ),
                }
            )

    fitted_keys = {
        (item.family, item.recipe, item.protocol_signature, item.timing_scope)
        for item in families.values()
    }
    for family in sorted({item.family for item in accepted}):
        if not any(item[0] == family for item in fitted_keys):
            unmodeled.append(
                {"family": family, "reason": "no_family_fit", "note": "all groups below min_samples"}
            )

    # Declared coverage of every family this module can model.  A family with no
    # measured train observation is *unmodeled*, and it is never answered from
    # another family's parameters (see ``predict_op_explained``).
    observed_counts = _count_by(accepted, lambda obs: obs.family)
    family_coverage: dict[str, Any] = {}
    for family in sorted(kind_table):
        keys = [key for key in families if families[key].family == family]
        observed = observed_counts.get(family, 0)
        if keys:
            status = "fitted"
        elif observed:
            status = "unmodeled_insufficient_samples"
        else:
            status = "unmodeled_no_measured_train_observations"
        family_coverage[family] = {
            "model_kind": kind_table[family],
            "status": status,
            "train_observations": observed,
            "fitted_keys": sorted(keys),
            "timing_scopes": sorted(
                {obs.timing_scope for obs in accepted if obs.family == family}
            ),
        }

    training_rows = [
        {
            "condition_key": obs.condition_key,
            "identity": obs.identity,
            "family": obs.family,
            "recipe": obs.recipe,
            "protocol_signature": obs.protocol_signature,
            "duration_ms": round(obs.duration_ms, 9),
            "flops": obs.flops,
            "hbm_bytes": obs.hbm_bytes,
            "elements": obs.elements,
            "raw_event_count": obs.raw_event_count,
            "run_id": obs.run_id,
        }
        for obs in sorted(accepted, key=lambda item: (item.identity, item.condition_key))
    ]
    provenance = {
        "source_runs": sorted({obs.run_id for obs in accepted if obs.run_id}),
        "condition_count": len(accepted),
        "repeat_merge": merge_stats,
        "raw_event_count": sum(obs.raw_event_count for obs in accepted),
        "protocol_signature_count": protocol_count,
        "timing_scope_count": scope_count,
        "package_trees": {
            tree or "unknown": count
            for tree, count in sorted(_count_by(accepted, lambda obs: obs.package_tree).items())
        },
        "accepted_package_trees": list(accepted_package_trees),
        "scope_samples": {
            scope: count
            for scope, count in sorted(_count_by(accepted, lambda obs: obs.timing_scope).items())
        },
        "timing_scopes": sorted({obs.timing_scope for obs in accepted if obs.timing_scope}),
        "hardware_keys": sorted({obs.hardware_key for obs in accepted if obs.hardware_key}),
        "dies": sorted({obs.die for obs in accepted if obs.die}),
        "training_hash": _stable_digest(training_rows),
        "training_hash_scope": "condition_key|identity|family|recipe|protocol|duration|flops|bytes|elements|raw_event_count|run_id",
        "git_head": git_head,
        "fitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "split_policy": "train_only",
        "target_estimators": sorted({obs.target_estimator for obs in accepted if obs.target_estimator}),
        "target_estimator_note": (
            "median_of_trial_medians is the acceptance target; pooled_repeat_median is a weaker "
            "estimator of the same quantity and is recorded as a limitation"
            if "pooled_repeat_median" in {obs.target_estimator for obs in accepted}
            else "median_of_trial_medians or declared_duration_ms"
        ),
        "heldout_guard": (
            heldout_guard.to_dict(status=guard_status, overlap=overlap)
            if heldout_guard is not None
            else {"status": "absent", "warning": "no heldout guard declared; only split=train enforced"}
        ),
        "rejections": {
            "total": sum(entry["count"] for entry in rejections.values()),
            "by_reason": {reason: entry["count"] for reason, entry in sorted(rejections.items())},
            "examples": {reason: entry["examples"] for reason, entry in sorted(rejections.items())},
        },
        "source_artifacts": list(source_artifacts),
        "family_coverage": family_coverage,
        "generator": "mindie_sim.a3_physical_calibration",
        "generator_schema": SCHEMA_VERSION,
        "cv_detail": cv_details,
    }
    fit_config = {
        "block_m_choices": list(BLOCK_M_CHOICES),
        "block_n_choices": list(BLOCK_N_CHOICES),
        "launch_bounds_ms": list(LAUNCH_BOUNDS_MS),
        "eta_bounds": list(ETA_BOUNDS),
        "l2_usable_fractions": list(L2_USABLE_FRACTIONS),
        "l2_bw_multiple_bounds": list(L2_BW_MULTIPLE_BOUNDS),
        "huber_k": HUBER_K,
        "irls_iterations": IRLS_ITERATIONS,
        "min_samples": min_samples,
        "cv_folds": cv_folds,
        "cv_method": "train_only_kfold",
        "selection_criterion": "train_wape_pct",
        "memory_model_rule": "blend requires >=%.1f pct train-only CV improvement" % MIN_CV_IMPROVEMENT_PCT,
        "family_aliases": dict(FAMILY_ALIASES),
        "model_forms": list(MODEL_FORMS),
        "tail_terms": ["m", "n", "k"],
        "min_tail_improvement_pct": MIN_TAIL_IMPROVEMENT_PCT,
        "shared_scope_min_improvement_pct": MIN_SHARED_SCOPE_IMPROVEMENT_PCT,
        "k_fill_max_k": K_FILL_MAX_K,
        "k_fill_bounds_ms": list(K_FILL_BOUNDS_MS),
        "model_form_descriptions": dict(FORM_DESCRIPTIONS),
        "model_selection_margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "cache_probe_min_samples": MIN_CACHE_PROBE_SAMPLES,
        "min_identifying_rows": MIN_IDENTIFYING_ROWS,
        "min_identifying_share": MIN_IDENTIFYING_SHARE,
        "max_trusted_cv_wape_pct": MAX_TRUSTED_CV_WAPE_PCT,
        "max_trusted_p90_ape_pct": MAX_TRUSTED_P90_APE_PCT,
        "ci_tolerance": CI_TOLERANCE,
        "vec_issue_max_per_aiv": VEC_ISSUE_MAX_PER_AIV,
        "vec_throughput_min_fraction": VEC_THROUGHPUT_MIN_FRACTION,
        "accepted_timing_scopes": list(ACCEPTED_TIMING_SCOPES),
        "protocol_core_fields": [
            "timer", "preallocated", "cache_policy", "compute_dtype", "weight_dtype",
            "activation_dtype", "accumulation_dtype", "quantization", "layout",
        ],
        "protocol_core_defaults": {
            "compute_dtype": "bf16",
            "weight_dtype": "bf16",
            "activation_dtype": "bf16",
            "accumulation_dtype": "backend_default",
            "quantization": "none",
            "layout": "contiguous",
            "cache_policy": DECLARED_CACHE_POLICY,
        },
        "protocol_core_excluded": [
            "timing_scope", "measurement_scope", "warmup", "warmup_scope", "iterations", "trials",
        ],
        "accepted_source_classes": list(ACCEPTED_SOURCE_CLASSES),
        "compute_dtype": compute_dtype,
        "attention": {
            "families": list(ATTENTION_KINDS),
            "model": (
                "t = launch[scope] + tail_tile_ms*tail_tiles + "
                "max(flops_eff/(peak*eta_compute*wave_eff), bytes/(bw*eta_memory))"
            ),
            "flops": "2*q*kv*heads*dim per matmul; 4*q*kv*heads*dim for a fused QK+PV call",
            "causal_work": (
                "visible_kv = kv - (q-1)/2 for a causal_bottom_right mask with kv >= q, "
                "kv otherwise (prior context fully visible, in-chunk triangular)"
            ),
            "bytes": (
                "Q read (q*heads*qk_dim) + KV read (kv*kv_heads*(qk_dim+v_dim)) + "
                "O write (q*heads*v_dim), each x dtype, plus workspace "
                "%.0f bytes per (query, head)" % ATTENTION_WORKSPACE_BYTES_PER_QUERY_HEAD
            ),
            "tiles": {
                "q": list(ATTENTION_Q_TILES),
                "kv": list(ATTENTION_KV_TILES),
                "dim": ATTENTION_DIM_TILE,
            },
            "tail_bounds_ms": list(ATTENTION_TAIL_BOUNDS_MS),
            "dispatch_overhead_ms": ATTENTION_DISPATCH_OVERHEAD_MS,
            "dispatch_launch_bounds_ms": list(DISPATCH_LAUNCH_BOUNDS_MS),
            "dispatch_source": (
                ".workstreams/bigops-a3/results/dispatch-overhead.json "
                "(19.8-20.2 us per-call device dispatch; 11.7-11.9 us batched per kernel)"
            ),
            "topologies": list(ATTENTION_TOPOLOGIES),
            "topology_rule": (
                "classified from (q_heads, kv_heads): mha if q_heads == kv_heads, "
                "shared_kv if kv_heads == 1 (absorbed MLA / MQA), gqa otherwise"
            ),
            "topology_split": (
                "pooled row vs per-topology rows (and per topology x phase when every group "
                "clears min_samples), scored out-of-fold on the same folds; adopted only when "
                "train-only CV improves by %.1f pp" % MIN_CV_IMPROVEMENT_PCT
            ),
            "kv_broadcast": {
                "choices": list(ATTENTION_KV_BROADCAST_CHOICES),
                "rule": (
                    "effective KV traffic is kv * max(kv_heads, ceil(q_heads / B)) * "
                    "(qk_dim + v_dim); None is the declared base formula (one read per KV "
                    "head). B is a CV variant, adopted only above the variant margin."
                ),
                "evidence": (
                    "MLA decode q=1 kv=65536 with 64 query heads and 1 KV head measures "
                    "9.84 ms against 142 MB of base KV bytes, which only a per-query-head "
                    "re-read explains"
                ),
            },
            "absorbed_mla": (
                "qk_dim and v_dim are separate measured axes: fused FLOPs are "
                "2*q*kv*heads*(qk_dim + v_dim) and are never assumed equal"
            ),
            "timing_scopes": [ATTENTION_DEVICE_SCOPE, ATTENTION_BLOCK_SCOPE],
            "scope_note": (
                "device_event_warm_operator is the per-call device event; "
                "block_interval_warm_operator is the completion-clock block interval whose "
                "elapsed time includes host submit and allocation gaps; they never share an "
                "intercept, a fit or a protocol signature"
            ),
            "never_gemm": (
                "attention families are fitted and predicted by this model only; they are "
                "never aliased onto, fitted from, or predicted by the gemm family"
            ),
        },
        "grouped_gemm": {
            "families": list(GROUPED_KINDS),
            "model": (
                "t = launch + per_expert_ms*experts**expert_scaling + "
                "max(flops_eff/(peak*eta_compute*wave_eff), bytes/(bw*eta_memory))"
            ),
            "flops": "2*tokens*n*k (tokens = routed tokens; each expert contracts its own block)",
            "bytes": (
                "experts*n*k (per-expert weight read) + tokens*k (per-expert activation read, "
                "optional convention) + tokens*n (output), each x dtype"
            ),
            "expert_scaling_choices": list(GROUPED_EXPERT_SCALING_CHOICES),
            "per_expert_bounds_ms": list(GROUPED_PER_EXPERT_BOUNDS_MS),
            "launch_capacity": GROUPED_LAUNCH_CAPACITY,
            "per_launch_bounds_ms": list(GROUPED_PER_LAUNCH_BOUNDS_MS),
            "launch_count_rule": "ceil(experts / %d) sequential launches" % GROUPED_LAUNCH_CAPACITY,
            "eta_memory_max": GROUPED_ETA_MEMORY_MAX,
            "eta_memory_note": (
                "a grouped family's memory efficiency is an effective bandwidth: the declared "
                "SKU HBM peak is conservative and measured weight streaming is >= 1.36 TB/s, so "
                "eta_memory may exceed 1.0 for this kind and is reported as such"
            ),
            "activation_read_default": GROUPED_DEFAULT_ACTIVATION_READ,
            "dense_transfer_baseline": (
                "every grouped fit is scored against a dense-GEMM transfer on the same rows "
                "and the same folds; the grouped fit is kept and the comparison is reported"
            ),
        },
    }
    return PhysicalCalibration(
        hardware_key=target_key,
        arch_family=target_arch,
        machine=limits,
        families=families,
        fit_config=fit_config,
        provenance=provenance,
        unmodeled=tuple(unmodeled),
    )


# ---------------------------------------------------------------------------
# Observation readers
# ---------------------------------------------------------------------------
def read_observations_json(path: str | Path) -> dict[str, Any]:
    """Read an observation document and validate its declared schema."""
    value = json.loads(Path(path).read_text())
    if isinstance(value, Mapping):
        declared = str(value.get("schema_version") or "")
        if declared and declared != OBSERVATION_SCHEMA:
            raise PhysicalCalibrationError("unsupported_observation_schema:%r" % (declared,))
        return dict(value)
    if isinstance(value, list):
        return {"schema_version": OBSERVATION_SCHEMA, "observations": value}
    raise PhysicalCalibrationError("observation_document_must_be_mapping_or_list")


_VECTOR_BYTES_PER_ELEMENT = {
    # Declared logical byte conventions for the vector families: tensors read and
    # written per element, times the dtype size (see the readers).
    "copy": 2.0,
    "cast": 2.0,
    "rmsnorm": 2.0,
    "softmax": 2.0,
    "masked_softmax": 2.0,
    "quantize": 2.0,
    "rope": 2.0,
    "add": 3.0,
    "swiglu": 3.0,
}
_SQLITE_COLUMNS = (
    "sample_key, op_family, recipe, shape_json, dtype, split, duration_median, measurement_count,"
    " run_ids_json, source_classes_json, timing_scope, protocol_json, hardware_key, environment_json"
)


def read_sqlite_observations(
    path: str | Path,
    *,
    hardware_key: str = "",
    arch_family: str = "",
    include_splits: Sequence[str] = ("train",),
    die: str = "",
) -> dict[str, Any]:
    """Fallback reader for a ``sample_statistics`` calibration store.

    Duration target is ``duration_median`` (median of per-trial event medians is
    not available in this store); ``raw_samples_ms`` and ``raw_event_count``
    come from the ``measurements`` table, so the artifact stays traceable to raw
    events.  Logical bytes follow the declared per-family convention: a GEMM
    touches ``m*k + k*n + m*n`` elements, ``copy`` reads and writes one array,
    ``add`` reads two and writes one.
    """
    connection = sqlite3.connect("file:%s?mode=ro" % Path(path), uri=True)
    try:
        rows = connection.execute(
            "select %s from sample_statistics order by sample_key" % _SQLITE_COLUMNS
        ).fetchall()
        raw_by_sample: dict[str, list[float]] = {}
        for sample_key, duration in connection.execute(
            "select sample_key, duration_ms from measurements order by sample_key, repeat_index"
        ):
            raw_by_sample.setdefault(sample_key, []).append(float(duration))
    finally:
        connection.close()
    observations = []
    skipped = {"split": 0, "family": 0, "shape": 0, "duration": 0}
    for row in rows:
        (sample_key, op_family, recipe, shape_json, dtype, split, duration_median,
         measurement_count, run_ids_json, source_classes_json, timing_scope,
         protocol_json, row_hardware_key, environment_json) = row
        if include_splits and split not in include_splits:
            skipped["split"] += 1
            continue
        if op_family not in MODEL_KINDS:
            skipped["family"] += 1
            continue
        kind = MODEL_KINDS[op_family]
        shape = json.loads(shape_json)
        itemsize = _reader_itemsize(dtype)
        if itemsize is None:
            # An undeclared dtype width is a rejection with a reason, never a
            # silent bf16 (2-byte) assumption.
            skipped["dtype"] = skipped.get("dtype", 0) + 1
            continue
        shape = {k: float(v) for k, v in shape.items() if v is not None}
        if kind == "gemm":
            m, n, k = shape.get("m"), shape.get("n"), shape.get("k")
            if not (m and n and k):
                skipped["shape"] += 1
                continue
            flops = 2.0 * m * n * k
            hbm_bytes = (m * k + k * n + m * n) * itemsize
            elements = None
        else:
            elements = shape.get("elements")
            if not elements:
                skipped["shape"] += 1
                continue
            flops = 0.0
            hbm_bytes = elements * itemsize * _VECTOR_BYTES_PER_ELEMENT[op_family]
        duration = float(duration_median) if duration_median else 0.0
        if duration <= 0:
            skipped["duration"] += 1
            continue
        protocol = json.loads(protocol_json or "{}")
        environment = json.loads(environment_json or "{}")
        sources = json.loads(source_classes_json or "[]")
        run_ids = json.loads(run_ids_json or "[]")
        observations.append(
            {
                "condition_key": sample_key,
                "hardware_key": str(row_hardware_key or hardware_key),
                "die": str(environment.get("die") or die or ""),
                "arch_family": arch_family,
                "family": op_family,
                "recipe": str(recipe or ""),
                "protocol_signature": protocol_signature(protocol),
                "run_id": str(run_ids[0]) if run_ids else "",
                "split": split,
                "shape": shape,
                "elements": elements,
                "flops": flops,
                "hbm_bytes": hbm_bytes,
                "duration_ms": duration,
        "target_estimator": "pooled_repeat_median",
                "raw_samples_ms": raw_by_sample.get(sample_key, []),
                "raw_event_count": int(measurement_count or 0),
                "dtype": str(dtype or ""),
                "source_class": str(sources[0]) if sources else "unknown",
                "timing_scope": str(timing_scope or ""),
                "protocol": protocol,
                "protocol_core_signature": protocol_core_signature(protocol),
            }
        )
    if not hardware_key:
        hardware_key = next((item["hardware_key"] for item in observations if item["hardware_key"]), "")
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "source": "sqlite:%s" % Path(path).name,
        "source_path": str(Path(path)),
        "hardware_key": hardware_key,
        "arch_family": arch_family,
        "skipped": skipped,
        "observations": observations,
    }


DECLARED_CACHE_POLICY = "same_preallocated_buffers_reused_no_cache_flush"

_DTYPE_ALIASES = {
    "bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16",
    "float32": "fp32", "fp32": "fp32", "float8": "fp8", "fp8": "fp8",
    "float8_e4m3": "fp8_e4m3", "float8_e4m3fn": "fp8_e4m3",
    "fp8_e4m3fn": "fp8_e4m3", "e4m3": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2", "fp8_e5m2": "fp8_e5m2", "e5m2": "fp8_e5m2",
    "int8": "int8", "int4": "int4", "int32": "int32", "bool": "bool",
}


def _reader_itemsize(dtype: Any) -> float | None:
    """Byte width for a reader; ``None`` means the dtype is not declared."""
    text = _normalise_dtype(dtype)
    return DTYPE_BYTES.get(text) if text else None


def _normalise_dtype(value: Any) -> str:
    text = str(value or "").strip().lower()
    return _DTYPE_ALIASES.get(text, text)


def protocol_core(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Timing-scope-independent protocol identity.

    A batched interval and a per-call interval of the same kernel differ in
    *dispatch treatment*, not in the physical protocol, so the scope, the
    measurement-scope prose and the repeat settings (warmup / iterations /
    trials) are excluded here; they are reported separately and each scope keeps
    its own launch intercept.  Missing dtype keys normalise to the declared
    defaults that the A2/A3 runners attest (bf16 / none / contiguous).
    """
    if not isinstance(protocol, Mapping) or not protocol:
        return {}
    return {
        "timer": str(protocol.get("timer") or ""),
        "preallocated": bool(protocol.get("preallocated", False)),
        # The old A3 store predates the explicit cache_policy field; every run in
        # this programme was measured under the declared policy below, so a
        # missing key normalises to it instead of splitting the protocol identity.
        "cache_policy": str(protocol.get("cache_policy") or DECLARED_CACHE_POLICY),
        "compute_dtype": _normalise_dtype(
            protocol.get("requested_compute_dtype") or protocol.get("compute_dtype") or "bf16"
        ),
        "weight_dtype": _normalise_dtype(protocol.get("weight_dtype") or "bf16"),
        "activation_dtype": _normalise_dtype(protocol.get("activation_dtype") or "bf16"),
        "accumulation_dtype": str(
            protocol.get("requested_accumulation_dtype")
            or protocol.get("accumulation_dtype")
            or "backend_default"
        ),
        "quantization": str(protocol.get("quantization") or "none"),
        "layout": str(protocol.get("layout") or "contiguous"),
    }


def protocol_core_signature(protocol: Mapping[str, Any]) -> str:
    """Stable digest of the scope-independent protocol identity."""
    core = protocol_core(protocol)
    if not core:
        return ""
    return "sha256:%s" % _stable_digest(core)[:16]


def protocol_signature(protocol: Mapping[str, Any]) -> str:
    """Stable digest of the timing-affecting protocol fields.

    Two conditions with different timing identity (timer, scope, warmup,
    iterations, trials, allocation, dtype, cache policy) must never share a fit.
    """
    if not isinstance(protocol, Mapping) or not protocol:
        return ""
    keys = (
        "timer", "timing_scope", "measurement_scope", "iterations", "trials", "warmup",
        "warmup_scope", "preallocated", "cache_policy", "requested_compute_dtype",
        "weight_dtype", "activation_dtype", "accumulation_dtype", "quantization", "layout",
    )
    selected = {key: protocol[key] for key in keys if key in protocol}
    return "sha256:%s" % _stable_digest(selected)[:16]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_git_head(repository: str | Path | None) -> str | None:
    """Resolve HEAD from ``.git`` files only (no subprocess, no network).

    Handles a linked worktree, where ``.git`` is a file containing
    ``gitdir: <path>`` rather than a directory.
    """
    if not repository:
        return None
    root = Path(repository)
    entry = root / ".git"
    try:
        if entry.is_file():
            text = entry.read_text().strip()
            if not text.startswith("gitdir:"):
                return None
            git_dir = Path(text[len("gitdir:"):].strip())
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()
        elif entry.is_dir():
            git_dir = entry
        else:
            return None
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head[len("ref:"):].strip()
        search = [git_dir]
        common = git_dir / "commondir"
        if common.is_file():
            shared = Path(common.read_text().strip())
            search.append(shared if shared.is_absolute() else (git_dir / shared).resolve())
        for directory in search:
            ref_file = directory / ref
            if ref_file.is_file():
                return ref_file.read_text().strip() or None
            packed = directory / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line and not line.startswith(("#", "^")):
                        sha, _, name = line.partition(" ")
                        if name.strip() == ref:
                            return sha.strip()
        return None
    except OSError:
        return None


def _load_observations(args: Any) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in args.observations or ():
        document = read_observations_json(path)
        rows.extend(document.get("observations") or ())
        sources.append("json:%s" % path)
    for path in getattr(args, "observations_attention", None) or ():
        document = read_attention_results(
            path,
            hardware_key=args.hardware_key or "",
            arch_family=args.arch_family or "",
        )
        rows.extend(document.get("observations") or ())
        sources.append("attention:%s" % path)
    for path in getattr(args, "observations_campaign", None) or ():
        document = read_campaign_results(
            path,
            hardware_key=args.hardware_key or "",
            arch_family=args.arch_family or "",
        )
        rows.extend(document.get("observations") or ())
        sources.append("campaign:%s" % path)
    for path in args.observations_sqlite or ():
        document = read_sqlite_observations(
            path,
            hardware_key=args.hardware_key or "",
            arch_family=args.arch_family or "",
        )
        rows.extend(document.get("observations") or ())
        sources.append("sqlite:%s" % path)
    return rows, sources


def _command_fit(args: Any) -> dict[str, Any]:
    rows, sources = _load_observations(args)
    if not rows:
        raise PhysicalCalibrationError("no_observations_loaded")
    machine = json.loads(Path(args.machine).read_text())
    guard = HeldoutGuard.from_files(args.heldout_guard) if args.heldout_guard else None
    calibration = fit_calibration(
        {"schema_version": OBSERVATION_SCHEMA, "observations": rows},
        machine,
        hardware_key=args.hardware_key or "",
        arch_family=args.arch_family or "",
        compute_dtype=args.compute_dtype,
        min_samples=args.min_samples,
        cv_folds=args.cv_folds,
        accepted_package_trees=args.accept_package_tree or (),
        heldout_guard=guard,
        git_head=args.git_head or _read_git_head(args.repo_root),
        source_artifacts=sources + list(args.source_artifact or ()),
    )
    payload = calibration.to_dict()
    if args.output:
        output = Path(args.output)
        if output.parent and not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {
        "status": "fitted",
        "output": str(args.output) if args.output else None,
        "hardware_key": calibration.hardware_key,
        "families": len(calibration.families),
        "provenance": calibration.provenance,
        "report": calibration.report(),
    }


def _command_predict(args: Any) -> dict[str, Any]:
    calibration = PhysicalCalibration.load(args.artifact)
    op = json.loads(Path(args.op).read_text())
    if "op" in op and isinstance(op["op"], Mapping):
        op = op["op"]
    machine = json.loads(Path(args.machine).read_text()) if args.machine else None
    prediction = calibration.predict_op_explained(op, machine, compute_dtype=args.compute_dtype)
    return prediction.to_dict()


def _command_report(args: Any) -> dict[str, Any]:
    calibration = PhysicalCalibration.load(args.artifact)
    return calibration.report()


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mindie_sim.a3_physical_calibration",
        description="Fit / query the physical (roofline) calibration artifact.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit an artifact from train observations")
    fit.add_argument("--observations", action="extend", nargs="+", default=[],
                     help="mindie-sim.observations/1 JSON document (repeatable)")
    fit.add_argument("--observations-sqlite", action="extend", nargs="+", default=[],
                     help="sample_statistics calibration store (repeatable fallback reader)")
    fit.add_argument("--observations-attention", action="extend", nargs="+", default=[],
                     help="attention-worker results JSON (device-event and block-interval scopes)")
    fit.add_argument("--observations-campaign", action="extend", nargs="+", default=[],
                     help="operator-campaign results.json (repeatable; batched and per-call scopes kept apart)")
    fit.add_argument("--machine", required=True, help="machine limits JSON")
    fit.add_argument("--output", default="", help="artifact output path")
    fit.add_argument("--accept-package-tree", action="extend", nargs="+", default=[],
                     help="only accept observations measured by these mindie_sim tree sha256 values")
    fit.add_argument("--heldout-guard", action="extend", nargs="+", default=[],
                     help="declared heldout split declaration / score / results JSON")
    fit.add_argument("--hardware-key", default="")
    fit.add_argument("--arch-family", default="")
    fit.add_argument("--compute-dtype", default="bf16")
    fit.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_FIT)
    fit.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    fit.add_argument("--source-artifact", action="extend", nargs="+", default=[])
    fit.add_argument("--git-head", default="")
    fit.add_argument("--repo-root", default=".")
    fit.set_defaults(handler=_command_fit)

    predict = sub.add_parser("predict", help="predict one operator duration")
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--op", required=True, help="operator mapping JSON")
    predict.add_argument("--machine", default="")
    predict.add_argument("--compute-dtype", default="bf16")
    predict.set_defaults(handler=_command_predict)

    report = sub.add_parser("report", help="print the artifact metrics and provenance")
    report.add_argument("--artifact", required=True)
    report.set_defaults(handler=_command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = args.handler(args)
    except PhysicalCalibrationError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0




_ATTENTION_OPERATORS = {
    "fusion_attention": "attention_fused",
    "fused_attention": "attention_fused",
    "npu_fusion_attention": "attention_fused",
    "attention_fused": "attention_fused",
    "attention_qk": "attention_qk",
    "attention_pv": "attention_pv",
}
ATTENTION_DEVICE_SCOPE = "device_event_warm_operator"
ATTENTION_BLOCK_SCOPE = "block_interval_warm_operator"


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )


def _attention_shape(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalise the attention worker's geometry to ``q/kv/heads/dim``."""
    shape = raw.get("shape") if isinstance(raw.get("shape"), Mapping) else raw
    if not isinstance(shape, Mapping):
        return None
    q = shape.get("q_tokens", shape.get("q", shape.get("q_len")))
    kv = shape.get("kv_tokens", shape.get("kv", shape.get("kv_len")))
    heads = shape.get("q_heads", shape.get("heads"))
    kv_heads = shape.get("kv_heads", heads)
    dim = shape.get("qk_dim", shape.get("dim", shape.get("head_dim")))
    if q is None or kv is None or heads is None or dim is None:
        return None
    out = {
        "q": float(q),
        "kv": float(kv),
        "heads": float(heads),
        "kv_heads": float(kv_heads),
        "dim": float(dim),
        "batch": float(shape.get("batch") or 1.0),
    }
    if shape.get("v_dim") is not None:
        out["v_dim"] = float(shape["v_dim"])
    return out


def read_attention_results(
    path: str | Path,
    *,
    hardware_key: str = "",
    arch_family: str = "",
    include_splits: Sequence[str] = ("train",),
) -> dict[str, Any]:
    """Read an attention-worker ``results.json`` (``a3_operator_worker_attention``).

    The worker reports three clocks per sample for one native call:
    ``cpu_submit_ms`` (host submit), ``device_interval_ms`` (device event) and
    ``wall_complete_ms`` (the whole call, *including* host submission and output
    allocation gaps).  Two of them are timing scopes, and they are never mixed:

    * ``device_event_warm_operator`` -- the device-event interval;
    * ``block_interval_warm_operator`` -- the completion-clock block interval,
      which carries the host/allocation gap and therefore gets its own launch
      intercept.

    An attention kernel is not a GEMM: the declared physical work is
    ``2 x q x kv x heads x dim`` per matmul, i.e. ``4 x q x kv x heads x dim`` for
    a fused QK+PV call, and the declared bytes are the Q read, the KV-cache read
    (K and V, on their own head count) and the O write, plus the measured
    workspace (two fp32 row-statistic buffers per query-head, 64 bytes).
    """
    value = json.loads(Path(path).read_text())
    results = value.get("cases") or value.get("results") or []
    document_protocol = value.get("protocol") if isinstance(value.get("protocol"), Mapping) else {}
    document_environment = value.get("environment") if isinstance(value.get("environment"), Mapping) else {}
    package_tree = str(
        (document_protocol.get("package_provenance") or {}).get("tree_sha256")
        or document_environment.get("mindie_sim_tree_sha256")
        or ""
    )
    observations: list[dict[str, Any]] = []
    skipped = {"split": 0, "family": 0, "shape": 0, "duration": 0, "failed_case": 0, "dtype": 0}
    for item in results:
        if not isinstance(item, Mapping):
            skipped["shape"] += 1
            continue
        if str(item.get("split") or "train") not in include_splits:
            skipped["split"] += 1
            continue
        if str(item.get("status") or "measured") != "measured":
            skipped["failed_case"] += 1
            continue
        operator = str(item.get("operator") or item.get("family") or "").lower()
        details = item.get("details") if isinstance(item.get("details"), Mapping) else {}
        if not operator:
            operator = str(details.get("kind") or "").lower()
        family = _ATTENTION_OPERATORS.get(operator)
        if family is None:
            skipped["family"] += 1
            continue
        geometry = item.get("geometry") if isinstance(item.get("geometry"), Mapping) else {}
        if not geometry and isinstance(details.get("geometry"), Mapping):
            geometry = details["geometry"]
        shape = _attention_shape(geometry)
        if shape is None:
            skipped["shape"] += 1
            continue
        mask = str(geometry.get("mask") or "")
        if not mask and isinstance(details.get("mask"), Mapping):
            mask = str(details["mask"].get("kind") or "")
        if not mask:
            declared = item.get("declared") if isinstance(item.get("declared"), Mapping) else {}
            mask = str(declared.get("mask") or "")
        contract = details.get("mask_contract")
        if not mask and isinstance(contract, Mapping):
            mask = str(contract.get("name") or "")
        shape["causal"] = "causal" in mask.lower()
        shape["mask"] = mask or "undeclared"
        descriptor = geometry.get("mask_descriptor")
        if isinstance(contract, Mapping) and contract.get("name"):
            shape["mask_contract"] = str(contract["name"])
        elif isinstance(descriptor, Mapping) and (
            descriptor.get("name") or descriptor.get("mask_form") or descriptor.get("kind")
        ):
            shape["mask_contract"] = str(
                descriptor.get("name")
                or descriptor.get("mask_form")
                or descriptor.get("kind")
            )
            if descriptor.get("materialized") is not None:
                shape["mask_materialized"] = bool(descriptor.get("materialized"))
        elif isinstance(geometry.get("mask_base_2d"), Sequence) and not isinstance(
            geometry.get("mask_base_2d"), (str, bytes)
        ):
            shape["mask_contract"] = "explicit_bool_mask"
        else:
            shape["mask_contract"] = "undeclared"
        shape["topology"] = attention_topology(shape["heads"], shape.get("kv_heads") or shape["heads"])
        shape["phase"] = attention_phase(shape["q"])
        dtype = _normalise_dtype(geometry.get("dtype") or details.get("dtype") or "bfloat16")
        itemsize = DTYPE_BYTES.get(dtype)
        if itemsize is None:
            skipped["dtype"] = skipped.get("dtype", 0) + 1
            continue
        q, kv, heads = shape["q"], shape["kv"], shape["heads"]
        kv_heads = shape.get("kv_heads") or heads
        dim, v_dim = shape["dim"], shape.get("v_dim") or shape["dim"]
        if family == "attention_qk":
            flops = 2.0 * q * kv * heads * dim
        elif family == "attention_pv":
            flops = 2.0 * q * kv * heads * v_dim
        else:
            flops = 2.0 * q * kv * heads * (dim + v_dim)
        hbm_bytes = (
            q * heads * dim
            + kv * kv_heads * (dim + v_dim)
            + q * heads * v_dim
        ) * itemsize + ATTENTION_WORKSPACE_BYTES_PER_QUERY_HEAD * q * heads
        trials = [trial for trial in (item.get("trials") or []) if isinstance(trial, Mapping)]
        device_trial_medians = []
        block_trial_medians = []
        samples = 0
        for trial in trials:
            trial_samples = [s for s in (trial.get("samples") or []) if isinstance(s, Mapping)]
            samples += len(trial_samples)
            device_median = _median(
                [s.get("device_interval_ms") for s in trial_samples
                 if s.get("device_interval_ms") is not None]
            )
            block_median = _median(
                [s.get("wall_complete_ms") for s in trial_samples
                 if s.get("wall_complete_ms") is not None]
            )
            if device_median is not None:
                device_trial_medians.append(device_median)
            if block_median is not None:
                block_trial_medians.append(block_median)
        case_key = str(item.get("case_id") or item.get("condition_key") or "")
        allocation = item.get("allocation") if isinstance(item.get("allocation"), Mapping) else {}
        preallocated = bool(allocation.get("preallocated"))
        if "preallocated" in details:
            preallocated = bool(details.get("preallocated"))
        declared_scope = str(item.get("timing_scope") or "")
        declared_duration = item.get("duration_ms")
        declared_samples = int(item.get("sample_count") or item.get("repeats_per_sample") or samples)
        base = {
            "hardware_key": hardware_key,
            "arch_family": arch_family,
            "die": "",
            "family": family,
            "recipe": "npu_fusion_attention",
            "run_id": str(item.get("run_id") or value.get("run_id") or ""),
            "split": str(item.get("split") or "train"),
            "shape": dict(shape),
            "elements": None,
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "dtype": dtype,
            "source_class": str(item.get("source_class") or "measured"),
            "package_tree": package_tree,
            "mask": shape["mask"],
        }
        emitted = 0
        scope_plans: list[tuple[str, str, float | None, int, str]] = []
        if declared_scope and declared_duration is not None and float(declared_duration) > 0:
            # The worker declared its own timing scope: use it verbatim, with the
            # duration it computed (a batched interval when it amortised calls).
            scope_plans.append(
                (declared_scope, str(document_protocol.get("timer") or "device_event"),
                 float(declared_duration), declared_samples, "declared_duration_ms")
            )
        else:
            scope_plans.append(
                (ATTENTION_DEVICE_SCOPE, "device_event", _median(device_trial_medians),
                 samples, "median_of_trial_medians")
            )
            scope_plans.append(
                (ATTENTION_BLOCK_SCOPE, "wall_complete", _median(block_trial_medians),
                 samples, "median_of_trial_medians")
            )
        per_call = item.get("per_call_median_ms")
        if per_call is not None and float(per_call) > 0 and "per_call" not in declared_scope:
            # A batched case also carries its per-call diagnostic; a case that is
            # already declared per-call must not be emitted twice.
            scope_plans.append(
                ("device_event_warm_per_call", "device_event", float(per_call),
                 len(item.get("per_call_diagnostic_ms") or ()) or 1, "declared_duration_ms")
            )
        for scope, clock, duration, events, estimator in scope_plans:
            if duration is None or duration <= 0:
                continue
            emitted += 1
            protocol = dict(document_protocol)
            protocol.update(
                {
                    "timer": clock,
                    "timing_scope": scope,
                    "preallocated": preallocated,
                    "requested_compute_dtype": dtype,
                    "worker_recipe": "npu_fusion_attention",
                    "measurement_scope": (
                        "one_native_call_device_event_interval"
                        if clock == "device_event"
                        else "one_native_call_completion_interval_including_host_and_allocation"
                    ),
                }
            )
            observations.append(
                dict(
                    base,
                    condition_key="%s|%s" % (case_key, clock),
                    protocol_signature=protocol_signature(protocol),
                    protocol_core_signature=protocol_core_signature(protocol),
                    duration_ms=float(duration),
                    raw_samples_ms=None,
                    raw_event_count=int(events),
                    timing_scope=scope,
                    target_estimator=estimator,
                    protocol=protocol,
                )
            )
        if not emitted:
            skipped["duration"] += 1
    if not hardware_key:
        hardware_key = str(document_environment.get("device_name") or "")
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "source": "attention_results:%s" % Path(path).name,
        "source_path": str(Path(path)),
        "hardware_key": hardware_key,
        "arch_family": arch_family,
        "protocol": dict(document_protocol),
        "package_tree": package_tree,
        "skipped": skipped,
        "observations": observations,
    }


def _campaign_family(recipe: str, shape: Mapping[str, Any], kind: str) -> str:
    """Map a campaign recipe/shape onto a modelled operator family."""
    token = ("%s %s %s" % (kind, recipe, shape)).lower()
    if {"m", "n", "k"} <= set(shape):
        return "gemm"
    grouped_axes = {"experts", "n", "k"} <= set(shape) and (
        "tokens" in shape
        or "routed_tokens" in shape
        or "tokens_per_expert" in shape
        or "expert_token_counts" in shape
    )
    if grouped_axes:
        # A grouped matmul is keyed by its routed-token axis, never by m/n/k: the
        # dense-GEMM branch above would otherwise claim it whenever the runner
        # reports the flattened shape.
        return "grouped_matmul"
    aliases = (
        ("rmsnorm", ("rms_norm", "rmsnorm")),
        ("swiglu", ("swiglu", "sigmoid_mul", "silu_mul")),
        # Checked before the generic softmax alias: a masked softmax is a distinct
        # kernel with its own mask traffic, so it keeps its own family fit.
        ("masked_softmax", ("masked_softmax", "scaled_masked_softmax")),
        ("softmax", ("softmax",)),
        ("quantize", ("quant",)),
        ("rope", ("rope", "rotary")),
        ("copy", ("copy",)),
        ("add", ("add", "residual")),
        ("cast", ("cast", "to_", "dtype_convert")),
    )
    for family, needles in aliases:
        if any(needle in token for needle in needles):
            return family
    return kind or recipe or "unknown"


def _campaign_elements(shape: Mapping[str, Any]) -> float | None:
    """Element count of a non-GEMM shape: ``elements`` or ``tokens x width``."""
    if shape.get("elements") is not None:
        return float(shape["elements"])
    if shape.get("tokens") is not None and shape.get("width") is not None:
        return float(shape["tokens"]) * float(shape["width"])
    if shape.get("rows") is not None and shape.get("cols") is not None:
        return float(shape["rows"]) * float(shape["cols"])
    if all(shape.get(key) is not None for key in ("heads", "q", "kv")):
        # Score-matrix work of a (masked) softmax: heads x q x kv elements.
        return float(shape["heads"]) * float(shape["q"]) * float(shape["kv"])
    return None


def read_campaign_results(
    path: str | Path,
    *,
    hardware_key: str = "",
    arch_family: str = "",
    include_splits: Sequence[str] = ("train",),
) -> dict[str, Any]:
    """Read an operator-campaign ``results.json`` (the A2/A3 runner schema).

    Emits the batched and the per-call interval of the same condition as two
    observations with distinct ``timing_scope`` values, because they are
    different physical protocols: a batched interval amortises dispatch over
    ``repeats_per_sample`` calls, a per-call interval includes it.  A2 runs
    report both (``duration_ms`` batched, ``per_call_median_ms`` per call) and
    carry ``trial_medians_ms``; A3 gap-fill runs report
    ``device_event_warm_operator_batched_amortized`` with two trials.

    Logical bytes follow the same declared convention as the other readers:
    a GEMM touches ``m*k + k*n + m*n`` elements, and each vector family reads /
    writes ``_VECTOR_BYTES_PER_ELEMENT`` tensors per element.
    """
    value = json.loads(Path(path).read_text())
    results = value.get("results") or value.get("cases") or []
    document_protocol = value.get("protocol") or {}
    document_environment = value.get("environment") if isinstance(value.get("environment"), Mapping) else {}
    package_tree = str(
        (document_protocol.get("package_provenance") or {}).get("tree_sha256")
        or document_environment.get("mindie_sim_tree_sha256")
        or ""
    )
    observations = []
    skipped = {"split": 0, "shape": 0, "duration": 0, "family": 0, "failed_case": 0, "dtype": 0}
    for item in results:
        if not isinstance(item, Mapping):
            skipped["shape"] += 1
            continue
        if str(item.get("split") or "train") not in include_splits:
            skipped["split"] += 1
            continue
        status = str(item.get("status") or "completed")
        if status not in ("completed", "measured"):
            # A failed case (NaN output, dispatch error) is not a measurement.
            skipped["failed_case"] = skipped.get("failed_case", 0) + 1
            continue
        details = item.get("details") if isinstance(item.get("details"), Mapping) else {}
        request = item.get("request") if isinstance(item.get("request"), Mapping) else {}
        shape_source = (
            details.get("shape")
            or request.get("shape")
            or item.get("geometry")
            or details.get("geometry")
            or {}
        )
        raw_shape = dict(shape_source) if isinstance(shape_source, Mapping) else {}
        protocol = dict(document_protocol)
        protocol.update(item.get("protocol") or {})
        recipe = str(
            details.get("kind")
            or item.get("recipe")
            or item.get("worker_recipe")
            or protocol.get("worker_recipe")
            or ""
        )
        kind = str(item.get("execution_kind") or protocol.get("execution_kind") or "primitive")
        declared_family = str(item.get("family") or details.get("group") or "")
        if declared_family in CAMPAIGN_FAMILY_ALIASES:
            declared_family = CAMPAIGN_FAMILY_ALIASES[declared_family]
        if declared_family in MODEL_KINDS:
            # A campaign that declares its operator family is authoritative: the
            # recipe string of one worker can cover several families.
            family = declared_family
        else:
            family = _campaign_family(recipe, raw_shape, kind)
            if family in CAMPAIGN_FAMILY_ALIASES:
                family = CAMPAIGN_FAMILY_ALIASES[family]
        if family not in MODEL_KINDS:
            skipped["family"] += 1
            continue
        # Propagate a die identity from the protocol block when the item has one.
        die = ""
        identity = protocol.get("die_identity") or item.get("die_identity")
        if isinstance(identity, Mapping):
            die = "phy%s" % identity.get("phy_id") if identity.get("phy_id") is not None else str(
                identity.get("logical") or ""
            )
        dtype = str(
            protocol.get("requested_compute_dtype")
            or details.get("dtype")
            or item.get("dtype")
            or "bf16"
        )
        itemsize = _reader_itemsize(dtype)
        if itemsize is None:
            skipped["dtype"] = skipped.get("dtype", 0) + 1
            continue
        if family == "gemm":
            shape = {
                key: float(raw_shape[key])
                for key in ("m", "n", "k")
                if raw_shape.get(key) is not None
            }
            m, n, k = shape.get("m"), shape.get("n"), shape.get("k")
            if not (m and n and k):
                skipped["shape"] += 1
                continue
            flops = 2.0 * m * n * k
            hbm_bytes = (m * k + k * n + m * n) * itemsize
            elements = None
        elif family in GROUPED_KINDS:
            experts = float(raw_shape.get("experts") or raw_shape.get("expert_count") or 0.0)
            group = (
                raw_shape.get("group")
                or raw_shape.get("tokens_per_expert")
                or (raw_shape.get("tokens") / experts if experts > 0 and raw_shape.get("tokens") else None)
            )
            tokens = float(
                raw_shape.get("routed_tokens")
                or raw_shape.get("tokens")
                or (experts * float(group) if group is not None else 0.0)
            )
            n = float(raw_shape.get("n") or 0.0)
            k = float(raw_shape.get("k") or 0.0)
            if not (experts > 0 and tokens > 0 and n > 0 and k > 0):
                skipped["shape"] += 1
                continue
            shape = {"tokens": tokens, "experts": experts, "n": n, "k": k}
            flops = 2.0 * tokens * n * k
            # Declared grouped byte convention: one weight read per expert, the
            # per-expert activation read, and the output write.
            hbm_bytes = (
                experts * n * k + tokens * k + tokens * n
            ) * itemsize
            elements = None
        else:
            elements = _campaign_elements(raw_shape)
            declared_read = details.get("actual_read_bytes")
            declared_write = details.get("actual_write_bytes")
            declared_bytes = None
            if declared_read is not None or declared_write is not None:
                declared_bytes = float(declared_read or 0.0) + float(declared_write or 0.0)
            if not elements and declared_bytes:
                # Byte-view references report bytes, not elements; express the
                # element count in the fitted dtype so the vector term stays
                # comparable across families.
                elements = declared_bytes / itemsize
            if not elements:
                skipped["shape"] += 1
                continue
            shape = {"elements": float(elements)}
            flops = 0.0
            hbm_bytes = (
                declared_bytes
                if declared_bytes
                else elements * itemsize * _VECTOR_BYTES_PER_ELEMENT.get(family, 2.0)
            )
        base = {
            "hardware_key": hardware_key,
            "arch_family": arch_family,
            "die": die,
            "family": family,
            "recipe": recipe,
            "run_id": str(item.get("run_id") or value.get("run_id") or ""),
            "split": str(item.get("split") or "train"),
            "shape": shape,
            "elements": elements,
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "dtype": dtype,
            "source_class": str(
                item.get("source_class")
                or protocol.get("source_class")
                # A case the runner itself marks "measured" is a device
                # measurement even when the document omits the source class.
                or ("measured" if status == "measured" else "unknown")
            ),
            "package_tree": package_tree,
        }
        condition_key = str(
            item.get("condition_key") or item.get("case_id") or item.get("measurement_id") or ""
        )
        campaign = item.get("campaign") if isinstance(item.get("campaign"), Mapping) else {}
        clock = campaign.get("clock_in_window") if isinstance(campaign.get("clock_in_window"), Mapping) else {}
        inferred_scope = ""
        batched_duration = item.get("duration_ms", item.get("batched_per_call_median_ms"))
        if not batched_duration:
            # The big-operator worker reports per-trial device-event medians and
            # one native call per sample instead of a pre-reduced duration.
            declared_medians = [
                float(value) for value in (item.get("trial_medians_ms") or ())
                if value is not None
            ]
            if declared_medians:
                batched_duration = _median(declared_medians)
                inferred_scope = "device_event_warm_operator"
            elif item.get("device_median_ms") is not None:
                batched_duration = float(item["device_median_ms"])
                inferred_scope = "device_event_warm_operator"
        declared_scope = str(
            item.get("timing_scope")
            or protocol.get("timing_mode")
            or protocol.get("timing_scope")
            or inferred_scope
            or "device_event_warm_batched"
        )
        if batched_duration and float(batched_duration) > 0:
            batched_protocol = dict(protocol, timing_scope=declared_scope)
            observations.append(
                dict(
                    base,
                    condition_key="%s|batched" % condition_key,
                    protocol_signature=protocol_signature(batched_protocol),
                    protocol_core_signature=protocol_core_signature(batched_protocol),
                    duration_ms=float(batched_duration),
                    trial_medians_ms=item.get("trial_medians_ms"),
                    raw_samples_ms=item.get("raw_samples_ms"),
                    raw_event_count=int(item.get("sample_count") or 0)
                    or int(item.get("trials_count") or 0)
                    or len(item.get("raw_samples_ms") or ())
                    or int(protocol.get("executed_trials") or protocol.get("trials") or 1),
                    timing_scope=declared_scope,
                    target_estimator="median_of_trial_medians",
                    protocol=batched_protocol,
                    observed_clock={"cur_freq_mhz": clock.get("cur_freq_mhz"),
                                    "rated_freq_mhz": clock.get("rated_freq_mhz"),
                                    "loaded_freq_floor_mhz": campaign.get("loaded_freq_floor_mhz")},
                )
            )
        else:
            skipped["duration"] += 1
        per_call = item.get("per_call_median_ms")
        if per_call and float(per_call) > 0:
            per_call_protocol = dict(protocol, timing_scope="device_event_warm_per_call")
            per_call_protocol.pop("timing_mode", None)
            diagnostics = [float(x) for x in (item.get("per_call_diagnostic_ms") or ())]
            observations.append(
                dict(
                    base,
                    condition_key="%s|per_call" % condition_key,
                    protocol_signature=protocol_signature(per_call_protocol),
                    protocol_core_signature=protocol_core_signature(per_call_protocol),
                    duration_ms=float(per_call),
                    raw_samples_ms=diagnostics,
                    raw_event_count=len(diagnostics) or 1,
                    timing_scope="device_event_warm_per_call",
                    target_estimator="declared_duration_ms",
                    protocol=per_call_protocol,
                    derived_from="per_call_diagnostic_ms",
                    observed_clock={"cur_freq_mhz": clock.get("cur_freq_mhz"),
                                    "rated_freq_mhz": clock.get("rated_freq_mhz"),
                                    "loaded_freq_floor_mhz": campaign.get("loaded_freq_floor_mhz")},
                )
            )
    if not hardware_key:
        hardware_key = str(document_environment.get("device_name") or "")
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "source": "campaign_results:%s" % Path(path).name,
        "source_path": str(Path(path)),
        "hardware_key": hardware_key,
        "arch_family": arch_family,
        "protocol": document_protocol,
        "package_tree": package_tree,
        "skipped": skipped,
        "observations": observations,
    }


def _mc2_leg_shape(
    leg: str, rows: float, local_experts: float, intermediate: float, hidden: float,
    itemsize: float,
) -> dict[str, Any] | None:
    """Grouped ``(tokens, experts, n, k)`` geometry of one MC2 compute leg.

    ``rows`` is the routed row count the timed rank actually fed to the grouped
    matmul -- the campaign records it per rank next to the interval, so the two
    are the same launch.  The weight geometry is the layer's, not the campaign's:
    gate/up produces ``2 x intermediate`` from ``hidden`` with ``split_item`` 3,
    down produces ``hidden`` from ``intermediate`` with ``split_item`` 2.
    """
    if leg == "gate_up":
        n, k = 2.0 * intermediate, hidden
    elif leg == "down":
        n, k = hidden, intermediate
    else:
        return None
    if not (rows > 0 and local_experts > 0 and n > 0 and k > 0):
        return None
    return {
        "shape": {"tokens": float(rows), "experts": float(local_experts),
                  "n": n, "k": k},
        "flops": 2.0 * rows * n * k,
        # Same declared grouped byte convention as ``read_campaign_results``:
        # one weight read per expert, the per-expert activation read, the write.
        "hbm_bytes": (local_experts * n * k + rows * k + rows * n) * itemsize,
    }


def read_mc2_grouped_results(
    path: str | Path,
    *,
    hardware_key: str = "",
    arch_family: str = "",
    machine_keys: Mapping[str, str] | None = None,
    campaigns: Sequence[str] = (),
    recipe: str = MC2_GROUPED_RECIPE,
    protocol: Mapping[str, Any] | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Read the MC2 campaign collation into grouped compute-leg observations.

    The document is the campaign's per-rank collation
    (``mc2-measurements.json``: ``runs`` plus one row per (plan, case, leg,
    rank)), not an operator-campaign ``results.json``: each leg is already a
    device-event median of 2 trials, so the reader reduces it to one observation
    per ``(plan, case, leg)`` with exactly the pair the campaign published.

    Declared extraction rules:

    * only ``kind == "gmm"`` rows of the declared ``campaigns`` are campaign
      conditions (a probe or diagnostic job is not a condition and is counted
      under ``skipped``);
    * the leg value is the **max over ranks of the per-rank median** -- the
      campaign's collective bound -- paired with the routed row count *of the
      rank that produced that median*, so duration and geometry are the same
      launch;
    * ``experts`` of the observation is the rank's **local** expert count
      (``global experts / ep``), because that is what the rank's list call
      launches; ``ep``, the global expert count and the rank count stay in the
      manifest;
    * the separately timed activation (``swiglu``) is extracted into ``legs``
      but is not ingested as a grouped observation;
    * a repeated ``(family, geometry)`` inside one machine keeps the first
      condition in sorted order and is counted as a duplicate instead of being
      fitted twice.

    ``machine_keys`` maps the document's machine label (``a3``/``a2``) to the
    hardware key of the artifact being built; rows of another machine are
    counted, not silently mixed.
    """
    value = json.loads(Path(path).read_text())
    if not isinstance(value, Mapping):
        raise PhysicalCalibrationError("mc2_document_must_be_a_mapping")
    if not machine_keys:
        raise PhysicalCalibrationError("mc2_machine_keys_required")
    allowed = {str(item) for item in (campaigns or ())}
    declared_protocol = dict(protocol or MC2_GROUPED_PROTOCOL)
    signature = protocol_signature({"timing_scope": MC2_GROUPED_SCOPE, **declared_protocol})
    core_signature = protocol_core_signature(declared_protocol)
    if not signature:
        raise PhysicalCalibrationError("mc2_protocol_signature_empty")
    runs: dict[tuple[str, str], Mapping[str, Any]] = {}
    for run in value.get("runs") or ():
        if isinstance(run, Mapping):
            runs[(str(run.get("machine") or ""), str(run.get("label") or ""))] = run
    cases: dict[tuple[str, str, str], dict[str, list[Mapping[str, Any]]]] = {}
    skipped = {
        "kind": 0, "timing_key": 0, "campaign": 0, "other_machine": 0,
        "plan_undeclared": 0, "rank_count": 0, "duration": 0, "rows": 0,
        "duplicate_geometry": 0,
    }
    for row in value.get("rows") or ():
        if not isinstance(row, Mapping):
            continue
        if str(row.get("kind") or "") != "gmm":
            skipped["kind"] += 1
            continue
        timing_key = str(row.get("timing_key") or "")
        leg = dict((item[0], item) for item in MC2_GROUPED_LEGS).get(timing_key)
        if leg is None and timing_key != MC2_ACTIVATION_LEG:
            # ``__case__`` bookkeeping rows and any other timing key are not legs.
            skipped["timing_key"] += 1
            continue
        machine = str(row.get("machine") or "")
        key = machine_keys.get(machine)
        if key is None:
            raise PhysicalCalibrationError("mc2_unknown_machine_label:%s" % (machine or "?",))
        if hardware_key and key != hardware_key:
            skipped["other_machine"] += 1
            continue
        if allowed and str(row.get("campaign") or "") not in allowed:
            skipped["campaign"] += 1
            continue
        plan = str(row.get("plan") or "")
        cases.setdefault((machine, plan, str(row.get("case_id") or "")), {}).setdefault(
            timing_key, []
        ).append(row)
    observations: list[dict[str, Any]] = []
    legs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for (machine, plan, case_id), by_leg in sorted(cases.items()):
        # Resolve the hardware key from the case's own machine label: the scan
        # above only decides which cases are in the corpus.
        machine_key = machine_keys[machine]
        any_row = next(iter(next(iter(by_leg.values()))))
        run = runs.get((machine, str(any_row.get("campaign") or "")))
        if run is None:
            skipped["plan_undeclared"] += 1
            continue
        ranks = int(run.get("ranks") or 0)
        if ranks <= 0:
            skipped["rank_count"] += 1
            continue
        experts = float(any_row.get("experts") or 0.0)
        hidden = float(any_row.get("hidden") or 0.0)
        intermediate = float(any_row.get("intermediate") or 0.0)
        local_experts = max(1.0, experts / float(ranks))
        for timing_key, family, leg_name in MC2_GROUPED_LEGS:
            rank_rows = [
                item for item in by_leg.get(timing_key) or ()
                if item.get("median_ms") is not None
            ]
            if not rank_rows:
                continue
            best = max(rank_rows, key=lambda item: (float(item["median_ms"]), int(item.get("rank") or 0)))
            duration = float(best["median_ms"])
            if duration <= 0:
                skipped["duration"] += 1
                continue
            rows = float(best.get("rows_per_rank") or 0.0)
            leg_itemsize = _reader_itemsize(best.get("dtype") or "bf16")
            if leg_itemsize is None:
                skipped["dtype"] = skipped.get("dtype", 0) + 1
                continue
            geometry = _mc2_leg_shape(
                leg_name, rows, local_experts, intermediate, hidden, leg_itemsize,
            )
            if geometry is None:
                skipped["rows"] += 1
                continue
            condition_key = "mc2:%s:%s:%s:%s" % (machine, plan, case_id, timing_key)
            leg_record = {
                "condition_key": condition_key,
                "machine": machine,
                "hardware_key": machine_key,
                "campaign": str(best.get("campaign") or ""),
                "plan": plan,
                "case_id": case_id,
                "leg": timing_key,
                "family": family,
                "ep": ranks,
                "global_experts": experts,
                "local_experts": local_experts,
                "routed_rows": rows,
                "tokens_per_expert": rows / local_experts if local_experts else 0.0,
                "duration_ms": duration,
                "per_rank_medians_ms": {
                    int(item.get("rank") or 0): float(item["median_ms"])
                    for item in rank_rows
                },
                "per_rank_rows": {
                    int(item.get("rank") or 0): item.get("rows_per_rank")
                    for item in rank_rows
                },
                "rank_of_max": int(best.get("rank") or 0),
                "samples": int(best.get("n_samples") or 0),
                "hidden": hidden,
                "intermediate": intermediate,
                "ingested": timing_key != MC2_ACTIVATION_LEG,
            }
            legs.append(leg_record)
            if timing_key == MC2_ACTIVATION_LEG:
                continue
            identity = (family, canonical_shape(geometry["shape"]))
            if identity in seen:
                skipped["duplicate_geometry"] += 1
                leg_record["ingested"] = False
                leg_record["duplicate_of"] = identity[1]
                continue
            seen.add(identity)
            observations.append(
                {
                    "condition_key": condition_key,
                    "hardware_key": machine_key,
                    "arch_family": arch_family,
                    "family": family,
                    "recipe": recipe,
                    "run_id": str(run.get("run_id") or ""),
                    "split": split,
                    "source_class": "measured",
                    "die": "",
                    "package_tree": "",
                    "shape": dict(geometry["shape"]),
                    "elements": None,
                    "flops": geometry["flops"],
                    "hbm_bytes": geometry["hbm_bytes"],
                    "duration_ms": duration,
                    "dtype": "bf16",
                    "timing_scope": MC2_GROUPED_SCOPE,
                    "raw_event_count": max(1, int(best.get("n_samples") or 0)),
                    "target_estimator": "max_over_ranks_per_rank_median",
                    "protocol_signature": signature,
                    "protocol_core_signature": core_signature,
                    "protocol": dict(declared_protocol, timing_scope=MC2_GROUPED_SCOPE),
                }
            )
        # The separately timed activation: recorded for the chain ledger, never
        # ingested (it is the ``swiglu`` vector family, not a grouped compute leg).
        activation_rows = [
            item for item in by_leg.get(MC2_ACTIVATION_LEG) or ()
            if item.get("median_ms") is not None
        ]
        if activation_rows:
            best = max(
                activation_rows,
                key=lambda item: (float(item["median_ms"]), int(item.get("rank") or 0)),
            )
            legs.append(
                {
                    "condition_key": "mc2:%s:%s:%s:%s" % (machine, plan, case_id, MC2_ACTIVATION_LEG),
                    "machine": machine,
                    "hardware_key": machine_key,
                    "campaign": str(best.get("campaign") or ""),
                    "plan": plan,
                    "case_id": case_id,
                    "leg": MC2_ACTIVATION_LEG,
                    "family": MC2_ACTIVATION_LEG,
                    "ep": ranks,
                    "global_experts": experts,
                    "local_experts": local_experts,
                    "routed_rows": float(best.get("rows_per_rank") or 0.0),
                    "duration_ms": float(best["median_ms"]),
                    "per_rank_medians_ms": {
                        int(item.get("rank") or 0): float(item["median_ms"])
                        for item in activation_rows
                    },
                    "per_rank_rows": {
                        int(item.get("rank") or 0): item.get("rows_per_rank")
                        for item in activation_rows
                    },
                    "rank_of_max": int(best.get("rank") or 0),
                    "samples": int(best.get("n_samples") or 0),
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "ingested": False,
                }
            )
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "source": "mc2_grouped_results:%s" % Path(path).name,
        "source_path": str(Path(path)),
        "hardware_key": hardware_key,
        "arch_family": arch_family,
        "campaigns": sorted(allowed),
        "recipe": recipe,
        "protocol": declared_protocol,
        "protocol_signature": signature,
        "protocol_core_signature": core_signature,
        "machines": dict(sorted((str(k), str(v)) for k, v in (value.get("machines") or {}).items())),
        "machine_keys": dict(sorted(dict(machine_keys).items())),
        "legs": legs,
        "skipped": skipped,
        "observations": observations,
    }


if __name__ == "__main__":
    raise SystemExit(main())
