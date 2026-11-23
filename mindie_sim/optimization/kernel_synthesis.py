"""Math x Hardware target-state kernel synthesis, Formula SDK, SRAM & wall analysis.

[Role] Own Layer-1 white-box FormulaKernelSpec / FusedPipelineRecipe evaluation
  and constructive target-state optimization moved out of ``export``:
  - Define ``FormulaTensorSpec``, ``FormulaKernelSpec``, ``FormulaKernelEvaluation``,
    ``FusedPipelineRecipe``, and ``evaluate_formula_kernel`` (alias
    ``simulate_formula_kernel``) pricing composed symbolic math equations
    through ``TileSolver`` (Ascend ``FP32`` 4 B/elem UB upcast + SIMT carveout)
    and canonical ``runtime.Engine`` L0 pipes (``MTE2``, ``Cube``, ``AIV FP32``,
    ``MTE3``, ``HBM``, ``Link``, single launch floor, wave fill/drain) with
    zero heuristic efficiency multipliers;
  - Discover provable algebraic fusions via ``rewrite.algebra`` (zero
    suffix tables or model branches), accept and mechanically verify optional
    Agent-supplied ``RewritePlan`` edits through ``rewrite.verify.verify_plan``;
  - Enforce strict mode separation (``allow_unbound_formula``), hard zero-saving
    status invariants, exact cross-stream overlap attribution, and macro
    ``HBM_CAPACITY_WALL`` / ``LATENCY_WALL`` space-time triggers.
[Invariants] Strictly ``optimization`` layer (zero imports from ``export``).
  Never degrades any operator duration (``T_after <= T_before``), never emits
  ``projected_us < l_star_us``, never uses heuristic efficiency divisions,
  and never emits unverified cross-collective or cross-reduction rewrites.
[Timeouts & Locks] Pure O(Archetype Ops) in-memory analysis; no I/O or locks.
[Side-Effect Policy] Functional over immutable inputs; optional JSON file read
  only when caller passes an explicit ``Path``/``str`` for ``rewrite_plan``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from math import prod
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Literal

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.capacity.ledger import (
    recover_ssm_snapshot_ring_breakdown,
)
from mindie_sim.prediction.collectives import (
    OneShotCollective,
    OneShotPlan,
    Topology,
)
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution.operator_replay import operator_costs
from mindie_sim.prediction.fusion.pool import register_recipe
from mindie_sim.prediction.fusion.ports import input_port_entries
from mindie_sim.prediction.offload.policy import _cached_p20_metrics
from mindie_sim.prediction.physical_recipe_base import (
    IO_HBM_CATEGORY,
    Phase,
    _resource_categories,
    io_weight,
    phased_program,
    tiled_phased_program,
)
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.program.binding import RecipeContext
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.results import OpOccurrence, StepRequest
from mindie_sim.prediction.runtime import tile_solver
from mindie_sim.prediction.runtime.exposed_observer import (
    ObservedInterval,
    attribute_stream0_intervals,
)
from mindie_sim.prediction.runtime.tile_solver import (
    LiveBufferSpec,
    TileCandidate,
    TilePipelineSpec,
    TileShape,
)
from mindie_sim.primitives.loader import resolve_dispatch_envelope, resolve_sim_rate_overlay
from mindie_sim.primitives.tilelang import try_load_tilelang_calibration
from mindie_sim.rewrite.algebra import (
    FP32_ACCUM_BYTES,
    MATMUL_SHAPE_RANK,
    MIN_FUSE_MEMBERS,
    UB_DOUBLE_BUFFER_SLOTS,
    AlgebraicCandidate,
    AlgebraicTrait,
    OpTraitInfo,
    SemanticAuditTrigger,
    classify_op_traits,
    discover_algebraic_rewrites_for_ops,
)
from mindie_sim.rewrite._oracle_prefetch import primitive_table_digest_token
from mindie_sim.rewrite.schema import (
    CollectiveTransform,
    Edit,
    FuseRegion,
    HoistOrDeduplicate,
    KernelBoundary,
    OpRef,
    PortRef,
    QuantizeContract,
    RewritePlan,
    SelectTemplate,
    _digest_default,
    _json_digest,
    decode_rewrite_plan,
    empty_plan,
    plan_sha256,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.operators import OpSpec, TensorPrecisionABI
from mindie_sim.schema.scenario import (
    CoDesignOpportunity,
    CrossHwOffloadSpec,
    MegaFusionBound,
    OptimizationMode,
    SpaceTimeTradeoffSpec,
    canonical_comparison_stages,
)
from mindie_sim.prediction.results import RankOps
from .causal_tail import (
    FUSED_MC2_ROUTED_LEVEL,
    FUSED_MC2_SHARED_LEVEL,
    SHIPPING_MOE_ACTIVATIONS,
    DieScheduleResult,
    MoERouteDecision,
    build_discrete_mso_candidate_nodes,
    enforce_moe_mutex_on_nodes,
    evaluate_disjoint_core_budgets,
    is_a5_hardware,
    is_persistent_whole_die_node,
    is_shared_expert_op_id,
    moe_route_legality,
    resolve_arch_family_tag,
    schedule_target_graph_on_engine,
    weighted_step_span_us,
)

SIMT_UB_CARVEOUT_BYTES = 16384

BindingState = Literal["bound_measured", "bound_implementation", "unbound_formula_projection"]

#: Small/medium superpod collective payload threshold (8 MiB).
UB_ONESHOT_SMALL_MSG_BYTES = 8 * 1024 * 1024
#: Minimum operator saving (us) and improvement ratio (3%) to stamp a tuning badge.
TARGET_OPT_MIN_GAIN_US = 0.25
TARGET_OPT_MIN_RATIO = 0.03
#: Strict zero-saving epsilon (us) for the Hard Status Invariant.
SAVING_POSITIVE_EPS_US = 1e-6
#: Overlap exposed duration threshold (us).
OVERLAP_EXPOSED_EPS_US = 1e-3
#: Causal timestamp epsilon (us) for detecting original same-stream overlap.
ORIG_OVERLAP_EPS_US = 1e-3
#: Maximum character length for compact ``masked_by`` labels.
MAX_MASKED_BY_LEN = 48
#: Minimum secondary overlap share (25%) to show dual maskers.
SECONDARY_MASKER_SHARE = 0.25
#: Time conversion factors.
MS_PER_SECOND = 1000.0
US_PER_MS = 1000.0
NS_PER_US = 1000.0
NS_PER_MS = 1_000_000.0
NS_PER_SECOND = 1_000_000_000.0
US_PER_SECOND = 1_000_000.0
#: Binary KiB / GiB divisors.
KIB_BYTES = 1024.0
GIB_BYTES = 1024 * 1024 * 1024
#: Minimum HBM utilization (%) to trigger HBM_CAPACITY_WALL.
HBM_WALL_UTIL_PCT = 88.0
#: Minimum free HBM headroom (GiB) to trigger LATENCY_WALL space-for-time trade-off.
LATENCY_WALL_FREE_GIB = 8.0
#: Minimum unused SLA slack (ms) indicating latency is not the active wall.
SLA_SLACK_WALL_MS = 3.0
#: Minimum relative gap from SLA ceiling to flag LATENCY_WALL.
SLA_NEAR_BOUND_RATIO = 0.92
#: Default tile K/MN block widths for Cube L1/L0 residency estimation.
CUBE_TILE_K_ELEMS = 64
CUBE_TILE_MN_ELEMS = 128
#: Default tile count ceiling for white-box ``FusedPipelineRecipe`` lowering.
FORMULA_TILE_CAP = 64
#: Minimum tile byte granule for white-box pipeline wave sizing.
FORMULA_MIN_TILE_BYTES = 4096
#: HBM read/write asymmetry threshold for selecting calibrated DRAM modes.
HBM_RW_ASYMMETRY_RATIO = 1.5
#: Number of FP32 bytes per element in L0C accumulator.
L0C_ACCUM_BYTES = 4
#: MXFP8 E8M0 block quant group width.
MXFP_BLOCK_GROUP = 32
#: Symmetric half split ratio.
HALF_SPLIT_RATIO = 0.5
#: ABI signature hash prefix length.
ABI_DIGEST_LEN = 12
#: Decimal GB/s conversion for Link.capacity_gbps.
BYTES_PER_GB = 1_000_000_000.0
#: Percent multiplier.
PERCENT_SCALE = 100.0
#: Default AIV vector ops per element when declarative formula omits explicit vector_ops_fp32.
DEFAULT_FORMULA_VEC_OPS_PER_ELEM = 4.0

#: Canonical status strings (kept aligned with ``export.models``).
STATUS_AT_FLOOR = "现成可用 / 已近物理极限 (Existing / At Floor)"
STATUS_NEW_OP = "[需新开发算子]"
STATUS_EXTEND_ABI = "[需扩展算子 ABI]"
STATUS_TUNE_PREFIX = "[需算子深度调优"
BACKEND_CANN_LABEL = "CANN aclnn/AscendC"
BACKEND_TILELANG_LABEL = "TileLang"
BACKEND_ASCENDC_LABEL = "AscendC"
BACKEND_CATLASS_LABEL = "Catlass"
BACKEND_TRITON_LABEL = "Triton-Ascend"
BACKEND_HCCL_LABEL = "HCCL / MC2"
FLOAT_PREC_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
QUANT_PREC_DTYPES = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "int8", "mxfp4", "fp4", "int4"})
TILER_EXT_FAMILIES = frozenset({
    "gemm", "matmul", "grouped_gemm", "grouped_matmul",
    "linear", "projection", "gate_up_projection", "down_projection",
    "dense_gate_up", "dense_down",
})
BOUND_NON_COLLAPSIBLE_FAMILIES = frozenset({
    "attn_res_prefix", "mhc", "hyper_connections", "short_convolution",
    "causal_conv1d_fn", "conv1d", "recurrent_scan", "recurrent_vector",
    "gdn_gating", "recover_ssm_commit", "moe_gating", "engram_rh2d", "ple",
})
P20_FALLBACK_H2D_BPS = 32_000_000_000.0
P20_FALLBACK_STARTUP_NS = 5_000.0
DEFAULT_BF16_BYTES = 2
L_STAR_INVERSION_TOL_US = 1e-2

_COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "allgather", "reducescatter", "reduce_scatter", "alltoall",
    "alltoallv", "mc2", "mc2_transport", "mc2_protocol", "dispatch", "combine",
    "collective", "hccl", "p2p", "tp_allreduce", "engram_ids", "engram_broadcast",
})
_COLLECTIVE_FAMILY_ALIASES = {
    "all_reduce": "allreduce",
    "tp_allreduce": "allreduce",
    "all_gather": "allgather",
    "reducescatter": "reduce_scatter",
}
_SUPPORTED_TARGET_OPCODES = (
    FuseRegion,
    SelectTemplate,
    CollectiveTransform,
    QuantizeContract,
    HoistOrDeduplicate,
)
_NON_MATRIX_PROGRAMS = frozenset({
    "TiledPointwise",
    "RingCollective",
    "PairwiseCollective",
    "OneShotCollective",
})
_EQUATION_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_INTRA_LINK_SCOPES = frozenset({"", "intra_package", "intra_node", "superpod"})
_BOUND_LABELS = {
    "memory_bound": "HBM Weight/KV BW Bound",
    "cube_bound": "Cube AIC Compute Bound",
    "vector_bound": "Vector AIV Compute Bound",
    "link_bound": "Exposed HCCL Link Bound",
    "launch_bound": "Kernel Launch / Latency Bound",
}


def _safe_elem_bytes(dtype: str) -> float:
    tok = str(dtype or "bf16").strip().lower()
    try:
        return float(dtype_bytes(tok))
    except ModelingError:
        return 2.0


def _strip_op_prefix(op_id: str) -> str:
    parts = str(op_id or "").split(".")
    for idx, part in enumerate(parts):
        if part.startswith("l") and part[1:].isdigit():
            return ".".join(parts[idx + 1 :])
    return parts[-1] if parts else str(op_id or "")


@dataclass(frozen=True, slots=True)
class FormulaTensorSpec:
    """Typed tensor operand declaration for Layer-1 ``FormulaKernelSpec``."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "bf16"
    role: Literal["input", "output", "weight", "workspace"] = "input"
    hbm_resident: bool = True

    @property
    def num_elements(self) -> int:
        dims = tuple(int(d) for d in self.shape if int(d) > 0)
        return prod(dims) if dims else 0

    @property
    def byte_size(self) -> float:
        return float(self.num_elements) * _safe_elem_bytes(self.dtype)


@dataclass(frozen=True, slots=True)
class PipeStageSpec:
    """One causal compute/communication pipe stage derived from the true operand DAG."""

    name: str
    pipe_kind: Literal["cube", "vector", "link"]
    share: float = 1.0
    after: tuple[str, ...] = ()
    row_reduction: bool = False


@dataclass(frozen=True, slots=True)
class FormulaKernelSpec:
    """Layer-1 declarative mathematical formula specification for unbound kernels."""

    name: str
    equations: tuple[str, ...]
    inputs: tuple[FormulaTensorSpec, ...] = ()
    outputs: tuple[FormulaTensorSpec, ...] = ()
    weights: tuple[FormulaTensorSpec, ...] = ()
    matrix_flops: tuple[tuple[str, float], ...] = ()
    vector_ops_fp32: float = 0.0
    link_bytes: float = 0.0
    row_reduction: bool = False
    uses_simt: bool = False
    atomic_ops: tuple[OpSpec, ...] = ()
    abi_signature_id: str = ""
    family: str = "formula_kernel"
    activation: str = ""
    activation_params: tuple[tuple[str, float], ...] = ()
    hbm_bytes_override: float | None = None
    mte2_bytes_override: float | None = None
    mte3_bytes_override: float | None = None
    pipe_stages: tuple[PipeStageSpec, ...] = ()

    def resolved_abi_signature(self) -> str:
        if self.abi_signature_id:
            return self.abi_signature_id
        payload = json.dumps(
            {
                "name": self.name,
                "equations": list(self.equations),
                "inputs": [(t.name, list(t.shape), t.dtype) for t in self.inputs],
                "outputs": [(t.name, list(t.shape), t.dtype) for t in self.outputs],
                "weights": [(t.name, list(t.shape), t.dtype) for t in self.weights],
                "matrix_flops": [[dt, float(fl)] for dt, fl in self.matrix_flops],
                "row_reduction": bool(self.row_reduction),
                "activation": self.activation,
                "activation_params": [[k, float(v)] for k, v in self.activation_params],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:ABI_DIGEST_LEN]
        return f"formula:{self.name}:{digest}"

    @classmethod
    def from_ops(
        cls,
        ops: Sequence[OpSpec],
        *,
        name: str = "",
        equations: Sequence[str] = (),
        saved_read_bytes: float = 0.0,
        saved_write_bytes: float = 0.0,
        preserve_port_ids: Sequence[str] = (),
    ) -> FormulaKernelSpec:
        """Construct a ``FormulaKernelSpec`` directly from absorbed atomic ``OpSpec``s."""
        ordered = tuple(ops)
        if not ordered:
            raise ModelingError("formula_kernel_spec_empty_ops")
        traits = tuple(classify_op_traits(op) for op in ordered)
        kernel_name = name or ordered[-1].recipe or ordered[-1].op_id
        matrix: dict[str, float] = defaultdict(float)
        for op in ordered:
            entries = op.matrix_flops or (((op.dtype, op.flops),) if op.flops else ())
            for dt, fl in entries:
                if fl > 0.0:
                    matrix[str(dt)] += float(fl)
        raw_mte2 = sum(float(op.mte2_bytes or 0.0) for op in ordered)
        raw_mte3 = sum(float(op.mte3_bytes or 0.0) for op in ordered)
        raw_hbm = sum(float(op.hbm_bytes or 0.0) for op in ordered)
        mte2 = max(0.0, raw_mte2 - max(0.0, float(saved_read_bytes)))
        mte3 = max(0.0, raw_mte3 - max(0.0, float(saved_write_bytes)))
        hbm = max(0.0, raw_hbm - max(0.0, float(saved_read_bytes)) - max(0.0, float(saved_write_bytes)))
        vec_ops = sum(float(op.vector_ops or 0.0) for op in ordered)
        link_b = sum(float(op.link_bytes or 0.0) for op in ordered)
        row_red = any(tr.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} for tr in traits)
        acts = [tr.activation_kind for tr in traits if tr.activation_kind]
        act_params = tuple(dict(p for tr in traits for p in tr.activation_params).items())
        dag_eqs, in_specs, out_specs, _elided = _compose_dag_equations_and_ports(
            ordered, traits, preserve_port_ids=preserve_port_ids,
        )
        comp_eqs = tuple(equations) if equations else dag_eqs
        pipe_stages = _derive_pipe_stages_from_ops(ordered, traits)
        return cls(
            name=kernel_name,
            equations=comp_eqs,
            inputs=in_specs,
            outputs=out_specs,
            matrix_flops=tuple(sorted(matrix.items())),
            vector_ops_fp32=vec_ops,
            link_bytes=link_b,
            row_reduction=row_red,
            atomic_ops=ordered,
            family="rewrite_fused",
            activation=acts[0] if acts else "",
            activation_params=act_params,
            hbm_bytes_override=hbm,
            mte2_bytes_override=mte2,
            mte3_bytes_override=mte3,
            pipe_stages=pipe_stages,
        )


@dataclass(frozen=True, slots=True)
class FormulaKernelEvaluation:
    """White-box Engine + TileSolver evaluation receipt for a ``FormulaKernelSpec``."""

    name: str
    abi_signature_id: str
    composed_equations: tuple[str, ...]
    projected_us: float
    l_star_us: float
    tile_spec: TilePipelineSpec | None
    ub_peak_bytes: int
    ub_budget_bytes: int
    l1_peak_bytes: int
    l0_peak_bytes: int
    sram_verified: bool
    binding_state: BindingState = "unbound_formula_projection"
    has_unbound_formula_ops: bool = True
    dominant_pipe: str = ""
    hbm_bytes: float = 0.0
    mte2_bytes: float = 0.0
    mte3_bytes: float = 0.0
    launch_us: float = 0.0
    tile_algorithm: str = "resident_1pass"
    tile_shape_str: str = ""
    l1_budget_bytes: int = 0
    l0_budget_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "abi_signature_id": self.abi_signature_id,
            "composed_equations": list(self.composed_equations),
            "projected_us": round(float(self.projected_us), 4),
            "l_star_us": round(float(self.l_star_us), 4),
            "ub_peak_bytes": int(self.ub_peak_bytes),
            "ub_budget_bytes": int(self.ub_budget_bytes),
            "l1_peak_bytes": int(self.l1_peak_bytes),
            "l0_peak_bytes": int(self.l0_peak_bytes),
            "sram_verified": bool(self.sram_verified),
            "binding_state": self.binding_state,
            "has_unbound_formula_ops": bool(self.has_unbound_formula_ops),
            "dominant_pipe": self.dominant_pipe,
            "hbm_bytes": float(self.hbm_bytes),
            "mte2_bytes": float(self.mte2_bytes),
            "mte3_bytes": float(self.mte3_bytes),
            "launch_us": round(float(self.launch_us), 4),
        }


@dataclass(frozen=True)
class FusedPipelineRecipe:
    """White-box L0-pipe recipe executing composed kernels directly on ``Engine``.

    Prices ``MTE2``, ``Cube`` (FP32 L0C accum), ``AIV FP32``, ``MTE3``, ``HBM``,
    and optional ``Link`` work with wave fill/drain and a single kernel launch
    floor from ``RecipeContext`` (``datapath`` + ``PrimitiveTable``), conserving
    100% of executed service with zero heuristic multipliers.
    """

    slots: int = UB_DOUBLE_BUFFER_SLOTS
    tile_cap: int = FORMULA_TILE_CAP
    row_reduction: bool = False
    tile_count: int = 0
    abi_signature_id: str = ""
    binding_state: BindingState = "unbound_formula_projection"
    traffic_rewrite_capable: bool = True
    pipe_stages: tuple[PipeStageSpec, ...] = ()

    def bind_context(self, context: RecipeContext) -> RecipeContext:
        """Bind calibrated L0 HBM R:W asymmetry rates when supplied by ``PrimitiveTable``."""
        op = context.op
        hbm_bytes = float(getattr(op, "hbm_bytes", 0.0) or 0.0)
        read_bytes = float(getattr(op, "mte2_bytes", 0.0) or 0.0)
        write_bytes = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
        if hbm_bytes <= 0.0 or "hbm" not in context.resource_names:
            return context
        hbm_read = max(0.0, hbm_bytes - write_bytes)
        hbm_write = min(hbm_bytes, write_bytes)
        if hbm_read <= 0.0 or hbm_write > HBM_RW_ASYMMETRY_RATIO * hbm_read:
            dram_rate = context.service_rate("hbm_write_dram") or context.service_rate("mte2_aiv_dram")
        elif hbm_write <= 0.0:
            dram_rate = context.service_rate("mte2_aiv_dram")
        elif read_bytes > HBM_RW_ASYMMETRY_RATIO * write_bytes:
            dram_rate = context.service_rate("hbm_rw21_dram") or context.service_rate("mte2_aiv_dram")
        else:
            dram_rate = (
                context.service_rate("hbm_rw11_dram")
                or context.service_rate("hbm_rw21_dram")
                or context.service_rate("mte2_aiv_dram")
            )
        if dram_rate is None or dram_rate <= 0.0:
            return context
        hbm_peak = float(context.rates[context.resource_names["hbm"]]) * MS_PER_SECOND
        updated_modes = dict(context.service_modes)
        updated_modes["hbm"] = ("hbm", min(dram_rate, hbm_peak))
        return replace(context, service_modes=updated_modes)

    def _lower_dag_pipe_stages(
        self, categories: set[str], has_load: bool,
    ) -> tuple[list[Phase], tuple[str, ...]]:
        """Emit compute/communication ``Phase`` DAG preserving true operand-DAG order."""
        active_kinds: set[str] = set()
        if categories & {"cube", "mte1"}:
            active_kinds.add("cube")
        if "vector" in categories:
            active_kinds.add("vector")
        if "link" in categories:
            active_kinds.add("link")
        if not active_kinds:
            return [], ("mte2_load",) if has_load else ()

        raw_stages: list[PipeStageSpec] = list(self.pipe_stages)
        if not raw_stages:
            prev_name: tuple[str, ...] = ()
            if "cube" in active_kinds:
                raw_stages.append(PipeStageSpec("cube_mma", "cube", 1.0, prev_name))
                prev_name = ("cube_mma",)
            if "vector" in active_kinds:
                raw_stages.append(PipeStageSpec("aiv_fp32", "vector", 1.0, prev_name, self.row_reduction))
                prev_name = ("aiv_fp32",)
            if "link" in active_kinds:
                raw_stages.append(PipeStageSpec("link_pipe", "link", 1.0, prev_name))

        # Project declared pipe DAG onto active resource categories while keeping transitive edges.
        active_names = {st.name for st in raw_stages if st.pipe_kind in active_kinds}
        eff_preds: dict[str, tuple[str, ...]] = {}
        for st in raw_stages:
            preds: list[str] = []
            for p_name in st.after:
                if p_name in active_names:
                    preds.append(p_name)
                elif p_name in eff_preds:
                    preds.extend(eff_preds[p_name])
            eff_preds[st.name] = tuple(dict.fromkeys(preds))

        active_stages = [st for st in raw_stages if st.name in active_names]
        covered_kinds = {st.pipe_kind for st in active_stages}
        tail_names = tuple(
            st.name for st in active_stages
            if not any(st.name in eff_preds[other.name] for other in active_stages)
        )
        for kind, default_name in (("cube", "cube_mma"), ("vector", "aiv_fp32"), ("link", "link_pipe")):
            if kind in active_kinds and kind not in covered_kinds:
                st_extra = PipeStageSpec(
                    default_name, kind, 1.0, tail_names,
                    row_reduction=(self.row_reduction if kind == "vector" else False),
                )
                active_stages.append(st_extra)
                eff_preds[st_extra.name] = tail_names
                tail_names = (st_extra.name,)

        share_totals: dict[str, float] = defaultdict(float)
        for st in active_stages:
            share_totals[st.pipe_kind] += max(0.0, float(st.share))

        phases: list[Phase] = []
        term_name: dict[str, str] = {}
        used_as_pred: set[str] = set()
        for st in active_stages:
            total = share_totals[st.pipe_kind]
            norm_share = (max(0.0, float(st.share)) / total) if total > 0.0 else (1.0 / len(active_stages))
            preds = tuple(dict.fromkeys(term_name[p] for p in eff_preds.get(st.name, ()) if p in term_name))
            used_as_pred.update(p for p in eff_preds.get(st.name, ()) if p in term_name)
            if not preds and has_load:
                preds = ("mte2_load",)
            if st.pipe_kind == "cube":
                cube_w = {role: norm_share for role in ("cube", "mte1") if role in categories}
                phases.append(Phase(st.name, cube_w, after=preds))
                term_name[st.name] = st.name
            elif st.pipe_kind == "vector":
                if st.row_reduction:
                    r_name = "aiv_reduce" if st.name == "aiv_fp32" else f"{st.name}_reduce"
                    e_name = "aiv_epilogue" if st.name == "aiv_fp32" else f"{st.name}_epilogue"
                    phases.append(Phase(r_name, {"vector": norm_share * HALF_SPLIT_RATIO}, after=preds))
                    phases.append(Phase(e_name, {"vector": norm_share * HALF_SPLIT_RATIO}, after=(r_name,)))
                    term_name[st.name] = e_name
                else:
                    phases.append(Phase(st.name, {"vector": norm_share}, after=preds))
                    term_name[st.name] = st.name
            elif st.pipe_kind == "link":
                phases.append(Phase(st.name, {"link": norm_share}, after=preds))
                term_name[st.name] = st.name
        sinks = tuple(dict.fromkeys(term_name[st.name] for st in active_stages if st.name not in used_as_pred))
        return phases, sinks or (("mte2_load",) if has_load else ())

    def lower(self, context: RecipeContext) -> KernelProgram:
        """Lower declared physical work into a finite multi-pipe tile pipeline."""
        categories = set(_resource_categories(context).values())
        if not categories:
            return KernelProgram((Stage("empty", keep_empty=True),), minimum_ms=context.minimum_ms)
        read_share = io_weight(context, "hbm_read")
        write_share = io_weight(context, "hbm_write")
        if read_share + write_share <= 0.0:
            read_share = write_share = HALF_SPLIT_RATIO
        phases: list[Phase] = []
        has_load = bool(categories & {"hbm", "mte2"}) and (read_share > 0.0 or "mte2" in categories)
        has_store = bool(categories & {"hbm", "mte3", "fixpipe"}) and (
            write_share > 0.0 or bool(categories & {"mte3", "fixpipe"})
        )
        if has_load:
            load_w: dict[str, float] = {}
            if "hbm" in categories and read_share > 0.0:
                load_w[IO_HBM_CATEGORY] = read_share if has_store else 1.0
            if "mte2" in categories:
                load_w["mte2"] = 1.0
            phases.append(Phase("mte2_load", load_w))
        pipe_phases, store_preds = self._lower_dag_pipe_stages(categories, has_load)
        phases.extend(pipe_phases)
        if has_store:
            store_w: dict[str, float] = {}
            if "hbm" in categories:
                hbm_w = write_share if has_load else 1.0
                if hbm_w > 0.0:
                    store_w[IO_HBM_CATEGORY] = hbm_w
            if "mte3" in categories:
                store_w["mte3"] = 1.0
            if "fixpipe" in categories:
                store_w["fixpipe"] = 1.0
            if store_w:
                phases.append(Phase("mte3_store", store_w, after=store_preds))
        if not phases:
            return phased_program(context, ())
        cap = self.tile_count if self.tile_count > 0 else self.tile_cap
        return tiled_phased_program(context, tuple(phases), slots=max(1, int(self.slots)), cap=cap)


@dataclass(frozen=True, slots=True)
class SkuTargetFloors:
    """Calibrated L0 launch and interconnect floors extracted from active SKU."""

    launch_vec_us: float
    launch_cube_us: float
    launch_norm_us: float
    oneshot_floor_us: float
    mc2_sync_us: float
    is_intra_pod: bool
    is_a5_ub: bool
    interconnect_label: str


@dataclass(frozen=True, slots=True)
class KernelSynthesisSpec:
    """Composed symbolic math equations + Ascend FP32 UB/L1/L0 SRAM spec."""

    spec_id: str
    kernel_name: str
    abi_tier: str
    suggested_backend: str
    status_tag: str
    opt_pillar: str
    constituent_op_ids: tuple[str, ...]
    constituent_traits: tuple[str, ...]
    symbolic_equations: tuple[str, ...]
    input_tensors: tuple[str, ...]
    output_tensors: tuple[str, ...]
    elided_intermediates: tuple[str, ...]
    elided_hbm_bytes: int
    tile_algorithm: str
    tile_shape: str
    ub_precision_mode: str
    ub_peak_bytes: int
    ub_budget_bytes: int
    l1_peak_bytes: int
    l1_budget_bytes: int
    l0_peak_bytes: int
    l0_budget_bytes: int
    sram_verified: bool
    l_star_us: float
    before_us: float
    target_us: float
    target_attainment_pct: float
    abi_signature_id: str = ""
    binding_state: BindingState = "unbound_formula_projection"
    exposed_before_us: float | None = None
    ub_wall_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "spec_id": self.spec_id,
            "kernel_name": self.kernel_name,
            "abi_tier": self.abi_tier,
            "suggested_backend": self.suggested_backend,
            "status_tag": self.status_tag,
            "opt_pillar": self.opt_pillar,
            "constituent_op_ids": list(self.constituent_op_ids),
            "constituent_traits": list(self.constituent_traits),
            "symbolic_equations": list(self.symbolic_equations),
            "input_tensors": list(self.input_tensors),
            "output_tensors": list(self.output_tensors),
            "elided_intermediates": list(self.elided_intermediates),
            "elided_hbm_bytes": int(self.elided_hbm_bytes),
            "tile_algorithm": self.tile_algorithm,
            "tile_shape": self.tile_shape,
            "ub_precision_mode": self.ub_precision_mode,
            "ub_peak_bytes": int(self.ub_peak_bytes),
            "ub_budget_bytes": int(self.ub_budget_bytes),
            "l1_peak_bytes": int(self.l1_peak_bytes),
            "l1_budget_bytes": int(self.l1_budget_bytes),
            "l0_peak_bytes": int(self.l0_peak_bytes),
            "l0_budget_bytes": int(self.l0_budget_bytes),
            "sram_verified": bool(self.sram_verified),
            "l_star_us": round(float(self.l_star_us), 3),
            "before_us": round(float(self.before_us), 3),
            "exposed_before_us": round(
                float(self.exposed_before_us if self.exposed_before_us is not None else self.before_us), 3
            ),
            "target_us": round(float(self.target_us), 3),
            "target_attainment_pct": round(float(self.target_attainment_pct), 1),
            "abi_signature_id": self.abi_signature_id,
            "binding_state": self.binding_state,
        }
        if self.ub_wall_warning:
            payload["ub_wall_warning"] = self.ub_wall_warning
        return payload

    def format_formula_block(self) -> str:
        eq_txt = " ; ".join(self.symbolic_equations)
        ub_kb = self.ub_peak_bytes / KIB_BYTES
        ub_cap_kb = self.ub_budget_bytes / KIB_BYTES
        l1_kb = self.l1_peak_bytes / KIB_BYTES
        l1_cap_kb = self.l1_budget_bytes / KIB_BYTES
        elided_kb = self.elided_hbm_bytes / KIB_BYTES
        warn_suffix = f" | WARN: {self.ub_wall_warning}" if self.ub_wall_warning else ""
        return (
            f"Math: {eq_txt} | "
            f"Tile: {self.tile_algorithm} {self.tile_shape} ({self.ub_precision_mode}) | "
            f"UB={ub_kb:.1f}/{ub_cap_kb:.1f}KiB, L1={l1_kb:.1f}/{l1_cap_kb:.1f}KiB "
            f"({'PASS' if self.sram_verified else 'FAIL'}) | "
            f"Elided HBM={elided_kb:.1f}KiB{warn_suffix}"
        )


@dataclass(frozen=True, slots=True)
class MacroWallTrigger:
    """High-level space-time co-design trigger emitted for Agent decisions (§6.4)."""

    wall_kind: Literal["HBM_CAPACITY_WALL", "LATENCY_WALL", "BALANCED"]
    tpot_ms: float
    tpot_slo_ms: float
    sla_slack_ms: float
    hbm_headroom_gib: float
    hbm_utilization_pct: float
    tradeoff_direction: str
    recommended_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_kind": self.wall_kind,
            "tpot_ms": round(float(self.tpot_ms), 3),
            "tpot_slo_ms": round(float(self.tpot_slo_ms), 3),
            "sla_slack_ms": round(float(self.sla_slack_ms), 3),
            "hbm_headroom_gib": round(float(self.hbm_headroom_gib), 2),
            "hbm_utilization_pct": round(float(self.hbm_utilization_pct), 1),
            "tradeoff_direction": self.tradeoff_direction,
            "recommended_actions": list(self.recommended_actions),
        }


@dataclass(frozen=True, slots=True)
class TargetOpNode:
    """Exporter-agnostic operator node for target-state graph optimization."""

    ordinal: int
    occurrence: OpOccurrence
    op_id: str
    archetype_id: str
    repeat_count: int
    family: str
    recipe: str
    stream: str
    stream_index: int
    task_type: str
    shape: tuple[int, ...]
    dtype: str
    predecessors: tuple[OpOccurrence, ...]
    start_us: float
    end_us: float
    dur_us: float
    exposed_dur_us: float
    is_overlapped: bool
    masked_by: str
    flops: float
    vector_ops: float
    hbm_read_bytes: float
    hbm_write_bytes: float
    hbm_total_bytes: float
    mte2_bytes: float
    mte3_bytes: float
    fixpipe_bytes: float
    link_bytes: float
    dominant_bound: str
    bound_display: str
    l_star_lean_us: float
    gap_us: float
    gap_ratio: float | None
    roofline_efficiency: float | None
    absorbed_source_ops: tuple[str, ...]
    semantic_atoms: tuple[str, ...]
    logical_role: str
    ascend_kernel: str
    kernel_backend: str
    kernel_status: str
    needs_new_kernel: bool
    new_kernel_spec: str
    canonical_op_key: str
    opt_pillar: str = ""
    opt_attainment_pct: float | None = None
    spec: OpSpec | None = None
    synthesis_spec: KernelSynthesisSpec | None = None
    binding_state: BindingState = "bound_implementation"
    is_shipping_abi: bool = True
    stream_core_budget: str = ""
    formula_eval: FormulaKernelEvaluation | None = None
    persistent_whole_die: bool = False
    source_ref: str = ""

    @property
    def exposed_us(self) -> float:
        return self.exposed_dur_us


@dataclass(frozen=True, slots=True)
class SolvedTargetGraph:
    """Verified target-state operator graph, synthesis specs, and audit triggers."""

    nodes: tuple[TargetOpNode, ...]
    demands_ns: tuple[tuple[int, int, int, int, int], ...]
    archetype_spans_us: Mapping[str, float]
    synthesis_specs: tuple[KernelSynthesisSpec, ...]
    semantic_audit_triggers: tuple[SemanticAuditTrigger, ...]
    applied_plan: RewritePlan
    rewrite_plan_sha256: str
    verification_passed: bool
    floors: SkuTargetFloors
    has_unbound_formula_ops: bool = False
    binding_state: BindingState = "bound_implementation"
    unbound_formula_ops: tuple[str, ...] = ()
    applied_edit_receipts: tuple[dict[str, Any], ...] = ()
    stream_core_budget: Mapping[tuple[int, int, int], Mapping[str, Any]] = field(default_factory=dict)
    stream_core_budget_summary: str = ""
    moe_route: Mapping[str, Any] = field(default_factory=dict)
    step_start_us: float = 0.0
    step_end_us: float = 0.0
    step_span_us: float = 0.0
    co_design_opportunities: tuple[CoDesignOpportunity, ...] = ()
    stage_waterfall: tuple[dict[str, Any], ...] = ()
    optimization_mode: OptimizationMode = "modeling"

    @property
    def exposed_us(self) -> float:
        return self.step_span_us


TargetGraphResult = SolvedTargetGraph


def _die_launch_us(die: Device, datapath: DatapathCapabilitySpec, primitive_table: Any = None) -> tuple[float, float, float]:
    """Read L0 kernel completion minimums from PrimitiveTable or Device declaration."""
    inv = dict(getattr(die, "kernel_invocations", ()) or ())

    def _inv_us(key: str) -> float:
        entry = inv.get(key)
        if isinstance(entry, Mapping) and entry.get("minimum_ms") is not None:
            val = float(entry["minimum_ms"]) * US_PER_MS
            if val > 0.0:
                return val
        launch = float(getattr(die, "launch_us", 0.0) or 0.0)
        return max(0.0, launch)

    launch_vec = _inv_us("vector")
    launch_cube = _inv_us("cube")
    launch_norm = _inv_us("norm") or launch_vec
    env = resolve_dispatch_envelope(primitive_table, datapath) if primitive_table is not None else None
    if env is not None:
        aiv_us = float(env.aiv_full_die_floor_ms) * US_PER_MS
        if aiv_us > 0.0:
            launch_vec = max(launch_vec, aiv_us) if launch_vec > 0.0 else aiv_us
            launch_norm = max(launch_norm, aiv_us)
        if launch_cube <= 0.0:
            launch_cube = float(env.mix_aic_floor_ms(0.0)) * US_PER_MS
    return launch_vec, launch_cube, launch_norm


def _min_link_latency_us(machine: Hardware) -> tuple[float, float, bool]:
    """Extract ``(min_intra_latency_us, min_any_latency_us, has_intra_links)`` from Hardware."""
    intra_lats: list[float] = []
    all_lats: list[float] = []
    for lnk in getattr(machine, "links", ()) or ():
        lat = float(getattr(lnk, "latency_us", 0.0) or 0.0)
        scope = str(getattr(lnk, "scope", "") or "")
        if lat > 0.0:
            all_lats.append(lat)
            if scope in _INTRA_LINK_SCOPES:
                intra_lats.append(lat)
    channels = getattr(machine, "transfer_channels", ()) or ()
    for ch in channels:
        name = str(getattr(ch, "name", "") or "")
        lat = float(getattr(ch, "alpha_us", 0.0) or getattr(ch, "latency_us", 0.0) or 0.0)
        if lat > 0.0:
            all_lats.append(lat)
            if "intra" in name or "package" in name or "superpod" in name:
                intra_lats.append(lat)
    return (
        min(intra_lats) if intra_lats else 0.0,
        min(all_lats) if all_lats else 0.0,
        bool(intra_lats),
    )


def extract_sku_floors(case: Case, *, primitive_table: Any = None) -> SkuTargetFloors:
    """Read SKU L0 kernel launch & interconnect floors purely from ``Case`` capabilities."""
    die = case.machine.dies[0]
    arch = resolve_arch_family_tag(case)
    ub_edges = ub_peer_edges_for_rank(die.rank, case.machine.links, case.machine.transfer_channels)
    datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
    launch_vec, launch_cube, launch_norm = _die_launch_us(die, datapath, primitive_table)
    intra_lat, any_lat, has_intra_links = _min_link_latency_us(case.machine)
    tp = max(1, int(case.runtime.parallel.tp))
    rpn = max(1, int(getattr(case.machine, "ranks_per_node", tp) or tp))
    is_a5 = arch == "Arch35"
    has_ub_direct = bool(is_a5 and (ub_edges or (datapath.has_l0c_to_ub and has_intra_links)))
    is_pod = bool(has_ub_direct or has_intra_links or rpn >= tp)
    link_alpha = intra_lat if (is_pod and intra_lat > 0.0) else any_lat
    oneshot = launch_vec + link_alpha
    if has_ub_direct and is_pod:
        label = "UB-Direct (CCU)"
    elif arch == "Arch22" and is_pod:
        label = "HCCS SuperPoD (AIV)"
    elif arch == "Arch20" and is_pod:
        label = "8P HCCS (AIV)"
    else:
        label = "Inter-Node Collective"
    return SkuTargetFloors(
        launch_vec_us=launch_vec,
        launch_cube_us=launch_cube,
        launch_norm_us=launch_norm,
        oneshot_floor_us=oneshot,
        mc2_sync_us=oneshot,
        is_intra_pod=is_pod,
        is_a5_ub=has_ub_direct and is_pod,
        interconnect_label=label,
    )


def _extract_hardware_supply(
    hardware: Case | Hardware | Device | DatapathCapabilitySpec,
    primitive_table: Any = None,
) -> tuple[DatapathCapabilitySpec, dict[str, float], dict[str, float], dict[str, tuple[str, float]], float, float, float, float]:
    """Resolve ``(datapath, rates_per_s, peaks, service_modes, launch_vec_us, launch_cube_us, link_bps, link_lat_us)``."""
    links: tuple[Any, ...] = ()
    channels: tuple[Any, ...] = ()
    die: Device | None = None
    if isinstance(hardware, Case):
        die = hardware.machine.dies[0]
        links = tuple(hardware.machine.links)
        channels = tuple(hardware.machine.transfer_channels)
    elif isinstance(hardware, Hardware):
        die = hardware.dies[0]
        links = tuple(hardware.links)
        channels = tuple(hardware.transfer_channels)
    elif isinstance(hardware, Device):
        die = hardware
    elif isinstance(hardware, DatapathCapabilitySpec):
        datapath = hardware
    else:
        raise ModelingError(f"formula_kernel_unsupported_hardware:{type(hardware).__name__}")

    if die is not None:
        ub_edges = ub_peer_edges_for_rank(die.rank, links, channels)
        datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
        rates = dict(die.rates)
        peaks = dict(die.compute_peaks)
        service_modes = dict(die.service_modes)
        launch_vec, launch_cube, _ = _die_launch_us(die, datapath, primitive_table)
    else:
        rates = {}
        hbm_rate = max(float(datapath.hbm_read_bytes_per_s or 0.0), float(datapath.hbm_write_bytes_per_s or 0.0))
        if hbm_rate > 0.0:
            rates["hbm"] = hbm_rate
            rates["mte2"] = hbm_rate
            rates["mte3"] = hbm_rate
        peaks = dict(datapath.compute_peaks)
        service_modes = {}
        stars = datapath.stars
        base_launch = float(stars.issue_interval_us) if stars is not None else 0.0
        launch_vec = launch_cube = base_launch

    if primitive_table is not None:
        sku = getattr(die, "sku", "") if die is not None else ""
        overlay = resolve_sim_rate_overlay(primitive_table, datapath, sku=sku)
        for role, val in overlay.rates:
            if val > 0.0:
                rates[role] = min(val, rates[role]) if rates.get(role, 0.0) > 0.0 else val
        for dt, pk in overlay.compute_peaks:
            if pk > 0.0:
                peaks[dt] = pk
        if overlay.hbm_read_bytes_per_s is not None:
            service_modes["mte2_aiv_dram"] = ("mte2_aiv" if "mte2_aiv" in rates else "mte2", overlay.hbm_read_bytes_per_s)
        if overlay.hbm_write_bytes_per_s is not None:
            service_modes["hbm_write_dram"] = ("hbm", overlay.hbm_write_bytes_per_s)
        if overlay.hbm_rw11_bytes_per_s is not None:
            service_modes["hbm_rw11_dram"] = ("hbm", overlay.hbm_rw11_bytes_per_s)
        if overlay.hbm_rw21_bytes_per_s is not None:
            service_modes["hbm_rw21_dram"] = ("hbm", overlay.hbm_rw21_bytes_per_s)

    link_bps = float(rates.get("link") or 0.0)
    link_lat_us = 0.0
    if links:
        def _link_bps(item: Any) -> float:
            b_per_ms = float(getattr(item, "capacity_bytes_per_ms", 0.0) or 0.0)
            if b_per_ms > 0.0:
                return b_per_ms * MS_PER_SECOND
            return float(getattr(item, "capacity_gbps", 0.0) or 0.0) * BYTES_PER_GB

        best = max(links, key=_link_bps)
        best_bps = _link_bps(best)
        if best_bps > 0.0:
            link_bps = best_bps
        link_lat_us = float(getattr(best, "latency_us", 0.0) or 0.0)
    return datapath, rates, peaks, service_modes, launch_vec, launch_cube, link_bps, link_lat_us


def _solve_formula_sram(
    spec: FormulaKernelSpec,
    datapath: DatapathCapabilitySpec,
    *,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> tuple[TilePipelineSpec | None, str, str, int, int, int, int, int, int, int, bool]:
    """Solve Ascend ``FP32`` (4 B/elem) UB/L1/L0 tiling for ``FormulaKernelSpec`` via ``TileSolver``."""
    bufs = dict(datapath.buffers)
    carveout = max(SIMT_UB_CARVEOUT_BYTES, int(datapath.simt_ub_carveout_bytes or 0))
    ub_raw = int(bufs.get("ub", 0) or 0)
    ub_budget = max(0, ub_raw - carveout)
    l1_budget = int(bufs.get("l1", 0) or 0)
    l0_budget = int(bufs.get("l0c", 0) or 0)
    has_cube = any(fl > 0.0 for _, fl in spec.matrix_flops)
    op_traits = tuple(classify_op_traits(op) for op in spec.atomic_ops)
    has_row_red = bool(spec.row_reduction) or any(
        tr.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE}
        for tr in op_traits
    )
    has_aiv = (
        any(tr.trait != AlgebraicTrait.BILINEAR_MATMUL for tr in op_traits)
        or spec.vector_ops_fp32 > 0.0
        or has_row_red
        or not has_cube
    )

    all_shapes = [t.shape for t in (*spec.inputs, *spec.outputs) if t.shape]
    in_b = max((_safe_elem_bytes(t.dtype) for t in spec.inputs), default=2.0)
    out_b = max((_safe_elem_bytes(t.dtype) for t in spec.outputs), default=2.0)
    internal_red_widths: list[int] = []
    internal_aiv_widths: list[int] = []
    internal_row_bytes: list[int] = []
    for op, tr in zip(spec.atomic_ops, op_traits):
        in_b = max(in_b, _safe_elem_bytes(tr.input_dtype))
        out_b = max(out_b, _safe_elem_bytes(tr.output_dtype))
        if tr.trait == AlgebraicTrait.BILINEAR_MATMUL or not op.local_shape:
            continue
        op_n = int(op.local_shape[-1])
        if op_n <= 0:
            continue
        op_in_b = max(_safe_elem_bytes(tr.input_dtype), float(FP32_ACCUM_BYTES))
        op_out_b = max(_safe_elem_bytes(tr.output_dtype), float(FP32_ACCUM_BYTES))
        if tr.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE}:
            internal_red_widths.append(op_n)
            internal_row_bytes.append(max(
                tr.row_working_bytes,
                int(op_n * (op_in_b + op_out_b + float(FP32_ACCUM_BYTES))),
            ))
        else:
            internal_aiv_widths.append(op_n)
            if has_row_red:
                internal_row_bytes.append(max(tr.row_working_bytes, int(op_n * (op_in_b + op_out_b))))

    width_pool = [int(sh[-1]) for sh in all_shapes] + internal_aiv_widths + internal_red_widths
    inner_dim = max(width_pool, default=CUBE_TILE_MN_ELEMS)
    if internal_red_widths:
        # The AIV reduction row spans the reducing op's own width, never an
        # unrelated boundary dimension (e.g. a downstream LM-head vocab axis).
        tile_n = max(internal_red_widths)
    elif has_row_red:
        tile_n = inner_dim  # equation-only row-reduction specification
    else:
        tile_n = min(inner_dim, CUBE_TILE_MN_ELEMS)
    if has_aiv:
        # Ascend AIV upcasts vector intermediates to FP32 (4 B/elem) inside UB.
        base_row_bytes = int(tile_n * (max(in_b, float(FP32_ACCUM_BYTES)) + max(out_b, float(FP32_ACCUM_BYTES))))
        if has_row_red:
            base_row_bytes += int(tile_n * FP32_ACCUM_BYTES)
        row_fp32_bytes = max(base_row_bytes, max(internal_row_bytes, default=0))
    else:
        row_fp32_bytes = 0
    l1_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_K_ELEMS * (in_b + out_b) * UB_DOUBLE_BUFFER_SLOTS) if has_cube else 0
    l0_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_MN_ELEMS * L0C_ACCUM_BYTES) if has_cube else 0

    candidates: list[TileCandidate] = []
    has_simd_ub_headroom = bool(has_aiv and int(datapath.simt_ub_carveout_bytes or 0) > 0 and ub_raw > ub_budget)
    cand_modes: tuple[tuple[int, bool], ...] = (
        ((UB_DOUBLE_BUFFER_SLOTS, True), (1, True), (1, False))
        if has_simd_ub_headroom else ((UB_DOUBLE_BUFFER_SLOTS, has_aiv), (1, has_aiv))
    )
    for slots, cand_simt in cand_modes:
        live: list[LiveBufferSpec] = []
        if has_aiv and row_fp32_bytes > 0:
            live.append(LiveBufferSpec("ub_fp32_work", "ub", row_fp32_bytes, 0, 1, owner_kind="aiv", slots=slots))
        if has_cube and l1_tile > 0:
            live.append(LiveBufferSpec("l1_cube_tile", "l1", l1_tile, 0, 1, owner_kind="aic", slots=1))
        if has_cube and l0_tile > 0:
            live.append(LiveBufferSpec("l0c_fp32_accum", "l0c", l0_tile, 0, 1, owner_kind="aic", slots=1))
        algo: Literal["resident_1pass", "online_reduce"] = (
            "resident_1pass" if slots == UB_DOUBLE_BUFFER_SLOTS else "online_reduce"
        )
        candidates.append(TileCandidate(
            algorithm=algo,
            tile_shape=TileShape(1, max(1, tile_n), CUBE_TILE_K_ELEMS if has_cube else 1),
            depth=slots,
            owner_partition=(active_aic or datapath.aic_count, active_aiv or datapath.aiv_count),
            work_items=1,
            tiles_per_owner=1,
            buffers=tuple(live),
            uses_simt=cand_simt,
            owner_axis="aic" if (has_cube and not has_aiv) else "aiv",
        ))
    tile_spec = TilePipelineSpec(spec.family or "formula_kernel", tuple(candidates), simt_ub_carveout_bytes=carveout)
    hw = SimpleNamespace(
        datapath=datapath,
        buffers=bufs,
        buffer_units={"ub": datapath.aiv_count, "l1": datapath.aic_count, "l0c": datapath.aic_count, "l2": 1},
        stream_owners=None,
    )
    max_ub_budget = ub_raw if has_simd_ub_headroom else ub_budget
    if has_aiv and ub_raw <= 0:
        return tile_spec, "unbounded", f"(M=1, N={tile_n}, K=1)", 0, 0, l1_tile, l1_budget, l0_tile, l0_budget, 1, False
    try:
        solved = tile_solver.solve(tile_spec, None, hw)
        peaks = {p.pool: int(p.peak_bytes) for p in solved.pool_peaks}
        caps = {p.pool: int(p.capacity_bytes) for p in solved.pool_peaks}
        ub_pk = peaks.get("ub", row_fp32_bytes) if has_aiv else 0
        eff_ub_budget = caps.get("ub", ub_budget) if has_aiv else ub_budget
        l1_pk = peaks.get("l1", l1_tile) if has_cube else 0
        l0_pk = peaks.get("l0c", l0_tile) if has_cube else 0
        ok = (
            (not has_aiv or ub_pk <= eff_ub_budget)
            and (not has_cube or (l1_pk <= l1_budget and l0_pk <= l0_budget))
        )
        sh = solved.tile_shape
        return (
            tile_spec, solved.algorithm, f"(M={sh.m}, N={sh.n}, K={sh.k})",
            ub_pk, eff_ub_budget, l1_pk, l1_budget, l0_pk, l0_budget, solved.depth, ok,
        )
    except tile_solver.TileInfeasibleError:
        fail_algo = "ub_infeasible" if (has_aiv and row_fp32_bytes > max_ub_budget) else "sram_infeasible"
        return (
            tile_spec, fail_algo, f"(M=1, N={tile_n}, K={CUBE_TILE_K_ELEMS if has_cube else 1})",
            row_fp32_bytes, max_ub_budget, l1_tile, l1_budget, l0_tile, l0_budget, 1, False,
        )


def _resolve_formula_work_bytes(spec: FormulaKernelSpec) -> tuple[float, float, float, float]:
    """Return ``(mte2_bytes, mte3_bytes, hbm_bytes, vector_ops_fp32)`` for ``spec``."""
    in_bytes = sum(t.byte_size for t in (*spec.inputs, *spec.weights) if t.hbm_resident)
    out_bytes = sum(t.byte_size for t in spec.outputs if t.hbm_resident)
    mte2 = float(spec.mte2_bytes_override) if spec.mte2_bytes_override is not None else in_bytes
    mte3 = float(spec.mte3_bytes_override) if spec.mte3_bytes_override is not None else out_bytes
    hbm = float(spec.hbm_bytes_override) if spec.hbm_bytes_override is not None else (mte2 + mte3)
    vec_ops = float(spec.vector_ops_fp32)
    if vec_ops <= 0.0 and not spec.matrix_flops and spec.equations:
        elems = max((t.num_elements for t in (*spec.inputs, *spec.outputs)), default=0)
        vec_ops = float(elems) * DEFAULT_FORMULA_VEC_OPS_PER_ELEM * float(len(spec.equations))
    return max(0.0, mte2), max(0.0, mte3), max(0.0, hbm), max(0.0, vec_ops)


def _compute_peak_for_dtype(peaks: Mapping[str, float], dtype: str) -> float:
    tok = str(dtype or "bf16").strip().lower()
    aliases = {
        "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
        "fp8_e4m3": "fp8", "fp8_e5m2": "fp8",
    }
    for key in (tok, aliases.get(tok, tok), "bf16", "fp16"):
        val = float(peaks.get(key, 0.0) or 0.0)
        if val > 0.0:
            return val
    return max((float(v) for v in peaks.values() if float(v) > 0.0), default=0.0)


def _build_formula_engine_inputs(
    spec: FormulaKernelSpec,
    rates: Mapping[str, float],
    peaks: Mapping[str, float],
    datapath: DatapathCapabilitySpec,
    *,
    mte2: float,
    mte3: float,
    hbm: float,
    vec_ops: float,
    link_bps: float,
    aic_eff: int,
    aiv_eff: int,
) -> tuple[list[Resource], dict[str, str], dict[str, float], dict[str, int], dict[str, float], dict[str, float], dict[str, float], float]:
    has_cube = any(fl > 0.0 for _, fl in spec.matrix_flops)
    total_flops = sum(float(fl) for _, fl in spec.matrix_flops if fl > 0.0)
    aic_scale = float(aic_eff) / float(max(1, datapath.aic_count))
    aiv_scale = float(aiv_eff) / float(max(1, datapath.aiv_count))
    cube_ms = 0.0
    for dt, fl in spec.matrix_flops:
        if fl <= 0.0:
            continue
        pk = _compute_peak_for_dtype(peaks, dt) * aic_scale
        if pk <= 0.0:
            raise ModelingError(f"formula_kernel_compute_peak_missing:{dt}")
        cube_ms += MS_PER_SECOND * float(fl) / pk

    hbm_rate = float(rates.get("hbm") or max(datapath.hbm_read_bytes_per_s, datapath.hbm_write_bytes_per_s) or 0.0)
    vec_rate = float(rates.get("vector") or 0.0) * aiv_scale
    mte2_key = "mte2_aic" if (has_cube and "mte2_aic" in rates) else ("mte2_aiv" if "mte2_aiv" in rates else "mte2")
    mte2_rate = float(rates.get(mte2_key) or hbm_rate or 0.0) * (aic_scale if mte2_key == "mte2_aic" else aiv_scale)
    mte3_rate = float(rates.get("mte3") or hbm_rate or 0.0) * aiv_scale
    if hbm > 0.0 and hbm_rate <= 0.0:
        raise ModelingError("formula_kernel_hbm_rate_missing")
    if vec_ops > 0.0 and vec_rate <= 0.0:
        raise ModelingError("formula_kernel_vector_rate_missing")
    if spec.link_bytes > 0.0 and link_bps <= 0.0:
        raise ModelingError("formula_kernel_link_rate_missing")

    res_list: list[Resource] = [Resource("matrix@0", 1.0, "peak_ms", "fair", aic_eff)]
    res_names: dict[str, str] = {"cube": "matrix@0"}
    res_rates: dict[str, float] = {"matrix@0": 1.0}
    res_units: dict[str, int] = {"cube": aic_eff}
    work_map: dict[str, float] = {}
    native_work: dict[str, float] = {}
    if cube_ms > 0.0:
        work_map["matrix@0"] = cube_ms
        native_work["cube"] = total_flops
    pipe_rows = (
        ("hbm", "memory@0", hbm_rate, "bytes", 1, hbm),
        ("mte2", "mte2@0", mte2_rate, "bytes", aic_eff if has_cube else aiv_eff, mte2),
        ("mte3", "mte3@0", mte3_rate, "bytes", aiv_eff, mte3),
        ("vector", "vector@0", vec_rate, "ops", aiv_eff, vec_ops),
        ("link", "link:0", link_bps, "bytes", 1, float(spec.link_bytes)),
    )
    for role, rname, rate_s, unit, units, amount in pipe_rows:
        if rate_s <= 0.0 or (role == "link" and amount <= 0.0):
            continue
        res_list.append(Resource(rname, rate_s / MS_PER_SECOND, unit, "fair", units))
        res_names[role] = rname
        res_rates[rname] = rate_s / MS_PER_SECOND
        res_units[role] = units
        if amount > 0.0:
            work_map[rname] = amount
            native_work[role] = amount

    pipe_times_us = {
        "cube": cube_ms * US_PER_MS,
        "vector": (vec_ops / vec_rate) * US_PER_SECOND if (vec_ops > 0.0 and vec_rate > 0.0) else 0.0,
        "hbm": (hbm / hbm_rate) * US_PER_SECOND if (hbm > 0.0 and hbm_rate > 0.0) else 0.0,
        "mte2": (mte2 / mte2_rate) * US_PER_SECOND if (mte2 > 0.0 and mte2_rate > 0.0) else 0.0,
        "mte3": (mte3 / mte3_rate) * US_PER_SECOND if (mte3 > 0.0 and mte3_rate > 0.0) else 0.0,
        "link": (float(spec.link_bytes) / link_bps) * US_PER_SECOND if (spec.link_bytes > 0.0 and link_bps > 0.0) else 0.0,
    }
    return res_list, res_names, res_rates, res_units, work_map, native_work, pipe_times_us, hbm_rate


def evaluate_formula_kernel(
    spec: FormulaKernelSpec,
    hardware: Case | Hardware | Device | DatapathCapabilitySpec,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Evaluate a Layer-1 ``FormulaKernelSpec`` through ``TileSolver`` + ``Engine`` L0 pipes."""
    datapath, rates, peaks, service_modes, launch_vec, launch_cube, link_bps, link_lat_us = (
        _extract_hardware_supply(hardware, primitive_table=primitive_table)
    )
    (
        tile_spec, algo, shape_str, ub_pk, ub_cap, l1_pk, l1_cap, l0_pk, l0_cap, depth, sram_ok,
    ) = _solve_formula_sram(spec, datapath, active_aic=active_aic, active_aiv=active_aiv)
    mte2, mte3, hbm, vec_ops = _resolve_formula_work_bytes(spec)
    has_cube = any(fl > 0.0 for _, fl in spec.matrix_flops)
    total_flops = sum(float(fl) for _, fl in spec.matrix_flops if fl > 0.0)
    aic_eff = max(1, int(active_aic or datapath.aic_count or 1))
    aiv_eff = max(1, int(active_aiv or datapath.aiv_count or 1))
    res_list, res_names, res_rates, res_units, work_map, native_work, pipe_times_us, hbm_rate = (
        _build_formula_engine_inputs(
            spec, rates, peaks, datapath,
            mte2=mte2, mte3=mte3, hbm=hbm, vec_ops=vec_ops,
            link_bps=link_bps, aic_eff=aic_eff, aiv_eff=aiv_eff,
        )
    )
    launch_us = launch_cube if has_cube else launch_vec
    setup_ms = (link_lat_us / US_PER_MS) if spec.link_bytes > 0.0 else 0.0
    op_shape = spec.outputs[0].shape if spec.outputs else (spec.inputs[0].shape if spec.inputs else ())
    op_stub = OpSpec(
        op_id=spec.name, family=spec.family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=op_shape, local_shape=op_shape,
        padded_shape=op_shape, launch_shape=op_shape,
        dtype=spec.outputs[0].dtype if spec.outputs else (spec.inputs[0].dtype if spec.inputs else "bf16"),
        flops=total_flops, matrix_flops=spec.matrix_flops, vector_ops=vec_ops,
        hbm_bytes=hbm, mte2_bytes=mte2, mte3_bytes=mte3, link_bytes=float(spec.link_bytes),
    )
    abi_id = spec.resolved_abi_signature()
    resolved_stages = spec.pipe_stages
    if not resolved_stages and spec.atomic_ops:
        resolved_stages = _derive_pipe_stages_from_ops(
            spec.atomic_ops, tuple(classify_op_traits(op) for op in spec.atomic_ops),
        )
    elif not resolved_stages and spec.equations:
        resolved_stages = _derive_pipe_stages_from_equations(
            spec.equations,
            has_cube=has_cube,
            has_vector=(vec_ops > 0.0),
            has_link=(spec.link_bytes > 0.0),
            row_reduction=spec.row_reduction,
        )
    recipe = FusedPipelineRecipe(
        slots=depth,
        row_reduction=spec.row_reduction,
        abi_signature_id=abi_id,
        binding_state="unbound_formula_projection",
        pipe_stages=resolved_stages,
    )
    ctx = RecipeContext(
        op=op_stub, work=work_map, resource_names=res_names, rates=res_rates,
        core_count=aic_eff if has_cube else aiv_eff,
        minimum_ms=launch_us / US_PER_MS, setup_ms=setup_ms,
        resource_units=res_units, compute_peaks=peaks, service_modes=service_modes,
        native_work=native_work, datapath=datapath, primitive_table=primitive_table,
    )
    bound_ctx = recipe.bind_context(ctx)
    if bound_ctx.service_modes.get("hbm") != ctx.service_modes.get("hbm") and "memory@0" in work_map:
        _, eff_hbm_bps = bound_ctx.service_modes["hbm"]
        if eff_hbm_bps > 0.0:
            work_map = {**work_map, "memory@0": hbm * (hbm_rate / eff_hbm_bps)}
            bound_ctx = replace(bound_ctx, work=work_map)

    dom_pipe, l_star_us = max(pipe_times_us.items(), key=lambda kv: kv[1])
    if l_star_us <= 0.0:
        dom_pipe = "launch"

    engine = Engine(res_list, backend="python")
    ready = engine.submit("ready")
    done = recipe.lower(bound_ctx).start(engine, spec.name, ready)
    engine.advance(until=(done,))
    projected_us = max(float(done.time) * US_PER_MS, l_star_us)
    return FormulaKernelEvaluation(
        name=spec.name, abi_signature_id=abi_id, composed_equations=spec.equations,
        projected_us=projected_us, l_star_us=l_star_us, tile_spec=tile_spec,
        ub_peak_bytes=ub_pk, ub_budget_bytes=ub_cap, l1_peak_bytes=l1_pk, l0_peak_bytes=l0_pk,
        sram_verified=sram_ok, binding_state="unbound_formula_projection",
        has_unbound_formula_ops=True, dominant_pipe=dom_pipe,
        hbm_bytes=hbm, mte2_bytes=mte2, mte3_bytes=mte3, launch_us=launch_us,
        tile_algorithm=algo, tile_shape_str=shape_str,
        l1_budget_bytes=l1_cap, l0_budget_bytes=l0_cap,
    )


simulate_formula_kernel = evaluate_formula_kernel


def evaluate_macro_wall_trigger(
    *,
    tpot_ms: float,
    tpot_slo_ms: float,
    hbm_used_bytes: float,
    hbm_usable_bytes: float,
    hbm_headroom_bytes: float,
) -> MacroWallTrigger:
    """Classify active macro deployment wall for Agent space-time trade-off guidance (§6.4)."""
    slo = max(tpot_slo_ms, SAVING_POSITIVE_EPS_US)
    slack_ms = slo - tpot_ms
    util_pct = (PERCENT_SCALE * hbm_used_bytes / hbm_usable_bytes) if hbm_usable_bytes > 0.0 else 0.0
    headroom_gib = hbm_headroom_bytes / float(GIB_BYTES)

    if slack_ms >= SLA_SLACK_WALL_MS and util_pct >= HBM_WALL_UTIL_PCT:
        return MacroWallTrigger(
            wall_kind="HBM_CAPACITY_WALL",
            tpot_ms=tpot_ms,
            tpot_slo_ms=tpot_slo_ms,
            sla_slack_ms=slack_ms,
            hbm_headroom_gib=headroom_gib,
            hbm_utilization_pct=util_pct,
            tradeoff_direction="compute_for_memory_and_offload",
            recommended_actions=(
                f"HBM 利用率 {util_pct:.1f}% 且 SLA 余量 {slack_ms:.2f} ms: 优先评估存储换并发（KV 分级池化/状态压缩/并行重分配）",
            ),
        )
    if tpot_ms >= slo * SLA_NEAR_BOUND_RATIO and headroom_gib >= LATENCY_WALL_FREE_GIB:
        return MacroWallTrigger(
            wall_kind="LATENCY_WALL",
            tpot_ms=tpot_ms,
            tpot_slo_ms=tpot_slo_ms,
            sla_slack_ms=slack_ms,
            hbm_headroom_gib=headroom_gib,
            hbm_utilization_pct=util_pct,
            tradeoff_direction="memory_for_compute",
            recommended_actions=(
                f"TPOT {tpot_ms:.2f} ms 逼近 SLA {tpot_slo_ms:.2f} ms 且 HBM 空余 {headroom_gib:.2f} GiB: 优先评估以存换算与片上融合降延迟",
            ),
        )
    return MacroWallTrigger(
        wall_kind="BALANCED",
        tpot_ms=tpot_ms,
        tpot_slo_ms=tpot_slo_ms,
        sla_slack_ms=slack_ms,
        hbm_headroom_gib=headroom_gib,
        hbm_utilization_pct=util_pct,
        tradeoff_direction="balanced",
        recommended_actions=(
            f"延迟 ({tpot_ms:.2f}/{tpot_slo_ms:.2f} ms) 与 HBM ({util_pct:.1f}%, 余量 {headroom_gib:.2f} GiB) 处于均衡区间",
        ),
    )


def attribute_node_overlap(
    nodes: Sequence[TargetOpNode],
) -> tuple[list[TargetOpNode], dict[str, float]]:
    """Attribute cross-stream/cross-archetype overlap via conserved ``ExposedObserver`` sweep-line."""
    span_by_arch: dict[str, float] = {n.archetype_id: 0.0 for n in nodes}
    if not nodes:
        return [], span_by_arch
    occ_to_id = {n.occurrence: str(i) for i, n in enumerate(nodes)}
    intervals: list[ObservedInterval] = []
    preds: dict[str, tuple[str, ...]] = {}
    min_ns: int | None = None
    max_ns = 0
    for idx, node in enumerate(nodes):
        s_ns = max(0, int(round(float(node.start_us) * NS_PER_US)))
        e_ns = max(s_ns, int(round(float(node.end_us) * NS_PER_US)))
        if e_ns > s_ns:
            min_ns = s_ns if min_ns is None else min(min_ns, s_ns)
            max_ns = max(max_ns, e_ns)
        oid = str(idx)
        label = (node.logical_role or node.ascend_kernel or _strip_op_prefix(node.op_id))[:MAX_MASKED_BY_LEN]
        intervals.append(ObservedInterval(
            occurrence_id=oid, stream=node.stream, stream_index=int(node.stream_index),
            start_ns=s_ns, end_ns=e_ns, ordinal=int(node.ordinal), label=label,
            replica=int(node.occurrence.replica),
        ))
        preds[oid] = tuple(occ_to_id[p] for p in node.predecessors if p in occ_to_id)
    start_i = min_ns if min_ns is not None else 0
    res = attribute_stream0_intervals(intervals, start_ns=start_i, terminal_ns=max_ns, causal_preds=preds)
    updated: list[TargetOpNode] = []
    for idx, node in enumerate(nodes):
        oid = str(idx)
        if node.dur_us <= 0.0:
            updated.append(replace(node, is_overlapped=False, exposed_dur_us=0.0, masked_by=""))
            continue
        exp_us = min(node.dur_us, float(res.exposed_ns_by_op.get(oid, 0)) / NS_PER_US)
        fully_hidden = exp_us <= OVERLAP_EXPOSED_EPS_US
        eff_exp = 0.0 if fully_hidden else exp_us
        masker = res.masked_by.get(oid, "") if (node.dur_us - eff_exp > OVERLAP_EXPOSED_EPS_US) else ""
        eff_gap = 0.0 if fully_hidden else max(0.0, eff_exp - node.l_star_lean_us)
        eff_ratio = None if fully_hidden else ((eff_exp / node.l_star_lean_us) if node.l_star_lean_us > 0.0 else None)
        b_disp = f"Overlapped by {masker} (0 exposed)" if (fully_hidden and masker) else node.bound_display
        updated.append(replace(
            node, is_overlapped=fully_hidden, exposed_dur_us=eff_exp, masked_by=masker,
            gap_us=eff_gap, gap_ratio=eff_ratio, bound_display=b_disp,
        ))
        span_by_arch[node.archetype_id] = span_by_arch.get(node.archetype_id, 0.0) + eff_exp
    return updated, span_by_arch


def reschedule_node_timelines(
    nodes: Sequence[TargetOpNode],
    *,
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]] | None = None,
    topology: Topology | None = None,
    case: Case | None = None,
    primitive_table: Any = None,
    stream_plan: Any = None,
    launch_floor_us: float = 0.0,
) -> list[TargetOpNode]:
    """Recompute exact causal ``(start_us, end_us)`` via Die + Topology Engine with strict FIFO."""
    if not nodes:
        return []
    if case is not None:
        return list(schedule_target_graph_on_engine(
            nodes,
            case,
            collective_plans=collective_plans,
            topology=topology,
            primitive_table=primitive_table,
            stream_plan=stream_plan,
            launch_floor_us=launch_floor_us,
        ).nodes)

    # Pure structural fallback when called without a Case: still enforces strict per-stream FIFO.
    by_arch: dict[str, list[tuple[int, TargetOpNode]]] = defaultdict(list)
    arch_order: list[str] = []
    for idx, node in enumerate(nodes):
        if node.archetype_id not in by_arch:
            arch_order.append(node.archetype_id)
        by_arch[node.archetype_id].append((idx, node))

    updated = list(nodes)
    cursor_us = 0.0
    end_by_occ: dict[Any, float] = {}
    stream_tail: dict[tuple[int, int, int, int], float] = {}
    for aid in arch_order:
        ordered = sorted(by_arch[aid], key=lambda item: (item[1].ordinal, item[1].start_us))
        arch_origin = cursor_us
        arch_max_end = arch_origin
        for idx, node in ordered:
            occ = node.occurrence
            pred_ready = max((end_by_occ.get(p, arch_origin) for p in node.predecessors), default=arch_origin)
            q_key = (int(occ.replica), int(occ.stage), int(occ.rank or 0), int(node.stream_index))
            stream_ready = stream_tail.get(q_key, arch_origin)
            new_s = max(arch_origin, pred_ready, stream_ready)
            new_e = new_s + max(0.0, float(node.dur_us))
            end_by_occ[occ] = new_e
            stream_tail[q_key] = new_e
            arch_max_end = max(arch_max_end, new_e)
            updated[idx] = replace(node, start_us=new_s, end_us=new_e)
        cursor_us = arch_max_end
    return updated


def _op_input_shape(op: OpSpec, trait: OpTraitInfo) -> tuple[int, ...]:
    sh = tuple(op.local_shape)
    if trait.trait == AlgebraicTrait.BILINEAR_MATMUL and len(sh) == MATMUL_SHAPE_RANK:
        return (int(sh[0]), int(sh[2]))
    return sh


def _op_output_shape(op: OpSpec, trait: OpTraitInfo) -> tuple[int, ...]:
    sh = tuple(op.local_shape)
    if trait.trait == AlgebraicTrait.BILINEAR_MATMUL and len(sh) == MATMUL_SHAPE_RANK:
        return (int(sh[0]), int(sh[1]))
    return sh


def _format_megamoe_equation(
    prev: str,
    out: str,
    shape_tag: str,
    param_txt: str,
    trait: OpTraitInfo,
    meta: Mapping[str, Any],
) -> str:
    """Render ``aclnnMegaMoe`` equation from resolved geometry (conditional EP + mode-3 shared)."""
    raw_geo = meta.get("megamoe_geometry")
    geo = raw_geo if isinstance(raw_geo, Mapping) else {}
    ep = int(geo.get("ep", 0) or (2 if trait.collective_axis == "ep" else 1))
    mc2_mode = int(geo.get("enable_fused_mc2", FUSED_MC2_ROUTED_LEVEL if ep > 1 else 1) or 0)
    has_ep = (ep > 1 and mc2_mode >= FUSED_MC2_ROUTED_LEVEL) or (not geo and trait.collective_axis == "ep")
    act = str(geo.get("activation") or trait.activation_kind or "swiglu")
    comp_dt = str(geo.get("compute_dtype") or trait.input_dtype or "bf16")
    shared_inter = int(geo.get("shared_intermediate", 0) or 0)
    has_shared = shared_inter > 0 or mc2_mode == FUSED_MC2_SHARED_LEVEL

    disp_step = f"h_disp = AllToAllDispatch({prev}, ep={ep})" if has_ep else f"h_disp = LocalTokenPermute({prev})"
    gmm1 = f"u = GroupedMatMul(h_disp, W_gate_up^T [{comp_dt}])"
    act_step = f"a, s = ActMxQuant({act}(u{param_txt}))"
    gmm2 = "y_loc = GroupedMatMul(a, W_down^T, scales=s)"
    comb_expr = "AllToAllCombine(y_loc, topk_w)" if has_ep else "LocalWeightedCombine(y_loc, topk_w)"
    if has_shared:
        shared_tag = f", inter={shared_inter}" if shared_inter > 0 else ""
        tail = f"y_routed = {comb_expr}; y_shared = SharedFFN({prev}{shared_tag}); {out} = y_routed + y_shared"
    else:
        tail = f"{out} = {comb_expr}"
    return f"{disp_step}; {gmm1}; {act_step}; {gmm2}; {tail} [{shape_tag}, Persistent Megakernel]"


def _format_moe_gating_equation(
    prev: str,
    out: str,
    shape_tag: str,
    meta: Mapping[str, Any],
) -> str:
    """Render ``aclnnMoeGatingTopKSoftmaxV2`` strictly from logits -> ``(weights, ids)``."""
    raw_mode = str(meta.get("router_mode") or meta.get("score_function") or "").strip().lower()
    score_fn = raw_mode if raw_mode in {"softmax", "sigmoid"} else "softmax/sigmoid"
    noaux = bool(meta.get("noaux_tc"))
    groups = int(meta.get("groups", 0) or 0)
    group_topk = int(meta.get("group_topk", 0) or 0)
    top_k = int(meta.get("top_k", 0) or 0)
    has_bias = bool(meta.get("correction_bias") or meta.get("bias") or noaux)
    norm_topk = meta.get("norm_topk")
    scale = meta.get("routed_scaling_factor")
    sel_name = "group_limited_topk" if (noaux or groups > 1) else "TopK"
    score_expr = f"{score_fn}({prev})"
    if has_bias:
        score_expr = f"{score_expr} + e_bias"
    extras: list[str] = []
    if top_k > 0:
        extras.append(f"k={top_k}")
    if groups > 1:
        extras.append(f"groups={groups}")
    if group_topk > 0:
        extras.append(f"group_topk={group_topk}")
    if norm_topk is not None:
        extras.append(f"norm={bool(norm_topk)}")
    if scale is not None and float(scale) != 1.0:
        extras.append(f"scale={float(scale):g}")
    extra_txt = (", " + ", ".join(extras)) if extras else ""
    return f"{out}, topk_ids = {sel_name}({score_expr}{extra_txt}) [{shape_tag}, UB FP32]"


def _format_dag_equation(
    op_id: str,
    shape: Sequence[int],
    trait: OpTraitInfo,
    input_syms: Sequence[str],
    out_sym: str,
    *,
    op_meta: Mapping[str, Any] | None = None,
) -> str:
    prev = input_syms[0] if input_syms else "x"
    rhs_second = input_syms[1] if len(input_syms) > 1 else ""
    all_args = ", ".join(input_syms) if input_syms else "x"
    out = out_sym
    k = trait.symbolic_op_kind
    meta = op_meta or {}
    gs = trait.group_size or MXFP_BLOCK_GROUP
    shape_tag = "x".join(str(d) for d in shape) if shape else "?"
    param_txt = (
        "; " + ", ".join(f"{pk}={pv:g}" for pk, pv in trait.activation_params)
        if trait.activation_params else ""
    )
    w_tag = _strip_op_prefix(op_id)
    w_sym = f"W_{w_tag}" if w_tag else "W"
    if k == "megamoe":
        return _format_megamoe_equation(prev, out, shape_tag, param_txt, trait, meta)
    if k == "moe_router_topk":
        return _format_moe_gating_equation(prev, out, shape_tag, meta)
    eq_map = {
        "view_copy": f"{out} = strided_view({prev}) [zero-copy metadata elision]",
        "grouped_matmul": f"{out} = GroupedMatMul({prev}, {w_sym}_e^T) [{shape_tag}, L0C FP32]",
        "matmul": f"{out} = {prev} @ {w_sym}^T [{shape_tag}, L0C FP32]",
        "allgather": f"{out} = AllGather({prev}, axis={trait.collective_axis or 'tp'})",
        "reduce_scatter": f"{out} = ReduceScatter({prev}, SUM, axis={trait.collective_axis or 'tp'})",
        "all_gather_matmul": f"{out} = AllGather({prev}) @ {w_sym}_col^T [MC2 pipelined]",
        "matmul_reduce_scatter": f"{out} = ReduceScatter({prev} @ {w_sym}_row^T, SUM) [MC2 pipelined]",
        "add": f"{out} = {prev} + {rhs_second or 'r'} [UB FP32]",
        "mul": f"{out} = {prev} * {rhs_second or 'g'} [UB FP32]",
        "rmsnorm": f"{out} = ({prev} / sqrt(mean({prev}^2) + eps)) * gamma [UB FP32]",
        "layernorm": f"{out} = (({prev} - mu) / sqrt(var + eps)) * gamma + beta [UB FP32]",
        "add_rmsnorm": f"{out} = RMSNorm({prev} + {rhs_second or 'r'}; gamma) [UB FP32]",
        "add_rmsnorm_quant": f"{out}, s = DynamicQuant(RMSNorm({prev} + {rhs_second or 'r'}; gamma)) [UB FP32]",
        "rmsnorm_quant": f"{out}, s = DynamicQuant(RMSNorm({prev}; gamma)) [UB FP32]",
        "norm_gate": f"{out} = RMSNorm({prev}; gamma) * SiLU({rhs_second or 'z'}) [UB FP32]",
        "l2norm": f"{out} = {prev} / (||{prev}||_2 + eps) [UB FP32]",
        "swiglu": f"{out} = SiLU({prev}_gate) * {prev}_up [UB FP32]",
        "situ": f"{out} = SiTU({prev}_gate, {prev}_up{param_txt}) [UB FP32, custom ABI]",
        "silu": f"{out} = {prev} * sigmoid({prev}) [UB FP32]",
        "gelu": f"{out} = GELU({prev}) [UB FP32]",
        "sigmoid": f"{out} = sigmoid({prev}) [UB FP32]",
        "dynamic_mx_quant": (
            f"s_b = 2^floor(log2(max|{prev}_{{b:b+{gs}}}|/448)); "
            f"{out} = cast_fp8({prev}/s_b) [E8M0 g={gs}, UB FP32]"
        ),
        "dynamic_quant": f"s = max|{prev}| / qmax; {out} = cast_{trait.output_dtype}({prev}/s) [UB FP32]",
        "gating": f"{out} = -exp(A_log) * softplus({prev} + dt_bias) [UB FP32]",
        "conv1d": f"{out}, S_next = CausalConv1d({prev}, S_conv, W_conv) [rolling UB]",
        "recurrent_scan": f"{out}, H_next = RecurrentStateUpdate({prev}, H_prev) [FP32 state]",
        "recover_ssm": "H_commit = AliasSlotSelect(H_slots, accept_idx) [2D slot alias]",
        "rotary": f"{out} = RoPE({all_args}, cos, sin) [UB FP32]",
        "kv_scatter": f"KV_cache[slot] = QuantScatter({prev}) [in-place cache write]",
        "attn_res": f"{out} = Sum_i softmax_i(Norm(h_i) . q_l) * h_i [UB FP32]",
        "mhc_mix": f"{out} = SinkhornMix({prev}, H_pre, H_post) [UB FP32]",
        "attention_update": f"{out}, lse = OnlineSoftmaxMerge({prev}, lse_prev) [FP32]",
        "cast": f"{out} = cast_{trait.output_dtype}({prev})",
    }
    return eq_map.get(k, f"{out} = {k}({all_args}{param_txt}) [{shape_tag}]")


def _extract_op_operand_ports(op: OpSpec) -> list[tuple[str, str, int]]:
    """Extract ordered ``(producer_id, port_name, version)`` activation operands preserving multiplicity."""
    ports: list[tuple[str, str, int]] = []
    meta = dict(op.metadata) if op.metadata else {}
    for key in ("typed_inputs", "input_ports"):
        raw = meta.get(key, ())
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for entry in raw:
                if not isinstance(entry, Mapping):
                    continue
                if str(entry.get("role") or "").strip().lower() in {"weight", "control", "barrier", "event"}:
                    continue
                prod_id = entry.get("producer")
                if isinstance(prod_id, str) and prod_id and prod_id != op.op_id:
                    pname = str(entry.get("port") or entry.get("output") or entry.get("name") or "output")
                    ver = int(entry.get("version", entry.get("state_version", 0)) or 0)
                    ports.append((prod_id, "output" if pname == "out" else pname, ver))
        if ports:
            break
    if not ports:
        for dep, port in input_port_entries(op.metadata):
            if dep and dep != op.op_id:
                pname = port.removeprefix(dep + ":")
                if pname != "weight":
                    ports.append((dep, "output" if pname == "out" else pname, 0))
    explicit_producers = {p[0] for p in ports}
    for dep in op.deps:
        if dep and dep != op.op_id and dep not in explicit_producers:
            ports.append((dep, "output", 0))
    return ports


def _compose_dag_equations_and_ports(
    ops: Sequence[OpSpec],
    traits: Sequence[OpTraitInfo],
    *,
    preserve_port_ids: Sequence[str] = (),
) -> tuple[tuple[str, ...], tuple[FormulaTensorSpec, ...], tuple[FormulaTensorSpec, ...], tuple[str, ...]]:
    """Build composed symbolic equations, input/output tensor specs, and elided intermediates from true DAG."""
    ordered = tuple(ops)
    if not ordered:
        return (), (), (), ()
    member_ids = {op.op_id for op in ordered}
    preserve_set = set(preserve_port_ids)
    operand_ports: dict[str, tuple[tuple[str, str, int], ...]] = {}
    internal_users: dict[str, list[str]] = {op.op_id: [] for op in ordered}
    unique_ext: list[tuple[tuple[str, str, int], OpSpec, OpTraitInfo]] = []
    seen_ext: set[tuple[str, str, int]] = set()

    for op, tr in zip(ordered, traits):
        ports = tuple(_extract_op_operand_ports(op))
        if not ports:
            ports = ((f"__ext__:{op.op_id}", "output", 0),)
        operand_ports[op.op_id] = ports
        seen_int: set[str] = set()
        for prod_id, pname, ver in ports:
            if prod_id in member_ids:
                if prod_id not in seen_int:
                    internal_users[prod_id].append(op.op_id)
                    seen_int.add(prod_id)
            else:
                ek = (prod_id, pname, ver)
                if ek not in seen_ext:
                    seen_ext.add(ek)
                    unique_ext.append((ek, op, tr))

    ext_sym: dict[tuple[str, str, int], str] = {}
    in_specs: list[FormulaTensorSpec] = []
    for idx, (ek, op, tr) in enumerate(unique_ext):
        sym = "x" if len(unique_ext) == 1 else f"x{idx}"
        ext_sym[ek] = sym
        sh = _op_input_shape(op, tr)
        if sh:
            in_specs.append(FormulaTensorSpec(sym, sh, tr.input_dtype, "input"))

    live_out_ids = [
        op.op_id for op in ordered
        if (not internal_users[op.op_id]) or (op.op_id in preserve_set)
    ]
    if not live_out_ids:
        live_out_ids = [ordered[-1].op_id]
    live_out_set = set(live_out_ids)

    sym_by_id: dict[str, str] = {}
    out_specs: list[FormulaTensorSpec] = []
    temp_idx = 1
    out_idx = 0
    for op, tr in zip(ordered, traits):
        if op.op_id in live_out_set:
            sym = "y" if len(live_out_ids) == 1 else f"y{out_idx}"
            out_idx += 1
            sym_by_id[op.op_id] = sym
            sh = _op_output_shape(op, tr)
            if sh:
                out_specs.append(FormulaTensorSpec(sym, sh, tr.output_dtype, "output"))
        else:
            sym_by_id[op.op_id] = f"t{temp_idx}"
            temp_idx += 1

    equations: list[str] = []
    for op, tr in zip(ordered, traits):
        in_syms_list: list[str] = []
        for prod_id, pname, ver in operand_ports[op.op_id]:
            if prod_id in member_ids:
                base_sym = sym_by_id[prod_id]
                if pname != "output" and ver:
                    in_syms_list.append(f"{base_sym}.{pname}@v{ver}")
                elif pname != "output":
                    in_syms_list.append(f"{base_sym}.{pname}")
                elif ver:
                    in_syms_list.append(f"{base_sym}@v{ver}")
                else:
                    in_syms_list.append(base_sym)
            else:
                in_syms_list.append(ext_sym[(prod_id, pname, ver)])
        equations.append(_format_dag_equation(
            op.op_id, op.local_shape, tr, tuple(in_syms_list), sym_by_id[op.op_id],
            op_meta=dict(op.metadata) if op.metadata else None,
        ))
    elided_ids = tuple(
        op.op_id for op in ordered
        if op.op_id not in live_out_set and internal_users[op.op_id]
    )
    return tuple(equations), tuple(in_specs), tuple(out_specs), elided_ids


def _condense_pipe_nodes(
    elem_nodes: Sequence[tuple[str, Literal["cube", "vector", "link"], float, bool, tuple[str, ...]]],
) -> tuple[PipeStageSpec, ...]:
    """Condense elementary pipe DAG without merging nodes whose dependency/consumer sets differ."""
    if not elem_nodes:
        return ()
    node_ids = {eid for eid, _, _, _, _ in elem_nodes}
    preds_by_id: dict[str, tuple[str, ...]] = {
        eid: tuple(dict.fromkeys(p for p in preds if p in node_ids and p != eid))
        for eid, _, _, _, preds in elem_nodes
    }
    succs_by_id: dict[str, list[str]] = {eid: [] for eid in node_ids}
    for eid, plist in preds_by_id.items():
        for p in plist:
            succs_by_id[p].append(eid)

    level_by_id: dict[str, int] = {}
    visiting: set[str] = set()

    def _level(eid: str) -> int:
        if eid in level_by_id:
            return level_by_id[eid]
        if eid in visiting:
            return 0
        visiting.add(eid)
        lvl = max((_level(p) + 1 for p in preds_by_id[eid]), default=0)
        visiting.discard(eid)
        level_by_id[eid] = lvl
        return lvl

    for eid in preds_by_id:
        _level(eid)

    GroupKey = tuple[int, Literal["cube", "vector", "link"], bool, frozenset[str], frozenset[str]]
    group_key_by_id: dict[str, GroupKey] = {}
    group_first_pos: dict[GroupKey, int] = {}
    group_order: list[GroupKey] = []
    group_work: dict[GroupKey, float] = defaultdict(float)
    group_red: dict[GroupKey, bool] = defaultdict(bool)
    group_preds: dict[GroupKey, list[GroupKey]] = defaultdict(list)

    for pos, (eid, pkind, _work, is_red, _preds) in enumerate(elem_nodes):
        gk: GroupKey = (
            level_by_id[eid],
            pkind,
            bool(is_red) if pkind == "vector" else False,
            frozenset(preds_by_id[eid]),
            frozenset(succs_by_id[eid]),
        )
        group_key_by_id[eid] = gk
        if gk not in group_first_pos:
            group_first_pos[gk] = pos
            group_order.append(gk)

    for eid, _pkind, work, is_red, _preds in elem_nodes:
        gk = group_key_by_id[eid]
        group_work[gk] += max(0.0, float(work))
        if is_red:
            group_red[gk] = True
        for p in preds_by_id[eid]:
            pgk = group_key_by_id[p]
            if pgk != gk and pgk not in group_preds[gk]:
                group_preds[gk].append(pgk)

    group_order.sort(key=lambda item: (item[0], {"link": 0, "vector": 1, "cube": 2}[item[1]], group_first_pos[item]))
    counts_by_kind: dict[str, int] = defaultdict(int)
    for gk in group_order:
        counts_by_kind[gk[1]] += 1

    base_names = {"cube": "cube_mma", "vector": "aiv_fp32", "link": "link_pipe"}
    seen_kind: dict[str, int] = defaultdict(int)
    stage_name: dict[GroupKey, str] = {}
    totals_by_kind: dict[str, float] = defaultdict(float)
    for gk in group_order:
        pkind = gk[1]
        idx = seen_kind[pkind]
        seen_kind[pkind] += 1
        stage_name[gk] = base_names[pkind] if counts_by_kind[pkind] == 1 else f"{base_names[pkind]}_{idx}"
        totals_by_kind[pkind] += group_work[gk]

    stages: list[PipeStageSpec] = []
    for gk in group_order:
        pkind = gk[1]
        tot = totals_by_kind[pkind]
        share = (group_work[gk] / tot) if tot > 0.0 else (1.0 / float(counts_by_kind[pkind]))
        after = tuple(stage_name[pgk] for pgk in group_preds[gk] if pgk in stage_name)
        stages.append(PipeStageSpec(
            name=stage_name[gk],
            pipe_kind=pkind,
            share=share,
            after=after,
            row_reduction=bool(group_red[gk]) if pkind == "vector" else False,
        ))
    return tuple(stages)


def _derive_pipe_stages_from_ops(
    ops: Sequence[OpSpec], traits: Sequence[OpTraitInfo],
) -> tuple[PipeStageSpec, ...]:
    """Derive causal ``PipeStageSpec`` DAG strictly from true operand dependencies."""
    ordered = tuple(ops)
    if not ordered:
        return ()
    member_ids = {op.op_id for op in ordered}
    elem_nodes: list[tuple[str, Literal["cube", "vector", "link"], float, bool, tuple[str, ...]]] = []
    exit_eid: dict[str, tuple[str, ...]] = {}

    for op, tr in zip(ordered, traits):
        ports = _extract_op_operand_ports(op)
        int_preds = tuple(dict.fromkeys(p[0] for p in ports if p[0] in member_ids))
        pred_eids = tuple(dict.fromkeys(eid for pid in int_preds for eid in exit_eid.get(pid, ())))
        fam = str(op.family or "").strip().lower()
        cube_w = sum(float(fl) for _, fl in op.matrix_flops) if op.matrix_flops else float(op.flops or 0.0)
        vec_w = float(op.vector_ops or 0.0)
        link_w = float(op.link_bytes or 0.0)
        is_red = tr.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE}

        if fam == "all_gather_matmul":
            l_id, c_id = f"{op.op_id}:link", f"{op.op_id}:cube"
            elem_nodes.append((l_id, "link", max(link_w, 1.0), False, pred_eids))
            elem_nodes.append((c_id, "cube", max(cube_w, 1.0), False, (l_id,)))
            exit_eid[op.op_id] = (c_id,)
            continue
        if fam == "matmul_reduce_scatter":
            c_id, l_id = f"{op.op_id}:cube", f"{op.op_id}:link"
            elem_nodes.append((c_id, "cube", max(cube_w, 1.0), False, pred_eids))
            elem_nodes.append((l_id, "link", max(link_w, 1.0), False, (c_id,)))
            exit_eid[op.op_id] = (l_id,)
            continue

        local_ids: list[str] = []
        if cube_w > 0.0 or tr.trait == AlgebraicTrait.BILINEAR_MATMUL:
            cid = f"{op.op_id}:cube"
            elem_nodes.append((cid, "cube", max(cube_w, 1.0), False, pred_eids))
            local_ids.append(cid)
        if vec_w > 0.0 or (not local_ids and tr.trait in {AlgebraicTrait.POINTWISE, AlgebraicTrait.ROW_REDUCTION}):
            vid = f"{op.op_id}:vec"
            v_preds = tuple(local_ids) if local_ids else pred_eids
            elem_nodes.append((vid, "vector", max(vec_w, 1.0), is_red, v_preds))
            local_ids = [vid]
        if link_w > 0.0 or (not local_ids and tr.trait == AlgebraicTrait.ASSOCIATIVE_COLLECTIVE):
            lid = f"{op.op_id}:link"
            l_preds = tuple(local_ids) if local_ids else pred_eids
            elem_nodes.append((lid, "link", max(link_w, 1.0), False, l_preds))
            local_ids = [lid]
        exit_eid[op.op_id] = tuple(local_ids) if local_ids else pred_eids

    return _condense_pipe_nodes(elem_nodes)


def _derive_pipe_stages_from_equations(
    equations: Sequence[str],
    *,
    has_cube: bool,
    has_vector: bool,
    has_link: bool,
    row_reduction: bool,
) -> tuple[PipeStageSpec, ...]:
    """Infer causal ``PipeStageSpec`` DAG from declarative ``FormulaKernelSpec.equations``."""
    elem_nodes: list[tuple[str, Literal["cube", "vector", "link"], float, bool, tuple[str, ...]]] = []
    producer_by_sym: dict[str, str] = {}
    for idx, eq in enumerate(equations):
        lhs, sep, rhs = str(eq).partition("=")
        rhs_txt = rhs if sep else lhs
        rhs_low = rhs_txt.lower()
        if any(tok in rhs_low for tok in ("allgather", "reducescatter", "allreduce", "alltoall")):
            pkind: Literal["cube", "vector", "link"] = "link"
        elif "@" in rhs_txt or "matmul" in rhs_low:
            pkind = "cube"
        else:
            pkind = "vector"
        if (pkind == "cube" and not has_cube) or (pkind == "vector" and not has_vector) or (pkind == "link" and not has_link):
            continue
        rhs_idents = _EQUATION_IDENT_RE.findall(rhs_txt)
        preds = tuple(dict.fromkeys(producer_by_sym[tok] for tok in rhs_idents if tok in producer_by_sym))
        eid = f"eq:{idx}"
        is_red = pkind == "vector" and (row_reduction or any(t in rhs_low for t in ("norm", "softmax", "quant")))
        elem_nodes.append((eid, pkind, 1.0, is_red, preds))
        if sep:
            for sym in _EQUATION_IDENT_RE.findall(lhs):
                producer_by_sym[sym] = eid
    return _condense_pipe_nodes(elem_nodes)


def _verify_sram_residency(
    members: Sequence[TargetOpNode],
    traits: Sequence[OpTraitInfo],
    datapath: DatapathCapabilitySpec | None,
) -> tuple[str, str, int, int, int, int, int, int, bool]:
    """Run ``TileSolver`` against Ascend UB (FP32 4B upcast + SIMT carveout), L1, and L0."""
    if datapath is None:
        return "unbounded", "(M=1, N=1, K=1)", 0, 0, 0, 0, 0, 0, False
    bufs = dict(datapath.buffers)
    carveout = max(SIMT_UB_CARVEOUT_BYTES, int(datapath.simt_ub_carveout_bytes or 0))
    ub_raw = int(bufs.get("ub", 0) or 0)
    ub_budget = max(0, ub_raw - carveout)
    l1_budget = int(bufs.get("l1", 0) or 0)
    l0_budget = int(bufs.get("l0c", 0) or 0)

    has_cube = any(t.trait == AlgebraicTrait.BILINEAR_MATMUL for t in traits)
    has_aiv = any(t.trait != AlgebraicTrait.BILINEAR_MATMUL for t in traits)
    max_row_ub = (
        max((t.row_working_bytes for t in traits if t.trait != AlgebraicTrait.BILINEAR_MATMUL and t.row_working_bytes > 0), default=0)
        if has_aiv else 0
    )
    if has_aiv and max_row_ub == 0:
        row_elems = max((int(m.shape[-1]) for m in members if m.shape), default=0)
        max_row_ub = row_elems * FP32_ACCUM_BYTES * UB_DOUBLE_BUFFER_SLOTS
    l1_tile = l0_tile = 0
    if has_cube:
        a_b = max((_safe_elem_bytes(t.input_dtype) for t in traits if t.trait == AlgebraicTrait.BILINEAR_MATMUL), default=2.0)
        w_b = max((_safe_elem_bytes(t.weight_dtype or t.input_dtype) for t in traits if t.trait == AlgebraicTrait.BILINEAR_MATMUL), default=1.0)
        l1_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_K_ELEMS * (a_b + w_b) * UB_DOUBLE_BUFFER_SLOTS)
        l0_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_MN_ELEMS * L0C_ACCUM_BYTES)
    if (has_aiv and (max_row_ub <= 0 or ub_budget <= 0)) or (not has_aiv and not has_cube):
        return "ub_infeasible", "(M=1, N=1, K=1)", max_row_ub, ub_budget, l1_tile, l1_budget, l0_tile, l0_budget, False

    hw = SimpleNamespace(
        datapath=datapath,
        buffers=bufs,
        buffer_units={"ub": datapath.aiv_count, "l1": datapath.aic_count, "l0c": datapath.aic_count, "l2": 1},
        stream_owners=None,
    )
    cands: list[TileCandidate] = []
    has_simd_ub_headroom = bool(has_aiv and int(datapath.simt_ub_carveout_bytes or 0) > 0 and ub_raw > ub_budget)
    cand_modes: tuple[tuple[int, bool], ...] = (
        ((UB_DOUBLE_BUFFER_SLOTS, True), (1, True), (1, False))
        if has_simd_ub_headroom else ((UB_DOUBLE_BUFFER_SLOTS, has_aiv), (1, has_aiv))
    )
    for slots, cand_simt in cand_modes:
        live_bufs: list[LiveBufferSpec] = []
        if has_aiv and max_row_ub > 0:
            live_bufs.append(LiveBufferSpec("ub_fp32_work", "ub", max_row_ub, 0, 1, owner_kind="aiv", slots=slots))
        if has_cube and l1_tile > 0:
            live_bufs.append(LiveBufferSpec("l1_cube_tile", "l1", l1_tile, 0, 1, owner_kind="aic", slots=1))
        if has_cube and l0_tile > 0:
            live_bufs.append(LiveBufferSpec("l0c_fp32_accum", "l0c", l0_tile, 0, 1, owner_kind="aic", slots=1))
        algo: Literal["resident_1pass", "online_reduce"] = (
            "resident_1pass" if slots == UB_DOUBLE_BUFFER_SLOTS else "online_reduce"
        )
        tile_n = max(1, max_row_ub // FP32_ACCUM_BYTES) if has_aiv else CUBE_TILE_MN_ELEMS
        cands.append(TileCandidate(
            algorithm=algo,
            tile_shape=TileShape(1 if has_aiv else CUBE_TILE_MN_ELEMS, tile_n, CUBE_TILE_K_ELEMS if has_cube else 1),
            depth=slots,
            owner_partition=(datapath.aic_count, datapath.aiv_count),
            work_items=1,
            tiles_per_owner=1,
            buffers=tuple(live_bufs),
            uses_simt=cand_simt,
            owner_axis="aiv" if has_aiv else "aic",
        ))
    max_ub_budget = ub_raw if has_simd_ub_headroom else ub_budget
    try:
        solved = tile_solver.solve(
            TilePipelineSpec("kernel_synthesis", tuple(cands), simt_ub_carveout_bytes=carveout),
            None,
            hw,
        )
        peaks = {p.pool: int(p.peak_bytes) for p in solved.pool_peaks}
        caps = {p.pool: int(p.capacity_bytes) for p in solved.pool_peaks}
        ub_pk = peaks.get("ub", max_row_ub) if has_aiv else 0
        eff_ub_budget = caps.get("ub", ub_budget) if has_aiv else ub_budget
        l1_pk = peaks.get("l1", l1_tile) if has_cube else 0
        l0_pk = peaks.get("l0c", l0_tile) if has_cube else 0
        ok = (
            (not has_aiv or ub_pk <= eff_ub_budget)
            and (not has_cube or (l1_pk <= l1_budget and l0_pk <= l0_budget))
        )
        sh = solved.tile_shape
        return (
            solved.algorithm, f"(M={sh.m}, N={sh.n}, K={sh.k})",
            ub_pk, eff_ub_budget, l1_pk, l1_budget, l0_pk, l0_budget, ok,
        )
    except tile_solver.TileInfeasibleError:
        fail_algo = "ub_infeasible" if (has_aiv and max_row_ub > max_ub_budget) else "sram_infeasible"
        return (
            fail_algo, f"(M=1, N={max(1, max_row_ub // FP32_ACCUM_BYTES)}, K=1)",
            max_row_ub, max_ub_budget, l1_tile, l1_budget, l0_tile, l0_budget, False,
        )


def synthesize_kernel_spec(
    *,
    spec_id: str,
    kernel_name: str,
    abi_tier: str,
    suggested_backend: str,
    status_tag: str,
    opt_pillar: str,
    members: Sequence[TargetOpNode],
    member_traits: Sequence[OpTraitInfo],
    datapath: DatapathCapabilitySpec | None,
    elided_hbm_bytes: int,
    l_star_us: float,
    before_us: float,
    target_us: float,
    eta_target: float = 1.0,
    abi_signature_id: str = "",
    binding_state: BindingState = "unbound_formula_projection",
    preserve_port_ids: Sequence[str] = (),
    exposed_before_us: float | None = None,
) -> KernelSynthesisSpec:
    """Compose symbolic math equations and TileSolver FP32 SRAM residency from true operand DAG."""
    member_ops = tuple(_node_to_op_spec(m, None) for m in members)
    equations, in_specs, out_specs, elided_ids = _compose_dag_equations_and_ports(
        member_ops, member_traits, preserve_port_ids=preserve_port_ids,
    )
    in_t = tuple(f"{t.name}[{t.dtype}, {list(t.shape)}]" for t in in_specs) or (
        f"{_strip_op_prefix(members[0].op_id)}.in[{members[0].dtype}, {list(members[0].shape)}]",
    )
    out_t = tuple(f"{t.name}[{t.dtype}, {list(t.shape)}]" for t in out_specs) or (
        f"{_strip_op_prefix(members[-1].op_id)}.out[{members[-1].dtype}, {list(members[-1].shape)}]",
    )
    elided_names = tuple(_strip_op_prefix(oid) for oid in elided_ids)
    algo, t_shape, ub_pk, ub_cap, l1_pk, l1_cap, l0_pk, l0_cap, ok = _verify_sram_residency(
        members, member_traits, datapath,
    )
    ub_warn: str | None = None
    if datapath is not None and any(tr.trait != AlgebraicTrait.BILINEAR_MATMUL for tr in member_traits):
        tl_cal = try_load_tilelang_calibration(str(datapath.arch_id))
        if tl_cal is not None:
            ub_warn = tl_cal.simt_vs_simd_ub_wall_warning(ub_pk)
    achieved = (
        min(1.0, l_star_us / target_us)
        if target_us > 0.0 and l_star_us > 0.0 else min(1.0, max(0.0, float(eta_target)))
    )
    return KernelSynthesisSpec(
        spec_id=spec_id,
        kernel_name=kernel_name,
        abi_tier=abi_tier,
        suggested_backend=suggested_backend,
        status_tag=status_tag,
        opt_pillar=opt_pillar,
        constituent_op_ids=tuple(m.op_id for m in members),
        constituent_traits=tuple(tr.trait.value for tr in member_traits),
        symbolic_equations=equations,
        input_tensors=in_t,
        output_tensors=out_t,
        elided_intermediates=elided_names,
        elided_hbm_bytes=max(0, int(elided_hbm_bytes)),
        tile_algorithm=algo,
        tile_shape=t_shape,
        ub_precision_mode="fp32_upcast",
        ub_peak_bytes=ub_pk,
        ub_budget_bytes=ub_cap,
        l1_peak_bytes=l1_pk,
        l1_budget_bytes=l1_cap,
        l0_peak_bytes=l0_pk,
        l0_budget_bytes=l0_cap,
        sram_verified=ok,
        l_star_us=l_star_us,
        before_us=before_us,
        target_us=target_us,
        target_attainment_pct=round(achieved * PERCENT_SCALE, 1),
        abi_signature_id=abi_signature_id,
        binding_state=binding_state,
        exposed_before_us=exposed_before_us,
        ub_wall_warning=ub_warn,
    )


def load_rewrite_plan(
    rewrite_plan: RewritePlan | Mapping[str, Any] | str | Path | None,
) -> RewritePlan | None:
    """Decode an optional Agent-supplied ``RewritePlan`` from object, dict, or JSON path."""
    if rewrite_plan is None:
        return None
    if isinstance(rewrite_plan, RewritePlan):
        return rewrite_plan
    if isinstance(rewrite_plan, (str, Path)):
        raw = json.loads(Path(rewrite_plan).read_text(encoding="utf-8"))
        return decode_rewrite_plan(raw)
    if isinstance(rewrite_plan, Mapping):
        return decode_rewrite_plan(rewrite_plan)
    raise ModelingError(f"unsupported_rewrite_plan_type:{type(rewrite_plan).__name__}")


def _node_to_op_spec(
    node: TargetOpNode,
    known_ids: set[str] | None = None,
    *,
    phase: str = "decode",
) -> OpSpec:
    deps = tuple(dict.fromkeys(
        p.op_id for p in node.predecessors
        if p.op_id != node.op_id and (known_ids is None or p.op_id in known_ids)
    ))
    if node.spec is not None:
        return replace(node.spec, op_id=node.op_id, deps=deps)
    matrix_flops = ((node.dtype, float(node.flops)),) if node.flops > 0.0 else ()
    return OpSpec(
        op_id=node.op_id,
        family=node.family,
        stage=node.occurrence.stage,
        layer=0,
        phase=phase,
        requests=(),
        semantic_shape=node.shape,
        local_shape=node.shape,
        padded_shape=node.shape,
        launch_shape=node.shape,
        recipe=node.recipe,
        deps=deps,
        stream=node.stream,
        dtype=node.dtype,
        flops=node.flops,
        matrix_flops=matrix_flops,
        vector_ops=node.vector_ops,
        hbm_bytes=node.hbm_total_bytes,
        mte2_bytes=node.mte2_bytes,
        mte3_bytes=node.mte3_bytes,
        link_bytes=node.link_bytes,
    )


class _TargetPricingContext:
    """Lazily constructed canonical ``RecipeBook`` + ``EvaluationContext`` + ``Topology``."""

    __slots__ = ("_case", "_primitive_table", "_book", "_eval_ctx")

    def __init__(self, case: Case, primitive_table: Any = None) -> None:
        self._case = case
        self._primitive_table = primitive_table
        self._book: RecipeBook | None = None
        self._eval_ctx: EvaluationContext | None = None

    def ensure(self) -> tuple[RecipeBook, EvaluationContext, Topology, bool]:
        if self._book is None or self._eval_ctx is None:
            self._book = RecipeBook.from_mapping(self._case.runtime.kernels)
            self._eval_ctx = EvaluationContext.create(
                self._case, self._book, primitive_table=self._primitive_table,
            )
        has_measured = isinstance(self._eval_ctx.cost.primitives, PrimitiveStore)
        return self._book, self._eval_ctx, self._eval_ctx.bindings.topology, has_measured


def _build_target_scenario(
    case: Case,
    arch_ops: Sequence[OpSpec],
    datapath: DatapathCapabilitySpec,
    *,
    phase: str = "decode",
    primitive_table: Any = None,
) -> Any:
    """Build a ``ResolvedScenario`` with canonical digests derived strictly from ``(case, arch_ops, step)``."""
    from mindie_sim.rewrite.apply import ResolvedScenario

    step = StepRequest(phase=phase, tokens=1, context=1)
    atomic_ops = (RankOps(0, tuple(arch_ops)),)
    base_sha = _json_digest(atomic_ops, default=_digest_default)
    framing_sha = _json_digest(
        (
            case.to_dict(),
            case.runtime.to_dict(),
            asdict(step),
            "kernel_synthesis",
            primitive_table_digest_token(primitive_table),
        ),
        default=_digest_default,
    )
    scenario_sha = _json_digest((base_sha, framing_sha), default=_digest_default)
    return ResolvedScenario(
        case=case,
        step=step,
        stack_id="kernel_synthesis",
        base_graph_sha256=base_sha,
        framing_sha256=framing_sha,
        scenario_sha256=scenario_sha,
        atomic_step=SimpleNamespace(case=case),  # type: ignore[arg-type]
        mainline_step=None,
        atomic_ops=atomic_ops,
        datapath=datapath,
        primitive_table=primitive_table,
    )


def _validate_supported_plan_opcodes(plan: RewritePlan) -> None:
    """Explicitly reject opcodes unsupported by ``solve_target_graph``."""
    for edit in plan.edits:
        if not isinstance(edit.operation, _SUPPORTED_TARGET_OPCODES):
            raise ModelingError(
                f"REWRITE_PLAN_UNSUPPORTED_OPCODE:{edit.edit_id}:{type(edit.operation).__name__}"
            )


def _verify_external_plan_digests(
    ext_plan: RewritePlan,
    case: Case,
    base_scenario: Any,
    *,
    step_request: StepRequest | None = None,
    stack_id: str | None = None,
    primitive_table: Any = None,
) -> bool:
    """Verify caller-supplied plan digests against target scenario or resolved public scenario.

    Returns ``True`` when the plan matched a public ``resolve_scenario`` digest (so
    downstream archetype application may rebind digests onto the archetype scenario),
    and ``False`` when the plan is unsealed or directly matched ``base_scenario``.
    """
    has_base = bool(ext_plan.base_graph_sha256)
    has_frame = bool(ext_plan.framing_sha256)
    if not has_base and not has_frame:
        return False
    base_match = (not has_base) or (ext_plan.base_graph_sha256 == base_scenario.base_graph_sha256)
    frame_match = (not has_frame) or (ext_plan.framing_sha256 == base_scenario.framing_sha256)
    if base_match and frame_match:
        return False
    if step_request is not None:
        from mindie_sim.rewrite.apply import DEFAULT_STACK_ID, resolve_scenario

        eff_stack = stack_id or DEFAULT_STACK_ID
        alt_scope = "body" if step_request.scope == "model" else "model"
        candidates = (step_request, replace(step_request, scope=alt_scope))
        last_base_ok = False
        for cand_step in candidates:
            resolved = resolve_scenario(
                case, cand_step, stack_id=eff_stack, primitive_table=primitive_table,
            )
            valid_bases = {resolved.base_graph_sha256}
            if resolved.mainline_step is not None:
                valid_bases.add(_json_digest(
                    ("mainline", resolved.mainline_step.graph.executable, resolved.mainline_step.case.runtime.to_dict()),
                    default=_digest_default,
                ))
            pub_base_ok = (not has_base) or (ext_plan.base_graph_sha256 in valid_bases)
            pub_frame_ok = (not has_frame) or (ext_plan.framing_sha256 == resolved.framing_sha256)
            if pub_base_ok and pub_frame_ok:
                return True
            last_base_ok = last_base_ok or pub_base_ok
        if not last_base_ok:
            raise ModelingError("rewrite_digest_mismatch:base_graph_sha256")
        raise ModelingError("rewrite_digest_mismatch:framing_sha256")
    if not base_match:
        raise ModelingError("rewrite_digest_mismatch:base_graph_sha256")
    raise ModelingError("rewrite_digest_mismatch:framing_sha256")


def _apply_and_verify_plan(
    case: Case,
    arch_ops: Sequence[OpSpec],
    plan: RewritePlan,
    datapath: DatapathCapabilitySpec,
    *,
    phase: str = "decode",
    primitive_table: Any = None,
    bind_empty_digests: bool = False,
    rebind_digests: bool = False,
) -> tuple[Any, Any, Any]:
    """Run canonical ``apply_plan`` + ``verify_plan`` and return ``(scenario, applied, receipt)``."""
    from mindie_sim.rewrite.apply import apply_plan
    from mindie_sim.rewrite.verify import verify_plan

    _validate_supported_plan_opcodes(plan)
    scenario = _build_target_scenario(
        case, arch_ops, datapath, phase=phase, primitive_table=primitive_table,
    )
    if rebind_digests:
        eff_plan = replace(
            plan,
            base_graph_sha256=scenario.base_graph_sha256,
            framing_sha256=scenario.framing_sha256,
        )
    elif bind_empty_digests:
        eff_plan = replace(
            plan,
            base_graph_sha256=plan.base_graph_sha256 or scenario.base_graph_sha256,
            framing_sha256=plan.framing_sha256 or scenario.framing_sha256,
        )
    else:
        eff_plan = plan
    applied = apply_plan(eff_plan, scenario)
    receipt = verify_plan(scenario, applied, scope="archetype")
    return scenario, applied, receipt


def _has_non_transport_work(op_spec: OpSpec) -> bool:
    """Return True if ``op_spec`` carries any local non-link compute or HBM/MTE traffic."""
    return bool(
        float(op_spec.flops or 0.0) > 0.0
        or any(float(fl) > 0.0 for _, fl in (op_spec.matrix_flops or ()))
        or float(op_spec.vector_ops or 0.0) > 0.0
        or float(op_spec.hbm_bytes or 0.0) > 0.0
        or float(op_spec.mte2_bytes or 0.0) > 0.0
        or float(op_spec.mte3_bytes or 0.0) > 0.0
    )


def _build_collective_plan(
    op_spec: OpSpec,
    pricing_ctx: _TargetPricingContext,
    case: Case,
    *,
    mode: str = "oneshot",
    topology_route: str | None = None,
) -> tuple[Topology, OpSpec, Any] | None:
    """Validate transport-only work & route and construct the bound ``Topology`` collective plan."""
    _, _, topology, _ = pricing_ctx.ensure()
    if _has_non_transport_work(op_spec):
        if topology_route is not None:
            raise ModelingError(f"{mode}_collective_requires_transport_only_work:{op_spec.op_id}")
        return None
    fam_raw = str(op_spec.family or "").strip().lower()
    norm_fam = _COLLECTIVE_FAMILY_ALIASES.get(fam_raw, fam_raw)
    meta = dict(op_spec.metadata) if op_spec.metadata else {}
    meta.pop("steps", None)
    if not meta.get("group_axis"):
        scope_l = str(op_spec.link_scope or "").strip().lower()
        meta["group_axis"] = scope_l if scope_l in {"tp", "dp", "ep", "cp", "pp", "sp"} else "tp"
    op_norm = replace(op_spec, family=norm_fam, metadata=tuple(meta.items()))

    if topology_route is not None:
        valid_modes = {"ring", "pairwise", "oneshot"}
        groups = topology.groups(
            meta.get("group_axis", "tp"), 0,
            int(meta.get("source_stage", op_norm.stage)), metadata=meta,
        )
        pairs = (
            (a, b) for group in groups for a, b in (
                zip(group, group[1:] + group[:1]) if mode == "ring"
                else ((a, b) for a in group for b in group)
            ) if a != b
        )
        if (
            mode not in valid_modes
            or topology_route not in topology.links
            or any(not any(name == topology_route for name, _, _ in topology.route(a, b)) for a, b in pairs)
        ):
            raise ModelingError(f"rewrite_collective_route_recipe_unbound:{op_spec.op_id}")

    if op_norm.link_bytes <= 0.0:
        return (topology, op_norm, None) if topology_route is not None else None

    plan_fn = {
        "oneshot": topology.oneshot_plan,
        "ring": topology.ring_plan,
        "pairwise": topology.pairwise_plan,
    }.get(mode)
    if plan_fn is None:
        if topology_route is not None:
            raise ModelingError(f"rewrite_collective_route_recipe_unbound:{op_spec.op_id}")
        return None

    try:
        plan = plan_fn(op_norm, 0)
    except ModelingError as exc:
        msg = str(exc)
        if "complete_local_group" in msg or "joint_collective_group_membership" in msg:
            dp_count = max(1, int(case.runtime.parallel.dp))
            participants = {r: float(op_norm.link_bytes) for r in range(dp_count)}
            try:
                plan = topology.joint_plan(op_norm, participants, algorithm=mode)
            except ModelingError:
                if topology_route is not None:
                    raise
                return None
        elif topology_route is not None:
            raise
        else:
            return None

    if not _collective_plan_has_work(plan, mode):
        return (topology, op_norm, None) if topology_route is not None else None
    if mode == "oneshot":
        plan.validate_path_model(OneShotCollective().path_model)
    return topology, op_norm, plan


def _collective_plan_has_work(plan: Any, mode: str) -> bool:
    """Whether ``plan`` carries wire work in the representation its algorithm uses.

    ``PairwiseGroup`` keeps routed traffic in ``volumes`` and materialises each
    message lazily when its round starts, so ``messages`` is never populated;
    ring / one-shot groups retain their routed ``messages`` eagerly.
    """
    groups = getattr(plan, "groups", ()) or ()
    if mode == "pairwise":
        return any(
            int(getattr(g, "rounds", 0) or 0) > 0
            and any(float(n) > 0.0 for _, n in getattr(g, "volumes", ()))
            for g in groups
        )
    return any(getattr(g, "messages", ()) for g in groups)


def _price_collective_on_topology(
    op_spec: OpSpec,
    pricing_ctx: _TargetPricingContext,
    case: Case,
    *,
    mode: str = "oneshot",
    topology_route: str | None = None,
    minimum_us: float = 0.0,
) -> float | None:
    """Price ``OneShotCollective`` / ``RingCollective`` / ``PairwiseCollective`` on true ``Topology``."""
    built = _build_collective_plan(
        op_spec, pricing_ctx, case, mode=mode, topology_route=topology_route,
    )
    if built is None:
        return None
    topology, op_norm, plan = built
    if plan is None:
        return float(minimum_us)
    engine = Engine(topology.resources_for_ranks(plan.ranks), backend="python")
    ready = engine.submit("ready")
    done = plan.start(engine, op_norm.op_id, ready)
    engine.advance(until=(done,))
    return max(float(done.time) * US_PER_MS, float(minimum_us))


def _price_bound_op_on_engine(
    op_spec: OpSpec,
    pricing_ctx: _TargetPricingContext,
    case: Case,
    *,
    scenario: Any = None,
    atomic_entry: Any = None,
    template_binding: Any = None,
    applied: Any = None,
) -> tuple[float, BindingState, str] | None:
    """Price strictly bound implementations (``RecipeBook``/Catalog/Extension) via ``Engine``."""
    from mindie_sim.optimization.dsl_solver import catalog_recipe_id, replay_catalog_binding

    book, eval_ctx, _, has_measured = pricing_ctx.ensure()
    bound_book = book
    target_op: OpSpec | None = None

    is_law2_gemm = (
        op_spec.family == "rewrite_fused"
        and dict(op_spec.metadata).get("law2_block_gemm") is True
        and len(op_spec.local_shape) == MATMUL_SHAPE_RANK
        and op_spec.flops > 0.0
    )
    if op_spec.family == "rewrite_fused" and applied is not None:
        f_edit = next((e for e in applied.plan.edits if op_spec.op_id == f"rewrite:{e.edit_id}"), None)
        if f_edit is None or not isinstance(f_edit.operation, FuseRegion) or len(f_edit.operation.kernel_boundaries) != 1:
            return None

    if template_binding is not None:
        cat = (
            replay_catalog_binding(scenario, atomic_entry, op_spec, template_binding, applied, None)
            if (scenario is not None and atomic_entry is not None and applied is not None)
            else None
        )
        if cat is not None:
            rec_id = catalog_recipe_id(0, op_spec, template_binding, cat[1])
            bound_book = register_recipe(book, rec_id, cat[0])
            target_op = replace(op_spec, recipe=rec_id)
        elif template_binding.template_id in book.recipes:
            target_op = replace(op_spec, recipe=template_binding.template_id)
        elif template_binding.template_id == "gemm" and is_law2_gemm:
            gemm_candidate = replace(op_spec, family="gemm", recipe="gemm")
            if not isinstance(book.select(gemm_candidate), FusedPipelineRecipe):
                target_op = gemm_candidate
        else:
            raise ModelingError(
                f"rewrite_template_recipe_unbound:{op_spec.op_id}:{template_binding.template_id}"
            )
    elif op_spec.family == "rewrite_fused":
        cat = (
            replay_catalog_binding(scenario, atomic_entry, op_spec, None, applied, None)
            if (scenario is not None and atomic_entry is not None and applied is not None)
            else None
        )
        if cat is not None:
            rec_id = catalog_recipe_id(0, op_spec, None, cat[1])
            bound_book = register_recipe(book, rec_id, cat[0])
            target_op = replace(op_spec, recipe=rec_id)
        elif op_spec.recipe in book.recipes:
            target_op = op_spec
        elif op_spec.recipe == "gemm" and is_law2_gemm:
            gemm_candidate = replace(op_spec, family="gemm", recipe="gemm")
            if not isinstance(book.select(gemm_candidate), FusedPipelineRecipe):
                target_op = gemm_candidate
    elif op_spec.recipe in book.recipes:
        target_op = op_spec

    if target_op is None:
        return None
    report = operator_costs(case, (target_op,), recipes=bound_book, evaluation=eval_ctx)
    rows = report.get("operators", ())
    if not rows:
        return None
    row = rows[0]
    program_name = str(row.get("program") or "")
    if (op_spec.flops > 0.0 or op_spec.matrix_flops) and program_name in _NON_MATRIX_PROGRAMS:
        raise ModelingError(f"WORK_CONSERVATION_MISMATCH:recipe_drops_matrix:{op_spec.op_id}:{program_name}")
    if op_spec.link_bytes > 0.0 and program_name == "TiledPointwise":
        raise ModelingError(f"WORK_CONSERVATION_MISMATCH:recipe_drops_link:{op_spec.op_id}:{program_name}")
    state: BindingState = "bound_measured" if has_measured else "bound_implementation"
    return float(row["duration_ms"]) * US_PER_MS, state, program_name


def _directional_fused_traffic(
    group_nodes: Sequence[TargetOpNode],
    fused_op: OpSpec,
) -> tuple[float, float, float, float, float, float]:
    """Return exact directional ``(hbm_r, hbm_w, mte2, mte3, useful_hbm, elided_hbm)``."""
    raw_hbm = sum(float(n.hbm_total_bytes) for n in group_nodes)
    raw_mte2 = sum(float(n.mte2_bytes) for n in group_nodes)
    raw_mte3 = sum(float(n.mte3_bytes) for n in group_nodes)
    raw_hbm_r = sum(float(n.hbm_read_bytes) for n in group_nodes)
    raw_hbm_w = sum(float(n.hbm_write_bytes) for n in group_nodes)

    saved_read = max(0.0, raw_mte2 - float(fused_op.mte2_bytes)) if raw_mte2 > 0.0 else (
        max(0.0, raw_hbm - float(fused_op.hbm_bytes))
    )
    saved_write = max(0.0, raw_mte3 - float(fused_op.mte3_bytes)) if raw_mte3 > 0.0 else 0.0
    mte2 = max(0.0, float(fused_op.mte2_bytes)) if raw_mte2 > 0.0 else max(0.0, raw_hbm_r - saved_read)
    mte3 = max(0.0, float(fused_op.mte3_bytes)) if raw_mte3 > 0.0 else max(0.0, raw_hbm_w - saved_write)
    hbm_r = max(0.0, raw_hbm_r - saved_read) if raw_hbm_r > 0.0 else mte2
    hbm_w = max(0.0, raw_hbm_w - saved_write) if raw_hbm_w > 0.0 else mte3
    useful_hbm = max(0.0, float(fused_op.hbm_bytes)) if raw_hbm > 0.0 else (hbm_r + hbm_w)
    elided_hbm = max(0.0, raw_hbm - useful_hbm)
    return hbm_r, hbm_w, mte2, mte3, useful_hbm, elided_hbm


def _compute_fused_demand_and_bound(
    group_demands: Sequence[tuple[int, int, int, int, int]],
    hbm_r: float,
    hbm_w: float,
    useful_hbm: float,
    peaks: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int, int], str]:
    hbm_peak = float(peaks.get("hbm_peak_bytes_per_s") or 1.0)
    hbm_r_rate = max(float(peaks.get("hbm_read_bytes_per_s") or hbm_peak), 1.0)
    hbm_w_rate = max(float(peaks.get("hbm_write_bytes_per_s") or hbm_peak), 1.0)
    alpha = float(peaks.get("hbm_rw_alpha", 1.0) if "hbm_rw_alpha" in peaks else 1.0)
    hbm_r_ns = int((hbm_r + alpha * hbm_w) / hbm_r_rate * NS_PER_SECOND) if useful_hbm > 0.0 else 0
    hbm_w_ns = int((hbm_w + alpha * hbm_r) / hbm_w_rate * NS_PER_SECOND) if useful_hbm > 0.0 else 0
    cube_ns = sum(d[0] for d in group_demands)
    vec_ns = sum(d[1] for d in group_demands)
    link_ns = sum(d[4] for d in group_demands)
    demand = (cube_ns, vec_ns, hbm_r_ns, hbm_w_ns, link_ns)
    hbm_ns = max(hbm_r_ns, hbm_w_ns)
    dom_bound = (
        "link_bound" if link_ns > max(cube_ns, vec_ns, hbm_ns) and link_ns > 0 else (
            "memory_bound" if hbm_ns >= max(cube_ns, vec_ns) and hbm_ns > 0 else (
                "cube_bound" if cube_ns >= vec_ns and cube_ns > 0 else "vector_bound"
            )
        )
    )
    return demand, dom_bound


def _resolve_candidate_metadata(
    cand: AlgebraicCandidate,
    members: Sequence[TargetOpNode],
    member_traits: Sequence[OpTraitInfo],
    floors: SkuTargetFloors,
) -> tuple[str, str, str, str, str, bool, BindingState]:
    """Return ``(suffix, role, kernel, backend, status, needs_new, binding_state)``."""
    anchor = members[0]
    has_gmm = any(t.is_grouped_matmul for t in member_traits)
    if cand.law == "law3_mc2":
        tag = "CCU" if floors.is_a5_ub else "AIV"
        return (
            _strip_op_prefix(members[-1].op_id if cand.abi_id == "aclnnAllGatherMatmulV2" else anchor.op_id),
            f"MC2 通算融合 ({cand.abi_id})",
            cand.abi_id,
            BACKEND_HCCL_LABEL,
            f"[通算融合: MC2 ({tag}) + L0 流水线]",
            False,
            "bound_implementation",
        )
    if cand.abi_id == "aclnnMlaPrologV3":
        return (
            "mla_prolog",
            "MLA Prolog 融合 (Q/KV 低秩投影 + Norm + RoPE + KV Write)",
            "aclnnMlaPrologV3",
            BACKEND_CANN_LABEL,
            "[MLA Prolog 融合: aclnnMlaPrologV3]",
            False,
            "bound_implementation",
        )
    if has_gmm:
        is_ext = not cand.shipping_supported
        acts = [t.activation_kind for t in member_traits if t.activation_kind]
        act_lbl = acts[0].upper() if acts else "ACT"
        raw_abi = cand.abi_id
        if floors.is_a5_ub and "dispatch_ffn_combine" in raw_abi.lower():
            raw_abi = f"aclnnMegaMoe_{act_lbl.lower()}_ext" if is_ext else "aclnnMegaMoe"
        kern = raw_abi if not is_ext else (
            raw_abi if ("dispatch_ffn_combine" in raw_abi or "megamoe" in raw_abi.lower())
            else f"GroupedMatmul{act_lbl.capitalize()}Quant"
        )
        status = (
            f"{STATUS_EXTEND_ABI} {kern} (非标激活 {act_lbl} 融合)"
            if is_ext else f"{STATUS_TUNE_PREFIX} -> L0 流水线目标]"
        )
        b_state: BindingState = "unbound_formula_projection" if is_ext else "bound_implementation"
        return (
            _strip_op_prefix(anchor.op_id),
            f"Fused Routed Expert GMM1 + {act_lbl} + Quant",
            kern,
            BACKEND_ASCENDC_LABEL if is_ext else BACKEND_CATLASS_LABEL,
            status,
            is_ext,
            b_state,
        )
    if cand.law == "law2_block_matmul":
        return (
            f"fused_block_{_strip_op_prefix(anchor.op_id)}",
            f"右分配律合并矩阵乘 ({' + '.join(_strip_op_prefix(m.op_id) for m in members)})",
            "aclnnMatmul (Right-Distributive Block-W)",
            BACKEND_CANN_LABEL,
            "[框架权重拼接融合 -> L0 流水线目标]",
            False,
            "bound_implementation",
        )
    is_new = not cand.shipping_supported
    status = (
        f"{STATUS_NEW_OP} {cand.abi_id} (UB FP32 驻留融合)"
        if is_new else f"[配置/算子融合: {cand.abi_id} -> L0 流水线目标]"
    )
    has_mm = any(t.trait == AlgebraicTrait.BILINEAR_MATMUL for t in member_traits)
    new_backend = BACKEND_CATLASS_LABEL if has_mm else BACKEND_TILELANG_LABEL
    return (
        f"fused_{_strip_op_prefix(members[-1].op_id)}",
        f"Fused UB Pipeline ({' + '.join(_strip_op_prefix(m.op_id) for m in members)})",
        cand.abi_id,
        new_backend if is_new else BACKEND_CANN_LABEL,
        status,
        is_new,
        "unbound_formula_projection" if is_new else "bound_implementation",
    )


def _build_fused_target_node(
    *,
    group_nodes: Sequence[TargetOpNode],
    fused_op: OpSpec,
    demand: tuple[int, int, int, int, int],
    dom_bound: str,
    new_op_id: str,
    new_family: str,
    suffix: str,
    role: str,
    kern: str,
    backend: str,
    status: str,
    needs_new: bool,
    b_state: BindingState,
    dur_us: float,
    l_star_us: float,
    hbm_r: float,
    hbm_w: float,
    mte2: float,
    mte3: float,
    useful_hbm: float,
    spec: KernelSynthesisSpec,
    formula_eval: FormulaKernelEvaluation | None = None,
    is_shipping_abi: bool = True,
    persistent_whole_die: bool = False,
) -> TargetOpNode:
    anchor = group_nodes[0]
    absorbed = tuple(dict.fromkeys(
        s for n in group_nodes for s in (n.absorbed_source_ops or (_strip_op_prefix(n.op_id),))
    ))
    sem_atoms = tuple(dict.fromkeys(a for n in group_nodes for a in n.semantic_atoms))
    group_occs = {n.occurrence for n in group_nodes}
    ext_preds = tuple(dict.fromkeys(p for n in group_nodes for p in n.predecessors if p not in group_occs))
    new_occ = replace(anchor.occurrence, op_id=new_op_id)
    fused_meta = dict(fused_op.metadata) if fused_op.metadata else {}
    raw_fixpipe = fused_meta.get("fixpipe_bytes")
    fused_fixpipe = (
        max(0.0, float(raw_fixpipe))
        if raw_fixpipe is not None
        else sum(float(n.fixpipe_bytes) for n in group_nodes)
    )
    return replace(
        anchor,
        occurrence=new_occ,
        op_id=new_op_id,
        family=new_family,
        recipe=suffix,
        shape=tuple(fused_op.local_shape) if fused_op.local_shape else anchor.shape,
        predecessors=ext_preds,
        start_us=min(n.start_us for n in group_nodes),
        end_us=min(n.start_us for n in group_nodes) + dur_us,
        dur_us=dur_us,
        exposed_dur_us=dur_us,
        flops=float(fused_op.flops),
        vector_ops=float(fused_op.vector_ops),
        hbm_read_bytes=hbm_r,
        hbm_write_bytes=hbm_w,
        hbm_total_bytes=useful_hbm,
        mte2_bytes=mte2,
        mte3_bytes=mte3,
        fixpipe_bytes=fused_fixpipe,
        link_bytes=float(fused_op.link_bytes),
        dominant_bound=dom_bound,
        bound_display=_BOUND_LABELS.get(dom_bound, dom_bound),
        l_star_lean_us=l_star_us,
        gap_us=max(0.0, dur_us - l_star_us),
        gap_ratio=(dur_us / l_star_us) if l_star_us > 0.0 else None,
        roofline_efficiency=min(1.0, l_star_us / dur_us) if dur_us > 0.0 and l_star_us > 0.0 else None,
        absorbed_source_ops=absorbed,
        semantic_atoms=sem_atoms,
        logical_role=role,
        ascend_kernel=kern,
        kernel_backend=backend,
        kernel_status=status,
        needs_new_kernel=needs_new,
        new_kernel_spec=spec.format_formula_block(),
        canonical_op_key=suffix,
        opt_pillar=spec.opt_pillar,
        opt_attainment_pct=spec.target_attainment_pct,
        spec=fused_op,
        synthesis_spec=spec,
        binding_state=b_state,
        is_shipping_abi=bool(is_shipping_abi),
        formula_eval=formula_eval,
        persistent_whole_die=bool(persistent_whole_die),
    )


def _external_read_member_ids(
    member_ids: Sequence[str],
    graph_ops: Sequence[OpSpec],
) -> tuple[str, ...]:
    """Return member IDs consumed by any external op in ``graph_ops`` across deps/ports/named inputs."""
    keys = set(member_ids)
    read_set: set[str] = set()
    for op in graph_ops:
        if op.op_id in keys:
            continue
        for dep in op.deps:
            if dep in keys:
                read_set.add(dep)
        for dep, _ in input_port_entries(op.metadata):
            if dep in keys:
                read_set.add(dep)
        for mkey, mval in op.metadata:
            if (
                mkey in {"typed_inputs", "input_ports"}
                and isinstance(mval, Sequence)
                and not isinstance(mval, (str, bytes))
            ):
                for entry in mval:
                    if isinstance(entry, Mapping) and entry.get("producer") in keys:
                        read_set.add(str(entry["producer"]))
            elif mkey.endswith("_producer") and isinstance(mval, str) and mval in keys:
                read_set.add(mval)
    return tuple(mid for mid in member_ids if mid in read_set)


def _node_precision_class(node: TargetOpNode) -> tuple[str, str, bool]:
    """Return ``(precision_class, signature, has_explicit_cast)`` for ABI legality checks."""
    spec = node.spec
    abi: TensorPrecisionABI | None = getattr(spec, "precision_abi", None) if spec is not None else None
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    if abi is not None:
        has_cast = bool(abi.explicit_cast_on_entry or abi.explicit_cast_on_exit)
        dts = {str(abi.input_dtype).lower(), str(abi.weight_dtype).lower(), str(abi.compute_dtype).lower()}
        if dts & QUANT_PREC_DTYPES:
            return "quant", abi.abi_signature, has_cast
        if dts & FLOAT_PREC_DTYPES:
            return "float", abi.abi_signature, has_cast
        return "other", abi.abi_signature, has_cast
    dts = {
        str(meta.get(k) or "").strip().lower()
        for k in ("activation_format", "weight_format", "compute_format")
        if meta.get(k)
    }
    if not dts:
        dts.add(str(node.dtype or "bf16").strip().lower())
    has_cast = bool(meta.get("explicit_cast_on_entry") or meta.get("explicit_cast_on_exit"))
    if dts & QUANT_PREC_DTYPES:
        return "quant", "|".join(sorted(dts)), has_cast
    if dts & FLOAT_PREC_DTYPES:
        return "float", "|".join(sorted(dts)), has_cast
    return "other", "|".join(sorted(dts)), has_cast


def _has_illegal_cross_precision_merge(group_nodes: Sequence[TargetOpNode]) -> bool:
    """Return True if matrix members mix FLOAT and FP8/INT8/MXFP4 without explicit cast."""
    matrix_members = [n for n in group_nodes if float(n.flops) > 0.0]
    if len(matrix_members) < MIN_FUSE_MEMBERS:
        return False
    classes = set()
    abis: list[TensorPrecisionABI] = []
    for node in matrix_members:
        p_cls, _sig, has_cast = _node_precision_class(node)
        if not has_cast and p_cls in {"float", "quant"}:
            classes.add(p_cls)
        abi = getattr(node.spec, "precision_abi", None) if node.spec is not None else None
        if abi is not None:
            abis.append(abi)
    if "float" in classes and "quant" in classes:
        return True
    if len(abis) >= MIN_FUSE_MEMBERS:
        base = abis[0]
        if any(not base.is_fusion_compatible(other) for other in abis[1:]):
            return True
    return False


def _validate_final_graph_invariants(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
) -> None:
    """Verify zero ``L* > T_Engine`` inversions and ABI legality on the selected target graph."""
    for node, dem in zip(nodes, demands):
        l_star_dem = float(max(dem)) / NS_PER_US if dem else 0.0
        l_star = max(float(node.l_star_lean_us), l_star_dem)
        if node.dur_us > 0.0 and l_star > float(node.dur_us) + L_STAR_INVERSION_TOL_US:
            raise ModelingError(
                f"lower_bound_inversion:{node.op_id}:dur_us={node.dur_us:.4f}<l_star_us={l_star:.4f}"
            )


def _collapse_candidate_group(
    group_nodes: Sequence[TargetOpNode],
    group_demands: Sequence[tuple[int, int, int, int, int]],
    cand: AlgebraicCandidate,
    member_traits: Sequence[OpTraitInfo],
    arch_ops: Sequence[OpSpec],
    floors: SkuTargetFloors,
    peaks: Mapping[str, Any],
    case: Case,
    datapath: DatapathCapabilitySpec,
    pricing_ctx: _TargetPricingContext,
    *,
    edit_id: str,
    verify_ops: Sequence[OpSpec] | None = None,
    phase: str = "decode",
    primitive_table: Any = None,
    allow_unbound_formula: bool = True,
) -> tuple[TargetOpNode, tuple[int, int, int, int, int], KernelSynthesisSpec, Edit] | None:
    """Collapse a verified candidate via canonical ``apply_plan`` + bound/formula pricing."""
    if _has_illegal_cross_precision_merge(group_nodes):
        return None
    suffix, role, kern, backend, status, needs_new, meta_state = _resolve_candidate_metadata(
        cand, group_nodes, member_traits, floors,
    )
    plan_ops = verify_ops if verify_ops is not None else arch_ops
    eff_preserve_ids = tuple(dict.fromkeys((
        *cand.preserve_port_ids,
        *_external_read_member_ids(cand.member_ids, plan_ops),
    )))
    refs = tuple(OpRef("atom", mid) for mid in cand.member_ids)
    edit = Edit(
        edit_id=edit_id,
        operation=FuseRegion(
            members=refs,
            preserve_ports=tuple(PortRef(OpRef("atom", mid), "output", 0) for mid in eff_preserve_ids),
            kernel_boundaries=(KernelBoundary(cand.abi_id, refs),),
            fusion_tier=cand.fusion_tier,
        ),
    )
    trial_plan = RewritePlan(edits=(edit,))
    try:
        scenario, applied, receipt = _apply_and_verify_plan(
            case, plan_ops, trial_plan, datapath, phase=phase, primitive_table=primitive_table,
            bind_empty_digests=True,
        )
    except ModelingError:
        return None
    if not receipt.passed:
        return None

    fused_op_id = f"rewrite:{edit_id}"
    fused_op = next((op for op in applied.executable[0].ops if op.op_id == fused_op_id), None)
    if fused_op is None:
        return None

    hbm_r, hbm_w, mte2, mte3, useful_hbm, elided_hbm = _directional_fused_traffic(group_nodes, fused_op)
    demand, dom_bound = _compute_fused_demand_and_bound(group_demands, hbm_r, hbm_w, useful_hbm, peaks)
    l_star_us = float(max(demand)) / NS_PER_US

    try:
        bound_res = _price_bound_op_on_engine(
            fused_op, pricing_ctx, case,
            scenario=scenario, atomic_entry=scenario.atomic_ops[0], applied=applied,
        )
    except ModelingError:
        bound_res = None
    eval_res: FormulaKernelEvaluation | None = None
    if bound_res is not None:
        projected_us, b_state, _prog = bound_res
        abi_sig = cand.abi_id
    else:
        if not allow_unbound_formula:
            return None
        group_ops = tuple(_node_to_op_spec(n, None, phase=phase) for n in group_nodes)
        formula_spec = replace(
            FormulaKernelSpec.from_ops(
                group_ops, name=kern, preserve_port_ids=eff_preserve_ids,
            ),
            hbm_bytes_override=useful_hbm,
            mte2_bytes_override=mte2,
            mte3_bytes_override=mte3,
        )
        eval_res = evaluate_formula_kernel(formula_spec, case, primitive_table=primitive_table)
        if not eval_res.sram_verified:
            return None
        projected_us = eval_res.projected_us
        l_star_us = max(l_star_us, eval_res.l_star_us)
        abi_sig = eval_res.abi_signature_id
        b_state = "unbound_formula_projection"

    before_sum_us = sum(n.dur_us for n in group_nodes)
    exposed_before_sum_us = sum(max(0.0, float(n.exposed_dur_us)) for n in group_nodes)
    t_target = max(projected_us, l_star_us)
    if cand.law == "law3_mc2":
        t_target = max(t_target, floors.mc2_sync_us + float(demand[4]) / NS_PER_US)
    dur_us = min(before_sum_us, t_target)
    if dur_us >= before_sum_us - SAVING_POSITIVE_EPS_US:
        return None

    achieved_eta = min(1.0, l_star_us / dur_us) if dur_us > 0.0 and l_star_us > 0.0 else 1.0
    spec = synthesize_kernel_spec(
        spec_id=cand.candidate_id,
        kernel_name=kern,
        abi_tier=cand.fusion_tier,
        suggested_backend=backend,
        status_tag=status,
        opt_pillar="pillar3_comm" if cand.law == "law3_mc2" else "pillar1_fusion",
        members=group_nodes,
        member_traits=member_traits,
        datapath=datapath,
        elided_hbm_bytes=int(round(elided_hbm)),
        l_star_us=l_star_us,
        before_us=before_sum_us,
        target_us=dur_us,
        eta_target=achieved_eta,
        abi_signature_id=abi_sig,
        binding_state=b_state,
        preserve_port_ids=eff_preserve_ids,
        exposed_before_us=exposed_before_sum_us,
    )
    if not spec.sram_verified:
        return None

    anchor = group_nodes[0]
    prefix = anchor.op_id[: len(anchor.op_id) - len(_strip_op_prefix(anchor.op_id))]
    new_op_id = f"{prefix}{suffix}"
    new_family = (
        "all_gather_matmul" if cand.abi_id == "aclnnAllGatherMatmulV2" else (
            "matmul_reduce_scatter" if cand.abi_id == "aclnnMatmulReduceScatterV2" else anchor.family
        )
    )
    eff_needs_new = bool(needs_new or meta_state == "unbound_formula_projection")
    is_persist = ("megamoe" in kern.lower()) or ("dispatch_ffn_combine" in kern.lower())
    fused_node = _build_fused_target_node(
        group_nodes=group_nodes, fused_op=fused_op, demand=demand, dom_bound=dom_bound,
        new_op_id=new_op_id, new_family=new_family, suffix=suffix, role=role, kern=kern,
        backend=backend, status=status, needs_new=eff_needs_new,
        b_state=b_state, dur_us=dur_us, l_star_us=l_star_us,
        hbm_r=hbm_r, hbm_w=hbm_w, mte2=mte2, mte3=mte3, useful_hbm=useful_hbm, spec=spec,
        formula_eval=eval_res, is_shipping_abi=not eff_needs_new, persistent_whole_die=is_persist,
    )
    return fused_node, demand, spec, edit


def _rewire_node_preds(
    nodes: Sequence[TargetOpNode], occ_remap: Mapping[OpOccurrence, OpOccurrence],
) -> list[TargetOpNode]:
    if not occ_remap:
        return list(nodes)
    valid_occs = {n.occurrence for n in nodes}
    return [
        replace(
            n,
            predecessors=tuple(dict.fromkeys(
                mapped for p in n.predecessors
                if (mapped := occ_remap.get(p, p)) != n.occurrence and mapped in valid_occs
            )),
        )
        for n in nodes
    ]


def _materialize_external_fused_op(
    op: OpSpec,
    member_ids: tuple[str, ...],
    node_by_id: Mapping[str, TargetOpNode],
    dem_by_id: Mapping[str, tuple[int, int, int, int, int]],
    template_binding: Any,
    scenario: Any,
    applied: Any,
    peaks: Mapping[str, Any],
    case: Case,
    datapath: DatapathCapabilitySpec,
    pricing_ctx: _TargetPricingContext,
    *,
    phase: str,
    primitive_table: Any,
    allow_unbound_formula: bool,
) -> tuple[TargetOpNode, tuple[int, int, int, int, int], KernelSynthesisSpec]:
    """Price and materialize an explicit caller ``FuseRegion`` applied by ``apply_plan``."""
    group_nodes = [node_by_id[mid] for mid in member_ids if mid in node_by_id]
    if _has_illegal_cross_precision_merge(group_nodes):
        raise ModelingError(f"cross_precision_abi_merge_forbidden:{op.op_id}")
    group_dems = [dem_by_id[mid] for mid in member_ids if mid in dem_by_id]
    group_ops = tuple(_node_to_op_spec(n, None, phase=phase) for n in group_nodes)
    member_traits = tuple(classify_op_traits(item) for item in group_ops)

    edit = next((e for e in applied.plan.edits if op.op_id == f"rewrite:{e.edit_id}"), None)
    freg = edit.operation if (edit is not None and isinstance(edit.operation, FuseRegion)) else None
    boundaries = freg.kernel_boundaries if freg is not None else ()
    preserve_ids = tuple(p.op.id for p in freg.preserve_ports) if freg is not None else ()
    tier = freg.fusion_tier if freg is not None else "K2_epilogue"
    if len(boundaries) != 1:
        all_abis = "+".join(b.abi_id for b in boundaries) if boundaries else (op.recipe or op.op_id)
        raise ModelingError(f"rewrite_fusion_recipe_unbound:{op.op_id}:{all_abis}")
    abi = boundaries[0].abi_id

    hbm_r, hbm_w, mte2, mte3, useful_hbm, elided_hbm = _directional_fused_traffic(group_nodes, op)
    demand, dom_bound = _compute_fused_demand_and_bound(group_dems, hbm_r, hbm_w, useful_hbm, peaks)
    l_star_us = float(max(demand)) / NS_PER_US

    bound_res = _price_bound_op_on_engine(
        op, pricing_ctx, case,
        scenario=scenario, atomic_entry=scenario.atomic_ops[0],
        template_binding=template_binding, applied=applied,
    )
    eval_res: FormulaKernelEvaluation | None = None
    if bound_res is not None:
        dur_us, b_state, _prog = bound_res
        abi_sig = template_binding.template_id if template_binding is not None else abi
        needs_new = False
        status = f"[配置/算子融合: {abi_sig} -> 绑定实现]"
    else:
        if not allow_unbound_formula:
            raise ModelingError(f"rewrite_fusion_recipe_unbound:{op.op_id}:{abi}")
        formula_spec = replace(
            FormulaKernelSpec.from_ops(group_ops, name=abi, preserve_port_ids=preserve_ids),
            hbm_bytes_override=useful_hbm,
            mte2_bytes_override=mte2,
            mte3_bytes_override=mte3,
        )
        eval_res = evaluate_formula_kernel(formula_spec, case, primitive_table=primitive_table)
        if not eval_res.sram_verified:
            raise ModelingError(
                f"MULTI_POOL_CAPACITY:{op.op_id}:{eval_res.tile_algorithm}"
                f":ub={eval_res.ub_peak_bytes}>{eval_res.ub_budget_bytes}"
            )
        dur_us = max(eval_res.projected_us, l_star_us)
        l_star_us = max(l_star_us, eval_res.l_star_us)
        abi_sig = eval_res.abi_signature_id
        b_state = "unbound_formula_projection"
        needs_new = True
        status = f"{STATUS_NEW_OP} {abi} (UB FP32 驻留融合)"

    before_sum_us = sum(n.dur_us for n in group_nodes)
    exposed_before_sum_us = sum(max(0.0, float(n.exposed_dur_us)) for n in group_nodes)
    achieved_eta = min(1.0, l_star_us / dur_us) if dur_us > 0.0 and l_star_us > 0.0 else 1.0
    has_mm_ext = any(t.trait == AlgebraicTrait.BILINEAR_MATMUL for t in member_traits)
    eff_backend = (
        (BACKEND_CATLASS_LABEL if has_mm_ext else BACKEND_TILELANG_LABEL)
        if needs_new else BACKEND_CANN_LABEL
    )
    spec = synthesize_kernel_spec(
        spec_id=f"plan_{edit.edit_id}" if edit is not None else op.op_id,
        kernel_name=abi,
        abi_tier=tier,
        suggested_backend=eff_backend,
        status_tag=status,
        opt_pillar="pillar1_fusion",
        members=group_nodes,
        member_traits=member_traits,
        datapath=datapath,
        elided_hbm_bytes=int(round(elided_hbm)),
        l_star_us=l_star_us,
        before_us=before_sum_us,
        target_us=dur_us,
        eta_target=achieved_eta,
        abi_signature_id=abi_sig,
        binding_state=b_state,
        preserve_port_ids=preserve_ids,
        exposed_before_us=exposed_before_sum_us,
    )
    if not spec.sram_verified and b_state == "unbound_formula_projection":
        raise ModelingError(
            f"MULTI_POOL_CAPACITY:{op.op_id}:{spec.tile_algorithm}"
            f":ub={spec.ub_peak_bytes}>{spec.ub_budget_bytes}"
        )
    fused_node = _build_fused_target_node(
        group_nodes=group_nodes, fused_op=op, demand=demand, dom_bound=dom_bound,
        new_op_id=op.op_id, new_family="rewrite_fused", suffix=abi,
        role=f"RewritePlan Fused ({' + '.join(_strip_op_prefix(m.op_id) for m in group_nodes)})",
        kern=abi, backend=eff_backend,
        status=status, needs_new=needs_new, b_state=b_state,
        dur_us=dur_us, l_star_us=l_star_us,
        hbm_r=hbm_r, hbm_w=hbm_w, mte2=mte2, mte3=mte3, useful_hbm=useful_hbm, spec=spec,
        formula_eval=eval_res, is_shipping_abi=not needs_new,
        persistent_whole_die=("megamoe" in abi.lower() or "dispatch_ffn_combine" in abi.lower()),
    )
    return fused_node, demand, spec


def _materialize_external_plan(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    ext_plan: RewritePlan,
    case: Case,
    datapath: DatapathCapabilitySpec,
    floors: SkuTargetFloors,
    peaks: Mapping[str, Any],
    pricing_ctx: _TargetPricingContext,
    *,
    phase: str,
    primitive_table: Any,
    allow_unbound_formula: bool,
    rebind_digests: bool = False,
) -> tuple[
    list[TargetOpNode],
    list[tuple[int, int, int, int, int]],
    list[KernelSynthesisSpec],
    set[str],
    dict[OpOccurrence, OpOccurrence],
    dict[OpOccurrence, tuple[Any, float]],
]:
    """Execute and price ALL supported caller ``RewritePlan`` edits via canonical ``apply_plan``."""
    all_known = {n.op_id for n in nodes}
    base_ops = [_node_to_op_spec(n, all_known, phase=phase) for n in nodes]
    scenario, applied, receipt = _apply_and_verify_plan(
        case, base_ops, ext_plan, datapath, phase=phase, primitive_table=primitive_table,
        bind_empty_digests=True, rebind_digests=rebind_digests,
    )
    if not receipt.passed:
        failed = next((g for g in receipt.gates if not g.passed), receipt.gates[0])
        raise ModelingError(f"REWRITE_PLAN_REJECTED:{failed.name}:{failed.reason}:{failed.detail}")

    node_by_id = {n.op_id: n for n in nodes}
    dem_by_id = {n.op_id: d for n, d in zip(nodes, demands)}
    absorbed_map = dict(applied.absorbed_atoms)
    template_map = {b.target_op: b for b in applied.template_bindings}
    collective_map = {c.target_op: c for c in applied.collective_rewrites}

    out_nodes: list[TargetOpNode] = []
    out_dems: list[tuple[int, int, int, int, int]] = []
    out_specs: list[KernelSynthesisSpec] = []
    touched_ids: set[str] = set()
    remap: dict[OpOccurrence, OpOccurrence] = {}
    col_plans: dict[OpOccurrence, tuple[Any, float]] = {}

    for op in applied.executable[0].ops:
        if op.op_id in absorbed_map:
            m_ids = absorbed_map[op.op_id]
            fused_node, fused_dem, kspec = _materialize_external_fused_op(
                op, m_ids, node_by_id, dem_by_id, template_map.get(op.op_id),
                scenario, applied, peaks, case, datapath, pricing_ctx,
                phase=phase, primitive_table=primitive_table,
                allow_unbound_formula=allow_unbound_formula,
            )
            out_nodes.append(fused_node)
            out_dems.append(fused_dem)
            out_specs.append(kspec)
            touched_ids.add(fused_node.op_id)
            for mid in m_ids:
                if mid in node_by_id:
                    remap[node_by_id[mid].occurrence] = fused_node.occurrence
            continue

        base_node = node_by_id[op.op_id]
        base_dem = dem_by_id[op.op_id]
        updated_node = replace(base_node, spec=op)
        if op.op_id in template_map:
            tb = template_map[op.op_id]
            bound_res = _price_bound_op_on_engine(
                op, pricing_ctx, case,
                scenario=scenario, atomic_entry=scenario.atomic_ops[0],
                template_binding=tb, applied=applied,
            )
            if bound_res is None:
                raise ModelingError(f"rewrite_template_recipe_unbound:{op.op_id}:{tb.template_id}")
            dur_us, b_state, _prog = bound_res
            dur_ns = int(round(dur_us * NS_PER_US))
            base_dem = tuple(min(int(d), dur_ns) for d in base_dem)  # type: ignore[assignment]
            eff_l_star = min(float(base_node.l_star_lean_us), dur_us)
            eff = min(1.0, eff_l_star / dur_us) if dur_us > 0.0 and eff_l_star > 0.0 else None
            updated_node = replace(
                updated_node,
                dur_us=dur_us, exposed_dur_us=dur_us,
                l_star_lean_us=eff_l_star,
                recipe=tb.template_id, ascend_kernel=tb.template_id,
                kernel_status=f"[模板切换: {tb.template_id}]", needs_new_kernel=False,
                gap_us=max(0.0, dur_us - eff_l_star),
                gap_ratio=(dur_us / eff_l_star) if eff_l_star > 0.0 else None,
                roofline_efficiency=eff, opt_pillar="pillar2_roofline",
                opt_attainment_pct=round(eff * PERCENT_SCALE, 1) if eff is not None else None,
                binding_state=b_state,
            )
            touched_ids.add(op.op_id)
        if op.op_id in collective_map:
            col_rec = collective_map[op.op_id]
            built = _build_collective_plan(
                op, pricing_ctx, case,
                mode=col_rec.mode, topology_route=col_rec.topology_route,
            )
            if built is None:
                raise ModelingError(f"rewrite_collective_route_recipe_unbound:{op.op_id}")
            topology, op_norm, plan = built
            if plan is None:
                col_us = float(floors.launch_vec_us)
            else:
                col_plans[updated_node.occurrence] = (plan, float(floors.launch_vec_us))
                eng = Engine(topology.resources_for_ranks(plan.ranks), backend="python")
                rdy = eng.submit("ready")
                done = plan.start(eng, op_norm.op_id, rdy)
                eng.advance(until=(done,))
                col_us = max(float(done.time) * US_PER_MS, float(floors.launch_vec_us))
            base_k = updated_node.ascend_kernel.split(" (", 1)[0]
            col_ns = int(round(col_us * NS_PER_US))
            base_dem = tuple(min(int(d), col_ns) for d in base_dem)  # type: ignore[assignment]
            eff_l_star_c = min(float(updated_node.l_star_lean_us), col_us)
            eff = min(1.0, eff_l_star_c / col_us) if col_us > 0.0 and eff_l_star_c > 0.0 else None
            updated_node = replace(
                updated_node,
                dur_us=col_us, exposed_dur_us=col_us,
                l_star_lean_us=eff_l_star_c,
                recipe=f"collective_{col_rec.mode}",
                ascend_kernel=f"{base_k} ({col_rec.mode}:{col_rec.topology_route})",
                kernel_status=f"[通信方案切换: {col_rec.mode} @ {col_rec.topology_route}]",
                needs_new_kernel=False,
                gap_us=max(0.0, col_us - eff_l_star_c),
                gap_ratio=(col_us / eff_l_star_c) if eff_l_star_c > 0.0 else None,
                roofline_efficiency=eff, opt_pillar="pillar3_comm",
                opt_attainment_pct=round(eff * PERCENT_SCALE, 1) if eff is not None else None,
                binding_state="bound_implementation",
            )
            touched_ids.add(op.op_id)
        out_nodes.append(updated_node)
        out_dems.append(base_dem)

    # Rewire predecessors from applied.executable OpSpec deps so HoistOrDeduplicate / FuseRegion updates hold.
    occ_by_id = {n.op_id: n.occurrence for n in out_nodes}
    rewired_nodes: list[TargetOpNode] = []
    for node, op in zip(out_nodes, applied.executable[0].ops):
        new_preds = tuple(dict.fromkeys(
            occ_by_id[d] for d in op.deps if d in occ_by_id and d != node.op_id
        ))
        rewired_nodes.append(replace(node, predecessors=new_preds))
    return rewired_nodes, out_dems, out_specs, touched_ids, remap, col_plans


def _apply_pillar_c_tiler_extensions(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    case: Case,
    *,
    phase: str,
    locked_ids: set[str] | None = None,
    primitive_table: Any = None,
    allow_unbound_formula: bool = True,
) -> tuple[list[TargetOpNode], list[CoDesignOpportunity]]:
    """Evaluate UB-legal Pillar C CATLASS/AscendC tile schedules strictly on unbound matrix ops.

    Never collapses bound single ops (``AttnResFwd``, ``HC combine``, ``Conv1d``,
    ``GDN/KDA scan``, ``RecoverSSM``, ``MoE gating``, ``PLE``) to launch floors.
    """
    locked = locked_ids or set()
    datapath = datapath_for_die(case.machine.dies[0])
    out: list[TargetOpNode] = []
    tiler_opps: list[CoDesignOpportunity] = []
    for node, dem in zip(nodes, demands):
        fam = str(node.family or "").strip().lower()
        if (
            not allow_unbound_formula
            or node.op_id in locked
            or node.opt_pillar in {"pillar1_fusion", "pillar3_comm"}
            or node.dur_us <= 0.0
            or fam in BOUND_NON_COLLAPSIBLE_FAMILIES
            or fam not in TILER_EXT_FAMILIES
            or node.flops <= 0.0
            or (node.spec is not None and getattr(node.spec, "owner_schedule", None) is not None)
        ):
            out.append(node)
            continue

        op_stub = _node_to_op_spec(node, set(), phase=phase)
        formula_spec = FormulaKernelSpec.from_ops((op_stub,), name=node.ascend_kernel or node.op_id)
        try:
            eval_res = evaluate_formula_kernel(formula_spec, case, primitive_table=primitive_table)
        except ModelingError:
            out.append(node)
            continue
        if not eval_res.sram_verified:
            out.append(node)
            continue

        l_star = max(float(node.l_star_lean_us), float(max(dem)) / NS_PER_US)
        t_target = max(float(eval_res.projected_us), l_star)
        t_after = min(float(node.dur_us), t_target)
        gain = float(node.dur_us) - t_after
        if gain < TARGET_OPT_MIN_GAIN_US or gain < float(node.dur_us) * TARGET_OPT_MIN_RATIO:
            out.append(replace(node, kernel_status=STATUS_AT_FLOOR))
            continue

        eff = min(1.0, l_star / t_after) if (t_after > 0.0 and l_star > 0.0) else None
        out.append(replace(
            node,
            dur_us=t_after, exposed_dur_us=t_after,
            kernel_backend=BACKEND_CATLASS_LABEL,
            kernel_status=f"{STATUS_TUNE_PREFIX} -> Catlass/AscendC Tiler]",
            needs_new_kernel=False,
            is_shipping_abi=True,
            formula_eval=eval_res,
            l_star_lean_us=l_star,
            gap_us=max(0.0, t_after - l_star),
            gap_ratio=(t_after / l_star) if l_star > 0.0 else None,
            roofline_efficiency=eff,
            opt_pillar="pillar2_roofline",
            opt_attainment_pct=round(eff * PERCENT_SCALE, 1) if eff is not None else None,
            binding_state="unbound_formula_projection",
        ))
        before_ms = float(node.dur_us) / US_PER_MS
        l_star_ms = l_star / US_PER_MS
        tiled_ms = t_after / US_PER_MS
        ub_live = max(0, int(eval_res.ub_peak_bytes))
        simt_gain_ms, simd_gain_ms, ub_warn = _resolve_tilelang_dual_tier_bounds(
            datapath=datapath,
            before_ms=before_ms,
            l_star_ms=l_star_ms,
            tiled_ms=tiled_ms,
            ub_live_bytes=ub_live,
            has_cube=True,
            has_aiv=(ub_live > 0),
            cube_dtype=str(node.dtype or "bf16"),
            vector_ops=float(node.vector_ops),
            is_vector_or_norm_region=False,
        )
        tiler_opps.append(CoDesignOpportunity(
            opportunity_id=f"tiler_{_strip_op_prefix(node.op_id)}",
            kind="tiler_tune",
            track="kernel_tune",
            readiness="tiler_extension",
            title=f"Catlass/AscendC Tiler Extension: {node.ascend_kernel or node.op_id}",
            evidence_refs=(node.op_id,),
            mega_fusion=MegaFusionBound(
                region_op_ids=(node.op_id,),
                exposed_before_ms=before_ms,
                boundary_lower_bound_ms=l_star_ms,
                tiled_engine_ms=tiled_ms,
                gain_ceiling_ms=before_ms - l_star_ms,
                gain_expected_ms=before_ms - tiled_ms,
                eliminated_gm_bytes=0,
                ub_live_bytes=ub_live,
                ub_capacity_bytes=max(int(eval_res.ub_budget_bytes), ub_live),
                requires_unabsorbed_split_cost_check=False,
                tilelang_simt_gain_ms=simt_gain_ms,
                tilelang_simd_or_ascendc_gain_ms=simd_gain_ms,
                ub_wall_warning=ub_warn,
            ),
        ))
    return out, tiler_opps


def _solve_one_archetype(
    aid: str,
    a_nodes: Sequence[TargetOpNode],
    a_dems: Sequence[tuple[int, int, int, int, int]],
    locked_ids: set[str],
    used_edit_ids: set[str],
    floors: SkuTargetFloors,
    p0: Mapping[str, Any],
    case: Case,
    datapath: DatapathCapabilitySpec,
    pricing_ctx: _TargetPricingContext,
    *,
    verify_ops: Sequence[OpSpec] | None = None,
    phase: str,
    safe_n_tok: float,
    primitive_table: Any,
    allow_unbound_formula: bool,
) -> tuple[
    list[TargetOpNode],
    list[tuple[int, int, int, int, int]],
    list[KernelSynthesisSpec],
    list[SemanticAuditTrigger],
    list[Edit],
    dict[OpOccurrence, OpOccurrence],
]:
    known_ids = {n.op_id for n in a_nodes}
    arch_ops = [_node_to_op_spec(n, known_ids, phase=phase) for n in a_nodes]
    traits_map = {op.op_id: classify_op_traits(op) for op in arch_ops}
    alg = discover_algebraic_rewrites_for_ops(
        arch_ops, datapath=datapath, allow_mc2=True, include_abi_extensions=allow_unbound_formula,
    )
    idx_by_id = {n.op_id: i for i, n in enumerate(a_nodes)}
    consumed_idxs: set[int] = set()
    replacements: dict[int, tuple[TargetOpNode, tuple[int, int, int, int, int]]] = {}
    unlocked_ops: set[str] = set()
    specs: list[KernelSynthesisSpec] = []
    edits: list[Edit] = []
    remap: dict[OpOccurrence, OpOccurrence] = {}

    for cand in alg.candidates:
        if set(cand.member_ids) & locked_ids:
            continue
        idxs = [idx_by_id[mid] for mid in cand.member_ids if mid in idx_by_id]
        if len(idxs) < MIN_FUSE_MEMBERS or any(i in consumed_idxs for i in idxs):
            continue
        idxs.sort()
        grp_nodes = [a_nodes[i] for i in idxs]
        grp_dems = [a_dems[i] for i in idxs]
        grp_traits = [traits_map[n.op_id] for n in grp_nodes]
        base_eid = f"{aid}_{cand.candidate_id}"
        eid = base_eid
        serial = 1
        while eid in used_edit_ids:
            eid = f"{base_eid}_{serial}"
            serial += 1
        collapsed = _collapse_candidate_group(
            grp_nodes, grp_dems, cand, grp_traits, arch_ops, floors, p0, case, datapath, pricing_ctx,
            edit_id=eid, verify_ops=verify_ops, phase=phase, primitive_table=primitive_table,
            allow_unbound_formula=allow_unbound_formula,
        )
        if collapsed is None:
            continue
        fused_node, fused_dem, kspec, edit = collapsed
        used_edit_ids.add(eid)
        consumed_idxs.update(idxs)
        replacements[idxs[0]] = (fused_node, fused_dem)
        unlocked_ops.update(cand.member_ids)
        specs.append(kspec)
        edits.append(edit)
        for i in idxs:
            remap[a_nodes[i].occurrence] = fused_node.occurrence

    out_nodes: list[TargetOpNode] = []
    out_dems: list[tuple[int, int, int, int, int]] = []
    for i, (n, d) in enumerate(zip(a_nodes, a_dems)):
        if i in replacements:
            out_nodes.append(replacements[i][0])
            out_dems.append(replacements[i][1])
        elif i not in consumed_idxs:
            out_nodes.append(n)
            out_dems.append(d)

    triggers: list[SemanticAuditTrigger] = []
    hbm_bw = max(float(p0.get("hbm_peak_bytes_per_s") or 1.0), 1.0)
    for trig in alg.semantic_audit_triggers:
        if set(trig.blocked_ops) <= unlocked_ops:
            continue
        rep = a_nodes[0].repeat_count if a_nodes else 1
        raw_us = sum(a_nodes[idx_by_id[oid]].dur_us for oid in trig.blocked_ops if oid in idx_by_id)
        hbm_save_us = (float(trig.avoidable_hbm_bytes) / hbm_bw) * US_PER_SECOND
        est_us = min(raw_us * HALF_SPLIT_RATIO, max(hbm_save_us, floors.launch_vec_us * max(1, len(trig.blocked_ops) - 1))) * rep
        triggers.append(replace(
            trig, potential_gain_us=est_us, potential_gain_us_per_tok_die=est_us / safe_n_tok,
        ))
    return out_nodes, out_dems, specs, triggers, edits, remap


def _verify_final_target_plan(
    nodes: Sequence[TargetOpNode],
    final_plan: RewritePlan,
    case: Case,
    datapath: DatapathCapabilitySpec,
    *,
    phase: str,
    primitive_table: Any,
) -> bool:
    """Verify the complete combined plan against the base target graph via canonical ``verify_plan``."""
    if not nodes:
        return True
    unique_ids = {n.op_id for n in nodes}
    if len(unique_ids) == len(nodes):
        base_ops = [_node_to_op_spec(n, unique_ids, phase=phase) for n in nodes]
        _, _, receipt = _apply_and_verify_plan(
            case, base_ops, final_plan, datapath, phase=phase, primitive_table=primitive_table,
        )
        return bool(receipt.passed)
    by_arch: dict[str, list[TargetOpNode]] = defaultdict(list)
    for n in nodes:
        by_arch[n.archetype_id].append(n)
    all_passed = True
    for aid, arch_nodes in by_arch.items():
        ids = {n.op_id for n in arch_nodes}
        arch_ops = [_node_to_op_spec(n, ids, phase=phase) for n in arch_nodes]
        arch_edits = tuple(e for e in final_plan.edits if e.edit_id.startswith(f"{aid}_"))
        sub_plan = RewritePlan(edits=arch_edits)
        _, _, receipt = _apply_and_verify_plan(
            case, arch_ops, sub_plan, datapath, phase=phase, primitive_table=primitive_table,
            bind_empty_digests=True,
        )
        all_passed = all_passed and bool(receipt.passed)
    return all_passed


def _reconcile_oneshot_link_ledger(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]],
    peaks_by_stage: Mapping[int, Mapping[str, Any]],
    default_peaks: Mapping[str, Any],
) -> tuple[list[TargetOpNode], list[tuple[int, int, int, int, int]]]:
    """Re-derive ``link_bytes`` / link demand / ``L*`` from executed ``OneShotPlan`` messages.

    A selected one-shot plan sends the full buffer to every peer, so the ring
    volume inherited from the source graph under-counts useful TX per rank.
    """
    out_nodes, out_dems = list(nodes), list(demands)
    for i, node in enumerate(out_nodes):
        entry = collective_plans.get(node.occurrence)
        if entry is None or not isinstance(entry[0], OneShotPlan):
            continue
        peaks = peaks_by_stage.get(node.occurrence.stage) or default_peaks
        rank = node.occurrence.rank
        if rank is None:
            die = peaks.get("die")
            rank = int(getattr(die, "rank", 0) or 0)
        acc = entry[0].accounting(source_ranks=(rank,))
        tx_useful = float(acc.useful_by_rank.get(rank, 0.0))
        tx_wire = float(acc.physical_wire_bytes) if float(acc.physical_wire_bytes) > 0.0 else tx_useful
        link_rate = float(peaks.get("link_peak_bytes_per_s") or 0.0)
        link_ns = int(tx_wire * NS_PER_SECOND / link_rate) if link_rate > 0.0 else int(out_dems[i][4])
        out_dems[i] = (*out_dems[i][:4], link_ns)
        l_star = float(max(out_dems[i])) / NS_PER_US
        dur = float(node.dur_us)
        eff = min(1.0, l_star / dur) if dur > 0.0 and l_star > 0.0 else None
        out_nodes[i] = replace(
            node, link_bytes=tx_wire, l_star_lean_us=l_star,
            gap_us=max(0.0, dur - l_star),
            gap_ratio=(dur / l_star) if l_star > 0.0 else None,
            roofline_efficiency=eff,
            opt_attainment_pct=round(eff * PERCENT_SCALE, 1) if eff is not None else None,
        )
    return out_nodes, out_dems


def _classify_moe_frame_role(node: TargetOpNode) -> str | None:
    if is_shared_expert_op_id(node.op_id):
        return "shared"
    fam = str(node.family or "").strip().lower()
    oid = str(node.op_id or "").strip().lower()
    if fam in {"dispatch", "alltoall", "alltoallv"} or ".dispatch" in oid:
        return "dispatch"
    if fam == "combine" or ".combine" in oid:
        return "combine"
    if fam == "grouped_gemm" or ".expert_" in oid:
        return "gmm"
    return None


def _has_external_cycle_bridge(
    arch_nodes: Sequence[TargetOpNode], member_occs: frozenset[OpOccurrence],
) -> bool:
    preds_by_occ = {n.occurrence: n.predecessors for n in arch_nodes}
    ext_after_member: set[OpOccurrence] = set()
    changed = True
    while changed:
        changed = False
        for node in arch_nodes:
            occ = node.occurrence
            if occ in member_occs or occ in ext_after_member:
                continue
            if any(p in member_occs or p in ext_after_member for p in preds_by_occ.get(occ, ())):
                ext_after_member.add(occ)
                changed = True
    return any(
        p in ext_after_member
        for node in arch_nodes if node.occurrence in member_occs
        for p in node.predecessors
    )


def _collapse_moe_archetype_group(
    members: Sequence[TargetOpNode],
    member_dems: Sequence[tuple[int, int, int, int, int]],
    decision: MoERouteDecision,
    floors: SkuTargetFloors,
    p0: Mapping[str, Any],
    *,
    level: int,
) -> tuple[TargetOpNode, tuple[int, int, int, int, int], dict[str, Any]]:
    first = members[0]
    member_occs = frozenset(m.occurrence for m in members)
    combines = [m for m in members if _classify_moe_frame_role(m) == "combine"]
    last_combine = combines[-1] if combines else members[-1]
    raw_read = sum(float(m.hbm_read_bytes) for m in members)
    raw_write = sum(float(m.hbm_write_bytes) for m in members)
    internal_writes = sum(float(m.hbm_write_bytes) for m in members if m.occurrence != last_combine.occurrence)
    elided_act = min(raw_read, raw_write, internal_writes)
    fused_read = max(0.0, raw_read - elided_act)
    fused_write = max(0.0, raw_write - elided_act)
    fused_hbm = fused_read + fused_write
    fused_flops = sum(float(m.flops) for m in members)
    fused_vec = sum(float(m.vector_ops) for m in members)
    fused_mte2 = max(0.0, sum(float(m.mte2_bytes) for m in members) - elided_act)
    fused_mte3 = max(0.0, sum(float(m.mte3_bytes) for m in members) - elided_act)
    fused_fix = sum(float(m.fixpipe_bytes) for m in members)
    fused_link = sum(float(m.link_bytes) for m in members)

    cube_peak = float(p0.get("cube_peak_flops_per_s") or 0.0)
    vec_peak = float(p0.get("vector_peak_ops_per_s") or 0.0)
    hbm_peak = float(p0.get("hbm_peak_bytes_per_s") or 0.0)
    link_peak = float(p0.get("link_peak_bytes_per_s") or 0.0)
    merged_mflops: dict[str, float] = defaultdict(float)
    for m in members:
        m_spec = getattr(m, "spec", None)
        entries = getattr(m_spec, "matrix_flops", ()) if m_spec is not None else ()
        if entries:
            for dt, fl in entries:
                if float(fl) > 0.0:
                    merged_mflops[str(dt)] += float(fl)
        elif float(m.flops) > 0.0:
            merged_mflops[str(m.dtype or "bf16")] += float(m.flops)
    die_obj = p0.get("die")
    if merged_mflops and die_obj is not None:
        cube_ns_f = 0.0
        for dt, fl in merged_mflops.items():
            try:
                cube_ns_f += fl * NS_PER_SECOND / float(die_obj.peak(dt))
            except (ModelingError, ZeroDivisionError):
                if cube_peak > 0.0:
                    cube_ns_f += fl * NS_PER_SECOND / cube_peak
        cube_ns = int(cube_ns_f) if cube_ns_f > 0.0 else sum(d[0] for d in member_dems)
    else:
        cube_ns = int(fused_flops * NS_PER_SECOND / cube_peak) if cube_peak > 0.0 else sum(d[0] for d in member_dems)
    vec_ns = int(fused_vec * NS_PER_SECOND / vec_peak) if vec_peak > 0.0 else sum(d[1] for d in member_dems)
    hbm_ns = int(fused_hbm * NS_PER_SECOND / hbm_peak) if hbm_peak > 0.0 else sum(d[2] for d in member_dems)
    link_ns = int(fused_link * NS_PER_SECOND / link_peak) if link_peak > 0.0 else sum(d[4] for d in member_dems)
    fused_dem = (cube_ns, vec_ns, hbm_ns, 0, link_ns)

    l_star_us = float(max(fused_dem)) / NS_PER_US
    fused_dur_us = max(l_star_us, float(floors.launch_cube_us) + float(floors.mc2_sync_us))
    raw_dur_us = sum(float(m.dur_us) for m in members)
    ext_preds = tuple(dict.fromkeys(p for m in members for p in m.predecessors if p not in member_occs))
    dom_map = (("cube_bound", cube_ns), ("vector_bound", vec_ns), ("memory_bound", hbm_ns), ("link_bound", link_ns))
    dom_bound = max(dom_map, key=lambda kv: kv[1])[0]
    eff = min(1.0, l_star_us / fused_dur_us) if (fused_dur_us > 0.0 and l_star_us > 0.0) else None
    new_oid = f"{first.archetype_id}.fused_moe_l{level}"
    new_occ = replace(first.occurrence, op_id=new_oid)
    bstate: BindingState = "unbound_formula_projection" if decision.needs_new_kernel else "bound_implementation"
    kstatus = STATUS_EXTEND_ABI if decision.needs_new_kernel else STATUS_AT_FLOOR
    fused_spec = (
        replace(
            first.spec, op_id=new_oid, flops=fused_flops,
            matrix_flops=tuple(sorted(merged_mflops.items())),
        )
        if first.spec is not None else None
    )

    fused_node = replace(
        first,
        occurrence=new_occ, op_id=new_oid, family="moe", recipe=decision.abi_id,
        stream="main", stream_index=0, task_type="MIX", predecessors=ext_preds,
        dur_us=fused_dur_us, exposed_dur_us=fused_dur_us, is_overlapped=False, masked_by="",
        flops=fused_flops, vector_ops=fused_vec, hbm_read_bytes=fused_read, hbm_write_bytes=fused_write,
        hbm_total_bytes=fused_hbm, mte2_bytes=fused_mte2, mte3_bytes=fused_mte3,
        fixpipe_bytes=fused_fix, link_bytes=fused_link, dominant_bound=dom_bound,
        l_star_lean_us=l_star_us, gap_us=max(0.0, fused_dur_us - l_star_us),
        gap_ratio=(fused_dur_us / l_star_us) if l_star_us > 0.0 else None,
        roofline_efficiency=eff,
        opt_attainment_pct=round(eff * PERCENT_SCALE, 1) if eff is not None else None,
        absorbed_source_ops=tuple(m.op_id for m in members),
        ascend_kernel=decision.abi_id,
        kernel_backend=BACKEND_ASCENDC_LABEL if decision.needs_new_kernel else BACKEND_CANN_LABEL,
        kernel_status=kstatus,
        needs_new_kernel=decision.needs_new_kernel, binding_state=bstate,
        is_shipping_abi=decision.is_shipping_abi, persistent_whole_die=True,
        opt_pillar="Pillar-1/MC2", spec=fused_spec,
    )
    receipt = {
        "edit_id": f"{first.archetype_id}_moe_fused_mc2_l{level}",
        "opcode": "MoERouteFusion",
        "archetype_id": first.archetype_id,
        "repeat_count": int(first.repeat_count),
        "input_occurrences": [m.op_id for m in members],
        "output_occurrences": [new_oid],
        "abi_id": decision.abi_id,
        "needs_new_kernel": bool(decision.needs_new_kernel),
        "is_shipping_abi": bool(decision.is_shipping_abi),
        "stream_core_budget": "",
        "before_us": round(raw_dur_us, 3),
        "after_us": round(fused_dur_us, 3),
        "saved_us": round(max(0.0, raw_dur_us - fused_dur_us), 3),
    }
    return fused_node, fused_dem, receipt


def _build_fused_moe_candidate(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    case: Case,
    floors: SkuTargetFloors,
    p0: Mapping[str, Any],
    *,
    locked_ids: set[str],
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]],
    allow_unbound_formula: bool,
    level: int,
) -> tuple[list[TargetOpNode], list[tuple[int, int, int, int, int]], MoERouteDecision, list[dict[str, Any]]] | None:
    """Build a legal persistent whole-die ``fused_mc2`` candidate from discrete MoE frames."""
    if not any(_classify_moe_frame_role(n) == "dispatch" for n in nodes):
        return None
    decision = moe_route_legality(
        case, enable_fused_mc2=level, multistream_overlap_shared_expert=False,
        nodes=nodes, allow_unbound_formula=allow_unbound_formula,
    )
    if not decision.legal:
        return None

    by_arch: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(nodes):
        by_arch[node.archetype_id].append(i)

    consumed: set[int] = set()
    replacements: dict[int, tuple[TargetOpNode, tuple[int, int, int, int, int]]] = {}
    remap: dict[OpOccurrence, OpOccurrence] = {}
    receipts: list[dict[str, Any]] = []

    for idxs in by_arch.values():
        arch_nodes = [nodes[i] for i in idxs]
        if any(is_persistent_whole_die_node(n) for n in arch_nodes):
            continue
        roles = {i: _classify_moe_frame_role(nodes[i]) for i in idxs}
        routed = [
            i for i in idxs
            if roles[i] in {"dispatch", "gmm", "combine"}
            and nodes[i].op_id not in locked_ids
            and nodes[i].occurrence not in collective_plans
        ]
        found_roles = {roles[i] for i in routed}
        if not {"dispatch", "gmm", "combine"} <= found_roles:
            continue
        frame = list(routed)
        if level == FUSED_MC2_SHARED_LEVEL:
            shared = [
                i for i in idxs
                if roles[i] == "shared" and nodes[i].op_id not in locked_ids
                and nodes[i].occurrence not in collective_plans
            ]
            frame.extend(shared)
            frame.sort()
        member_occs = frozenset(nodes[i].occurrence for i in frame)
        if _has_external_cycle_bridge(arch_nodes, member_occs):
            continue
        fused_node, fused_dem, rec = _collapse_moe_archetype_group(
            [nodes[i] for i in frame], [demands[i] for i in frame], decision, floors, p0, level=level,
        )
        consumed.update(frame)
        replacements[frame[0]] = (fused_node, fused_dem)
        receipts.append(rec)
        for i in frame:
            remap[nodes[i].occurrence] = fused_node.occurrence

    if not replacements:
        return None
    out_nodes: list[TargetOpNode] = []
    out_dems: list[tuple[int, int, int, int, int]] = []
    for i, (node, dem) in enumerate(zip(nodes, demands)):
        if i in replacements:
            out_nodes.append(replacements[i][0])
            out_dems.append(replacements[i][1])
        elif i not in consumed:
            if int(node.stream_index) == 1 and is_shared_expert_op_id(node.op_id):
                node = replace(node, stream="main", stream_index=0)
            out_nodes.append(node)
            out_dems.append(dem)
    return _rewire_node_preds(out_nodes, remap), out_dems, decision, receipts


def _select_best_moe_and_die_schedule(
    enveloped: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    case: Case,
    floors: SkuTargetFloors,
    p0: Mapping[str, Any],
    *,
    locked_ids: set[str],
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]],
    topology: Topology | None,
    primitive_table: Any,
    allow_unbound_formula: bool,
) -> tuple[DieScheduleResult, list[tuple[int, int, int, int, int]], MoERouteDecision | None, list[dict[str, Any]]]:
    """Enforce MoE mutex and pick the lowest-makespan legal MoE + core-split schedule on Engine."""
    col_plans = collective_plans or None

    def _run_die_engine(cand_nodes: Sequence[TargetOpNode]) -> DieScheduleResult:
        shared = schedule_target_graph_on_engine(
            cand_nodes, case, collective_plans=col_plans, topology=topology,
            primitive_table=primitive_table, launch_floor_us=floors.launch_vec_us,
        )
        return evaluate_disjoint_core_budgets(
            cand_nodes, case, shared_result=shared, collective_plans=col_plans,
            topology=topology, primitive_table=primitive_table, launch_floor_us=floors.launch_vec_us,
        )

    base_nodes, base_moe = enforce_moe_mutex_on_nodes(
        enveloped, case, allow_unbound_formula=allow_unbound_formula,
    )
    best_sched = _run_die_engine(base_nodes)
    best_dems = list(demands)
    best_moe = base_moe
    best_receipts: list[dict[str, Any]] = []
    best_w = weighted_step_span_us(best_sched.nodes)

    if base_moe is None or any(is_persistent_whole_die_node(n) for n in base_nodes):
        return best_sched, best_dems, best_moe, best_receipts

    mso_leg = moe_route_legality(
        case, enable_fused_mc2=0, multistream_overlap_shared_expert=True,
        nodes=base_nodes, allow_unbound_formula=allow_unbound_formula,
    )
    if mso_leg.legal:
        mso_nodes = build_discrete_mso_candidate_nodes(base_nodes)
        if mso_nodes is not None and any(
            int(n.stream_index) != int(b.stream_index) for n, b in zip(mso_nodes, base_nodes)
        ) and not any(
            n.op_id in locked_ids and int(n.stream_index) != int(b.stream_index)
            for n, b in zip(mso_nodes, base_nodes)
        ):
            mso_sched = _run_die_engine(mso_nodes)
            mso_w = weighted_step_span_us(mso_sched.nodes)
            if mso_w + SAVING_POSITIVE_EPS_US < best_w:
                moved_ops = [n.op_id for n, b in zip(mso_nodes, base_nodes) if int(n.stream_index) != int(b.stream_index)]
                best_receipts = [{
                    "edit_id": "moe_discrete_mso_stream1",
                    "opcode": "MoERouteMSO",
                    "archetype_id": "step",
                    "repeat_count": 1,
                    "input_occurrences": moved_ops,
                    "output_occurrences": moved_ops,
                    "abi_id": mso_leg.abi_id,
                    "needs_new_kernel": False,
                    "is_shipping_abi": True,
                    "stream_core_budget": mso_sched.stream_core_budget_summary,
                    "before_us": round(best_w, 3),
                    "after_us": round(mso_w, 3),
                    "saved_us": round(max(0.0, best_w - mso_w), 3),
                }]
                best_sched, best_dems, best_moe, best_w = mso_sched, list(demands), mso_leg, mso_w

    for lvl in (FUSED_MC2_SHARED_LEVEL, FUSED_MC2_ROUTED_LEVEL, 1):
        fused_cand = _build_fused_moe_candidate(
            base_nodes, demands, case, floors, p0, locked_ids=locked_ids,
            collective_plans=collective_plans, allow_unbound_formula=allow_unbound_formula, level=lvl,
        )
        if fused_cand is None:
            continue
        f_nodes, f_dems, f_dec, f_recs = fused_cand
        f_sched = _run_die_engine(f_nodes)
        f_w = weighted_step_span_us(f_sched.nodes)
        if f_w + SAVING_POSITIVE_EPS_US < best_w:
            best_sched, best_dems, best_moe, best_receipts, best_w = f_sched, f_dems, f_dec, f_recs, f_w

    return best_sched, best_dems, best_moe, best_receipts


def _build_applied_edit_receipts(
    final_plan: RewritePlan,
    specs: Sequence[KernelSynthesisSpec],
    indexed_nodes: Sequence[TargetOpNode],
    die_sched: DieScheduleResult,
    extra_receipts: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Emit structured producer receipts for verified rewrite, MoE, and core-split edits."""
    node_by_id = {n.op_id: n for n in indexed_nodes}
    spec_by_out: dict[str, KernelSynthesisSpec] = {
        s.constituent_op_ids[0]: s for s in specs if s.constituent_op_ids
    }
    receipts: list[dict[str, Any]] = []
    for edit in final_plan.edits:
        in_ops = [ref.op_id for ref in getattr(edit, "members", ())] or (
            [edit.target.op_id] if hasattr(edit, "target") and hasattr(edit.target, "op_id") else []
        )
        out_id = str(getattr(edit, "fused_op_id", "") or (in_ops[0] if in_ops else ""))
        node = node_by_id.get(out_id)
        kspec = spec_by_out.get(in_ops[0]) if in_ops else None
        before_us = float(kspec.before_us) if kspec is not None else (float(node.dur_us) if node is not None else 0.0)
        after_us = float(node.dur_us) if node is not None else (float(kspec.target_us) if kspec is not None else 0.0)
        receipts.append({
            "edit_id": str(edit.edit_id),
            "opcode": type(edit).__name__,
            "archetype_id": str(node.archetype_id) if node is not None else "",
            "repeat_count": int(node.repeat_count) if node is not None else 1,
            "input_occurrences": list(in_ops),
            "output_occurrences": [out_id] if out_id else [],
            "abi_id": str((node.ascend_kernel if node is not None else "") or (kspec.abi_signature_id if kspec is not None else "")),
            "needs_new_kernel": bool(node.needs_new_kernel) if node is not None else False,
            "is_shipping_abi": bool(node.is_shipping_abi) if node is not None else True,
            "stream_core_budget": die_sched.stream_core_budget_summary,
            "before_us": round(before_us, 3),
            "after_us": round(after_us, 3),
            "saved_us": round(max(0.0, before_us - after_us), 3),
        })
    receipts.extend(dict(r) for r in extra_receipts)
    if die_sched.stream_core_budget_summary and "disjoint" in die_sched.stream_core_budget_summary:
        receipts.append({
            "edit_id": "die_stream_core_budget_disjoint",
            "opcode": "StreamCoreBudget",
            "archetype_id": "step",
            "repeat_count": 1,
            "input_occurrences": [],
            "output_occurrences": [],
            "abi_id": "stream_core_budget_disjoint",
            "needs_new_kernel": False,
            "is_shipping_abi": True,
            "stream_core_budget": die_sched.stream_core_budget_summary,
            "before_us": 0.0,
            "after_us": round(float(die_sched.step_span_us), 3),
            "saved_us": 0.0,
        })
    return tuple(receipts)


def _p20_host_h2d_params(case: Case, primitive_table: Any) -> tuple[float, float]:
    """Read calibrated ``P20_host_dma_sync`` H2D bandwidth (B/s) and startup (ns)."""
    h2d_bps: float | None = None
    startup_ns: float | None = None
    cell_fn = getattr(primitive_table, "cell", None)
    if callable(cell_fn):
        try:
            cell = cell_fn("P20_host_dma_sync")
            bw_band = cell.band("host_h2d_bytes_per_s")
            if bw_band.p50 is not None and float(bw_band.p50) > 0.0:
                h2d_bps = float(bw_band.p50)
            st_band = cell.band("pinned_h2d_startup_ns")
            if st_band.p50 is not None and float(st_band.p50) > 0.0:
                startup_ns = float(st_band.p50)
        except ModelingError:
            pass
    if h2d_bps is None or startup_ns is None:
        sku = str(getattr(case.machine, "sku", "") or getattr(case.machine, "name", "") or "")
        cached = _cached_p20_metrics(sku)
        if h2d_bps is None and cached.get("h2d_bps"):
            h2d_bps = float(cached["h2d_bps"])
        if startup_ns is None and cached.get("h2d_startup_ns"):
            startup_ns = float(cached["h2d_startup_ns"])
    return (
        h2d_bps if (h2d_bps is not None and h2d_bps > 0.0) else P20_FALLBACK_H2D_BPS,
        startup_ns if (startup_ns is not None and startup_ns >= 0.0) else P20_FALLBACK_STARTUP_NS,
    )


def _peak_step_int(raw: Any, default: int) -> int:
    if isinstance(raw, (tuple, list)):
        vals = [int(x) for x in raw if x is not None]
        return max(vals) if vals else int(default)
    return int(raw) if raw is not None else int(default)


def _peak_sequences_per_dp(step_request: StepRequest | None, case: Case) -> int:
    if step_request is not None:
        return max(1, _peak_step_int(getattr(step_request, "sequences", 1), 1))
    return max(1, len(case.workload.requests))


def _is_recover_ssm_commit_node(node: TargetOpNode) -> bool:
    """True strictly for post-accept RecoverSSM commit ops (never the [B, K+1] verify scan)."""
    meta = dict(getattr(node.spec, "metadata", ()) or ()) if node.spec is not None else {}
    stage = str(meta.get("recover_ssm_stage") or "").strip().lower()
    if stage:
        return stage == "commit"
    op_lower = str(node.op_id or "").lower()
    rec_lower = str(node.recipe or "").lower()
    if "verify" in op_lower or "verify" in rec_lower or "recurrent_gated_delta_rule" in rec_lower:
        return False
    return (
        "recover_ssm_commit" in op_lower
        or "recover_ssm_commit" in rec_lower
        or "aclnnrecoverssmstate" in rec_lower
    )


def _max_pp_stage_bytes(case: Case, target_plans: Sequence[Any], per_layer_bytes: Any) -> int:
    """Return the bottleneck single-stage weight byte sum across pipeline stages."""
    pp = max(1, int(case.runtime.parallel.pp))
    if pp <= 1:
        return sum(int(per_layer_bytes(lp)) for lp in target_plans)
    total_layers = max(1, int(getattr(case.model, "layers", len(case.model.layer_plans)) or 1))
    custom = tuple(getattr(case.runtime, "pp_layer_partition", ()) or ())
    if len(custom) == pp and sum(int(x) for x in custom) == total_layers:
        counts = [int(x) for x in custom]
    else:
        base, rem = divmod(total_layers, pp)
        counts = [base + (1 if idx < rem else 0) for idx in range(pp)]
    layer_to_stage: dict[int, int] = {}
    cursor = 0
    for st, cnt in enumerate(counts):
        for lyr in range(cursor, cursor + cnt):
            layer_to_stage[lyr] = st
        cursor += cnt
    stage_totals = [0] * pp
    for lp in target_plans:
        st = layer_to_stage.get(int(getattr(lp, "index", 0)), 0)
        stage_totals[st] += int(per_layer_bytes(lp))
    return max(stage_totals, default=0)


def _engram_lookup_h2d_bytes(nodes: Sequence[TargetOpNode]) -> float:
    """Extract host-link lookup H2D bytes only, excluding device GEMMs like ``engram_gate*``."""
    rh2d_nodes = [
        n for n in nodes
        if str(n.family).lower() == "engram_rh2d"
        or str(n.recipe).lower() in {"engram_cpu_decode_pinned_staging_h2d", "ple_host_uva_lookup"}
    ]
    if rh2d_nodes:
        total = 0.0
        for n in rh2d_nodes:
            meta = dict(getattr(n.spec, "metadata", ()) or ()) if n.spec is not None else {}
            rwork = meta.get("resource_work")
            h2d = float(rwork.get("h2d", 0.0)) if isinstance(rwork, Mapping) else 0.0
            if h2d <= 0.0:
                h2d = max(float(n.hbm_write_bytes), float(n.mte3_bytes), float(n.hbm_total_bytes))
            total += h2d * max(1, int(n.repeat_count))
        return total

    gather_nodes = [
        n for n in nodes
        if str(n.family).lower() == "engram_gather"
        or str(n.recipe).lower() in {"catalog_engram_device_gather", "ple_device_lookup"}
    ]
    total = 0.0
    for n in gather_nodes:
        meta = dict(getattr(n.spec, "metadata", ()) or ()) if n.spec is not None else {}
        h2d = float(meta.get("logical_table_read_bytes") or 0.0)
        if h2d <= 0.0 and meta.get("response_bytes_per_lookup") and meta.get("lookups"):
            h2d = float(meta["response_bytes_per_lookup"]) * float(meta["lookups"])
        if h2d <= 0.0:
            h2d = max(float(n.hbm_read_bytes), float(n.mte2_bytes), float(n.hbm_total_bytes))
        total += h2d * max(1, int(n.repeat_count))
    return total


def _resident_free_headroom_bytes(case: Case, step_request: StepRequest | None) -> int:
    try:
        mem = Memory(case)
        seqs = _peak_sequences_per_dp(step_request, case)
        ctx = max(0, _peak_step_int(getattr(step_request, "context", 0), 0)) if step_request is not None else 0
        q_tok = max(1, _peak_step_int(getattr(step_request, "tokens", 1), 1)) if step_request is not None else 1
        receipt = mem.resident_memory_receipt(
            phase=str(getattr(step_request, "phase", "decode") or "decode"),
            sequence_tokens=(ctx + q_tok) if ctx > 0 else None,
            batch_per_dp=seqs,
            query_tokens=q_tok,
        )
        return min((int(d.free_hbm_headroom_bytes) for d in receipt.dies), default=0)
    except (ModelingError, ValueError, ZeroDivisionError):
        return 0


def _collect_space_time_opportunities(
    case: Case,
    nodes: Sequence[TargetOpNode],
    *,
    step_request: StepRequest | None,
) -> list[CoDesignOpportunity]:
    """Build typed ``SpaceTimeTradeoffSpec`` records backed by ``ResidentMemoryLedger``."""
    out: list[CoDesignOpportunity] = []
    headroom = _resident_free_headroom_bytes(case, step_request)
    meta = dict(getattr(case.runtime, "metadata", ()) or ())
    k_prop = int(meta.get("draft_proposal_tokens", 0) or 0)
    if k_prop <= 0 and step_request is not None and step_request.phase == "decode":
        k_prop = max(0, _peak_step_int(getattr(step_request, "tokens", 1), 1) - 1)
    seqs = _peak_sequences_per_dp(step_request, case)
    plans = tuple(getattr(case.model, "layer_plans", ()) or ())
    linear_plans = [lp for lp in plans if str(getattr(lp, "attention", "")).lower() in {"kda", "gdn"}]
    recover_us = sum(
        float(n.exposed_dur_us) * max(1, int(n.repeat_count))
        for n in nodes
        if _is_recover_ssm_commit_node(n)
    )
    if linear_plans and (k_prop > 0 or recover_us > 0.0):
        eff_k = max(1, k_prop)
        try:
            bd = recover_ssm_snapshot_ring_breakdown(case, seqs, eff_k)
            delta_b = int(bd.delta_hbm_vs_recover_ssm_bytes)
        except (ModelingError, ValueError):
            delta_b = 0
        direction = "space_for_time" if delta_b >= 0 else "time_for_space"
        feasible = bool(delta_b <= headroom) if delta_b > 0 else True
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_recover_ssm_snapshot_ring",
            kind="space_time_tradeoff",
            track="framework",
            readiness="proposed_pr",
            title=f"RecoverSSM Snapshot Ring [B={seqs}, K+1={eff_k + 1}] vs {len(linear_plans)}-Layer Recompute",
            evidence_refs=("ResidentMemoryLedger", "recover_ssm_snapshot_ring_breakdown"),
            space_time=SpaceTimeTradeoffSpec(
                direction=direction,
                mechanism="recover_ssm_snapshot_ring",
                delta_hbm_bytes=delta_b,
                free_hbm_headroom_bytes=headroom,
                delta_step_ms=-(recover_us / US_PER_MS),
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=feasible,
            ),
        ))

    tp = max(1, int(case.runtime.parallel.tp))
    mla_plans = [lp for lp in plans if int(getattr(lp, "kv_lora_rank", 0) or 0) > 0]
    if mla_plans:
        def _mla_absorbed_layer_bytes(lp: Any) -> int:
            kv_r = int(getattr(lp, "kv_lora_rank", 0) or 0)
            n_heads = max(1, int(getattr(lp, "num_heads", 1) or 1))
            v_dim = max(1, int(getattr(lp, "v_head_dim", 0) or getattr(lp, "head_dim", 1) or 1))
            qk_nope = max(1, int(getattr(lp, "qk_nope_head_dim", 0) or getattr(lp, "head_dim", 1) or 1))
            return (kv_r * n_heads * (qk_nope + v_dim) * DEFAULT_BF16_BYTES) // tp

        prep_bytes = _max_pp_stage_bytes(case, mla_plans, _mla_absorbed_layer_bytes)
        feasible_mla = bool(prep_bytes <= headroom)
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_absorbed_kv_projection",
            kind="space_time_tradeoff",
            track="framework",
            readiness="shipping",
            title=f"Absorbed MLA Projection Weights vs Factorized ({len(mla_plans)} MLA Layers)",
            evidence_refs=("ResidentMemoryLedger",),
            space_time=SpaceTimeTradeoffSpec(
                direction="space_for_time",
                mechanism="absorbed_kv_projection",
                delta_hbm_bytes=int(prep_bytes),
                free_hbm_headroom_bytes=headroom,
                delta_step_ms=0.0,
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=feasible_mla,
            ),
        ))
    elif linear_plans:
        hidden = max(1, int(getattr(case.model, "hidden_size", 0) or 1))

        def _linear_prep_layer_bytes(lp: Any) -> int:
            lh = max(1, int(getattr(lp, "linear_heads", 1) or 1))
            ld = max(1, int(getattr(lp, "linear_head_dim", 1) or 1))
            return (hidden * lh * ld * DEFAULT_BF16_BYTES) // tp

        prep_f_bytes = _max_pp_stage_bytes(case, linear_plans, _linear_prep_layer_bytes)
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_prepared_weight_materialization",
            kind="space_time_tradeoff",
            track="framework",
            readiness="shipping",
            title=f"Prepared Materialized F = F_b @ F_a vs Factorized ({len(linear_plans)} Linear Layers)",
            evidence_refs=("ResidentMemoryLedger",),
            space_time=SpaceTimeTradeoffSpec(
                direction="space_for_time",
                mechanism="prepared_weight_materialization",
                delta_hbm_bytes=int(prep_f_bytes),
                free_hbm_headroom_bytes=headroom,
                delta_step_ms=0.0,
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=bool(prep_f_bytes <= headroom),
            ),
        ))
    return out


def _collect_cross_hw_offload_opportunities(
    case: Case,
    nodes: Sequence[TargetOpNode],
    *,
    primitive_table: Any,
) -> list[CoDesignOpportunity]:
    """Build ``CrossHwOffloadSpec`` for Engram/PLE Host DDR async H2D prefetch vs hide window."""
    plans = tuple(getattr(case.model, "layer_plans", ()) or ())
    engram_layers = [int(lp.index) for lp in plans if bool(getattr(lp, "engram", False))]
    engram_cfg = getattr(case.model, "engram", None)
    engram_tables = tuple(getattr(engram_cfg, "tables", ()) or ()) if engram_cfg is not None else ()
    engram_nodes = [
        n for n in nodes
        if str(n.family).lower() in {"engram_rh2d", "engram_gather", "ple", "engram"}
        or "engram" in n.op_id.lower()
        or "ple_" in n.op_id.lower()
    ]
    if not engram_layers and not engram_tables and not engram_nodes:
        return []

    first_k = (
        min(engram_layers) if engram_layers
        else (min((int(t.layer_id) for t in engram_tables), default=1) if engram_tables else 1)
    )
    hide_us = 0.0
    for node in nodes:
        lyr = getattr(node.spec, "layer", None) if node.spec is not None else None
        if lyr is not None and 0 <= int(lyr) < first_k:
            hide_us += float(node.exposed_dur_us) * max(1, int(node.repeat_count))
    if hide_us <= 0.0:
        hide_us = sum(
            float(n.exposed_dur_us) * max(1, int(n.repeat_count))
            for n in nodes if n not in engram_nodes
        ) * HALF_SPLIT_RATIO

    xfer_bytes = _engram_lookup_h2d_bytes(engram_nodes)
    freed_hbm = 0
    if engram_cfg is not None and callable(getattr(engram_cfg, "table_storage_bytes", None)):
        try:
            freed_hbm = int(round(float(engram_cfg.table_storage_bytes())))
        except (ModelingError, ValueError):
            freed_hbm = 0
    if freed_hbm <= 0:
        for tbl in engram_tables:
            rows = max(1, int(getattr(tbl, "rows", 0) or 1))
            width = max(1, int(getattr(tbl, "width", 0) or case.model.hidden_size))
            freed_hbm += rows * width * DEFAULT_BF16_BYTES
    if freed_hbm <= 0:
        freed_hbm = max(int(round(xfer_bytes)), len(engram_layers) * max(1, case.model.hidden_size) * DEFAULT_BF16_BYTES)
    if xfer_bytes <= 0.0:
        xfer_bytes = float(max(1, len(engram_layers)) * max(1, case.model.hidden_size) * DEFAULT_BF16_BYTES)

    h2d_bps, startup_ns = _p20_host_h2d_params(case, primitive_table)
    transfer_ms = (startup_ns / NS_PER_MS) + (xfer_bytes / h2d_bps) * MS_PER_SECOND
    hide_ms = max(0.0, hide_us / US_PER_MS)
    net_exposed_ms = max(0.0, transfer_ms - hide_ms)
    return [CoDesignOpportunity(
        opportunity_id="cross_hw_engram_ple_host_ddr_prefetch",
        kind="cross_hw_offload_multistream",
        track="framework",
        readiness="shipping",
        title=f"Engram/PLE Host DDR Async H2D Prefetch vs NPU Layers 0..{max(0, first_k - 1)} Hide Window",
        evidence_refs=("P20_host_dma_sync",),
        cross_hw=CrossHwOffloadSpec(
            target_component="engram_ple_table",
            source_tier="host_ddr",
            transfer_engine="h2d",
            compute_hide_window_ms=hide_ms,
            transfer_ms=transfer_ms,
            net_exposed_ms=net_exposed_ms,
            freed_hbm_bytes=int(freed_hbm),
            delta_concurrency_per_dp=0,
            delta_tps_per_die=0.0,
        ),
    )]


def _resolve_tilelang_dual_tier_bounds(
    *,
    datapath: DatapathCapabilitySpec,
    before_ms: float,
    l_star_ms: float,
    tiled_ms: float,
    ub_live_bytes: int,
    has_cube: bool,
    has_aiv: bool,
    cube_dtype: str = "bf16",
    vector_ops: float = 0.0,
    is_vector_or_norm_region: bool = True,
) -> tuple[float | None, float | None, str | None]:
    """Compute calibrated TileLang ``T.SimtVF`` quick-win vs ``T.SimdVF / AscendC`` ceiling gains."""
    cal = try_load_tilelang_calibration(str(datapath.arch_id))
    if cal is None:
        return None, None, None
    simd_target_ms = max(l_star_ms, tiled_ms)
    ceil_gain_ms = before_ms - l_star_ms
    simd_gain_ms = min(ceil_gain_ms, before_ms - simd_target_ms)
    ub_warn = (
        cal.simt_vs_simd_ub_wall_warning(ub_live_bytes)
        if (is_vector_or_norm_region or has_aiv) else None
    )
    if (is_vector_or_norm_region or has_aiv) and ub_live_bytes > cal.simt_usable_ub_bytes:
        # T.SimtVF exceeds the 216 KiB UB ceiling (ACL 107000) and yields zero quick-win gain.
        return min(0.0, simd_gain_ms), simd_gain_ms, ub_warn

    disp_ms = cal.aclgraph_issue_us / US_PER_MS
    cube_tl_ms = 0.0
    if has_cube:
        eta_cube = cal.cube_attainment(
            cube_dtype, blockscaled=(str(cube_dtype or "").strip().lower() == "mxfp8"),
        )
        cube_tl_ms = tiled_ms / eta_cube
    if has_cube and not has_aiv:
        t_quick_ms = max(simd_target_ms, cube_tl_ms, disp_ms)
    else:
        dma_simt_ms = tiled_ms / cal.dma_attainment(uses_simt=True, rw_pattern="rw21")
        vec_simt_ms = 0.0
        if vector_ops > 0.0:
            aiv_count = max(1, int(datapath.aiv_count))
            simt_ops_per_ms = (
                float(aiv_count)
                * cal.vector_ops_per_s_per_aiv(uses_simt=True, op_kind="ub_fma")
                / MS_PER_SECOND
            )
            vec_simt_ms = float(vector_ops) / simt_ops_per_ms
        t_quick_ms = max(simd_target_ms, dma_simt_ms, vec_simt_ms, cube_tl_ms, disp_ms)
    raw_simt_gain = before_ms - t_quick_ms
    if simd_gain_ms >= 0.0:
        simt_gain_ms = min(simd_gain_ms, max(0.0, raw_simt_gain))
    else:
        simt_gain_ms = min(simd_gain_ms, raw_simt_gain)
    return simt_gain_ms, simd_gain_ms, ub_warn


def _collect_codesign_opportunities(
    case: Case,
    nodes: Sequence[TargetOpNode],
    specs: Sequence[KernelSynthesisSpec],
    tiler_opps: Sequence[CoDesignOpportunity],
    *,
    primitive_table: Any,
    step_request: StepRequest | None,
) -> tuple[CoDesignOpportunity, ...]:
    """Assemble typed ``MegaFusionBound``, ``SpaceTimeTradeoffSpec`` & ``CrossHwOffloadSpec`` records."""
    datapath = datapath_for_die(case.machine.dies[0])
    node_by_spec_id = {
        n.synthesis_spec.spec_id: n
        for n in nodes
        if n.synthesis_spec is not None and n.synthesis_spec.spec_id
    }
    opps: list[CoDesignOpportunity] = []
    for spec in specs:
        if not spec.sram_verified or len(spec.constituent_op_ids) < MIN_FUSE_MEMBERS:
            continue
        exposed_us = spec.exposed_before_us if spec.exposed_before_us is not None else spec.before_us
        before_ms = max(0.0, float(exposed_us) / US_PER_MS)
        tiled_ms = max(0.0, float(spec.target_us) / US_PER_MS)
        l_star_ms = min(tiled_ms, max(0.0, float(spec.l_star_us) / US_PER_MS))
        ub_live = max(0, int(spec.ub_peak_bytes))
        ub_cap = max(int(spec.ub_budget_bytes), ub_live)
        matched = node_by_spec_id.get(spec.spec_id)
        has_cube = (
            AlgebraicTrait.BILINEAR_MATMUL.value in spec.constituent_traits
            or (matched is not None and float(matched.flops) > 0.0)
        )
        has_aiv = any(t != AlgebraicTrait.BILINEAR_MATMUL.value for t in spec.constituent_traits) or not has_cube
        simt_gain_ms, simd_gain_ms, ub_warn = _resolve_tilelang_dual_tier_bounds(
            datapath=datapath,
            before_ms=before_ms,
            l_star_ms=l_star_ms,
            tiled_ms=tiled_ms,
            ub_live_bytes=ub_live,
            has_cube=has_cube,
            has_aiv=has_aiv,
            cube_dtype=str(matched.dtype or "bf16") if matched is not None else "bf16",
            vector_ops=float(matched.vector_ops) if matched is not None else 0.0,
            is_vector_or_norm_region=has_aiv,
        )
        is_shipping = spec.binding_state == "bound_implementation"
        opps.append(CoDesignOpportunity(
            opportunity_id=f"mega_{spec.spec_id}",
            kind="mega_fusion",
            track="framework" if is_shipping else "kernel_new",
            readiness="shipping" if is_shipping else "synthesized_target",
            title=f"{spec.kernel_name} ({spec.suggested_backend})",
            evidence_refs=tuple(spec.constituent_op_ids),
            mega_fusion=MegaFusionBound(
                region_op_ids=tuple(spec.constituent_op_ids),
                exposed_before_ms=before_ms,
                boundary_lower_bound_ms=l_star_ms,
                tiled_engine_ms=tiled_ms,
                gain_ceiling_ms=before_ms - l_star_ms,
                gain_expected_ms=before_ms - tiled_ms,
                eliminated_gm_bytes=max(0, int(spec.elided_hbm_bytes)),
                ub_live_bytes=ub_live,
                ub_capacity_bytes=ub_cap,
                requires_unabsorbed_split_cost_check=True,
                tilelang_simt_gain_ms=simt_gain_ms,
                tilelang_simd_or_ascendc_gain_ms=simd_gain_ms,
                ub_wall_warning=ub_warn or spec.ub_wall_warning,
            ),
        ))
    opps.extend(tiler_opps)
    opps.extend(_collect_space_time_opportunities(case, nodes, step_request=step_request))
    opps.extend(_collect_cross_hw_offload_opportunities(case, nodes, primitive_table=primitive_table))
    return tuple(opps)


def _run_archetype_rewrites(
    work_nodes: Sequence[TargetOpNode],
    work_dems: Sequence[tuple[int, int, int, int, int]],
    locked_ids: set[str],
    used_edit_ids: set[str],
    floors: SkuTargetFloors,
    p0: Mapping[str, Any],
    case: Case,
    datapath: DatapathCapabilitySpec,
    pricing_ctx: _TargetPricingContext,
    *,
    phase: str,
    safe_n_tok: float,
    primitive_table: Any,
    allow_unbound_formula: bool,
) -> tuple[
    list[TargetOpNode],
    list[tuple[int, int, int, int, int]],
    list[KernelSynthesisSpec],
    list[SemanticAuditTrigger],
    list[Edit],
    dict[OpOccurrence, OpOccurrence],
]:
    by_arch: dict[str, tuple[list[TargetOpNode], list[tuple[int, int, int, int, int]]]] = {}
    arch_order: list[str] = []
    for node, dem in zip(work_nodes, work_dems):
        if node.archetype_id not in by_arch:
            by_arch[node.archetype_id] = ([], [])
            arch_order.append(node.archetype_id)
        by_arch[node.archetype_id][0].append(node)
        by_arch[node.archetype_id][1].append(dem)
    work_unique_ids = {n.op_id for n in work_nodes}
    verify_work_ops = (
        [_node_to_op_spec(n, work_unique_ids, phase=phase) for n in work_nodes]
        if len(work_unique_ids) == len(work_nodes) else None
    )
    all_nodes: list[TargetOpNode] = []
    all_dems: list[tuple[int, int, int, int, int]] = []
    all_specs: list[KernelSynthesisSpec] = []
    all_triggers: list[SemanticAuditTrigger] = []
    auto_edits: list[Edit] = []
    remap: dict[OpOccurrence, OpOccurrence] = {}
    for aid in arch_order:
        a_nodes, a_dems = by_arch[aid]
        o_nodes, o_dems, o_specs, o_trigs, o_edits, o_remap = _solve_one_archetype(
            aid, a_nodes, a_dems, locked_ids, used_edit_ids, floors, p0, case, datapath, pricing_ctx,
            verify_ops=verify_work_ops, phase=phase, safe_n_tok=safe_n_tok, primitive_table=primitive_table,
            allow_unbound_formula=allow_unbound_formula,
        )
        all_nodes.extend(o_nodes)
        all_dems.extend(o_dems)
        all_specs.extend(o_specs)
        all_triggers.extend(o_trigs)
        auto_edits.extend(o_edits)
        remap.update(o_remap)
    return all_nodes, all_dems, all_specs, all_triggers, auto_edits, remap


def _build_stage_waterfall(
    opt_mode: OptimizationMode,
    stage_scheds: Mapping[str, DieScheduleResult],
) -> tuple[dict[str, Any], ...]:
    """Build canonical stage waterfall strictly from winning Engine schedules (zero scalar clamping)."""
    stage_specs = canonical_comparison_stages(opt_mode)
    before_us = weighted_step_span_us(stage_scheds["before"].nodes)
    rows: list[dict[str, Any]] = []
    prev_us = before_us
    for st in stage_specs:
        cur_us = weighted_step_span_us(stage_scheds[st.stage].nodes)
        rows.append({
            "stage": st.stage,
            "owner": st.owner,
            "step_span_us": cur_us,
            "delta_from_prev_us": cur_us - prev_us,
            "cumulative_saved_us": before_us - cur_us,
            "allowed_readiness": tuple(sorted(st.allowed_readiness)),
            "allowed_rewrite_tracks": st.allowed_rewrite_tracks,
            "engine_uid": "unified_die_engine",
        })
        prev_us = cur_us
    return tuple(rows)


def _enforce_engine_monotonicity(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    case: Case,
    floors: SkuTargetFloors,
    p0: Mapping[str, Any],
    datapath: DatapathCapabilitySpec,
    pricing_ctx: _TargetPricingContext,
    die_sched: DieScheduleResult,
    final_dems: list[tuple[int, int, int, int, int]],
    final_plan: RewritePlan,
    all_specs: list[KernelSynthesisSpec],
    tiler_opps: list[CoDesignOpportunity],
    extra_receipts: list[dict[str, Any]],
    moe_dec: MoERouteDecision | None,
    *,
    opt_mode: OptimizationMode,
    phase: str,
    safe_n_tok: float,
    base_sha: str,
    frame_sha: str,
    topology: Topology | None,
    primitive_table: Any,
    has_ext_edits: bool,
    compute_waterfall: bool,
) -> tuple[
    DieScheduleResult,
    list[tuple[int, int, int, int, int]],
    RewritePlan,
    list[KernelSynthesisSpec],
    list[CoDesignOpportunity],
    list[dict[str, Any]],
    MoERouteDecision | None,
    dict[str, DieScheduleResult],
]:
    """Compare stage Engine schedules and return the exact winning (sched, dems, plan, specs) tuple."""
    if not compute_waterfall and has_ext_edits:
        return die_sched, final_dems, final_plan, all_specs, tiler_opps, extra_receipts, moe_dec, {}

    orig_sched = schedule_target_graph_on_engine(
        nodes, case, collective_plans=None, topology=topology,
        primitive_table=primitive_table, launch_floor_us=floors.launch_vec_us,
    )
    orig_bundle = (orig_sched, list(demands), empty_plan(base_sha, frame_sha), [], [], [], None)
    orig_us = weighted_step_span_us(orig_sched.nodes)

    cfg_sched, cfg_dems, cfg_moe, cfg_receipts = _select_best_moe_and_die_schedule(
        nodes, demands, case, floors, p0, locked_ids=set(),
        collective_plans={}, topology=topology,
        primitive_table=primitive_table, allow_unbound_formula=False,
    )
    cfg_bundle = (
        (cfg_sched, list(cfg_dems), empty_plan(base_sha, frame_sha), [], [], list(cfg_receipts), cfg_moe)
        if weighted_step_span_us(cfg_sched.nodes) <= orig_us + SAVING_POSITIVE_EPS_US
        else orig_bundle
    )
    stage_scheds: dict[str, DieScheduleResult] = {
        "before": orig_sched,
        "config_only": cfg_bundle[0],
    }

    best_prior = cfg_bundle
    if opt_mode == "modeling":
        fwk_nodes, fwk_dems, fwk_specs, _, fwk_edits, fwk_remap = _run_archetype_rewrites(
            nodes, demands, set(), set(), floors, p0, case, datapath, pricing_ctx,
            phase=phase, safe_n_tok=safe_n_tok, primitive_table=primitive_table,
            allow_unbound_formula=False,
        )
        fwk_rewired = _rewire_node_preds(fwk_nodes, fwk_remap)
        fwk_sched, fwk_final_dems, fwk_moe, fwk_receipts = _select_best_moe_and_die_schedule(
            fwk_rewired, fwk_dems, case, floors, p0, locked_ids=set(),
            collective_plans={}, topology=topology,
            primitive_table=primitive_table, allow_unbound_formula=False,
        )
        fwk_plan = (
            RewritePlan(base_graph_sha256=base_sha, framing_sha256=frame_sha, edits=tuple(fwk_edits))
            if fwk_edits else empty_plan(base_sha, frame_sha)
        )
        if weighted_step_span_us(fwk_sched.nodes) <= weighted_step_span_us(best_prior[0].nodes) + SAVING_POSITIVE_EPS_US:
            best_prior = (
                fwk_sched, list(fwk_final_dems), fwk_plan, list(fwk_specs), [], list(fwk_receipts), fwk_moe,
            )
        stage_scheds["framework_rewrite"] = best_prior[0]

    if has_ext_edits:
        stage_scheds["after"] = die_sched
        return die_sched, final_dems, final_plan, all_specs, tiler_opps, extra_receipts, moe_dec, stage_scheds

    after_us = weighted_step_span_us(die_sched.nodes)
    prior_us = weighted_step_span_us(best_prior[0].nodes)
    if after_us > prior_us + SAVING_POSITIVE_EPS_US:
        win_sched, win_dems, win_plan, win_specs, win_tiler, win_rcpts, win_moe = best_prior
        stage_scheds["after"] = win_sched
        return win_sched, win_dems, win_plan, win_specs, win_tiler, win_rcpts, win_moe, stage_scheds

    stage_scheds["after"] = die_sched
    return die_sched, final_dems, final_plan, all_specs, tiler_opps, extra_receipts, moe_dec, stage_scheds


def _build_solved_target_graph(
    *,
    indexed: tuple[TargetOpNode, ...],
    final_dems: Sequence[tuple[int, int, int, int, int]],
    die_sched: DieScheduleResult,
    all_specs: Sequence[KernelSynthesisSpec],
    all_triggers: Sequence[SemanticAuditTrigger],
    final_plan: RewritePlan,
    verification_passed: bool,
    floors: SkuTargetFloors,
    eff_allow_unbound: bool,
    extra_receipts: Sequence[dict[str, Any]],
    moe_dec: MoERouteDecision | None,
    codesign_opps: tuple[CoDesignOpportunity, ...],
    stage_waterfall: tuple[dict[str, Any], ...],
    opt_mode: OptimizationMode,
) -> SolvedTargetGraph:
    edit_receipts = _build_applied_edit_receipts(final_plan, all_specs, indexed, die_sched, extra_receipts)
    unbound_ops = tuple(n.op_id for n in indexed if n.binding_state == "unbound_formula_projection")
    has_unbound = bool(unbound_ops)
    if not eff_allow_unbound and has_unbound:
        raise ModelingError(f"strict_simulation_unbound_formula_ops:{','.join(unbound_ops)}")
    graph_state: BindingState = "unbound_formula_projection" if has_unbound else "bound_implementation"
    return SolvedTargetGraph(
        nodes=indexed,
        demands_ns=tuple(final_dems),
        archetype_spans_us=die_sched.archetype_spans_us,
        synthesis_specs=tuple(all_specs),
        semantic_audit_triggers=tuple(all_triggers),
        applied_plan=final_plan,
        rewrite_plan_sha256=plan_sha256(final_plan),
        verification_passed=verification_passed,
        floors=floors,
        has_unbound_formula_ops=has_unbound,
        binding_state=graph_state,
        unbound_formula_ops=unbound_ops,
        applied_edit_receipts=edit_receipts,
        stream_core_budget=die_sched.stream_core_budget,
        stream_core_budget_summary=die_sched.stream_core_budget_summary,
        moe_route=moe_dec.to_dict() if moe_dec is not None else {},
        step_start_us=die_sched.step_start_us,
        step_end_us=die_sched.step_end_us,
        step_span_us=die_sched.step_span_us,
        co_design_opportunities=codesign_opps,
        stage_waterfall=stage_waterfall,
        optimization_mode=opt_mode,
    )


def solve_target_graph(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    case: Case,
    peaks_by_stage: Mapping[int, Mapping[str, Any]],
    *,
    phase: str = "decode",
    rewrite_plan: RewritePlan | Mapping[str, Any] | str | Path | None = None,
    n_tok_per_die: float = 1.0,
    primitive_table: Any = None,
    allow_unbound_formula: bool = True,
    optimization_mode: OptimizationMode | None = None,
    step_request: StepRequest | None = None,
    stack_id: str | None = None,
    scenario_case: Case | None = None,
    compute_waterfall_and_codesign: bool = True,
) -> SolvedTargetGraph:
    """Solve target-state graph on ONE Engine + ONE PrimitiveTable across causal stages."""
    opt_mode: OptimizationMode = (
        optimization_mode if optimization_mode is not None
        else ("modeling" if allow_unbound_formula else "simulation")
    )
    if opt_mode == "simulation" and allow_unbound_formula:
        raise ModelingError("simulation_mode_forbids_unbound_formula_ops")
    eff_allow_unbound = bool(allow_unbound_formula and opt_mode == "modeling")

    floors = extract_sku_floors(case, primitive_table=primitive_table)
    die = case.machine.dies[0]
    ub_edges = ub_peer_edges_for_rank(die.rank, case.machine.links, case.machine.transfer_channels)
    datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
    p0 = peaks_by_stage.get(0) or next(iter(peaks_by_stage.values()), {})
    ext_plan = load_rewrite_plan(rewrite_plan)
    pricing_ctx = _TargetPricingContext(case, primitive_table=primitive_table)
    safe_n_tok = max(float(n_tok_per_die), SAVING_POSITIVE_EPS_US)

    all_known_ids = {n.op_id for n in nodes}
    base_ops_all = [_node_to_op_spec(n, all_known_ids, phase=phase) for n in nodes]
    base_scenario = _build_target_scenario(
        case, base_ops_all, datapath, phase=phase, primitive_table=primitive_table,
    )
    all_specs: list[KernelSynthesisSpec] = []
    executed_ext_edits: list[Edit] = []
    locked_ids: set[str] = set()
    global_remap: dict[OpOccurrence, OpOccurrence] = {}
    collective_plans: dict[OpOccurrence, tuple[Any, float]] = {}

    rebind_ext = (
        _verify_external_plan_digests(
            ext_plan, scenario_case or case, base_scenario,
            step_request=step_request, stack_id=stack_id, primitive_table=primitive_table,
        )
        if ext_plan is not None else False
    )
    if ext_plan is not None and ext_plan.edits:
        work_nodes, work_dems, ext_specs, locked_ids, ext_remap, ext_col_plans = _materialize_external_plan(
            nodes, demands, ext_plan, case, datapath, floors, p0, pricing_ctx,
            phase=phase, primitive_table=primitive_table,
            allow_unbound_formula=eff_allow_unbound, rebind_digests=rebind_ext,
        )
        all_specs.extend(ext_specs)
        executed_ext_edits.extend(ext_plan.edits)
        global_remap.update(ext_remap)
        collective_plans.update(ext_col_plans)
    else:
        work_nodes = list(nodes)
        work_dems = list(demands)

    used_edit_ids = {e.edit_id for e in executed_ext_edits}
    all_nodes, all_dems, auto_specs, all_triggers, auto_edits, auto_remap = _run_archetype_rewrites(
        work_nodes, work_dems, locked_ids, used_edit_ids, floors, p0, case, datapath, pricing_ctx,
        phase=phase, safe_n_tok=safe_n_tok, primitive_table=primitive_table,
        allow_unbound_formula=eff_allow_unbound,
    )
    all_specs.extend(auto_specs)
    global_remap.update(auto_remap)

    base_sha = base_scenario.base_graph_sha256
    frame_sha = base_scenario.framing_sha256
    combined_edits = (*executed_ext_edits, *auto_edits)
    final_plan = (
        RewritePlan(base_graph_sha256=base_sha, framing_sha256=frame_sha, edits=combined_edits)
        if combined_edits else empty_plan(base_sha, frame_sha)
    )
    verification_passed = _verify_final_target_plan(
        nodes, final_plan, case, datapath, phase=phase, primitive_table=primitive_table,
    )

    rewired = _rewire_node_preds(all_nodes, global_remap)
    tuned_nodes, tiler_opps = _apply_pillar_c_tiler_extensions(
        rewired, all_dems, case, phase=phase, locked_ids=locked_ids,
        primitive_table=primitive_table, allow_unbound_formula=eff_allow_unbound,
    )
    if collective_plans:
        tuned_nodes, all_dems = _reconcile_oneshot_link_ledger(
            tuned_nodes, all_dems, collective_plans, peaks_by_stage, p0,
        )
    _, _, topology, _ = pricing_ctx.ensure() if collective_plans else (None, None, None, False)
    die_sched, final_dems, moe_dec, extra_receipts = _select_best_moe_and_die_schedule(
        tuned_nodes, all_dems, case, floors, p0, locked_ids=locked_ids,
        collective_plans=collective_plans, topology=topology,
        primitive_table=primitive_table, allow_unbound_formula=eff_allow_unbound,
    )
    die_sched, final_dems, final_plan, all_specs, tiler_opps, extra_receipts, moe_dec, stage_scheds = (
        _enforce_engine_monotonicity(
            nodes, demands, case, floors, p0, datapath, pricing_ctx,
            die_sched, final_dems, final_plan,
            all_specs, tiler_opps, extra_receipts, moe_dec,
            opt_mode=opt_mode, phase=phase, safe_n_tok=safe_n_tok,
            base_sha=base_sha, frame_sha=frame_sha, topology=topology,
            primitive_table=primitive_table, has_ext_edits=bool(executed_ext_edits),
            compute_waterfall=compute_waterfall_and_codesign,
        )
    )
    indexed = tuple(replace(n, ordinal=idx) for idx, n in enumerate(die_sched.nodes))
    _validate_final_graph_invariants(indexed, final_dems)
    stage_waterfall = (
        _build_stage_waterfall(opt_mode, stage_scheds)
        if compute_waterfall_and_codesign else ()
    )
    codesign_opps = (
        _collect_codesign_opportunities(
            case, indexed, all_specs, tiler_opps,
            primitive_table=primitive_table, step_request=step_request,
        )
        if compute_waterfall_and_codesign else ()
    )
    return _build_solved_target_graph(
        indexed=indexed, final_dems=final_dems, die_sched=die_sched,
        all_specs=all_specs, all_triggers=all_triggers, final_plan=final_plan,
        verification_passed=verification_passed, floors=floors,
        eff_allow_unbound=eff_allow_unbound, extra_receipts=extra_receipts,
        moe_dec=moe_dec, codesign_opps=codesign_opps,
        stage_waterfall=stage_waterfall, opt_mode=opt_mode,
    )


__all__ = [
    "BACKEND_ASCENDC_LABEL",
    "BACKEND_CATLASS_LABEL",
    "BACKEND_TILELANG_LABEL",
    "BindingState",
    "PipeStageSpec",
    "FormulaTensorSpec",
    "FormulaKernelSpec",
    "FormulaKernelEvaluation",
    "FusedPipelineRecipe",
    "SkuTargetFloors",
    "KernelSynthesisSpec",
    "MacroWallTrigger",
    "TargetOpNode",
    "SolvedTargetGraph",
    "TargetGraphResult",
    "STATUS_AT_FLOOR",
    "STATUS_NEW_OP",
    "STATUS_EXTEND_ABI",
    "STATUS_TUNE_PREFIX",
    "extract_sku_floors",
    "evaluate_formula_kernel",
    "simulate_formula_kernel",
    "evaluate_macro_wall_trigger",
    "attribute_node_overlap",
    "reschedule_node_timelines",
    "synthesize_kernel_spec",
    "load_rewrite_plan",
    "solve_target_graph",
]
