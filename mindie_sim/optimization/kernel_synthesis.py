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
  - Discover provable Tier-P2 algebraic fusions via ``rewrite.algebra`` (zero
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
from dataclasses import dataclass, replace
import hashlib
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.physical_recipe_base import (
    IO_HBM_CATEGORY,
    Phase,
    _resource_categories,
    io_weight,
    phased_program,
    tiled_phased_program,
)
from mindie_sim.prediction.program.binding import RecipeContext
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.results import OpOccurrence, StepRequest
from mindie_sim.prediction.runtime import tile_solver
from mindie_sim.prediction.runtime.tile_solver import (
    LiveBufferSpec,
    TileCandidate,
    TilePipelineSpec,
    TileShape,
)
from mindie_sim.primitives.loader import resolve_dispatch_envelope, resolve_sim_rate_overlay
from mindie_sim.rewrite.algebra import (
    FP32_ACCUM_BYTES,
    MIN_FUSE_MEMBERS,
    UB_DOUBLE_BUFFER_SLOTS,
    AlgebraicCandidate,
    AlgebraicTrait,
    OpTraitInfo,
    SemanticAuditTrigger,
    classify_op_traits,
    discover_algebraic_rewrites_for_ops,
)
from mindie_sim.rewrite.schema import (
    Edit,
    FuseRegion,
    KernelBoundary,
    OpRef,
    PortRef,
    RewritePlan,
    decode_rewrite_plan,
    empty_plan,
    plan_sha256,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import RankOps

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
BACKEND_TRITON_LABEL = "Triton-Ascend"
BACKEND_HCCL_LABEL = "HCCL / MC2"

_COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "allgather", "reducescatter", "reduce_scatter", "alltoall",
    "alltoallv", "mc2", "mc2_transport", "mc2_protocol", "dispatch", "combine",
    "collective", "hccl", "p2p", "tp_allreduce", "engram_ids", "engram_broadcast",
})
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
        comp_eqs = tuple(equations) if equations else tuple(
            _equation_for_op_spec(op, tr, idx) for idx, (op, tr) in enumerate(zip(ordered, traits))
        )
        in_spec = FormulaTensorSpec("x", tuple(ordered[0].local_shape), traits[0].input_dtype, "input")
        out_spec = FormulaTensorSpec("y", tuple(ordered[-1].local_shape), traits[-1].output_dtype, "output")
        return cls(
            name=kernel_name,
            equations=comp_eqs,
            inputs=(in_spec,) if in_spec.shape else (),
            outputs=(out_spec,) if out_spec.shape else (),
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
        if categories & {"cube", "mte1"}:
            cube_w = {role: 1.0 for role in ("cube", "mte1") if role in categories}
            phases.append(Phase("cube_mma", cube_w, after=(phases[-1].name,) if phases else ()))
        if "vector" in categories:
            prev = (phases[-1].name,) if phases else ()
            if self.row_reduction:
                phases.append(Phase("aiv_reduce", {"vector": HALF_SPLIT_RATIO}, after=prev))
                phases.append(Phase("aiv_epilogue", {"vector": HALF_SPLIT_RATIO}, after=("aiv_reduce",)))
            else:
                phases.append(Phase("aiv_fp32", {"vector": 1.0}, after=prev))
        if "link" in categories:
            phases.append(Phase("link_pipe", {"link": 1.0}, after=(phases[-1].name,) if phases else ()))
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
                phases.append(Phase("mte3_store", store_w, after=(phases[-1].name,) if phases else ()))
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "target_us": round(float(self.target_us), 3),
            "target_attainment_pct": round(float(self.target_attainment_pct), 1),
            "abi_signature_id": self.abi_signature_id,
            "binding_state": self.binding_state,
        }

    def format_formula_block(self) -> str:
        eq_txt = " ; ".join(self.symbolic_equations)
        ub_kb = self.ub_peak_bytes / KIB_BYTES
        ub_cap_kb = self.ub_budget_bytes / KIB_BYTES
        l1_kb = self.l1_peak_bytes / KIB_BYTES
        l1_cap_kb = self.l1_budget_bytes / KIB_BYTES
        elided_kb = self.elided_hbm_bytes / KIB_BYTES
        return (
            f"Math: {eq_txt} | "
            f"Tile: {self.tile_algorithm} {self.tile_shape} ({self.ub_precision_mode}) | "
            f"UB={ub_kb:.1f}/{ub_cap_kb:.1f}KiB, L1={l1_kb:.1f}/{l1_cap_kb:.1f}KiB "
            f"({'PASS' if self.sram_verified else 'FAIL'}) | "
            f"Elided HBM={elided_kb:.1f}KiB"
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
    opt_target_eta_pct: float | None = None
    spec: OpSpec | None = None
    synthesis_spec: KernelSynthesisSpec | None = None
    binding_state: BindingState = "bound_implementation"


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
    ub_edges = ub_peer_edges_for_rank(die.rank, case.machine.links, case.machine.transfer_channels)
    datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
    launch_vec, launch_cube, launch_norm = _die_launch_us(die, datapath, primitive_table)
    intra_lat, any_lat, has_intra_links = _min_link_latency_us(case.machine)
    tp = max(1, int(case.runtime.parallel.tp))
    rpn = max(1, int(getattr(case.machine, "ranks_per_node", tp) or tp))
    has_ub_direct = bool(ub_edges) or bool(datapath.has_l0c_to_ub and has_intra_links)
    is_pod = bool(has_ub_direct or has_intra_links or rpn >= tp)
    link_alpha = intra_lat if (is_pod and intra_lat > 0.0) else any_lat
    oneshot = launch_vec + link_alpha
    if has_ub_direct and is_pod:
        label = "UB-Direct (CCU / AIV One-Shot)"
    elif is_pod:
        label = "Intra-Pod Direct (AIV One-Shot)"
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

    link_bps = 0.0
    link_lat_us = 0.0
    if links:
        best = max(links, key=lambda item: float(getattr(item, "capacity_gbps", 0.0) or 0.0))
        link_bps = float(getattr(best, "capacity_gbps", 0.0) or 0.0) * BYTES_PER_GB
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
    all_shapes = [t.shape for t in (*spec.inputs, *spec.outputs) if t.shape]
    inner_dim = max((int(sh[-1]) for sh in all_shapes), default=CUBE_TILE_MN_ELEMS)
    tile_n = inner_dim if spec.row_reduction else min(inner_dim, CUBE_TILE_MN_ELEMS)
    in_b = max((_safe_elem_bytes(t.dtype) for t in spec.inputs), default=2.0)
    out_b = max((_safe_elem_bytes(t.dtype) for t in spec.outputs), default=2.0)
    # Ascend AIV upcasts vector intermediates to FP32 (4 B/elem) inside UB.
    row_fp32_bytes = int(tile_n * (max(in_b, float(FP32_ACCUM_BYTES)) + max(out_b, float(FP32_ACCUM_BYTES))))
    if spec.row_reduction:
        row_fp32_bytes += int(tile_n * FP32_ACCUM_BYTES)
    l1_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_K_ELEMS * (in_b + out_b) * UB_DOUBLE_BUFFER_SLOTS) if has_cube else 0
    l0_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_MN_ELEMS * L0C_ACCUM_BYTES) if has_cube else 0

    candidates: list[TileCandidate] = []
    for slots in (UB_DOUBLE_BUFFER_SLOTS, 1):
        live: list[LiveBufferSpec] = []
        if row_fp32_bytes > 0 and ub_raw > 0:
            live.append(LiveBufferSpec("ub_fp32_work", "ub", row_fp32_bytes, 0, 1, owner_kind="aiv", slots=slots))
        if has_cube and l1_budget > 0 and l1_tile > 0:
            live.append(LiveBufferSpec("l1_cube_tile", "l1", min(l1_tile, l1_budget), 0, 1, owner_kind="aic", slots=1))
        if has_cube and l0_budget > 0 and l0_tile > 0:
            live.append(LiveBufferSpec("l0c_fp32_accum", "l0c", min(l0_tile, l0_budget), 0, 1, owner_kind="aic", slots=1))
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
            uses_simt=True,
            owner_axis="aic" if (has_cube and row_fp32_bytes == 0) else "aiv",
        ))
    tile_spec = TilePipelineSpec(spec.family or "formula_kernel", tuple(candidates), simt_ub_carveout_bytes=carveout)
    hw = SimpleNamespace(
        datapath=datapath,
        buffers=bufs,
        buffer_units={"ub": datapath.aiv_count, "l1": datapath.aic_count, "l0c": datapath.aic_count, "l2": 1},
        stream_owners=None,
    )
    if ub_raw <= 0:
        return tile_spec, "unbounded", f"(M=1, N={tile_n}, K=1)", 0, 0, l1_tile, l1_budget, l0_tile, l0_budget, 1, False
    try:
        solved = tile_solver.solve(tile_spec, None, hw)
        peaks = {p.pool: int(p.peak_bytes) for p in solved.pool_peaks}
        ub_pk = peaks.get("ub", row_fp32_bytes)
        l1_pk = peaks.get("l1", l1_tile)
        l0_pk = peaks.get("l0c", l0_tile)
        ok = ub_pk <= ub_budget and (l1_budget == 0 or l1_pk <= l1_budget) and (l0_budget == 0 or l0_pk <= l0_budget)
        sh = solved.tile_shape
        return (
            tile_spec, solved.algorithm, f"(M={sh.m}, N={sh.n}, K={sh.k})",
            ub_pk, ub_budget, l1_pk, l1_budget, l0_pk, l0_budget, solved.depth, ok,
        )
    except tile_solver.TileInfeasibleError:
        return (
            tile_spec, "ub_infeasible", f"(M=1, N={tile_n}, K={CUBE_TILE_K_ELEMS if has_cube else 1})",
            row_fp32_bytes, ub_budget, l1_tile, l1_budget, l0_tile, l0_budget, 1, False,
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
    recipe = FusedPipelineRecipe(
        slots=depth,
        row_reduction=spec.row_reduction,
        abi_signature_id=abi_id,
        binding_state="unbound_formula_projection",
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
                "以算换存: 优先启用 MLA 矩阵吸收 / KV Cache 低比特量化 (C8/FP8) 释放单卡 HBM",
                "内存卸载: 评估 KV Offloading / Host DRAM-SSD Prefix Cache 分级池化与 Engram Host Offload",
                "投机状态权衡: 对比周期检查点重算 vs 多槽 SSM 状态快照，扩大可承载 Batch B 吃满 SLA 余量",
                "并行重分配: 适度调低 DP/调高 TP 或切换高压缩权重量化以突破 HBM 并发上限",
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
                "以存换算: 利用充裕 HBM 余量预展开高频投影/索引权重或增大 L1/UB 驻留分块",
                "投机多槽快照: 分配 [K+1] 多槽状态缓冲并做 in-place 2D 槽位别名消除回滚拷贝",
                "降低单步通信/算力瓶颈: 评估 Attention Local-TP / 提升 TP 或 EP 分摊权重读取延迟",
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
            "当前延迟与 HBM 利用率处于均衡区间，优先落地已验证的片上融合与超节点通信展开方案",
        ),
    )


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((max(0.0, a), max(a, b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        p_lo, p_hi = merged[-1]
        if lo <= p_hi:
            merged[-1] = (p_lo, max(p_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _exposed_length_outside(
    interval: tuple[float, float], cover: Sequence[tuple[float, float]],
) -> float:
    lo, hi = interval
    if hi <= lo:
        return 0.0
    cur = lo
    exposed = 0.0
    for c_lo, c_hi in cover:
        if c_hi <= cur:
            continue
        if c_lo >= hi:
            break
        if c_lo > cur:
            exposed += min(hi, c_lo) - cur
        cur = max(cur, c_hi)
        if cur >= hi:
            return 0.0
    return exposed + max(0.0, hi - cur)


def _short_primary_label(node: TargetOpNode) -> str:
    fam = node.family.lower()
    if fam in {"grouped_gemm", "grouped_moe", "mc2"} or "megamoe" in node.recipe.lower():
        return "MoE Routed GMM / MegaMoE"
    if fam in {"allgather", "all_gather"}:
        return "Collective AllGather"
    if fam in {"indexer", "index_pool"}:
        return "Sparse Indexer"
    role = node.logical_role or node.ascend_kernel or _strip_op_prefix(node.op_id)
    return role[:MAX_MASKED_BY_LEN]


def _find_dominant_masker(
    interval: tuple[float, float], claimed_labeled: Sequence[tuple[float, float, str]],
) -> str:
    lo, hi = interval
    overlap_by_lbl: dict[str, float] = defaultdict(float)
    for c_lo, c_hi, lbl in claimed_labeled:
        ov = min(hi, c_hi) - max(lo, c_lo)
        if ov > OVERLAP_EXPOSED_EPS_US:
            overlap_by_lbl[lbl] += ov
    if not overlap_by_lbl:
        return "主计算流并发算子"
    ordered = sorted(overlap_by_lbl.items(), key=lambda kv: -kv[1])
    top_lbl, top_ov = ordered[0]
    if len(ordered) > 1 and ordered[1][1] > top_ov * SECONDARY_MASKER_SHARE:
        return f"{top_lbl} + {ordered[1][0]}"[:MAX_MASKED_BY_LEN]
    return top_lbl


def _overlap_priority_key(node: TargetOpNode) -> tuple[Any, ...]:
    if node.stream_index == 1:
        return (2, node.start_us, node.ordinal)
    if node.stream_index == 2:
        return (1, -node.end_us, -node.dur_us, node.ordinal)
    return (0, -(node.hbm_total_bytes + node.flops), -node.end_us, -node.dur_us, node.ordinal)


def attribute_node_overlap(
    nodes: Sequence[TargetOpNode],
) -> tuple[list[TargetOpNode], dict[str, float]]:
    """Attribute cross-stream/branch overlap with 100% exact archetype-span conservation."""
    by_arch: dict[str, list[tuple[int, TargetOpNode]]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        by_arch[node.archetype_id].append((idx, node))

    updated: list[TargetOpNode] = list(nodes)
    span_by_arch: dict[str, float] = {}
    for aid, indexed in by_arch.items():
        all_ivs = [(n.start_us, n.end_us) for _, n in indexed if n.dur_us > 0.0]
        span_by_arch[aid] = sum(hi - lo for lo, hi in _merge_intervals(all_ivs))
        ordered = sorted(indexed, key=lambda item: _overlap_priority_key(item[1]))
        claimed: list[tuple[float, float]] = []
        claimed_labeled: list[tuple[float, float, str]] = []
        for idx, node in ordered:
            if node.dur_us <= 0.0:
                updated[idx] = replace(node, is_overlapped=False, exposed_dur_us=0.0, masked_by="")
                continue
            iv = (node.start_us, node.end_us)
            exp_us = _exposed_length_outside(iv, claimed)
            masked_us = max(0.0, node.dur_us - exp_us)
            fully_hidden = exp_us <= OVERLAP_EXPOSED_EPS_US
            eff_exp = 0.0 if fully_hidden else exp_us
            masker = _find_dominant_masker(iv, claimed_labeled) if masked_us > OVERLAP_EXPOSED_EPS_US else ""
            eff_gap = 0.0 if fully_hidden else max(0.0, eff_exp - node.l_star_lean_us)
            eff_ratio = None if fully_hidden else (
                (eff_exp / node.l_star_lean_us) if node.l_star_lean_us > 0.0 else None
            )
            b_disp = f"Overlapped by {masker} (0 exposed)" if fully_hidden else node.bound_display
            updated[idx] = replace(
                node,
                is_overlapped=fully_hidden,
                exposed_dur_us=eff_exp,
                masked_by=masker,
                gap_us=eff_gap,
                gap_ratio=eff_ratio,
                bound_display=b_disp,
            )
            claimed = _merge_intervals([*claimed, iv])
            claimed_labeled.append((iv[0], iv[1], _short_primary_label(node)))
    return updated, span_by_arch


def reschedule_node_timelines(nodes: Sequence[TargetOpNode]) -> list[TargetOpNode]:
    """Recompute exact causal ``(start_us, end_us)`` preserving parallel branches."""
    by_arch: dict[str, list[tuple[int, TargetOpNode]]] = defaultdict(list)
    arch_order: list[str] = []
    for idx, node in enumerate(nodes):
        if node.archetype_id not in by_arch:
            arch_order.append(node.archetype_id)
        by_arch[node.archetype_id].append((idx, node))

    updated = list(nodes)
    cursor_us = 0.0
    for aid in arch_order:
        ordered = sorted(by_arch[aid], key=lambda item: (item[1].start_us, item[1].ordinal))
        end_by_occ: dict[Any, float] = {}
        stream_history: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
        arch_origin = cursor_us
        arch_max_end = arch_origin
        for idx, node in ordered:
            pred_ready = max((end_by_occ.get(p, arch_origin) for p in node.predecessors), default=arch_origin)
            s_key = (node.stream, node.stream_index)
            stream_ready = arch_origin
            for orig_s, orig_e, new_e in stream_history[s_key]:
                if node.start_us >= orig_e - ORIG_OVERLAP_EPS_US and orig_e > orig_s:
                    stream_ready = max(stream_ready, new_e)
            new_s = max(pred_ready, stream_ready)
            new_e = new_s + node.dur_us
            end_by_occ[node.occurrence] = new_e
            arch_max_end = max(arch_max_end, new_e)
            stream_history[s_key].append((node.start_us, node.end_us, new_e))
            updated[idx] = replace(node, start_us=new_s, end_us=new_e)
        cursor_us = arch_max_end
    return updated


def _equation_for_shape(shape: Sequence[int], trait: OpTraitInfo, step_idx: int) -> str:
    prev = "x" if step_idx == 0 else f"t{step_idx}"
    out = f"t{step_idx + 1}"
    k = trait.symbolic_op_kind
    gs = trait.group_size or MXFP_BLOCK_GROUP
    shape_tag = "x".join(str(d) for d in shape) if shape else "?"
    param_txt = (
        "; " + ", ".join(f"{pk}={pv:g}" for pk, pv in trait.activation_params)
        if trait.activation_params else ""
    )
    eq_map = {
        "view_copy": f"{out} = strided_view({prev}) [zero-copy metadata elision]",
        "grouped_matmul": f"{out} = GroupedMatMul({prev}, W_e^T) [{shape_tag}, L0C FP32]",
        "matmul": f"{out} = {prev} @ W^T [{shape_tag}, L0C FP32]",
        "allgather": f"{out} = AllGather({prev}, axis={trait.collective_axis or 'tp'})",
        "reduce_scatter": f"{out} = ReduceScatter({prev}, SUM, axis={trait.collective_axis or 'tp'})",
        "all_gather_matmul": f"{out} = AllGather({prev}) @ W_col^T [MC2 pipelined]",
        "matmul_reduce_scatter": f"{out} = ReduceScatter({prev} @ W_row^T, SUM) [MC2 pipelined]",
        "add": f"{out} = {prev} + r [UB FP32]",
        "mul": f"{out} = {prev} * g [UB FP32]",
        "rmsnorm": f"{out} = ({prev} / sqrt(mean({prev}^2) + eps)) * gamma [UB FP32]",
        "layernorm": f"{out} = (({prev} - mu) / sqrt(var + eps)) * gamma + beta [UB FP32]",
        "add_rmsnorm": f"{out} = RMSNorm({prev} + r; gamma) [UB FP32]",
        "add_rmsnorm_quant": f"{out}, s = DynamicQuant(RMSNorm({prev} + r; gamma)) [UB FP32]",
        "rmsnorm_quant": f"{out}, s = DynamicQuant(RMSNorm({prev}; gamma)) [UB FP32]",
        "norm_gate": f"{out} = RMSNorm({prev}; gamma) * SiLU(z) [UB FP32]",
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
        "rotary": f"{out} = RoPE({prev}, cos, sin) [UB FP32]",
        "kv_scatter": f"KV_cache[slot] = QuantScatter({prev}) [in-place cache write]",
        "attn_res": f"{out} = Sum_i softmax_i(Norm(h_i) . q_l) * h_i [UB FP32]",
        "mhc_mix": f"{out} = SinkhornMix({prev}, H_pre, H_post) [UB FP32]",
        "attention_update": f"{out}, lse = OnlineSoftmaxMerge({prev}, lse_prev) [FP32]",
        "cast": f"{out} = cast_{trait.output_dtype}({prev})",
    }
    return eq_map.get(k, f"{out} = {k}({prev}{param_txt}) [{shape_tag}]")


def _equation_for_atom(node: TargetOpNode, trait: OpTraitInfo, step_idx: int) -> str:
    return _equation_for_shape(node.shape, trait, step_idx)


def _equation_for_op_spec(op: OpSpec, trait: OpTraitInfo, step_idx: int) -> str:
    return _equation_for_shape(op.local_shape, trait, step_idx)


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
    max_row_ub = max((t.row_working_bytes for t in traits if t.row_working_bytes > 0), default=0)
    if max_row_ub == 0:
        row_elems = max((int(m.shape[-1]) for m in members if m.shape), default=0)
        max_row_ub = row_elems * FP32_ACCUM_BYTES * UB_DOUBLE_BUFFER_SLOTS
    l1_tile = l0_tile = 0
    if has_cube:
        a_b = max((_safe_elem_bytes(t.input_dtype) for t in traits if t.trait == AlgebraicTrait.BILINEAR_MATMUL), default=2.0)
        w_b = max((_safe_elem_bytes(t.weight_dtype or t.input_dtype) for t in traits if t.trait == AlgebraicTrait.BILINEAR_MATMUL), default=1.0)
        l1_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_K_ELEMS * (a_b + w_b) * UB_DOUBLE_BUFFER_SLOTS)
        l0_tile = int(CUBE_TILE_MN_ELEMS * CUBE_TILE_MN_ELEMS * L0C_ACCUM_BYTES)
    if max_row_ub <= 0 or ub_budget <= 0:
        return "ub_infeasible", "(M=1, N=1, K=1)", max_row_ub, ub_budget, l1_tile, l1_budget, l0_tile, l0_budget, False

    hw = SimpleNamespace(
        datapath=datapath,
        buffers=bufs,
        buffer_units={"ub": datapath.aiv_count, "l1": datapath.aic_count, "l0c": datapath.aic_count, "l2": 1},
        stream_owners=None,
    )
    cands: list[TileCandidate] = []
    for slots in (UB_DOUBLE_BUFFER_SLOTS, 1):
        live_bufs = [LiveBufferSpec("ub_fp32_work", "ub", max_row_ub, 0, 1, owner_kind="aiv", slots=slots)]
        if has_cube and l1_budget > 0 and l1_tile > 0:
            live_bufs.append(LiveBufferSpec("l1_cube_tile", "l1", min(l1_tile, l1_budget), 0, 1, owner_kind="aic", slots=1))
        if has_cube and l0_budget > 0 and l0_tile > 0:
            live_bufs.append(LiveBufferSpec("l0c_fp32_accum", "l0c", min(l0_tile, l0_budget), 0, 1, owner_kind="aic", slots=1))
        algo: Literal["resident_1pass", "online_reduce"] = (
            "resident_1pass" if slots == UB_DOUBLE_BUFFER_SLOTS else "online_reduce"
        )
        cands.append(TileCandidate(
            algorithm=algo,
            tile_shape=TileShape(1, max(1, max_row_ub // FP32_ACCUM_BYTES), CUBE_TILE_K_ELEMS if has_cube else 1),
            depth=slots,
            owner_partition=(datapath.aic_count, datapath.aiv_count),
            work_items=1,
            tiles_per_owner=1,
            buffers=tuple(live_bufs),
            uses_simt=True,
            owner_axis="aiv",
        ))
    try:
        solved = tile_solver.solve(
            TilePipelineSpec("kernel_synthesis", tuple(cands), simt_ub_carveout_bytes=carveout),
            None,
            hw,
        )
        peaks = {p.pool: int(p.peak_bytes) for p in solved.pool_peaks}
        ub_pk = peaks.get("ub", max_row_ub)
        l1_pk = peaks.get("l1", l1_tile)
        l0_pk = peaks.get("l0c", l0_tile)
        ok = ub_pk <= ub_budget and (l1_budget == 0 or l1_pk <= l1_budget) and (l0_budget == 0 or l0_pk <= l0_budget)
        sh = solved.tile_shape
        return (
            solved.algorithm, f"(M={sh.m}, N={sh.n}, K={sh.k})",
            ub_pk, ub_budget, l1_pk, l1_budget, l0_pk, l0_budget, ok,
        )
    except tile_solver.TileInfeasibleError:
        return (
            "ub_infeasible", f"(M=1, N={max(1, max_row_ub // FP32_ACCUM_BYTES)}, K=1)",
            max_row_ub, ub_budget, l1_tile, l1_budget, l0_tile, l0_budget, False,
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
) -> KernelSynthesisSpec:
    """Compose symbolic math equations and TileSolver FP32 SRAM residency for a fused kernel."""
    equations = tuple(
        _equation_for_atom(node, tr, idx)
        for idx, (node, tr) in enumerate(zip(members, member_traits))
    )
    in_t = (f"{_strip_op_prefix(members[0].op_id)}.in[{members[0].dtype}, {list(members[0].shape)}]",)
    out_t = (f"{_strip_op_prefix(members[-1].op_id)}.out[{members[-1].dtype}, {list(members[-1].shape)}]",)
    elided_names = tuple(_strip_op_prefix(m.op_id) for m in members[:-1])
    algo, t_shape, ub_pk, ub_cap, l1_pk, l1_cap, l0_pk, l0_cap, ok = _verify_sram_residency(
        members, member_traits, datapath,
    )
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


def _node_to_op_spec(node: TargetOpNode, known_ids: set[str]) -> OpSpec:
    deps = tuple(dict.fromkeys(p.op_id for p in node.predecessors if p.op_id in known_ids and p.op_id != node.op_id))
    if node.spec is not None:
        return replace(node.spec, op_id=node.op_id, deps=deps)
    return OpSpec(
        op_id=node.op_id,
        family=node.family,
        stage=node.occurrence.stage,
        layer=0,
        phase="decode",
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
        vector_ops=node.vector_ops,
        hbm_bytes=node.hbm_total_bytes,
        mte2_bytes=node.mte2_bytes,
        mte3_bytes=node.mte3_bytes,
        link_bytes=node.link_bytes,
    )


def _verify_arch_plan(
    case: Case,
    arch_ops: Sequence[OpSpec],
    plan: RewritePlan,
    datapath: DatapathCapabilitySpec,
) -> bool:
    """Mechanically verify a ``RewritePlan`` against all structural/capacity/conservation gates."""
    if not plan.edits or not arch_ops:
        return True
    from mindie_sim.rewrite.apply import ResolvedScenario, apply_plan
    from mindie_sim.rewrite.verify import verify_plan

    scenario = ResolvedScenario(
        case=case,
        step=StepRequest(phase="decode", tokens=1, context=1),
        stack_id="kernel_synthesis",
        base_graph_sha256=plan.base_graph_sha256,
        framing_sha256=plan.framing_sha256,
        scenario_sha256="kernel_synthesis",
        atomic_step=SimpleNamespace(case=case),  # type: ignore[arg-type]
        mainline_step=None,
        atomic_ops=(RankOps(0, tuple(arch_ops)),),
        datapath=datapath,
    )
    applied = apply_plan(plan, scenario)
    receipt = verify_plan(scenario, applied, scope="archetype")
    if not receipt.passed:
        failed = next((g for g in receipt.gates if not g.passed), receipt.gates[0])
        raise ModelingError(f"REWRITE_PLAN_REJECTED:{failed.name}:{failed.reason}:{failed.detail}")
    return True


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
        tag = "CCU/AIV" if floors.is_a5_ub else "AIV"
        return (
            _strip_op_prefix(members[-1].op_id if cand.abi_id == "aclnnAllGatherMatmulV2" else anchor.op_id),
            f"MC2 通算融合 ({cand.abi_id})",
            cand.abi_id,
            BACKEND_HCCL_LABEL,
            f"[通信方案优化: MC2 切 {tag} + L0 流水线建模]",
            False,
            "bound_implementation",
        )
    if cand.abi_id == "aclnnMlaPrologV3":
        return (
            "mla_prolog",
            "MLA Prolog 融合 (Q/KV 低秩投影 + Norm + RoPE + KV Write)",
            "aclnnMlaPrologV3",
            BACKEND_CANN_LABEL,
            "[配置开启 MLAPO + L0 流水线建模]",
            False,
            "bound_implementation",
        )
    if has_gmm:
        is_ext = not cand.shipping_supported
        acts = [t.activation_kind for t in member_traits if t.activation_kind]
        act_lbl = acts[0].upper() if acts else "ACT"
        kern = cand.abi_id if not is_ext else (
            cand.abi_id if "dispatch_ffn_combine" in cand.abi_id else f"GroupedMatmul{act_lbl.capitalize()}Quant"
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
            BACKEND_CANN_LABEL,
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
    return (
        f"fused_{_strip_op_prefix(members[-1].op_id)}",
        f"Fused UB Pipeline ({' + '.join(_strip_op_prefix(m.op_id) for m in members)})",
        cand.abi_id,
        BACKEND_TRITON_LABEL if is_new else BACKEND_CANN_LABEL,
        status,
        is_new,
        "unbound_formula_projection" if is_new else "bound_implementation",
    )


def _collapse_candidate_group(
    group_nodes: Sequence[TargetOpNode],
    group_demands: Sequence[tuple[int, int, int, int, int]],
    cand: AlgebraicCandidate,
    member_traits: Sequence[OpTraitInfo],
    floors: SkuTargetFloors,
    peaks: Mapping[str, Any],
    case: Case,
    datapath: DatapathCapabilitySpec,
    *,
    primitive_table: Any = None,
    allow_unbound_formula: bool = True,
) -> tuple[TargetOpNode, tuple[int, int, int, int, int], KernelSynthesisSpec] | None:
    """Collapse a verified candidate into one fused ``TargetOpNode`` via ``Engine`` + ``TileSolver``."""
    suffix, role, kern, backend, status, needs_new, b_state = _resolve_candidate_metadata(
        cand, group_nodes, member_traits, floors,
    )
    if b_state == "unbound_formula_projection" and not allow_unbound_formula:
        return None
    anchor = group_nodes[0]
    raw_hbm = sum(n.hbm_total_bytes for n in group_nodes)
    avoid = min(raw_hbm, float(cand.avoidable_hbm_bytes))
    useful_hbm = max(0.0, raw_hbm - avoid)
    if useful_hbm <= 0.0 and raw_hbm > 0.0:
        useful_hbm = max(n.hbm_total_bytes for n in group_nodes)
    hbm_r = useful_hbm * HALF_SPLIT_RATIO
    hbm_w = useful_hbm * HALF_SPLIT_RATIO

    group_ops = tuple(_node_to_op_spec(n, {m.op_id for m in group_nodes}) for n in group_nodes)
    equations = tuple(_equation_for_atom(n, tr, idx) for idx, (n, tr) in enumerate(zip(group_nodes, member_traits)))
    formula_spec = replace(
        FormulaKernelSpec.from_ops(group_ops, name=kern, equations=equations),
        hbm_bytes_override=useful_hbm,
        mte2_bytes_override=hbm_r,
        mte3_bytes_override=hbm_w,
    )
    eval_res = evaluate_formula_kernel(formula_spec, case, primitive_table=primitive_table)
    if not eval_res.sram_verified:
        return None

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
    l_star_us = max(float(max(demand)) / NS_PER_US, eval_res.l_star_us)

    before_sum_us = sum(n.dur_us for n in group_nodes)
    t_target = max(eval_res.projected_us, l_star_us)
    if cand.law == "law3_mc2":
        t_target = max(t_target, floors.mc2_sync_us + float(link_ns) / NS_PER_US)
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
        elided_hbm_bytes=int(round(max(0.0, raw_hbm - useful_hbm))),
        l_star_us=l_star_us,
        before_us=before_sum_us,
        target_us=dur_us,
        eta_target=achieved_eta,
        abi_signature_id=eval_res.abi_signature_id,
        binding_state=b_state,
    )
    if not spec.sram_verified:
        return None

    prefix = anchor.op_id[: len(anchor.op_id) - len(_strip_op_prefix(anchor.op_id))]
    new_op_id = f"{prefix}{suffix}"
    absorbed = tuple(dict.fromkeys(
        s for n in group_nodes for s in (n.absorbed_source_ops or (_strip_op_prefix(n.op_id),))
    ))
    sem_atoms = tuple(dict.fromkeys(a for n in group_nodes for a in n.semantic_atoms))
    group_occs = {n.occurrence for n in group_nodes}
    ext_preds = tuple(dict.fromkeys(p for n in group_nodes for p in n.predecessors if p not in group_occs))
    new_occ = replace(anchor.occurrence, op_id=new_op_id)
    hbm_ns = max(hbm_r_ns, hbm_w_ns)
    dom_bound = (
        "link_bound" if link_ns > max(cube_ns, vec_ns, hbm_ns) and link_ns > 0 else (
            "memory_bound" if hbm_ns >= max(cube_ns, vec_ns) and hbm_ns > 0 else (
                "cube_bound" if cube_ns >= vec_ns and cube_ns > 0 else "vector_bound"
            )
        )
    )
    new_family = (
        "all_gather_matmul" if cand.abi_id == "aclnnAllGatherMatmulV2" else (
            "matmul_reduce_scatter" if cand.abi_id == "aclnnMatmulReduceScatterV2" else anchor.family
        )
    )
    fused_node = replace(
        anchor,
        occurrence=new_occ,
        op_id=new_op_id,
        family=new_family,
        recipe=suffix,
        predecessors=ext_preds,
        start_us=min(n.start_us for n in group_nodes),
        end_us=min(n.start_us for n in group_nodes) + dur_us,
        dur_us=dur_us,
        exposed_dur_us=dur_us,
        flops=sum(n.flops for n in group_nodes),
        vector_ops=sum(n.vector_ops for n in group_nodes),
        hbm_read_bytes=hbm_r,
        hbm_write_bytes=hbm_w,
        hbm_total_bytes=useful_hbm,
        mte2_bytes=hbm_r,
        mte3_bytes=hbm_w,
        fixpipe_bytes=0.0,
        link_bytes=sum(n.link_bytes for n in group_nodes),
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
        opt_target_eta_pct=spec.target_attainment_pct,
        synthesis_spec=spec,
        binding_state=b_state,
    )
    return fused_node, demand, spec


def _rewire_node_preds(
    nodes: Sequence[TargetOpNode], occ_remap: Mapping[OpOccurrence, OpOccurrence],
) -> list[TargetOpNode]:
    if not occ_remap:
        return list(nodes)
    return [
        replace(
            n,
            predecessors=tuple(dict.fromkeys(
                occ_remap.get(p, p) for p in n.predecessors if occ_remap.get(p, p) != n.occurrence
            )),
        )
        for n in nodes
    ]


def _extract_external_candidates(
    ext_plan: RewritePlan | None,
    arch_ops: Sequence[OpSpec],
    traits_map: Mapping[str, OpTraitInfo],
    case: Case,
    datapath: DatapathCapabilitySpec,
) -> list[AlgebraicCandidate]:
    """Verify external Agent ``FuseRegion`` edits through ``verify_plan`` and lift to candidates."""
    if ext_plan is None or not ext_plan.edits:
        return []
    op_ids = {op.op_id for op in arch_ops}
    matching_edits: list[Edit] = []
    for edit in ext_plan.edits:
        op = edit.operation
        if isinstance(op, FuseRegion):
            m_ids = tuple(ref.id for ref in op.members)
            if m_ids and all(mid in op_ids for mid in m_ids):
                matching_edits.append(edit)
    if not matching_edits:
        return []
    sub_plan = RewritePlan(
        base_graph_sha256=ext_plan.base_graph_sha256,
        framing_sha256=ext_plan.framing_sha256,
        edits=tuple(matching_edits),
    )
    _verify_arch_plan(case, arch_ops, sub_plan, datapath)
    out: list[AlgebraicCandidate] = []
    op_by_id = {op.op_id: op for op in arch_ops}
    for edit in matching_edits:
        freg = edit.operation
        assert isinstance(freg, FuseRegion)
        m_ids = tuple(ref.id for ref in freg.members)
        abi = freg.kernel_boundaries[0].abi_id if freg.kernel_boundaries else f"AgentFused_{edit.edit_id}"
        avoid = sum(int(op_by_id[mid].hbm_bytes or 0) for mid in m_ids[:-1]) // 2
        out.append(AlgebraicCandidate(
            candidate_id=f"agent_{edit.edit_id}",
            law="law1_ub_chain",
            abi_id=abi,
            fusion_tier=freg.fusion_tier,
            member_ids=m_ids,
            preserve_port_ids=tuple(p.op.id for p in freg.preserve_ports),
            shipping_supported=(freg.fusion_tier == "K1_shipping"),
            avoidable_hbm_bytes=avoid,
            traits=tuple(traits_map[mid].trait.value for mid in m_ids if mid in traits_map),
        ))
    return out


def _apply_pillar23_envelopes(
    nodes: Sequence[TargetOpNode],
    demands: Sequence[tuple[int, int, int, int, int]],
    floors: SkuTargetFloors,
    case: Case,
    *,
    phase: str,
    primitive_table: Any = None,
    allow_unbound_formula: bool = True,
) -> list[TargetOpNode]:
    """Price remaining single operators via white-box L0-pipe ``Engine`` + link alpha."""
    is_decode = str(phase or "decode").lower() == "decode"
    out: list[TargetOpNode] = []
    for node, dem in zip(nodes, demands):
        if node.opt_pillar in {"pillar1_fusion", "pillar3_comm"} or node.dur_us <= 0.0:
            out.append(node)
            continue
        fam = node.family.lower()
        if fam in {"engram_rh2d", "h2d", "offload"} or node.task_type == "HOST":
            out.append(replace(node, kernel_status=STATUS_AT_FLOOR, needs_new_kernel=False))
            continue

        rec = node.recipe.lower()
        kern_l = node.ascend_kernel.lower()
        l_link = float(dem[4]) / NS_PER_US
        is_mc2 = fam in {"all_gather_matmul", "matmul_reduce_scatter"}
        is_megamoe = fam in {"grouped_moe", "mc2"} or "megamoe" in rec or "dispatch_ffn_combine" in kern_l
        is_pure_comm = (node.task_type == "HCCL" or fam in _COLLECTIVE_FAMILIES) and not is_mc2 and not is_megamoe

        if is_pure_comm:
            # Pure L0 one-shot collective: SKU launch + link alpha + wire transfer time.
            t_after = min(node.dur_us, floors.oneshot_floor_us + l_link) if floors.is_intra_pod else node.dur_us
            gain = node.dur_us - t_after
            if floors.is_intra_pod and gain >= TARGET_OPT_MIN_GAIN_US and gain >= node.dur_us * TARGET_OPT_MIN_RATIO:
                tag = "CCU/AIV UB-Direct One-Shot" if floors.is_a5_ub else "AIV Intra-Pod One-Shot"
                is_small_sp = fam in {"allgather", "reduce_scatter"} and node.link_bytes <= UB_ONESHOT_SMALL_MSG_BYTES
                status = (
                    f"[通信方案优化: 切 {tag} / 小Batch Decode 可关 SP]"
                    if (is_small_sp and is_decode) else f"[通信方案优化: 切 {tag}]"
                )
                base_k = node.ascend_kernel.split(" (", 1)[0]
                eff = min(1.0, node.l_star_lean_us / t_after) if t_after > 0.0 and node.l_star_lean_us > 0.0 else None
                out.append(replace(
                    node, dur_us=t_after, exposed_dur_us=t_after,
                    ascend_kernel=f"{base_k} ({tag})", kernel_status=status, needs_new_kernel=False,
                    gap_us=max(0.0, t_after - node.l_star_lean_us),
                    gap_ratio=(t_after / node.l_star_lean_us) if node.l_star_lean_us > 0.0 else None,
                    roofline_efficiency=eff,
                    opt_pillar="pillar3_comm",
                    opt_target_eta_pct=round(eff * PERCENT_SCALE, 1) if eff is not None else None,
                ))
            else:
                out.append(replace(node, kernel_status=STATUS_AT_FLOOR, needs_new_kernel=False))
            continue

        op_stub = _node_to_op_spec(node, set())
        formula_spec = FormulaKernelSpec.from_ops((op_stub,), name=node.ascend_kernel or node.op_id)
        try:
            eval_res = evaluate_formula_kernel(formula_spec, case, primitive_table=primitive_table)
        except ModelingError:
            out.append(replace(node, kernel_status=STATUS_AT_FLOOR, needs_new_kernel=False))
            continue

        t_target = max(eval_res.projected_us, node.l_star_lean_us)
        mc2_kernel = node.ascend_kernel
        needs_new = fam == "attn_res_prefix" and allow_unbound_formula
        b_state: BindingState = "unbound_formula_projection" if needs_new else node.binding_state
        if is_mc2:
            t_target = max(t_target, floors.oneshot_floor_us + l_link) + floors.mc2_sync_us
            pillar = "pillar3_comm"
            if fam == "all_gather_matmul" and "allgathermatmul" not in kern_l:
                mc2_kernel = "aclnnAllGatherMatmulV2"
            elif fam == "matmul_reduce_scatter" and "matmulreducescatter" not in kern_l:
                mc2_kernel = "aclnnMatmulReduceScatterV2"
            status_lbl = f"[通信方案优化: MC2 切 {'CCU/AIV' if floors.is_a5_ub else 'AIV'} + L0 流水线]"
        elif is_megamoe:
            t_target = max(t_target, floors.oneshot_floor_us + l_link)
            pillar = "pillar2_roofline"
            status_lbl = f"{STATUS_TUNE_PREFIX} -> L0 流水线目标]"
        else:
            pillar = "pillar2_roofline"
            status_lbl = (
                f"{STATUS_NEW_OP} UB 驻留 Block-AttnRes 融合前缀"
                if needs_new else f"{STATUS_TUNE_PREFIX} -> L0 流水线目标]"
            )

        t_after = min(node.dur_us, t_target)
        gain = node.dur_us - t_after
        if gain >= TARGET_OPT_MIN_GAIN_US and gain >= node.dur_us * TARGET_OPT_MIN_RATIO:
            eff = min(1.0, node.l_star_lean_us / t_after) if t_after > 0.0 and node.l_star_lean_us > 0.0 else None
            out.append(replace(
                node, dur_us=t_after, exposed_dur_us=t_after, ascend_kernel=mc2_kernel,
                kernel_status=status_lbl, needs_new_kernel=needs_new,
                gap_us=max(0.0, t_after - node.l_star_lean_us),
                gap_ratio=(t_after / node.l_star_lean_us) if node.l_star_lean_us > 0.0 else None,
                roofline_efficiency=eff,
                opt_pillar=pillar,
                opt_target_eta_pct=round(eff * PERCENT_SCALE, 1) if eff is not None else None,
                binding_state=b_state,
            ))
        else:
            out.append(replace(node, kernel_status=STATUS_AT_FLOOR, needs_new_kernel=False))
    return out


def _solve_one_archetype(
    aid: str,
    a_nodes: Sequence[TargetOpNode],
    a_dems: Sequence[tuple[int, int, int, int, int]],
    ext_plan: RewritePlan | None,
    floors: SkuTargetFloors,
    p0: Mapping[str, Any],
    case: Case,
    datapath: DatapathCapabilitySpec,
    *,
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
    arch_ops = [_node_to_op_spec(n, known_ids) for n in a_nodes]
    traits_map = {op.op_id: classify_op_traits(op) for op in arch_ops}
    alg = discover_algebraic_rewrites_for_ops(
        arch_ops, datapath=datapath, allow_mc2=True, include_abi_extensions=allow_unbound_formula,
    )
    ext_cands = _extract_external_candidates(ext_plan, arch_ops, traits_map, case, datapath)
    ext_claimed = {mid for c in ext_cands for mid in c.member_ids}
    merged_cands = [*ext_cands, *(c for c in alg.candidates if not (set(c.member_ids) & ext_claimed))]
    idx_by_id = {n.op_id: i for i, n in enumerate(a_nodes)}
    consumed_idxs: set[int] = set()
    replacements: dict[int, tuple[TargetOpNode, tuple[int, int, int, int, int]]] = {}
    unlocked_ops: set[str] = set()
    specs: list[KernelSynthesisSpec] = []
    edits: list[Edit] = []
    remap: dict[OpOccurrence, OpOccurrence] = {}

    for cand in merged_cands:
        idxs = [idx_by_id[mid] for mid in cand.member_ids if mid in idx_by_id]
        if len(idxs) < MIN_FUSE_MEMBERS or any(i in consumed_idxs for i in idxs):
            continue
        idxs.sort()
        grp_nodes = [a_nodes[i] for i in idxs]
        grp_dems = [a_dems[i] for i in idxs]
        grp_traits = [traits_map[n.op_id] for n in grp_nodes]
        collapsed = _collapse_candidate_group(
            grp_nodes, grp_dems, cand, grp_traits, floors, p0, case, datapath,
            primitive_table=primitive_table, allow_unbound_formula=allow_unbound_formula,
        )
        if collapsed is None:
            continue
        fused_node, fused_dem, kspec = collapsed
        consumed_idxs.update(idxs)
        replacements[idxs[0]] = (fused_node, fused_dem)
        unlocked_ops.update(cand.member_ids)
        specs.append(kspec)
        refs = tuple(OpRef("atom", mid) for mid in cand.member_ids)
        edits.append(Edit(
            edit_id=f"{aid}_{cand.candidate_id}",
            operation=FuseRegion(
                members=refs,
                preserve_ports=tuple(PortRef(OpRef("atom", mid), "output", 0) for mid in cand.preserve_port_ids),
                kernel_boundaries=(KernelBoundary(cand.abi_id, refs),),
                fusion_tier=cand.fusion_tier,
            ),
        ))
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
) -> SolvedTargetGraph:
    """Solve target-state graph via Tier-P2 algebra + white-box L0-pipe ``Engine`` pricing."""
    floors = extract_sku_floors(case, primitive_table=primitive_table)
    die = case.machine.dies[0]
    ub_edges = ub_peer_edges_for_rank(die.rank, case.machine.links, case.machine.transfer_channels)
    datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
    p0 = peaks_by_stage.get(0) or next(iter(peaks_by_stage.values()), {})
    ext_plan = load_rewrite_plan(rewrite_plan)

    by_arch: dict[str, tuple[list[TargetOpNode], list[tuple[int, int, int, int, int]]]] = {}
    arch_order: list[str] = []
    for node, dem in zip(nodes, demands):
        if node.archetype_id not in by_arch:
            by_arch[node.archetype_id] = ([], [])
            arch_order.append(node.archetype_id)
        by_arch[node.archetype_id][0].append(node)
        by_arch[node.archetype_id][1].append(dem)

    all_nodes: list[TargetOpNode] = []
    all_dems: list[tuple[int, int, int, int, int]] = []
    all_specs: list[KernelSynthesisSpec] = []
    all_triggers: list[SemanticAuditTrigger] = []
    applied_edits: list[Edit] = []
    global_remap: dict[OpOccurrence, OpOccurrence] = {}
    safe_n_tok = max(float(n_tok_per_die), SAVING_POSITIVE_EPS_US)

    if ext_plan is not None and ext_plan.edits:
        all_known = {n.op_id for n in nodes}
        _verify_arch_plan(case, [_node_to_op_spec(n, all_known) for n in nodes], ext_plan, datapath)

    for aid in arch_order:
        a_nodes, a_dems = by_arch[aid]
        o_nodes, o_dems, o_specs, o_trigs, o_edits, o_remap = _solve_one_archetype(
            aid, a_nodes, a_dems, ext_plan, floors, p0, case, datapath,
            safe_n_tok=safe_n_tok, primitive_table=primitive_table,
            allow_unbound_formula=allow_unbound_formula,
        )
        all_nodes.extend(o_nodes)
        all_dems.extend(o_dems)
        all_specs.extend(o_specs)
        all_triggers.extend(o_trigs)
        applied_edits.extend(o_edits)
        global_remap.update(o_remap)

    rewired = _rewire_node_preds(all_nodes, global_remap)
    enveloped = _apply_pillar23_envelopes(
        rewired, all_dems, floors, case,
        phase=phase, primitive_table=primitive_table, allow_unbound_formula=allow_unbound_formula,
    )
    scheduled = reschedule_node_timelines(enveloped)
    attributed, arch_spans = attribute_node_overlap(scheduled)
    indexed = tuple(replace(n, ordinal=idx) for idx, n in enumerate(attributed))
    unbound_ops = tuple(n.op_id for n in indexed if n.binding_state == "unbound_formula_projection")
    has_unbound = bool(unbound_ops)
    graph_state: BindingState = "unbound_formula_projection" if has_unbound else "bound_implementation"

    final_plan = (
        RewritePlan(
            base_graph_sha256=ext_plan.base_graph_sha256 if ext_plan is not None else "",
            framing_sha256=ext_plan.framing_sha256 if ext_plan is not None else "",
            edits=tuple(applied_edits),
        )
        if applied_edits else (ext_plan or empty_plan("", ""))
    )
    return SolvedTargetGraph(
        nodes=indexed,
        demands_ns=tuple(all_dems),
        archetype_spans_us=arch_spans,
        synthesis_specs=tuple(all_specs),
        semantic_audit_triggers=tuple(all_triggers),
        applied_plan=final_plan,
        rewrite_plan_sha256=plan_sha256(final_plan),
        verification_passed=True,
        floors=floors,
        has_unbound_formula_ops=has_unbound,
        binding_state=graph_state,
        unbound_formula_ops=unbound_ops,
    )


__all__ = [
    "BindingState",
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
