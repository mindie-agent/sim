"""Tensor contracts and physical phases for hyper connections.

[Role] Derive arithmetic, ownership, and memory traffic from caller geometry.
[Invariants] The same phase graph prepares work, describes it, and lowers it
  to the Engine. L2 reuse is local to an invocation's ordered history passes.
[Timeouts & Locks] Bounded by physical owners and operator phases. No locks.
[Side-Effect Policy] No external I/O or device runtime. Cache admission is
  evaluated in a fresh analytical view for each plan or lowering.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Literal

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, RecipeContext, Stage
from ...results import FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.memory_physics import (
    DEFAULT_SECTOR_BYTES,
    AddressRun,
    AddressRunPlan,
    AdmissionMemoryView,
    HbmBusSpec, MemoryAccessPattern,
    ReuseKey,
    VisibilityContract, bind_memory_access,
)
from ..contracts import OperatorPlanningContext

FAMILY = "hyper_connections"
ARCH22 = ArchId("npu_arch_2201")
ARCH35 = ArchId("npu_arch_3510")
FP32_BYTES = 4
RESIDUAL_RANK = 3
RMS_SCALAR_OPS = 3
MHC_COEFFICIENT_SCALARS = 3
ENGRAM_PRODUCT_OPS_PER_ELEMENT = 5
# Two inverse RMS values and their scaled dot, signed root with clamp, then sigmoid and mask.
ENGRAM_GATE_SCALAR_OPS = 20
ATTNRES_SOFTMAX_OPS_PER_ROW = 5
HC_SINKHORN_STREAMS = 4
SINKHORN_DEFAULT_ITERS = 20
SINKHORN_INIT_ENTRY_OPS = 3
SILU_OPS_PER_ELEMENT = 4
MHC_COLLAPSE_MAX_ROWS = 72
TEXT_GATED_EZ1008_MIN_ROWS = 513
TEXT_GATED_LARGE_MIN_ROWS = 512
TEXT_GATED_SMALL_SPLITK_FACTOR = 14
TEXT_GATED_SMALL_MAX_STREAMS = 10
TEXT_GATED_SMALL_HIDDEN_ALIGNMENT = 256
TEXT_GATED_SMALL_MAX_HIDDEN = 4096
TEXT_GATED_SMALL_RANK_ALIGNMENT = 64
TEXT_GATED_SMALL_MAX_RANK = 896
TEXT_GATED_TASK_ALIGNMENT = 32
TEXT_GATED_FULL_TASK_ROWS = 128
ATTNRES_UB_OVERHEAD_BYTES = 16384
ATTNRES_FP32_SCRATCH_ROWS = 4
ENGRAM_REDUCTIONS = 3
ENGRAM_CHUNK_BYTES_PER_ELEMENT = 12

VLLM_REVISION = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
VLLM_CALLER_SOURCES: tuple[SourceRef, ...] = (
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp",
              "29bd7606b274d6a9289ce02b6e6cc03ff947c3a163850263cfea622b53b5d26c"),
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_tiling.cpp",
              "b77dc41317386e5ac4d9b6b71e9ef5249c52895b5d37163c54b0cd603ff4c361"),
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_legacy_plan.h",
              "47c863f70950dcc4ca833aa2e5d2db2cfdb980e4c6475198ee0711d74d3149c3"),
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_reload.h",
              "46b3a1fcec42b5bac511021d1059b4e91ddbb78426563b5a0343d1a9e73db611"),
    SourceRef("vllm-ascend", VLLM_REVISION, "vllm_ascend/ops/triton/engram_gate.py",
              "aa52205e9207af272e6179ebf1b6bbd552de65ef6f1472db6de893aded29e47b"),
)

MHC_KINDS = frozenset({"hc_pre", "hc_post", "hc_pre_sinkhorn", "mhc_post_collapse"})
LEGACY_ALIASES = {
    "npu_hc_pre_v2_arch35": "hc_pre",
    "npu_hc_pre_v3_arch35": "hc_pre",
    "npu_hc_post_arch35": "hc_post",
    "mhc_post_collapse_norm_arch35": "mhc_post_collapse",
    "text_gated_residual_arch35": "text_gated_residual",
}
ABI_NAMES = {
    "hc_pre": frozenset({"npu_hc_pre", "npu_hc_pre_v2", "npu_hc_pre_v3"}),
    "hc_post": frozenset({"npu_hc_post"}),
    "hc_pre_sinkhorn": frozenset({"aclnnHcPreSinkhorn"}),
    "mhc_post_collapse": frozenset({"mhc_post_collapse_norm"}),
    "text_gated_residual": frozenset({"aclnnTextGatedResidual"}),
    "attn_res_fwd": frozenset({"npu_attn_res_fwd"}),
    "engram_gate": frozenset({"fused_engram_gate"}),
}
OWNED_RECIPES = frozenset(ABI_NAMES) | LEGACY_ALIASES.keys()
FAMILY_DEFAULTS = {"attention_residual": "attn_res_fwd", "engram_gate": "engram_gate"}
HISTORY_KEY = ReuseKey(die=0, backing_allocation="attnres_history_bank", tensor_version=0)


def gap(code: str, detail: str) -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(
        capability=FAMILY, entry=f"{FAMILY}.resolve_operator_plan", code=code, message=detail,
    )


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"hyper_connections_invalid_{name}:{value}")


def _dtype(value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ModelingError(f"hyper_connections_invalid_dtype:{value}")


def _ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor

@dataclass(frozen=True, slots=True)
class MhcContract:
    rows: int
    hc_mult: int
    hidden_size: int
    n_mix: int | None = None
    sinkhorn_iters: int = SINKHORN_DEFAULT_ITERS
    split_k: int = 1
    mhc_aux_cores: int | None = None
    dtype: str = "bf16"
    has_collapse: bool = False
    unbounded_rows: bool = False

    def __post_init__(self) -> None:
        _integer("rows", self.rows)
        _integer("hc_mult", self.hc_mult, 1)
        _integer("hidden_size", self.hidden_size, 1)
        _integer("split_k", self.split_k, 1)
        _integer("sinkhorn_iters", self.sinkhorn_iters, 1)
        _dtype(self.dtype, {"bf16", "fp16", "fp32"})
        width = (2 + self.hc_mult) * self.hc_mult
        if self.n_mix is None:
            object.__setattr__(self, "n_mix", width)
        _integer("n_mix", self.n_mix, width)
        if self.split_k > self.hc_mult * self.hidden_size:
            raise ModelingError("hc_split_k_exceeds_input_width")
        if self.mhc_aux_cores is not None:
            _integer("mhc_aux_cores", self.mhc_aux_cores)
        if self.has_collapse and not self.unbounded_rows and self.rows > MHC_COLLAPSE_MAX_ROWS:
            raise ModelingError(f"mhc_post_collapse_row_cutoff_exceeded:{self.rows}")

@dataclass(frozen=True, slots=True)
class TextGatedResidualContract:
    rows: int
    hc_mult: int
    hidden_size: int
    d_dim: int
    rank: int
    has_gate: bool = False
    has_gamma: bool = False
    dtype: str = "bf16"
    stack_binding: Literal["ez1008", "cann_current"] = "cann_current"
    strict_native: bool = False

    def __post_init__(self) -> None:
        _integer("rows", self.rows)
        for name in ("hc_mult", "hidden_size", "d_dim", "rank"):
            _integer(name, getattr(self, name), 1)
        _dtype(self.dtype, {"bf16", "fp16"})
        if self.stack_binding not in {"ez1008", "cann_current"}:
            raise ModelingError(f"text_gated_residual_invalid_stack_binding:{self.stack_binding}")

@dataclass(frozen=True, slots=True)
class TextGatedGuardDecision:
    selected_template: str
    splitk_factor: int = 1
    m_task: int = 0
    used_aic: int = 0
    excluded_fusion: str | None = None


def evaluate_text_gated_guard(
    contract: TextGatedResidualContract, arch_id: ArchId, aic_cores: int,
) -> TextGatedGuardDecision:
    """Evaluate the declared stack's row and native tiling domain."""
    m, c, h, r = contract.rows, contract.hc_mult, contract.hidden_size, contract.rank
    _integer("aic_cores", aic_cores)
    if m == 0:
        return TextGatedGuardDecision("empty_noop")
    if arch_id not in {ARCH22, ARCH35}:
        raise gap("text_gated_residual_arch_unsupported", str(arch_id))
    reason = None
    if contract.stack_binding == "ez1008":
        if m < TEXT_GATED_EZ1008_MIN_ROWS:
            reason = "ez1008_m_le_512"
    elif m < TEXT_GATED_LARGE_MIN_ROWS:
        if arch_id == ARCH22:
            task = _ceil(m, 2) if m >= TEXT_GATED_FULL_TASK_ROWS else m
            if m >= TEXT_GATED_SMALL_RANK_ALIGNMENT:
                task = _ceil(task, TEXT_GATED_TASK_ALIGNMENT) * TEXT_GATED_TASK_ALIGNMENT
            return TextGatedGuardDecision("native_a2_small", 1, task, min(aic_cores, _ceil(m, task)))
        if not 2 <= c <= TEXT_GATED_SMALL_MAX_STREAMS:
            reason = "a5_small_hc_mult"
        elif not (TEXT_GATED_SMALL_HIDDEN_ALIGNMENT <= h <= TEXT_GATED_SMALL_MAX_HIDDEN
                  and h % TEXT_GATED_SMALL_HIDDEN_ALIGNMENT == 0):
            reason = "a5_small_hidden_size"
        elif not (TEXT_GATED_SMALL_RANK_ALIGNMENT <= r <= TEXT_GATED_SMALL_MAX_RANK
                  and r % TEXT_GATED_SMALL_RANK_ALIGNMENT == 0):
            reason = "a5_small_rank"
        elif aic_cores == 0:
            reason = "a5_small_aic_cores"
        else:
            return TextGatedGuardDecision("native_a5_small", TEXT_GATED_SMALL_SPLITK_FACTOR, m, aic_cores)
    if reason is not None:
        if contract.strict_native:
            code = "ez1008_row_floor_not_met" if reason == "ez1008_m_le_512" else reason
            raise ModelingError(f"text_gated_residual_{code}")
        return TextGatedGuardDecision(
            "decomposed_fallback", 1, m, min(m, aic_cores), excluded_fusion=reason,
        )
    task = TEXT_GATED_FULL_TASK_ROWS
    if _ceil(m, task) < aic_cores:
        task //= 2
    return TextGatedGuardDecision("native_large", 1, task, min(aic_cores, _ceil(m, task)))

@dataclass(frozen=True, slots=True)
class AttnResContract:
    tokens: int
    hidden_size: int
    capacity: int
    valid_history: int
    token_stride: int | None = None
    fuse_add: bool = True
    output_norm: bool = False
    cached_rows: int = 0
    dtype: str = "bf16"
    ub_overhead_bytes: int = ATTNRES_UB_OVERHEAD_BYTES

    def __post_init__(self) -> None:
        _integer("tokens", self.tokens)
        _integer("hidden_size", self.hidden_size, 1)
        _integer("capacity", self.capacity, 1)
        _integer("valid_history", self.valid_history)
        _integer("cached_rows", self.cached_rows)
        _integer("ub_overhead_bytes", self.ub_overhead_bytes)
        _dtype(self.dtype, {"bf16"})
        stride = self.capacity * self.hidden_size
        if self.token_stride is None:
            object.__setattr__(self, "token_stride", stride)
        _integer("token_stride", self.token_stride, stride)
        if not self.cached_rows <= self.valid_history <= self.capacity:
            raise ModelingError("attnres_history_out_of_capacity")


def attnres_max_resident_rows(
    hidden: int, ub_bytes: int, *, fuse_add: bool = True,
    ub_overhead_bytes: int = ATTNRES_UB_OVERHEAD_BYTES,
    physical_capacity_cap: int | None = None,
) -> int:
    """Compute the legacy resident admission bound and optional physical cap."""
    _integer("hidden_size", hidden, 1)
    _integer("ub_bytes", ub_bytes)
    _integer("ub_overhead_bytes", ub_overhead_bytes)
    row = 2 * hidden
    fixed = ATTNRES_FP32_SCRATCH_ROWS * hidden * FP32_BYTES + 2 * row + ub_overhead_bytes
    limit = max(0, (ub_bytes - fixed - (row if fuse_add else 0)) // (row + FP32_BYTES))
    if physical_capacity_cap is not None:
        _integer("physical_capacity_cap", physical_capacity_cap)
        limit = min(limit, physical_capacity_cap)
    return limit

@dataclass(frozen=True, slots=True)
class EngramGateContract:
    tokens: int
    hidden_size: int
    active_tokens: int | None = None
    chunk_dim: int | None = None
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        _integer("tokens", self.tokens)
        _integer("hidden_size", self.hidden_size, 1)
        _dtype(self.dtype, {"bf16"})
        if self.active_tokens is None:
            object.__setattr__(self, "active_tokens", self.tokens)
        if self.chunk_dim is None:
            object.__setattr__(self, "chunk_dim", self.hidden_size)
        _integer("active_tokens", self.active_tokens)
        _integer("chunk_dim", self.chunk_dim, 1)
        if self.active_tokens > self.tokens:
            raise ModelingError("engram_gate_active_tokens_exceed_tokens")


def sinkhorn_vector_ops(tokens: int, hc_mult: int, iters: int) -> int:
    """Count row-softmax initialization and subsequent row/column half-steps."""
    for label, value, minimum in (("tokens", tokens, 0), ("hc_mult", hc_mult, 1), ("sinkhorn_iters", iters, 0)):
        _integer(label, value, minimum)
    adds = hc_mult * (hc_mult - 1)
    entries = hc_mult * hc_mult
    # The initial row softmax is the first normalization. Later denominators
    # add epsilon, while initialization also shifts logits and stabilizes entries.
    initialization = adds + SINKHORN_INIT_ENTRY_OPS * entries
    half_steps = 2 * iters
    return tokens * (initialization + half_steps * (adds + entries) + max(0, half_steps - 1) * hc_mult)

@dataclass(frozen=True, slots=True)
class Phase:
    name: str
    after: tuple[str, ...]
    owner_kind: Literal["aic", "aiv"]
    owners: tuple[int, ...]
    work: Work = field(default_factory=Work)
    access: MemoryAccessPattern | None = None
    metadata: tuple[tuple[str, int | str], ...] = ()

@dataclass(frozen=True, slots=True)
class PhaseProgram:
    """A picklable staged program whose access bindings use declared pipe service."""

    phases: tuple[Phase, ...]
    aic_count: int
    aiv_count: int
    l2_capacity: int
    l2_units: int

    def _memory(self) -> AdmissionMemoryView:
        return AdmissionMemoryView({0: self.l2_capacity}, l2_buffer_units_by_die={0: self.l2_units})

    def phase_work(self) -> tuple[Work, ...]:
        """Prepare the same aligned copy, sector, and L2 work the stages issue."""
        memory = self._memory()
        work = []
        for phase in self.phases:
            native = phase.work
            access = phase.access
            if access is not None:
                # Unit rates expose the byte ledgers independently of timing.
                ledger = bind_memory_access(
                    access, memory, hbm_bus=HbmBusSpec(0, 1, 1, 0),
                    copy_hit_bytes_per_ms=1, copy_miss_bytes_per_ms=1, l2_bytes_per_ms=1,
                    pipe_row_id=access.pipe, l2_row_id="l2", commit=True,
                ).ledger
                native += Work(resources={access.pipe: ledger.aligned_transfer_bytes,
                                          "hbm": ledger.dram_bus_transaction_bytes, "l2": ledger.l2_hit_bytes})
            work.append(native)
        return tuple(work)

    def __call__(self, ctx: RecipeContext) -> KernelProgram:
        memory = self._memory()
        stages = []
        for phase in self.phases:
            demand = ctx.bind(phase.work)
            roles = list(phase.work.resources) + (["cube"] if phase.work.matrix else [])
            metadata = dict(phase.metadata)
            access = phase.access
            if access is not None:
                pipe = access.pipe if access.pipe in ctx.resource_names else "mte2"
                name = ctx.resource_names[pipe]
                units = ctx.resource_units.get(pipe, ctx.resource_units.get(name, 1))
                nominal = ctx.rates[name] / units
                hit_rate = ctx.service_rate(f"{pipe}_l2")
                miss_rate = ctx.service_rate(f"{pipe}_dram")
                bound = ctx.bind_access(
                    access, memory, commit=True,
                    copy_hit_bytes_per_ms=nominal if hit_rate is None else hit_rate / (1000 * units),
                    copy_miss_bytes_per_ms=nominal if miss_rate is None else miss_rate / (1000 * units),
                )
                # Standard execution exposes a shared HBM row. The copy
                # constraint is normalized to one physical pipe unit.
                demand[name] = bound.stage_work(resource_rates={name: nominal}).get(name, 0)
                ledger = bound.ledger
                for role, amount in (("hbm", ledger.dram_bus_transaction_bytes), ("l2", ledger.l2_hit_bytes)):
                    if amount and (row := ctx.resource_names.get(role)) is not None:
                        demand[row] = amount
                roles.append(pipe)
                metadata.update(logical_bytes=ledger.logical_tensor_bytes, l2_hit_bytes=ledger.l2_hit_bytes)
            domain = self.aic_count if phase.owner_kind == "aic" else self.aiv_count
            placement = {}
            for role in roles:
                if role in {"hbm", "l2"}:
                    continue
                resource = ctx.resource_names[ctx.physical_role(role)]
                if resource not in demand:
                    continue
                units = ctx.resource_units.get(ctx.physical_role(role), domain)
                placement[resource] = tuple(sorted({
                    unit for owner in phase.owners for unit in unit_group(owner, domain, units)
                }))
            stages.append(Stage(phase.name, demand, after=phase.after, placement=placement,
                                coupled=access is not None, metadata=metadata))
        return KernelProgram(tuple(stages))


class PhaseGraph:
    """Assemble a bounded invocation DAG with explicit AIC and AIV ownership."""

    def __init__(self, ctx: OperatorPlanningContext) -> None:
        self.ctx = ctx
        self.phases: list[Phase] = []
        self.aic = self._owners("aic")
        self.aiv = self._owners("aiv")
        spec = ctx.datapath.memory_access
        self.sector = spec.sector_bytes if spec is not None else DEFAULT_SECTOR_BYTES

    def _owners(self, kind: Literal["aic", "aiv"]) -> tuple[int, ...]:
        available = getattr(self.ctx, f"effective_{kind}_cores")
        owners = self.ctx.stream_owners
        if owners is not None and owners.isolation == "disjoint":
            return tuple(getattr(owners, kind))[:available]
        return tuple(range(available))

    def add(self, name, after=(), *, work=None, owners=None, aic=False,
            metadata=(), access=None) -> str:
        domain = self.aic if aic else self.aiv
        owners = domain if owners is None else owners
        if not owners:
            kind = "aic" if aic else "aiv"
            raise gap(f"hyper_connections_requires_{kind}_owners", name)
        self.phases.append(Phase(name, after, "aic" if aic else "aiv", owners,
                                 work or Work(), access, metadata))
        return name

    def transfer(self, name, size, after=(), *, aic=False, write=False, owners=None,
                 run=None, reuse=None, distance=None) -> str:
        if owners is None:
            owners = self.aic if aic else self.aiv
        pipe = ("fixpipe" if aic else "mte3") if write else ("mte2_aic" if aic else "mte2_aiv")
        runs = AddressRunPlan((run or AddressRun(0, 1, size, size),), sector_bytes=self.sector)
        access = MemoryAccessPattern(
            tensor_id=reuse.backing_allocation if reuse else name, version=0,
            die=0, owner=owners[0] if owners else 0, direction="write" if write else "read",
            pipe=pipe, logical_bytes=size, runs=runs, reuse_key=reuse,
            reuse_distance_bytes=distance, visibility=VisibilityContract(live_out=write),
        )
        return self.add(name, after, owners=owners, aic=aic, access=access)

    def vector(self, name, amount, after, *, owners=None, metadata=()) -> str:
        return self.add(name, after, work=Work(resources={"vector": amount}),
                        owners=owners, metadata=metadata)

    def finish(self, implementation, dtype, sources, metadata) -> OperatorPlan:
        buffers = dict(self.ctx.datapath.buffers)
        l2 = buffers.get("l2", self.ctx.geometry.buffers.get("l2", 0))
        program = PhaseProgram(tuple(self.phases), self.ctx.datapath.aic_count,
                               self.ctx.datapath.aiv_count, l2,
                               self.ctx.geometry.buffer_units.get("l2", 1))
        total = Work()
        phases = []
        for phase, work in zip(program.phases, program.phase_work(), strict=True):
            total += work
            items = [WorkItem("cube", "flop", flops, precision) for precision, flops in work.matrix.items()]
            items.extend(WorkItem(role, "op" if role == "vector" else "bytes", value, dtype)
                         for role, value in work.resources.items() if value)
            phases.append(PipelinePhase(phase.name, phase.after, tuple(items)))
        spec = FusedPipelineSpec(FAMILY, implementation, tuple(phases), sources)
        return OperatorPlan(total, program, metadata, pipeline_spec=spec)


def _matrix(graph, name, m, k, n, dtype, after, owners, *, gram=0) -> str:
    elt = int(dtype_bytes(dtype))
    l1 = graph.add(f"{name}_operands", after, aic=True, owners=owners,
                   work=Work(resources={"mte1": (m + n) * k * elt}))
    work = Work(matrix={dtype: 2 * m * k * n})
    if gram:
        work += Work(matrix={"fp32": gram})
    return graph.add(name, (l1,), aic=True, owners=owners, work=work)


def _partition(graph: PhaseGraph, contract: MhcContract, pre: bool):
    m, c, h = contract.rows, contract.hc_mult, contract.hidden_size
    cores = len(graph.aiv)
    aux = contract.mhc_aux_cores
    if aux is None:
        aux = 0
        if pre and cores > 1:
            primary_work = h * (2 * c - 1)
            sink_work = sinkhorn_vector_ops(1, c, contract.sinkhorn_iters)
            aux = min(range(1, cores), key=lambda n: (
                max(_ceil(m, cores - n) * primary_work, _ceil(m, n) * sink_work), n,
            ))
    if aux and aux >= cores:
        raise ModelingError("mhc_aux_leaves_no_primary_cores")
    return graph.aiv[:cores - aux], graph.aiv[cores - aux:] if aux else ()


def model_mhc(graph: PhaseGraph, kind: str, t: MhcContract) -> dict:
    m, c, h = t.rows, t.hc_mult, t.hidden_size
    if kind == "hc_pre_sinkhorn" and (c, t.sinkhorn_iters) != (HC_SINKHORN_STREAMS, SINKHORN_DEFAULT_ITERS):
        raise ModelingError("hc_pre_sinkhorn_requires_hc4_and_iters20")
    pre = kind in {"hc_pre", "hc_pre_sinkhorn"}
    sink_ops = sinkhorn_vector_ops(m, c, t.sinkhorn_iters) if pre else 0
    mix = t.n_mix
    metadata = {"split_k": t.split_k, "sinkhorn_ops": sink_ops,
                "projection_flops": 2 * m * c * h * mix if pre else 0}
    if not m:
        return {**metadata, "mhc_aux_cores": 0, "primary_aiv_cores": 0}
    primary, aux = _partition(graph, t, pre)
    primary = primary[:m]
    metadata.update(mhc_aux_cores=len(aux), primary_aiv_cores=len(graph.aiv) - len(aux))
    elt = int(dtype_bytes(t.dtype))
    x_bytes = m * c * h * elt
    coefficients = m * c * (c + 1) * FP32_BYTES
    if not pre:
        read = graph.transfer("post_inputs", x_bytes + m * h * elt + coefficients, owners=primary)
        # Each output stream j contracts input stream i with comb[i, j],
        # then adds layer_output * post[j] to the same owner-local row.
        post = graph.vector("stream_contraction", m * c * h * (2 * c + 1), (read,),
                            owners=primary, metadata=(("coefficient_orientation", "input_to_output"),))
        size = x_bytes
        if t.has_collapse:
            post = graph.vector("collapse_and_norm", m * (h * (2 * c - 1 + 2 + 2) + 2),
                                (post,), owners=primary)
            size = m * h * elt
        graph.transfer("post_output", size, (post,), write=True, owners=primary)
        return metadata

    aic = graph.aic[:m * t.split_k]
    load = graph.transfer("projection_inputs", x_bytes + c * h * mix * elt, aic=True, owners=aic)
    split = t.split_k
    projection = _matrix(
        graph, "split_k_partials" if split > 1 else "mix_projection", m, c * h, mix,
        t.dtype, (load,), aic, gram=m * (2 * c * h - split),
    )
    partials = m * (mix + 1) * split * FP32_BYTES
    publish = graph.transfer("projection_workspace_write", partials, (projection,), aic=True, write=True, owners=aic)
    read = graph.transfer("projection_workspace_read", partials, (publish,), owners=primary)
    if split > 1:
        read = graph.vector("split_k_merge", m * (mix + 1) * (split - 1), (read,), owners=primary)
    coef = graph.transfer("coefficient_parameters", (mix + MHC_COEFFICIENT_SCALARS) * FP32_BYTES,
                          (read,), owners=primary)
    ready = graph.vector("norm_scale_bias", m * (RMS_SCALAR_OPS + MHC_COEFFICIENT_SCALARS * mix + 2 * c * (2 + 2)),
                         (coef,), owners=primary)
    if aux:
        handoff = graph.transfer("sinkhorn_workspace_write", m * c * c * FP32_BYTES,
                                 (ready,), write=True, owners=primary)
        sink_input = graph.transfer("sinkhorn_workspace_read", m * c * c * FP32_BYTES,
                                    (handoff,), owners=aux[:m])
    else:
        sink_input = ready
    sink_owners = aux[:m] or primary
    sink = graph.vector("sinkhorn", sink_ops, (sink_input,), owners=sink_owners,
                        metadata=(("stream", "mhc_aux" if aux else "main"),))
    graph.transfer("comb_output", m * c * c * FP32_BYTES, (sink,), write=True, owners=sink_owners)
    residual = graph.transfer("residual_for_pre_mix", x_bytes, (ready,), owners=primary)
    mixed = graph.vector("pre_mix", m * h * (2 * c - 1), (residual,), owners=primary)
    graph.transfer("pre_and_post_gate_outputs", m * h * elt + m * c * FP32_BYTES,
                   (mixed,), write=True, owners=primary)
    return metadata


def model_text_gate(graph: PhaseGraph, t: TextGatedResidualContract, decision: TextGatedGuardDecision) -> dict:
    m, c, h, d, r = t.rows, t.hc_mult, t.hidden_size, t.d_dim, t.rank
    template = decision.selected_template
    metadata = {"selected_template": template, "splitk_factor": decision.splitk_factor,
                "m_task": decision.m_task, "used_aic": decision.used_aic,
                "excluded_fusion": decision.excluded_fusion}
    if not m:
        return metadata
    aic = graph.aic[:decision.used_aic]
    aiv = graph.aiv[:m]
    elt = int(dtype_bytes(t.dtype))
    split, raw = decision.splitk_factor, r + (c if t.has_gate else 0)
    residual_bytes = m * d * elt
    gamma_bytes = d * FP32_BYTES if t.has_gamma else 0
    x = graph.transfer("text_input", residual_bytes + gamma_bytes, owners=aiv)
    rms_groups = c if d == c * h else 1
    rms = m * (2 * d + d - rms_groups + rms_groups * RMS_SCALAR_OPS + d * int(t.has_gamma))
    norm = graph.vector("input_rms", rms, (x,), owners=aiv)
    xn = graph.transfer("normalized_workspace_write", residual_bytes, (norm,), write=True, owners=aiv)
    inp = graph.transfer("down_and_gate_inputs", residual_bytes + d * raw * elt, (xn,), aic=True, owners=aic)
    down = _matrix(graph, "down_and_gate_partials" if split > 1 else "down_and_gate",
                   m, d, raw, t.dtype, (inp,), aic)
    down_bytes = m * raw * (FP32_BYTES * split if split > 1 else elt)
    write = graph.transfer("down_workspace_write", down_bytes, (down,), aic=True, write=True, owners=aic)
    read = graph.transfer("down_workspace_read", down_bytes, (write,), owners=aiv)
    if split > 1:
        read = graph.vector("text_split_k_merge", m * raw * (split - 1), (read,), owners=aiv)
        raw_out = graph.transfer("merged_down_write", m * raw * elt, (read,), write=True, owners=aiv)
        read = graph.transfer("merged_down_read", m * raw * elt, (raw_out,), owners=aiv)
    activation = graph.vector("down_activation", m * r * SILU_OPS_PER_ELEMENT, (read,), owners=aiv)
    lowrank = graph.transfer("lowrank_workspace_write", m * r * elt,
                             (activation,), write=True, owners=aiv)
    up_input = graph.transfer("up_inputs", (m * r + r * d) * elt,
                              (lowrank,), aic=True, owners=aic)
    up = _matrix(graph, "up_projection", m, r, d, t.dtype, (up_input,), aic)
    up_bytes = m * d * elt
    up_write = graph.transfer("up_workspace_write", up_bytes, (up,), aic=True, write=True, owners=aic)
    gate_bytes, mixed_bytes = m * c * elt * int(t.has_gate), m * h * elt
    if gate_bytes:
        activation = graph.transfer("inject_output", gate_bytes, (activation,), write=True, owners=aiv)
    inputs = graph.transfer("gated_reduction_inputs", up_bytes + residual_bytes + gate_bytes,
                            (up_write, activation), owners=aiv)
    mix_ops = m * h * (c - 1 + c * int(t.has_gate))
    if template == "decomposed_fallback":
        gated = graph.vector("gated_reductions", 2 * mix_ops, (inputs,), owners=aiv)
        materialized = graph.transfer("decomposed_gate_write", 2 * mixed_bytes, (gated,), write=True, owners=aiv)
        inputs = graph.transfer("decomposed_residual_inputs", 2 * mixed_bytes, (materialized,), owners=aiv)
        mixed = graph.vector("residual_add", m * h, (inputs,), owners=aiv)
    else:
        mixed = graph.vector("residual_add_and_gated_reduction", m * d + mix_ops, (inputs,), owners=aiv)
    graph.transfer("text_gated_output", mixed_bytes, (mixed,), write=True, owners=aiv)
    return {**metadata, "projection_flops": 2 * m * d * (raw + r)}


def model_attnres(graph: PhaseGraph, t: AttnResContract, ub: int, physical_cap: int | None,
                  bank_write: bool = False) -> dict:
    m, h, j = t.tokens, t.hidden_size, t.valid_history
    limit = attnres_max_resident_rows(h, ub, fuse_add=t.fuse_add,
                                     ub_overhead_bytes=t.ub_overhead_bytes, physical_capacity_cap=physical_cap)
    resident = j + 1 <= limit
    cached = j if resident else t.cached_rows
    template = "resident" if resident else "cooperative_reload" if 0 < cached < j else "reload"
    row = 2 * h
    metadata = {"resident_row_limit": limit, "selected_template": template,
                "reread_bytes": m * (j - cached) * row, "history_read_bytes": m * j * row}
    if m and not graph.aiv:
        raise gap("hyper_connections_requires_aiv_owners", "attn_res_fwd")
    if not m:
        return metadata
    stride = 2 * t.token_stride
    distance = AddressRunPlan((AddressRun(0, m, j * row, stride),),
                              sector_bytes=graph.sector).aligned_copy_bytes()
    scores = []
    per_owner = _ceil(m, len(graph.aiv))
    for index, start in enumerate(range(0, m, per_owner)):
        rows = min(m - start, per_owner)
        owners = (graph.aiv[index],)
        inp = graph.transfer(f"prefix_input_{index}", rows * row * (1 + int(t.fuse_add)) + 2 * row,
                             owners=owners)
        deps = (inp,)
        if j:
            history = graph.transfer(f"history_pass1_{index}", rows * j * row, owners=owners,
                                     run=AddressRun(start * stride, rows, j * row, stride),
                                     reuse=HISTORY_KEY, distance=distance)
            deps += (history,)
        score_per_row = 2 * (h - 1) + (2 + 2) * h + RMS_SCALAR_OPS
        softmax = ATTNRES_SOFTMAX_OPS_PER_ROW * (j + 1) - 1
        fuse = (2 + 2) * h if t.fuse_add else 0
        score = graph.vector(f"score_softmax_{index}", rows * ((j + 1) * score_per_row + softmax + fuse),
                             deps, owners=owners)
        scores.append((score, index, start, rows, owners))
    score_ready = tuple(score for score, *_ in scores)
    for score, index, start, rows, owners in scores:
        deps = (score,)
        if j > cached:
            reread = graph.transfer(f"history_pass2_{index}", rows * (j - cached) * row, score_ready,
                                    owners=owners, run=AddressRun(start * stride + cached * row,
                                                                 rows, (j - cached) * row, stride),
                                    reuse=HISTORY_KEY, distance=distance)
            deps = (reread,)
        if t.output_norm:
            deps += (graph.transfer(f"output_gamma_{index}", row, owners=owners),)
        norm = (2 + 2) * h + 2 * h + h + RMS_SCALAR_OPS if t.output_norm else 0
        weighted = graph.vector(f"weighted_sum_{index}", rows * (2 * (j + 1) * h + norm),
                                deps, owners=owners)
        graph.transfer(f"attnres_output_{index}", rows * row * (1 + int(t.fuse_add) + int(bank_write)),
                       (weighted,), write=True, owners=owners)
    return metadata


def model_engram(graph: PhaseGraph, t: EngramGateContract, ub: int) -> dict:
    m, h, active = t.tokens, t.hidden_size, t.active_tokens
    chunk = min(t.chunk_dim, h, max(1, (ub - ENGRAM_REDUCTIONS * FP32_BYTES) // ENGRAM_CHUNK_BYTES_PER_ELEMENT))
    reductions = active * ENGRAM_REDUCTIONS * (h - 1)
    metadata = {"chunk_dim": chunk, "hidden_slices": _ceil(h, chunk), "sinkhorn_ops": 0,
                "active_tokens": active, "reduction_adds": reductions}
    if not m:
        return metadata
    aiv = graph.aiv[:m]
    masked = graph.transfer("token_mask", m, owners=aiv)
    elt = int(dtype_bytes(t.dtype))
    if active:
        active_owners = aiv[:active]
        inp = graph.transfer("engram_x_key_weights", active * h * elt * (2 + 2),
                             (masked,), owners=active_owners)
        reduce = graph.vector("hidden_slice_partial_reductions", reductions + active * h * ENGRAM_PRODUCT_OPS_PER_ELEMENT,
                              (inp,), owners=active_owners)
        gate = graph.vector("rms_signed_root_sigmoid_mask", active * ENGRAM_GATE_SCALAR_OPS,
                            (reduce,), owners=active_owners)
        value = graph.transfer("engram_value_and_spilled_x", active * h * elt * (1 + int(chunk < h)),
                               (gate,), owners=active_owners)
        out = graph.vector("engram_gated_add", 2 * active * h, (value,), owners=active_owners)
        graph.transfer("engram_active_output", active * h * elt, (out,), write=True, owners=active_owners)
    if active < m:
        inp = graph.transfer("engram_inactive_input", (m - active) * h * elt, (masked,), owners=aiv[:m - active])
        graph.transfer("engram_inactive_output", (m - active) * h * elt,
                       (inp,), write=True, owners=aiv[:m - active])
    return metadata


def recipe_kind(op: OpSpec) -> str:
    recipe = op.recipe or FAMILY_DEFAULTS.get(op.family) or dict(op.metadata).get("variant", op.family)
    return LEGACY_ALIASES.get(recipe, recipe)

@dataclass(frozen=True, slots=True)
class HyperConnectionsStrategy:
    """Bind the common physics to one upstream ABI and UB capacity."""

    arch_id: ArchId
    abi: str
    sources: tuple[SourceRef, ...]
    ub_bytes: int
    datapath_name: str
    exclusive_abis: frozenset[str]

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        kind = recipe_kind(op)
        if ctx.compute_arch.arch_id != self.arch_id or kind not in ABI_NAMES:
            return False
        if op.recipe in LEGACY_ALIASES and self.arch_id != ARCH35:
            return False
        abi = ctx.requested_abi
        if abi is None:
            return True
        if abi == "fusedSmall":
            return kind == "text_gated_residual" and abi in self.exclusive_abis
        return abi in ABI_NAMES[kind] or abi == kind or abi in self.exclusive_abis

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        if not self.accepts(op, ctx):
            raise gap("hyper_connections_strategy_unsupported",
                      f"{ctx.compute_arch.arch_id}:{op.recipe}:{ctx.requested_abi}")
        if op.source_boundary == "call":
            raise gap("hyper_connections_requires_expanded_kernel", op.op_id)
        kind = recipe_kind(op)
        graph = PhaseGraph(ctx)
        declared_ub = dict(ctx.datapath.buffers).get("ub", ctx.geometry.buffers.get("ub", self.ub_bytes))
        ub = min(self.ub_bytes, declared_ub)
        contract, options = operator_contract(op, kind)
        if kind in MHC_KINDS:
            metadata = model_mhc(graph, kind, contract)
        elif kind == "text_gated_residual":
            decision = evaluate_text_gated_guard(contract, self.arch_id, len(graph.aic))
            if ctx.requested_abi == "fusedSmall" and decision.selected_template not in {"native_a5_small", "empty_noop"}:
                raise gap("text_gated_residual_fused_small_abi_unsupported",
                          decision.excluded_fusion or decision.selected_template)
            metadata = model_text_gate(graph, contract, decision)
        elif kind == "attn_res_fwd":
            metadata = model_attnres(graph, contract, ub, **options)
        else:
            metadata = model_engram(graph, contract, ub)
        if graph.phases and ctx.datapath.usable_paths:
            for source, target in (("gm", "ub"), ("ub", "gm")):
                if not ctx.datapath.has_path(source, target):
                    raise gap("hyper_connections_datapath_unsupported", f"{source}->{target}")
        metadata.update(arch_id=str(self.arch_id), datapath=self.datapath_name,
                        ub_bytes=ub, implementation=kind, governed_family=FAMILY)
        return graph.finish(kind, contract.dtype, self.sources + VLLM_CALLER_SOURCES, metadata)


def operator_contract(op: OpSpec, kind: str):
    """Decode local tensors and the established flat and nested caller metadata."""
    meta = dict(op.metadata)
    for key in ("fusion_tensor_view", "hc_geometry", "attn_res_view", "contract"):
        if isinstance(view := meta.get(key), Mapping):
            meta.update(view)
    shape = op.local_shape or op.semantic_shape
    meta.setdefault("dtype", op.dtype or "bf16")
    rank = RESIDUAL_RANK if kind in MHC_KINDS or kind == "text_gated_residual" else 2
    if len(shape) != rank:
        raise ModelingError(f"hyper_connections_requires_rank{rank}:{op.op_id}")
    options = {}
    if rank == RESIDUAL_RANK:
        m, c, h = shape
        if kind == "text_gated_residual":
            schema = TextGatedResidualContract
            # Explicit channels identify the native [B,S,D] input.
            channels = meta.get("hc_mult", meta.get("channels", c))
            if "channels" in meta or channels != c:
                _integer("channels", channels, 1)
                if h % channels:
                    raise ModelingError("text_gated_residual_d_dim_not_divisible_by_channels")
                m, c, h = m * c, channels, h // channels
            meta.setdefault("rank", meta.get("hc_lowrank"))
            meta.setdefault("d_dim", c * h)
            if op.recipe in LEGACY_ALIASES:
                meta = {"stack_binding": "ez1008", "has_gate": True, **meta}
        else:
            schema = MhcContract
            meta["has_collapse"] = kind == "mhc_post_collapse" or meta.get("has_collapse", False)
        meta.update(rows=m, hc_mult=c, hidden_size=h)
    else:
        meta.update(tokens=shape[0], hidden_size=shape[-1])
        if kind == "attn_res_fwd":
            schema = AttnResContract
            meta.setdefault("capacity", meta.get("bank_capacity"))
            meta.setdefault("valid_history", meta.get("valid_blocks", 0) if meta.get("mix", True) else 0)
            meta.setdefault("token_stride", meta.get("block_token_stride"))
            options = {"physical_cap": meta.get("physical_capacity_cap"),
                       "bank_write": meta.get("block_write_idx", -1) >= 0}
        else:
            schema = EngramGateContract
            if (mask := meta.get("active_mask")) is not None:
                if len(mask) != shape[0] or any(value not in (0, 1) for value in mask):
                    raise ModelingError("engram_gate_invalid_active_mask")
                count = sum(mask)
                if meta.setdefault("active_tokens", count) != count:
                    raise ModelingError("engram_gate_active_mask_count_mismatch")
    return schema(**{item.name: meta[item.name] for item in fields(schema) if item.name in meta}), options
