"""Structural attention contracts, physical ledgers and executable phases.

[Role] Share attention geometry, KV access accounting, context-parallel
  ownership, panel selection and resource-program construction.
[Invariants] Selection work is charged once per query and head. Sector
  transactions precede L2 classification. Prepared Work and executable
  phase work have the same resource totals.
[Timeouts & Locks] Finite tiling search and streaming gather replay.
[Side-Effect Policy] No I/O. An explicit admission view may be updated by
  gather replay. Planners otherwise use private state.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import islice
from math import ceil, isfinite

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan, Work
from ...ownership import read_role, unit_group
from ...program import KernelProgram, ParallelProgram, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.l2_residency import AdmissionMemoryView, bind_memory_access
from ...runtime.memory_physics import AddressRun, AddressRunPlan, HbmBusSpec, MemoryAccessPattern, ReuseKey
from ...runtime.tile_solver import LiveBufferSpec, TileCandidate, TilePipelineSpec, TileShape, TileSolution, solve
from ..contracts import OperatorPlanningContext
from ._cp_tiling import (
    CACHE_GQA_SITE, CACHE_MLA_SITE, COMPLETE_KV_SITE, COMPLETE_KV_SPARSE_PREFILL,
    HISTORY_DECODE_SITE, HISTORY_PREFILL_SITE, VLLM_REVISION, ContextParallelPlan,
    lower_cp_collectives,
)

FIA_ABI = "aclnnFusedInferAttentionScore"
GBSA_ABI = "aclnnGenericBlockSparseAttention"
C8_ABI = "aclnnKvQuantSparseFlashAttentionV2"
NOPE_ABI = "aclnnSparseFlashMla"
MINDIE_V1_ABI = "mindie_sd.block_sparse_attention_v1"
MINDIE_V2_ABI = "mindie_sd.block_sparse_attention_v2"
MINDIE_V3_ABI = "mindie_sd.block_sparse_attention_v3"
MINDIE_ABIS = frozenset({MINDIE_V1_ABI, MINDIE_V2_ABI, MINDIE_V3_ABI})
SUPPORTED_ABIS = frozenset({FIA_ABI, GBSA_ABI, C8_ABI, NOPE_ABI}) | MINDIE_ABIS
ARCH22 = "npu_arch_2201"
ARCH35 = "npu_arch_3510"
QUANT_MODE_BF16 = 0
QUANT_MODE_INT8_C8 = 1
QUANT_MODE_MINDIE_FP8 = 1
QUANT_MODE_MXFP4_OCP = 2
QUANT_MODE_MXFP4_CX = 3
QUANT_MODE_TQ4 = 3
QUANT_MODE_FP8_C8 = 5
QUANT_MODE_MXFP8 = 6
SUPPORTED_QUANT_MODES = frozenset({
    QUANT_MODE_BF16, QUANT_MODE_INT8_C8, QUANT_MODE_MXFP4_OCP, QUANT_MODE_TQ4,
    QUANT_MODE_FP8_C8, QUANT_MODE_MXFP8,
})
C8_MODES = frozenset({QUANT_MODE_INT8_C8, QUANT_MODE_TQ4, QUANT_MODE_FP8_C8})
MINDIE_V3_MODES = frozenset({
    QUANT_MODE_BF16, QUANT_MODE_MINDIE_FP8, QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP4_CX,
})
TOKEN_MAJOR_LAYOUTS = frozenset({"BSH", "PA_BLK_BSND", "TND"})
PAGED_LAYOUTS = frozenset({"PA_BLK_BSND", "PA_BLK_BHSD"})
KV_LAYOUTS = TOKEN_MAJOR_LAYOUTS | {"BNSD", "PA_BLK_BHSD"}
FIA_TILE = 128
FIA_PROBABILITY_SLOTS = 3
WIDE_PANEL = 256
ARCH35_L0C_BYTES = 256 * 1024
L0C_QUAD_SLOTS = 4
PROBABILITY_READY = 3
PV_COMPLETE = 4
BF16_BYTES = 2
FP32_BYTES = 4
INDEX_BYTES = 4
SCALE_ADDRESS_BYTES = 8
SOFTMAX_OPS = 5
C8_LATENT_DIM = 512
C8_ROPE_DIM = 64
C8_SCALE_COUNT = 4
C8_ROW_BYTES = C8_LATENT_DIM + C8_ROPE_DIM * BF16_BYTES + C8_SCALE_COUNT * FP32_BYTES
TQ4_ROW_BYTES = C8_LATENT_DIM // 2 + C8_ROPE_DIM * BF16_BYTES + BF16_BYTES
SECTOR_BYTES = 32


def unmodeled(code: str, detail: str) -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(
        capability="attention", entry="operators.attention.build", code=code, message=detail,
        hints=({"schema": "mindie-unmodeled-structure-hint/1", "mechanism": "attention",
                "required": detail},),
    )


@dataclass(frozen=True, slots=True)
class AttentionContract:
    """Rank-local Q/K/V and selection geometry before architecture selection.

    Each sequence is (query_count, total_kv_length). Causal queries are
    end-aligned in that length. An explicit sparse list contains physical
    byte addresses for one shared KV head, one list per query. For packed
    KV, L2 reuse requires explicit addresses on a named backing allocation
    at its declared producer version.
    """

    query_tokens: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    selected_tokens: int | None = None
    sequences: tuple[tuple[int, int], ...] = ()
    phase: str = "decode"
    causal: bool = False
    window_size: int = 0
    kv_layout: str = "BSH"
    quant_mode: int = QUANT_MODE_BF16
    abi: str = FIA_ABI
    block_size: int = FIA_TILE
    splits: int = 1
    return_lse: bool = False
    batch_size: int = 1
    shared_prefix: bool = False
    precompute_scale_addr: bool = False
    ssmax: bool = False
    metadata_reuse: bool = False
    panel_mode: str = "shipping"
    cp_plan: ContextParallelPlan | tuple[ContextParallelPlan, ...] | None = None
    selection_lists: tuple[tuple[int, ...], ...] = ()
    fia_template: FiaPanelPlan | None = None
    kv_tensor_id: str = ""
    kv_version: int = 0
    query_block_size: int | None = None
    dst_type_max: float = 0.0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (
            self.query_tokens, self.num_heads, self.num_kv_heads, self.head_dim, self.v_head_dim,
            self.block_size, self.splits, self.batch_size,
        )):
            raise ModelingError("attention_positive_integer_geometry_required")
        if self.num_heads % self.num_kv_heads != 0:
            raise ModelingError("attention_head_group_divisibility_required")
        if (self.phase not in {"prefill", "decode"} or self.kv_layout not in KV_LAYOUTS
                or self.window_size < 0 or self.panel_mode not in {"shipping", "tile_solver_candidate"}):
            raise ModelingError("attention_phase_layout_window_or_panel_mode_invalid")
        modes = MINDIE_V3_MODES if self.abi == MINDIE_V3_ABI else SUPPORTED_QUANT_MODES
        if self.abi not in SUPPORTED_ABIS or type(self.quant_mode) is not int or self.quant_mode not in modes:
            raise unmodeled("attention_abi_or_quant_mode", f"Bind attention ABI {self.abi} and quantization {self.quant_mode}")
        if (type(self.kv_version) is not int or self.kv_version < 0 or not isinstance(self.kv_tensor_id, str)
                or (self.query_block_size is not None
                    and (type(self.query_block_size) is not int or self.query_block_size < 1))
                or not isfinite(self.dst_type_max) or self.dst_type_max < 0):
            raise ModelingError("attention_invalid_storage_contract")
        sequences = tuple(tuple(pair) for pair in self.sequences)
        object.__setattr__(self, "sequences", sequences)
        if sequences and (
            any(len(pair) != 2 or any(type(n) is not int or n < 0 for n in pair) for pair in sequences)
            or any(self.causal and q > kv for q, kv in sequences)
            or sum(q for q, _ in sequences) != self.query_tokens
        ):
            raise ModelingError("attention_sequence_geometry_invalid")
        if self.selected_tokens is not None and (type(self.selected_tokens) is not int or self.selected_tokens < 0):
            raise ModelingError("attention_selected_tokens_invalid")
        lists = tuple(tuple(rows) for rows in self.selection_lists)
        object.__setattr__(self, "selection_lists", lists)
        if lists and (self.num_kv_heads != 1 or len(lists) != self.query_tokens
                      or any(type(a) is not int or a < 0 for rows in lists for a in rows)):
            raise ModelingError("attention_selection_list_geometry_invalid")
        if not (sequences or lists or self.selected_tokens is not None):
            raise ModelingError("attention_selection_geometry_required")
        cp_plans = (self.cp_plan,) if isinstance(self.cp_plan, ContextParallelPlan) else tuple(self.cp_plan or ())
        if any(p.phase != self.phase or (p.case == COMPLETE_KV_SPARSE_PREFILL and self.return_lse)
               for p in cp_plans):
            raise ModelingError("attention_cp_consumer_contract_mismatch")
        object.__setattr__(self, "cp_plan", cp_plans or None)

    def row_counts(self) -> Iterable[int]:
        """Yield the cardinality of each query's visible selection."""
        if self.selection_lists:
            yield from map(len, self.selection_lists)
            return
        for q, kv in self.sequences or ((self.query_tokens, self.selected_tokens),):
            selected = kv if self.selected_tokens is None else self.selected_tokens
            for index in range(q):
                visible = kv - q + index + 1 if self.sequences and self.causal else kv
                yield min(visible, selected, self.window_size or visible)

    @property
    def useful_flops(self) -> int:
        return 2 * self.num_heads * sum(self.row_counts()) * (self.head_dim + self.v_head_dim)


def bsh_kv_access_plan(
    hkv_local: int, head_dim: int, elem_bytes: int, layout: str, heads_per_owner: int = 1,
    *, rows: int = 1, base_byte: int = 0, first_head: int = 0, block_size: int = FIA_TILE,
    tokens_per_head: int | None = None,
) -> AddressRunPlan:
    """Map one owner's adjacent head slice to physical K or V tensor runs."""
    if (layout not in KV_LAYOUTS or min(hkv_local, head_dim, elem_bytes, heads_per_owner, block_size) < 1
            or min(base_byte, rows, first_head) < 0 or first_head + heads_per_owner > hkv_local
            or (tokens_per_head is not None and tokens_per_head < rows)):
        raise ModelingError("attention_kv_address_geometry_invalid")
    width = head_dim * elem_bytes
    if layout in TOKEN_MAJOR_LAYOUTS:
        return AddressRunPlan((AddressRun(base_byte + first_head * width, rows,
                                          heads_per_owner * width, hkv_local * width, layout),))
    page_rows = block_size if layout == "PA_BLK_BHSD" else max(1, tokens_per_head or rows)
    return AddressRunPlan(tuple(
        AddressRun(base_byte + (start * hkv_local + head * page_rows) * width,
                   min(page_rows, rows - start), width, width, layout)
        for start in range(0, rows, page_rows)
        for head in range(first_head, first_head + heads_per_owner)
    ))


@dataclass(frozen=True, slots=True)
class GatherAccounting:
    row_bytes: int
    rows: int
    logical_bytes: int
    sector_bytes: int
    aligned_transfer_bytes: int
    l2_hit_bytes: int
    dram_sector_bytes: int


class _PaddedGatherRuns(AddressRunPlan):
    """Contiguous GM rows can share sectors while UB pads each destination row."""

    __slots__ = ()

    def aligned_copy_bytes(self) -> int:
        alignment = self.alignment_bytes
        return sum(run.rows * ceil(run.row_bytes / alignment) * alignment for run in self.runs)


def packed_selection_runs(selection: Iterable[int] | AddressRunPlan, row_bytes: int) -> AddressRunPlan:
    runs = selection if isinstance(selection, AddressRunPlan) else _selection_runs(selection, row_bytes)
    if runs.scale_bytes or any(
        run.row_bytes != row_bytes or run.proven_alias_group is not None for run in runs.runs
    ):
        raise ModelingError("attention_c8_selection_requires_physical_rows")
    return _PaddedGatherRuns(runs.runs, alignment_bytes=SECTOR_BYTES, sector_bytes=runs.sector_bytes)


def packed_gather_run_plans(runs: AddressRunPlan) -> Iterable[AddressRunPlan]:
    """Keep each gather independent while sharing sectors within a contiguous run."""
    for run in runs.runs:
        yield _PaddedGatherRuns((run,), alignment_bytes=runs.alignment_bytes, sector_bytes=runs.sector_bytes)


def _selection_runs(addresses: Iterable[int], row_bytes: int) -> AddressRunPlan:
    runs = []
    start = end = None
    for address in addresses:
        if type(address) is not int or address < 0:
            raise ModelingError("attention_gather_address_invalid")
        if address != end:
            if start is not None:
                runs.append(AddressRun(start, (end - start) // row_bytes, row_bytes, row_bytes))
            start = address
        end = address + row_bytes
    if start is not None:
        runs.append(AddressRun(start, (end - start) // row_bytes, row_bytes, row_bytes))
    return AddressRunPlan(tuple(runs), alignment_bytes=SECTOR_BYTES)


def c8_gather_accounting(
    selection_lists: Iterable[Iterable[int] | AddressRunPlan] | None = None, *,
    quant_mode: int = QUANT_MODE_INT8_C8, query_tokens: int = 1, selected_tokens: int = 0,
    memory_view: AdmissionMemoryView | None = None, l2_capacity_bytes: int = 0,
    tensor_id: str = "kv", die: int = 0, owner: int = 0,
) -> GatherAccounting:
    """Replay byte-address selections in query order through admission memory.

    Geometry-only queries have independent cold traffic. An explicit list
    can share resident sectors in a caller-supplied or local admission view.
    """
    if quant_mode not in C8_MODES or min(query_tokens, selected_tokens, l2_capacity_bytes) < 0:
        raise ModelingError("attention_c8_gather_geometry_invalid")
    row_bytes = TQ4_ROW_BYTES if quant_mode == QUANT_MODE_TQ4 else C8_ROW_BYTES
    stride = ceil(row_bytes / SECTOR_BYTES) * SECTOR_BYTES
    proven_addresses = selection_lists is not None
    if selection_lists is None:
        run = AddressRun(0, selected_tokens, row_bytes, stride)
        selection_lists = (AddressRunPlan((run,), alignment_bytes=SECTOR_BYTES) for _ in range(query_tokens))
    view = memory_view if proven_addresses else None
    if proven_addresses and view is None and l2_capacity_bytes:
        view = AdmissionMemoryView({die: l2_capacity_bytes})
    key = ReuseKey(die, tensor_id, 0)
    bus = HbmBusSpec(die, 1, 1, 0)
    rows = sectors = aligned = hits = dram = 0
    for selection in selection_lists:
        plan = packed_selection_runs(selection, row_bytes)
        count = sum(run.rows for run in plan.runs)
        rows += count
        # Independent gather runs in one query preserve repeat accesses before L2
        # fill; physically contiguous rows were already coalesced into one AddressRun.
        for run_plan in packed_gather_run_plans(plan):
            sectors += run_plan.dram_sector_and_service()[0]
            pattern = MemoryAccessPattern(tensor_id, 0, die, owner, "read", "mte2_aiv",
                                          run_plan.runs[0].rows * row_bytes, run_plan, reuse_key=key)
            ledger = bind_memory_access(
                pattern, view, hbm_bus=bus, copy_hit_bytes_per_ms=1, copy_miss_bytes_per_ms=1,
                l2_bytes_per_ms=1, pipe_row_id="mte2_aiv", l2_row_id="l2", commit=False,
            ).ledger
            aligned += ledger.aligned_transfer_bytes
            hits += ledger.l2_hit_bytes
            dram += ledger.dram_sector_bytes
        if view is not None:
            full_pattern = MemoryAccessPattern(tensor_id, 0, die, owner, "read", "mte2_aiv",
                                               count * row_bytes, plan, reuse_key=key)
            view.admit_access(full_pattern, plan.sector_coverage().measure_bytes())
    return GatherAccounting(row_bytes, rows, rows * row_bytes, sectors, aligned, hits, dram)


@dataclass(frozen=True, slots=True)
class FiaPanelPlan:
    """Physical BMM1 K panels, BMM2 output panels and independent slot depths."""

    k_panels: tuple[int, ...]
    v_panels: tuple[int, ...]
    q_l1_slots: int
    k_l1_slots: int
    v_l1_slots: int
    p_l1_slots: int
    l0a_slots: int
    l0b_slots: int
    l0c_slots: int


def panels(width: int, tile: int) -> tuple[int, ...]:
    return tuple(min(tile, width - start) for start in range(0, width, tile))


def shipping_fia_panel_plan(contract: AttentionContract, arch_id: str) -> FiaPanelPlan:
    """Replay the pinned FIA branch's panel split and buffer policies."""
    dk, dv = contract.head_dim, contract.v_head_dim
    if arch_id not in {ARCH22, ARCH35}:
        raise unmodeled("attention_fia_arch", f"Bind a FIA template for {arch_id}")
    if contract.fia_template is not None:
        plan = contract.fia_template
        if (sum(plan.k_panels) != dk or sum(plan.v_panels) != dv
                or min((*plan.k_panels, *plan.v_panels, plan.q_l1_slots, plan.k_l1_slots,
                        plan.v_l1_slots, plan.p_l1_slots, plan.l0a_slots, plan.l0b_slots,
                        plan.l0c_slots)) < 1):
            raise ModelingError("attention_fia_template_geometry_mismatch")
        return plan
    if arch_id == ARCH22:
        if max(dk, dv) > WIDE_PANEL:
            raise unmodeled("attention_arch22_wide_shipping_template", "Bind an explicit Arch22 FIA template")
        return FiaPanelPlan((dk,), (dv,), 1, 2, 2, 2, 2, 2, 2)
    accumulator_bytes = max(FIA_TILE, dv) * FIA_TILE * FP32_BYTES
    l0c_slots = L0C_QUAD_SLOTS if accumulator_bytes * L0C_QUAD_SLOTS <= ARCH35_L0C_BYTES else 2
    return FiaPanelPlan(
        panels(dk, FIA_TILE) if dk > WIDE_PANEL else (dk,),
        panels(dv, FIA_TILE) if dv > WIDE_PANEL else (dv,),
        1 if dk > WIDE_PANEL else 2, 2, 2, FIA_PROBABILITY_SLOTS, 2, 2, l0c_slots,
    )


def solve_wide_head_tiling(
    contract: AttentionContract, hardware: Geometry, *, s1: int = FIA_TILE, s2: int = FIA_TILE,
) -> TileSolution:
    """Screen full residency, full L0B, double-buffered panels and one-slot panels."""
    dk, dv = contract.head_dim, contract.v_head_dim
    width = max(dk, dv)
    candidates = []
    for algorithm, l1_cols, l0_cols, operand_slots in (
        ("resident_1pass", width, width, 1),
        ("online_reduce", WIDE_PANEL, width, 1),
        ("chunk_recurrence", WIDE_PANEL, WIDE_PANEL, 2),
        ("row_split_2pass", WIDE_PANEL, WIDE_PANEL, 1),
    ):
        buffers = (
            LiveBufferSpec("q", "l1", s1 * dk * BF16_BYTES, 0, PV_COMPLETE, "aic"),
            LiveBufferSpec("k", "l1", s2 * min(dk, l1_cols) * BF16_BYTES, 0, PROBABILITY_READY, "aic", slots=2),
            LiveBufferSpec("v", "l1", s2 * min(dv, l1_cols) * BF16_BYTES, 1, PV_COMPLETE, "aic", slots=2),
            LiveBufferSpec("p", "l1", s1 * s2 * BF16_BYTES, PROBABILITY_READY, PV_COMPLETE, "aic"),
            LiveBufferSpec("l0b", "l0b", s2 * min(width, l0_cols) * BF16_BYTES,
                           1, PV_COMPLETE, "aic", slots=operand_slots),
        )
        flops = 2 * s1 * s2 * (dk + dv)
        candidates.append(TileCandidate(
            algorithm, TileShape(s1, s2, min(width, l0_cols), dk, dv), operand_slots,
            (0, 0), 1, 1, buffers, logical_flops=flops, issued_flops=flops,
        ))
    return solve(TilePipelineSpec("attention", tuple(candidates)), None, hardware)


@dataclass(frozen=True, slots=True)
class AttentionPhase:
    name: str
    work: Work
    after: tuple[str, ...]
    slot: int | None = None
    memory: object = None


@dataclass(frozen=True, slots=True)
class _AttentionProgramBuilder:
    phases: tuple[AttentionPhase, ...]
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    aic_count: int
    aiv_count: int

    def __call__(self, context: RecipeContext) -> KernelProgram | ParallelProgram:
        split = next((index for index, p in enumerate(self.phases) if p.name == "split_read"), None)
        if split is None:
            return self._group(context, self.phases)
        return KernelProgram((
            Stage("split_producers", program=self._group(context, self.phases[:split])),
            Stage("split_consumers", after=("split_producers",),
                  program=self._group(context, self.phases[split:])),
        ))

    def _group(self, context: RecipeContext, phases: tuple[AttentionPhase, ...]) -> ParallelProgram:
        phase_names = {phase.name for phase in phases}
        count = len(self.owners)
        programs = []
        for index, (core, aiv_group) in enumerate(self.owners):
            pair = (core, aiv_group)
            placement = _placement(context, pair, self.aic_count, self.aiv_count)
            stages = []
            for phase in phases:
                share = Work(
                    {role: _owner_share(amount, index, count) for role, amount in phase.work.matrix.items()},
                    {role: _owner_share(amount, index, count) for role, amount in phase.work.resources.items()},
                )
                bound = replace(phase, work=share, after=tuple(dep for dep in phase.after if dep in phase_names))
                stages.append(_stage(context, bound, placement))
            programs.append(KernelProgram(tuple(stages)))
        return ParallelProgram(tuple(programs))


def _allocations(context, buffers, pair, aic_count):
    core, aiv_group = pair
    return tuple(
        context.allocate(pool, pool, amount, aiv_group if pool == "ub" else
                         unit_group(core, aic_count, context.pool_units.get(pool, aic_count)))
        for pool, amount in buffers if pool in context.pool_names
    )


def _placement(context, pair, aic_count, aiv_count):
    core, aiv_group = pair
    placement = dict(context.owned(core, aic_count, ("mte1", "mte2_aic", "mte2", "fixpipe", "cube")))
    for role in ("mte2_aiv", "vector", "mte3"):
        physical = context.physical_role(role)
        units = context.resource_units.get(physical, 1)
        if physical in context.resource_names and units > 1:
            placement.update(context.placement(role, tuple(sorted({
                unit for aiv in aiv_group for unit in unit_group(aiv, aiv_count, units)
            }))))
    return placement


def _stage(context, phase, placement, **kwargs):
    demand = context.bind(phase.work)
    return Stage(
        phase.name, demand, after=phase.after,
        placement={name: units for name, units in placement.items() if name in demand},
        metadata={} if phase.slot is None else {"operand_slot": phase.slot},
        keep_empty=True, **kwargs,
    )


def _owner_share(amount: float, index: int, count: int) -> float:
    if int(amount) != amount:
        return amount / count
    base, extra = divmod(int(amount), count)
    return base + (index < extra)


def owner_pairs(ctx: OperatorPlanningContext, work_items: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    owners = ctx.stream_owners
    aics = (owners.aic or owners.pool_aic) if owners else ()
    aivs = (owners.aiv or owners.pool_aiv) if owners else ()
    active_aic = frozenset((aics or range(ctx.effective_aic_cores))[:ctx.effective_aic_cores])
    active_aiv = frozenset((aivs or range(ctx.effective_aiv_cores))[:ctx.effective_aiv_cores])
    pairs = tuple(islice(
        ((aic, active) for aic, aiv_group in ctx.datapath.aic_aiv_pairs
         if aic in active_aic and (active := tuple(aiv for aiv in aiv_group if aiv in active_aiv))),
        work_items,
    ))
    if not pairs:
        raise unmodeled("attention_owner_pairs", "Bind reachable AIC/AIV owners for the attention stream")
    return pairs


def make_plan(
    op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext,
    phases: tuple[AttentionPhase, ...], sources: tuple[SourceRef, ...], *,
    metadata: dict | None = None, work_items: int | None = None, program=None,
) -> OperatorPlan:
    if program is None:
        if work_items is None:
            work_items = contract.query_tokens * contract.num_kv_heads * contract.splits
        program = _AttentionProgramBuilder(
            phases, owner_pairs(ctx, work_items), ctx.datapath.aic_count, ctx.datapath.aiv_count,
        )
    work = Work()
    projection = []
    for phase in phases:
        work += phase.work
        items = tuple(WorkItem("cube", "flop", amount, dt) for dt, amount in phase.work.matrix.items() if amount)
        items += tuple(WorkItem(role, "op" if role == "vector" else "bytes", amount)
                       for role, amount in phase.work.resources.items() if amount)
        projection.append(PipelinePhase(phase.name, phase.after, items, slot=phase.slot))
    implementation = op.recipe or contract.abi
    return OperatorPlan(
        Work(work.matrix, {role: amount for role, amount in work.resources.items() if amount}),
        program,
        {"implementation": implementation, "abi": contract.abi, "execution_class": "mixed_aic_aiv",
         "arch_id": str(ctx.compute_arch.arch_id), "useful_flops": contract.useful_flops,
         **(metadata or {})},
        pipeline_spec=FusedPipelineSpec("attention", implementation, tuple(projection), sources),
    )


def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan, /) -> FusedPipelineDescriptor:
    """Bind the shared pipeline projection to this source occurrence."""
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(
        ctx.occurrence, spec.family, spec.implementation, (op.op_id,), (),
        spec.phases, spec.sources, spec.repeats,
    )


def read_work(ctx: OperatorPlanningContext, bytes_: int, *, aiv: bool = False, hbm: int | None = None,
              l2: int = 0) -> Work:
    roles = ctx.geometry.resource_units or ctx.geometry.rates
    pipe = read_role(roles, aiv=0 if aiv else None) if roles else ("mte2_aiv" if aiv else "mte2_aic")
    return Work(resources={pipe: bytes_, "hbm": bytes_ if hbm is None else hbm, "l2": l2})


def output_phases(
    ctx: OperatorPlanningContext, contract: AttentionContract, dtype_bytes: int,
    previous: tuple[str, ...], partials: int, *, queries: int | None = None,
) -> tuple[AttentionPhase, ...]:
    """Publish completed rows and explicitly reduce split-KV partials."""
    rows = (contract.query_tokens if queries is None else queries) * contract.num_heads
    phases = []
    if partials > rows:
        payload = partials * (contract.v_head_dim + 2) * FP32_BYTES
        phases.extend((
            AttentionPhase("split_write", Work(resources={"mte3": payload, "hbm": payload}), previous),
            AttentionPhase("split_read", read_work(ctx, payload, aiv=True), ("split_write",)),
            AttentionPhase("split_merge", Work(resources={
                "vector": partials * (2 * contract.v_head_dim + SOFTMAX_OPS),
            }), ("split_read",)),
        ))
        previous = ("split_merge",)
    result = rows * (contract.v_head_dim * dtype_bytes + (FP32_BYTES if contract.return_lse else 0))
    phases.append(AttentionPhase("output", Work(resources={"mte3": result, "hbm": result}), previous))
    return tuple(phases)
