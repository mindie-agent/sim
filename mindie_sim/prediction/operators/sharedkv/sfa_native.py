"""Arch35 split-G 2-AIC/4-AIV pipelined sparse MLA attention (BF16 & C8).

[Role] Lower aclnnSparseFlashAttention and aclnnKvQuantSparseFlashAttention
  on npu_arch_3510 using the native split-G (G>64 -> 2 AIC + 4 AIV per query)
  3-slot 128-row pipelined GM/L2 KV ring.
[Invariants] Work declared in OperatorPlan.work matches lowered stages
  identically. Split-G halves query heads across adjacent AICs while 4 AIVs
  gather/dequantize 32 rows each per 128-row tile. Zero hardcoded setup constants
  or empirical service binders; supply rates come from the active datapath or L0 table.
[Timeouts & Locks] Finite O(tiles) pipeline over selected rows (<= topk).
[Side-Effect Policy] Pure planning and program construction.
Upstream-ABI: aclnnKvQuantSparseFlashAttention
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from math import ceil
from typing import Literal

from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...runtime.l2_residency import AdmissionMemoryView, bind_memory_access
from ...runtime.memory_physics import AddressRun, AddressRunPlan, HbmBusSpec, MemoryAccessPattern, ReuseKey
from ..attention.common import (
    BF16_BYTES,
    C8_ABI,
    C8_MODES,
    C8_V1_ABI,
    FP32_BYTES,
    INDEX_BYTES,
    PAGED_LAYOUTS,
    QUANT_MODE_BF16,
    QUANT_MODE_TQ4,
    SECTOR_BYTES,
    SFA_V1_ABI,
    SOFTMAX_OPS,
    AttentionContract,
    AttentionPhase,
    _stage,
    make_plan,
    owner_pairs,
    packed_gather_run_plans,
    packed_selection_runs,
    read_work,
    unmodeled,
    workspace_write_work,
)
from ..contracts import OperatorPlanningContext

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SFA_TILING_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp",
    "e87df78987dabb50064fc8361b51463dbd3490f7ba1c5d66a0b85c9721df0259",
)
SFA_KERNEL_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla_arch35.h",
    "651400b9a8aa23290fb5646c38aa7c9ebca85e352f484ce409b7fe426f52ec66",
)
C8_TILING_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp",
    "2c75acce70974165d5b087416d724bcd0d09909b2e8dd693370b5860de9dbc6e",
)
C8_KERNEL_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_kernel_mla_arch35.h",
    "89f70d3db0344bdd44bef92b3108007225a84f342e0995ed724654dbada278fa",
)
SOURCES: tuple[SourceRef, ...] = (
    SFA_TILING_SOURCE,
    SFA_KERNEL_SOURCE,
    C8_TILING_SOURCE,
    C8_KERNEL_SOURCE,
)

SFA_TILE_ROWS = 128
SFA_RING_SLOTS = 3
SFA_SPLIT_G_HEADS = 64
C8_PACKED_ROW_BYTES = 656
TQ4_PACKED_ROW_BYTES = 392
C8_DEQUANT_OPS_PER_ELEM = 11
SFA_OUTPUT_MERGE_OPS_PER_ELEM = 3
SFA_SSMAX_EXTRA_OPS = 2
FLOPS_PER_FMA = 2
SPLIT_G_AICS_PER_QUERY = 2
SPLIT_G_AIVS_PER_QUERY = 4
NON_SPLIT_G_AIVS_PER_QUERY = 2
NON_SPLIT_G_ROWS_PER_LOOP = 2
HALF_DIVISOR = 2


def _l2_capacity_bytes(ctx: OperatorPlanningContext) -> int:
    return int(ctx.geometry.buffers.get("l2", dict(ctx.datapath.buffers).get("l2", 0)))


def _packed_row_bytes(contract: AttentionContract) -> int:
    if contract.abi == SFA_V1_ABI:
        if contract.quant_mode != QUANT_MODE_BF16:
            raise unmodeled(
                "attention_sfa_v1_quant_mode_unsupported",
                f"{SFA_V1_ABI} requires non-quantized quant_mode={QUANT_MODE_BF16}, got {contract.quant_mode}",
            )
        return contract.head_dim * BF16_BYTES
    if contract.quant_mode == QUANT_MODE_BF16:
        return contract.head_dim * BF16_BYTES
    if contract.quant_mode == QUANT_MODE_TQ4:
        return TQ4_PACKED_ROW_BYTES
    if contract.quant_mode in C8_MODES:
        return C8_PACKED_ROW_BYTES
    raise unmodeled(
        "attention_sfa_quant_mode_unsupported",
        f"{contract.abi} does not support quant_mode={contract.quant_mode}",
    )


@dataclass(frozen=True, slots=True)
class _SfaSelectionRead:
    pattern: MemoryAccessPattern
    bound: object
    capacity: int
    hbm: str
    ctx: OperatorPlanningContext

    def start(self, engine, name, ready, *, owner=""):
        view = getattr(engine, "admission_memory_view", None)
        if view is None:
            views = getattr(engine, "_fia_admission", None)
            if views is None:
                views = engine._fia_admission = {}
            die = self.pattern.die
            key = (self.hbm, die)
            view = views.get(key)
            if view is None:
                if self.ctx.admission_memory_view is not None:
                    view = copy.deepcopy(self.ctx.admission_memory_view)
                else:
                    view = AdmissionMemoryView(
                        {die: self.capacity}, arch_id=str(self.ctx.compute_arch.arch_id),
                    )
                views[key] = view
        return engine.submit_bound_access(
            name, self.bound, after=(ready,), owner=owner, memory_state=view, pattern=self.pattern,
        )


def _partition_tile_rows(
    n_rows: int, aivs: tuple[int, ...], aics_per_group: int,
) -> tuple[tuple[int, int, int], ...]:
    """Partition selected tile rows across AIVs matching ``CalSparseCalSize``."""
    if n_rows <= 0 or not aivs:
        return ()
    if aics_per_group > 1 and len(aivs) >= SPLIT_G_AIVS_PER_QUERY:
        first_aic = ceil(n_rows / HALF_DIVISOR)
        second_aic = n_rows - first_aic
        v0_end = ceil(first_aic / HALF_DIVISOR)
        v2_end = first_aic + ceil(second_aic / HALF_DIVISOR)
        spans = (
            (aivs[0], 0, v0_end),
            (aivs[1], v0_end, first_aic),
            (aivs[2], first_aic, v2_end),
            (aivs[3], v2_end, n_rows),
        )
    elif len(aivs) >= NON_SPLIT_G_AIVS_PER_QUERY:
        s2_loops = ceil(ceil(n_rows / NON_SPLIT_G_AIVS_PER_QUERY) / NON_SPLIT_G_ROWS_PER_LOOP)
        split = min(s2_loops * NON_SPLIT_G_ROWS_PER_LOOP, n_rows)
        spans = ((aivs[0], 0, split), (aivs[1], split, n_rows))
    else:
        spans = ((aivs[0], 0, n_rows),)
    return tuple((aiv, lo, hi) for aiv, lo, hi in spans if hi > lo)


def _selection_aiv_steps(
    selection_slice: tuple[int, ...], contract: AttentionContract, is_quant: bool, row_bytes: int,
) -> tuple[tuple[str, int, AddressRunPlan], ...]:
    """Build per-AIV tensor read steps; BF16 separates ``keyGm`` and ``keyRopeGm``."""
    if not selection_slice:
        return ()
    if is_quant:
        plans = tuple(packed_gather_run_plans(packed_selection_runs(selection_slice, row_bytes)))
        return tuple((contract.kv_tensor_id, row_bytes, plan) for plan in plans)
    latent_b = contract.v_head_dim * BF16_BYTES
    rope_b = max(0, contract.head_dim - contract.v_head_dim) * BF16_BYTES
    lat_plans = tuple(packed_gather_run_plans(packed_selection_runs(selection_slice, latent_b)))
    if rope_b <= 0:
        return tuple((contract.kv_tensor_id, latent_b, plan) for plan in lat_plans)
    rope_addrs = tuple((addr // latent_b) * rope_b for addr in selection_slice)
    rope_plans = tuple(packed_gather_run_plans(packed_selection_runs(rope_addrs, rope_b)))
    rope_tid = f"{contract.kv_tensor_id}:rope"
    steps: list[tuple[str, int, AddressRunPlan]] = []
    for idx in range(max(len(lat_plans), len(rope_plans))):
        if idx < len(lat_plans):
            steps.append((contract.kv_tensor_id, latent_b, lat_plans[idx]))
        if idx < len(rope_plans):
            steps.append((rope_tid, rope_b, rope_plans[idx]))
    return tuple(steps)


def _lower_selection_gather(
    context, selection_slice: tuple[int, ...], grp_aivs: tuple[int, ...],
    aics_per_group: int, contract: AttentionContract, is_quant: bool, row_bytes: int,
    ctx: OperatorPlanningContext,
):
    """Lower per-AIV partitioned selection gathers onto Engine completion-time L2."""
    die = int(ctx.geometry.rank or 0)
    hbm_name = context.resource_names["hbm"]
    pipe_role = "mte2_aiv" if "mte2_aiv" in context.resource_names else "mte2"
    pipe_units = context.resource_units.get(pipe_role, 1)
    bus = HbmBusSpec(die, 1, 1, 0, row_r=hbm_name, row_w=hbm_name, ctr_r=hbm_name, ctr_w=hbm_name)
    l2_cap = _l2_capacity_bytes(ctx)
    aiv_programs = []
    for aiv_id, lo, hi in _partition_tile_rows(len(selection_slice), grp_aivs, aics_per_group):
        owner = unit_group(aiv_id, ctx.datapath.aiv_count, pipe_units)[0]
        steps = _selection_aiv_steps(selection_slice[lo:hi], contract, is_quant, row_bytes)
        read_stages = []
        prev_name: str | None = None
        for idx, (alloc_id, width, plan) in enumerate(steps):
            key = ReuseKey(die, alloc_id, contract.kv_version)
            pat = MemoryAccessPattern(
                key.backing_allocation, key.tensor_version, die, owner, "read", pipe_role,
                plan.runs[0].rows * width, plan, reuse_key=key,
            )
            bound = context.bind_access(pat, hbm_bus=bus)
            st_name = "read" if idx == 0 else f"read-{idx}"
            read_stages.append(Stage(
                st_name,
                after=(prev_name,) if prev_name is not None else (),
                program=_SfaSelectionRead(pat, bound, l2_cap, hbm_name, ctx),
            ))
            prev_name = st_name
        if read_stages:
            aiv_programs.append(KernelProgram(tuple(read_stages)))
    if len(aiv_programs) == 1:
        return aiv_programs[0]
    return ParallelProgram(tuple(aiv_programs))


@dataclass(frozen=True, slots=True)
class _Arch35SfaProgram:
    contract: AttentionContract
    ctx: OperatorPlanningContext
    dtype: str
    row_bytes: int
    is_quant: bool
    counts: tuple[int, ...]
    groups: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]
    ring_l2_resident: bool
    has_l2_rate: bool
    phases: tuple[AttentionPhase, ...] = ()
    tile_gathers: tuple[tuple[tuple[int, int, int, int], ...], ...] = ()

    def _tile_phases(
        self, rows: int, heads_per_aic: int, aics_per_group: int,
        *, gather_override: tuple[int, int, int, int] | None = None,
    ) -> tuple[AttentionPhase, ...]:
        c, ctx = self.contract, self.ctx
        gather_l2 = 0
        gather_miss = 0
        if gather_override is not None:
            gather_copy, gather_hbm, gather_l2, gather_miss = gather_override
        elif self.is_quant:
            stride = ceil(self.row_bytes / SECTOR_BYTES) * SECTOR_BYTES
            run = AddressRunPlan((AddressRun(0, rows, self.row_bytes, stride),))
            gather_copy = run.aligned_copy_bytes()
            gather_hbm = run.dram_sector_and_service()[0]
        else:
            latent_b = c.v_head_dim * BF16_BYTES
            rope_b = max(0, c.head_dim - c.v_head_dim) * BF16_BYTES
            lat_run = AddressRunPlan((AddressRun(0, rows, latent_b, ceil(latent_b / SECTOR_BYTES) * SECTOR_BYTES),))
            gather_copy = lat_run.aligned_copy_bytes()
            gather_hbm = lat_run.dram_sector_and_service()[0]
            if rope_b > 0:
                rope_run = AddressRunPlan((AddressRun(0, rows, rope_b, ceil(rope_b / SECTOR_BYTES) * SECTOR_BYTES),))
                gather_copy += rope_run.aligned_copy_bytes()
                gather_hbm += rope_run.dram_sector_and_service()[0]
        kv_bf16_bytes = rows * c.head_dim * BF16_BYTES
        out: list[AttentionPhase] = [
            AttentionPhase("sparse_index", read_work(ctx, rows * INDEX_BYTES, aiv=True), ()),
        ]
        gather_deps: tuple[str, ...] = ("sparse_index",)
        if c.kv_layout in PAGED_LAYOUTS:
            pages = ceil(rows / max(1, c.block_size))
            out.append(AttentionPhase("block_table", read_work(ctx, pages * INDEX_BYTES, aiv=True), ()))
            gather_deps += ("block_table",)
        if gather_override is not None and gather_miss > 0 and "mte2_aiv_dram" in ctx.geometry.resource_units:
            gather_hit = max(0, gather_copy - gather_miss)
            g_res: dict[str, int] = {"hbm": gather_hbm, "mte2_aiv_dram": gather_miss}
            if gather_hit > 0:
                g_res["mte2_aiv"] = gather_hit
            if gather_l2 and self.has_l2_rate:
                g_res["l2"] = gather_l2
            gather_work = Work(resources=g_res)
        else:
            gather_work = read_work(
                ctx, gather_copy, aiv=True, hbm=gather_hbm,
                l2=gather_l2 if self.has_l2_rate else 0,
            )
        out.append(AttentionPhase("packed_kv_gather", gather_work, gather_deps))
        prev = "packed_kv_gather"
        if self.is_quant:
            out.append(AttentionPhase(
                "kv_dequant",
                Work(resources={"vector": rows * c.v_head_dim * C8_DEQUANT_OPS_PER_ELEM}),
                (prev,),
            ))
            prev = "kv_dequant"
        pairs = heads_per_aic * rows * aics_per_group
        total_heads = heads_per_aic * aics_per_group
        prob_bytes = pairs * BF16_BYTES
        if aics_per_group > 1:
            out.append(AttentionPhase(
                "kv_publish",
                workspace_write_work(ctx, "mte3", kv_bf16_bytes, l2_resident=self.ring_l2_resident),
                (prev,),
            ))
            kv_read_bytes = kv_bf16_bytes * aics_per_group
            out.append(AttentionPhase(
                "kv_ring_read",
                read_work(
                    ctx, kv_read_bytes,
                    hbm=0 if self.ring_l2_resident else kv_read_bytes,
                    l2=(kv_read_bytes if (self.ring_l2_resident and self.has_l2_rate) else 0),
                ),
                ("kv_publish",),
            ))
            qk_dep = "kv_ring_read"
        else:
            # Non-split-G (1 AIC + local 2 AIVs): push KV directly UB->L1 without GM workspace.
            out.append(AttentionPhase("kv_to_l1", Work(resources={"mte3": kv_bf16_bytes}), (prev,)))
            qk_dep = "kv_to_l1"
        out.append(AttentionPhase(
            "qk",
            Work(
                {self.dtype: FLOPS_PER_FMA * pairs * c.head_dim},
                {"mte1": (total_heads + rows * aics_per_group) * c.head_dim * BF16_BYTES},
            ),
            (qk_dep,),
        ))
        out.append(AttentionPhase("score_fixpipe", Work(resources={"fixpipe": pairs * FP32_BYTES}), ("qk",)))
        sm_ops = SOFTMAX_OPS + (SFA_SSMAX_EXTRA_OPS if c.ssmax else 0)
        out.append(AttentionPhase("online_softmax", Work(resources={"vector": pairs * sm_ops}), ("score_fixpipe",)))
        out.append(AttentionPhase("probability_transfer", Work(resources={"mte3": prob_bytes}), ("online_softmax",)))
        out.append(AttentionPhase(
            "pv",
            Work(
                {self.dtype: FLOPS_PER_FMA * pairs * c.v_head_dim},
                {"mte1": prob_bytes + rows * aics_per_group * c.v_head_dim * BF16_BYTES},
            ),
            ("probability_transfer",),
        ))
        out.append(AttentionPhase(
            "pv_fixpipe",
            Work(resources={"fixpipe": total_heads * c.v_head_dim * FP32_BYTES}),
            ("pv",),
        ))
        out.append(AttentionPhase(
            "online_output",
            Work(resources={"vector": total_heads * c.v_head_dim * SFA_OUTPUT_MERGE_OPS_PER_ELEM}),
            ("pv_fixpipe",),
        ))
        return tuple(out)

    def _query_head_load(self) -> AttentionPhase:
        q_bytes = self.contract.num_heads * self.contract.head_dim * BF16_BYTES
        return AttentionPhase("query_load", read_work(self.ctx, q_bytes), ())

    def _query_finish(self) -> tuple[AttentionPhase, ...]:
        c = self.contract
        out_bytes = c.num_heads * (c.v_head_dim * BF16_BYTES + (FP32_BYTES if c.return_lse else 0))
        return (
            AttentionPhase("normalize", Work(resources={"vector": c.num_heads * c.v_head_dim}), ()),
            AttentionPhase("output", Work(resources={"mte3": out_bytes, "hbm": out_bytes}), ("normalize",)),
        )

    def __call__(self, context):
        group_places = []
        for grp in self.groups:
            aics = tuple(p[0] for p in grp)
            aivs = tuple(v for p in grp for v in p[1])
            place: dict[str, tuple[int, ...]] = {}
            for role in ("cube", "mte1", "mte2_aic", "fixpipe"):
                place.update(context.placement(role, aics))
            for role in ("vector", "mte2_aiv", "mte3"):
                place.update(context.placement(role, aivs))
            group_places.append(place)

        aics_per_group = len(self.groups[0])
        heads_per_aic = ceil(self.contract.num_heads / aics_per_group)
        q_load_phase = self._query_head_load()
        fin_phases = self._query_finish()

        def build_query(q_idx: int, grp_idx: int):
            place = group_places[grp_idx]
            rows = self.counts[q_idx]
            n_tiles = max(1, ceil(rows / SFA_TILE_ROWS)) if rows > 0 else 0
            q_st = _stage(context, q_load_phase, place)
            if n_tiles == 0:
                f_st = tuple(_stage(context, replace(p, after=(q_st.name,) if i == 0 else p.after), place)
                             for i, p in enumerate(fin_phases))
                return KernelProgram((q_st, *f_st))

            q_gathers = self.tile_gathers[q_idx] if q_idx < len(self.tile_gathers) else ()
            q_selection = (
                self.contract.selection_lists[q_idx]
                if (self.contract.selection_lists and self.contract.kv_tensor_id
                    and self.has_l2_rate and "hbm" in context.resource_names
                    and q_idx < len(self.contract.selection_lists))
                else ()
            )
            grp_aivs = tuple(v for p in self.groups[grp_idx] for v in p[1])

            def tile_fn(t_idx: int, total_rows=rows, pl=place, qg=q_gathers, qsel=q_selection, gaivs=grp_aivs):
                t_start = t_idx * SFA_TILE_ROWS
                t_rows = min(SFA_TILE_ROWS, total_rows - t_start)
                ov = qg[t_idx] if t_idx < len(qg) else None
                stages = []
                for ph in self._tile_phases(t_rows, heads_per_aic, aics_per_group, gather_override=ov):
                    if ph.name == "packed_kv_gather" and qsel:
                        st = Stage(
                            ph.name, after=ph.after,
                            program=_lower_selection_gather(
                                context, qsel[t_start : t_start + t_rows], gaivs,
                                aics_per_group, self.contract, self.is_quant, self.row_bytes, self.ctx,
                            ),
                        )
                    else:
                        st = _stage(context, ph, pl)
                    if ph.name in {"kv_ring_read", "kv_to_l1"}:
                        st = replace(st, previous=("pv",))
                    stages.append(st)
                return tuple(stages)

            loop = Stage("s2_ring", after=(q_st.name,),
                         program=PipelineProgram(n_tiles, min(n_tiles, SFA_RING_SLOTS), tile_fn))
            f0 = _stage(context, replace(fin_phases[0], after=(loop.name,)), place)
            f1 = _stage(context, replace(fin_phases[1], after=(f0.name,)), place)
            return KernelProgram((q_st, loop, f0, f1))

        n_groups = len(self.groups)
        return ParallelProgram(tuple(
            PipelineProgram(
                len(indices), 1,
                lambda i, g=g_idx, indices=indices: (Stage("query", program=build_query(indices[i], g)),),
            )
            for g_idx in range(n_groups) if (indices := range(g_idx, len(self.counts), n_groups))
        ))


def _preview_selection_gathers(
    counts: tuple[int, ...],
    groups: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...],
    aics_per_query: int,
    contract: AttentionContract,
    is_quant: bool,
    row_bytes: int,
    ctx: OperatorPlanningContext,
    view: AdmissionMemoryView,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Preview tile gather L2 hits/misses in wave-step lockstep matching Engine execution."""
    die = int(ctx.geometry.rank or 0)
    bus = HbmBusSpec(die, 1, 1, 0)
    n_groups = len(groups)
    per_q_tiles: list[list[list[int]]] = [
        [[0, 0, 0, 0] for _ in range(0, rows, SFA_TILE_ROWS)] for rows in counts
    ]
    for wave_start in range(0, len(counts), n_groups):
        wave = range(wave_start, min(len(counts), wave_start + n_groups))
        max_tiles = max((len(per_q_tiles[q_idx]) for q_idx in wave), default=0)
        for t_idx in range(max_tiles):
            t_start = t_idx * SFA_TILE_ROWS
            q_aiv_steps: list[tuple[int, tuple[tuple[tuple[str, int, AddressRunPlan], ...], ...]]] = []
            max_aivs = max_steps = 0
            for q_idx in wave:
                if t_idx >= len(per_q_tiles[q_idx]):
                    continue
                t_rows = min(SFA_TILE_ROWS, counts[q_idx] - t_start)
                sr = contract.selection_lists[q_idx][t_start : t_start + t_rows]
                grp_aivs = tuple(v for p in groups[q_idx % n_groups] for v in p[1])
                aiv_lists = tuple(
                    _selection_aiv_steps(sr[lo:hi], contract, is_quant, row_bytes)
                    for _, lo, hi in _partition_tile_rows(len(sr), grp_aivs, aics_per_query)
                )
                max_aivs = max(max_aivs, len(aiv_lists))
                max_steps = max(max_steps, max((len(s) for s in aiv_lists), default=0))
                q_aiv_steps.append((q_idx, aiv_lists))
            for step_idx in range(max_steps):
                for aiv_slot in range(max_aivs):
                    for q_idx, aiv_lists in q_aiv_steps:
                        if aiv_slot >= len(aiv_lists) or step_idx >= len(aiv_lists[aiv_slot]):
                            continue
                        alloc_id, width, plan = aiv_lists[aiv_slot][step_idx]
                        key = ReuseKey(die, alloc_id, contract.kv_version)
                        pat = MemoryAccessPattern(
                            key.backing_allocation, key.tensor_version, die, 0, "read", "mte2_aiv",
                            plan.runs[0].rows * width, plan, reuse_key=key,
                        )
                        led = bind_memory_access(
                            pat, view, hbm_bus=bus, copy_hit_bytes_per_ms=1, copy_miss_bytes_per_ms=1,
                            l2_bytes_per_ms=1, pipe_row_id="mte2_aiv", l2_row_id="l2", commit=True,
                        ).ledger
                        acc = per_q_tiles[q_idx][t_idx]
                        acc[0] += led.aligned_transfer_bytes
                        acc[1] += led.dram_sector_bytes
                        acc[2] += led.l2_hit_bytes
                        acc[3] += led.dram_miss_copy_bytes
    return tuple(tuple((a[0], a[1], a[2], a[3]) for a in q_list) for q_list in per_q_tiles)


def build_arch35_split_g_sfa(
    op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext, sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    """Build split-G 2-AIC/4-AIV 3-slot pipelined SFA plan (BF16 or C8) on npu_arch_3510."""
    if contract.num_kv_heads != 1:
        raise unmodeled(
            "attention_sfa_kv_heads_unsupported",
            f"{contract.abi} requires num_kv_heads == 1, got {contract.num_kv_heads}",
        )
    if contract.abi == SFA_V1_ABI and contract.quant_mode != QUANT_MODE_BF16:
        raise unmodeled(
            "attention_sfa_v1_quant_mode_unsupported",
            f"{SFA_V1_ABI} requires non-quantized quant_mode={QUANT_MODE_BF16}, got {contract.quant_mode}",
        )
    if contract.quant_mode == QUANT_MODE_TQ4:
        raise unmodeled("attention_arch35_tq4_unsupported", "TQ4 packed latent cache is supported on Arch22 only, not Arch35")
    if contract.abi in {C8_ABI, C8_V1_ABI} and contract.quant_mode not in C8_MODES:
        raise unmodeled(
            "attention_c8_quant_mode_unsupported",
            f"{contract.abi} requires a C8 quant_mode, got {contract.quant_mode}",
        )
    is_quant = contract.quant_mode in C8_MODES
    row_bytes = _packed_row_bytes(contract)
    counts = tuple(contract.row_counts())
    desired_aics = SPLIT_G_AICS_PER_QUERY if contract.num_heads > SFA_SPLIT_G_HEADS else 1
    raw_owners = owner_pairs(ctx, max(desired_aics, len(counts) * desired_aics))
    aics_per_query = max(1, min(desired_aics, len(raw_owners)))
    heads_per_aic = ceil(contract.num_heads / aics_per_query)
    n_groups = max(1, len(raw_owners) // aics_per_query)
    groups = tuple(
        raw_owners[g * aics_per_query : (g + 1) * aics_per_query]
        for g in range(n_groups)
    )
    ring_bytes = n_groups * SFA_RING_SLOTS * SFA_TILE_ROWS * contract.head_dim * BF16_BYTES
    l2_cap = _l2_capacity_bytes(ctx)
    ring_l2_resident = aics_per_query > 1 and l2_cap > 0 and ring_bytes <= l2_cap
    has_l2 = "l2" in ctx.geometry.rates
    prog = _Arch35SfaProgram(
        contract, ctx, op.dtype or "bf16", row_bytes, is_quant, counts, groups, ring_l2_resident, has_l2,
    )
    bucket: dict[str, AttentionPhase] = {}

    def _acc(phase: AttentionPhase, repeats: int = 1) -> None:
        w = phase.work.scaled(repeats) if repeats != 1 else phase.work
        old = bucket.get(phase.name)
        bucket[phase.name] = replace(phase, work=old.work + w if old else w)

    q_load = prog._query_head_load()
    fin = prog._query_finish()
    die = int(ctx.geometry.rank or 0)
    # Planning must never mutate the live AdmissionMemoryView; preview on a private
    # scratch snapshot while Engine delivery settles fills at transfer completion.
    sel_l2: AdmissionMemoryView | None = None
    if contract.selection_lists and contract.kv_tensor_id and has_l2:
        if ctx.admission_memory_view is not None:
            sel_l2 = copy.deepcopy(ctx.admission_memory_view)
        elif l2_cap > 0:
            sel_l2 = AdmissionMemoryView({die: l2_cap}, arch_id=str(ctx.compute_arch.arch_id))
    tile_gathers: tuple[tuple[tuple[int, int, int, int], ...], ...] = ()
    if sel_l2 is not None and contract.selection_lists:
        tile_gathers = _preview_selection_gathers(
            counts, groups, aics_per_query, contract, is_quant, row_bytes, ctx, sel_l2,
        )
    for q_idx, rows in enumerate(counts):
        _acc(q_load)
        if tile_gathers and q_idx < len(tile_gathers):
            for t_idx, ov in enumerate(tile_gathers[q_idx]):
                t_rows = min(SFA_TILE_ROWS, rows - t_idx * SFA_TILE_ROWS)
                for ph in prog._tile_phases(t_rows, heads_per_aic, aics_per_query, gather_override=ov):
                    _acc(ph, 1)
        else:
            full, rem = divmod(rows, SFA_TILE_ROWS)
            if full:
                for ph in prog._tile_phases(SFA_TILE_ROWS, heads_per_aic, aics_per_query):
                    _acc(ph, full)
            if rem:
                for ph in prog._tile_phases(rem, heads_per_aic, aics_per_query):
                    _acc(ph, 1)
        for ph in fin:
            _acc(ph)

    phases = tuple(bucket.values())
    total_rows = sum(counts)
    cold_hbm = bucket["packed_kv_gather"].work.resources.get("hbm", 0) if "packed_kv_gather" in bucket else 0
    l2_shared = int(bucket["kv_ring_read"].work.resources.get("mte2_aic", 0)) if (
        ring_l2_resident and "kv_ring_read" in bucket
    ) else 0
    leaf_sources = (C8_KERNEL_SOURCE, C8_TILING_SOURCE) if is_quant else (SFA_KERNEL_SOURCE, SFA_TILING_SOURCE)
    plan_sources = leaf_sources + tuple(src for src in sources if src not in leaf_sources)
    metadata = {
        "source_program": leaf_sources[0].to_dict(),
        "kv_logical_bytes": total_rows * row_bytes,
        "kv_sector_bytes": cold_hbm,
        "kv_dram_sector_bytes": cold_hbm,
        "l2_shared_kv_bytes": l2_shared,
        "kv_row_bytes": row_bytes,
        "kv_gather_rows": total_rows,
        "split_g_aics_per_query": aics_per_query,
        "ring_l2_resident": ring_l2_resident,
    }
    return make_plan(
        op, contract, ctx, phases, plan_sources, metadata=metadata,
        program=replace(prog, phases=phases, tile_gathers=tile_gathers),
    )
