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

from dataclasses import dataclass, replace
from math import ceil

from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...runtime.memory_physics import AddressRun, AddressRunPlan
from ..attention.common import (
    BF16_BYTES,
    C8_MODES,
    FP32_BYTES,
    INDEX_BYTES,
    PAGED_LAYOUTS,
    QUANT_MODE_BF16,
    QUANT_MODE_TQ4,
    SECTOR_BYTES,
    SOFTMAX_OPS,
    AttentionContract,
    AttentionPhase,
    _stage,
    make_plan,
    owner_pairs,
    read_work,
    workspace_write_work,
)
from ..contracts import OperatorPlanningContext

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


def _l2_capacity_bytes(ctx: OperatorPlanningContext) -> int:
    return int(ctx.geometry.buffers.get("l2", dict(ctx.datapath.buffers).get("l2", 0)))


def _packed_row_bytes(contract: AttentionContract) -> int:
    if contract.quant_mode == QUANT_MODE_BF16:
        return contract.head_dim * BF16_BYTES
    if contract.quant_mode == QUANT_MODE_TQ4:
        return TQ4_PACKED_ROW_BYTES
    return C8_PACKED_ROW_BYTES


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

    def _tile_phases(self, rows: int, heads_per_aic: int, aics_per_group: int) -> tuple[AttentionPhase, ...]:
        c, ctx = self.contract, self.ctx
        if self.is_quant:
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
        out.append(AttentionPhase("packed_kv_gather", read_work(ctx, gather_copy, aiv=True, hbm=gather_hbm), gather_deps))
        prev = "packed_kv_gather"
        if self.is_quant:
            out.append(AttentionPhase(
                "kv_dequant",
                Work(resources={"vector": rows * c.v_head_dim * C8_DEQUANT_OPS_PER_ELEM}),
                (prev,),
            ))
            prev = "kv_dequant"
        out.append(AttentionPhase(
            "kv_publish",
            workspace_write_work(ctx, "mte3", kv_bf16_bytes, l2_resident=self.ring_l2_resident),
            (prev,),
        ))
        pairs = heads_per_aic * rows * aics_per_group
        total_heads = heads_per_aic * aics_per_group
        prob_bytes = pairs * BF16_BYTES
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
        out.append(AttentionPhase(
            "qk",
            Work(
                {self.dtype: FLOPS_PER_FMA * pairs * c.head_dim},
                {"mte1": (total_heads + rows * aics_per_group) * c.head_dim * BF16_BYTES},
            ),
            ("kv_ring_read",),
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

            def tile_fn(t_idx: int, total_rows=rows, pl=place):
                t_rows = min(SFA_TILE_ROWS, total_rows - t_idx * SFA_TILE_ROWS)
                stages = []
                for ph in self._tile_phases(t_rows, heads_per_aic, aics_per_group):
                    st = _stage(context, ph, pl)
                    if ph.name == "kv_ring_read":
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


def build_arch35_split_g_sfa(
    op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext, sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    """Build split-G 2-AIC/4-AIV 3-slot pipelined SFA plan (BF16 or C8) on npu_arch_3510."""
    is_quant = contract.quant_mode in C8_MODES
    row_bytes = _packed_row_bytes(contract)
    counts = tuple(contract.row_counts())
    aics_per_query = SPLIT_G_AICS_PER_QUERY if contract.num_heads > SFA_SPLIT_G_HEADS else 1
    heads_per_aic = ceil(contract.num_heads / aics_per_query)
    raw_owners = owner_pairs(ctx, max(aics_per_query, len(counts) * aics_per_query))
    n_groups = max(1, len(raw_owners) // aics_per_query)
    groups = tuple(
        raw_owners[g * aics_per_query : (g + 1) * aics_per_query]
        for g in range(n_groups)
    )
    ring_bytes = n_groups * SFA_RING_SLOTS * SFA_TILE_ROWS * contract.head_dim * BF16_BYTES
    l2_cap = _l2_capacity_bytes(ctx)
    ring_l2_resident = l2_cap > 0 and ring_bytes <= l2_cap
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
    for rows in counts:
        _acc(q_load)
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
    metadata = {
        "source_program": sources[0].to_dict(),
        "kv_logical_bytes": total_rows * row_bytes,
        "kv_sector_bytes": cold_hbm,
        "kv_dram_sector_bytes": cold_hbm,
        "l2_shared_kv_bytes": l2_shared,
        "kv_row_bytes": row_bytes,
        "kv_gather_rows": total_rows,
        "split_g_aics_per_query": aics_per_query,
        "ring_l2_resident": ring_l2_resident,
    }
    return make_plan(op, contract, ctx, phases, sources, metadata=metadata, program=replace(prog, phases=phases))
