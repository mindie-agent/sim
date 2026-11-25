"""Arch35 linear-attention strategies over explicit Cube/AIV routes.

[Role] Lower persistent GDN and FlashKDA, staged Chunk32, and decode ABIs.
[Invariants] A persistent head owns one AIC and two reachable AIVs. State
  lives in FP32 UB tiles while operand snapshots use the declared L1 route.
[Timeouts & Locks] Finite pipelines issue only a bounded number of chunks.
[Side-Effect Policy] Pure planning. No device context or source-file reads.
Upstream-ABI: aclnnChunkGatedDeltaRuleArch35
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from heapq import heapify, heapreplace
from itertools import chain
from typing import Literal

from mindie_sim.schema.common import ModelingError, align_up
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.operators.linear_attention.gdn_chunk32 import PR22147_SOURCE
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext, Stage
from ...runtime.ring_buffer import RingBuffer, lower_generation
from ..contracts import OperatorPlanningContext
from ._gdn_native import (
    ARCH35_GDN_AIC_MTE2_QK_PANELS,
    ARCH35_GDN_AIV_MTE2_QKV_PANELS,
    ARCH35_GDN_COMPACT_THRESHOLD,
    RECURRENT_GDN_ABI,
    arch35_gdn_chunk_sync_ms,
    arch35_gdn_extra_sync_stage,
    arch35_gdn_vf_vector_ops,
    build_native_recurrent_gdn,
    compact_lane_pipeline,
)
from ..kda.triton_kda import TRITON_KDA_CHUNK_ABIS, TRITON_KDA_RECURRENT_ABIS
TRITON_KDA_ARCH22_ABIS = TRITON_KDA_CHUNK_ABIS | TRITON_KDA_RECURRENT_ABIS
from .common import (
    CANN_GDN_ABI, CHUNK32_KERNELS, CHUNK32_LEAVES, DIAG_BLOCK_WIDTH, FLASH_KDA_TEMPLATE_BY_ABI,
    FLA_GDN_ABIS, FP32_BYTES, GDN_CHUNK64_TOKENS, KDA_MAX_AIC_CORES, KDA_WORKSPACE_SLOTS,
    CausalConvContract, LinearAttentionContract, Phase, RecoverSsmCommitPlan,
    analyze_state_residency, bind_phase, build_causal_conv1d_plan, build_recover_ssm_plan,
    ceil_div, compute_wy_chunk_work, element_bytes,
    flash_kda_workspace_groups, make_plan, missing, owner_units,
    transfer, validate_contract, verify_native_gdn_abi,
)
from .gdn_domain import GDN_CHUNK32_HEAD_DIM_V, GDN_HEAD_DIM_K, can_run_gdn_chunk32

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
OPS_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
DSL_REVISION = "5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer", revision=OPS_REVISION,
        relative_path="attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_arch35.h",
        source_sha256="914151d3e01c167dc7f62f51c049fff9850b681039c4744e9cb68bf4067360da",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision=OPS_REVISION,
        relative_path="attention/flash_kda/op_dsl/arch35/flash_kda.py",
        source_sha256="598c722eaa5df24b37c19dad6e969f06221389f1f0558f0efbe56db2bc73999a",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision=OPS_REVISION,
        relative_path="mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_update.h",
        source_sha256="812584ad8fd5b8dbb76fa53fb15648a15d2942fec7c6d23a1fa866355fea13c3",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels", revision=DSL_REVISION,
        relative_path="samples/attention/flash_kda/flash_kda_metadata.py",
        source_sha256="5033e923d9d63602708918691f75fcf8f210a0782e850d7822090362f4242377",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels", revision=DSL_REVISION,
        relative_path="samples/attention/flash_kda/flash_kda_persistent.py",
        source_sha256="16f01982db4fc4f053f238e353637ac314cdc4d589ed948a32ee391297d50b08",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels", revision=DSL_REVISION,
        relative_path="samples/attention/flash_kda/flash_kda_staged.py",
        source_sha256="41957c35307c32dc5326278a01e176325f264bd63d414c059166f12bfd47ac17",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision=OPS_REVISION,
        relative_path="attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_basic_block.h",
        source_sha256="b8c938e623b4207c5f95b39fd4b8c8422b201b32c2b473918c42a11069c5f582",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels", revision=DSL_REVISION,
        relative_path="samples/attention/fused_recurrent_kda_snapshot/fused_recurrent_kda_snapshot.py",
        source_sha256="e0e78d2a77126389605ea73972eb8c4da5a3727b877c95cbf019ea644a528352",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels", revision=DSL_REVISION,
        relative_path="samples/attention/commit_recurrent_kda/commit_recurrent_kda.py",
        source_sha256="81e2ff105aba687f5fb6fc8eadae0149365bb85cf8c01e2889c66a6a906973c4",
    ),
)
ABI = "aclnnChunkGatedDeltaRuleArch35"
FRACTAL_DIM = 16
CHUNK_RING_DEPTH = 2
CHUNK_WORK_CACHE_SIZE = 128
TRIANGULAR_FMA_DIVISOR = 3
GDN_INVERSE_OPERAND_PANELS = 4
WY_PREPROCESSED_KEY_PANELS = 5
STAGED_KEY_WORKSPACES = 3
LOW_PRECISION_BYTES = 2
COMMIT_VALUE_TILE = 128
STAGED_STATE_WAVE_START = 3
KDA_PACKED_INVERSE_PRODUCTS = 5
GDN_CUBE_DIM = 128
GDN_STREAM_VECTOR_COST = 256
GDN_STREAM_STATE_OPS_PER_ELEMENT = 7
GDN_SCORE_PRODUCTS = 5
GDN_STATE_PRODUCTS = 3
GDN_CUBE_FLOPS_PER_CHUNK = 2 * (
    GDN_SCORE_PRODUCTS * GDN_CHUNK64_TOKENS**2 * GDN_CUBE_DIM
    + GDN_STATE_PRODUCTS * GDN_CHUNK64_TOKENS * GDN_CUBE_DIM**2
    + 2 * DIAG_BLOCK_WIDTH**3
)
GDN_SOURCES = (SOURCES[0], SOURCES[-3])
CHUNK32_SOURCES = (SourceRef(**PR22147_SOURCE),)


def _pairs(ctx: OperatorPlanningContext, c: LinearAttentionContract) -> tuple[tuple[int, tuple[int, ...]], ...]:
    path = ctx.datapath
    aics, aivs = owner_units(ctx)
    if not path.has_l0c_to_ub or not path.has_ub_to_l1:
        raise missing("linear_attention_direct_routes_required", "Arch35 prefill requires L0C to UB and UB to L1")
    pairs = tuple(
        (aic, tuple(aiv for aiv in paired if aiv in aivs)[:2])
        for aic, paired in path.aic_aiv_pairs if aic in aics
        and sum(aiv in aivs for aiv in paired) >= 2
    )
    if not pairs:
        raise missing("linear_attention_no_paired_owners", "Arch35 prefill requires paired AIC and AIV owners")
    fixpipe = path.route("l0c", "ub", "fp32").path
    pack = path.route("ub", "l1", c.dtype).path
    tile_elements = c.bv * c.bk
    fixpipe_tile = align_up(tile_elements * FP32_BYTES, fixpipe.alignment_bytes)
    pack_tile = align_up(tile_elements * element_bytes(c.dtype), pack.alignment_bytes)
    for aic, aiv_pair in pairs:
        for aiv in aiv_pair:
            owner = {"aic": aic, "aiv": aiv}
            path.route("l0c", "ub", "fp32", owner_group=owner, tile=fixpipe_tile)
            path.route("ub", "l1", c.dtype, owner_group=owner, tile=pack_tile)
    path.route("l0c", "gm", c.dtype)
    return pairs


def _chunk_phases(c: LinearAttentionContract, ctx: OperatorPlanningContext, rows: int, *,
                  kda: bool, snapshot: bool) -> tuple[Phase, ...]:
    w = compute_wy_chunk_work(rows, c.dk, c.dv, c.bt)
    elem = element_bytes(c.dtype)
    key, value, score = rows * c.dk, rows * c.dv, rows * rows
    state = c.dk * c.dv
    gate_width = c.dk if kda else 1
    gate = rows * gate_width
    state_elem = element_bytes(c.state_dtype)
    inputs = (2 * key + value) * elem + gate * element_bytes(c.gate_dtype) + rows * element_bytes(c.beta_dtype)
    # The CANN WY kernel zero-pads Q/K/V to its fixed 64x128 operands.
    gdn_wy = not kda and c.bt == GDN_CHUNK64_TOKENS
    issued = compute_wy_chunk_work(c.bt, GDN_CUBE_DIM, GDN_CUBE_DIM, c.bt) if gdn_wy else w
    cube_dtype = "bf16" if gdn_wy else c.dtype
    cube_key = c.bt * GDN_CUBE_DIM if gdn_wy else key
    cube_value = c.bt * GDN_CUBE_DIM if gdn_wy else value
    cube_score = c.bt**2 if gdn_wy else score
    cube_state = GDN_CUBE_DIM**2 if gdn_wy else state
    inverse = issued.offdiag_inv_flops
    inverse_operand_bytes = GDN_INVERSE_OPERAND_PANELS * DIAG_BLOCK_WIDTH**2 * LOW_PRECISION_BYTES if inverse else 0
    diagonal = w.diag_inv_ops
    if kda:
        blocks = c.bt // FRACTAL_DIM
        diagonal = blocks * FRACTAL_DIM * (FRACTAL_DIM - 1) * (FRACTAL_DIM - 2) // TRIANGULAR_FMA_DIVISOR
        inverse = 2 * KDA_PACKED_INVERSE_PRODUCTS * c.bt * c.bt**2
        inverse_operand_bytes = 2 * KDA_PACKED_INVERSE_PRODUCTS * c.bt**2 * LOW_PRECISION_BYTES
    matrix_saves = 2 * rows * c.bt * elem if c.save_matrices else 0
    state_save = state * element_bytes(c.stored_chunk_state_dtype) if c.save_intermediate_state else 0

    inverse_dtype = "fp16" if kda else cube_dtype
    initialize = transfer(ctx, read=state * state_elem * c.use_initial_state) + Work(resources={
        "vector": state if c.use_initial_state else 2 * state, "mte3": state * elem,
    })
    vf_gate, vf_inv, vf_cast, vf_state = (
        arch35_gdn_vf_vector_ops(rows, c.dk, c.dv) if gdn_wy else (0, 0, 0, 0)
    )
    gate_vec = vf_gate if gdn_wy else (w.gate_ops + (2 * rows * (gate_width - 1) if kda else 0))
    inv_vec = vf_inv if gdn_wy else (w.mask_ops + diagonal)
    load_work = (
        transfer(ctx, read=inputs) + Work(resources={
            "mte2_aic": ARCH35_GDN_AIC_MTE2_QK_PANELS * key * elem,
            "mte2_aiv": ARCH35_GDN_AIV_MTE2_QKV_PANELS * key * elem,
        })
        if gdn_wy else transfer(ctx, read=inputs)
    )
    return (
        Phase("state_begin", initialize),
        Phase("load_tokens", load_work),
        Phase("KKT_QKT", Work({cube_dtype: issued.kkt_flops + issued.qkt_flops},
                              {"mte1": 2 * cube_key * elem, "fixpipe": 2 * cube_score * FP32_BYTES}), ("load_tokens",)),
        Phase("gate_operands", Work(resources={
            "vector": gate_vec,
            "mte3": (WY_PREPROCESSED_KEY_PANELS * key + value) * elem,
        }), ("load_tokens",)),
        Phase("mask_and_diagonal_inverse", Work(resources={
            "vector": inv_vec, "mte3": 2 * score * elem,
        }), ("KKT_QKT", "gate_operands")),
        Phase("offdiagonal_inverse", Work({inverse_dtype: inverse}, {
            "mte1": inverse_operand_bytes, "fixpipe": inverse_operand_bytes // 2,
        }), ("mask_and_diagonal_inverse",)),
        Phase("Kcd_U", Work({cube_dtype: issued.kcd_flops + issued.u_flops}, {
            "mte1": (cube_score + cube_key + cube_value) * elem, "fixpipe": (cube_key + cube_value) * elem,
        }), ("offdiagonal_inverse",)),
        Phase("save_matrices", Work(resources={"hbm": matrix_saves, "fixpipe": matrix_saves}),
              ("offdiagonal_inverse",)),
        Phase("save_entry_state", transfer(ctx, write=state_save), ("state_begin",)),
        Phase("R_O_inter", Work({cube_dtype: issued.r_flops + issued.o_inter_flops}, {
            "mte1": (2 * cube_key + cube_state) * elem, "fixpipe": cube_value * FP32_BYTES,
        }), ("Kcd_U", "save_entry_state")),
        Phase("R32_to_R16_UB_to_L1", Work(resources={"vector": vf_cast if gdn_wy else w.cast_ops, "mte3": value * elem}),
              ("R_O_inter",)),
        Phase("delta_L0C_to_UB", Work({cube_dtype: issued.delta_flops}, {
            "mte1": (cube_value + cube_key) * elem, "fixpipe": cube_state * FP32_BYTES,
        }), ("R32_to_R16_UB_to_L1",)),
        Phase("O_L0C_to_GM", Work({cube_dtype: issued.o_intra_flops}, {
            "mte1": (cube_score + cube_value) * elem, "fixpipe": value * elem, "hbm": value * elem,
        }), ("delta_L0C_to_UB",)),
        Phase("S_next", Work(resources={
            "vector": vf_state if gdn_wy else (w.state_update_ops + state * (snapshot or kda)),
            "mte3": state * elem * snapshot,
        }), ("delta_L0C_to_UB",)),
        Phase("state_end", transfer(ctx, write=state * state_elem * c.output_final_state),
              ("S_next", "O_L0C_to_GM", "save_matrices")),
    )


def _chunk_counts(c: LinearAttentionContract) -> Counter[tuple[int, bool]]:
    counts: Counter[tuple[int, bool]] = Counter()
    for length in c.lengths:
        full, tail = divmod(length, c.bt)
        counts[c.bt, True] += max(0, full - (not tail)) * c.hv
        counts[tail or c.bt, False] += c.hv
    return counts


def _lane_spans(c: LinearAttentionContract, cores: int) -> tuple[list[tuple[int, int, int]], ...]:
    lanes: tuple[list[tuple[int, int, int]], ...] = tuple([] for _ in range(cores))
    heap = [(0, i) for i in range(cores)]
    heapify(heap)
    for length in sorted(c.lengths, key=lambda length: -ceil_div(length, c.bt)):
        chunks = ceil_div(length, c.bt)
        for _ in range(c.hv):
            load, lane = heap[0]
            spans = lanes[lane]
            if spans and spans[-1][1] == length:
                _, _, previous_count = spans[-1]
                spans[-1] = load + chunks, length, previous_count + 1
            else:
                spans.append((load + chunks, length, 1))
            heapreplace(heap, (load + chunks, lane))
    return lanes


def _build_persistent(c: LinearAttentionContract, ctx: OperatorPlanningContext,
                      pairs: tuple[tuple[int, tuple[int, ...]], ...]) -> OperatorPlan:
    kda = c.operation == "kda_prefill"
    state_ring = CHUNK_RING_DEPTH * c.dv * c.dk * LOW_PRECISION_BYTES if kda else 0
    report = analyze_state_residency(c, ctx, persistent=True, live_copies=2, l1_state_ring_bytes=state_ring)

    @lru_cache(maxsize=CHUNK_WORK_CACHE_SIZE)
    def chunk(rows: int, snapshot: bool) -> tuple[Phase, ...]:
        return _chunk_phases(c, ctx, rows, kda=kda, snapshot=snapshot)

    counts = _chunk_counts(c)
    phase_work: dict[str, Work] = {}
    for (rows, snapshot), count in counts.items():
        if not count:
            continue
        for phase in chunk(rows, snapshot):
            if phase.name not in {"state_begin", "state_end"}:
                phase_work[phase.name] = phase_work.get(phase.name, Work()) + phase.work.scaled(count)
    owners = len(c.lengths) * c.hv
    totals = tuple(replace(p, work=phase_work.get(p.name, p.work.scaled(owners))) for p in chunk(c.bt, False))
    lanes = _lane_spans(c, min(len(pairs), owners))

    def lower(rc: RecipeContext) -> ParallelProgram:
        def lane_program(
            aic: int, aivs: tuple[int, ...], spans: list[tuple[int, int, int]],
        ) -> PipelineProgram | KernelProgram:
            ends = tuple(end for end, _, _ in spans)
            ring = RingBuffer(allocation="prepared_chunk", depth=CHUNK_RING_DEPTH)
            events = lower_generation(ring, 0, "prepare", ("last_operand_use",),
                                      producer_extra_after=("Kcd_U",))
            credit, write, publish, read, last = ring.to_stages()

            def tile(index: int) -> tuple[Stage, ...]:
                position = bisect_right(ends, index)
                end, length, repeat = spans[position]
                chunks = ceil_div(length, c.bt)
                offset = (index - (end - repeat * chunks)) % chunks
                first, final = offset == 0, offset + 1 == chunks
                rows = min(c.bt, length - offset * c.bt)
                stages: list[Stage] = []
                for phase in chunk(rows, not final):
                    if phase.name == "state_begin":
                        if not first:
                            phase = replace(phase, work=Work())
                    elif phase.name == "state_end":
                        if not final:
                            phase = replace(phase, work=Work())
                    elif phase.name == "R_O_inter":
                        stages.extend((credit, write, publish))
                        phase = replace(phase, after=(events.publish, "save_entry_state"))
                    stage = bind_phase(phase, rc, ctx, (aic,), aivs)
                    if stage.name == "R_O_inter":
                        stage = replace(stage, previous=("S_next",))
                    elif stage.name == "save_entry_state":
                        stage = replace(stage, previous=("S_next",))
                    elif stage.name == "state_begin" and first:
                        stage = replace(stage, previous=("state_end",))
                    stages.append(stage)
                chunk_sync_ms = arch35_gdn_chunk_sync_ms(rc) if (not kda and c.bt == GDN_CHUNK64_TOKENS) else 0.0
                if chunk_sync_ms > 0.0:
                    stages.append(arch35_gdn_extra_sync_stage("chunk_crosscore_sync", chunk_sync_ms, after=("S_next",)))
                    for s_i, st in enumerate(stages):
                        if st.name in {"R_O_inter", "save_entry_state"}:
                            stages[s_i] = replace(st, previous=("chunk_crosscore_sync",))
                stages.extend((replace(read, after=("state_end",)), last))
                return tuple(stages)

            # Compact only inside single sequences so inter-sequence state_begin/state_end are never dropped.
            if len(spans) == 1 and spans[0][2] == 1:
                pipe = compact_lane_pipeline(rc, ends[-1], CHUNK_RING_DEPTH, tile)
            else:
                seq_stages: list[Stage] = []
                cursor = 0
                for _, length, repeat in spans:
                    s_chunks = ceil_div(length, c.bt)
                    for r_i in range(repeat):
                        base = cursor
                        if s_chunks > ARCH35_GDN_COMPACT_THRESHOLD:
                            sub = compact_lane_pipeline(
                                rc, s_chunks, CHUNK_RING_DEPTH,
                                lambda idx, b=base: tile(b + idx),
                            )
                        else:
                            sub = PipelineProgram(s_chunks, CHUNK_RING_DEPTH, lambda idx, b=base: tile(b + idx))
                        after = (seq_stages[-1].name,) if seq_stages else ()
                        seq_stages.append(Stage(f"seq_{len(seq_stages)}", after=after, program=sub))
                        cursor += s_chunks
                pipe = seq_stages[0].program if len(seq_stages) == 1 else KernelProgram(tuple(seq_stages))
            allocations = []
            for pool, size, kind, ids in (
                ("ub", report.owner_ub_bytes, ctx.datapath.aiv_count, aivs),
                ("l1", state_ring, ctx.datapath.aic_count, (aic,)),
            ):
                if size and pool in rc.pool_names:
                    units = rc.pool_units.get(pool, kind)
                    physical = tuple(sorted({unit for i in ids for unit in unit_group(i, kind, units)}))
                    allocations.append(rc.allocate(f"resident_{pool}", pool, size, physical))
            if not allocations:
                return pipe
            buffers = tuple(allocations)
            return KernelProgram((
                Stage("resident_allocate", acquire=buffers),
                Stage("chunks", after=("resident_allocate",), program=pipe),
                Stage("resident_release", after=("chunks",), free=buffers),
            ))

        return ParallelProgram(tuple(lane_program(aic, aivs, spans)
                                     for (aic, aivs), spans in zip(pairs, lanes, strict=False)))

    sources = (SOURCES[1], SOURCES[3], SOURCES[4]) if kda else GDN_SOURCES
    implementation = "flash_kda_persistent" if kda else ABI
    offdiag = sum(sum(p.work.matrix.values()) for p in totals if p.name == "offdiagonal_inverse")
    return make_plan(ctx, implementation, totals, sources=sources, program=lower, metadata={
        "template": "persistent", "persistent_ub_resident": report.persistent_ub_resident,
        "total_chunks": report.total_chunks, "chunk_tokens": c.bt, "chunk_state_bank": 0,
        "state_tile_bytes": report.state_tile_bytes, "owner_ub_bytes": report.owner_ub_bytes,
        "saved_entry_state_bytes": report.total_chunks * c.hv * c.dk * c.dv
        * element_bytes(c.stored_chunk_state_dtype) * c.save_intermediate_state,
        "l1_state_ring_bytes": report.l1_state_ring_bytes, "aic_per_head": 1, "aiv_per_head": 2,
        "offdiag_inv_flops": offdiag,
    })


def _staged_work(c: LinearAttentionContract, ctx: OperatorPlanningContext, *, kda: bool,
                 loads: int, stores: int) -> tuple[tuple[Phase, ...], dict[str, int]]:
    report = analyze_state_residency(c, ctx, persistent=kda)
    size = element_bytes(c.dtype)
    state = element_bytes(c.state_dtype) * c.hv * c.dk * c.dv
    tokens = sum(c.lengths) * c.hv
    prepare, recurrence, output = Work(), Work(), Work()
    score_bytes = 0
    offdiag = 0
    for (rows, snapshot), count in _chunk_counts(c).items():
        if not count:
            continue
        by_name = {p.name: p.work.scaled(count) for p in _chunk_phases(c, ctx, rows, kda=kda, snapshot=snapshot)}
        for name in ("load_tokens", "KKT_QKT", "gate_operands", "mask_and_diagonal_inverse",
                     "offdiagonal_inverse", "Kcd_U"):
            prepare += by_name[name]
        if not kda:
            # The output kernel recomputes QK and owns QH. The state kernel only needs WH.
            dtype = c.dtype
            inter = by_name["R_O_inter"]
            recurrence += Work({dtype: inter.matrix[dtype] - 2 * rows * c.dk * c.dv * count},
                               {**inter.resources, "mte1": rows * c.dk * size * count})
            for name in ("R32_to_R16_UB_to_L1", "delta_L0C_to_UB", "S_next"):
                recurrence += by_name[name]
            qk_operands = 2 * c.bt * c.dk
            qh_operands = (c.bt + c.dv) * c.dk
            output += by_name["O_L0C_to_GM"] + Work(
                {dtype: 2 * c.bt * (c.bt + c.dv) * c.dk * count + 2 * rows * (c.bt - rows) * c.dv * count},
                {"mte1": (qk_operands + qh_operands + rows * (c.bt - rows)) * size * count},
            ) + transfer(ctx, read=2 * rows * c.dk * size * count, aiv=False)
        offdiag += sum(by_name["offdiagonal_inverse"].matrix.values())
        score_bytes += rows * c.bt * size * count

    # Stage 1 publishes decayed K/Q, U, Kg, decay/gamma and the score workspace.
    # Aqk aliases the staged score workspace when returned to the caller.
    operands = tokens * (STAGED_KEY_WORKSPACES * c.dk + c.dv) * size
    gamma = (report.total_chunks * c.hv * c.dk if kda else tokens) * FP32_BYTES
    residual = tokens * c.dv * size
    bank = report.l2_hit_bytes + report.dram_miss_bytes
    saved = score_bytes * ((1 if kda else 2) if c.save_matrices else 0)
    prepare += transfer(ctx, write=operands + gamma) + Work(
        resources={"hbm": score_bytes + saved, "fixpipe": score_bytes + saved},
    )
    if kda:
        phases = (
            Phase("prepare_workspace_group", prepare),
            Phase("SyncAll.workspace", Work(), ("prepare_workspace_group",)),
        )
        # Stage 2 writes H_c before mutating its UB state, including staging at group boundaries.
        bank_bytes = report.total_chunks * c.hv * c.dk * c.dv * FP32_BYTES
        return phases, {"offdiag_inv_flops": offdiag, "total_chunks": report.total_chunks,
                        "chunk_state_bank": bank_bytes,
                        "saved_entry_state_bytes": bank_bytes * c.save_intermediate_state}
    recurrence += (
        transfer(ctx, read=operands, aiv=False)
        + transfer(ctx, read=gamma + state * loads,
                   write=report.compulsory_store_bytes + state * stores + residual)
        + Work(resources={"vector": (2 * len(c.lengths) - loads) * c.hv * c.dk * c.dv,
                          "mte3": len(c.lengths) * c.hv * c.dk * c.dv * size})
    )
    output += transfer(ctx, read=score_bytes + residual + bank, aiv=False, hits=report.l2_hit_bytes)
    phases = (
        Phase(CHUNK32_KERNELS[0], prepare),
        Phase(CHUNK32_KERNELS[1], recurrence, (CHUNK32_KERNELS[0],)),
        Phase(CHUNK32_KERNELS[2], output, (CHUNK32_KERNELS[1],)),
    )
    return phases, {
        "offdiag_inv_flops": offdiag, "total_chunks": report.total_chunks,
        "chunk_state_bank": report.chunk_bank_bytes, "l2_hit_bytes": report.l2_hit_bytes,
        "compulsory_store_bytes": report.compulsory_store_bytes, "reload_bytes": report.reload_bytes,
        "dram_miss_bytes": report.dram_miss_bytes, "evicted_chunks": report.evicted_chunks,
        "saved_entry_state_bytes": report.chunk_bank_bytes if c.save_intermediate_state else 0,
    }


def _flash_kda_state_wave(c: LinearAttentionContract, ctx: OperatorPlanningContext,
                          sizes: Counter[int], save_entry: bool) -> tuple[Phase, ...]:
    size = element_bytes(c.dtype)
    head_chunks = sum(sizes.values()) * c.hv
    state_bytes = c.dk * c.dv * FP32_BYTES
    work = Work()
    for rows, count in sizes.items():
        selected = {"R_O_inter", "R32_to_R16_UB_to_L1", "delta_L0C_to_UB", "S_next", "O_L0C_to_GM"}
        for phase in _chunk_phases(c, ctx, rows, kda=True, snapshot=False):
            if phase.name in selected:
                work += phase.work.scaled(count * c.hv)
        operands = rows * ((STAGED_KEY_WORKSPACES * c.dk + c.dv) + c.bt) * size
        work += transfer(ctx, read=operands * count * c.hv, aiv=False)
    work += transfer(ctx, read=head_chunks * c.dk * FP32_BYTES)
    return (
        Phase("save_entry_state_uncached", transfer(ctx, write=head_chunks * state_bytes * save_entry)),
        Phase("state_and_output_group", work, ("save_entry_state_uncached",)),
    )


def _flash_kda_state_group(c: LinearAttentionContract, ctx: OperatorPlanningContext,
                           loads: int, finals: int, continued: bool, carry: int):
    elements = c.hv * c.dk * c.dv
    entry = elements * FP32_BYTES
    state = elements * element_bytes(c.state_dtype)
    load = Phase("load_state_group", transfer(ctx, read=loads * (entry if continued else state))
                 + Work(resources={"vector": (2 * len(c.lengths) - loads) * elements,
                                   "mte3": elements * len(c.lengths) * element_bytes(c.dtype)}),
                 ("SyncAll.workspace",))
    store = Phase("store_state_group", transfer(ctx, write=finals * state + carry * entry),
                  ("state_and_output_group",))
    wave_count = ceil_div(max(c.lengths), c.bt)

    def wave(index: int):
        offset = index * c.bt
        sizes = Counter(min(c.bt, length - offset) for length in c.lengths if length > offset)
        return _flash_kda_state_wave(c, ctx, sizes, save_entry=not continued or index > 0)

    sizes = Counter()
    for length in c.lengths:
        full, tail = divmod(length, c.bt)
        if full:
            sizes[c.bt] += full
        if tail:
            sizes[tail] += 1
    save, recur = _flash_kda_state_wave(c, ctx, sizes, save_entry=True)
    if continued:
        save = replace(save, work=transfer(ctx, write=entry * (sum(sizes.values()) - len(c.lengths))))
    return (load, replace(save, after=(load.name,)), recur, store), wave_count, wave


def _staged_shapes(c: LinearAttentionContract, kda: bool, slots: int, group_limit: int):
    if not kda:
        yield c.lengths, len(c.lengths) * c.use_initial_state, len(c.lengths) * c.output_final_state, False, 0
        return
    chunk_counts = tuple(ceil_div(n, c.bt) for n in c.lengths)
    for start, consumed, counts in flash_kda_workspace_groups(chunk_counts, slots // c.hv, group_limit):
        rows = tuple(min(n * c.bt, c.lengths[start + i] - consumed * c.bt)
                     for i, n in enumerate(counts) if n)
        loads = len(rows) * (c.use_initial_state or consumed > 0)
        carry = sum(bool(n) and consumed + n < chunk_counts[start + i] for i, n in enumerate(counts))
        finals = (len(rows) - carry) * c.output_final_state
        yield rows, loads, finals, consumed > 0, carry


def _build_staged(c: LinearAttentionContract, ctx: OperatorPlanningContext, *,
                  kda: bool, pairs: tuple[tuple[int, tuple[int, ...]], ...],
                  slots: int = KDA_WORKSPACE_SLOTS, group_limit: int = 1) -> OperatorPlan:
    leaf = None if kda else CHUNK32_LEAVES.get(c.abi, c.abi)
    if leaf not in CHUNK32_KERNELS:
        leaf = None

    @lru_cache(maxsize=CHUNK_WORK_CACHE_SIZE)
    def group(key):
        rows, loads, finals, continued, carry = key
        group_contract = replace(c, lengths=rows)
        phases, details = _staged_work(group_contract, ctx, kda=kda, loads=loads, stores=finals)
        waves, wave = 0, None
        if kda:
            state_phases, waves, wave = _flash_kda_state_group(group_contract, ctx, loads, finals, continued, carry)
            phases += state_phases
        elif leaf:
            phases = tuple(replace(p, after=()) for p in phases if p.name == leaf)
        return phases, details, waves, wave

    counts = Counter()
    spans = []
    end = 0
    for key in _staged_shapes(c, kda, slots, group_limit):
        counts[key] += 1
        end += 1
        if spans and spans[-1][1] == key:
            spans[-1] = (end, key)
        else:
            spans.append((end, key))
    workspace = Counter()
    phase_work = {}
    for key, count in counts.items():
        phases, payload, _, _ = group(key)
        workspace.update({name: value * count for name, value in payload.items()})
        for phase in phases:
            phase_work[phase.name] = phase_work.get(phase.name, Work()) + phase.work.scaled(count)
    totals = tuple(replace(p, work=phase_work[p.name]) for p in group(spans[0][1])[0])
    ends = tuple(end for end, _ in spans)

    def lower(rc: RecipeContext) -> KernelProgram:
        def tile(index: int) -> tuple[Stage, ...]:
            phases, details, waves, wave = group(spans[bisect_right(ends, index)][1])
            used = pairs[:min(len(pairs), details["total_chunks"] * c.hv)]
            aics = tuple(aic for aic, _ in used)
            aivs = tuple(chain.from_iterable(pair for _, pair in used))
            if wave is None:
                return tuple(bind_phase(p, rc, ctx, aics, aivs) for p in phases)

            def state_tile(chunk: int) -> tuple[Stage, ...]:
                return tuple(bind_phase(p, rc, ctx, aics, aivs) for p in wave(chunk))

            before = tuple(bind_phase(p, rc, ctx, aics, aivs) for p in phases[:STAGED_STATE_WAVE_START])
            store = replace(bind_phase(phases[-1], rc, ctx, aics, aivs), after=("state_chunks",))
            return (*before, Stage("state_chunks", after=(before[-1].name,),
                                   program=PipelineProgram(waves, 1, state_tile)), store)

        if not kda:
            return KernelProgram(tile(0))
        return KernelProgram((Stage("workspace_groups", program=PipelineProgram(ends[-1], 1, tile)),))

    sources = (SOURCES[1], SOURCES[3], SOURCES[5]) if kda else CHUNK32_SOURCES
    boundaries = ("flash_kda_staged",) if kda else (leaf,) if leaf else CHUNK32_KERNELS
    implementation = "flash_kda_staged" if kda else c.abi if leaf else "gdn_chunk32"
    return make_plan(ctx, implementation, totals, sources=sources,
                     launch_count=len(boundaries), program=lower, metadata={
                         "template": "staged", "launch_boundaries": boundaries, "workspace_groups": ends[-1],
                         "persistent_ub_resident": False, "chunk_tokens": c.bt,
                         "saved_outputs_l2_cache_disabled": kda, **workspace,
                     })


def _recurrent_work(c: LinearAttentionContract, ctx: OperatorPlanningContext, tokens: int,
                    *, kda: bool, save_tokens: bool) -> tuple[Phase, ...]:
    elem = element_bytes(c.dtype)
    width = c.dk if kda else 1
    state = tokens * c.dk * c.dv
    inputs = tokens * ((2 * c.dk + c.dv) * elem + width * element_bytes(c.gate_dtype) + element_bytes(c.beta_dtype))
    contraction = 2 * state
    return (
        Phase("load_tokens", transfer(ctx, read=inputs)),
        Phase("decay", Work(resources={"vector": state + tokens * width}), ("load_tokens",)),
        Phase("key_dot", Work(resources={"vector": contraction}), ("decay",)),
        Phase("rank1_update", Work(resources={"vector": 2 * state + 2 * tokens * c.dv}), ("key_dot",)),
        Phase("query_dot", Work(resources={"vector": contraction}), ("rank1_update",)),
        Phase("publish_output", transfer(ctx, write=tokens * c.dv * elem), ("query_dot",)),
        Phase("publish_token_state", transfer(ctx, write=state * element_bytes(c.state_dtype) * save_tokens), ("query_dot",)),
    )


def _build_recurrent(c: LinearAttentionContract, ctx: OperatorPlanningContext, *,
                     streaming: bool = False) -> OperatorPlan:
    kda = c.operation == "kda_recurrent"
    save_tokens = kda and (c.output_final_state or c.save_intermediate_state)
    nheads = len(c.lengths) * c.hv
    ctx.datapath.route("gm", "ub", c.dtype)
    ctx.datapath.route("ub", "gm", c.state_dtype)
    report = analyze_state_residency(c, ctx, persistent=True)
    sequences = tuple(sorted(c.lengths))
    tokens = sum(sequences) * c.hv
    state = nheads * c.dk * c.dv * element_bytes(c.state_dtype)

    @lru_cache(maxsize=CHUNK_WORK_CACHE_SIZE)
    def token_work(count: int) -> tuple[Phase, ...]:
        return _recurrent_work(c, ctx, count, kda=kda, save_tokens=save_tokens)

    read = Phase("load_initial_state", transfer(ctx, read=state * c.use_initial_state))
    write = Phase("publish_final_state", transfer(ctx, write=state * (not kda and c.output_final_state)),
                  ("publish_output",))
    token_phases = token_work(tokens)
    totals = (read, replace(token_phases[0], after=(read.name,)), *token_phases[1:], write)
    owners = nheads * ceil_div(c.dv, c.bv)

    def lower(rc: RecipeContext) -> KernelProgram:
        aics, aivs = owner_units(ctx)
        aivs = aivs[:owners]

        def tile(index: int) -> tuple[Stage, ...]:
            active = (len(sequences) - bisect_right(sequences, index)) * c.hv
            stages = tuple(bind_phase(p, rc, ctx, aics, aivs) for p in token_work(active))
            if not save_tokens:
                return stages
            # Each head writes to the packed slot for sequence_start + token_index.
            checkpoint = replace(stages[-1], metadata={"phase": stages[-1].name, "token_index": index})
            return (*stages[:-1], checkpoint)

        return KernelProgram((
            bind_phase(read, rc, ctx, aics, aivs),
            Stage("tokens", after=(read.name,), program=PipelineProgram(sequences[-1], 1, tile)),
            replace(bind_phase(write, rc, ctx, aics, aivs), after=("tokens",)),
        ))

    implementation = ABI if streaming else "fused_recurrent_kda" if kda else "gdn_recurrent"
    sources = GDN_SOURCES if streaming else (SOURCES[-2],) if kda else ()
    return make_plan(ctx, implementation, totals, sources=sources, program=lower, metadata={
        "owner_ub_bytes": report.owner_ub_bytes, "state_tile_bytes": report.state_tile_bytes,
        "state_contraction_ops": 2 * tokens * c.dk * c.dv, "persistent_ub_resident": True,
        "token_state_slots": tokens if save_tokens else 0,
        "token_state_bytes": tokens * c.dk * c.dv * element_bytes(c.state_dtype) if save_tokens else 0,
    })


def _build_gdn(c: LinearAttentionContract, ctx: OperatorPlanningContext) -> OperatorPlan:
    verify_native_gdn_abi(c, ctx, abi_id=CANN_GDN_ABI)
    if max(c.dk, c.dv) > GDN_CUBE_DIM:
        raise missing("linear_attention_gdn_dimension_unsupported", "CANN Arch35 GDN supports Dk and Dv up to 128")
    streaming, wy = [], []
    for length in c.lengths:
        stream_cost = length * c.dk * c.dv * GDN_STREAM_STATE_OPS_PER_ELEMENT * GDN_STREAM_VECTOR_COST
        cube_cost = ceil_div(length, c.bt) * GDN_CUBE_FLOPS_PER_CHUNK
        (streaming if stream_cost < cube_cost else wy).append(length)

    routes = {"streaming_exact_sequences": len(streaming), "chunked_wy_sequences": len(wy)}
    if not wy:
        plan = _build_recurrent(c, ctx, streaming=True)
        return replace(plan, metadata={**plan.metadata, **routes, "template": "streaming_exact",
                                       "total_chunks": 0, "offdiag_inv_flops": 0, "chunk_state_bank": 0})
    chunked = replace(c, lengths=tuple(wy))
    plan = _build_persistent(chunked, ctx, _pairs(ctx, chunked))
    if not streaming:
        return replace(plan, metadata={**plan.metadata, **routes})

    stream = _build_recurrent(replace(c, lengths=tuple(streaming)), ctx, streaming=True)
    # Independent tasks share one launch and contend for the same physical units.
    stream_phases = tuple(replace(p, phase_id=f"stream_{p.phase_id}", after=tuple(f"stream_{d}" for d in p.after))
                          for p in stream.pipeline_spec.phases)
    spec = replace(plan.pipeline_spec, phases=stream_phases + plan.pipeline_spec.phases)
    return replace(plan, work=plan.work + stream.work, pipeline_spec=spec,
                   program=lambda rc: ParallelProgram((stream.program(rc), plan.program(rc))),
                   metadata={**plan.metadata, **routes, "template": "mixed"})


def _build_commit(c: LinearAttentionContract, ctx: OperatorPlanningContext) -> OperatorPlan:
    if c.metadata.get("ssm_family", c.metadata.get("transition_family", c.metadata.get("family", "kda"))) != "kda":
        raise ModelingError("commit_recurrent_kda_requires_kda_transition")

    def bytes_read(commit: RecoverSsmCommitPlan, total_tokens: int) -> int:
        # Each value partition reads its correction slice and the whole K16/decay32 pair.
        rec_override = c.metadata.get("record_bytes_per_token", c.metadata.get("record_bytes"))
        tile_v = min(c.bv, COMMIT_VALUE_TILE)
        parts = int(c.metadata.get("row_partitions", ceil_div(c.dv, tile_v)))
        if parts < 1:
            raise ModelingError("commit_recurrent_kda_row_partitions_invalid")
        if rec_override is not None and parts <= 1:
            return int(round(total_tokens * float(rec_override)))
        per_token = FP32_BYTES * c.dv + (LOW_PRECISION_BYTES + FP32_BYTES) * c.dk * parts
        return total_tokens * c.hv * per_token

    return build_recover_ssm_plan(c, ctx, sources=SOURCES[-1:], replay_bytes=bytes_read)


def chunk32_setup_ms(context: RecipeContext) -> float:
    from mindie_sim.primitives import NS_TO_MS

    table = context.primitive_table
    if table is None:
        return 0.0
    cell = table.cell("P12_intracore_flag")
    band = cell.metrics.get("flag_ns") if cell.supported else None
    return band.p50 * NS_TO_MS if band is not None and band.p50 is not None else 0.0


class Arch35LinearAttentionStrategy:
    arch_id = ArchId("npu_arch_3510")
    abi = ABI
    sources = SOURCES

    def chunk32_timing_policy(
        self, context: RecipeContext, _step_sync_ms: float = 0.0,
    ) -> tuple[float, float]:
        from mindie_sim.primitives import lower_primitive
        from mindie_sim.primitives.lowering import INTRA_BLOCK_MODE

        table = context.primitive_table
        if table is None:
            raise ModelingError("gdn_chunk32_requires_primitive_table:P13_crosscore_sync:mode4_intra_block_ns")
        sync = lower_primitive("P13_crosscore_sync", {"mode": INTRA_BLOCK_MODE}, table, context=context)
        return chunk32_setup_ms(context), sync.analytical_ms

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        c = validate_contract(op, ctx)
        if isinstance(c, CausalConvContract):
            return c.mode == "incremental" or (ctx.requested_abi or dict(op.metadata).get("abi") or op.recipe) == "causal_conv1d_update"
        return (c.abi not in FLA_GDN_ABIS | TRITON_KDA_ARCH22_ABIS | {"aclnnChunkGatedDeltaRuleArch22"}
                and c.operation != "recover_ssm_verify")

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        c = validate_contract(op, ctx)
        if not self.accepts(op, ctx):
            raise missing("linear_attention_arch35_abi_unsupported", "The requested ABI is unavailable on Arch35")
        if isinstance(c, CausalConvContract):
            ctx.datapath.route("gm", "ub", c.dtype)
            ctx.datapath.route("ub", "gm", c.state_dtype)
            return build_causal_conv1d_plan(
                c, ctx, implementation="causal_conv1d_update", sources=(SOURCES[2],),
                weight_copies=dict(op.metadata).get("token_block_count", 1),
            )
        if c.operation == "recover_ssm_commit":
            return _build_commit(c, ctx)
        if not c.lengths:
            return make_plan(ctx, c.abi or c.operation, (),
                             metadata={"total_chunks": 0, "state_contraction_ops": 0})
        if c.abi == RECURRENT_GDN_ABI:
            return build_native_recurrent_gdn(c, ctx)
        if c.operation.endswith("_recurrent"):
            return _build_recurrent(c, ctx)
        if c.operation == "gdn_prefill" and c.bt != DIAG_BLOCK_WIDTH:
            return _build_gdn(c, ctx)
        if c.operation == "gdn_chunk32" or c.bt == DIAG_BLOCK_WIDTH:
            if not can_run_gdn_chunk32(
                c.hk, c.hv, c.dk, c.dv, activation_dtype=c.dtype, state_dtype=c.state_dtype,
            ):
                raise missing(
                    "gdn_chunk32_domain_unsupported",
                    f"gdn_chunk32 requires bf16, fp32 state, dk={GDN_HEAD_DIM_K}, dv={GDN_CHUNK32_HEAD_DIM_V}",
                )
            return _build_staged(c, ctx, kda=False, pairs=_pairs(ctx, c))
        pairs = _pairs(ctx, c)

        from mindie_sim.templates.dsl_catalog import resolve_flash_kda_route

        pairs = pairs[:KDA_MAX_AIC_CORES]
        request = FLASH_KDA_TEMPLATE_BY_ABI.get(c.abi, c.kda_template_request)
        m = c.metadata
        sequence_layout = ctx.layout.upper() if ctx.requests else ""
        if sequence_layout in {"BSND", "BNSD", "BSH", "TND"}:
            layout, has_cu = 1, sequence_layout == "TND"
        else:
            legacy_layout = op.layout.upper()
            layout = m.get("layout_kind", int(legacy_layout in {"BSND", "TND"}))
            has_cu = m.get("has_cu", legacy_layout == "TND")
        slots = m.get("workspace_slots", KDA_WORKSPACE_SLOTS)
        selected, group = resolve_flash_kda_route(
            c.lengths, c.hv, len(pairs), requested_template=request, device_name=m.get("device_name", ""),
            route_coefficients=m.get("route_coefficients", ()), layout_kind=layout, workspace_slots=slots,
            max_chunk_group=m.get("max_chunk_group"), metadata_word_capacity=m.get("metadata_word_capacity"),
            has_cu=has_cu,
            calibration_enabled=m.get("calibration_enabled", True), enable_grad=m.get("enable_grad", False),
            query_heads=c.hq,
        )
        if selected == "staged":
            return _build_staged(c, ctx, kda=True, pairs=pairs, slots=slots, group_limit=group)
        return _build_persistent(c, ctx, pairs)


STRATEGY = Arch35LinearAttentionStrategy()
