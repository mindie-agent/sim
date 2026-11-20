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

from mindie_sim.schema.common import ModelingError, align_up
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext, Stage
from ...runtime.ring_buffer import RingBuffer, lower_generation
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    DIAG_BLOCK_WIDTH, FP32_BYTES, KDA_MAX_AIC_CORES, KDA_ROUTE_PARAMETERS, KDA_WORKSPACE_SLOTS,
    CausalConvContract, LinearAttentionContract, Phase, RecoverSsmCommitPlan,
    analyze_state_residency, bind_phase, build_causal_conv1d_plan, build_recover_ssm_plan,
    ceil_div, compute_wy_chunk_work, element_bytes, flash_kda_group_limit,
    flash_kda_workspace_groups, make_plan, missing, owner_units,
    select_flash_kda_template, transfer, validate_contract,
)

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
        repository="cann/cannbot-dsl", revision=DSL_REVISION,
        relative_path="samples/attention/flash_kda/flash_kda_metadata.py",
        source_sha256="5033e923d9d63602708918691f75fcf8f210a0782e850d7822090362f4242377",
    ),
    SourceRef(
        repository="cann/cannbot-dsl", revision=DSL_REVISION,
        relative_path="samples/attention/flash_kda/flash_kda_persistent.py",
        source_sha256="16f01982db4fc4f053f238e353637ac314cdc4d589ed948a32ee391297d50b08",
    ),
    SourceRef(
        repository="cann/cannbot-dsl", revision=DSL_REVISION,
        relative_path="samples/attention/flash_kda/flash_kda_staged.py",
        source_sha256="41957c35307c32dc5326278a01e176325f264bd63d414c059166f12bfd47ac17",
    ),
    SourceRef(
        repository="cann/cannbot-dsl", revision=DSL_REVISION,
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
STATE_UPDATE_OPS_PER_ELEMENT = 3
LOW_PRECISION_BYTES = 2
COMMIT_VALUE_TILE = 128
SHORT_TRAIN_MIN_LENGTH = 8192
SHORT_TRAIN_CHUNK_GROUP = 13
KDA_PACKED_INVERSE_PRODUCTS = 5
CHUNK32_KERNELS = (
    "gdn_prepare_arch35_chunk32", "gdn_state_arch35_chunk32", "gdn_output_arch35_chunk32",
)


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
    inverse = w.offdiag_inv_flops
    inverse_operand_bytes = GDN_INVERSE_OPERAND_PANELS * DIAG_BLOCK_WIDTH**2 * LOW_PRECISION_BYTES if inverse else 0
    diagonal = w.diag_inv_ops
    if kda:
        blocks = c.bt // FRACTAL_DIM
        diagonal = blocks * FRACTAL_DIM * (FRACTAL_DIM - 1) * (FRACTAL_DIM - 2) // TRIANGULAR_FMA_DIVISOR
        inverse = 2 * KDA_PACKED_INVERSE_PRODUCTS * c.bt * c.bt**2
        inverse_operand_bytes = 2 * KDA_PACKED_INVERSE_PRODUCTS * c.bt**2 * LOW_PRECISION_BYTES
    matrix_saves = 2 * rows * c.bt * elem if c.save_matrices else 0
    state_save = state * element_bytes(c.stored_chunk_state_dtype) if c.save_intermediate_state else 0

    initialize = transfer(ctx, read=state * state_elem * c.use_initial_state) + Work(resources={
        "vector": state if c.use_initial_state else 2 * state, "mte3": state * elem,
    })
    return (
        Phase("state_begin", initialize),
        Phase("load_tokens", transfer(ctx, read=inputs)),
        Phase("KKT_QKT", Work({c.dtype: w.kkt_flops + w.qkt_flops},
                              {"mte1": 2 * key * elem, "fixpipe": 2 * score * FP32_BYTES}), ("load_tokens",)),
        Phase("gate_operands", Work(resources={
            "vector": w.gate_ops + (2 * rows * (gate_width - 1) if kda else 0),
            "mte3": (WY_PREPROCESSED_KEY_PANELS * key + value) * elem,
        }), ("load_tokens",)),
        Phase("mask_and_diagonal_inverse", Work(resources={
            "vector": w.mask_ops + diagonal, "mte3": 2 * score * elem,
        }), ("KKT_QKT", "gate_operands")),
        Phase("offdiagonal_inverse", Work({"fp16": inverse}, {
            "mte1": inverse_operand_bytes, "fixpipe": inverse_operand_bytes // 2,
        }), ("mask_and_diagonal_inverse",)),
        Phase("Kcd_U", Work({c.dtype: w.kcd_flops + w.u_flops}, {
            "mte1": (score + key + value) * elem, "fixpipe": (key + value) * elem,
        }), ("offdiagonal_inverse",)),
        Phase("save_matrices", Work(resources={"hbm": matrix_saves, "fixpipe": matrix_saves}),
              ("offdiagonal_inverse",)),
        Phase("save_entry_state", transfer(ctx, write=state_save), ("state_begin",)),
        Phase("R_O_inter", Work({c.dtype: w.r_flops + w.o_inter_flops}, {
            "mte1": (2 * key + state) * elem, "fixpipe": value * FP32_BYTES,
        }), ("Kcd_U", "save_entry_state")),
        Phase("R32_to_R16_UB_to_L1", Work(resources={"vector": w.cast_ops, "mte3": value * elem}),
              ("R_O_inter",)),
        Phase("delta_L0C_to_UB", Work({c.dtype: w.delta_flops}, {
            "mte1": (value + key) * elem, "fixpipe": state * FP32_BYTES,
        }), ("R32_to_R16_UB_to_L1",)),
        Phase("O_L0C_to_GM", Work({c.dtype: w.o_intra_flops}, {
            "mte1": (score + value) * elem, "fixpipe": value * elem, "hbm": value * elem,
        }), ("delta_L0C_to_UB",)),
        Phase("S_next", Work(resources={
            "vector": w.state_update_ops + state * (snapshot or kda), "mte3": state * elem * snapshot,
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
                stages.extend((replace(read, after=("state_end",)), last))
                return tuple(stages)

            pipe = PipelineProgram(ends[-1], CHUNK_RING_DEPTH, tile)
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

    sources = (SOURCES[1], SOURCES[3], SOURCES[4]) if kda else SOURCES[:1]
    implementation = "flash_kda_persistent" if kda else ABI
    offdiag = sum(compute_wy_chunk_work(rows, c.dk, c.dv, c.bt).offdiag_inv_flops * count
                  for (rows, _), count in counts.items())
    return make_plan(ctx, implementation, totals, sources=sources, program=lower, metadata={
        "template": "persistent", "persistent_ub_resident": report.persistent_ub_resident,
        "total_chunks": report.total_chunks, "chunk_tokens": c.bt, "chunk_state_bank": 0,
        "state_tile_bytes": report.state_tile_bytes, "owner_ub_bytes": report.owner_ub_bytes,
        "saved_entry_state_bytes": report.total_chunks * c.hv * c.dk * c.dv
        * element_bytes(c.stored_chunk_state_dtype) * c.save_intermediate_state,
        "l1_state_ring_bytes": report.l1_state_ring_bytes, "aic_per_head": 1, "aiv_per_head": 2,
        "offdiag_inv_flops": offdiag if not kda else sum(
            p.work.matrix.get("fp16", 0) for p in totals if p.name == "offdiagonal_inverse"
        ),
    })


def _staged_work(c: LinearAttentionContract, ctx: OperatorPlanningContext, *, kda: bool,
                 loads: int, stores: int) -> tuple[tuple[Phase, ...], dict[str, int]]:
    report = analyze_state_residency(c, ctx)
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
        for name in ("R_O_inter", "R32_to_R16_UB_to_L1", "delta_L0C_to_UB", "S_next"):
            recurrence += by_name[name]
        output += by_name["O_L0C_to_GM"]
        offdiag += by_name["offdiagonal_inverse"].matrix.get("fp16", 0)
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
    recurrence += (
        transfer(ctx, read=operands, aiv=False)
        + transfer(ctx, read=gamma + state * loads, write=report.compulsory_store_bytes + state * stores)
        + Work(resources={"vector": (2 * len(c.lengths) - loads) * c.hv * c.dk * c.dv,
                          "mte3": len(c.lengths) * c.hv * c.dk * c.dv * size})
    )
    output += transfer(ctx, read=score_bytes, aiv=False)
    bank_read = transfer(ctx, read=bank, aiv=False, hits=report.l2_hit_bytes)
    if kda:
        phases = (
            Phase("prepare_workspace_group", prepare),
            Phase("SyncAll.workspace", Work(), ("prepare_workspace_group",)),
            Phase("state_and_output_group", recurrence + bank_read + output, ("SyncAll.workspace",)),
            Phase("save_entry_state_uncached", transfer(ctx, write=report.chunk_bank_bytes * c.save_intermediate_state),
                  ("state_and_output_group",)),
        )
    else:
        recurrence += transfer(ctx, write=residual)
        output += bank_read + transfer(ctx, read=residual, aiv=False)
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


def _staged_shapes(c: LinearAttentionContract, kda: bool, slots: int, group_limit: int):
    if not kda:
        yield c.lengths, len(c.lengths) * c.use_initial_state, len(c.lengths) * c.output_final_state
        return
    chunk_counts = tuple(ceil_div(n, c.bt) for n in c.lengths)
    for start, consumed, counts in flash_kda_workspace_groups(chunk_counts, slots // c.hv, group_limit):
        rows = tuple(min(n * c.bt, c.lengths[start + i] - consumed * c.bt)
                     for i, n in enumerate(counts) if n)
        loads = len(rows) * (c.use_initial_state or consumed > 0)
        stores = sum(bool(n) and (c.output_final_state or consumed + n < chunk_counts[start + i])
                     for i, n in enumerate(counts))
        yield rows, loads, stores


def _build_staged(c: LinearAttentionContract, ctx: OperatorPlanningContext, *,
                  kda: bool, pairs: tuple[tuple[int, tuple[int, ...]], ...],
                  slots: int = KDA_WORKSPACE_SLOTS, group_limit: int = 1) -> OperatorPlan:
    @lru_cache(maxsize=CHUNK_WORK_CACHE_SIZE)
    def group(key):
        rows, loads, stores = key
        return _staged_work(replace(c, lengths=rows), ctx, kda=kda, loads=loads, stores=stores)

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
        phases, payload = group(key)
        workspace.update({name: value * count for name, value in payload.items()})
        for phase in phases:
            phase_work[phase.name] = phase_work.get(phase.name, Work()) + phase.work.scaled(count)
    totals = tuple(replace(p, work=phase_work[p.name]) for p in group(spans[0][1])[0])
    ends = tuple(end for end, _ in spans)

    def lower(rc: RecipeContext) -> KernelProgram:
        def tile(index: int) -> tuple[Stage, ...]:
            phases, details = group(spans[bisect_right(ends, index)][1])
            used = pairs[:min(len(pairs), details["total_chunks"] * c.hv)]
            aics = tuple(aic for aic, _ in used)
            aivs = tuple(chain.from_iterable(pair for _, pair in used))
            return tuple(bind_phase(phase, rc, ctx, aics, aivs) for phase in phases)

        if not kda:
            return KernelProgram(tile(0))
        return KernelProgram((Stage("workspace_groups", program=PipelineProgram(ends[-1], 1, tile)),))

    sources = (SOURCES[1], SOURCES[3], SOURCES[5]) if kda else SOURCES[:1]
    boundaries = ("flash_kda_staged",) if kda else CHUNK32_KERNELS
    return make_plan(ctx, "flash_kda_staged" if kda else "gdn_chunk32", totals, sources=sources,
                     launch_count=len(boundaries), program=lower, metadata={
                         "template": "staged", "launch_boundaries": boundaries, "workspace_groups": ends[-1],
                         "persistent_ub_resident": False, "chunk_tokens": c.bt,
                         "saved_outputs_l2_cache_disabled": kda, **workspace,
                     })


def _build_recurrent(c: LinearAttentionContract, ctx: OperatorPlanningContext) -> OperatorPlan:
    kda = c.operation == "kda_recurrent"
    ctx.datapath.route("gm", "ub", c.dtype)
    ctx.datapath.route("ub", "gm", c.state_dtype)
    record = analyze_state_residency(c, ctx, persistent=True)
    tokens = sum(c.lengths) * c.hv
    elem = element_bytes(c.dtype)
    state = len(c.lengths) * c.hv * c.dk * c.dv * element_bytes(c.state_dtype)
    width = c.dk if kda else 1
    inputs = tokens * ((2 * c.dk + c.dv) * elem + width * element_bytes(c.gate_dtype) + element_bytes(c.beta_dtype))
    contraction = 2 * tokens * c.dk * c.dv
    state_ops = STATE_UPDATE_OPS_PER_ELEMENT * tokens * c.dk * c.dv + tokens * (width + 2 * c.dv)
    phases = (
        Phase("load_state_and_tokens", transfer(ctx, read=inputs + c.use_initial_state * state)),
        Phase("key_dot", Work(resources={"vector": contraction}), ("load_state_and_tokens",)),
        Phase("gate_rank1_update", Work(resources={"vector": state_ops}), ("key_dot",)),
        Phase("query_dot", Work(resources={"vector": contraction}), ("gate_rank1_update",)),
        Phase("publish", transfer(ctx, write=tokens * c.dv * elem + c.output_final_state * state), ("query_dot",)),
    )
    return make_plan(ctx, "fused_recurrent_kda" if kda else "gdn_recurrent", phases,
                     owners=len(c.lengths) * c.hv * ceil_div(c.dv, c.bv),
                     metadata={"owner_ub_bytes": record.owner_ub_bytes, "state_tile_bytes": record.state_tile_bytes,
                               "state_contraction_ops": contraction, "persistent_ub_resident": True})


def _build_commit(c: LinearAttentionContract, ctx: OperatorPlanningContext) -> OperatorPlan:
    if c.metadata.get("ssm_family", c.metadata.get("transition_family", c.metadata.get("family", "kda"))) != "kda":
        raise ModelingError("commit_recurrent_kda_requires_kda_transition")

    def bytes_read(commit: RecoverSsmCommitPlan) -> int:
        # Each value partition reads its correction slice and the whole K16/decay32 pair.
        tile_v = min(c.bv, COMMIT_VALUE_TILE)
        parts = int(c.metadata.get("row_partitions", ceil_div(c.dv, tile_v)))
        if parts < 1:
            raise ModelingError("commit_recurrent_kda_row_partitions_invalid")
        per_token = FP32_BYTES * c.dv + (LOW_PRECISION_BYTES + FP32_BYTES) * c.dk * parts
        return commit.a * len(c.lengths) * c.hv * per_token

    return build_recover_ssm_plan(c, ctx, sources=SOURCES[-1:], replay_bytes=bytes_read)


class Arch35LinearAttentionStrategy:
    arch_id = ArchId("npu_arch_3510")
    abi = ABI
    sources = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        c = validate_contract(op, ctx)
        if isinstance(c, CausalConvContract):
            return c.mode == "incremental" or (ctx.requested_abi or dict(op.metadata).get("abi") or op.recipe) == "causal_conv1d_update"
        return c.abi != "aclnnChunkGatedDeltaRuleArch22" and c.operation != "recover_ssm_verify"

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
        if c.operation.endswith("_recurrent"):
            return _build_recurrent(c, ctx)
        pairs = _pairs(ctx, c)
        if c.operation == "gdn_chunk32" or c.bt == DIAG_BLOCK_WIDTH:
            return _build_staged(c, ctx, kda=False, pairs=pairs)
        if c.operation == "kda_prefill":
            pairs = pairs[:KDA_MAX_AIC_CORES]
            request = {
                "flash_kda_persistent": "persistent", "flash_kda_staged": "staged",
            }.get(c.abi, c.kda_template_request)
            m = c.metadata
            layout = m.get("layout_kind", int(op.layout.upper() in {"BSND", "TND"}))
            has_cu = m.get("has_cu", op.layout.upper() == "TND")
            slots = m.get("workspace_slots", KDA_WORKSPACE_SLOTS)
            max_group = m.get("max_chunk_group")
            calibrated = m.get("calibration_enabled", True)
            if (max_group is None and m.get("enable_grad") is True and layout == 0 and not has_cu
                    and len(c.lengths) == 1 and c.hq == c.hv == len(pairs) == KDA_MAX_AIC_CORES
                    and c.lengths[0] >= SHORT_TRAIN_MIN_LENGTH and not c.lengths[0] % c.bt):
                max_group = SHORT_TRAIN_CHUNK_GROUP
            selected = select_flash_kda_template(
                c, core_num=len(pairs), template_request=request,
                route_coefficients=m.get("route_coefficients", KDA_ROUTE_PARAMETERS), layout_kind=layout,
                workspace_slots=slots, max_chunk_group=max_group, metadata_word_capacity=m.get("metadata_word_capacity"),
                has_cu=has_cu, calibration_enabled=calibrated,
            )
            if selected == "staged":
                chunks = tuple(ceil_div(n, c.bt) for n in c.lengths)
                group = flash_kda_group_limit(chunks, c.hv, len(pairs), layout, max_group,
                                              request == "auto" and calibrated)
                return _build_staged(c, ctx, kda=True, pairs=pairs, slots=slots, group_limit=group)
        return _build_persistent(c, ctx, pairs)


STRATEGY: ArchStrategy = Arch35LinearAttentionStrategy()
