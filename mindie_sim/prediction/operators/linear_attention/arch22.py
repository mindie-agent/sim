"""Arch22 staged GDN, staged KDA and MemBase recurrent execution.

[Role] Model the GM workspace handoffs and intra-kernel SyncAll boundaries.
[Invariants] CANN GDN reuses one BF16 GM state slot per sequence and head.
[Timeouts & Locks] Groups are issued through a bounded runtime pipeline.
[Side-Effect Policy] Pure planning. No device context or source-file reads.
Upstream-ABI: aclnnChunkGatedDeltaRuleArch22
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from itertools import chain

from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...program import KernelProgram, PipelineProgram, RecipeContext, Stage
from ...runtime.ring_buffer import RingBuffer, lower_generation
from ..contracts import OperatorPlanningContext
from ..kda.triton_kda import (
    TRITON_KDA_CHUNK_ABIS, TRITON_KDA_RECURRENT_ABIS, build_triton_chunk_kda, build_triton_recurrent_kda,
)
from ..matrix.refill_steady import compact_periodic_pipeline
from ..sharedkv.l0_sync import p03_fixpipe_setup_ms, p12_flag_ms, p13_paired_ms, p13_syncall_ms
from ..sharedkv.split_core import p01_cold_hbm_group_cap
from ._gdn_native import RECURRENT_GDN_ABI, build_native_recurrent_gdn
from .common import (
    FLA_GDN_ABIS, FP32_BYTES, CausalConvContract, LinearAttentionContract, Phase, analyze_state_residency,
    bind_phase, ceil_div, compute_wy_chunk_work, element_bytes, make_plan, missing,
    owner_units, transfer, validate_contract,
)

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/chunk_gated_delta_rule/op_kernel/arch22/chunk_gated_delta_rule_arch22.h",
        source_sha256="56c72f3db58221fa12763e10c0d2925ac4d8f5679891851e5249e07ffd376419",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/chunk_gated_delta_rule/op_kernel/arch22/chunk_gated_delta_rule_stage2_arch22.h",
        source_sha256="a4189e2fe9d448fe4c279f66165578a2f946e435b7c861b1dd024e0999a2f471",
    ),
    SourceRef(
        repository="cann/cann-recipes-infer",
        revision="3c9266727c2a25bd17d503ad37d189a134ff364b",
        relative_path="ops/cannbot_dsl/fused_recurrent_kda.py",
        source_sha256="725d325763e2f248f62f72397f54c55a5006706ee32137b128eb2b3900bda49a",
    ),
)
ABI = "aclnnChunkGatedDeltaRuleArch22"
GROUP_CHUNKS_PER_AIC = 2
GROUP_WORK_CACHE_SIZE = 128
GDN_GROUP_BARRIERS = 3
WY_PREPARE_KEY_PANELS = 5
WY_PREPARE_SCORE_PANELS = 4
WY_RECURRENCE_KEY_PANELS = 3
STATE_UPDATE_OPS_PER_ELEMENT = 3
# Arch22 chunk_gated_delta_rule_stage{1,2,3}_arch22.h op counts (zero fitted timings)
ARCH22_STAGE1_INV_FMAS_PER_HEAD = 31_744
ARCH22_STAGE1_INV_BCAST_MULT = 3
ARCH22_STAGE1_GBK_OPS_PER_ELEM = 4
ARCH22_STAGE1_INV_FLAG_ROUNDS = 31 * 4  # 31 dependent scalar/vector sync rounds per head-chunk
ARCH22_STAGE1_SCRATCH_READ_KEY_PANELS = 6  # stage1_arch22.h:298-799 (queryCon,keyCon,gBK,vBeta,kkWs,attnWs)
ARCH22_STAGE1_SCRATCH_READ_VAL_PANELS = 2
ARCH22_STAGE1_SCRATCH_READ_SCORE_PANELS = 4
ARCH22_STAGE1_SCRATCH_WRITE_KEY_PANELS = 4
ARCH22_STAGE1_SCRATCH_WRITE_VAL_PANELS = 1
ARCH22_STAGE1_EXP_SCORE_MULT = 3  # 3*C^2 + 3*C + 1 exp ops across stages 1-3 (stage1_arch22.h:433,619)
ARCH22_STAGE1_EXP_SFU_WEIGHT = 4  # P07/P08 SFU ratio fallback
ARCH22_STAGE1_CROSSCORE_BATCH = 4
ARCH22_STAGE1_CROSSCORE_SYNCS = 6  # 6 CrossCore notifications per 4-head-chunk batch
ARCH22_STAGE1_MATMULS_PER_HEAD_CHUNK = 5  # kkt, qkt, offdiag_inv, kcd, u (stage1_arch22.h)
ARCH22_STAGE3_CROSSCORE_SYNCS = 2  # 2 CrossCore notifications per head-chunk (stage3_arch22.h:114)
ARCH22_STAGE3_MATMULS_PER_HEAD_CHUNK = 1  # masked_qkt @ v_inner (stage3_arch22.h:178)
ARCH22_STAGE3_MASK_OPS_PER_SCORE = 12
ARCH22_STAGE2_PAIRED_SYNCS_PER_CHUNK = 2  # 3 one-way CrossCore flags -> 2 paired handoffs (stage2_arch22.h:116-135)
ARCH22_STAGE2_MATMULS_PER_CHUNK = 3  # CalVPrime, CalAttnInter, CalStateNew (stage2_arch22.h:126-131)
ARCH22_STAGE2_AIV_CAST_MUL_OPS = 3  # AIV0 bf16->fp32->mul->bf16 (stage2_arch22.h:157)
ARCH22_COMPACT_GROUP_THRESHOLD = 20
ARCH22_COMPACT_WINDOW = 2
ARCH35_ABIS = frozenset({
    "aclnnChunkGatedDeltaRuleArch35", "gdn_prefill_arch35", "flash_kda_persistent",
    "flash_kda_arch35", "causal_conv1d_update", "commit_recurrent_kda",
})
ATTENTION_OPERATIONS = frozenset({"gdn_prefill", "kda_prefill", "gdn_recurrent", "kda_recurrent"})


def _rolling_chunk_phases(c: LinearAttentionContract, ctx: OperatorPlanningContext, rows: int,
                          load_state: bool) -> tuple[Phase, ...]:
    elem = element_bytes(c.dtype)
    heads = c.hv
    tokens = rows * heads
    # Both Arch22 Stage2 inputs and its rolling finalState use BF16.
    state_bytes = heads * c.dk * c.dv * element_bytes("bf16")
    key, value = tokens * c.dk * elem, tokens * c.dv * elem
    w = compute_wy_chunk_work(rows, c.dk, c.dv, c.bt)
    return (
        Phase("read_rolling_state",
              transfer(ctx, read=state_bytes * load_state + heads * FP32_BYTES)),
        Phase("R_O_inter",
              transfer(ctx, read=2 * key + 2 * value + 2 * state_bytes * load_state, aiv=False)
              + Work({c.dtype: heads * (w.r_flops + w.o_inter_flops)}, {
                  "mte1": 2 * key + value + 2 * state_bytes, "fixpipe": 2 * value, "hbm": 2 * value,
              }), ("read_rolling_state",)),
        Phase("cal_decayed_state",
              Work(resources={"vector": heads * (w.cast_ops + w.state_update_ops + ARCH22_STAGE2_AIV_CAST_MUL_OPS * c.dk * c.dv)}),
              ("R_O_inter",)),
        Phase("write_decayed_state",
              transfer(ctx, write=state_bytes),
              ("cal_decayed_state",)),
        Phase("write_rolling_state",
              transfer(ctx, read=key + state_bytes, aiv=False) + Work({c.dtype: heads * w.delta_flops}, {
                  "mte1": key + value, "fixpipe": state_bytes, "hbm": state_bytes,
              }), ("write_decayed_state",)),
    )


def _work_for_group(c: LinearAttentionContract, ctx: OperatorPlanningContext, rows: int,
                    load_state: bool, store_state: bool, *, rolling: bool) -> tuple[tuple[Phase, ...], dict[str, int]]:
    kda = c.operation == "kda_prefill"
    elem, state = element_bytes(c.dtype), element_bytes(c.state_dtype)
    tokens = rows * c.hv
    full, tail = divmod(rows, c.bt)
    sizes = ((c.bt, full), (tail, 1)) if tail else ((c.bt, full),)
    work = [(n, count * c.hv, compute_wy_chunk_work(n, c.dk, c.dv, c.bt))
            for n, count in sizes if count]
    scores = sum(n * n * count for n, count, _ in work)
    workspace = {
        "gCum": tokens * (c.dk if kda else 1) * FP32_BYTES,
        "kCumDecay": tokens * c.dk * elem,
        "vInner": tokens * c.dv * elem,
        "qPrime": tokens * c.dk * elem,
        "kg": tokens * c.dk * elem,
        "qkt": scores * elem,
    }
    prepare_vec = sum(n * (w.gate_ops + w.mask_ops + w.diag_inv_ops) for _, n, w in work)
    if rolling:
        head_chunks = sum(count for _, count, _ in work)
        exp_ops = head_chunks * (ARCH22_STAGE1_EXP_SCORE_MULT * c.bt * c.bt + ARCH22_STAGE1_EXP_SCORE_MULT * c.bt + 1)
        prepare_vec += head_chunks * (
            ARCH22_STAGE1_INV_BCAST_MULT * ARCH22_STAGE1_INV_FMAS_PER_HEAD
            + ARCH22_STAGE1_GBK_OPS_PER_ELEM * c.bt * (2 * c.dk + c.dv)
        ) + exp_ops * ARCH22_STAGE1_EXP_SFU_WEIGHT
    prepare_cube = sum(n * (w.kkt_flops + w.qkt_flops + w.offdiag_inv_flops
                            + w.kcd_flops + w.u_flops) for _, n, w in work)
    state_cube = sum(n * (w.r_flops + w.delta_flops + w.o_inter_flops) for _, n, w in work)
    state_vec = sum(n * (w.cast_ops + w.state_update_ops) for _, n, w in work)
    output_cube = sum(n * w.o_intra_flops for _, n, w in work)
    report = analyze_state_residency(replace(c, lengths=(rows,)), ctx, persistent=rolling)
    workspace.update(chunk_state_bank=report.chunk_bank_bytes, l2_hit_bytes=report.l2_hit_bytes,
                     offdiag_inv_flops=sum(n * w.offdiag_inv_flops for _, n, w in work),
                     dram_miss_bytes=report.dram_miss_bytes, evicted_chunks=report.evicted_chunks,
                     compulsory_store_bytes=report.compulsory_store_bytes, reload_bytes=report.reload_bytes)
    gate_elements = tokens * (c.dk if kda else 1)
    inputs = tokens * ((2 * c.dk + c.dv) * elem + element_bytes(c.beta_dtype))
    inputs += gate_elements * element_bytes(c.gate_dtype)
    v_bytes = workspace["vInner"]
    state_bytes = c.hv * c.dk * c.dv * state
    bank_transfer = report.l2_hit_bytes + report.dram_miss_bytes
    matrix_save = 2 * workspace["qkt"] if c.save_matrices else 0
    # Stage 2 publishes the entering chunk states, residual R, and O_inter.
    cube_workspace = workspace["kCumDecay"] + v_bytes + workspace["qkt"] + matrix_save
    vector_workspace = workspace["gCum"] + workspace["qPrime"] + workspace["kg"]
    recurrence_inputs = workspace["kCumDecay"] + v_bytes + workspace["qPrime"] + workspace["kg"]
    s1_scratch_read = (
        (ARCH22_STAGE1_SCRATCH_READ_KEY_PANELS * tokens * c.dk
         + ARCH22_STAGE1_SCRATCH_READ_VAL_PANELS * tokens * c.dv
         + ARCH22_STAGE1_SCRATCH_READ_SCORE_PANELS * scores) * elem
        if rolling else 0
    )
    s1_scratch_write = (
        (ARCH22_STAGE1_SCRATCH_WRITE_KEY_PANELS * tokens * c.dk
         + ARCH22_STAGE1_SCRATCH_WRITE_VAL_PANELS * tokens * c.dv) * elem
        + scores * elem // 2
        if rolling else 0
    )
    prepare = transfer(ctx, read=inputs + s1_scratch_read, write=vector_workspace + s1_scratch_write) + Work(
        {c.dtype: prepare_cube}, {
            "hbm": cube_workspace,
            "mte1": (tokens * (WY_PREPARE_KEY_PANELS * c.dk + c.dv) + WY_PREPARE_SCORE_PANELS * scores) * elem,
            "fixpipe": cube_workspace + scores * FP32_BYTES,
            "vector": prepare_vec + (gate_elements if kda else 0),
        },
    )
    if rolling:
        recurrence = Work()
        for chunk_rows, count in sizes:
            if not count:
                continue
            initial = not load_state and chunk_rows == sizes[0][0]
            if initial:
                for phase in _rolling_chunk_phases(c, ctx, chunk_rows, False):
                    recurrence += phase.work
            for phase in _rolling_chunk_phases(c, ctx, chunk_rows, True):
                recurrence += phase.work.scaled(count - initial)
    else:
        recurrence = (
            transfer(ctx, read=recurrence_inputs + bank_transfer, aiv=False, hits=report.l2_hit_bytes)
            + transfer(ctx, read=state_bytes * load_state + report.total_chunks * c.hv * FP32_BYTES,
                       write=report.compulsory_store_bytes)
            + Work({c.dtype: state_cube}, {
                "mte1": (tokens * (WY_RECURRENCE_KEY_PANELS * c.dk + c.dv)
                         + c.dk * c.dv * report.total_chunks * c.hv) * elem,
                "fixpipe": 2 * v_bytes + bank_transfer + state_bytes * store_state,
                "hbm": 2 * v_bytes + bank_transfer + state_bytes * store_state,
                "vector": state_vec,
            })
        )
    # Stage 3 combines the GM residual and scores with the existing O_inter.
    output_vec = ARCH22_STAGE3_MASK_OPS_PER_SCORE * scores if rolling else 0
    output = (
        transfer(ctx, read=workspace["qkt"] + workspace["gCum"], write=workspace["qkt"])
        + transfer(ctx, read=workspace["qkt"] + 2 * v_bytes, aiv=False)
        + Work({c.dtype: output_cube}, {"hbm": v_bytes, "fixpipe": v_bytes,
                                       "mte1": workspace["qkt"] + v_bytes,
                                       **({"vector": output_vec} if output_vec else {})})
    )
    phases = (
        Phase("RunStage1", prepare, ("SyncAll.entry",)),
        Phase("SyncAll.prepare", Work(), ("RunStage1",)),
        Phase("RunStage2", recurrence, ("SyncAll.prepare",)),
        Phase("SyncAll.state", Work(), ("RunStage2",)),
        Phase("RunStage3", output, ("SyncAll.state",)),
        Phase("SyncAll.output", Work(), ("RunStage3",)),
    )
    return phases, workspace


def _lower_rolling_group(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, rc: RecipeContext,
    pairs: tuple[tuple[int, tuple[int, ...]], ...], used: tuple[tuple[int, tuple[int, ...]], ...],
    cores: tuple[int, ...], phases: tuple[Phase, ...], bound: tuple[Stage, ...],
    key: tuple[int, bool, bool], chunks: int, sequence: int,
    position: int, index: int, spans: list[tuple[int, tuple[int, bool, bool], int]],
    ends: tuple[int, ...], groups: int,
) -> tuple[Stage, ...]:
    """Lower Arch22 rolling GDN stages 1-3 with P01 cold-core caps and P12/P13 syncs."""
    rows, load_state, _ = key
    # Stage 2 binds headsPerCore = ceil(Nv / AIC) -> min(AIC, Nv) AICs + AIV0 subblocks
    # (chunk_gated_delta_rule_stage2_arch22.h:91-115).
    stage2_used = pairs[:min(len(pairs), c.hv)]
    stage2_cores = tuple(aic for aic, _ in stage2_used)
    stage2_aiv0 = tuple(peers[0] for _, peers in stage2_used)
    stage3_aiv0 = tuple(peers[0] for _, peers in used)
    stage2_caps = p01_cold_hbm_group_cap(rc, len(stage2_cores))
    full_caps = p01_cold_hbm_group_cap(rc, len(cores))
    fixpipe_setup_ms = p03_fixpipe_setup_ms(rc)
    chunk_sync_ms = (
        ARCH22_STAGE2_PAIRED_SYNCS_PER_CHUNK * p13_paired_ms(rc)
        + ARCH22_STAGE2_MATMULS_PER_CHUNK * fixpipe_setup_ms
    )
    sync_all_ms = p13_syncall_ms(rc)
    slot_bytes = c.dk * c.dv * element_bytes("bf16")
    slot = {"state_slot": "rolling_final_state", "sequence": sequence,
            "base_byte": sequence * c.hv * slot_bytes, "head_stride_bytes": slot_bytes}
    first_group = (position == 0 or spans[position - 1][2] != sequence) and (
        index == 0 or index == ends[position - 1]
    )

    def _with_caps(st: Stage, cap_map: dict[str, float]) -> Stage:
        if not cap_map:
            return st
        applied = {k: v for k, v in cap_map.items() if k in st.work}
        return replace(st, caps={**st.caps, **applied}) if applied else st

    def chunk_tile(chunk: int) -> tuple[Stage, ...]:
        valid = min(c.bt, rows - chunk * c.bt)
        loaded = load_state or chunk > 0
        stages = tuple(replace(_with_caps(bind_phase(p, rc, ctx, stage2_cores, stage2_aiv0), stage2_caps),
                               metadata={"phase": p.name, **slot})
                       for p in _rolling_chunk_phases(c, ctx, valid, loaded))
        if chunk_sync_ms > 0.0:
            read = replace(stages[0], previous=("stage2_crosscore_sync",),
                           metadata={**stages[0].metadata,
                                     "initial_state_read": first_group and chunk == 0 and loaded})
            sync = Stage("stage2_crosscore_sync", after=("write_rolling_state",),
                         latency=chunk_sync_ms, keep_empty=True)
            return (read, *stages[1:], sync)
        read = replace(stages[0], previous=("write_rolling_state",),
                       metadata={**stages[0].metadata,
                                 "initial_state_read": first_group and chunk == 0 and loaded})
        return (read, *stages[1:])

    stage2_prog = (
        compact_periodic_pipeline(
            rc, chunks, 1, chunk_tile,
            head=ARCH22_COMPACT_WINDOW, tail=ARCH22_COMPACT_WINDOW,
            threshold=ARCH22_COMPACT_WINDOW * 2,
        )
        if groups > ARCH22_COMPACT_GROUP_THRESHOLD else PipelineProgram(chunks, 1, chunk_tile)
    )
    stage1_jobs_per_core = ceil_div(chunks * c.hv, max(1, len(used)))
    stage1_cc_ms = (
        ceil_div(stage1_jobs_per_core, ARCH22_STAGE1_CROSSCORE_BATCH) * ARCH22_STAGE1_CROSSCORE_SYNCS * p13_paired_ms(rc)
        + stage1_jobs_per_core * ARCH22_STAGE1_MATMULS_PER_HEAD_CHUNK * fixpipe_setup_ms
    )
    stage1_sync_ms = stage1_jobs_per_core * ARCH22_STAGE1_INV_FLAG_ROUNDS * p12_flag_ms(rc) + stage1_cc_ms
    s1_capped = _with_caps(bound[0], full_caps)
    stage1_bound = (
        Stage(s1_capped.name, after=s1_capped.after, program=KernelProgram((
            replace(s1_capped, after=()),
            Stage("stage1_sync", after=(s1_capped.name,), latency=stage1_sync_ms, keep_empty=True),
        ))) if stage1_sync_ms > 0.0 else s1_capped
    )
    stage2 = replace(bound[2], work={}, placement={}, program=stage2_prog)
    stage3_cc_ms = stage1_jobs_per_core * (
        ARCH22_STAGE3_CROSSCORE_SYNCS * p13_paired_ms(rc)
        + ARCH22_STAGE3_MATMULS_PER_HEAD_CHUNK * fixpipe_setup_ms
    )
    s3_raw = _with_caps(bind_phase(phases[4], rc, ctx, cores, stage3_aiv0), full_caps)
    stage3 = (
        Stage(s3_raw.name, after=s3_raw.after, program=KernelProgram((
            replace(s3_raw, after=()),
            Stage("stage3_sync", after=(s3_raw.name,), latency=stage3_cc_ms, keep_empty=True),
        ))) if stage3_cc_ms > 0.0 else s3_raw
    )
    return (
        stage1_bound, replace(bound[1], latency=sync_all_ms), stage2,
        replace(bound[3], latency=sync_all_ms), stage3, replace(bound[5], latency=sync_all_ms),
    )


def _build_staged(c: LinearAttentionContract, ctx: OperatorPlanningContext) -> OperatorPlan:
    aics, aivs = owner_units(ctx)
    pairs = tuple((aic, peers[:2]) for aic, peers in ctx.datapath.aic_aiv_pairs
                  if aic in aics and len(peers) >= 2 and all(aiv in aivs for aiv in peers[:2]))
    if not pairs:
        raise missing("linear_attention_no_mixed_owners", "Staged prefill needs paired AIC and AIV owners")
    gdn = c.operation == "gdn_prefill"
    rolling = gdn and c.abi not in FLA_GDN_ABIS
    group_rows = GROUP_CHUNKS_PER_AIC * len(pairs) * c.bt
    if not gdn:
        group_rows = c.bt * max(1, len(pairs) // c.hv)
    groups = 0
    counts: Counter[tuple[int, bool, bool]] = Counter()
    spans: list[tuple[int, tuple[int, bool, bool], int]] = []
    active_slots = (tuple(i for i, view in enumerate(ctx.requests) if not view.padding)
                    if ctx.requests else range(len(c.lengths)))
    for sequence, length in zip(active_slots, c.lengths, strict=True):
        first = min(length, group_rows)
        if length == first:
            series = ((1, (first, c.use_initial_state, c.output_final_state)),)
        else:
            middle, tail = divmod(length - first - 1, group_rows)
            series = ((1, (first, c.use_initial_state, True)),
                      (middle, (group_rows, True, True)),
                      (1, (tail + 1, True, c.output_final_state)))
        for count, key in series:
            if count:
                groups += count
                counts[key] += count
                spans.append((groups, key, sequence))

    @lru_cache(maxsize=GROUP_WORK_CACHE_SIZE)
    def group(key):
        return _work_for_group(c, ctx, *key, rolling=rolling)

    phase_work: dict[str, Work] = {}
    workspace: Counter[str] = Counter()
    for key, count in counts.items():
        phases, payload = group(key)
        for phase in phases:
            phase_work[phase.name] = phase_work.get(phase.name, Work()) + phase.work.scaled(count)
        workspace.update({name: count * amount for name, amount in payload.items()})
    sample, _ = group(spans[0][1])
    totals = (Phase("SyncAll.entry", Work()), *(replace(p, work=phase_work[p.name]) for p in sample))
    ends = tuple(end for end, _, _ in spans)

    def lower(rc: RecipeContext) -> KernelProgram:
        def tile(index: int) -> tuple[Stage, ...]:
            position = bisect_right(ends, index)
            _, key, sequence = spans[position]
            phases, _ = group(key)
            chunks = ceil_div(key[0], c.bt)
            used = pairs[:min(len(pairs), chunks * c.hv)]
            cores = tuple(aic for aic, _ in used)
            vectors = tuple(chain.from_iterable(peers for _, peers in used))
            ring = RingBuffer(allocation="group_workspace", pool="gm", depth=1)
            events = lower_generation(ring, 0, "publish_workspace", ("consume_workspace",),
                                      producer_extra_after=("SyncAll.state",))
            credit, write, publish, read, last = ring.to_stages()
            bound = tuple(bind_phase(p, rc, ctx, cores, vectors) for p in phases)
            if rolling:
                bound = _lower_rolling_group(
                    c, ctx, rc, pairs, used, cores, phases, bound,
                    key, chunks, sequence, position, index, spans, ends, groups,
                )
            return (
                replace(bound[0], after=()), *bound[1:4], credit, write, publish,
                replace(bound[4], after=(events.publish,)), replace(read, after=("RunStage3",)),
                last, replace(bound[5], after=(events.last_use,)),
            )

        entry_latency = p13_syncall_ms(rc) if rolling else 0.0
        return KernelProgram((
            Stage("SyncAll.entry", latency=entry_latency, keep_empty=True),
            Stage("chunk_groups", after=("SyncAll.entry",), program=PipelineProgram(groups, 1, tile)),
        ))

    rolling_slot = c.dk * c.dv * element_bytes("bf16") if rolling else 0
    implementation = (ABI if rolling else c.abi) if gdn else "flash_kda_staged"
    return make_plan(
        ctx, implementation, totals, sources=SOURCES[:2] if rolling else () if gdn else SOURCES[-1:],
        metadata={"template": "staged", "chunk_tokens": c.bt, "sync_all_count": 1 + GDN_GROUP_BARRIERS * groups,
                  "launch_boundaries": ("ChunkGatedDeltaRule::Process",) if gdn else ("flash_kda_staged",),
                  "total_chunks": sum(ceil_div(length, c.bt) for length in c.lengths),
                  "persistent_ub_resident": False, "rolling_state_slot_bytes": rolling_slot,
                  "rolling_state_gm_bytes": rolling_slot * (len(ctx.requests) or len(c.lengths)) * c.hv,
                  **workspace},
        program=lower,
    )


def _build_recurrent(c: LinearAttentionContract, ctx: OperatorPlanningContext) -> OperatorPlan:
    kda = c.operation == "kda_recurrent"
    count = sum(c.lengths) * c.hv
    size = element_bytes(c.dtype)
    gate_width = c.dk if kda else 1
    inputs = count * ((2 * c.dk + c.dv) * size + gate_width * element_bytes(c.gate_dtype)
                      + element_bytes(c.beta_dtype))
    outputs = count * c.dv * size
    slot_bytes = c.dk * c.dv * element_bytes(c.state_dtype)
    state = len(c.lengths) * c.hv * slot_bytes
    report = analyze_state_residency(c, ctx, persistent=True)
    contraction = 2 * count * c.dk * c.dv
    # Key and query each contract over the full state. The update decays each
    # element and applies the outer product after forming beta * (V - S @ K).
    update = STATE_UPDATE_OPS_PER_ELEMENT * count * c.dk * c.dv + count * (gate_width + 2 * c.dv)
    owners = len(c.lengths) * c.hv * ceil_div(c.dv, c.bv)
    save_tokens = kda and (c.output_final_state or c.save_intermediate_state)
    state_read = Phase("load_initial_state", transfer(ctx, read=c.use_initial_state * state))
    if save_tokens:
        ctx.datapath.route("ub", "gm", c.state_dtype)
        sequence_lengths = tuple(sorted(c.lengths))

        def token_phases(active):
            return (
                Phase("load_tokens", transfer(ctx, read=inputs // count * active)),
                Phase("key_dot_rank1_update_query_dot", Work(resources={"vector": (2 * contraction + update) // count * active}),
                      ("load_tokens",)),
                Phase("publish_output", transfer(ctx, write=outputs // count * active), ("key_dot_rank1_update_query_dot",)),
                Phase("publish_token_state", transfer(ctx, write=slot_bytes * active), ("key_dot_rank1_update_query_dot",)),
            )

        def lower(rc):
            aics, aivs = (units[:owners] for units in owner_units(ctx))

            def tile(index):
                active = (len(sequence_lengths) - bisect_right(sequence_lengths, index)) * c.hv
                stages = tuple(bind_phase(p, rc, ctx, aics, aivs) for p in token_phases(active))
                return (*stages[:-1], replace(stages[-1], metadata={**stages[-1].metadata, "token_index": index}))

            return KernelProgram((
                bind_phase(state_read, rc, ctx, aics, aivs),
                Stage("tokens", after=(state_read.name,), program=PipelineProgram(sequence_lengths[-1], 1, tile)),
            ))

        load, *rest = token_phases(count)
        phases = (state_read, replace(load, after=(state_read.name,)), *rest)
    else:
        lower = None
        phases = (
            Phase("load_state_and_tokens", transfer(ctx, read=inputs + c.use_initial_state * state)),
            Phase("key_dot_rank1_update_query_dot", Work(resources={"vector": 2 * contraction + update}),
                  ("load_state_and_tokens",)),
            Phase("publish", transfer(ctx, write=outputs + c.output_final_state * state),
                  ("key_dot_rank1_update_query_dot",)),
        )
    return make_plan(ctx, "fused_recurrent_kda" if kda else "gdn_recurrent", phases,
                     sources=SOURCES[-1:] if kda else (),
                     owners=owners, program=lower,
                     metadata={"state_tile_bytes": report.state_tile_bytes, "owner_ub_bytes": report.owner_ub_bytes,
                               "state_contraction_ops": contraction, "persistent_ub_resident": True,
                               "token_state_slots": count if save_tokens else 0,
                               "token_state_bytes": count * slot_bytes if save_tokens else 0})


class Arch22LinearAttentionStrategy:
    arch_id = ArchId("npu_arch_2201")
    abi = ABI
    sources = SOURCES

    def chunk32_timing_policy(
        self, _context: RecipeContext, _step_sync_ms: float,
    ) -> tuple[float, float]:
        return 0.0, 0.0

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        c = validate_contract(op, ctx)
        if isinstance(c, CausalConvContract):
            return False
        direct_routes = {"l0c->ub", "ub->l1", ("l0c", "ub"), ("ub", "l1")}
        required_routes = c.metadata.get("required_routes") or ()
        if isinstance(required_routes, str):
            required_routes = (required_routes,)
        required_routes = tuple(tuple(route) if isinstance(route, list) else route for route in required_routes)
        return (c.operation in ATTENTION_OPERATIONS and c.abi not in ARCH35_ABIS
                and not direct_routes.intersection(required_routes)
                and not c.metadata.get("requires_l0c_to_ub") and not c.metadata.get("requires_ub_to_l1")
                and c.kda_template_request != "persistent" and not c.metadata.get("chunk32"))

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        c = validate_contract(op, ctx)
        if not self.accepts(op, ctx) or isinstance(c, CausalConvContract):
            raise missing("linear_attention_arch22_abi_unsupported",
                          "The requested ABI or data route is unavailable on Arch22")
        if not c.lengths:
            return make_plan(ctx, c.abi or c.operation, (),
                             metadata={"total_chunks": 0, "state_contraction_ops": 0})
        if c.abi in TRITON_KDA_CHUNK_ABIS:
            return build_triton_chunk_kda(c, ctx)
        if c.abi in TRITON_KDA_RECURRENT_ABIS:
            return build_triton_recurrent_kda(c, ctx)
        if c.abi == RECURRENT_GDN_ABI:
            return build_native_recurrent_gdn(c, ctx)
        if c.operation.endswith("_prefill"):
            return _build_staged(c, ctx)
        return _build_recurrent(c, ctx)


STRATEGY = Arch22LinearAttentionStrategy()
