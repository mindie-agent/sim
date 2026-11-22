"""Arch35 GDN ``Chunk32`` (PR #22147) Prepare, State, and Output source planners.

[Role]
  Model the ``fla-npu`` ``chunk_gated_delta_rule_fwd`` ``BT=32`` path (``PR #22147``,
  ``use_exp2=False``, ``use_beta_sigmoid_in_kernel=True``) across all three
  mixed AIC+2AIV kernels (Prepare/WY, ChunkFwdH, ChunkFwdO) for supported GQA
  head ratios ``Hv/Hk in {1, 2, 3, 4}``.

[Invariants]
  * ``BT=32`` fits the ``32x32`` unit-triangular WY inverse in a single UB VCS
    tile, eliminating the ``Chunk64`` ``4x 64^3`` FP32 Cube block products and
    ``y_dump``/``y_reload`` GM workspace round-trip.
  * ``ChunkFwdH`` groups chunks contiguously per ``(seq, key_head, sub_group)``
    recurrence chain with ``slots=1`` serial state dependency and compacts long
    chunk chains via ``compact_periodic_pipeline``.
  * Every ``OperatorPlan`` strictly conserves ``Work`` across ``hbm``,
    ``mte2_aic``, ``mte2_aiv``, ``mte1``, ``cube``, ``vector``, ``fixpipe``, and
    ``mte3`` via ``mixed_core`` ownership and dynamic ``geometry.buffers`` checks.

[Timeouts & Locks]
  O(sequences + cores) plan construction and O(1) periodic compaction; no I/O.

[Side-Effect Policy]
  Pure immutable ``OpSpec`` and ``OperatorPlan`` construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Callable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ..contracts import OperatorPlanningContext
from .common import missing, owner_units
from .gdn_domain import (
    FLA_GDN_CHUNK_SOURCE,
    GDN_CHUNK32_HEAD_DIM_V,
    GDN_CHUNK32_TOKENS,
    GDN_HEAD_DIM_K,
    can_run_gdn_chunk32,
)
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from .gdn_prepare import PackedChunks
from mindie_sim.prediction.operators.vector.mixed_core import phases, require_pair, resident, split_reads
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage
from mindie_sim.prediction.operators.matrix.refill_steady import (
    PeriodicPipelineSpec,
    compact_periodic_peer_group,
    compact_periodic_pipeline,
)

PREPARE_RECIPE = "gdn_prepare_arch35_chunk32"
STATE_RECIPE = "gdn_state_arch35_chunk32"
OUTPUT_RECIPE = "gdn_output_arch35_chunk32"

PR22147_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "57de212f6a",
    "relative_path": "vllm_ascend/ops/gdn.py",
    "source_sha256": "b170b30626cf46bc4ec1e47080f9487c1e5443e51add42bb8c67bfd20a79fac4",
}

BF16_BYTES = 2
FP32_BYTES = 4
MAX_TASK_GROUP_HEADS = 4
PIPELINE_WINDOW_SLOTS = 2
RECURRENCE_SERIAL_SLOTS = 1
COMPACT_CHUNK_THRESHOLD = 16
COMPACT_TAIL_CHUNKS = 2
L2NORM_OPS_PER_ELEM = 5
SIGMOID_SCALE_OPS_PER_ELEM = 6
GATE_CUMSUM_OPS_PER_ROW = 4
CAUSAL_DECAY_OPS_PER_PAIR = 7
VCS32_TRIANGULAR_OPS = 66464
STATE_DECAY_OPS_PER_ELEM = 4
OUTPUT_FUSE_OPS_PER_ELEM = 5
CHUNK32_BUFFERS = {
    "ub": 160 * 1024, "l1": 192 * 1024, "l0a": 16 * 1024,
    "l0b": 32 * 1024, "l0c": 32 * 1024,
}


def _check_buffers(geometry: Geometry) -> dict[str, int]:
    available = dict(geometry.buffers or {})
    for pool, required in CHUNK32_BUFFERS.items():
        cap = int(available.get(pool, 0))
        if cap > 0 and required > cap:
            raise ModelingError(f"gdn_chunk32_buffer_capacity_exceeded:{pool}")
    return {p: sz for p, sz in CHUNK32_BUFFERS.items() if available.get(p, 0) > 0}


def _validate_chunk32_params(
    params: Mapping[str, Any],
) -> tuple[tuple[int, ...], int, int, int, int]:
    lengths = tuple(int(v) for v in (params.get("lengths") or ()))
    if not lengths or min(lengths) < 1:
        raise ValueError("gdn_chunk32_requires_positive_lengths")
    heads, keys = int(params.get("heads", 0)), int(params.get("key_heads", 0))
    dk = int(params.get("dk", GDN_HEAD_DIM_K))
    dv = int(params.get("dv", GDN_CHUNK32_HEAD_DIM_V))
    chunk = int(params.get("chunk", GDN_CHUNK32_TOKENS))
    dtype = str(params.get("dtype", "bf16")).lower()
    if chunk != GDN_CHUNK32_TOKENS:
        raise ValueError("gdn_chunk32_requires_chunk_32")
    if not can_run_gdn_chunk32(keys, heads, dk, dv, activation_dtype=dtype, state_dtype="fp32"):
        raise ValueError("gdn_chunk32_domain_unsupported")
    return lengths, heads, keys, dk, dv


def _build_mixed_plan(
    geometry: Geometry,
    cores: tuple[int, ...],
    owners: tuple[Sequence[Any], ...],
    work_fn: Callable[[Any], tuple[Work, Work]],
    *,
    aiv_first: bool,
    slots: int,
    recurrence_sync: bool = False,
    first_label: str,
    second_label: str,
    extra_meta: Mapping[str, Any],
) -> OperatorPlan:
    from mindie_sim.prediction.operators.linear_attention import chunk32_plan_timing

    require_pair(geometry)
    buf_sizes = _check_buffers(geometry)
    total_work = Work()
    for items in owners:
        for item in items:
            aiv_w, aic_w = work_fn(item)
            total_work += split_reads(aiv_w, 0, geometry.resource_units)
            total_work += split_reads(aic_w, aic_w.resources["mte2"], geometry.resource_units)

    def program(context):
        setup_ms, sync_ms = chunk32_plan_timing(context, recurrence_sync=recurrence_sync)

        def tile_for(core: int, items: Sequence[Any]):
            def tile(idx: int):
                aiv_w, aic_w = work_fn(items[idx])
                half = aiv_w.scaled(0.5)
                if aiv_first:
                    l0, l1 = f"{first_label}_0", f"{first_label}_1"
                    base = (
                        phases(l0, half, context, core, aiv=0),
                        phases(l1, half, context, core, aiv=1),
                        phases(second_label, aic_w, context, core, aiv=None, after=(l0, l1)),
                    )
                    tail_after = (second_label,)
                else:
                    s0, s1 = f"{second_label}_0", f"{second_label}_1"
                    base = (
                        phases(first_label, aic_w, context, core, aiv=None),
                        phases(s0, half, context, core, aiv=0, after=(first_label,)),
                        phases(s1, half, context, core, aiv=1, after=(first_label,)),
                    )
                    tail_after = (s0, s1)
                if recurrence_sync:
                    return (*base, Stage("recurrence_sync", after=tail_after, latency=sync_ms))
                return base
            return tile

        specs = tuple(
            PeriodicPipelineSpec(
                len(items), slots, tile_for(core, items),
                head=1, tail=COMPACT_TAIL_CHUNKS, threshold=COMPACT_CHUNK_THRESHOLD,
            )
            for core, items in zip(cores, owners, strict=True)
        )
        pipes = compact_periodic_peer_group(context, specs, compactor=compact_periodic_pipeline)
        branches = tuple(
            resident(context, core, buf_sizes, KernelProgram((
                Stage("setup", latency=setup_ms, keep_empty=setup_ms <= 0.0),
                Stage("tiles", after=("setup",), program=pipe),
            ))) if items else KernelProgram(())
            for core, items, pipe in zip(cores, owners, pipes, strict=True)
        )
        return ParallelProgram(branches)

    return OperatorPlan(total_work, program, {
        "execution_class": "mixed_aic_aiv",
        "chunk_size": GDN_CHUNK32_TOKENS,
        "source_program": FLA_GDN_CHUNK_SOURCE,
        **dict(extra_meta),
    })


def _active_cores(geometry: Geometry, ctx: OperatorPlanningContext | None) -> tuple[int, ...]:
    if ctx is None or not ctx.requests:
        return tuple(range(max(1, int(geometry.compute_units))))
    aics, aivs = owner_units(ctx)
    paired = tuple(aic for aic, peers in ctx.datapath.aic_aiv_pairs
                   if aic in aics and len(peers) >= 2 and all(peer in aivs for peer in peers[:2]))
    if not paired:
        raise missing("linear_attention_no_mixed_owners", "Chunk32 needs paired AIC and AIV owners")
    return paired


def _tile_spans(cores: tuple[int, ...], count: int) -> tuple[range, ...]:
    return tuple(range(lane, count, len(cores)) for lane in range(len(cores)))


def make_prepare_op(params: Mapping[str, Any]) -> OpSpec:
    lengths, heads, keys, dk, dv = _validate_chunk32_params(params)
    chunks = PackedChunks.from_lengths(lengths, GDN_CHUNK32_TOKENS)
    tokens, gqa = sum(lengths), heads // keys
    gate_dtype = str(params.get("gate_dtype", "fp32")).lower()
    beta_dtype = str(params.get("beta_dtype", "bf16")).lower()
    output_a = bool(params.get("output_a", True))
    use_beta_sig = bool(params.get("use_beta_sigmoid_in_kernel", True))
    semantic = (1, heads, tokens, dk)
    padded = (1, heads, chunks.padded_tokens, dk)
    meta = {
        "implementation": PREPARE_RECIPE, "lengths": lengths,
        "heads": heads, "key_heads": keys, "head_dim_k": dk, "head_dim_v": dv,
        "chunk": GDN_CHUNK32_TOKENS, "gqa_ratio": gqa, "dtype": "bf16",
        "gate_dtype": gate_dtype, "beta_dtype": beta_dtype,
        "use_qk_l2norm_in_kernel": True, "use_beta_sigmoid_in_kernel": use_beta_sig,
        "use_exp2": False, "output_a": output_a,
        "dispatch": {
            "implementation": PREPARE_RECIPE, "chunk": GDN_CHUNK32_TOKENS,
            "gate_dtype": gate_dtype, "beta_dtype": beta_dtype,
            "use_exp2": False, "use_beta_sigmoid_in_kernel": use_beta_sig,
        },
        "native_source": FLA_GDN_CHUNK_SOURCE, "pr_source": PR22147_SOURCE,
    }
    bt = GDN_CHUNK32_TOKENS
    gate_bytes = FP32_BYTES if gate_dtype == "fp32" else BF16_BYTES
    beta_bytes = FP32_BYTES if beta_dtype == "fp32" else BF16_BYTES
    aiv_read = float(keys * tokens * (2 * dk * BF16_BYTES + gqa * dv * BF16_BYTES + gqa * (gate_bytes + beta_bytes)))
    a_write = keys * tokens * gqa * bt * BF16_BYTES if output_a else 0
    aiv_write = float(keys * tokens * (2 * dk * BF16_BYTES + 2 * FP32_BYTES + gqa * FP32_BYTES) + a_write)
    aiv_vec = float(keys * (
        tokens * (2 * dk * L2NORM_OPS_PER_ELEM + gqa * (dk + dv) * SIGMOID_SCALE_OPS_PER_ELEM)
        + chunks.total * gqa * (bt * GATE_CUMSUM_OPS_PER_ROW + bt * bt * CAUSAL_DECAY_OPS_PER_PAIR + VCS32_TRIANGULAR_OPS)
    ))
    aic_read = float(keys * tokens * (dk + gqa * (dk + dv)) * BF16_BYTES)
    wu_write = float(keys * tokens * gqa * (dk + dv) * BF16_BYTES)
    flops = float(chunks.total * keys * (2 * bt * bt * dk + gqa * (2 * bt * dk * bt + 2 * bt * dv * bt)))
    return OpSpec(
        op_id=f"gdn-prepare-c32:h{heads}:hk{keys}:c{chunks.total}",
        family="gdn_prepare", phase="prefill", stage=0, layer=0, requests=(),
        semantic_shape=semantic, local_shape=semantic,
        padded_shape=padded, launch_shape=padded,
        flops=flops, matrix_flops=(("bf16", flops),), vector_ops=aiv_vec,
        hbm_bytes=aiv_read + aiv_write + aic_read + wu_write,
        mte2_bytes=aiv_read + aic_read, mte3_bytes=aiv_write,
        dtype="bf16", recipe=PREPARE_RECIPE, source_boundary="kernel",
        metadata=tuple(sorted(meta.items())),
    )


def build_prepare(op: OpSpec, geometry: Geometry, *, ctx: OperatorPlanningContext | None = None) -> OperatorPlan:
    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in meta["lengths"])
    heads, keys = int(meta["heads"]), int(meta["key_heads"])
    dk, dv, gqa = int(meta["head_dim_k"]), int(meta["head_dim_v"]), heads // keys
    bt = GDN_CHUNK32_TOKENS
    chunks = PackedChunks.from_lengths(lengths, bt)
    gate_bytes = FP32_BYTES if meta.get("gate_dtype", "fp32") == "fp32" else BF16_BYTES
    beta_bytes = FP32_BYTES if meta.get("beta_dtype", "bf16") == "fp32" else BF16_BYTES
    output_a = bool(meta.get("output_a", True))
    cores = _active_cores(geometry, ctx)

    def group_work(group_id: int) -> tuple[Work, Work]:
        _, _, _, rows = chunks.range(group_id // keys)
        qk_bytes = 2 * rows * dk * BF16_BYTES
        v_bytes = gqa * rows * dv * BF16_BYTES
        gb_bytes = gqa * rows * (gate_bytes + beta_bytes)
        aiv_read = float(qk_bytes + v_bytes + gb_bytes)
        a_write = gqa * rows * bt * BF16_BYTES if output_a else 0
        aiv_write = float(qk_bytes + 2 * rows * FP32_BYTES + gqa * rows * FP32_BYTES + a_write)
        aiv_vec = float(
            2 * rows * dk * L2NORM_OPS_PER_ELEM
            + gqa * rows * (dk + dv) * SIGMOID_SCALE_OPS_PER_ELEM
            + gqa * (bt * GATE_CUMSUM_OPS_PER_ROW + bt * bt * CAUSAL_DECAY_OPS_PER_PAIR
                     + VCS32_TRIANGULAR_OPS)
        )
        aiv = Work(resources={
            "hbm": aiv_read + aiv_write, "mte2": aiv_read,
            "vector": aiv_vec, "mte3": aiv_write,
        })
        aic_read = float(rows * dk * BF16_BYTES + gqa * rows * (dk + dv) * BF16_BYTES)
        wu_write = float(gqa * rows * (dk + dv) * BF16_BYTES)
        l0_bytes = float((2 * bt * dk + gqa * 2 * (bt * bt + bt * dk + bt * dv)) * BF16_BYTES)
        flops = float(2 * bt * bt * dk + gqa * (2 * bt * dk * bt + 2 * bt * dv * bt))
        aic = Work(
            matrix={"bf16": flops},
            resources={"hbm": aic_read + wu_write, "mte2": aic_read,
                       "mte1": l0_bytes, "fixpipe": wu_write},
        )
        return aiv, aic

    return _build_mixed_plan(
        geometry, cores, _tile_spans(cores, chunks.total * keys), group_work,
        aiv_first=True, slots=PIPELINE_WINDOW_SLOTS,
        first_label="aiv_vcs32", second_label="aic_wy32",
        extra_meta={"gqa_ratio": gqa},
    )


def make_state_op(params: Mapping[str, Any]) -> OpSpec:
    lengths, heads, keys, dk, dv = _validate_chunk32_params(params)
    chunks = PackedChunks.from_lengths(lengths, GDN_CHUNK32_TOKENS)
    gate_dtype = str(params.get("gate_dtype", "fp32")).lower()
    use_init = bool(params.get("use_initial_state", True))
    out_final = bool(params.get("output_final_state", True))
    state_v_first = bool(params.get("state_v_first", True))
    shape = (len(lengths), heads, dv, dk) if state_v_first else (len(lengths), heads, dk, dv)
    meta = {
        "implementation": STATE_RECIPE, "lengths": lengths, "heads": heads,
        "key_heads": keys, "dk": dk, "dv": dv, "head_dim_k": dk, "head_dim_v": dv,
        "chunk": GDN_CHUNK32_TOKENS, "dtype": "bf16", "gate_dtype": gate_dtype,
        "use_initial_state": use_init, "output_final_state": out_final,
        "state_v_first": state_v_first, "use_exp2": False,
        "dispatch": {
            "implementation": STATE_RECIPE, "state_v_first": state_v_first,
            "use_exp2": False, "gate_dtype": gate_dtype,
            "use_initial_state": use_init, "output_final_state": out_final,
        },
        "native_source": FLA_GDN_CHUNK_SOURCE,
    }
    bt = GDN_CHUNK32_TOKENS
    tokens = sum(lengths)
    seq_chunks = tuple((n + bt - 1) // bt for n in lengths)
    state_fp32, state_bf16 = float(dk * dv * FP32_BYTES), float(dk * dv * BF16_BYTES)
    total_snaps = sum((1 if use_init else 0) + (nc - 1) for nc in seq_chunks)
    total_aic_states = sum((nc if use_init else nc - 1) for nc in seq_chunks)
    aiv_read = (len(lengths) * heads * state_fp32 if use_init else 0.0) + float(heads * tokens * (dv * BF16_BYTES + FP32_BYTES))
    aiv_write = (len(lengths) * heads * state_fp32 if out_final else 0.0) + total_snaps * heads * state_bf16 + float(heads * tokens * dv * BF16_BYTES)
    aiv_vec = float(heads * (chunks.total * dk * dv + tokens * dv) * STATE_DECAY_OPS_PER_ELEM)
    aic_read = float((keys * tokens * dk + heads * tokens * (dk + dv)) * BF16_BYTES) + total_aic_states * heads * state_bf16
    flops = float(chunks.total * heads * 2 * (2 * bt * dv * dk))
    return OpSpec(
        op_id=f"gdn-state-c32:h{heads}:c{chunks.total}",
        family="gdn_state", phase="prefill", stage=0, layer=0, requests=(),
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        flops=flops, matrix_flops=(("bf16", flops),), vector_ops=aiv_vec,
        hbm_bytes=aiv_read + aiv_write + aic_read,
        mte2_bytes=aiv_read + aic_read, mte3_bytes=aiv_write,
        dtype="fp32", recipe=STATE_RECIPE, source_boundary="kernel",
        metadata=tuple(sorted(meta.items())),
    )


def build_state(op: OpSpec, geometry: Geometry, *, ctx: OperatorPlanningContext | None = None) -> OperatorPlan:
    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in meta["lengths"])
    heads, keys, dk, dv = int(meta["heads"]), int(meta["key_heads"]), int(meta["dk"]), int(meta["dv"])
    bt, use_init, out_final = GDN_CHUNK32_TOKENS, bool(meta.get("use_initial_state", True)), bool(meta.get("output_final_state", True))
    cores = _active_cores(geometry, ctx)
    group = max(1, min(MAX_TASK_GROUP_HEADS, heads // keys, (len(lengths) * heads + len(cores) - 1) // len(cores)))
    groups_per_key = (heads // keys + group - 1) // group
    chunks = PackedChunks.from_lengths(lengths, bt)
    state_fp32, state_bf16 = float(dk * dv * FP32_BYTES), float(dk * dv * BF16_BYTES)

    def group_work(item: tuple[int, int]) -> tuple[Work, Work]:
        chunk_idx, sub_idx = item
        active_h = min(group, heads // keys - sub_idx * group)
        seq_idx, chunk_local, _, rows = chunks.range(chunk_idx)
        first_chunk, last_chunk = chunk_local == 0, chunk_local == (lengths[seq_idx] + bt - 1) // bt - 1
        init_read = active_h * state_fp32 if (first_chunk and use_init) else 0.0
        final_write = active_h * state_fp32 if (last_chunk and out_final) else 0.0
        snap_count = (1 if (first_chunk and use_init) else 0) + (1 if not last_chunk else 0)
        aiv_read = init_read + float(active_h * rows * (dv * BF16_BYTES + FP32_BYTES))
        aiv_write = final_write + snap_count * active_h * state_bf16 + float(active_h * rows * dv * BF16_BYTES)
        aiv_vec = float(active_h * (dk * dv + rows * dv) * STATE_DECAY_OPS_PER_ELEM)
        aiv = Work(resources={"hbm": aiv_read + aiv_write, "mte2": aiv_read, "vector": aiv_vec, "mte3": aiv_write})
        aic_read = float((rows * dk + active_h * rows * (dk + dv)) * BF16_BYTES + (1 if (use_init or not first_chunk) else 0) * active_h * state_bf16)
        l0_bytes, flops = float(active_h * 2 * (bt * dk + dk * dv + bt * dv) * BF16_BYTES), float(active_h * 2 * (2 * bt * dv * dk))
        return aiv, Work(matrix={"bf16": flops}, resources={"hbm": aic_read, "mte2": aic_read, "mte1": l0_bytes})

    chunk_starts = tuple(accumulate(((n + bt - 1) // bt for n in lengths), initial=0))
    chains_per_sequence = keys * groups_per_key

    def state_tiles(chains: range) -> tuple[tuple[int, int], ...]:
        tiles = []
        for chain in chains:
            sequence, head_group = divmod(chain, chains_per_sequence)
            sub_group = head_group % groups_per_key
            tiles.extend((chunk, sub_group) for chunk in range(chunk_starts[sequence], chunk_starts[sequence + 1]))
        return tuple(tiles)

    return _build_mixed_plan(
        geometry, cores, tuple(state_tiles(span) for span in _tile_spans(cores, len(lengths) * chains_per_sequence)),
        group_work,
        aiv_first=False, slots=RECURRENCE_SERIAL_SLOTS,
        recurrence_sync=True,
        first_label="state_aic", second_label="state_aiv", extra_meta={"task_group_size": group},
    )


def make_output_op(params: Mapping[str, Any]) -> OpSpec:
    lengths, heads, keys, dk, dv = _validate_chunk32_params(params)
    gate_dtype = str(params.get("gate_dtype", "fp32")).lower()
    state_v_first = bool(params.get("state_v_first", True))
    group = min(MAX_TASK_GROUP_HEADS, heads // keys)
    shape = (1, sum(lengths), heads, dv)
    meta = {
        "implementation": OUTPUT_RECIPE, "lengths": lengths, "heads": heads,
        "key_heads": keys, "dk": dk, "dv": dv, "head_dim_k": dk, "head_dim_v": dv,
        "chunk": GDN_CHUNK32_TOKENS, "dtype": "bf16", "gate_dtype": gate_dtype,
        "task_group_size": group, "use_exp2": False,
        "state_v_first": state_v_first,
        "dispatch": {
            "implementation": OUTPUT_RECIPE, "output_token_first": True,
            "use_exp2": False, "state_v_first": state_v_first,
            "gate_dtype": gate_dtype, "task_group_size": group,
        },
        "native_source": FLA_GDN_CHUNK_SOURCE,
    }
    bt = GDN_CHUNK32_TOKENS
    chunks = PackedChunks.from_lengths(lengths, bt)
    tokens = sum(lengths)
    groups_per_key = (heads // keys + group - 1) // group
    aic_read = float((keys * groups_per_key * 2 * tokens * dk + heads * (chunks.total * dk * dv + tokens * dv)) * BF16_BYTES)
    flops = float(heads * (chunks.total * (2 * bt * bt * dk + 2 * bt * dv * dk) + tokens * 2 * dv * bt))
    aiv_read = float(heads * tokens * FP32_BYTES)
    aiv_write = float(heads * tokens * dv * BF16_BYTES)
    aiv_vec = float(heads * (chunks.total * bt * bt * CAUSAL_DECAY_OPS_PER_PAIR + tokens * dv * OUTPUT_FUSE_OPS_PER_ELEM))
    return OpSpec(
        op_id="gdn_output_c32", family="gdn_output", phase="prefill",
        stage=0, layer=0, requests=(), dtype="bf16", layout="BSND",
        flops=flops, matrix_flops=(("bf16", flops),), vector_ops=aiv_vec,
        hbm_bytes=aic_read + aiv_read + aiv_write,
        mte2_bytes=aic_read + aiv_read, mte3_bytes=aiv_write,
        recipe=OUTPUT_RECIPE, source_boundary="kernel",
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        metadata=tuple(sorted(meta.items())),
    )


def build_output(op: OpSpec, geometry: Geometry, *, ctx: OperatorPlanningContext | None = None) -> OperatorPlan:
    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in meta["lengths"])
    heads, keys = int(meta["heads"]), int(meta["key_heads"])
    dk, dv, group = int(meta["dk"]), int(meta["dv"]), int(meta["task_group_size"])
    bt = GDN_CHUNK32_TOKENS
    chunks = PackedChunks.from_lengths(lengths, bt)
    groups_per_key = (heads // keys + group - 1) // group
    total_groups = chunks.total * keys * groups_per_key
    cores = _active_cores(geometry, ctx)

    def group_work(gid: int) -> tuple[Work, Work]:
        chunk_idx = gid // (keys * groups_per_key)
        sub_idx = (gid % (keys * groups_per_key)) % groups_per_key
        active_h = min(group, heads // keys - sub_idx * group)
        _, _, _, rows = chunks.range(chunk_idx)
        aic_read = float((2 * rows * dk + active_h * (dk * dv + rows * dv)) * BF16_BYTES)
        l0_bytes = float(active_h * (2 * bt * dk + dk * dv + bt * bt + bt * dv) * BF16_BYTES)
        flops = float(active_h * (2 * bt * bt * dk + 2 * bt * dv * dk + 2 * rows * dv * bt))
        aic = Work(matrix={"bf16": flops},
                   resources={"hbm": aic_read, "mte2": aic_read, "mte1": l0_bytes})
        aiv_read = float(active_h * rows * FP32_BYTES)
        aiv_write = float(active_h * rows * dv * BF16_BYTES)
        aiv_vec = float(active_h * (
            bt * bt * CAUSAL_DECAY_OPS_PER_PAIR + rows * dv * OUTPUT_FUSE_OPS_PER_ELEM
        ))
        aiv = Work(resources={
            "hbm": aiv_read + aiv_write, "mte2": aiv_read,
            "vector": aiv_vec, "mte3": aiv_write,
        })
        return aiv, aic

    return _build_mixed_plan(
        geometry, cores, _tile_spans(cores, total_groups), group_work,
        aiv_first=False, slots=PIPELINE_WINDOW_SLOTS,
        first_label="out_aic", second_label="out_aiv",
        extra_meta={"task_group_size": group},
    )


@dataclass(frozen=True)
class _KernelNamespace:
    make_op: Any
    build: Any


PREPARE = _KernelNamespace(make_prepare_op, build_prepare)
STATE = _KernelNamespace(make_state_op, build_state)
OUTPUT = _KernelNamespace(make_output_op, build_output)

_BUILDERS = {PREPARE_RECIPE: build_prepare, STATE_RECIPE: build_state, OUTPUT_RECIPE: build_output}


def build(op: OpSpec, geometry: Geometry, *, ctx: OperatorPlanningContext | None = None) -> OperatorPlan:
    return _BUILDERS[op.recipe](op, geometry, ctx=ctx)


def recipes() -> dict[str, PlannedRecipe]:
    """Registered ``PlannedRecipe``s for the three ``Chunk32`` GDN kernels."""
    return {recipe_id: PlannedRecipe(builder) for recipe_id, builder in _BUILDERS.items()}
