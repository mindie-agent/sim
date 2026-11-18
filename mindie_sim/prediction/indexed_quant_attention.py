"""Explicit packed-cache INT8 sparse MLA adaptation on the shared pipeline.

[Role] Expand declared per-128 FP32-scaled INT8 rows before BF16 QK/PV.
[Invariants] K and V alias one packed cache. A row contains 512 INT8 values,
  four FP32 scales and zero or 64 BF16 rotary values. UB padding is local.
  The arch22 source admits rope64; rope0 is an analytic adaptation. Query
  ownership spans the batch, never a separate whole-call charge per sequence.
[Timeouts & Locks] Pure finite phase planning; common bounded query pipeline.
[Side-Effect Policy] No captures, timing fits, hardware operations or I/O.
"""
from dataclasses import replace
from functools import lru_cache, partial

from mindie_sim.schema.common import ModelingError
from .indexed_attention import (
    _align, bind_sustained_dram, build_pipeline, owner_row_tiles, _phases,
)
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage


VECTOR_BUFFER_BYTES = (64 + 32 + 32 + 4 + 32 + 8 + 8) * 1024 + (3 * 3 + 2) * 512


SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "caller": "vllm_ascend/attention/utils.py:69-80; attention/sfa_v1.py:1215-1276",
    "phase_reference": "csrc/attention/kv_quant_sparse_flash_attention/op_kernel/"
                       "kv_quant_sparse_flash_attention_service_vector_mla.h:652-851",
    "phase_sha256": "3eed5ab746e056319bf873a488a3306cffbabe8dd19f70294da85685f4e3cfd3",
    "scope": "arch22_packed_latent512_rope64; rope0_is_analytic_adaptation",
    "capture_binary": "not_established",
}
#: CANN 9.2.1 ``arch35`` sparse scheduler on the same service template.
ARCH35_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "cann-9.2.1-arch35-sfa",
    "caller": "torch_npu.npu_kv_quant_sparse_flash_attention(tile_size=128)",
    "phase_reference": "op_kernel/"
                       "kv_quant_sparse_flash_attention_service_vector_mla.h:652-851",
    "scope": "arch35_c8_tile128_656B_row; sparse_block_size1; one_1c2v_group_per_query_row",
    "capture_binary": "not_established",
}
#: Explicit schedule names; the arch22 512-key three-slot review stays the default.
SCHEDULE_ARCH22 = "arch22_key512_slots3"
SCHEDULE_ARCH35 = "arch35_c8_tile128_slots2"
#: Declared packed row: 512 INT8 payload, four FP32 scales, 64 BF16 rotary lanes.
C8_PAYLOAD_BYTES = 512
C8_SCALE_BYTES = 16
C8_SCALE_GROUP = 128
C8_PACKED_ROW_BYTES = C8_PAYLOAD_BYTES + C8_SCALE_BYTES + 64 * 2
#: Declared row alignment and the bounded pure-phase caches.
ROW_ALIGNMENT_BYTES = 32
C8_TILE_CACHE_SIZE = 4096
OWNER_TILE_CACHE_SIZE = 8
#: The caller passes one key cache and one value cache that alias those bytes.
C8_ALIAS_READS = 2
#: ``sparse_indices -> block_table`` resolves two dependent INT32 scalar reads.
C8_ADDRESS_LOOKUPS = 2
C8_ADDRESS_ENTRY_BYTES = 4
#: ``arch35`` C8 key tile and the declared two KV/accumulator slots per owner.
C8_KEY_TILE = 128
C8_TILE_SLOTS = 2
#: C8 INT8 -> FP16 -> FP32 * scale -> BF16 -> ND-to-NZ vector service.
#: Weights are active lanes per value element of one declared source call; the
#: rotary lanes are copied, not dequantized.
C8_DEQUANT_PASSES = (
    ("int8_unpack", 2), ("int8_widen", 2), ("int8_to_fp16", 1),
    ("fp16_to_fp32", 1), ("scale_broadcast", 1), ("fp32_scale_mul", 1),
    ("fp32_to_bf16", 1), ("nz_lane_permute", 2),
)
#: Online softmax per score pair: exponent, running max, running sum and the
#: rescale fold.  The output rescale is the accumulator lane pass, not a repeat.
C8_SOFTMAX_PAIR_OPS = 4
C8_SOFTMAX_ROW_OPS = 26
C8_ACCUM_FIRST_OPS = 1
C8_ACCUM_PASS_OPS = 2
C8_FP32_BYTES = 4
C8_BF16_BYTES = 2
C8_HEAD_ALIGN = 16
#: Source scalar/softmax state per owner: FP32 sinks, two accumulators, scores.
C8_UB_PER_OWNER = 4 * 128 * C8_FP32_BYTES + 2 * C8_HEAD_ALIGN * 512 * C8_FP32_BYTES


def packed_row_bytes(rope):
    """One packed C8 row: INT8 payload, FP32 group scales and BF16 rotary."""
    return C8_PAYLOAD_BYTES + C8_SCALE_BYTES + int(rope) * C8_BF16_BYTES


def packed_cache_declaration(rope):
    """Declared packed row layout shared by both reviewed schedules."""
    row_bytes = packed_row_bytes(rope)
    return {
        "payload_bytes": C8_PAYLOAD_BYTES, "scale_bytes": C8_SCALE_BYTES,
        "scale_dtype": "fp32", "group_size": C8_SCALE_GROUP,
        "rotary_bytes": int(rope) * C8_BF16_BYTES, "rotary_dtype": "bf16",
        "row_bytes": row_bytes,
        "local_padded_row_bytes": -(-row_bytes // ROW_ALIGNMENT_BYTES)
        * ROW_ALIGNMENT_BYTES,
        "key_value_alias": True,
    }


def _dequant_lane_ops(latent):
    return sum(weight for _name, weight in C8_DEQUANT_PASSES) * latent


def recipe(*, scratch_policy, compact=True, operand_modes=True,
           schedule=SCHEDULE_ARCH22, tile_size=None, slots=None):
    """Select explicitly; this recipe never changes ordinary BF16 defaults."""
    if schedule == SCHEDULE_ARCH35 and tile_size is None:
        tile_size = C8_KEY_TILE
    if schedule == SCHEDULE_ARCH35 and slots is None:
        slots = C8_TILE_SLOTS
    return PlannedRecipe(partial(build, scratch_policy=scratch_policy,
                                 compact=compact, operand_modes=operand_modes,
                                 schedule=schedule, tile_size=tile_size, slots=slots))


def phases(keys, heads, latent, rope, *, first, last, scratch_policy,
           operand_modes=False):
    """Alias-aware gather plus element work of the declared conversion chain."""
    common = _phases(keys, heads, latent, rope, first=first, last=last,
                     scratch_policy=scratch_policy, operand_modes=operand_modes)
    # DataCopyPad transfers valid bytes, pads only the local destination. The
    # retained kernel's arbitrary-index/page reads are accounted as two INT32
    # fields per selected row; dependent scalar response remains unmodeled.
    packed_bytes = keys * (latent + latent // 128 * 4 + rope * 2)
    gather = Work(resources={"hbm": packed_bytes + keys * 8,
                             "mte2_aiv": packed_bytes + keys * 8})
    # INT8 -> FP16 -> FP32, multiply scale, FP32 -> BF16, ND -> NZ copy.
    # A scale is broadcast to eight FP32 lanes before repeated vector Mul.
    # These are typed element service counts, not a measured instruction CPI.
    # Rotary data is already BF16. Its source Copy to NZ adds a lane move,
    # not an INT8 dequantization pass.
    convert = Work(resources={"vector": keys * (5 * latent + latent // 128 * 8 + rope)})
    return (("gather", gather), ("dequantize", convert), *common[1:])


def phase_unit_weights(name, key, rope=0):
    if name not in {"gather", "dequantize", "publish"}:
        return {}
    keys = key[0]
    # MergeKv assigns pairs: ceil(ceil(rows/2)/2) pairs to AIV0.
    first = min(keys, ((keys + 3) // 4) * 2)
    if name == "publish":
        # Each paired AIV publishes its own valid rows plus one 512-byte
        # valid-size descriptor, including the sibling with no KV payload.
        width = 512 + rope
        total = keys * width * 2 + 1024
        return {"mte3": ((first * width * 2 + 512) / total,
                         ((keys - first) * width * 2 + 512) / total)}
    role = "mte2_aiv" if name == "gather" else "vector"
    return {role: (first / keys, (keys - first) / keys)}


def phase_unit_weights_rope64(name, key):
    return phase_unit_weights(name, key, rope=64)


def build(op, geometry, *, scratch_policy, compact=True, operand_modes=True,
          schedule=SCHEDULE_ARCH22, tile_size=None, slots=None):
    shape = dict(op.metadata).get("indexed_attention_geometry")
    if (not shape or shape["query_dtype"] != "bf16" or shape["cache_dtype"] != "int8"
            or shape["latent"] != 512 or shape["rope"] not in {0, 64}
            or shape["window"] or shape["compression_ratio"] != 1 or shape["cp"] != 1
            or shape.get("sparse_block_size", 1) != 1 or not 1 <= shape["heads"] <= 64):
        raise ModelingError("packed_int8_indexed_attention_requires_bf16_q_int8_latent512_rope0_or64")
    if scratch_policy not in {"resident", "offchip"}:
        raise ModelingError("indexed_attention_scratch_policy_required")
    if any(geometry.resource_units.get(role) != geometry.compute_units * ratio
           for role, ratio in (("vector", 2), ("mte2_aiv", 2), ("mte3", 2),
                               ("mte2_aic", 1), ("mte1", 1), ("fixpipe", 1))):
        raise ModelingError("packed_int8_indexed_attention_requires_paired_aic_aiv_pipes")
    if (geometry.buffer_units.get("ub") != geometry.compute_units * 2
            or geometry.buffers.get("ub", 0) < VECTOR_BUFFER_BYTES):
        raise ModelingError("packed_int8_indexed_attention_requires_declared_vector_scratch")
    if schedule == SCHEDULE_ARCH35:
        return arch35_plan(shape, geometry, tile_size=tile_size, slots=slots)
    if schedule != SCHEDULE_ARCH22:
        raise ModelingError("packed_int8_indexed_attention_schedule_unknown")
    plan = build_pipeline(
        shape, geometry, scratch_policy=scratch_policy, compact=compact,
        operand_modes=operand_modes, phase_factory=phases,
        source_items=tuple(SOURCE.items()), paired_vector_stages=("dequantize",),
        vector_buffer_bytes=VECTOR_BUFFER_BYTES,
        phase_unit_weights=phase_unit_weights_rope64 if shape["rope"] else phase_unit_weights)
    row_bytes = packed_row_bytes(shape["rope"])
    return replace(plan, metadata={
        "kind": "indexed_attention_packed_int8",
        "coverage": "source_phases_declared_dispatch" if shape["rope"] else "analytic_adaptation",
        "source_execution_domain": "mixed", "source": SOURCE,
        "packed_cache": {**packed_cache_declaration(shape["rope"]),
                         "row_bytes": row_bytes,
                         "local_padded_row_bytes": -(-row_bytes // ROW_ALIGNMENT_BYTES)
                         * ROW_ALIGNMENT_BYTES},
        "kernel_schedule": SCHEDULE_ARCH22,
        "vector_buffer_bytes_per_aiv": VECTOR_BUFFER_BYTES,
        "buffer_basis": "retained_InitBuffers_preload3; no_rope_adaptation_keeps_allocation",
        "unmodeled_features": [
            "captured_kernel_dispatch_and_binary", "captured_valid_index_values",
            "scalar_index_and_page_lookup_response", "cross_core_flag_latency",
            "external_cache_eviction", "conversion_instruction_issue_rate"],
        "abstraction": "shared_finite_pipeline_with_fluid_compact_contention",
    })


def arch35_tile_phases(keys, heads, latent, rope, *, first, last, private_hbm=0):
    """One ``arch35`` C8 key tile: address, gather/dequant, QK, PV, accumulate.

    The ``2-AIV`` ``index/page lookup -> packed-row gather -> C8 dequant + NZ
    publish`` block is one source subtile loop over a double-buffered UB slot,
    so its steady service is the busiest of the copy, vector and publish pipes.
    The AIC ``QK -> Fixpipe -> online softmax -> PV`` chain stays explicit
    because every step consumes the result of the previous one.
    """
    padded_h = _align(heads, C8_HEAD_ALIGN)
    width = latent + rope
    row_bytes = packed_row_bytes(rope)
    first_size = heads * width * C8_BF16_BYTES if first else 0
    output_bytes = heads * latent * C8_BF16_BYTES if last else 0
    # QK and PV are each one matrix-pipeline unit: MTE1 operand loads, the cube
    # MMA and the Fixpipe result write overlap inside the unit, so the steady
    # service of the unit is its busiest declared pipe, never their sum.
    qk_work = Work(matrix={"bf16": 2 * padded_h * keys * width}, resources={
        "mte1": (padded_h + keys) * width * C8_BF16_BYTES,
        "fixpipe": padded_h * keys * C8_FP32_BYTES})
    pv_work = Work(matrix={"bf16": 2 * padded_h * keys * latent}, resources={
        "mte1": (padded_h * keys + keys * latent) * C8_BF16_BYTES,
        "fixpipe": padded_h * latent * C8_FP32_BYTES})
    return (
        ("kv_stage", Work(resources={
            # One aliased KV row plus the two dependent INT32 scalar lookups.
            "mte2_aiv": keys * (row_bytes
                                + C8_ADDRESS_LOOKUPS * C8_ADDRESS_ENTRY_BYTES),
            "vector": keys * _dequant_lane_ops(latent),
            "mte3": keys * width * C8_BF16_BYTES * C8_ALIAS_READS,
            "hbm": private_hbm}), ()),
        ("query", Work(resources={"hbm": first_size, "mte2_aic": first_size}), ()),
        ("qk", qk_work, ("kv_stage", "query") if first else ("kv_stage",)),
        ("softmax", Work(resources={
            "vector": C8_SOFTMAX_PAIR_OPS * heads * keys
                      + C8_SOFTMAX_ROW_OPS * heads}), ("qk",)),
        ("pv", pv_work, ("softmax",)),
        ("accumulate", Work(resources={"vector": heads * latent * (
            C8_ACCUM_FIRST_OPS if first else C8_ACCUM_PASS_OPS)}), ("pv",)),
        ("output", Work(resources={
            "hbm": output_bytes, "mte3": output_bytes}), ("accumulate",)),
    )


@lru_cache(maxsize=C8_TILE_CACHE_SIZE)
def cached_tile(keys, heads, latent, rope, first, last, private_hbm=0):
    """Bounded cache of the pure per-tile phase payloads."""
    return arch35_tile_phases(keys, heads, latent, rope, first=first, last=last,
                              private_hbm=private_hbm)


def tile_stream(sequences, topk, compute_units, tile_size, pool_size, pool_tail):
    """One finite owner -> query row -> key tile stream for every owner."""
    rows = sum(query for query, _ in sequences)
    owners = min(rows, compute_units)
    return owners, tuple(
        tuple(owner_row_tiles(owner, sequences, compute_units, topk, tile_size,
                              pool_size=pool_size, pool_tail=pool_tail))
        for owner in range(owners))


def _row_partition(shape, sequences, topk):
    from mindie_sim.catalog.index_pool import selected_tokens
    from .batch_prefix_attention import (
        nested_prefix_union, prefix_members, selected_prefix_count,
    )
    pool = shape.get("index_pool_size", 1)
    tail = shape.get("index_pool_tail", False)
    by_key, private, maximum = {}, [], 0
    for (query, context), (key, extent) in zip(sequences, prefix_members(shape, sequences)):
        row_counts = []
        for offset in range(query):
            visible = context + offset + 1
            selected = selected_tokens(visible, topk, pool, tail)
            # The calibrated estimate samples historical complete pools. Any
            # selected causal tail is a known interval, including its identity.
            history = visible // pool
            picked = min(history, topk // pool)
            prefix_pools, rest = divmod(extent, pool)
            population = context // pool
            whole = selected_prefix_count(picked, prefix_pools, history, population)
            # A prefix ending inside a complete pool can supply only its
            # overlapping lanes. Select it if the declared history forces it.
            partial = (rest if prefix_pools < history and picked == history else 0)
            suffix = (max(0, extent - history * pool) if tail else 0)
            shared = whole * pool + partial + suffix
            if shared:
                by_key.setdefault(key, []).append((extent, shared))
            maximum = max(maximum, shared)
            row_counts.append(selected - shared)
        private.append(tuple(row_counts))
    return sum(nested_prefix_union(g) for g in by_key.values()), tuple(private), maximum


def prefix_partition(shape, sequences, topk, *, sequence_keys=False):
    shared, private, maximum = _row_partition(shape, sequences, topk)
    result = shared, sum(map(sum, private)), maximum
    if sequence_keys:
        return (*result, tuple(max(counts, default=0) for counts in private))
    return result


def _priced_streams(streams, sequences, units, private):
    from .indexed_attention import sequence_query_positions
    result = []
    for owner, parts in enumerate(streams):
        priced = []
        for (sequence, pos), (_, tiles) in zip(
                sequence_query_positions(owner, sequences, units), parts, strict=True):
            selected = sum(k for k, _, _ in tiles)
            count = private[sequence][pos]
            start = 0
            for keys, first, last in tiles:
                stop = start + keys
                owned = stop * count // selected - start * count // selected
                priced.append((keys, first, last, owned))
                start = stop
        result.append(tuple(priced))
    return tuple(result)


def arch35_plan(shape, geometry, *, tile_size=None, slots=None):
    """Declared ``arch35`` C8 schedule: one ``1c2v`` group per query row."""
    tile_size = C8_KEY_TILE if tile_size is None else tile_size
    slots = C8_TILE_SLOTS if slots is None else slots
    if tile_size != C8_KEY_TILE or slots != C8_TILE_SLOTS:
        raise ModelingError("arch35_c8_requires_tile128_and_two_slots")
    sequences = tuple(tuple(row) for row in shape.get(
        "sequences", ((shape.get("query", 0), shape.get("context", 0)),)))
    heads, latent, rope = shape["heads"], shape["latent"], shape["rope"]
    topk, paired = shape["topk"], max(1, int(shape.get("paired_calls", 1)))
    compute_units = geometry.compute_units
    _owners, streams = tile_stream(
        sequences, topk, compute_units, tile_size,
        shape.get("index_pool_size", 1), shape.get("index_pool_tail", False))
    kv_slot_bytes = tile_size * (latent + rope) * C8_BF16_BYTES
    accumulator_bytes = _align(heads, C8_HEAD_ALIGN) * latent * C8_FP32_BYTES
    if (geometry.buffer_units.get("l1", 0) < compute_units
            or geometry.buffer_units.get("l0c", 0) < compute_units
            or C8_TILE_SLOTS * kv_slot_bytes > geometry.buffers.get("l1", 0)
            or C8_TILE_SLOTS * accumulator_bytes > geometry.buffers.get("l0c", 0)):
        raise ModelingError("arch35_c8_requires_two_kv_and_accumulator_slots_per_owner")
    shared_rows, private, _ = _row_partition(shape, sequences, topk)
    private_rows = sum(map(sum, private))
    streams = _priced_streams(streams, sequences, compute_units, private)
    external = shared_rows * packed_row_bytes(rope)
    from collections import Counter
    counts = Counter(tile for parts in streams for tile in parts)
    call_work = Work()
    for tile, count in counts.items():
        for _name, work, _after in _tile_phases(tile, heads, latent, rope):
            call_work += work.scaled(count)
    fill = Work(resources={"hbm": external} if external else {})
    stats_bytes = int(shape.get("statistics_bytes", 0) or 0)
    extra = stats_bytes + int(shape.get("padding_output_bytes", 0))
    stats_work = Work(resources={"hbm": extra, "mte3": extra}) if extra else Work()
    shape = {**shape, "waves": -(-sum(q for q, _ in sequences) // compute_units)}
    return OperatorPlan(
        call_work.scaled(paired) + fill + stats_work,
        _arch35_program(streams, heads, latent, rope, kv_slot_bytes,
                        accumulator_bytes, fill, paired,
                        stats_work),
        _arch35_metadata(shape, external, private_rows, sequences, tile_size,
                         slots, paired))


def _tile_phases(key, heads, latent, rope):
    keys, first, last, private = key
    return cached_tile(keys, heads, latent, rope, first, last,
                       private * packed_row_bytes(rope))


def _arch35_program(streams, heads, latent, rope, kv_slot_bytes, accumulator_bytes,
                    fill, paired, stats_work):
    """Cold prefix fill once, then the declared paired query-owner programs."""
    def program(context):
        context = bind_sustained_dram(context)

        def one(owner, parts):
            flat = parts
            allocations = tuple(
                [context.allocate(f"c8kv{owner}_{index}", "l1", kv_slot_bytes, (owner,))
                 for index in range(C8_TILE_SLOTS)]
                + [context.allocate(f"c8acc{owner}_{index}", "l0c", accumulator_bytes,
                                    (owner,))
                   for index in range(C8_TILE_SLOTS)])

            @lru_cache(maxsize=OWNER_TILE_CACHE_SIZE)
            def stages(key):
                result = []
                for name, work, after in _tile_phases(key, heads, latent, rope):
                    roles = set(work.resources) | ({"cube"} if work.matrix else set())
                    # The two declared KV slots bound the live tiles; inside a
                    # row the online-softmax accumulator is a strict recurrence.
                    previous = (("accumulate",) if name == "accumulate" and not key[1]
                                else ())
                    result.append(Stage(
                        name, context.bind(work), after=tuple(after),
                        previous=previous,
                        placement=context.owned(
                            owner, context.core_count, roles - {"hbm"})))
                return tuple(result)

            return KernelProgram((
                Stage("reserve", acquire=allocations),
                Stage("tiles", after=("reserve",), program=PipelineProgram(
                    len(flat), C8_TILE_SLOTS, lambda index: stages(flat[index]))),
                Stage("release", after=("tiles",), free=allocations)))

        call = ParallelProgram(tuple(one(owner, parts)
                                     for owner, parts in enumerate(streams)))
        calls = tuple(Stage(f"call{index}", after=(
            ("prefix_fill",) if index == 0 else (f"call{index - 1}",)), program=call)
            for index in range(paired))
        stages = (Stage("prefix_fill", work=context.bind(fill)), *calls)
        if stats_work.resources:
            stages = (*stages, Stage(
                "lse_statistics", work=context.bind(stats_work),
                after=(f"call{paired - 1}",)))
        return KernelProgram(stages)

    return program


def _arch35_metadata(shape, external, private_rows, sequences, tile_size, slots,
                     paired):
    return {
        "kind": "indexed_attention_packed_int8",
        "coverage": "source_phases_declared_dispatch",
        "source_execution_domain": "mixed", "source": ARCH35_SOURCE,
        "kernel_schedule": SCHEDULE_ARCH35,
        "packed_cache": packed_cache_declaration(shape["rope"]),
        "c8_schedule": {
            "key_tile": tile_size, "tile_slots": slots,
            "query_rows": sum(query for query, _ in sequences),
            "waves": shape["waves"],
            "head_tile": _align(shape["heads"], C8_HEAD_ALIGN),
            "dequant_lane_ops_per_value": sum(
                weight for _name, weight in C8_DEQUANT_PASSES),
            "external_fill_bytes": external, "private_rows": private_rows,
            "paired_calls": paired,
            "key_tile_owners": "one_1c2v_group_per_query_row",
        },
        "unmodeled_features": [
            "captured_kernel_dispatch_and_binary", "captured_valid_index_values",
            "cross_core_flag_latency", "conversion_instruction_issue_rate"],
        "abstraction": "finite_tile_pipeline_with_sustained_dram_and_l2_prefix_reuse",
    }
