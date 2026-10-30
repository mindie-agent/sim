"""Compact indexed-attention gather and matrix/vector pipelines.

[Role] Price query-specific sparse MLA phases on a shared batch owner grid.
[Invariants] Query indices are independent. Cache-resident scratch still consumes
  copy service and finite cache capacity; it is not another HBM transaction.
  Matrix padding, QK/softmax/PV ordering and finite producer slots are explicit.
  The finite stage recurrence is compressed, not an instruction simulator.
[Timeouts & Locks] O(query * key_tiles) arithmetic and O(owners + slots) state.
[Side-Effect Policy] Pure plans and shared Engine submissions; no observed times.
"""
from collections import Counter, OrderedDict
from bisect import bisect_right
from dataclasses import replace
from functools import lru_cache, partial

from mindie_sim.schema.common import ModelingError
from mindie_sim.catalog.index_pool import selected_tokens
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .sparse_attention import selected_query_owners
from .program import KernelProgram, ParallelProgram, Stage


SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8fd0bc4885cb52e58791aaa8efacc571b9bd115d",
    "branch": "arch22 sparse_flash_attention MLA V_TEMPLATE",
    "scope": "BF16, single latent KV head, per-query indices, no dense window",
}
IMPLEMENTATION = "ascend_sparse_flash_attention_arch22"


def recipe(*, scratch_policy, key_tile=512, slots=3, compact=True,
           operand_modes=False, rope0_analytic=False):
    return PlannedRecipe(partial(
        build, scratch_policy=scratch_policy, key_tile=key_tile,
        slots=slots, compact=compact, operand_modes=operand_modes,
        rope0_analytic=rope0_analytic))


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(_selected_build)}


def _selected_build(op, geometry):
    sequences = dict(op.metadata).get("indexed_attention_geometry", {}).get("sequences", ())
    if len(sequences) > 1:
        raise ModelingError("indexed_attention_requires_single_contiguous_sequence")
    if any(geometry.resource_units.get(role) != geometry.compute_units * ratio
           for role, ratio in (("vector", 2), ("mte2_aiv", 2), ("mte3", 2),
                               ("mte2_aic", 1), ("mte1", 1), ("fixpipe", 1))):
        raise ModelingError("indexed_attention_source_requires_paired_aic_aiv_pipes")
    return build(op, geometry, operand_modes=True,
                 scratch_policy=dict(op.metadata)["indexed_source_options"]["scratch_policy"])


def lower_op(op, *, selection, view, views, plan, parallel):
    """Bind the declared TND branch without inferring a captured binary."""
    policy = selection.options.get("scratch_policy", "offchip")
    if policy not in {"resident", "offchip"}:
        raise ModelingError("indexed_attention_scratch_policy_required")
    if (len(views) != 1 or parallel.cp != 1 or parallel.pcp != 1
            or view.draft_proposal_tokens
            or (view.query_ranges and view.query_ranges != ((0, view.q_len),))):
        raise ModelingError("indexed_attention_requires_single_contiguous_sequence")
    if (plan.cache_kind != "dense" or plan.mla_variant != "latent_mla"
            or not plan.index_topk or plan.noncausal_attention):
        raise ModelingError("indexed_attention_requires_causal_dense_latent_selection")
    metadata = {**dict(op.metadata), "indexed_source_options": {"scratch_policy": policy},
                "unmodeled_source_mechanisms": (
                    "scalar_index_and_page_lookup_dependencies", "cross_core_flag_latency",
                    "atomic_response_latency", "external_cache_eviction"),
                "implementation_binding": {
                    "selector": "indexed_attention", "implementation": IMPLEMENTATION,
                    "branch_scope": "TND_BF16_block1_MLA_V_TEMPLATE",
                    "capture_dispatch": "not_established",
                }}
    return replace(op, recipe=IMPLEMENTATION, source_boundary="kernel",
                   metadata=tuple(metadata.items()))


def batch_analytic_ops(ops, outputs, *, runtime, views, prefix):
    """Use one query-owner grid for the explicitly selected analytic pipeline.

    Native source bindings retain their narrower admission contract. Explicit
    per-request/family recipes also keep their chosen call boundaries.
    """
    if runtime is None or len(outputs) < 2 or len(outputs) != len(views):
        return outputs
    kernels = runtime.kernels
    program = kernels.get("programs", {}).get("indexed_attention", {})
    if (not isinstance(program, dict) or program.get("kind") not in {
            "indexed_attention_pipeline", "indexed_attention_packed_int8"}
            or "attention" in kernels.get("families", {})
            or "indexed_attention" in kernels.get("lowerings", {})
            or any(view.query_ranges and view.query_ranges != ((0, view.q_len),)
                   for view in views)):
        return outputs
    ids = set(outputs)
    members = [op for op in ops if op.op_id in ids]
    if (len(members) != len(outputs) or any(op.plan_required for op in members)
            or any(op.recipe in kernels.get("recipes", {}) for op in members)):
        return outputs
    shapes = [dict(op.metadata).get("indexed_attention_geometry") for op in members]
    if any(shape is None or shape["window"] or shape["compression_ratio"] != 1
           or shape["cp"] != 1 for shape in shapes):
        return outputs
    common = [{key: value for key, value in shape.items() if key not in {"query", "context"}}
              for shape in shapes]
    if any(shape != common[0] for shape in common[1:]):
        return outputs
    sequences = tuple((shape["query"], shape["context"]) for shape in shapes)
    total_queries = sum(query for query, _ in sequences)
    meta = dict(members[0].metadata)
    # Active TND descriptors concatenate the query axis only. Context/indices
    # remain separate declarations; this descriptor is not a native ABI claim.
    if "attention_tensor_view" in meta:
        view = dict(meta["attention_tensor_view"])
        for key in ("query_shape", "query_rope_shape", "output_shape"):
            if key in view:
                view[key] = (total_queries, *view[key][1:])
        view["batch_count"] = len(sequences)
        meta["attention_tensor_view"] = view
    meta.update({
        "indexed_attention_geometry": {**common[0], "sequences": sequences},
        "indexed_batch_boundary": "analytic_single_kernel_shared_query_owner_grid",
        "indexed_batch_members": tuple(op.recipe for op in members),
        "indexed_sequence_requests": tuple(view.request_id for view in views),
    })
    for key in ("logical_pairs", "kv_sweeps", "indexed_gather_rows",
                "dense_window_sweep_rows", "selection_index_read_bytes"):
        meta[key] = sum(dict(op.metadata).get(key, 0) for op in members)
    shape = (len(sequences), total_queries, max(q + c for q, c in sequences),
             common[0]["heads"], common[0]["latent"] + common[0]["rope"], common[0]["latent"])
    batched = replace(
        members[0], op_id=prefix + ".latent_attention_batch", recipe="latent_attention_batch",
        deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        flops=sum(op.flops for op in members), vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members), metadata=tuple(meta.items()))
    first = next(i for i, op in enumerate(ops) if op.op_id in ids)
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.insert(first, batched)
    return [batched.op_id]


def _align(value, size):
    return (value + size - 1) // size * size


def _phases(keys, heads, latent, rope, *, first, last, scratch_policy,
            operand_modes=False):
    """Physical tile work; padding is confined to matrix operands/results."""
    padded_h, padded_k = _align(heads, 16), _align(keys, 16)
    width = latent + rope
    query_bytes = heads * width * 2
    payload = keys * width * 2
    # Key and rope are separate, 32-byte-aligned row transfers. The reviewed
    # 512+64 geometry has no row-transaction amplification.
    transactions = keys * (_align(latent * 2, 32) + _align(rope * 2, 32))
    score = padded_h * padded_k * 4
    probability = heads * padded_k * 2
    partial_output = padded_h * latent * 4
    output = heads * latent * 2 if last else 0
    offchip = scratch_policy == "offchip"

    def scratch(read=0, write=0, *, vector=False, fixpipe=False):
        resource = "mte2_aiv" if vector else "mte2_aic"
        if not offchip:
            resource += "_cached"
        work = {resource: read} if read else {}
        if write:
            work["fixpipe" if fixpipe else "mte3"] = write
        if offchip:
            work["hbm"] = read + write
        else:
            if read:
                work["l2_read"] = read
            if write:
                work["l2_write"] = write
        return work

    def atomic(write, *, fixpipe=False):
        work = scratch(write=write, fixpipe=fixpipe)
        # Destination-side read/modify/write is not another MTE2 load.
        role = "hbm" if offchip else "l2_read"
        work[role] = work.get(role, 0) + write
        return work

    def operands(a, b):
        # L0A and L0B instructions share MTE1 but have different service rates.
        return ({"mte1_l0a": a, "mte1_l0b": b} if operand_modes
                else {"mte1": a + b})

    qk_inner = operands(padded_h * ((keys + 127) // 128) * width * 2,
                        padded_k * width * 2)
    pv_reads = keys * latent * 2 + probability
    pv_inner = operands(padded_h * (latent // 128) * padded_k * 2,
                        latent * padded_k * 2)
    return (
        ("gather", Work(resources={
            "hbm": transactions + keys * 8,
            "mte2_aiv": transactions + keys * 8})),
        ("publish", Work(resources=scratch(write=payload + 1024))),
        ("qk_load", Work(resources=scratch(read=payload)) + Work(resources={
            "hbm": query_bytes, "mte2_aic": query_bytes, **qk_inner})),
        ("qk", Work(matrix={"bf16": 2 * padded_h * padded_k * width})),
        ("scores", Work(resources=scratch(write=score, fixpipe=True))),
        ("softmax", Work(resources={
            **scratch(read=heads * padded_k * 4 + 1024, vector=True),
            "vector": 8 * heads * padded_k + 26 * heads})),
        ("probability", Work(resources=scratch(write=probability))),
        ("pv_load", Work(resources={**scratch(read=pv_reads), **pv_inner})),
        ("pv", Work(matrix={"bf16": 2 * padded_h * padded_k * latent})),
        ("nupdate", Work(resources={
            **(scratch(write=partial_output) if first else atomic(partial_output)),
            "vector": padded_h * (latent if first else 128)})),
        ("partial", Work(resources=atomic(partial_output, fixpipe=True))),
        ("update", Work(resources={
            **scratch(read=heads * latent * 4 if last else 0, vector=True),
            "vector": heads * latent * 5 if last else 0})),
        ("output", Work(resources={"hbm": output, "mte3": output})),
    )


def build(op, geometry, *, scratch_policy, key_tile=512, slots=3, compact=True,
          operand_modes=False, rope0_analytic=False):
    meta = dict(op.metadata)
    shape = meta.get("indexed_attention_geometry")
    if not shape:
        raise ModelingError("indexed_attention_requires_frontend_geometry")
    if (scratch_policy not in {"resident", "offchip"} or key_tile != 512
            or slots != 3):
        raise ModelingError("indexed_attention_requires_declared_source_tile_and_scratch_policy")
    if (shape["window"] or shape["compression_ratio"] != 1
            or shape["cp"] != 1 or shape["query_dtype"] != "bf16"
            or shape["cache_dtype"] != "bf16"
            or shape.get("sparse_block_size", 1) != 1
            or (shape["latent"], shape["rope"]) not in (
                {(512, 64), (512, 0)} if rope0_analytic else {(512, 64)})
            or not 1 <= shape["heads"] <= 16):
        raise ModelingError("indexed_attention_source_geometry_outside_reviewed_domain")
    extension = shape["rope"] == 0
    source = ({**SOURCE, "scope": "analytic latent512/rope0 geometry extension",
               "capture_binary": "not_established", "native_source_binding": False}
              if extension else SOURCE)
    plan = build_pipeline(shape, geometry, scratch_policy=scratch_policy,
                          key_tile=key_tile, slots=slots, compact=compact,
                          operand_modes=operand_modes, source_items=tuple(source.items()))
    if extension:
        return replace(plan, metadata={
            "kind": "indexed_attention_pipeline", "coverage": "analytic_geometry_extension",
            "geometry_extension": {"latent": 512, "rope": 0},
            "source_reference": SOURCE, "kernel_dispatch": "not_established",
            "unmodeled_features": ["captured_kernel_schedule", "scalar_index_response",
                                   "cross_core_flag_latency", "external_cache_eviction"],
        })
    return plan


def build_pipeline(shape, geometry, *, scratch_policy, key_tile=512, slots=3,
                   compact=True, operand_modes=False, phase_factory=_phases,
                   source_items=tuple(SOURCE.items()), paired_vector_stages=(),
                   vector_buffer_bytes=0, phase_unit_weights=None):
    """Common finite query pipeline; callers own representation admission."""
    sequences = tuple(tuple(row) for row in shape.get(
        "sequences", ((shape.get("query", 0), shape.get("context", 0)),)))
    if (not sequences or any(len(row) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in row) for row in sequences)):
        raise ModelingError("indexed_attention_requires_integer_sequence_geometry")
    query, topk = sum(row[0] for row in sequences), shape["topk"]
    heads, latent, rope = shape["heads"], shape["latent"], shape["rope"]
    owners = selected_query_owners(query, geometry.compute_units)
    # The source reserves four KV rings and three score/probability/output
    # rings per owner. Cache residency is an explicit conditional placement.
    hpad = _align(heads, 16)
    per_owner_scratch = (4 * key_tile * (latent + rope) * 2
                         + 3 * hpad * (key_tile * 6 + latent * 4))
    if scratch_policy == "resident":
        if (geometry.buffer_units.get("l2", 1) != 1
                or geometry.buffers.get("l2", 0) < owners * per_owner_scratch):
            raise ModelingError("indexed_attention_resident_scratch_requires_shared_cache_capacity")

    return _plan(sequences, topk, heads, latent, rope, geometry.compute_units,
                 scratch_policy, key_tile, slots, compact, operand_modes,
                 phase_factory, source_items, tuple(paired_vector_stages), vector_buffer_bytes,
                 phase_unit_weights, shape.get("index_pool_size", 1),
                 shape.get("index_pool_tail", False))


def query_positions(owner, query, compute_units):
    """InitCalcParamsEach assigns contiguous ceil(Q/AIC) query intervals."""
    span = (query + compute_units - 1) // compute_units
    return range(owner * span, min(query, (owner + 1) * span))


def sequence_query_positions(owner, sequences, compute_units):
    """Flatten query ownership, never the independent sequence KV/context axes."""
    ends, total = [], 0
    for query, _ in sequences:
        total += query
        ends.append(total)
    for position in query_positions(owner, total, compute_units):
        sequence = bisect_right(ends, position)
        start = ends[sequence - 1] if sequence else 0
        yield sequence, position - start


@lru_cache(maxsize=4)
def _plan(sequences, topk, heads, latent, rope, compute_units,
          scratch_policy, key_tile, slots, compact, operand_modes,
          phase_factory, source_items, paired_vector_stages, vector_buffer_bytes,
          phase_unit_weights, index_pool_size, index_pool_tail):
    # Repeated model layers share geometry, never dependencies or Engine state.
    # Cache only bounded phase histograms and pure nominal timing calculations.
    query = sum(row[0] for row in sequences)
    owners = selected_query_owners(query, compute_units)
    per_owner_scratch = (4 * key_tile * (latent + rope) * 2
                         + 3 * _align(heads, 16) * (key_tile * 6 + latent * 4))

    def tile_keys(owner):
        for sequence, position in sequence_query_positions(owner, sequences, compute_units):
            context = sequences[sequence][1]
            count = selected_tokens(context + position + 1, topk,
                                    index_pool_size, index_pool_tail)
            for start in range(0, count, key_tile):
                keys = min(key_tile, count - start)
                yield keys, start == 0, start + keys == count

    @lru_cache(maxsize=2048)
    def phases(key):
        keys, first, last = key
        return phase_factory(keys, heads, latent, rope, first=first, last=last,
                             scratch_policy=scratch_policy, operand_modes=operand_modes)

    owner_work = []
    for owner in range(owners):
        work = Work()
        for key, count in Counter(tile_keys(owner)).items():
            for _, phase in phases(key):
                work += phase.scaled(count)
        owner_work.append(work)
    total = Work()
    for work in owner_work:
        total += work

    nominal_cache = OrderedDict()

    def lower(ctx):
        # Copy latency/issue limits are deliberately not guessed from residuals.
        required = {"mte2_aic", "mte2_aiv", "mte1", "mte3", "fixpipe"}
        if scratch_policy == "resident":
            required |= {"mte2_aic_cached", "mte2_aiv_cached", "l2_read", "l2_write"}
        for role in required:
            if ctx.physical_role(role) not in ctx.resource_names:
                raise ModelingError(f"indexed_attention_physical_copy_rate_required:{role}")
        if operand_modes and any(ctx.physical_role(role) != "mte1"
                                 for role in ("mte1_l0a", "mte1_l0b")):
            raise ModelingError("indexed_attention_operands_must_share_mte1")

        signature = tuple(tuple(sorted(mapping.items())) for mapping in (
            ctx.rates, ctx.resource_names, ctx.resource_units,
            ctx.service_modes, ctx.compute_peaks))
        if signature not in nominal_cache:
            nominal_cache[signature] = {}
            if len(nominal_cache) > 4:
                nominal_cache.popitem(last=False)
        nominal_cache.move_to_end(signature)
        nominal = nominal_cache[signature]

        def lower_owner(owner):
            def bind(phase, name=None, key=None):
                work = ctx.bind(phase)
                roles = set(phase.resources) | ({"cube"} if phase.matrix else set())
                placement = ctx.owned(
                    owner, compute_units, roles - {"hbm", "l2_read", "l2_write"})
                # Gather splits selected KV rows across both AIVs. For <=16
                # heads the source gives all softmax/output rows to AIV0.
                single_roles = (set() if name in paired_vector_stages
                                or (name is None and paired_vector_stages) else {"vector"})
                if name not in {None, "gather", "publish"}:
                    single_roles |= {"mte2_aiv", "mte2_aiv_cached", "mte3"}
                for role in single_roles & roles:
                    resource = ctx.resource_names[ctx.physical_role(role)]
                    placement[resource] = placement[resource][:1]
                if phase_unit_weights and key is not None:
                    for role, fractions in phase_unit_weights(name, key).items():
                        resource = ctx.resource_names[ctx.physical_role(role)]
                        if resource in placement:
                            placement[resource] = tuple(
                                unit for unit, fraction in zip(placement[resource], fractions)
                                if fraction)
                return work, placement

            def phase_caps(work, name, key):
                # An uneven local partition has the supply of its busiest
                # physical unit; it cannot borrow an inactive sibling's pipe.
                result = {}
                if phase_unit_weights:
                    for role, fractions in phase_unit_weights(name, key).items():
                        resource = ctx.resource_names[ctx.physical_role(role)]
                        if work.get(resource, 0):
                            result[resource] = ctx.rates[resource] / (
                                ctx.resource_units[ctx.physical_role(role)] * max(fractions))
                return result

            def duration(work, placement):
                limits = {}
                for resource, amount in work.items():
                    units = len(placement.get(resource, ()))
                    role = next(role for role, name in ctx.resource_names.items()
                                if name == resource)
                    fraction = units / ctx.resource_units.get(role, compute_units) if units else 1
                    limits[resource] = amount / (ctx.rates[resource] * fraction)
                return max(limits.values(), default=0)

            def stage_list(key):
                result = []
                for name, phase in phases(key):
                    work, placement = bind(phase, name, key)
                    result.append(Stage(
                        name, work, after=(result[-1].name,) if result else (),
                        previous=("partial",) if name == "nupdate" and not key[1] else (),
                        placement=placement, caps=phase_caps(work, name, key), coupled=True))
                return tuple(result)

            buffers = ()
            if scratch_policy == "resident":
                buffers = (ctx.allocate(f"scratch{owner}", "l2", per_owner_scratch, (0,)),)
            if vector_buffer_bytes:
                from .ownership import unit_group
                buffers += (ctx.allocate(
                    f"vector_scratch{owner}", "ub", vector_buffer_bytes,
                    unit_group(owner, compute_units, ctx.pool_units["ub"])),)
            if not compact:
                from .program import PipelineProgram
                # Bounded verification control; input phases are regenerated
                # lazily, so even this path retains no full task history.
                keys_seq = tuple(tile_keys(owner))
                count = len(keys_seq)
                program = PipelineProgram(
                    count, slots, lambda idx, keys_seq=keys_seq: stage_list(keys_seq[idx]))
                return KernelProgram((
                    Stage("reserve", acquire=buffers),
                    Stage("tiles", after=("reserve",), program=program),
                    Stage("release", after=("tiles",), free=buffers)))

            # Ordered phase recurrence with finite in-flight tiles. Shared
            # physical pipes impose an additional duty bound across streams.
            # Compact mode is a fluid approximation; the expanded control
            # retains exact phase arbitration for counterexample testing.
            all_work, all_placement = bind(owner_work[owner])
            if phase_unit_weights:
                # The aggregate may use only units used by its real phases.
                # A short packed-cache tail can leave the sibling vector
                # entirely idle even though publish emits metadata on both.
                actual_units = {}
                for key in set(tile_keys(owner)):
                    for name, phase in phases(key):
                        _, placement = bind(phase, name, key)
                        for resource, units in placement.items():
                            actual_units.setdefault(resource, set()).update(units)
                all_placement = {resource: tuple(sorted(units))
                                 for resource, units in actual_units.items()}
            if owner not in nominal:
                phase_times = {}
                stream_tail = {}
                retired = [0.0] * slots
                end = 0.0
                for count, key in enumerate(tile_keys(owner)):
                    if key not in phase_times:
                        times = []
                        for name, phase in phases(key):
                            work, placement = bind(phase, name, key)
                            caps = phase_caps(work, name, key)
                            times.append((name, max(duration(work, placement),
                                max((work[r] / cap for r, cap in caps.items()), default=0))))
                        phase_times[key] = tuple(times)
                    ready = retired[count % slots]
                    prior_partial = stream_tail.get("partial", 0)
                    for name, elapsed in phase_times[key]:
                        if name == "nupdate" and not key[1]:
                            ready = max(ready, prior_partial)
                        ready = max(ready, stream_tail.get(name, 0)) + elapsed
                        stream_tail[name] = ready
                    retired[count % slots] = ready
                    end = max(end, ready)
                unit_demand = Counter()
                for key, count in Counter(tile_keys(owner)).items():
                    for name, phase in phases(key):
                        work, placement = bind(phase, name, key)
                        weights = {ctx.resource_names[ctx.physical_role(role)]: tuple(
                            value for value in fractions if value)
                            for role, fractions in (phase_unit_weights(name, key).items()
                                                    if phase_unit_weights else ())}
                        for resource, units in placement.items():
                            fractions = weights.get(resource, (1 / len(units),) * len(units))
                            for unit, fraction in zip(units, fractions):
                                unit_demand[resource, unit] += work.get(resource, 0) * count * fraction
                for (resource, _), amount in unit_demand.items():
                    role = next(role for role, name in ctx.resource_names.items()
                                if name == resource)
                    end = max(end, amount * ctx.resource_units.get(
                        role, compute_units) / ctx.rates[resource])
                nominal[owner] = max(end, duration(all_work, all_placement))
            end = nominal[owner]
            caps = {key: amount / end for key, amount in all_work.items()} if end else {}
            return KernelProgram((
                Stage("reserve", acquire=buffers),
                Stage("pipeline", all_work, after=("reserve",), caps=caps,
                      placement=all_placement, coupled=True),
                Stage("release", after=("pipeline",), free=buffers)))

        if ctx.record_mapping:
            ctx.record_mapping({
                "program": "indexed_attention_pipeline", "source": dict(source_items),
                "scratch_policy": scratch_policy,
                "scratch_bytes": owners * per_owner_scratch, "key_tile": key_tile,
                "sparse_block_size": 1,
                "owners": owners, "producer_slots": slots, "compact": compact,
                "query_schedule": "contiguous_ceil_query_over_aic",
                "sequences": sequences,
                "batch_boundary": ("analytic_shared_query_owner_grid" if len(sequences) > 1
                                   else "single_sequence"),
                "source_binding": ("analytic_batch_adaptation; capture_dispatch_not_established"
                                   if len(sequences) > 1 else "declared_single_sequence_pipeline"),
                "operand_modes": operand_modes,
                "abstraction": "finite stage recurrence with shared-pipe duty bound and fluid contention",
                "unmodeled": ["scalar_index_response", "cross_core_flag_latency",
                              "external_cache_eviction", "atomic_response_latency"],
                "output_accumulation": "integer_bias_update_then_atomic_fixpipe; final_vector_read_once",
                "vector_ownership": "two_gather_AIVs; one_softmax_and_output_AIV",
                "paired_vector_stages": paired_vector_stages,
                "vector_buffer_bytes_per_aiv": vector_buffer_bytes,
            })
        return ParallelProgram(tuple(lower_owner(owner) for owner in range(owners)))

    return OperatorPlan(total, lower)
