"""Coarse source phases for the SCFA SparseAttnSharedkv branch.

[Role] Price only the demonstrated SparseAttnSharedkv SCFA branch.
[Invariants] Original-window and selected compressed rows are separate typed
  regions; one query has one final output after the combined S2 list. Resident
  GM reads use the declared cached service mode. SWA/CFA are not this plan.
[Timeouts & Locks] Streaming O(query/key-tiles) arithmetic, two slots per owner.
[Side-Effect Policy] Pure plans and ordinary Engine submissions; no observations.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache, partial

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .sparse_attention import selected_query_owners


SOURCE = {
    "repository": "vllm-ascend",
    "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "status": "five retained source SHA256s independently verified against pinned git blobs",
    "scope": "SparseAttnSharedkv SCFA only, TND/PA_ND, D512 BF16, one KV head",
    "dispatch": "sparse_attn_sharedkv.cpp routes SCFA_TEMPLATE to Scfa; all other modes to Swa",
    "capture_identity": "not asserted",
}


def recipe(*, scratch_policy="offchip", slots=2):
    return PlannedRecipe(partial(build, scratch_policy=scratch_policy, slots=slots))


def _align(value, size):
    return (value + size - 1) // size * size


def _scratch_bytes(heads):
    # Host tiling allocates two MM1/probability/MM2/accumulator regions,
    # four gathered-KV regions and the declared (currently unused) valid-size GM.
    return 4 * 512 * 512 * 2 + 2 * _align(heads, 16) * 512 * 14 + 4096


def source_tiles(query, context, window, ratio, topk, owner, owners):
    """Lazy combined original-then-compressed S2 tiles for one owner."""
    for position in range(owner, query, owners):
        original = min(context + position + 1, window)
        selected = min((context + position + 1) // ratio, topk)
        total_tiles = (original + 511) // 512 + (selected + 511) // 512
        index = 0
        for branch, count in (("ori", original), ("cmp", selected)):
            for start in range(0, count, 512):
                yield branch, min(512, count - start), index == 0, index + 1 == total_tiles
                index += 1


def _phases(branch, rows, heads, width, *, first, last, scratch_policy):
    """Coarse physical transfers and arithmetic for one SCFA S2 tile.

    Original KV is read directly twice by the AIC. Only compressed KV is
    gathered through paired AIVs into scratch. Scalar index bytes are an
    external lower bound; they do not invent a private DMA or response delay.
    """
    padded_rows = _align(rows, 16)
    stride_rows = _align(rows, 32)
    padded_h = _align(heads, 16)
    payload = rows * width * 2
    score_write = padded_h * padded_rows * 4
    score_read = heads * stride_rows * 4
    probability_write = heads * stride_rows * 2
    probability_read = heads * rows * 2
    pv_workspace = padded_h * width * 4
    vector_workspace = heads * width * 4
    offchip = scratch_policy == "offchip"

    def gm(read=0, write=0, read_role="mte2_aic", write_role="mte3"):
        if read and not offchip:
            read_role += "_cached"
        work = {read_role: read} if read else {}
        if write:
            work[write_role] = write
        if offchip:
            work["hbm"] = read + write
        else:
            if read:
                work["l2_read"] = read
            if write:
                work["l2_write"] = write
        return Work(resources=work)

    result = []
    if branch == "cmp":
        # CalcParams fixes V0's 512-row block; valid-prefix indices give AIV0
        # the first 256 rows and AIV1 the remainder, never a pooled half-rate.
        for aiv, count in enumerate((min(rows, 256), max(0, rows - 256))):
            if count:
                amount = count * width * 2
                result.extend((
                    (f"gather{aiv}", Work(resources={
                        "hbm": amount + count * 8, "mte2_aiv": amount})),
                    (f"publish{aiv}", gm(write=amount))))
        kv_read = gm(read=payload)
    else:
        kv_read = Work(resources={"hbm": payload, "mte2_aic": payload})
    qk_inner = (padded_h * ((rows + 127) // 128) + padded_rows) * width * 2
    pv_inner = (4 * padded_h * padded_rows + padded_rows * width) * 2
    accum_read = vector_workspace * (2 if not first else 1)
    if last:
        # External output is never a resident temporary, regardless of where
        # the preceding FP32 accumulator was served.
        output = heads * width * 2
        accum_store = Work(resources={"hbm": output, "mte3": output})
    else:
        accum_store = gm(write=vector_workspace)
    accum = Work(resources={
        "vector": heads * width * ((3 if not first else 1) + (2 if last else 0))})
    result.extend((
        ("qk_load", kv_read + Work(resources={
            "hbm": heads * width * 2, "mte2_aic": heads * width * 2})),
        ("qk_operands", Work(resources={"mte1": qk_inner})),
        ("qk", Work(matrix={"bf16": 2 * padded_h * padded_rows * width})),
        ("scores", gm(write=score_write, write_role="fixpipe")),
        ("score_read", gm(read=score_read, read_role="mte2_aiv")),
        ("softmax", Work(resources={
            "vector": 8 * heads * rows + 26 * heads})),
        ("probability", gm(write=probability_write)),
        ("pv_load", kv_read + gm(read=probability_read)),
        ("pv_operands", Work(resources={"mte1": pv_inner})),
        ("pv", Work(matrix={"bf16": 2 * padded_h * padded_rows * width})),
        ("pv_store", gm(write=pv_workspace, write_role="fixpipe")),
        ("accum_read", gm(read=accum_read, read_role="mte2_aiv")),
        ("accum", accum),
        ("accum_store", accum_store),
    ))
    return tuple(result)


@dataclass(frozen=True)
class _OwnerStream:
    """A fresh lazy tile iterator per invocation, including repeated replays."""
    tile_count: int
    tiles: object
    stages: object
    slots: int

    def start(self, engine, name, ready, *, owner=""):
        iterator = iter(self.tiles())
        return PipelineProgram(self.tile_count, self.slots,
                               lambda _: self.stages(next(iterator))).start(
                                   engine, name, ready, owner=owner)


def build(op, geometry, *, scratch_policy="offchip", slots=2):
    meta = dict(op.metadata)
    shape = meta.get("sharedkv_attention_geometry")
    if not shape:
        raise ModelingError("sharedkv_attention_requires_frontend_geometry")
    if scratch_policy not in {"resident", "offchip"} or slots != 2:
        raise ModelingError("sharedkv_attention_requires_declared_scratch_policy")
    integer_fields = ("query", "context", "window", "cmp_ratio", "topk", "heads", "width")
    if any(isinstance(shape.get(key), bool) or not isinstance(shape.get(key), int)
           or shape[key] < 0 for key in integer_fields):
        raise ModelingError("sharedkv_attention_source_geometry_outside_reviewed_domain")
    declarations = {
        "branch": "SCFA", "query_layout": "TND", "cache_layout": "PA_ND",
        "sequences": 1, "views": 1, "kv_heads": 1, "cp": 1, "pcp": 1, "sparse_block_size": 1,
        "query_dtype": "bf16", "cache_dtype": "bf16", "output_dtype": "bf16",
        "sinks_dtype": "fp32", "return_softmax_lse": False,
        "metadata_s2_split": False, "selected_rows": "valid_prefix",
        "owner_schedule": "one_query_per_core",
    }
    if (op.phase != "prefill" or shape["query"] < 1
            or any(shape.get(key) != value for key, value in declarations.items())
            or shape["heads"] not in {4, 8, 12, 16} or shape["width"] != 512
            or not (0 < shape["window"] <= 512 and shape["cmp_ratio"] in {4, 128}
                    and shape["topk"] in {512, 1024})):
        raise ModelingError("sharedkv_attention_source_geometry_outside_reviewed_domain")
    owners = selected_query_owners(shape["query"], geometry.compute_units)
    per_owner_scratch = _scratch_bytes(shape["heads"])
    if scratch_policy == "resident":
        if (geometry.buffer_units.get("l2", 1) != 1
                or geometry.buffers.get("l2", 0) < owners * per_owner_scratch):
            raise ModelingError("sharedkv_attention_resident_scratch_requires_shared_cache_capacity")
    return _plan(shape["query"], shape["context"], shape["window"],
                 shape["cmp_ratio"], shape["topk"], shape["heads"], shape["width"],
                 geometry.compute_units, scratch_policy, slots)


@lru_cache(maxsize=4)
def _plan(query, context, window, ratio, topk, heads, width, compute_units,
          scratch_policy, slots):
    owners = selected_query_owners(query, compute_units)
    per_owner_scratch = _scratch_bytes(heads)

    def tiles(owner):
        return source_tiles(query, context, window, ratio, topk, owner, compute_units)

    @lru_cache(maxsize=1024)
    def phases(key):
        branch, rows, first, last = key
        return _phases(branch, rows, heads, width, first=first, last=last,
                       scratch_policy=scratch_policy)

    owner_counts = tuple(Counter(tiles(owner)) for owner in range(owners))
    # Both paired AIVs initialize their local sinks, even when the odd AIV has
    # no softmax/output rows. Initialization is once per enabled owner.
    sink_work = Work(resources={"hbm": heads * 4, "mte2_aiv": heads * 4})
    total = sink_work.scaled(2 * owners)
    for counts in owner_counts:
        for key, count in counts.items():
            for _, phase in phases(key):
                total += phase.scaled(count)

    def lower(ctx):
        required = {"mte2_aic", "mte2_aiv", "mte1", "mte3", "fixpipe"}
        if scratch_policy == "resident":
            required |= {"mte2_aic_cached", "mte2_aiv_cached", "l2_read", "l2_write"}
        for role in required:
            if ctx.physical_role(role) not in ctx.resource_names:
                raise ModelingError(f"sharedkv_attention_physical_copy_rate_required:{role}")
        if scratch_policy == "resident":
            for role in ("mte2_aic", "mte2_aiv"):
                if ctx.physical_role(role + "_cached") != ctx.physical_role(role):
                    raise ModelingError(f"sharedkv_attention_cached_mode_must_share_pipe:{role}")
            if (ctx.physical_role("l2_read") != ctx.physical_role("l2_write")
                    or ctx.resource_units.get(ctx.physical_role("l2_read")) != 1):
                raise ModelingError("sharedkv_attention_requires_shared_l2_service")
        # This source uses one AIC paired with exactly two AIVs. Pooling a
        # single-AIV phase across both private pipes can borrow idle capacity.
        for role in ("vector", "mte2_aiv", "mte3"):
            if ctx.resource_units.get(ctx.physical_role(role)) != 2 * compute_units:
                raise ModelingError(f"sharedkv_attention_requires_paired_aiv_ownership:{role}")
        for role in ("cube", "mte2_aic", "mte1", "fixpipe"):
            if ctx.resource_units.get(ctx.physical_role(role)) != compute_units:
                raise ModelingError(f"sharedkv_attention_requires_aic_ownership:{role}")

        def lower_owner(owner):
            def bind(phase, name):
                work = ctx.bind(phase)
                roles = set(phase.resources) | ({"cube"} if phase.matrix else set())
                placement = ctx.owned(owner, compute_units,
                                      roles - {"hbm", "l2_read", "l2_write"})
                subblock = 1 if name in {"gather1", "publish1", "sinks1"} else 0
                for role in ({"vector", "mte2_aiv", "mte2_aiv_cached", "mte3"} & roles):
                    resource = ctx.resource_names[ctx.physical_role(role)]
                    placement[resource] = (placement[resource][subblock],)
                return work, placement

            def stages(key):
                emitted = []
                previous = None
                publishes = tuple(name for name, _ in phases(key) if name.startswith("publish"))
                for name, phase in phases(key):
                    work, placement = bind(phase, name)
                    if name.startswith("gather"):
                        after = ()
                    elif name.startswith("publish"):
                        after = ("gather" + name[-1],)
                    elif name == "qk_load":
                        after = publishes
                    else:
                        after = (previous,)
                    prior = ("accum_store",) if name == "accum_read" and not key[2] else ()
                    emitted.append(Stage(name, work, after=after, previous=prior,
                                         placement=placement))
                    previous = name
                return tuple(emitted)

            buffers = ()
            if scratch_policy == "resident":
                buffers = (ctx.allocate(f"sharedkv_scratch{owner}", "l2", per_owner_scratch, (0,)),)
            initialization = []
            for subblock in range(2):
                name = f"sinks{subblock}"
                work, placement = bind(sink_work, name)
                initialization.append(Stage(name, work, after=("reserve",), placement=placement))
            if ctx.record_mapping:
                ctx.record_mapping({
                    "program": "sharedkv_attention_pipeline", "source": SOURCE,
                    "scratch_policy": scratch_policy, "owners": owners,
                    "owner_schedule": "one_query_per_core", "metadata_s2_split": False,
                    "producer_slots": slots, "sparse_block_size": 1,
                    "cmp_ratio": ratio, "topk_rows": topk, "window_rows": window,
                    "branch": "SCFA", "selected_rows": "valid_prefix",
                    "abstraction": "bounded coarse phase DAG with source AIC/AIV ownership",
                    "external_output_policy": "hbm_write", "external_kv_policy": "offchip",
                    "unmodeled": ["scalar_index_response", "original_page_table_control",
                                  "invalid_index_probes", "local_initialization_instructions",
                                  "cross_core_flag_latency", "external_cache_eviction",
                                  "inner_matrix_overlap", "local_buffer_capacity"]})
            stream = _OwnerStream(sum(owner_counts[owner].values()), partial(tiles, owner), stages, slots)
            return KernelProgram((
                Stage("reserve", acquire=buffers), *initialization,
                Stage("pipeline", after=("sinks0", "sinks1"), program=stream),
                Stage("release", after=("pipeline",), free=buffers)))

        return ParallelProgram(tuple(lower_owner(owner) for owner in range(owners)))

    return OperatorPlan(total, lower)
