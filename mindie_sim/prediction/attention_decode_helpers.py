"""Geometry, work accounting and stage helpers for staged GQA decode.

[Role] Validate attention geometry and build bounded buffer and phase records.
[Invariants] Preserve causal work, serial dependencies and physical ownership;
  split outputs are materialized once and buffers retain their original lifetimes.
[Timeouts & Locks] Pure finite planning and bounded virtual-engine callbacks;
  no waits or locks.
[Side-Effect Policy] No files, measurements or device access. Merge programs
  submit only to the caller's virtual engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineTemplate, RecipeContext, Stage,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from .attention_kv_access import KVCacheAccess, kv_access

if TYPE_CHECKING:
    from .attention_decode import StagedGQADecode


_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
    # Quantized cache payloads retain their storage dtype.  Conversion and
    # scale work are separate declarations, not a re-labelled BF16 tensor.
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2",
    "int8": "int8", "uint8": "uint8", "int4": "int4",
    "fp4": "fp4_e2m1", "fp4_e2m1": "fp4_e2m1",
}


_FP32_BYTES = 4


def _partial_row_bytes(dim: int) -> int:
    """Per output row: FP32 partial accumulator plus max and sum scalars."""
    return dim * _FP32_BYTES + 2 * _FP32_BYTES


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"attention_decode_{name}_must_be_positive_int")
    return value


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"attention_decode_{name}_must_be_nonnegative_int")
    return value


def _dtype(value, name: str) -> str:
    if not isinstance(value, str):
        raise ModelingError(f"attention_decode_{name}_must_be_dtype")
    try:
        return _DTYPE_ALIASES[value.strip().lower()]
    except KeyError:
        raise ModelingError(f"attention_decode_unsupported_{name}:{value}") from None


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def causal_pairs(query: int, context: int) -> int:
    """Causal key/query pairs for one query head of a speculative decode block."""
    return query * (context + 1) + query * (query - 1) // 2


@dataclass(frozen=True)
class _Contribution:
    split: int
    tile_start: int
    tile_count: int
    owner: int | None = None


@dataclass(frozen=True)
class _Spec:
    requests: tuple[tuple[int, int], ...]
    query_heads: int
    kv_heads: int
    head_dim: int
    query_dtype: str
    kv_dtype: str
    key_tile: int
    query_tile: int
    kv_splits: int
    slots: int
    outputs: tuple[_Output, ...]
    kv_cache: KVCacheAccess = KVCacheAccess()

    @property
    def group(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def query_bytes(self) -> int:
        return int(dtype_bytes(self.query_dtype))


@dataclass(frozen=True)
class _Output:
    request: int
    kv_head: int
    query_tile: int
    row_start: int
    row_end: int
    token_first: int
    token_last: int
    horizon: int
    contributions: tuple[_Contribution, ...]

    @property
    def rows(self) -> int:
        return self.row_end - self.row_start


def _split_ranges(tiles: int, requested: int) -> tuple[tuple[int, int], ...]:
    """Contiguous, nonempty tile ranges, at most ``requested`` partitions."""
    if tiles <= 0:
        return ()
    count = min(requested, tiles)
    base, extra = divmod(tiles, count)
    ranges, start = [], 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        ranges.append((start, start + size))
        start += size
    return tuple(ranges)


def _spec(op: OpSpec, recipe: StagedGQADecode) -> _Spec:
    if getattr(op, "phase", "decode") not in {"decode", "draft"}:
        raise ModelingError("attention_decode_only_supports_decode_phase")
    metadata = dict(getattr(op, "metadata", ()))
    raw = metadata.get("gqa_decode")
    if not isinstance(raw, Mapping):
        raise ModelingError("attention_decode_gqa_decode_metadata_required")

    requests_raw = raw.get("requests")
    if not isinstance(requests_raw, (list, tuple)):
        raise ModelingError("attention_decode_requests_must_be_sequence")
    requests = []
    for item in requests_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ModelingError("attention_decode_request_must_be_query_context_pair")
        query = _nonnegative_int(item[0], "query_len")
        context = _nonnegative_int(item[1], "context")
        requests.append((query, context))
    requests = tuple(requests)

    heads = _positive_int(raw.get("query_heads"), "query_heads")
    kv_heads = _positive_int(raw.get("kv_heads"), "kv_heads")
    dim = _positive_int(raw.get("head_dim"), "head_dim")
    if heads % kv_heads:
        raise ModelingError("attention_decode_query_heads_not_divisible_by_kv_heads")
    query_dtype = _dtype(raw.get("query_dtype"), "query_dtype")
    kv_dtype = _dtype(raw.get("kv_dtype"), "kv_dtype")
    if raw.get("causal") is not True:
        raise ModelingError("attention_decode_causal_true_required")
    if raw.get("window", 0) != 0:
        raise ModelingError("attention_decode_windowed_decode_not_selected")
    if raw.get("cp", 1) != 1:
        raise ModelingError("attention_decode_context_parallel_not_selected")

    group = heads // kv_heads
    # A fixed split count is a mapping declaration, not a source selector.
    # The optional occupancy mapping shares cores between independent query
    # tiles first, then partitions their KV loops. Short contexts still cap the
    # split count at real key tiles; partial writes/merges are never removed.
    query_tasks = kv_heads * sum(_ceil_div(query * group, recipe.query_tile)
                                for query, _ in requests if query)
    splits = (max(recipe.kv_splits, _ceil_div(recipe.target_cores, query_tasks))
              if recipe.target_cores and query_tasks else recipe.kv_splits)
    outputs = []
    for request_index, (query, context) in enumerate(requests):
        if query == 0:
            continue
        rows = query * group
        length = context + query
        tile_count = _ceil_div(length, recipe.key_tile)
        partitions = _split_ranges(tile_count, splits)
        for kv_head in range(kv_heads):
            for query_tile, row_start in enumerate(
                    range(0, rows, recipe.query_tile)):
                row_end = min(rows, row_start + recipe.query_tile)
                token_first = row_start // group
                token_last = (row_end - 1) // group
                horizon = context + token_last + 1
                cutoff = _ceil_div(horizon, recipe.key_tile)
                contributions = tuple(
                    _Contribution(index, begin, min(end, cutoff) - begin)
                    for index, (begin, end) in enumerate(partitions)
                    if min(end, cutoff) > begin
                )
                if not contributions:
                    continue
                outputs.append(_Output(
                    request_index, kv_head, query_tile, row_start, row_end,
                    token_first, token_last, horizon, contributions,
                ))
    if recipe.kv_mapping in {"fai_flash_decode", "fai_host", "fai_host_arch35"}:
        from .fai_decode_mapping import (
            host_partitions,
            host_partitions_arch35,
            partitions,
        )
        if recipe.kv_mapping == "fai_host_arch35":
            mapping = host_partitions_arch35(
                requests, heads, kv_heads, recipe.target_cores,
                key_tile=recipe.key_tile, query_tile=recipe.query_tile,
                head_dim=dim,
            )
        else:
            mapper = host_partitions if recipe.kv_mapping == "fai_host" else partitions
            mapping = mapper(requests, heads, kv_heads, recipe.target_cores,
                             key_tile=recipe.key_tile, query_tile=recipe.query_tile)
        outputs = [replace(output, contributions=tuple(
            _Contribution(index, start, end - start, owner)
            for index, (owner, start, end) in enumerate(
                mapping[output.request, output.kv_head]))) for output in outputs]
        splits = max(len(output.contributions) for output in outputs)
    return _Spec(
        requests, heads, kv_heads, dim, query_dtype, kv_dtype,
        recipe.key_tile, recipe.query_tile, splits, recipe.slots,
        tuple(outputs),
        kv_access(op, query_dtype=query_dtype, kv_dtype=kv_dtype,
                  head_dim=dim),
    )


def _rows_per_token(output: _Output, group: int, token: int) -> int:
    return max(0, min(output.row_end, (token + 1) * group)
               - max(output.row_start, token * group))


def _output_pairs(output: _Output, context: int, group: int) -> int:
    """Valid causal pairs, independently of key tiling."""
    return sum(_rows_per_token(output, group, token) * (context + token + 1)
               for token in range(output.token_first, output.token_last + 1))


def _output_row_tiles(output: _Output, context: int, group: int,
                      key_tile: int) -> int:
    """Sum of active query rows over key tiles (online-softmax work)."""
    return sum(_rows_per_token(output, group, token)
               * _ceil_div(context + token + 1, key_tile)
               for token in range(output.token_first, output.token_last + 1))


def _total_work(spec: _Spec, *, include_mte1: bool, read: str = "mte2",
                vread: str | None = None, write: str | None = "mte3",
                operand: str | None = "mte1") -> Work:
    cache = spec.kv_cache
    vread = vread or read
    producer_reads = vector_reads = writes = mte3_extra = vector = mte1 = 0
    flops = 0
    for output in spec.outputs:
        _query, context = spec.requests[output.request]
        pairs = _output_pairs(output, context, spec.group)
        row_tiles = _output_row_tiles(output, context, spec.group, spec.key_tile)
        contributions = len(output.contributions)
        q_bytes = contributions * output.rows * spec.head_dim * spec.query_bytes
        key_payload = output.horizon * cache.key_payload_row
        value_payload = output.horizon * cache.value_payload_row
        output_bytes = output.rows * spec.head_dim * spec.query_bytes
        partial = contributions * output.rows * _partial_row_bytes(spec.head_dim)
        page_meta = 0
        page_scale = 0
        if cache.paged:
            for contribution in output.contributions:
                start = contribution.tile_start * spec.key_tile
                end = min((contribution.tile_start + contribution.tile_count)
                          * spec.key_tile, output.horizon)
                pages = cache.pages(start, end)
                page_meta += pages * cache.page_meta_bytes
                if cache.scale_scope == "page":
                    page_scale += pages * cache.page_scale_bytes
        row_scale = (cache.row_scale_bytes(output.horizon)
                     if cache.scale_scope == "row" else 0)
        producer_reads += (q_bytes + page_meta
                           + (0 if cache.key_needs_conversion else key_payload)
                           + (0 if cache.value_needs_conversion else value_payload))
        vector_reads += (page_scale + row_scale
                         + (key_payload if cache.key_needs_conversion else 0)
                         + (value_payload if cache.value_needs_conversion else 0))
        writes += output_bytes
        if cache.key_needs_conversion:
            mte3_extra += output.horizon * cache.key_operand_row
        if cache.value_needs_conversion:
            mte3_extra += output.horizon * cache.value_operand_row
        if include_mte1:
            mte1 += q_bytes
            mte1 += (output.horizon * cache.key_operand_row
                     if cache.key_needs_conversion else key_payload)
            mte1 += (output.horizon * cache.value_operand_row
                     if cache.value_needs_conversion else value_payload)
        flops += 4 * pairs * spec.head_dim
        vector += 7 * pairs + (3 * spec.head_dim + 10) * row_tiles
        vector += cache.conversion_ops(output.horizon)
        if include_mte1:
            # Each probability tile becomes the left operand of PV. It never
            # leaves the on-chip hierarchy or becomes an HBM read.
            mte1 += pairs * spec.query_bytes
        if contributions == 1:
            vector += output.rows * (spec.head_dim + 1)
        else:
            vector_reads += partial
            writes += partial
            vector += (contributions * output.rows * (3 * spec.head_dim + 9)
                       + output.rows * (spec.head_dim + 1))
    resources = {"hbm": producer_reads + vector_reads + writes, "vector": vector}
    if read:
        resources[read] = resources.get(read, 0) + producer_reads
    if vread:
        resources[vread] = resources.get(vread, 0) + vector_reads
    if write and (writes or mte3_extra):
        resources[write] = (resources.get(write, 0) + writes + mte3_extra)
    if include_mte1 and operand:
        resources[operand] = mte1
    return Work(matrix={spec.query_dtype: flops}, resources=resources)


def _placement(context: RecipeContext, core: int, *roles: str) -> dict:
    present = tuple(role for role in roles if role in context.resource_names)
    return context.owned(core, context.core_count, present) if present else {}


def _read_role(context: RecipeContext, *, vector=False) -> str:
    return read_role(context.resource_names, aiv=0 if vector else None)


def _write_role(context: RecipeContext) -> str | None:
    return "mte3" if "mte3" in context.resource_names else None


def _operand_role(context: RecipeContext) -> str | None:
    return "mte1" if "mte1" in context.resource_names else None


def _allocation(context: RecipeContext, core: int, pool: str, name: str,
                size: float) -> Allocation | None:
    if pool not in context.pool_names:
        return None
    if size <= 0:
        return None
    capacity = context.pool_capacities.get(pool)
    if capacity is not None and size > capacity:
        raise ModelingError(
            f"attention_decode_buffer_exceeds_pool:{pool}:{name}:{size}>{capacity}")
    units = unit_group(core, context.core_count,
                       context.pool_units.get(pool, 1))
    return Allocation(name, context.pool_names[pool], float(size), units)


def _allocations(items) -> tuple[Allocation, ...]:
    return tuple(item for item in items if item is not None)


def _tile_counts(output: _Output, context: int, group: int,
                 tile_start: int, tile_end: int) -> tuple[int, int]:
    """Return (valid query/key pairs, active query rows) for one key tile."""
    pairs = row_count = 0
    for token in range(output.token_first, output.token_last + 1):
        rows = _rows_per_token(output, group, token)
        if not rows:
            continue
        valid = max(0, min(tile_end, context + token + 1) - tile_start)
        if valid:
            pairs += rows * valid
            row_count += rows
    return pairs, row_count


@dataclass(frozen=True)
class _AttentionPipelineTemplate(PipelineTemplate):
    """Attention slot buffers are tile-scoped in the finite phase table.

    The detailed producer already names slot buffers ``..._s0/_s1`` and the
    Engine scopes them per tile.  Marking them ``panel`` preserves that
    identity for both residue stages without inferring ownership from buffer
    names.
    """

    @property
    def allocation_bindings(self):
        rows = []
        seen = set()
        stages = []
        for phase in self.phases:
            stages.extend(phase.stages)
            for group in phase.residue_stages:
                stages.extend(group)
        for stage in stages:
            for allocation, action in (
                    *((item, "acquire") for item in stage.acquire),
                    *((item, "free") for item in stage.free)):
                key = (allocation.name, action)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "name": allocation.name, "scope": "panel",
                    "owner": self.owner, "ordinal": 0,
                    "pool": allocation.pool, "size": allocation.size,
                    "units": tuple(allocation.units),
                    "action": action,
                })
        return tuple(rows)


def _fold_interior_phase(phase: PipelinePhase) -> PipelinePhase:
    """Return one interior key-tile phase unchanged.

    Summing repeated work into one task per stage would collapse the slot
    window's overlap into a single stream. Keep each phase intact so elapsed
    service and buffer lifetimes follow the declared pipeline schedule.
    """
    return phase


def _attention_pipeline_template(tile, key_of, count, *, rows, key_tile,
                                 slots, core):
    """Group the real key-tile stream into finite head/full/tail phases.

    ``full`` carries one Stage set per slot residue; its repeated body is
    selected by absolute tile index, so a full phase never replays slot0
    acquire/free on slot1.  Head and partial tail stages remain explicit.
    """
    if count <= 0:
        return None
    slots = max(1, int(slots))
    keys = [tuple(key_of(index)) for index in range(count)]

    tail_start = count
    while tail_start > 0 and keys[tail_start - 1][0] < key_tile:
        tail_start -= 1
    if tail_start == 0:
        return None

    full_start = 0
    periodic = False
    while full_start < tail_start:
        if tail_start - full_start >= slots and all(
                keys[index] == keys[index + slots]
                for index in range(full_start, tail_start - slots)):
            periodic = True
            break
        full_start += 1
    if not periodic:
        return None
    full_end = tail_start
    full_count = full_end - full_start
    if full_count < slots or full_count % slots:
        return None
    base_stages = tuple(tile(full_start))
    for offset in range(1, slots):
        other = tuple(tile(full_start + offset))
        if len(other) != len(base_stages):
            return None
        for left, right in zip(base_stages, other):
            if (left.name, left.work, left.caps, left.coupled,
                    left.latency, left.after, left.previous,
                    left.placement) != (
                    right.name, right.work, right.caps, right.coupled,
                    right.latency, right.after, right.previous,
                    right.placement):
                # The consumer selects phase.stages, so every residue must
                # carry the same physical work and dependency ordering.
                return None

    phases = []

    def add_run(kind, start, end):
        phases.append(PipelinePhase(
            kind, tuple(tile(start)), end - start,
            key=keys[start], end=end))

    index = 0
    while index < full_start:
        end = index + 1
        while end < full_start and keys[end] == keys[index]:
            end += 1
        add_run("head", index, end)
        index = end
    if full_count:
        phases.append(PipelinePhase(
            "full", tuple(tile(full_start)), full_count,
            key=(*keys[full_start][:3], None),
            residue_stages=tuple(
                tuple(tile(full_start + ((offset - full_start) % slots)))
                for offset in range(slots)),
            end=full_end))
    index = full_end
    while index < count:
        end = index + 1
        while end < count and keys[end] == keys[index]:
            end += 1
        add_run("tail", index, end)
        index = end
    if sum(phase.count for phase in phases) != count:
        raise ModelingError("attention_decode_phase_count_mismatch")
    phases = [_fold_interior_phase(phase) for phase in phases]
    return _AttentionPipelineTemplate(core, slots, tuple(phases))


@dataclass(frozen=True)
class _MergeStream:
    """Serialize per-owner merges; each starts only after its split barrier."""

    context: RecipeContext
    recipe: StagedGQADecode
    spec: _Spec
    tasks: tuple[tuple[int, _Output], ...]
    barriers: Mapping[int, object]
    core: int = 0

    def start(self, engine, name, ready, *, owner=""):
        if not self.tasks:
            return ready
        complete = engine.future(name + ":complete")
        cursor = [0]

        def advance(predecessor):
            index = cursor[0]
            if index >= len(self.tasks):
                engine.resolve(complete)
                return
            cursor[0] += 1
            output_index, output = self.tasks[index]
            barrier = self.barriers[output_index]
            gate = engine.submit(
                f"{name}:gate{index}", after=(predecessor, barrier), owner=owner,
            )

            def launch(_):
                program = _merge_program(self.context, self.spec, output,
                                         self.core)
                end = program.start(engine, f"{name}:merge{index}::task",
                                    gate, owner=owner)
                engine.submit(
                    f"{name}:next{index}", after=(end,), owner=owner,
                    on_complete=lambda _: advance(end),
                )

            engine.submit(f"{name}:start{index}", after=(gate,),
                          on_complete=launch, owner=owner)

        advance(ready)
        return complete


def _merge_program(context: RecipeContext, spec: _Spec,
                   output: _Output, core: int) -> KernelProgram:
    dim, rows = spec.head_dim, output.rows
    contributions = len(output.contributions)
    partial = contributions * rows * _partial_row_bytes(dim)
    out_bytes = rows * dim * spec.query_bytes
    read = _read_role(context, vector=True)
    write = _write_role(context)
    ub_partial = _allocation(context, core, "ub", "merge_partial",
                             rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES))
    ub_out = _allocation(context, core, "ub", "merge_out",
                         rows * dim * _FP32_BYTES)
    buffers = _allocations((ub_partial, ub_out))
    return KernelProgram((
        Stage("reserve", acquire=buffers),
        Stage("merge_load", context.bind(Work(resources={
            "hbm": partial, read: partial,
        })), after=("reserve",), placement=_placement(context, core, read)),
        Stage("merge", context.bind(Work(resources={
            "vector": contributions * rows * (3 * dim + 9),
        })), after=("merge_load",),
              placement=_placement(context, core, "vector")),
        Stage("normalize", context.bind(Work(resources={"vector": rows * (dim + 1)})),
              after=("merge",), placement=_placement(context, core, "vector")),
        Stage("merge_store", context.bind(Work(resources={
            "hbm": out_bytes, "mte3": out_bytes,
        })), after=("normalize",), free=buffers,
              placement=_placement(context, core, write, "vector")),
    ))
