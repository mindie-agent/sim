"""Staged analytic GQA decode on the common finite resource engine.

[Role]
  Lower the declared ``metadata['gqa_decode']`` geometry into finite
  ``(batch, kv_head, query_tile, kv_split)`` owners.  Each owner loads a
  resident query tile, streams bounded key/value tiles through
  QK -> softmax -> PV -> online accumulation, and stores a partial or final
  output.  Split partials are merged by a separate, barrier-gated program.

[Invariants]
  * Real request lengths and the causal triangular tail determine work; no
    padded cache capacity or query-head multiplier is applied to KV.
  * One key tile is loaded once per contributing split and query tile.  K/V
    traffic is never multiplied by the GQA query-head group.
  * QK and PV together are ``4 * pairs * query_heads * head_dim`` FLOPs.
  * A split partial is materialized only when at least two split producers
    contribute to one output.  One contributing split writes the final output
    directly.  The final output is written exactly once.
  * On-chip operands, scores, probabilities and accumulators use advertised
    pools when present and are released by the owning task.  They are never
    added to HBM traffic.
  * Owners are physical placements; their task streams cannot borrow idle units.

[Timeouts & Locks]
  Planning and lowering are pure arithmetic.  Key tiles are generated lazily
  through a bounded pipeline window; no per-tile list is materialized.

[Side-Effect Policy]
  No device runtimes, measurements, files or external I/O.  Programs submit
  only to the caller's virtual ``Engine``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import (
    KernelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

IMPLEMENTATION = "staged_gqa_decode"
RECIPE_ID = IMPLEMENTATION

# Mechanism-only references.  These pin the source features being represented;
# they are not a claim of native instruction, compiler or dispatch fidelity.
SOURCE_REFERENCES = (
    "/Users/maoxx241/code/refs/ops-transformer/attention/"
    "fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h",
    "/Users/maoxx241/code/refs/ops-transformer/attention/"
    "fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h",
)

_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
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
class StagedGQADecode:
    """Finite analytic GQA decode recipe.

    ``key_tile`` is the K/V row tile.  ``query_tile`` bounds the flattened
    ``(token, query_head_in_group)`` rows in one owner task.  ``kv_splits`` is
    an explicit requested partition count, reduced to real key tiles.
    ``slots`` is the bounded key/value pipeline window.
    Optional ``target_cores`` increases partitions to fill the declared owner
    budget after batch/head/query parallelism. It is an analytic mapping;
    captured dispatch should continue to declare its explicit split count.
    ``kv_mapping='fai_flash_decode'`` instead uses the pinned FAI host budget
    across query boundaries and a global producer/merge barrier. Device stages
    remain analytical; selecting host ownership does not select a native kernel.
    """

    key_tile: int = 128
    query_tile: int = 64
    kv_splits: int = 1
    slots: int = 2
    target_cores: int | None = None
    kv_mapping: str = "uniform"

    def __post_init__(self):
        _positive_int(self.key_tile, "key_tile")
        _positive_int(self.query_tile, "query_tile")
        _positive_int(self.kv_splits, "kv_splits")
        _positive_int(self.slots, "slots")
        if self.target_cores is not None:
            _positive_int(self.target_cores, "target_cores")
        if self.kv_mapping not in {"uniform", "fai_flash_decode"}:
            raise ModelingError("attention_decode_unknown_kv_mapping")
        if self.kv_mapping == "fai_flash_decode" and self.kv_splits != 1:
            raise ModelingError("fai_decode_mapping_owns_splits:remove_kv_splits")

    def spec(self, op: OpSpec) -> "_Spec":
        return _spec(op, self)

    def work(self, op: OpSpec, *, include_mte1: bool = False) -> Work:
        """Physical work of the lowered mechanism, independent of service rates."""
        return _total_work(self.spec(op), include_mte1=include_mte1)

    def build(self, op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
        """Verification binding using this recipe's options and work ledger."""
        spec = self.spec(op)
        mte1 = geometry is not None and "mte1" in geometry.resource_units
        return OperatorPlan(
            self.work(op, include_mte1=mte1),
            lambda context: _GQADecodeProgram(context, self, spec),
        )

    def lower(self, context: RecipeContext) -> KernelProgram:
        """Lower one kernel setup and the finite staged body."""
        spec = self.spec(context.op)
        return KernelProgram(
            (
                Stage("setup", latency=context.setup_ms),
                Stage("body", after=("setup",),
                      program=_GQADecodeProgram(context, self, spec)),
            ),
            minimum_ms=context.minimum_ms,
        )


@dataclass(frozen=True)
class _Contribution:
    split: int
    tile_start: int
    tile_count: int
    owner: int | None = None


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

    @property
    def group(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def query_bytes(self) -> int:
        return int(dtype_bytes(self.query_dtype))

    @property
    def kv_bytes(self) -> int:
        return int(dtype_bytes(self.kv_dtype))


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
    if recipe.kv_mapping == "fai_flash_decode":
        from .fai_decode_mapping import partitions
        mapping = partitions(requests, heads, kv_heads, recipe.target_cores,
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


def _total_work(spec: _Spec, *, include_mte1: bool) -> Work:
    reads = writes = vector = mte1 = 0
    flops = 0
    for output in spec.outputs:
        _query, context = spec.requests[output.request]
        pairs = _output_pairs(output, context, spec.group)
        row_tiles = _output_row_tiles(output, context, spec.group, spec.key_tile)
        contributions = len(output.contributions)
        q_bytes = contributions * output.rows * spec.head_dim * spec.query_bytes
        kv_bytes = (output.horizon * 2 * spec.head_dim * spec.kv_bytes)
        output_bytes = output.rows * spec.head_dim * spec.query_bytes
        partial = contributions * output.rows * _partial_row_bytes(spec.head_dim)
        reads += q_bytes + kv_bytes
        writes += output_bytes
        if include_mte1:
            mte1 += q_bytes + kv_bytes
        flops += 4 * pairs * spec.head_dim
        vector += 7 * pairs + (3 * spec.head_dim + 10) * row_tiles
        if include_mte1:
            # Each probability tile becomes the left operand of PV. It never
            # leaves the on-chip hierarchy or becomes an HBM read.
            mte1 += pairs * spec.query_bytes
        if contributions == 1:
            vector += output.rows * (spec.head_dim + 1)
        else:
            reads += partial
            writes += partial
            vector += (contributions * output.rows * (3 * spec.head_dim + 9)
                       + output.rows * (spec.head_dim + 1))
    resources = {
        "hbm": reads + writes,
        "mte2": reads,
        "mte3": writes,
        "vector": vector,
    }
    if include_mte1:
        resources["mte1"] = mte1
    return Work(matrix={spec.query_dtype: flops}, resources=resources)


def build(op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
    """Independent verification binding for the default recipe options."""
    return StagedGQADecode().build(op, geometry)


# ---------------------------------------------------------------------------
# Placement, buffer and tile helpers.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Executable programs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProducerTaskProgram:
    context: RecipeContext
    recipe: StagedGQADecode
    spec: _Spec
    output: _Output
    contribution: _Contribution
    core: int
    signal: Callable[[], None] | None = None

    def start(self, engine, name, ready, *, owner=""):
        program = _producer_program(
            self.context, self.recipe, self.spec, self.output,
            self.contribution, self.core,
        )
        end = program.start(engine, name, ready, owner=owner)
        if self.signal is None:
            return end
        return engine.submit(
            name + ":split_produced", after=(end,),
            on_complete=lambda _: self.signal(), owner=owner,
        )


def _producer_program(context: RecipeContext, recipe: StagedGQADecode,
                      spec: _Spec, output: _Output,
                      contribution: _Contribution, core: int) -> KernelProgram:
    _query, context_len = spec.requests[output.request]
    dim, rows = spec.head_dim, output.rows
    q_bytes = rows * dim * spec.query_bytes
    read = _read_role(context)
    write = _write_role(context)
    operand = _operand_role(context)
    cube = ("cube",)
    vector = ("vector",)
    copy_in = (read,)
    copy_out = (write,) if write else ()
    operand_role = (operand,) if operand else ()

    q_l1 = _allocation(context, core, "l1", "q_l1", q_bytes)
    q_l0a = _allocation(context, core, "l0a", "q_l0a", q_bytes)
    ub_state = _allocation(context, core, "ub", "online_state",
                           rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES))
    persistent = _allocations((q_l1, q_l0a, ub_state))

    q_work = {"hbm": q_bytes, read: q_bytes}
    if operand:
        q_work["mte1"] = q_bytes
    q_place = {**_placement(context, core, *copy_in),
               **_placement(context, core, *operand_role)}
    stages = [
        Stage("q_load", context.bind(Work(resources=q_work)),
              acquire=persistent, placement=q_place),
    ]

    max_tile = min(recipe.key_tile, max(1, output.horizon))

    def tile(index: int):
        start = (contribution.tile_start + index) * recipe.key_tile
        end = min(start + recipe.key_tile, output.horizon)
        valid = max(0, end - start)
        pairs, active_rows = _tile_counts(
            output, context_len, spec.group, start, end)
        k_bytes = valid * dim * spec.kv_bytes
        slot = index % recipe.slots
        suffix = f"s{slot}"
        k_l1 = _allocation(context, core, "l1", f"k_l1_{suffix}", k_bytes)
        k_l0b = _allocation(context, core, "l0b", f"k_l0b_{suffix}", k_bytes)
        scores = _allocation(context, core, "l0c", f"scores_{suffix}",
                             rows * max_tile * _FP32_BYTES)
        probs = _allocation(context, core, "ub", f"probs_{suffix}",
                            rows * max_tile * _FP32_BYTES)
        p_l0a = _allocation(context, core, "l0a", f"p_l0a_{suffix}",
                           rows * valid * spec.query_bytes)
        v_l1 = _allocation(context, core, "l1", f"v_l1_{suffix}", k_bytes)
        v_l0b = _allocation(context, core, "l0b", f"v_l0b_{suffix}", k_bytes)
        pv_acc = _allocation(context, core, "l0c", f"pv_acc_{suffix}",
                             rows * dim * _FP32_BYTES)
        k_acquire = _allocations((k_l1, k_l0b))
        score_acquire = _allocations((scores,))
        prob_acquire = _allocations((probs,))
        v_acquire = _allocations((v_l1, v_l0b, pv_acc))
        k_release = _allocations((k_l1, k_l0b))
        score_release = _allocations((scores,))
        v_release = _allocations((v_l1, v_l0b, pv_acc))
        prob_release = _allocations((probs,))

        load_work = {"hbm": k_bytes, read: k_bytes}
        if operand:
            load_work["mte1"] = k_bytes
        qk = Work(
            matrix={spec.query_dtype: 2 * dim * pairs},
            resources={"vector": 2 * pairs},
        )
        softmax = Work(resources={"vector": 5 * pairs + active_rows})
        vload_work = {"hbm": k_bytes, read: k_bytes}
        if operand:
            vload_work["mte1"] = k_bytes
        pv = Work(matrix={spec.query_dtype: 2 * dim * pairs})
        # Combine the tile-local numerator with the running numerator using
        # their maxima: two scalings plus one add per output element.
        online = Work(resources={"vector": active_rows * (3 * dim + 9)})
        return (
            Stage("k_load", context.bind(Work(resources=load_work)),
                  acquire=k_acquire,
                  placement={**_placement(context, core, *copy_in),
                             **_placement(context, core, *operand_role)}),
            Stage("qk", context.bind(qk), after=("k_load",),
                  acquire=score_acquire, free=k_release,
                  placement={**_placement(context, core, *cube),
                             **_placement(context, core, *vector)}),
            Stage("softmax", context.bind(softmax), after=("qk",),
                  acquire=prob_acquire, free=score_release,
                  placement=_placement(context, core, *vector)),
            Stage("v_load", context.bind(Work(resources=vload_work)),
                  after=("softmax",), acquire=v_acquire,
                  placement={**_placement(context, core, *copy_in),
                             **_placement(context, core, *operand_role)}),
            Stage("probability_operand", context.bind(Work(resources=(
                      {"mte1": pairs * spec.query_bytes} if operand else {}))),
                  after=("softmax",), acquire=_allocations((p_l0a,)),
                  placement=_placement(context, core, *operand_role)),
            Stage("pv", context.bind(pv), after=("v_load", "probability_operand"),
                  free=v_release + _allocations((p_l0a,)),
                  placement=_placement(context, core, *cube)),
            Stage("online_update", context.bind(online), after=("pv",),
                  free=prob_release, previous=("online_update",),
                  placement=_placement(context, core, *vector)),
        )

    stages.append(Stage(
        "key_tiles", after=("q_load",),
        program=PipelineProgram(contribution.tile_count, recipe.slots, tile),
    ))

    contributions = len(output.contributions)
    if contributions == 1:
        out_bytes = rows * dim * spec.query_bytes
        final = Work(resources={
            "hbm": out_bytes, "mte3": out_bytes,
            "vector": rows * (dim + 1),
        })
        stages.append(Stage(
            "final_store", after=("key_tiles",),
            work=context.bind(final), free=persistent,
            placement={**_placement(context, core, *copy_out),
                       **_placement(context, core, *vector)},
        ))
    else:
        partial_bytes = rows * _partial_row_bytes(dim)
        stages.append(Stage(
            "partial_store", after=("key_tiles",),
            work=context.bind(Work(resources={
                "hbm": partial_bytes, "mte3": partial_bytes,
            })),
            free=persistent,
            placement=_placement(context, core, *copy_out),
        ))
    return KernelProgram(tuple(stages))


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
    def part(_):
        return (
            Stage("merge_load", context.bind(Work(resources={
                "hbm": partial / contributions, read: partial / contributions,
            })), placement=_placement(context, core, read)),
            Stage("merge", context.bind(Work(resources={
                "vector": rows * (3 * dim + 9),
            })), after=("merge_load",),
                  placement=_placement(context, core, "vector")),
        )
    return KernelProgram((
        Stage("reserve", acquire=buffers),
        Stage("parts", after=("reserve",),
              program=PipelineProgram(contributions, 1, part)),
        Stage("normalize", context.bind(Work(resources={"vector": rows * (dim + 1)})),
              after=("parts",), placement=_placement(context, core, "vector")),
        Stage("merge_store", context.bind(Work(resources={
            "hbm": out_bytes, "mte3": out_bytes,
        })), after=("normalize",), free=buffers,
              placement=_placement(context, core, write, "vector")),
    ))


@dataclass(frozen=True)
class _GQADecodeProgram:
    context: RecipeContext
    recipe: StagedGQADecode
    spec: _Spec

    def start(self, engine, name, ready, *, owner=""):
        outputs = self.spec.outputs
        barriers, remaining = {}, {}
        for index, output in enumerate(outputs):
            if len(output.contributions) > 1:
                barriers[index] = engine.future(
                    f"{name}:split_barrier:{index}")
                remaining[index] = len(output.contributions)

        def signal(index):
            remaining[index] -= 1
            if remaining[index] == 0:
                engine.resolve(barriers[index])

        tasks = tuple(
            (output_index, contribution)
            for output_index, output in enumerate(outputs)
            for contribution in output.contributions
        )
        cores = self.context.core_count
        streams = [[] for _ in range(cores)]
        for index, (output_index, contribution) in enumerate(tasks):
            core = index % cores if contribution.owner is None else contribution.owner
            if core >= cores:
                raise ModelingError("attention_decode_mapping_exceeds_hardware_cores")
            streams[core].append((output_index, contribution))
        producer_events = []
        for core, stream in enumerate(streams):
            if not stream:
                continue
            count = len(stream)

            def tile(index: int, core: int = core, stream=tuple(stream)):
                output_index, contribution = stream[index]
                callback = ((lambda index=output_index: signal(index))
                            if output_index in barriers else None)
                return (Stage("task", program=_ProducerTaskProgram(
                    self.context, self.recipe, self.spec,
                    outputs[output_index], contribution, core, callback,
                )),)

            producer_events.append(PipelineProgram(count, 1, tile).start(
                engine, f"{name}:owner{core}:producer", ready,
                owner=f"{owner}:gqa{core}" if owner else f"gqa{core}",
            ))

        merges = tuple((index, output) for index, output in enumerate(outputs)
                       if len(output.contributions) > 1)
        merge_events = []
        merge_ready = (engine.submit(name + ":all_producers", after=tuple(producer_events),
                                     owner=owner)
                       if self.recipe.kv_mapping == "fai_flash_decode" else ready)
        for core in range(min(len(merges), cores)):
            stream = _MergeStream(
                self.context, self.recipe, self.spec,
                merges[core::cores], barriers, core,
            )
            merge_events.append(stream.start(
                engine, f"{name}:owner{core}:merge", merge_ready,
                owner=f"{owner}:gqa_merge{core}" if owner else f"gqa_merge{core}",
            ))
        return engine.submit(
            name + ":complete", after=tuple(producer_events + merge_events),
            owner=owner,
        )


__all__ = [
    "IMPLEMENTATION",
    "RECIPE_ID",
    "SOURCE_REFERENCES",
    "StagedGQADecode",
    "build",
    "causal_pairs",
]
