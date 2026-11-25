"""Staged analytic GQA decode on the common finite resource engine.

[Role] Lower GQA decode to physical owners by request, KV head, query tile
  and KV split. A resident query streams through QK, softmax, PV and update.
[Invariants] Real lengths and causal tails determine work. Each contributing
  split loads K/V once per query tile, shared across its query heads.
  A barrier gates the merge when multiple splits contribute to an output.
  Each output is written once. Owners use their assigned pools and service.
[Timeouts & Locks] Key tiles are generated lazily in a bounded window.
[Side-Effect Policy] Programs submit only to the caller's virtual engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from mindie_sim.prediction.mechanism import (
    Geometry, OperatorPlan, PlannedRecipe, Work,
)
from mindie_sim.prediction.ownership import read_role
from .attention_kv_access import (
    FixpipeHandoffPolicy, KVCacheAccess, gqa_fixpipe_policy, kv_access, paged_fia_plan,
)
from mindie_sim.prediction.program import (
    KernelProgram, PipelineProgram, RecipeContext, Stage,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .attention_pipeline import (
    DECODE_ONLINE_STATE_OPS, DECODE_OUTPUT_UPDATE_OPS_PER_VALUE,
    DECODE_SOFTMAX_OPS_PER_PAIR, _allocation, _allocations,
    _compile_decode_peer_group, _partial_row_bytes, _placement,
    _producer_program, _read_role, _rows_per_token, _write_role,
    decode_handoff_workspace_bytes,
)

IMPLEMENTATION = "staged_gqa_decode"
RECIPE_ID = IMPLEMENTATION

DEFAULT_DECODE_KEY_TILE = 128
DEFAULT_DECODE_QUERY_TILE = 64
FP32_BYTES = 4
_MX_SCALE_GROUP_SIZE = 32
_MX_V_SCALE_TOKEN_GROUP = 32
_QFA_MXFP8_ENTRY = "cann_ops_transformer.ops.quant_flash_attn"
_MXFP8_KV_TAGS = frozenset({"mxfp8", "qk_mxfp8_dynamic_v_mxfp8_per_channel"})

# Mechanism-only references pinning represented source features.
SOURCE_REFERENCES = (
    "repository://refs/ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h",
    "repository://refs/ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h",
)


_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
    # Quantized cache payloads retain their storage dtype.  Conversion and
    # scale work are separate declarations, not a re-labelled BF16 tensor.
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2",
    "mxfp8": "fp8_e4m3", "qk_mxfp8_dynamic_v_mxfp8_per_channel": "fp8_e4m3",
    "int8": "int8", "c8": "int8", "uint8": "uint8", "int4": "int4",
    "fp4": "fp4_e2m1", "fp4_e2m1": "fp4_e2m1",
}


def _resolve_arch35_mxfp8_access(
    raw_kv_dtype: Any, query_dtype: str, kv_dtype: str, dim: int, v_dim: int,
    cache: KVCacheAccess, recipe: StagedGQADecode,
) -> tuple[str, KVCacheAccess, bool]:
    """Use direct PA_NZ MXFP8 Cube MMA without AIV BF16 dequant writeback on Arch35."""
    raw_norm = str(raw_kv_dtype or "").strip().lower()
    is_mxfp8_kv = (
        raw_norm in _MXFP8_KV_TAGS
        or (recipe.kv_mapping == "fai_host_arch35" and kv_dtype == "fp8_e4m3"
            and cache.scale_scope == "row" and cache.key_scale_row > 0)
    )
    if not is_mxfp8_kv:
        return query_dtype, cache, False
    k_row_bytes = dim + _ceil_div(dim, _MX_SCALE_GROUP_SIZE)
    direct = replace(
        cache,
        key_payload_row=k_row_bytes,
        value_payload_row=v_dim,
        key_operand_row=k_row_bytes,
        value_operand_row=v_dim,
        key_scale_row=0,
        value_scale_row=0,
        scale_scope="none",
        dequant_ops_per_element=0,
        key_needs_conversion=False,
        value_needs_conversion=False,
        query_dtype="fp8_e4m3",
    )
    return "fp8_e4m3", direct, True


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
    v_head_dim: int = 0
    output_dtype: str = ""
    mxfp8_qfa: bool = False

    @property
    def group(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def query_bytes(self) -> int:
        return int(dtype_bytes(self.query_dtype))

    @property
    def output_bytes(self) -> int:
        return int(dtype_bytes(self.output_dtype or self.query_dtype))


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
    v_dim = _positive_int(raw.get("v_head_dim", dim), "v_head_dim")
    if heads % kv_heads:
        raise ModelingError("attention_decode_query_heads_not_divisible_by_kv_heads")
    query_dtype = _dtype(raw.get("query_dtype"), "query_dtype")
    kv_dtype = _dtype(raw.get("kv_dtype"), "kv_dtype")
    view = metadata.get("attention_tensor_view")
    raw_output_dtype = raw.get("output_dtype")
    if raw_output_dtype is None and isinstance(view, Mapping):
        raw_output_dtype = view.get("output_dtype")
    output_dtype = (
        _dtype(raw_output_dtype, "output_dtype")
        if raw_output_dtype is not None else query_dtype
    )
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
        active_indices = [idx for idx, (q, _) in enumerate(requests) if q > 0]
        active_requests = tuple(requests[idx] for idx in active_indices)
        if active_requests:
            from .fai_decode_mapping import (
                host_partitions,
                host_partitions_arch35,
                partitions,
            )
            if recipe.kv_mapping == "fai_host_arch35":
                raw_mapping = host_partitions_arch35(
                    active_requests, heads, kv_heads, recipe.target_cores,
                    key_tile=recipe.key_tile, query_tile=recipe.query_tile,
                    head_dim=dim,
                )
            else:
                mapper = host_partitions if recipe.kv_mapping == "fai_host" else partitions
                raw_mapping = mapper(active_requests, heads, kv_heads, recipe.target_cores,
                                     key_tile=recipe.key_tile, query_tile=recipe.query_tile)
            mapping = {
                (active_indices[active_pos], kv_head): spans
                for (active_pos, kv_head), spans in raw_mapping.items()
            }
            outputs = [replace(output, contributions=tuple(
                _Contribution(index, start, end - start, owner)
                for index, (owner, start, end) in enumerate(
                    mapping[output.request, output.kv_head]))) for output in outputs]
            splits = max((len(output.contributions) for output in outputs), default=1)
    access = kv_access(
        op, query_dtype=query_dtype, kv_dtype=kv_dtype,
        head_dim=dim, v_head_dim=v_dim,
    )
    effective_query_dtype, access, is_mxfp8_qfa = _resolve_arch35_mxfp8_access(
        raw.get("kv_dtype"), query_dtype, kv_dtype, dim, v_dim, access, recipe,
    )
    return _Spec(
        requests, heads, kv_heads, dim, effective_query_dtype, kv_dtype,
        recipe.key_tile, recipe.query_tile, splits, recipe.slots,
        tuple(outputs),
        access,
        v_head_dim=v_dim,
        output_dtype=output_dtype,
        mxfp8_qfa=is_mxfp8_qfa,
    )


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
                operand: str | None = "mte1",
                handoff: FixpipeHandoffPolicy | None = None) -> Work:
    cache = spec.kv_cache
    vread = vread or read
    producer_reads = vector_reads = writes = mte3_extra = vector = mte1 = 0
    handoff_bytes = 0
    flops = 0
    for output in spec.outputs:
        _query, context = spec.requests[output.request]
        pairs = _output_pairs(output, context, spec.group)
        row_tiles = _output_row_tiles(output, context, spec.group, spec.key_tile)
        contributions = len(output.contributions)
        q_bytes = contributions * output.rows * spec.head_dim * spec.query_bytes
        key_payload = output.horizon * cache.key_payload_row
        value_payload = output.horizon * cache.value_payload_row
        v_dim = spec.v_head_dim or spec.head_dim
        if spec.mxfp8_qfa:
            # Per-channel E8M0 V scales are broadcast per 32-token group across all v_dim channels.
            value_payload += sum(
                _ceil_div(
                    max(0, min((c.tile_start + c.tile_count) * spec.key_tile, output.horizon)
                        - c.tile_start * spec.key_tile),
                    _MX_V_SCALE_TOKEN_GROUP,
                ) * v_dim
                for c in output.contributions
            )
        output_bytes = output.rows * v_dim * spec.output_bytes
        partial = contributions * output.rows * _partial_row_bytes(v_dim)
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
        # Probabilities written from UB to L1 before the MTE1 PV operand copy.
        mte3_extra += pairs * spec.query_bytes
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
        flops += 2 * pairs * (spec.head_dim + v_dim)
        handoff_bytes += (pairs + row_tiles * v_dim) * FP32_BYTES
        vector += (2 + DECODE_SOFTMAX_OPS_PER_PAIR) * pairs + (
            DECODE_OUTPUT_UPDATE_OPS_PER_VALUE * v_dim + DECODE_ONLINE_STATE_OPS + 1
        ) * row_tiles
        vector += cache.conversion_ops(output.horizon)
        if include_mte1:
            # Each probability tile becomes the left operand of PV. It never
            # leaves the on-chip hierarchy or becomes an HBM read.
            mte1 += pairs * spec.query_bytes
        if contributions == 1:
            vector += output.rows * (v_dim + 1)
        else:
            vector_reads += partial
            writes += partial
            vector += (contributions * output.rows * (DECODE_OUTPUT_UPDATE_OPS_PER_VALUE * v_dim + DECODE_ONLINE_STATE_OPS)
                       + output.rows * (v_dim + 1))
    resources = {"hbm": producer_reads + vector_reads + writes, "vector": vector}
    if read:
        resources[read] = resources.get(read, 0) + producer_reads
    if vread:
        resources[vread] = resources.get(vread, 0) + vector_reads
    if write and (writes or mte3_extra):
        resources[write] = (resources.get(write, 0) + writes + mte3_extra)
    if include_mte1 and operand:
        resources[operand] = mte1
    if handoff is not None and handoff.enabled and handoff_bytes > 0:
        resources["fixpipe"] = resources.get("fixpipe", 0) + handoff_bytes
        if handoff.via_gm:
            if vread:
                resources[vread] = resources.get(vread, 0) + handoff_bytes
            if handoff.mem_tier:
                resources[handoff.mem_tier] = resources.get(handoff.mem_tier, 0) + 2 * handoff_bytes
    return Work(matrix={spec.query_dtype: flops}, resources=resources)


@dataclass(frozen=True)
class _StreamBinding:
    owners: Any
    datapath: Any
    kwargs: Mapping[str, Any]

    def bind(self, program):
        from mindie_sim.prediction.streams import apply_stream_owners_to_program
        return apply_stream_owners_to_program(
            program, self.owners, self.datapath, **self.kwargs,
        )


@dataclass(frozen=True)
class _MergeStream:
    """Serialize per-owner merges; each starts only after its split barrier."""

    context: RecipeContext
    spec: _Spec
    tasks: tuple[tuple[int, _Output], ...]
    barriers: Mapping[int, object]
    core: int = 0
    stream_binding: _StreamBinding | None = None

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
                if self.stream_binding is not None:
                    program = self.stream_binding.bind(program)
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
    dim, rows = spec.v_head_dim or spec.head_dim, output.rows
    contributions = len(output.contributions)
    partial = contributions * rows * _partial_row_bytes(dim)
    out_bytes = rows * dim * spec.output_bytes
    read = _read_role(context, vector=True)
    write = _write_role(context)
    ub_partial = _allocation(context, core, "ub", "merge_partial",
                             rows * _partial_row_bytes(dim))
    ub_out = _allocation(context, core, "ub", "merge_out",
                         rows * dim * FP32_BYTES)
    buffers = _allocations((ub_partial, ub_out))
    return KernelProgram((
        Stage("reserve", acquire=buffers),
        Stage("merge_load", context.bind(Work(resources={
            "hbm": partial, read: partial,
        })), after=("reserve",), placement=_placement(context, core, read)),
        Stage("merge", context.bind(Work(resources={
            "vector": contributions * rows * (DECODE_OUTPUT_UPDATE_OPS_PER_VALUE * dim + DECODE_ONLINE_STATE_OPS),
        })), after=("merge_load",),
              placement=_placement(context, core, "vector")),
        Stage("normalize", context.bind(Work(resources={"vector": rows * (dim + 1)})),
              after=("merge",), placement=_placement(context, core, "vector")),
        Stage("merge_store", context.bind(Work(resources={
            "hbm": out_bytes, **({write: out_bytes} if write else {}),
        })), after=("normalize",), free=buffers,
              placement=_placement(context, core, *(([write] if write else ())))),
    ))


@dataclass(frozen=True)
class StagedGQADecode:
    """Bound the K/V tiles, flattened query rows and live pipeline slots.

    ``kv_splits`` requests partitions over real key tiles. ``target_cores``
    can increase partitions after batch/head/query parallelism. FAI mappings
    use the source host budget across query tiles and gate merges on all
    producers. ``fai_host`` also handles speculative queries for which the
    source occupancy threshold disables flash splitting.
    """

    key_tile: int = DEFAULT_DECODE_KEY_TILE
    query_tile: int = DEFAULT_DECODE_QUERY_TILE
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
        if self.kv_mapping not in {"uniform", "fai_flash_decode", "fai_host", "fai_host_arch35"}:
            raise ModelingError("attention_decode_unknown_kv_mapping")
        if self.kv_mapping in {"fai_flash_decode", "fai_host", "fai_host_arch35"} and self.kv_splits != 1:
            raise ModelingError("fai_decode_mapping_owns_splits:remove_kv_splits")

    def spec(self, op: OpSpec) -> "_Spec":
        return _spec(op, self)

    def work(self, op: OpSpec, *, include_mte1: bool = False,
             read: str = "mte2", vread: str | None = None,
             write: str | None = "mte3", operand: str | None = "mte1",
             handoff: FixpipeHandoffPolicy | None = None) -> Work:
        """Physical work of the lowered mechanism, independent of service rates.

        ``read``/``vread`` are the actual producer and vector read pipes.  The
        default generic pair keeps the historical verification ledger; a
        bound geometry supplies the split AIC/AIV pipe names.
        """
        return _total_work(self.spec(op), include_mte1=include_mte1,
                           read=read, vread=vread, write=write, operand=operand,
                           handoff=handoff)

    def kv_access(self, op: OpSpec) -> KVCacheAccess:
        """Declared KV storage/page/scale access for this operator geometry."""
        return self.spec(op).kv_cache

    def native_plan(self, op: OpSpec) -> dict | None:
        """Typed ordinary paged-FIA ports and geometry for the actual op."""
        spec = self.spec(op)
        cache = spec.kv_cache
        if not cache.paged or cache.key_format is None or cache.value_format is None:
            return None
        active_indices = [idx for idx, (q, _) in enumerate(spec.requests) if q > 0]
        if not active_indices:
            return None
        active_map = {orig: compact for compact, orig in enumerate(active_indices)}
        active_requests = tuple(spec.requests[idx] for idx in active_indices)
        v_dim = spec.v_head_dim or spec.head_dim
        plan = paged_fia_plan(
            requests=active_requests,
            query_lengths=tuple(query for query, _ in active_requests),
            kv_lengths=tuple(context + query
                             for query, context in active_requests),
            page_size=cache.page_size,
            query_heads=spec.query_heads,
            kv_heads=spec.kv_heads,
            head_dim=spec.head_dim,
            v_head_dim=v_dim,
            shared_format=cache.shared_row,
            query_dtype=spec.query_dtype,
            output_dtype=spec.output_dtype or spec.query_dtype,
            key_format=cache.key_format,
            value_format=cache.value_format,
            phase=getattr(op, "phase", "decode"),
            mask="causal",
            return_lse=False,
            output_prepare=None,
        )
        if spec.mxfp8_qfa:
            q_total = sum(query for query, _ in active_requests)
            kv_lens = tuple(context + query for query, context in active_requests)
            total_pages = plan["geometry"]["max_pages"] if len(active_requests) == 1 else sum(
                plan["geometry"]["page_counts"]
            )
            k_scale_groups = _ceil_div(spec.head_dim, _MX_SCALE_GROUP_SIZE)
            v_scale_groups = max(1, cache.page_size // _MX_V_SCALE_TOKEN_GROUP)
            plan["kind"] = "paged_qfa_mxfp8"
            plan["abi"] = {
                "namespace": "cann_ops_transformer",
                "entry": _QFA_MXFP8_ENTRY,
                "aclnn_symbol": None,
                "source_state": "vllm_ascend_attention_c8_mxfp",
                "binary_identity": "unresolved",
                "basis": "arch35_mxfp8_quant_flash_attn_pa_nz",
            }
            plan["layouts"]["key"] = "PA_NZ"
            plan["layouts"]["value"] = "PA_NZ"
            plan["ports"]["key_cache"]["layout"] = "PA_NZ"
            plan["ports"]["value_cache"]["layout"] = "PA_NZ"
            plan["ports"]["query_scale"] = {
                "role": "query_scale", "direction": "in",
                "shape": (q_total, spec.query_heads, k_scale_groups),
                "dtype": "fp8_e8m0", "layout": "TND", "required": True,
            }
            plan["ports"]["key_scale"] = {
                "role": "key_scale", "direction": "in",
                "shape": (total_pages, cache.page_size, spec.kv_heads, k_scale_groups),
                "dtype": "fp8_e8m0", "layout": "PA_NZ", "required": True,
            }
            plan["ports"]["value_scale"] = {
                "role": "value_scale", "direction": "in",
                "shape": (total_pages, v_scale_groups, spec.kv_heads, v_dim),
                "dtype": "fp8_e8m0", "layout": "PA_NZ", "required": True,
            }
            kv_scale_bytes = sum(
                kv * spec.kv_heads * k_scale_groups
                + _ceil_div(kv, _MX_V_SCALE_TOKEN_GROUP) * spec.kv_heads * v_dim
                for kv in kv_lens
            )
            plan["work"]["kv_scale_bytes"] = kv_scale_bytes
            plan["work"]["kv_valid_bytes"] += kv_scale_bytes
        outputs = []
        full_tiles = tail_tiles = 0
        for output in spec.outputs:
            for contribution in output.contributions:
                last_start = (contribution.tile_start
                              + contribution.tile_count - 1) * spec.key_tile
                last_valid = min(spec.key_tile,
                                 output.horizon - last_start)
                for tile in range(contribution.tile_count):
                    start = (contribution.tile_start + tile) * spec.key_tile
                    valid = min(spec.key_tile, output.horizon - start)
                    if valid < spec.key_tile:
                        tail_tiles += 1
                    else:
                        full_tiles += 1
                outputs.append({
                    "request": active_map[output.request], "kv_head": output.kv_head,
                    "tiles": contribution.tile_count,
                    "tile_start": contribution.tile_start,
                    "last_valid": last_valid,
                    "tail": last_valid < spec.key_tile,
                })
        plan["strategy"] = {
            "key_tile": spec.key_tile,
            "query_tile": spec.query_tile,
            "kv_splits": spec.kv_splits,
            "slots": spec.slots,
            "key_tiles": full_tiles + tail_tiles,
            "full_tiles": full_tiles,
            "tail_tiles": tail_tiles,
            "outputs": outputs,
            "transactions": {
                "page_reads": plan["work"]["pages"],
                "key_page_reads": plan["work"]["key_page_reads"],
                "value_page_reads": plan["work"]["value_page_reads"],
                "static_kv_read": True,
                "state_writes": 0,
            },
            "compute": {"qk": "fia", "softmax": "online",
                        "pv": "fia", "causal": True},
        }
        plan["cache_access"] = cache.report()
        plan["operator"] = {"op_id": op.op_id, "stage": op.stage,
                            "layer": op.layer, "phase": op.phase}
        return plan

    def build(self, op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
        """Verification binding using this recipe's options and work ledger."""
        spec = self.spec(op)
        if geometry is None:
            read, vread, write, operand = "mte2", "mte2", "mte3", "mte1"
            mte1 = False
            handoff = FixpipeHandoffPolicy()
        else:
            read = read_role(geometry.resource_units)
            vread = read_role(geometry.resource_units, aiv=0)
            write = "mte3" if "mte3" in geometry.resource_units else None
            operand = "mte1" if "mte1" in geometry.resource_units else None
            mte1 = operand is not None
            handoff = gqa_fixpipe_policy(
                geometry, decode_handoff_workspace_bytes(spec, geometry.compute_units),
            )
            if spec.kv_cache.needs_conversion and operand is None:
                raise ModelingError(
                    "attention_decode_conversion_requires_mte1_operand_copy")
            if spec.kv_cache.needs_conversion and write is None:
                raise ModelingError(
                    "attention_decode_conversion_requires_mte3_result_copy")
        metadata = {"kv_cache_access": spec.kv_cache.report()}
        native = self.native_plan(op)
        if native is not None:
            metadata["native_paged_fia"] = native
        return OperatorPlan(
            self.work(op, include_mte1=mte1, read=read, vread=vread,
                      write=write, operand=operand, handoff=handoff),
            lambda context: _GQADecodeProgram(context, self, spec),
            metadata=metadata,
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


def build(op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
    """Independent verification binding for the default recipe options."""
    return StagedGQADecode().build(op, geometry)


@dataclass(frozen=True)
class _ProducerTaskProgram:
    context: RecipeContext
    recipe: StagedGQADecode
    spec: _Spec
    output: _Output
    contribution: _Contribution
    core: int
    signal: Callable[[], None] | None = None
    peer_counts: tuple[int, ...] | None = None
    compiled: KernelProgram | None = None
    stream_binding: _StreamBinding | None = None

    def start(self, engine, name, ready, *, owner=""):
        program = (
            self.compiled if self.compiled is not None else _producer_program(
                self.context, self.recipe, self.spec, self.output,
                self.contribution, self.core,
                peer_counts=self.peer_counts,
            )
        )
        if self.stream_binding is not None:
            program = self.stream_binding.bind(program)
        end = program.start(engine, name, ready, owner=owner)
        if self.signal is None:
            return end
        return engine.submit(
            name + ":split_produced", after=(end,),
            on_complete=lambda _: self.signal(), owner=owner,
        )


@dataclass(frozen=True)
class _GQADecodeProgram:
    context: RecipeContext
    recipe: StagedGQADecode
    spec: _Spec
    stream_binding: _StreamBinding | None = None

    def bind_stream_owners(self, owners, datapath, **kwargs):
        return replace(self, stream_binding=_StreamBinding(owners, datapath, dict(kwargs)))

    def _resolve_stream(self, tasks):
        binding = self.stream_binding
        if binding is None and self.context.stream_owners is not None and self.context.datapath is not None:
            from mindie_sim.prediction.streams import MAIN_STREAM
            binding = _StreamBinding(
                self.context.stream_owners,
                self.context.datapath,
                {
                    "resource_names": self.context.resource_names,
                    "resource_units": self.context.resource_units,
                    "rates": self.context.rates,
                    "stream_id": getattr(self.context.op, "stream", None) or MAIN_STREAM,
                    "quota_resources": None,
                    "pool_names": self.context.pool_names,
                    "pool_units": self.context.pool_units,
                },
            )
        owners = binding.owners if binding is not None else self.context.stream_owners
        base_context = (
            replace(self.context, stream_owners=None)
            if binding is not None and self.context.stream_owners is not None
            else self.context
        )
        if owners is None:
            return binding, None, base_context, self.context.core_count
        stream_id = (
            binding.kwargs.get("stream_id", "main")
            if binding is not None else (getattr(self.context.op, "stream", None) or "main")
        )
        if tasks and int(owners.aic_limit) <= 0:
            raise ModelingError(f"stream_zero_aic_budget:{stream_id}:gqa_decode")
        if tasks and int(owners.aiv_limit) <= 0:
            raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:gqa_decode")
        cores = max(1, min(self.context.core_count, int(owners.aic_limit)))
        return binding, owners, base_context, cores

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
        binding, owners, base_context, cores = self._resolve_stream(tasks)
        streams = [[] for _ in range(cores)]
        for index, (output_index, contribution) in enumerate(tasks):
            if contribution.owner is None:
                core = index % cores
            elif owners is not None:
                core = contribution.owner % cores
            else:
                core = contribution.owner
                if core >= cores:
                    raise ModelingError("attention_decode_mapping_exceeds_hardware_cores")
            streams[core].append((output_index, contribution))
        peer_counts = tuple(c.tile_count for s in streams for _, c in s if c.tile_count > 0)
        compiled_group = _compile_decode_peer_group(
            base_context, self.recipe, self.spec, streams,
            compactor=compact_periodic_pipeline,
        )
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
                    base_context, self.recipe, self.spec,
                    outputs[output_index], contribution, core, callback,
                    peer_counts=peer_counts, compiled=compiled_group.get((core, index)),
                    stream_binding=binding,
                )),)

            producer_events.append(PipelineProgram(count, 1, tile).start(
                engine, f"{name}:owner{core}:producer", ready,
                owner=f"{owner}:gqa{core}" if owner else f"gqa{core}",
            ))

        merges = tuple((index, output) for index, output in enumerate(outputs)
                       if len(output.contributions) > 1)
        merge_events = []
        is_fai_mapping = self.recipe.kv_mapping in {
            "fai_flash_decode", "fai_host", "fai_host_arch35"
        }
        merge_ready = (
            engine.submit(
                name + ":all_producers", after=tuple(producer_events),
                owner=owner,
            )
            if is_fai_mapping else ready
        )
        for core in range(min(len(merges), cores)):
            stream = _MergeStream(
                base_context, self.spec,
                merges[core::cores], barriers, core,
                stream_binding=binding,
            )
            merge_events.append(stream.start(
                engine, f"{name}:owner{core}:merge", merge_ready,
                owner=f"{owner}:gqa_merge{core}" if owner else f"gqa_merge{core}",
            ))
        return engine.submit(
            name + ":complete", after=tuple(producer_events + merge_events),
            owner=owner,
        )


_CONTRACT_FP32_BYTES = 4
_CONTRACT_ONLINE_META_BYTES = 8


def _build_decode_contract(spec) -> dict[str, Any]:
    cache, slots = spec.kv_cache, spec.slots
    outputs = tuple(spec.outputs)
    rows = max((item.rows for item in outputs), default=0)
    horizon = max((item.horizon for item in outputs), default=0)
    full = min(spec.key_tile, max(1, horizon))
    v_dim = spec.v_head_dim or spec.head_dim
    q_bytes = rows * spec.head_dim * spec.query_bytes
    online = rows * (v_dim * _CONTRACT_FP32_BYTES + _CONTRACT_ONLINE_META_BYTES)
    scores = rows * full * _CONTRACT_FP32_BYTES
    p_l0a = rows * full * spec.query_bytes
    pv_acc = rows * v_dim * _CONTRACT_FP32_BYTES
    key, value = full * cache.key_payload_row, full * cache.value_payload_row
    key_op = full * cache.key_operand_row if cache.needs_conversion else key
    val_op = full * cache.value_operand_row if cache.needs_conversion else value
    k_scale = full * cache.key_scale_row if cache.scale_scope == "row" else 0
    v_scale = full * cache.value_scale_row if cache.scale_scope == "row" else 0
    pages = cache.pages(0, horizon) if cache.paged else 0
    page_scale = pages * cache.page_scale_bytes if cache.scale_scope == "page" else 0
    k_conv = key_op if cache.needs_conversion else 0
    v_conv = val_op if cache.needs_conversion else 0
    cap = {
        "l1": q_bytes + slots * (key + value + p_l0a),
        "l0a": q_bytes + slots * p_l0a,
        "l0b": slots * (key_op + val_op),
        "l0c": slots * (scores + pv_acc),
        "ub": online + page_scale + slots * (scores + k_conv + v_conv + k_scale + v_scale),
    }
    cap["total"] = sum(cap.values())
    sl = "0..slots-1"
    rows_spec = [
        ("q_l1", "l1", q_bytes, "persistent", "q_load", "final"),
        ("q_l0a", "l0a", q_bytes, "persistent", "q_load", "final"),
        ("online_state", "ub", online, "persistent", "q_load", "final"),
        ("scores", "l0c", scores, sl, "qk", "softmax"),
        ("probs", "ub", scores, sl, "softmax", "online_update"),
        ("p_l1", "l1", p_l0a, sl, "probability_store", "probability_operand"),
        ("p_l0a", "l0a", p_l0a, sl, "probability_operand", "pv"),
        ("pv_acc", "l0c", pv_acc, sl, "v_load", "pv"),
        ("k_l1", "l1", key, sl, "k_load", "dequant_k" if cache.needs_conversion else "qk"),
        ("v_l1", "l1", value, sl, "v_load", "dequant_v" if cache.needs_conversion else "pv"),
        ("k_l0b", "l0b", key_op, sl, "operand_k", "qk"),
        ("v_l0b", "l0b", val_op, sl, "operand_v", "pv"),
    ]
    if cache.needs_conversion:
        rows_spec.extend((
            ("k_conv", "ub", k_conv, sl, "dequant_k", "operand_k"),
            ("v_conv", "ub", v_conv, sl, "dequant_v", "operand_v"),
        ))
    if k_scale:
        rows_spec.append(("k_scale", "ub", k_scale, sl, "k_scale_load", "dequant_k"))
    if v_scale:
        rows_spec.append(("v_scale", "ub", v_scale, sl, "v_scale_load", "dequant_v"))
    if page_scale:
        rows_spec.append(("page_scale", "ub", page_scale, "persistent", "page_access", "page_scale_release"))
    buffers = tuple(
        {"name": n, "pool": p, "bytes": b, "slot": s, "acquire": a, "free": f}
        for n, p, b, s, a, f in rows_spec
    )
    rep = cache.report()
    return {
        "kv_cache_access": rep,
        "conversion": {
            "needs_conversion": cache.needs_conversion,
            "query_dtype": cache.query_dtype,
            "key_format": rep["key_format"],
            "value_format": rep["value_format"],
            "dequant_ops_per_element": cache.dequant_ops_per_element,
            "elements_per_valid_row": (
                (cache.head_dim - cache.key_suffix)
                + ((cache.v_head_dim or cache.head_dim) - cache.value_suffix)
            ),
            "operand_copy_pipe": "mte1" if cache.needs_conversion else "",
            "scope": "on_chip_operand_materialization_no_hbm_writeback",
        },
        "workspace": {
            "owner_slots": slots, "rows_per_owner": rows, "horizon": horizon,
            "full_tile_rows": full, "capacity_bytes": cap,
            "basis": "max declared output rows/tile per owner; slots reused",
        },
        "buffer_lifecycle": buffers,
    }


class _StagedGQADecodePlan(OperatorPlan):
    """OperatorPlan adapter whose public program name stays recipe-like."""


_StagedGQADecodePlan.__name__ = "StagedGQADecode"


def staged_gqa_decode(**options) -> PlannedRecipe:
    """Build a recipe-named staged decode plan with physical contract metadata."""
    source = StagedGQADecode(**options)

    def _build(op, geometry):
        plan = source.build(op, geometry)
        metadata = dict(plan.metadata)
        metadata["decode_physical_contract"] = _build_decode_contract(source.spec(op))
        return _StagedGQADecodePlan(plan.work, plan.program, metadata, plan.native)

    return PlannedRecipe(_build)


def planned_recipe(**options) -> PlannedRecipe:
    """Bind the declared options to the staged decode work and program."""
    return PlannedRecipe(StagedGQADecode(**options).build)


from ..matrix.refill_steady import compact_periodic_pipeline


__all__ = [
    "IMPLEMENTATION",
    "KVCacheAccess",
    "RECIPE_ID",
    "SOURCE_REFERENCES",
    "StagedGQADecode",
    "build",
    "causal_pairs",
    "compact_periodic_pipeline",
    "planned_recipe",
    "staged_gqa_decode",
]
