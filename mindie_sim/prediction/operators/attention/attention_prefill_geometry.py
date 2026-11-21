"""Task geometry and work ledgers for staged GQA prefill.

[Role] Define request tiles, lazy task order and per-task resource work.
[Invariants] Executed rectangles and logical causal pairs remain distinct.
  KV traffic follows the task's actual head sharing and key-tile range.
[Timeouts & Locks] Enumerates tasks and tiles without retaining their history.
[Side-Effect Policy] Pure geometry and arithmetic.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .attention_kv_access import KVCacheAccess
from mindie_sim.prediction.mechanism import Work

IMPLEMENTATION = "staged_gqa_prefill"


_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2",
    "int8": "int8", "uint8": "uint8", "int4": "int4",
    "fp4": "fp4_e2m1", "fp4_e2m1": "fp4_e2m1",
}
_FP32_BYTES = 4
_NATIVE_QUERY_TILE = 128
_QK_PV_FLOPS_PER_DIM = 4
_SOFTMAX_OPS_PER_PAIR = 7
_ONLINE_UPDATE_OPS_PER_DIM = 3
_SOFTMAX_AND_UPDATE_OPS_PER_ROW = 10
_PREFILL_PHASES = {"prefill", "draft_prefill"}
# Source CVDiff Souter start (prompt_flash_attention_tiling.cpp 6415-6443).
# The query panel is chosen by the aligned head dimension, not by a request
# length or a measured duration.  Explicit recipe options still override it.
_SOURCE_QUERY_TILE_LARGE_HEAD = 64      # alignedHeadSize >= 200
_SOURCE_QUERY_TILE_DEFAULT = 128        # 32 <= alignedHeadSize < 200
_SOURCE_LARGE_HEAD_THRESHOLD = 200


def source_query_tile(head_dim: int) -> int:
    """CVDiff Souter start for the declared head dimension (source tiling)."""
    dim = _positive_int(head_dim, "head_dim")
    return (_SOURCE_QUERY_TILE_LARGE_HEAD
            if dim >= _SOURCE_LARGE_HEAD_THRESHOLD
            else _SOURCE_QUERY_TILE_DEFAULT)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"attention_prefill_{name}_must_be_positive_int")
    return value


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"attention_prefill_{name}_must_be_nonnegative_int")
    return value


def _dtype(value, name: str) -> str:
    if not isinstance(value, str):
        raise ModelingError(f"attention_prefill_{name}_must_be_dtype")
    try:
        return _DTYPE_ALIASES[value.strip().lower()]
    except KeyError:
        raise ModelingError(f"attention_prefill_unsupported_{name}:{value}") from None


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def logical_causal_pairs(rows: int, context: int, first_token: int = 0) -> int:
    """Valid causal pairs for contiguous query rows starting at ``first_token``."""
    return rows * (context + first_token + 1) + rows * (rows - 1) // 2


# Unselected native path.  GQA's resident set is PRELOAD_N=3.  Simplified
# runInfo[4] is four task descriptors, not four K/V panels; that cursor runs
# only when issue_kind is explicit.  Depth 3 here is not a 589824 B refusal.
NATIVE_S2_PRELOAD = 3


def _explicit_reconstruct(metadata: Mapping):
    """Top-level flag only.  Absence does not select Simplified or GQA."""
    if "reconstruct" not in metadata:
        return None
    value = metadata["reconstruct"]
    if value is not True and value is not False:
        raise ModelingError("attention_prefill_reconstruct_must_be_bool_or_absent")
    return value


@dataclass(frozen=True)
class _Task:
    request: int
    query_head: int
    kv_head: int
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    horizon: int
    key_tiles: int
    key_start: int = 0
    heads: int = 1

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
    query_tile: int
    key_tile: int
    slots: int
    s2_preload: int
    softmax_slots: int
    softmax_outer: int
    native_fia: bool
    reconstruct: bool | None
    issue_kind: str | None
    request_ends: tuple[int, ...]
    request_tile_counts: tuple[int, ...]
    kv_cache: KVCacheAccess = KVCacheAccess()

    @property
    def group(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def query_bytes(self) -> int:
        return int(dtype_bytes(self.query_dtype))

    @property
    def kv_bytes(self) -> int:
        return int(dtype_bytes(self.kv_dtype))

    @property
    def task_count(self) -> int:
        return self.request_ends[-1] if self.request_ends else 0


def _task_at(spec: _Spec, index: int) -> _Task:
    """Resolve one global cyclic task index without materializing task rows."""
    if index < 0 or index >= spec.task_count:
        raise IndexError("attention_prefill_task_index_out_of_range")
    request = bisect_right(spec.request_ends, index)
    base = spec.request_ends[request - 1] if request else 0
    local = index - base
    tiles = spec.request_tile_counts[request]
    query_head, tile = divmod(local, tiles)
    query, context = spec.requests[request]
    token_start = tile * spec.query_tile
    token_end = min(query, token_start + spec.query_tile)
    horizon = context + token_end
    return _Task(request, query_head, query_head // spec.group,
                 token_start, token_end, token_start, token_end, horizon,
                 _ceil_div(horizon, spec.key_tile), 0, 1)


def _iter_tasks(spec: _Spec):
    if spec.native_fia:
        for _index, task in _native_tasks(spec):
            yield task
        return
    for index in range(spec.task_count):
        yield _task_at(spec, index)


def _native_tasks(spec: _Spec):
    """Native TND S1G M tiles: time x G flattened into 128-row tiles.

    A tile covers ``128/G`` query tokens and all G query heads that share one
    KV head.  K/V is loaded once for the tile, not once per query head.
    """
    index = 0
    width = spec.group
    for request, (query, context) in enumerate(spec.requests):
        total = query * width
        for kv_head in range(spec.kv_heads):
            for m in range(_ceil_div(total, _NATIVE_QUERY_TILE)):
                row_start = m * _NATIVE_QUERY_TILE
                row_end = min(total, row_start + _NATIVE_QUERY_TILE)
                token_start = row_start // width
                token_end = _ceil_div(row_end, width)
                horizon = context + token_end
                yield index, _Task(
                    request, kv_head * width, kv_head, row_start, row_end,
                    token_start, token_end, horizon,
                    _ceil_div(horizon, spec.key_tile), 0, width)
                index += 1


def _task_at_for(spec: _Spec, index: int) -> _Task:
    if spec.native_fia:
        for position, (_idx, task) in enumerate(_native_tasks(spec)):
            if position == index:
                return task
        raise IndexError("attention_prefill_native_task_index_out_of_range")
    return _task_at(spec, index)


def _time_major_tasks(spec: _Spec):
    """S1G flattening order: outer query tile, inner query head.

    The registered TND GQA tiler walks ``[sequence, group]`` time-major, not
    the head-major order used by the cyclic executor.  Owner costs are S2
    tiles in this order.
    """
    if spec.native_fia:
        for index, task in _native_tasks(spec):
            yield index, task
        return
    for request, (query, _context) in enumerate(spec.requests):
        base = spec.request_ends[request - 1] if request else 0
        tiles = spec.request_tile_counts[request]
        for tile in range(tiles):
            for head in range(spec.query_heads):
                index = base + head * tiles + tile
                yield index, _task_at(spec, index)


def _task_pairs(task: _Task, context: int) -> tuple[int, int]:
    """Return (logical valid pairs, physical full-rectangle pairs)."""
    heads = max(1, task.heads)
    rows = task.rows // heads
    logical = heads * logical_causal_pairs(rows, context, task.token_start)
    return logical, task.rows * task.horizon


def _task_work(spec: _Spec, task: _Task, *, include_mte1: bool, partial: bool,
               read: str, vread: str, write: str | None,
                operand: str | None):
    cache = spec.kv_cache
    _query, context = spec.requests[task.request]
    rows, dim = task.rows, spec.head_dim
    q_bytes = rows * dim * spec.query_bytes
    if partial:
        out_bytes = rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES)
        vector_out = rows * (dim + 2)
    else:
        out_bytes = rows * dim * spec.query_bytes
        vector_out = rows * (dim + 1)
    load_start = task.key_start * spec.key_tile
    load_end = min((task.key_start + task.key_tiles) * spec.key_tile,
                   task.horizon)
    kv_rows = max(0, load_end - load_start)
    key_payload = kv_rows * cache.key_payload_row
    value_payload = kv_rows * cache.value_payload_row
    page_meta = page_scale = 0
    if cache.paged:
        pages = cache.pages(load_start, load_end)
        page_meta = pages * cache.page_meta_bytes
        if cache.scale_scope == "page":
            page_scale = pages * cache.page_scale_bytes
    row_scale = (cache.row_scale_bytes(kv_rows)
                 if cache.scale_scope == "row" else 0)
    producer = (q_bytes + page_meta
                + (0 if cache.key_needs_conversion else key_payload)
                + (0 if cache.value_needs_conversion else value_payload))
    vector_reads = (page_scale + row_scale
                    + (key_payload if cache.key_needs_conversion else 0)
                    + (value_payload if cache.value_needs_conversion else 0))
    mte3_extra = 0
    if cache.key_needs_conversion:
        mte3_extra += kv_rows * cache.key_operand_row
    if cache.value_needs_conversion:
        mte3_extra += kv_rows * cache.value_operand_row
    mte1 = 0
    if include_mte1:
        mte1 += q_bytes * task.key_tiles
        mte1 += (kv_rows * cache.key_operand_row
                 if cache.key_needs_conversion else key_payload)
        mte1 += (kv_rows * cache.value_operand_row
                 if cache.value_needs_conversion else value_payload)
    vector = vector_out + cache.conversion_ops(kv_rows)
    flops = 0
    for tile in range(task.key_tiles):
        valid = min(spec.key_tile, task.horizon
                    - (task.key_start + tile) * spec.key_tile)
        rect = rows * valid
        if include_mte1:
            mte1 += rect * spec.query_bytes
        flops += _QK_PV_FLOPS_PER_DIM * dim * rect
        vector += _SOFTMAX_OPS_PER_PAIR * rect + rows * (
            _ONLINE_UPDATE_OPS_PER_DIM * dim + _SOFTMAX_AND_UPDATE_OPS_PER_ROW
        )
    return {
        "producer": producer, "vector_reads": vector_reads,
        "writes": out_bytes, "mte3_extra": mte3_extra,
        "vector": vector, "mte1": mte1, "flops": flops,
    }


def _total_work(spec: _Spec, *, include_mte1: bool, read: str = "mte2",
                vread: str | None = None, write: str | None = "mte3",
                operand: str | None = "mte1", plan=None) -> Work:
    cache = spec.kv_cache
    vread = vread or read
    producer = vector_reads = writes = mte3_extra = vector = mte1 = 0
    flops = 0
    if plan is None:
        for task in _iter_tasks(spec):
            part = _task_work(
                spec, task, include_mte1=include_mte1, partial=False,
                read=read, vread=vread, write=write, operand=operand)
            producer += part["producer"]
            vector_reads += part["vector_reads"]
            writes += part["writes"]
            mte3_extra += part["mte3_extra"]
            vector += part["vector"]
            mte1 += part["mte1"]
            flops += part["flops"]
    else:
        task_owners = {}
        for core, ranges in enumerate(plan["ranges"]):
            for ti, key_start, key_end in ranges:
                task_owners.setdefault(ti, []).append(core)
                base = _task_at_for(spec, ti)
                item = replace(base, key_start=key_start,
                               key_tiles=key_end - key_start)
                part = _task_work(
                    spec, item, include_mte1=include_mte1,
                    partial=key_start > 0 or key_end < base.key_tiles,
                    read=read, vread=vread, write=write, operand=operand)
                producer += part["producer"]
                vector_reads += part["vector_reads"]
                writes += part["writes"]
                mte3_extra += part["mte3_extra"]
                vector += part["vector"]
                mte1 += part["mte1"]
                flops += part["flops"]
        for ti, owners in sorted(task_owners.items()):
            if len(owners) < 2:
                continue
            base = _task_at_for(spec, ti)
            rows, dim = base.rows, spec.head_dim
            fragments = len(owners)
            frag_rows = _ceil_div(rows, 2)
            for fragment in range(2):
                this_rows = min(frag_rows, rows - fragment * frag_rows)
                if this_rows <= 0:
                    continue
                part = fragments * this_rows * (dim * _FP32_BYTES
                                                + 2 * _FP32_BYTES)
                out = this_rows * dim * spec.query_bytes
                vector_reads += part
                writes += out
                vector += fragments * this_rows * (dim + 2)
    resources = {"hbm": producer + vector_reads + writes, "vector": vector}
    if read:
        resources[read] = resources.get(read, 0) + producer
    if vread:
        resources[vread] = resources.get(vread, 0) + vector_reads
    if write and (writes or mte3_extra):
        resources[write] = (resources.get(write, 0) + writes + mte3_extra)
    if include_mte1 and operand:
        resources[operand] = mte1
    return Work(matrix={spec.query_dtype: flops}, resources=resources)

def _report(spec: _Spec) -> dict:
    tasks = logical = physical = kv_read = q_read = out_write = 0
    kv_payload = kv_scale = page_table = 0
    q_l1_l0a = 0
    key_loads = 0
    cache = spec.kv_cache
    for task in _iter_tasks(spec):
        _query, context = spec.requests[task.request]
        tasks += 1
        task_logical, task_physical = _task_pairs(task, context)
        logical += task_logical
        physical += task_physical
        key_loads += task.key_tiles
        task_q = task.rows * spec.head_dim * spec.query_bytes
        q_read += task_q
        q_l1_l0a += task_q * task.key_tiles
        out_write += task.rows * spec.head_dim * spec.query_bytes
        payload = cache.payload_bytes(task.horizon)
        kv_payload += payload
        if cache.scale_scope == "row":
            kv_scale += cache.row_scale_bytes(task.horizon)
        if cache.paged:
            page_start = task.key_start * spec.key_tile
            page_end = min((task.key_start + task.key_tiles) * spec.key_tile,
                           task.horizon)
            pages = cache.pages(page_start, page_end)
            page_table += pages * cache.page_meta_bytes
            if cache.scale_scope == "page":
                kv_scale += pages * cache.page_scale_bytes
        kv_read += payload
    if spec.native_fia:
        # M tiles cut [time, group] and tails can split a head row; logical
        # causal pairs are still per original query head and KV group.
        logical = sum(spec.query_heads * logical_causal_pairs(query, context)
                      for query, context in spec.requests)
    return {
        "implementation": IMPLEMENTATION,
        "tasks": tasks,
        "key_tile_loads": key_loads,
        "logical_pairs": logical,
        "executed_rectangle_pairs": physical,
        "causal_masked_pairs": physical - logical,
        "q_bytes": q_read,
        "q_l1_l0a_bytes": q_l1_l0a,
        "kv_bytes": kv_payload + kv_scale + page_table,
        "kv_payload_bytes": kv_payload,
        "kv_scale_bytes": kv_scale,
        "page_table_bytes": page_table,
        "output_bytes": out_write,
        "query_tile": spec.query_tile,
        "key_tile": spec.key_tile,
        "slots": spec.slots,
        "s2_preload": spec.s2_preload,
        "reconstruct": spec.reconstruct,
        "issue_kind": spec.issue_kind,
        "source_state": "pinned_source_mechanism_not_captured_dispatch",
    }
