"""Cache-state algebra and physical KV prefix ownership across the runtime.
[Role] Exercise byte accounting, cache transactions, prefix reads and die-local residency.
[Invariants] GM work and writeback bytes are conserved. Cancellation restores LRU order.
  Shared pages pool per replica and key. Private tails, recurrent state and draft caches are per request.
[Timeouts & Locks] Bounded CPU simulation with no threads or waits.
[Side-Effect Policy] Reads packaged profiles and mutates local simulation state.
"""
from __future__ import annotations
from dataclasses import replace
import pytest
from _support import axes
from _support.builders import dense_case, dense_model, linear_model, view
from _support.hardware import machine
from _support.linear_cases import sku_planning_context, template_context, template_datapath
from mindie_sim.backends.ascend import profiles
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.cache_format import CacheFormat
from mindie_sim.catalog.cache_ownership import (
    SHARING_PRIVATE, batch_cache_footprint, cache_component_ranges,
    draft_cache_components,
)
from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.prediction.cache_path.geometry import (
    Span, full_line_spans, intersect_spans, lines_of_byte_spans,
    normalize_spans, span, subtract_spans, take_first_spans, union_spans,
)
from mindie_sim.prediction.cache_path import (
    AccessSummary, CacheAccessGeometry, CacheGeometry, CachePolicy, CacheState,
)
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.capacity.features import batch_cache_bytes
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.efficiency import capacity_window
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.operators.linear_attention.common import LinearAttentionContract, analyze_state_residency
from mindie_sim.prediction.runtime.memory_physics import AddressRun, AddressRunPlan, AdmissionMemoryView, MemoryAccessPattern, ReuseKey
from mindie_sim.primitives import load_primitives
from mindie_sim.templates.fuse_region import Lowering
from mindie_sim.prediction.prefix_reads import (
    CacheAccess, apply_prefix_read_plan, plan_prefix_reads,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import RequestSpec, WorkloadSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
pytestmark = pytest.mark.tier0
LINE = 128
WRITE_BACK = CachePolicy(write_policy="write_back")
WRITE_THROUGH = CachePolicy(write_policy="write_through")
BYPASS = CachePolicy(bypass=True)
SKU_GEOMETRIES = tuple(axes.REPRESENTATIVE_SKUS.items())
GENERATIONS = tuple(axes.REPRESENTATIVE_SKUS)
def state(capacity=256):
    return CacheState(CacheGeometry(capacity_bytes=capacity, line_bytes=LINE,
                                    transfer_bytes=32))
def buffered(*, rows=1, row_bytes=LINE, offset=0, stride=None, partial="masked", page=0):
    return CacheAccessGeometry(
        offset_bytes=offset, rows=rows, row_bytes=row_bytes,
        stride_bytes=row_bytes if stride is None else stride,
        page_bytes=page, partial_line=partial)
def reserve(cache, identity="A", span=None, *, write=False, policy=None):
    return cache.prepare_access(identity, span if span is not None else buffered(),
                                write=write, policy=policy if policy is not None else CachePolicy())
def commit(reservation):
    reservation.publish(); reservation.finish()
def access(request="a", *, context=999, q_len=1, prefix=128, key="k", layer=0):
    return CacheAccess(request, context, q_len=q_len, prefix_tokens=prefix,
                       prefix_key=key, row_bytes=128, scale_bytes=16, layer=layer)
def shared_case():
    base = dense_case(layers=2)
    return replace(base, runtime=replace(base.runtime, prefix_cache_mode="shared"))
def test_cold_warm_and_bypass_separate_physical_from_semantic_work():
    cache = state(); cold = cache.access("kv:layer0:request0", 128); warm = cache.access("kv:layer0:request0", 128)
    assert (cold.logical_bytes, cold.gm_bytes, cold.onchip_bytes) == (128, 128, 128); assert (
        cold.cache_read_bytes, cold.hbm_read_bytes) == (0, 128)
    # The hit still performs GM and on-chip work; only the external fill vanishes.
    assert (warm.logical_bytes, warm.gm_bytes, warm.onchip_bytes) == (128, 128, 128)
    assert (warm.cache_read_bytes, warm.hbm_read_bytes, warm.hit_lines) == (128, 0, 1)
    assert (cache.resident_bytes, cache.dirty_bytes) == (128, 0)
    assert cache.access("kv:layer0:request0", 128, policy=BYPASS).hbm_read_bytes == 128
    cold_cache, warm_cache = state(0), state(); cold_reads, warm_reads = [], []
    for _ in range(2):
        cold_reads.append(cold_cache.access("kv", 128).hbm_read_bytes)
        warm_reads.append(warm_cache.access("kv", 128).hbm_read_bytes)
    assert (cold_reads, warm_reads) == ([128, 128], [128, 0])
    for item in (cold_cache, warm_cache):
        assert item.counters["gm_bytes"] == item.counters["onchip_bytes"] == 256
    bypass = state(); first = bypass.access("kv", 128, policy=CachePolicy(bypass=True, ready_ms=0.002))
    assert first.hbm_read_bytes == first.onchip_bytes == first.staging_bytes == 128
    assert (first.ready_ms, first.bypassed, first.cold_path) == (0.002, True, False)
    assert bypass.access("kv", 128, policy=BYPASS).hbm_read_bytes == 128; assert bypass.resident_bytes == 0
@pytest.mark.parametrize("generation,sku", SKU_GEOMETRIES, ids=GENERATIONS)
def test_stateful_reuse_holds_across_declared_l2_geometries(generation, sku):
    capacity = int(profiles()[sku]["device"]["buffers"]["l2"]); cache = state(capacity); cold = cache.access("kv:layer0:request0", 512); warm = cache.access("kv:layer0:request0", 512)
    assert (cold.hbm_read_bytes, warm.hbm_read_bytes) == (512, 0)
    assert (cold.gm_bytes, warm.gm_bytes, warm.onchip_bytes) == (512, 512, 512)
    assert (warm.hit_lines, warm.miss_lines) == (4, 0)
    assert cache.geometry.capacity_lines == capacity // LINE
def test_write_back_streaming_eviction_and_dirty_guard_conserve_bytes():
    cache = state(); cache.access("A", 256, write=True, policy=WRITE_BACK); evicted = cache.access("B", 256, write=True, policy=WRITE_BACK)
    assert evicted.evicted_bytes == evicted.writeback_bytes == 256
    assert evicted.hbm_write_bytes == evicted.direct_write_bytes + evicted.writeback_bytes
    assert cache.flush().bytes == 256 and cache.dirty_bytes == 256
    clean = state(); clean.access("A", 256, write=True)
    assert clean.access("B", 256, write=True).writeback_bytes == 0
    over = state(128); result = over.access("A", 256, write=True, policy=WRITE_BACK)
    assert (result.direct_write_bytes, result.writeback_bytes) == (128, 0)
    assert over.resident("A") == (1, 1) and over.dirty_bytes == 128; assert (
        result.hbm_write_bytes + over.flush().bytes) == 256
    partial = state(256); partial.access("A", 256)
    assert partial.resident("A") == (2, 0)
    assert partial.access("A", 128, write=True, policy=WRITE_BACK).direct_write_bytes == 0
    assert partial.resident("A") == (2, 1) and partial.flush().bytes == 128
    stream = state(128); first, second = stream.access("stream", 300), stream.access("stream", 300)
    assert (first.hbm_read_bytes, first.fill_bytes) == (384, 128)
    assert (second.hit_lines, second.miss_lines, second.hbm_read_bytes) == (1, 2, 256)
    assert stream.resident_bytes == 128
    cache = state(128); cache.access("A", 128, write=True, policy=WRITE_BACK)
    with pytest.raises(ModelingError):
        cache.clone()
    plan = cache.flush()
    for action in (lambda: cache.access("B", 128), cache.flush, cache.invalidate,
                   cache.reset, cache.clone):
        with pytest.raises(ModelingError):
            action()
    plan.cancel()
    assert cache.dirty_bytes == 128
    with pytest.raises(ModelingError):
        cache.invalidate()
    for write in (False, True):
        with pytest.raises(ModelingError):
            cache.access("A", 128, write=write, policy=BYPASS)
    assert cache.resident("A") == (1, 1)
    held = reserve(cache)
    with pytest.raises(ModelingError):
        cache.clone()
    with pytest.raises(ModelingError):
        cache.reset()
    held.cancel()
    assert cache.flush().bytes == 128
    clean = state(256); clean.access("A", 128)
    assert clean.clone().resident("A") == (1, 0); assert clean.invalidate() == 1 and clean.invalidate() == 0
    clean.access("A", 128)
    assert clean.access("A", 128, policy=BYPASS).hbm_read_bytes == 128; assert clean.resident("A") == (1, 0)
    assert clean.access("A", 128, write=True, policy=BYPASS).hbm_write_bytes == 128
    assert clean.resident("A") == (0, 0) and clean.resident_bytes == 0
def test_byte_validity_ranges_holes_and_rmw_protocols():
    cache = state(128); masked = reserve(cache, span=buffered(row_bytes=32), write=True, policy=WRITE_BACK)
    assert masked.result.hbm_write_bytes == 0
    commit(masked)
    assert cache.dirty_bytes == 32
    partial = reserve(cache, span=buffered(offset=64, row_bytes=32))
    assert partial.result.hbm_read_bytes == 128
    commit(partial)
    assert cache.dirty_bytes == 32  # only [0,32) was modified
    full = reserve(cache)
    assert (full.result.hbm_read_bytes, full.result.hit_lines) == (0, 1)
    commit(full); rmw = reserve(cache, span=buffered(row_bytes=64), write=True, policy=WRITE_THROUGH)
    assert (rmw.result.hbm_read_bytes, rmw.result.hbm_write_bytes) == (0, 128)
    ranges = state(256); low = reserve(ranges)
    assert low.result.hbm_read_bytes == 128
    commit(low); high = reserve(ranges, span=buffered(offset=128))
    assert (high.result.hbm_read_bytes, high.result.hit_lines) == (128, 0)
    commit(high)
    assert ranges.resident_ranges("A")[0] == (Span(0, 2, 2, 1),); assert (
        ranges.access("A", 128).hbm_read_bytes) == 0; assert ranges.access("A", 256).hit_lines == 2
    strided = state(4096); stride = buffered(rows=2, row_bytes=32, stride=1025)
    assert stride.summary(LINE).line_ranges == ((0, 1), (8, 1))
    reservation = reserve(strided, "S", stride)
    assert reservation.result.hbm_read_bytes == 256
    commit(reservation)
    assert strided.resident_ranges("S")[0] == (Span(0, 1, 1, 1), Span(8, 1, 1, 1))
    overlap = state(256); commit(reserve(overlap, span=buffered(row_bytes=256))); update = reserve(overlap, span=buffered(offset=128), write=True, policy=WRITE_BACK)
    assert (update.result.hit_lines, update.result.hbm_write_bytes) == (1, 0)
    commit(update)
    assert overlap.resident_ranges("A")[1] == (Span(1, 1, 1, 1),)
    assert overlap.flush().bytes == 128
    interleaved = state(1024)
    commit(reserve(interleaved, span=buffered(offset=128, row_bytes=1),
                   write=True, policy=WRITE_BACK))
    shape = buffered(rows=3, row_bytes=64, stride=256, partial="read_modify_write"); summary = shape.summary(LINE, write=True)
    assert (summary.line_segments, full_line_spans(summary.byte_segments, LINE),
            summary.rmw_read_lines) == ((Span(0, 1, 2, 3),), (), 3)
    hole = reserve(interleaved, span=shape, write=True, policy=WRITE_BACK)
    assert hole.result.hbm_read_bytes == 384
    commit(hole)
    for offset in (64, 320, 576):
        high = reserve(interleaved, span=buffered(offset=offset, row_bytes=64))
        assert (high.result.hbm_read_bytes, high.result.hit_lines) == (0, 1)
        commit(high)
def test_rmw_reads_writes_and_geometry_are_line_exact():
    zero = state(0)
    overlap = reserve(zero, span=buffered(rows=2, row_bytes=64, stride=32,
                                         partial="read_modify_write"),
                      write=True, policy=WRITE_THROUGH)
    assert (overlap.result.hbm_read_bytes, overlap.result.hbm_write_bytes) == (128, 128); assert (
        overlap.result.gm_bytes, overlap.result.onchip_bytes) == (128, 128)
    sparse = reserve(zero, "S", buffered(rows=2, row_bytes=1, stride=128))
    assert (sparse.result.logical_bytes, sparse.result.gm_bytes) == (2, 64); assert (
        sparse.result.onchip_bytes, sparse.result.hbm_read_bytes) == (64, 256)
    for rows, row_bytes, stride, expected in ((4097, 64, 256, 524416), (2, 64, 32, 128)):
        result = reserve(state(1 << 20),
                         span=buffered(rows=rows, row_bytes=row_bytes, stride=stride,
                                       partial="read_modify_write"),
                         write=True, policy=WRITE_THROUGH).result
        assert (result.hbm_read_bytes, result.hbm_write_bytes) == (expected, expected)
    contiguous = buffered(row_bytes=300).summary(LINE)
    assert (contiguous.line_count, contiguous.logical_bytes) == (3, 300); assert (
        contiguous.partial_first, contiguous.partial_last) == (False, True)
    strided = buffered(rows=4, row_bytes=64, stride=128).summary(LINE)
    assert (strided.line_count, strided.repeated) == (4, True)
    paged = buffered(rows=3, row_bytes=100, stride=256, page=1024).summary(LINE)
    assert (paged.line_count, paged.page_count) == (3, 1); assert buffered(
        offset=20, rows=2, row_bytes=100, stride=128,
        partial="read_modify_write").summary(LINE, write=True).rmw_read_lines == 2
    assert buffered(offset=20, rows=2, row_bytes=100, stride=128).summary(
        LINE, write=True).rmw_read_lines == 0
    repeated = buffered(rows=65537, row_bytes=64, stride=256).summary(LINE)
    assert repeated.line_segments == (Span(0, 1, 2, 65537),); assert repeated.ranges_complete is False; assert (
        repeated.coverage_bytes == repeated.gm_bytes == 4194368)
    huge = buffered(row_bytes=134217728).summary(LINE)
    assert huge.byte_segments == (Span(0, 134217728, 134217728, 1),); assert (
        huge.line_segments) == (Span(0, 1048576, 1048576, 1),); assert huge.ranges_complete is True
def test_large_strided_access_retains_only_capacity_prefix():
    cache = state(256); large = buffered(rows=1000001, row_bytes=64, stride=256); reservation = reserve(cache, span=large)
    assert reservation.result.hbm_read_bytes == 128000128; assert (
        reservation.result.fill_bytes, reservation.result.miss_lines) == (256, 1000001)
    commit(reservation)
    assert cache.resident("A") == (2, 0); assert cache.resident_ranges("A")[0] == (Span(0, 1, 2, 2),)
    assert reserve(cache, span=buffered(offset=128, row_bytes=64)).result.hbm_read_bytes == 128
    rmw = state(256)
    write = reserve(rmw, span=buffered(rows=1000001, row_bytes=64, stride=256,
                                       partial="read_modify_write"),
                    write=True, policy=WRITE_BACK)
    assert (write.result.hbm_read_bytes, write.result.direct_write_bytes) == (
        128000128, 999999 * 128)
    commit(write)
    assert rmw.resident("A") == (2, 2) and rmw.dirty_bytes == 128; assert (
        rmw.counters["direct_write_bytes"]) == 999999 * 128
    assert rmw.flush().bytes == 256
def test_span_algebra_and_async_intersection_match_byte_sets():
    cases = (
        (), (Span(0, 1, 1, 1),), (Span(0, 3, 8, 3),),
        (Span(0, 2, 5, 3), Span(3, 2, 5, 3)),
        (Span(0, 2, 10, 2), Span(5, 1, 3, 3)),
        (Span(0, 1, 1, 1), Span(3, 1, 1, 1), Span(5, 1, 1, 1)),
        (Span(0, 64, 256, 3), Span(128, 1, 1, 1)),
    )
    for left in cases:
        for right in cases:
            left_bytes, right_bytes = oracle_bytes(left), oracle_bytes(right)
            whole = left_bytes | right_bytes
            assert oracle_bytes(union_spans(left, right)) == whole
            assert oracle_bytes(subtract_spans(left, right)) == left_bytes - right_bytes
            assert oracle_bytes(intersect_spans(left, right)) == left_bytes & right_bytes
            for limit in (0, 3, len(whole)):
                assert oracle_bytes(take_first_spans(left + right, limit)) == (
                    set(sorted(whole)[:limit]))
            for line in (1, 4, 16):
                touched = {byte // line for byte in whole}
                assert oracle_bytes(lines_of_byte_spans(left + right, line)) == touched
                full = {number for number in touched
                        if set(range(number * line, (number + 1) * line)) <= whole}
                assert oracle_bytes(full_line_spans(left + right, line)) == full
    for offset, rows, row_bytes, stride in (
            (0, 3, 64, 256), (20, 2, 100, 128), (5, 4, 32, 64),
            (3, 5, 48, 128), (127, 2, 7, 256)):
        summary = buffered(offset=offset, rows=rows, row_bytes=row_bytes,
                           stride=stride, partial="read_modify_write").summary(16, write=True)
        request_lines = oracle_bytes(summary.line_segments)
        full = oracle_bytes(full_line_spans(summary.byte_segments, 16))
        assert summary.rmw_read_lines == len(request_lines - full)
    left = Span(0, 100000007, 200000011, 2); right = Span(1, 100000007, 300000013, 2); got = intersect_spans((left,), (right,)); pairs = []
    for left_index in range(left.count):
        left_start = left.start + left_index * left.stride
        for right_index in range(right.count):
            right_start = right.start + right_index * right.stride
            lo = max(left_start, right_start)
            hi = min(left_start + left.width, right_start + right.width)
            if hi > lo:
                pairs.append(span(lo, hi - lo))
    assert got == normalize_spans(pairs)
    assert got == (Span(1, 100000006, 100000006, 1), Span(300000014, 4, 4, 1))
    assert intersect_spans((Span(0, 1, 6, 5),), (Span(1, 1, 10, 3),)) == ()
    assert oracle_bytes(intersect_spans((Span(0, 1, 6, 5),), (Span(4, 1, 10, 3),))) == {24}
def test_access_summary_and_unseen_formats_predict_without_objects():
    cache = state(2048); summary = AccessSummary()
    for index in range(64):
        summary.add(cache.access(f"request{index}", 512))
    assert summary.accesses == 64 and summary.logical_bytes == 64 * 512; assert (
        summary.resident_peak_bytes == 2048 and cache.resident_identities <= 16); assert (
        summary.onchip_bytes == summary.gm_bytes == summary.logical_bytes)
    assert "resident_bytes" not in cache.snapshot()["counters"]
    for width in (1, 37, 513, 1025):
        for rows in (1, 3):
            fmt = CacheFormat("int8", group_size=128, scale_dtype="fp32")
            stored = fmt.stored_bytes(width, rows)
            result = state(512).access(f"unseen:{width}:{rows}", stored)
            assert result.hbm_read_bytes == 128 * ((stored + 127) // 128); assert (
                result.gm_bytes) == 32 * ((stored + 31) // 32)
            assert result.logical_bytes == stored
    assert CacheFormat("fp4_e2m1").row_bytes(3) == 2; assert CacheFormat("fp4_e2m1").row_bytes(5) == 3
    assert CacheFormat("uint8", packed_elements=2).row_bytes(5) == 3
    with pytest.raises(ModelingError):
        CacheFormat("fp4_e2m1", packed_elements=2)
    packed = CacheFormat("fp4_e2m1", group_size=16, scale_dtype="bf16")
    assert packed.row_bytes(576) == 288 + 72
    assert packed.stored_bytes(576, 2) == 720; assert packed.transaction_count(576, 2, 128) == 6
    assert packed.transaction_bytes(576, 2, 128) == 768
def test_reservation_rollback_capacity_and_victim_drain():
    cache = state(256); cache.access("A", 128, write=True, policy=WRITE_BACK); reservation = reserve(cache, write=True, policy=WRITE_BACK)
    assert cache.resident("A") == (0, 0)
    reservation.cancel()
    assert cache.resident("A") == (1, 1); assert (cache.resident_bytes, cache.dirty_bytes) == (128, 128)
    clean = state(128); clean.access("A", 128); bypass = reserve(clean, write=True, policy=BYPASS)
    assert clean.resident("A") == (0, 0)
    bypass.cancel()
    assert clean.resident("A") == (1, 0) and clean.resident_bytes == 128
    lru = state(384)
    for identity in ("A", "B", "C"):
        lru.access(identity, 128)
    victim = reserve(lru, "D")
    assert tuple(lru._resident) == ("B", "C")
    victim.cancel()
    assert tuple(lru._resident) == ("A", "B", "C")
    first, second = reserve(lru, "A"), reserve(lru, "B")
    assert tuple(lru._resident) == ("C",)
    first.cancel()
    assert tuple(lru._resident) == ("A", "C")
    second.cancel()
    assert tuple(lru._resident) == ("A", "B", "C")
    partial_victim = state(256); partial_victim.access("A", 256, write=True, policy=WRITE_BACK); victim = reserve(partial_victim, "B")
    assert partial_victim.resident("A") == (1, 1)
    victim.cancel()
    assert partial_victim.resident("A") == (2, 2)
    assert partial_victim.dirty_bytes == 256 and partial_victim.flush().bytes == 256
    cache = state(128); first = reserve(cache)
    with pytest.raises(ModelingError):
        reserve(cache, "B")
    commit(first); second = reserve(cache, "B")
    assert second.victims and cache.resident("A") == (0, 0)
    commit(second)
    assert cache.resident("B") == (1, 0)
    dirty = state(128); dirty.access("A", 128, write=True, policy=WRITE_BACK); victim = reserve(dirty, "B")
    with pytest.raises(ModelingError, match="cache_reservation_drain_incomplete"):
        victim.publish()
    victim.mark_drained(); commit(victim)
    assert dirty.resident("A") == (0, 0) and dirty.flush().bytes == 0
    zero = state(256); empty = reserve(zero, span=buffered(rows=0, row_bytes=0))
    assert empty.held_identities == ()
    nonempty = reserve(zero)
    assert zero.snapshot()["reserved_bytes"] == 128
    commit(empty)
    assert zero.snapshot()["reserved_bytes"] == 128 and zero.inflight_identities == 1
    commit(nonempty)
    assert zero.inflight_identities == 0 and zero.resident("A") == (1, 0)
    rollback = state(128); rollback.access("A", 128); bypass = reserve(rollback, write=True, policy=BYPASS)
    assert rollback.snapshot()["reserved_bytes"] == 128
    with pytest.raises(ModelingError, match="cache_capacity_reserved_exhausted"):
        reserve(rollback, "B")
    bypass.cancel()
    assert rollback.resident("A") == (1, 0)
    commit(reserve(rollback, "B"))
    assert rollback.resident("A") == (0, 0) and rollback.resident("B") == (1, 0)
    published = state(128); published.access("A", 128); commit(reserve(published, write=True, policy=BYPASS))
    assert published.resident("A") == (0, 0)
    assert published.snapshot()["reserved_bytes"] == 0
def test_shared_prefix_footprint_replica_cow_private_window_and_linear_state():
    case = shared_case(); requests = (view("a", 1, 200, prefix=128), view("b", 1, 200, prefix=128))
    footprint = batch_cache_footprint(
        case, requests=requests, replica=0, phase="decode", padded=False)
    row = 2 * 2 * 16 * 2
    assert footprint.shared_once_total == pytest.approx(2 * 128 * row)
    assert footprint.private_total == pytest.approx(2 * 2 * (201 - 128) * row)
    assert footprint.total == pytest.approx(footprint.shared_once_total + footprint.private_total)
    duplicate = batch_cache_footprint(
        case, requests=requests, replica=1, phase="decode", padded=False)
    assert duplicate.total == footprint.total
    aligned = cache_component_ranges(
        case.model, case.model.layer_plans[0], case.runtime.parallel,
        sequence_tokens=201, prefix_tokens=128, interleave=128, rank=None,
        capacity_padded=False)
    partial = cache_component_ranges(
        case.model, case.model.layer_plans[0], case.runtime.parallel,
        sequence_tokens=201, prefix_tokens=129, interleave=128, rank=None,
        capacity_padded=False)
    shared = [item.valid_bytes for item in aligned if item.sharing != SHARING_PRIVATE]; tails = [sum(item.valid_bytes for item in ranges
                 if item.sharing == SHARING_PRIVATE) for ranges in (aligned, partial)]
    assert shared == [128 * 128] and tails[0] == pytest.approx(tails[1])
    private, request = dense_case(layers=2), view("a", 1, 200)
    sole = batch_cache_footprint(
        private, requests=(request,), replica=0, phase="decode", padded=False)
    expected = sum(sum(value for value in layer_cache_components(
        private.model, layer, private.runtime.parallel, request.kv_len,
        capacity_padded=False).values()) for layer in private.model.layer_plans)
    assert sole.shared_once_total == 0 and sole.total == pytest.approx(expected)
    different = batch_cache_footprint(
        case, requests=(view("a", 1, 200, prefix=128),
                        replace(view("b", 1, 200, prefix=128),
                                prefix_cache_key="other")),
        replica=0, phase="decode", padded=False)
    same = batch_cache_footprint(
        case, requests=(view("a", 1, 200, prefix=128),
                        view("b", 1, 200, prefix=128)),
        replica=0, phase="decode", padded=False)
    assert different.shared_once_total == pytest.approx(2 * same.shared_once_total)
    windowed = replace(dense_model(layers=2), layer_plans=tuple(
        replace(layer, window=64) for layer in dense_model(layers=2).layer_plans))
    live = batch_cache_footprint(
        replace(case, model=windowed), requests=(view("a", 1, 400, prefix=256),),
        replica=0, phase="decode", padded=False)
    assert live.private_total <= 2 * 64 * 64 * 2
    linear = replace(dense_case(layers=1), model=linear_model("kda", layers=1))
    runtime = replace(linear.runtime, recover_ssm=True, metadata=(("draft_proposal_tokens", 5),))
    stateful = batch_cache_footprint(
        replace(linear, runtime=runtime), requests=(view("a", 1, 200, prefix=128),),
        replica=0, phase="decode", padded=False)
    assert stateful.state_total > 0 and stateful.shared_once_total == 0
    target = dense_model(layers=1)
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
                      num_heads=4, num_kv_heads=2, head_dim=16, q_width=64,
                      kv_width=32, vocab_size=128, attention_kind="gqa",
                      kv_cache_elements=32, target_shared_embedding=True,
                      target_shared_lm_head=True)
    draft = replace(draft, layer_plans=native_draft_layer_plans(target, draft))
    components = draft_cache_components(
        replace(target, auxiliary=AuxiliarySpec(mtp=draft)),
        ParallelSpec(tp=1), draft_kind="mtp", sequence_tokens=8)
    assert components["draft_kv"] > 0; assert any(name.startswith("draft.") for name in components)
def test_prefix_read_union_mode_causal_rows_and_applied_delta():
    plan = plan_prefix_reads((access("a"), access("b")),
                             mode="batch_shared", kernel_reuse="tile")
    rows = 128 + 2 * (1000 - 128)
    assert plan.logical_query_bytes == pytest.approx(2 * (1000 * 128 + 16))
    assert plan.unique_hbm_bytes == pytest.approx(rows * 128 + 2 * 16 * rows / (2 * 1000))
    assert plan.l2_to_core_bytes == pytest.approx(2 * (1000 * 128 + 16)); assert (
        plan.query_read_bytes, plan.read_reuse_applied) == (256, True)
    assert plan.unique_hbm_bytes < plan.logical_query_bytes
    per_sequence = plan_prefix_reads((access("a"), access("b")),
                                     mode="per_sequence", kernel_reuse="tile")
    assert per_sequence.unique_hbm_bytes == pytest.approx(per_sequence.logical_query_bytes); assert (
        per_sequence.read_reuse_applied) is False
    assert "not_batch_shared" in per_sequence.reason
    plain = plan_prefix_reads((access("a"), access("b")),
                              mode="batch_shared", kernel_reuse="tile")
    keys = plan_prefix_reads((access("a"), access("b", key="other")),
                             mode="batch_shared", kernel_reuse="tile")
    layers = plan_prefix_reads((access("a"), access("b", layer=1)),
                               mode="batch_shared", kernel_reuse="tile")
    assert keys.unique_hbm_bytes > plain.unique_hbm_bytes; assert layers.unique_hbm_bytes > plain.unique_hbm_bytes
    with pytest.raises(ModelingError, match="shared_prefix_kernel_reuse_unsupported"):
        plan_prefix_reads((access("a"), access("b")), mode="batch_shared",
                          kernel_reuse="native_per_sequence")
    auto = plan_prefix_reads((access("a"), access("b")), mode="auto",
                             kernel_reuse="native_per_sequence")
    assert auto.read_reuse_applied is False and auto.unique_hbm_bytes == auto.logical_query_bytes
    causal = plan_prefix_reads((access("a", context=9, q_len=3, prefix=0),),
                               mode="auto", kernel_reuse="tile")
    assert causal.logical_query_bytes == pytest.approx((10 + 11 + 12) * 128 + 3 * 16)
    assert causal.query_read_bytes == pytest.approx(3 * 128)
    op = OpSpec("index", "attention_index", "decode", 0, 0, ("a", "b"),
                (1000,), (1000,), (1024,), (1000,),
                hbm_bytes=1000.0 * 128 + 32, mte2_bytes=1000.0 * 128 + 32,
                metadata=(("component", "index_k"),))
    applied = apply_prefix_read_plan(op, plan)
    assert dict(applied.metadata)["prefix_read_reuse_applied"] is True
    assert applied.hbm_bytes == pytest.approx(
        op.hbm_bytes - plan.logical_query_bytes + plan.unique_hbm_bytes)
    assert applied.mte2_bytes == pytest.approx(
        op.mte2_bytes - plan.logical_query_bytes + plan.l2_to_core_bytes)
def test_shared_prefix_window_charges_pool_pages_and_releases_owners():
    base = dense_case(layers=2)
    case = replace(
        base, runtime=replace(base.runtime, prefix_cache_mode="shared"),
        workload=WorkloadSpec((RequestSpec("r", 0.0, 128, 2, prefix_tokens=128),)))
    window = capacity_window(
        case, phase="decode", context=128, tokens_per_seq=1, sequences=2, output_budget=8)
    active = window["window"]["active_tokens_per_sequence"]
    footprint = batch_cache_bytes(
        case, sequences=2, context=128, tokens_per_seq=active - 128,
        prefix_tokens=128, phase="decode", padded=True)
    assert window["active_kv_bytes"] == pytest.approx(sum(footprint.total_bytes))
    assert window["peak_bytes"] == pytest.approx(window["active_kv_bytes"])
    private = capacity_window(
        replace(case, runtime=replace(case.runtime, prefix_cache_mode="private")),
        phase="decode", context=128, tokens_per_seq=1, sequences=2, output_budget=8)
    assert private["active_kv_bytes"] == pytest.approx(2 * sum(Memory(case).cache_bytes(active)))
    memory = Memory(dense_case(layers=1))
    assert memory.retain_shared_pages(0, "owner", (1024.0,), physical_key="p1") is True
    assert memory.retain_shared_pages(0, "owner", (2048.0,), physical_key="p2") is True
    assert len(memory.shared_pages) == 2; assert memory.release_shared_pages(0, "owner") is True; assert (
        memory.shared_pages == {} and len(memory.release_events) == 2)
    assert memory.release_shared_pages(0, "owner") is False
    linear = replace(
        dense_case(layers=1), model=linear_model("kda", 1),
        runtime=replace(dense_case(layers=1).runtime, prefix_cache_mode="shared"),
        workload=WorkloadSpec((RequestSpec("r", 0.0, 128, 1024, prefix_tokens=64),)))
    long = capacity_window(
        linear, phase="decode", context=128, tokens_per_seq=1, sequences=2, output_budget=1024)
    short = capacity_window(
        linear, phase="decode", context=128, tokens_per_seq=1, sequences=2, output_budget=1)
    assert long["window"]["active_tokens_per_sequence"] == 128 + 1024
    assert long["active_kv_bytes"] == pytest.approx(short["active_kv_bytes"])
    footprint = batch_cache_bytes(linear, sequences=2, context=128, tokens_per_seq=1,
                                  prefix_tokens=64, phase="decode", padded=True)
    assert long["active_kv_bytes"] == pytest.approx(sum(footprint.total_bytes))
def oracle_bytes(spans):
    """Independent finite byte-set expansion of the compact span encoding."""; result = set()
    for item in normalize_spans(spans):
        for index in range(item.count):
            start = item.start + index * item.stride
            result.update(range(start, start + item.width))
    return result

def test_admission_epoch_observes_resident_identity_and_eviction_order():
    sector = AddressRunPlan().sector_bytes
    view = AdmissionMemoryView({0: 4 * sector, 1: 4 * sector})
    engine = Engine([])
    engine.attach_admission_memory_view(view)

    def warm(name):
        key = ReuseKey(0, name, 0)
        pattern = MemoryAccessPattern(name, 0, 0, 0, "read", "mte2_aic", sector,
                                      AddressRunPlan((AddressRun(0, 1, sector, sector),)), key)
        view.admit_access(pattern, sector)

    other_die = view.residency_fingerprint(1)
    warm("a")
    a_epoch = engine._memory_epoch()
    warm("b")
    both = engine._memory_epoch()
    warm("a")
    assert engine._memory_epoch() != both
    assert view.residency_fingerprint(1) == other_die
    other = AdmissionMemoryView({0: 4 * sector, 1: 4 * sector})
    engine.attach_admission_memory_view(other)
    view = other
    warm("b")
    assert engine._memory_epoch() != a_epoch

def test_idle_replica_preserves_the_active_die_knee():
    active = machine("Ascend950DT_9582")
    idle = replace(machine("Ascend910_9362").dies[0], rank=1)
    mixed = replace(active, dies=(*active.dies, idle))
    size, working_set = 8192, 144 * 1024**2
    runs = AddressRunPlan((AddressRun(0, 1, size, size),))
    pattern = MemoryAccessPattern("weights", 0, 0, 0, "read", "mte2_aiv", size, runs,
                                  ReuseKey(0, "weights", 0), working_set)
    fingerprints, previews = [], []
    for hardware in (active, mixed):
        case = replace(dense_case(dp=len(hardware.dies)), machine=hardware)
        admission = Execution(case, CostModel(case)).admission_memory_view
        assert admission.admit_access(pattern, size) == (0, size)
        previews.append(admission.preview_access(pattern, size))
        fingerprints.append(admission.residency_fingerprint(0))
    assert 0 < previews[0][0] < size and previews[0] == previews[1] and fingerprints[0] == fingerprints[1]
    assert admission.memory_access_for(1) == idle.memory_access

def test_reset_engine_rebinds_memory_to_a_cold_capacity_domain():
    case = replace(dense_case(), machine=machine(axes.REPRESENTATIVE_SKUS["a5"]))
    memory = Memory(case)
    execution = Execution(case, CostModel(case), memory=memory)
    engine, ledger, admission = execution.engine, execution.ledger, execution.admission_memory_view
    admission.grant_reservation(0, "live", LINE)
    execution.reset_engine()
    assert execution.engine is not engine and execution.ledger is not ledger
    assert execution.engine.admission_memory_view is execution.admission_memory_view is not admission
    assert execution.admission_memory_view.reserved_bytes(0) == 0
    assert execution.engine.memory is execution.memory is not memory
    assert execution.memory.runtime_state is execution.engine and memory.runtime_state is engine
    assert execution.memory.ledger_domain is not memory.ledger_domain
    assert execution.memory.rank_capacity == memory.rank_capacity

@pytest.mark.parametrize("sku", ["Ascend910_9362", "Ascend950DT_9582"])
def test_template_admission_uses_working_set_sweep_at_capacity(sku):
    path = template_datapath(sku)
    context = replace(template_context(path, None), primitive_table=load_primitives(sku))
    view = Lowering(path, context=context).state
    capacity = view.effective_l2_bytes(0)
    at_capacity = view._hit_fraction(capacity, capacity)
    just_over = view._hit_fraction(capacity + 1, capacity)
    assert at_capacity == 1.0
    assert 0 < view._hit_fraction(capacity + capacity // 8, capacity) < just_over < at_capacity

def test_staged_state_residency_sees_live_stream_pressure_on_its_die():
    _, ctx = sku_planning_context(axes.REPRESENTATIVE_SKUS["a5"])
    die = 1
    contract = LinearAttentionContract("gdn_prefill", (64,), 1, 1, 1, 64, 64)
    capacity = 12 * 1024
    view = AdmissionMemoryView({die: capacity})
    ctx = replace(ctx, occurrence=replace(ctx.occurrence, rank=die), admission_memory_view=view)
    free = analyze_state_residency(contract, ctx)
    view.enter_stream(die, "weights")
    before = view.residency_fingerprint()
    shared = analyze_state_residency(contract, ctx)
    assert free.reload_bytes == 0
    assert shared.reload_bytes == shared.chunk_bank_bytes > 0
    assert view.residency_fingerprint() == before
