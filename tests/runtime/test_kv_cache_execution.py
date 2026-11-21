"""Engine-bound KV cache execution: programs, lowering ports and receipts.
[Role]
  Pin event-time cache execution: access operations, ``CacheAccessProgram``
  cold/writeback paths, ``CacheExecution`` receipts, ``CacheLowering`` ports
  and the Builder store/load row ABI and typed view feeding those programs.
[Invariants]
  * State publishes at completion: a reserved fill is not hittable and a flush
    clears dirty bytes only when its write completes; HBM work is conserved.
  * A named source/output port is a real Event, never a generic ready or a
    declared alias; a missing later port or resource leaves Engine and state; untouched because the whole graph is preflighted before submission.
  * Receipts keep launch identity, source, version and completion across
    re-execution and engines; ghost or mismatched lookups raise.
  * Builder stores preserve declared source scope, row ABI and line-rounded
    output; identity is the only reuse key.
[Timeouts & Locks] Pure in-memory Engine runs; no threads, waits or I/O. [Side-Effect Policy] No filesystem, environment or network access; no production patch.
"""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import pytest
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.prediction.cache_execution import (
    CacheAccessOperation, CacheAccessRequest, CacheExecution,
)
from mindie_sim.prediction.cache_path import (
    CacheAccessGeometry, CacheAccessProgram, CacheGeometry, CachePolicy,
    CacheState, cache_work_stages,
)
from mindie_sim.prediction.cache_representation import (
    CacheLowering, cache_identity, load as legacy_load, store as legacy_store,
)
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]; LINE = 128
PACKED = ROOT / "examples/cases/longctx-decode-20261030/v41-a5-tp1-dp8-ep8.json"
PLAIN = ROOT / "examples/cases/v4f-a3-w8a8-tp8.json"
WRITE_BACK = CachePolicy(write_policy="write_back"); WRITE_THROUGH = CachePolicy(write_policy="write_through"); BYPASS = CachePolicy(bypass=True)
def cache(capacity=4096):
    return CacheState(CacheGeometry(capacity_bytes=capacity, line_bytes=LINE, transfer_bytes=32))
def engine(pool_bytes=1 << 20, mte2_rate=2000.0):
    return Engine([Resource("hbm", 1000.0, unit="bytes"),
                   Resource("mte2", mte2_rate), Resource("mte3", 2000.0),
                   Resource("vector", 1000.0)], pools={"ub": pool_bytes})
def buffered(*, rows=1, row_bytes=LINE, offset=0, stride=None, partial="masked", page=0):
    return CacheAccessGeometry(offset_bytes=offset, rows=rows, row_bytes=row_bytes,
        stride_bytes=row_bytes if stride is None else stride, page_bytes=page, partial_line=partial)
def request(identity="A", *, write=False, policy=None, geometry=None, **kwargs):
    return CacheAccessRequest(
        identity, geometry if geometry is not None else buffered(), write=write,
        policy=policy if policy is not None else CachePolicy(), **kwargs)
def served(e): return dict(e.serviced)
def run(cache_state, access, *, name="access", release=0, source=None, **kwargs):
    e = engine()
    done = CacheExecution(cache_state).access(
        access, e, name, e.submit("ready", release=release), source=source, **kwargs)
    e.advance(until=(done,), first=False)
    return e, done
def check_raises(match, call):
    with pytest.raises(ModelingError, match=match):
        call()
def builder(path=PACKED):
    case = parse_case(json.loads(Path(path).read_text(encoding="utf-8"))); plan = case.model.layer_plans[0]
    return Builder("l0", plan, 0, (RequestView("r0", 1, 100, False, "decode"),),
                   case.model, "decode"), case, plan
def lower(cache_state=None, path=PACKED):
    built, case, plan = builder(path)
    return CacheLowering(built, cache_state if cache_state is not None else cache()), built, case, plan
def packed(cache_state=None, *, rows=5, width=None, region="window", deps=(), **kwargs):
    lowering, built, case, plan = lower(cache_state); identity = cache_identity(region, layer=0, request="r0", block=0)
    lowering.store("kv_store", deps, rows,
                   width if width is not None else plan.cache_row_elements,
                   region, identity=identity,
                   source_residency=kwargs.pop("source_residency", "on_chip"), **kwargs)
    return lowering, built, case, plan
def test_event_time_publication_and_failed_ready_abort():
    store, load = request("A", write=True, policy=WRITE_BACK), request("A")
    for store_release, load_release, hbm in ((10, 0, 128), (0, 10, 0)):
        state, e = cache(128), engine()
        events = (CacheExecution(state).access(load, e, "load",
                                               e.submit("r_load", release=load_release)),
                  CacheExecution(state).access(store, e, "store",
                                               e.submit("r_store", release=store_release)))
        e.advance(until=events, first=False)
        assert served(e)["hbm"] == hbm and state.resident("A") == (1, 1)
    state, e = cache(256), engine(); competitor = e.submit("external_hbm", {"hbm": 128.0}, release=0, latency=0.0); done = CacheExecution(state).access(request("A"), e, "read", e.submit("ready")); e.advance(until=(done,), first=False)
    assert served(e)["hbm"] == 256.0 and competitor.done and state.resident("A") == (1, 0)
    with pytest.raises(ModelingError):
        CacheExecution(state, pool=None).access(request("B"), engine(), "x", engine().submit("r"))
    small = engine(pool_bytes=64)
    with pytest.raises(ExecutionError):
        CacheExecution(state).access(request("B"), small, "x", small.submit("r"))
    state = cache(128); e = engine(); done = CacheExecution(state).access(request("A"), e, "read", e.submit("ready")); e.advance(horizon=0.1)
    assert state.resident_bytes == 0 and state.snapshot()["reserved_bytes"] == 128; assert state.counters["accesses"] == 0
    e.advance(horizon=0.15)
    assert state.resident_bytes == 128 and state.counters["accesses"] == 0
    e.advance(until=(done,), first=False)
    assert state.counters["accesses"] == 1 and state.resident("A") == (1, 0)
    enough = engine(pool_bytes=128); done = CacheExecution(state).access(request("B"), enough, "normal", enough.submit("ready")); enough.advance(until=(done,), first=False)
    assert dict(enough.peak_allocated)["ub"] == 128.0 and dict(enough.allocated)["ub"] == 0
    geometry = buffered(row_bytes=64, partial="read_modify_write"); state = cache(128); reservation = state.prepare_access("A", geometry, write=True, policy=WRITE_THROUGH); result = reservation.result
    reservation.cancel(); e = engine(); done = CacheAccessProgram(result, pool="ub").start(e, "program", e.submit("ready")); e.advance(until=(done,), first=False)
    assert served(e)["hbm"] == 256.0 and served(e)["mte2"] == 64.0
    with pytest.raises(ModelingError):
        CacheAccessProgram(result, pool=None).start(engine(), "no_pool", engine().submit("r"))
    e, _ = run(cache(128), request("A", geometry=geometry, write=True, policy=WRITE_THROUGH))
    assert served(e)["hbm"] == 256.0 and served(e)["mte3"] == 64.0
    load_state = cache(128); load = load_state.prepare_access("L", buffered(row_bytes=64)); load_result = load.result; load.cancel()
    e = engine(mte2_rate=1000.0); program = CacheAccessProgram(load_result, pool="ub", source_required=True)
    with pytest.raises(ModelingError):
        program.start(e, "missing", e.submit("ready"))
    source = e.submit("source", release=10); done = program.start(e, "load", e.submit("ready2"), source=source); e.advance(until=(done,), first=False)
    assert done.started >= 10.0 and done.time == pytest.approx(10.192); assert served(e)["hbm"] == 128.0
def test_state_access_projection_and_runtime_receipts():
    access = request("index@r0", geometry=buffered(rows=2, row_bytes=32, stride=1025))
    projected = CacheExecution(cache()).state_access(
        access, occurrence="indexer_prolog:index_pool_tail_store",
        storage="index_pool_tail", request_id="r0", layer=0, rank=0, state_version=3)
    assert projected["span"] == [[0, 1], [8, 1]] and projected["bytes"] == 64; assert projected["direction"] == "read" and projected["position"] == "cache"; assert projected["evidence"] == "declaration" and projected["runtime_bound"] is False
    state, access, execution = cache(256), request("A"), None; execution = CacheExecution(state); e = engine(); first_event = execution.access(access, e, "first", e.submit("first_ready"))
    e.advance(until=(first_event,), first=False); first = execution.binding("first"); second = execution.access(access, e, "second", e.submit("second_ready", release=10)); e.advance(until=(second,), first=False)
    second_receipt = execution.binding("second"); third = execution.access(access, e, "first", e.submit("third_ready", release=20)); e.advance(until=(third,), first=False); old = execution.binding(first.occurrence)
    assert second_receipt.occurrence != first.occurrence; assert old.event is first_event and old.completed == first.completed; assert old.output == first.output and old.state_version == first.state_version; assert execution.binding(second_receipt.occurrence).event is second
    check_raises("cache_access_occurrence_unknown:ghost",
                 lambda: execution.state_access(access, occurrence="ghost", storage="A"))
    check_raises("cache_access_binding_mismatch:second",
                 lambda: execution.state_access(access, occurrence="second",
                                                storage="A", binding=old))
    receipt = execution.state_access(access, occurrence=first.occurrence, storage="A")
    assert receipt["runtime_bound"] and not receipt["output_actual"]; assert receipt["grant"] is None and receipt["grant_connected"] is False; assert receipt["staging"]["size"] == 128 and "T08_grant_not_connected" in receipt["gaps"]
    source = e.submit("real_source", release=0); sourced = execution.access(access, e, "sourced", e.submit("sourced_ready"), source=source); e.advance(until=(sourced,), first=False)
    assert execution.state_access(access, occurrence="sourced", storage="A")["source"] is source
    state = cache(128); blocker = state.prepare_access("A", buffered()); access, e = request("B"), engine(); done = CacheExecution(state).access(access, e, "B", e.submit("ready"))
    completed = []
    successor = e.submit("successor", after=(done,),
                         on_complete=lambda _event: completed.append(1))
    with pytest.raises(ModelingError):
        e.advance(until=(done,), first=False)
    e.advance(horizon=1.0, first=False)
    assert completed == [] and not done.done and not successor.done; assert done.sequence not in e._promises and dict(e.allocated)["ub"] == 0.0; assert state.snapshot()["reserved_bytes"] == 128
    blocker.cancel(); retry = CacheExecution(state).access(access, e, "B_retry", e.submit("retry_ready")); e.advance(until=(retry,), first=False)
    assert retry.done and state.resident("B") == (1, 0)
def test_flush_writeback_and_dirty_victim_terminal_boundary():
    state = cache(128); result = state.access("A", 256, write=True, policy=WRITE_BACK); e = engine(); access = CacheAccessProgram(result, pool="ub").start(e, "access", e.submit("ready"))
    plan = state.flush()
    assert plan.bytes == 128 and state.dirty_bytes == 128
    flush = plan.start(e, "flush", access)
    assert state.dirty_bytes == 128  # completion boundary, not submission
    e.advance(until=(flush,), first=False)
    assert state.dirty_bytes == 0 and state.counters["hbm_write_bytes"] == 256
    assert served(e)["hbm"] == 256.0
    empty_engine, empty = engine(), state.flush()
    assert empty.bytes == 0
    empty_engine.advance(until=(empty.start(empty_engine, "flush2", empty_engine.submit("ready")),), first=False)
    assert empty_engine.serviced.get("hbm", 0.0) == 0.0
    stale_state = cache(128); stale_state.access("A", 128, write=True, policy=WRITE_BACK); stale_plan = stale_state.flush(); stale_plan.cancel()
    with pytest.raises(ModelingError):
        stale_plan.start(engine(), "stale", engine().submit("ready"))
    evict_state = cache(256); evict_state.access("A", 256, write=True, policy=WRITE_BACK); evicted = evict_state.access("B", 256, write=True, policy=WRITE_BACK); evict_engine = engine()
    evict_engine.advance(until=(CacheAccessProgram(evicted, pool="ub").start(
        evict_engine, "evict", evict_engine.submit("ready")),), first=False)
    assert served(evict_engine)["hbm"] == 256.0  # victim drain; dirty tail flushes later
    state = cache(256); access = request("A", write=True, policy=WRITE_THROUGH); e = engine(); done = CacheExecution(state).access(access, e, "write", e.submit("ready"))
    e.advance(horizon=0.07)
    for policy in (BYPASS, None):
        with pytest.raises(ModelingError):
            state.prepare_access("A", buffered(), policy=policy or CachePolicy())
    e.advance(until=(done,), first=False); bypass = state.prepare_access("A", buffered(), policy=BYPASS)
    assert bypass.result.hbm_read_bytes == 128
    bypass.publish()
def test_builder_lowering_store_binds_typed_view_and_real_ports():
    lowering, built, case, plan = packed()
    lowering.load("kv_read", ("kv_store",), 5, plan.cache_row_elements,
                  "window", identity=lowering.requests[0][1].identity)
    e = engine(); events = lowering.execute(e); e.advance(until=events, first=False); store_event, load_event = events
    producer, access = lowering.requests[0][1], lowering.requests[1][1]
    assert load_event.started >= store_event.time; assert served(e)["hbm"] == 2816.0 and served(e)["mte2"] == 7840.0; assert served(e)["mte3"] == 2720.0 and served(e)["vector"] == 10320.0; assert producer.producer.endswith("l0.kv_store") and producer.output.endswith(".output"); assert producer.view.layout == "packed_row"; assert producer.view.payload_bytes + producer.view.scale_bytes == 2720; assert access.identity == producer.identity
    projected = lowering.state_access("kv_store", occurrence="kv_store", storage="window")
    assert projected["producer"] == producer.producer and projected["runtime_bound"]; assert projected["storage"] == "window" and projected["payload_bytes"] == producer.view.payload_bytes
    binding = lowering.execution.binding("kv_store")
    assert binding.producer == producer.producer and binding.output == producer.output; assert binding.storage == "window" and binding.completed == store_event.time; assert binding.state_version < lowering.state.version; assert binding.staging["size"] == 128 and binding.grant is None
    metadata = dict(built.ops[0].metadata)
    assert metadata["cache_typed_view"]["producer"] == producer.producer; assert metadata["cache_access_program"]["producer"] == producer.producer
    resolved = lowering.state.prepare_access(
        producer.identity, producer.geometry, write=True, policy=producer.policy)
    e2 = engine()
    program = CacheAccessProgram(
        resolved.result, pool="ub", source_bytes=producer.source_bytes,
        convert_ops=producer.convert_ops, source_residency=producer.source_residency,
        write_resource="mte3").start(e2, "program", e2.submit("ready"))
    e2.advance(until=(program,), first=False)
    assert served(e2) == {"hbm": 2816.0, "mte2": 5120.0, "mte3": 2720.0, "vector": 10320.0}
    resolved.cancel()
def test_lowering_ports_join_and_preflight_before_submission():
    lowering, _, _, plan = lower(); lowering.load("kv_read", ("src",), 1, 64, "mla_kv", identity="A"); e = engine(); generic = e.submit("generic_ready", release=10)
    check_raises("cache_lowering_port_unbound:src", lambda: lowering.execute(e, ready=generic)); source = e.submit("src_event", release=10); events = lowering.execute(e, ready=generic, ready_by_name={"src": source}); e.advance(until=events, first=False)
    assert events[0].started == 10.0 and events[0].time == pytest.approx(10.192); assert served(e)["hbm"] == 128.0; assert lowering.execution.binding("kv_read").completed == 10.192
    alias, _, _, plan = packed(); store_op = alias.requests[0][1].producer
    alias.load("kv_read", (store_op + ".output",), 5, plan.cache_row_elements,
               "mla_kv", identity="A")
    alias_engine = engine(); check_raises("cache_lowering_port_unregistered", lambda: alias.execute(alias_engine))
    assert alias_engine._sequence == 0
    order, _, _, plan = packed(); order.load("later", ("earlier",), 5, plan.cache_row_elements, "mla_kv", identity="A"); order.load("earlier", (), 5, plan.cache_row_elements, "mla_kv", identity="A"); order_engine = engine()
    check_raises("cache_lowering_dependency_order:earlier", lambda: order.execute(order_engine))
    assert order_engine._sequence == 0
    missing, _, _, plan = packed(); missing.load("kv_read", ("missing",), 5, plan.cache_row_elements, "mla_kv", identity="A"); missing_engine = engine(); snapshot, version = missing.state.snapshot(), missing.state.version
    check_raises("cache_lowering_port_unbound:missing",
                 lambda: missing.execute(missing_engine,
                                         ready=missing_engine.submit("generic", release=10)))
    assert missing_engine.completed_count == 0; assert all(value == 0.0 for value in missing_engine.serviced.values()); assert missing.state.snapshot() == snapshot and missing.state.version == version; assert not missing._executions
    resource, _, _, plan = packed(); resource.load("first", (), 1, 64, "mla_kv", identity="A")
    resource.store("second", (), 5, plan.cache_row_elements, "window",
                   identity="B", source_residency="on_chip")
    sparse_engine = Engine([Resource("hbm", 1000.0, unit="bytes"),
                            Resource("mte2", 2000.0), Resource("mte3", 2000.0)],
                           pools={"ub": 1 << 20})
    snapshot, version = resource.state.snapshot(), resource.state.version; check_raises("vector_resource_missing", lambda: resource.execute(sparse_engine))
    assert sparse_engine._sequence == 0 and sparse_engine.completed_count == 0; assert resource.state.snapshot() == snapshot and resource.state.version == version
    joined, _, _, plan = lower(); joined.load("kv_read", (), 1, 64, "mla_kv", identity="A"); e = engine()
    events = joined.execute(e, ready=e.submit("generic", release=10),
                            ready_by_name={"kv_read": e.submit("named", release=0)})
    e.advance(until=events, first=False)
    assert events[0].started == 10.0 and events[0].time == pytest.approx(10.192)
    state = cache(256); access = request("A", geometry=buffered(row_bytes=64), source_port="src"); e = engine(mte2_rate=1000.0); ready, execution = e.submit("ready"), CacheExecution(state)
    check_raises("cache_access_source_port_unbound:src",
                 lambda: execution.access(access, e, "declared", ready))
    source = e.submit("src", release=5); done = execution.access(access, e, "declared", ready, source=source); e.advance(until=(done,), first=False)
    assert done.started == 5.0 and done.time == pytest.approx(5.192)
    assert execution.binding("declared").source is source
    via_deps = request("B", geometry=buffered(row_bytes=64), source_port="src"); deps_engine = engine(); deps_execution = CacheExecution(cache(256)); deps_source = deps_engine.submit("src", release=0)
    bound = deps_execution.access(via_deps, deps_engine, "via_deps",
                                  deps_engine.submit("ready"), deps={"src": deps_source})
    deps_engine.advance(until=(bound,), first=False)
    assert bound.done and deps_execution.binding("via_deps").source is deps_source
    with pytest.raises(ModelingError, match="cache_access_dependency_unbound:x"):
        deps_execution.access(request("C", deps=("x",)), deps_engine, "no_deps",
                              deps_engine.submit("r2"))
    with pytest.raises(ModelingError, match="cache_access_dependency_unbound:y"):
        deps_execution.access(request("D", deps=("y",)), deps_engine, "missing_dep",
                              deps_engine.submit("r3"), deps={})
    with pytest.raises(ModelingError, match="cache_access_operation_source_missing"):
        deps_execution.access(request("E"), deps_engine, "required",
                              deps_engine.submit("r4"), source_required=True)
def test_failed_issued_stage_releases_staging_and_reservation(monkeypatch):
    state = cache(128); access = request("A"); e = engine(); done = CacheExecution(state).access(access, e, "aborted", e.submit("ready"))
    completed = []
    successor = e.submit("successor", after=(done,),
                         on_complete=lambda _event: completed.append(1))
    real_submit = e.submit; injected = {"done": False}
    def fail_publish(name, *args, **kwargs):
        if name.endswith(":publish") and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("injected_stage_submission_failure")
        return real_submit(name, *args, **kwargs)
    monkeypatch.setattr(e, "submit", fail_publish)
    with pytest.raises(RuntimeError, match="injected_stage_submission_failure"):
        e.advance(until=(done,), first=False)
    e.advance(horizon=1.0, first=False)
    assert completed == [] and not done.done and not successor.done
    assert done.sequence not in e._promises and dict(e.allocated)["ub"] == 0.0
    assert state.snapshot()["reserved_bytes"] == 0 and state.inflight_identities == 0
    assert state.resident("A") == (0, 0)
    monkeypatch.setattr(e, "submit", real_submit); retry = CacheExecution(state).access(access, e, "retry", e.submit("retry_ready")); e.advance(until=(retry,), first=False)
    assert retry.done and state.resident("A") == (1, 0)


def test_inflight_write_holds_identity_capacity():
    state = cache(128); access = request("A", write=True, policy=WRITE_THROUGH); e = engine(); done = CacheExecution(state).access(access, e, "A", e.submit("ready"))
    e.advance(horizon=0.07)
    assert state.resident("A") == (1, 0) and not done.done
    with pytest.raises(ModelingError, match="cache_capacity_inflight_held"):
        state.prepare_access("B", buffered(), write=True, policy=WRITE_BACK)
    e.advance(until=(done,), first=False); state.prepare_access("B", buffered(), write=True, policy=WRITE_BACK).cancel()

def test_bypass_write_holds_identity_until_completion():
    state = cache(128)
    policy = CachePolicy(bypass=True, write_policy="write_through")
    access = request("A", write=True, policy=policy)
    e = engine()
    done = CacheExecution(state).access(access, e, "write", e.submit("ready"))
    e.advance(horizon=0.07)
    assert not done.done and state.inflight_identities == 1
    with pytest.raises(ModelingError, match="cache_identity_access_in_flight"):
        state.prepare_access("A", CacheAccessGeometry(rows=1, row_bytes=128))
    e.advance(until=(done,), first=False)
    assert done.time == pytest.approx(0.192) and state.inflight_identities == 0
    read = CacheExecution(state).access(request("A"), e, "read", e.submit("read_ready"))
    e.advance(until=(read,), first=False)
    assert read.done and state.resident("A") == (1, 0)
    assert served(e)["hbm"] == 256.0

def test_occurrences_cross_engines_and_sources_cannot_be_fabricated():
    state, access = cache(256), request("A"); first_engine, first_execution = engine(), CacheExecution(state); first = first_execution.access(access, first_engine, "first", first_engine.submit("r")); first_engine.advance(until=(first,), first=False)
    receipt = first_execution.binding("first"); other_source = first_engine.submit("other_source", release=0)
    check_raises("cache_access_source_not_recorded",
                 lambda: first_execution.state_access(access, occurrence=receipt.occurrence,
                                                      storage="A", source=other_source))
    second_engine, second_execution = engine(), CacheExecution(cache(256)); second = second_execution.access(access, second_engine, "first", second_engine.submit("r")); second_engine.advance(until=(second,), first=False); second_receipt = second_execution.binding("first")
    assert second_receipt.occurrence != receipt.occurrence; assert second_execution.binding(receipt.occurrence) is None; assert receipt.event is first and receipt.completed == first.time
    same_engine, same_execution = engine(), CacheExecution(cache(256)); first_event = same_execution.access(access, same_engine, "first", same_engine.submit("a")); same_engine.advance(until=(first_event,), first=False)
    second_event = same_execution.access(access, same_engine, "second",
                                         same_engine.submit("b", release=10))
    same_engine.advance(until=(second_event,), first=False); first_view = same_execution.state_access(access, occurrence="first", storage="A"); second_view = same_execution.state_access(access, occurrence="second", storage="A")
    assert first_view["completed"] == first_event.time and first_view["started"] == 0.0; assert second_view["completed"] == second_event.time and second_view["started"] == 10.0; assert second_view["state_version"] > first_view["state_version"]; assert first_view["span"] == second_view["span"] == [[0, 1]]
def test_sparse_staging_partial_lines_and_outstanding_credit():
    empty = CacheGeometry(capacity_bytes=0, line_bytes=LINE, transfer_bytes=32); sparse = buffered(rows=2, row_bytes=1, stride=128); reservation = CacheState(empty).prepare_access("S", sparse)
    assert reservation.result.gm_bytes == reservation.result.onchip_bytes == 64; assert reservation.result.staging_bytes == 64 and reservation.result.hbm_read_bytes == 256; assert CacheAccessOperation(CacheState(empty), request("S", geometry=sparse)).staging_bytes() == 64
    mini = engine(pool_bytes=64)
    done = CacheExecution(CacheState(empty)).access(
        request("S", geometry=sparse), mini, "sparse", mini.submit("ready"))
    mini.advance(until=(done,), first=False)
    assert dict(mini.peak_allocated)["ub"] == 64.0 and dict(mini.allocated)["ub"] == 0.0
    program_engine = engine(pool_bytes=64)
    program = CacheAccessProgram(reservation.result, pool="ub").start(
        program_engine, "program", program_engine.submit("ready"))
    program_engine.advance(until=(program,), first=False)
    assert served(program_engine)["hbm"] == 256.0
    reservation.cancel()
    for policy, expected in (("masked", 0), ("read_modify_write", 256)):
        geometry = buffered(offset=20, rows=2, row_bytes=100, stride=128, partial=policy)
        e, _ = run(cache(512), request("A", geometry=geometry, write=True,
                                       policy=WRITE_THROUGH))
        assert served(e).get("hbm", 0.0) == 256.0 + expected
    access = request("A", geometry=buffered(row_bytes=2720), outstanding=4, response_ms=0.001); e, _ = run(cache(4096), access)
    assert served(e)["hbm"] == 2816.0
    lowering, _, _, plan = packed(cache(4096), rows=5, outstanding=4, response_ms=0.001)
    lowering.store("credit_store", (), 5, plan.cache_row_elements, "window",
                   identity="B", source_residency="unresolved",
                   outstanding=4, response_ms=0.001)
    credit = lowering.requests[1][1]; e = engine(); done = CacheExecution(lowering.state).access(credit, e, "credit", e.submit("ready")); e.advance(until=(done,), first=False)
    reservation = lowering.state.prepare_access(
        credit.identity, credit.geometry, write=True, policy=credit.policy)
    program_engine = engine()
    program = CacheAccessProgram(
        reservation.result, pool="ub", source_bytes=credit.source_bytes,
        convert_ops=credit.convert_ops, source_residency=credit.source_residency,
        write_resource="mte3", outstanding=credit.outstanding,
        response_ms=credit.response_ms).start(
            program_engine, "credit_program", program_engine.submit("ready"))
    program_engine.advance(until=(program,), first=False)
    assert served(program_engine) == served(e) and program.time == pytest.approx(done.time)
    reservation.cancel()
def test_lowering_source_conversion_ready_reversal_and_stage_dag():
    for residency, source_hbm in (("on_chip", 0), ("unresolved", 5120)):
        lowering, built, case, plan = packed(source_residency=residency)
        lowering.load("kv_read", ("kv_store",), 5, plan.cache_row_elements, "window",
                      identity=lowering.requests[0][1].identity)
        assert built.ops[0].mte2_bytes == 5120 and built.ops[0].vector_ops == 10320
        e = engine()
        external, store_ready = e.submit("external"), e.submit("r_store")
        events = lowering.execute(e, ready=external, ready_by_name={
            "rotary": external, "kv_store": store_ready, "kv_read": e.submit("r_load")})
        e.advance(until=events, first=False)
        assert served(e)["hbm"] == source_hbm + 2816
        assert served(e)["mte2"] == 5120 + 2720
        assert served(e)["mte3"] == 2720 and served(e)["vector"] == 10320
    lowering, built, case, plan = packed()
    lowering.load("kv_read", ("kv_store",), 5, plan.cache_row_elements, "window",
                  identity=lowering.requests[0][1].identity)
    def iteration(*, store_first):
        e = engine()
        store_ready = e.submit("r_store", release=0 if store_first else 10)
        load_ready = e.submit("r_load", release=0 if not store_first else 10)
        external = e.submit("external", release=0)
        events = lowering.execute(e, ready=external, ready_by_name={
            "rotary": external, "kv_store": store_ready, "kv_read": load_ready})
        e.advance(until=events, first=False)
        return served(e)
    first, second = iteration(store_first=True), iteration(store_first=False)
    assert first["hbm"] == 2816.0 and second["hbm"] == 2816.0; assert second["mte2"] == 7840.0 and second["mte3"] == 2720.0; assert second["vector"] == 10320.0 and lowering.state.resident_bytes == 2816
    for residency, labels, finish in (
            ("on_chip", ("source_onchip", "convert", "stage", "writeback", "retire"), 17.056),
            ("unresolved", ("source_hbm", "source_onchip", "convert", "stage",
                            "writeback", "retire"), 22.176)):
        lowering, _, _, plan = packed(rows=5, source_residency=residency)
        access = lowering.requests[0][1]
        assert access.source_bytes == 5120
        reservation = lowering.state.prepare_access(
            access.identity, access.geometry, write=True, policy=access.policy)
        stages = cache_work_stages(
            reservation.result, source_bytes=access.source_bytes,
            source_residency=residency, convert_ops=access.convert_ops)
        assert tuple(item.label for item in stages) == labels
        after = {item.label: item.after for item in stages}
        assert after["source_onchip"] == (("ready",) if residency == "on_chip"
                                          else ("source_hbm",))
        assert (after.get("source_hbm") == ("ready",)) if residency == "unresolved" else True
        assert after["convert"] == ("source_onchip",) and after["stage"] == ("convert",)
        assert after["writeback"] == ("stage",) and after["retire"] == ("writeback",)
        reservation.cancel()
        operation = CacheAccessOperation(lowering.state, access, write_resource="mte3")
        e = engine()
        done = operation.start(e, "packed", e.submit("ready"))
        e.advance(until=(done,), first=False)
        assert done.time == pytest.approx(finish)
        assert operation.stage_events["source_onchip"].started == pytest.approx(
            0.0 if residency == "on_chip" else 5.12)
        assert operation.stage_events["retire"].time == pytest.approx(finish)
        program_state = cache()
        program_reservation = program_state.prepare_access(
            access.identity, access.geometry, write=True, policy=access.policy)
        program_engine = engine()
        program = CacheAccessProgram(
            program_reservation.result, pool="ub", source_bytes=access.source_bytes,
            convert_ops=access.convert_ops, source_residency=residency,
            write_resource="mte3").start(program_engine, "packed_program",
                                         program_engine.submit("ready"))
        program_engine.advance(until=(program,), first=False)
        assert program.time == pytest.approx(finish) and served(program_engine) == served(e)
        program_reservation.cancel()
    lowering, _, _, plan = packed()
    lowering.load("kv_read", ("kv_store",), 5, plan.cache_row_elements, "window",
                  identity=lowering.requests[0][1].identity)
    e = engine(); external = e.submit("external", release=0); e.submit("competitor", {"hbm": 128.0}, release=0); first = lowering.execute(e, ready=external, ready_by_name={
        "rotary": external, "kv_store": e.submit("first_store", release=0),
        "kv_read": e.submit("first_load", release=10)})
    second = lowering.execute(e, ready=external, ready_by_name={
        "rotary": external, "kv_store": e.submit("second_store", release=20),
        "kv_read": e.submit("second_load", release=30)})
    e.advance(until=tuple(first) + tuple(second), first=False)
    assert served(e)["hbm"] == 5760.0 and served(e)["mte2"] == 15680.0; assert served(e)["mte3"] == 5440.0 and served(e)["vector"] == 20640.0; assert lowering.state.resident_bytes == 2816
    built, case, plan = builder(); lowering = CacheLowering(built, cache(128))
    lowering.store("kv_store", ("rotary",), 4, 64, "mla_kv", identity="A",
                   write_policy="write_back", source_residency="on_chip")
    assert built.ops[-1].hbm_bytes == 512.0
    for index in range(2):
        e = engine()
        ready = e.submit("ready", release=0)
        events = lowering.execute(e, ready=ready, ready_by_name={"rotary": ready})
        e.advance(until=events, first=False)
        direct = served(e)["hbm"]
        flush = lowering.flush(e, f"flush{index}", e.submit("flush_ready", release=10))
        e.advance(until=(flush,), first=False)
        assert direct == 384.0 and served(e)["hbm"] == 512.0
        assert lowering.state.dirty_bytes == 0
        assert dict(e.peak_allocated)["ub"] == 128.0 and dict(e.allocated)["ub"] == 0
    state = cache(128); state.access("A", 128, write=True, policy=WRITE_BACK); e = engine(); operation = CacheAccessOperation(state, request("B"))
    done = operation.start(e, "B", e.submit("ready")); e.advance(horizon=0.2)
    assert operation.stage_events["drain"].time <= operation.stage_events["fill"].started; assert state.snapshot()["resident_bytes"] <= 128; assert state.snapshot()["reserved_bytes"] == 128 and state.resident("A") == (0, 0)
    e.advance(until=(done,), first=False)
    assert state.resident("B") == (1, 0) and state.resident_bytes == 128; assert served(e)["hbm"] == 256.0 and state.counters["hbm_write_bytes"] == 128
def test_store_source_scope_row_abi_and_line_rounded_write():
    for residency in ("on_chip", "unresolved"):
        built, case, plan = builder()
        state, rows = cache(), 5
        identity = cache_identity("window", layer=0, request="r0", block=0)
        legacy_store(built, "kv_store", ("rotary",), rows, plan.cache_row_elements,
                     "window", state=state, identity=identity, source_residency=residency)
        op = built.ops[-1]
        fmt = cache_format(case.model, "window")
        stored, source = rows * fmt.row_bytes(plan.cache_row_elements), rows * plan.cache_row_elements * 2
        ledger = dict(op.metadata)["cache_hbm_write_ledger"]
        assert ledger["hbm_write_bytes"] == ledger["direct_write_bytes"] + ledger["writeback_bytes"]
        assert dict(op.metadata)["cache_access"]["staging_bytes"] == 128
        assert "cache_execution_gap" in dict(op.metadata)
        if residency == "on_chip":
            assert op.mte2_bytes == source and op.mte3_bytes == stored
            assert op.hbm_bytes == pytest.approx(128 * ((stored + 127) // 128))
            assert op.hbm_bytes > stored
            assert dict(op.metadata)["cache_source_residency"] == "on_chip"
        else:
            assert op.hbm_bytes == source + 128 * ((stored + 127) // 128)
            assert op.mte2_bytes == source and op.mte3_bytes == stored
            assert dict(op.metadata)["cache_source_residency"] == "unresolved"
            assert dict(op.metadata)["cache_representation"]["source_read_scope"] == (
                "unresolved_source_read_counted_in_HBM")
    built, case, plan = builder(); state = cache()
    legacy_store(built, "plain_store", ("rotary",), 3, 64, "mla_kv", state=state,
                 identity="plain", source_residency="on_chip")
    op = built.ops[-1]
    assert op.local_shape == (3, 128) and op.vector_ops == 3 * 128; assert op.dtype == "bf16" and op.hbm_bytes == 384; assert op.mte2_bytes == op.mte3_bytes == 384
    metadata = dict(case.model.metadata); representations = dict(metadata.get("cache_representations", {}))
    representations["window"] = {"dtype": "fp4_e2m1"}; metadata["cache_representations"] = representations
    model = replace(case.model, metadata=tuple(metadata.items()))
    custom = Builder("l0", plan, 0, (RequestView("r0", 1, 100, False, "decode"),),
                     model, "decode")
    lowering = CacheLowering(custom, cache())
    lowering.store("kv_store", (), 5, 64, "window", identity="W",
                   source_residency="unresolved")
    expected = 5 * 64 * dtype_bytes(model.activation_dtype)
    assert lowering.requests[0][1].source_bytes == expected != int(custom.ops[-1].mte2_bytes)
    built, case, plan = builder(PLAIN); legacy_store(built, "kv_store", ("rotary",), 2, plan.cache_row_elements, "window")
    op = built.ops[-1]
    assert dict(op.metadata)["cache_resolution"] == "legacy_no_cache_declaration"; assert op.local_shape == (2, 1024) and op.padded_shape == (8, 1024); assert op.hbm_bytes == 2 * (2 * 1024); assert op.mte2_bytes == op.mte3_bytes == 2 * 1024 and op.vector_ops == 2 * 1024
    built, case, plan = builder(); legacy_store(built, "kv_store", ("rotary",), 2, plan.cache_row_elements, "window")
    op = built.ops[-1]; fmt = cache_format(case.model, "window")
    stored, source = 2 * fmt.row_bytes(plan.cache_row_elements), 2 * plan.cache_row_elements * 2
    assert op.hbm_bytes == source + stored; assert op.mte2_bytes == source and op.mte3_bytes == stored
def test_identity_geometry_state_and_unidentified_cold_path():
    first = cache_identity("window", layer=0, request="r0", block=0)
    assert first != cache_identity("window", layer=0, request="r0", block=1); assert cache_identity("window", layer=0, request="a,b") != cache_identity(
        "window", layer=0, request="a")
    assert cache_identity("window", layer=1, request="r0") != first
    with pytest.raises(ModelingError):
        cache_identity("window")
    built, case, plan = builder(); state = cache()
    legacy_store(built, "kv_store", ("rotary",), 5, plan.cache_row_elements, "window",
                 state=state, identity=first, source_residency="on_chip")
    legacy_load(built, "kv_read", ("kv_store",), 5, plan.cache_row_elements,
                "window", state=state, identity=first)
    written, read = built.ops
    assert read.hbm_bytes == 0 and read.mte2_bytes == written.mte3_bytes; assert dict(read.metadata)["cache_access"]["hit_lines"] == dict(
        written.metadata)["cache_access"]["hbm_write_lines"]
    other_built, other_case, other_plan = builder(); other_state = cache()
    legacy_store(other_built, "kv_store", (), 1, 64, "mla_kv", state=other_state,
                 identity=first, source_residency="on_chip")
    legacy_load(other_built, "kv_read", (), 1, 64, "mla_kv", state=other_state,
                identity=cache_identity("window", layer=0, request="r0", block=1))
    assert other_built.ops[-1].hbm_bytes == 128; assert dict(written.metadata)["cache_geometry"]["capacity_bytes"] == 4096
    with pytest.raises(ModelingError):
        legacy_store(built, "kv_bad", ("rotary",), 1, 64, "mla_kv", state=state,
                     geometry=CacheGeometry(128, LINE, 32), identity="A")
    with pytest.raises(ModelingError):
        legacy_load(built, "kv_no_geometry", ("kv_store",), 1, 64, "mla_kv")
    with pytest.raises(ModelingError):
        legacy_store(built, "kv_no_geometry", ("rotary",), 1, 64, "mla_kv",
                     source_residency="on_chip")
    with pytest.raises(ModelingError):
        legacy_store(built, "kv_identity_only", ("rotary",), 1, 64, "mla_kv", identity="A")
    attached, _, plan = builder(); attached_state = cache()
    attached.cache_state = attached_state
    legacy_store(attached, "kv_store", ("rotary",), 5, plan.cache_row_elements, "window",
                 identity="kv:layer0:block0", source_residency="on_chip")
    legacy_load(attached, "kv_read", ("kv_store",), 5, plan.cache_row_elements,
                "window", identity="kv:layer0:block0")
    assert dict(attached.ops[0].metadata)["cache_resolution"] == "builder_cache_state"; assert attached.ops[-1].hbm_bytes == 0 and attached_state.resident_bytes > 0
    cold, _, plan = builder()
    cold_state = cache()
    legacy_store(cold, "kv_store", ("rotary",), 1, 64, "mla_kv", state=cold_state,
                 source_residency="on_chip")
    legacy_load(cold, "kv_read", ("kv_store",), 1, 64, "mla_kv", state=cold_state)
    for op in cold.ops:
        metadata = dict(op.metadata)
        assert metadata["cache_access"]["cold_path"] is True
        assert metadata["cache_cold_path"] == "unidentified_storage"
        assert cold.ops[-1].hbm_bytes == 128 and cold_state.resident_bytes == 0
    ephemeral, _, plan = builder()
    legacy_store(ephemeral, "kv_store", ("rotary",), 5, plan.cache_row_elements, "window",
                 geometry=CacheGeometry(4096, LINE, 32), identity="A",
                 source_residency="on_chip")
    metadata = dict(ephemeral.ops[-1].metadata)
    assert metadata["cache_resolution"] == "ephemeral_cold"; assert metadata["cache_access"]["cold_path"] is False; assert ephemeral.ops[-1].hbm_bytes == 2816
    with pytest.raises(ModelingError):
        legacy_store(ephemeral, "kv_direct", ("rotary",), 5, plan.cache_row_elements,
                     "window", bypass=True, ready_ms=0.001)
def test_write_back_store_and_load_carry_dirty_eviction_ledger():
    built, case, plan = builder()
    state = cache(128)
    legacy_store(built, "kv_store", ("rotary",), 4, 64, "mla_kv", state=state,
                 identity="A", write_policy="write_back", source_residency="on_chip")
    op = built.ops[-1]
    metadata = dict(op.metadata)
    assert op.hbm_bytes == 384; assert metadata["cache_hbm_write_ledger"] == {
        "direct_write_bytes": 384, "writeback_bytes": 0, "hbm_write_bytes": 384}
    assert state.resident("A") == (1, 1) and state.dirty_bytes == 128
    e = engine()
    plan_write = state.flush()
    e.advance(until=(plan_write.start(e, "flush", e.submit("ready")),), first=False)
    assert plan_write.bytes == 128 and op.hbm_bytes + plan_write.bytes == 512
    dirty = cache(128)
    dirty.access("A", 128, write=True, policy=WRITE_BACK)
    built, case, plan = builder()
    legacy_load(built, "kv_read", ("kv_store",), 1, 64, "mla_kv", state=dirty,
                identity="B")
    load_op = built.ops[-1]
    metadata = dict(load_op.metadata)
    assert load_op.hbm_bytes == 256 and load_op.mte2_bytes == 128; assert metadata["cache_access"]["hbm_write_bytes"] == 128; assert metadata["cache_hbm_write_ledger"]["writeback_bytes"] == 128
