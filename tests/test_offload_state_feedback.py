"""Bound state-access tier projection for target/draft/padding.

[Role]
  Prove issued KV/state traffic is projected from actual occurrence-bound
  target/draft/padding access records, not from model layer counts, total
  produced tokens or cache-capacity differences.

[Invariants]
  * Per-rank DCP rows and full-ring index-tail rows keep their actual spans.
  * An empty fragment has zero state transfer; a positive opaque access is
    honoured without inventing a model layer.
  * Target and draft producers are distinct occurrences and dependencies.
  * Padding has no request state access while its physical graph work remains.

[Timeouts & Locks]
  Small in-memory Engine/Execution fixtures; no device, files or locks.

[Side-Effect Policy]
  Local cases and local Engine events only.
"""
from __future__ import annotations

from dataclasses import replace
import gc
import weakref
from types import SimpleNamespace

import pytest

from mindie_sim import predict
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.offload import (
    OffloadPolicy,
    StateAccess,
    _accesses_from_ops,
    bind_offload_access_events,
    project_state_tiers,
    release_offload_access,
    schedule_draft_kv_writeback,
    schedule_kv_layer_offload,
    schedule_kv_layer_writeback,
    schedule_kv_iteration_offload,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.compose import compose_case
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_draft_and_media import draft_case
from test_prediction_system import case, linear_work


def _with_offload(case_spec, *, ddr=0.5, ssd=0.25):
    metadata = tuple(dict(case_spec.runtime.metadata).items()) + (
        ("offload", {"kvc_ddr_ratio": ddr, "kvc_ssd_ratio": ssd}),)
    return replace(case_spec, runtime=replace(case_spec.runtime, metadata=metadata))


def _execution(case_spec):
    return Execution(case_spec, CostModel(case_spec), include_serving_io=False)


def _work(views=None, ops=()):
    if views is None:
        views = (RequestView("r", 1, 4, True, "decode"),)
    return SimpleNamespace(plan=IterationPlan(tuple(views), iteration=0),
                           ops=tuple(ops))


def _access(*, bytes, rank=None, occurrence="occ", request="r", scope="target",
            direction="write", storage="gqa_kv", layer=0, stage=0):
    return StateAccess(
        occurrence=occurrence, producer=occurrence, request=request,
        stage=stage, layer=layer, direction=direction, position="cache",
        output=storage, storage=storage, span=((0, 1),), rank=rank,
        scope=scope, bytes=bytes,
    )


def _serviced(execution, events):
    execution.engine.advance(until=tuple(events), first=False)
    return dict(execution.engine.serviced)


def test_offload_uses_the_same_memory_instance_as_the_scheduler():
    case_spec = _with_offload(parse_case(case(output=1)))
    memory = Memory(case_spec)
    execution = _execution(case_spec)
    scheduler = Scheduler(case_spec, list(case_spec.workload.requests), memory)
    scheduler.bind_execution(execution)
    assert scheduler.memory is memory
    assert execution.memory is memory
    access = _access(bytes=32, occurrence="shared-memory:writer")
    producer = execution.engine.submit("producer", latency=1.0)
    schedule_kv_layer_writeback(
        execution, "same", 0, 0, 0, _work(), producer,
        state_accesses=(access,))
    assert execution._offload_memory is memory
    assert scheduler.grant_state([("shared-grant", 4096.0)]) is True
    assert memory.granted[0][0] == 4096.0
    assert execution._offload_memory.granted[0][0] == 4096.0
    assert release_offload_access(execution, _work()) == 0


def test_completed_batches_release_registry_and_ring_history_is_bounded():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    for round_index in range(32):
        work = _work()
        access = _access(bytes=1024, occurrence="ring",
                         direction="write", storage="index_pool_tail")
        producer = execution.engine.submit(f"ring:{round_index}", latency=1.0)
        events = schedule_kv_layer_writeback(
            execution, f"ring-{round_index}", 0, 0, 0, work, producer,
            state_accesses=(access,))
        assert len(events) == 2
        no_new = schedule_kv_iteration_offload(
            execution, f"ring-fb-{round_index}", 0, work, producer, (),
            {"ring": producer}, scheduled_layers={(0, 0)},
            state_accesses=(access,))
        assert no_new == ()
        assert release_offload_access(execution, work) == 1
        execution.engine.advance(until=events, first=False)
    assert not execution._offload_access_batches
    assert not getattr(execution, "_offload_actual_accesses", {})


def test_independent_schedulers_and_memories_on_the_same_case_are_isolated():
    case_spec = _with_offload(parse_case(case(output=1)))
    memory_a = Memory(case_spec)
    memory_b = Memory(case_spec)
    execution_a = _execution(case_spec)
    execution_b = _execution(case_spec)
    scheduler_a = Scheduler(case_spec, list(case_spec.workload.requests), memory_a)
    scheduler_b = Scheduler(case_spec, list(case_spec.workload.requests), memory_b)
    scheduler_a.bind_execution(execution_a)
    scheduler_b.bind_execution(execution_b)
    assert scheduler_b.grant_state([("only-b", 4096.0)]) is True
    assert memory_a.granted[0][0] == 0.0
    assert memory_b.granted[0][0] == 4096.0
    event = execution_a.engine.submit("a", latency=1.0)
    access = _access(bytes=32, occurrence="a")
    schedule_kv_layer_writeback(
        execution_a, "a", 0, 0, 0, _work(), event, state_accesses=(access,))
    assert execution_a._offload_memory is memory_a
    assert execution_b._offload_memory is memory_b
    assert memory_a.granted[0][0] == 0.0


def test_standalone_contexts_do_not_retain_case_or_memory():
    refs = []
    for _ in range(32):
        parsed = parse_case(case(output=1))
        memory = Memory(parsed)
        scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
        refs.append((weakref.ref(parsed), weakref.ref(memory)))
    del scheduler, parsed, memory
    gc.collect()
    assert all(case_ref() is None and memory_ref() is None
               for case_ref, memory_ref in refs)


def test_handoff_json_nested_span_normalizes_to_a_stable_identity():
    policy = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25)
    payload = {
        "occurrence": "dcp:rank0", "producer": "dcp:rank0",
        "request": "r", "stage": 0, "layer": 0, "direction": "write",
        "rank": 0, "span": [[0, 1]], "storage": "gqa_kv", "scope": "target",
        "bytes": 32,
    }
    projected, = project_state_tiers((payload,), policy)
    assert projected["span"] == ((0, 1),)
    assert (projected["ddr_bytes"], projected["ssd_bytes"],
            projected["hbm_bytes"]) == (16.0, 8.0, 24.0)
    # The same normalized value is usable as a set/dedupe key.
    assert len({projected["span"], ((0, 1),)}) == 1


def test_dcp_per_rank_rows_project_to_actual_tier_bytes():
    policy = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25)
    rank0 = _access(bytes=32, rank=0, occurrence="dcp:rank0")
    rank1 = _access(bytes=32, rank=1, occurrence="dcp:rank1")
    projected = project_state_tiers((rank0, rank1), policy)
    assert [(row["rank"], row["ddr_bytes"], row["ssd_bytes"], row["hbm_bytes"])
            for row in projected] == [
        (0, 16.0, 8.0, 24.0), (1, 16.0, 8.0, 24.0)]
    assert rank0.key != rank1.key

    case_spec = _with_offload(parse_case(case(output=1)))
    for access in (rank0, rank1):
        execution = _execution(case_spec)
        producer = execution.engine.submit("producer", latency=1.0)
        events = schedule_kv_layer_writeback(
            execution, "dcp", 0, 0, 0, _work(), producer,
            state_accesses=(access,))
        serviced = _serviced(execution, events)
        assert serviced["offload_ddr@r0:s0"] == pytest.approx(16.0)
        assert serviced["offload_ssd@r0:s0"] == pytest.approx(8.0)
        assert serviced["memory@r0:s0"] == pytest.approx(24.0)


def _gdn_pad_first_op():
    contract = {
        "phase": "decode", "direction": "token", "target_kind": "recurrent",
        "read_versions": (None, 16), "checkpoints": (0, 1),
        "publish_versions": (None, 17), "targets": ("", "r"),
        "state_layout": "V,K",
    }
    return SimpleNamespace(
        op_id="gdn:recurrent", family="recurrent_vector", stage=0, layer=0,
        requests=("pad", "r"), mte3_bytes=0.0, hbm_bytes=0.0,
        metadata=(("heads", 8), ("dk", 128), ("dv", 128),
                  ("state_element_bytes", 4.0),
                  ("gdn_state_contract", contract)))


def test_issued_access_registry_stops_fallback_duplicate_and_splits_writers():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    a = _access(bytes=32, occurrence="writer:A")
    b = _access(bytes=64, occurrence="writer:B")
    event_a = execution.engine.submit("writer:A", latency=1.0)
    event_b = execution.engine.submit("writer:B", latency=2.0)
    work = _work()
    first = schedule_kv_layer_writeback(
        execution, "one", 0, 0, 0, work, event_a,
        state_accesses=(a,))
    assert _serviced(execution, first)["offload_ddr@r0:s0"] == pytest.approx(16.0)
    # The old scheduled_layers hint must not suppress B, but must not reissue A.
    no_new = schedule_kv_iteration_offload(
        execution, "fallback", 0, work, event_a, (),
        {"writer:A": event_a}, scheduled_layers={(0, 0)},
        state_accesses=(a, b))
    assert no_new == ()
    later = schedule_kv_iteration_offload(
        execution, "fallback2", 0, work, event_a, (),
        {"writer:A": event_a, "writer:B": event_b}, scheduled_layers={(0, 0)},
        state_accesses=(a, b))
    serviced = _serviced(execution, later)
    assert serviced["offload_ddr@r0:s0"] == pytest.approx(16.0 + 32.0)
    assert serviced["offload_ssd@r0:s0"] == pytest.approx(8.0 + 16.0)
    # A already issued, so exactly B's 64 B is charged once.
    assert len(later) == 2


def test_prefetch_strategies_union_source_and_grant_hard_edges():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    access = _access(bytes=32, occurrence="fetch", direction="read")
    input_ready = execution.engine.submit("input", latency=0.0)
    source_ready = execution.engine.submit("source", release=3.0)
    grant_ready = execution.engine.submit("grant", release=5.0)
    early = execution.engine.submit("early", latency=0.0)
    unrelated_late = execution.engine.submit("unrelated-late", release=10.0)
    for version in ("v1", "v2", "v3", "v4"):
        execution.offload_policy = replace(
            execution.offload_policy, kvc_prefetch_version=version)
        events = schedule_kv_layer_offload(
            execution, f"edge-{version}", 0, 0, 0, _work(), input_ready,
            (input_ready, unrelated_late), {0: early},
            state_accesses=(access,), source_ready=source_ready,
            grant_ready=grant_ready)
        assert events
        execution.engine.advance(until=(events[-1],), first=False)
        assert min(event.time for event in events) >= 5.0
        assert max(event.time for event in events) < 10.0
        reader = execution.engine.submit(
            f"reader-{version}", latency=0.0, after=events)
        execution.engine.advance(until=(reader,), first=False)
        assert reader.time >= max(event.time for event in events)


def test_draft_read_fetch_is_anchored_before_its_reader():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    execution.offload_policy = replace(
        execution.offload_policy, kvc_prefetch_version="v1")
    read = _access(bytes=64, occurrence="draft:read", scope="draft",
                   direction="read", storage="draft_kv", layer=-1)
    source = execution.engine.submit("source", latency=1.0)
    fetch = schedule_kv_layer_offload(
        execution, "draft", 0, 0, -1, _work(), source, (source,), {},
        state_accesses=(read,))
    assert fetch
    reader = execution.engine.submit("reader", latency=0.0, after=fetch)
    execution.engine.advance(until=(reader,), first=False)
    assert fetch[0].time >= source.time
    assert reader.time >= fetch[0].time
    assert execution.engine.serviced["offload_ddr@r0:s0"] == pytest.approx(32.0)
    assert execution.engine.serviced["offload_ssd@r0:s0"] == pytest.approx(16.0)


def test_gdn_pad_first_keeps_the_real_slot_and_its_tier_bytes():
    policy = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25)
    op = _gdn_pad_first_op()
    pad = RequestView("pad", 1, 16, True, "decode", padding=True)
    real = RequestView("r", 1, 16, True, "decode")
    work = SimpleNamespace(plan=IterationPlan((pad, real)), ops=(op,))
    accesses = _accesses_from_ops(work, (op,))
    assert len(accesses) == 2
    read_access = next(item for item in accesses if item.direction == "read")
    write_access = next(item for item in accesses if item.direction == "write")
    assert (read_access.request, read_access.bytes) == ("r", 524288.0)
    assert (write_access.request, write_access.bytes) == ("r", 524288.0)
    for access in (read_access, write_access):
        projected, = project_state_tiers((access,), policy)
        assert (projected["ddr_bytes"], projected["ssd_bytes"],
                projected["hbm_bytes"]) == (262144.0, 131072.0, 393216.0)
    assert len(project_state_tiers(
        (read_access, write_access), policy, expected_bytes=1048576.0)) == 2
    with pytest.raises(ValueError, match="total_bytes_mismatch"):
        project_state_tiers((read_access, write_access), policy,
                            expected_bytes=1024.0)

    # Swap physical order: padding remains zero, r remains at its own slot.
    swapped = SimpleNamespace(**{**op.__dict__, "requests": ("r", "pad")})
    swapped_contract = dict(dict(op.metadata)["gdn_state_contract"])
    swapped_contract.update(read_versions=(16, None), checkpoints=(1, 0),
                            publish_versions=(17, None), targets=("r", ""))
    swapped = SimpleNamespace(**{**swapped.__dict__, "metadata": tuple({
        **dict(op.metadata), "gdn_state_contract": swapped_contract}.items())})
    swapped_work = SimpleNamespace(plan=IterationPlan((real, pad)), ops=(swapped,))
    swapped_accesses = _accesses_from_ops(swapped_work, (swapped,))
    assert len(swapped_accesses) == 2
    assert all(item.request == "r" and item.bytes == 524288.0
               for item in swapped_accesses)


def test_full_ring_index_tail_projects_the_actual_written_row():
    policy = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25)
    access = _access(bytes=1024, occurrence="index_pool_tail_store",
                     storage="index_pool_tail")
    projected, = project_state_tiers((access,), policy)
    assert (projected["ddr_bytes"], projected["ssd_bytes"],
            projected["hbm_bytes"]) == (512.0, 256.0, 768.0)
    work = SimpleNamespace(plan=IterationPlan((RequestView(
        "r", 1, 4, True, "decode"),)), ops=(_index_tail_op(),))
    discovered = _accesses_from_ops(work, work.ops)
    assert len(discovered) == 1
    assert discovered[0].bytes == 1024.0
    assert discovered[0].storage == "index_pool_tail"


def _index_tail_op():
    return SimpleNamespace(
        op_id="index_pool_tail_store", family="cache_store", stage=0, layer=0,
        requests=("r",), mte3_bytes=1024.0, hbm_bytes=2048.0,
        metadata=(("cache_kind", "index_pool_tail"), ("retained_rows", 1),
                  ("index_width", 128), ("cache_visible", True)))


def test_empty_fragment_is_zero_and_positive_opaque_access_is_not_invented():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    input_ready = execution.engine.submit("input", latency=1.0)
    empty = _work(ops=(_index_tail_op(),))
    events = schedule_kv_iteration_offload(
        execution, "empty", 0, empty, input_ready, (), {})
    assert events == ()
    assert execution.engine.serviced.get("offload_ddr@r0:s0", 0.0) == 0.0

    positive = _access(bytes=32, occurrence="opaque:state_store")
    other = _execution(case_spec)
    input_ready = other.engine.submit("input", latency=1.0)
    work = _work(ops=(SimpleNamespace(
        op_id="opaque:state_store", family="opaque", stage=0, layer=0,
        requests=("r",), mte3_bytes=32.0, hbm_bytes=64.0,
        metadata=(("state_write_bytes", 32.0),)),))
    producer = other.engine.submit("opaque:state_store", latency=1.0)
    events = schedule_kv_iteration_offload(
        other, "positive", 0, work, input_ready, work.ops,
        {"opaque:state_store": producer})
    serviced = _serviced(other, events)
    assert serviced["offload_ddr@r0:s0"] == pytest.approx(16.0)
    # The discovery helper is bound to the op, not to a model layer list.
    discovered = _accesses_from_ops(work, work.ops)
    assert len(discovered) == 1 and discovered[0].bytes == 32.0
    assert discovered[0].occurrence == "opaque:state_store"
    assert positive.bytes == 32.0


def _writer_op(name, size):
    return SimpleNamespace(
        op_id=name, family="state", stage=0, layer=0, requests=("r",),
        mte3_bytes=0.0, hbm_bytes=0.0,
        metadata=(("state_write_bytes", float(size)), ("state_layout", "gqa_kv")))


def test_two_iterations_charge_the_same_request_and_ring_access_independently():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    access0 = _access(bytes=32, occurrence="call0", request="r")
    access1 = _access(bytes=32, occurrence="call0", request="r",
                      direction="write")
    # AccessKey intentionally does not contain producer; each work invocation
    # has its own registry, so the second writer must be charged again.
    work0, work1 = _work(), _work()
    event0 = execution.engine.submit("writer:0", latency=1.0)
    event1 = execution.engine.submit("writer:1", latency=2.0)
    first = schedule_kv_layer_writeback(
        execution, "batch0", 0, 0, 0, work0, event0,
        state_accesses=(access0,))
    no_new = schedule_kv_iteration_offload(
        execution, "batch0-fallback", 0, work0, event0, (),
        {"call0": event0}, scheduled_layers={(0, 0)},
        state_accesses=(access0,))
    assert no_new == ()
    assert release_offload_access(execution, work0) > 0
    second = schedule_kv_layer_writeback(
        execution, "batch1", 0, 0, 0, work1, event1,
        state_accesses=(access1,))
    assert _serviced(execution, first + second)["offload_ddr@r0:s0"] == pytest.approx(32.0)
    assert _serviced(execution, first + second)["offload_ssd@r0:s0"] == pytest.approx(16.0)

    ring0 = _access(bytes=1024, occurrence="ring", direction="write",
                    storage="index_pool_tail", layer=0)
    ring1 = _access(bytes=1024, occurrence="ring", direction="write",
                    storage="index_pool_tail", layer=0)
    ring_work0, ring_work1 = _work(), _work()
    before_ring = _serviced(execution, first + second)
    ring_events0 = schedule_kv_layer_writeback(
        execution, "ring0", 0, 0, 0, ring_work0, event0,
        state_accesses=(ring0,))
    ring_events1 = schedule_kv_layer_writeback(
        execution, "ring1", 0, 0, 0, ring_work1, event1,
        state_accesses=(ring1,))
    assert len(ring_events0) == 2 and len(ring_events1) == 2
    serviced = _serviced(execution, ring_events0 + ring_events1)
    assert serviced["offload_ddr@r0:s0"] - before_ring["offload_ddr@r0:s0"] == pytest.approx(1024.0)
    assert serviced["offload_ssd@r0:s0"] - before_ring["offload_ssd@r0:s0"] == pytest.approx(512.0)


def test_omitted_layer_access_waits_for_its_own_writer_event():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    work = _work(ops=(_writer_op("A", 32), _writer_op("B", 64)))
    event_a = execution.engine.submit("A", latency=1.0)
    event_b = execution.engine.submit("B", latency=2.0)
    first = schedule_kv_layer_writeback(
        execution, "old-layer", 0, 0, 0, work, event_a,
        access_events={"A": event_a})
    assert _serviced(execution, first)["offload_ddr@r0:s0"] == pytest.approx(16.0)
    no_new = schedule_kv_iteration_offload(
        execution, "old-fallback", 0, work, event_a, (),
        {"A": event_a}, scheduled_layers={(0, 0)})
    assert no_new == ()
    second = schedule_kv_layer_writeback(
        execution, "old-layer-B", 0, 0, 0, work, event_b,
        access_events={"B": event_b})
    assert _serviced(execution, second)["offload_ddr@r0:s0"] == pytest.approx(48.0)


def test_real_pp2_draft_store_is_not_the_stage0_id_transfer():
    from dataclasses import replace
    from mindie_sim.schema.workload import IterationPlan, RequestView

    raw = case(pp=2, output=2)
    raw["runtime"]["metadata"] = {
        "draft_mode": "mtp", "draft_proposal_tokens": 2,
        "draft_accept_tokens": 1,
    }
    base = parse_case(raw)
    from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
    auxiliary = draft_case().model.auxiliary
    draft = replace(auxiliary.mtp, layer_plans=native_draft_layer_plans(
        base.model, auxiliary.mtp))
    target = replace(base, model=replace(
        base.model, auxiliary=replace(auxiliary, mtp=draft)))
    view = RequestView("r", 1, 4, True, "decode",
                       draft_proposal_tokens=2, draft_accepted_tokens=1)
    work = Frontend()(IterationPlan((view,)), target.model, target.runtime,
                      target.machine)
    accesses = _accesses_from_ops(work)
    draft_accesses = [item for item in accesses if item.scope == "draft"]
    assert draft_accesses
    assert all(item.stage == 1 for item in draft_accesses)
    assert all(item.occurrence != "draft_proposal_transfer"
               for item in draft_accesses)
    assert any("cache" in item.occurrence for item in draft_accesses)

    case_spec = _with_offload(target)
    execution = _execution(case_spec)
    producer = execution.engine.submit("draft:last-store", latency=1.0)
    events = schedule_draft_kv_writeback(
        execution, "pp2-real", 0, 1, work, producer,
        state_accesses=tuple(draft_accesses),
        access_events={item.producer: producer for item in draft_accesses})
    assert events
    assert all("draft_kv_writeback" in event.name for event in events)
    serviced = _serviced(execution, events)
    assert serviced["offload_ddr@r0:s1"] > 0


def test_multiple_target_producers_require_separate_writeback_calls():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    producer = execution.engine.submit("producer", latency=1.0)
    a = _access(bytes=32, occurrence="target:a")
    b = _access(bytes=32, occurrence="target:b")
    with pytest.raises(Exception, match="one_producer_event_per_call"):
        schedule_kv_layer_writeback(
            execution, "two", 0, 0, 0, _work(), producer,
            state_accesses=(a, b))


def test_target_and_draft_have_separate_448_byte_write_sets():
    policy = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25)
    target = _access(bytes=256, occurrence="target:cache_store")
    draft = _access(bytes=192, occurrence="draft:cache_store", scope="draft")
    projected = project_state_tiers((target, draft), policy)
    assert sum(row["ddr_bytes"] for row in projected) == pytest.approx(224.0)
    assert sum(row["ssd_bytes"] for row in projected) == pytest.approx(112.0)
    assert sum(row["hbm_bytes"] for row in projected) == pytest.approx(336.0)

    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    target_producer = execution.engine.submit("target:store", release=1.0)
    draft_producer = execution.engine.submit("draft:store", release=2.0)
    id_transfer = execution.engine.submit("stage0:proposal-id", release=3.0)
    target_events = schedule_kv_layer_writeback(
        execution, "i0", 0, 0, 0, _work(), target_producer,
        state_accesses=(target,))
    draft_events = schedule_draft_kv_writeback(
        execution, "i0", 0, 0, _work(), draft_producer,
        state_accesses=(draft,))
    execution.engine.advance(until=(target_events + draft_events), first=False)
    assert all("draft_kv_writeback" not in event.name for event in target_events)
    assert all("draft_kv_writeback" in event.name for event in draft_events)
    serviced = _serviced(execution, target_events + draft_events)
    assert serviced["offload_ddr@r0:s0"] == pytest.approx(224.0)
    assert serviced["offload_ssd@r0:s0"] == pytest.approx(112.0)
    assert serviced["memory@r0:s0"] == pytest.approx(336.0)
    assert target_events[0].time >= target_producer.time
    assert draft_events[0].time >= draft_producer.time
    assert all("proposal-id" not in event.name
               for event in target_events + draft_events)
    assert target.key != draft.key

    # Proposal=0/disabled preparation: no draft access means no transfer.
    no_draft_execution = _execution(case_spec)
    producer = no_draft_execution.engine.submit("producer", latency=1.0)
    assert schedule_draft_kv_writeback(
        no_draft_execution, "i1", 0, 0, _work(), producer,
        state_accesses=()) == ()


def test_padding_has_no_state_access_and_physical_graph_work_remains():
    case_spec = _with_offload(parse_case(case(output=1)))
    active = _access(bytes=32, occurrence="active:cache_store", request="active")
    padding_only = _execution(case_spec)
    input_ready = padding_only.engine.submit("producer", latency=1.0)
    assert schedule_kv_layer_writeback(
        padding_only, "pad", 0, 0, 0, _work(), input_ready,
        state_accesses=()) == ()
    assert padding_only.engine.serviced.get("offload_ddr@r0:s0", 0.0) == 0.0

    active_execution = _execution(case_spec)
    input_ready = active_execution.engine.submit("producer", latency=1.0)
    active_events = schedule_kv_layer_writeback(
        active_execution, "active", 0, 0, 0, _work(), input_ready,
        state_accesses=(active,))
    mixed_execution = _execution(case_spec)
    input_ready = mixed_execution.engine.submit("producer", latency=1.0)
    mixed_events = schedule_kv_layer_writeback(
        mixed_execution, "mixed", 0, 0, 0, _work(), input_ready,
        state_accesses=(active,))
    assert _serviced(active_execution, active_events)["offload_ddr@r0:s0"] == pytest.approx(
        _serviced(mixed_execution, mixed_events)["offload_ddr@r0:s0"])

    physical = _execution(case_spec)
    padding = RequestView("padding", 4, 0, False, "decode", padding=True)
    step = physical.add(linear_work(IterationPlan((padding,), iteration=0),
                                    case_spec.model, case_spec.runtime,
                                    case_spec.machine), 0, 0.0,
                        emitted_tokens={})
    physical.engine.advance(until=(step.completion,), first=False)
    assert step.padded_tokens == 4
    assert step.operators == 1
    assert physical.engine.serviced.get("offload_ddr@r0:s0", 0.0) == 0.0
    assert physical.engine.serviced["matrix@r0:s0"] > 0


def _state_op(name, family="state", **metadata):
    return SimpleNamespace(
        op_id=name, family=family, stage=0, layer=0, requests=("r",),
        mte3_bytes=0.0, hbm_bytes=0.0, metadata=tuple(metadata.items()))


def test_fallback_uses_the_actual_producer_output_event_not_the_call_id():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    work = _work()
    call0 = _state_op("call0", "attention")
    store_b = _state_op("storeB", "state")
    access = StateAccess(
        occurrence="call0", producer="storeB", request="r", stage=0, layer=0,
        direction="write", output="out", storage="kv/key",
        span=((0, 1),), bytes=64.0)
    e1 = execution.engine.submit("call0", release=1.0)
    e_b = execution.engine.submit("storeB.out", release=5.0)

    # Only the outer call0 ID exists: the real storeB output is still absent,
    # so there is no emission and no issued registration.
    assert schedule_kv_iteration_offload(
        execution, "only-call0", 0, work, e1, (call0, store_b),
        {"call0": e1}, state_accesses=(access,)) == ()
    assert execution.engine.serviced.get("offload_ddr@r0:s0", 0.0) == 0.0
    assert release_offload_access(execution, work) == 0

    # The named storeB output Event is the real writer.  Its display name is
    # irrelevant; the bound access/producer identity selects it.
    events = schedule_kv_iteration_offload(
        execution, "bound", 0, work, e1, (call0, store_b),
        {"call0": e1}, state_accesses=(access,),
        access_events={("storeB", "out"): e_b})
    assert events
    execution.engine.advance(until=events, first=False)
    assert min(event.time for event in events) >= 5.0
    serviced = dict(execution.engine.serviced)
    assert serviced["offload_ddr@r0:s0"] == pytest.approx(32.0)
    assert serviced["offload_ssd@r0:s0"] == pytest.approx(16.0)
    assert serviced["memory@r0:s0"] == pytest.approx(48.0)
    assert schedule_kv_iteration_offload(
        execution, "same-batch-repeat", 0, work, e1, (call0, store_b),
        {"storeB": e_b}, state_accesses=(access,)) == ()
    assert release_offload_access(execution, work) == 1

    # A per-work binding registration is consumed identically by fallback.
    work2 = _work()
    access2 = replace(access)
    assert bind_offload_access_events(
        execution, work2, {access2.key: e_b}) == 1
    again = schedule_kv_iteration_offload(
        execution, "registered", 0, work2, e1, (call0, store_b),
        {"call0": e1}, state_accesses=(access2,))
    assert again
    execution.engine.advance(until=again, first=False)
    assert execution.engine.serviced["offload_ddr@r0:s0"] == pytest.approx(64.0)
    assert release_offload_access(execution, work2) == 1
    assert not getattr(execution, "_offload_access_events", {})


def test_legacy_metadata_fallback_and_explicit_calls_share_the_binding_result():
    case_spec = _with_offload(parse_case(case(output=1)))
    legacy_op = _state_op(
        "call0", "cache_store", cache_kind="out", state_producer="storeB")
    legacy_op = SimpleNamespace(**{**legacy_op.__dict__, "mte3_bytes": 64.0})
    work = _work(ops=(legacy_op,))
    store_b = _state_op("storeB", "state")
    e1 = execution = None

    def run(use_explicit):
        nonlocal e1, execution
        execution = _execution(case_spec)
        e1 = execution.engine.submit("call0", release=1.0)
        e_b = execution.engine.submit("displayed-as-anything", release=5.0)
        ids = {"call0": e1}
        common = dict(physical_ops=(legacy_op, store_b), ids=ids,
                      state_accesses=None)
        if use_explicit:
            explicit = StateAccess(
                occurrence="call0", producer="storeB", request="r", stage=0,
                layer=0, direction="write", output="out", storage="out",
                span=((0, 1),), bytes=64.0)
            common.update(state_accesses=(explicit,))
        assert schedule_kv_iteration_offload(
            execution, "legacy-before", 0, work, e1, *(), **common) == ()
        ids["storeB"] = e_b
        events = schedule_kv_iteration_offload(
            execution, "legacy-after", 0, work, e1, *(), **common)
        execution.engine.advance(until=events, first=False)
        return (len(events), execution.engine.serviced["offload_ddr@r0:s0"],
                execution.engine.serviced["offload_ssd@r0:s0"])

    legacy = run(False)
    explicit = run(True)
    assert legacy == explicit
    assert legacy[1:] == (pytest.approx(32.0), pytest.approx(16.0))


def test_same_layer_writes_use_each_physical_producer_event_once():
    case_spec = _with_offload(parse_case(case(output=1)))
    execution = _execution(case_spec)
    a = StateAccess(
        occurrence="layer:call0", producer="storeA", request="r", stage=0,
        layer=0, direction="write", output="key", storage="gqa_kv",
        span=((0, 1),), bytes=32.0)
    b = StateAccess(
        occurrence="layer:call0", producer="storeB", request="r", stage=0,
        layer=0, direction="write", output="key", storage="gqa_kv",
        span=((0, 1),), bytes=64.0)
    assert a.key != b.key
    e_a = execution.engine.submit("display-a", release=1.0)
    e_b = execution.engine.submit("display-b", release=5.0)
    work = _work()
    first = schedule_kv_layer_writeback(
        execution, "layer", 0, 0, 0, work, e_a,
        state_accesses=(a, b), access_events={("storeA", "key"): e_a})
    execution.engine.advance(until=first, first=False)
    assert min(event.time for event in first) >= 1.0
    assert max(event.time for event in first) < 5.0
    assert execution.engine.serviced["offload_ddr@r0:s0"] == pytest.approx(16.0)
    assert execution.engine.serviced["offload_ssd@r0:s0"] == pytest.approx(8.0)
    second = schedule_kv_layer_writeback(
        execution, "layer", 0, 0, 0, work, e_b,
        state_accesses=(a, b), access_events={("storeB", "key"): e_b})
    execution.engine.advance(until=second, first=False)
    assert max(event.time for event in second) >= 5.0
    assert len(second) == len(first) + 2
    assert execution.engine.serviced["offload_ddr@r0:s0"] == pytest.approx(48.0)
    assert execution.engine.serviced["offload_ssd@r0:s0"] == pytest.approx(24.0)
    assert schedule_kv_iteration_offload(
        execution, "same", 0, work, e_a, (), {},
        state_accesses=(a, b)) == ()


def test_gdn_contract_slots_are_parsed_before_aggregate_validation():
    op = _gdn_pad_first_op()
    with_aggregate = SimpleNamespace(**{
        **op.__dict__,
        "metadata": op.metadata + (("state_read_bytes", 524288.0),
                                   ("state_write_bytes", 524288.0))})
    pad = RequestView("pad", 1, 16, True, "decode", padding=True)
    real = RequestView("r", 1, 16, True, "decode")
    work = SimpleNamespace(plan=IterationPlan((pad, real)),
                           ops=(with_aggregate,))
    accesses = _accesses_from_ops(work, (with_aggregate,))
    assert len(accesses) == 2
    assert {(item.direction, item.bytes) for item in accesses} == {
        ("read", 524288.0), ("write", 524288.0)}
    assert all(item.request == "r" for item in accesses)

    bad = SimpleNamespace(**{
        **op.__dict__,
        "metadata": op.metadata + (("state_read_bytes", 1.0),)})
    with pytest.raises(ValueError, match="state_access_contract_total_mismatch"):
        _accesses_from_ops(
            SimpleNamespace(plan=IterationPlan((pad, real)), ops=(bad,)),
            (bad,))

    policy = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25)
    alias = replace(accesses[0], direction="alias", bytes=0.0)
    projected = project_state_tiers((alias,), policy)
    assert (projected[0]["ddr_bytes"], projected[0]["ssd_bytes"],
            projected[0]["hbm_bytes"]) == (0.0, 0.0, 0.0)
    execution = _execution(_with_offload(parse_case(case(output=1))))
    producer = execution.engine.submit("grant-only", latency=1.0)
    assert schedule_kv_layer_writeback(
        execution, "grant-only", 0, 0, 0, _work(), producer,
        state_accesses=(replace(accesses[0], bytes=0.0),)) == ()
    assert execution.engine.serviced.get("offload_ddr@r0:s0", 0.0) == 0.0
