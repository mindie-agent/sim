"""Captured launch, descriptor queues, and eager serving transfers.

[Role] Exercise capture diagnostics, STARS retirement, D2H barriers and serving I/O.
[Invariants] Conserve launches, bytes, credits and shared root-complex bandwidth.
[Timeouts & Locks] Bounded in-memory cases with virtual time.
[Side-Effect Policy] Read packaged hardware profiles and construct local runtime instances.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from _support.axes import ALL_SKUS
from _support.builders import dense_case, hardware
from _support.collective_cases import cohort_case
from mindie_sim import api as sdk_api
from mindie_sim.backends import ascend
from mindie_sim.prediction.collectives import RootComplexKey
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import (
    EagerSplitBarrier,
    Execution,
    SplitBarrierSpec,
    StarsDescriptorQueue,
    compute_uncovered_stars_ns,
    is_zero_kernel_view,
)
from mindie_sim.prediction.efficiency.exposed_program import _role_times
from mindie_sim.prediction.graph_analysis import _exposed_breakdown, graph_costs
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.device_input import StarsSpec
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

def _assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected

LEGACY_ARCH_SKU = "Ascend910_9362"
DUAL_DIE_LARGE_SKU = "Ascend910_9382"
NEXTGEN_ARCH_SKU = "Ascend950DT_9582"

TINY_MODEL_ID = "qwen3-32b"
MOE_MODEL_ID = "deepseek-v3"

def _make_machine(sku: str, *, num_devices: int = 2, ranks_per_node: int | None = None) -> Hardware:
    raw: dict = {"sku": sku, "num_devices": num_devices}
    if ranks_per_node is not None:
        raw["ranks_per_node"] = ranks_per_node
    return Hardware.from_mapping(ascend(raw))

def _tiny_case(*, stars_spec: StarsSpec | None = None, tp: int = 2):
    case = sdk_api.compose_case(
        TINY_MODEL_ID, hardware=LEGACY_ARCH_SKU, options={"dp": 1, "tp": tp, "pp": 1, "layers": 1},
    )
    if stars_spec is not None:
        new_dies = tuple(
            replace(die, dispatch_sync=replace(die.dispatch_sync, stars=stars_spec))
            for die in case.machine.dies
        )
        case = replace(case, machine=replace(case.machine, dies=new_dies))
    return case

@pytest.mark.parametrize("sku", ALL_SKUS)
def test_profile_datapath_capabilities(sku):
    dp = datapath_for_die(_make_machine(sku, num_devices=2).dies[0])
    assert len(ALL_SKUS) == 16 and dp.subtile_wave and dp.stars_spec is not None
    assert dp.stars_spec.queue_depth > 0 and dp.stars_spec.lanes >= 1
    assert dp.memory_access is not None and dp.memory_access.sector_bytes == 32
    assert dp.hbm_read_bytes_per_s > 0.0 and dp.hbm_write_bytes_per_s > 0.0
    assert 0.0 <= dp.hbm_rw_alpha <= 1.0

def test_datapath_route_capabilities_and_validation():
    dp_legacy = datapath_for_die(_make_machine(LEGACY_ARCH_SKU).dies[0])
    dp_next = datapath_for_die(_make_machine(NEXTGEN_ARCH_SKU).dies[0])
    assert dp_legacy.simt_ub_carveout_bytes == 0
    assert dp_next.simt_ub_carveout_bytes == 32768
    assert dp_next.usable_bytes("ub") - dp_next.usable_bytes("ub", kernel="simt") == 32768

    route_ok = dp_next.route("l0c", "ub", representation="fp32", owner_group=(0, 1), tile=(16, 16))
    assert route_ok.path.unit == "fixpipe"
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported:l0c->ub"):
        dp_legacy.route("l0c", "ub")
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_representation_unknown:no_such_representation"):
        dp_next.route("l0c", "ub", representation="no_such_representation")
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_owner_unreachable"):
        dp_next.route("l0c", "ub", owner_group=(0, 47))
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_tile"):
        dp_next.route("l0c", "ub", representation="fp32", tile=(1, 1))
    # The reverse route uses named owners and validates both buffers.
    assert dp_next.route("ub", "l1", owner_group={"aic": 1, "aiv": 2}).path.source == "ub"
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_owner_unreachable"):
        dp_next.route("ub", "l1", owner_group={"aic": 2, "aiv": 1})
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_tile_exceeds_buffer:ub:262144"):
        dp_next.route("ub", "l1", representation="fp16", tile=(256, 512))

def _stars_spec(depth, issue_us, *, first_us=0, scope="die", policy="descriptor_ready"):
    return StarsSpec(queue_scope=scope, queue_depth=depth, lanes=1, issue_interval_us=issue_us,
                     first_issue_us=first_us, release_policy=policy, evidence_id="test:stars")

def _queue(depth, issue_us, **kwargs):
    return StarsDescriptorQueue(_stars_spec(depth, issue_us, **kwargs), die_id=0)

def _submit(queue, engine, ready, occurrence, stream, resource, work, *, replay=0, op=None):
    return queue.submit_kernel(engine, replay=replay, stream=stream, kernel_occurrence=occurrence,
                               descriptor_ready=ready, work={resource: work}, op=op)

def test_stars_covered_behind_compute_and_bounded_replay_retirement():
    queue = _queue(8, 5)
    engine = Engine([Resource("aic", 1.0, "ms", "fifo")])
    ready = engine.submit("r")
    dones = [_submit(queue, engine, ready, i, "main", "aic", 0.050)[0] for i in range(6)]
    engine.advance()
    _assert_fields(queue.summary(), descriptor_count=6, uncovered_stars_ns=5_000, active_records=0)
    assert dones[-1].time == pytest.approx(0.005 + 6 * 0.050, rel=1e-9)

    # Completed depth-one replays retain only aggregate clock evidence.
    queue = _queue(1, 5, first_us=2)
    engine = Engine([Resource("aic", 1.0, "ms", "fifo")])
    ready = engine.submit("r0")
    for index in range(400):
        ready, _ = _submit(queue, engine, ready, index, "main", "aic", 0.010, replay=index)
    engine.advance()
    _assert_fields(queue.summary(), descriptor_count=400, descriptor_union_ns=400 * 7_000)
    assert len(queue._retired_desc_ns) == len(queue._retired_comp_ns) == 0
    assert len(queue._replay_first_charged) <= 1

@pytest.mark.parametrize(
    "depth,issue,second_stream,first_work,second_work,expected",
    [(1, 0, "s1", 1.0, 1.0, (1.0, 2.0)), (4, 1, "s0", 0.100, 0.001, (0.101, 0.102))],
    ids=["zero-issue-credit-depth", "same-stream-order"],
)
def test_stars_credit_and_stream_order(depth, issue, second_stream, first_work, second_work, expected):
    queue = _queue(depth, issue)
    engine = Engine([Resource("u0", 1.0, "ms"), Resource("u1", 1.0, "ms")])
    ready = engine.submit("ready")
    first, left = _submit(queue, engine, ready, 0, "s0", "u0", first_work)
    second, right = _submit(queue, engine, ready, 1, second_stream, "u1", second_work)
    engine.advance()
    assert (first.time, second.time) == pytest.approx(expected, rel=1e-9)
    assert left is not None and right is not None
    assert left.issue_done_event is None and left.kernel_done_event is None
    assert queue.summary()["active_records"] == 0

def test_stars_credit_is_held_until_the_whole_kernel_group_retires():
    queue = _queue(1, 0)
    engine = Engine([Resource("u0", 1.0, "ms"), Resource("u1", 1.0, "ms")])
    ready = engine.submit("rg")
    record, grant = queue.issue_descriptor(
        engine, replay=0, stream="s0", kernel_occurrence=0, descriptor_ready=ready)
    entry = engine.submit("g0:entry", {"u0": 0.010}, after=(record.issue_done_event,))
    tail = engine.submit("g0:tail", {"u0": 0.090}, after=(entry,))
    queue.bind_kernel_execution(engine, record, grant,
                                kernel_start=engine.start_boundary(entry, "g0:start"), kernel_done=tail)
    done, _ = _submit(queue, engine, ready, 1, "s1", "u1", 0.020)
    engine.advance()
    assert done.time == pytest.approx(0.010 + 0.090 + 0.020, rel=1e-9)

@pytest.mark.parametrize("policy,depth,index,start", [
    ("descriptor_ready", 4, 2, 0.020), ("stream_predecessor_done", 4, 1, 0.060),
    ("descriptor_ready", 2, 2, 0.060),
])
def test_stars_release_policies_and_backpressure(policy, depth, index, start):
    queue = _queue(depth, 10, policy=policy)
    engine = Engine([Resource("aic", 1.0, "ms", "fifo")])
    ready = engine.submit("ready")
    records = [_submit(queue, engine, ready, i, "s0", "aic", 0.050)[1] for i in range(3)]
    engine.advance()
    assert all(record is not None and record.issue_interval_ms is not None for record in records)
    assert records[index].issue_interval_ms[0] == pytest.approx(start, abs=1e-9)

def test_stars_multistream_union_and_zero_kernel_views():
    queue = _queue(8, 5)
    engine = Engine([Resource(f"u{i}", 1.0, "ms") for i in range(3)])
    ready = engine.submit("ready")
    reshape = SimpleNamespace(family="reshape", metadata=())
    marked = SimpleNamespace(family="matmul", metadata=(("zero_kernel_view", True),))
    assert is_zero_kernel_view(reshape) is True and is_zero_kernel_view(marked) is True
    assert _submit(queue, engine, ready, 0, "s0", "u0", 0.0, op=reshape)[1] is None
    for index in range(3):
        _submit(queue, engine, ready, index + 1, f"s{index}", f"u{index}", 0.050)
    engine.advance()
    _assert_fields(queue.summary(), descriptor_count=3, uncovered_stars_ns=5_000)
    assert compute_uncovered_stars_ns([(0.0, 0.015)], [(0.005, 0.055), (0.010, 0.065)]) == 5_000

def _piecewise_work():
    shape = (8, 16, 64)
    view = OpSpec("op_view", "rotary", "decode", 0, 0, ("req0",), shape, shape, shape, shape,
                  recipe="rotary_position_embedding", dtype="bf16", stream="main",
                  metadata=(("zero_kernel_view", True),))
    segments = tuple(replace(view, op_id=f"op_seg{index}", deps=("op_seg0",) if index else (),
                             metadata=(("graph_captured", True), ("graph_segment", index)))
                     for index in range(2))
    plan = IterationPlan(iteration=0, requests=(RequestView("req0", 1, 16, True, "decode"),))
    return IterationWork(ops=(view, *segments), plan=plan, stage_count=1, mode="decode",
                         valid_tokens=1, padded_tokens=1, graph_bucket=1, recipe="default",
                         replay_scope="piecewise")

def test_piecewise_segments_charge_initial_descriptor_delay_on_every_die():
    case = _tiny_case(stars_spec=_stars_spec(32, 5, first_us=25), tp=2)
    execution = Execution(case, CostModel(case), include_submission=False)
    execution.add(_piecewise_work(), replica=0, release_ms=0.0)
    execution.engine.advance()
    summary = execution.summary()
    assert summary["submission"]["included"] is False
    assert summary["submission"]["host_service_ms"] == 0
    stage = summary["stars"]["r0:s0"]
    _assert_fields(stage, descriptor_count=4, die_count=2, total_descriptor_ns=2 * 2 * 30_000)
    for die in ("d0", "d1"):
        assert stage["dies"][die]["descriptor_union_ns"] == 2 * 30_000

@pytest.mark.parametrize("scope,expected", [("die", (2, 2)), ("package", (4,))])
def test_stars_queue_scope_on_two_replicas(scope, expected):
    spec = _stars_spec(32, 8, scope=scope)
    base = sdk_api.compose_case(TINY_MODEL_ID, hardware=LEGACY_ARCH_SKU,
                                options={"dp": 2, "tp": 1, "pp": 1, "ranks_per_node": 2, "layers": 1})
    dies = tuple(replace(die, dispatch_sync=replace(die.dispatch_sync, stars=spec)) for die in base.machine.dies)
    case = replace(base, machine=replace(base.machine, dies=dies))
    execution = Execution(case, CostModel(case), include_submission=False)
    for replica in range(2):
        execution.add(_piecewise_work(), replica=replica, release_ms=0.0)
    execution.engine.advance()
    stages = execution.summary()["stars"]
    for replica, count in enumerate(expected):
        _assert_fields(stages[f"r{replica}:s0"], descriptor_count=count, descriptor_union_ns=count * 8_000)

@pytest.mark.parametrize("local_experts", [1, 4])
def test_eager_split_barrier_orders_collective_and_fences_single_expert(local_experts):
    machine = _make_machine(DUAL_DIE_LARGE_SKU, num_devices=4, ranks_per_node=4)
    die = machine.dies[0]
    single = local_experts == 1
    spec = SplitBarrierSpec(
        ep_size=16, num_local_experts=local_experts, split_dtype_bytes=8, pinned=True, non_blocking=True,
        host_consume_us=5.0, histogram_ms=0.020, allgather_ms=0.040,
        **({"sync_overhead_us": 10.0} if single else {"repeat_interleave_ms": 0.030}),
    )
    engine = Engine([Resource("other_stream", 1.0, "ms")])
    ready = engine.submit("ready")
    independent = engine.submit("indep", {"other_stream": 0.500}, after=(ready,))
    result = EagerSplitBarrier.submit(engine, machine, die, spec, ready,
                                      independent_device_events=(independent,))
    engine.advance()
    if single:
        assert result.explicit_device_sync is not None
        assert result.explicit_device_sync.time == pytest.approx(0.500 + 0.010, rel=1e-9)
        assert result.dispatch_ready.time == pytest.approx(0.500 + 0.010 + 0.005, rel=1e-9)
    else:
        assert result.explicit_device_sync is None
        assert RootComplexKey(0, die.root_complex_id, "d2h").resource_name() in result.d2h_resources
        assert result.allgather_counts.time == pytest.approx(0.020 + 0.040, rel=1e-9)
        assert result.device_branch_done.time == pytest.approx(0.020 + 0.040 + 0.030, rel=1e-9)
        assert 0.090 <= result.dispatch_ready.time < 0.200

def test_eager_split_barrier_pinned_and_pageable_order_independence():
    machine = _make_machine(DUAL_DIE_LARGE_SKU, num_devices=4, ranks_per_node=4)
    die0 = machine.dies[0]
    # 3) Pinned vs pageable on the SAME Engine is 100% order-independent
    big_pin = SplitBarrierSpec(ep_size=1_000_000, num_local_experts=2, split_dtype_bytes=8, pinned=True)
    big_pag = replace(big_pin, pinned=False)

    def _run_pair(first_pinned: bool) -> tuple[float, float]:
        eng = Engine([])
        t0 = eng.submit("t0")
        first = EagerSplitBarrier.submit(eng, machine, die0, big_pin if first_pinned else big_pag, t0, name="first")
        t1 = eng.submit("t1", after=(first.dispatch_ready,))
        second = EagerSplitBarrier.submit(eng, machine, die0, big_pag if first_pinned else big_pin, t1, name="second")
        eng.advance()
        d_first = first.input_splits_d2h.time
        d_second = second.input_splits_d2h.time - t1.time
        return (d_first, d_second) if first_pinned else (d_second, d_first)

    pin_a, pag_a = _run_pair(True)
    pin_b, pag_b = _run_pair(False)
    assert math.isclose(pin_a, pin_b, rel_tol=1e-9)
    assert math.isclose(pag_a, pag_b, rel_tol=1e-9)
    assert pag_a > pin_a * 2.0


def test_eager_moe_dispatch_wires_split_barriers_and_counts_collective():
    def _run_eager_moe_dispatch(num_experts: int) -> dict:
        case = sdk_api.compose_case(
            MOE_MODEL_ID, hardware=LEGACY_ARCH_SKU,
            options={"dp": 1, "tp": 4, "ep": 4, "pp": 1, "layers": 4},
        )
        ex = Execution(case, CostModel(case), include_submission=True)
        shape = (4, 1, 512)
        op = OpSpec(
            op_id="l3.moe_dispatch", family="alltoall", recipe="expert_dispatch",
            phase="prefill", stage=0, layer=3, stream="main", dtype="bf16",
            link_bytes=4096, link_scope="ep", requests=("req0",),
            metadata=(("moe_ep_size", 4), ("moe_num_experts", num_experts)),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        )
        plan = IterationPlan(
            iteration=0,
            requests=(RequestView("req0", q_len=4, context=16, samples=True, phase="prefill"),),
        )
        work = IterationWork(
            plan=plan, ops=(op,), stage_count=1, mode="prefill", valid_tokens=4,
            padded_tokens=4, graph_bucket=None, recipe="default", replay_scope="none",
        )
        ex.add(work, replica=0, release_ms=0.0)
        ex.engine.advance()
        barriers = ex.summary()["eager_split_barriers"]
        assert len(barriers) == 1
        return barriers[0]

    b_single = _run_eager_moe_dispatch(num_experts=4)
    b_multi = _run_eager_moe_dispatch(num_experts=16)
    assert b_single["input_splits_bytes"] == 32 and b_single["output_splits_bytes"] == 32
    assert b_single["total_d2h_bytes"] == 256 and b_single["participating_ranks"] == [0, 1, 2, 3]
    assert b_single["explicit_device_sync"] is True and b_multi["explicit_device_sync"] is False
    # Real 4-layer raw_eager prefill at TP=EP=4 submits per-rank splits (256 B) + counts AllGather
    from mindie_sim import compose_case, parse_case
    case = parse_case(compose_case(MOE_MODEL_ID, LEGACY_ARCH_SKU, tp=4, ep=4, layers=4, raw_eager=True))
    gc_eager = graph_costs(
        case, target_graphs(case, phase="prefill", tokens=16, scope="body"),
        external_inputs={"analysis_input": 0}, include_submission=True, isolated=False,
    )
    ex_eager = gc_eager["execution"]
    eb = ex_eager["eager_split_barriers"][0]
    assert eb["total_d2h_bytes"] == 256 and eb["participating_ranks"] == [0, 1, 2, 3]
    assert eb["counts_allgather_mode"] == "lowered_collective"
    assert math.isclose(ex_eager["serviced"]["host_rc:n0:rc0:d2h"], 256.0, rel_tol=1e-9)

def test_serving_io_couples_concurrent_replicas_to_shared_root_complex():
    """Solo H2D caps at 59.5 GB/s per die while 8 concurrent replicas cap at 238.8 GB/s RC."""
    payload_bytes = 64_000_000
    base = sdk_api.compose_case(
        TINY_MODEL_ID, hardware=DUAL_DIE_LARGE_SKU,
        options={"dp": 8, "tp": 1, "pp": 1, "ranks_per_node": 8, "layers": 1},
    )
    io_cfg = {"h2d": {"fixed_bytes": payload_bytes, "bytes_per_s": 500.0e9, "latency_us": 0.0}}
    case = replace(base, runtime=replace(base.runtime, serving_io=io_cfg))
    solo_gbps = float(case.machine.host_solo_bw_gbps("h2d") or 0.0)
    rc_gbps = float(case.machine.host_root_complex_ceiling_gbps("h2d") or 0.0)
    assert math.isclose(solo_gbps, 59.5, rel_tol=1e-6) and math.isclose(rc_gbps, 238.8, rel_tol=1e-6)
    for replicas, bandwidth in ((1, solo_gbps), (8, rc_gbps)):
        ex = Execution(case, CostModel(case), include_submission=False, include_serving_io=True)
        ready = ex.engine.submit("ready")
        dones = [ex.serving_io.transfer(ex, "h2d", f"r{rep}", rep, ready, tokens=1, requests=1)
                 for rep in range(replicas)]
        ex.engine.advance()
        assert max(ev.time for ev in dones) == pytest.approx(
            replicas * payload_bytes / (bandwidth * 1_000_000), rel=1e-4)

@pytest.fixture
def aclgraph(profiler_api):
    from mindie_profiler import aclgraph_multistream

    return aclgraph_multistream

def _region(api, *, start=0, second_stream=1, region=None, branch=None):
    settings = {"op": "dispatch", "stream_id": second_stream, "start_us": start, "end_us": start + 100}
    return api.ParallelRegion(
        "region", api.StreamInterval("AllReduce", 0, 0, 100),
        api.StreamInterval(**{**settings, **(branch or {})}), **(region or {}),
    )

@pytest.mark.parametrize("region,start,stream,code,cause", [
    ({}, 0, 0, "A1", "missing_npu_stream_switch_or_default_stream_fallback"),
    ({"graph_mode": "full", "event_causality_observed": True}, 96, 1, "A2", "premature_event_record_wait"),
    ({"graph_mode": "piecewise", "piece_boundary_between": True}, 96, 1, "A2",
     "piecewise_aclgraph_split_barrier"),
    ({}, 0, 1, "OK", "overlap_holds"),
])
def test_capture_diagnoses_collapse_and_boundaries(aclgraph, region, start, stream, code, cause):
    diag = aclgraph.diagnose_region(_region(aclgraph, start=start, second_stream=stream, region=region))
    assert (diag.code, diag.cause) == (code, cause)
    if code == "A1":
        assert diag.overlap_pct == 0 and "stream_switch" in " ".join(diag.recommendations)
    if cause == "premature_event_record_wait":
        assert "Notify_Wait" in diag.evidence

@pytest.mark.parametrize("branch", [
    {"waits_on_capture_root": False}, {"joined_at_exit": False},
    {"aliases_buffer": True, "events_ordered": False},
])
def test_capture_diagnoses_dangling_and_unordered_aliases(aclgraph, branch):
    assert aclgraph.diagnose_region(_region(aclgraph, branch=branch)).code == "A5"

@pytest.mark.parametrize("mode,steals_vector_cores", [("AIV", True), ("CCU_SCHED", False)])
def test_capture_diagnoses_collective_core_oversubscription(aclgraph, mode, steals_vector_cores):
    def interval(name, stream, blocks):
        return aclgraph.StreamInterval(name, stream, 0, 100, aiv_blocks=blocks, io_bytes=4096,
                                       hccl_expansion_mode=mode)
    region = aclgraph.ParallelRegion("region", interval("AllReduce", 0, 40), interval("dispatch", 1, 30),
                                    aiv_cores=64)
    diag = aclgraph.diagnose_region(region)
    assert diag.code == "A3"
    assert ("hccl_aiv" in diag.cause) is steals_vector_cores
    assert diag.net_speedup_us > 0
    if not steals_vector_cores:
        # Two 100 us branches overlap, with one 1.5 us event pair.
        assert diag.net_speedup_us == pytest.approx(2 * 100 - (100 + 1.5))
        assert diag.details["event_us"] == 1.5

def test_capture_event_overhead_erodes_parallel_gain(aclgraph):
    region = _region(aclgraph, region={"event_pairs": 60}, branch={"io_bytes": 400_000_000})
    region = replace(region, branch_a=replace(region.branch_a, io_bytes=400_000_000))
    observed = aclgraph.diagnose_region(region)
    assert observed.details["event_us"] == 60 * 1.5
    causal = aclgraph.diagnose_region({
        "name": "producer-consumer-event",
        "nodes": [
            {"id": "consumer", "stream_id": 0, "start_us": 0, "duration_us": 4},
            {"id": "producer", "stream_id": 1, "start_us": 2, "duration_us": 2},
            {"id": "tail", "stream_id": 1, "start_us": 4, "duration_us": 2},
        ],
        "events": [{"id": "ready", "record": "producer", "wait": "consumer"}],
    })
    assert causal.code == "A4" and "event" in causal.cause
    assert causal.net_speedup_us == pytest.approx((4 + 2 + 2) - (4 + 1.5 + 4))
    assert causal.details["potential_gain_us"] == pytest.approx((4 + 2 + 2) - 6)

def _envelope_size(payload):
    """Strict UTF-8/BPE upper bound for one serialized MCP payload."""
    default = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    compact = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    bound = max(len(default.encode("utf-8")), len(compact.encode("utf-8")))
    return {"utf8_bytes": bound, "token_estimate": bound}

@pytest.mark.requires_profiler
def test_wrapper_updatable_and_breakable_models() -> None:
    from mindie_profiler.aclgraph_multistream import (
        ACLGraphWrapperModel, BreakableACLGraphWrapperModel, UpdatableGraphModel,
    )

    wrapper = ACLGraphWrapperModel(
        captured_batch_sizes=8, subgraphs_per_step=32, active_streams_per_graph=16
    )
    assert wrapper.budget.error_code == "207008" and wrapper.replay_sync_required
    assert wrapper.superkernel.eligible
    assert wrapper.launch_floor_us == pytest.approx(1.35)
    assert BreakableACLGraphWrapperModel(pieces=4).join_overhead_us == pytest.approx(4.5)
    updatable = UpdatableGraphModel(enable_enpu=True)
    assert updatable.replay_order() == ("update", "replay")
    assert not updatable.sync_before_replay

@pytest.mark.requires_profiler
def test_capture_queue_budget_and_exhaustion() -> None:
    from mindie_profiler.aclgraph_multistream import StreamResourceExhausted, check_capture_budget

    fits = check_capture_budget(8, 32, 4)
    assert fits.required == 1024 and fits.fits and fits.error_code == ""
    exhausted = check_capture_budget(8, 32, 16)
    assert not exhausted.fits and exhausted.error_code == "207008"
    assert exhausted.recommendation == "prefer_full_or_full_decode_only"
    with pytest.raises(StreamResourceExhausted, match="207008") as exc:
        exhausted.raise_if_exhausted()
    assert exc.value.error_code == "207008"
    assert check_capture_budget(2, 2, 2, capacity=2).error_code == "207008"

@pytest.mark.requires_profiler
def test_capture_error_classifier() -> None:
    from mindie_profiler.aclgraph_multistream import classify_capture_error

    assert classify_capture_error("error code is 207008 Stream resources are insufficient") == "stream_resource_207008"
    assert classify_capture_error("Alloc sq cq fail") == "old_hdk_alloc_sq_cq"
    assert classify_capture_error("other") == "unknown"

def test_superkernel_eligibility_and_dcci(aclgraph):
    check_superkernel_eligibility = aclgraph.check_superkernel_eligibility
    effective_launch_floor_us = aclgraph.effective_launch_floor_us
    static = check_superkernel_eligibility(static_decode=True)
    assert static.eligible
    assert static.optimize_options == {"dcci_after_kernel_end": [".*"]}
    assert not check_superkernel_eligibility(static_decode=False).eligible
    assert not check_superkernel_eligibility(static_decode=True, dynamic_tiling_host_sync=True).eligible
    assert not check_superkernel_eligibility(static_decode=True, capturable_ops=False).eligible
    assert effective_launch_floor_us("full") < effective_launch_floor_us("piecewise")
    assert effective_launch_floor_us("full", superkernel=True) < effective_launch_floor_us("full")
    with pytest.raises(aclgraph.ACLGraphModelError, match="unknown graph mode"):
        effective_launch_floor_us("magic")

@pytest.mark.requires_profiler
def test_mcp_envelopes_bounded_and_hostile() -> None:
    from mindie_profiler.aclgraph_mcp import call_tool

    region = {
        "name": "AscendSharedExperts Gate-Up || Dispatch",
        "branch_a": {"op": "gate_up", "stream_id": 0, "start_us": 0, "end_us": 100, "aic_blocks": 20},
        "branch_b": {"op": "dispatch", "stream_id": 1, "start_us": 0, "end_us": 100, "aiv_blocks": 20},
    }
    budget = {"captured_batch_sizes": 8, "subgraphs_per_step": 32, "active_streams_per_graph": 16}
    calls = [
        ("aclgraph_diagnose", {"region": region}),
        ("aclgraph_diagnose", {"region": {**region, "name": "x" * 5000}}),
        ("aclgraph_sq_budget", budget),
        ("aclgraph_superkernel", {"static_decode": False}),
        ("aclgraph_launch_floor", {"mode": "piecewise"}), ("aclgraph_launch_floor", {"mode": "x" * 5000}),
        ("nope", {}),
    ]
    for name, arguments in calls:
        result = call_tool(name, arguments)
        size = _envelope_size(result)
        assert size["utf8_bytes"] <= 250
        assert size["token_estimate"] <= 250
        assert isinstance(result, dict)
    assert call_tool("aclgraph_diagnose", {"region": region})["code"] == "OK"
    assert call_tool("aclgraph_sq_budget", budget)["err"] == "207008"
    assert call_tool("aclgraph_superkernel", {"static_decode": False})["eligible"] == 0

def _captured_cohort():
    raw = cohort_case(2)
    raw["runtime"].update(graph_mode="full", graph_capture_mode="full", graph_capture_keys=[1, 2, 4])
    return raw

@pytest.mark.parametrize("explicit_idle", [False, True])
def test_idle_capture_rows_launch_physical_work_and_keep_logical_zero(explicit_idle):
    graphs = target_graphs(_captured_cohort(), phase="decode", tokens=1, context=4,
                           sequences=[1, 0], scope="body", padded_idle=explicit_idle)
    active, idle = (next(op for op in graphs[rank] if op.family == "attention") for rank in (0, 1))
    assert idle.launch_shape[0] == 1
    _assert_fields(dict(idle.metadata), graph_active_query_rows=0, graph_physical_query_rows=1)
    assert dict(active.metadata)["graph_active_query_rows"] == 1
    if not explicit_idle:
        assert active.launch_shape[0] == 1
        assert (active.requests, idle.requests) == (("analysis0:0",), ("analysis1:padding",))
        assert len(graphs[1]) == len(graphs[0])

def test_physical_global_rank_scope_preserves_tp_owner_identity():
    from types import SimpleNamespace
    from mindie_sim.prediction.frontend.body_spec import physical_global_ranks
    from mindie_sim.schema.parallel import ParallelSpec
    scope = physical_global_ranks(SimpleNamespace(parallel=ParallelSpec(dp=4, tp=4)), (1, 3))
    assert scope[3] == (12, 13, 14, 15)
    assert scope[1] == (4, 5, 6, 7)
    assert scope[3][0] == 12
    assert scope[1][2] == 6

def test_body_geometry_exposes_physical_global_rank_scope_without_relabeling():
    from mindie_sim.prediction.frontend.body_spec import body_forward
    out = body_forward(_captured_cohort(), phase="decode", tokens=1, context=4,
                       sequences=[1, 0], event_limit=0, compact=True)
    assert out["geometry"]["valid_query_tokens_by_replica"] == {0: 1, 1: 0}
    assert out["geometry"]["graph_query_tokens_by_replica"] == {0: 1, 1: 1}
    assert out["geometry"]["physical_global_rank_basis"] == ("dp_replica * tp + tp_rank")
    assert out["geometry"]["physical_global_ranks_by_replica"][1] == (1,)
    assert out["profile"]["by_rank"][1]["operator_events"] > 0

def _clock_record(replica, family, start, end, *, link=0.0, flops=0.0, vector=0.0, hbm=0.0):
    op = SimpleNamespace(
        family=family, link_bytes=link, flops=flops, vector_ops=vector,
        hbm_bytes=hbm, mte2_bytes=0.0, mte3_bytes=0.0, metadata=())
    return {"replica": replica, "op": op,
            "ready": SimpleNamespace(time=start), "complete": SimpleNamespace(time=end)}

def test_graph_costs_publishes_conserving_device_step_breakdown():
    case = replace(dense_case(), machine=hardware(1, rates=(("hbm", 1.0e12), ("vector", 1.0e12))))
    ops = target_operators(case, phase="decode", tokens=1, context=8, sequences=1, scope="body")
    report = graph_costs(
        case, {0: tuple(ops)}, external_inputs={"analysis_input": 0},
        include_submission=False, isolated=False)
    assert report["exposed_breakdown_basis"]["available"] is True
    assert report["exposed_breakdown_basis"]["boundary"] == "device_step"
    basis = report["exposed_breakdown_basis"]
    assert "clock_reconciliation_residual_ms" in basis
    assert "clock_order_clamped_ms" in basis
    assert sum(report["exposed_breakdown_ms"].values()) == pytest.approx(report["duration_ms"], abs=1e-12)

def test_exposed_breakdown_reconciles_within_limit_and_rejects_large_gap():
    records = (_clock_record(0, "gemm", 0.0, 5.0, flops=10.0),
               _clock_record(0, "alltoall", 5.0, 8.0, link=100.0))
    result = _exposed_breakdown(records, 8.0, 0.0, 0)
    values = result["exposed_breakdown_ms"]
    assert all(value >= 0 for value in values.values())
    assert sum(values.values()) == pytest.approx(8.0, abs=1e-12)
    basis = result["exposed_breakdown_basis"]
    assert basis["available"] is True
    assert "clock_reconciliation_residual_ms" in basis
    assert "clock_order_clamped_ms" in basis
    # An execution boundary whose intervals cannot cover the span is still
    # declared unavailable rather than rescaled.
    unavailable = _exposed_breakdown(records, 9.0, 0.0, 0)
    assert unavailable["exposed_breakdown_ms"] is None
    assert unavailable["exposed_breakdown_basis"]["reason"] == ("unreconciled_execution_boundary")
    # A gap larger than a quarter of the reported duration is a real defect.
    from mindie_sim.prediction.efficiency.exposed_breakdown import ExposedClocks, finish_breakdown
    with pytest.raises(ModelingError, match="exposed_attribution_clock_mismatch"):
        finish_breakdown(ExposedClocks(2.0, 2.0, 2.0, 2.0), duration_ms=10.0, boundary="device_step")

def test_scoped_resource_names_map_to_semantic_roles():
    times = _role_times(
        {"memory@r0:s0": 100.0, "matrix@r0:s0": 200.0,
         "vector@r0:s0": 50.0, "link:fabric": 10.0},
        {"memory@r0:s0": 100.0, "matrix@r0:s0": 100.0,
         "vector@r0:s0": 100.0, "link:fabric": 100.0})
    assert times == {"cube_ms": 2.0, "vector_ms": 0.5, "memory_ms": 1.0, "communication_ms": 0.1}
