"""Causal core of ``runtime.Engine``: timing, ownership, arbitration and lifetimes.

[Role] Pin the replacement execution engine with small independent oracles:
  exact dependency/stream/fair-share timings, per-unit ownership and residency,
  owner-rate partition templates that must equal literal disjoint arbitration,
  shared read-only rate views, pending-cohort stage contexts, and transaction
  address hazards, queue backpressure and nested buffer lifetimes.
[Invariants] Every expected time is hand-derived from capacities (comments
  show the arithmetic); serviced work is conserved exactly; a unit never lends
  idle capacity or buffer bytes to work pinned elsewhere; cached owner-rate
  templates are schedule-identical to the uncached literal arbitration; live
  tasks/programs/allocations return to zero after every drain.
[Timeouts & Locks] Pure in-memory Engine runs, no threads; the module runs
  well under the tier0 5 s budget.
[Side-Effect Policy] No filesystem, environment or network access; the one
  production monkeypatch (``engine.unit_rates``) is scoped by ``monkeypatch``.
"""
from __future__ import annotations

from dataclasses import replace
import math
from random import Random

import pytest

from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction import engine as engine_module
from mindie_sim.prediction import ownership
from mindie_sim.prediction.runtime.coupled import coupled_rates
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, ExecutionError, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.ownership import unit_rates
from mindie_sim.prediction.operators.vector import TiledPointwise
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, RecipeContext, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transactions import QueuedStore
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

INF = float("inf")


def hardware_case():
    """One declared device: bf16 2e6 flop/ms, memory/vector 1e6 units/ms."""
    return {
        "machine": {"name": "arbitrary", "num_devices": 1,
                    "device": {"memory_bytes": 1024**3, "compute_units": 4,
                               "compute_peaks": {"bf16": 2e6, "fp32": 1e6},
                               "rates": {"memory": 1e6, "vector": 1e6}}},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8, "num_heads": 1,
                  "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def copy_op(work=2000, **changes):
    return replace(OpSpec(
        "copy", "copy", "prefill", 0, 0, ("r",),
        (work // 2,), (work // 2,), (work // 2,), (work // 2,),
        hbm_bytes=work, dtype="bf16", layout="contiguous", recipe="copy",
    ), **changes)


def iteration(ops):
    return IterationWork(tuple(ops), IterationPlan((RequestView("r", 1, 0, True, "prefill"),)),
                         1, 1, 1, None, "prefill", "resource_service_control")


# --- Exact timing oracles -------------------------------------------------


def test_dependency_stream_and_independent_resource_overlap():
    engine = Engine([Resource("cube", 10), Resource("hbm", 20)])
    load = engine.submit("load", {"hbm": 100})
    independent = engine.submit("other", {"cube": 40})
    compute = engine.submit("compute", {"cube": 60}, after=(load,))
    engine.advance(until=(compute, independent), first=False)
    # load 100/20=5; other 40/10=4 overlaps; compute starts at 5 -> 5+6=11.
    assert (load.time, independent.time, compute.time) == (5, 4, 11)
    assert engine.serviced == {"cube": 100, "hbm": 100}
    assert engine.summary()["live_tasks"] == 0


@pytest.mark.parametrize("caps,expected", [
    # Fair share 50/50 until short (100) ends at 2; long has 200 left at 100/ms.
    ({}, (2, 4)),
    # Narrow capped at 20 leaves 80 for wide: wide 100/80=1.25; narrow 100/20=5.
    ({"hbm": 20}, (5, 1.25)),
])
def test_fair_service_rebalances_and_capped_supply_leaves_capacity(caps, expected):
    engine = Engine([Resource("hbm", 100)])
    first = engine.submit("first", {"hbm": 100}, caps=caps)
    second = engine.submit("second", {"hbm": 300 if not caps else 100})
    engine.advance(until=(first, second), first=False)
    assert (first.time, second.time) == expected
    assert engine.serviced["hbm"] == (400 if not caps else 200)


def test_host_submission_pipelines_with_device_completion():
    engine = Engine([Resource("host", 1, policy="fifo"), Resource("cube", 1)])
    device = submit = None
    for index in range(3):
        submit = engine.submit(f"submit{index}", {"host": 2}, after=(() if submit is None else (submit,)))
        device = engine.submit(f"kernel{index}", {"cube": 20},
                               after=(submit,) + (() if device is None else (device,)))
    engine.advance(until=(device,))
    # First submit 2ms, then three serial 20ms kernels; later submits hide.
    assert device.time == 62


def test_allocation_lifetime_applies_backpressure_until_the_consumer_finishes():
    engine = Engine([Resource("load", 1), Resource("compute", 1)], pools={"L1": 128})
    completions = []
    for index in range(2):
        buffer = Allocation(f"tile{index}", "L1", 128)
        load = engine.submit(f"load{index}", {"load": 2}, acquire=(buffer,))
        completions.append(engine.submit(f"compute{index}", {"compute": 3}, after=(load,),
                                         free=(buffer,)))
    engine.advance(until=completions, first=False)
    assert [event.time for event in completions] == [5, 10]
    assert engine.peak_allocated == {"L1": 128}
    assert engine.allocated == {"L1": 0}


def test_incremental_arrivals_do_not_restart_work_or_retain_completed_history():
    engine = Engine([Resource("wire", 10)], trace_limit=2)
    first = engine.submit("first", {"wire": 100})
    engine.advance(horizon=3)
    second = engine.submit("second", {"wire": 10})
    engine.advance(until=(second,))
    # 30 done by 3ms; then 5/ms each: second ends at 5, first at 5 + 60/10 = 11.
    assert second.time == 5
    engine.advance(until=(first,))
    assert first.time == 11
    for index in range(1000):
        event = engine.submit(str(index), {"wire": 10}, after=(first,))
        engine.advance(until=(event,))
        first = event
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["peak_live_tasks"] == 2
    assert len(engine.trace) == 2
    assert len(first._waiters) == 0


def test_observation_envelope_does_not_fabricate_resource_work():
    engine = Engine([Resource("cube", 10)])
    event = engine.submit("kernel", {"cube": 10}, latency=3)
    consumer = engine.submit("consumer", {"cube": 10}, after=(event,))
    engine.advance(until=(consumer,))
    assert (event.time, consumer.time) == (3, 4)
    assert engine.serviced["cube"] == 20


@pytest.mark.parametrize("offset", [0, 1000])
def test_tiny_positive_work_and_absolute_arrival_offset(offset):
    engine = Engine([Resource("wire", 1)])
    engine.now = offset
    event = engine.submit("tiny", {"wire": 2e-9})
    engine.advance(until=(event,))
    assert event.time - offset == pytest.approx(2e-9, rel=1e-4)
    assert engine.serviced["wire"] == 2e-9


def test_float_scale_tail_service_and_zero_work_callbacks_are_conserved():
    # Amounts span subnormal through large; service and dependent callbacks
    # observe all work even when time rounds.
    for exponent in (-1074, -1070, -1022, -500, -20, 0, 20, 500, 1000):
        amount = math.ldexp(1.0, exponent)
        engine = Engine([Resource("wire", amount if exponent < -1000 else amount / 3)], trace_limit=10)
        first = engine.submit("first", {"wire": amount})
        callbacks = []
        end = engine.submit("tail", after=(first,), on_complete=lambda e: callbacks.append(e.time))
        engine.advance(until=(end,))
        assert engine.serviced["wire"] == amount
        assert callbacks == [end.time] and first.time == end.time
        assert engine.summary()["live_tasks"] == 0


@pytest.mark.parametrize("slots,expected", [(1, 50_000), (2, 30_002)])
def test_lazy_finite_pipeline_includes_fill_drain_and_bounded_buffer_slots(slots, expected):
    # One slot serialises 2+3 per tile; two slots are cube-bound 3/tile + 2 fill.
    engine = Engine([Resource("load", 1), Resource("cube", 1)])
    plan = PipelineProgram(10_000, slots, lambda i: (
        Stage("load", {"load": 2}), Stage("compute", {"cube": 3}, after=("load",)),
    ))
    complete = plan.start(engine, "gemm", engine.submit("ready"))
    engine.advance(until=(complete,))
    assert complete.time == expected
    assert engine.serviced == {"load": 20_000, "cube": 30_000}
    assert engine.summary()["peak_live_tasks"] <= 8
    assert engine.summary()["live_programs"] == 0


# --- Unit ownership, contention and residency ------------------------------


@pytest.mark.parametrize("rate,count,caps,placements,expected", [
    # Disjoint groups redistribute capped supply only inside the group; units
    # 3 and 7 stay idle although a and d could consume more (sum 600 < 800).
    (800, 8, {"a": INF, "b": 30, "c": INF, "d": 10000, "e": 40},
     {"a": (0, 1), "b": (0, 1), "c": (2,), "d": (4, 5, 6), "e": (4, 5, 6)},
     {"a": 170, "b": 30, "c": 100, "d": 260, "e": 40}),
    # Shared units bind wide/8 + pair/2 = 100; capped unit2 cannot lend.
    (800, 8, {"wide": INF, "pair": INF, "cap": 25}, {"pair": (0, 1), "cap": (2,)},
     {"wide": 160, "pair": 160, "cap": 25}),
    # Partial chain overlap keeps the joint per-unit constraint.
    (400, 4, {"a": 20, "b": INF, "c": INF}, {"a": (0, 1), "b": (1, 2), "c": (2, 3)},
     {"a": 20, "b": 100, "c": 100}),
])
def test_owner_rates_respect_joint_per_unit_constraints(rate, count, caps, placements, expected):
    rates = unit_rates(rate, count, caps, placements)
    assert rates == pytest.approx(expected)
    permuted = {key: tuple(reversed(units)) for key, units in placements.items()}
    assert unit_rates(rate, count, caps, permuted) == pytest.approx(rates)


def test_grouped_owner_release_reallocates_only_its_physical_partition():
    engine = Engine([Resource("copy", 800, units=8)])
    fast = engine.submit("fast", {"copy": 170}, placement={"copy": (0, 1)})
    capped = engine.submit("capped", {"copy": 60}, caps={"copy": 30}, placement={"copy": (0, 1)})
    other = engine.submit("other", {"copy": 300}, placement={"copy": (4, 5, 6)})
    engine.advance(until=(fast, capped, other), first=False)
    assert (fast.time, capped.time, other.time) == pytest.approx((1, 2, 1))
    assert engine.serviced["copy"] == 530
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


@pytest.mark.parametrize("placements,expected", [(((0,), (1,)), 1), (((0,), (0,)), 2)])
def test_idle_units_cannot_supply_work_owned_by_one_unit(placements, expected):
    engine = Engine([Resource("vector", 400, units=4)])
    events = [engine.submit(str(i), {"vector": 100}, placement={"vector": units})
              for i, units in enumerate(placements)]
    engine.advance(until=events, first=False)
    assert engine.now == pytest.approx(expected)
    assert engine.serviced["vector"] == 200


def test_overlapping_groups_share_the_intersection_and_conserve_total_work():
    engine = Engine([Resource("matrix", 200, units=2)])
    group = engine.submit("pair", {"matrix": 200}, placement={"matrix": (0, 1)})
    single = engine.submit("single", {"matrix": 100}, placement={"matrix": (1,)})
    engine.advance(until=(group, single), first=False)
    # Both run at 200/3 until single ends at 1.5; pair's last 100 uses both units.
    assert (single.time, group.time) == pytest.approx((1.5, 2))
    assert engine.serviced["matrix"] == 300


@pytest.mark.parametrize("owners,expected", [((0, 1), 1), ((0, 0), 2)])
def test_buffers_are_independent_per_unit_and_never_aggregated(owners, expected):
    engine = Engine([], pools={"ub": 64}, pool_units={"ub": 4})
    events = []
    for index, unit in enumerate(owners):
        allocation = Allocation(f"resident{index}", "ub", 64, (unit,))
        events.append(engine.submit(str(index), latency=1, acquire=(allocation,), free=(allocation,)))
    engine.advance(until=events, first=False)
    assert engine.now == expected
    assert engine.peak_allocated["ub"] == 64 and engine.allocated["ub"] == 0
    assert not engine.unit_allocated
    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        engine.submit("cannot_borrow", acquire=(Allocation("large", "ub", 65, (0,)),))


def test_nested_pipeline_preserves_unit_identity_and_releases_every_live_allocation():
    engine = Engine([Resource("vector", 200, units=2)], pools={"ub": 64}, pool_units={"ub": 2})

    def tile(index):
        allocation = Allocation(f"tile{index}", "ub", 64, (1,))
        return (Stage("resident", acquire=(allocation,), free=(allocation,),
                      program=KernelProgram((Stage("work", {"vector": 100},
                                                   placement={"vector": (1,)}),))),)
    end = PipelineProgram(200, 2, tile).start(engine, "pipeline", engine.submit("ready"))
    engine.advance(until=(end,))
    assert end.time == pytest.approx(200)
    assert engine.serviced["vector"] == 20000
    assert engine.summary()["peak_allocated_per_unit"] == {"ub:unit1": 64}
    assert not engine.allocations and not engine.unit_allocated
    assert engine.peak_tasks < 30


def test_cross_iteration_handoff_allows_prefetch_but_protects_reused_output():
    engine = Engine([Resource("copy", 1), Resource("matrix", 1), Resource("vector", 1)])

    def tile(index):
        return (Stage("load", {"copy": 1}, previous=("matrix",)),
                Stage("matrix", {"matrix": 1}, after=("load",), previous=("output",)),
                Stage("output", {"vector": 4}, after=("matrix",)))
    end = PipelineProgram(2, 2, tile).start(engine, "head", engine.submit("ready"))
    engine.advance(until=(end,))
    # load0[0,1] matrix0[1,2] output0[2,6]; load1[2,3] overlaps output0,
    # matrix1 waits until 6, output1[7,11].
    assert end.time == 11
    assert engine.serviced == {"copy": 2, "matrix": 2, "vector": 8}


# --- Owner-rate templates == literal disjoint arbitration -----------------


def literal_rates(rate, count, caps, placements):
    result = ownership._partition_rates(rate, count, caps, placements)
    return ownership.unit_rates(rate, count, caps, placements) if result is None else result


def test_ordinal_cache_preserves_exact_rates_order_and_rebinding():
    random = Random(31030)
    ownership._partition_template.cache_clear()
    for _ in range(500):
        units = random.choice((1, 2, 4, 8, 32, 64))
        keys = random.sample(range(1000, 10000), random.randrange(1, units * 3 + 1))
        caps = {key: random.choice((INF, 0, 1 / 7, 33.3, 10000)) for key in keys}
        placements = {key: (random.randrange(units),) for key in keys}
        rate = random.choice((1 / 3, 1200.5, 4224000000))
        expected = ownership._partition_rates(rate, units, caps, placements)
        actual = ownership._cached_partition_rates(rate, units, caps, placements)
        assert list(actual.items()) == list(expected.items())
        # New IDs cannot retain a prior task or change tie ordering.
        names = [f"new:{key}" for key in keys]
        rebound = ownership._cached_partition_rates(
            rate, units, dict(zip(names, caps.values())), dict(zip(names, placements.values())))
        index = {key: i for i, key in enumerate(keys)}
        assert list(rebound.items()) == [(names[index[k]], v) for k, v in expected.items()]
    assert ownership._partition_template.cache_info().currsize <= 512


def test_overlapping_or_reordered_groups_do_not_reuse_disjoint_topology():
    caps = {91: 30.1, 55: INF, 872: INF}
    first = {91: (0, 1), 55: (0, 1), 872: (2, 3)}
    original = ownership.unit_rates(99.1, 4, caps, first)
    assert list(original.items()) == list(ownership._partition_rates(99.1, 4, caps, first).items())
    for placement in ({91: (0, 1), 55: (1, 2), 872: (2, 3)}, {91: (0, 1), 55: (1, 0), 872: (2, 3)}, {}):
        if ownership._partition_rates(99.1, 4, caps, placement) is None:
            assert ownership._cached_partition_rates(99.1, 4, caps, placement) is None
    changed = ownership.unit_rates(99.1, 4, {**caps, 91: 1.2}, first)
    assert changed[91] == 1.2 and changed != original
    assert ownership.unit_rates(99.1, 8, caps, first) != original


def test_large_live_cohort_does_not_enter_retained_template_cache():
    ownership._partition_template.cache_clear()
    caps = {index: INF for index in range(1024)}
    placements = {index: (index % 64,) for index in caps}
    actual = ownership.unit_rates(123.7, 64, caps, placements)
    assert list(actual.items()) == list(ownership._partition_rates(123.7, 64, caps, placements).items())
    assert ownership._partition_template.cache_info().currsize == 0


def test_coupled_detection_after_ordinary_contenders_and_empty_coupled_task():
    engine = Engine([Resource("read", 27.3, units=2), Resource("compute", 19.1, units=2)])
    engine.submit("ordinary", {"read": 7}, placement={"read": (0,)})
    engine.submit("empty-coupled", latency=3, coupled=True)
    engine.submit("coupled", {"read": 11, "compute": 9}, coupled=True,
                  placement={"read": (1,), "compute": (0,)})
    engine._settle()
    expected = coupled_rates(engine.resources, {key: engine._tasks[key] for key in engine._active})
    assert engine._rates() == expected
    engine.advance()
    assert engine.summary()["live_tasks"] == 0
    assert engine.serviced == {"read": 18.0, "compute": 9.0}


def _owner_schedule(seed):
    r = Random(seed)
    engine = Engine([Resource("copy", 97.3, units=4), Resource("vector", 300.1, units=4)],
                    pools={"ub": 64}, pool_units={"ub": 4}, trace_limit=4000)
    ready = engine.submit("late", latency=7)
    ends, callbacks = [], []
    for owner in range(4):
        def tile(index, owner=owner):
            buf = Allocation(f"{owner}:{index}", "ub", 32, (owner,))
            return (Stage("read", {"copy": r.randrange(1, 8)}, acquire=(buf,), placement={"copy": (owner,)},
                          caps={"copy": 13.7 if owner % 2 else 1000}),
                    Stage("compute", {"vector": r.randrange(1, 8)}, after=("read",),
                          placement={"vector": (owner,)}, free=(buf,)))
        end = PipelineProgram(7, 2, tile).start(engine, f"owner{owner}", ready)
        engine.submit(f"callback{owner}", after=(end,),
                      on_complete=lambda event, owner=owner: callbacks.append((owner, event.time)))
        ends.append(end)
    engine.submit("overlap", {"copy": 9}, release=7.02, placement={"copy": (0, 1)}, caps={"copy": 18.3})
    engine.advance(until=ends, first=False)
    engine.advance()
    return engine.summary(), engine.trace, callbacks


def test_actual_engine_schedule_service_buffers_and_callbacks_are_identical(monkeypatch):
    cached = [_owner_schedule(seed) for seed in range(8)]
    monkeypatch.setattr(engine_module, "unit_rates", literal_rates)
    assert [_owner_schedule(seed) for seed in range(8)] == cached
    assert all(not result[0]["live_tasks"] and not result[0]["live_programs"] for result in cached)


# --- Execution rate views and pending-cohort stage contexts ---------------


def test_contexts_share_readonly_rates_and_see_new_pd_namespace():
    shared = Engine(())
    raw = hardware_case()
    raw["machine"]["architecture"] = "declared"
    raw["machine"]["links"] = [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                                "members": [0], "semantics": "shared"}]
    parsed = parse_case(raw)
    prefill = Execution(parsed, CostModel(parsed), engine=shared, namespace="prefill:")
    first = prefill._context(copy_op(0, flops=1e6), 0, "none")
    second = prefill._context(copy_op(0, flops=2e6), 0, "none")
    assert first.rates is second.rates
    before = dict(first.rates)
    decode = Execution(parsed, CostModel(parsed), engine=shared, namespace="decode:")
    other = decode._context(copy_op(0, flops=1e6), 0, "none")
    assert first.rates["decode:matrix@r0:s0"] == other.rates["decode:matrix@r0:s0"]
    assert all(first.rates[key] == value for key, value in before.items())
    assert len(first.rates) > len(before)


def test_pending_context_storage_scales_by_stage_while_each_replica_keeps_its_rate():
    raw = hardware_case()
    raw["machine"]["num_devices"] = 2
    raw["runtime"]["dp"] = 2
    parsed = parse_case(raw)
    second = parsed.machine.dies[1]
    second = replace(second, rates=tuple({**dict(second.rates), "hbm": 2e6}.items()))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(parsed.machine.dies[0], second)))
    contexts = {0: [], 1: []}

    class ObservedExecution(Execution):
        def _context(self, op, replica, scope, **options):
            context = super()._context(op, replica, scope, **options)
            contexts[replica].append(context)
            return context

    execution = ObservedExecution(parsed, CostModel(parsed), include_submission=False)
    ops = tuple(copy_op(2000, op_id=str(i)) for i in range(128))
    steps = execution.add_group({replica: iteration(ops) for replica in contexts}, 7)
    shared_fields = ("resource_names", "resource_units", "work_units", "compute_peaks",
                     "pool_names", "pool_units", "pool_capacities", "service_modes", "store_queues")
    for replica, records in contexts.items():
        for name in shared_fields:
            assert len({id(getattr(record, name)) for record in records}) == 1
        assert len({id(record.native_work) for record in records}) == len(ops)
        assert records[0].resource_names["hbm"] == f"memory@r{replica}:s0"
    execution.engine.advance(until=tuple(step.completion for step in steps.values()), first=False)
    # 128 copies x 2000 bytes at 1e6 vs 2e6 bytes/ms after the 7 ms release.
    assert steps[0].completion.time == 7 + 128 * 2
    assert steps[1].completion.time == 7 + 128
    assert execution.engine.serviced["memory@r0:s0"] == 256000
    assert execution.engine.serviced["memory@r1:s0"] == 256000
    assert not execution.engine.summary()["live_tasks"]


def test_operator_precision_alias_extends_only_its_local_binding():
    raw = hardware_case()
    raw["machine"]["device"]["compute_peaks"]["fp16"] = 4e6
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    op = copy_op(0, flops=2000, matrix_flops=(("float16", 2000),), dtype="float16")
    context = execution._context(op, 0, "none")
    assert context.compute_peaks["float16"] == 4e6
    assert "float16" not in execution.stage_contexts[0, 0].compute_peaks
    step = execution.add(iteration((op,)), 0, 3)
    execution.engine.advance(until=(step.completion,))
    assert step.completion.time == pytest.approx(3.5)
    assert execution.engine.serviced["matrix@r0:s0"] == .5


# --- Transactions: address hazards, backpressure, lifetimes ---------------


def queued_store(**changes):
    return replace(QueuedStore(
        rows=2, cols=2, row_stride_bytes=4, element_bytes=2, offset_bytes=0, participants=1,
        issue_resource="issue", issue_rate=1000, memory_resource="memory",
        granule_bytes=8, outstanding=2, response_ms=.01), **changes)


def run_store(program, *, bandwidth=1e20, background=0):
    engine = Engine([Resource("issue", 1000), Resource("memory", bandwidth)], trace_limit=100)
    if background:
        engine.submit("other_memory_consumer", {"memory": background})
    done = program.start(engine, "store", engine.submit("ready"))
    engine.advance(until=(done,))
    return done.time, engine


@pytest.mark.parametrize("changes,expected", [
    # Two slots: issues at 1,2,12,22 us; one shared granule returns 11,21,31,41.
    ({}, .041),
    # Two independent granules permit responses at 11,21,22,32 us.
    ({"row_stride_bytes": 8}, .032),
    ({"row_stride_bytes": 8, "outstanding": 1}, .044),
])
def test_same_address_block_serializes_responses_not_all_independent_transfers(changes, expected):
    elapsed, engine = run_store(queued_store(**changes))
    assert elapsed == pytest.approx(expected)
    assert engine.serviced == {"issue": 4, "memory": 8}
    assert engine.summary()["live_tasks"] == 0


def test_memory_contention_keeps_slots_occupied_until_real_completion():
    uncontended, _ = run_store(queued_store(row_stride_bytes=8), bandwidth=100)
    contended, engine = run_store(queued_store(row_stride_bytes=8), bandwidth=100, background=100)
    assert contended > uncontended
    transfers = [row for row in engine.trace if row["task"].endswith(":response")]
    assert sum(row["serviced"]["memory"] for row in transfers) == 8
    # A fixed timer must not release the slot while the payload is still queued.
    first = next(row for row in engine.trace if row["task"].endswith("request0:response"))
    third = next(row for row in engine.trace if row["task"].endswith("request2:issue"))
    assert third["start_ms"] >= first["end_ms"]


def test_queue_working_set_is_bounded_and_nested_stage_releases_after_its_child():
    allocation = Allocation("output", "ub", 16)
    program = KernelProgram((Stage("write", acquire=(allocation,), free=(allocation,),
                                   program=queued_store(rows=1024, row_stride_bytes=128)),))
    engine = Engine([Resource("issue", 1000), Resource("memory", 1e20)], pools={"ub": 16}, trace_limit=5)
    done = program.start(engine, "kernel", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced == {"issue": 2048, "memory": 4096}
    assert engine.peak_tasks < 20
    assert engine.peak_allocated["ub"] == 16 and engine.allocated["ub"] == 0
    writes = [row for row in engine.trace if row["task"].endswith(":response")]
    freed = next(row for row in engine.trace if row["task"].endswith(":free"))
    assert freed["start_ms"] >= max(row["end_ms"] for row in writes)
    assert not engine._tasks and not engine._promises


def test_distinct_unit_address_phases_do_not_create_per_tile_global_barriers():
    rates = {"hbm": 1e20, "vector": 1e20, "issue": 3}
    engine = Engine([Resource(key, value) for key, value in rates.items()])
    op = vector_op(op_id="store", deps=(), stage=0, layer=0, requests=(), elements=81,
                   ops_per_element=0, dtype="bf16", family="store", phase="primitive",
                   recipe="tiled", shape=(9, 9))
    context = RecipeContext(op, {}, {key: key for key in rates}, rates, 3)
    recipe = TiledPointwise(0, ({
        "bytes_per_element": 2, "resource": "issue",
        "queue": {"granule_bytes": 16, "outstanding": 2, "response_ms": 5, "row_control_ms": 1},
    },), tile_rows=3, tile_cols=3)
    done = recipe.lower(context).start(engine, "pointwise", engine.submit("ready"))
    engine.advance(until=(done,))
    # Tile spans for units 0/1/2 are [38,38,35], [38,35,38], [35,38,34]:
    # independent workers take 111; a barrier at each tile would give 114.
    assert done.time == pytest.approx(111)
    assert engine.serviced["issue"] == 81 and engine.serviced["hbm"] == 162
