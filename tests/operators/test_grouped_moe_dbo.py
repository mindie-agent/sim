"""Exercise dual-batch MoE through the governed resolver and causal engine.

[Role] Check caller partitions, cohort transport, and in-operator execution.
[Invariants] Cohorts own every input row exactly once, conserve each expert's
  work, and overlap without violating dispatch, matmul, or buffer dependencies.
[Timeouts & Locks] Small finite DAGs and bounded engine traces.
[Side-Effect Policy] Analytical execution with the session-isolated C11 cache.
"""
from collections import Counter
from dataclasses import dataclass, replace
from graphlib import TopologicalSorter
import pickle

import pytest

from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS
from _support.hardware import isa_of, machine
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.operators import grouped_moe, mc2
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.mc2.common import PeerRoute, Transport
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.results import FusedPipelineDescriptor, OpOccurrence
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier0

CORES = 2
HALF_BATCH = 16
WIDTH = 32
LINK_BYTES = 8192
COHORTS = (tuple(range(HALF_BATCH)), tuple(range(HALF_BATCH, 2 * HALF_BATCH)))
SKEWED = ((HALF_BATCH, 0), (0, HALF_BATCH))
BALANCED = ((HALF_BATCH // 2,) * 2,) * 2
CHAIN = ("dispatch", "gmm1", "swiglu_quant", "gmm2", "combine")
AIC_PIPES = ("cube", "mte1", "mte2_aic", "fixpipe")
AIV_PIPES = ("mte2_aiv", "vector", "mte3")


@pytest.fixture
def geometry(arch):
    sku = next(sku for sku in REPRESENTATIVE_SKUS.values() if isa_of(sku) == arch)
    geometry = Geometry.from_devices(machine(sku).dies[:1])
    units = {**dict.fromkeys(AIC_PIPES, CORES), **dict.fromkeys(AIV_PIPES, 2 * CORES),
             **dict.fromkeys(("link", "l2", "hbm"), 1)}
    return replace(geometry, compute_units=CORES, resource_units=units,
                   datapath=replace(geometry.datapath, aic_count=CORES, aiv_count=2 * CORES))


def make_op(*, mode="dual_batch_moe", rows=SKEWED, ids=COHORTS, **overrides):
    totals = tuple(map(sum, zip(*rows, strict=True)))
    split = dict(cohort0_row_ids=ids[0], cohort1_row_ids=ids[1],
                 cohort0_expert_rows=rows[0], cohort1_expert_rows=rows[1])
    shape = (sum(totals), WIDTH, WIDTH)
    metadata = dict(mode=mode, requested_abi="aclnnMegaMoe", group_rows=totals, f=WIDTH,
                    dbo_split=split, enable_dbo=True, base_m=HALF_BATCH,
                    base_n=2 * WIDTH, top_k=1, ep=2,
                    dispatch_link_bytes=LINK_BYTES, combine_link_bytes=LINK_BYTES)
    metadata.update(overrides)
    return OpSpec("routed", "grouped_moe", "decode", 0, 0, (), shape, shape, shape, shape,
                  dtype="bf16", recipe=mode, metadata=tuple(metadata.items()))


def resolve(op, geometry):
    ctx = OperatorPlanningContext(OpOccurrence(0, 0, 0, op.op_id), geometry, geometry.datapath)
    return grouped_moe.resolve_operator_plan(op, ctx)


def context_for(op, geometry):
    return RecipeContext(
        op, {}, {role: role for role in geometry.resource_units}, {}, CORES,
        compute_peaks={"bf16": 1000}, datapath=geometry.datapath,
        resource_units=geometry.resource_units, pool_names={"ub": "ub"},
        pool_units={"ub": 2 * CORES},
    )


def stages_for(op, geometry):
    resolved = resolve(op, geometry)
    context = context_for(op, geometry)
    return resolved, context, resolved.recipe.program(context).stages


def dependencies(stages):
    direct = {stage.name: stage.after for stage in stages}
    ancestors = {}
    for name in TopologicalSorter(direct).static_order():
        ancestors[name] = set(direct[name]).union(*(ancestors[parent] for parent in direct[name]))
    return ancestors


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("fault,reason", [
    ("overlap", "row_ownership"), ("missing", "row_ownership"),
    ("outside", "row_ownership"), ("expert_sum", "expert_rows_not_conserved"),
    ("parent", "parent_mismatch"), ("capacity", "routing_capacity"),
])
def test_partition_rejects_invalid_row_or_expert_ownership(geometry, fault, reason):
    split = dict(dict(make_op().metadata)["dbo_split"])
    overrides = {}
    if fault == "overlap":
        split["cohort1_row_ids"] = (COHORTS[0][-1], *COHORTS[1][1:])
    elif fault == "missing":
        split["cohort1_row_ids"] = COHORTS[1][1:]
    elif fault == "outside":
        split["cohort1_row_ids"] = (*COHORTS[1][:-1], 2 * HALF_BATCH)
    elif fault == "expert_sum":
        split["cohort1_expert_rows"] = (1, HALF_BATCH - 1)
    elif fault == "parent":
        split["top_k"] = 2
    else:
        split["ep"] = 1
        split["cohort0_expert_rows"] = (HALF_BATCH, 1)
        overrides["ep"] = 1
    with pytest.raises(ModelingError, match=reason):
        resolve(make_op(dbo_split=split, **overrides), geometry)


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("choice", [None, False])
def test_shipping_default_requires_caller_enablement(geometry, choice):
    for mode in ("dual_batch_moe", "mega_moe"):
        op = make_op(mode=mode, enable_dbo=choice)
        op = replace(op, metadata=tuple((key, val) for key, val in op.metadata
                                        if key != "enable_dbo" or choice is not None))
        if mode == "dual_batch_moe":
            with pytest.raises(UnmodeledCapabilityError, match="dbo_requires_explicit_enablement") as error:
                resolve(op, geometry)
            hint = error.value.to_dict()["onboarding_hint"]
            assert hint["schema"] == "mindie-unmodeled-structure-hint/1"
        else:
            resolved, _, stages = stages_for(op, geometry)
            assert resolved.recipe.metadata["enable_dbo"] is False
            assert all(phase.cohort is None for phase in resolved.descriptor.phases)
            assert set(CHAIN) <= {stage.name for stage in stages}
            assert "join_restore" not in {stage.name for stage in stages}


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("mode", ["dual_batch_moe", "mega_moe"])
def test_enabled_plan_covers_both_cohorts_and_pickles(geometry, mode):
    op = make_op(mode=mode, rows=BALANCED, ids=(COHORTS[1], COHORTS[0]))
    resolved, context, stages = stages_for(op, geometry)
    plan, descriptor = resolved.recipe, resolved.descriptor
    assert isinstance(plan, OperatorPlan)
    assert isinstance(descriptor, FusedPipelineDescriptor)
    assert descriptor.owner == resolved.occurrence
    assert descriptor.member_op_ids == (op.op_id,)
    assert pickle.loads(pickle.dumps(descriptor)) == descriptor
    restored = pickle.loads(pickle.dumps(plan))
    assert restored.work == plan.work
    assert restored.program(context).stages == stages
    assert restored.pipeline_spec.phases == descriptor.phases

    logical = {stage.name: stage for stage in stages}
    ancestry = dependencies(stages)
    assert set(logical) == {phase.phase_id for phase in descriptor.phases}
    matrix, resources = Counter(), Counter()
    for phase in descriptor.phases:
        stage = logical[phase.phase_id]
        assert phase.after == stage.after
        assert phase.cohort == stage.metadata.get("cohort_id")
        assert phase.slot == stage.metadata.get("slot")
        work = Work({item.dtype: item.amount for item in phase.work if item.resource == "cube"},
                    {item.resource: item.amount for item in phase.work if item.resource != "cube"})
        assert stage.work == context.bind(work)
        matrix.update(work.matrix)
        resources.update(work.resources)
    assert plan.work == Work(dict(matrix), dict(resources))

    partitions = []
    for cohort, owned in enumerate((COHORTS[1], COHORTS[0])):
        dispatch, gmm, act, down, combine = (f"c{cohort}_{name}" for name in CHAIN)
        for upstream, downstream in zip((dispatch, gmm, act, down), (gmm, act, down, combine)):
            assert upstream in ancestry[downstream]
        meta = logical[dispatch].metadata
        assert meta["row_ids"] == owned
        assert (meta["row_start"], meta["row_end"]) == (min(owned), max(owned) + 1)
        partitions.append((set(owned), meta["expert_rows"]))
    assert partitions[0][0].isdisjoint(partitions[1][0])
    assert partitions[0][0] | partitions[1][0] == set(range(2 * HALF_BATCH))
    assert tuple(map(sum, zip(*(rows for _, rows in partitions)))) == (HALF_BATCH,) * 2
    assert "c0_dispatch" in ancestry["c1_dispatch"]
    assert "c0_gmm1" not in ancestry["c1_dispatch"]
    assert logical["join_restore"].after == ("c0_combine", "c1_combine")


@pytest.mark.parametrize("arch", ARCH_IDS)
def test_empty_cohort_emits_no_launch_transfer_or_slot(geometry):
    rows = ((HALF_BATCH, HALF_BATCH), (0, 0))
    op = make_op(rows=rows, ids=(COHORTS[0] + COHORTS[1], ()),
                 cohort_link_bytes=((LINK_BYTES, LINK_BYTES), (0, 0)))
    resolved, _, stages = stages_for(op, geometry)
    assert {phase.cohort for phase in resolved.descriptor.phases} == {0, None}
    assert all(not stage.name.startswith("c1_") for stage in stages)
    assert stages[-1].after == ("c0_combine",)
    assert resolved.recipe.work.resources["link"] == 2 * LINK_BYTES
    with pytest.raises(ModelingError, match="empty_dbo_cohort_has_work"):
        resolve(make_op(rows=rows, ids=(COHORTS[0] + COHORTS[1], ()),
                        cohort_link_bytes=((LINK_BYTES - 1, LINK_BYTES), (1, 0))), geometry)


@pytest.mark.parametrize("arch", ARCH_IDS)
def test_equal_cohort_sizes_recompute_fractal_and_wave_tails_from_each_histogram(geometry):
    for rows, row_fractals, imbalance in ((SKEWED, 1, 1), (BALANCED, 2, 0)):
        op = make_op(rows=rows)
        _, _, stages = stages_for(op, geometry)
        by_name = {stage.name: stage for stage in stages}
        for cohort in range(len(rows)):
            for name, n, k in (("gmm1", 2 * WIDTH, WIDTH), ("gmm2", WIDTH, WIDTH)):
                meta = by_name[f"c{cohort}_{name}"].metadata
                assert meta["useful_flops"] == 2 * HALF_BATCH * n * k
                assert meta["issued_fractals"] == row_fractals * (n // 16) * (k // 16)
                assert meta["max_waves"] == 1
                assert meta["wave_tail_imbalance"] == imbalance


def execute(geometry, program, backend, *, link_rate=1, serialize=False, numeric_only=False, reserve_ms=0):
    if serialize:
        stages = tuple(replace(stage, after=("c0_combine",)) if stage.name == "c1_dispatch:local"
                       else stage for stage in program.stages)
        program = replace(program, stages=stages)
    rates = {**dict.fromkeys(AIC_PIPES + AIV_PIPES, 512), "cube": 96,
             "hbm": 2048, "l2": 2048, "link": link_rate}
    engine = Engine(
        [Resource(role, rates[role], units=units,
                  policy="fifo" if role == "link" else "fair")
         for role, units in geometry.resource_units.items()],
        pools={"ub": geometry.datapath.usable_bytes("ub")}, pool_units={"ub": 2 * CORES},
        trace_limit=len(program.stages) + 4, backend=backend,
    )
    if numeric_only:
        # Dependencies encode slot lifetimes, and the full replay proves this
        # fixture has enough capacity to grant every allocation immediately.
        program = replace(program, stages=tuple(replace(stage, acquire=(), free=())
                                                for stage in program.stages))
    if reserve_ms:
        units = program.stages[0].acquire[0].units
        busy = Allocation("peer-routing", "ub", geometry.datapath.usable_bytes("ub"), units)
        acquired = engine.submit("peer-reserve", acquire=(busy,))
        engine.advance(until=(acquired,))
        engine.submit("peer-release", release=reserve_ms, free=(busy,))
    complete = program.start(engine, "moe", engine.submit("ready"))
    engine.advance(until=(complete,))
    summary = engine.summary()
    assert summary["live_tasks"] == summary["live_programs"] == 0
    assert not any(summary["live_allocated"].values())
    assert complete.time == engine.now > 0
    if numeric_only:
        assert engine.accel_backend == "c_native", engine.accel_status()
        assert engine.accel_status()["native_steps"] > 0
    trace = {row["task"].removeprefix("moe:"): row for row in engine.trace}
    return complete.time, dict(engine.serviced), trace


def assert_replay_equal(native, python):
    assert native[0] == pytest.approx(python[0], rel=1e-12)
    assert native[1] == pytest.approx(python[1], rel=1e-12)
    assert native[2].keys() == python[2].keys()
    for name, row in native[2].items():
        expected = python[2][name]
        assert row["demand"] == expected["demand"]
        assert row["placement"] == expected["placement"]
        for clock in ("start_ms", "end_ms"):
            assert row[clock] == pytest.approx(expected[clock], abs=1e-10, rel=1e-12)


@pytest.mark.parametrize("arch", ARCH_IDS)
def test_engine_masks_link_service_with_cube_work_under_backpressure(geometry):
    op = make_op(rows=BALANCED)
    resolved, context, stages = stages_for(op, geometry)
    program = resolved.recipe.program(context)
    expected_service = context.bind(resolved.recipe.work)
    slot_claims = {item.name for stage in stages for item in stage.acquire}
    slot_releases = {item.name for stage in stages for item in stage.free}
    assert slot_claims == slot_releases
    assert len(slot_claims) >= 2

    baseline = execute(geometry, program, "python")
    for backend in ("python", "c"):
        result = execute(geometry, program, backend)
        duration, service, trace = result
        serial_duration, serial_service, serial = execute(geometry, program, backend, serialize=True)
        slower_duration, slower_service, _ = execute(geometry, program, backend, link_rate=0.5)
        assert service == serial_service == slower_service == pytest.approx(expected_service)
        assert duration < serial_duration
        assert duration < slower_duration
        link = trace["c1_dispatch:transport"]
        cubes = [row for name, row in trace.items()
                 if name.startswith("c0_gmm1:") and "cube" in row["demand"]]
        assert cubes and all(len(row["placement"]["cube"]) == 1 for row in cubes)
        assert any(max(link["start_ms"], row["start_ms"]) < min(link["end_ms"], row["end_ms"])
                   for row in cubes)
        assert serial["c1_dispatch:transport"]["start_ms"] >= max(
            row["end_ms"] for name, row in serial.items() if name.startswith("c0_combine:"))
        ancestry = dependencies(stages)
        for stage in stages:
            if stage.name in trace:
                for pred in ancestry[stage.name] & trace.keys():
                    assert trace[pred]["end_ms"] <= trace[stage.name]["start_ms"] + 1e-10
        assert_replay_equal(result, baseline)
    native = execute(geometry, program, "c", numeric_only=True)
    assert_replay_equal(native, baseline)
    native_serial = execute(geometry, program, "c", numeric_only=True, serialize=True)
    assert native[0] < native_serial[0]
    assert native_serial[1] == pytest.approx(expected_service)


@dataclass(frozen=True)
class EmbeddedTransportStage:
    name: str
    after: tuple
    work: Work
    kind: str
    metadata: tuple
    owner: int = 0
    ub_acquire: tuple | None = None
    ub_release: tuple | None = None


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("row_ids", [COHORTS, tuple(reversed(COHORTS))])
def test_embedded_mc2_preserves_cohort_rows_and_wire_work(geometry, row_ids):
    route = Transport("declared", "direct_peer", (PeerRoute(0, 1, ("link",)),))
    cohorts = tuple(mc2.Mc2CohortSlice(cohort, min(ids), max(ids) + 1, HALF_BATCH,
                                     HALF_BATCH, HALF_BATCH // 2, HALF_BATCH // 2)
                    for cohort, ids in enumerate(row_ids))
    assignments = tuple((c.cohort_id, src, dst, c.remote_assignments)
                        for c in cohorts for src, dst in ((0, 1), (1, 0)))
    dispatch_layout = mc2.dispatch_row_layout(WIDTH, input_dtype="bf16",
                                             payload_dtype="bf16", quant_mode="none")
    status_bytes = 32
    calls = []

    def transport_builder(name, after, amount, metadata):
        tag = dict(metadata)
        cohort = cohorts[tag["cohort_id"]]
        dispatch = "_dispatch:" in name
        layout = dispatch_layout if dispatch else mc2.combine_row_layout(WIDTH)
        contract = mc2.Mc2Contract(
            "dispatch_v2" if dispatch else "combine_v2", route.abi, route.algorithm,
            (0, 1), 0, 1, True, True, "bf16", output_dtype="bf16", row_layout=layout,
            experts=2, status_bytes=status_bytes, cohorts=cohorts, assignments=assignments,
        )
        phases = mc2.build_cohort_transport_stages(contract, cohort, route, after=after)
        phase_names = {phase.name: f"{name}:{phase.name}" for phase in phases}
        assert sum(p.work.resources.get("link", 0) for p in phases) == amount
        for phase in phases:
            meta = dict(phase.metadata)
            assert (meta["cohort_id"], meta["row_start"], meta["row_end"]) == (
                cohort.cohort_id, cohort.row_start, cohort.row_end)
        calls.append((name, cohort.cohort_id, amount))
        return tuple(EmbeddedTransportStage(
            phase_names[p.name], tuple(phase_names.get(prev, prev) for prev in p.after),
            p.work, p.owner, p.metadata, owner=cohort.cohort_id,
        ) for p in phases)

    per_cohort = HALF_BATCH // 2 * 512 + status_bytes
    op = make_op(ids=row_ids, cohort_transport_builder=transport_builder,
                 dispatch_link_bytes=2 * per_cohort, combine_link_bytes=2 * per_cohort)
    resolved, _, stages = stages_for(op, geometry)
    assert calls == [(f"c{cid}_{kind}:transport", cid, per_cohort)
                     for cid in range(2) for kind in ("dispatch", "combine")]
    assert resolved.recipe.work.resources["link"] == 4 * per_cohort
    graph = dependencies(stages)
    for row in (s for s in stages if ":transport:" in s.name):
        cohort = cohorts[row.metadata["cohort_id"]]
        assert (row.metadata["row_start"], row.metadata["row_end"]) == (cohort.row_start, cohort.row_end)
        assert row.metadata["row_ids"] == row_ids[cohort.cohort_id]
        if row.name.endswith("_consume"):
            assert row.name.removesuffix("_consume").replace("_send_", "_status_") in graph[row.name]
    for cid in range(2):
        assert any(name.endswith("_slot_release") for name in graph[f"c{cid}_dispatch"])
        assert any(name.endswith("_slot_release") for name in graph[f"c{cid}_combine"])


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("backend", ["python", "c"])
def test_routing_waits_for_physical_ub_credit(geometry, backend):
    op = make_op()
    resolved, context, _ = stages_for(op, geometry)
    program = resolved.recipe.program(context)
    ready_at = 7
    baseline = execute(geometry, program, backend)
    delayed = execute(geometry, program, backend, reserve_ms=ready_at)
    assert delayed[2]["c0_dispatch:local"]["start_ms"] == ready_at
    assert delayed[2]["c0_dispatch:transport"]["start_ms"] >= delayed[2]["c0_dispatch:local"]["end_ms"]
    assert delayed[0] == pytest.approx(baseline[0] + ready_at, abs=1e-10)
    assert delayed[1] == pytest.approx(context.bind(resolved.recipe.work), rel=1e-12)


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
@pytest.mark.parametrize("histograms,first_wave_fractals", [(SKEWED, 8), (BALANCED, 16)])
def test_each_cohort_recomputes_its_last_partial_wave(geometry, histograms, first_wave_fractals):
    wave_rows = 12
    op = make_op(rows=histograms, wave_rows=wave_rows)
    _, _, stages = stages_for(op, geometry)
    stage_by_name = {stage.name: stage for stage in stages}
    for cohort in range(2):
        prefix = f"c{cohort}_"
        waves = [stage_by_name[prefix + f"wave{wave}_gmm1"] for wave in range(2)]
        assert prefix + "wave2_gmm1" not in stage_by_name
        assert [stage.metadata["useful_flops"] for stage in waves] == [
            2 * m * (2 * WIDTH) * WIDTH for m in (wave_rows, HALF_BATCH - wave_rows)]
        assert [stage.metadata["issued_fractals"] for stage in waves] == [first_wave_fractals, 8]
        assert all(stage.metadata["cohort_id"] == cohort for stage in waves)
