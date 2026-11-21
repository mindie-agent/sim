"""Exercise dual-batch MoE through the governed resolver and causal engine.

[Role] Check caller partitions, cohort transport, and in-operator execution.
[Invariants] Cohorts own every input row exactly once, conserve each expert's
  work, and overlap without violating dispatch, matmul, or buffer dependencies.
[Timeouts & Locks] Small finite DAGs and bounded engine traces.
[Side-Effect Policy] Analytical execution with the session-isolated C11 cache.
"""
from collections import Counter
from dataclasses import fields, replace
from graphlib import TopologicalSorter
import pickle

import pytest

from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS
from _support.hardware import isa_of, machine
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.operators import grouped_moe, mc2
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.grouped_moe.lowering import StageDemand
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
    metadata["tokens"] = sum(map(len, ids))
    if overrides.get("enable_dbo", True) and "cohort_transport_builder" not in overrides:
        transfers = tuple(overrides.get(key, LINK_BYTES) for key in ("dispatch_link_bytes", "combine_link_bytes"))
        first = tuple(amount // 2 for amount in transfers)
        metadata["cohort_link_bytes"] = (first, tuple(amount - cut for amount, cut in zip(transfers, first)))
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
        assert tuple(map(int, meta["row_ids"].split(","))) == owned
        assert (meta["row_start"], meta["row_end"]) == (cohort * HALF_BATCH, (cohort + 1) * HALF_BATCH)
        assert meta["restore_map"] == meta["row_ids"]
        partitions.append((set(owned), tuple(map(int, meta["expert_rows"].split(",")))))
    assert partitions[0][0].isdisjoint(partitions[1][0])
    assert partitions[0][0] | partitions[1][0] == set(range(2 * HALF_BATCH))
    assert tuple(map(sum, zip(*(rows for _, rows in partitions)))) == (HALF_BATCH,) * 2
    assert "c0_dispatch" in ancestry["c1_dispatch"]
    assert "c0_gmm1" not in ancestry["c1_dispatch"]
    assert logical["join_restore"].after == ("c0_ub_release", "c1_ub_release")


@pytest.mark.parametrize("arch", ARCH_IDS)
def test_empty_cohort_emits_no_launch_transfer_or_slot(geometry):
    rows = ((HALF_BATCH, HALF_BATCH), (0, 0))
    op = make_op(rows=rows, ids=(COHORTS[0] + COHORTS[1], ()),
                 cohort_link_bytes=((LINK_BYTES, LINK_BYTES), (0, 0)))
    resolved, _, stages = stages_for(op, geometry)
    assert {phase.cohort for phase in resolved.descriptor.phases} == {0, None}
    assert all(not stage.name.startswith("c1_") for stage in stages)
    assert stages[-1].after == ("c0_ub_release",)
    assert resolved.recipe.work.resources["link"] == 2 * LINK_BYTES


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


def execute(geometry, program, backend, *, link_rate=1, serialize=False, reserve_ms=0):
    if serialize:
        stages = tuple(replace(stage, after=("c0_combine",)) if stage.name == "c1_ub_admit"
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
    if backend == "c":
        assert engine.accel_status()["native_steps"] > 0, engine.accel_status()
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
    assert all(not stage.work for stage in stages if stage.acquire or stage.free)

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


def explicit_mc2_transport(sends, receives):
    route = Transport("declared", "direct_peer", (PeerRoute(0, 1, ("link",)),))
    status_bytes = 32
    calls = []
    supports_row_ids = any(field.name == "row_ids" for field in fields(mc2.Mc2CohortSlice))

    def transport_builder(name, after, amount, metadata):
        tag = dict(metadata)
        cid = tag["cohort_id"]
        dispatch = tag["transport_direction"] == "dispatch"
        send, receive = (sends[cid], receives[cid]) if dispatch else (receives[cid], sends[cid])
        cohort = mc2.Mc2CohortSlice(
            cid, tag["row_start"], tag["row_end"], send, receive, send, 0,
            **({"row_ids": tag["row_ids"]} if supports_row_ids else {}),
        )
        assignments = ((cid, 0, 1, send), (cid, 1, 0, receive))
        layout = (mc2.dispatch_row_layout(WIDTH, input_dtype="bf16", payload_dtype="bf16", quant_mode="none")
                  if dispatch else mc2.combine_row_layout(WIDTH))
        contract = mc2.Mc2Contract(
            "dispatch_v2" if dispatch else "combine_v2", route.abi, route.algorithm,
            (0, 1), 0, 1, True, True, "bf16", output_dtype="bf16", row_layout=layout,
            experts=2, status_bytes=status_bytes, cohorts=(cohort,), assignments=assignments,
        )
        phases = mc2.build_cohort_transport_stages(contract, cohort, route, after=after)
        phase_names = {phase.name: f"{name}:{phase.name}" for phase in phases}
        wire_bytes = sum(p.work.resources.get("link", 0) for p in phases)
        assert wire_bytes == (send + receive) * 512 + (bool(send) + bool(receive)) * status_bytes
        for phase in phases:
            meta = dict(phase.metadata)
            assert (meta["cohort_id"], meta["row_start"], meta["row_end"]) == (
                cid, cohort.row_start, cohort.row_end)
            if supports_row_ids:
                assert meta["cohort_row_ids"] == tag["row_ids"]
        calls.append((name, cid, wire_bytes))
        # The grouped endpoint owns local pack/unpack. Reuse the MC2 peer
        # protocol and its readiness fences at that boundary.
        return tuple(StageDemand(
            phase_names[p.name], tuple(phase_names.get(prev, prev) for prev in p.after),
            Work() if p.name.endswith(("_pack", "_unpack")) else p.work,
            p.owner, cid, p.metadata,
        ) for p in phases)

    return transport_builder, calls


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("row_ids", [
    COHORTS, tuple(reversed(COHORTS)), (tuple(range(0, 32, 2)), tuple(range(1, 32, 2))),
])
def test_embedded_mc2_preserves_cohort_rows_and_wire_work(geometry, row_ids):
    builder, calls = explicit_mc2_transport((HALF_BATCH, HALF_BATCH), (HALF_BATCH, HALF_BATCH))
    per_cohort = 2 * (HALF_BATCH * 512 + 32)
    op = make_op(ids=row_ids, cohort_transport_builder=builder,
                 dispatch_link_bytes=2 * per_cohort, combine_link_bytes=2 * per_cohort)
    resolved, _, stages = stages_for(op, geometry)
    assert calls == [(f"c{cid}_{kind}:transport", cid, per_cohort)
                     for cid in range(2) for kind in ("dispatch", "combine")]
    assert resolved.recipe.work.resources["link"] == 4 * per_cohort
    graph = dependencies(stages)
    for row in (s for s in stages if ":transport:" in s.name):
        cid = row.metadata["cohort_id"]
        assert (row.metadata["row_start"], row.metadata["row_end"]) == (cid * HALF_BATCH, (cid + 1) * HALF_BATCH)
        expected = ",".join(map(str, row_ids[cid]))
        assert row.metadata["row_ids"] == row.metadata["restore_map"] == expected
        assert row.metadata["cohort_row_ids"] == row.metadata["cohort_restore_map"] == expected
        if row.name.endswith("_consume"):
            assert row.name.replace("_consume", "_ready") in graph[row.name]
    for cid in range(2):
        assert any(name.endswith("_slot_release") for name in graph[f"c{cid}_dispatch"])
        assert any(name.endswith("_slot_release") for name in graph[f"c{cid}_combine"])


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("row_ids,rows", [
    (((0, 2), (1, 3)), ((0, 0), (1, 5))),
    (((), (0, 1, 2, 3)), ((5, 1), (0, 0))),
])
def test_off_rank_cohorts_separate_wire_work_received_rows_and_local_outputs(geometry, row_ids, rows):
    send = tuple(map(len, row_ids))
    receive = tuple(map(sum, rows))
    builder, calls = explicit_mc2_transport(send, receive)
    per_cohort = tuple((a + b) * 512 + (bool(a) + bool(b)) * 32 for a, b in zip(send, receive))
    op = make_op(ids=row_ids, rows=rows, cohort_transport_builder=builder,
                 dispatch_link_bytes=sum(per_cohort), combine_link_bytes=sum(per_cohort))
    resolved, _, stages = stages_for(op, geometry)
    by_name = {stage.name: stage for stage in stages}
    for cid in range(2):
        prefix = f"c{cid}_"
        dispatch, combine = (by_name[prefix + phase + ":local"] for phase in ("dispatch", "combine"))
        assert dispatch.metadata["source_tokens"] == combine.metadata["output_tokens"] == send[cid]
        assert dispatch.metadata["row_ids"] == combine.metadata["restore_map"] == ",".join(map(str, row_ids[cid]))
        assert combine.work.get("mte3", 0) == 2 * WIDTH * send[cid]
        if receive[cid]:
            assert by_name[prefix + "gmm1"].metadata["useful_flops"] == 2 * receive[cid] * WIDTH * (2 * WIDTH)
            assert by_name[prefix + "gmm2"].metadata["useful_flops"] == 2 * receive[cid] * WIDTH * WIDTH
        else:
            assert prefix + "gmm1" not in by_name and prefix + "gmm2" not in by_name
    assert [wire for _, _, wire in calls] == [amount for amount in per_cohort for _ in range(2)]
    assert resolved.recipe.work.resources["link"] == 2 * sum(per_cohort)
    assert sum(by_name[f"c{cid}_combine:local"].metadata["output_tokens"] for cid in range(2)) == 4


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
def test_wave_transport_uses_the_directed_cut_supplied_by_the_caller(geometry):
    wire = {
        (0, 0): (96, 0), (0, 1): (0, 64),
        (1, 0): (32, 64), (1, 1): (128, 0),
    }

    def transport_builder(name, after, _amount, metadata):
        tag = dict(metadata)
        amounts = wire[tag["cohort_id"], tag["wave"]]
        direction = tag["transport_direction"]
        assert tag["wave_row_start"] == tag["wave"]
        assert tag["wave_row_end"] == tag["wave"] + 1
        value = amounts[direction == "combine"]
        return (StageDemand(name, after, Work(resources={"link": value})),) if value else ()

    op = make_op(
        rows=((2, 0), (0, 2)), ids=((0, 2), (1, 3)), wave_rows=1,
        cohort_link_bytes=((96, 64), (160, 64)),
        dispatch_link_bytes=256, combine_link_bytes=128,
        cohort_transport_builder=transport_builder,
    )
    resolved, _, stages = stages_for(op, geometry)
    actual = {}
    for stage in stages:
        if "link" in stage.work:
            tag = stage.metadata
            cid = tag["cohort_id"]
            assert tag["wave_expert_rows"] == ("1,0", "0,1")[cid]
            actual[cid, tag["wave"], tag["transport_direction"]] = stage.work["link"]
    assert actual == {(cid, wave, direction): amount for (cid, wave), pair in wire.items()
                      for direction, amount in zip(("dispatch", "combine"), pair) if amount}
    assert resolved.recipe.work.resources["link"] == 384


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
