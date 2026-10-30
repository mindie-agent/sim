"""Source-selected KDA prefill FWD_H state pipeline.

The arithmetic oracle enumerates the retained source's per-(sequence, head)
chunk intervals.  It shares no closed form with ``kda_state.build``.  The
engine controls exercise owner supply, delay-ready submission, serial state
handoffs, buffer/task release and the ordinary model frontend selection.
"""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_kda import IMPLEMENTATION as OLD_IMPLEMENTATION
from mindie_sim.prediction.family_kda_state import IMPLEMENTATION, SOURCE
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operator_analysis import (
    operator_costs, physical_signature, target_operators,
)
from mindie_sim.prediction.operators import kda_state
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.recurrent import linear
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import WorkloadSpec
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

K = V = 128
CHUNK = 64


def params(lengths=(64,), heads=2, **overrides):
    value = {
        "lengths": list(lengths), "heads": heads, "key_heads": heads,
        "dk": K, "dv": V, "chunk": CHUNK, "dtype": "bf16",
        "gate_dtype": "fp32", "state_dtype": "fp32",
    }
    value.update(overrides)
    return value


def hardware_case(*, units=2, split_copy=False, rates=None, compute_bf16=None):
    base = {
        "memory": 1e12, "vector": 2e12,
        "mte1": 1e12, "fixpipe": 1e12, "mte3": 1e12,
    }
    resource_units = {"vector": 2 * units}
    if split_copy:
        resource_units.update(mte2_aic=units, mte2_aiv=2 * units)
        base.update(mte2_aic=1e12, mte2_aiv=2e12)
    else:
        base["mte2"] = 1e12
    if rates:
        base.update(rates)
    return {
        "machine": {
            "name": "kda-source-control", "num_devices": 1,
            "device": {
                "memory_bytes": 64 * 1024 ** 3,
                "compute_units": units,
                "compute_peaks": {"bf16": compute_bf16 or units * 1e14,
                                  "fp32": units * 2.5e13},
                "rates": base,
                "resource_units": resource_units,
            },
        },
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                                   "output_tokens": 1}]},
        "runtime": {},
    }


def execute(value, case, *, trace_limit=0):
    op = kda_state.make_op(value)
    parsed = parse_case(case)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(kda_state.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes,
                          trace_limit=trace_limit)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one")
    result = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution.engine


# ---------------------------------------------------------------------------
# Independent source-interval oracle.
# ---------------------------------------------------------------------------

def tasks_for(lengths, heads, cores):
    total = len(lengths) * heads
    slots = 2 if total > cores else 1
    stride = cores * slots
    result = []
    for core in range(cores):
        base = core * slots
        rounds = []
        while base < total:
            round_tasks = []
            for slot in range(slots):
                index = base + slot
                if index >= total:
                    continue
                sequence, head = divmod(index, heads)
                round_tasks.append((core, slot, sequence, head))
            if round_tasks:
                rounds.append(tuple(round_tasks))
            base += stride
        if rounds:
            result.append((core, tuple(rounds)))
    return result


def oracle(lengths, heads, cores):
    """Enumerate every source chunk interval without using the planner."""
    lengths = tuple(lengths)
    work = {"matrix": 0, "mte2": 0, "mte3": 0, "hbm": 0, "vector": 0,
            "mte1": 0, "fixpipe": 0, "aic_mte2": 0}
    counts = {"streams": len(lengths) * heads, "init_reads": 0,
              "init_writes": 0, "state_reads": 0, "state_writes": 0,
              "h_reads": 0, "h_writes": 0, "final_writes": 0}
    for _core, rounds in tasks_for(lengths, heads, cores):
        for round_tasks in rounds:
            for _core, _slot, sequence, _head in round_tasks:
                length = lengths[sequence]
                chunks = (length + CHUNK - 1) // CHUNK
                counts["init_reads"] += 1
                counts["init_writes"] += 1
                counts["final_writes"] += 1
                for index in range(chunks):
                    valid = min(CHUNK, length - index * CHUNK)
                    first, last = index == 0, index == chunks - 1
                    h_snapshot = K * V * 2
                    state = K * V * 4
                    v_work = valid * V * 4
                    k_work = valid * K * 2
                    h_work = K * V * 4
                    if first:
                        work["mte2"] += state
                        work["mte3"] += h_snapshot
                        work["vector"] += K * V
                        counts["state_reads"] += 1
                    work["matrix"] += 2 * valid * K * V
                    work["mte2"] += valid * K * 2 + h_snapshot
                    work["aic_mte2"] += valid * K * 2 + h_snapshot
                    work["mte1"] += 2 * (valid * K + K * V)
                    work["fixpipe"] += v_work
                    counts["h_reads"] += 1

                    work["mte2"] += valid * V * 2 + v_work + k_work
                    work["mte3"] += 2 * valid * V * 2 + k_work
                    work["vector"] += 4 * valid * V

                    work["matrix"] += 2 * K * valid * V
                    work["mte2"] += k_work + valid * V * 2
                    work["aic_mte2"] += k_work + valid * V * 2
                    work["mte1"] += 2 * (valid * K + valid * V)
                    work["fixpipe"] += h_work

                    work["mte2"] += h_work + K * 4 + state
                    work["mte3"] += state + (0 if last else h_snapshot)
                    work["vector"] += 2 * K + K * V * (2 + (0 if last else 1))
                    counts["state_reads"] += 1
                    counts["state_writes"] += 1
                    counts["h_writes"] += 1
    work["hbm"] = work["mte2"] + work["mte3"] + work["fixpipe"]
    return work, counts


def plan_ledger(value, cores, *, split_copy=False):
    op = kda_state.make_op(value)
    units = {"vector": 2 * cores}
    if split_copy:
        units.update(mte2_aic=cores, mte2_aiv=2 * cores)
    return kda_state.build(op, Geometry(cores, units, {})), op


def assert_matches(plan, expected, *, split_copy):
    assert plan.work.matrix == ({"bf16": expected["matrix"]}
                                if expected["matrix"] else {})
    resources = plan.work.resources
    if split_copy:
        assert resources.get("mte2_aic", 0) == pytest.approx(expected["aic_mte2"])
        assert (resources.get("mte2_aic", 0) + resources.get("mte2_aiv", 0)
                == pytest.approx(expected["mte2"]))
    else:
        assert resources.get("mte2", 0) == pytest.approx(expected["mte2"])
    for name in ("mte3", "hbm", "vector", "mte1", "fixpipe"):
        assert resources.get(name, 0) == pytest.approx(expected[name])
    assert resources["hbm"] == pytest.approx(
        expected["mte2"] + expected["mte3"] + expected["fixpipe"])


@pytest.mark.parametrize("lengths,heads,cores", [
    ((64,), 1, 1),
    ((65,), 2, 2),
    ((128,), 3, 4),
    ((65, 1), 2, 4),
    ((193,), 2, 2),
    ((63, 1, 65), 3, 5),
])
def test_source_interval_work_matches_plan(lengths, heads, cores):
    value = params(lengths=lengths, heads=heads)
    for split in (False, True):
        plan, _ = plan_ledger(value, cores, split_copy=split)
        assert_matches(plan, oracle(lengths, heads, cores)[0], split_copy=split)


def test_ragged_tail_is_real_and_not_padded_chunk_work():
    for length in (65, 127, 193):
        value = params(lengths=(length,), heads=1)
        plan, _ = plan_ledger(value, 1)
        chunks = (length + CHUNK - 1) // CHUNK
        padded = chunks * CHUNK
        expected = oracle((length,), 1, 1)[0]
        assert plan.work.matrix["bf16"] == 4 * length * K * V
        assert plan.work.matrix["bf16"] < 4 * padded * K * V
        assert_matches(plan, expected, split_copy=False)


# ---------------------------------------------------------------------------
# Source ownership, independent state streams and idle-supply controls.
# ---------------------------------------------------------------------------

def test_contract_summary_records_the_selected_source_branch():
    op = kda_state.make_op(params(lengths=(65, 1), heads=2))
    meta = dict(op.metadata)
    assert op.recipe == kda_state.RECIPE
    assert op.source_boundary == "kernel"
    assert meta["source_phase"] == "FWD_H"
    assert meta["state_dtype"] == "fp32"
    assert meta["internal_state_layout"] == "K,V"
    assert meta["public_state_layout"] == "V,K"
    assert meta["caller_state_transpose"] == "outside_selected_fwd_h_phase"
    dispatch = meta["dispatch"]
    assert dispatch["k_gated"] is True and dispatch["scalar_gated"] is False
    assert dispatch["use_exp2"] is True
    assert SOURCE["revision"] == "ae923905948f360f4b7ab06df52d0789a0c9316e"
    assert SOURCE["sha256"]["vllm_ascend/ops/kimi_kda.py"].startswith("35365b3b")


def test_source_stride_ownership_and_idle_cores_are_not_added():
    expected = tasks_for((65, 1), 2, 4)
    actual = [
        (group.core, tuple(tuple((task.core, task.slot, task.sequence, task.head)
                                 for task in round_tasks)
                           for round_tasks in group.rounds))
        for group in kda_state.head_ownership((65, 1), 2, 4)
    ]
    assert actual == expected
    # One head means one core; a 32-core device cannot split its serial state
    # chain or borrow another core's vector/copy service.
    one = execute(params(lengths=(128,), heads=1),
                  hardware_case(units=1, rates={
                      "memory": 1e30, "mte2": 1e30, "mte3": 1e30,
                      "mte1": 1e30, "fixpipe": 1e30, "vector": 2e9}), )
    many = execute(params(lengths=(128,), heads=1),
                   hardware_case(units=32, rates={
                       "memory": 1e30, "mte2_aic": 32e30, "mte2_aiv": 64e30,
                       "mte3": 64e30, "mte1": 32e30, "fixpipe": 32e30,
                       "vector": 64e9}, split_copy=True))
    assert many.now == pytest.approx(one.now, rel=1e-12)


def test_two_sequences_keep_two_state_streams_and_initialization_writes():
    one_plan, one_op = plan_ledger(params(lengths=(128,), heads=1), 2)
    two_plan, two_op = plan_ledger(params(lengths=(64, 64), heads=1), 2)
    one_expected, one_counts = oracle((128,), 1, 2)
    two_expected, two_counts = oracle((64, 64), 1, 2)
    assert one_counts["streams"] == 1 and two_counts["streams"] == 2
    assert two_counts["init_reads"] == two_counts["init_writes"] == 2
    assert two_counts["final_writes"] == 2
    assert one_counts["init_reads"] == one_counts["init_writes"] == 1
    assert one_counts["final_writes"] == 1
    assert two_plan.work.matrix == one_plan.work.matrix
    assert two_plan.work.resources["vector"] == one_plan.work.resources["vector"]
    assert (two_plan.work.resources["mte2"] - one_plan.work.resources["mte2"]
            == K * V * 4)
    assert (two_plan.work.resources["hbm"] - one_plan.work.resources["hbm"]
            == K * V * 4)
    assert_matches(one_plan, one_expected, split_copy=False)
    assert_matches(two_plan, two_expected, split_copy=False)
    assert dict(one_op.metadata)["head_tasks"] == 1
    assert dict(two_op.metadata)["head_tasks"] == 2

    engine = execute(params(lengths=(64, 64), heads=1),
                     hardware_case(units=2), trace_limit=10000)
    assert sum(row["task"].endswith(":init") for row in engine.trace) == 2
    assert sum(row["task"].endswith(":states0:store") for row in engine.trace) == 2


def test_source_owner_supply_exposes_vector_and_copy_constraints():
    fast_case = hardware_case(units=1, rates={
        "memory": 1e30, "mte2": 1e30, "mte3": 1e30, "mte1": 1e30,
        "fixpipe": 1e30, "vector": 2e6}, compute_bf16=1e15)
    slow_case = hardware_case(units=1, rates={
        "memory": 1e30, "mte2": 1e30, "mte3": 1e30, "mte1": 1e30,
        "fixpipe": 1e30, "vector": 1e6}, compute_bf16=1e15)
    value = params(lengths=(128,), heads=1)
    fast = execute(value, fast_case)
    slow = execute(value, slow_case)
    expected_vector = oracle((128,), 1, 1)[0]["vector"]
    assert fast.now == pytest.approx(expected_vector * 1000 / 2e6, rel=1e-6)
    assert slow.now == pytest.approx(2 * fast.now, rel=1e-6)

    # A slow copy pipe also binds the owner and is not rescued by idle cores.
    copy_fast = execute(value, hardware_case(units=1, rates={
        "memory": 1e30, "mte2": 1e5, "mte3": 1e5, "mte1": 1e30,
        "fixpipe": 1e30, "vector": 1e30}, compute_bf16=1e15))
    copy_slow = execute(value, hardware_case(units=1, rates={
        "memory": 1e30, "mte2": 0.5e5, "mte3": 0.5e5, "mte1": 1e30,
        "fixpipe": 1e30, "vector": 1e30}, compute_bf16=1e15))
    assert copy_slow.now == pytest.approx(2 * copy_fast.now, rel=1e-9)


# ---------------------------------------------------------------------------
# Engine contract: delay-ready, causal chunk chain, bounded live state.
# ---------------------------------------------------------------------------

def test_actual_engine_obeys_input_readiness_and_conserves_service():
    value = params(lengths=(65, 1), heads=2)
    op = kda_state.make_op(value)
    for split in (False, True):
        case = hardware_case(units=4, split_copy=split)
        report = verify_operator(case, op, kda_state.build, input_ready_ms=3)
        assert report["status"] == "passed", report
        assert report["input_dependency_obeyed"] is True
        assert report["buffers_released"] is True
        assert report["work_mismatches"] == {}
        row = report["result"]
        assert row["duration_ms"] > 0
        expected = oracle((65, 1), 2, 4)[0]
        assert row["serviced"]["matrix@r0:s0"] == pytest.approx(
            1000 * expected["matrix"] / (4 * 1e14), rel=1e-12)
        assert row["serviced"]["memory@r0:s0"] == pytest.approx(expected["hbm"])
        if split:
            assert row["serviced"]["mte2_aic@r0:s0"] == pytest.approx(
                expected["aic_mte2"])
            assert row["serviced"]["mte2_aiv@r0:s0"] == pytest.approx(
                expected["mte2"] - expected["aic_mte2"])
        else:
            assert row["serviced"]["mte2@r0:s0"] == pytest.approx(
                expected["mte2"])
        assert row["live_tasks"] == 0
        assert not any(row["remaining_buffers"].values())


def test_causal_c1_delta_c2_state_handoffs_by_chunk():
    engine = execute(params(lengths=(128,), heads=1),
                     hardware_case(units=1, rates={
                         "memory": 1e30, "mte2": 1e30, "mte3": 1e30,
                         "mte1": 1e30, "fixpipe": 1e30, "vector": 1e9}),
                     trace_limit=10000)

    def event(chunk, name):
        return next(row for row in engine.trace
                    if f":tile{chunk}:{name}" in row["task"])

    init_end = max(row["end_ms"] for row in engine.trace
                   if row["task"].endswith(":init"))
    assert init_end <= event(0, "c1s0:load")["start_ms"]
    for chunk in (0, 1):
        assert event(chunk, "c1s0:store")["end_ms"] <= event(chunk, "deltas0:load")["start_ms"]
        assert event(chunk, "deltas0:store")["end_ms"] <= event(chunk, "c2s0:load")["start_ms"]
        assert event(chunk, "c2s0:store")["end_ms"] <= event(chunk, "states0:load")["start_ms"]
        for phase in ("c1s0", "c2s0"):
            assert event(chunk, phase + ":l0")["end_ms"] <= event(
                chunk, phase + ":compute")["start_ms"]
    assert event(0, "states0:store")["end_ms"] <= event(1, "c1s0:load")["start_ms"]


def test_short_slot_refills_before_other_stream_finishes_and_h0_is_global():
    lengths = (1, 128, 1, 1, 1, 1)
    groups = kda_state.head_ownership(lengths, 1, 2)
    queues = kda_state.slot_queues(groups[0])
    actual = []
    for iteration in range(3):
        actual.append(tuple((entry[0].sequence, entry[1])
                            for queue in queues
                            if (entry := queue.at(iteration)) is not None))
    assert actual == [((0, 0), (1, 0)), ((4, 0), (1, 1)), ((5, 0),)]
    engine = execute(params(lengths=lengths, heads=1),
                     hardware_case(units=2), trace_limit=10000)
    init_end = max(row["end_ms"] for row in engine.trace
                   if row["task"].endswith(":init"))
    c1_rows = [row for row in engine.trace if row["task"].endswith(":c1s0:load")
               or row["task"].endswith(":c1s1:load")]
    assert min(row["start_ms"] for row in c1_rows) >= init_end
    core0 = [row for row in engine.trace if ":scan:program:group0:" in row["task"]]
    next_short = next(row for row in core0
                      if row["task"].endswith(":tile1:c1s0:load"))
    long_final = next(row for row in core0
                      if row["task"].endswith(":tile1:states1:store"))
    assert next_short["start_ms"] < long_final["end_ms"]
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_long_chunk_streams_keep_live_state_bounded_and_release_everything():
    short = execute(params(lengths=(64 * 8,), heads=1),
                    hardware_case(units=1))
    long = execute(params(lengths=(64 * 32,), heads=1),
                   hardware_case(units=1))
    for engine in (short, long):
        summary = engine.summary()
        assert summary["live_tasks"] == 0
        assert summary["live_programs"] == 0
        assert not any(summary["live_allocated"].values())
    assert short.summary()["peak_live_tasks"] == long.summary()["peak_live_tasks"]
    assert long.completed_count > 3 * short.completed_count


# ---------------------------------------------------------------------------
# Ordinary model frontend and graph selection.
# ---------------------------------------------------------------------------

def frontend_model(value):
    model = load_model("kimi-k3")
    layer = replace(model.layer_plans[0], linear_heads=2,
                    linear_key_heads=2, linear_value_heads=2,
                    intermediate_size=512)
    return replace(model, layers=1, layer_plans=(layer,), name=value,
                   hidden_size=256, num_heads=2, num_kv_heads=2, head_dim=128,
                   intermediate_size=512, vocab_size=256)


def frontend_case(model_name="kimi-k3"):
    model = frontend_model(model_name)
    hardware = Hardware.from_mapping({
        "name": "kda-frontend", "num_devices": 1,
        "device": {"memory_bytes": 64 * 1024 ** 3, "compute_units": 4,
                   "compute_peaks": {"bf16": 1e14, "fp32": 2.5e13},
                   "rates": {"memory": 1e12, "vector": 2e12, "mte2": 1e12,
                             "mte3": 1e12, "mte1": 1e12, "fixpipe": 1e12},
                   "resource_units": {"vector": 8}},
    })
    workload = WorkloadSpec.from_mapping({"requests": [
        {"id": "r", "prompt_tokens": 65, "output_tokens": 1}]})
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": IMPLEMENTATION}}})
    return Case(hardware, model, workload, runtime)


def test_ordinary_frontend_selects_one_structural_source_launch():
    case = frontend_case()
    ops = target_operators(case, phase="prefill", tokens=65, sequences=2)
    scans = [op for op in ops if op.recipe == kda_state.RECIPE]
    assert len(scans) == 1
    scan = scans[0]
    meta = dict(scan.metadata)
    assert scan.source_boundary == "kernel"
    assert scan.requests == ("analysis0", "analysis1")
    assert meta["lengths"] == (65, 65)
    assert meta["total_tokens"] == 130
    assert len(meta["chunks_per_sequence"]) == 2
    assert {op.family for op in ops if op.plan_required} == {
        "recurrent_scan"}
    row = operator_costs(case, (scan,))["operators"][0]
    assert row["program"] == "OperatorPlan"
    assert row["duration_ms"] > 0
    assert row["live_tasks"] == 0
    graph = graph_costs(case, target_graphs(
        case, phase="prefill", tokens=65, sequences=2, scope="model"), isolated=False)
    selected = [item for item in graph["timeline"] if item["recipe"] == kda_state.RECIPE]
    assert len(selected) == 1
    outputs = [item for item in graph["timeline"] if item["family"] == "recurrent_output"]
    assert len(outputs) == 2
    assert all(selected[0]["id"] in item["predecessors"] for item in outputs)
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0


def test_graph_and_isolation_are_model_label_independent():
    first = frontend_case("kimi-k3")
    second = frontend_case("label-only-control")
    one = next(op for op in target_operators(first, phase="prefill", tokens=65)
               if op.recipe == kda_state.RECIPE)
    two = next(op for op in target_operators(second, phase="prefill", tokens=65)
               if op.recipe == kda_state.RECIPE)
    assert physical_signature(one) == physical_signature(two)
    first_graph = graph_costs(first, target_graphs(
        first, phase="prefill", tokens=65, scope="model"))
    second_graph = graph_costs(second, target_graphs(
        second, phase="prefill", tokens=65, scope="model"))
    assert first_graph["duration_ms"] == pytest.approx(
        second_graph["duration_ms"], rel=1e-12)
    first_rows = [row for row in first_graph["isolated"]["operators"]
                  if row["operator"]["recipe"] == kda_state.RECIPE]
    assert len(first_rows) == 1
    assert first_rows[0]["duration_ms"] > 0
    assert first_rows[0]["live_tasks"] == 0
    assert not any(first_rows[0]["remaining_buffers"].values())


def test_generic_and_old_explicit_kda_selection_are_unchanged():
    case = frontend_case()
    generic = replace(case, runtime=replace(case.runtime, kernels={}))
    old = replace(case, runtime=replace(case.runtime, kernels={"lowerings": {
        "kda": OLD_IMPLEMENTATION}}))
    generic_scan = next(op for op in target_operators(
        generic, phase="prefill", tokens=65) if op.family == "recurrent_scan")
    old_scan = next(op for op in target_operators(
        old, phase="prefill", tokens=65) if op.family == "recurrent_scan")
    assert generic_scan.recipe == "recurrent_scan0"
    assert old_scan.recipe == "recurrent_scan0"
    assert generic_scan.source_boundary == old_scan.source_boundary == ""
    assert physical_signature(generic_scan) == physical_signature(old_scan)


def test_mixed_prefill_decode_binds_only_the_prefill_scan():
    model = load_model("kimi-k3")
    layer = model.layer_plans[0]
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": IMPLEMENTATION}}})
    views = (RequestView("p", 65, 0, False, "prefill"),
             RequestView("d", 1, 65, False, "decode"))
    result = linear(
        prefix="mixed", plan=layer, views=views, stage=0, deps="input",
        model=model, parallel=ParallelSpec(), quant=model.quant,
        mode="prefill", padded_tokens=66,
        policy=AttentionPolicy(recurrent_chunk=64), runtime=runtime)
    source = [op for op in result.ops if op.recipe == kda_state.RECIPE]
    generic = [op for op in result.ops if op.family == "recurrent_scan"
               and op.recipe != kda_state.RECIPE]
    assert len(source) == 1 and source[0].requests == ("p",)
    assert dict(source[0].metadata)["lengths"] == (65,)
    assert len(generic) == 1 and generic[0].recipe == "recurrent_scan1"
    assert generic[0].source_boundary == ""

    decode_only = linear(
        prefix="decode", plan=layer,
        views=(RequestView("d", 1, 65, False, "decode"),), stage=0,
        deps="input", model=model, parallel=ParallelSpec(), quant=model.quant,
        mode="decode", padded_tokens=1,
        policy=AttentionPolicy(recurrent_chunk=64), runtime=runtime)
    assert not any(op.recipe == kda_state.RECIPE for op in decode_only.ops)


@pytest.mark.parametrize("change,error", [
    ({"activation_dtype": "fp16"}, "kda_state_source_requires_bf16_inputs"),
    ({"linear_key_heads": 1}, "kda_state_source_requires_equal_key_value_heads"),
    ({"linear_key_head_dim": 64}, "kda_state_source_requires_dk_dv_128"),
])
def test_unsupported_source_domain_is_a_visible_error(change, error):
    from mindie_sim.schema.common import ModelingError

    model = load_model("kimi-k3")
    fields = {"linear_key_heads": 2, "linear_value_heads": 2,
              "linear_heads": 2}
    fields.update({k: v for k, v in change.items()
                   if k.startswith("linear_")})
    layer = replace(model.layer_plans[0], **fields)
    model = replace(model, layers=1, layer_plans=(layer,), **{
        k: v for k, v in change.items() if not k.startswith("linear_")})
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": IMPLEMENTATION}}})
    with pytest.raises(ModelingError, match=error):
        linear(
            prefix="bad", plan=layer,
            views=(RequestView("r", 64, 0, False, "prefill"),), stage=0,
            deps="input", model=model, parallel=ParallelSpec(),
            quant=model.quant, mode="prefill", padded_tokens=64,
            policy=AttentionPolicy(recurrent_chunk=64), runtime=runtime)


def test_wrong_chunk_and_pcp_are_rejected_by_the_source_hook():
    from mindie_sim.schema.common import ModelingError

    model = load_model("kimi-k3")
    layer = model.layer_plans[0]
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": IMPLEMENTATION}}})
    with pytest.raises(ModelingError, match="kda_state_requires_chunk_size:64"):
        linear(prefix="chunk", plan=layer,
               views=(RequestView("r", 64, 0, False, "prefill"),), stage=0,
               deps="input", model=model, parallel=ParallelSpec(),
               quant=model.quant, mode="prefill", padded_tokens=64,
               policy=AttentionPolicy(recurrent_chunk=32), runtime=runtime)
    with pytest.raises(ModelingError, match="kda_state_requires_pcp1"):
        linear(prefix="pcp", plan=layer,
               views=(RequestView("r", 64, 0, False, "prefill"),), stage=0,
               deps="input", model=model, parallel=ParallelSpec(pcp=2),
               quant=model.quant, mode="prefill", padded_tokens=64,
               policy=AttentionPolicy(recurrent_chunk=64), runtime=runtime)


def test_state_and_snapshot_addresses_are_disjoint_per_stream():
    value = params(lengths=(65, 1), heads=2)
    op = kda_state.make_op(value)
    meta = dict(op.metadata)
    total_chunks = meta["total_chunks"]
    prefixes = [0]
    for count in meta["chunks_per_sequence"]:
        prefixes.append(prefixes[-1] + count)
    states = [
        (sequence * 2 + head) * K * V
        for sequence in range(2) for head in range(2)
    ]
    assert len(set(states)) == 4
    assert sorted(states) == list(range(0, 4 * K * V, K * V))
    snapshots = sorted(
        kda_state.h_snapshot_offset(head, total_chunks, prefixes[sequence],
                                    chunk, 0, 0)
        for sequence in range(2) for head in range(2)
        for chunk in range(meta["chunks_per_sequence"][sequence])
    )
    assert len(snapshots) == total_chunks * 2
    assert len(set(snapshots)) == len(snapshots)
    for begin, end in zip(snapshots, snapshots[1:]):
        assert begin + K * V <= end
    assert kda_state.state_offset(1, 1, 0, 0, 2) == states[-1]
    assert kda_state.state_offset(1, 1, 1, 0, 2) == states[-1] + V


@pytest.mark.parametrize("override,error", [
    ({"state_dtype": "bf16"}, "kda_state_requires_fp32_state"),
    ({"gate_dtype": "bf16"}, "kda_state_requires_fp32_gates"),
    ({"use_initial_state": False}, "kda_state_requires_flag:use_initial_state"),
    ({"output_final_state": False},
     "kda_state_requires_flag:output_final_state"),
    ({"state_v_first": False}, "kda_state_requires_flag:state_v_first"),
])
def test_unsupported_source_flags_are_rejected(override, error):
    from mindie_sim.schema.common import ModelingError

    with pytest.raises(ModelingError, match=error):
        kda_state.make_op(params(**override))
