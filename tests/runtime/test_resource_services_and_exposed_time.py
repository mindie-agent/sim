"""Shared-capacity services, physical traffic conservation, and exposed time.

[Role]
  Pin the numerical resource laws behind the runtime: declared HBM service
  curves and shared-capacity scheduling, GM cacheline supply, matrix hierarchy
  reuse (shared HBM fill vs private MTE2 refill), M2 physical traffic
  conservation, and four-clock max-plus exposed-time attribution.
[Invariants]
  * A curve extrapolates only through declared physics, never a latency floor;
    a fixed owner cannot borrow an idle sibling's bandwidth.
  * Work is conserved per physical resource and per clock bucket; percentages
    use ``math.fsum`` with a deterministic residual assignment.
  * Every expected time/byte count is hand-derived from declared rates, knots
    or datasheet supply; no prediction is read back as its own expectation.
[Timeouts & Locks] Pure in-memory Engine/observer runs; no threads or locks.
[Side-Effect Policy] No filesystem, environment or network writes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from mindie_sim import operator_costs, predict
from mindie_sim.backends.ascend import materialize, profiles
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.observations.service_fitting import fit_service_curve
from mindie_sim.prediction.batch_prefix_attention import apply_attention_prefix_reads
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction import (
    exposed_breakdown as eb, exposed_program as exposed, memory_traffic as traffic,
    resource_attribution as attribution, roofline_graph as roof)
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.matrix_traffic import (
    MatrixGeometry, MatrixSchedule, plan_matrix_traffic)
from mindie_sim.prediction.mechanism import (
    Geometry as PlanGeometry, OperatorPlan, PlannedRecipe, Work)
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators.kda_recurrent import Params, summarize_work
from mindie_sim.prediction.physical_recipe_base import Phase, _phase_amounts
from mindie_sim.prediction.program import KernelProgram, RecipeBook, RecipeContext, Stage
from mindie_sim.prediction.runtime import exposed_intervals as intervals
from mindie_sim.prediction.runtime.exposed_observer import ExposedObserver
from mindie_sim.prediction.services import ResourceService, ServiceCurve
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.physical_mechanisms import MemoryHierarchySpec
from mindie_sim.schema.workload import IterationPlan, RequestView

from _support.hardware import hardware_case
from _support.kernel_invocation import copy_op, curve

pytestmark = pytest.mark.tier0

M2_SKU = "Ascend950DT_9582"
CACHELINE_SKU, CACHELINE_ROLE = "Ascend950PR_958b", "gm_hbm_cacheline_128b"
REFILL_MODEL = "deepseek-v3.2"
CACHELINE_RATE = 488281250.0
SHARED_PREFIX_ROWS, QUERY_ROWS, SHARED_ROW_BYTES = 4096, 1024, 1024
QO_BYTES = 16 * 1024 * 1024

def execute(ops, recipe, raw=None, release_ms=0):
    parsed = parse_case(raw or hardware_case())
    execution = Execution(parsed, CostModel(parsed),
                          recipes=RecipeBook({"copy": recipe}), trace_limit=200)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    step = execution.add(IterationWork(tuple(ops), plan, 1, 1, 1, None, "prefill",
                                       "resource_service_control"), 0, release_ms)
    execution.engine.advance(until=(step.completion,))
    return step, execution

def copy_row(operations, recipe, raw=None, **kwargs):
    return operator_costs(raw or hardware_case(), tuple(operations),
                          recipes=RecipeBook({"copy": recipe}), **kwargs)["operators"][0]

def packet():
    samples = ((1000, 2, "a", "train"), (1000, 2, "b", "train"),
               (3000, 4, "a", "train"), (3000, 4, "b", "train"),
               (2000, 3.1, "c", "heldout"), (5000, 6.2, "c", "heldout"))
    return {**curve().to_dict(), "samples": [
        {"work": w, "duration_ms": d, "run_id": r, "repeat_index": 0,
         "source": f"oracle:{r}:{w}", "split": s} for w, d, r, s in samples]}

def cacheline_run(lines, *, owner=0):
    device = Hardware.from_mapping({"backend": "ascend", "sku": M2_SKU,
                                    "num_devices": 1, "device": {}}).dies[0]
    rates = {name: float(rate) / 1000.0 for name, rate in device.rates}
    units = {name: int(device.units(name)) for name in rates}
    engine = Engine([Resource(name, rate, units=units[name]) for name, rate in rates.items()])
    context = RecipeContext(None, {}, {name: name for name in rates}, rates, 64,
                            resource_units=units)
    work = Work(resources={CACHELINE_ROLE: lines})
    done = KernelProgram((Stage("line", context.bind(work),
                                placement=context.owned(owner, 64, work.resources)),)).start(
        engine, "line", engine.submit("ready", release=0))
    engine.advance(until=(done,))
    return done.time * 1e6, engine.serviced, device

def matrix_execution(m, n, k, *, tile=(1, 1, 1), reuse=True, mte2_rate=100,
                     hbm_rate=100, quant="bf16", cores=2):
    operation = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                        m=m, n=n, k=k,
                        quant=QuantSpec("bf16") if quant == "bf16" else quant_preset(quant),
                        category="dense_mlp", phase="prefill")
    rates = {"cube": 100, "hbm": hbm_rate, "mte2": mte2_rate, "mte3": 100}
    if operation.vector_ops:
        rates["vector"] = 100
    names = {name: name for name in rates}
    units = {"cube": cores, "hbm": 1, "mte2": cores, "mte3": cores, "vector": cores}
    engine = Engine([Resource(name, rate, units=units[name]) for name, rate in rates.items()])
    context = RecipeContext(
        operation, {"cube": operation.flops,
                    **({"vector": operation.vector_ops} if operation.vector_ops else {})},
        names, rates, cores, resource_units=units)
    done = TiledMatmul(*tile, slots=2, reuse_inputs=reuse, reuse_weights=reuse).lower(
        context).start(engine, "matrix", engine.submit("ready"))
    engine.advance(until=(done,))
    return engine, done.time

def mechanism_op(family="elementwise", recipe="residual_add", shape=(8, 7168)):
    return OpSpec("op", family, "decode", 0, 0, ("r",), shape, shape, shape, shape,
                  recipe=recipe, dtype="bf16")

def exposed_table():
    roofline = eb.roofline_clocks
    return (
        (roofline(cube_ms=10, vector_ms=0, memory_ms=4, communication_ms=0), 10, (10, 0, 0, 0)),
        (roofline(cube_ms=4, vector_ms=0, memory_ms=10, communication_ms=0), 10, (4, 0, 6, 0)),
        (eb.advance_clocks((eb.cube_clocks(4),), eb.vector_clocks(3)), 7, (4, 3, 0, 0)),
        (eb.join_clocks((eb.cube_clocks(4), eb.vector_clocks(7))), 7, (4, 3, 0, 0)),
        (eb.join_clocks((eb.communication_clocks(3),
                      roofline(cube_ms=5, vector_ms=0, memory_ms=7, communication_ms=0))),
         7, (5, 0, 2, 0)),
        (eb.advance_clocks((eb.control_clocks(2),),
                        roofline(cube_ms=5, vector_ms=0, memory_ms=3, communication_ms=0)),
         7, (5, 2, 0, 0)),
    )

def test_fit_excludes_heldout_weights_runs_equally_and_rejects_conflicts():
    source = packet()
    fitted = fit_service_curve(source)
    changed = deepcopy(source)
    for sample in changed["samples"]:
        if sample["split"] == "heldout":
            sample["duration_ms"] *= 100
    second = fit_service_curve(changed)
    assert all((fitted["model"] == second["model"] and fitted["model_id"] == second["model_id"], fitted["model"]["startup_ms"] == 1))
    assert [row["predicted_ms"] for row in fitted["heldout"]] == [3, None]
    assert fitted["heldout"][0]["relative_error"] != second["heldout"][0]["relative_error"]
    assert ServiceCurve.from_mapping(
        json.loads(json.dumps(fitted["model"]))).model_id == fitted["model_id"]
    source["samples"] = [
        {"work": work, "duration_ms": duration, "run_id": run, "repeat_index": repeat,
         "source": f"oracle:{run}:{work}:{repeat}", "split": "train"}
        for work, base in ((1000, 1), (3000, 3))
        for run, duration, count in (("a", base + 1, 1), ("b", base + 3, 19))
        for repeat in range(count)]
    weighted = fit_service_curve(source)
    # Run means (2, 4) and (4, 6): a 19-repeat run cannot outweigh its peer.
    assert all((weighted["model"]["knots"] == ((1000, 3), (3000, 5)), weighted["model"]["startup_ms"] == 2))
    source["samples"] += deepcopy(source["samples"])
    assert fit_service_curve(source) == weighted
    source["samples"][-1]["split"] = "heldout"
    with pytest.raises(ValueError, match="conflicting_primitive_repeat_or_split"):
        fit_service_curve(source)
    negative = fit_service_curve({**packet(), "samples": [
        {"work": work, "duration_ms": duration, "run_id": "n", "repeat_index": 0,
         "source": "oracle", "split": "train"}
        for work, duration in ((1000, 1), (3000, 5))]})
    assert negative["model"]["startup_ms"] == 0

def test_unseen_work_interpolates_and_serializes_through_public_paths():
    model = curve(startup_ms=1, knots=((1000, 3), (3000, 9), (5000, 17)))
    operation, recipe = copy_op(4000), ResourceService(model)
    row = copy_row((operation,), recipe, input_ready_ms=7)
    step, execution = execute((operation,), recipe, release_ms=7)
    # 4000 is halfway between the 9 ms and 17 ms knots; readiness is 7.
    assert all((row["duration_ms"] == 13 and step.completion.time == 20, row["service_before_input"] == {}))
    assert all((row["declared_service"] == row["serviced"] == {"memory@r0:s0": 4000}, execution.work["hbm_bytes"] == 4000))
    assert row["basis"]["service_models"][0]["queries"] == {"interpolated": 1}
    assert all(event["start_ms"] >= 7 for event in execution.engine.trace if event["demand"])
    raw = hardware_case()
    raw["workload"]["requests"][0].update(prompt_tokens=2, output_tokens=3)
    raw["runtime"]["kernels"] = {"recipes": {"copy": {
        "kind": "resource_service", "model": curve().to_dict(),
        "outside_domain": "analytic"}}}
    def lower(plan, model, runtime, machine):
        return IterationWork((copy_op(plan.valid_tokens * 1000),), plan, 1, plan.valid_tokens,
                             plan.valid_tokens, None, "prefill", "service_control")
    result = predict(json.loads(json.dumps(raw)), lower=lower)
    assert all((result["requests"][0]["ttft_ms"] == 3, result["requests"][0]["finish_ms"] == 7))
    assert all((result["execution"]["work"]["hbm_bytes"] == 4000, result["execution"]["serviced"]["memory@r0:s0"] == 4000))

def test_shared_and_capped_capacity_release_only_unused_bandwidth():
    left, right = copy_op(stream="left"), copy_op(op_id="right", stream="right")
    parallel, execution = execute((left, right), ResourceService(curve()))
    serial, _ = execute((left, replace(right, deps=(left.op_id,))), ResourceService(curve()))
    # Both start for 1 ms; 4000 B on one 1000 B/ms pipe need 4 ms more.
    assert all((parallel.completion.time == 5 and serial.completion.time == 6, execution.engine.serviced["memory@r0:s0"] == 4000))
    assert execution.cost.summary()["service_models"][0]["queries"] == {"interpolated": 2}
    slow = copy_op(1000, stream="slow")
    fast = copy_op(3000, op_id="fast", stream="fast", recipe="analytic")
    step, execution = execute((slow, fast), ResourceService(
        curve(startup_ms=0, knots=((1000, 4), (3000, 12)))))
    # Slow producer supplies 250 B/ms; the remaining 750 B/ms serves 3000 B.
    ends = {}
    for event in execution.engine.trace:
        if event["demand"]:
            name = event["task"].split(":")[2]
            ends[name] = max(ends.get(name, 0.0), event["end_ms"])
    assert all((step.completion.time == 4 and ends == {"copy": 4, "fast": 4}, execution.engine.serviced["memory@r0:s0"] == 4000))

def test_curve_ownership_domain_identity_and_unit_conditions():
    for work in (500, 4000):
        row = copy_row((copy_op(work),), ResourceService(curve()))
        assert row["duration_ms"] == work / 1000
        assert row["basis"]["service_models"][0]["queries"] == {"analytic_outside_domain": 1}
        with pytest.raises(ValueError, match="service_work_outside_domain"):
            execute((copy_op(work),), ResourceService(curve(), outside_domain="error"))
    for changes in ({"dtype": "fp32"}, {"layout": "strided"},
                    {"metadata": (("kernel", "another_kernel"),)},
                    {"metadata": (("implementation", {"block_threads": 128}),)},
                    {"metadata": (("service_axes", {"stride_bytes": 128}),)}):
        with pytest.raises(ValueError, match="service_identity_mismatch"):
            execute((copy_op(**changes),), ResourceService(curve()))
    model = curve(active_units=1, startup_ms=.5, knots=((1000, 1.5), (3000, 3.5)))
    row = copy_row((copy_op(),), ResourceService(model, units=(0,)))
    # One of four owners supplies 250 B/ms; 2000 B need 8 ms plus .5 ms startup.
    assert row["duration_ms"] == 8.5 and row["serviced"] == {"memory@r0:s0": 2000}
    assert row["basis"]["service_models"][0]["capacity_limited_queries"] == 1
    with pytest.raises(ValueError, match="service_active_units_mismatch"):
        execute((copy_op(),), ResourceService(model))
    raw = hardware_case()
    raw["model"]["name"] = "unseen-model"
    step, _ = execute((copy_op(op_id="new", layer=17, requests=("new-request",)),),
                      ResourceService(curve()), raw)
    assert step.completion.time == 3
    raw["runtime"]["vllm_version"] = "different-runtime"
    with pytest.raises(ValueError, match="service_identity_mismatch"):
        execute((copy_op(),), ResourceService(curve()), raw)
    with pytest.raises(ValueError, match="service_identity_mismatch"):
        copy_row((copy_op(),), ResourceService(curve()), replay_scope="full")
    operation = copy_op(metadata=(("service_axes", {"tile": (16, 32)}),))
    step, _ = execute((operation,), ResourceService(
        json.loads(json.dumps(curve(op=operation).to_dict()))))
    assert step.completion.time == 3
    with pytest.raises(ValueError, match="service_work_unit_mismatch"):
        execute((copy_op(),), ResourceService(curve(work_unit="ops")))
    with pytest.raises(ValueError, match="service_curve_requires_one_explicit_resource_phase"):
        execute((copy_op(vector_ops=200),), ResourceService(curve()))
    raw = hardware_case()
    raw["machine"]["device"]["service_modes"] = {"alias": {"resource": "hbm", "rate": 1e6}}
    with pytest.raises(ValueError, match="service_curve_requires_one_explicit_resource_phase"):
        execute((copy_op(metadata=(("resource_work", {"alias": 1000}),)),),
                ResourceService(curve()), raw)

def test_typed_matrix_and_byte_mode_curves_keep_their_native_work():
    raw = hardware_case()
    raw["machine"]["device"].update(
        service_modes={"strided": {"resource": "vector", "rate": 5e5}},
        work_units={"strided": "bytes"})
    case = parse_case(raw)
    operation = copy_op(hbm_bytes=0, flops=2000, matrix_flops=(("bf16", 2000),))
    matrix = curve(case, operation, resource="cube", work_unit="flops")
    row = copy_row((operation,), ResourceService(matrix), case)
    assert all((row["duration_ms"] == 3, row["operator"]["matrix_flops"] == {"bf16": 2000}))
    assert row["serviced"] == {"matrix@r0:s0": 1}
    # Equal precision peaks do not turn two instruction modes into one primitive.
    mixed = replace(operation, matrix_flops=(("bf16", 1000), ("fp32", 1000)))
    with pytest.raises(ValueError, match="service_curve_requires_one_compute_precision"):
        copy_row((mixed,), ResourceService(matrix), case)
    operation = copy_op(hbm_bytes=0, metadata=(("resource_work", {"strided": 2000}),))
    mode = curve(case, operation, resource="strided", work_unit="bytes", startup_ms=1,
                 knots=((1000, 5), (3000, 13)))
    row = copy_row((operation,), ResourceService(mode), case)
    assert all((row["duration_ms"] == 9, row["operator"]["metadata"]["resource_work"] == {"strided": 2000}))
    assert row["declared_service"] == row["serviced"] == {"vector@r0:s0": 4000}

def test_phase_calibration_composes_with_analytic_work_and_source_binding():
    operation = copy_op(source_boundary="kernel")
    identity = {**curve().identity, "kernel": "load"}
    service = ResourceService(curve(identity=identity, boundary="resource_phase"))
    load, compute, store = (Work(resources={"hbm": 1000}), Work(resources={"vector": 500}),
                            Work(resources={"hbm": 1000}))
    def build(operation, geometry):
        def program(context):
            return KernelProgram((
                Stage("load", program=service.lower(context.for_phase(load, kernel="load"))),
                Stage("compute", context.bind(compute), after=("load",)),
                Stage("store", context.bind(store), after=("compute",))))
        return OperatorPlan(load + compute + store, program)
    result = verify_operator(hardware_case(), operation, build, input_ready_ms=7)
    assert result["status"] == "passed" and result["result"]["duration_ms"] == 3.5
    assert result["result"]["serviced"] == {"memory@r0:s0": 2000, "vector@r0:s0": 500}
    step, _ = execute((operation,), PlannedRecipe(build), release_ms=7)
    assert step.completion.time == 10.5
    with pytest.raises(ValueError, match="source_plan_binding_required"):
        execute((operation,), ResourceService(curve()))
    parsed = parse_case(hardware_case())
    execution = Execution(parsed, CostModel(parsed))
    phase = execution._context(copy_op(), 0, "none").for_phase(load, kernel="load")
    with pytest.raises(ValueError, match="service_boundary_mismatch"):
        ResourceService(curve(identity=identity)).lower(phase)
    context = replace(execution._context(build(operation, None).work.apply(operation), 0, "none"),
                      setup_ms=.25, minimum_ms=7)
    done = build(operation, None).lower(context).start(
        execution.engine, "wrapped", execution.engine.submit("ready"))
    execution.engine.advance(until=(done,))
    assert done.time == 7

@pytest.mark.parametrize("lines", (1, 1025))
def test_gm_cacheline_supply_matches_the_declared_hbm_share(lines):
    elapsed, serviced, device = cacheline_run(lines)
    # One 128 B line at 488.28125 lines/us costs exactly 1e9/rate us.
    assert elapsed == pytest.approx(lines * 1e9 / CACHELINE_RATE, rel=1e-12)
    assert serviced.get(CACHELINE_ROLE) == lines and sum(serviced.values()) == lines
    assert float(dict(device.rates)[CACHELINE_ROLE]) == 31250000000.0
    assert device.units(CACHELINE_ROLE) == 64
    assert dict(device.work_units)[CACHELINE_ROLE] == "cachelines"
    for unpriced in ("scalar_gm_access", "scalar_alu_issue"):
        assert unpriced not in dict(device.rates)
    # One equal per-AIV HBM share: 1025 lines take 5.9977 us on one AIV.
    profile = profiles()[CACHELINE_SKU]["device"]
    service = profile["compute_arch"]["instruction_services"][CACHELINE_ROLE]
    owners, hbm = profile["resource_units"]["vector"], profile["rates"]["hbm"]
    assert service["per_owner_rate"] == pytest.approx(hbm / (128 * owners))
    assert 1025 / service["per_owner_rate"] * 1e9 == pytest.approx(5997.7, abs=.2)

def test_shared_hbm_hits_preserve_private_copies_without_hiding_a_bottleneck():
    shared, _ = matrix_execution(1, 2, 1)
    uncached, _ = matrix_execution(1, 2, 1, reuse=False)
    # BF16 A is 2 B and B is 4 B. Each of two owners needs its own 2 B A and
    # 2 B B, even when the shared cache fetches A from HBM once.
    assert all((shared.serviced == {"cube": 4, "hbm": 10, "mte2": 8, "mte3": 4}, uncached.serviced == {"cube": 4, "hbm": 12, "mte2": 8, "mte3": 4}))
    shared, shared_ms = matrix_execution(1, 2, 1, mte2_rate=2)
    uncached, uncached_ms = matrix_execution(1, 2, 1, reuse=False, mte2_rate=2)
    # Four local bytes per owner at 1 B/ms, then 4 FLOP/100 and 4 B/100.
    assert all((shared_ms == uncached_ms == pytest.approx(4.08), shared.serviced["hbm"] < uncached.serviced["hbm"]))
    _, limited_shared = matrix_execution(1, 2, 1, hbm_rate=2)
    _, limited_uncached = matrix_execution(1, 2, 1, reuse=False, hbm_rate=2)
    assert limited_shared == pytest.approx(5.04) and limited_uncached == pytest.approx(6.04)

def test_tail_tiles_and_packed_scales_stay_in_their_memory_boundaries():
    shared, _ = matrix_execution(3, 5, 7, tile=(2, 3, 4))
    uncached, _ = matrix_execution(3, 5, 7, tile=(2, 3, 4), reuse=False)
    # Two M tiles and two N tiles include both tails; K panels add no whole
    # operand duplicate in this declared tile model.
    activation, weight, output = 3 * 7 * 2, 5 * 7 * 2, 3 * 5 * 2
    local_reads = 2 * activation + 2 * weight
    assert [shared.serviced["mte2"], uncached.serviced["mte2"]] == pytest.approx(
        [local_reads] * 2)
    assert shared.serviced["hbm"] == pytest.approx(activation + weight + output)
    assert uncached.serviced["hbm"] == pytest.approx(local_reads + output)
    assert [shared.serviced["cube"], uncached.serviced["cube"]] == pytest.approx(
        [2 * 3 * 5 * 7] * 2)
    shared, _ = matrix_execution(2, 4, 3, tile=(1, 2, 3), quant="w8a8_int8")
    uncached, _ = matrix_execution(2, 4, 3, tile=(1, 2, 3), reuse=False, quant="w8a8_int8")
    # INT8 A 6 B plus two FP32 activation scales 8 B; B 12 B plus a BF16
    # per-channel runtime weight scale 8 B; BF16 output 16 B.
    activation, packed_weight, output = 14, 20, 16
    assert all((shared.serviced["hbm"] == activation + packed_weight + output, shared.serviced["mte2"] == 2 * activation + 2 * packed_weight))
    assert all((uncached.serviced["hbm"] == 2 * activation + 2 * packed_weight + output, shared.serviced["mte2"] == uncached.serviced["mte2"]))
    assert shared.serviced["vector"] == uncached.serviced["vector"] == 4 * 2 * 4

def test_traffic_plans_conserve_shared_prefix_and_selected_rows():
    machine = Hardware.from_mapping(materialize(
        {"backend": "ascend", "sku": M2_SKU, "num_devices": 8, "ranks_per_node": 8}))
    die, pair = machine.dies[0], next(
        link for link in machine.links if "d2d_intra_node" in link.channel)
    # L2 128 MiB, 256x256 base tile, and 0.85 of the 56 TB/s per-pair service.
    assert (die.physical_mechanisms.memory_hierarchy.l2_capacity_bytes,
            (256, 256) in die.physical_mechanisms.matrix.base_tiles) == (134217728, True)
    assert (pair.semantics, pair.capacity_bytes_per_ms,
            pair.peak_capacity_bytes_per_ms) == ("per_pair", 47_500_000, 56_000_000)
    with pytest.raises(ModelingError, match="unmodeled_operator_mechanism"):
        RecipeBook().select(mechanism_op(family="unmodeled_test", recipe="unknown"))
    hierarchy, spec = MemoryHierarchySpec(134217728, 0.75, 128, True, {}), traffic.AttentionAccessSpec
    dense = traffic.plan_attention_traffic(
        spec(q_lengths=(128,), context_lengths=(4096,), unique_rows=4224,
             refill_rows=8192, row_bytes=576, phase="prefill"),
        hierarchy, traffic.AttentionSchedule(query_tile=128))
    assert (dense.hbm_read_bytes, dense.private_refill_bytes,
            dense.l2_hit_bytes) == (4224 * 576, 8192 * 576, (8192 - 4224) * 576)
    sparse = traffic.plan_attention_traffic(
        spec(q_lengths=(1,), context_lengths=(32768,), unique_rows=32768,
             refill_rows=2048, selected_rows=2048, row_bytes=576,
             phase="decode", topk=2048), hierarchy, traffic.AttentionSchedule())
    assert (sparse.hbm_read_bytes, sparse.residency_basis) == (
        2048 * 576, "unknown_dynamic_topk_per_query_gather")
    prefix = traffic.plan_index_key_traffic(
        traffic.IndexAccessSpec(query_rows=4, prefix_tokens=4096,
                                private_tokens=(128, 128, 128, 128), row_bytes=576),
        hierarchy, traffic.IndexSchedule(batch_tile=True))
    assert prefix.hbm_read_bytes == (4096 + 512) * 576 and prefix.l2_hit_bytes > 0
    plan = plan_matrix_traffic(
        mechanism_op(shape=(32, 32, 7168)), MatrixGeometry(32, 32, 7168),
        MatrixSchedule(32, 32, 256, 16, 16, 16, 28, 1, 1, 2, 2, "gm"), hierarchy)
    assert plan.hbm_read_bytes == 32 * 7168 * 2 + 7168 * 32 * 2
    assert plan.partial_bytes > 0 and plan.private_refill_bytes >= plan.hbm_read_bytes
    work = summarize_work(Params.parse({
        "lengths": (1,), "heads": 4, "key_heads": 2, "dk": 128, "dv": 128,
        "dtype": "bf16", "state_dtype": "fp32"}), PlanGeometry(
            32, {"vector": 64, "mte2_aiv": 64, "mte2_aic": 32, "mte3": 64},
            {"ub": 253952}, {"ub": 64}))
    assert work.resources["hbm"] > 0 and work.resources["vector"] > 0
    assert work.resources["mte2_aiv"] > 0

def test_critical_path_link_and_exposed_hbm_are_raw():
    graph = roof.PhysicalWorkGraph((
        roof.PhysicalWorkNode("load", hbm_ms=10.0),
        roof.PhysicalWorkNode("compute", cube_ms=20.0, predecessors=("load",))))
    result = roof.critical_path_roofline(
        graph, roof.RooflineTargets(cube=0.8, vector=0.8, hbm=0.8, link=0.8))
    assert all((math.isclose(result.critical_path_ideal_ms, 30.0), result.physical_ideal_ms >= 30.0))
    assert result.target_duration_ms >= result.physical_ideal_ms
    repeated = roof.graph_from_occurrences(tuple(
        {"id": f"copy{index}", "hbm_ms": 1.0,
         "predecessors": (() if index == 0 else (f"copy{index - 1}",))} for index in range(3)))
    total = roof.critical_path_roofline(
        repeated, roof.RooflineTargets(cube=1.0, vector=1.0, hbm=1.0, link=1.0))
    assert all((total.critical_path_ideal_ms == pytest.approx(3.0), total.resource_totals["hbm"] == pytest.approx(3.0)))
    utilization = roof.link_utilization({"link:a": 120.0}, {"link:a": 100.0}, 1.0)
    assert math.isclose(utilization.maximum, 1.2) and utilization.violations == ("link:a",)
    assert attribution.estimate_exposed_hbm(
        10.0, 8.0, 4.0, cube_cover_fraction=0.5, vector_cover_fraction=0.5) == 4.0
    serial = roof.graph_from_occurrences((
        {"id": "cube", "cube_ms": 1.0, "predecessors": ()},
        {"id": "memory", "hbm_ms": 1.0, "predecessors": ("cube",)}))
    attributed = attribution.attribute_stage_dag(serial, duration_ms=2.0)
    assert attributed["available"] is True
    assert attributed["buckets"]["cube"]["exposed_ms"] == pytest.approx(1.0)
    assert attributed["buckets"]["hbm"]["exposed_ms"] == pytest.approx(1.0)
    assert attributed["terminal_clocks"].all_ms == pytest.approx(2.0)

def test_shared_prefix_and_asymmetric_link_work_are_not_redistributed():
    views = tuple(SimpleNamespace(
        request_id=str(i), q_len=QUERY_ROWS, context=SHARED_PREFIX_ROWS,
        prefix_tokens=SHARED_PREFIX_ROWS, prefix_key="shared",
        kv_len=SHARED_PREFIX_ROWS + QUERY_ROWS, phase="prefill") for i in range(2))
    q_read = out_write = QO_BYTES // 2
    kv_precharged = 2 * (SHARED_PREFIX_ROWS + QUERY_ROWS) * SHARED_ROW_BYTES
    shape = (QUERY_ROWS, SHARED_PREFIX_ROWS + QUERY_ROWS, 1, 1)
    operation = OpSpec("attention", "attention", "prefill", 0, 0, ("r",), shape, shape,
                       shape, shape, hbm_bytes=kv_precharged + QO_BYTES,
                       mte2_bytes=kv_precharged + q_read, mte3_bytes=out_write,
                       recipe="flash_attention")
    updated = apply_attention_prefix_reads(
        operation, views=views, sweeps_by_view=(SHARED_PREFIX_ROWS + QUERY_ROWS,) * 2,
        shared_sweeps_by_view=(SHARED_PREFIX_ROWS,) * 2, row_bytes=SHARED_ROW_BYTES,
        layer_index=0, runtime=SimpleNamespace(prefix_cache_mode="shared",
                                               prefix_read_mode="auto"),
        kernel_reuse="tile", allow_recipe_rebind=False,
        kv_hbm_bytes=kv_precharged, kv_mte2_bytes=kv_precharged)
    cold_kv = (SHARED_PREFIX_ROWS + 2 * QUERY_ROWS) * SHARED_ROW_BYTES
    # Replacement unions the external fill but keeps query/output traffic and
    # every per-tile MTE2 refill.
    assert (updated.hbm_bytes, updated.hbm_bytes - cold_kv) == pytest.approx(
        (cold_kv + QO_BYTES, QO_BYTES))
    assert updated.mte3_bytes == pytest.approx(out_write)
    assert updated.mte2_bytes == pytest.approx(kv_precharged + q_read)
    context = SimpleNamespace(
        op=OpSpec("collective", "allreduce", "decode", 0, 0, (), (), (), (), ()),
        work={"link:a": 9_000_000.0, "link:b": 1_000_000.0},
        resource_names={"link": "link"})
    amounts = _phase_amounts(context, (Phase("first", {"link": 0.5}),
                                      Phase("second", {"link": 0.5})))
    # Each physical resource conserves its own work; neither link is rescaled.
    assert (amounts[0]["link:a"], amounts[1]["link:a"]) == pytest.approx(
        (4_500_000.0, 4_500_000.0))
    assert (amounts[0]["link:b"], amounts[1]["link:b"]) == pytest.approx((500_000.0, 500_000.0))
    assert (sum(row.get("link:a", 0.0) for row in amounts),
            sum(row.get("link:b", 0.0) for row in amounts)) == pytest.approx((9e6, 1e6))

def test_split_pipes_receive_private_refill_service():
    case = parse_case(compose_case(REFILL_MODEL, hardware=M2_SKU, tp=8, ep=1, dp=1, pp=1,
                                   layers=1, prompt_tokens=512, output_tokens=2,
                                   concurrency=1))
    operations = target_operators(case, phase="decode", tokens=1, context=4096,
                                  sequences=1, scope="body")
    attention = next(item for item in operations if item.family == "attention")
    traffic = dict(dict(attention.metadata)["attention_traffic_plan"])
    assert attention.mte2_bytes >= traffic["private_refill_bytes"]
    row = operator_costs(case, (attention,), input_ready_ms=7)["operators"][0]
    assert row["serviced"]["mte2_aic@r0:s0"] == pytest.approx(attention.mte2_bytes)
    norm = next(item for item in operations if item.family in {"rmsnorm", "add_rmsnorm"})
    norm_row = operator_costs(case, (norm,), input_ready_ms=7)["operators"][0]
    assert norm_row["serviced"].get("mte2_aiv@r0:s0", 0.0) > 0.0

@pytest.mark.parametrize("clocks,duration,expected", exposed_table())
def test_exposed_table_examples_conserve_four_clocks(clocks, duration, expected):
    values = eb.finish_breakdown(clocks, duration_ms=duration,
                              boundary="device_step")["exposed_breakdown_ms"]
    assert tuple(values[name] for name in
                 ("cube", "vector", "memory", "communication")) == pytest.approx(expected)

def test_zero_unavailable_and_invalid_clock_results_are_explicit():
    clocks = eb.roofline_clocks(cube_ms=3.0, vector_ms=5.0, memory_ms=7.0,
                               communication_ms=11.0)
    result = eb.finish_breakdown(clocks, duration_ms=11.0, boundary="device_step",
                                 control_ms=2.0, opaque_control_ms=1.0)
    values, percentages = result["exposed_breakdown_ms"], result["exposed_breakdown_pct"]
    assert math.fsum(values.values()) == pytest.approx(11.0)
    assert math.fsum(percentages.values()) == pytest.approx(
        100.0, abs=eb.ATTRIBUTION_PERCENT_TOLERANCE)
    assert all(value >= 0 for value in values.values())
    basis = result["exposed_breakdown_basis"]
    assert basis["vector_control_ms"] == 2.0 and basis["opaque_control_ms"] == 1.0
    assert basis["counterfactual_speedup"] is False
    zero = eb.finish_breakdown(eb.ExposedClocks(0, 0, 0, 0), duration_ms=0,
                            boundary="device_step")
    assert all((set(zero["exposed_breakdown_ms"].values()) == {0}, zero["exposed_breakdown_basis"]["available"] is True))
    unavailable = eb.unavailable_breakdown("operator_only", boundary="operator_sum")
    assert all((unavailable["exposed_breakdown_ms"] is None, unavailable["exposed_breakdown_basis"]["available"] is False))
    assert unavailable["exposed_breakdown_basis"]["reason"] == "operator_only"
    with pytest.raises(ModelingError, match="clock_mismatch"):
        eb.finish_breakdown(eb.cube_clocks(5), duration_ms=10, boundary="device_step")
    with pytest.raises(ModelingError, match="clock_order_invalid"):
        eb.ExposedClocks(1, 2, 0, 0)
    with pytest.raises(ModelingError, match="clock_invalid"):
        eb.ExposedClocks(-1, 0, 0, 0)
    clocks = eb.roofline_clocks(cube_ms=2, vector_ms=0, memory_ms=3, communication_ms=0)
    with pytest.raises(ModelingError, match="clock_mismatch"):
        eb.finish_breakdown(clocks, duration_ms=10, boundary="device_step")
    even = eb.ExposedClocks(3.0, 3.0, 3.0, 3.0)
    first = eb.finish_breakdown(even, duration_ms=3.0, boundary="device_step")
    second = eb.finish_breakdown(even, duration_ms=3.0, boundary="device_step")
    assert first == second and first["exposed_breakdown_pct"]["cube"] == pytest.approx(100.0)

def test_serial_parallel_and_control_program_attribution():
    observer = ExposedObserver()
    observer.begin_scope("serial", 0)
    serial = observer.finish_scope("serial", observer.program_scope("serial", 0, (
        exposed.StageDescriptor("cube", cube_ms=4), exposed.StageDescriptor("vector", vector_ms=3))))
    assert serial["exposed_breakdown_ms"] == {
        "cube": 4.0, "vector": 3.0, "memory": 0.0, "communication": 0.0}
    observer = ExposedObserver()
    observer.begin_scope("parallel", 0, inputs={"cube": None, "vector": None})
    D = exposed.StageDescriptor
    observer.on_submit("cube", declaration=D("cube", cube_ms=4))
    observer.on_submit("vector", declaration=D("vector", vector_ms=7))
    observer.on_resolve("cube", moment_ms=4)
    observer.on_resolve("vector", moment_ms=7)
    joined = observer.finish_scope("parallel", ("cube", "vector"))
    assert all((joined["exposed_breakdown_ms"]["cube"] == pytest.approx(4), joined["exposed_breakdown_ms"]["vector"] == pytest.approx(3)))
    observer = ExposedObserver()
    observer.begin_scope("control", 0)
    control = observer.finish_scope("control", observer.program_scope("control", 0, (
        exposed.StageDescriptor("control", control_ms=2, opaque_control_ms=1),
        exposed.StageDescriptor("coupled", cube_ms=5, memory_ms=3))))
    basis = control["exposed_breakdown_basis"]
    assert basis["vector_control_ms"] <= control["exposed_breakdown_ms"]["vector"]
    assert all((basis["opaque_control_ms"] <= control["exposed_breakdown_ms"]["vector"], basis["method"] == "ordered_critical_span_v1"))
    class Program:
        stages = tuple(SimpleNamespace(work=work, latency=0.0, after=after, coupled=False)
                       for work, after in (({"cube": 100.0}, ()), ({"hbm": 100.0}, ("c0",))))
    rates = {"cube": 100.0, "hbm": 50.0}
    assert [item.role for item in exposed.program_descriptors(Program(), rates)] == ["cube", "memory"]
    declarations = exposed.declarations_from_program(Program(), rates)
    assert all((declarations[0]["cube_ms"] == pytest.approx(1.0), declarations[1]["memory_ms"] == pytest.approx(2.0)))

def test_service_intervals_and_periodic_skips_share_one_clock_law():
    interval = intervals.interval_clocks(delivered_work={"hbm": 100.0, "cube": 100.0},
                               offered_capacities={"hbm": 25.0, "cube": 100.0})
    assert interval.clocks.cube_ms == pytest.approx(1.0)
    assert (interval.clocks.without_communication_ms
            - interval.clocks.compute_ms) == pytest.approx(3.0)
    blocked = intervals.interval_clocks(delivered_work={"hbm": 100.0}, offered_capacities={"hbm": 0.0})
    assert blocked.blockers == ("hbm",) and blocked.clocks.all_ms == 0
    assert intervals.residual_capacity({"hbm": 100.0}, {"hbm": 30.0}) == {"hbm": 70.0}
    assert all((intervals.service_role("link:fabric") == "communication", intervals.service_role("memory@r0") == "memory"))
    clocks, control, opaque = intervals.bind_interval(interval=blocked, control_ms=2.0, opaque_control_ms=1.0)
    assert all((clocks.compute_ms - clocks.cube_ms == pytest.approx(2.0), control == 2.0 and opaque == 1.0))
    with pytest.raises(ModelingError, match="control_invalid"):
        intervals.bind_interval(interval=interval, control_ms=-1.0)
    observer = ExposedObserver(max_events=8)
    observer.begin_scope("periodic", 0)
    for index in range(20):
        observer.on_submit(f"extra{index}", declaration=exposed.StageDescriptor("cube", cube_ms=1))
    observer.on_submit("stage0", declaration=exposed.StageDescriptor(
        "coupled", cube_ms=2, vector_ms=3, memory_ms=1))
    observer.on_periodic_skip(
        correspondence={"stage0": "stage0"}, repeat=3,
        clock_delta={"all_ms": 3, "without_communication_ms": 3, "compute_ms": 3,
                     "cube_ms": 2})
    result = observer.finish_scope("periodic", "stage0")
    assert all((len(observer.scope_events("periodic")) <= 8, result["exposed_breakdown_basis"]["periodic_skips"] == 3))
    assert sum(result["exposed_breakdown_ms"].values()) == pytest.approx(
        result["exposed_breakdown_basis"]["duration_ms"])
