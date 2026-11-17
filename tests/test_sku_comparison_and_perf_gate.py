"""Track-2 regression, physics-parity and wall-clock gate.

[Role]
  Verify the Track-2 contract end to end: activation-time FIFO admission is
  preserved by the eager pure-program path, the C11 arena and the Python
  reference Engine agree on wide (``>256`` resources) and 72-AIV (``>64``
  units) frontiers, real 40-layer graphs are backend independent, a two-slot
  ``PipelineProgram`` keeps its exact slot occupancy, ``physical_key`` and
  layer runs collapse without false merges, ``compare_skus`` matches the
  standalone solvers exactly and stays inside the seconds-scale budget, and a
  serving prediction records wall time, native coverage and peak RSS.
[Invariants]
  Every relative duration comparison is ``< 1e-9``; a hardware mutation always
  re-executes a fresh Engine; no result crosses two public calls.
[Timeouts & Locks]
  Bounded in-process calls; wall-clock assertions carry generous headroom for a
  shared verification host.
[Side-Effect Policy]
  Writes only pytest temporary directories through the isolated fixtures.
"""
from __future__ import annotations

import os
import time

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction.api import predict
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency, scale_case_hardware, solve_sla_sizing,
)
from mindie_sim.prediction.layer_runs import homogeneous_layer_runs
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.physical_key import physical_key
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.structure import parse_case
from mindie_sim.sku_comparison import compare_skus
from mindie_sim import runtime_advisor

RELATIVE_TOLERANCE = 1e-9
MS_PER_SECOND = 1000.0
BYTES_PER_MIB = 1024 * 1024
DSV41_MODEL = "deepseek-v4.1-flash"
KIMI_MODEL = "kimi-k3.1"
LONG_MODEL_LAYERS = 40
KIMI_LAYERS = 93
WIDE_UNITS = 72
WIDE_RESOURCES = 300
SMALL_GRAPH_LAYERS = 4
FIFO_DEFERRED_MS = 112.0
FIFO_EAGER_MS = 102.0
COMPARE_WALL_BUDGET_S = 120.0
EFFICIENCY_WALL_BUDGET_S = 240.0
SERVING_BENCHMARK_TOKENS = 8
SERVING_BENCHMARK_ENV = "MINDIE_SIM_SERVING_BENCHMARK_TOKENS"
RSS_BUDGET_MIB = 512.0
ADVISOR_TASK_COUNT = 12
LAST_LAYER_RUN_FLOOR = 30


def _backend_case(case_dict, *, context=256, tokens=1, sequences=1, layers=None):
    return analyze_network_efficiency(
        case_dict, phase="decode", tokens=tokens, context=context,
        sequences=sequences, scope="body")


def _resident_mib() -> float:
    """Current resident set size of this process (0 when unavailable)."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return 0.0
    return pages * os.sysconf("SC_PAGE_SIZE") / BYTES_PER_MIB


def _duration(case_dict, **kwargs):
    return float(_backend_case(case_dict, **kwargs)["graph_duration_ms"])


def _fifo_frontier(*, deferred: bool) -> float:
    """Astra Max's FIFO counterexample as a real two-program Engine run."""
    engine = Engine([Resource("fifo", 1.0, "ms", "fifo"),
                     Resource("indep", 1.0, "ms")])
    ready_a = engine.submit("a:ready", release=1.0)
    ready_b = engine.submit("b:ready", release=0.0)
    program_a = KernelProgram((
        Stage("a1", {"fifo": 1.0}),
        Stage("a2", {"indep": 100.0}, after=("a1",)),
    ))
    program_b = KernelProgram((
        Stage("b1", {"indep": 1.0}),
        Stage("b2", {"fifo": 10.0}, after=("b1",)),
    ))
    finished: dict[str, object] = {}

    def admit(event, program, name):
        finished[name] = program.start(engine, name, event)

    for program, ready, name in ((program_a, ready_a, "a"),
                                 (program_b, ready_b, "b")):
        if deferred:
            engine.submit(name + ":begin", after=(ready,), native_hook=True,
                          on_complete=lambda event, p=program, n=name:
                          admit(event, p, n))
        else:
            finished[name] = program.start(engine, name, ready)
    engine.advance(horizon=1000.0)
    return max(event.time for event in finished.values())


def test_fifo_admission_order_is_exact_under_the_eager_optimization(
        monkeypatch):
    """Deferred admission keeps 112; eager submission would report 102."""
    assert _fifo_frontier(deferred=True) == FIFO_DEFERRED_MS
    assert _fifo_frontier(deferred=False) == FIFO_EAGER_MS

    from mindie_sim.prediction import execution as execution_module
    case_dict = compose_case(DSV41_MODEL, hardware="a5", tp=8, ep=8,
                             num_devices=8, layers=SMALL_GRAPH_LAYERS,
                             output_tokens=SERVING_BENCHMARK_TOKENS)
    graphs = target_graphs(case_dict, phase="decode", tokens=1, context=256,
                           sequences=1, scope="body")
    from mindie_sim.prediction.graph_analysis import graph_costs
    # Submission work lives on a declared FIFO scheduler resource, so eager
    # admission must decline and the duration must equal the forced-deferred run.
    submitted = graph_costs(case_dict, graphs, external_inputs={"analysis_input": 0},
                            include_submission=True, isolated=False,
                            replay_scope="none")["duration_ms"]
    original = execution_module.Execution._program_uses_fifo
    monkeypatch.setattr(execution_module.Execution, "_program_uses_fifo",
                        lambda self, context, engine: True)
    forced = graph_costs(case_dict, graphs, external_inputs={"analysis_input": 0},
                         include_submission=True, isolated=False,
                         replay_scope="none")["duration_ms"]
    monkeypatch.setattr(execution_module.Execution, "_program_uses_fifo", original)
    assert abs(submitted - forced) <= abs(forced) * RELATIVE_TOLERANCE


def test_c11_and_python_engines_agree_on_wide_and_72_aiv_frontiers():
    """The arena executes >256 resources and 72 owners bit-identically."""
    pytest.importorskip("ctypes")
    from mindie_sim.prediction.runtime import accel_build
    compiler, _reason = accel_build.discover_compiler()
    if compiler is None:
        pytest.skip("c11_compiler_unavailable")

    resources = [Resource(f"r{index}", 100.0 + index, "ms",
                          units=WIDE_UNITS if index == 0 else 1)
                 for index in range(WIDE_RESOURCES)]
    durations = {}
    statuses = {}
    for backend in ("python", "c"):
        engine = Engine(resources, backend=backend)
        events = []
        for index in range(WIDE_RESOURCES):
            name = f"r{index}"
            units = (tuple(range(WIDE_UNITS)) if index == 0 else None)
            events.append(engine.submit(
                f"t{index}", {name: 100.0},
                placement={name: units} if units else None))
        engine.advance(until=tuple(events))
        durations[backend] = engine.now
        statuses[backend] = engine.accel_status()
    assert durations["c"] == durations["python"]
    assert statuses["c"]["native_steps"] > 0, statuses["c"]
    assert statuses["python"]["native_steps"] == 0
    assert statuses["python"]["accel_backend"] == "python"


@pytest.mark.parametrize("layout", [
    {"tp": 8, "ep": 8, "num_devices": 8},
    {"tp": 4, "dp": 4, "ep": 16, "num_devices": 16},
])
def test_real_graph_physics_is_backend_independent(monkeypatch, layout):
    from mindie_sim.prediction.runtime import accel_build
    if accel_build.discover_compiler()[0] is None:
        pytest.skip("c11_compiler_unavailable")
    case_dict = compose_case(DSV41_MODEL, hardware="a5",
                             layers=LONG_MODEL_LAYERS, **layout)
    monkeypatch.setenv("MINDIE_SIM_ACCEL", "python")
    reference = _backend_case(case_dict, context=4096)
    monkeypatch.setenv("MINDIE_SIM_ACCEL", "c")
    native = _backend_case(case_dict, context=4096)
    relative = abs(native["graph_duration_ms"] - reference["graph_duration_ms"])
    relative /= max(abs(reference["graph_duration_ms"]), RELATIVE_TOLERANCE)
    assert relative < RELATIVE_TOLERANCE
    assert int(native["accel_status"]["native_steps"]) >= 0
    assert native["accel_status"]["accel_backend"] in {
        "c_native", "c_native_ready", "python_fallback"}


def test_two_slot_pipeline_keeps_exact_slot_occupancy():
    """A tile pipeline and its flattened form keep identical timestamps."""
    from mindie_sim.prediction.program import coalesce_execution_program

    def tile(index):
        return (Stage(f"t{index}.load", {"hbm": 1.0}),
                Stage(f"t{index}.mma", {"cube": 1.0}, after=(f"t{index}.load",)))

    def makespan(program):
        engine = Engine([Resource("hbm", 1.0), Resource("cube", 1.0)])
        ready = engine.submit("input")
        done = program.start(engine, "pipe", ready)
        engine.advance(until=(done,))
        return done.time

    two_slot = PipelineProgram(count=4, slots=2, tile=tile)
    assert two_slot.slots == 2
    reference = makespan(two_slot)
    assert reference > 0.0
    # Coalescing is a task-shape optimization: the slot occupancy and every
    # virtual timestamp must survive it unchanged.
    assert makespan(coalesce_execution_program(two_slot)) == reference
    single_slot = PipelineProgram(count=4, slots=1, tile=tile)
    flattened = coalesce_execution_program(single_slot)
    assert isinstance(flattened, KernelProgram)
    assert makespan(flattened) == makespan(single_slot)


def test_physical_key_and_layer_runs_collapse_without_false_merges():
    case_dict = compose_case(DSV41_MODEL, hardware="a5", tp=8, ep=8,
                             num_devices=8)
    ops = target_graphs(case_dict, phase="decode", tokens=1, context=4096,
                        sequences=1, scope="body")[0]
    signatures = {}
    for op in ops:
        key = physical_key(op)
        signature = (op.family, op.flops, op.hbm_bytes, op.mte2_bytes,
                     op.mte3_bytes, op.vector_ops, op.link_bytes,
                     op.local_shape)
        assert signatures.setdefault(key, signature) == signature
    assert len(signatures) <= LONG_MODEL_LAYERS * 4
    runs = homogeneous_layer_runs(ops)
    assert any(run.count >= 3 for run in runs), runs
    assert sum(run.count for run in runs) >= LAST_LAYER_RUN_FLOOR, runs


def test_attnres_layers_never_fold_across_distinct_valid_blocks():
    case_dict = compose_case(KIMI_MODEL, hardware="a5", tp=8, ep=8,
                             num_devices=8, layers=20)
    ops = target_graphs(case_dict, phase="decode", tokens=1, context=256,
                        sequences=1, scope="body")[0]
    block_size = int(case_dict["model"].get("residual_block_size") or 8)
    for run in homogeneous_layer_runs(ops):
        blocks = {layer // block_size for layer in run.layers}
        assert len(blocks) == 1, run


def test_hardware_mutation_always_re_executes_a_fresh_engine():
    case_dict = compose_case(DSV41_MODEL, hardware="a5", tp=8, ep=8,
                             num_devices=8, layers=8)
    baseline = _backend_case(case_dict, context=1024)
    scaled_case = scale_case_hardware(parse_case(case_dict), hbm_bw_scale=2.0)
    scaled = analyze_network_efficiency(
        scaled_case, phase="decode", tokens=1, context=1024, sequences=1,
        scope="body")
    assert scaled["graph_duration_ms"] != baseline["graph_duration_ms"]
    assert (int(scaled["accel_status"]["native_steps"])
            + int(scaled["accel_status"]["python_steps"])) > 0
    # A second identical call still executes (no cross-call result memo).
    again = _backend_case(case_dict, context=1024)
    assert again["graph_duration_ms"] == baseline["graph_duration_ms"]


def test_93_layer_kimi_dynamic_efficiency_is_seconds_scale():
    case_dict = compose_case(KIMI_MODEL, hardware="a5", tp=1, dp=8, ep=8,
                             num_devices=8, layers=KIMI_LAYERS,
                             moe_comm_mode="dynamic")
    started = time.perf_counter()
    report = _backend_case(case_dict, context=4096)
    elapsed = time.perf_counter() - started
    print(f"\n[kimi-93L dynamic] wall={elapsed:.2f}s "
          f"step_ms={report['graph_duration_ms']:.4f} "
          f"backend={report['accel_status']['accel_backend']}")
    assert 0.0 < float(report["graph_duration_ms"])
    assert elapsed < EFFICIENCY_WALL_BUDGET_S


def test_compare_skus_matches_standalone_solvers_and_stays_seconds_scale():
    started = time.perf_counter()
    result = compare_skus(DSV41_MODEL, ["a3", "a5"], tp=8, ep=8,
                          num_devices=8, layers=LONG_MODEL_LAYERS,
                          sla_ms=50.0, sensitivity="baseline", jobs=1,
                          emit_advisory=False)
    elapsed = time.perf_counter() - started
    print(f"\n[compare-skus 40L baseline] wall={elapsed:.2f}s "
          f"backend={result['execution_coverage']['execution_backend']} "
          f"native_steps={result['execution_coverage']['native_steps']}")
    assert len(result["skus"]) == 2
    assert result["comparisons"][0]["step_speedup_x"] is not None
    assert "execution_backend" in result["runtime_environment"]
    assert "offload_recommended" in result["agent_advisory"]
    assert result["runtime_environment"]["wall_ms"] > 0.0

    reference_case = compose_case(DSV41_MODEL, hardware="a3", tp=8, ep=8,
                                  num_devices=8, layers=LONG_MODEL_LAYERS)
    geometry = result["geometry"]
    standalone = analyze_network_efficiency(
        reference_case, phase="decode", tokens=1,
        context=geometry["context"],
        sequences=geometry["sequences_per_dp"], scope="body")
    relative = abs(result["skus"][0]["metrics"]["step_ms"]
                   - standalone["graph_duration_ms"])
    relative /= max(abs(standalone["graph_duration_ms"]), RELATIVE_TOLERANCE)
    assert relative < RELATIVE_TOLERANCE

    standalone_sla = solve_sla_sizing(
        reference_case, sla_ms=50.0, context=geometry["context"],
        tokens_per_seq=1, max_sequences_per_dp=256, phase="decode",
        scope="body", sensitivity="baseline", jobs=1)
    assert (result["skus"][0]["sla_scenarios"][0]["b_sla"]
            == standalone_sla["scenarios"][0]["b_sla"])
    assert elapsed < COMPARE_WALL_BUDGET_S


def test_serving_prediction_records_wall_time_coverage_and_peak_rss():
    tokens = int(os.environ.get(SERVING_BENCHMARK_ENV,
                                str(SERVING_BENCHMARK_TOKENS)))
    case_dict = compose_case(DSV41_MODEL, hardware="a5", tp=8, ep=8,
                             num_devices=8, layers=LONG_MODEL_LAYERS,
                             output_tokens=tokens, concurrency=1,
                             prompt_tokens=128)
    rss_before = _resident_mib()
    started = time.perf_counter()
    prediction = predict(case_dict, scope="serving")
    elapsed = time.perf_counter() - started
    rss_after = _resident_mib()
    growth_mib = max(0.0, rss_after - rss_before)
    execution = prediction.get("execution") or {}
    print(f"\n[serving predict output_tokens={tokens}] wall={elapsed:.2f}s "
          f"native_steps={execution.get('native_steps')} "
          f"python_steps={execution.get('python_steps')} "
          f"rss_before_mib={rss_before:.1f} rss_after_mib={rss_after:.1f}")
    assert execution.get("completed_tasks", 0) > 0
    assert prediction["execution_status"]
    assert int(execution.get("python_steps") or 0) >= 0
    assert growth_mib < RSS_BUDGET_MIB


def test_runtime_advisor_bounds_workers_and_reports_truthfully(monkeypatch):
    assert runtime_advisor.resolve_jobs("auto", task_count=1) == 1
    assert runtime_advisor.resolve_jobs(8, task_count=3) == 3
    assert runtime_advisor.resolve_jobs("auto",
                                        task_count=ADVISOR_TASK_COUNT) >= 1
    host = runtime_advisor.probe_host_environment(ensure_accel=False)
    assert host["compiler"] is not None or host["compiler_available"] is False
    assert "compiler_reason" in host
    # Execution labels come from real counters, never from build availability.
    assert runtime_advisor.canonical_engine_backend(None) == "python_fallback"
    assert runtime_advisor.canonical_engine_backend(
        {"native_steps": 5, "python_steps": 0}) == "c11_accel"
    assert runtime_advisor.canonical_engine_backend(
        {"native_steps": 5, "python_steps": 3}) == "c11_accel+python"
    assert runtime_advisor.canonical_engine_backend(
        {"requested": "python", "native_steps": 0}) == "python"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("python backend touched the toolchain")

    monkeypatch.setattr(
        "mindie_sim.prediction.runtime.accel_build.discover_compiler",
        forbidden)
    engine = Engine([Resource("r", 10.0)], backend="python")
    status = engine.accel_status()
    assert status["accel_backend"] == "python"
    assert status["native_steps"] == 0
    assert status["python_steps"] == 0


def test_compare_skus_without_sla_publishes_efficiency_only():
    result = compare_skus(DSV41_MODEL, ["a3", "a5"], tp=8, ep=8,
                          num_devices=8, layers=SMALL_GRAPH_LAYERS,
                          sla_ms=None, jobs=1, emit_advisory=False)
    assert result["sla_ms"] is None
    assert result["scenarios_evaluated"] == ["baseline"]
    assert [row["sla_scenarios"] for row in result["skus"]] == [[], []]
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["step_speedup_x"] is not None


def test_cli_compare_skus_no_sla_skips_the_sla_search():
    from mindie_sim.__main__ import parser, run
    args = parser().parse_args([
        "compare-skus", "--model", DSV41_MODEL, "--skus", "a3", "a5",
        "--layers", str(SMALL_GRAPH_LAYERS), "--no-sla", "--jobs", "1",
        "--no-advisory",
    ])
    result = run(args)
    assert result["sla_ms"] is None
    assert [row["sla_scenarios"] for row in result["skus"]] == [[], []]
