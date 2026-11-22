"""Public CLI routing, typed SDK, composed cases, and execution receipts.
[Role] Exercise CLI and Python entry points against the same simulation engine.
[Invariants] Composition conserves physical work. Receipts use executed commit times.
[Timeouts & Locks] Bounded CLI subprocess and CPU-only analytical simulation.
[Side-Effect Policy] Files and isolated runtime state stay under pytest temp roots.
"""
from __future__ import annotations

from copy import deepcopy
from operator import itemgetter
import json
from pathlib import Path
import subprocess
import sys

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.operator_cases import declared_case
import mindie_sim
from mindie_sim import api as sdk_api, cli as cli_mod, predict
from mindie_sim.__main__ import main as module_main
from mindie_sim.backends.ascend import materialize
from mindie_sim.cli.parser import resolve_route
from mindie_sim.compose import compose_case as compose_raw_case
from mindie_sim.compose.runtime import capture_keys
from mindie_sim.prediction.results import StepRequest
from mindie_sim.compose import HARDWARE_ALIASES, apply_va_recommended, resolve_hardware_sku, va_recommended_config
from mindie_sim.prediction.efficiency import analyze_network_efficiency, extract_hardware_peaks
from mindie_sim.prediction.graph import GraphPolicy
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.prediction.scheduler import DraftPolicy, Scheduler
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView
from mindie_sim.public import ExecutionEnvironment, simulate_graph, simulate_native_call, simulate_operator
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.api import run_cohort, simulate_service
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend

pytestmark = pytest.mark.tier1

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_DEFAULTS = dict(dp=1, tp=1, pp=1, layers=1)
SDK_STEP = StepRequest(phase="decode", tokens=1, context=64, sequences=1, scope="body")
PREFILL = dict(phase="prefill", tokens=128, scope="body")
DECODE = dict(phase="decode", tokens=1, context=128, scope="body")
VECTOR_BYTES = 64
DRAM_SERVICE_SCALE = 1 / 0.8316
def _cli_json(output, *args):
    assert module_main([*args, "--output", str(output)]) == 0
    return json.loads(output.read_text(encoding="utf-8"))
def test_cli_entrypoint_parity_disambiguation_and_timeout_propagation(tmp_path, monkeypatch):
    import mindie_sim.public.cli as pub_cli
    case = compose_raw_case("qwen3-32b", REPRESENTATIVE_SKUS["a2"], **SDK_DEFAULTS)
    case_path, module_path = tmp_path / "case.json", tmp_path / "module.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    path, decode = str(case_path), ("--phase", "decode", "--tokens", "1", "--context", "64")
    step = ("predict", "step", path, *decode)
    direct = _cli_json(tmp_path / "direct.json", *step)
    proc = subprocess.run([sys.executable, "-m", "mindie_sim", *step, "--output", str(module_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert direct == json.loads(module_path.read_text(encoding="utf-8"))
    assert direct["span_ms"] > 0 and direct["graph_id"] and direct["operators"]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "step").write_text("{}", encoding="utf-8")
    for spelling, short_tail, long_tail, command in ((("predict", "step"), (), (path,), "predict-step"),
        (("calibrate", "record"), ("samples.jsonl",), ("store.sqlite", "samples.jsonl"), "calibrate"),
        (("calibrate", "bind-profiler"), ("samples.jsonl",), ("cat.sqlite", "store.sqlite", "b.jsonl"), "bind-profiler"),):
        group, subcommand = spelling
        route, rest = resolve_route([*spelling, *short_tail])
        assert (route.command, route.path, rest) == (group, (group,), (subcommand, *short_tail))
        route, rest = resolve_route([*spelling, *long_tail])
        assert (route.command, route.path, rest) == (command, spelling, long_tail)
    tiers = _cli_json(tmp_path / "tiers.json", "predict", "tiers", path, *decode)
    span = tiers["should_achieve"]["span_ms"]
    assert span > 0 and 0 <= tiers["certified_lower_bound_ms"] <= span + 1e-6
    assert tiers["aligned_table"]["graph_id"] == tiers["mainline"]["graph_id"]
    opt = _cli_json(tmp_path / "opt.json", "optimize", path, *decode, "--max-trials", "4")
    assert opt["schema"] == "mindie-sim.five-level-report/1"
    lean, dsl, framework, mainline, unmerge = (opt["ladder"][key] for key in (
        "l_star_lean_ns", "a_star_dsl_ns", "a_star_framework_ns", "m_mainline_ns", "u_unmerge_ns"))
    assert 0 <= lean <= dsl <= framework <= min(mainline, unmerge)
    assert _cli_json(tmp_path / "ops.json", *step, "--view", "operators")["operator_sum_ms"] > 0
    efficiency = _cli_json(tmp_path / "eff.json", *step, "--view", "efficiency",
                           "--primitives", ":memory:", "--include-submission")
    assert efficiency["graph_duration_ms"] > 0
    def raise_timeout(*args, **kwargs):
        raise TimeoutError("simulated_cli_timeout")
    for module, symbol, call, args in ((cli_mod, "run", cli_mod._main, ["predict", "step", path]),
        (pub_cli, "load_extensions", pub_cli.extensions_main, ["list"]),
        (pub_cli, "load_user_profile", pub_cli.profile_main, ["inspect", "dummy.json"]),):
        monkeypatch.setattr(module, symbol, raise_timeout)
        with pytest.raises(TimeoutError, match="simulated_cli_timeout"):
            call(args)
@pytest.mark.parametrize("sku", SKUS)
def test_typed_sdk_three_tier_evaluation_and_exports(sku):
    import mindie_sim.prediction.results as results_mod
    import mindie_sim.prediction.spine as spine_mod
    assert not hasattr(spine_mod, "_scale_case_efficiency") and not hasattr(results_mod, "DEFAULT_TARGET_EFFICIENCY")
    case = sdk_api.compose_case("qwen3-32b", hardware=sku, options=SDK_DEFAULTS)
    tiers = sdk_api.evaluate_three_tiers(case, SDK_STEP)
    span, bound = tiers.should_achieve.span_ms, min(tiers.mainline.span_ms, tiers.unmerge.span_ms)
    assert span > 0 and bound > 0 and 0 <= tiers.certified_lower_bound_ms <= span + 1e-6 and span <= bound + 1e-6
    aligned = tiers.aligned_table
    assert aligned is not None and aligned.graph_id == tiers.mainline.graph_id and len(aligned.tier_spans_ms) == 3
    lad = sdk_api.optimize_ladder(case, SDK_STEP, max_trials=4).ladder
    assert 0 <= lad.l_star_lean_ns <= lad.a_star_dsl_ns <= lad.a_star_framework_ns <= min(lad.m_mainline_ns, lad.u_unmerge_ns)
    assert len(mindie_sim.__all__) == len(set(mindie_sim.__all__))
def test_typed_sdk_service_sla_sweep_and_production_capture_resolution():
    import pickle
    from dataclasses import replace
    from mindie_sim.prediction.results import SlaRequest, SweepRequest
    from mindie_sim.stack_registry import LoweringRequest, default_stack_registry
    options = dict(SDK_DEFAULTS, prompt_tokens=32, output_tokens=2, concurrency=1, ttft_slo_ms=5000.0)
    case = sdk_api.compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a2"], options=options)
    service = sdk_api.evaluate_service(case)
    assert service.completed_requests >= 1 and service.emitted_tokens >= 2 and service.span_ms > 0
    assert pickle.loads(pickle.dumps(service)) == service
    sla_req = SlaRequest(step=StepRequest(phase="decode", tokens=1, context=128, sequences=1, scope="body"),
        latency_ms=50.0, max_sequences_per_dp=16, output_budget=32,)
    best = sdk_api.solve_sla(case, sla_req).best
    assert best is not None and best.fits_memory is True
    assert min(best.sequences_per_dp, best.span_ms, best.throughput_tokens_s) > 0
    for mode, counts in (("topology", (1, 2)), ("pd", (4,))):
        sweep = sdk_api.sweep_topology(case, SweepRequest(sla=sla_req, device_counts=counts, mode=mode))
        assert sweep.evaluated_count >= 1 and sweep.frontier
        assert all(pt.throughput_tokens_s > 0 and pt.span_ms > 0 for pt in sweep.frontier)
    speculative = json.dumps({"draft_len": 2, "acceptance": [0.8, 0.6]})
    eager = replace(case.runtime, graph_capture_mode="none", graph_capture_keys=(), metadata=(("speculative", speculative),))
    request = LoweringRequest(step=StepRequest("decode", context=64, sequences=2), tier="mainline",
                              production_capture=True, stack="raw_eager")
    resolved = default_stack_registry().resolve_base(replace(case, runtime=eager), request)
    assert resolved.case.runtime.graph_capture_mode == "full"
    assert resolved.case.runtime.graph_capture_keys == tuple(capture_keys(max(16, case.runtime.max_num_seqs or 16), 2))
    step_ev = sdk_api.evaluate_step(case, StepRequest("decode", context=64))
    with_preds = [row for row in step_ev.operators if row.predecessors and row.predecessors[0].op_id != "analysis_input"]
    assert with_preds and all(pred.rank == 0 for pred in with_preds[0].predecessors)

def test_top_level_optimize_ladder_replays_default_routed_schedule():
    case = compose_raw_case("deepseek-v4-flash", REPRESENTATIVE_SKUS["a5"], tp=4, ep=4)
    report = mindie_sim.optimize_ladder(case, StepRequest("decode", context=64))
    ladder, verification = report.ladder, report.verification
    assert mindie_sim.optimize_ladder is sdk_api.optimize_ladder and verification.passed and verification.scope == "full"
    assert 0 < ladder.l_star_lean_ns <= ladder.a_star_dsl_ns <= ladder.a_star_framework_ns <= min(
        ladder.m_mainline_ns, ladder.u_unmerge_ns)

from _support.axes import ALIASES_BY_SKU, CATALOG_COMPOSE_CASES, COMPOSE_HARDWARE  # noqa: E402

def test_resolve_hardware_sku_aliases_and_errors():
    for expected, aliases in ALIASES_BY_SKU.items():
        assert all(resolve_hardware_sku(alias) == expected for alias in aliases)
    assert {"a2", "a3", "a5", "910b4", "910_9392"} <= HARDWARE_ALIASES.keys()
    with pytest.raises(ModelingError, match="unknown_hardware_sku"):
        resolve_hardware_sku("unknown_gpu_999")

@pytest.mark.parametrize("model_name,tp,layers", CATALOG_COMPOSE_CASES)
@pytest.mark.parametrize("hw_alias,expected_sku,dp", COMPOSE_HARDWARE)
def test_compose_catalog_geometry_matches_hardware(model_name, hw_alias, expected_sku, tp, dp, layers):
    geometry = dict(tp=tp, dp=dp, ep=tp * dp, prompt_tokens=2048, output_tokens=64)
    case = compose_raw_case(model_name, hw_alias, concurrency=2, **geometry)
    ranks = min(tp * dp, 16 if expected_sku.startswith("Ascend910_93") else 8)
    assert itemgetter("sku", "num_devices", "ranks_per_node")(case["machine"]) == (expected_sku, tp * dp, ranks)
    assert [req["id"] for req in case["workload"]["requests"]] == ["req-0", "req-1"]
    assert parse_case(case).machine.sku == expected_sku
    fast = compose_raw_case(model_name, hw_alias, concurrency=1, layers=layers, **geometry)
    assert predict(fast, scope="body", phase="decode", tokens=1, context=2048, sequences=1)["duration_ms"] > 0

def test_compose_case_synthetic_layer_and_expert_overrides():
    case = compose_raw_case("qwen3.5-35b-a3b", "a3", tp=2, dp=2, ep=4, layers=8,
                            n_routed_experts=32, num_experts_per_tok=4, quant="w8a8")
    model = case["model"]
    assert itemgetter("layers", "n_routed_experts", "num_experts", "num_experts_per_tok", "top_k")(model) == (8, 32, 32, 4, 4)
    assert len(model["layer_plans"]) == 8
    moe = [plan for plan in model["layer_plans"] if plan.get("ffn") == "moe"]
    assert moe and all(itemgetter("num_experts", "top_k")(plan) == (32, 4) for plan in moe)
    parsed = parse_case(case).model
    assert (parsed.layers, parsed.num_experts, parsed.top_k) == (8, 32, 4)
    assert predict(case, scope="body", phase="decode", tokens=1, context=1024, sequences=2)["duration_ms"] > 0

def test_compose_case_execution_controls_and_long_context():
    def compose(model="qwen3.5-35b-a3b", tp=2, **options):
        return compose_raw_case(model, "a3", tp=tp, dp=1, ep=tp, layers=2, **options)
    views = (RequestView("req-0", 1, 2048, False, "decode"),)
    for capture in ("full", "piecewise"):
        assert GraphPolicy().select(views, parse_case(compose(graph_capture_mode=capture)).runtime) == (1, capture)
    for model, tp, mode, acceptance in (("qwen3.5-35b-a3b", 2, "mtp", 1.0), ("kimi-k3", 4, "dspark", 1.5),):
        raw = compose(model, tp, mtp_proposal_tokens=2, mtp_acceptance_length=acceptance)
        case = parse_case(raw)
        policy = DraftPolicy.for_request(case, case.workload.requests[0])
        assert policy.mode == raw["runtime"]["metadata"]["draft_mode"] == mode
        assert (policy.proposal, policy.mean) == pytest.approx((2, acceptance))
    sp = compose("kimi-k3", 4, sp=True)
    for runtime in (sp["runtime"], compose("qwen3.5-35b-a3b", 4, sp=True)["runtime"]):
        assert runtime["metadata"]["sp"] is True and runtime["kernels"]["sequence_parallel"]["padding"] == "right_to_tp"
    assert predict(sp, scope="body", phase="decode", tokens=1, context=2048, sequences=1)["duration_ms"] > 0
    long = compose("qwen3-30b-a3b", prompt_tokens=32768, output_tokens=128)
    assert min(long["runtime"]["max_model_len"], parse_case(long).runtime.max_model_len) >= 32768 + 128

@pytest.mark.parametrize("model,options,error", (
    ("qwen3.5-35b-a3b", dict(tp=1, dp=1, ep=1, layers=2, sp=True), "sequence_parallel"),
    ("qwen3.5-35b-a3b", dict(tp=4, dp=2, ep=3), "invalid_ep_partition"),
    ("qwen3.5-35b-a3b", dict(tp=4, dp=2, num_devices=4), "insufficient_num_devices"),
    ("qwen3.5-35b-a3b", dict(n_routed_experts=8, num_experts_per_tok=16), "customize_model_spec_moe_invalid:E=8:K=16"),
    ("qwen3.5-35b-a3b", dict(mtp_proposal_tokens=2, mtp_acceptance_length=5.0), "invalid_mtp_acceptance_length"),
    ("glm-5.3-flash", dict(tp=4, dp=1, ep=4, layers=2), "ep_requires_moe_model"),))
def test_compose_case_validation_errors(model, options, error):
    with pytest.raises(ModelingError, match=error):
        compose_raw_case(model, "a3", **options)

def test_cli_compose_efficiency_and_sla_end_to_end(tmp_path):
    case_path = tmp_path / "case.json"
    composed = _cli_json(case_path, "compose", "--model", "qwen3.5-35b-a3b", "--hardware", "a3",
                         "--tp", "2", "--dp", "1", "--ep", "2", "--layers", "2", "--experts", "32",
                         "--top-k", "4", "--prompt-tokens", "2048", "--output-tokens", "64", "--concurrency", "2")
    assert composed["machine"]["sku"] == "Ascend910_9392"
    assert itemgetter("layers", "n_routed_experts", "num_experts_per_tok")(composed["model"]) == (2, 32, 4)
    eff_args = ("efficiency", str(case_path), "--phase", "decode", "--tokens", "1", "--context", "2048", "--sequences", "2")
    report = _cli_json(tmp_path / "efficiency.json", *eff_args)
    fitted = _cli_json(tmp_path / "fitted.json", *eff_args, "--fit-targets")
    assert report["schema"] == fitted["schema"] == "mindie-sim.efficiency/1" and fitted["fitted_roofline_regimes"]
    total = report["operator_sum_ms"]
    assert total > 0
    bounds = report["breakdown_by_bound"].values()
    families = report["breakdown_by_family"]
    for rows in (bounds, families.values() if isinstance(families, dict) else families):
        assert sum(row["total_ms"] for row in rows) == pytest.approx(total, rel=1e-6)
    assert sum(row["share_of_operator_sum"] for row in bounds) == pytest.approx(1.0, rel=1e-6)
    sla_args = ("sla", str(case_path), "--sla-ms", "50.0", "--context", "4096", "--max-sequences-per-dp", "4")
    plain = _cli_json(tmp_path / "sla.json", *sla_args)
    scaled = _cli_json(tmp_path / "scaled.json", *sla_args, "--hbm-bw-scale", "2.0")
    assert plain["schema"] == scaled["schema"] == "mindie-sim.hardware-sla/1" and plain["sensitivity"] == "baseline"
    for report, ids in ((plain, ["baseline"]), (scaled, ["baseline", "scaled"])):
        assert [row.get("scenario_id", row.get("id")) for row in report["scenarios"]] == ids
    baseline, improved = scaled["scenarios"]
    assert improved["scales"]["hbm_bw_scale"] == pytest.approx(2.0)
    base_ms, scaled_ms = (row.get("step_ms_at_b1", row.get("b1_step_ms")) for row in (baseline, improved))
    base_seqs, scaled_seqs = (row.get("b_sla", row.get("max_sequences_per_dp")) for row in (baseline, improved))
    assert scaled_ms <= base_ms + 1e-9 and scaled_seqs >= base_seqs

def test_cli_sweep_modes(tmp_path):
    case_path = tmp_path / "case.json"
    case = compose_raw_case("qwen3.5-35b-a3b", "a3", tp=2, dp=2, ep=4, layers=1, prompt_tokens=128, output_tokens=16)
    case["runtime"]["max_num_seqs"] = 4
    case_path.write_text(json.dumps(case), encoding="utf-8")
    args = ("sweep", str(case_path), "--mode")
    topo = _cli_json(tmp_path / "topo.json", *args, "topology", "--num-devices", "4", "--context", "512", "--sla-ms", "0.5", "--output-budget", "16")
    pd = _cli_json(tmp_path / "pd.json", *args, "pd", "--num-devices", "8", "--prompt-tokens", "128",
                   "--output-tokens", "16", "--ttft-sla-ms", "2000.0", "--tpot-sla-ms", "0.5")
    assert topo["schema"] == "mindie-sim.topology-sweep/1" and "pd" in pd["schema"]

EXCHANGE_LATENCY_US = (("a5-pod-dt", (8.364917647058823, 16.222870588235295), (9.864917647058826, 17.7228705882353)),
    ("a5-server-dt", (31.6008, 61.2864), (346.6008, 376.2864)),)

@pytest.mark.parametrize("hardware,ideal_us,duration_us", EXCHANGE_LATENCY_US)
def test_expert_exchange_charges_payload_and_topology(hardware, ideal_us, duration_us):
    case = compose_raw_case("glm5.3", hardware, dp=64, tp=1, ep=64, layers=4,
                            prompt_tokens=131072, output_tokens=1024, concurrency=1216,
                            mtp_proposal_tokens=5, mtp_acceptance_length=4.5, tpot_slo_ms=20.0,
                            exclude_optimizations=("cann_megamoe_fusion",))
    report = analyze_network_efficiency(case, **(DECODE | dict(tokens=6, context=131072)), sequences=19, run_graph=False)
    rows = {row["recipe"]: row for row in report["operators"]}
    # Remote rows/rank multiply the FP8-plus-scale dispatch or BF16 combine payload.
    remote_rows = 6 * 19 * 8 * (64 - 1) / 64
    for i, (recipe, row_bytes) in enumerate((("expert_dispatch", 6144 + 48 * 4), ("expert_combine", 6144 * 2))):
        assert rows[recipe]["link_bytes"] == pytest.approx(remote_rows * row_bytes)
        assert rows[recipe]["t_link_ideal_ms"] * 1000 == pytest.approx(ideal_us[i], rel=1e-3)
        assert rows[recipe]["duration_ms"] * 1000 == pytest.approx(duration_us[i], rel=1e-3)
    op_ids = [row["op_id"] for row in report["operators"]]
    assert any(op.endswith(".pack.quantize") for op in op_ids) and not any(op.endswith(".expert_up.quantize") for op in op_ids)

def test_va_recommended_compose_and_apply_helper() -> None:
    config = va_recommended_config("deepseek-v3.2", "a3")
    assert itemgetter("quant", "graph_capture_mode", "sp")(config) == ("w8a8", "full", True)
    assert config["env"]["VLLM_ASCEND_ENABLE_MLAPO"] == "1"
    assert va_recommended_config("kimi-k3", "a5")["quant"] is None
    ds_case = compose_raw_case("deepseek-v3.2", hardware="a3", tp=4, dp=2, ep=8, layers=4, va_recommended=True)
    assert ds_case["runtime"]["metadata"]["va_env"]["VLLM_ASCEND_ENABLE_MLAPO"] == "1"
    assert ds_case["runtime"]["kernels"]["lowerings"].get("add_rmsnorm") == "vllm_ascend_native_norm"
    kimi_case = compose_raw_case("kimi-k3", hardware="a5", tp=4, dp=2, ep=8, layers=8, va_recommended=True)
    assert kimi_case["model"]["quant"] == compose_raw_case("kimi-k3", hardware="a5", tp=4, dp=2, ep=8, layers=8)["model"]["quant"]
    kernels = kimi_case["runtime"]["kernels"]
    assert kimi_case["runtime"]["metadata"]["sp"] is True and "sequence_parallel" in kernels
    kimi_low = kernels["lowerings"]
    kda_binding = kimi_low["kda"]["cases"][0]["binding"]
    assert kda_binding["implementation"] == "vllm_ascend_recurrent_kda_arch35"
    assert itemgetter("use_beta_sigmoid", "beta_dtype")(kda_binding["options"]) == (True, "bf16")
    assert kimi_low.get("attention_residual") == "vllm_ascend_attn_res_prefix"
    enriched_glm = apply_va_recommended(compose_raw_case("glm-5.3", "a3", tp=4, dp=2, ep=8, layers=4))
    for case in (ds_case, enriched_glm):
        runtime = case["runtime"]
        assert runtime["graph_capture_mode"] == "full" and runtime["metadata"]["va_recommended"] is True
        assert runtime["kernels"]["lowerings"].get("mla_prolog") == "vllm_ascend_mla_prolog_v3"
    for case in (ds_case, kimi_case, enriched_glm):
        assert target_graphs(case, phase="decode", tokens=1, context=1024, sequences=4)[0]

def _prefill_efficiency(case):
    return analyze_network_efficiency(case, context=0, sequences=1, run_graph=False, **PREFILL)

def _compose_features(model, hardware="a3", **options):
    return compose_raw_case(model, hardware, layers=4, prompt_tokens=128, output_tokens=8, **options)

def test_shared_expert_replication_conserves_math_and_removes_wire():
    def shared_ops(graph):
        return [op for op in graph if ".shared" in op.op_id and op.family in ("gate_up_projection", "down_projection")]
    tp, dp = 4, 4
    geometry = dict(tp=tp, dp=dp, ep=tp * dp, quant="w8a8")
    on, off = (_compose_features("deepseek-r1", **geometry, va_recommended=True, enable_shared_expert_dp=value)
               for value in (True, False))
    graph_on, graph_off = (target_graphs(case, **PREFILL) for case in (on, off))
    assert len(graph_on) == len(graph_off) == dp
    for rank in graph_off:
        large, split = shared_ops(graph_on[rank]), shared_ops(graph_off[rank])
        assert [op.family for op in large] == [op.family for op in split] == ["gate_up_projection", "down_projection"]
        for reshaped, sharded, axis in zip(large, split, (1, 2), strict=True):
            assert reshaped.local_shape[0] * tp == sharded.local_shape[0]
            assert reshaped.local_shape[axis] == sharded.local_shape[axis] * tp
            assert reshaped.flops == pytest.approx(sharded.flops, rel=1e-12)
    assert not any(op.link_bytes for graph in graph_on.values() for op in graph if ".shared" in op.op_id)
    assert sum(bool(op.link_bytes) for graph in graph_off.values() for op in graph if ".shared" in op.op_id) == 2 * dp
    enabled, disabled = map(_prefill_efficiency, (on, off))
    assert enabled["total_flops"] == pytest.approx(disabled["total_flops"], rel=1e-12)
    wire = 2 * PREFILL["tokens"] * on["model"]["hidden_size"] * 2 * (tp - 1) / tp
    assert disabled["total_link_bytes"] - enabled["total_link_bytes"] == pytest.approx(wire, rel=1e-12)
    assert enabled["operator_sum_ms"] < disabled["operator_sum_ms"]
    for value, expected in ((True, []), (False, [wire])):
        raw = _compose_features("deepseek-r1", **geometry, sp=False, enable_shared_expert_dp=value)
        links = [op.link_bytes for op in target_operators(raw, **PREFILL) if op.family == "allreduce" and ".shared" in op.op_id]
        assert links == pytest.approx(expected, rel=1e-12)

@pytest.mark.parametrize("model,tp,options", (("glm-5.1", 1, {"va_recommended": True}), ("glm-5.3", 2, {"sp": True})))
def test_context_partition_conserves_pairs_and_expert_exchange(model, tp, options):
    pcp = 8
    def compose(parts, **extras):
        return _compose_features(model, tp=tp, dp=1, pcp=parts, ep=tp * parts, **extras)
    case, base = compose(pcp, **options), compose(1, **options)
    ops, base_ops = (target_operators(raw, **PREFILL) for raw in (case, base))
    attention, original = (next(op for op in rows if op.layer == 0 and op.family == "attention") for rows in (ops, base_ops))
    counts = [dict(op.metadata) for op in (attention, original)]
    local, total = (data.get("local_pairs", data.get("logical_pairs")) for data in counts)
    tokens = PREFILL["tokens"]
    assert pcp * local == total == tokens * (tokens + 1) // 2
    assert pcp * attention.flops == pytest.approx(original.flops, rel=1e-12)
    if tp == 1:
        assert attention.local_shape[0] == tokens // pcp == 16
    plain = compose(pcp, sp=False) if options.get("sp") else None
    plain_ops = target_operators(plain, **PREFILL) if plain is not None else ()
    for raw, rows in ((case, ops), (plain, plain_ops)) if plain is not None else ((case, ops),):
        parallel = raw["runtime"]["parallel"]
        ep = parallel["ep"]
        assert ep == parallel["tp"] * parallel["dp"] * parallel["pcp"]
        exchange = [op for op in rows if op.recipe in ("expert_dispatch", "expert_combine")]
        assert exchange
        for op in exchange:
            meta = dict(op.metadata)
            assert "steps" not in meta
            expected = op.local_shape[0] * (ep - 1) / ep * meta["row_bytes"] / meta["source_assignment_partitions"]
            assert op.link_bytes == pytest.approx(expected, rel=1e-12)
    if plain is not None:
        _assert_sequence_parallel_work(case, plain, ops, plain_ops, tp)

def _assert_sequence_parallel_work(sp, plain, sp_ops, plain_ops, tp):
    norm = next(op for op in sp_ops if op.layer == 0 and op.recipe == "rmsnorm" and "norm1" in op.op_id)
    quant = next(op for op in sp_ops if op.layer == 0 and op.recipe == "dynamic_quantize" and "q_a.quantize" in op.op_id)
    fused = next(op for op in plain_ops if op.layer == 0 and op.recipe == "rmsnorm_quantize")
    assert norm.local_shape[0] * tp == fused.local_shape[0]
    assert norm.vector_ops * tp + quant.vector_ops == pytest.approx(fused.vector_ops, rel=1e-12)
    residual, full = (next(op.vector_ops for op in rows if op.recipe == "residual_add") for rows in (sp_ops, plain_ops))
    assert residual * tp == pytest.approx(full, rel=1e-12)
    for layer in range(4):
        wire = []
        for rows, expected in ((sp_ops, {"allgather", "reduce_scatter"}), (plain_ops, {"allreduce"})):
            links = [op for op in rows if op.layer == layer and op.family in ("allgather", "reduce_scatter", "allreduce")
                     and dict(op.metadata).get("group_axis") == "tp"]
            assert {op.family for op in links} == expected
            wire.append(sum(op.link_bytes for op in links))
        assert wire[0] == pytest.approx(wire[1], rel=1e-12)
    enabled, disabled = map(_prefill_efficiency, (sp, plain))
    assert enabled["operator_sum_ms"] - disabled["operator_sum_ms"] == pytest.approx(0.0183449315728843, abs=1e-8)
    assert enabled["total_link_bytes"] > disabled["total_link_bytes"]

def test_pipeline_preserves_compute_and_activation_bank_traffic():
    base, pipeline = (_compose_features("kimi-k3", "a5", tp=4, dp=1, pp=stages, ep=4,
                                        mtp_proposal_tokens=2, va_recommended=True) for stages in (1, 4))
    original, partitioned = (target_operators(case, **PREFILL) for case in (base, pipeline))
    assert {op.stage for op in partitioned} == set(range(4))
    for metric in ("flops", "hbm_bytes"):
        total, split = (sum(getattr(op, metric) for op in ops if op.layer >= 0) for ops in (original, partitioned))
        assert split == pytest.approx(total, rel=1e-12)
    stage_work = [sum(op.flops for op in partitioned if op.stage == stage and op.layer >= 0) for stage in range(4)]
    assert 0 < min(stage_work) and max(stage_work) < sum(stage_work) / 2
    assert sum(stage_work) == pytest.approx(sum(op.flops for op in partitioned if op.layer >= 0), rel=1e-12)
    shape = (PREFILL["tokens"] // 4, pipeline["model"]["hidden_size"])
    payload = shape[0] * shape[1] * 2
    for recipe, count in (("pipeline_send", 3), ("pipeline_residual_send", 6)):
        sends = [op for op in partitioned if op.family == "pp_p2p" and op.recipe == recipe]
        assert [op.local_shape for op in sends] == [shape] * count
        assert [op.link_bytes for op in sends] == pytest.approx([payload] * count, rel=1e-12)
        assert sum(op.link_bytes for op in sends) == pytest.approx(count * payload, rel=1e-12)

def test_pipeline_and_context_partition_scale_speculative_verify_rows():
    base, verify = (_compose_features("kimi-k3", "a5", tp=4, dp=1, pp=4, pcp=2, ep=8, sp=False,
                                      mtp_proposal_tokens=proposal) for proposal in (0, 2))
    proposal = verify["runtime"]["metadata"]["draft_proposal_tokens"]
    assert proposal == 2
    single, expanded = target_operators(base, **DECODE), target_operators(verify, **(DECODE | {"tokens": 1 + proposal}))
    routed_b, routed_a = ([op for op in ops if op.recipe == "megamoe_arch35"] for ops in (single, expanded))
    suffixes = (".shared.gate_up", ".shared.down") if routed_b else (".expert_up", ".expert_down")
    for suffix in suffixes:
        before, after = ([op for op in ops if op.op_id.endswith(suffix)] for ops in (single, expanded))
        assert len(before) == len(after) == 3
        assert all(b.local_shape[0] == 3 * a.local_shape[0] and b.flops == pytest.approx(3 * a.flops, rel=1e-12) and b.hbm_bytes > a.hbm_bytes for a, b in zip(before, after, strict=True))
    assert not routed_b or (len(routed_b) == len(routed_a) == 3 and all(b.flops == pytest.approx(3 * a.flops, rel=1e-12) and dict(b.metadata)["megamoe_geometry"]["routed_rows"] == pytest.approx(3 * dict(a.metadata)["megamoe_geometry"]["routed_rows"], rel=1e-12) for a, b in zip(routed_b, routed_a, strict=True)))
    rows = [next(op.local_shape[0] for op in ops if op.family == "attention") for ops in (single, expanded)]
    assert rows[1] == 3 * rows[0]

def test_quantized_mc2_conserves_flops_and_halves_expert_weight_payload():
    cases = [_compose_features("deepseek-v3.2", tp=2, dp=4, pcp=2, ep=16, va_recommended=True, quant=quant)
             for quant in ("w8a8", "w4a8")]
    parallel = cases[1]["runtime"]["parallel"]
    assert parallel["ep"] == parallel["tp"] * parallel["dp"] * parallel["pcp"]
    baseline, reduced = map(_prefill_efficiency, cases)
    assert reduced["total_flops"] == pytest.approx(baseline["total_flops"], rel=1e-12)
    assert all(reduced[key] < baseline[key] for key in ("total_hbm_bytes", "operator_sum_ms"))
    programs = []
    for native in deepcopy(cases):
        native["runtime"]["kernels"]["native_moe_protocol"] = {"kind": "arch35_fullmesh", "max_inflight_per_rank": 4}
        programs.append(target_operators(native, **PREFILL))
    weights = []
    for ops, dtype in zip(programs, ("int8", "int4"), strict=True):
        expert = [dict(op.metadata) for op in ops if op.op_id.endswith((".expert_up", ".expert_down"))]
        assert expert and all(metadata["weight_format"] == dtype for metadata in expert)
        weights.append(expert)
    rate = extract_hardware_peaks(cases[1])["hbm_peak_bytes_per_s"] / 1000
    for field, scale, tolerance in (("weight_payload_bytes", 1, 1e-12), ("weight_bytes", rate, 4e-3)):
        for original, packed in zip(*weights, strict=True):
            assert packed[field] == pytest.approx(0.5 * original[field], rel=tolerance)
        base, small = (sum(op[field] for op in rows) / scale for rows in weights)
        assert small == pytest.approx(0.5 * base, rel=tolerance) and small < base
    assert not any(op.recipe in ("expert_dispatch", "expert_combine") for op in programs[1])
    groups = [dict(op.metadata)["device_kernel_group"] for op in programs[1] if op.family == "mc2_protocol"]
    assert {group["kernel"] for group in groups} == {"MoeDistributeDispatchV3", "MoeDistributeCombineV3"}
    assert {group["id"].rsplit(".", 1)[-1] for group in groups} == {"mc2_dispatch", "mc2_combine"}

@pytest.mark.parametrize("model", ("deepseek-v4-pro", "qwen3.5-397b-a17b"))
def test_composite_quantized_pipeline_preserves_send_and_completion_boundaries(model):
    parallel = dict(tp=4, dp=2, pp=2, pcp=2, ep=16)
    case = _compose_features(model, **parallel, sp=True, va_recommended=True, quant="w4a8")
    assert case["runtime"]["metadata"]["sp"] is True
    assert {key: case["runtime"]["parallel"][key] for key in parallel} == parallel
    ops = target_operators(case, **PREFILL)
    assert {op.stage for op in ops} == {0, 1}
    send, = (op for op in ops if op.family == "pp_p2p" and op.recipe == "pipeline_send")
    layer = int(send.op_id.split(".l", 1)[1].split(".", 1)[0])
    hc = max(1, parse_case(case).model.layer_plans[layer].hc_mult)
    assert send.link_bytes == pytest.approx(send.local_shape[0] * case["model"]["hidden_size"] * 2 * hc, rel=1e-12)
    for suffix in (".expert_up", ".expert_down"):
        expert = [op for op in ops if op.op_id.endswith(suffix)]
        assert expert and all(dict(op.metadata)["weight_format"] == "int4" and op.hbm_bytes > 0 for op in expert)
    result = predict(case, sequences=1, **DECODE)
    duration = result["duration_ms"]
    assert (result["complete_ms"] - result["input_ready_ms"], result["body_forward_ms"]) == pytest.approx(
        (duration, duration), rel=1e-12)
    peak = extract_hardware_peaks(case)["hbm_peak_bytes_per_s"]
    hbm_floor = sum(op.hbm_bytes for op in target_operators(case, **DECODE)) / peak * 1000
    assert 0 < hbm_floor < duration < predict(case, context=0, sequences=1, **PREFILL)["duration_ms"]

def test_pipeline_retains_cross_layer_attention_residual_bank():
    for pp in (1, 2):
        case = compose_raw_case("kimi-k3", "a3", tp=8, pp=pp, ep=8, layers=4, va_recommended=True)
        ops = target_operators(case, **(PREFILL | {"scope": "model"}))
        views = [dict(op.metadata)["attn_res_view"] for op in ops if op.layer == 2 and "attn_res_view" in dict(op.metadata)]
        assert [view["valid_blocks"] for view in views] == [1, 1]

def test_short_prefill_owns_one_replica_token_across_context_partitions():
    short = compose_raw_case("deepseek-v3.2", "a3", dp=2, tp=2, ep=4, pcp=4, layers=4)
    ops = target_graphs(short, phase="prefill", tokens=1, sequences=1, scope="body")[0]
    assert next(op for op in ops if "expert_up" in op.op_id).semantic_shape[0] == 8

def test_cross_dp_sp_body_selection_preserves_body_defaults():
    case = compose_raw_case("qwen3.5-35b-a3b", "a3", tp=2, dp=2, ep=4, sp=True, layers=2, prompt_tokens=64, output_tokens=8)
    ops = target_operators(case, **PREFILL)
    selections = {row["family"]: row for row in ops.default_selection["selections"]}
    decision = itemgetter("selection", "reason")
    grouped_program = {"kind": "steady_matmul", "owner_cohorts": True}
    assert decision(selections["grouped_gemm"]) == ("default_installed", "preserve_existing_grouped_mapping_domain")
    assert decision(selections["matmul"]) == ("default_installed", "compact_finite_k_pipeline_preserving_128_output_grid")
    assert {"programs", "families"} <= ops.effective_kernels.keys()
    assert selections["grouped_gemm"]["program"] == ops.effective_kernels["families"]["grouped_gemm"] == grouped_program
    disabled = target_operators(case, **PREFILL, apply_body_defaults=False)
    assert disabled.default_selection["reason"] == "body_defaults_disabled" and "programs" not in disabled.effective_kernels
    explicit = {"kind": "steady_matmul", "explicit": True}
    case["runtime"]["kernels"]["families"] = {"grouped_gemm": explicit}
    preserved, = [row for row in target_operators(case, **PREFILL).default_selection["selections"]
                  if row["family"] == "grouped_gemm"]
    assert decision(preserved) == ("preserved", "explicit_implementation_preserved") and preserved["program"] == explicit

@pytest.fixture
def execution_environment():
    return ExecutionEnvironment.from_hardware(Hardware.from_mapping(materialize({"sku": "Ascend950DT_9582"})))

def _op(identifier, *deps):
    shape = (VECTOR_BYTES,)
    return OpSpec(identifier, "vector", "decode", 0, 0, (), shape, shape, shape, shape,
                  dtype="bf16", deps=deps, hbm_bytes=VECTOR_BYTES, vector_ops=VECTOR_BYTES)

def _assert_vector_charge(serviced, copies):
    work = copies * VECTOR_BYTES
    assert itemgetter("memory@r0:s0", "vector@r0:s0")(serviced) == pytest.approx((work * DRAM_SERVICE_SCALE, work))

def test_public_operator_report_preserves_duration_after_input_release(execution_environment):
    ops = (_op("a"),)
    baseline = simulate_operator(execution_environment, ops)
    row = baseline["operators"][0]
    assert baseline["schema"] == "mindie-sim.operators/1" and row["ready_ms"] == 0
    times = (row["complete_ms"], baseline["timeline"][0]["complete_ms"], baseline["operator_sum_ms"])
    assert times == pytest.approx((row["duration_ms"],) * 3)
    assert isinstance(row["runtime_uid"], str) and row["runtime_uid"]
    _assert_vector_charge(row["serviced"], 1)
    assert set(row["serviced"]) == {"memory@r0:s0", "vector@r0:s0"}
    delayed = simulate_operator(execution_environment, ops, input_ready_ms=5.0)
    ready, complete = itemgetter("ready_ms", "complete_ms")(delayed["operators"][0])
    assert ready == pytest.approx(5.0) and complete >= 5.0
    assert delayed["operator_sum_ms"] == pytest.approx(baseline["operator_sum_ms"])

def test_public_graph_serializes_dependencies_and_obeys_external_release(execution_environment):
    result = simulate_graph(execution_environment, (_op("a"), _op("b", "a")))
    assert itemgetter("operator_occurrences", "operator_events", "zero_work_joins")(result) == (2, 2, [])
    assert isinstance(result["runtime_uid"], str) and result["runtime_uid"] and result["duration_ms"] > 0
    assert result["complete_ms"] == pytest.approx(result["charge_ready_ms"] + result["duration_ms"])
    by_id = {row["physical_id"]: row for row in result["timeline"]}
    assert by_id["a"]["complete_ms"] <= by_id["b"]["ready_ms"]
    _assert_vector_charge(result["execution"]["serviced"], 2)
    external = simulate_graph(execution_environment, (_op("a", "producer"),), external_inputs={"producer": 5.0})
    assert min(external["charge_ready_ms"], external["timeline"][0]["ready_ms"]) >= 5.0

def test_public_native_call_charges_each_invocation_once(execution_environment):
    single, twice = (simulate_native_call(execution_environment, (_op("a"),), invocations=count, mode="eager")
                     for count in (1, 2))
    assert twice["physical_operators"] == 1
    assert twice["expanded_operators"] == twice["operator_occurrences"] == len(twice["timeline"]) == 2
    assert isinstance(twice["runtime_uid"], str) and twice["runtime_uid"]
    assert twice["duration_ms"] == pytest.approx(2 * single["duration_ms"])
    _assert_vector_charge(twice["execution"]["serviced"], 2)

def _pipeline_case():
    raw = declared_case(pp=2, requests=[{"id": request, "prompt_tokens": 1, "output_tokens": 2} for request in ("a", "b")])
    raw["runtime"]["pipeline_microbatches"] = 2
    return raw

def test_service_timings_follow_commits_and_resources_drain():
    result = simulate_service(_pipeline_case(), event_limit=8, trace_limit=8)
    commits = {}
    for trace in result["traces"]:
        for request, when in (trace.get("declared_commit_ms") or {}).items():
            commits.setdefault(request, []).append(float(when))
        assert not trace.get("fallback_commit_ms")
    assert sorted(commits) == ["a", "b"]
    for row in result["requests"]:
        times = commits[row["request_id"]]
        assert row["status"] == "served" and row["generated_tokens"] == row["output_tokens"] == row["committed_tokens"] == 2
        timing = itemgetter("first_token_abs_ms", "finish_ms", "e2e_ms")(row)
        assert timing == pytest.approx((min(times), max(times), max(times) - row["arrival_ms"]))
    assert itemgetter("production_entry", "metric_boundary")(result) == ("simulate_service", "device_token_commit")
    execution, memory = result["execution"], result["memory"]
    assert result["metrics"]["served_requests"] == 2 and result["status"] == "modeled" and execution["cohort_stop"] == "drained"
    assert execution["advance_calls"] >= 1 and isinstance(execution["runtime_uid"], str) and execution["runtime_uid"]
    for key in ("live_grants_bytes_per_replica_stage", "live_cache_bytes_per_replica_stage"):
        assert all(value == 0 for replica in memory[key] for value in replica)
    assert all(value == 0 for replica in memory["physical_rank_reader_bytes_per_replica_stage"]
               for stage in replica for value in stage)

def test_run_cohort_advances_only_the_caller_runtime():
    case = parse_case(_pipeline_case())
    memory = Memory(case)
    scheduler = Scheduler(case, list(case.workload.requests), memory, 0)
    execution = Execution(case, CostModel(case), include_serving_io=True, memory_policy=memory.policy, memory=memory)
    run = run_cohort(case, execution, [scheduler], memory, Frontend(), trace_limit=0)
    assert run["engine"] is execution.engine and run["serving_end"] == execution.engine.now
    assert run["stop_reason"] == "drained" and scheduler.done is True
    assert execution.memory is execution.engine.memory is memory

def test_token_quota_preserves_declared_request_length():
    result = simulate_service(_pipeline_case(), event_limit=4, token_quotas={"a": 1, "b": 1})
    assert result["status"] == "incomplete" and result["execution"]["cohort_stop"] == "token_quota"
    assert all(row["generated_tokens"] <= row["output_tokens"] == 2 for row in result["requests"])

def test_isolated_worker_runs_and_times_out():
    from mindie_sim.sweep import sweep_topologies_iter
    raw = compose_raw_case("qwen3-32b", "a3", tp=1, ep=1, pp=1, num_devices=1, layers=2, prompt_tokens=8, output_tokens=2, concurrency=1, max_num_seqs=1)
    kw = {"sla_ms": 30.0, "context": 8, "output_budget": 2, "max_sequences_per_dp": 1}
    rows = list(sweep_topologies_iter(raw, two_stage=True, finalist_top_k=1, isolate_workers=True, candidate_timeout_s=120.0, num_devices=1, **kw))
    assert any(row["evaluation_stage"] == "b1_screen" for row in rows) and any(row["evaluation_stage"] == "full_sla" for row in rows)
    timed_out = list(sweep_topologies_iter(raw, isolate_workers=True, candidate_timeout_s=1e-9, num_devices=1, **kw))
    assert [row["status"] for row in timed_out] == ["timeout"] and timed_out[0]["feasibility_status"] == "candidate_timeout"


def test_agent_first_report_and_solve_cli_flags_and_model_target_ceiling(tmp_path):
    import mindie_sim
    from mindie_sim.cli import _main as cli_main
    from mindie_sim.cli.parser import build_parser

    p = build_parser()
    rep_args = p.parse_args([
        "report", "--model", "qwen3-32b", "--hardware", "a5",
        "--mode", "simulation", "--rewrite-plan", str(tmp_path / "plan.json"),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rep_args.mode == "simulation" and rep_args.rewrite_plan.endswith("plan.json")

    sol_args = p.parse_args([
        "solve", "--model", "qwen3-32b", "--hardware", "a5",
        "--mode", "simulation", "--rewrite-plan", str(tmp_path / "plan.json"),
    ])
    assert sol_args.mode == "simulation" and sol_args.rewrite_plan.endswith("plan.json")

    qs_out = tmp_path / "quickstart.json"
    assert cli_main(["quickstart", "--json", "-o", str(qs_out)]) == 0
    qs_payload = json.loads(qs_out.read_text(encoding="utf-8"))
    assert qs_payload["schema"] == "mindie-sim.quickstart/1"
    assert any("--rewrite-plan" in item["cmd"] for item in qs_payload["golden_commands"])

    ceiling = mindie_sim.model_target_ceiling(
        "qwen3-32b", "950dt-96g-pod",
        num_devices=4, layers=2, prompt_tokens=512, output_tokens=64,
        tpot_slo_ms=50.0, max_sequences_per_dp=4,
    )
    assert ceiling["analysis_mode"] == "modeling" and ceiling["allow_unbound_formula"] is True
    assert ceiling["feasible"] is True and ceiling["iso_sla_comparison"]["after"]["sustainable_tps_per_die"] > 0.0

    # End-to-end turnkey report with --from-script (verifies ParsedLaunchScript wiring).
    script = tmp_path / "serve.sh"
    script.write_text(
        "export HCCL_OP_EXPANSION_MODE=AI_CPU\n"
        "vllm serve /weights/qwen3-32b-bf16 --tensor-parallel-size 2 --data-parallel-size 1 --max-num-seqs 2 --enforce-eager\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "rep_out"
    rec_path = tmp_path / "receipt.json"
    rc = cli_main([
        "report", "--from-script", str(script), "--hardware", "a5",
        "--dp", "1", "--tp", "2", "--ep", "1", "--concurrency", "2",
        "--prompt-tokens", "512", "--output-tokens", "32",
        "--out-dir", str(out_dir), "--json", "--receipt", str(rec_path),
    ])
    assert rc == 0 and rec_path.is_file()
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec["ok"] is True and rec["mode"] == "modeling"
    brief_path = Path(rec["agent_brief"]["generated_files"]["agent_brief_json"])
    opt_sh_path = Path(rec["agent_brief"]["generated_files"]["optimized_serve_sh"])
    assert brief_path.is_file() and brief_path.stat().st_size < 8192
    assert opt_sh_path.is_file() and "FULL_DECODE_ONLY" in opt_sh_path.read_text(encoding="utf-8")


def test_public_graph_mapping_cli_geometry_and_report_parity(tmp_path, capsys):
    from mindie_sim.cli import _main as cli_main
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.frontend.speculation import compute_speculative_accept_step
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.public.api import simulate_graph
    from mindie_sim.structure import export_structure

    # 1. public.simulate_graph handles rank-keyed TargetGraphSet mappings without crashing.
    raw = compose_case("qwen3-32b", "a3", tp=2, layers=2, prompt_tokens=256, output_tokens=32)
    graphs = target_graphs(parse_case(raw), phase="decode", tokens=1, context=256, sequences=1, scope="model")
    rep = simulate_graph(raw, graphs)
    assert float(rep["duration_ms"]) > 0.0

    # 2. CLI graph command emits canonical phase/context/execution_geometry and threads media_tokens into structure.
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(raw), encoding="utf-8")
    graph_out = tmp_path / "graph_out.json"
    struct_out = tmp_path / "struct_out.json"
    rc_g = cli_main([
        "graph", str(case_path), "--phase", "prefill", "--tokens", "64", "--context", "0",
        "--sequences", "1", "--scope", "model", "--media-tokens-per-sequence", "0",
        "--export-structure", str(struct_out), "-o", str(graph_out),
    ])
    err_g = capsys.readouterr().err
    assert rc_g == 0, err_g
    assert graph_out.is_file() and struct_out.is_file()
    g_data = json.loads(graph_out.read_text(encoding="utf-8"))
    s_data = json.loads(struct_out.read_text(encoding="utf-8"))
    assert g_data["phase"] == "prefill" and g_data["context"] == 0
    assert g_data["execution_geometry"]["phase"] == "prefill"
    assert g_data["execution_geometry"]["media_tokens_per_sequence"] == 0
    assert s_data["execution"]["media_tokens_per_sequence"] == 0
    from mindie_sim.structure.export import _structure_inputs
    media_case_path = Path(__file__).resolve().parents[2] / "examples/cases/qwen3.8-27b-media.json"
    media_raw = json.loads(media_case_path.read_text(encoding="utf-8"))
    graphs_media, opts_media = _structure_inputs(
        media_raw, phase="prefill", tokens=5, context=0, sequences=1, media_tokens_per_sequence=4,
    )
    graphs_lang, opts_lang = _structure_inputs(
        media_raw, phase="prefill", tokens=5, context=0, sequences=1, media_tokens_per_sequence=0,
    )
    assert opts_media["execution"]["media_tokens_per_sequence"] == 4
    assert opts_lang["execution"]["media_tokens_per_sequence"] == 0
    assert len(graphs_media[0]) > len(graphs_lang[0])
    direct_s = export_structure(raw, phase="prefill", tokens=64, context=0, sequences=1, media_tokens_per_sequence=0)
    assert direct_s["execution"]["media_tokens_per_sequence"] == 0

    # 3. CLI report uses geometric speculative accept length & honors explicit --weight-quant bf16 --cache-dtype bf16.
    out_dir = tmp_path / "rep_bf16"
    rec_path = tmp_path / "rep_bf16_rec.json"
    rc_r = cli_main([
        "report", "--model", "deepseek-v3.2", "--hardware", "a5",
        "--dp", "1", "--tp", "8", "--ep", "8", "--concurrency", "1",
        "--prompt-tokens", "256", "--output-tokens", "32",
        "--weight-quant", "bf16", "--cache-dtype", "bf16",
        "--spec-tokens", "3", "--mode", "simulation",
        "--no-trace", "--no-xlsx", "--out-dir", str(out_dir), "--json", "--receipt", str(rec_path),
    ])
    err_r = capsys.readouterr().err
    assert rc_r == 0, err_r
    assert rec_path.is_file()
    rec_r = json.loads(rec_path.read_text(encoding="utf-8"))
    scen = rec_r["agent_brief"]["scenario_and_parallelism"]
    assert scen["weight_quant"].upper().startswith("BF16")
    assert scen["cache_dtype"].upper() == "BF16"
    assert scen["avg_accepted_tokens"] == pytest.approx(compute_speculative_accept_step(0.70, 3), rel=1e-6)


