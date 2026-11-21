"""Unified CLI routing, ``python -m mindie_sim`` parity, and typed Python SDK.

[Role] Verify ``mindie_sim.cli:main`` and ``python -m mindie_sim`` share one
  command route, legacy vs canonical disambiguation works without filesystem
  sniffing, ``TimeoutError`` propagates cleanly, and ``mindie_sim.api``
  three-tier evaluation runs causally on ``runtime.Engine``.
[Invariants] Legacy ``predict step`` (file literally named ``step``) and
  ``calibrate record samples.jsonl`` stay unambiguous. ``TimeoutError`` is
  never masked by ``except OSError:``. ``mindie_sim.__all__`` has zero
  duplicate exports.
[Timeouts & Locks] Fast Tier-1 subprocess + in-memory checks (<5s).
[Side-Effect Policy] Writes only inside pytest ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from _support.axes import SKUS
import mindie_sim
import mindie_sim.api as sdk_api
import mindie_sim.cli as cli_mod
from mindie_sim.cli_parser import resolve_route
from mindie_sim.compose import compose_case as compose_raw_case
from mindie_sim.prediction.results import StepRequest

pytestmark = pytest.mark.tier1

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_entrypoint_parity_disambiguation_and_timeout_propagation(tmp_path, monkeypatch):
    raw_case = compose_raw_case(
        model="qwen3-32b",
        hardware="Ascend910B2",
        dp=1, tp=1, pp=1, layers=1,
    )
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(raw_case), encoding="utf-8")

    out_direct = tmp_path / "step_direct.json"
    out_module = tmp_path / "step_module.json"
    rc_direct = cli_mod.main([
        "predict", "step", str(case_path),
        "--phase", "decode", "--tokens", "1", "--context", "64",
        "--output", str(out_direct),
    ])
    assert rc_direct == 0

    proc = subprocess.run(
        [
            sys.executable, "-m", "mindie_sim",
            "predict", "step", str(case_path),
            "--phase", "decode", "--tokens", "1", "--context", "64",
            "--output", str(out_module),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    direct_payload = json.loads(out_direct.read_text(encoding="utf-8"))
    module_payload = json.loads(out_module.read_text(encoding="utf-8"))
    assert direct_payload == module_payload
    assert direct_payload["span_ms"] > 0.0
    assert direct_payload["graph_id"]
    assert len(direct_payload["operators"]) > 0

    # Legacy file literally named "step" routes to legacy `predict` (not `predict-step`).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "step").write_text("{}", encoding="utf-8")
    route_legacy, rest_legacy = resolve_route(["predict", "step"])
    assert route_legacy.command == "predict"
    assert rest_legacy == ("step",)

    route_canon, rest_canon = resolve_route(["predict", "step", str(case_path)])
    assert route_canon.command == "predict-step"
    assert rest_canon == (str(case_path),)

    # Legacy calibrate <store="record"> <samples> vs canonical calibrate record <store> <samples>.
    cal_legacy, cal_legacy_rest = resolve_route(["calibrate", "record", "samples.jsonl"])
    assert cal_legacy.command == "calibrate"
    assert cal_legacy.path == ("calibrate",)
    assert cal_legacy_rest == ("record", "samples.jsonl")

    cal_canon, cal_canon_rest = resolve_route(["calibrate", "record", "store.sqlite", "samples.jsonl"])
    assert cal_canon.command == "calibrate"
    assert cal_canon.path == ("calibrate", "record")
    assert cal_canon_rest == ("store.sqlite", "samples.jsonl")

    # Legacy calibrate <store="bind-profiler"> <samples> vs canonical calibrate bind-profiler <cat> <store> <bindings>.
    bp_legacy, bp_legacy_rest = resolve_route(["calibrate", "bind-profiler", "samples.jsonl"])
    assert bp_legacy.command == "calibrate"
    assert bp_legacy.path == ("calibrate",)
    assert bp_legacy_rest == ("bind-profiler", "samples.jsonl")

    bp_canon, bp_canon_rest = resolve_route(["calibrate", "bind-profiler", "cat.sqlite", "store.sqlite", "b.jsonl"])
    assert bp_canon.command == "bind-profiler"
    assert bp_canon.path == ("calibrate", "bind-profiler")
    assert bp_canon_rest == ("cat.sqlite", "store.sqlite", "b.jsonl")

    # Canonical three-tier & certified optimize CLI commands (zero 0.80x scaling).
    out_tiers = tmp_path / "tiers.json"
    rc_tiers = cli_mod.main([
        "predict", "tiers", str(case_path),
        "--phase", "decode", "--tokens", "1", "--context", "64",
        "--output", str(out_tiers),
    ])
    assert rc_tiers == 0
    tiers_payload = json.loads(out_tiers.read_text(encoding="utf-8"))
    assert tiers_payload["certified_lower_bound_ms"] >= 0.0
    assert tiers_payload["should_achieve"]["span_ms"] > 0.0
    assert tiers_payload["should_achieve"]["span_ms"] >= tiers_payload["certified_lower_bound_ms"] - 1e-6
    assert tiers_payload["aligned_table"]["graph_id"] == tiers_payload["mainline"]["graph_id"]

    out_opt = tmp_path / "opt.json"
    rc_opt = cli_mod.main([
        "optimize", str(case_path),
        "--phase", "decode", "--tokens", "1", "--context", "64",
        "--max-trials", "4", "--output", str(out_opt),
    ])
    assert rc_opt == 0
    opt_payload = json.loads(out_opt.read_text(encoding="utf-8"))
    assert opt_payload["schema"] == "mindie-sim.five-level-report/1"
    lad = opt_payload["ladder"]
    assert 0 <= lad["l_star_lean_ns"] <= lad["a_star_dsl_ns"] <= lad["a_star_framework_ns"] <= min(lad["m_mainline_ns"], lad["u_unmerge_ns"])

    # predict step --view operators and --view efficiency (including --primitives :memory: & --include-submission).
    out_ops = tmp_path / "ops_view.json"
    assert cli_mod.main([
        "predict", "step", str(case_path), "--view", "operators",
        "--phase", "decode", "--tokens", "1", "--context", "64", "--output", str(out_ops),
    ]) == 0
    assert json.loads(out_ops.read_text(encoding="utf-8"))["operator_sum_ms"] > 0.0

    out_eff = tmp_path / "eff_view.json"
    assert cli_mod.main([
        "predict", "step", str(case_path), "--view", "efficiency",
        "--phase", "decode", "--tokens", "1", "--context", "64",
        "--primitives", ":memory:", "--include-submission", "--output", str(out_eff),
    ]) == 0
    assert json.loads(out_eff.read_text(encoding="utf-8"))["graph_duration_ms"] > 0.0

    # TimeoutError must propagate directly in both cli.py and public/cli.py.
    def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("simulated_cli_timeout")

    monkeypatch.setattr(cli_mod, "run", _raise_timeout)
    with pytest.raises(TimeoutError, match="simulated_cli_timeout"):
        cli_mod._main(["predict", "step", str(case_path)])

    import mindie_sim.public.cli as pub_cli

    monkeypatch.setattr(pub_cli, "load_extensions", _raise_timeout)
    with pytest.raises(TimeoutError, match="simulated_cli_timeout"):
        pub_cli.extensions_main(["list"])
    monkeypatch.setattr(pub_cli, "load_user_profile", _raise_timeout)
    with pytest.raises(TimeoutError, match="simulated_cli_timeout"):
        pub_cli.profile_main(["inspect", "dummy.json"])


@pytest.mark.parametrize("sku", SKUS)
def test_typed_sdk_three_tier_evaluation_and_exports(sku):
    import mindie_sim.prediction.results as results_mod
    import mindie_sim.prediction.spine as spine_mod

    assert not hasattr(spine_mod, "_scale_case_efficiency")
    assert not hasattr(results_mod, "DEFAULT_TARGET_EFFICIENCY")

    case = sdk_api.compose_case(
        "qwen3-32b", hardware=sku, options={"dp": 1, "tp": 1, "pp": 1, "layers": 1}
    )
    req = StepRequest(phase="decode", tokens=1, context=64, sequences=1, scope="body")
    tiers = sdk_api.evaluate_three_tiers(case, req)
    assert tiers.unmerge.span_ms > 0.0
    assert tiers.mainline.span_ms > 0.0
    assert tiers.should_achieve.span_ms > 0.0
    assert 0.0 <= tiers.certified_lower_bound_ms <= tiers.should_achieve.span_ms + 1e-6
    assert tiers.should_achieve.span_ms <= min(tiers.mainline.span_ms, tiers.unmerge.span_ms) + 1e-6
    assert tiers.aligned_table is not None
    assert tiers.aligned_table.graph_id == tiers.mainline.graph_id
    assert len(tiers.aligned_table.tier_spans_ms) == 3

    report = sdk_api.optimize_ladder(case, req, max_trials=4)
    lad = report.ladder
    assert 0 <= lad.l_star_lean_ns <= lad.a_star_dsl_ns <= lad.a_star_framework_ns <= min(lad.m_mainline_ns, lad.u_unmerge_ns)

    assert len(mindie_sim.__all__) == len(set(mindie_sim.__all__))


def test_typed_sdk_service_sla_sweep_and_production_capture_resolution():
    import pickle
    from dataclasses import replace
    from mindie_sim.prediction.results import SlaRequest, SweepRequest
    from mindie_sim.stack_registry import LoweringRequest, default_stack_registry

    case = sdk_api.compose_case(
        "qwen3-32b",
        hardware="Ascend910B2",
        options={
            "dp": 1, "tp": 1, "pp": 1, "layers": 1,
            "prompt_tokens": 32, "output_tokens": 2, "concurrency": 1,
        },
    )

    # evaluate_service returns non-zero served requests, output tokens, and span, and pickles cleanly.
    svc = sdk_api.evaluate_service(case)
    assert svc.completed_requests >= 1
    assert svc.emitted_tokens >= 2
    assert svc.span_ms > 0.0
    assert pickle.loads(pickle.dumps(svc)) == svc

    # solve_sla returns non-None best candidate with real concurrency & throughput.
    sla_req = SlaRequest(
        step=StepRequest(phase="decode", tokens=1, context=128, sequences=1, scope="body"),
        latency_ms=50.0,
        max_sequences_per_dp=16,
        output_budget=32,
    )
    sla = sdk_api.solve_sla(case, sla_req)
    assert sla.best is not None
    assert sla.best.sequences_per_dp > 0
    assert sla.best.span_ms > 0.0
    assert sla.best.throughput_tokens_s > 0.0
    assert sla.best.fits_memory is True

    # sweep_topology (topology & pd modes) returns non-empty Pareto frontier with real metrics.
    topo = sdk_api.sweep_topology(case, SweepRequest(sla=sla_req, device_counts=(1, 2), mode="topology"))
    assert topo.evaluated_count >= 1
    assert len(topo.frontier) >= 1
    assert all(pt.throughput_tokens_s > 0.0 and pt.span_ms > 0.0 for pt in topo.frontier)

    pd_res = sdk_api.sweep_topology(case, SweepRequest(sla=sla_req, device_counts=(4,), mode="pd"))
    assert pd_res.evaluated_count >= 1
    assert len(pd_res.frontier) >= 1
    assert all(pt.throughput_tokens_s > 0.0 and pt.span_ms > 0.0 for pt in pd_res.frontier)

    # production_capture=True resolves JSON speculative SpeculativeSpec matching scheduler.py.
    from mindie_sim.compose_runtime import capture_keys

    eager_case = replace(
        case,
        runtime=replace(
            case.runtime,
            graph_capture_mode="none",
            graph_capture_keys=(),
            metadata=(("speculative", json.dumps({"draft_len": 2, "acceptance": [0.8, 0.6]})),),
        ),
    )
    resolved = default_stack_registry().resolve_base(
        eager_case,
        LoweringRequest(
            step=StepRequest(phase="decode", tokens=1, context=64, sequences=2),
            tier="mainline",
            production_capture=True,
            stack="raw_eager",
        ),
    )
    assert resolved.case.runtime.graph_capture_mode == "full"
    assert resolved.case.runtime.graph_capture_keys == tuple(capture_keys(max(16, int(case.runtime.max_num_seqs or 16)), 2))

    # In-graph predecessor occurrences preserve producer physical rank (not None).
    step_ev = sdk_api.evaluate_step(case, StepRequest(phase="decode", tokens=1, context=64, sequences=1))
    with_preds = [row for row in step_ev.operators if row.predecessors and row.predecessors[0].op_id != "analysis_input"]
    assert with_preds and all(pred.rank == 0 for pred in with_preds[0].predecessors)
