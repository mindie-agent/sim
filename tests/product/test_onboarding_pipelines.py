"""Orthogonal Tier-1 tests for turnkey SKU & Operator onboarding pipelines.

[Role] Verify all 4 New-SKU stages (``probe-soc``, ``calibrate-l0-and-fabric``,
  ``validate-512-400k``, ``verify-gates``, ``run-all``), all 4 New-Operator
  stages (``scaffold``, ``validate-contract``, ``generate-cliff-plan``,
  ``bench-and-score``, ``certify``, ``run-all``), Lock-2 sealed scoring,
  Lock-3 ``TilingAxes`` / ``validate_abi`` 2D cliff synthesis, true L1/L2/L3
  ``WhiteboxSpineRunner`` separation & Stage-2 ``p50`` threading, real
  coordinator/profiler wire contracts, and adversarial fail-closed gates.
[Invariants] Strictly isolates ``HOME``, ``USERPROFILE``, ``APPDATA``,
  ``LOCALAPPDATA``, ``XDG_*``, ``MINDIE_*``, and ``TMPDIR``; zero raw IPv4
  literals, zero host mutations, zero device contexts, independent hardware
  oracle specs, independent test runner tables, and zero ``service_regimes``.
[Timeouts & Locks] Deterministic in-memory and ``tmp_path`` subprocess checks
  with bounded timeouts (< 15s); no global locks.
[Side-Effect Policy] Writes temporary fixtures/receipts exclusively under
  pytest ``tmp_path``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from mindie_sim.cli import main as mindie_sim_main
from mindie_sim.observations.multilevel_gate import EngineInterval, GateCase, bind_engine_interval
from mindie_sim.onboarding import (
    ANCHOR_CONTEXT_GRID,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    AbiDivisibilityRule,
    OperatorCliffParameters,
    TilingAxes,
    TimeoutBudget,
    WhiteboxSpineRunner,
    bench_and_score_operator,
    calibrate_l0_and_fabric,
    certify_operator,
    check_lean_non_inversion,
    evaluate_abi_rejections,
    generate_operator_cliff_plan,
    probe_soc,
    run_operator_onboarding,
    run_sku_onboarding,
    scaffold_operator,
    score_sealed_gate_cases,
    synthesize_operator_cliff_plan,
    synthesize_sku_512_400k_plan,
    validate_512_400k,
    validate_operator_contract,
    verify_receipt_sha256,
    verify_sku_gates,
)
from mindie_sim.onboarding._gate_runners import acquire_coordinator_host_queue_lease
from mindie_sim.onboarding._operator_contract import (
    _build_default_planning_context,
    _verify_resolved_operator_dual_bound,
)
from mindie_sim.optimization.dual_bound import EventDagSpec, ResourceChoiceProgram, certify_lower_bound
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.operators.contracts import OperatorFamilySpec, ResolvedOperator
from mindie_sim.prediction.operators.registry import FamilyRegistry
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.primitives.schema import parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from _support.axes import REPRESENTATIVE_SKUS

pytestmark = pytest.mark.tier1
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPEATS = 12
A2_SKU = REPRESENTATIVE_SKUS["a2"]
A2_ORACLE = {
    "arch_id": "npu_arch_2201", "aic": 20, "aiv": 40,
    "l2": 100_663_296, "ub": 196_608, "dies": 1,
    "hbm_bw": 1.2e12, "mte2_aiv": 2.4e10,
}


@pytest.fixture(autouse=True)
def _isolate_host_env(monkeypatch, tmp_path):
    iso = tmp_path / "isolated_home"
    iso.mkdir(parents=True, exist_ok=True)
    for var in (
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
        "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME", "MINDIE_DIAGNOSTICS_ROOT",
        "REMOTE_DEV_AUTH_DIR", "UV_CACHE_DIR", "TMPDIR",
    ):
        monkeypatch.setenv(var, str(iso))


def _git_repo_with_blob(tmp_path: Path, name: str, rel_path: str, content: bytes) -> tuple[Path, str, str]:
    repo = tmp_path / "refs" / name
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e.invalid"}
    for cmd in (["init", "-q"], ["add", rel_path], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env, capture_output=True)
    rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    return repo, rev, hashlib.sha256(content).hexdigest()


def _synthetic_repo(tmp_path: Path, *, family: str = "demo_op") -> Path:
    syn = tmp_path / f"syn_repo_{family}"
    for sub in ("scripts", "mindie_sim", "tests/operators"):
        (syn / sub).mkdir(parents=True)
    (syn / "mindie_sim/__init__.py").write_text('"""Stub."""\n', encoding="utf-8")
    (syn / "scripts/check_arch_governance.py").write_text("print('arch governance violations: 0')\n", encoding="utf-8")
    (syn / "scripts/check_test_orthogonality.py").write_text("print('summary: 0 findings (none); mode=--strict; modules=1/96')\n", encoding="utf-8")
    (syn / f"tests/operators/test_{family}.py").write_text('"""Test."""\nimport pytest\npytestmark = pytest.mark.tier0\n', encoding="utf-8")
    (syn / "tests/TAXONOMY.toml").write_text(
        'schema_version = 1\ndomains = ["operators"]\nmax_lines = 600\nmax_modules = 8\n'
        f'[[module]]\npath = "operators/test_{family}.py"\ntier = "tier0"\n'
        f'owns = ["mindie_sim/prediction/operators/{family}", "mindie_sim/templates/{family}_spec.py"]\n',
        encoding="utf-8",
    )
    return syn


def _synthetic_profiler_root(tmp_path: Path, supported: tuple[str, ...] = ("demo_op",), *, abi: str = "aclnnDemoOpV1", tag: str = "main") -> Path:
    prof = tmp_path / f"syn_profiler_{tag}"
    pkg = prof / "mindie_profiler/bench"
    pkg.mkdir(parents=True, exist_ok=True)
    (prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "operators.py").write_text(
        "class BenchUnmodeledOperatorError(RuntimeError):\n    pass\n"
        f"SUPPORTED = {set(supported)!r}\nABI = {abi!r}\n"
        "def resolve_builder_strict(family, recipe, impl, dtypes=(), meta=None):\n"
        "    if family in SUPPORTED or recipe in SUPPORTED:\n        return (f'{family}_builder', ABI)\n"
        "    raise BenchUnmodeledOperatorError(f'unmodeled:{family}:{recipe}')\n",
        encoding="utf-8",
    )
    return prof


def _lean_witness_row(case_id: str, work_ns: int, *, sim_ns: int, p99_ns: int) -> dict[str, Any]:
    prog = ResourceChoiceProgram(num_resources=1, global_work=(work_ns,), site_modes=())
    dag = EventDagSpec(node_quantum_ns=(work_ns,), edges=())
    cert, _ = certify_lower_bound(prog, dag)
    return {
        "case_id": case_id, "l_star_lean_ns": cert.l_star_lean_ns, "t_sim_ns": sim_ns, "t_silicon_p99_ns": p99_ns,
        "program": {"num_resources": 1, "global_work": [work_ns], "site_modes": []},
        "dag": {"node_quantum_ns": [work_ns], "edges": []},
        "certificate": {
            "farkas": {"scale_d": cert.farkas.scale_d, "lambda_num": list(cert.farkas.lambda_num), "mu_num": list(cert.farkas.mu_num), "lower_bound_ns": cert.farkas.lower_bound_ns},
            "path_edges": list(cert.path_edges), "sink_node_id": cert.sink_node_id, "l_star_lean_ns": cert.l_star_lean_ns,
        },
    }


def _independent_engine_runner(pred_by_id: dict[str, int], *, default_ns: int = 20_000, clock_op: str = "elapsed_span"):
    """Test runner whose predictions come strictly from an explicit lookup table (never case.measured_ns)."""
    table = dict(pred_by_id)

    class _Runner:
        def _exec(self, case: GateCase) -> EngineInterval:
            pred_ns = int(table.get(case.case_id, default_ns))
            eng = Engine([Resource("device", 1_000_000.0, "ns", "fifo")], trace_limit=32)
            s_ev = eng.submit(case.boundary.start_event, {"device": 100.0})
            e_ev = eng.submit(case.boundary.end_event, {"device": float(pred_ns)}, after=(s_ev,))
            eng.advance(until=(e_ev,))
            return bind_engine_interval(case, eng, s_ev, e_ev, diagnostics=(("clock_operator", clock_op), ("busy_union_ns", pred_ns), ("elapsed_span_ns", pred_ns)))

        def run_l1(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l2(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l3(self, case: GateCase) -> EngineInterval: return self._exec(case)

    return _Runner()


def _plan_fixture_bundle(
    plan: dict[str, Any], *, levels: tuple[str, ...] = ("L1", "L2", "L3"),
    clock_op: str = "elapsed_span", scale: float = 1.02,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, preds, lean_rows, sku = [], {}, [], str(plan["sku"])
    for idx, pt in enumerate(plan["train_points"]):
        pid, meas = str(pt["point_id"]), 10_000 + idx * 100
        sha = hashlib.sha256(f"train:{sku}:{pid}:{idx}".encode()).hexdigest()
        preds[pid] = int(round(meas * scale))
        rows.append({
            "case_id": pid, "point_id": pid, "sku": sku, "split": "train", "level": "L1",
            "acquisition_root": "acq-train-src", "run_uid": f"run-tr-{idx}", "raw_sha256": sha,
            "capture_uri": f"capture://train/{sha}", "event_ids": [f"tr-s-{idx}", f"tr-e-{idx}"], "device_ranks": [0],
            "context_len": pt["value"] if pt["axis"] == "context_len" else None,
            "tokens": pt["value"] if pt["axis"] == "tokens" else None,
            "batch_size": pt["batch_size"], "query_tokens": pt["query_tokens"],
            "parallel_topology": pt["parallel_topology"], "topk": pt["topk"],
            "measured_ns": meas, "clock_operator": clock_op,
            "busy_union_ns": meas if clock_op == "busy_union" else meas - 500,
            "elapsed_span_ns": meas if clock_op == "elapsed_span" else meas + 500,
        })
    for idx, pt in enumerate(plan["heldout_points"]):
        pid, lvl, meas = str(pt["point_id"]), levels[idx % len(levels)], 20_000 + idx * 250
        sha, pred = hashlib.sha256(f"held:{sku}:{pid}:{lvl}:{idx}".encode()).hexdigest(), int(round(meas * scale))
        preds[pid] = pred
        rows.append({
            "case_id": pid, "point_id": pid, "sku": sku, "split": "heldout", "level": lvl,
            "acquisition_root": f"acq-held-{lvl.lower()}", "run_uid": f"run-h-{lvl.lower()}-{idx}",
            "raw_sha256": sha, "capture_uri": f"capture://held/{lvl.lower()}/{sha}",
            "event_ids": [f"hd-s-{idx}", f"hd-e-{idx}"], "device_ranks": [0],
            "context_len": pt["value"] if pt["axis"] == "context_len" else None,
            "tokens": pt["value"] if pt["axis"] == "tokens" else None,
            "batch_size": pt["batch_size"], "query_tokens": pt["query_tokens"],
            "parallel_topology": pt["parallel_topology"], "topk": pt["topk"],
            "measured_ns": meas, "clock_operator": clock_op,
            "busy_union_ns": meas if clock_op == "busy_union" else meas - 500,
            "elapsed_span_ns": meas if clock_op == "elapsed_span" else meas + 500,
        })
        lean_rows.append(_lean_witness_row(pid, int(meas * 0.7), sim_ns=pred, p99_ns=int(meas * 1.05)))
    abi_rows = [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"]} for pt in plan["abi_rejection_points"]]
    return rows, preds, lean_rows, abi_rows


def _raw_l0_samples(sku: str, *, scale: float = 1.0, cann_version: str = "9.2.0") -> dict[str, Any]:
    return {
        "device_name": sku, "cann_version": cann_version,
        "P01_datacopy": {
            "hbm_read_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.72 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "hbm_write_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.70 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "mte2_aiv_hit_bytes_per_s_per_core": [A2_ORACLE["mte2_aiv"] * scale * (0.75 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
        },
        "P20_host_dma_sync": {
            "host_h2d_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "host_d2h_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
        },
    }


def _a2_probe_payload(sku: str = A2_SKU, *, endpoint: str = "lab-a2") -> dict[str, Any]:
    return {
        "aclrt_soc_name": sku, "endpoint_alias": endpoint, "dies_per_package": A2_ORACLE["dies"],
        "platform_config_ini": (
            f"[SoCInfo]\nsoc_version={sku}\ncompute_arch={A2_ORACLE['arch_id']}\n"
            f"ai_core_cnt={A2_ORACLE['aic']}\nvector_core_cnt={A2_ORACLE['aiv']}\n"
            f"l2_size={A2_ORACLE['l2']}\nub_size={A2_ORACLE['ub']}\n"
        ),
        "npu_smi_common": f"Chip ID : 0\nAicore Count : {A2_ORACLE['aic']}\nAicore Freq(MHZ) : 1650\n",
    }


def test_probe_soc_strict_ini_conflict_and_new_sku_run_all(tmp_path):
    synthetic_ip = ".".join(("198", "51", "100", "22"))
    ok_probe = tmp_path / "probe_ok.json"
    ok_probe.write_text(json.dumps(_a2_probe_payload(endpoint=f"lab-{synthetic_ip}-box")), encoding="utf-8")
    res_ok = probe_soc(probe_json=ok_probe)
    assert res_ok["ok"] is True and res_ok["mode"] == "validated_existing"
    assert res_ok["probe"]["capture_uri"].startswith("capture://") and len(res_ok["ini_sha256"]) == 64
    assert synthetic_ip not in json.dumps(res_ok)

    # Finding #13: existing SKU without platform_config_ini OR with dies conflict MUST fail closed!
    no_ini = tmp_path / "no_ini.json"
    no_ini.write_text(json.dumps({"aclrt_soc_name": A2_SKU, "aic_count": A2_ORACLE["aic"]}), encoding="utf-8")
    assert probe_soc(probe_json=no_ini)["error"] == "silicon_probe_incomplete"
    bad_dies = tmp_path / "bad_dies.json"
    bad_dies.write_text(json.dumps({**_a2_probe_payload(), "dies_per_package": 2}), encoding="utf-8")
    assert probe_soc(probe_json=bad_dies)["error"] == "silicon_profile_conflict"

    # Unknown SoC without full capabilities fails closed (NO silent B4 fallback).
    inc_probe = tmp_path / "inc.json"
    inc_probe.write_text(json.dumps({"aclrt_soc_name": "AscendFuture_9999", "aic_count": 32}), encoding="utf-8")
    assert probe_soc(probe_json=inc_probe, allow_scaffold=True)["error"] == "silicon_capabilities_incomplete"

    # Finding #12: Brand-new scaffolded SKU completes ALL 4 stages of run-all!
    new_sku = "AscendFuture_9999"
    full_probe = tmp_path / "probe_new.json"
    full_probe.write_text(json.dumps({
        "aclrt_soc_name": new_sku, "arch_preset": A2_ORACLE["arch_id"], "architecture": "A2",
        "aic_count": 28, "aiv_count": 56, "hbm_bytes": 64 * 1024**3,
        "l2_bytes": 200 * 1024**2, "ub_bytes": 196_608, "dies_per_package": 1, "ranks_per_node": 8,
    }), encoding="utf-8")
    new_plan = synthesize_sku_512_400k_plan(sku=new_sku, core_count=28, l2_bytes=200 * 1024**2).to_dict()
    obs, preds, leans, abis = _plan_fixture_bundle(new_plan)
    raw_l0 = tmp_path / "new_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(new_sku)), encoding="utf-8")
    run_new = run_sku_onboarding(
        probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU,
        profiler_runner=lambda _p: obs, abi_rejection_results=abis,
        runner=_independent_engine_runner(preds), lean_bounds_json=leans,
        repo_root=_synthetic_repo(tmp_path, family="new_sku"), out_dir=tmp_path / "new_sku_out",
    )
    assert run_new["ok"] is True and run_new["status"] == "certified" and run_new["sku"] == new_sku


def test_canonical_sealer_capture_unwrap_cann_and_sku_mismatch(tmp_path):
    assert not (REPO_ROOT / "mindie_sim/onboarding/_primitive_sealer.py").exists()
    with pytest.raises(ModelingError, match="timeout_budget_outer_too_small"):
        TimeoutBudget((("l0", 300), ("fabric", 200)), cleanup_grace_s=60, outer_timeout_s=500)
    uncal = calibrate_l0_and_fabric(sku=A2_SKU, endpoint_alias="lab-a2")
    assert uncal["ok"] is False and uncal["status"] == "uncalibrated" and uncal["calibrated_count"] == 0
    assert mindie_sim_main(["onboard-sku", "calibrate-l0-and-fabric", "--sku", A2_SKU]) == 2

    # Finding #11: Canonical {"tag","payload"} .capture.json unwraps cleanly!
    real_cap = REPO_ROOT / "data/primitives/captures/arch22_a3_9362.capture.json"
    cap_sku = json.loads(real_cap.read_text(encoding="utf-8"))["payload"]["device_name"]
    cal_cap = calibrate_l0_and_fabric(sku=cap_sku, raw_l0_json=real_cap, cann_version="9.2.0")
    assert cal_cap["ok"] is True and cal_cap["calibrated_count"] >= 3

    # Finding #9: Conflicting CANN, missing primitives_json, wrong-SKU table, and wrong expected CANN all fail!
    raw_l0 = tmp_path / "raw_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU, cann_version="9.2.0")), encoding="utf-8")
    with pytest.raises(ModelingError, match="cann_version_provenance_conflict"):
        calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_l0, cann_version="9.3.0")
    syn = _synthetic_repo(tmp_path, family="prim_check")
    plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    obs, preds, leans, abis = _plan_fixture_bundle(plan)
    val = validate_512_400k(sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, runner=_independent_engine_runner(preds))
    assert val["ok"] is True
    assert verify_sku_gates(sku=A2_SKU, validation_report_json=val, lean_bounds_json=leans, repo_root=syn)["ok"] is False
    assert verify_sku_gates(sku=A2_SKU, validation_report_json=val, lean_bounds_json=leans, primitives_json=cal_cap["primitives_payload"], repo_root=syn)["ok"] is False
    cal_a2 = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_l0, cann_version="9.2.0")
    assert verify_sku_gates(sku=A2_SKU, validation_report_json=val, lean_bounds_json=leans, primitives_json=cal_a2["primitives_payload"], expected_cann_version="9.3.0", repo_root=syn)["ok"] is False

    def _timeout_runner(_sku: str, _budget: int) -> dict[str, Any]:
        raise TimeoutError("l0_timeout")

    with pytest.raises(TimeoutError, match="l0_timeout"):
        calibrate_l0_and_fabric(sku=A2_SKU, l0_runner=_timeout_runner)


def test_lock3_tiling_axes_controls_grid_and_abi_rejection():
    # Default 512..400K disjoint Anchor + Cliff + 2D grid + topk=2049 routing.
    std = synthesize_operator_cliff_plan(
        OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 24, 192 * 1024 * 1024, 4096, topk_threshold=2048),
        sku=A2_SKU,
    )
    assert set(ANCHOR_CONTEXT_GRID) == {pt.value for pt in std.train_points if pt.axis == "context_len"}
    assert set(OFF_GRID_CLIFF_HELDOUT_GRID).issubset({pt.value for pt in std.heldout_points if pt.axis == "context_len"})
    assert any(pt.topk == 2049 for pt in std.abi_rejection_points) and not any(pt.topk == 2049 for pt in std.heldout_points)

    # Finding #20: Custom TilingAxes controls train filtering, B/q axes, and phase thresholds!
    axes = TilingAxes(
        tile_dims=(128,),
        batch_axis=(1, 3), query_tokens_axis=(1, 7),
        phase_thresholds=(("context_len", 12345),),
        abi_divisibility=(AbiDivisibilityRule(axis="context_len", above_threshold=0, divisor=3, rationale="ctx_div3"),),
    )
    plan = synthesize_operator_cliff_plan(
        OperatorCliffParameters("custom:v1", "custom", 128, 24, 192 * 1024 * 1024, 4096, tiling_axes=axes),
        sku=A2_SKU,
    )
    assert all(pt.value % 3 == 0 for pt in plan.train_points)
    assert 3 in {pt.batch_size for pt in plan.heldout_points} and 7 in {pt.query_tokens for pt in plan.heldout_points}
    assert {12344, 12345, 12346}.intersection({pt.value for pt in plan.heldout_points if pt.axis == "context_len"}) == {12345}
    assert {12344, 12346}.issubset({pt.value for pt in plan.abi_rejection_points})

    # Finding #21: evaluate_abi_rejections fails closed on missing, rejected=False, or wrong error class!
    rejs = plan.to_dict()["abi_rejection_points"]
    assert evaluate_abi_rejections(rejs, None)["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": "no_memory"} for pt in rejs])["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": False, "abi_error": pt["abi_expected_error"]} for pt in rejs])["ok"] is False


def test_lock2_sealed_scorer_adversarial_hardening(tmp_path):
    plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    obs, preds, leans, abis = _plan_fixture_bundle(plan)
    held_ids = [r["case_id"] for r in obs if r["split"] == "heldout"]
    p1, p2 = held_ids[0], held_ids[1]

    # 1. plan_only & threshold > 10% (Finding #6) MUST fail closed!
    assert validate_512_400k(sku=A2_SKU)["ok"] is False
    assert mindie_sim_main(["onboard-sku", "validate-512-400k", "--sku", A2_SKU]) == 2
    with pytest.raises(ModelingError, match="threshold_pct_exceeds_strict_release_cap"):
        validate_512_400k(sku=A2_SKU, threshold_pct=30.0)
    assert mindie_sim_main(["onboard-sku", "validate-512-400k", "--sku", A2_SKU, "--threshold-pct", "30"]) == 2

    # 2. Caller predicted_ns, clock label mismatch, AND clock value mismatch (Finding #3)!
    with pytest.raises(ModelingError, match="caller_predicted_duration_forbidden"):
        score_sealed_gate_cases([{**obs[-1], "predicted_ns": 20_000}], sku=A2_SKU, runner=_independent_engine_runner(preds))
    with pytest.raises(ModelingError, match="clock_operator_mismatch"):
        score_sealed_gate_cases(obs[:2], sku=A2_SKU, runner=_independent_engine_runner(preds, clock_op="busy_union"))
    bad_clk_val = {**obs[-1], "clock_operator": "elapsed_span", "elapsed_span_ns": 30_000, "busy_union_ns": 10_000, "measured_ns": 10_000}
    with pytest.raises(ModelingError, match="clock_operator_measured_value_mismatch"):
        score_sealed_gate_cases([bad_clk_val], sku=A2_SKU, runner=_independent_engine_runner(preds))

    # 3. Finding #4: Single row cannot satisfy 2 cells & coordinate mismatch fails closed!
    alias_row = {**next(r for r in obs if r["case_id"] == p2), "case_id": p1, "point_id": p2}
    res_alias = score_sealed_gate_cases([alias_row], sku=A2_SKU, runner=_independent_engine_runner(preds), expected_point_ids=[p1, p2])
    assert res_alias["ok"] is False and p1 in res_alias["missing_plan_points"]
    bad_coord = [dict(r, batch_size=999) if r["case_id"] == p1 else r for r in obs]
    with pytest.raises(ModelingError, match="observation_plan_coordinate_mismatch"):
        validate_512_400k(sku=A2_SKU, profiler_runner=lambda _p: bad_coord, abi_rejection_results=abis, runner=_independent_engine_runner(preds))

    # 4. Finding #5: Training digest reuse under fresh acquisition_root & missing event_ids fail closed!
    reused_sha = [dict(r, raw_sha256=obs[0]["raw_sha256"]) if r["case_id"] == p1 else r for r in obs]
    with pytest.raises(ModelingError, match="train_heldout_capture_digest_overlap"):
        validate_512_400k(sku=A2_SKU, profiler_runner=lambda _p: reused_sha, abi_rejection_results=abis, runner=_independent_engine_runner(preds))
    with pytest.raises(ModelingError, match="observation_event_ids_missing"):
        score_sealed_gate_cases([{k: v for k, v in obs[-1].items() if k != "event_ids"}], sku=A2_SKU, runner=_independent_engine_runner(preds))

    # 5. Finding #21, [0%, 0%, 27%] outlier + diagnose_tail_cluster, and service_regimes rejection!
    assert validate_512_400k(sku=A2_SKU, profiler_runner=lambda _p: obs, runner=_independent_engine_runner(preds))["ok"] is False
    bad_preds = {**preds, p1: int(next(r["measured_ns"] for r in obs if r["case_id"] == p1) * 1.27)}
    val_fail = validate_512_400k(sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, runner=_independent_engine_runner(bad_preds))
    assert val_fail["ok"] is False and p1 in val_fail["failing_cases"]
    raw_l0 = tmp_path / "l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_l0)
    gate_fail = verify_sku_gates(sku=A2_SKU, validation_report_json=val_fail, primitives_json=cal["primitives_payload"], lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="tail_diag"))
    assert gate_fail["ok"] is False and gate_fail["diagnosis_triggered"] is True
    with pytest.raises(ModelingError, match="whitebox_service_regimes_forbidden"):
        score_sealed_gate_cases([{**obs[-1], "service_regimes": {"a": 1}}], sku=A2_SKU, runner=_independent_engine_runner(preds))


def test_fabricated_reports_joined_lean_and_positive_sku_certify(tmp_path):
    syn = _synthetic_repo(tmp_path, family="fab_chk")
    # Finding #1: Fabricated SKU & operator reports MUST fail closed!
    fake_val = {"sku": "NonExistent_9999", "ok": True, "residuals": [{"case_id": f"c{i}", "split": "heldout", "level": lv, "acquisition_root": f"r-{lv}"} for i, lv in enumerate(("L1", "L2", "L3"))]}
    assert verify_sku_gates(sku="NonExistent_9999", validation_report_json=fake_val, lean_bounds_json=[_lean_witness_row("unrelated", 1, sim_ns=1, p99_ns=1)], repo_root=syn, run_repo_gates=False)["ok"] is False
    assert certify_operator(operator_id="fake:v1", sku=A2_SKU, contract_report={"ok": True}, score_report={"ok": False, "sku": A2_SKU, "residuals": [{}], "heldout_summary": {"max_ape_pct": 0.0}}, lean_bounds_json=[_lean_witness_row("unrelated", 1, sim_ns=1, p99_ns=1)], repo_root=syn, run_repo_gates=False)["ok"] is False

    # Finding #7: Unrelated Lean case_id & bytes-as-time bug fixed!
    ok_u, _, v_u = check_lean_non_inversion([_lean_witness_row("unrelated", 10, sim_ns=20, p99_ns=20)], expected_case_timings={"real-case": (20_000, 20_000)})
    assert ok_u is False and any("lean_witness_unrelated_case_id" in v for v in v_u)
    ctx = _build_default_planning_context(OpSpec("op0", "demo", "demo", "decode", "fp16", (), (1, 64), (1, 64), (1, 64), (1, 64)))
    dual_ok, bound_ns, _ = _verify_resolved_operator_dual_bound(SimpleNamespace(recipe=SimpleNamespace(work=Work(resources={"hbm": 4096.0}))), ctx)
    assert dual_ok is True and 1 <= bound_ns < 100

    # Positive end-to-end 4-stage SKU certification on A2_SKU!
    plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    obs, preds, leans, abis = _plan_fixture_bundle(plan)
    probe_f = tmp_path / "a2_probe.json"
    probe_f.write_text(json.dumps(_a2_probe_payload()), encoding="utf-8")
    raw_l0 = tmp_path / "a2_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    res = run_sku_onboarding(sku=A2_SKU, probe_json=probe_f, raw_l0_json=raw_l0, profiler_runner=lambda _p: obs, abi_rejection_results=abis, runner=_independent_engine_runner(preds), lean_bounds_json=leans, repo_root=syn, out_dir=tmp_path / "sku_ok")
    assert res["ok"] is True and res["status"] == "certified" and verify_receipt_sha256(res["stages"]["verify-gates"]["receipt"]) is True


def test_operator_pipeline_strict_owners_abi_crosscheck_and_certify(tmp_path):
    repo_dir, rev, sha = _git_repo_with_blob(tmp_path, "ops-transformer", "kernels/demo_op.cpp", b"// demo\n")
    syn = _synthetic_repo(tmp_path, family="demo_op")
    prof_ok = _synthetic_profiler_root(tmp_path, supported=("demo_op",), abi="aclnnDemoOpV1", tag="ok")
    prof_bad = _synthetic_profiler_root(tmp_path, supported=("demo_op",), abi="aclnnMatmulReduceScatter", tag="bad_abi")
    scaf = scaffold_operator(family="demo_op", upstream_abi_arch22="aclnnDemoOpV1", upstream_abi_arch35="aclnnDemoOpV2", source_refs=[{"repository": "ops-transformer", "revision": rev, "relative_path": "kernels/demo_op.cpp", "source_sha256": sha}], target_root=syn, overwrite=True)

    # Finding #19: Module-level constant revision resolves & unmounted refs_root fails closed!
    arch22 = syn / "mindie_sim/prediction/operators/demo_op/arch22.py"
    arch22.write_text(arch22.read_text(encoding="utf-8").replace(f'revision="{rev}"', "revision=DEMO_REV").replace("from __future__ import annotations\n", f'from __future__ import annotations\nDEMO_REV = "{rev}"\n'), encoding="utf-8")
    target_rel = scaf["package_rel"]

    def _good_resolve(op: OpSpec, ctx) -> ResolvedOperator:
        return ResolvedOperator(occurrence=ctx.occurrence, op=op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(work=Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: (), pipeline_spec={"stage": "v"}, metadata={"implementation": "aclnnDemoOpV1"}))

    def _empty_resolve(op: OpSpec, ctx) -> ResolvedOperator:
        return ResolvedOperator(occurrence=ctx.occurrence, op=op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(work=Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: ()))

    reg_ok = FamilyRegistry([OperatorFamilySpec("demo_op", frozenset({"demo_op"}), frozenset({"demo_op"}), _good_resolve)])
    reg_empty = FamilyRegistry([OperatorFamilySpec("demo_op", frozenset({"demo_op"}), frozenset({"demo_op"}), _empty_resolve)])

    # Findings #17, #18, #19:
    assert validate_operator_contract(target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_ok, family_registry=reg_empty)["ok"] is False
    bad_abi = validate_operator_contract(target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_bad, family_registry=reg_ok)
    assert bad_abi["ok"] is False and any("profiler_abi_mismatch" in v for v in bad_abi["violations"])
    unmounted = validate_operator_contract(target_path=target_rel, repo_root=syn, refs_root=tmp_path / "empty_refs", profiler_root=prof_ok, family_registry=reg_ok)
    assert unmounted["ok"] is False and any("source_repo_not_mounted" in v for v in unmounted["violations"])

    # Positive 4-stage operator certification + Lean inversion rejection!
    plan_doc = generate_operator_cliff_plan(operator_id="demo_op:v1", family="demo_op", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048)["plan"]
    op_obs, op_preds, op_leans, op_abis = _plan_fixture_bundle(plan_doc, levels=("L1",))
    op_run = run_operator_onboarding(
        target_path=target_rel, operator_id="demo_op:v1", family="demo_op", sku=A2_SKU,
        tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048,
        captured_json=op_obs, abi_rejection_results=op_abis, lean_bounds_json=op_leans,
        repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_ok,
        family_registry=reg_ok, runner=_independent_engine_runner(op_preds), out_dir=tmp_path / "op_out",
    )
    assert op_run["ok"] is True and op_run["status"] == "certified" and verify_receipt_sha256(op_run["stages"]["certify"]["receipt"]) is True
    inv_leans = [dict(op_leans[0], l_star_lean_ns=op_leans[0]["t_sim_ns"] + 5000), *op_leans[1:]]
    assert certify_operator(operator_id="demo_op:v1", sku=A2_SKU, contract_report=op_run["stages"]["validate-contract"], score_report=op_run["stages"]["bench-and-score"], lean_bounds_json=inv_leans, repo_root=syn)["ok"] is False


def test_live_coordinator_schema_device_forwarding_and_profiler_module(tmp_path):
    # Findings #14, #15, #16: Real mindie.lease wire schema (--json), granted device ID [3], and -m profiler module!
    log_file = tmp_path / "coord.jsonl"
    fake_coord = tmp_path / "fake_coord.py"
    fake_coord.write_text(
        f"#!{sys.executable}\nimport json, sys\nargs = sys.argv[1:]\nact = args[args.index('--action') + 1]\n"
        "req = json.loads(args[args.index('--json') + 1])\n"
        "if act == 'acquire':\n"
        "    assert 'npu_count' in req and 'ttl_seconds' in req and 'ttl_s' not in req\n"
        "    out = {'outcome': 'success', 'status': 'ready', 'data': {'lease_id': 'lq-9', 'fence_token': 77, 'devices': [3]}}\n"
        "else:\n"
        "    assert req.get('lease_id') == 'lq-9' and req.get('fence_token') == 77\n"
        "    out = {'outcome': 'success', 'status': 'released', 'data': {'released': True}}\n"
        f"open({str(log_file)!r}, 'a', encoding='utf-8').write(act + '\\n')\nprint(json.dumps(out))\n",
        encoding="utf-8",
    )
    fake_coord.chmod(0o755)
    with pytest.raises(RuntimeError, match="boom"):
        with acquire_coordinator_host_queue_lease("lab-a2", coordinator_bin=str(fake_coord)) as lease:
            assert lease["lease_id"] == "lq-9" and lease["fence_token"] == 77 and lease["devices"] == [3]
            raise RuntimeError("boom")
    assert log_file.read_text(encoding="utf-8").splitlines() == ["acquire", "release"]

    prof_argv_log = tmp_path / "prof_argv.json"
    fake_prof = tmp_path / "fake_prof_root"
    pkg = fake_prof / "mindie_profiler/bench"
    pkg.mkdir(parents=True)
    (fake_prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "bench_sku_microbench.py").write_text(
        f"import json, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\nPath({str(prof_argv_log)!r}).write_text(json.dumps(args), encoding='utf-8')\n"
        "Path(args[args.index('--observations') + 1]).write_text(json.dumps({'observations': []}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    plan_doc = generate_operator_cliff_plan(operator_id="live:v1", family="live", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"])["plan"]
    bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, endpoint_alias="lab-a2", profiler_root=fake_prof, coordinator_bin=str(fake_coord))
    argv_seen = json.loads(prof_argv_log.read_text(encoding="utf-8"))
    assert argv_seen[argv_seen.index("--allowed-phy-ids") + 1] == "3" and argv_seen[argv_seen.index("--sku") + 1] == A2_SKU
    assert log_file.read_text(encoding="utf-8").splitlines() == ["acquire", "release", "acquire", "release"]

    done_sku = subprocess.run([sys.executable, "-B", str(REPO_ROOT / "tools/onboard_sku.py"), "calibrate-l0-and-fabric", "--sku", A2_SKU], capture_output=True, text=True, check=False)
    assert done_sku.returncode == 2 and json.loads(done_sku.stdout)["status"] == "uncalibrated"


def test_whitebox_spine_runner_l1_l2_l3_separation_and_p50_threading(tmp_path):
    # Findings #2, #3, #8, #10: Real WhiteboxSpineRunner separates L1 < L2 < L3, honors busy_union, and threads Stage-2 p50!
    real_case = {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    base_rows = [
        {"case_id": "wb-l1", "point_id": "wb-l1", "sku": A2_SKU, "split": "heldout", "level": "L1", "acquisition_root": "acq-wb-l1", "run_uid": "r1", "raw_sha256": hashlib.sha256(b"wb-l1").hexdigest(), "event_ids": ["s1", "e1"], "device_ranks": [0], "measured_ns": 2500, "clock_operator": "busy_union", "busy_union_ns": 2500, "elapsed_span_ns": 2600, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "target_family": "rmsnorm", "sim_case": real_case},
        {"case_id": "wb-l2", "point_id": "wb-l2", "sku": A2_SKU, "split": "heldout", "level": "L2", "acquisition_root": "acq-wb-l2", "run_uid": "r2", "raw_sha256": hashlib.sha256(b"wb-l2").hexdigest(), "event_ids": ["s2", "e2"], "device_ranks": [0], "measured_ns": 260_000, "clock_operator": "busy_union", "busy_union_ns": 260_000, "elapsed_span_ns": 265_000, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "target_layer": 0, "sim_case": real_case},
        {"case_id": "wb-l3", "point_id": "wb-l3", "sku": A2_SKU, "split": "heldout", "level": "L3", "acquisition_root": "acq-wb-l3", "run_uid": "r3", "raw_sha256": hashlib.sha256(b"wb-l3").hexdigest(), "event_ids": ["s3", "e3"], "device_ranks": [0], "measured_ns": 7_500_000, "clock_operator": "busy_union", "busy_union_ns": 7_500_000, "elapsed_span_ns": 7_600_000, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "sim_case": real_case},
    ]
    res_base = score_sealed_gate_cases(base_rows, sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    by_lvl = {r["level"]: r for r in res_base["residuals"]}
    assert 0 < by_lvl["L1"]["predicted_ns"] < by_lvl["L2"]["predicted_ns"] < by_lvl["L3"]["predicted_ns"]
    assert 0 < by_lvl["L3"]["engine_busy_union_ns"] <= by_lvl["L3"]["engine_elapsed_span_ns"]

    raw_full, raw_half = tmp_path / "full.json", tmp_path / "half.json"
    raw_full.write_text(json.dumps(_raw_l0_samples(A2_SKU, scale=1.0)), encoding="utf-8")
    raw_half.write_text(json.dumps(_raw_l0_samples(A2_SKU, scale=0.5)), encoding="utf-8")
    tbl_full = parse_primitive_table(calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_full)["primitives_payload"], verify_envelope=True)
    tbl_half = parse_primitive_table(calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_half)["primitives_payload"], verify_envelope=True)
    p_full = score_sealed_gate_cases([base_rows[2]], sku=A2_SKU, primitive_table=tbl_full)["residuals"][0]["predicted_ns"]
    p_half = score_sealed_gate_cases([base_rows[2]], sku=A2_SKU, primitive_table=tbl_half)["residuals"][0]["predicted_ns"]
    assert p_half > p_full

    # Finding #8: verify_sku_gates(multilevel_manifest_json=..., runner=None) replays via default WhiteboxSpineRunner!
    manifest_f = tmp_path / "ml_manifest.json"
    manifest_f.write_text(json.dumps({
        "manifest_id": "m-live-replay", "schema_version": "mindie-multilevel-manifest/1",
        "cases": [
            {
                "case_id": r["case_id"], "level": r["level"], "split": "heldout",
                "boundary": {
                    "raw_sha256": r["raw_sha256"], "acquisition_uid": r["acquisition_root"],
                    "run_uid": r["run_uid"], "parent_digests": [], "event_ids": r["event_ids"],
                    "device_ranks": r["device_ranks"], "clock_domain": "device_ns",
                    "start_event": "start", "end_event": "end",
                    "boundary_kind": {"L1": "device_kernel", "L2": "fx_region", "L3": "model_step"}[r["level"]],
                    "unit": "ns", "capture_uri": f"capture://ml/{r['raw_sha256']}",
                },
                "input_digest": r["raw_sha256"], "stack_digest": hashlib.sha256(b"st").hexdigest(),
                "implementation": "whitebox",
                "cohort": {"level": r["level"], "cohort_name": "wb", "sku": A2_SKU, "phase": "decode", "mode": "eager", "shape_regime": "cliff", "parallel_topology": "tp1"},
                "measured_ns": by_lvl[r["level"]]["predicted_ns"],
                "payload": {k: v for k, v in r.items() if k in {"clock_operator", "phase", "context_len", "query_tokens", "batch_size", "target_family", "target_layer", "sim_case"}},
            }
            for r in base_rows
        ],
    }), encoding="utf-8")
    ml_leans = [_lean_witness_row(r["case_id"], max(1, by_lvl[r["level"]]["predicted_ns"] // 2), sim_ns=by_lvl[r["level"]]["predicted_ns"], p99_ns=by_lvl[r["level"]]["predicted_ns"] + 10) for r in base_rows]
    ml_gate = verify_sku_gates(
        sku=A2_SKU, multilevel_manifest_json=manifest_f, runner=None,
        primitives_json=calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_full)["primitives_payload"],
        lean_bounds_json=ml_leans, repo_root=_synthetic_repo(tmp_path, family="ml_live"),
    )
    assert ml_gate["ok"] is True and ml_gate["status"] == "certified"
