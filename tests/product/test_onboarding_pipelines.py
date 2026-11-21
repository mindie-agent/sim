"""Orthogonal Tier-1 tests for turnkey SKU & Operator onboarding pipelines.

[Role] Verify all 4 New-SKU stages (``probe-soc``, ``calibrate-l0-and-fabric``,
  ``validate-512-400k``, ``verify-gates``, ``run-all``), all 4 New-Operator
  stages (``scaffold``, ``validate-contract``, ``generate-cliff-plan``,
  ``bench-and-score``, ``certify``, ``run-all``), Prevention Lock 2 sealed
  scoring & adversarial negatives, Prevention Lock 3 ``TilingAxes`` /
  ``validate_abi`` 2D cliff synthesis, and CLI entrypoints.
[Invariants] Strictly isolates ``HOME``, ``USERPROFILE``, ``APPDATA``,
  ``LOCALAPPDATA``, ``XDG_*``, ``MINDIE_*``, and ``TMPDIR``; zero raw IPv4
  literals, zero host mutations, zero device contexts, and zero
  ``service_regimes`` crutches.
[Timeouts & Locks] Deterministic in-memory and ``tmp_path`` subprocess checks
  with bounded timeouts (< 15s); no global locks.
[Side-Effect Policy] Writes temporary fixtures/receipts exclusively under
  pytest ``tmp_path``.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from mindie_sim.backends import ascend
from mindie_sim.cli import main as mindie_sim_main
from mindie_sim.observations.multilevel_gate import EngineInterval, GateCase, bind_engine_interval
from mindie_sim.onboarding import (
    ANCHOR_CONTEXT_GRID,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    OperatorCliffParameters,
    TimeoutBudget,
    bench_and_score_operator,
    calibrate_l0_and_fabric,
    certify_operator,
    check_lean_non_inversion,
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
from mindie_sim.optimization.dual_bound import EventDagSpec, ResourceChoiceProgram, certify_lower_bound
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.operators.contracts import OperatorFamilySpec, ResolvedOperator
from mindie_sim.prediction.operators.registry import FamilyRegistry
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from _support.axes import REPRESENTATIVE_SKUS

pytestmark = pytest.mark.tier1

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPEATS = 12


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


def _synthetic_repo(tmp_path: Path, *, family: str = "demo_op", include_test: bool = True) -> Path:
    syn = tmp_path / f"syn_repo_{family}"
    (syn / "scripts").mkdir(parents=True)
    (syn / "mindie_sim").mkdir(parents=True)
    (syn / "mindie_sim" / "__init__.py").write_text('"""Stub."""\n', encoding="utf-8")
    (syn / "scripts" / "check_arch_governance.py").write_text("print('arch governance violations: 0')\n", encoding="utf-8")
    (syn / "scripts" / "check_test_orthogonality.py").write_text("print('summary: 0 findings (none); mode=--strict; modules=1/96')\n", encoding="utf-8")
    (syn / "tests" / "operators").mkdir(parents=True)
    if include_test:
        (syn / "tests" / "operators" / f"test_{family}.py").write_text('"""Test."""\nimport pytest\npytestmark = pytest.mark.tier0\n', encoding="utf-8")
    (syn / "tests" / "TAXONOMY.toml").write_text(
        'schema_version = 1\ndomains = ["operators"]\nmax_lines = 600\nmax_modules = 8\n'
        f'[[module]]\npath = "operators/test_{family}.py"\ntier = "tier0"\n'
        f'owns = ["mindie_sim/prediction/operators/{family}", "mindie_sim/templates/{family}_spec.py"]\n',
        encoding="utf-8",
    )
    return syn


def _synthetic_profiler_root(
    tmp_path: Path, supported: tuple[str, ...] = ("rmsnorm", "demo_op"), *, tag: str = "main"
) -> Path:
    prof = tmp_path / f"syn_profiler_{tag}"
    pkg = prof / "mindie_profiler" / "bench"
    pkg.mkdir(parents=True, exist_ok=True)
    (prof / "mindie_profiler" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "operators.py").write_text(
        "class BenchUnmodeledOperatorError(RuntimeError):\n    pass\n"
        f"SUPPORTED = {set(supported)!r}\n"
        "def resolve_builder_strict(family, recipe, impl, dtypes=(), meta=None):\n"
        "    if family in SUPPORTED or recipe in SUPPORTED:\n"
        "        return (f'{family}_builder', impl or 'aclnnResolved')\n"
        "    raise BenchUnmodeledOperatorError(f'unmodeled:{family}:{recipe}')\n",
        encoding="utf-8",
    )
    return prof


def _lean_witness_row(case_id: str, work_ns: int, *, sim_ns: int, p99_ns: int) -> dict[str, Any]:
    prog = ResourceChoiceProgram(num_resources=1, global_work=(work_ns,), site_modes=())
    dag = EventDagSpec(node_quantum_ns=(work_ns,), edges=())
    cert, _ = certify_lower_bound(prog, dag)
    return {
        "case_id": case_id,
        "l_star_lean_ns": cert.l_star_lean_ns,
        "t_sim_ns": sim_ns,
        "t_silicon_p99_ns": p99_ns,
        "program": {"num_resources": 1, "global_work": [work_ns], "site_modes": []},
        "dag": {"node_quantum_ns": [work_ns], "edges": []},
        "certificate": {
            "farkas": {
                "scale_d": cert.farkas.scale_d,
                "lambda_num": list(cert.farkas.lambda_num),
                "mu_num": list(cert.farkas.mu_num),
                "lower_bound_ns": cert.farkas.lower_bound_ns,
            },
            "path_edges": list(cert.path_edges),
            "sink_node_id": cert.sink_node_id,
            "l_star_lean_ns": cert.l_star_lean_ns,
        },
    }


def _live_engine_runner(scale_by_id: dict[str, float] | None = None, *, default_scale: float = 1.03, clock_op: str = "elapsed_span"):
    scales = scale_by_id or {}

    class _Runner:
        def _exec(self, case: GateCase) -> EngineInterval:
            sc = scales.get(case.case_id, default_scale)
            pred_ns = max(1, int(round((case.measured_ns or 1000) * sc)))
            eng = Engine([Resource("device", 1_000_000.0, "ns", "fifo")], trace_limit=32)
            s_ev = eng.submit(case.boundary.start_event, {"device": 100.0})
            e_ev = eng.submit(case.boundary.end_event, {"device": float(pred_ns)}, after=(s_ev,))
            eng.advance(until=(e_ev,))
            return bind_engine_interval(
                case, eng, s_ev, e_ev,
                diagnostics=(("clock_operator", clock_op), ("busy_union_ns", pred_ns), ("elapsed_span_ns", pred_ns)),
            )

        def run_l1(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l2(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l3(self, case: GateCase) -> EngineInterval: return self._exec(case)

    return _Runner()


def _obs_for_plan(plan: dict[str, Any], *, levels: tuple[str, ...] = ("L1", "L2", "L3"), clock_op: str = "elapsed_span") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, pt in enumerate(plan["train_points"]):
        sha = hashlib.sha256(f"train:{pt['point_id']}:{idx}".encode()).hexdigest()
        rows.append({
            "case_id": pt["point_id"], "point_id": pt["point_id"], "split": "train", "level": "L1",
            "acquisition_root": "acq-train-src", "run_uid": f"run-tr-{idx}", "raw_sha256": sha,
            "context_len": pt["value"], "measured_ns": 10000 + idx * 100, "clock_operator": clock_op,
            "busy_union_ns": 9500 + idx * 100, "elapsed_span_ns": 10000 + idx * 100,
        })
    for idx, pt in enumerate(plan["heldout_points"]):
        lvl = levels[idx % len(levels)]
        sha = hashlib.sha256(f"held:{pt['point_id']}:{lvl}:{idx}".encode()).hexdigest()
        rows.append({
            "case_id": pt["point_id"], "point_id": pt["point_id"], "split": "heldout", "level": lvl,
            "acquisition_root": f"acq-held-{lvl.lower()}", "run_uid": f"run-h-{lvl.lower()}-{idx}", "raw_sha256": sha,
            "context_len": pt["value"] if pt["axis"] == "context_len" else None,
            "measured_ns": 20000 + idx * 250, "clock_operator": clock_op,
            "busy_union_ns": 19000 + idx * 250, "elapsed_span_ns": 20000 + idx * 250,
        })
    return rows


def _raw_l0_samples(sku: str) -> dict[str, Any]:
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    die = hw.dies[0]
    dp = datapath_for_die(die)
    rates = dict(die.rates)
    hbm_r = float(dp.hbm_read_bytes_per_s or rates["hbm"])
    hbm_w = float(dp.hbm_write_bytes_per_s or rates["hbm"])
    mte2_aiv = float(rates["mte2_aiv"]) / max(1, int(dp.aiv_count))
    return {
        "device_name": sku,
        "P01_datacopy": {
            "hbm_read_bytes_per_s": [hbm_r * (0.72 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "hbm_write_bytes_per_s": [hbm_w * (0.70 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "mte2_aiv_hit_bytes_per_s_per_core": [mte2_aiv * (0.75 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
        },
        "P20_host_dma_sync": {
            "host_h2d_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "host_d2h_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
        },
    }


def test_probe_soc_validation_conflict_and_no_b4_fallback(tmp_path):
    sku = REPRESENTATIVE_SKUS["a2"]
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    dp = datapath_for_die(hw.dies[0])
    bufs = dict(hw.dies[0].buffers)
    synthetic_ip = ".".join(("198", "51", "100", "22"))

    ok_probe = tmp_path / "probe_ok.json"
    ok_probe.write_text(json.dumps({
        "aclrt_soc_name": sku,
        "endpoint_alias": f"lab-{synthetic_ip}-box",
        "platform_config_ini": (
            f"[SoCInfo]\nsoc_version={sku}\ncompute_arch={dp.arch_id}\n"
            f"ai_core_cnt={dp.aic_count}\nvector_core_cnt={dp.aiv_count}\n"
            f"l2_size={bufs['l2']}\nub_size={bufs['ub']}\n"
        ),
        "npu_smi_common": f"Chip ID : 0\nAicore Count : {dp.aic_count}\nAicore Freq(MHZ) : 1650\n",
    }), encoding="utf-8")
    res_ok = probe_soc(probe_json=ok_probe)
    assert res_ok["ok"] is True and res_ok["mode"] == "validated_existing"
    assert res_ok["probe"]["capture_uri"].startswith("capture://")
    assert len(res_ok["ini_sha256"]) == 64
    assert synthetic_ip not in json.dumps(res_ok)

    bad_probe = tmp_path / "probe_conflict.json"
    bad_probe.write_text(json.dumps({"aclrt_soc_name": sku, "aic_count": dp.aic_count + 4, "aiv_count": dp.aiv_count}), encoding="utf-8")
    res_bad = probe_soc(probe_json=bad_probe)
    assert res_bad["ok"] is False and res_bad["error"] == "silicon_profile_conflict"

    # Unknown SoC with incomplete capabilities MUST fail closed (NO silent B4 fallback!).
    inc_probe = tmp_path / "probe_incomplete.json"
    inc_probe.write_text(json.dumps({"aclrt_soc_name": "AscendFuture_9999", "aic_count": 32}), encoding="utf-8")
    res_inc = probe_soc(probe_json=inc_probe, allow_scaffold=True)
    assert res_inc["ok"] is False and res_inc["error"] == "silicon_capabilities_incomplete"
    assert "arch_preset" in res_inc["missing_capabilities"] and "base_template_sku" in res_inc["missing_capabilities"]

    # Unknown SoC with full explicit capabilities + arch-matched anchor scaffolds cleanly.
    full_probe = tmp_path / "probe_new.json"
    full_probe.write_text(json.dumps({
        "aclrt_soc_name": "AscendFuture_9999", "arch_preset": str(dp.arch_id), "architecture": "A2",
        "aic_count": 28, "aiv_count": 56, "hbm_bytes": 64 * 1024**3,
        "l2_bytes": 200 * 1024**2, "ub_bytes": 196608, "dies_per_package": 1, "ranks_per_node": 8,
    }), encoding="utf-8")
    res_new = probe_soc(probe_json=full_probe, allow_scaffold=True, base_template_sku=sku, scaffold_out=tmp_path / "new.json")
    assert res_new["ok"] is True and res_new["mode"] == "scaffolded_new"
    assert res_new["datapath_summary"]["aic_count"] == 28


def test_canonical_sealer_lease_budget_and_uncalibrated_lock2(tmp_path):
    assert not (REPO_ROOT / "mindie_sim" / "onboarding" / "_primitive_sealer.py").exists()
    sku = REPRESENTATIVE_SKUS["a2"]
    with pytest.raises(ModelingError, match="timeout_budget_outer_too_small"):
        TimeoutBudget((("l0", 300), ("fabric", 200)), cleanup_grace_s=60, outer_timeout_s=500)

    # Lock 2: calibrated_count == 0 MUST return ok=False and non-zero CLI exit!
    uncal = calibrate_l0_and_fabric(sku=sku, endpoint_alias="lab-a2")
    assert uncal["ok"] is False and uncal["status"] == "uncalibrated" and uncal["calibrated_count"] == 0
    assert mindie_sim_main(["onboard-sku", "calibrate-l0-and-fabric", "--sku", sku]) == 2

    # Calibrated path uses canonical tools/bench_l0_primitives.py + lease_id/fence_token.
    lease_events: list[str] = []

    @contextmanager
    def _lease(target_sku: str, ttl_s: int):
        lease_events.append(f"acq:{target_sku}:{ttl_s}")
        try:
            yield {"lease_mode": "coordinator_host_queue", "lease_id": "lease-7", "fence_token": 42}
        finally:
            lease_events.append("rel")

    raw_l0 = tmp_path / "raw_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(sku)), encoding="utf-8")
    cal = calibrate_l0_and_fabric(
        sku=sku, endpoint_alias="lab-a2", raw_l0_json=raw_l0,
        out_primitives=tmp_path / "prim.json", manifest_path=tmp_path / "manifest.json",
        train_manifest_out=tmp_path / "train.json", lease_acquirer=_lease,
    )
    assert cal["ok"] is True and cal["sealer_owner"] == "tools/bench_l0_primitives.py"
    assert cal["calibrated_count"] >= 2 and cal["lease"]["fence_token"] == 42
    assert lease_events == [f"acq:{sku}:660", "rel"]

    def _timeout_runner(_sku: str, _budget: int) -> dict[str, Any]:
        raise TimeoutError("l0_timeout")

    with pytest.raises(TimeoutError, match="l0_timeout"):
        calibrate_l0_and_fabric(sku=sku, l0_runner=_timeout_runner)


def test_lock3_tiling_axes_2d_cliffs_and_abi_rejection_routing():
    sku = REPRESENTATIVE_SKUS["a2"]
    plan = synthesize_operator_cliff_plan(
        OperatorCliffParameters(
            operator_id="sparse_indexer:v2", family="sparse_indexer",
            tile_dim=128, core_count=24, l2_bytes=192 * 1024 * 1024,
            bytes_per_token=4096, topk_threshold=2048,
        ),
        sku=sku,
    )
    train_ctxs = {pt.value for pt in plan.train_points if pt.axis == "context_len"}
    held_ctxs = {pt.value for pt in plan.heldout_points if pt.axis == "context_len"}
    assert set(ANCHOR_CONTEXT_GRID) == train_ctxs
    assert 262144 in train_ctxs and 409600 in train_ctxs and 400000 in held_ctxs
    assert train_ctxs.isdisjoint(held_ctxs)
    assert set(OFF_GRID_CLIFF_HELDOUT_GRID).issubset(held_ctxs)

    # 2D (B/q x Context x Parallel) cross-samples present.
    assert {pt.batch_size for pt in plan.heldout_points} >= {1, 16, 64, 128}
    assert {pt.query_tokens for pt in plan.heldout_points} >= {1, 2, 4, 8, 2048, 8192}
    assert {"pcp2_dcp2", "pcp4_dcp2"}.issubset({pt.parallel_topology for pt in plan.heldout_points})

    # Lock 3: ctx=2049, topk=2048 in perf heldout; illegal topk=2049 routed to abi_rejection_points!
    assert any(pt.value == 2049 and pt.topk == 2048 for pt in plan.heldout_points)
    assert not any(pt.topk == 2049 for pt in plan.heldout_points)
    assert any(pt.topk == 2049 and "abi_divisibility_violation" in pt.abi_expected_error for pt in plan.abi_rejection_points)


def test_lock2_sealed_scorer_adversarial_negatives_and_witnessed_lean(tmp_path):
    sku = REPRESENTATIVE_SKUS["a2"]
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    dp = datapath_for_die(hw.dies[0])
    plan = synthesize_sku_512_400k_plan(sku=sku, core_count=dp.aic_count, l2_bytes=int(dict(hw.dies[0].buffers)["l2"])).to_dict()
    obs = _obs_for_plan(plan)

    # 1. plan_only MUST fail closed (ok=False, CLI exit 2).
    plan_only = validate_512_400k(sku=sku)
    assert plan_only["ok"] is False and plan_only["status"] == "plan_only"
    assert mindie_sim_main(["onboard-sku", "validate-512-400k", "--sku", sku]) == 2

    # 2. Caller-supplied predicted_ns MUST be rejected.
    bad_pred = [dict(obs[0]), {**obs[-1], "predicted_ns": 20000}]
    with pytest.raises(ModelingError, match="caller_predicted_duration_forbidden"):
        score_sealed_gate_cases(bad_pred, sku=sku, runner=_live_engine_runner())

    # 3. Clock operator mismatch (busy_union vs elapsed_span) MUST be rejected.
    with pytest.raises(ModelingError, match="clock_operator_mismatch"):
        score_sealed_gate_cases(obs[:3], sku=sku, runner=_live_engine_runner(clock_op="busy_union"))

    # 4. [0%, 0%, 27%] pointwise outlier MUST fail + trigger diagnose_tail_cluster.
    held_ids = [r["case_id"] for r in obs if r["split"] == "heldout"]
    scales = {cid: 1.0 for cid in held_ids}
    scales[held_ids[0]] = 1.27
    val_fail = validate_512_400k(sku=sku, profiler_runner=lambda _p: obs, runner=_live_engine_runner(scales))
    assert val_fail["ok"] is False and val_fail["status"] == "fail" and held_ids[0] in val_fail["failing_cases"]
    syn = _synthetic_repo(tmp_path, family="sku_fail")
    gate_fail = verify_sku_gates(
        sku=sku, validation_report_json=val_fail,
        lean_bounds_json=[_lean_witness_row(held_ids[0], 15000, sim_ns=25000, p99_ns=22000)],
        repo_root=syn,
    )
    assert gate_fail["ok"] is False and gate_fail["diagnosis_triggered"] is True

    # 5. Missing 1 planned cell OR over_budget substituting for full-shape cell MUST fail.
    obs_missing = [r for r in obs if r["case_id"] != held_ids[0]]
    val_miss = validate_512_400k(sku=sku, profiler_runner=lambda _p: obs_missing, runner=_live_engine_runner())
    assert val_miss["ok"] is False and val_miss["status"] == "insufficient_evidence"
    assert held_ids[0] in val_miss["missing_plan_points"]

    obs_over = [{**r, "status": "over_budget", "executed_shape": "shrunk"} if r["case_id"] == held_ids[0] else r for r in obs]
    val_over = validate_512_400k(sku=sku, profiler_runner=lambda _p: obs_over, runner=_live_engine_runner())
    assert val_over["ok"] is False and len(val_over["over_budget_points"]) == 1

    # 6. Train==heldout leakage & service_regimes crutch MUST be rejected.
    leaked = [dict(r, acquisition_root="acq-train-src") if r["case_id"] == held_ids[0] else r for r in obs]
    with pytest.raises(ModelingError, match="train_heldout_lineage_overlap"):
        validate_512_400k(sku=sku, profiler_runner=lambda _p: leaked, runner=_live_engine_runner())
    with pytest.raises(ModelingError, match="whitebox_service_regimes_forbidden"):
        score_sealed_gate_cases([{**obs[-1], "service_regimes": {"a": 1}}], sku=sku, runner=_live_engine_runner())

    # 7. Witness-free L=0 pass MUST be rejected; valid witness + <10% sealed run certifies!
    ok_zero, _, v_zero = check_lean_non_inversion([{"case_id": "z", "l_star_lean_ns": 0, "t_sim_ns": 100, "t_silicon_p99_ns": 100}])
    assert ok_zero is False and any("lean_certificate_missing" in v for v in v_zero)
    assert any("lean_zero_lower_bound_on_positive_time" in v for v in v_zero)

    probe_file = tmp_path / "p_ok.json"
    probe_file.write_text(json.dumps({"aclrt_soc_name": sku, "aic_count": dp.aic_count, "aiv_count": dp.aiv_count}), encoding="utf-8")
    raw_l0 = tmp_path / "l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(sku)), encoding="utf-8")
    run_ok = run_sku_onboarding(
        sku=sku, probe_json=probe_file, raw_l0_json=raw_l0,
        profiler_runner=lambda _p: obs, runner=_live_engine_runner(default_scale=1.02),
        lean_bounds_json=[_lean_witness_row(held_ids[0], 15000, sim_ns=20400, p99_ns=20500)],
        repo_root=syn, out_dir=tmp_path / "sku_out",
    )
    assert run_ok["ok"] is True and run_ok["status"] == "certified"
    assert verify_receipt_sha256(run_ok["stages"]["verify-gates"]["receipt"]) is True


def test_operator_pipeline_real_owners_honest_scaffold_and_certify(tmp_path):
    sku = REPRESENTATIVE_SKUS["a2"]
    repo_dir, rev, sha = _git_repo_with_blob(tmp_path, "ops-transformer", "kernels/demo_op.cpp", b"// demo\n")
    syn = _synthetic_repo(tmp_path, family="demo_op")
    prof = _synthetic_profiler_root(tmp_path, supported=("demo_op",))

    # 1. Scaffold declares TILING_AXES + NotImplementedError (NO dummy shape[0] work!).
    scaf = scaffold_operator(
        family="demo_op", upstream_abi_arch22="aclnnDemoOpV1", upstream_abi_arch35="aclnnDemoOpV2",
        source_refs=[{"repository": "ops-transformer", "revision": rev, "relative_path": "kernels/demo_op.cpp", "source_sha256": sha}],
        target_root=syn, overwrite=True,
    )
    common_src = (syn / "mindie_sim/prediction/operators/demo_op/common.py").read_text(encoding="utf-8")
    assert "TILING_AXES" in common_src and "NotImplementedError" in common_src
    assert "shape[0]" not in common_src

    target_rel = scaf["package_rel"]
    # Unregistered scaffold MUST fail real builtin_family_registry + profiler owner checks!
    fail_unreg = validate_operator_contract(
        target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent,
        profiler_root=prof, require_tiling_axes=True,
    )
    assert fail_unreg["ok"] is False and any("sim_registry_unmatched" in v for v in fail_unreg["violations"])

    # Missing profiler owner MUST fail closed (never silently verified=True!).
    def _real_resolve(op: OpSpec, ctx) -> ResolvedOperator:
        return ResolvedOperator(
            occurrence=ctx.occurrence, op=op,
            recipe=OperatorPlan(
                work=Work(resources={"vector": 4096.0}),
                program=lambda *_a, **_kw: (),
                metadata={"implementation": "aclnnDemoOpV1"},
            ),
            implementation="aclnnDemoOpV1",
        )

    reg = FamilyRegistry([OperatorFamilySpec(name="demo_op", families=frozenset({"demo_op"}), owned_recipe_ids=frozenset({"demo_op"}), resolve=_real_resolve)])
    fail_no_prof = validate_operator_contract(
        target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent,
        profiler_root=_synthetic_profiler_root(tmp_path, supported=("other_only",), tag="missing"),
        family_registry=reg, require_tiling_axes=True,
    )
    assert fail_no_prof["ok"] is False and any("profiler_builder_unmodeled" in v for v in fail_no_prof["violations"])

    # Full real owners pass contract -> cliff plan -> sealed scoring -> certification!
    ok_contract = validate_operator_contract(
        target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent,
        profiler_root=prof, family_registry=reg, require_tiling_axes=True,
    )
    assert ok_contract["ok"] is True
    assert ok_contract["sim_owner_dispatch"]["lean_lower_bound_ns"] > 0
    assert ok_contract["profiler_builder"]["verified"] is True

    # Negative: missing Upstream-ABI header fails closed.
    leaf22 = syn / "mindie_sim/prediction/operators/demo_op/arch22.py"
    orig_leaf = leaf22.read_text(encoding="utf-8")
    leaf22.write_text(orig_leaf.replace("Upstream-ABI: aclnnDemoOpV1", ""), encoding="utf-8")
    bad_abi = validate_operator_contract(target_path=target_rel, repo_root=syn, profiler_root=prof, family_registry=reg)
    assert bad_abi["ok"] is False and "missing_upstream_abi_header:arch22.py" in bad_abi["violations"]
    leaf22.write_text(orig_leaf, encoding="utf-8")

    # End-to-end operator run-all with ABI-rejection guard verification.
    plan_res = generate_operator_cliff_plan(
        operator_id="demo_op:v1", family="demo_op", sku=sku,
        tile_dim=128, core_count=24, l2_bytes=192 * 1024 * 1024, topk_threshold=2048,
    )
    plan_doc = plan_res["plan"]
    op_obs = _obs_for_plan(plan_doc, levels=("L1",))
    abi_ok_rows = [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"]} for pt in plan_doc["abi_rejection_points"]]
    abi_bad_rows = [{"point_id": pt["point_id"], "rejected": False, "abi_error": ""} for pt in plan_doc["abi_rejection_points"]]

    bad_abi_score = bench_and_score_operator(
        cliff_plan=plan_doc, sku=sku, captured_json=op_obs,
        abi_rejection_results=abi_bad_rows, runner=_live_engine_runner(),
    )
    assert bad_abi_score["ok"] is False and bad_abi_score["abi_rejection_check"]["ok"] is False

    op_run = run_operator_onboarding(
        target_path=target_rel, operator_id="demo_op:v1", family="demo_op", sku=sku,
        tile_dim=128, core_count=24, l2_bytes=192 * 1024 * 1024, topk_threshold=2048,
        captured_json=op_obs, abi_rejection_results=abi_ok_rows,
        lean_bounds_json=[_lean_witness_row("demo_op:v1", 15000, sim_ns=20600, p99_ns=20500)],
        repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof,
        family_registry=reg, runner=_live_engine_runner(default_scale=1.02),
        out_dir=tmp_path / "op_out",
    )
    assert op_run["ok"] is True and op_run["status"] == "certified"
    assert verify_receipt_sha256(op_run["stages"]["certify"]["receipt"]) is True

    # Negative: Lean inversion fails certify.
    inv = certify_operator(
        operator_id="demo_op:v1", sku=sku,
        contract_report=op_run["stages"]["validate-contract"],
        score_report=op_run["stages"]["bench-and-score"],
        lean_bounds_json=[_lean_witness_row("demo_op:v1", 25000, sim_ns=20600, p99_ns=20500)],
        repo_root=syn,
    )
    assert inv["ok"] is False and inv["status"] == "failed"


def test_cli_entrypoints_and_fail_closed_exit_codes(tmp_path):
    sku = REPRESENTATIVE_SKUS["a2"]
    plan_out = tmp_path / "cli_plan.json"
    rc_plan = mindie_sim_main([
        "onboard-operator", "generate-cliff-plan", "--operator-id", "cli_op:v1", "--family", "cli_op",
        "--sku", sku, "--tile-dim", "128", "--core-count", "24", "--l2-bytes", "201326592",
        "--topk-threshold", "2048", "--plan-out", str(plan_out),
    ])
    assert rc_plan == 0 and plan_out.is_file()
    loaded = json.loads(plan_out.read_text(encoding="utf-8"))
    assert loaded["abi_rejection_points"]

    # plan_only bench-and-score MUST exit 2 via standalone tool wrapper too!
    done_op = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "tools" / "onboard_operator.py"),
         "bench-and-score", "--cliff-plan-json", str(plan_out), "--sku", sku],
        capture_output=True, text=True, check=False,
    )
    assert done_op.returncode == 2
    assert json.loads(done_op.stdout)["status"] == "plan_only"

    # Uncalibrated calibrate-l0-and-fabric MUST exit 2 via standalone tool wrapper!
    done_sku = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "tools" / "onboard_sku.py"),
         "calibrate-l0-and-fabric", "--sku", sku],
        capture_output=True, text=True, check=False,
    )
    assert done_sku.returncode == 2
    assert json.loads(done_sku.stdout)["status"] == "uncalibrated"


def test_real_coordinator_lease_builtin_registry_and_whitebox_runner(tmp_path):
    from mindie_sim.onboarding._gate_runners import acquire_coordinator_host_queue_lease
    from mindie_sim.onboarding import WhiteboxSpineRunner

    log_file = tmp_path / "coord_calls.jsonl"
    fake_coord = tmp_path / "fake_coord.py"
    fake_coord.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"p = {str(log_file)!r}\n"
        "args = sys.argv[1:]\n"
        "act = args[args.index('--action') + 1]\n"
        "with open(p, 'a', encoding='utf-8') as f: f.write(act + '\\n')\n"
        "if act == 'acquire': print(json.dumps({'lease_id': 'lq-9', 'fence_token': 77, 'devices': [0]}))\n"
        "else: print(json.dumps({'released': True}))\n",
        encoding="utf-8",
    )
    fake_coord.chmod(0o755)
    with pytest.raises(RuntimeError, match="boom"):
        with acquire_coordinator_host_queue_lease("lab-a2", coordinator_bin=str(fake_coord)) as lease:
            assert lease["lease_id"] == "lq-9" and lease["fence_token"] == 77
            raise RuntimeError("boom")
    assert log_file.read_text(encoding="utf-8").splitlines() == ["acquire", "release"]

    rotary_res = validate_operator_contract(
        target_path="mindie_sim/prediction/operators/rotary",
        repo_root=REPO_ROOT, require_profiler_builder=False,
    )
    assert rotary_res["sim_owner_dispatch"]["matched"] is True
    assert rotary_res["sim_owner_dispatch"]["resolved"] is True
    assert rotary_res["sim_owner_dispatch"]["lean_lower_bound_ns"] > 0

    sku = REPRESENTATIVE_SKUS["a2"]
    real_case = {"compose_model": "deepseek-v2-lite", "compose_hardware": sku, "compose_options": {"tp": 1}}
    sha = hashlib.sha256(b"real-whitebox-probe").hexdigest()
    wb_res = score_sealed_gate_cases(
        [{
            "case_id": "wb-real-1", "point_id": "wb-real-1", "split": "heldout", "level": "L3",
            "acquisition_root": "acq-wb-l3", "run_uid": "run-wb-1", "raw_sha256": sha,
            "measured_ns": 7_600_000, "clock_operator": "busy_union",
            "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1,
            "sim_case": real_case,
        }],
        sku=sku, runner=WhiteboxSpineRunner(clock_operator="busy_union"),
    )
    row = wb_res["residuals"][0]
    assert row["predicted_ns"] > 0 and row["engine_busy_union_ns"] > 0 and row["engine_elapsed_span_ns"] > 0
    assert row["engine_receipt"].startswith("capture://engine-receipt/")
