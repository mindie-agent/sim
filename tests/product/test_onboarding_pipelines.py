"""Orthogonal Tier-1 tests for turnkey SKU and Operator onboarding pipelines.

[Role] Verify all 4 New-SKU & 4 New-Operator onboarding stages, canonical L0
  sealing, ISA-scoped ``DAV_2201`` vs ``DAV_3510`` ABI rules, single-Engine
  white-box L1/L2/L3 evaluation (including ``kv_store(seq_len=1)`` safety and
  GEMM/``grouped_gemm`` operand binding), Lean 4 ``OptimalityCertificate``
  non-inversion, ``HostQueue`` lease cleanup, tail diagnosis, and CLI roundtrips.
[Invariants] Isolates host env/TMPDIR; zero raw IPv4, zero host mutations,
  zero device contexts, independent runners/oracles, zero ``service_regimes``.
[Timeouts & Locks] Deterministic in-memory and ``tmp_path`` checks; no locks.
[Side-Effect Policy] Writes temporary fixtures exclusively under ``tmp_path``.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.backends import ascend_profiles
from mindie_sim.cli import main as mindie_sim_main
from mindie_sim.cli.onboarding import operator_main, sku_main
from mindie_sim.observations.multilevel import (
    CaptureBoundary, CohortKey, EngineInterval, GateCase, bind_engine_interval,
)
from mindie_sim.onboarding import (
    ANCHOR_CONTEXT_GRID, ARCH_PRESET_22, ARCH_PRESET_35, OFF_GRID_CLIFF_HELDOUT_GRID,
    OperatorCliffParameters, TimeoutBudget, WhiteboxSpineRunner,
    acquire_coordinator_host_queue_lease, assign_plan_point_level, bench_and_score_operator,
    build_profiler_subgraph_plan, calibrate_l0_and_fabric, certify_operator,
    check_lean_non_inversion, eligible_replay_cases, evaluate_abi_rejections,
    family_default_tiling_axes, generate_operator_cliff_plan, load_canonical_l0_sealer,
    normalize_profiler_records, probe_soc, program_dag_sha256, required_ranks_for_plan,
    resolve_point_measured_binding, run_operator_onboarding, run_profiler_bench_subprocess,
    run_sku_onboarding, sanitize_public_warning, scaffold_operator, score_sealed_gate_cases,
    synthesize_operator_cliff_plan, synthesize_sku_512_400k_plan, temporary_scaffold_profile,
    validate_512_400k, validate_abi, validate_host_queue_grant, validate_operator_contract,
    validate_parallel_topology, verify_cliff_plan_integrity, verify_idle_die_telemetry,
    verify_receipt_sha256, verify_sku_gates,
)
from mindie_sim.onboarding.operator_pipeline import _build_default_planning_context, _verify_resolved_operator_dual_bound
from mindie_sim.optimization.dual_bound import DagEdge, EventDagSpec, ResourceChoiceProgram, certify_lower_bound
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.operators.contracts import OperatorFamilySpec, ResolvedOperator
from mindie_sim.prediction.operators.registry import FamilyRegistry
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.primitives.schema import parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier1
REPO_ROOT, SAMPLE_REPEATS, A2_SKU, A5_SKU = Path(__file__).resolve().parents[2], 12, REPRESENTATIVE_SKUS["a2"], REPRESENTATIVE_SKUS["a5"]
A2_ORACLE = {"arch_id": "npu_arch_2201", "aic": 20, "aiv": 40, "l2": 100_663_296, "ub": 196_608, "dies": 1, "hbm_bw": 1.2e12, "mte2_aiv": 2.4e10}


@pytest.fixture(autouse=True)
def _isolate_host_env(monkeypatch, tmp_path):
    iso = tmp_path / "isolated_home"
    iso.mkdir(parents=True, exist_ok=True)
    for var in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME", "MINDIE_DIAGNOSTICS_ROOT", "REMOTE_DEV_AUTH_DIR", "UV_CACHE_DIR", "TMPDIR"):
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
    files = (
        ("mindie_sim/__init__.py", '"""Stub."""\n'),
        ("scripts/check_arch_governance.py", "print('arch governance violations: 0')\n"),
        ("scripts/check_test_orthogonality.py", 'import json\nprint(json.dumps({"findings": []}))\n'),
        ("scripts/check_template_provenance.py", 'import json\nprint(json.dumps({"checked": 1, "violations": []}))\n'),
        (f"tests/operators/test_{family}.py", '"""Test."""\nimport pytest\npytestmark = pytest.mark.tier0\n'),
        ("tests/TAXONOMY.toml", f'schema_version = 1\ndomains = ["operators"]\nmax_lines = 600\nmax_modules = 8\n[[module]]\npath = "operators/test_{family}.py"\ntier = "tier0"\nowns = ["mindie_sim/prediction/operators/{family}", "mindie_sim/templates/{family}_spec.py"]\n'),
    )
    for rel, txt in files:
        (syn / rel).write_text(txt, encoding="utf-8")
    return syn


def _synthetic_profiler_root(tmp_path: Path, supported: tuple[str, ...] = ("demo_op",), *, abi: str = "aclnnDemoOpV1", tag: str = "main") -> Path:
    prof = tmp_path / f"syn_profiler_{tag}"
    pkg = prof / "mindie_profiler" / "bench"
    pkg.mkdir(parents=True, exist_ok=True)
    (prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "operators.py").write_text(
        f"class BenchUnmodeledOperatorError(RuntimeError):\n    pass\nSUPPORTED = {set(supported)!r}\nABI = {abi!r}\n"
        "def resolve_builder_strict(family, recipe, impl, dtypes=(), meta=None):\n"
        "    if family in SUPPORTED or recipe in SUPPORTED:\n        return (f'{family}_builder', ABI)\n"
        "    raise BenchUnmodeledOperatorError(f'unmodeled:{family}:{recipe}')\n",
        encoding="utf-8",
    )
    return prof


def _lean_witness_row(case_id: str, work_ns: int, *, sim_ns: int, p99_ns: int) -> dict[str, Any]:
    cert, _ = certify_lower_bound(ResourceChoiceProgram(1, (work_ns,), ()), EventDagSpec((work_ns,), ()))
    p_dict, d_dict = {"num_resources": 1, "global_work": [work_ns], "site_modes": []}, {"node_quantum_ns": [work_ns], "edges": []}
    return {
        "case_id": case_id, "l_star_lean_ns": cert.l_star_lean_ns, "t_sim_ns": sim_ns, "t_silicon_p99_ns": p99_ns,
        "program": p_dict, "dag": d_dict, "program_dag_sha256": program_dag_sha256(p_dict, d_dict),
        "certificate": {"farkas": {"scale_d": cert.farkas.scale_d, "lambda_num": list(cert.farkas.lambda_num), "mu_num": list(cert.farkas.mu_num), "lower_bound_ns": cert.farkas.lower_bound_ns}, "path_edges": list(cert.path_edges), "sink_node_id": cert.sink_node_id, "l_star_lean_ns": cert.l_star_lean_ns},
    }


def _independent_engine_runner(pred_by_id: dict[str, int], *, default_ns: int = 20_000, clock_op: str = "elapsed_span", prog_sha_by_id: dict[str, str] | None = None):
    table, p_map = dict(pred_by_id), dict(prog_sha_by_id or {})

    class _Runner:
        def _exec(self, case: GateCase) -> EngineInterval:
            pred_ns = int(table.get(case.case_id, default_ns))
            eng = Engine([Resource("device", 1_000_000.0, "ns", "fifo")], trace_limit=32)
            s_ev = eng.submit(case.boundary.start_event, {"device": 100.0})
            e_ev = eng.submit(case.boundary.end_event, {"device": float(pred_ns)}, after=(s_ev,))
            eng.advance(until=(e_ev,))
            half = max(1, pred_ns // 2)
            fb_sha = program_dag_sha256({"num_resources": 1, "global_work": [half], "site_modes": []}, {"node_quantum_ns": [half], "edges": []})
            return bind_engine_interval(case, eng, s_ev, e_ev, diagnostics=(("clock_operator", clock_op), ("busy_union_ns", pred_ns), ("elapsed_span_ns", pred_ns), ("program_dag_sha256", p_map.get(case.case_id, fb_sha))))

        run_l1 = run_l2 = run_l3 = _exec

    return _Runner()


def _plan_fixture_bundle(plan: dict[str, Any], *, levels: tuple[str, ...] | None = None, scale: float = 1.02, impl: str = "whitebox_spine", capture_dir: Path | None = None, overrides: dict[str, tuple[int, int]] | None = None) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    rows, preds, lean_rows, prog_shas, ov_map = [], {}, [], {}, dict(overrides or {})
    sku, family, target_id = str(plan["sku"]), str(plan.get("family") or ""), str(plan.get("target_id") or "")
    for split, pts in (("train", plan["train_points"]), ("heldout", plan["heldout_points"])):
        for idx, pt in enumerate(pts):
            pid = str(pt["point_id"])
            lvl = assign_plan_point_level(plan, pt) if levels is None else ("L1" if split == "train" else levels[idx % len(levels)])
            base_m = 10_000 + idx * 100 if split == "train" else 20_000 + idx * 250
            meas, pred = ov_map[pid] if pid in ov_map else (base_m, int(round(base_m * scale)))
            p99 = int(meas * 1.05)
            ranks = list(range(validate_parallel_topology(str(pt.get("parallel_topology") or "tp1"))))
            acq = "acq-train-src" if split == "train" else f"acq-held-{lvl.lower()}"
            cap_bytes = json.dumps({"case_id": pid, "sku": sku, "split": split, "level": lvl, "measured_ns": meas, "measured_p99_ns": p99}, sort_keys=True).encode("utf-8")
            sha = hashlib.sha256(cap_bytes).hexdigest()
            if capture_dir is not None:
                capture_dir.mkdir(parents=True, exist_ok=True)
                (capture_dir / f"{sha}.json").write_bytes(cap_bytes)
            preds[pid] = pred
            row: dict[str, Any] = {
                "case_id": pid, "point_id": pid, "sku": sku, "split": split, "level": lvl, "acquisition_root": acq,
                "run_uid": f"run-{split[0]}-{lvl.lower()}-{idx}", "raw_sha256": sha, "capture_uri": f"capture://{split}/{sha}.json",
                "event_ids": [f"{split[0]}-s-{idx}", f"{split[0]}-e-{idx}"], "device_ranks": ranks,
                "context_len": pt["value"] if pt["axis"] == "context_len" else None, "tokens": pt["value"] if pt["axis"] == "tokens" else None,
                "batch_size": pt["batch_size"], "query_tokens": pt["query_tokens"], "parallel_topology": pt["parallel_topology"], "topk": pt["topk"],
                "measured_ns": meas, "measured_p99_ns": p99, "clock_operator": "elapsed_span", "implementation": impl, "operator_id": target_id,
            }
            if family and family != "sku_suite":
                row["target_family"], row["upstream_abi"] = family, impl
            if pt["axis"] not in {"context_len", "tokens"}:
                row[str(pt["axis"])] = pt["value"]
            rows.append(row)
            if split == "heldout":
                lw = _lean_witness_row(pid, int(meas * 0.7), sim_ns=pred, p99_ns=p99)
                prog_shas[pid] = lw["program_dag_sha256"]
                lean_rows.append(lw)
    abi_rows = [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"]} for pt in plan["abi_rejection_points"]]
    return rows, preds, lean_rows, abi_rows, prog_shas


def _raw_l0_samples(sku: str, *, scale: float = 1.0, cann_version: str = "9.2.0") -> dict[str, Any]:
    out: dict[str, Any] = {
        "device_name": sku,
        "P01_datacopy": {"hbm_read_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.72 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "hbm_write_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.70 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "mte2_aiv_hit_bytes_per_s_per_core": [A2_ORACLE["mte2_aiv"] * scale * (0.75 + 0.001 * i) for i in range(SAMPLE_REPEATS)]},
        "P20_host_dma_sync": {"host_h2d_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "host_d2h_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)]},
    }
    return {**out, "cann_version": cann_version} if cann_version else out


def _a2_probe_payload(sku: str = A2_SKU, *, endpoint: str = "prod-cluster-01.corp") -> dict[str, Any]:
    return {"aclrt_soc_name": sku, "endpoint_alias": endpoint, "dies_per_package": A2_ORACLE["dies"], "platform_config_ini": f"[SoCInfo]\nsoc_version={sku}\ncompute_arch={A2_ORACLE['arch_id']}\nai_core_cnt={A2_ORACLE['aic']}\nvector_core_cnt={A2_ORACLE['aiv']}\nl2_size={A2_ORACLE['l2']}\nub_size={A2_ORACLE['ub']}\n", "npu_smi_common": f"Chip ID : 0\nAicore Count : {A2_ORACLE['aic']}\nAicore Freq(MHZ) : 1650\n"}


@contextmanager
def _mock_lease(devs: list[int], ttl: int, fence: int = 1):
    yield {"lease_mode": "coordinator_host_queue", "lease_id": "l-mock", "fence_token": fence, "devices": devs, "ttl_seconds": ttl}


def test_sku_stage1_probe_soc_conflicts_and_end_to_end_scaffold(tmp_path):
    synthetic_ip = ".".join(("198", "51", "100", "22"))
    ok_probe = tmp_path / "probe_ok.json"
    ok_probe.write_text(json.dumps(_a2_probe_payload(endpoint=f"prod-cluster-{synthetic_ip}.corp")), encoding="utf-8")
    res_ok = probe_soc(probe_json=ok_probe)
    assert res_ok["ok"] is True and res_ok["probe"]["endpoint_alias"].startswith("ep-")
    assert "prod-cluster" not in json.dumps(res_ok) and synthetic_ip not in json.dumps(res_ok)
    for bad_name, payload in (
        ("soc_conflict", {**_a2_probe_payload(), "soc_name": A2_SKU, "aclrt_soc_name": "OtherSilicon"}),
        ("smi_hbm", {**_a2_probe_payload(), "npu_smi_common": f"Chip ID : 0\nAicore Count : {A2_ORACLE['aic']}\nHBM Capacity(MB) : 1\n"}),
        ("bad_fab", {**_a2_probe_payload(), "fabric_channels": {"d2d_intra_node": 1.0}}),
    ):
        p = tmp_path / f"{bad_name}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        assert probe_soc(probe_json=p)["error"] == "silicon_profile_conflict", bad_name

    new_sku = "AscendFuture_9999"
    full_probe = tmp_path / "probe_new.json"
    full_probe.write_text(json.dumps({
        "aclrt_soc_name": new_sku, "arch_preset": A2_ORACLE["arch_id"], "architecture": "A2",
        "aic_count": 28, "aiv_count": 56, "hbm_bytes": 64 * 1024**3, "l2_bytes": 200 * 1024**2,
        "ub_bytes": 196_608, "dies_per_package": 1, "ranks_per_node": 8,
    }), encoding="utf-8")
    prof_scaf = probe_soc(probe_json=full_probe, allow_scaffold=True, base_template_sku=A2_SKU)["scaffold_profile"][new_sku]
    new_plan = synthesize_sku_512_400k_plan(sku=new_sku, core_count=28, l2_bytes=200 * 1024**2, profile_override=prof_scaf).to_dict()
    cap_dir, raw_l0 = tmp_path / "new_caps", tmp_path / "new_l0.json"
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(new_plan, capture_dir=cap_dir)
    raw_l0.write_text(json.dumps(_raw_l0_samples(new_sku)), encoding="utf-8")

    run_new = run_sku_onboarding(
        probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU,
        profiler_runner=lambda _p: obs, abi_rejection_results=abis, capture_root=cap_dir,
        require_capture_files=True, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas),
        lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="new_sku"), out_dir=tmp_path / "new_sku_out",
    )
    assert run_new["ok"] is True and run_new["status"] == "offline_reverify"
    assert verify_receipt_sha256(run_new["stages"]["verify-gates"]["receipt"]) is True

    warn_gate = verify_sku_gates(
        sku=new_sku, validation_report_json=run_new["stages"]["validate-512-400k"],
        primitives_json=run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"],
        lean_bounds_json=leans, profile_override=prof_scaf,
        stage2_lease={"lease_mode": "coordinator_host_queue", "lease_digest": "lease-0123456789abcdef", "fence_token": 7, "device_count": 1, "receipt_unconfirmed": True, "receipt_warning": "release_delayed:audit-node.internal.example"},
        repo_root=_synthetic_repo(tmp_path, family="warn_s"),
    )
    assert warn_gate["ok"] is True and warn_gate["receipt"]["receipt_unconfirmed"] is True
    assert "audit-node" not in json.dumps(warn_gate["receipt"]) and verify_receipt_sha256(warn_gate["receipt"]) is True


def test_sku_stage2_canonical_sealer_timeout_idle_and_cann_provenance(tmp_path):
    assert load_canonical_l0_sealer(tmp_path / "no_tools").__name__ == "mindie_sim.primitives.sealer"
    assert mindie_sim_main(["onboard-sku", "calibrate-l0-and-fabric", "--sku", A2_SKU]) == 2
    with pytest.raises(ModelingError, match="timeout_budget_outer_too_small"):
        TimeoutBudget((("l0", 300), ("fabric", 200)), cleanup_grace_s=60, outer_timeout_s=500)

    a5_cap = REPO_ROOT / "data/primitives/captures/arch35_a5_950dt.capture.json"
    a5_payload = json.loads(a5_cap.read_text(encoding="utf-8"))["payload"]
    a5_cap_sku = a5_payload["device_name"]
    a5_cap_cann = str(a5_payload["runtime_cann_version"])
    with pytest.raises(ModelingError, match="cann_version_provenance_conflict"):
        calibrate_l0_and_fabric(sku=a5_cap_sku, raw_l0_json=a5_cap, cann_version="9.9.9")
    cal_a5 = calibrate_l0_and_fabric(sku=a5_cap_sku, raw_l0_json=a5_cap, cann_version=a5_cap_cann, campaign_id="lease-audituser-0abc")
    assert cal_a5["ok"] is True and "audituser" not in json.dumps(cal_a5) and cal_a5["train_manifest"]["campaign_id"].startswith("camp-")

    over_l0 = tmp_path / "over_l0.json"
    over_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU, scale=2.5)), encoding="utf-8")
    cal_over = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=over_l0)
    assert any(c["reason"].startswith("p99_exceeds_nominal") for c in cal_over["metric_conflicts"])
    assert any(r["reason"] == "missing_samples" for r in cal_over["coverage_matrix"])

    idle = verify_idle_die_telemetry("No running processes found in NPU 0\n", "Aicore Usage Rate(%) : 0\nHBM Usage Rate(%) : 4\nAicore Count : 20\nAicore Freq(MHZ) : 1650\n", npu_id=0)
    assert idle["aicore_util_pct"] == 0 and idle["hbm_util_pct"] == 4
    with pytest.raises(ModelingError, match="onboarding_npu_not_idle"):
        verify_idle_die_telemetry("PID 1234 python\n", "Aicore Usage Rate(%) : 0\nHBM Usage Rate(%) : 0\n", npu_id=0)
    with pytest.raises(ModelingError, match="onboarding_npu_busy"):
        verify_idle_die_telemetry("No running processes found in NPU 0\n", "Aicore Usage Rate(%) : 15\nHBM Usage Rate(%) : 0\n", npu_id=0)


def test_cliff_planner_dav2201_vs_dav3510_abi_and_topology():
    axes35 = family_default_tiling_axes("sparse_indexer", tile_dim=128, arch_preset=ARCH_PRESET_35)
    axes22 = family_default_tiling_axes("sparse_indexer", tile_dim=128, arch_preset=ARCH_PRESET_22)
    assert validate_abi({"context_len": 1, "topk": 64}, axes35, arch_preset=ARCH_PRESET_35)[0] is True
    assert validate_abi({"context_len": 2049, "topk": 2049}, axes35, arch_preset=ARCH_PRESET_35)[0] is True
    assert validate_abi({"context_len": 2049, "topk": 2049}, axes22, arch_preset=ARCH_PRESET_22)[1].startswith("abi_divisibility_violation:topk=2049")
    assert validate_abi({"topk": 8193}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_max_violation:topk=8193")
    assert validate_abi({"topk": "2049"}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_type_violation")

    sp35 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 32, 192 * 1024**2), sku=A5_SKU)
    sp22 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, 100 * 1024**2), sku=A2_SKU)
    assert any(pt.topk == 2049 for pt in sp35.heldout_points) and not any(pt.topk == 2049 for pt in sp35.abi_rejection_points)
    assert any(pt.topk == 8193 for pt in sp35.abi_rejection_points) and any(pt.topk == 2049 for pt in sp22.abi_rejection_points)
    assert set(ANCHOR_CONTEXT_GRID) == {pt.value for pt in sp22.train_points if pt.axis == "context_len"}
    assert set(OFF_GRID_CLIFF_HELDOUT_GRID).issubset({pt.value for pt in sp22.heldout_points if pt.axis == "context_len"})
    assert required_ranks_for_plan(sp22.to_dict()) == 4
    with pytest.raises(ModelingError, match="dcp_pcp_domain_invalid"):
        validate_parallel_topology("tp2_pcp4_dcp2")
    with pytest.raises(ModelingError, match="cliff_plan_sha256_tampered"):
        verify_cliff_plan_integrity({**sp22.to_dict(), "arch_preset": ARCH_PRESET_35})
    with pytest.raises(ModelingError, match="unknown_sku_profile"):
        synthesize_sku_512_400k_plan(sku="NoSuchSKU", core_count=20, l2_bytes=A2_ORACLE["l2"])

    rejs = sp22.to_dict()["abi_rejection_points"]
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"]} for pt in rejs])["ok"] is True
    assert evaluate_abi_rejections(rejs, [{"point_id": rejs[0]["point_id"], "rejected": False, "abi_error": ""}])["ok"] is False


def test_sku_stage3_stage4_lineage_lean_tolerance_and_tail_diagnosis(tmp_path):
    plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    cap_dir, raw_l0 = tmp_path / "caps", tmp_path / "l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_l0)["primitives_payload"]
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(plan, capture_dir=cap_dir)

    # Disjoint train vs heldout lineage and L1/L2/L3 level roots
    leaked = [{**r, "acquisition_root": "acq-train-src"} if r["point_id"] == "offgrid_ctx_768" else r for r in obs]
    with pytest.raises(ModelingError, match="train_heldout_lineage_leakage"):
        score_sealed_gate_cases(leaked, sku=A2_SKU, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas))
    shared_lvl = [{**r, "acquisition_root": "acq-held-l1"} if r["level"] == "L2" else r for r in obs]
    with pytest.raises(ModelingError, match="multilevel_shared_acquisition_root"):
        score_sealed_gate_cases(shared_lvl, sku=A2_SKU, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas))
    with pytest.raises(ModelingError, match="caller_prediction_forbidden"):
        score_sealed_gate_cases([{**obs[0], "predicted_ns": 1234}], sku=A2_SKU, runner=_independent_engine_runner(preds))
    with pytest.raises(ModelingError, match="whitebox_service_regimes_forbidden"):
        score_sealed_gate_cases([{**obs[0], "service_regimes": {"k": 1}}], sku=A2_SKU, runner=_independent_engine_runner(preds))
    with pytest.raises(ModelingError, match="atol_ns_exceeds_strict_release_cap"):
        score_sealed_gate_cases([obs[-1]], sku=A2_SKU, atol_ns=2100, runner=_independent_engine_runner(preds))

    # Physical tolerance max(10%, 2.0 us): 4000ns vs 5500ns passes, vs 6500ns fails
    tiny_row = {**obs[-1], "case_id": "tiny-4us", "point_id": "tiny-4us", "measured_ns": 4000, "measured_p99_ns": 4200, "device_ranks": [0]}
    assert score_sealed_gate_cases([tiny_row], sku=A2_SKU, runner=_independent_engine_runner({"tiny-4us": 5500}))["ok"] is True
    assert score_sealed_gate_cases([tiny_row], sku=A2_SKU, runner=_independent_engine_runner({"tiny-4us": 6500}))["ok"] is False

    # Passing full Stage 3 + Stage 4 via validation report and capture file verification
    syn = _synthetic_repo(tmp_path, family="sku_s4")
    val_ok = validate_512_400k(sku=A2_SKU, primitives_json=cal, profiler_runner=lambda _p: obs, abi_rejection_results=abis, capture_root=cap_dir, require_capture_files=True, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas))
    gate_ok = verify_sku_gates(sku=A2_SKU, validation_report_json=val_ok, primitives_json=cal, lean_bounds_json=leans, repo_root=syn)
    assert val_ok["ok"] is True and gate_ok["ok"] is True and verify_receipt_sha256(gate_ok["receipt"]) is True

    # Lean 4 non-inversion rejection
    inv_leans = [{**leans[0], "l_star_lean_ns": leans[0]["t_sim_ns"] + 500}, *leans[1:]]
    gate_inv = verify_sku_gates(sku=A2_SKU, validation_report_json=val_ok, primitives_json=cal, lean_bounds_json=inv_leans, repo_root=syn)
    assert gate_inv["ok"] is False and any(g["name"] == "lean_optimality_non_inversion" and not g["passed"] for g in gate_inv["receipt"]["gates"])

    # >=10% cliff failure auto-triggers diagnose_tail_cluster.py
    fail_preds = {**preds, "offgrid_ctx_200000": int(next(r["measured_ns"] for r in obs if r["point_id"] == "offgrid_ctx_200000") * 1.45)}
    val_fail = validate_512_400k(sku=A2_SKU, primitives_json=cal, profiler_runner=lambda _p: obs, abi_rejection_results=abis, runner=_independent_engine_runner(fail_preds, prog_sha_by_id=pshas))
    diag_out = tmp_path / "tail_diag.json"
    gate_fail = verify_sku_gates(sku=A2_SKU, validation_report_json=val_fail, primitives_json=cal, lean_bounds_json=leans, repo_root=syn, diagnosis_out=diag_out)
    assert gate_fail["ok"] is False and "diagnosis" in gate_fail and diag_out.is_file()


def test_operator_onboarding_4stages_contract_template_and_certify(tmp_path):
    repo_dir, rev, sha = _git_repo_with_blob(tmp_path, "ops-transformer", "kernels/demo_op.cpp", b"// demo\n")
    syn = _synthetic_repo(tmp_path, family="demo_op")
    prof_ok = _synthetic_profiler_root(tmp_path, supported=("demo_op",), abi="aclnnDemoOpV1", tag="ok")

    with pytest.raises(ModelingError, match="operator_family_forbidden_hardware_token"):
        scaffold_operator(family="arch22_bad", upstream_abi_arch22="aclnnA2", upstream_abi_arch35="aclnnA5", source_refs=[{"repository": "ops-transformer", "revision": rev, "relative_path": "kernels/demo_op.cpp", "source_sha256": sha}], target_root=syn)
    scaf = scaffold_operator(
        family="demo_op", upstream_abi_arch22="aclnnDemoOpV1", upstream_abi_arch35="aclnnDemoOpV2",
        source_refs=[{"repository": "ops-transformer", "revision": rev, "relative_path": "kernels/demo_op.cpp", "source_sha256": sha}],
        target_root=syn, overwrite=True,
    )
    target_rel = scaf["package_rel"]

    def _real_resolve(op: OpSpec, rctx) -> ResolvedOperator:
        prog = KernelProgram((Stage("vec_main", work={"vector": 4096.0}),))
        return ResolvedOperator(occurrence=rctx.occurrence, op=op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: prog, pipeline_spec={"stage": "v"}))

    reg_ok = FamilyRegistry([OperatorFamilySpec("demo_op", frozenset({"demo_op"}), frozenset({"demo_op", "demo_op:aclnnDemoOpV1"}), _real_resolve)])
    stub_chk = validate_operator_contract(target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_ok, family_registry=reg_ok)
    assert stub_chk["ok"] is False and any("operator_target_unimplemented_stub" in v for v in stub_chk["violations"])

    # Real built-in template lowers & passes Lean dual bound; borrowed template_id rejected
    cat_chk = validate_operator_contract(target_path=REPO_ROOT / "mindie_sim/templates/catlass_catalog.py", repo_root=REPO_ROOT, template_id="catlass.basic_pingpong", require_profiler_builder=False, require_taxonomy=False, require_tiling_axes=False)
    assert cat_chk["sim_owner_dispatch"]["lowered"] is True and cat_chk["sim_owner_dispatch"]["lean_status"] == "verified"
    fake_tpl = syn / "mindie_sim/templates/fake_spec.py"
    fake_tpl.parent.mkdir(parents=True, exist_ok=True)
    fake_tpl.write_text(f'"""Stub.\n\nUpstream-ABI: aclnnRmsNorm\n"""\nTEMPLATE_ID = "catlass.basic_pingpong"\nSOURCES = (SourceRef(repository="ops-transformer", revision="{rev}", relative_path="kernels/demo_op.cpp", source_sha256="{sha}"),)\nSemanticWorkItem = TensorRangeObligation = TilingAxes = object\n', encoding="utf-8")
    borrowed = validate_operator_contract(target_path=fake_tpl, repo_root=syn, refs_root=repo_dir.parent, template_id="catlass.basic_pingpong", require_profiler_builder=False, require_taxonomy=False)
    assert borrowed["ok"] is False and any("template_id_not_owned_by_target" in v for v in borrowed["violations"])

    probe_op = OpSpec("op0", "demo_op", "demo_op", "decode", "bf16", (), (1, 64), (1, 64), (1, 64), (1, 64))
    ctx = _build_default_planning_context(probe_op)
    zero_res = ResolvedOperator(occurrence=ctx.occurrence, op=probe_op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: (), pipeline_spec={"stage": "v"}))
    assert _verify_resolved_operator_dual_bound(zero_res, ctx)[2] == "resolved_operator_program_zero_executable_events"

    # Implement stubs and run all 4 operator stages end-to-end
    common_py = syn / target_rel / "common.py"
    common_py.write_text(common_py.read_text(encoding="utf-8").replace("raise NotImplementedError(", "return ("), encoding="utf-8")
    plan_doc = generate_operator_cliff_plan(operator_id="demo_op:aclnnDemoOpV1", family="demo_op", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048)["plan"]
    op_caps = tmp_path / "op_caps"
    op_obs, op_preds, op_leans, op_abis, op_shas = _plan_fixture_bundle(plan_doc, levels=("L1",), impl="aclnnDemoOpV1", capture_dir=op_caps)
    op_run = run_operator_onboarding(
        target_path=target_rel, operator_id="demo_op:aclnnDemoOpV1", family="demo_op", sku=A2_SKU,
        tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048,
        captured_json=op_obs, abi_rejection_results=op_abis, lean_bounds_json=op_leans,
        capture_root=op_caps, require_capture_files=True, repo_root=syn, refs_root=repo_dir.parent,
        profiler_root=prof_ok, family_registry=reg_ok, runner=_independent_engine_runner(op_preds, prog_sha_by_id=op_shas),
        out_dir=tmp_path / "op_out",
    )
    assert op_run["ok"] is True and op_run["status"] == "offline_reverify"
    assert verify_receipt_sha256(op_run["stages"]["certify"]["receipt"]) is True
    with pytest.raises(ModelingError, match="cliff_plan_sha256_tampered"):
        bench_and_score_operator(cliff_plan={**plan_doc, "arch_preset": ARCH_PRESET_35}, sku=A2_SKU, captured_json=op_obs)
    assert certify_operator(operator_id="demo_op:wrong", sku=A2_SKU, contract_report=op_run["stages"]["validate-contract"], score_report=op_run["stages"]["bench-and-score"], lean_bounds_json=op_leans, repo_root=syn)["ok"] is False


def test_host_queue_leases_profiler_normalizer_and_cli_roundtrip(tmp_path):
    for bad_g, req_d, min_t, pat in (
        ({"lease_id": "l1", "fence_token": 1, "devices": [], "ttl_seconds": 600}, 1, 120, "host_queue_lease_device_count_mismatch"),
        ({"lease_id": "l1", "fence_token": 1, "devices": [0, 1, 2, 3], "ttl_seconds": 1}, 4, 120, "host_queue_lease_ttl_below_subprocess_plus_grace"),
    ):
        with pytest.raises(ModelingError, match=pat):
            validate_host_queue_grant(bad_g, required_devices=req_d, min_ttl_seconds=min_t)

    rel_log, c_short, c_warn = tmp_path / "rel_log.txt", tmp_path / "c_short.py", tmp_path / "c_warn.py"
    c_short.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\nact = sys.argv[sys.argv.index('--action') + 1]\nif act == 'acquire':\n    print(json.dumps({{'outcome':'success','status':'ready','data':{{'lease_id':'lease-audituser-1','fence_token':5,'devices':[0,1,2]}}}}))\nelse:\n    Path({str(rel_log)!r}).write_text('released', encoding='utf-8')\n    print(json.dumps({{'outcome':'success','status':'ok'}}))\n", encoding="utf-8")
    c_warn.write_text(f"#!{sys.executable}\nimport json, sys\nact = sys.argv[sys.argv.index('--action') + 1]\nif act == 'acquire':\n    print(json.dumps({{'outcome':'success','status':'ready','receipt_unconfirmed':True,'receipt_warning':'PermissionError: audituser','data':{{'lease_id':'lease-audituser-2','fence_token':7,'devices':[0,1,2,3]}}}}))\nelse:\n    print(json.dumps({{'outcome':'success','status':'ok'}}))\n", encoding="utf-8")
    c_short.chmod(0o755)
    c_warn.chmod(0o755)
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        with acquire_coordinator_host_queue_lease("audit-node.internal.example", npu_count=4, ttl_seconds=180, min_subprocess_timeout_s=120, coordinator_bin=str(c_short)):
            pass
    assert rel_log.read_text(encoding="utf-8") == "released"
    with acquire_coordinator_host_queue_lease("audit-node.internal.example", npu_count=4, ttl_seconds=180, min_subprocess_timeout_s=120, coordinator_bin=str(c_warn)) as linfo:
        assert linfo["devices"] == [0, 1, 2, 3]
    assert linfo["receipt_unconfirmed"] is True and "audituser" not in sanitize_public_warning("PermissionError: audituser")
    with pytest.raises(ModelingError, match="host_queue_lease_"):
        calibrate_l0_and_fabric(sku=A2_SKU, live_bench=True, lease_acquirer=lambda _s, _t: _mock_lease([], 1, fence=0))

    # Subgraph plan lowering + profiler@57dc39c1 kv_store(seq_len=1) safety on L3 offgrid_ctx_200000
    b4_plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=20, l2_bytes=A2_ORACLE["l2"]).to_dict()
    b4_sub = build_profiler_subgraph_plan(b4_plan, sku=A2_SKU)
    b4_by_id = {c["case_id"]: c for c in b4_sub["cases"]}
    assert b4_sub["world_size"] == 4 and {c["level"] for c in b4_sub["cases"]} == {"L1", "L2", "L3"}
    assert b4_by_id["anchor_ctx_512"]["shape"]["seq_len"] == 513 and b4_by_id["anchor_ctx_512"]["shape"]["query_layout"] == "BNSD"
    assert b4_by_id["cross_b16_ctx_4097"]["nodes"][1]["builder"] == "grouped_matmul"
    assert len(b4_by_id["offgrid_ctx_200000"]["nodes"]) == 15 and len(b4_by_id["cross_tp2_pcp2_dcp4_ctx_400000"]["nodes"]) == 23

    l3_c200 = b4_by_id["offgrid_ctx_200000"]
    l3_ns_kv1 = {
        "offgrid_ctx_200000:s0.l0.kv_store": {"tokens": 1, "width": 576, "seq_len": 1, "dtype": "bf16"},
        **{n["node_id"]: ({**n["shape"], "dtype": "bf16", "weight_dtype": "bf16"} if n["family"] in {"gemm", "grouped_gemm", "gate_up_projection", "down_projection"} else n["shape"]) for n in l3_c200["nodes"] if n["op_id"] != "s0.l0.kv_store"},
    }
    l3_norm, _ = normalize_profiler_records([
        {"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_200000", "run_id": "r200.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "timing_mode": "eager", "duration_ms": 1.5, "sku": A2_SKU},
        {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "offgrid_ctx_200000", "run_id": "r200", "rank": 0, "status": "ok", "actual_shape": {"tokens": 1, "width": 2048, "dtype": "bf16"}, "node_shapes": l3_ns_kv1, "timed_nodes": [n["node_id"] for n in l3_c200["nodes"]]},
    ], plan_doc=b4_plan, sku=A2_SKU, capture_out_dir=tmp_path / "l3_caps")
    assert l3_norm[0]["context_len"] == 200000
    assert score_sealed_gate_cases(l3_norm, sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="elapsed_span"))["residuals"][0]["predicted_ns"] > 0

    # Subprocess roundtrip with partial ABI exit code 2
    abi_plan = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, A2_ORACLE["l2"]), sku=A2_SKU).to_dict()
    abi_pt = abi_plan["abi_rejection_points"][0]
    fake_prof = tmp_path / "fake_prof_sub"
    (fake_prof / "mindie_profiler/bench").mkdir(parents=True)
    (fake_prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (fake_prof / "mindie_profiler/__main__.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\n"
        "plan = json.loads(Path(args[args.index('--plan') + 1]).read_text(encoding='utf-8'))\n"
        "obs_p, out_p = Path(args[args.index('--observations') + 1]), Path(args[args.index('--out') + 1])\n"
        "rk, ws = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))\n"
        "c0 = next(c for c in plan['cases'] if c['split'] != 'abi_rejection')\n"
        "obs_p.write_text(json.dumps({'schema':'mindie-bench-observation/1','record_type':'observation','case_id':c0['case_id'],'run_id':f'r{rk}.base.eager.t0','trial':0,'repeat_index':0,'rank':rk,'split':c0['split'],'timing_mode':'eager','duration_ms':0.02 + 0.001*rk,'sku':plan['sku']}) + '\\n', encoding='utf-8')\n"
        f"recs = [{{'case_id': c0['case_id'], 'status': 'ok', 'actual_shape': c0.get('shape') or {{}}, 'timed_nodes': [n['node_id'] for n in c0['nodes']]}}, {{'case_id': {abi_pt['point_id']!r}, 'status': 'unsupported_binding', 'reason': {abi_pt['abi_expected_error']!r}}}]\n"
        "Path(f'{out_p}.rank{rk}.json' if ws > 1 else str(out_p)).write_text(json.dumps({'schema':'mindie-bench-subgraph-run/1','status':'partial','run_id':f'r{rk}','records':recs}), encoding='utf-8')\n"
        "sys.exit(2)\n",
        encoding="utf-8",
    )
    obs_l, abi_l = run_profiler_bench_subprocess(abi_plan, sku=A2_SKU, profiler_root=fake_prof, allowed_phy_ids=[0, 1, 2, 3], fence_token=7)
    assert len(obs_l) == 1 and obs_l[0]["device_ranks"] == [0, 1, 2, 3] and any(a["point_id"] == abi_pt["point_id"] for a in abi_l)

    # CLI roundtrips (mindie-sim + standalone sku_main / operator_main)
    p_json = tmp_path / "cli_probe.json"
    p_json.write_text(json.dumps(_a2_probe_payload()), encoding="utf-8")
    p_out, plan_out = tmp_path / "cli_probe_out.json", tmp_path / "cli_plan_out.json"
    assert sku_main(["probe-soc", "--probe-json", str(p_json), "-o", str(p_out)]) == 0 and json.loads(p_out.read_text(encoding="utf-8"))["ok"] is True
    assert operator_main(["generate-cliff-plan", "--operator-id", "rmsnorm:v1", "--family", "rmsnorm", "--sku", A2_SKU, "--tile-dim", "128", "--core-count", "20", "--l2-bytes", str(A2_ORACLE["l2"]), "--plan-out", str(plan_out)]) == 0
    assert plan_out.is_file()


def test_whitebox_runner_l1_l2_l3_gemm_grouped_gemm_and_isolation(tmp_path):
    b4_case = {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    base_l1 = {
        "case_id": "wb-l1", "point_id": "wb-l1", "sku": A2_SKU, "split": "heldout", "level": "L1",
        "acquisition_root": "acq-wb-l1", "run_uid": "r1", "raw_sha256": hashlib.sha256(b"wb-l1").hexdigest(),
        "event_ids": ["s1", "e1"], "device_ranks": [0], "measured_ns": 2500, "measured_p99_ns": 2600,
        "clock_operator": "busy_union", "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1,
        "target_family": "rmsnorm", "sim_case": b4_case,
    }
    with pytest.raises(ModelingError, match="whitebox_materialized_sku_mismatch"):
        score_sealed_gate_cases([{**base_l1, "sku": A5_SKU}], sku=A5_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    mr_l3 = {**base_l1, "case_id": "wb-mr", "point_id": "wb-mr", "level": "L3", "parallel_topology": "tp2_pcp2_dcp2", "device_ranks": [0, 1, 2, 3], "target_family": None, "sim_case": {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU}}
    assert score_sealed_gate_cases([mr_l3], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0

    gqa_case = {"compose_model": "qwen3-32b", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    with pytest.raises(ModelingError, match="whitebox_batch_x_kv_heads_indivisible"):
        score_sealed_gate_cases([{**base_l1, "case_id": "bx-odd", "point_id": "bx-odd", "batch_x_kv_heads": 2559, "sim_case": gqa_case}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    assert score_sealed_gate_cases([{**base_l1, "case_id": "bx-1kv", "point_id": "bx-1kv", "batch_x_kv_heads": 2559}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0

    # Attention & L2 operand parity checks (layout, kv_len, block_table, sequences, q_len, GEMM matrix_groups/dtype)
    attn_l1 = {**base_l1, "case_id": "wb-attn", "point_id": "wb-attn", "context_len": 512, "target_family": "attention", "target_op_id": "s0.l0.paged_mla_decode"}
    for bad_bind in (
        {"measured_binding": {"family": "attention", "recipe": "fused_infer_attention", "paged": False}},
        {"measured_binding": {"family": "attention", "recipe": "ascend_paged_mla_decode_arch22", "query_layout": "TND"}},
        {"actual_shape": {"page_count": 5, "block_table": [1, 4]}, "measured_binding": {"family": "attention", "recipe": "ascend_paged_mla_decode_arch22"}},
    ):
        with pytest.raises(ModelingError, match="whitebox_measured_binding_mismatch"):
            score_sealed_gate_cases([{**attn_l1, **bad_bind}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))

    l2_b16 = {**base_l1, "case_id": "l2-mg", "point_id": "l2-mg", "level": "L2", "context_len": 4097, "batch_size": 16, "target_family": None, "target_op_ids": ["s0.l0.paged_mla_decode", "s0.l0.v_expand"]}
    l2_tn = ["l2-mg:s0.l0.paged_mla_decode", "l2-mg:s0.l0.v_expand"]
    l2_attn_ok = {"sequences": 16, "q_len": 1, "seq_len": 4098, "page_count": 33, "block_table": [16, 33], "heads": 16, "kv_heads": 1, "query_layout": "BNSD"}
    l2_v_ok = {"m": 256, "n": 128, "k": 512, "e": 16, "matrix_groups": 16, "dtype": "bf16", "weight_dtype": "bf16"}
    for pat, patch in (
        ("matrix_groups", {"measured_binding": {"node_bindings": [{"op_id": "s0.l0.v_expand", "matrix_groups": 1}]}}),
        ("l2_op_set_mismatch", {"timed_nodes": ["l2-mg:s0.l0.paged_mla_decode"]}),
        ("dtype", {"actual_shape": {"node_shapes": {"s0.l0.paged_mla_decode": l2_attn_ok, "s0.l0.v_expand": {**l2_v_ok, "dtype": "int8"}}}, "timed_nodes": l2_tn}),
        ("sequences|q_len|block_table_sequences", {"actual_shape": {"node_shapes": {"s0.l0.paged_mla_decode": {**l2_attn_ok, "sequences": 1, "q_len": 16, "block_table": [1, 33]}, "s0.l0.v_expand": l2_v_ok}}, "timed_nodes": l2_tn}),
    ):
        with pytest.raises(ModelingError, match=pat):
            score_sealed_gate_cases([{**l2_b16, **patch}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))

    # L1 GEMM & grouped_gemm off-grid cliff (127, graph_capture_mode="none") + weight_dtype/quant_mode parity
    for g_fam in ("gemm", "grouped_gemm"):
        g_plan = synthesize_operator_cliff_plan(OperatorCliffParameters(f"{g_fam}:v1", g_fam, 128, 20, A2_ORACLE["l2"]), sku=A2_SKU).to_dict()
        g_pt = next(p for p in g_plan["heldout_points"] if p["point_id"] == "tile_128_minus1_127")
        g_sh = {k: v for k, v in resolve_point_measured_binding(g_plan, g_pt, sku=A2_SKU)["expected_shape"].items() if k in {"m", "n", "k", "e", "matrix_groups", "dtype"}}
        g_sh["weight_dtype"] = "bf16"
        g_rows, _ = normalize_profiler_records([
            {"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": g_pt["point_id"], "run_id": f"g-{g_fam}.base.eager.t0", "acquisition_id": f"acq-{g_fam}", "trial": 0, "rank": 0, "split": "heldout", "timing_mode": "eager", "duration_ms": 0.05, "sku": A2_SKU},
            {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": g_pt["point_id"], "run_id": f"g-{g_fam}", "acquisition_id": f"acq-{g_fam}", "rank": 0, "status": "ok", "actual_shape": g_sh, "timed_nodes": [f"{g_pt['point_id']}:op0"]},
        ], plan_doc=g_plan, sku=A2_SKU)
        g_sc = score_sealed_gate_cases(g_rows, sku=A2_SKU, expected_family=g_fam, runner=WhiteboxSpineRunner(clock_operator="elapsed_span"))
        assert g_sc["residuals"][0]["predicted_ns"] > 0
        for k_bad, v_bad in (("weight_dtype", "int8"), ("quant_mode", "w8a8")):
            with pytest.raises(ModelingError, match=k_bad):
                score_sealed_gate_cases([{**g_rows[0], "actual_shape": {**g_sh, k_bad: v_bad}}], sku=A2_SKU, expected_family=g_fam, runner=WhiteboxSpineRunner(clock_operator="elapsed_span"))

    # Indexer topk parity + L2 real DAG Lean OptimalityCertificate + calibrated PrimitiveTable + thread isolation
    idx_row = {**base_l1, "case_id": "idx-2048", "point_id": "idx-2048", "context_len": 2049, "topk": 2048, "target_family": "indexer", "target_op_id": "s0.l2.index_quant_lightning", "measured_binding": {"target_op_id": "s0.l2.index_quant_lightning", "family": "indexer", "sparse_count": 2048, "query_dtype": "bf16", "cache_dtype": "bf16"}}
    with pytest.raises(ModelingError, match="topk|query_dtype"):
        score_sealed_gate_cases([{**idx_row, "sim_case": {"compose_model": "deepseek-v4.1-flash", "compose_hardware": A2_SKU}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    assert score_sealed_gate_cases([{**idx_row, "sim_case": {"compose_model": "deepseek-v4.1-flash", "compose_hardware": A2_SKU, "model_overrides": {"index_topk": 2048, "index_dtype": "bf16"}}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0

    wb_r = WhiteboxSpineRunner(clock_operator="busy_union")
    l2_dep = {**base_l1, "case_id": "l2-dep", "point_id": "l2-dep", "level": "L2", "target_family": None, "target_op_ids": ["s0.l0.norm1", "s0.l0.q_proj"]}
    l2_ind = {**base_l1, "case_id": "l2-ind", "point_id": "l2-ind", "level": "L2", "target_family": None, "target_op_ids": ["s0.l0.q_proj", "s0.l0.kv_a"]}
    r_dep, r_ind = score_sealed_gate_cases([l2_dep], sku=A2_SKU, runner=wb_r)["residuals"][0], score_sealed_gate_cases([l2_ind], sku=A2_SKU, runner=wb_r)["residuals"][0]
    assert r_dep["program_dag_sha256"] and r_dep["program_dag_sha256"] != r_ind["program_dag_sha256"]
    dep_gc = GateCase("l2-dep", "L2", "heldout", CaptureBoundary(base_l1["raw_sha256"], "acq-l2", "r1", (), ("s1", "e1"), (0,), "device_ns", "s1", "e1", "fx_region", "ns", r_dep["capture_uri"]), "in", "st", "wb", CohortKey("L2", "c", A2_SKU, "decode", "eager", "cliff", "tp1"), 2500, payload=(("clock_operator", "busy_union"), ("context_len", 64), ("phase", "decode"), ("sim_case", b4_case), ("target_op_ids", ["s0.l0.norm1", "s0.l0.q_proj"])))
    dep_diag = dict(wb_r.run_l2(dep_gc).diagnostics)
    lp, ld = dep_diag["lean_program"], dep_diag["lean_dag"]
    d_cert, _ = certify_lower_bound(ResourceChoiceProgram(lp["num_resources"], tuple(lp["global_work"]), ()), EventDagSpec(tuple(ld["node_quantum_ns"]), tuple(DagEdge(e["src"], e["dst"], e["wire_ns"]) for e in ld["edges"])))
    lean_ok, _, lean_v = check_lean_non_inversion([{"case_id": "l2-dep", "l_star_lean_ns": d_cert.l_star_lean_ns, "t_sim_ns": dep_diag["busy_union_ns"], "t_silicon_p99_ns": max(dep_diag["busy_union_ns"], 2600), "program": lp, "dag": ld, "certificate": {"farkas": {"scale_d": d_cert.farkas.scale_d, "lambda_num": list(d_cert.farkas.lambda_num), "mu_num": list(d_cert.farkas.mu_num), "lower_bound_ns": d_cert.farkas.lower_bound_ns}, "path_edges": list(d_cert.path_edges), "sink_node_id": d_cert.sink_node_id, "l_star_lean_ns": d_cert.l_star_lean_ns}}], expected_case_timings={"l2-dep": (dep_diag["busy_union_ns"], 2500, max(dep_diag["busy_union_ns"], 2600), dep_diag["program_dag_sha256"])})
    assert lean_ok is True and not lean_v

    raw_full = tmp_path / "full.json"
    raw_full.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal_tbl = parse_primitive_table(calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_full)["primitives_payload"])
    cal_r = WhiteboxSpineRunner(clock_operator="busy_union", primitive_table=cal_tbl)
    assert score_sealed_gate_cases([base_l1], sku=A2_SKU, runner=cal_r)["residuals"][0]["predicted_ns"] > 0
    gc_cf = GateCase("cf-drop", "L1", "train", dep_gc.boundary, "i", "s", "w", dep_gc.cohort, 0, is_counterfactual=True, has_matched_capture=False)
    assert eligible_replay_cases([dep_gc, gc_cf]) == [dep_gc]

    bar, seen_rec = threading.Barrier(2), {}
    prof22, prof35 = copy.deepcopy(ascend_profiles()[A2_SKU]), copy.deepcopy(ascend_profiles()[A5_SKU])

    def _t_iso(tag: str, pr: dict[str, Any], fail_early: bool) -> None:
        try:
            with temporary_scaffold_profile("AuditScaffold", pr):
                bar.wait(timeout=5)
                if fail_early:
                    raise RuntimeError("boom")
                bar.wait(timeout=5)
                pl = synthesize_sku_512_400k_plan(sku="AuditScaffold", core_count=20, l2_bytes=A2_ORACLE["l2"]).to_dict()
                seen_rec[tag] = resolve_point_measured_binding(pl, pl["train_points"][0], sku="AuditScaffold")["measured_binding"]["recipe"]
        except RuntimeError:
            bar.wait(timeout=5)

    t1, t2 = threading.Thread(target=_t_iso, args=("a5", prof35, True)), threading.Thread(target=_t_iso, args=("a2", prof22, False))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert seen_rec.get("a2") == "ascend_paged_mla_decode_arch22" and "AuditScaffold" not in ascend_profiles()
