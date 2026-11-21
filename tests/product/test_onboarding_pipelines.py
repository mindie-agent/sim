"""Orthogonal Tier-1 tests for turnkey SKU & Operator onboarding pipelines.

[Role] Verify all 4 New-SKU & 4 New-Operator onboarding stages, mandatory raw
  capture truth, unspoofable ``silicon_certified`` vs ``offline_reverify``,
  ISA-scoped ``DAV_2201`` vs ``DAV_3510`` ABI rules, single-Engine white-box
  SKU/2D/multi-rank/real-DAG enforcement, ``bench subgraph`` dispatch + real
  profiler roundtrip, unified ``HostQueue`` lease validation, and privacy.
[Invariants] Isolates host env/TMPDIR; zero raw IPv4, zero host mutations,
  zero device contexts, independent oracles/runners, zero ``service_regimes``.
[Timeouts & Locks] Deterministic in-memory and ``tmp_path`` checks; no locks.
[Side-Effect Policy] Writes temporary fixtures exclusively under ``tmp_path``.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy, hashlib, json, subprocess, sys, threading
from pathlib import Path
from typing import Any
import pytest

from mindie_sim.backends import ascend_profiles
from mindie_sim.cli import main as mindie_sim_main
from mindie_sim.observations.multilevel_gate import CaptureBoundary, CohortKey, EngineInterval, GateCase, bind_engine_interval
from mindie_sim.observations.multilevel_stream import verify_capture_file_consistency
from mindie_sim.onboarding import (
    ANCHOR_CONTEXT_GRID, OFF_GRID_CLIFF_HELDOUT_GRID, OperatorCliffParameters, TimeoutBudget, WhiteboxSpineRunner,
    bench_and_score_operator, calibrate_l0_and_fabric, certify_operator, evaluate_abi_rejections, family_default_tiling_axes,
    generate_operator_cliff_plan, probe_soc, program_dag_sha256, required_ranks_for_plan, run_operator_onboarding,
    run_sku_onboarding, scaffold_operator, score_sealed_gate_cases, synthesize_operator_cliff_plan, synthesize_sku_512_400k_plan,
    validate_512_400k, validate_abi, validate_operator_contract, validate_parallel_topology, verify_receipt_sha256, verify_sku_gates,
)
from mindie_sim.onboarding._gate_runners import acquire_coordinator_host_queue_lease, load_canonical_l0_sealer, normalize_profiler_records, run_profiler_bench_subprocess, validate_host_queue_grant
from mindie_sim.onboarding._lean_and_receipts import check_lean_non_inversion
from mindie_sim.onboarding._operator_contract import _build_default_planning_context, _verify_resolved_operator_dual_bound
from mindie_sim.onboarding._profiler_dispatch import _parse_json_or_jsonl_records, _split_disjoint_acquisitions, _validate_dispatch_exit, build_profiler_subgraph_plan
from mindie_sim.onboarding._profiler_plan import assign_plan_point_level, resolve_point_measured_binding
from mindie_sim.onboarding._sku_calibration import temporary_scaffold_profile
from mindie_sim.onboarding._whitebox_runner import eligible_replay_cases
from mindie_sim.onboarding.cliff_planner import verify_cliff_plan_integrity
from mindie_sim.onboarding.contracts import sanitize_public_warning
from mindie_sim.onboarding.tiling_axes import ARCH_PRESET_22, ARCH_PRESET_35
from mindie_sim.optimization.dual_bound import DagEdge, EventDagSpec, ResourceChoiceProgram, certify_lower_bound
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.operators.contracts import OperatorFamilySpec, ResolvedOperator
from mindie_sim.prediction.operators.registry import FamilyRegistry
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.primitives.schema import parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from _support.axes import REPRESENTATIVE_SKUS

pytestmark = pytest.mark.tier1
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPEATS = 12
A2_SKU = REPRESENTATIVE_SKUS["a2"]
A5_SKU = REPRESENTATIVE_SKUS["a5"]
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
    return repo, subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(), hashlib.sha256(content).hexdigest()


def _synthetic_repo(tmp_path: Path, *, family: str = "demo_op") -> Path:
    syn = tmp_path / f"syn_repo_{family}"
    for sub in ("scripts", "mindie_sim", "tests/operators"):
        (syn / sub).mkdir(parents=True)
    for rel, txt in (
        ("mindie_sim/__init__.py", '"""Stub."""\n'), ("scripts/check_arch_governance.py", "print('arch governance violations: 0')\n"),
        ("scripts/check_test_orthogonality.py", "print('summary: 0 findings (none); mode=--strict; modules=1/96')\n"), ("scripts/check_template_provenance.py", "print('template provenance violations: 0')\n"),
        (f"tests/operators/test_{family}.py", '"""Test."""\nimport pytest\npytestmark = pytest.mark.tier0\n'),
        ("tests/TAXONOMY.toml", f'schema_version = 1\ndomains = ["operators"]\nmax_lines = 600\nmax_modules = 8\n[[module]]\npath = "operators/test_{family}.py"\ntier = "tier0"\nowns = ["mindie_sim/prediction/operators/{family}", "mindie_sim/templates/{family}_spec.py"]\n'),
    ):
        (syn / rel).write_text(txt, encoding="utf-8")
    return syn


def _synthetic_profiler_root(tmp_path: Path, supported: tuple[str, ...] = ("demo_op",), *, abi: str = "aclnnDemoOpV1", tag: str = "main") -> Path:
    prof, pkg = tmp_path / f"syn_profiler_{tag}", tmp_path / f"syn_profiler_{tag}" / "mindie_profiler/bench"
    pkg.mkdir(parents=True, exist_ok=True); (prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8"); (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "operators.py").write_text(f"class BenchUnmodeledOperatorError(RuntimeError):\n    pass\nSUPPORTED = {set(supported)!r}\nABI = {abi!r}\ndef resolve_builder_strict(family, recipe, impl, dtypes=(), meta=None):\n    if family in SUPPORTED or recipe in SUPPORTED:\n        return (f'{{family}}_builder', ABI)\n    raise BenchUnmodeledOperatorError(f'unmodeled:{{family}}:{{recipe}}')\n", encoding="utf-8")
    return prof


def _lean_witness_row(case_id: str, work_ns: int, *, sim_ns: int, p99_ns: int) -> dict[str, Any]:
    cert, _ = certify_lower_bound(ResourceChoiceProgram(1, (work_ns,), ()), EventDagSpec((work_ns,), ()))
    p_dict, d_dict = {"num_resources": 1, "global_work": [work_ns], "site_modes": []}, {"node_quantum_ns": [work_ns], "edges": []}
    return {"case_id": case_id, "l_star_lean_ns": cert.l_star_lean_ns, "t_sim_ns": sim_ns, "t_silicon_p99_ns": p99_ns, "program": p_dict, "dag": d_dict, "program_dag_sha256": program_dag_sha256(p_dict, d_dict), "certificate": {"farkas": {"scale_d": cert.farkas.scale_d, "lambda_num": list(cert.farkas.lambda_num), "mu_num": list(cert.farkas.mu_num), "lower_bound_ns": cert.farkas.lower_bound_ns}, "path_edges": list(cert.path_edges), "sink_node_id": cert.sink_node_id, "l_star_lean_ns": cert.l_star_lean_ns}}


def _independent_engine_runner(pred_by_id: dict[str, int], *, default_ns: int = 20_000, clock_op: str = "elapsed_span", busy_override: dict[str, int] | None = None, prog_sha_by_id: dict[str, str] | None = None):
    table, b_map, p_map = dict(pred_by_id), dict(busy_override or {}), dict(prog_sha_by_id or {})

    class _Runner:
        def _exec(self, case: GateCase) -> EngineInterval:
            pred_ns, busy_ns = int(table.get(case.case_id, default_ns)), int(b_map.get(case.case_id, table.get(case.case_id, default_ns)))
            eng = Engine([Resource("device", 1_000_000.0, "ns", "fifo")], trace_limit=32)
            s_ev = eng.submit(case.boundary.start_event, {"device": 100.0})
            e_ev = eng.submit(case.boundary.end_event, {"device": float(pred_ns)}, after=(s_ev,))
            eng.advance(until=(e_ev,))
            diag = (("clock_operator", clock_op), ("busy_union_ns", busy_ns), ("elapsed_span_ns", pred_ns), ("program_dag_sha256", p_map.get(case.case_id, program_dag_sha256({"num_resources": 1, "global_work": [max(1, pred_ns // 2)], "site_modes": []}, {"node_quantum_ns": [max(1, pred_ns // 2)], "edges": []}))))
            return bind_engine_interval(case, eng, s_ev, e_ev, diagnostics=diag)

        def run_l1(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l2(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l3(self, case: GateCase) -> EngineInterval: return self._exec(case)

    return _Runner()


def _plan_fixture_bundle(plan: dict[str, Any], *, levels: tuple[str, ...] | None = None, clock_op: str = "elapsed_span", scale: float = 1.02, impl: str = "whitebox_spine", capture_dir: Path | None = None, omit_capture_p99: bool = False, overrides: dict[str, tuple[int, int]] | None = None) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    rows, preds, lean_rows, prog_shas, ov_map = [], {}, [], {}, dict(overrides or {})
    sku, family, target_id = str(plan["sku"]), str(plan.get("family") or ""), str(plan.get("target_id") or "")
    for split, pts in (("train", plan["train_points"]), ("heldout", plan["heldout_points"])):
        for idx, pt in enumerate(pts):
            pid = str(pt["point_id"])
            lvl = assign_plan_point_level(plan, pt) if levels is None else ("L1" if split == "train" else levels[idx % len(levels)])
            meas, pred = ov_map[pid] if pid in ov_map else ((10_000 + idx * 100 if split == "train" else 20_000 + idx * 250), int(round((10_000 + idx * 100 if split == "train" else 20_000 + idx * 250) * scale)))
            p99 = int(meas * 1.05)
            ranks, ev_ids = list(range(validate_parallel_topology(str(pt.get("parallel_topology") or "tp1")))), [f"{split[0]}-s-{idx}", f"{split[0]}-e-{idx}"]
            acq = "acq-train-src" if split == "train" else f"acq-held-{lvl.lower()}"
            cap_obj: dict[str, Any] = {"case_id": pid, "sku": sku, "split": split, "level": lvl, "acquisition_root": acq, "idx": idx, "start_event": "start", "end_event": "end", "clock_domain": "device_ns", "device_ranks": ranks, "clock_operator": clock_op, "event_ids": ev_ids, "measured_ns": meas, "busy_union_ns": meas if clock_op == "busy_union" else None, "elapsed_span_ns": meas if clock_op == "elapsed_span" else None}
            if lvl in {"L2", "L3"}:
                cap_obj["timed_nodes"], cap_obj["actual_shape"] = [f"{pid}:op0", f"{pid}:op1"], {pt["axis"]: pt["value"], "batch_size": pt["batch_size"], "query_tokens": pt["query_tokens"]}
            if not omit_capture_p99:
                cap_obj["measured_p99_ns"] = p99
            cap_bytes = json.dumps(cap_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            sha = hashlib.sha256(cap_bytes).hexdigest()
            if capture_dir is not None:
                capture_dir.mkdir(parents=True, exist_ok=True); (capture_dir / f"{sha}.json").write_bytes(cap_bytes)
            preds[pid] = pred
            row: dict[str, Any] = {
                "case_id": pid, "point_id": pid, "sku": sku, "split": split, "level": lvl, "acquisition_root": acq, "run_uid": f"run-{split[0]}-{lvl.lower()}-{idx}", "raw_sha256": sha, "capture_uri": f"capture://{split}/{lvl.lower()}/{sha}.json",
                "start_event": "start", "end_event": "end", "clock_domain": "device_ns", "event_ids": ev_ids, "device_ranks": ranks,
                "context_len": pt["value"] if pt["axis"] == "context_len" else None, "tokens": pt["value"] if pt["axis"] == "tokens" else None,
                "batch_size": pt["batch_size"], "query_tokens": pt["query_tokens"], "parallel_topology": pt["parallel_topology"], "topk": pt["topk"],
                "measured_ns": meas, "measured_p99_ns": p99, "clock_operator": clock_op, "busy_union_ns": meas if clock_op == "busy_union" else meas - 500, "elapsed_span_ns": meas if clock_op == "elapsed_span" else meas + 500,
                "implementation": impl, "operator_id": target_id,
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
    abi_rows = [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"], "executed_shape": {"batch_size": pt["batch_size"], "query_tokens": pt["query_tokens"], pt["axis"]: pt["value"], **({"topk": pt["topk"]} if pt.get("topk") is not None else {})}} for pt in plan["abi_rejection_points"]]
    return rows, preds, lean_rows, abi_rows, prog_shas


def _raw_l0_samples(sku: str, *, scale: float = 1.0, cann_version: str = "9.2.0") -> dict[str, Any]:
    out: dict[str, Any] = {"device_name": sku, "P01_datacopy": {"hbm_read_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.72 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "hbm_write_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.70 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "mte2_aiv_hit_bytes_per_s_per_core": [A2_ORACLE["mte2_aiv"] * scale * (0.75 + 0.001 * i) for i in range(SAMPLE_REPEATS)]}, "P20_host_dma_sync": {"host_h2d_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "host_d2h_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)]}}
    return {**out, "cann_version": cann_version} if cann_version else out


def _a2_probe_payload(sku: str = A2_SKU, *, endpoint: str = "prod-cluster-01.corp") -> dict[str, Any]:
    return {"aclrt_soc_name": sku, "endpoint_alias": endpoint, "dies_per_package": A2_ORACLE["dies"], "platform_config_ini": f"[SoCInfo]\nsoc_version={sku}\ncompute_arch={A2_ORACLE['arch_id']}\nai_core_cnt={A2_ORACLE['aic']}\nvector_core_cnt={A2_ORACLE['aiv']}\nl2_size={A2_ORACLE['l2']}\nub_size={A2_ORACLE['ub']}\n", "npu_smi_common": f"Chip ID : 0\nAicore Count : {A2_ORACLE['aic']}\nAicore Freq(MHZ) : 1650\n"}


def test_probe_soc_conflicts_capture_gate_and_unspoofable_status(tmp_path):
    synthetic_ip = ".".join(("198", "51", "100", "22"))
    ok_probe = tmp_path / "probe_ok.json"
    ok_probe.write_text(json.dumps(_a2_probe_payload(endpoint=f"prod-cluster-{synthetic_ip}.corp")), encoding="utf-8")
    res_ok = probe_soc(probe_json=ok_probe)
    assert res_ok["ok"] is True and res_ok["probe"]["endpoint_alias"].startswith("ep-") and "prod-cluster" not in json.dumps(res_ok) and synthetic_ip not in json.dumps(res_ok)
    for bad_name, payload in (("soc_conflict", {**_a2_probe_payload(), "soc_name": A2_SKU, "aclrt_soc_name": "OtherSilicon"}), ("smi_hbm", {**_a2_probe_payload(), "npu_smi_common": f"Chip ID : 0\nAicore Count : {A2_ORACLE['aic']}\nHBM Capacity(MB) : 1\n"}), ("bad_fab", {**_a2_probe_payload(), "fabric_channels": {"d2d_intra_node": 1.0}})):
        p = tmp_path / f"{bad_name}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        assert probe_soc(probe_json=p)["error"] == "silicon_profile_conflict", bad_name
    new_sku, full_probe = "AscendFuture_9999", tmp_path / "probe_new.json"
    full_probe.write_text(json.dumps({"aclrt_soc_name": new_sku, "arch_preset": A2_ORACLE["arch_id"], "architecture": "A2", "aic_count": 28, "aiv_count": 56, "hbm_bytes": 64 * 1024**3, "l2_bytes": 200 * 1024**2, "ub_bytes": 196_608, "dies_per_package": 1, "ranks_per_node": 8}), encoding="utf-8")
    prof_scaf = probe_soc(probe_json=full_probe, allow_scaffold=True, base_template_sku=A2_SKU)["scaffold_profile"][new_sku]
    new_plan, cap_dir, raw_l0 = synthesize_sku_512_400k_plan(sku=new_sku, core_count=28, l2_bytes=200 * 1024**2, profile_override=prof_scaf).to_dict(), tmp_path / "new_caps", tmp_path / "new_l0.json"
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(new_plan, capture_dir=cap_dir)
    raw_l0.write_text(json.dumps(_raw_l0_samples(new_sku)), encoding="utf-8")
    with pytest.raises(ModelingError, match="capture_file_missing"):
        run_sku_onboarding(probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas), lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="no_cap"))
    with pytest.raises(ModelingError, match="release_certification_requires_capture_files"):
        run_sku_onboarding(probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, require_capture_files=False, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas), lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="no_cap2"))
    run_new = run_sku_onboarding(probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, capture_root=cap_dir, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas), lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="new_sku"), out_dir=tmp_path / "new_sku_out")
    assert run_new["ok"] is True and run_new["status"] == "offline_reverify"
    off768_row = next(r for r in obs if r["point_id"] == "offgrid_ctx_768")
    orig_cap_path = cap_dir / f"{off768_row['raw_sha256']}.json"
    saved_cap_bytes, decoy_dir = orig_cap_path.read_bytes(), cap_dir / "0000_decoys"
    decoy_dir.mkdir(parents=True, exist_ok=True)
    for j in range(4100):
        (decoy_dir / f"d_{j:04d}.json").write_bytes(b"{}")
    nested_f = cap_dir / "audit-node.internal.example/lease-audituser/trial-001.json"
    nested_f.parent.mkdir(parents=True, exist_ok=True); nested_f.write_bytes(saved_cap_bytes); orig_cap_path.unlink()
    off768_row["capture_uri"] = "capture://audit-node.internal.example/lease-audituser/trial-001.json"
    prim_new, prof_new = run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"], run_new["stages"]["probe-soc"]["scaffold_profile"][new_sku]
    val_nested = validate_512_400k(sku=new_sku, primitives_json=prim_new, profiler_runner=lambda _p: obs, abi_rejection_results=abis, profile_override=prof_new, capture_root=cap_dir, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas))
    assert val_nested["ok"] is True and "audit-node" not in json.dumps(val_nested) and "trial-001" not in json.dumps(val_nested)
    assert verify_sku_gates(sku=new_sku, validation_report_json=val_nested, primitives_json=prim_new, lean_bounds_json=leans, profile_override=prof_new, capture_root=cap_dir, repo_root=_synthetic_repo(tmp_path, family="nested_s"))["status"] == "offline_reverify"
    nested_f.unlink()
    with pytest.raises(ModelingError, match="capture_file_missing"):
        verify_sku_gates(sku=new_sku, validation_report_json=val_nested, primitives_json=prim_new, lean_bounds_json=leans, profile_override=prof_new, capture_root=cap_dir, repo_root=_synthetic_repo(tmp_path, family="nested_del"))
    orig_cap_path.write_bytes(saved_cap_bytes); empty_caps, empty_repo = tmp_path / "empty_caps", _synthetic_repo(tmp_path, family="empty_c"); empty_caps.mkdir()
    for cap_arg in (empty_caps, None):
        with pytest.raises(ModelingError, match="capture_file_missing"):
            verify_sku_gates(sku=new_sku, validation_report_json=run_new["stages"]["validate-512-400k"], primitives_json=prim_new, lean_bounds_json=leans, profile_override=prof_new, capture_root=cap_arg, repo_root=empty_repo)
    spoof = verify_sku_gates(sku=new_sku, validation_report_json=run_new["stages"]["validate-512-400k"], primitives_json=prim_new, lean_bounds_json=leans, profile_override=prof_new, capture_root=cap_dir, acquisition_mode="live_silicon", repo_root=_synthetic_repo(tmp_path, family="spoof_s"))
    assert spoof["ok"] is False and spoof["status"] != "silicon_certified"
    warn_gate = verify_sku_gates(sku=new_sku, validation_report_json=run_new["stages"]["validate-512-400k"], primitives_json=prim_new, lean_bounds_json=leans, profile_override=prof_new, capture_root=cap_dir, stage2_lease={"lease_mode": "coordinator_host_queue", "lease_digest": "lease-0123456789abcdef", "fence_token": 7, "device_count": 1, "receipt_unconfirmed": True, "receipt_warning": "release_delayed:audit-node.internal.example"}, repo_root=_synthetic_repo(tmp_path, family="warn_s"))
    assert warn_gate["ok"] is True and warn_gate["receipt"]["receipt_unconfirmed"] is True and "audit-node" not in json.dumps(warn_gate["receipt"]) and verify_receipt_sha256(warn_gate["receipt"]) is True


def test_sealer_cann_manifest_conflicts_and_campaign_privacy(tmp_path):
    assert load_canonical_l0_sealer(tmp_path / "no_tools").__name__ == "mindie_sim.primitives.sealer"
    with pytest.raises(ModelingError, match="timeout_budget_outer_too_small"):
        TimeoutBudget((("l0", 300), ("fabric", 200)), cleanup_grace_s=60, outer_timeout_s=500)
    assert mindie_sim_main(["onboard-sku", "calibrate-l0-and-fabric", "--sku", A2_SKU]) == 2
    a5_cap = REPO_ROOT / "data/primitives/captures/arch35_a5_950dt.capture.json"
    a5_cap_sku = json.loads(a5_cap.read_text(encoding="utf-8"))["payload"]["device_name"]
    with pytest.raises(ModelingError, match="cann_version_provenance_conflict"):
        calibrate_l0_and_fabric(sku=a5_cap_sku, raw_l0_json=a5_cap, cann_version="9.3.0")
    cal_a5 = calibrate_l0_and_fabric(sku=a5_cap_sku, raw_l0_json=a5_cap, cann_version="9.2.0", campaign_id="lease-audituser-0abc123abc")
    assert cal_a5["ok"] is True and "audituser" not in json.dumps(cal_a5) and cal_a5["train_manifest"]["campaign_id"].startswith("camp-")
    over_l0 = tmp_path / "over_l0.json"
    over_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU, scale=2.5)), encoding="utf-8")
    cal_over = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=over_l0)
    assert any(c["reason"].startswith("p99_exceeds_nominal") for c in cal_over["metric_conflicts"]) and any(r["reason"] == "missing_samples" for r in cal_over["coverage_matrix"])


def test_lock3_dav2201_vs_dav3510_lightning_abi_and_strict_rejections():
    axes35, axes22 = family_default_tiling_axes("sparse_indexer", tile_dim=128, arch_preset=ARCH_PRESET_35), family_default_tiling_axes("sparse_indexer", tile_dim=128, arch_preset=ARCH_PRESET_22)
    assert validate_abi({"context_len": 1, "topk": 64}, axes35, arch_preset=ARCH_PRESET_35)[0] is True and validate_abi({"context_len": 2049, "topk": 2049}, axes35, arch_preset=ARCH_PRESET_35)[0] is True
    assert validate_abi({"context_len": 2049, "topk": 2049}, axes22, arch_preset=ARCH_PRESET_22)[1].startswith("abi_divisibility_violation:topk=2049")
    assert validate_abi({"topk": 8193}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_max_violation:topk=8193") and validate_abi({"topk": "2049"}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_type_violation")
    sp35 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 32, 192 * 1024**2), sku=A5_SKU)
    sp22 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, 100 * 1024**2), sku=A2_SKU)
    sp22_2050 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, 100 * 1024**2, topk_threshold=2050), sku=A2_SKU)
    assert not any(pt.topk == 2049 for pt in sp35.abi_rejection_points) and any(pt.topk == 2049 for pt in sp35.heldout_points)
    assert any(pt.topk == 8193 for pt in sp35.abi_rejection_points) and any(pt.topk == 2049 for pt in sp22.abi_rejection_points)
    assert not any(pt.topk == 2049 for pt in sp22_2050.heldout_points) and any(pt.topk == 2049 for pt in sp22_2050.abi_rejection_points)
    assert set(ANCHOR_CONTEXT_GRID) == {pt.value for pt in sp22.train_points if pt.axis == "context_len"} and set(OFF_GRID_CLIFF_HELDOUT_GRID).issubset({pt.value for pt in sp22.heldout_points if pt.axis == "context_len"})
    with pytest.raises(ModelingError, match="dcp_pcp_domain_invalid"):
        validate_parallel_topology("tp2_pcp4_dcp2")
    assert required_ranks_for_plan(sp22.to_dict()) == 4
    rejs = sp22.to_dict()["abi_rejection_points"]
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"]} for pt in rejs])["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": "abi_divisibility_violation:batch_size=4:note_topk=2049", "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": pt["value"], "topk": pt["topk"]}} for pt in rejs])["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": "abi_divisibility_violation:topk=2051:above=2048:divisor=1024", "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": pt["value"], "topk": pt["topk"]}} for pt in rejs])["ok"] is False


def test_capture_duration_override_clock_plan_ranks_and_capture_p99(tmp_path):
    plan, cap_dir = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict(), tmp_path / "caps"
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(plan, capture_dir=cap_dir)
    cap75 = {"case_id": "clk-75", "sku": A2_SKU, "split": "heldout", "acquisition_root": "acq-held-l1", "start_event": "start", "end_event": "end", "clock_domain": "device_ns", "device_ranks": [0], "clock_operator": "busy_union", "event_ids": ["s0", "e0"], "busy_union_ns": 7500, "samples_ns": [7500] * 12}
    cap75_b = json.dumps(cap75, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cap75_sha = hashlib.sha256(cap75_b).hexdigest()
    (cap_dir / f"{cap75_sha}.json").write_bytes(cap75_b)
    tampered = {**obs[-1], "case_id": "clk-75", "point_id": "clk-75", "raw_sha256": cap75_sha, "capture_uri": f"capture://heldout/{cap75_sha}.json", "event_ids": ["s0", "e0"], "device_ranks": [0], "clock_operator": "busy_union", "measured_ns": 110, "busy_union_ns": 110}
    with pytest.raises(ModelingError, match="capture_measured_duration_mismatch"):
        score_sealed_gate_cases([tampered], sku=A2_SKU, capture_root=cap_dir, require_capture_files=True, runner=_independent_engine_runner({"clk-75": 110}, clock_op="busy_union"))
    off768 = next(r for r in obs if r["point_id"] == "offgrid_ctx_768")
    with pytest.raises(ModelingError, match="clock_operator_plan_mismatch"):
        score_sealed_gate_cases([{**off768, "clock_operator": "busy_union", "measured_ns": off768["busy_union_ns"]}], sku=A2_SKU, expected_points=plan["heldout_points"], runner=_independent_engine_runner(preds, clock_op="busy_union"))
    mr_row = next(r for r in obs if r["parallel_topology"] == "tp2_pcp2_dcp2")
    with pytest.raises(ModelingError, match="observation_participating_ranks_mismatch"):
        score_sealed_gate_cases([{**mr_row, "device_ranks": [0]}], sku=A2_SKU, expected_points=plan["heldout_points"], runner=_independent_engine_runner(preds))
    nop99_dir, raw_l0 = tmp_path / "caps_nop99", tmp_path / "l0.json"
    obs_np, preds_np, leans_np, abis_np, pshas_np = _plan_fixture_bundle(plan, capture_dir=nop99_dir, omit_capture_p99=True)
    raw_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_l0)
    val_np = validate_512_400k(sku=A2_SKU, primitives_json=cal["primitives_payload"], profiler_runner=lambda _p: obs_np, abi_rejection_results=abis_np, capture_root=nop99_dir, runner=_independent_engine_runner(preds_np, prog_sha_by_id=pshas_np))
    gate_np = verify_sku_gates(sku=A2_SKU, validation_report_json=val_np, primitives_json=cal["primitives_payload"], lean_bounds_json=leans_np, capture_root=nop99_dir, repo_root=_synthetic_repo(tmp_path, family="nop99"))
    assert gate_np["ok"] is False and any(g["name"] == "lean_optimality_non_inversion" and "lean_captured_p99_missing" in g["detail"] for g in gate_np["receipt"]["gates"])


def test_operator_stub_template_ownership_identity_and_certify(tmp_path):
    repo_dir, rev, sha = _git_repo_with_blob(tmp_path, "ops-transformer", "kernels/demo_op.cpp", b"// demo\n")
    syn, prof_ok = _synthetic_repo(tmp_path, family="demo_op"), _synthetic_profiler_root(tmp_path, supported=("demo_op",), abi="aclnnDemoOpV1", tag="ok")
    scaf = scaffold_operator(family="demo_op", upstream_abi_arch22="aclnnDemoOpV1", upstream_abi_arch35="aclnnDemoOpV2", source_refs=[{"repository": "ops-transformer", "revision": rev, "relative_path": "kernels/demo_op.cpp", "source_sha256": sha}], target_root=syn, overwrite=True)
    target_rel = scaf["package_rel"]
    def _real_resolve(op: OpSpec, rctx) -> ResolvedOperator:
        prog = KernelProgram((Stage("vec_main", work={"vector": 4096.0}),))
        return ResolvedOperator(occurrence=rctx.occurrence, op=op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: prog, pipeline_spec={"stage": "v"}))
    reg_ok = FamilyRegistry([OperatorFamilySpec("demo_op", frozenset({"demo_op"}), frozenset({"demo_op", "demo_op:aclnnDemoOpV1"}), _real_resolve)])
    stub_chk = validate_operator_contract(target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_ok, family_registry=reg_ok)
    assert stub_chk["ok"] is False and any("operator_target_unimplemented_stub" in v for v in stub_chk["violations"])
    fake_rms = syn / "mindie_sim/templates/rmsnorm_spec.py"
    fake_rms.parent.mkdir(parents=True, exist_ok=True)
    fake_rms.write_text(f'"""Stub.\n\nUpstream-ABI: aclnnRmsNorm\n"""\nTEMPLATE_ID = "catlass.basic_pingpong"\nSOURCES = (SourceRef(repository="ops-transformer", revision="{rev}", relative_path="kernels/demo_op.cpp", source_sha256="{sha}"),)\nSemanticWorkItem = TensorRangeObligation = TilingAxes = object\n', encoding="utf-8")
    borrowed = validate_operator_contract(target_path=fake_rms, repo_root=syn, refs_root=repo_dir.parent, template_id="catlass.basic_pingpong", require_profiler_builder=False, require_taxonomy=False)
    assert borrowed["ok"] is False and any("template_id_not_owned_by_target" in v for v in borrowed["violations"])
    cat_chk = validate_operator_contract(target_path=REPO_ROOT / "mindie_sim/templates/catlass_catalog.py", repo_root=REPO_ROOT, template_id="catlass.basic_pingpong", require_profiler_builder=False, require_taxonomy=False, require_tiling_axes=False)
    assert cat_chk["sim_owner_dispatch"]["lowered"] is True and cat_chk["sim_owner_dispatch"]["lean_status"] == "verified"
    probe_op = OpSpec("op0", "demo_op", "demo_op", "decode", "bf16", (), (1, 64), (1, 64), (1, 64), (1, 64))
    ctx = _build_default_planning_context(probe_op)
    zero_res = ResolvedOperator(occurrence=ctx.occurrence, op=probe_op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: (), pipeline_spec={"stage": "v"}))
    assert _verify_resolved_operator_dual_bound(zero_res, ctx)[2] == "resolved_operator_program_zero_executable_events"
    common_py = syn / target_rel / "common.py"
    common_py.write_text(common_py.read_text(encoding="utf-8").replace('raise NotImplementedError(', 'return ('), encoding="utf-8")
    plan_doc, op_caps = generate_operator_cliff_plan(operator_id="demo_op:aclnnDemoOpV1", family="demo_op", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048)["plan"], tmp_path / "op_caps"
    op_obs, op_preds, op_leans, op_abis, op_shas = _plan_fixture_bundle(plan_doc, levels=("L1",), impl="aclnnDemoOpV1", capture_dir=op_caps)
    op_run = run_operator_onboarding(target_path=target_rel, operator_id="demo_op:aclnnDemoOpV1", family="demo_op", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048, captured_json=op_obs, abi_rejection_results=op_abis, lean_bounds_json=op_leans, capture_root=op_caps, repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_ok, family_registry=reg_ok, runner=_independent_engine_runner(op_preds, prog_sha_by_id=op_shas), out_dir=tmp_path / "op_out")
    assert op_run["ok"] is True and op_run["status"] == "offline_reverify" and verify_receipt_sha256(op_run["stages"]["certify"]["receipt"]) is True
    assert certify_operator(operator_id="demo_op:unimplemented", sku=A2_SKU, contract_report=op_run["stages"]["validate-contract"], score_report=op_run["stages"]["bench-and-score"], lean_bounds_json=op_leans, capture_root=op_caps, repo_root=syn)["ok"] is False
    with pytest.raises(ModelingError, match="capture_file_missing"):
        certify_operator(operator_id="demo_op:aclnnDemoOpV1", sku=A2_SKU, contract_report=op_run["stages"]["validate-contract"], score_report=op_run["stages"]["bench-and-score"], lean_bounds_json=op_leans, repo_root=syn)
    with pytest.raises(ModelingError, match="release_certification_requires_capture_files"):
        certify_operator(operator_id="demo_op:aclnnDemoOpV1", sku=A2_SKU, contract_report=op_run["stages"]["validate-contract"], score_report=op_run["stages"]["bench-and-score"], lean_bounds_json=op_leans, require_capture_files=False, repo_root=syn)
    op_spoof = certify_operator(operator_id="demo_op:aclnnDemoOpV1", sku=A2_SKU, contract_report=op_run["stages"]["validate-contract"], score_report=op_run["stages"]["bench-and-score"], lean_bounds_json=op_leans, capture_root=op_caps, acquisition_mode="live_silicon", repo_root=syn)
    assert op_spoof["ok"] is False and op_spoof["status"] != "silicon_certified"


@contextmanager
def _bad_acq(devs: list[int], ttl: int, fence: int = 1):
    yield {"lease_mode": "coordinator_host_queue", "lease_id": "l-bad", "fence_token": fence, "devices": devs, "ttl_seconds": ttl}


def test_host_queue_unified_grant_cleanup_and_profiler_roundtrip(tmp_path):
    for bad_g, req_d, min_t, pat in (
        ({"lease_id": "l1", "fence_token": 1, "devices": [], "ttl_seconds": 600}, 1, 120, "host_queue_lease_device_count_mismatch"),
        ({"lease_id": "l1", "fence_token": 1, "devices": [0, 1, 2, 3, 4], "ttl_seconds": 600}, 4, 120, "host_queue_lease_device_count_mismatch"),
        ({"lease_id": "l1", "fence_token": 1, "devices": [0, 1, 2, 3], "ttl_seconds": 1}, 4, 120, "host_queue_lease_ttl_below_subprocess_plus_grace"),
    ):
        with pytest.raises(ModelingError, match=pat):
            validate_host_queue_grant(bad_g, required_devices=req_d, min_ttl_seconds=min_t)
    rel_log, c_short, c_warn = tmp_path / "rel_log.txt", tmp_path / "c_short.py", tmp_path / "c_warn.py"
    c_short.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\nact = args[args.index('--action') + 1]\nif act == 'acquire':\n    print(json.dumps({{'outcome':'success','status':'ready','data':{{'lease_id':'lease-audituser-1','fence_token':5,'devices':[0,1,2]}}}}))\nelse:\n    Path({str(rel_log)!r}).write_text('released', encoding='utf-8')\n    print(json.dumps({{'outcome':'success','status':'ok'}}))\n", encoding="utf-8")
    c_warn.write_text(f"#!{sys.executable}\nimport json, sys\nargs = sys.argv[1:]\nact = args[args.index('--action') + 1]\nif act == 'acquire':\n    print(json.dumps({{'outcome':'success','status':'ready','receipt_unconfirmed':True,'receipt_warning':'PermissionError: audituser','data':{{'lease_id':'lease-audituser-2','fence_token':7,'devices':[0,1,2,3]}}}}))\nelse:\n    print(json.dumps({{'outcome':'success','status':'ok'}}))\n", encoding="utf-8")
    c_short.chmod(0o755); c_warn.chmod(0o755)
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        with acquire_coordinator_host_queue_lease("audit-node.internal.example", npu_count=4, ttl_seconds=180, min_subprocess_timeout_s=120, coordinator_bin=str(c_short)):
            pass
    assert rel_log.read_text(encoding="utf-8") == "released" and "audituser" not in sanitize_public_warning("PermissionError: audituser") and "permissionerror" not in sanitize_public_warning("PermissionError: audituser")
    with acquire_coordinator_host_queue_lease("audit-node.internal.example", npu_count=4, ttl_seconds=180, min_subprocess_timeout_s=120, coordinator_bin=str(c_warn)) as linfo:
        assert linfo["devices"] == [0, 1, 2, 3]
    assert linfo["receipt_unconfirmed"] is True and "audituser" not in linfo["receipt_warning"] and "permissionerror" not in linfo["receipt_warning"]
    ran_cb: list[bool] = []
    plan_doc = generate_operator_cliff_plan(operator_id="live:v1", family="live", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"])["plan"]
    for pat, fn_cb in (
        ("host_queue_lease_", lambda: calibrate_l0_and_fabric(sku=A2_SKU, live_bench=True, l0_runner=lambda _s, _t: (ran_cb.append(True) or {}), lease_acquirer=lambda _s, _t: _bad_acq([], 1, fence=0))),
        ("host_queue_lease_device_count_mismatch", lambda: calibrate_l0_and_fabric(sku=A2_SKU, live_bench=True, endpoint_alias="audit-node.internal.example", l0_runner=lambda _s, _t: (ran_cb.append(True) or {}), coordinator_bin=str(c_short))),
        ("host_queue_lease_device_count_mismatch", lambda: validate_512_400k(sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", profiler_runner=lambda _p: (ran_cb.append(True) or []), coordinator_bin=str(c_short))),
        ("host_queue_lease_device_count_mismatch", lambda: bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", bench_dispatcher=lambda _p, _t: (ran_cb.append(True) or []), coordinator_bin=str(c_short))),
        ("host_queue_lease_device_count_mismatch", lambda: bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, lease_acquirer=lambda _s, _t: _bad_acq([0, 1, 2, 3, 4], 900))),
    ):
        with pytest.raises(ModelingError, match=pat):
            fn_cb()
    assert not ran_cb

    # H6 & R5 #1-#7: Lowered 1-layer L3, BNSD/513/5-page/grouped_matmul L2, world-size-separated L3, receipt actual_shape join, strict ABI & multi-rank critical path!
    b4_plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=20, l2_bytes=A2_ORACLE["l2"]).to_dict()
    b4_sub = build_profiler_subgraph_plan(b4_plan, sku=A2_SKU)
    b4_by_id = {c["case_id"]: c for c in b4_sub["cases"]}
    assert b4_sub["world_size"] == 4 and {c["level"] for c in b4_sub["cases"]} == {"L1", "L2", "L3"}
    assert all(c["nodes"][0]["family"] != "sku_suite" and c["nodes"][0].get("builder") for c in b4_sub["cases"])
    assert b4_by_id["anchor_ctx_512"]["shape"]["seq_len"] == 513 and b4_by_id["anchor_ctx_512"]["shape"]["page_count"] == 5 and b4_by_id["anchor_ctx_512"]["shape"]["query_layout"] == "BNSD"
    assert b4_by_id["cross_b16_ctx_4097"]["level"] == "L2" and b4_by_id["cross_b16_ctx_4097"]["nodes"][1]["builder"] == "grouped_matmul" and b4_by_id["cross_b16_ctx_4097"]["nodes"][1]["shape"]["e"] == 16
    assert b4_by_id["cores_wave_batch_x_kv_heads_128_minus1_2559"]["shape"]["sequences"] == 2559 and len(b4_by_id["offgrid_ctx_200000"]["nodes"]) == 15 and len(b4_by_id["cross_tp2_pcp2_dcp4_ctx_400000"]["nodes"]) == 19
    n200 = {n["op_id"]: n for n in b4_by_id["offgrid_ctx_200000"]["nodes"]}
    n400 = {n["op_id"]: n for n in b4_by_id["cross_tp2_pcp2_dcp4_ctx_400000"]["nodes"]}
    assert n200["s0.l0.kv_store"]["shape"]["width"] == 576 and n400["s0.l0.dcp_merge"]["nbytes"] == 4104 and n400["s0.l0.dcp_merge"]["group_size"] == 4 and n400["s0.l0.dcp_merge"]["dtype"] == "bf16" and n400["s0.l0.output_reduce"]["nbytes"] == 4096 and n400["s0.l0.output_reduce"]["dtype"] == "bf16" and n400["s0.l0.dcp_update"]["dtype"] == "fp32" and n400["s0.l0.dcp_update"]["shape"]["partial_count"] == 4
    assert b4_by_id["abi_reject_topk_attr_2049"]["shape"]["top_k"] == 2049 and b4_by_id["abi_reject_topk_attr_2049"]["shape"]["enforce_exact_topk"] is True
    b4_splits = _split_disjoint_acquisitions(b4_plan, domain=None, dry_run=False)
    assert [t for t, _, _ in b4_splits] == ["train", "heldout-l1", "heldout-l2", "heldout-l3-w1", "heldout-l3-w4"] and b4_splits[3][2] == b4_splits[4][2] and len({s[2] for s in b4_splits[:4]}) == 4
    b16_full = {**b4_by_id["cross_b16_ctx_4097"]["shape"], "m": 256, "n": 128, "k": 512, "e": 16, "matrix_groups": 16, "dtype": "bf16"}
    l3_ns400 = {n["node_id"]: n["shape"] for n in b4_by_id["cross_tp2_pcp2_dcp4_ctx_400000"]["nodes"]}
    l3_full = {**b4_by_id["cross_tp2_pcp2_dcp4_ctx_400000"]["shape"], "dtype": "fp32", "partial_count": 4, "node_shapes": l3_ns400}
    l3_tn = [n["node_id"] for n in b4_by_id["cross_tp2_pcp2_dcp4_ctx_400000"]["nodes"]]
    for pat, recs in (
        ("profiler_actual_geometry_mismatch", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": {"sequences": 1, "q_len": 16, "tokens": 16, "seq_len": 4098, "kv_heads": 1, "heads": 16, "head_dim": 512}}]),
        ("profiler_bench_receipt_failed.*missing_status", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "actual_shape": b4_by_id["anchor_ctx_512"]["shape"]}]),
        ("profiler_actual_geometry_missing", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": {"tokens": 16, "seq_len": 4098}}]),
        ("profiler_bench_receipt_missing", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "diagnostic", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": b4_by_id["anchor_ctx_512"]["shape"], "timed_nodes": ["anchor_ctx_512:op0"]}]),
        ("profiler_bench_receipt_missing", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "acquisition_id": "acq-a", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "split": "heldout", "acquisition_id": "acq-b", "status": "ok", "actual_shape": b4_by_id["anchor_ctx_512"]["shape"], "timed_nodes": ["anchor_ctx_512:op0"]}]),
        ("profiler_actual_shape_missing", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True, "requested_shape": b4_by_id["anchor_ctx_512"]["shape"]}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "status": "ok", "requested_shape": b4_by_id["anchor_ctx_512"]["shape"]}]),
        ("page_count_block_table_conflict", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": {**b4_by_id["anchor_ctx_512"]["shape"], "page_count": 5, "block_table": [1, 4]}}]),
        ("profiler_plan_level_mismatch", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_200000", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "level": "L1", "boundary": "model_step", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "offgrid_ctx_200000", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": b4_by_id["offgrid_ctx_200000"]["shape"], "timed_nodes": [n["node_id"] for n in b4_by_id["offgrid_ctx_200000"]["nodes"]]}]),
        ("profiler_actual_geometry_missing.*node_shapes", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_200000", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "model_step", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "offgrid_ctx_200000", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": {**b4_by_id["offgrid_ctx_200000"]["shape"], "m": 16, "n": 128, "k": 512, "e": 16, "matrix_groups": 16}, "timed_nodes": [n["node_id"] for n in b4_by_id["offgrid_ctx_200000"]["nodes"]]}]),
        ("profiler_observation_status_invalid|profiler_observation_unqualified", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "status": "debug_only", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}]),
        ("profiler_case_complete_failed", [{"schema": "mindie-bench-observation/1", "record_type": "case_complete", "case_id": "anchor_ctx_512", "run_id": "r1", "status": "unqualified", "summary": {}}]),
        ("profiler_sample_duration_invalid", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.0, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": b4_by_id["anchor_ctx_512"]["shape"], "timed_nodes": ["anchor_ctx_512:op0"]}]),
        ("profiler_timed_nodes_foreign_case_prefix", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": b4_by_id["anchor_ctx_512"]["shape"], "timed_nodes": ["other:op0"]}]),
        ("profiler_timed_nodes_plan_mismatch", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": b16_full, "timed_nodes": ["cross_b16_ctx_4097:s0.l0.paged_mla_decode"]}]),
        ("profiler_timed_nodes_order_violation|foreign_case_prefix", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": b16_full, "timed_nodes": ["other:s0.l0.v_expand", "other:s0.l0.paged_mla_decode"]}]),
        ("profiler_timed_nodes_order_violation", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": b16_full, "timed_nodes": ["cross_b16_ctx_4097:s0.l0.v_expand", "cross_b16_ctx_4097:s0.l0.paged_mla_decode"]}]),
        ("profiler_timed_nodes_duplicate", [r for rk in range(4) for r in ({"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}.base.eager.t0", "acquisition_id": "acq-mr", "trial": 0, "repeat_index": 0, "rank": rk, "split": "heldout", "boundary": "model_step", "timing_mode": "eager", "duration_ms": 0.1, "sample_id": f"s-{rk}", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}", "rank": rk, "status": "ok", "actual_shape": l3_full, "timed_nodes": l3_tn if rk < 3 else l3_tn + [l3_tn[-1]]})]),
        ("profiler_plan_level_mismatch", [r for rk in range(4) for r in ({"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}.base.eager.t0", "acquisition_id": "acq-mr", "trial": 0, "repeat_index": 0, "rank": rk, "split": "heldout", "level": "L3" if rk < 3 else "L1", "boundary": "model_step", "timing_mode": "eager", "duration_ms": 0.1, "sample_id": f"s-{rk}", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}", "rank": rk, "status": "ok", "actual_shape": l3_full, "timed_nodes": l3_tn})]),
        ("profiler_actual_geometry_mismatch|profiler_rank_actual_shape_mismatch", [r for rk in range(4) for r in ({"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}.base.eager.t0", "acquisition_id": "acq-mr", "trial": 0, "repeat_index": 0, "rank": rk, "split": "heldout", "boundary": "model_step", "timing_mode": "eager", "duration_ms": 0.1, "sample_id": f"s-{rk}", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}", "rank": rk, "status": "ok", "actual_shape": l3_full if rk < 3 else {**l3_full, "partial_count": 2}, "timed_nodes": l3_tn})]),
        ("multirank_incomplete_rank_samples", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}.base.eager.t0", "acquisition_id": "acq-mr", "trial": 0, "repeat_index": rep, "rank": rk, "split": "heldout", "boundary": "model_step", "timing_mode": "eager", "duration_ms": 0.03 + 0.01 * (rk == 3), "sample_id": f"s-{rk}-{rep}", "sku": A2_SKU, "validated": True, "qualified": True} for rk in range(4) for rep in (range(4) if rk < 3 else range(1))]),
        ("profiler_abi_rejection_unexpectedly_accepted", [{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "abi_reject_topk_attr_2049", "run_id": "r1.base.eager.t0", "trial": 0, "split": "abi_rejection", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True, "actual_shape": {"top_k": 2048, "seq_len": 2049}}]),
    ):
        with pytest.raises(ModelingError, match=pat):
            normalize_profiler_records(recs, plan_doc=b4_plan, sku=A2_SKU)
    leak_cap = tmp_path / "leak_caps"
    with pytest.raises(ModelingError, match="private_host_path_rejected"):
        normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "anchor_ctx_512", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "train", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "anchor_ctx_512", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": {**b4_by_id["anchor_ctx_512"]["shape"], "source_file": "/".join(("", "Users", "audituser", "private", "model.bin"))}, "timed_nodes": ["anchor_ctx_512:op0"]}], plan_doc=b4_plan, sku=A2_SKU, capture_out_dir=leak_cap)
    assert not list(leak_cap.glob("*"))
    with pytest.raises(ModelingError, match="profiler_bench_record_not_mapping"):
        _parse_json_or_jsonl_records("[{}, 123]")


def test_capture_invariants_and_profiler_dispatch_roundtrip(tmp_path):
    plan_doc = generate_operator_cliff_plan(operator_id="live:v1", family="live", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"])["plan"]
    b4_plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=20, l2_bytes=A2_ORACLE["l2"]).to_dict()
    b4_by_id = {c["case_id"]: c for c in build_profiler_subgraph_plan(b4_plan, sku=A2_SKU)["cases"]}
    b16_tn = [n["node_id"] for n in b4_by_id["cross_b16_ctx_4097"]["nodes"]]
    l2_cap, leak_rep = tmp_path / "l2_caps", tmp_path / "leak_rep.json"
    b16_ns = {n["node_id"]: n["shape"] for n in b4_by_id["cross_b16_ctx_4097"]["nodes"]}
    l2_norm, _ = normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r2.base.stream_saturated.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "stream_saturated", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r2", "rank": 0, "status": "ok", "actual_shape": b4_by_id["cross_b16_ctx_4097"]["shape"], "node_shapes": b16_ns, "timed_nodes": b16_tn}], plan_doc=b4_plan, sku=A2_SKU, capture_out_dir=l2_cap)
    rms127_norm, _ = normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "tile_128_minus1_127", "run_id": "r3.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.02, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "tile_128_minus1_127", "run_id": "r3", "rank": 0, "status": "ok", "actual_shape": {"tokens": 127, "width": 2048}, "timed_nodes": ["tile_128_minus1_127:op0"]}], plan_doc=b4_plan, sku=A2_SKU, capture_out_dir=l2_cap)
    assert l2_norm[0]["clock_operator"] == "elapsed_span" and rms127_norm[0]["tokens"] == 127
    stripped_l2 = {k: v for k, v in l2_norm[0].items() if k not in {"timed_nodes", "actual_shape", "measured_binding", "plan_key"}}
    for bad_cap_row, pat in (({**stripped_l2, "target_op_ids": ["s0.l0.norm1"]}, "capture_target_op_ids_mismatch|l2_op_set_mismatch"), ({**rms127_norm[0], "acquisition_root": "acq-forged"}, "capture_acquisition_root_mismatch"), ({**rms127_norm[0], "target_family": "attention", "target_op_id": "s0.l0.paged_mla_decode", "measured_binding": {}}, "capture_target_family_mismatch|l1_plan_target_mismatch"), ({**rms127_norm[0], "debug_cross_level": True}, "observation_debug_cross_level_forbidden"), ({**rms127_norm[0], "busy_union_ns": 999_999, "elapsed_span_ns": 20_000}, "observation_busy_union_exceeds_elapsed_span")):
        with pytest.raises(ModelingError, match=pat):
            score_sealed_gate_cases([bad_cap_row], sku=A2_SKU, capture_root=l2_cap, require_capture_files=True, runner=WhiteboxSpineRunner(clock_operator="elapsed_span"))
    with pytest.raises(ModelingError, match="l2_timed_nodes_missing"):
        score_sealed_gate_cases([stripped_l2], sku=A2_SKU, expected_points=b4_plan["heldout_points"], runner=WhiteboxSpineRunner(clock_operator="elapsed_span"))
    with pytest.raises(ModelingError, match="profiler_actual_geometry_mismatch.*dtype|profiler_actual_geometry_missing.*dtype"):
        normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r4.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r4", "rank": 0, "status": "ok", "actual_shape": b4_by_id["cross_b16_ctx_4097"]["shape"], "node_shapes": {**b16_ns, b16_tn[1]: {**b16_ns[b16_tn[1]], "dtype": "int8"}}, "timed_nodes": b16_tn}], plan_doc=b4_plan, sku=A2_SKU)
    with pytest.raises(ModelingError, match="duplicate_heldout_physical_capture"):
        score_sealed_gate_cases([{**rms127_norm[0], "case_id": "d1", "point_id": "d1", "run_uid": "u1", "event_ids": ["a", "b"]}, {**rms127_norm[0], "case_id": "d2", "point_id": "d2", "run_uid": "u2", "event_ids": ["a"]}], sku=A2_SKU, runner=_independent_engine_runner({"d1": 20_000, "d2": 20_000}))
    with pytest.raises(ModelingError, match="duplicate_train_physical_capture"):
        score_sealed_gate_cases([{**rms127_norm[0], "case_id": "t1", "point_id": "t1", "split": "train", "run_uid": "u1"}, {**rms127_norm[0], "case_id": "t2", "point_id": "t2", "split": "train", "run_uid": "u2"}], sku=A2_SKU, runner=_independent_engine_runner({"t1": 20_000, "t2": 20_000}))
    for bad_cap_obj, pat in (({"clock_operator": "elapsed_span", "elapsed_span_ns": 10000, "start_event": "s", "end_event": "e", "device_ranks": [0], "event_ids": ["s", "e"]}, "capture_case_id_missing"), ({"case_id": "c_l4", "split": "heldout", "level": "L4", "acquisition_root": "acq-x", "clock_operator": "elapsed_span", "elapsed_span_ns": 10000, "start_event": "s", "end_event": "e", "device_ranks": [0], "event_ids": ["s", "e"]}, "capture_level_invalid"), ({"case_id": "c_dbg", "split": "heldout", "level": "L1", "acquisition_root": "acq-x", "debug_cross_level": True, "clock_operator": "elapsed_span", "elapsed_span_ns": 10000, "start_event": "s", "end_event": "e", "device_ranks": [0], "event_ids": ["s", "e"]}, "capture_debug_cross_level_forbidden"), ({"case_id": "c_st", "split": "heldout", "level": "L1", "acquisition_root": "acq-x", "status": "ok", "metadata": {"status": "failed"}, "clock_operator": "elapsed_span", "elapsed_span_ns": 10000, "start_event": "s", "end_event": "e", "device_ranks": [0], "event_ids": ["s", "e"]}, "capture_status_invalid"), ({"case_id": "c_notn", "split": "heldout", "level": "L2", "acquisition_root": "acq-x", "clock_operator": "elapsed_span", "elapsed_span_ns": 10000, "start_event": "s", "end_event": "e", "device_ranks": [0], "event_ids": ["s", "e"]}, "capture_timed_nodes_missing")):
        cb = json.dumps(bad_cap_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        csha = hashlib.sha256(cb).hexdigest()
        (l2_cap / f"{csha}.json").write_bytes(cb)
        cid_b = str(bad_cap_obj.get("case_id") or "c_noid")
        with pytest.raises(ModelingError, match=pat):
            score_sealed_gate_cases([{**rms127_norm[0], "case_id": cid_b, "point_id": cid_b, "level": str(bad_cap_obj.get("level") or "L1") if str(bad_cap_obj.get("level") or "L1") in {"L1", "L2", "L3"} else "L1", "acquisition_root": "acq-x", "raw_sha256": csha, "capture_uri": f"capture://x/{csha}.json", "event_ids": ["s", "e"], "device_ranks": [0], "start_event": "s", "end_event": "e", "measured_ns": 10000, "elapsed_span_ns": 10000, "busy_union_ns": 9000}], sku=A2_SKU, capture_root=l2_cap, require_capture_files=True, runner=_independent_engine_runner({cid_b: 10000}))
    bar, seen_rec, prof22, prof35 = threading.Barrier(2), {}, copy.deepcopy(ascend_profiles()[A2_SKU]), copy.deepcopy(ascend_profiles()[A5_SKU])
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
    t1.start(); t2.start(); t1.join(timeout=10); t2.join(timeout=10)
    p950 = synthesize_sku_512_400k_plan(sku="Audit950Scaffold", core_count=20, l2_bytes=A2_ORACLE["l2"], profile_override=prof22).to_dict()
    assert seen_rec.get("a2") == "ascend_paged_mla_decode_arch22" and "AuditScaffold" not in ascend_profiles() and resolve_point_measured_binding(p950, p950["train_points"][0], sku="Audit950Scaffold", profile_override=prof22)["measured_binding"]["recipe"] == "ascend_paged_mla_decode_arch22"
    for pat, bad_fn in (("cliff_plan_sha256_tampered", lambda: verify_cliff_plan_integrity({**b4_plan, "arch_preset": ARCH_PRESET_35})), ("unknown_sku_profile", lambda: synthesize_sku_512_400k_plan(sku="NoSuchSKU", core_count=20, l2_bytes=A2_ORACLE["l2"]))):
        with pytest.raises(ModelingError, match=pat):
            bad_fn()
    _validate_dispatch_exit(0, [{"status": "dry_run"}], [], {}, "", is_remote=False, dry_run=True); _validate_dispatch_exit(0, [{"status": "dry_run"}], [], {}, "", is_remote=True, dry_run=True)
    lw_bad = _lean_witness_row("w-bad", 1000, sim_ns=2000, p99_ns=2000); lw_bad["certificate"]["sink_node_id"] = -0.5
    assert check_lean_non_inversion([lw_bad], expected_case_timings={"w-bad": (2000, 1500, 2000, lw_bad["program_dag_sha256"])})[0] is False
    with pytest.raises(ModelingError, match="private_host_path_rejected"):
        validate_512_400k(sku=A2_SKU, profiler_runner=lambda _: [], abi_rejection_results=[{"point_id": "abi_reject_topk_attr_2049", "rejected": True, "abi_error": "/".join(("", "Users", "audituser", "p.json"))}], report_out=leak_rep)
    assert not leak_rep.exists()
    with pytest.raises(ModelingError, match="clock_operator_plan_mismatch"):
        normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_768", "run_id": "r1.base.stream_saturated.t0", "trial": 0, "split": "heldout", "boundary": "device_kernel", "timing_mode": "stream_saturated", "duration_ms": 0.0075, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}], plan_doc=plan_doc, sku=A2_SKU)
    cap_out = tmp_path / "prof_caps"
    norm_obs, norm_abi = normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_768", "run_id": "run001.base.eager.t0", "trial": 0, "split": "heldout", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.020, "sample_id": "obs-0", "rank": None, "sku": A2_SKU, "validated": True, "qualified": True, "identity": {"level": "operator", "implementation": "aclnnLive"}}, {"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_768", "run_id": "run001.base.stream_saturated.t0", "trial": 0, "split": "heldout", "boundary": "device_kernel", "timing_mode": "stream_saturated", "duration_ms": 0.0075, "sample_id": "obs-1", "rank": None, "sku": A2_SKU, "validated": True, "qualified": True, "identity": {"level": "operator", "implementation": "aclnnLive"}}, {"schema": "mindie-bench-observation/1", "record_type": "case_complete", "case_id": "offgrid_ctx_768", "run_id": "run001", "split": "heldout", "status": "ok", "summary": {"op": "live", "actual_shape": {"context_len": 768, "batch_size": 1, "query_tokens": 1}, "participating_ranks": [2]}}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "offgrid_ctx_768", "run_id": "run001", "status": "ok", "actual_shape": {"context_len": 768, "batch_size": 1, "query_tokens": 1}, "timed_nodes": ["offgrid_ctx_768:op0"]}, {"schema": "mindie-bench-receipt/1", "case_id": "abi_1", "status": "unsupported_binding", "reason": "abi_divisibility_violation:topk=2049", "executed_shape": {"context_len": 2049, "topk": 2049, "batch_size": 1, "query_tokens": 1}}, {"schema": "mindie-bench-receipt/1", "case_id": "abi_1", "status": "unsupported_binding", "reason": "kernel_symbol_missing", "executed_shape": {"context_len": 2049, "topk": 2049, "batch_size": 1, "query_tokens": 1}}], plan_doc=plan_doc, sku=A2_SKU, endpoint_alias="audit-node.internal.example", capture_out_dir=cap_out)
    assert len(norm_obs) == 1 and norm_obs[0]["device_ranks"] == [2] and norm_obs[0]["context_len"] == 768 and norm_obs[0]["clock_operator"] == "elapsed_span" and "sim_case" in norm_obs[0] and len(norm_abi) == 2 and "audit-node" not in json.dumps(norm_obs) and norm_obs[0]["run_uid"].startswith("run-")
    verify_capture_file_consistency("offgrid_ctx_768", CaptureBoundary(norm_obs[0]["raw_sha256"], norm_obs[0]["acquisition_root"], norm_obs[0]["run_uid"], (), tuple(norm_obs[0]["event_ids"]), tuple(norm_obs[0]["device_ranks"]), "device_monotonic", "k_start", "k_end", "device_kernel", "ns", norm_obs[0]["capture_uri"]), cap_out / f"{norm_obs[0]['raw_sha256']}.json")
    abi_plan = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, A2_ORACLE["l2"]), sku=A2_SKU).to_dict()
    abi_pt = abi_plan["abi_rejection_points"][0]; abi_pid, abi_err, abi_by_id = abi_pt["point_id"], abi_pt["abi_expected_error"], {abi_pt["point_id"]: abi_pt}
    for rc_v, bad_recs in ((2, [{"case_id": abi_pid, "status": "unsupported_binding", "reason": abi_err, "requested_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"]}}]), (2, [{"case_id": abi_pid, "status": "unsupported_binding", "reason": abi_err, "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"], "sparse_count": abi_pt["topk"] - 1}}]), (0, [{"case_id": "offgrid_ctx_768", "status": "failed", "reason": "kernel_symbol_missing"}]), (2, [{"case_id": abi_pid, "status": "unsupported_binding", "reason": abi_err, "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"]}}, {"case_id": abi_pid, "status": "unsupported_binding", "reason": "kernel_symbol_missing", "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"]}}])):
        with pytest.raises(ModelingError, match="profiler_bench_unexpected_partial_failure|profiler_shape_alias_conflict"):
            _validate_dispatch_exit(rc_v, [{"status": "ok" if rc_v == 0 else "partial"}], bad_recs, abi_by_id, "", is_remote=False)
    fake_prof = tmp_path / "fake_prof_sub"; (fake_prof / "mindie_profiler/bench").mkdir(parents=True); (fake_prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (fake_prof / "mindie_profiler/__main__.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\nassert args[:2] == ['bench', 'subgraph']\n"
        "plan = json.loads(Path(args[args.index('--plan') + 1]).read_text(encoding='utf-8'))\n"
        "obs_p, out_p = Path(args[args.index('--observations') + 1]), Path(args[args.index('--out') + 1])\n"
        "rk, ws = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))\n"
        f"has_abi = any(c['case_id'] == {abi_pid!r} for c in plan['cases'])\n"
        "c0 = next(c for c in plan['cases'] if c['split'] != 'abi_rejection')\n"
        "cid, sp, rid = c0['case_id'], c0['split'], f\"{plan['plan_id']}_u{rk}\"\n"
        "obs_p.write_text(json.dumps({'schema':'mindie-bench-observation/1','record_type':'observation','plan_id':plan['plan_id'],'case_id':cid,'run_id':f'{rid}.base.eager.t0','trial':0,'repeat_index':0,'rank':rk,'split':sp,'timing_mode':'eager','duration_ms':0.02 + 0.001*rk,'sample_id':f's{rk}','sku':plan['sku'],'validated':True,'qualified':True}) + '\\n', encoding='utf-8')\n"
        f"recs = [{{'case_id': cid, 'status': 'ok', 'actual_shape': c0.get('shape') or {{}}, 'timed_nodes': [n['node_id'] for n in c0['nodes']]}}] + ([{{'case_id': {abi_pid!r}, 'status': 'unsupported_binding', 'reason': {abi_err!r}, 'executed_shape': {{'batch_size': 1, 'query_tokens': 1, 'context_len': {abi_pt['value']}, 'topk': {abi_pt['topk']}}}}}] if has_abi else [])\n"
        "Path(f'{out_p}.rank{rk}.json' if ws > 1 else str(out_p)).write_text(json.dumps({'schema':'mindie-bench-subgraph-run/1','status':'partial' if has_abi else 'ok','plan_id':plan['plan_id'],'run_id':rid,'records':recs}), encoding='utf-8')\n"
        "sys.exit(2 if has_abi else 0)\n", encoding="utf-8",
    )
    obs_l, abi_l = run_profiler_bench_subprocess(abi_plan, sku=A2_SKU, profiler_root=fake_prof, allowed_phy_ids=[0, 1, 2, 3], fence_token=7)
    ho_mr, ho_w1 = next(o for o in obs_l if o["case_id"] == "cross_tp2_pcp2_dcp2_ctx_200000"), next(o for o in obs_l if o["split"] == "heldout" and o["device_ranks"] == [0])
    assert ho_mr["device_ranks"] == [0, 1, 2, 3] and ho_mr["acquisition_root"] == ho_w1["acquisition_root"] and len(abi_l) == 1 and abi_l[0]["point_id"] == abi_pid
    for fn_call in (lambda: validate_512_400k(sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", profiler_root=fake_prof, lease_acquirer=lambda _s, _t: _bad_acq([0, 1, 2, 3], 600)), lambda: bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", profiler_root=fake_prof, lease_acquirer=lambda _s, _t: _bad_acq([0, 1, 2, 3], 900))):
        with pytest.raises(ModelingError, match="onboarding_endpoint_unknown|onboarding_json_read_failed"):
            fn_call()


def test_whitebox_sku_check_2d_kv_heads_dcp_compose_and_real_dag(tmp_path):
    # Items #4, #5, #7: B4 sim_case rejected on A5 plan; 2D bxkv checks local_kv_heads; dcp compose works; dependent vs independent L2 DAGs differ!
    b4_case = {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    base_l1 = {"case_id": "wb-l1", "point_id": "wb-l1", "sku": A2_SKU, "split": "heldout", "level": "L1", "acquisition_root": "acq-wb-l1", "run_uid": "r1", "raw_sha256": hashlib.sha256(b"wb-l1").hexdigest(), "event_ids": ["s1", "e1"], "device_ranks": [0], "measured_ns": 2500, "measured_p99_ns": 2600, "clock_operator": "busy_union", "busy_union_ns": 2500, "elapsed_span_ns": 2600, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "target_family": "rmsnorm", "sim_case": b4_case}
    with pytest.raises(ModelingError, match="whitebox_materialized_sku_mismatch"):
        score_sealed_gate_cases([{**base_l1, "sku": A5_SKU}], sku=A5_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))

    mr_l3 = {**base_l1, "case_id": "wb-mr", "point_id": "wb-mr", "level": "L3", "parallel_topology": "tp2_pcp2_dcp2", "device_ranks": [0, 1, 2, 3], "target_family": None, "sim_case": {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU}}
    assert score_sealed_gate_cases([mr_l3], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0
    gqa_case = {"compose_model": "qwen3-32b", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    with pytest.raises(ModelingError, match="whitebox_batch_x_kv_heads_indivisible"):
        score_sealed_gate_cases([{**base_l1, "case_id": "bx-odd", "point_id": "bx-odd", "batch_x_kv_heads": 2559, "sim_case": gqa_case}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    assert score_sealed_gate_cases([{**base_l1, "case_id": "bx-1kv", "point_id": "bx-1kv", "batch_x_kv_heads": 2559}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0
    attn_l1 = {**base_l1, "case_id": "wb-attn", "point_id": "wb-attn", "context_len": 512, "target_family": "attention", "target_op_id": "s0.l0.paged_mla_decode"}
    for bad_bind in ({"measured_binding": {"builder": "fused_infer_attention", "family": "attention", "recipe": "fused_infer_attention", "paged": False, "heads": 16, "kv_heads": 2, "head_dim": 128}}, {"measured_binding": {"builder": "absorbed_mla", "family": "attention", "recipe": "ascend_paged_mla_decode_arch22", "target_op_id": "s0.l0.paged_mla_decode", "query_layout": "TND"}}, {"measured_binding": {"builder": "absorbed_mla", "family": "attention", "recipe": "ascend_paged_mla_decode_arch22", "target_op_id": "s0.l0.paged_mla_decode", "kv_len": 512, "page_count": 4}}, {"actual_shape": {"page_count": 5, "block_table": [1, 4]}, "measured_binding": {"builder": "absorbed_mla", "family": "attention", "recipe": "ascend_paged_mla_decode_arch22", "target_op_id": "s0.l0.paged_mla_decode"}}):
        with pytest.raises(ModelingError, match="whitebox_measured_binding_mismatch"):
            score_sealed_gate_cases([{**attn_l1, **bad_bind}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    l2_b16 = {**base_l1, "case_id": "l2-mg", "point_id": "l2-mg", "level": "L2", "context_len": 4097, "batch_size": 16, "target_family": None, "target_op_ids": ["s0.l0.paged_mla_decode", "s0.l0.v_expand"]}
    for pat, patch in (("matrix_groups", {"measured_binding": {"node_bindings": [{"op_id": "s0.l0.v_expand", "matrix_groups": 1}]}}), ("l2_op_set_mismatch", {"timed_nodes": ["l2-mg:s0.l0.paged_mla_decode"]}), ("order_violation", {"timed_nodes": ["l2-mg:s0.l0.v_expand", "l2-mg:s0.l0.paged_mla_decode"]}), ("l2_timed_nodes_missing", {"actual_shape": {"m": 256, "n": 128, "k": 512, "e": 16, "matrix_groups": 16, "dtype": "bf16"}}), ("gemm_n_conflict|weight_n", {"actual_shape": {"m": 256, "n": 64, "k": 512, "e": 16, "matrix_groups": 16, "dtype": "bf16"}, "measured_binding": {"node_bindings": [{"op_id": "s0.l0.v_expand", "m": 256, "n": 128, "k": 512, "matrix_groups": 16, "dtype": "bf16"}]}}), ("dtype", {"actual_shape": {"m": 256, "n": 128, "k": 512, "e": 16, "matrix_groups": 16, "dtype": "int8"}, "timed_nodes": ["l2-mg:s0.l0.paged_mla_decode", "l2-mg:s0.l0.v_expand"]})):
        with pytest.raises(ModelingError, match=pat):
            score_sealed_gate_cases([{**l2_b16, **patch}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    l3_base = {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU, "model_overrides": {"layers": 1}}
    l3_c400 = next(c for c in build_profiler_subgraph_plan(synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=20, l2_bytes=A2_ORACLE["l2"]).to_dict(), sku=A2_SKU)["cases"] if c["case_id"] == "cross_tp2_pcp2_dcp4_ctx_400000")
    l3_ns = {n["op_id"]: n["shape"] for n in l3_c400["nodes"] if n["family"] in {"gemm", "grouped_gemm", "gate_up_projection", "down_projection"}}
    l3_mr = {"parallel_topology": "tp2_pcp2_dcp4", "device_ranks": [0, 1, 2, 3]}
    for pat, patch in (("l3_op_set_mismatch", {"timed_nodes": ["l3-sub:s0.l0.norm1", "l3-sub:s0.l0.paged_mla_decode"]}), ("node_shapes_missing:s0.l0.q_proj", {**l3_mr, "actual_shape": {"dtype": "fp32", "partial_count": 4}}), ("dtype_missing", {**l3_mr, "actual_shape": {"partial_count": 4, "node_shapes": l3_ns}}), ("partial_count_missing", {**l3_mr, "actual_shape": {"dtype": "fp32", "node_shapes": l3_ns}}), ("dtype", {**l3_mr, "actual_shape": {"dtype": "bf16", "partial_count": 4, "node_shapes": l3_ns}}), ("partial_count", {**l3_mr, "actual_shape": {"dtype": "fp32", "partial_count": 2, "node_shapes": l3_ns}}), ("weight_n:s0.l0.q_proj", {**l3_mr, "actual_shape": {"dtype": "fp32", "partial_count": 4, "node_shapes": {**l3_ns, "s0.l0.q_proj": {"m": 1, "n": 64, "k": 2048}}}})):
        with pytest.raises(ModelingError, match=pat):
            score_sealed_gate_cases([{**base_l1, "case_id": "l3-sub", "point_id": "l3-sub", "level": "L3", "target_family": None, "sim_case": l3_base, **patch}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    with pytest.raises(ModelingError, match="engine_busy_union_exceeds_elapsed_span"):
        score_sealed_gate_cases([{**base_l1, "case_id": "b-hi", "point_id": "b-hi"}], sku=A2_SKU, runner=_independent_engine_runner({"b-hi": 40_000}, clock_op="busy_union", busy_override={"b-hi": 80_000}))
    idx_row = {**base_l1, "case_id": "idx-2048", "point_id": "idx-2048", "context_len": 2049, "topk": 2048, "target_family": "indexer", "target_op_id": "s0.l2.index_quant_lightning", "measured_binding": {"target_op_id": "s0.l2.index_quant_lightning", "family": "indexer", "heads": 32, "head_dim": 128, "kv_heads": 1, "sparse_count": 2048, "query_dtype": "bf16", "cache_dtype": "bf16", "kv_len": 2050}}
    with pytest.raises(ModelingError, match="topk|query_dtype"):
        score_sealed_gate_cases([{**idx_row, "sim_case": {"compose_model": "deepseek-v4.1-flash", "compose_hardware": A2_SKU}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    assert score_sealed_gate_cases([{**idx_row, "sim_case": {"compose_model": "deepseek-v4.1-flash", "compose_hardware": A2_SKU, "model_overrides": {"index_topk": 2048, "index_dtype": "bf16"}}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0
    assert score_sealed_gate_cases([{**base_l1, "case_id": "tiny-4us", "point_id": "tiny-4us", "measured_ns": 4000, "busy_union_ns": 4000, "elapsed_span_ns": 4100}], sku=A2_SKU, runner=_independent_engine_runner({"tiny-4us": 5500}, clock_op="busy_union"))["ok"] is True
    assert score_sealed_gate_cases([{**base_l1, "case_id": "tiny-4us-fail", "point_id": "tiny-4us-fail", "measured_ns": 4000, "busy_union_ns": 4000, "elapsed_span_ns": 4100}], sku=A2_SKU, runner=_independent_engine_runner({"tiny-4us-fail": 6500}, clock_op="busy_union"))["ok"] is False
    l2_dep = {**base_l1, "case_id": "l2-dep", "point_id": "l2-dep", "level": "L2", "capture_uri": f"capture://audit-node.internal.example/lease-audituser-0abc/{base_l1['raw_sha256']}.json", "target_family": None, "target_op_ids": ["s0.l0.norm1", "s0.l0.q_proj"]}
    l2_ind = {**base_l1, "case_id": "l2-ind", "point_id": "l2-ind", "level": "L2", "target_family": None, "target_op_ids": ["s0.l0.q_proj", "s0.l0.kv_a"]}
    wb_r = WhiteboxSpineRunner(clock_operator="busy_union")
    r_dep, r_ind = score_sealed_gate_cases([l2_dep], sku=A2_SKU, runner=wb_r)["residuals"][0], score_sealed_gate_cases([l2_ind], sku=A2_SKU, runner=wb_r)["residuals"][0]
    assert r_dep["program_dag_sha256"] and r_dep["program_dag_sha256"] != r_ind["program_dag_sha256"] and "audit-node" not in r_dep["capture_uri"] and "audituser" not in r_dep["capture_uri"]
    dep_gc = GateCase("l2-dep", "L2", "heldout", CaptureBoundary(base_l1["raw_sha256"], "acq-l2", "r1", (), ("s1", "e1"), (0,), "device_ns", "s1", "e1", "fx_region", "ns", r_dep["capture_uri"]), "in", "st", "wb", CohortKey("L2", "c", A2_SKU, "decode", "eager", "cliff", "tp1"), 2500, payload=(("clock_operator", "busy_union"), ("context_len", 64), ("phase", "decode"), ("sim_case", b4_case), ("target_op_ids", ["s0.l0.norm1", "s0.l0.q_proj"])))
    dep_diag = dict(wb_r.run_l2(dep_gc).diagnostics)
    lp, ld = dep_diag["lean_program"], dep_diag["lean_dag"]
    assert ld["edges"] and "wire_ns" in ld["edges"][0]
    d_cert, _ = certify_lower_bound(ResourceChoiceProgram(lp["num_resources"], tuple(lp["global_work"]), ()), EventDagSpec(tuple(ld["node_quantum_ns"]), tuple(DagEdge(e["src"], e["dst"], e["wire_ns"]) for e in ld["edges"])))
    lean_ok, _, lean_v = check_lean_non_inversion([{"case_id": "l2-dep", "l_star_lean_ns": d_cert.l_star_lean_ns, "t_sim_ns": dep_diag["busy_union_ns"], "t_silicon_p99_ns": max(dep_diag["busy_union_ns"], 2600), "program": lp, "dag": ld, "certificate": {"farkas": {"scale_d": d_cert.farkas.scale_d, "lambda_num": list(d_cert.farkas.lambda_num), "mu_num": list(d_cert.farkas.mu_num), "lower_bound_ns": d_cert.farkas.lower_bound_ns}, "path_edges": list(d_cert.path_edges), "sink_node_id": d_cert.sink_node_id, "l_star_lean_ns": d_cert.l_star_lean_ns}}], expected_case_timings={"l2-dep": (dep_diag["busy_union_ns"], 2500, max(dep_diag["busy_union_ns"], 2600), dep_diag["program_dag_sha256"])})
    assert lean_ok is True and not lean_v
    raw_full = tmp_path / "full.json"; raw_full.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal_full = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_full)["primitives_payload"]
    par_r = WhiteboxSpineRunner(clock_operator="busy_union", primitive_table=parse_primitive_table(cal_full), max_workers=2)
    par_batch = score_sealed_gate_cases([{**base_l1, "case_id": "p0", "point_id": "p0", "raw_sha256": hashlib.sha256(b"p0").hexdigest()}, {**base_l1, "case_id": "p1", "point_id": "p1", "context_len": 128, "raw_sha256": hashlib.sha256(b"p1").hexdigest()}], sku=A2_SKU, runner=par_r)
    assert len(par_batch["residuals"]) == 2 and all(r["predicted_ns"] > 0 for r in par_batch["residuals"])
    gc64 = GateCase("stale-x", "L1", "heldout", CaptureBoundary(hashlib.sha256(b"s64").hexdigest(), "acq-s", "r1", (), ("s1", "e1"), (0,), "device_ns", "s1", "e1", "device_kernel", "ns", "capture://s/64.json"), "i", "s", "w", CohortKey("L1", "c", A2_SKU, "decode", "eager", "cliff", "tp1"), 2500, payload=(("clock_operator", "busy_union"), ("context_len", 64), ("phase", "decode"), ("target_family", "attention"), ("sim_case", b4_case)))
    gc1024 = GateCase("stale-x", "L1", "heldout", CaptureBoundary(hashlib.sha256(b"s1024").hexdigest(), "acq-s", "r2", (), ("s1", "e1"), (0,), "device_ns", "s1", "e1", "device_kernel", "ns", "capture://s/1024.json"), "i", "s", "w", CohortKey("L1", "c", A2_SKU, "decode", "eager", "cliff", "tp1"), 2500, payload=(("clock_operator", "busy_union"), ("context_len", 1024), ("phase", "decode"), ("target_family", "attention"), ("sim_case", b4_case)))
    gc_cf = GateCase("cf-drop", "L1", "train", gc64.boundary, "i", "s", "w", gc64.cohort, 0, is_counterfactual=True, has_matched_capture=False)
    gc_l2_nb = GateCase("l2-nb", "L2", "heldout", dep_gc.boundary, "i", "s", "w", dep_gc.cohort, 2500, block_capture_present=False)
    assert eligible_replay_cases([gc64, gc_cf, gc_l2_nb]) == [gc64]
    par_r.prefetch_cases([gc64, dep_gc])
    assert par_r.run_l1(gc1024).predicted_ns > par_r.run_l1(gc64).predicted_ns
    with pytest.raises(ModelingError, match="whitebox_duplicate_prefetch_case"):
        par_r.prefetch_cases([gc64, gc64])
    with pytest.raises(ModelingError, match="atol_ns_exceeds_strict_release_cap"):
        score_sealed_gate_cases([base_l1], sku=A2_SKU, atol_ns=2100, runner=_independent_engine_runner({"wb-l1": 2500}, clock_op="busy_union"))
    full_plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    cap_dir, cap_dir_a = tmp_path / "captures", tmp_path / "captures_a"
    l1_atol_ov = {str(pt["point_id"]): (5000, 6900) for pt in full_plan["heldout_points"] if assign_plan_point_level(full_plan, pt) == "L1"}
    m_rows, m_preds, m_leans, m_abis, m_shas = _plan_fixture_bundle(full_plan, capture_dir=cap_dir, overrides=l1_atol_ov)
    ml_cases = [{"case_id": r["case_id"], "point_id": r["point_id"], "level": r["level"], "split": r["split"], "boundary": {"raw_sha256": r["raw_sha256"], "acquisition_uid": r["acquisition_root"], "run_uid": r["run_uid"], "parent_digests": [], "event_ids": r["event_ids"], "device_ranks": r["device_ranks"], "clock_domain": "device_ns", "start_event": "start", "end_event": "end", "boundary_kind": {"L1": "device_kernel", "L2": "fx_region", "L3": "model_step"}[r["level"]], "unit": "ns", "capture_uri": r["capture_uri"]}, "input_digest": r["raw_sha256"], "stack_digest": hashlib.sha256(b"st").hexdigest(), "implementation": "whitebox", "cohort": {"level": r["level"], "cohort_name": "wb", "sku": A2_SKU, "phase": "decode", "mode": "eager", "shape_regime": "cliff", "parallel_topology": r["parallel_topology"]}, "measured_ns": r["measured_ns"], "measured_p99_ns": r["measured_p99_ns"], "clock_operator": r["clock_operator"], "payload": {k: r[k] for k in ("context_len", "tokens", "batch_size", "query_tokens", "parallel_topology", "topk", "batch_x_kv_heads") if r.get(k) is not None}} for r in m_rows]
    full_f = tmp_path / "ml_full.json"
    full_f.write_text(json.dumps({"manifest_id": "m-full", "schema_version": "mindie-multilevel-manifest/1", "cases": ml_cases, "abi_rejection_results": m_abis}), encoding="utf-8")
    with pytest.raises(ModelingError, match="capture_file_missing"):
        verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=full_f, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=_synthetic_repo(tmp_path, family="ml_nocap"))
    syn_ok = _synthetic_repo(tmp_path, family="ml_ok")
    with pytest.raises(ModelingError, match="capture_level_mismatch"):
        validate_512_400k(sku=A2_SKU, primitives_json=cal_full, profiler_runner=lambda _p: [{**r, "level": "L1", "acquisition_root": "acq-held-l1"} if r["level"] == "L3" else ({**r, "level": "L3", "acquisition_root": "acq-held-l3"} if r["point_id"] == "offgrid_ctx_768" else r) for r in m_rows], abi_rejection_results=m_abis, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas))
    with pytest.raises(ModelingError, match="observation_plan_level_mismatch"):
        score_sealed_gate_cases([{**r, "level": "L1", "acquisition_root": "acq-held-l1"} if r["level"] == "L3" else ({**r, "level": "L3", "acquisition_root": "acq-held-l3"} if r["point_id"] == "offgrid_ctx_768" else r) for r in m_rows], sku=A2_SKU, expected_points=full_plan["heldout_points"], expected_train_points=full_plan["train_points"], expected_levels={str(pt["point_id"]): assign_plan_point_level(full_plan, pt) for s in ("train_points", "heldout_points") for pt in full_plan[s]}, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas))
    val_atol = validate_512_400k(sku=A2_SKU, primitives_json=cal_full, profiler_runner=lambda _p: m_rows, abi_rejection_results=m_abis, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas))
    rep_ok = verify_sku_gates(sku=A2_SKU, validation_report_json=val_atol, capture_root=cap_dir, primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=syn_ok)
    full_ok = verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=full_f, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=syn_ok)
    assert val_atol["ok"] is True and rep_ok["ok"] is True and full_ok["ok"] is True and full_ok["status"] == "offline_reverify"
    from mindie_sim.onboarding._lean_and_receipts import sign_stage_report
    dbg_val = {**val_atol, "residuals": [{**r, "status": "debug_only"} if r["level"] == "L3" else r for r in val_atol["residuals"]]}
    dbg_val["report_hmac"] = sign_stage_report("sku_validation", dbg_val)
    assert verify_sku_gates(sku=A2_SKU, validation_report_json=dbg_val, capture_root=cap_dir, primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=syn_ok)["ok"] is False
    bad_atol_val = {**val_atol, "atol_ns": 2100}
    bad_atol_val["report_hmac"] = sign_stage_report("sku_validation", bad_atol_val)
    with pytest.raises(ModelingError, match="atol_ns_exceeds_strict_release_cap"):
        verify_sku_gates(sku=A2_SKU, validation_report_json=bad_atol_val, capture_root=cap_dir, primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=syn_ok)
    assert verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=full_f, capture_root=cap_dir, runner=_independent_engine_runner({**m_preds, next(iter(l1_atol_ov)): 7500}, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=syn_ok)["ok"] is False
    raw_a = tmp_path / "raw_a.json"; raw_a.write_text(json.dumps(_raw_l0_samples(A2_SKU, scale=1.01)), encoding="utf-8")
    cal_a = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_a)["primitives_payload"]
    a_rows, a_preds, _, a_abis, a_shas = _plan_fixture_bundle(full_plan, scale=1.01, capture_dir=cap_dir_a)
    val_a = validate_512_400k(sku=A2_SKU, primitives_json=cal_a, profiler_runner=lambda _p: a_rows, abi_rejection_results=a_abis, capture_root=cap_dir_a, runner=_independent_engine_runner(a_preds, prog_sha_by_id=a_shas))
    cross_spoof = verify_sku_gates(sku=A2_SKU, validation_report_json=val_a, multilevel_manifest_json=full_f, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, acquisition_mode="live_silicon", repo_root=_synthetic_repo(tmp_path, family="ml_spoof"))
    assert cross_spoof["ok"] is False and cross_spoof["status"] != "silicon_certified"
