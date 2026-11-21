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
import hashlib, json, subprocess, sys
from pathlib import Path
from typing import Any
import pytest

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
from mindie_sim.onboarding._profiler_dispatch import _split_disjoint_acquisitions, _validate_dispatch_exit, build_profiler_subgraph_plan
from mindie_sim.onboarding.contracts import sanitize_public_warning
from mindie_sim.onboarding.tiling_axes import ARCH_PRESET_22, ARCH_PRESET_35
from mindie_sim.optimization.dual_bound import DagEdge, EventDagSpec, ResourceChoiceProgram, certify_lower_bound
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.operators.contracts import OperatorFamilySpec, ResolvedOperator
from mindie_sim.prediction.operators.registry import FamilyRegistry
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
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
    (syn / "mindie_sim/__init__.py").write_text('"""Stub."""\n', encoding="utf-8")
    (syn / "scripts/check_arch_governance.py").write_text("print('arch governance violations: 0')\n", encoding="utf-8")
    (syn / "scripts/check_test_orthogonality.py").write_text("print('summary: 0 findings (none); mode=--strict; modules=1/96')\n", encoding="utf-8")
    (syn / "scripts/check_template_provenance.py").write_text("print('template provenance violations: 0')\n", encoding="utf-8")
    (syn / f"tests/operators/test_{family}.py").write_text('"""Test."""\nimport pytest\npytestmark = pytest.mark.tier0\n', encoding="utf-8")
    (syn / "tests/TAXONOMY.toml").write_text(f'schema_version = 1\ndomains = ["operators"]\nmax_lines = 600\nmax_modules = 8\n[[module]]\npath = "operators/test_{family}.py"\ntier = "tier0"\nowns = ["mindie_sim/prediction/operators/{family}", "mindie_sim/templates/{family}_spec.py"]\n', encoding="utf-8")
    return syn


def _synthetic_profiler_root(tmp_path: Path, supported: tuple[str, ...] = ("demo_op",), *, abi: str = "aclnnDemoOpV1", tag: str = "main") -> Path:
    prof, pkg = tmp_path / f"syn_profiler_{tag}", tmp_path / f"syn_profiler_{tag}" / "mindie_profiler/bench"
    pkg.mkdir(parents=True, exist_ok=True)
    (prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
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


def _plan_fixture_bundle(plan: dict[str, Any], *, levels: tuple[str, ...] = ("L1", "L2", "L3"), clock_op: str = "elapsed_span", scale: float = 1.02, impl: str = "whitebox_spine", capture_dir: Path | None = None, omit_capture_p99: bool = False) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    rows, preds, lean_rows, prog_shas = [], {}, [], {}
    sku, family, target_id = str(plan["sku"]), str(plan.get("family") or ""), str(plan.get("target_id") or "")
    for split, pts in (("train", plan["train_points"]), ("heldout", plan["heldout_points"])):
        for idx, pt in enumerate(pts):
            pid, lvl = str(pt["point_id"]), ("L1" if split == "train" else levels[idx % len(levels)])
            meas = (10_000 + idx * 100) if split == "train" else (20_000 + idx * 250)
            pred, p99 = int(round(meas * scale)), int(meas * 1.05)
            ranks, ev_ids = list(range(validate_parallel_topology(str(pt.get("parallel_topology") or "tp1")))), [f"{split[0]}-s-{idx}", f"{split[0]}-e-{idx}"]
            cap_obj: dict[str, Any] = {"case_id": pid, "sku": sku, "split": split, "level": lvl, "idx": idx, "start_event": "start", "end_event": "end", "clock_domain": "device_ns", "device_ranks": ranks, "clock_operator": clock_op, "event_ids": ev_ids, "measured_ns": meas, "busy_union_ns": meas if clock_op == "busy_union" else None, "elapsed_span_ns": meas if clock_op == "elapsed_span" else None}
            if not omit_capture_p99:
                cap_obj["measured_p99_ns"] = p99
            cap_bytes = json.dumps(cap_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            sha = hashlib.sha256(cap_bytes).hexdigest()
            if capture_dir is not None:
                capture_dir.mkdir(parents=True, exist_ok=True)
                (capture_dir / f"{sha}.json").write_bytes(cap_bytes)
            preds[pid] = pred
            acq = "acq-train-src" if split == "train" else f"acq-held-{lvl.lower()}"
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
    out: dict[str, Any] = {
        "device_name": sku,
        "P01_datacopy": {"hbm_read_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.72 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "hbm_write_bytes_per_s": [A2_ORACLE["hbm_bw"] * scale * (0.70 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "mte2_aiv_hit_bytes_per_s_per_core": [A2_ORACLE["mte2_aiv"] * scale * (0.75 + 0.001 * i) for i in range(SAMPLE_REPEATS)]},
        "P20_host_dma_sync": {"host_h2d_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)], "host_d2h_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)]},
    }
    return {**out, "cann_version": cann_version} if cann_version else out


def _a2_probe_payload(sku: str = A2_SKU, *, endpoint: str = "prod-cluster-01.corp") -> dict[str, Any]:
    return {
        "aclrt_soc_name": sku, "endpoint_alias": endpoint, "dies_per_package": A2_ORACLE["dies"],
        "platform_config_ini": (
            f"[SoCInfo]\nsoc_version={sku}\ncompute_arch={A2_ORACLE['arch_id']}\n"
            f"ai_core_cnt={A2_ORACLE['aic']}\nvector_core_cnt={A2_ORACLE['aiv']}\n"
            f"l2_size={A2_ORACLE['l2']}\nub_size={A2_ORACLE['ub']}\n"
        ),
        "npu_smi_common": f"Chip ID : 0\nAicore Count : {A2_ORACLE['aic']}\nAicore Freq(MHZ) : 1650\n",
    }


def test_probe_soc_conflicts_capture_gate_and_unspoofable_status(tmp_path):
    # Items #1, #13, #14, #17: Opaque ep/camp IDs, cross-source conflict detection, mandatory captures, unspoofable status!
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
        "aic_count": 28, "aiv_count": 56, "hbm_bytes": 64 * 1024**3,
        "l2_bytes": 200 * 1024**2, "ub_bytes": 196_608, "dies_per_package": 1, "ranks_per_node": 8,
    }), encoding="utf-8")
    new_plan = synthesize_sku_512_400k_plan(sku=new_sku, core_count=28, l2_bytes=200 * 1024**2).to_dict()
    cap_dir = tmp_path / "new_caps"
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(new_plan, capture_dir=cap_dir)
    raw_l0 = tmp_path / "new_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(new_sku)), encoding="utf-8")
    # Missing capture files, omitted capture_root on signed report, require_capture_files=False, & live_silicon spoof MUST fail!
    with pytest.raises(ModelingError, match="capture_file_missing"):
        run_sku_onboarding(probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas), lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="no_cap"))
    with pytest.raises(ModelingError, match="release_certification_requires_capture_files"):
        run_sku_onboarding(probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, require_capture_files=False, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas), lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="no_cap2"))
    run_new = run_sku_onboarding(probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU, profiler_runner=lambda _p: obs, abi_rejection_results=abis, capture_root=cap_dir, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas), lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="new_sku"), out_dir=tmp_path / "new_sku_out")
    assert run_new["ok"] is True and run_new["status"] == "offline_reverify"
    # H13 / R5-#8: 4,100 unrelated files sorting before nested trial-001.json still resolve Stage 3 -> Stage 4 via private index (zero leak)!
    off768_row = next(r for r in obs if r["point_id"] == "offgrid_ctx_768")
    orig_cap_path = cap_dir / f"{off768_row['raw_sha256']}.json"
    saved_cap_bytes = orig_cap_path.read_bytes()
    decoy_dir = cap_dir / "0000_decoys"
    decoy_dir.mkdir(parents=True, exist_ok=True)
    for j in range(4100):
        (decoy_dir / f"d_{j:04d}.json").write_bytes(b"{}")
    nested_f = cap_dir / "audit-node.internal.example/lease-audituser/trial-001.json"
    nested_f.parent.mkdir(parents=True, exist_ok=True)
    nested_f.write_bytes(saved_cap_bytes)
    orig_cap_path.unlink()
    off768_row["capture_uri"] = "capture://audit-node.internal.example/lease-audituser/trial-001.json"
    val_nested = validate_512_400k(sku=new_sku, primitives_json=run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"], profiler_runner=lambda _p: obs, abi_rejection_results=abis, profile_override=run_new["stages"]["probe-soc"]["scaffold_profile"][new_sku], capture_root=cap_dir, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas))
    assert val_nested["ok"] is True and "audit-node" not in json.dumps(val_nested) and "trial-001" not in json.dumps(val_nested)
    nested_gate = verify_sku_gates(sku=new_sku, validation_report_json=val_nested, primitives_json=run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"], lean_bounds_json=leans, profile_override=run_new["stages"]["probe-soc"]["scaffold_profile"][new_sku], capture_root=cap_dir, repo_root=_synthetic_repo(tmp_path, family="nested_s"))
    assert nested_gate["ok"] is True and nested_gate["status"] == "offline_reverify"
    nested_f.unlink()
    with pytest.raises(ModelingError, match="capture_file_missing"):
        verify_sku_gates(sku=new_sku, validation_report_json=val_nested, primitives_json=run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"], lean_bounds_json=leans, profile_override=run_new["stages"]["probe-soc"]["scaffold_profile"][new_sku], capture_root=cap_dir, repo_root=_synthetic_repo(tmp_path, family="nested_del"))
    orig_cap_path.write_bytes(saved_cap_bytes)
    empty_caps, empty_repo = tmp_path / "empty_caps", _synthetic_repo(tmp_path, family="empty_c")
    empty_caps.mkdir()
    for cap_arg in (empty_caps, None):
        with pytest.raises(ModelingError, match="capture_file_missing"):
            verify_sku_gates(sku=new_sku, validation_report_json=run_new["stages"]["validate-512-400k"], primitives_json=run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"], lean_bounds_json=leans, profile_override=run_new["stages"]["probe-soc"]["scaffold_profile"][new_sku], capture_root=cap_arg, repo_root=empty_repo)
    spoof = verify_sku_gates(sku=new_sku, validation_report_json=run_new["stages"]["validate-512-400k"], primitives_json=run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"], lean_bounds_json=leans, profile_override=run_new["stages"]["probe-soc"]["scaffold_profile"][new_sku], capture_root=cap_dir, acquisition_mode="live_silicon", repo_root=_synthetic_repo(tmp_path, family="spoof_s"))
    assert spoof["ok"] is False and spoof["status"] != "silicon_certified"
    warn_gate = verify_sku_gates(sku=new_sku, validation_report_json=run_new["stages"]["validate-512-400k"], primitives_json=run_new["stages"]["calibrate-l0-and-fabric"]["primitives_payload"], lean_bounds_json=leans, profile_override=run_new["stages"]["probe-soc"]["scaffold_profile"][new_sku], capture_root=cap_dir, stage2_lease={"lease_mode": "coordinator_host_queue", "lease_digest": "lease-0123456789abcdef", "fence_token": 7, "device_count": 1, "receipt_unconfirmed": True, "receipt_warning": "release_delayed:audit-node.internal.example"}, repo_root=_synthetic_repo(tmp_path, family="warn_s"))
    assert warn_gate["ok"] is True and warn_gate["receipt"]["receipt_unconfirmed"] is True and "audit-node" not in json.dumps(warn_gate["receipt"]) and verify_receipt_sha256(warn_gate["receipt"]) is True


def test_sealer_cann_manifest_conflicts_and_campaign_privacy(tmp_path):
    # Items #12, #15, #16, #17: Wheel sealer, CANN manifest check, conflict vs missing, opaque campaign_id!
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
    assert any(c["reason"].startswith("p99_exceeds_nominal") for c in cal_over["metric_conflicts"])
    assert any(r["reason"] == "missing_samples" for r in cal_over["coverage_matrix"])


def test_lock3_dav2201_vs_dav3510_lightning_abi_and_strict_rejections():
    # Item #6 (Astra #8): Upstream DAV_3510 accepts ctx=2049, topk=2049; DAV_2201 rejects topk=2049!
    axes35 = family_default_tiling_axes("sparse_indexer", tile_dim=128, arch_preset=ARCH_PRESET_35)
    axes22 = family_default_tiling_axes("sparse_indexer", tile_dim=128, arch_preset=ARCH_PRESET_22)
    assert validate_abi({"context_len": 1, "topk": 64}, axes35, arch_preset=ARCH_PRESET_35)[0] is True
    assert validate_abi({"context_len": 2049, "topk": 2049}, axes35, arch_preset=ARCH_PRESET_35)[0] is True
    assert validate_abi({"context_len": 2049, "topk": 2049}, axes22, arch_preset=ARCH_PRESET_22)[1].startswith("abi_divisibility_violation:topk=2049")
    assert validate_abi({"topk": 8193}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_max_violation:topk=8193")
    assert validate_abi({"topk": "2049"}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_type_violation")

    sp35 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 32, 192 * 1024**2), sku=A5_SKU)
    sp22 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, 100 * 1024**2), sku=A2_SKU)
    sp22_2050 = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, 100 * 1024**2, topk_threshold=2050), sku=A2_SKU)
    assert not any(pt.topk == 2049 for pt in sp35.abi_rejection_points) and any(pt.topk == 2049 for pt in sp35.heldout_points)
    assert any(pt.topk == 8193 for pt in sp35.abi_rejection_points) and any(pt.topk == 2049 for pt in sp22.abi_rejection_points)
    assert not any(pt.topk == 2049 for pt in sp22_2050.heldout_points) and any(pt.topk == 2049 for pt in sp22_2050.abi_rejection_points)
    assert set(ANCHOR_CONTEXT_GRID) == {pt.value for pt in sp22.train_points if pt.axis == "context_len"}
    assert set(OFF_GRID_CLIFF_HELDOUT_GRID).issubset({pt.value for pt in sp22.heldout_points if pt.axis == "context_len"})

    with pytest.raises(ModelingError, match="dcp_pcp_domain_invalid"):
        validate_parallel_topology("tp2_pcp4_dcp2")
    assert required_ranks_for_plan(sp22.to_dict()) == 4

    # Strict ABI rejection matching: requires executed_shape, exact primary axis + value + rule!
    rejs = sp22.to_dict()["abi_rejection_points"]
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"]} for pt in rejs])["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": "abi_divisibility_violation:batch_size=4:note_topk=2049", "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": pt["value"], "topk": pt["topk"]}} for pt in rejs])["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": "abi_divisibility_violation:topk=2051:above=2048:divisor=1024", "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": pt["value"], "topk": pt["topk"]}} for pt in rejs])["ok"] is False


def test_capture_duration_override_clock_plan_ranks_and_capture_p99(tmp_path):
    # Items #2, #5, #7: Capture duration/p99 cannot be overridden by row; plan clock & multi-rank participation enforced!
    plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    cap_dir = tmp_path / "caps"
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(plan, capture_dir=cap_dir)

    # Row setting measured_ns=110 against hash-verified 7500 ns capture MUST fail capture_measured_duration_mismatch!
    cap75 = {"case_id": "clk-75", "sku": A2_SKU, "split": "heldout", "start_event": "start", "end_event": "end", "clock_domain": "device_ns", "device_ranks": [0], "clock_operator": "busy_union", "event_ids": ["s0", "e0"], "busy_union_ns": 7500, "samples_ns": [7500] * 12}
    cap75_b = json.dumps(cap75, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cap75_sha = hashlib.sha256(cap75_b).hexdigest()
    (cap_dir / f"{cap75_sha}.json").write_bytes(cap75_b)
    tampered = {**obs[-1], "case_id": "clk-75", "point_id": "clk-75", "raw_sha256": cap75_sha, "capture_uri": f"capture://heldout/{cap75_sha}.json", "event_ids": ["s0", "e0"], "device_ranks": [0], "clock_operator": "busy_union", "measured_ns": 110, "busy_union_ns": 110}
    with pytest.raises(ModelingError, match="capture_measured_duration_mismatch"):
        score_sealed_gate_cases([tampered], sku=A2_SKU, capture_root=cap_dir, require_capture_files=True, runner=_independent_engine_runner({"clk-75": 110}, clock_op="busy_union"))

    # Planned elapsed_span point (offgrid_ctx_768) rejects busy_union row!
    off768 = next(r for r in obs if r["point_id"] == "offgrid_ctx_768")
    with pytest.raises(ModelingError, match="clock_operator_plan_mismatch"):
        score_sealed_gate_cases([{**off768, "clock_operator": "busy_union", "measured_ns": off768["busy_union_ns"]}], sku=A2_SKU, expected_points=plan["heldout_points"], runner=_independent_engine_runner(preds, clock_op="busy_union"))

    # Multi-rank 4-rank cell rejects device_ranks=[0]!
    mr_row = next(r for r in obs if r["parallel_topology"] == "tp2_pcp2_dcp2")
    with pytest.raises(ModelingError, match="observation_participating_ranks_mismatch"):
        score_sealed_gate_cases([{**mr_row, "device_ranks": [0]}], sku=A2_SKU, expected_points=plan["heldout_points"], runner=_independent_engine_runner(preds))

    # Capture lacking both p99 and repeat samples MUST fail Lean lean_captured_p99_missing!
    nop99_dir = tmp_path / "caps_nop99"
    obs_np, preds_np, leans_np, abis_np, pshas_np = _plan_fixture_bundle(plan, capture_dir=nop99_dir, omit_capture_p99=True)
    raw_l0 = tmp_path / "l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_l0)
    val_np = validate_512_400k(sku=A2_SKU, primitives_json=cal["primitives_payload"], profiler_runner=lambda _p: obs_np, abi_rejection_results=abis_np, capture_root=nop99_dir, runner=_independent_engine_runner(preds_np, prog_sha_by_id=pshas_np))
    gate_np = verify_sku_gates(sku=A2_SKU, validation_report_json=val_np, primitives_json=cal["primitives_payload"], lean_bounds_json=leans_np, capture_root=nop99_dir, repo_root=_synthetic_repo(tmp_path, family="nop99"))
    assert gate_np["ok"] is False and any(g["name"] == "lean_optimality_non_inversion" and "lean_captured_p99_missing" in g["detail"] for g in gate_np["receipt"]["gates"])


def test_operator_stub_template_ownership_identity_and_certify(tmp_path):
    # Items #3, #5, #7: Stub fails; borrowed TEMPLATE_ID/profiler fails; real catlass_catalog passes; unowned variant fails certify!
    repo_dir, rev, sha = _git_repo_with_blob(tmp_path, "ops-transformer", "kernels/demo_op.cpp", b"// demo\n")
    syn = _synthetic_repo(tmp_path, family="demo_op")
    prof_ok = _synthetic_profiler_root(tmp_path, supported=("demo_op",), abi="aclnnDemoOpV1", tag="ok")
    scaf = scaffold_operator(family="demo_op", upstream_abi_arch22="aclnnDemoOpV1", upstream_abi_arch35="aclnnDemoOpV2", source_refs=[{"repository": "ops-transformer", "revision": rev, "relative_path": "kernels/demo_op.cpp", "source_sha256": sha}], target_root=syn, overwrite=True)
    target_rel = scaf["package_rel"]

    def _real_resolve(op: OpSpec, rctx) -> ResolvedOperator:
        prog = KernelProgram((Stage("vec_main", work={"vector": 4096.0}),))
        return ResolvedOperator(occurrence=rctx.occurrence, op=op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: prog, pipeline_spec={"stage": "v"}))

    reg_ok = FamilyRegistry([OperatorFamilySpec("demo_op", frozenset({"demo_op"}), frozenset({"demo_op", "demo_op:aclnnDemoOpV1"}), _real_resolve)])
    # Unimplemented scaffold MUST fail validate_operator_contract!
    stub_chk = validate_operator_contract(target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_ok, family_registry=reg_ok)
    assert stub_chk["ok"] is False and any("operator_target_unimplemented_stub" in v for v in stub_chk["violations"])

    # Borrowed TEMPLATE_ID="catlass.basic_pingpong" on rmsnorm_spec.py MUST fail template_id_not_owned_by_target!
    fake_rms = syn / "mindie_sim/templates/rmsnorm_spec.py"
    fake_rms.parent.mkdir(parents=True, exist_ok=True)
    fake_rms.write_text(f'"""Stub.\n\nUpstream-ABI: aclnnRmsNorm\n"""\nTEMPLATE_ID = "catlass.basic_pingpong"\nSOURCES = (SourceRef(repository="ops-transformer", revision="{rev}", relative_path="kernels/demo_op.cpp", source_sha256="{sha}"),)\nSemanticWorkItem = TensorRangeObligation = TilingAxes = object\n', encoding="utf-8")
    borrowed = validate_operator_contract(target_path=fake_rms, repo_root=syn, refs_root=repo_dir.parent, template_id="catlass.basic_pingpong", require_profiler_builder=False, require_taxonomy=False)
    assert borrowed["ok"] is False and any("template_id_not_owned_by_target" in v for v in borrowed["violations"])

    # Real catlass_catalog.py passes real TemplateCatalog.lower (48 stages, verified Lean dual bound)!
    cat_chk = validate_operator_contract(target_path=REPO_ROOT / "mindie_sim/templates/catlass_catalog.py", repo_root=REPO_ROOT, template_id="catlass.basic_pingpong", require_profiler_builder=False, require_taxonomy=False, require_tiling_axes=False)
    assert cat_chk["sim_owner_dispatch"]["lowered"] is True and cat_chk["sim_owner_dispatch"]["lean_status"] == "verified"

    # Zero-event recipe MUST fail dual bound!
    probe_op = OpSpec("op0", "demo_op", "demo_op", "decode", "bf16", (), (1, 64), (1, 64), (1, 64), (1, 64))
    ctx = _build_default_planning_context(probe_op)
    zero_res = ResolvedOperator(occurrence=ctx.occurrence, op=probe_op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: (), pipeline_spec={"stage": "v"}))
    assert _verify_resolved_operator_dual_bound(zero_res, ctx)[2] == "resolved_operator_program_zero_executable_events"

    # Implement demo_op common.py (removing NotImplementedError stubs) -> full 4-stage run passes as offline_reverify!
    common_py = syn / target_rel / "common.py"
    common_py.write_text(common_py.read_text(encoding="utf-8").replace('raise NotImplementedError(', 'return ('), encoding="utf-8")
    plan_doc = generate_operator_cliff_plan(operator_id="demo_op:aclnnDemoOpV1", family="demo_op", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048)["plan"]
    op_caps = tmp_path / "op_caps"
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


def test_host_queue_unified_grant_cleanup_and_profiler_roundtrip(tmp_path):
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        validate_host_queue_grant({"lease_id": "l1", "fence_token": 1, "devices": [], "ttl_seconds": 600}, required_devices=1, min_ttl_seconds=120)
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        validate_host_queue_grant({"lease_id": "l1", "fence_token": 1, "devices": [0, 1, 2, 3, 4], "ttl_seconds": 600}, required_devices=4, min_ttl_seconds=120)
    with pytest.raises(ModelingError, match="host_queue_lease_ttl_below_subprocess_plus_grace"):
        validate_host_queue_grant({"lease_id": "l1", "fence_token": 1, "devices": [0, 1, 2, 3], "ttl_seconds": 1}, required_devices=4, min_ttl_seconds=120)
    rel_log, c_short = tmp_path / "rel_log.txt", tmp_path / "c_short.py"
    c_short.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\nact = args[args.index('--action') + 1]\nif act == 'acquire':\n    print(json.dumps({{'outcome':'success','status':'ready','data':{{'lease_id':'lease-audituser-1','fence_token':5,'devices':[0,1,2]}}}}))\nelse:\n    Path({str(rel_log)!r}).write_text('released', encoding='utf-8')\n    print(json.dumps({{'outcome':'success','status':'ok'}}))\n", encoding="utf-8")
    c_short.chmod(0o755)
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        with acquire_coordinator_host_queue_lease("audit-node.internal.example", npu_count=4, ttl_seconds=180, min_subprocess_timeout_s=120, coordinator_bin=str(c_short)):
            pass
    assert rel_log.read_text(encoding="utf-8") == "released" and "audituser" not in sanitize_public_warning("PermissionError: audituser") and "permissionerror" not in sanitize_public_warning("PermissionError: audituser")
    c_warn = tmp_path / "c_warn.py"
    c_warn.write_text(f"#!{sys.executable}\nimport json, sys\nargs = sys.argv[1:]\nact = args[args.index('--action') + 1]\nif act == 'acquire':\n    print(json.dumps({{'outcome':'success','status':'ready','receipt_unconfirmed':True,'receipt_warning':'PermissionError: audituser','data':{{'lease_id':'lease-audituser-2','fence_token':7,'devices':[0,1,2,3]}}}}))\nelse:\n    print(json.dumps({{'outcome':'success','status':'ok'}}))\n", encoding="utf-8")
    c_warn.chmod(0o755)
    with acquire_coordinator_host_queue_lease("audit-node.internal.example", npu_count=4, ttl_seconds=180, min_subprocess_timeout_s=120, coordinator_bin=str(c_warn)) as linfo:
        assert linfo["devices"] == [0, 1, 2, 3]
    assert linfo["receipt_unconfirmed"] is True and "audituser" not in linfo["receipt_warning"] and "permissionerror" not in linfo["receipt_warning"]

    # H11: Both custom lease_acquirer AND default coordinator_bin block live callbacks (l0_runner, profiler_runner, bench_dispatcher)!
    @contextmanager
    def _bad_acq(devs: list[int], ttl: int, fence: int = 1):
        yield {"lease_mode": "coordinator_host_queue", "lease_id": "l-bad", "fence_token": fence, "devices": devs, "ttl_seconds": ttl}

    ran_cb: list[bool] = []
    with pytest.raises(ModelingError, match="host_queue_lease_"):
        calibrate_l0_and_fabric(sku=A2_SKU, live_bench=True, l0_runner=lambda _s, _t: (ran_cb.append(True) or {}), lease_acquirer=lambda _s, _t: _bad_acq([], 1, fence=0))
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        calibrate_l0_and_fabric(sku=A2_SKU, live_bench=True, endpoint_alias="audit-node.internal.example", l0_runner=lambda _s, _t: (ran_cb.append(True) or {}), coordinator_bin=str(c_short))
    plan_doc = generate_operator_cliff_plan(operator_id="live:v1", family="live", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"])["plan"]
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        validate_512_400k(sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", profiler_runner=lambda _p: (ran_cb.append(True) or []), coordinator_bin=str(c_short))
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", bench_dispatcher=lambda _p, _t: (ran_cb.append(True) or []), coordinator_bin=str(c_short))
    assert not ran_cb
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, lease_acquirer=lambda _s, _t: _bad_acq([0, 1, 2, 3, 4], 900))

    # H6 & R5 #1-#7: Lowered 1-layer L3, BNSD/513/5-page/grouped_matmul L2, world-size-separated L3, receipt actual_shape join, strict ABI & multi-rank critical path!
    b4_plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=20, l2_bytes=A2_ORACLE["l2"]).to_dict()
    b4_sub = build_profiler_subgraph_plan(b4_plan, sku=A2_SKU)
    b4_by_id = {c["case_id"]: c for c in b4_sub["cases"]}
    assert b4_sub["world_size"] == 4 and {c["level"] for c in b4_sub["cases"]} == {"L1", "L2", "L3"}
    assert all(c["nodes"][0]["family"] != "sku_suite" and c["nodes"][0].get("builder") for c in b4_sub["cases"])
    assert b4_by_id["anchor_ctx_512"]["shape"]["seq_len"] == 513 and b4_by_id["anchor_ctx_512"]["shape"]["page_count"] == 5 and b4_by_id["anchor_ctx_512"]["shape"]["query_layout"] == "BNSD"
    assert b4_by_id["cross_b16_ctx_4097"]["level"] == "L2" and b4_by_id["cross_b16_ctx_4097"]["nodes"][1]["builder"] == "grouped_matmul" and b4_by_id["cross_b16_ctx_4097"]["nodes"][1]["shape"]["e"] == 16
    assert b4_by_id["cores_wave_batch_x_kv_heads_128_minus1_2559"]["shape"]["sequences"] == 2559 and len(b4_by_id["offgrid_ctx_200000"]["nodes"]) == 15 and len(b4_by_id["cross_tp2_pcp2_dcp4_ctx_400000"]["nodes"]) == 19
    assert b4_by_id["abi_reject_topk_attr_2049"]["shape"]["top_k"] == 2049 and b4_by_id["abi_reject_topk_attr_2049"]["shape"]["enforce_exact_topk"] is True
    b4_splits = _split_disjoint_acquisitions(b4_plan, domain=None, dry_run=False)
    assert [t for t, _, _ in b4_splits] == ["train", "heldout-l1", "heldout-l2", "heldout-l3-w1", "heldout-l3-w4"] and b4_splits[3][2] == b4_splits[4][2] and len({s[2] for s in b4_splits[:4]}) == 4
    with pytest.raises(ModelingError, match="profiler_actual_geometry_mismatch"):
        normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_b16_ctx_4097", "run_id": "r1.base.eager.t0", "trial": 0, "rank": 0, "split": "heldout", "boundary": "fx_region", "timing_mode": "eager", "duration_ms": 0.25, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}, {"schema": "mindie-bench-subgraph-run/1", "record_type": "receipt", "case_id": "cross_b16_ctx_4097", "run_id": "r1", "rank": 0, "status": "ok", "actual_shape": {"sequences": 1, "q_len": 16, "tokens": 16, "seq_len": 4098, "kv_heads": 1, "heads": 16, "head_dim": 512}}], plan_doc=b4_plan, sku=A2_SKU)
    with pytest.raises(ModelingError, match="multirank_incomplete_rank_samples"):
        normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "cross_tp2_pcp2_dcp4_ctx_400000", "run_id": f"mr{rk}.base.eager.t0", "acquisition_id": "acq-mr", "trial": 0, "repeat_index": rep, "rank": rk, "split": "heldout", "boundary": "model_step", "timing_mode": "eager", "duration_ms": 0.03 + 0.01 * (rk == 3), "sample_id": f"s-{rk}-{rep}", "sku": A2_SKU, "validated": True, "qualified": True} for rk in range(4) for rep in (range(4) if rk < 3 else range(1))], plan_doc=b4_plan, sku=A2_SKU)
    with pytest.raises(ModelingError, match="profiler_abi_rejection_unexpectedly_accepted"):
        normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "abi_reject_topk_attr_2049", "run_id": "r1.base.eager.t0", "trial": 0, "split": "abi_rejection", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.05, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True, "actual_shape": {"top_k": 2048, "seq_len": 2049}}], plan_doc=b4_plan, sku=A2_SKU)
    with pytest.raises(ModelingError, match="clock_operator_plan_mismatch"):
        normalize_profiler_records([{"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_768", "run_id": "r1.base.stream_saturated.t0", "trial": 0, "split": "heldout", "boundary": "device_kernel", "timing_mode": "stream_saturated", "duration_ms": 0.0075, "sample_id": "s0", "sku": A2_SKU, "validated": True, "qualified": True}], plan_doc=plan_doc, sku=A2_SKU)
    cap_out = tmp_path / "prof_caps"
    norm_obs, norm_abi = normalize_profiler_records(
        [
            {"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_768", "run_id": "run001.base.eager.t0", "trial": 0, "split": "heldout", "boundary": "device_kernel", "timing_mode": "eager", "duration_ms": 0.020, "sample_id": "obs-0", "rank": None, "sku": A2_SKU, "validated": True, "qualified": True, "identity": {"level": "operator", "implementation": "aclnnLive"}},
            {"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "offgrid_ctx_768", "run_id": "run001.base.stream_saturated.t0", "trial": 0, "split": "heldout", "boundary": "device_kernel", "timing_mode": "stream_saturated", "duration_ms": 0.0075, "sample_id": "obs-1", "rank": None, "sku": A2_SKU, "validated": True, "qualified": True, "identity": {"level": "operator", "implementation": "aclnnLive"}},
            {"schema": "mindie-bench-observation/1", "record_type": "case_complete", "case_id": "offgrid_ctx_768", "run_id": "run001", "split": "heldout", "status": "ok", "summary": {"op": "live", "actual_shape": {"context_len": 768, "batch_size": 1, "query_tokens": 1}, "participating_ranks": [2]}},
            {"schema": "mindie-bench-receipt/1", "case_id": "abi_1", "status": "unsupported_binding", "reason": "abi_divisibility_violation:topk=2049", "executed_shape": {"context_len": 2049, "topk": 2049, "batch_size": 1, "query_tokens": 1}},
            {"schema": "mindie-bench-receipt/1", "case_id": "abi_1", "status": "unsupported_binding", "reason": "kernel_symbol_missing", "executed_shape": {"context_len": 2049, "topk": 2049, "batch_size": 1, "query_tokens": 1}},
        ],
        plan_doc=plan_doc, sku=A2_SKU, endpoint_alias="audit-node.internal.example", capture_out_dir=cap_out,
    )
    assert len(norm_obs) == 1 and norm_obs[0]["device_ranks"] == [2] and norm_obs[0]["context_len"] == 768 and norm_obs[0]["clock_operator"] == "elapsed_span" and "sim_case" in norm_obs[0] and len(norm_abi) == 2
    assert "audit-node" not in json.dumps(norm_obs) and norm_obs[0]["run_uid"].startswith("run-")
    verify_capture_file_consistency("offgrid_ctx_768", CaptureBoundary(norm_obs[0]["raw_sha256"], norm_obs[0]["acquisition_root"], norm_obs[0]["run_uid"], (), tuple(norm_obs[0]["event_ids"]), tuple(norm_obs[0]["device_ranks"]), "device_monotonic", "k_start", "k_end", "device_kernel", "ns", norm_obs[0]["capture_uri"]), cap_out / f"{norm_obs[0]['raw_sha256']}.json")

    abi_plan = synthesize_operator_cliff_plan(OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 20, A2_ORACLE["l2"]), sku=A2_SKU).to_dict()
    abi_pt = abi_plan["abi_rejection_points"][0]
    abi_pid, abi_err = abi_pt["point_id"], abi_pt["abi_expected_error"]
    abi_by_id = {abi_pid: abi_pt}
    for rc_v, bad_recs in (
        (2, [{"case_id": abi_pid, "status": "unsupported_binding", "reason": abi_err, "requested_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"]}}]),
        (2, [{"case_id": abi_pid, "status": "unsupported_binding", "reason": abi_err, "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"], "sparse_count": abi_pt["topk"] - 1}}]),
        (0, [{"case_id": "offgrid_ctx_768", "status": "failed", "reason": "kernel_symbol_missing"}]),
        (2, [{"case_id": abi_pid, "status": "unsupported_binding", "reason": abi_err, "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"]}}, {"case_id": abi_pid, "status": "unsupported_binding", "reason": "kernel_symbol_missing", "executed_shape": {"batch_size": 1, "query_tokens": 1, "context_len": abi_pt["value"], "topk": abi_pt["topk"]}}]),
    ):
        with pytest.raises(ModelingError, match="profiler_bench_unexpected_partial_failure|profiler_shape_alias_conflict"):
            _validate_dispatch_exit(rc_v, [{"status": "ok" if rc_v == 0 else "partial"}], bad_recs, abi_by_id, "", is_remote=False)
    fake_prof = tmp_path / "fake_prof_sub"
    (fake_prof / "mindie_profiler/bench").mkdir(parents=True)
    (fake_prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (fake_prof / "mindie_profiler/__main__.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\nassert args[:2] == ['bench', 'subgraph']\n"
        "plan = json.loads(Path(args[args.index('--plan') + 1]).read_text(encoding='utf-8'))\n"
        "obs_p, out_p = Path(args[args.index('--observations') + 1]), Path(args[args.index('--out') + 1])\n"
        "rk, ws = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))\n"
        f"has_abi = any(c['case_id'] == {abi_pid!r} for c in plan['cases'])\n"
        "c0 = next(c for c in plan['cases'] if c['split'] != 'abi_rejection')\n"
        "cid, sp, rid = c0['case_id'], c0['split'], f\"{plan['plan_id']}_u{rk}\"\n"
        "obs_p.write_text(json.dumps({'schema':'mindie-bench-observation/1','record_type':'observation','plan_id':plan['plan_id'],'case_id':cid,'run_id':f'{rid}.base.eager.t0','trial':0,'repeat_index':0,'rank':rk,'split':sp,'timing_mode':'eager','duration_ms':0.02 + 0.001*rk,'sample_id':f's{rk}','sku':plan['sku'],'validated':True,'qualified':True}) + '\\n', encoding='utf-8')\n"
        f"recs = [{{'case_id': cid, 'status': 'ok'}}] + ([{{'case_id': {abi_pid!r}, 'status': 'unsupported_binding', 'reason': {abi_err!r}, 'executed_shape': {{'batch_size': 1, 'query_tokens': 1, 'context_len': {abi_pt['value']}, 'topk': {abi_pt['topk']}}}}}] if has_abi else [])\n"
        "Path(f'{out_p}.rank{rk}.json' if ws > 1 else str(out_p)).write_text(json.dumps({'schema':'mindie-bench-subgraph-run/1','status':'partial' if has_abi else 'ok','plan_id':plan['plan_id'],'run_id':rid,'records':recs}), encoding='utf-8')\n"
        "sys.exit(2 if has_abi else 0)\n",
        encoding="utf-8",
    )
    obs_l, abi_l = run_profiler_bench_subprocess(abi_plan, sku=A2_SKU, profiler_root=fake_prof, allowed_phy_ids=[0, 1, 2, 3], fence_token=7)
    ho_mr = next(o for o in obs_l if o["case_id"] == "cross_tp2_pcp2_dcp2_ctx_200000")
    ho_w1 = next(o for o in obs_l if o["split"] == "heldout" and o["device_ranks"] == [0])
    assert ho_mr["device_ranks"] == [0, 1, 2, 3] and ho_mr["acquisition_root"] == ho_w1["acquisition_root"] and len(abi_l) == 1 and abi_l[0]["point_id"] == abi_pid
    with pytest.raises(ModelingError, match="onboarding_endpoint_unknown|onboarding_json_read_failed"):
        validate_512_400k(sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", profiler_root=fake_prof, lease_acquirer=lambda _s, _t: _bad_acq([0, 1, 2, 3], 600))
    with pytest.raises(ModelingError, match="onboarding_endpoint_unknown|onboarding_json_read_failed"):
        bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, endpoint_alias="audit-node.internal.example", profiler_root=fake_prof, lease_acquirer=lambda _s, _t: _bad_acq([0, 1, 2, 3], 900))


def test_whitebox_sku_check_2d_kv_heads_dcp_compose_and_real_dag(tmp_path):
    # Items #4, #5, #7: B4 sim_case rejected on A5 plan; 2D bxkv checks local_kv_heads; dcp compose works; dependent vs independent L2 DAGs differ!
    b4_case = {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    base_l1 = {"case_id": "wb-l1", "point_id": "wb-l1", "sku": A2_SKU, "split": "heldout", "level": "L1", "acquisition_root": "acq-wb-l1", "run_uid": "r1", "raw_sha256": hashlib.sha256(b"wb-l1").hexdigest(), "event_ids": ["s1", "e1"], "device_ranks": [0], "measured_ns": 2500, "measured_p99_ns": 2600, "clock_operator": "busy_union", "busy_union_ns": 2500, "elapsed_span_ns": 2600, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "target_family": "rmsnorm", "sim_case": b4_case}
    with pytest.raises(ModelingError, match="whitebox_materialized_sku_mismatch"):
        score_sealed_gate_cases([{**base_l1, "sku": A5_SKU}], sku=A5_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))

    # Multi-rank composed replay with tp2_pcp2_dcp2 passes `dcp=2` cleanly (no `unexpected keyword argument 'cp'`)!
    mr_l3 = {**base_l1, "case_id": "wb-mr", "point_id": "wb-mr", "level": "L3", "parallel_topology": "tp2_pcp2_dcp2", "device_ranks": [0, 1, 2, 3], "target_family": None, "sim_case": {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU}}
    mr_res = score_sealed_gate_cases([mr_l3], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    assert mr_res["residuals"][0]["predicted_ns"] > 0

    # 2D batch_x_kv_heads: indivisible 2559 on a GQA model (local_kv_heads=8) FAILS; 2559 on 1-kv-head model lowers cleanly!
    gqa_case = {"compose_model": "qwen3-32b", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    with pytest.raises(ModelingError, match="whitebox_batch_x_kv_heads_indivisible"):
        score_sealed_gate_cases([{**base_l1, "case_id": "bx-odd", "point_id": "bx-odd", "batch_x_kv_heads": 2559, "sim_case": gqa_case}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    assert score_sealed_gate_cases([{**base_l1, "case_id": "bx-1kv", "point_id": "bx-1kv", "batch_x_kv_heads": 2559}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0

    # H8 & R5 #2/#3/#4 & #259: Reject TND vs BNSD, 512 vs 513 KV, 4 vs 5 pages, matrix_groups=1 vs 16, 5-op L3, INT8/512 vs BF16/2048 indexer; honor max(10%, 2.0 us)!
    attn_l1 = {**base_l1, "case_id": "wb-attn", "point_id": "wb-attn", "context_len": 512, "target_family": "attention", "target_op_id": "s0.l0.paged_mla_decode"}
    for bad_bind in (
        {"measured_binding": {"builder": "fused_infer_attention", "family": "attention", "recipe": "fused_infer_attention", "paged": False, "heads": 16, "kv_heads": 2, "head_dim": 128}},
        {"measured_binding": {"builder": "absorbed_mla", "family": "attention", "recipe": "ascend_paged_mla_decode_arch22", "target_op_id": "s0.l0.paged_mla_decode", "query_layout": "TND"}},
        {"measured_binding": {"builder": "absorbed_mla", "family": "attention", "recipe": "ascend_paged_mla_decode_arch22", "target_op_id": "s0.l0.paged_mla_decode", "kv_len": 512, "page_count": 4}},
    ):
        with pytest.raises(ModelingError, match="whitebox_measured_binding_mismatch"):
            score_sealed_gate_cases([{**attn_l1, **bad_bind}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    with pytest.raises(ModelingError, match="matrix_groups"):
        score_sealed_gate_cases([{**base_l1, "case_id": "l2-mg", "point_id": "l2-mg", "level": "L2", "context_len": 4097, "batch_size": 16, "target_family": None, "target_op_ids": ["s0.l0.paged_mla_decode", "s0.l0.v_expand"], "measured_binding": {"node_bindings": [{"op_id": "s0.l0.v_expand", "matrix_groups": 1}]}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    with pytest.raises(ModelingError, match="l3_op_set_mismatch"):
        score_sealed_gate_cases([{**base_l1, "case_id": "l3-sub", "point_id": "l3-sub", "level": "L3", "target_family": None, "timed_nodes": ["l3-sub:s0.l0.norm1", "l3-sub:s0.l0.paged_mla_decode"], "sim_case": {**b4_case, "model_overrides": {"layers": 1}}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    idx_row = {**base_l1, "case_id": "idx-2048", "point_id": "idx-2048", "context_len": 2049, "topk": 2048, "target_family": "indexer", "target_op_id": "s0.l2.index_quant_lightning", "measured_binding": {"target_op_id": "s0.l2.index_quant_lightning", "family": "indexer", "heads": 32, "head_dim": 128, "kv_heads": 1, "sparse_count": 2048, "query_dtype": "bf16", "cache_dtype": "bf16", "kv_len": 2050}}
    with pytest.raises(ModelingError, match="topk|query_dtype"):
        score_sealed_gate_cases([{**idx_row, "sim_case": {"compose_model": "deepseek-v4.1-flash", "compose_hardware": A2_SKU}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    assert score_sealed_gate_cases([{**idx_row, "sim_case": {"compose_model": "deepseek-v4.1-flash", "compose_hardware": A2_SKU, "model_overrides": {"index_topk": 2048, "index_dtype": "bf16"}}}], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"] > 0
    assert score_sealed_gate_cases([{**base_l1, "case_id": "tiny-4us", "point_id": "tiny-4us", "measured_ns": 4000, "busy_union_ns": 4000}], sku=A2_SKU, runner=_independent_engine_runner({"tiny-4us": 5500}, clock_op="busy_union"))["ok"] is True
    assert score_sealed_gate_cases([{**base_l1, "case_id": "tiny-4us-fail", "point_id": "tiny-4us-fail", "measured_ns": 4000, "busy_union_ns": 4000}], sku=A2_SKU, runner=_independent_engine_runner({"tiny-4us-fail": 6500}, clock_op="busy_union"))["ok"] is False

    # Dependent L2 (`norm1 -> q_proj`) emits `wire_ns` & passes check_lean_non_inversion; private capture URI scrubbed to opaque locator!
    l2_dep = {**base_l1, "case_id": "l2-dep", "point_id": "l2-dep", "level": "L2", "capture_uri": f"capture://audit-node.internal.example/lease-audituser-0abc/{base_l1['raw_sha256']}.json", "target_family": None, "target_op_ids": ["s0.l0.norm1", "s0.l0.q_proj"]}
    l2_ind = {**base_l1, "case_id": "l2-ind", "point_id": "l2-ind", "level": "L2", "target_family": None, "target_op_ids": ["s0.l0.q_proj", "s0.l0.kv_a"]}
    wb_r = WhiteboxSpineRunner(clock_operator="busy_union")
    r_dep = score_sealed_gate_cases([l2_dep], sku=A2_SKU, runner=wb_r)["residuals"][0]
    r_ind = score_sealed_gate_cases([l2_ind], sku=A2_SKU, runner=wb_r)["residuals"][0]
    assert r_dep["program_dag_sha256"] and r_dep["program_dag_sha256"] != r_ind["program_dag_sha256"] and "audit-node" not in r_dep["capture_uri"] and "audituser" not in r_dep["capture_uri"]
    dep_gc = GateCase("l2-dep", "L2", "heldout", CaptureBoundary(base_l1["raw_sha256"], "acq-l2", "r1", (), ("s1", "e1"), (0,), "device_ns", "s1", "e1", "fx_region", "ns", r_dep["capture_uri"]), "in", "st", "wb", CohortKey("L2", "c", A2_SKU, "decode", "eager", "cliff", "tp1"), 2500, payload=(("clock_operator", "busy_union"), ("context_len", 64), ("phase", "decode"), ("sim_case", b4_case), ("target_op_ids", ["s0.l0.norm1", "s0.l0.q_proj"])))
    dep_diag = dict(wb_r.run_l2(dep_gc).diagnostics)
    lp, ld = dep_diag["lean_program"], dep_diag["lean_dag"]
    assert ld["edges"] and "wire_ns" in ld["edges"][0]
    d_cert, _ = certify_lower_bound(ResourceChoiceProgram(lp["num_resources"], tuple(lp["global_work"]), ()), EventDagSpec(tuple(ld["node_quantum_ns"]), tuple(DagEdge(e["src"], e["dst"], e["wire_ns"]) for e in ld["edges"])))
    lean_ok, _, lean_v = check_lean_non_inversion([{"case_id": "l2-dep", "l_star_lean_ns": d_cert.l_star_lean_ns, "t_sim_ns": dep_diag["busy_union_ns"], "t_silicon_p99_ns": max(dep_diag["busy_union_ns"], 2600), "program": lp, "dag": ld, "certificate": {"farkas": {"scale_d": d_cert.farkas.scale_d, "lambda_num": list(d_cert.farkas.lambda_num), "mu_num": list(d_cert.farkas.mu_num), "lower_bound_ns": d_cert.farkas.lower_bound_ns}, "path_edges": list(d_cert.path_edges), "sink_node_id": d_cert.sink_node_id, "l_star_lean_ns": d_cert.l_star_lean_ns}}], expected_case_timings={"l2-dep": (dep_diag["busy_union_ns"], 2500, max(dep_diag["busy_union_ns"], 2600), dep_diag["program_dag_sha256"])})
    assert lean_ok is True and not lean_v

    # Full manifest gate rejects partial/missing captures & cross-run live report spoofing, passes verified full manifest!
    raw_full = tmp_path / "full.json"
    raw_full.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal_full = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_full)["primitives_payload"]
    full_plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    cap_dir, cap_dir_a = tmp_path / "captures", tmp_path / "captures_a"
    m_rows, m_preds, m_leans, m_abis, m_shas = _plan_fixture_bundle(full_plan, capture_dir=cap_dir)
    ml_cases = [{"case_id": r["case_id"], "point_id": r["point_id"], "level": r["level"], "split": r["split"], "boundary": {"raw_sha256": r["raw_sha256"], "acquisition_uid": r["acquisition_root"], "run_uid": r["run_uid"], "parent_digests": [], "event_ids": r["event_ids"], "device_ranks": r["device_ranks"], "clock_domain": "device_ns", "start_event": "start", "end_event": "end", "boundary_kind": {"L1": "device_kernel", "L2": "fx_region", "L3": "model_step"}[r["level"]], "unit": "ns", "capture_uri": r["capture_uri"]}, "input_digest": r["raw_sha256"], "stack_digest": hashlib.sha256(b"st").hexdigest(), "implementation": "whitebox", "cohort": {"level": r["level"], "cohort_name": "wb", "sku": A2_SKU, "phase": "decode", "mode": "eager", "shape_regime": "cliff", "parallel_topology": r["parallel_topology"]}, "measured_ns": r["measured_ns"], "measured_p99_ns": r["measured_p99_ns"], "clock_operator": r["clock_operator"], "payload": {k: r[k] for k in ("context_len", "tokens", "batch_size", "query_tokens", "parallel_topology", "topk", "batch_x_kv_heads") if r.get(k) is not None}} for r in m_rows]
    full_f = tmp_path / "ml_full.json"
    full_f.write_text(json.dumps({"manifest_id": "m-full", "schema_version": "mindie-multilevel-manifest/1", "cases": ml_cases, "abi_rejection_results": m_abis}), encoding="utf-8")
    with pytest.raises(ModelingError, match="capture_file_missing"):
        verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=full_f, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=_synthetic_repo(tmp_path, family="ml_nocap"))
    full_ok = verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=full_f, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=_synthetic_repo(tmp_path, family="ml_ok"))
    assert full_ok["ok"] is True and full_ok["status"] == "offline_reverify"
    raw_a = tmp_path / "raw_a.json"
    raw_a.write_text(json.dumps(_raw_l0_samples(A2_SKU, scale=1.01)), encoding="utf-8")
    cal_a = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_a)["primitives_payload"]
    a_rows, a_preds, _, a_abis, a_shas = _plan_fixture_bundle(full_plan, scale=1.01, capture_dir=cap_dir_a)
    val_a = validate_512_400k(sku=A2_SKU, primitives_json=cal_a, profiler_runner=lambda _p: a_rows, abi_rejection_results=a_abis, capture_root=cap_dir_a, runner=_independent_engine_runner(a_preds, prog_sha_by_id=a_shas))
    cross_spoof = verify_sku_gates(sku=A2_SKU, validation_report_json=val_a, multilevel_manifest_json=full_f, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, acquisition_mode="live_silicon", repo_root=_synthetic_repo(tmp_path, family="ml_spoof"))
    assert cross_spoof["ok"] is False and cross_spoof["status"] != "silicon_certified"
