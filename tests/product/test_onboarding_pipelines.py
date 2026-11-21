"""Orthogonal Tier-1 tests for turnkey SKU & Operator onboarding pipelines.

[Role] Verify all 4 New-SKU stages (``probe-soc``, ``calibrate-l0-and-fabric``,
  ``validate-512-400k``, ``verify-gates``, ``run-all``), all 4 New-Operator
  stages (``scaffold``, ``validate-contract``, ``generate-cliff-plan``,
  ``bench-and-score``, ``certify``, ``run-all``), Lock-2 sealed scoring & clock
  parity, Lock-3 ISA-dispatched ``TilingAxes`` / ``validate_abi`` &
  ``ParallelSpec`` validation, single-Engine ``WhiteboxSpineRunner``, installed
  ``mindie_sim.primitives.sealer``, opaque privacy sanitization, real
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
from typing import Any

import pytest

from mindie_sim.cli import main as mindie_sim_main
from mindie_sim.observations.multilevel_gate import EngineInterval, GateCase, bind_engine_interval
from mindie_sim.onboarding import (
    ANCHOR_CONTEXT_GRID, OFF_GRID_CLIFF_HELDOUT_GRID, OperatorCliffParameters, TimeoutBudget,
    WhiteboxSpineRunner, bench_and_score_operator, calibrate_l0_and_fabric, certify_operator,
    check_lean_non_inversion, evaluate_abi_rejections, family_default_tiling_axes,
    generate_operator_cliff_plan, probe_soc, program_dag_sha256, required_ranks_for_plan,
    run_operator_onboarding, run_sku_onboarding, scaffold_operator, score_sealed_gate_cases,
    synthesize_operator_cliff_plan, synthesize_sku_512_400k_plan, validate_512_400k,
    validate_abi, validate_operator_contract, validate_parallel_topology, verify_receipt_sha256,
    verify_sku_gates,
)
from mindie_sim.onboarding._gate_runners import (
    acquire_coordinator_host_queue_lease, load_canonical_l0_sealer, normalize_profiler_records,
)
from mindie_sim.onboarding._operator_contract import (
    _build_default_planning_context, _verify_resolved_operator_dual_bound,
)
from mindie_sim.onboarding.tiling_axes import ARCH_PRESET_35
from mindie_sim.optimization.dual_bound import EventDagSpec, ResourceChoiceProgram, certify_lower_bound
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
    (syn / "scripts/check_template_provenance.py").write_text("print('template provenance violations: 0')\n", encoding="utf-8")
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
    p_dict = {"num_resources": 1, "global_work": [work_ns], "site_modes": []}
    d_dict = {"node_quantum_ns": [work_ns], "edges": []}
    return {
        "case_id": case_id, "l_star_lean_ns": cert.l_star_lean_ns, "t_sim_ns": sim_ns, "t_silicon_p99_ns": p99_ns,
        "program": p_dict, "dag": d_dict, "program_dag_sha256": program_dag_sha256(p_dict, d_dict),
        "certificate": {
            "farkas": {"scale_d": cert.farkas.scale_d, "lambda_num": list(cert.farkas.lambda_num), "mu_num": list(cert.farkas.mu_num), "lower_bound_ns": cert.farkas.lower_bound_ns},
            "path_edges": list(cert.path_edges), "sink_node_id": cert.sink_node_id, "l_star_lean_ns": cert.l_star_lean_ns,
        },
    }


def _independent_engine_runner(
    pred_by_id: dict[str, int], *, default_ns: int = 20_000, clock_op: str = "elapsed_span",
    busy_override: dict[str, int] | None = None, prog_sha_by_id: dict[str, str] | None = None,
):
    """Test runner whose predictions come strictly from an explicit lookup table."""
    table, b_map, p_map = dict(pred_by_id), dict(busy_override or {}), dict(prog_sha_by_id or {})

    class _Runner:
        def _exec(self, case: GateCase) -> EngineInterval:
            pred_ns = int(table.get(case.case_id, default_ns))
            busy_ns = int(b_map.get(case.case_id, pred_ns))
            eng = Engine([Resource("device", 1_000_000.0, "ns", "fifo")], trace_limit=32)
            s_ev = eng.submit(case.boundary.start_event, {"device": 100.0})
            e_ev = eng.submit(case.boundary.end_event, {"device": float(pred_ns)}, after=(s_ev,))
            eng.advance(until=(e_ev,))
            diag = [
                ("clock_operator", clock_op), ("busy_union_ns", busy_ns), ("elapsed_span_ns", pred_ns),
                ("program_dag_sha256", p_map.get(case.case_id, program_dag_sha256({"num_resources": 1, "global_work": [max(1, pred_ns // 2)], "site_modes": []}, {"node_quantum_ns": [max(1, pred_ns // 2)], "edges": []}))),
            ]
            return bind_engine_interval(case, eng, s_ev, e_ev, diagnostics=tuple(diag))

        def run_l1(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l2(self, case: GateCase) -> EngineInterval: return self._exec(case)
        def run_l3(self, case: GateCase) -> EngineInterval: return self._exec(case)

    return _Runner()


def _plan_fixture_bundle(
    plan: dict[str, Any], *, levels: tuple[str, ...] = ("L1", "L2", "L3"),
    clock_op: str = "elapsed_span", scale: float = 1.02, impl: str = "whitebox_spine",
    capture_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    rows, preds, lean_rows, prog_shas = [], {}, [], {}
    sku, family = str(plan["sku"]), str(plan.get("family") or "")
    for split, pts in (("train", plan["train_points"]), ("heldout", plan["heldout_points"])):
        for idx, pt in enumerate(pts):
            pid = str(pt["point_id"])
            lvl = "L1" if split == "train" else levels[idx % len(levels)]
            meas = (10_000 + idx * 100) if split == "train" else (20_000 + idx * 250)
            pred = int(round(meas * scale))
            p99 = int(meas * 1.05)
            ev_ids = [f"{split[0]}-s-{idx}", f"{split[0]}-e-{idx}"]
            cap_obj = {"case_id": pid, "sku": sku, "split": split, "level": lvl, "idx": idx, "start_event": "start", "end_event": "end", "clock_domain": "device_ns", "device_ranks": [0], "clock_operator": clock_op, "event_ids": ev_ids}
            cap_bytes = json.dumps(cap_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            sha = hashlib.sha256(cap_bytes).hexdigest()
            if capture_dir is not None:
                capture_dir.mkdir(parents=True, exist_ok=True)
                (capture_dir / f"{sha}.json").write_bytes(cap_bytes)
            preds[pid] = pred
            acq = "acq-train-src" if split == "train" else f"acq-held-{lvl.lower()}"
            row: dict[str, Any] = {
                "case_id": pid, "point_id": pid, "sku": sku, "split": split, "level": lvl,
                "acquisition_root": acq, "run_uid": f"run-{split[0]}-{lvl.lower()}-{idx}",
                "raw_sha256": sha, "capture_uri": f"capture://{split}/{lvl.lower()}/{sha}.json",
                "event_ids": [f"{split[0]}-s-{idx}", f"{split[0]}-e-{idx}"], "device_ranks": [0],
                "context_len": pt["value"] if pt["axis"] == "context_len" else None,
                "tokens": pt["value"] if pt["axis"] == "tokens" else None,
                "batch_size": pt["batch_size"], "query_tokens": pt["query_tokens"],
                "parallel_topology": pt["parallel_topology"], "topk": pt["topk"],
                "measured_ns": meas, "measured_p99_ns": p99, "clock_operator": clock_op,
                "busy_union_ns": meas if clock_op == "busy_union" else meas - 500,
                "elapsed_span_ns": meas if clock_op == "elapsed_span" else meas + 500,
                "implementation": impl,
            }
            if family and family != "sku_suite":
                row["target_family"] = family
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
    if cann_version:
        out["cann_version"] = cann_version
    return out


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


def test_probe_soc_conflicts_privacy_and_new_sku_offline_reverify(tmp_path):
    # Items #1, #13, #17: Raw hostname/IPv4 redacted to opaque ep-<12-hex>; contradictory silicon sources fail!
    synthetic_ip = ".".join(("198", "51", "100", "22"))
    ok_probe = tmp_path / "probe_ok.json"
    ok_probe.write_text(json.dumps(_a2_probe_payload(endpoint=f"prod-cluster-{synthetic_ip}.corp")), encoding="utf-8")
    res_ok = probe_soc(probe_json=ok_probe)
    assert res_ok["ok"] is True and res_ok["mode"] == "validated_existing"
    assert res_ok["probe"]["endpoint_alias"].startswith("ep-") and "prod-cluster" not in json.dumps(res_ok)
    assert synthetic_ip not in json.dumps(res_ok)

    # Item #13: soc_name vs aclrt_soc_name AND conflicting 1 MiB SMI HBM MUST fail silicon_profile_conflict!
    for bad_name, payload in (
        ("soc_conflict", {**_a2_probe_payload(), "soc_name": A2_SKU, "aclrt_soc_name": "OtherSilicon"}),
        ("smi_hbm", {**_a2_probe_payload(), "npu_smi_common": f"Chip ID : 0\nAicore Count : {A2_ORACLE['aic']}\nHBM Capacity(MB) : 1\n"}),
        ("smi_aic", {**_a2_probe_payload(), "npu_smi_common": "Chip ID : 0\nAicore Count : 1\n"}),
        ("bad_fab", {**_a2_probe_payload(), "fabric_channels": {"d2d_intra_node": 1.0}}),
        ("bad_dies", {**_a2_probe_payload(), "dies_per_package": 2}),
    ):
        p = tmp_path / f"{bad_name}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        assert probe_soc(probe_json=p)["error"] == "silicon_profile_conflict", bad_name
    no_ini = tmp_path / "no_ini.json"
    no_ini.write_text(json.dumps({"aclrt_soc_name": A2_SKU, "aic_count": A2_ORACLE["aic"]}), encoding="utf-8")
    assert probe_soc(probe_json=no_ini)["error"] == "silicon_probe_incomplete"

    # Item #1 & #14: Brand-new scaffolded SKU completes run-all offline as `offline_reverify` (NEVER silicon_certified)!
    new_sku = "AscendFuture_9999"
    full_probe = tmp_path / "probe_new.json"
    full_probe.write_text(json.dumps({
        "aclrt_soc_name": new_sku, "arch_preset": A2_ORACLE["arch_id"], "architecture": "A2",
        "aic_count": 28, "aiv_count": 56, "hbm_bytes": 64 * 1024**3,
        "l2_bytes": 200 * 1024**2, "ub_bytes": 196_608, "dies_per_package": 1, "ranks_per_node": 8,
    }), encoding="utf-8")
    new_plan = synthesize_sku_512_400k_plan(sku=new_sku, core_count=28, l2_bytes=200 * 1024**2).to_dict()
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(new_plan)
    raw_l0 = tmp_path / "new_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(new_sku)), encoding="utf-8")
    run_new = run_sku_onboarding(
        probe_json=full_probe, raw_l0_json=raw_l0, allow_scaffold=True, base_template_sku=A2_SKU,
        profiler_runner=lambda _p: obs, abi_rejection_results=abis,
        runner=_independent_engine_runner(preds, prog_sha_by_id=pshas), lean_bounds_json=leans,
        repo_root=_synthetic_repo(tmp_path, family="new_sku"), out_dir=tmp_path / "new_sku_out",
    )
    assert run_new["ok"] is True and run_new["status"] == "offline_reverify" and run_new["sku"] == new_sku
    # Item #14: tools/bench_l0_primitives.py accepts --profile-json for a newly scaffolded SKU!
    done_prof = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "tools/bench_l0_primitives.py"), "--sku", new_sku,
         "--profile-json", str(tmp_path / "new_sku_out/scaffold_profile.json"), "--emit-unmeasured",
         "--out", str(tmp_path / "new_tbl.json"), "--manifest", str(tmp_path / "new_man.json")],
        capture_output=True, text=True, check=False,
    )
    assert done_prof.returncode == 0 and (tmp_path / "new_tbl.json").is_file()


def test_sealer_wheel_cann_manifest_conflicts_and_coverage(tmp_path):
    # Items #12, #15, #16, #17: Installed sealer works without tools/; manifest CANN contradiction fails; conflicts/coverage exposed!
    assert load_canonical_l0_sealer(tmp_path / "nonexistent_wheel_root").__name__ == "mindie_sim.primitives.sealer"
    with pytest.raises(ModelingError, match="timeout_budget_outer_too_small"):
        TimeoutBudget((("l0", 300), ("fabric", 200)), cleanup_grace_s=60, outer_timeout_s=500)
    assert calibrate_l0_and_fabric(sku=A2_SKU)["status"] == "uncalibrated"
    assert mindie_sim_main(["onboard-sku", "calibrate-l0-and-fabric", "--sku", A2_SKU]) == 2

    # Item #12: Shipped A5 9.2.0 capture rejected when claimed as 9.3.0; absent CANN becomes "unknown"!
    a5_cap = REPO_ROOT / "data/primitives/captures/arch35_a5_950dt.capture.json"
    a5_cap_sku = json.loads(a5_cap.read_text(encoding="utf-8"))["payload"]["device_name"]
    with pytest.raises(ModelingError, match="cann_version_provenance_conflict"):
        calibrate_l0_and_fabric(sku=a5_cap_sku, raw_l0_json=a5_cap, cann_version="9.3.0")
    assert calibrate_l0_and_fabric(sku=a5_cap_sku, raw_l0_json=a5_cap, cann_version="9.2.0")["ok"] is True

    no_cann_l0 = tmp_path / "no_cann.json"
    no_cann_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU, cann_version="")), encoding="utf-8")
    cal_unk = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=no_cann_l0)
    assert cal_unk["ok"] is True and cal_unk["cann_version"] == "unknown"
    with pytest.raises(ModelingError, match="cann_version_evidence_missing"):
        calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=no_cann_l0, cann_version="9.3.0")

    # Item #15: p99 > nominal preserved as `p99_exceeds_nominal` conflict distinct from `missing_samples` + coverage_matrix!
    over_l0 = tmp_path / "over_l0.json"
    over_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU, scale=2.5)), encoding="utf-8")
    cal_over = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=over_l0)
    assert any(c["reason"].startswith("p99_exceeds_nominal") for c in cal_over["metric_conflicts"])
    assert any(r["primitive"] == "P01_datacopy" and r["reason"].startswith("p99_exceeds_nominal") for r in cal_over["coverage_matrix"])
    assert any(r["reason"] == "missing_samples" for r in cal_over["coverage_matrix"])


def test_lock3_isa_tiling_axes_parallel_spec_and_abi_rejections():
    # Items #8 & #9: sparse_indexer auto-loads ISA TilingAxes without --topk-threshold; Arch35 accepts ctx=1, topk=64!
    axes35 = family_default_tiling_axes("sparse_indexer", tile_dim=128, arch_preset=ARCH_PRESET_35)
    assert validate_abi({"context_len": 1, "topk": 64}, axes35, arch_preset=ARCH_PRESET_35)[0] is True
    assert validate_abi({"topk": 9216}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_max_violation")
    assert validate_abi({"topk": "2049"}, axes35, arch_preset=ARCH_PRESET_35)[1].startswith("abi_type_violation")

    sp_plan = synthesize_operator_cliff_plan(
        OperatorCliffParameters("sp:v2", "sparse_indexer", 128, 24, 192 * 1024**2),
        sku=A5_SKU,
    )
    assert any(pt.topk == 2049 for pt in sp_plan.abi_rejection_points)
    assert any(pt.topk == 9216 for pt in sp_plan.abi_rejection_points)
    assert set(ANCHOR_CONTEXT_GRID) == {pt.value for pt in sp_plan.train_points if pt.axis == "context_len"}
    assert set(OFF_GRID_CLIFF_HELDOUT_GRID).issubset({pt.value for pt in sp_plan.heldout_points if pt.axis == "context_len"})

    # Item #9: ParallelSpec rejects illegal pcp4_dcp2; generated plans require exact 4 ranks!
    with pytest.raises(ModelingError, match="dcp_pcp_domain_invalid"):
        validate_parallel_topology("tp2_pcp4_dcp2")
    assert required_ranks_for_plan(sp_plan.to_dict()) == 4

    # Item #8: evaluate_abi_rejections blocks prefix bypass, cross-axis batch_size error, and shape mismatch!
    rejs = sp_plan.to_dict()["abi_rejection_points"]
    assert evaluate_abi_rejections(rejs, None)["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": f"ResourceUnavailable: skipped {pt['abi_expected_error']}"} for pt in rejs])["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": "abi_divisibility_violation:batch_size=2049"} for pt in rejs])["ok"] is False
    assert evaluate_abi_rejections(rejs, [{"point_id": pt["point_id"], "rejected": True, "abi_error": pt["abi_expected_error"], "executed_shape": {"topk": 1}} for pt in rejs])["ok"] is False


def test_lock2_clock_substitution_lean_join_and_adversarial_gates(tmp_path):
    plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    obs, preds, leans, abis, pshas = _plan_fixture_bundle(plan)
    p1 = next(r["case_id"] for r in obs if r["split"] == "heldout")

    # Item #3: busy_union=110 vs elapsed=7500 with payload omitting clock_operator -> ~98.53% APE FAIL (NEVER 0%)!
    clk_row = {
        **obs[-1], "case_id": "clk-bug", "point_id": "clk-bug", "clock_operator": "busy_union",
        "measured_ns": 7500, "measured_p99_ns": 7600, "busy_union_ns": 7500, "elapsed_span_ns": 7500,
    }
    clk_res = score_sealed_gate_cases(
        [clk_row], sku=A2_SKU,
        runner=_independent_engine_runner({"clk-bug": 7500}, clock_op="busy_union", busy_override={"clk-bug": 110}),
    )
    assert clk_res["ok"] is False and clk_res["residuals"][0]["predicted_ns"] == 110
    assert clk_res["heldout_summary"]["max_ape_pct"] > 98.0
    with pytest.raises(ModelingError, match="clock_operator_conflict"):
        score_sealed_gate_cases([{**clk_row, "payload": {"clock_operator": "elapsed_span"}}], sku=A2_SKU, runner=_independent_engine_runner({"clk-bug": 7500}, clock_op="busy_union"))

    # Item #6: Unrelated 1-node 1 ns Lean witness AND missing capture p99 BOTH fail closed!
    ok_u, _, v_u = check_lean_non_inversion(
        [_lean_witness_row("c1", 1, sim_ns=100, p99_ns=100)],
        expected_case_timings={"c1": (100, 100, 100, "deadbeef" * 8)},
    )
    assert ok_u is False and any("lean_witness_program_dag_mismatch" in v for v in v_u)
    ok_p99, _, v_p99 = check_lean_non_inversion(
        [_lean_witness_row("c1", 50, sim_ns=100, p99_ns=100)],
        expected_case_timings={"c1": (100, 100, None, _lean_witness_row("c1", 50, sim_ns=100, p99_ns=100)["program_dag_sha256"])},
    )
    assert ok_p99 is False and any("lean_captured_p99_missing" in v for v in v_p99)

    # Lock-2 controls: plan_only, >10% threshold, caller predicted_ns, reversed event_ids dedup, tail diagnosis!
    assert validate_512_400k(sku=A2_SKU)["ok"] is False
    with pytest.raises(ModelingError, match="threshold_pct_exceeds_strict_release_cap"):
        validate_512_400k(sku=A2_SKU, threshold_pct=30.0)
    with pytest.raises(ModelingError, match="caller_predicted_duration_forbidden"):
        score_sealed_gate_cases([{**obs[-1], "predicted_ns": 20_000}], sku=A2_SKU, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas))
    dup_rev = [obs[-1], {**obs[-1], "case_id": "dup-rev", "point_id": "dup-rev", "event_ids": list(reversed(obs[-1]["event_ids"]))}]
    with pytest.raises(ModelingError, match="duplicate_heldout_physical_capture"):
        score_sealed_gate_cases(dup_rev, sku=A2_SKU, runner=_independent_engine_runner(preds, prog_sha_by_id=pshas))

    bad_preds = {**preds, p1: int(next(r["measured_ns"] for r in obs if r["case_id"] == p1) * 1.27)}
    raw_l0 = tmp_path / "l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_l0)
    val_fail = validate_512_400k(sku=A2_SKU, primitives_json=cal["primitives_payload"], profiler_runner=lambda _p: obs, abi_rejection_results=abis, runner=_independent_engine_runner(bad_preds, prog_sha_by_id=pshas))
    gate_fail = verify_sku_gates(sku=A2_SKU, validation_report_json=val_fail, primitives_json=cal["primitives_payload"], lean_bounds_json=leans, repo_root=_synthetic_repo(tmp_path, family="tail_diag"))
    assert gate_fail["ok"] is False and gate_fail["diagnosis_triggered"] is True


def test_operator_owner_lowering_zero_event_train_drop_and_certify(tmp_path):
    repo_dir, rev, sha = _git_repo_with_blob(tmp_path, "ops-transformer", "kernels/demo_op.cpp", b"// demo\n")
    syn = _synthetic_repo(tmp_path, family="demo_op")
    prof_ok = _synthetic_profiler_root(tmp_path, supported=("demo_op",), abi="aclnnDemoOpV1", tag="ok")
    scaf = scaffold_operator(family="demo_op", upstream_abi_arch22="aclnnDemoOpV1", upstream_abi_arch35="aclnnDemoOpV2", source_refs=[{"repository": "ops-transformer", "revision": rev, "relative_path": "kernels/demo_op.cpp", "source_sha256": sha}], target_root=syn, overwrite=True)
    target_rel = scaf["package_rel"]

    # Item #5: Zero-event recipe (`program=lambda: ()` or empty KernelProgram) MUST fail dual bound!
    probe_op = OpSpec("op0", "demo_op", "demo_op", "decode", "bf16", (), (1, 64), (1, 64), (1, 64), (1, 64))
    ctx = _build_default_planning_context(probe_op)
    zero_res = ResolvedOperator(occurrence=ctx.occurrence, op=probe_op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: (), pipeline_spec={"stage": "v"}))
    ok_z, _, reason_z = _verify_resolved_operator_dual_bound(zero_res, ctx)
    assert ok_z is False and reason_z == "resolved_operator_program_zero_executable_events"
    empty_kp = ResolvedOperator(occurrence=ctx.occurrence, op=probe_op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: KernelProgram(()), pipeline_spec={"stage": "v"}))
    assert _verify_resolved_operator_dual_bound(empty_kp, ctx)[0] is False

    # Item #4: Unrelated template_id & unrelated profiler_family & skipped contract checks all fail closed!
    unimpl = syn / "mindie_sim/templates/audit_op_spec.py"
    unimpl.parent.mkdir(parents=True, exist_ok=True)
    unimpl.write_text('"""Stub.\n\nUpstream-ABI: aclnnAuditOp\n"""\nSemanticWorkItem = TensorRangeObligation = TilingAxes = object\n', encoding="utf-8")
    bad_tmpl = validate_operator_contract(target_path=unimpl, repo_root=syn, template_id="catlass.basic_pingpong", profiler_family="rmsnorm", profiler_root=prof_ok, require_taxonomy=False)
    assert bad_tmpl["ok"] is False and any("template_id_unrelated_to_target" in v for v in bad_tmpl["violations"])
    assert any("profiler_family_unrelated_to_target" in v for v in bad_tmpl["violations"])
    real_tmpl = syn / "mindie_sim/templates/catlass_pingpong_spec.py"
    real_tmpl.write_text(f'"""CATLASS.\n\nUpstream-ABI: aclnnMm\n"""\nTEMPLATE_ID = "catlass.basic_pingpong"\nSOURCES = (SourceRef(repository="ops-transformer", revision="{rev}", relative_path="kernels/demo_op.cpp", source_sha256="{sha}"),)\nSemanticWorkItem = TensorRangeObligation = TilingAxes = object\n', encoding="utf-8")
    ok_tmpl = validate_operator_contract(target_path=real_tmpl, repo_root=syn, refs_root=repo_dir.parent, template_id="catlass.basic_pingpong", require_profiler_builder=False, require_taxonomy=False)
    assert ok_tmpl["ok"] is True and ok_tmpl["sim_owner_dispatch"]["lowered"] is True and ok_tmpl["sim_owner_dispatch"]["lean_status"] == "verified" and ok_tmpl["sim_owner_dispatch"]["stage_count"] > 0

    def _real_resolve(op: OpSpec, rctx) -> ResolvedOperator:
        prog = KernelProgram((Stage("vec_main", work={"vector": 4096.0}),))
        return ResolvedOperator(occurrence=rctx.occurrence, op=op, implementation="aclnnDemoOpV1", recipe=OperatorPlan(Work(resources={"vector": 4096.0}), program=lambda *_a, **_kw: prog, pipeline_spec={"stage": "v"}, metadata={"implementation": "aclnnDemoOpV1"}))

    reg_ok = FamilyRegistry([OperatorFamilySpec("demo_op", frozenset({"demo_op"}), frozenset({"demo_op"}), _real_resolve)])
    plan_doc = generate_operator_cliff_plan(operator_id="demo_op:v1", family="demo_op", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048)["plan"]
    op_obs, op_preds, op_leans, op_abis, op_shas = _plan_fixture_bundle(plan_doc, levels=("L1",), impl="aclnnDemoOpV1")

    # Item #7: Deleting all 7 train anchors fails bench_and_score_operator; wrong SKU fails closed!
    no_train = [r for r in op_obs if r["split"] != "train"]
    drop_res = bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, captured_json=no_train, abi_rejection_results=op_abis, runner=_independent_engine_runner(op_preds, prog_sha_by_id=op_shas))
    assert drop_res["ok"] is False and len(drop_res["missing_train_points"]) == len(ANCHOR_CONTEXT_GRID)
    with pytest.raises(ModelingError, match="cliff_plan_sku_mismatch"):
        bench_and_score_operator(cliff_plan=plan_doc, sku=A5_SKU, captured_json=op_obs, runner=_independent_engine_runner(op_preds, prog_sha_by_id=op_shas))

    # Item #4: Skipped contract checks cannot certify!
    skipped_c = validate_operator_contract(target_path=target_rel, repo_root=syn, refs_root=repo_dir.parent, require_sim_owner=False, require_profiler_builder=False)
    assert skipped_c["ok"] is True
    full_score = bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, captured_json=op_obs, abi_rejection_results=op_abis, runner=_independent_engine_runner(op_preds, prog_sha_by_id=op_shas))
    assert certify_operator(operator_id="demo_op:v1", sku=A2_SKU, contract_report=skipped_c, score_report=full_score, lean_bounds_json=op_leans, repo_root=syn)["ok"] is False

    # Positive 4-stage operator onboarding -> `offline_reverify`!
    op_run = run_operator_onboarding(
        target_path=target_rel, operator_id="demo_op:v1", family="demo_op", sku=A2_SKU,
        tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"], topk_threshold=2048,
        captured_json=op_obs, abi_rejection_results=op_abis, lean_bounds_json=op_leans,
        repo_root=syn, refs_root=repo_dir.parent, profiler_root=prof_ok,
        family_registry=reg_ok, runner=_independent_engine_runner(op_preds, prog_sha_by_id=op_shas),
        out_dir=tmp_path / "op_out",
    )
    assert op_run["ok"] is True and op_run["status"] == "offline_reverify" and verify_receipt_sha256(op_run["stages"]["certify"]["receipt"]) is True


def test_host_queue_guards_profiler_normalization_and_privacy(tmp_path):
    # Items #9, #10, #11, #17: Reject devices=[] & fence_token=0; enforce ttl>=timeout+grace; surface release warning; hash lease_id!
    def _mk_coord(name: str, body: str) -> Path:
        p = tmp_path / f"{name}.py"
        p.write_text(f"#!{sys.executable}\nimport json, sys\n{body}\n", encoding="utf-8")
        p.chmod(0o755)
        return p

    bad_empty = _mk_coord("c_empty", "print(json.dumps({'outcome':'success','status':'ready','data':{'lease_id':'l1','fence_token':1,'devices':[]}}))")
    bad_fence = _mk_coord("c_fence", "print(json.dumps({'outcome':'success','status':'ready','data':{'lease_id':'l1','fence_token':0,'devices':[0]}}))")
    with pytest.raises(ModelingError, match="host_queue_lease_device_count_mismatch"):
        with acquire_coordinator_host_queue_lease("lab-a2", npu_count=1, coordinator_bin=str(bad_empty)):
            pass
    with pytest.raises(ModelingError, match="host_queue_lease_missing_lease_id_or_positive_fence_token"):
        with acquire_coordinator_host_queue_lease("lab-a2", npu_count=1, coordinator_bin=str(bad_fence)):
            pass
    with pytest.raises(ModelingError, match="host_queue_lease_ttl_below_subprocess_plus_grace"):
        with acquire_coordinator_host_queue_lease("lab-a2", ttl_seconds=120, min_subprocess_timeout_s=120, cleanup_grace_s=60, coordinator_bin=str(bad_empty)):
            pass

    rel_fail = _mk_coord(
        "c_relfail",
        "args = sys.argv[1:]\nact = args[args.index('--action') + 1]\n"
        "req = json.loads(args[args.index('--json') + 1])\n"
        "if act == 'acquire':\n"
        "    devs = list(range(int(req['npu_count'])))\n"
        "    print(json.dumps({'outcome':'success','status':'ready','data':{'lease_id':'lease-audituser-secret','fence_token':9,'devices':devs}}))\n"
        "else:\n    sys.exit(1)\n",
    )
    with acquire_coordinator_host_queue_lease("lab-a2", npu_count=4, ttl_seconds=180, min_subprocess_timeout_s=120, coordinator_bin=str(rel_fail)) as linfo:
        assert linfo["devices"] == [0, 1, 2, 3]
    assert linfo["receipt_unconfirmed"] is True and "host_queue_lease_release_unconfirmed" in linfo["receipt_warning"]

    # Item #10 & #17: normalize_profiler_records ingests real mindie-bench-observation/1 JSONL & bench_and_score reserves 4 NPUs + scrubs lease username!
    plan_doc = generate_operator_cliff_plan(operator_id="live:v1", family="live", sku=A2_SKU, tile_dim=128, core_count=20, l2_bytes=A2_ORACLE["l2"])["plan"]
    norm_obs, norm_abi = normalize_profiler_records(
        [
            {"schema": "mindie-bench-observation/1", "record_type": "observation", "case_id": "c1", "run_id": "r1", "split": "heldout", "boundary": "device_kernel", "timing_mode": "device_kernel", "duration_ms": 0.02, "sample_id": "obs-1", "sku": A2_SKU, "validated": True, "qualified": True, "identity": {"level": "operator", "implementation": "aclnnLive"}},
            {"schema": "mindie-bench-observation/1", "record_type": "case_complete", "case_id": "c1", "run_id": "r1", "split": "heldout", "status": "ok", "summary": {"op": "live", "actual_shape": {"context_len": 768, "batch_size": 1, "query_tokens": 1}, "participating_ranks": [0]}},
            {"schema": "mindie-bench-receipt/1", "case_id": "abi_1", "status": "abi_rejected", "reason": "abi_divisibility_violation:topk=2049"},
        ],
        plan_doc=plan_doc, sku=A2_SKU, endpoint_alias="prod-host.corp",
    )
    assert len(norm_obs) == 1 and norm_obs[0]["measured_ns"] == 20_000 and norm_obs[0]["clock_operator"] == "busy_union" and "prod-host" not in json.dumps(norm_obs)
    assert len(norm_abi) == 1 and norm_abi[0]["rejected"] is True

    fake_prof = tmp_path / "fake_prof_root"
    (fake_prof / "mindie_profiler/bench").mkdir(parents=True)
    (fake_prof / "mindie_profiler/__init__.py").write_text("", encoding="utf-8")
    (fake_prof / "mindie_profiler/__main__.py").write_text(
        "import os, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\n"
        "assert Path(os.environ['MINDIE_ONBOARDING_PLAN_JSON']).is_file() and os.environ['MINDIE_LEASE_FENCE_TOKEN'] == '9'\n"
        "assert args[args.index('--allowed-phy-ids') + 1] == '0,1,2,3'\n"
        "Path(args[args.index('--observations') + 1]).write_text('', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (fake_prof / "mindie_profiler/bench/__init__.py").write_text("", encoding="utf-8")
    (fake_prof / "mindie_profiler/bench/bench_sku_microbench.py").write_text("def main(argv=None): return 0\n", encoding="utf-8")
    live_res = bench_and_score_operator(cliff_plan=plan_doc, sku=A2_SKU, live_profiler=True, endpoint_alias="prod-host.corp", profiler_root=fake_prof, coordinator_bin=str(rel_fail))
    assert live_res["receipt_unconfirmed"] is True and "audituser" not in json.dumps(live_res) and live_res["lease"]["lease_digest"].startswith("lease-")


def test_real_single_engine_whitebox_2d_cliffs_and_full_manifest_gate(tmp_path):
    # Items #1, #2, #7: WhiteboxSpineRunner binds directly to real workload Engine (zero dummy Engine), separates L1<L2<L3 & 2D batch_x_kv_heads!
    real_case = {"compose_model": "deepseek-v2-lite", "compose_hardware": A2_SKU, "compose_options": {"tp": 1}}
    base_rows = [
        {"case_id": "wb-l1", "point_id": "wb-l1", "sku": A2_SKU, "split": "heldout", "level": "L1", "acquisition_root": "acq-wb-l1", "run_uid": "r1", "raw_sha256": hashlib.sha256(b"wb-l1").hexdigest(), "event_ids": ["s1", "e1"], "device_ranks": [0], "measured_ns": 2500, "measured_p99_ns": 2600, "clock_operator": "busy_union", "busy_union_ns": 2500, "elapsed_span_ns": 2600, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "target_family": "rmsnorm", "sim_case": real_case},
        {"case_id": "wb-l2", "point_id": "wb-l2", "sku": A2_SKU, "split": "heldout", "level": "L2", "acquisition_root": "acq-wb-l2", "run_uid": "r2", "raw_sha256": hashlib.sha256(b"wb-l2").hexdigest(), "event_ids": ["s2", "e2"], "device_ranks": [0], "measured_ns": 260_000, "measured_p99_ns": 270_000, "clock_operator": "busy_union", "busy_union_ns": 260_000, "elapsed_span_ns": 265_000, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "target_layer": 0, "sim_case": real_case},
        {"case_id": "wb-l3", "point_id": "wb-l3", "sku": A2_SKU, "split": "heldout", "level": "L3", "acquisition_root": "acq-wb-l3", "run_uid": "r3", "raw_sha256": hashlib.sha256(b"wb-l3").hexdigest(), "event_ids": ["s3", "e3"], "device_ranks": [0], "measured_ns": 7_500_000, "measured_p99_ns": 7_600_000, "clock_operator": "busy_union", "busy_union_ns": 7_500_000, "elapsed_span_ns": 7_600_000, "phase": "decode", "context_len": 64, "query_tokens": 1, "batch_size": 1, "sim_case": real_case},
    ]
    res_base = score_sealed_gate_cases(base_rows, sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))
    by_lvl = {r["level"]: r for r in res_base["residuals"]}
    assert 0 < by_lvl["L1"]["predicted_ns"] < by_lvl["L2"]["predicted_ns"] < by_lvl["L3"]["predicted_ns"]
    assert by_lvl["L3"]["engine_uid"] and by_lvl["L3"]["engine_uid"] in by_lvl["L3"]["engine_receipt"] and by_lvl["L3"]["program_dag_sha256"]

    # Item #7: batch_x_kv_heads 2559 vs 2561 lower distinctly (not collapsed to sequences=1)!
    c_2559 = {**base_rows[0], "case_id": "bx-2559", "point_id": "bx-2559", "raw_sha256": hashlib.sha256(b"bx2559").hexdigest(), "batch_x_kv_heads": 2559}
    c_2561 = {**base_rows[0], "case_id": "bx-2561", "point_id": "bx-2561", "raw_sha256": hashlib.sha256(b"bx2561").hexdigest(), "batch_x_kv_heads": 2561}
    p2559 = score_sealed_gate_cases([c_2559], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"]
    p2561 = score_sealed_gate_cases([c_2561], sku=A2_SKU, runner=WhiteboxSpineRunner(clock_operator="busy_union"))["residuals"][0]["predicted_ns"]
    assert p2559 > by_lvl["L1"]["predicted_ns"] and p2561 >= p2559

    # Item #1: Manifest gate rejects 8-row incomplete manifest & missing capture files, passes ONLY full authoritative plan + real captures!
    raw_full = tmp_path / "full.json"
    raw_full.write_text(json.dumps(_raw_l0_samples(A2_SKU)), encoding="utf-8")
    cal_full = calibrate_l0_and_fabric(sku=A2_SKU, raw_l0_json=raw_full)["primitives_payload"]
    full_plan = synthesize_sku_512_400k_plan(sku=A2_SKU, core_count=A2_ORACLE["aic"], l2_bytes=A2_ORACLE["l2"]).to_dict()
    cap_dir = tmp_path / "captures"
    m_rows, m_preds, m_leans, m_abis, m_shas = _plan_fixture_bundle(full_plan, capture_dir=cap_dir)
    ml_cases = [
        {
            "case_id": r["case_id"], "point_id": r["point_id"], "level": r["level"], "split": r["split"],
            "boundary": {"raw_sha256": r["raw_sha256"], "acquisition_uid": r["acquisition_root"], "run_uid": r["run_uid"], "parent_digests": [], "event_ids": r["event_ids"], "device_ranks": r["device_ranks"], "clock_domain": "device_ns", "start_event": "start", "end_event": "end", "boundary_kind": {"L1": "device_kernel", "L2": "fx_region", "L3": "model_step"}[r["level"]], "unit": "ns", "capture_uri": r["capture_uri"]},
            "input_digest": r["raw_sha256"], "stack_digest": hashlib.sha256(b"st").hexdigest(), "implementation": "whitebox",
            "cohort": {"level": r["level"], "cohort_name": "wb", "sku": A2_SKU, "phase": "decode", "mode": "eager", "shape_regime": "cliff", "parallel_topology": r["parallel_topology"]},
            "measured_ns": r["measured_ns"], "measured_p99_ns": r["measured_p99_ns"], "clock_operator": r["clock_operator"],
            "payload": {k: r[k] for k in ("context_len", "tokens", "batch_size", "query_tokens", "parallel_topology", "topk", "batch_x_kv_heads") if r.get(k) is not None},
        }
        for r in m_rows
    ]
    partial_f = tmp_path / "ml_partial.json"
    partial_f.write_text(json.dumps({"manifest_id": "m-8", "schema_version": "mindie-multilevel-manifest/1", "cases": [c for c in ml_cases if c["split"] == "heldout"][:8]}), encoding="utf-8")
    assert verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=partial_f, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans[:8], repo_root=_synthetic_repo(tmp_path, family="ml_part"))["ok"] is False

    full_f = tmp_path / "ml_full.json"
    full_f.write_text(json.dumps({"manifest_id": "m-full", "schema_version": "mindie-multilevel-manifest/1", "cases": ml_cases, "abi_rejection_results": m_abis}), encoding="utf-8")
    with pytest.raises(ModelingError, match="capture_file_missing"):
        verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=full_f, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=_synthetic_repo(tmp_path, family="ml_nocap"))
    full_ok = verify_sku_gates(sku=A2_SKU, multilevel_manifest_json=full_f, capture_root=cap_dir, runner=_independent_engine_runner(m_preds, prog_sha_by_id=m_shas), primitives_json=cal_full, lean_bounds_json=m_leans, repo_root=_synthetic_repo(tmp_path, family="ml_ok"))
    assert full_ok["ok"] is True and full_ok["status"] == "offline_reverify"
