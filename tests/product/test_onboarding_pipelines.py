"""End-to-end and fail-closed tests for SKU and Operator onboarding pipelines.

[Role] Verify all four New-SKU stages (``probe-soc``, ``calibrate-l0-and-fabric``,
  ``validate-512-400k``, ``verify-gates``, ``run-all``), all four New-Operator
  stages (``scaffold``, ``validate-contract``, ``generate-cliff-plan``,
  ``bench-and-score``, ``certify``, ``run-all``), CLI roundtrips, and every
  fail-closed guardrail.
[Invariants] Isolated environment and ``tmp_path`` workspaces only; never
  touches real host ``$HOME`` or live NPU devices. Tests reject
  ``silicon_profile_conflict``, ``train==heldout`` and cross-level lineage
  leakage, ``service_regimes`` crutches, missing/ungrounded ``Upstream-ABI:``,
  ``SourceRef`` SHA-256 mismatch, Lean dual-bound inversions, and auto-trigger
  ``diagnose_tail_cluster.py`` on ``>= 10.0%`` cliff tails.
[Timeouts & Locks] Bounded local subprocesses and temporary Git repos; no locks.
[Side-Effect Policy] Writes exclusively inside pytest ``tmp_path``.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.backends import ascend
from mindie_sim.onboarding import (
    ANCHOR_CONTEXT_GRID,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    TimeoutBudget,
    bench_and_score_operator,
    calibrate_l0_and_fabric,
    certify_operator,
    generate_operator_cliff_plan,
    probe_soc,
    run_operator_onboarding,
    run_sku_onboarding,
    scaffold_operator,
    validate_512_400k,
    validate_operator_contract,
    verify_receipt_sha256,
    verify_sku_gates,
)
from mindie_sim.optimization.dual_bound import (
    EventDagSpec,
    ResourceChoiceProgram,
    certify_lower_bound,
)
from mindie_sim.primitives.schema import parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Hardware

pytestmark = pytest.mark.tier1

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPEATS = 12


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    tmp_sub = tmp_path / "sub_tmp"
    tmp_sub.mkdir(exist_ok=True)
    return {
        **os.environ,
        "TMPDIR": str(tmp_sub),
        "TEMP": str(tmp_sub),
        "TMP": str(tmp_sub),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _make_clean_synthetic_repo(tmp_path: Path, family: str | None = None) -> Path:
    repo = tmp_path / "syn_repo"
    for sub in ("scripts", "mindie_sim/prediction/operators", "tests/operators", "tests/_support"):
        (repo / sub).mkdir(parents=True)
    for name in ("check_arch_governance.py", "check_test_orthogonality.py", "check_template_provenance.py", "diagnose_tail_cluster.py"):
        shutil.copy2(REPO_ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "mindie_sim" / "__init__.py").write_text('"""Root."""\n', encoding="utf-8")
    (repo / "tests" / "operators" / "test_baseline_law.py").write_text(
        '"""Baseline."""\nimport pytest\npytestmark = pytest.mark.tier0\ndef test_ok():\n    assert True\n',
        encoding="utf-8",
    )
    tax = (
        'schema_version = 1\ndomains = ["operators", "product"]\nsupport_dirs = ["_support"]\n'
        'max_lines = 600\nmax_modules = 16\n\n[[module]]\npath = "operators/test_baseline_law.py"\ntier = "tier0"\n'
    )
    if family:
        tax += f'\n[[module]]\npath = "operators/test_{family}.py"\ntier = "tier0"\nowns = ["mindie_sim/prediction/operators/{family}"]\n'
    (repo / "tests" / "TAXONOMY.toml").write_text(tax, encoding="utf-8")
    return repo


def _make_upstream_git_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    up = tmp_path / "refs" / "ops-sample"
    src_file = up / "ops" / "sample_kernel.cpp"
    src_file.parent.mkdir(parents=True)
    content = b"// upstream kernel\nvoid aclnnSampleFuseV1() {}\nvoid aclnnSampleFuseV2() {}\n"
    src_file.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Tester"],
        ["git", "add", "ops/sample_kernel.cpp"],
        ["git", "commit", "-q", "-m", "init"],
    ):
        res = subprocess.run(cmd, cwd=str(up), capture_output=True, text=True, check=False)
        assert res.returncode == 0, res.stderr
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(up), capture_output=True, text=True, check=True).stdout.strip()
    return up, rev, "ops/sample_kernel.cpp", sha256


def _make_lean_bounds(case_ids: list[str], *, invert_sim: bool = False, invert_p99: bool = False) -> list[dict]:
    prog = ResourceChoiceProgram(num_resources=2, global_work=(800, 0), site_modes=(((0, 600),),))
    dag = EventDagSpec(node_quantum_ns=(600,), edges=())
    cert, _ = certify_lower_bound(prog, dag)
    return [
        {
            "case_id": cid,
            "l_star_lean_ns": cert.l_star_lean_ns,
            "t_sim_ns": 700 if invert_sim else 1000,
            "t_silicon_p99_ns": 750 if invert_p99 else 1050,
            "program": {"num_resources": 2, "global_work": [800, 0], "site_modes": [[[0, 600]]]},
            "dag": {"node_quantum_ns": [600], "edges": []},
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
        for cid in case_ids
    ]


def _sku_observations(sku: str, *, cliff_error_ratio: float = 0.03) -> list[dict]:
    rows = [
        {
            "case_id": f"{sku}:fia:anchor:{seq}", "level": "L1", "split": "train",
            "grid_kind": "anchor", "operator": "fused_infer_attention", "context_len": seq,
            "acquisition_root": "acq-train-sku-01",
            "measured_p50_ns": 10000 + seq, "predicted_ns": int(round((10000 + seq) * 1.02)),
        }
        for seq in ANCHOR_CONTEXT_GRID
    ]
    rows.extend(
        {
            "case_id": f"{sku}:fia:cliff:{seq}", "level": "L1", "split": "heldout",
            "grid_kind": "off_grid_cliff", "operator": "fused_infer_attention", "context_len": seq,
            "acquisition_root": "acq-heldout-l1-root",
            "measured_p50_ns": 12000 + seq,
            "predicted_ns": int(round((12000 + seq) * (1.0 + cliff_error_ratio))),
        }
        for seq in OFF_GRID_CLIFF_HELDOUT_GRID
    )
    for lvl, root in (("L2", "acq-heldout-l2-root"), ("L3", "acq-heldout-l3-root")):
        rows.append({
            "case_id": f"{sku}:{lvl.lower()}:cliff:24576", "level": lvl, "split": "heldout",
            "grid_kind": "off_grid_cliff", "operator": "block" if lvl == "L2" else "step",
            "context_len": 24576, "acquisition_root": root,
            "measured_p50_ns": 50000, "predicted_ns": int(round(50000 * (1.0 + cliff_error_ratio))),
        })
    return rows


def _raw_l0_samples(sku: str) -> dict:
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    die = hw.dies[0]
    dp = datapath_for_die(die)
    rates = dict(die.rates)
    aiv = max(1, int(dp.aiv_count))
    hbm_r = float(dp.hbm_read_bytes_per_s or rates["hbm"]) * 0.72
    hbm_w = float(dp.hbm_write_bytes_per_s or rates["hbm"]) * 0.70
    mte2_aiv = (float(rates["mte2_aiv"]) / aiv) * 0.75
    return {
        "device_name": sku,
        "cold_ring_l2_multiplier": 3,
        "P01_datacopy": {
            "hbm_read_bytes_per_s": [hbm_r * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "hbm_write_bytes_per_s": [hbm_w * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "mte2_aiv_hit_bytes_per_s_per_core": [mte2_aiv * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
        },
        "P20_host_dma_sync": {
            "host_h2d_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
            "host_d2h_bytes_per_s": [2.0e10 * (1.0 + 0.001 * i) for i in range(SAMPLE_REPEATS)],
        },
    }


def test_sku_probe_soc_validation_conflict_and_new_sku_scaffold(tmp_path):
    sku = REPRESENTATIVE_SKUS["a2"]
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    dp = datapath_for_die(hw.dies[0])
    bufs = dict(hw.dies[0].buffers)

    synthetic_ip = ".".join(("198", "51", "100", "22"))
    ok_probe_file = tmp_path / "probe_ok.json"
    ok_probe_file.write_text(json.dumps({
        "aclrt_soc_name": sku,
        "endpoint_alias": f"lab-{synthetic_ip}-box",
        "platform_config_ini": (
            f"[SoCInfo]\nsoc_version={sku}\nai_core_cnt={dp.aic_count}\n"
            f"vector_core_cnt={dp.aiv_count}\nl2_size={bufs['l2']}\nub_size={bufs['ub']}\n"
        ),
        "npu_smi_common": f"Chip ID : 0\nAicore Count : {dp.aic_count}\nAicore Freq(MHZ) : 1650\n",
    }), encoding="utf-8")
    res_ok = probe_soc(probe_json=ok_probe_file)
    assert res_ok["ok"] is True and res_ok["mode"] == "validated_existing"
    assert res_ok["probe"]["capture_uri"].startswith("capture://")
    assert synthetic_ip not in json.dumps(res_ok)
    assert res_ok["datapath_summary"]["buffer_units_l2"] == 1

    bad_probe_file = tmp_path / "probe_conflict.json"
    bad_probe_file.write_text(json.dumps({
        "aclrt_soc_name": sku, "aic_count": dp.aic_count + 4, "aiv_count": dp.aiv_count, "l2_bytes": bufs["l2"],
    }), encoding="utf-8")
    res_bad = probe_soc(probe_json=bad_probe_file)
    assert res_bad["ok"] is False and res_bad["error"] == "silicon_profile_conflict"
    assert any(c["field"] == "aic_count" for c in res_bad["conflicts"])

    new_probe_file = tmp_path / "probe_new.json"
    new_probe_file.write_text(json.dumps({
        "aclrt_soc_name": "Ascend950DT_9599", "arch_preset": "npu_arch_3510",
        "aic_count": 40, "aiv_count": 80, "hbm_bytes": 128 * 1024**3,
        "l2_bytes": 160 * 1024**2, "dies_per_package": 1, "ranks_per_node": 8,
        "fabric_channels": {"d2d_intra_node": 640.0},
    }), encoding="utf-8")
    scaffold_out = tmp_path / "new_sku_scaffold.json"
    res_new = probe_soc(probe_json=new_probe_file, scaffold_out=scaffold_out, allow_scaffold=True)
    assert res_new["ok"] is True and res_new["mode"] == "scaffolded_new" and scaffold_out.is_file()
    assert res_new["datapath_summary"]["aic_count"] == 40
    assert res_new["datapath_summary"]["buffer_units_l2"] == 1


def test_sku_calibrate_l0_timeout_budget_lease_idle_and_sealing(tmp_path):
    sku = REPRESENTATIVE_SKUS["a3"]
    with pytest.raises(ModelingError, match="timeout_budget_outer_too_small"):
        TimeoutBudget((("l0", 360), ("fabric", 240)), cleanup_grace_s=60, outer_timeout_s=620)

    lease_events: list[str] = []

    @contextmanager
    def _mock_lease(target_sku: str, outer_s: int):
        lease_events.append(f"enter:{target_sku}:{outer_s}")
        try:
            yield {"lease_id": "lease-001", "fence_token": 7}
        finally:
            lease_events.append("exit")

    raw_l0 = tmp_path / "raw_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(sku)), encoding="utf-8")
    out_prim = tmp_path / "primitives" / "arch22_target.json"
    out_man = tmp_path / "primitives" / "manifest.json"
    out_train = tmp_path / "train_manifest.json"
    res = calibrate_l0_and_fabric(
        sku=sku, endpoint_alias="lab-a3-ep", raw_l0_json=raw_l0,
        out_primitives=out_prim, manifest_path=out_man, train_manifest_out=out_train,
        idle_precheck={
            "npu_id": 0, "info_text": "No running processes found in NPU 0",
            "common_text": "Aicore Usage Rate(%) : 0\nHBM Usage Rate(%) : 1\nAicore Count : 20\nAicore Freq(MHZ) : 1800\n",
        },
        lease_acquirer=_mock_lease,
    )
    assert res["ok"] is True and lease_events == [f"enter:{sku}:660", "exit"] and res["calibrated_count"] >= 2
    table = parse_primitive_table(json.loads(out_prim.read_text(encoding="utf-8")), verify_envelope=True)
    assert table.cell("P01_datacopy").status == "calibrated"
    assert table.cell("P20_host_dma_sync").status == "calibrated"
    assert json.loads(out_train.read_text(encoding="utf-8"))["split"] == "train"

    with pytest.raises(ModelingError, match="onboarding_npu_busy"):
        calibrate_l0_and_fabric(
            sku=sku, raw_l0_json=raw_l0,
            idle_precheck={"npu_id": 0, "info_text": "No running processes found in NPU 0", "common_text": "Aicore Usage Rate(%) : 45\nHBM Usage Rate(%) : 5\n"},
        )
    bad_ring = tmp_path / "bad_ring.json"
    bad_ring.write_text(json.dumps({**_raw_l0_samples(sku), "cold_ring_l2_multiplier": 2}), encoding="utf-8")
    with pytest.raises(ModelingError, match="cold_ring_l2_multiplier_too_small"):
        calibrate_l0_and_fabric(sku=sku, raw_l0_json=bad_ring)

    def _timeout_runner(_sku: str, _budget: int):
        raise TimeoutError("simulated_l0_timeout")

    with pytest.raises(TimeoutError, match="simulated_l0_timeout"):
        calibrate_l0_and_fabric(sku=sku, l0_runner=_timeout_runner)


def test_sku_validate_512_400k_and_verify_gates_positive_and_fail_closed(tmp_path):
    sku = REPRESENTATIVE_SKUS["a5"]
    syn_repo = _make_clean_synthetic_repo(tmp_path)
    obs_ok = tmp_path / "obs_ok.json"
    obs_ok.write_text(json.dumps({"observations": _sku_observations(sku, cliff_error_ratio=0.04)}), encoding="utf-8")

    val_ok = validate_512_400k(sku=sku, observations_json=obs_ok, plan_out=tmp_path / "plan.json")
    assert val_ok["ok"] is True and val_ok["covered_levels"] == ["L1", "L2", "L3"]
    assert val_ok["missing_cliff_contexts"] == []

    # Reject train==heldout leakage, L1==L2 root overlap, and service_regimes crutch.
    leaked = _sku_observations(sku)
    leaked[0]["acquisition_root"] = "acq-heldout-l1-root"
    obs_leaked = tmp_path / "obs_leaked.json"
    obs_leaked.write_text(json.dumps({"observations": leaked}), encoding="utf-8")
    with pytest.raises(ModelingError, match="onboarding_lineage_split_leakage"):
        validate_512_400k(sku=sku, observations_json=obs_leaked)

    overlap = _sku_observations(sku)
    overlap[-1]["acquisition_root"] = "acq-heldout-l1-root"
    obs_overlap = tmp_path / "obs_overlap.json"
    obs_overlap.write_text(json.dumps({"observations": overlap}), encoding="utf-8")
    with pytest.raises(ModelingError, match="onboarding_cross_level_root_overlap"):
        validate_512_400k(sku=sku, observations_json=obs_overlap)

    crutch = _sku_observations(sku)
    crutch[0]["service_regimes"] = {"fixed_us": 12.0}
    obs_crutch = tmp_path / "obs_crutch.json"
    obs_crutch.write_text(json.dumps({"observations": crutch}), encoding="utf-8")
    with pytest.raises(ModelingError, match="service_regimes_crutch_forbidden"):
        validate_512_400k(sku=sku, observations_json=obs_crutch)

    bounds_ok = _make_lean_bounds([f"{sku}:fia:cliff:2049"])
    rcpt_ok = verify_sku_gates(
        sku=sku, validation_report_json=val_ok, lean_bounds_json=bounds_ok,
        repo_root=syn_repo, out_receipt=tmp_path / "sku_receipt.json",
    )
    assert rcpt_ok["ok"] is True and rcpt_ok["status"] == "certified"
    assert verify_receipt_sha256(rcpt_ok) is True and rcpt_ok["diagnosis_report"] is None

    # >= 10.0% cliff failure auto-triggers scripts/diagnose_tail_cluster.py.
    obs_tail = tmp_path / "obs_tail.json"
    obs_tail.write_text(json.dumps({"observations": _sku_observations(sku, cliff_error_ratio=0.18)}), encoding="utf-8")
    val_tail = validate_512_400k(sku=sku, observations_json=obs_tail)
    diag_out = tmp_path / "diag.json"
    rcpt_tail = verify_sku_gates(
        sku=sku, validation_report_json=val_tail, lean_bounds_json=bounds_ok,
        repo_root=syn_repo, diagnosis_out=diag_out,
    )
    assert val_tail["ok"] is False and rcpt_tail["ok"] is False and diag_out.is_file()
    assert rcpt_tail["diagnosis_report"]["schema"] == "mindie-sim.tail-cluster-diagnosis/1"
    assert rcpt_tail["diagnosis_report"]["cluster_count"] >= 1

    # Lean dual-bound inversion fails closed.
    rcpt_inv = verify_sku_gates(
        sku=sku, validation_report_json=val_ok,
        lean_bounds_json=_make_lean_bounds([f"{sku}:fia:cliff:2049"], invert_p99=True),
        repo_root=syn_repo,
    )
    assert rcpt_inv["ok"] is False
    lean_gate = next(g for g in rcpt_inv["gates"] if g["name"] == "lean_optimality_non_inversion")
    assert lean_gate["passed"] is False and "lean_silicon_p99_inversion" in lean_gate["detail"]


def test_operator_scaffold_contract_cliff_plan_and_certification(tmp_path):
    sku = REPRESENTATIVE_SKUS["a3"]
    family = "sample_fuse"
    syn_repo = _make_clean_synthetic_repo(tmp_path, family=family)
    up_repo, rev, rel_path, sha = _make_upstream_git_repo(tmp_path)
    profiler_root = tmp_path / "mock_profiler"
    bench_ops = profiler_root / "mindie_profiler" / "bench" / "operators_custom.py"
    bench_ops.parent.mkdir(parents=True)
    bench_ops.write_text('CUSTOM_BUILDERS = {"sample_fuse": lambda *_a: None}\n', encoding="utf-8")

    refs = [{"repository": "cann/ops-sample", "revision": rev, "relative_path": rel_path, "source_sha256": sha}]
    scaf = scaffold_operator(
        family=family, upstream_abi_arch22="aclnnSampleFuseV1", upstream_abi_arch35="aclnnSampleFuseV2",
        source_refs=refs, target_root=syn_repo, profiler_builder="sample_fuse",
    )
    assert scaf["ok"] is True

    target_pkg = syn_repo / "mindie_sim" / "prediction" / "operators" / family
    contract_ok = validate_operator_contract(
        target_path=target_pkg, repo_root=syn_repo,
        repo_map={"cann/ops-sample": up_repo}, profiler_root=profiler_root,
    )
    assert contract_ok["ok"] is True, contract_ok["violations"]
    assert contract_ok["has_semantic_work_item"] and contract_ok["has_tensor_range_obligation"]
    assert contract_ok["lean_rewrite_probe"]["passed"] and contract_ok["profiler_builder_verified"]

    plan = generate_operator_cliff_plan(
        operator_id="sample_fuse_op", family=family, sku=sku, tile_dim=128, core_count=20,
        l2_bytes=96 * 1024 * 1024, bytes_per_token=4096, topk_threshold=2048,
    )["plan"]
    heldout_axes = {(p["axis"], p["value"]) for p in plan["heldout_points"]}
    train_axes = {(p["axis"], p["value"]) for p in plan["train_points"]}
    assert heldout_axes.isdisjoint(train_axes)
    assert {("tile_m", 127), ("tile_m", 129), ("wave_m", 2559), ("wave_m", 2561), ("topk", 2047), ("topk", 2049)} <= heldout_axes
    assert ("tile_m", 128) in train_axes
    assert all(("context_len", c) in heldout_axes for c in OFF_GRID_CLIFF_HELDOUT_GRID)
    assert all(("context_len", c) in train_axes for c in ANCHOR_CONTEXT_GRID)

    captured = [
        {
            "case_id": pt["point_id"], "level": "L1", "split": pt["split"],
            "grid_kind": pt["grid_kind"], "operator": family,
            "context_len": pt["value"] if pt["axis"] == "context_len" else None,
            "acquisition_root": plan["train_lineage_root"] if pt["split"] == "train" else plan["heldout_lineage_root"],
            "measured_p50_ns": 20000 + pt["value"],
            "predicted_ns": int(round((20000 + pt["value"]) * 1.03)),
        }
        for pt in (*plan["train_points"], *plan["heldout_points"])
    ]
    score_ok = bench_and_score_operator(cliff_plan=plan, sku=sku, captured_json=captured)
    assert score_ok["ok"] is True
    cert_ok = certify_operator(
        operator_id="sample_fuse_op", sku=sku, contract_report=contract_ok, score_report=score_ok,
        lean_bounds_json=_make_lean_bounds(["sample_fuse_op:cliff"]), repo_root=syn_repo,
    )
    assert cert_ok["ok"] is True and cert_ok["status"] == "certified" and verify_receipt_sha256(cert_ok)

    # Negative contract cases: missing Upstream-ABI & SHA-256 mismatch fail closed.
    leaf22 = target_pkg / "arch22.py"
    orig_leaf = leaf22.read_text(encoding="utf-8")
    leaf22.write_text(orig_leaf.replace("Upstream-ABI: aclnnSampleFuseV1", "No-ABI-Header"), encoding="utf-8")
    bad_abi = validate_operator_contract(
        target_path=target_pkg, repo_root=syn_repo, repo_map={"cann/ops-sample": up_repo}, profiler_root=profiler_root
    )
    assert bad_abi["ok"] is False and any("missing_upstream_abi" in v for v in bad_abi["violations"])

    leaf22.write_text(orig_leaf.replace(sha, "0" * 64), encoding="utf-8")
    bad_sha = validate_operator_contract(
        target_path=target_pkg, repo_root=syn_repo, repo_map={"cann/ops-sample": up_repo}, profiler_root=profiler_root
    )
    assert bad_sha["ok"] is False and any("source_sha256_mismatch" in v for v in bad_sha["violations"])
    leaf22.write_text(orig_leaf, encoding="utf-8")

    run_inv = run_operator_onboarding(
        operator_id="sample_fuse_op", target_path=target_pkg, sku=sku, tile_dim=128, core_count=20,
        l2_bytes=96 * 1024 * 1024, bytes_per_token=4096, topk_threshold=2048,
        captured_json=captured, lean_bounds_json=_make_lean_bounds(["sample_fuse_op:cliff"], invert_sim=True),
        repo_root=syn_repo, repo_map={"cann/ops-sample": up_repo}, profiler_root=profiler_root,
    )
    assert run_inv["ok"] is False

    # Template target scaffolding + real check_template_provenance.py verification.
    tmpl_repo = _make_clean_synthetic_repo(tmp_path / "tmpl_syn", family="sample_tmpl")
    scaf_tmpl = scaffold_operator(
        family="sample_tmpl", upstream_abi_arch22="aclnnSampleFuseV1", upstream_abi_arch35="aclnnSampleFuseV2",
        source_refs=refs, target_root=tmpl_repo, target_kind="template", profiler_builder="sample_fuse",
    )
    assert scaf_tmpl["ok"] is True and scaf_tmpl["target_kind"] == "template"
    tmpl_contract = validate_operator_contract(
        target_path=tmpl_repo / "mindie_sim" / "templates" / "sample_tmpl_spec.py",
        repo_root=tmpl_repo, repo_map={"cann/ops-sample": up_repo}, profiler_root=profiler_root,
        require_taxonomy=False,
    )
    assert tmpl_contract["ok"] is True, tmpl_contract["violations"]
    assert tmpl_contract["template_provenance"]["passed"] is True


def test_cli_roundtrips_for_sku_and_operator_pipelines(tmp_path):
    env = _isolated_env(tmp_path)
    sku = REPRESENTATIVE_SKUS["a2"]
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    dp = datapath_for_die(hw.dies[0])
    syn_repo_sku = _make_clean_synthetic_repo(tmp_path / "sku_syn")
    syn_repo_op = _make_clean_synthetic_repo(tmp_path / "op_syn", family="cli_fuse")
    up_repo, rev, rel_path, sha = _make_upstream_git_repo(tmp_path)

    probe_file = tmp_path / "cli_probe.json"
    probe_file.write_text(json.dumps({"aclrt_soc_name": sku, "aic_count": dp.aic_count, "aiv_count": dp.aiv_count}), encoding="utf-8")
    raw_l0 = tmp_path / "cli_l0.json"
    raw_l0.write_text(json.dumps(_raw_l0_samples(sku)), encoding="utf-8")
    obs_file = tmp_path / "cli_obs.json"
    obs_file.write_text(json.dumps({"observations": _sku_observations(sku, cliff_error_ratio=0.02)}), encoding="utf-8")
    lean_file = tmp_path / "cli_lean.json"
    lean_file.write_text(json.dumps({"bounds": _make_lean_bounds([f"{sku}:fia:cliff:2049"])}), encoding="utf-8")

    proc_sku = subprocess.run(
        [
            sys.executable, "-B", "-m", "mindie_sim", "onboard-sku", "run-all",
            "--probe-json", str(probe_file), "--raw-l0-json", str(raw_l0),
            "--observations-json", str(obs_file), "--lean-bounds-json", str(lean_file),
            "--repo-root", str(syn_repo_sku), "--out-dir", str(tmp_path / "sku_run"),
        ],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False,
    )
    assert proc_sku.returncode == 0, proc_sku.stdout + proc_sku.stderr
    sku_out = json.loads(proc_sku.stdout)
    assert sku_out["ok"] is True and verify_receipt_sha256(sku_out["receipt"]) is True

    bad_probe = tmp_path / "cli_bad_probe.json"
    bad_probe.write_text(json.dumps({"aclrt_soc_name": sku, "aic_count": dp.aic_count + 8}), encoding="utf-8")
    proc_conflict = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "tools" / "onboard_sku.py"), "probe-soc", "--probe-json", str(bad_probe)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False,
    )
    assert proc_conflict.returncode == 2
    assert json.loads(proc_conflict.stdout)["error"] == "silicon_profile_conflict"

    refs_file = tmp_path / "refs.json"
    refs_file.write_text(json.dumps([
        {"repository": "cann/ops-sample", "revision": rev, "relative_path": rel_path, "source_sha256": sha}
    ]), encoding="utf-8")
    proc_scaf = subprocess.run(
        [
            sys.executable, "-B", str(REPO_ROOT / "tools" / "onboard_operator.py"), "scaffold",
            "--family", "cli_fuse", "--upstream-abi-arch22", "aclnnSampleFuseV1",
            "--upstream-abi-arch35", "aclnnSampleFuseV2", "--source-refs-json", str(refs_file),
            "--target-root", str(syn_repo_op),
        ],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False,
    )
    assert proc_scaf.returncode == 0, proc_scaf.stderr

    proc_plan = subprocess.run(
        [
            sys.executable, "-B", "-m", "mindie_sim", "onboard-operator", "generate-cliff-plan",
            "--operator-id", "cli_fuse_op", "--family", "cli_fuse", "--sku", sku,
            "--tile-dim", "128", "--core-count", str(dp.aic_count),
            "--l2-bytes", str(dict(hw.dies[0].buffers)["l2"]), "--topk-threshold", "2048",
        ],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False,
    )
    assert proc_plan.returncode == 0, proc_plan.stderr
    plan_payload = json.loads(proc_plan.stdout)
    assert plan_payload["ok"] is True and plan_payload["heldout_point_count"] > len(OFF_GRID_CLIFF_HELDOUT_GRID)

    fail_run = run_sku_onboarding(probe_json=bad_probe, out_dir=tmp_path / "sku_fail", run_repo_gates=False)
    assert fail_run["ok"] is False and verify_receipt_sha256(fail_run["receipt"]) is True
