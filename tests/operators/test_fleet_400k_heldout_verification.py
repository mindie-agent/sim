"""Verification tests for all-SKU 512..400K Anchor + Off-Grid Cliff Heldout bundle.

[Role] Lock cryptographic provenance, zero-drop multiset reconciliation, strict
  P50/P99 Lean L* non-inversion (L* > 0), observed board clock replay, DCP/PCP
  component gating, privacy hygiene, P01/P18 calibration physics, and white-box
  operator conservation across all 4 reachable Ascend SKUs plus unmeasured 950PR.
[Invariants]
  * Disjoint Anchor (512..409600) and Off-Grid Cliff Heldout (768..400000) grids
    and zero Anchor/Heldout geometry overlap across cases + blocked_cases.
  * Inclusive full float64 tolerance: |pred_ns - meas_ns| <= max(0.10 * meas_ns, 2000.0).
  * Finite > 0 guards on every duration and Lean L* bound; 0 P50/P99/sim inversions.
  * Zero privacy leaks (no IPv4, home paths, or hostnames) in bundle or producer.
[Timeouts & Locks] Tier-0 in-memory verification (<5s); read-only I/O; no locks.
[Side-Effect Policy] Read-only checks against sealed repo data and live simulator.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from mindie_sim import compose_case, graph_costs, parse_case
from mindie_sim.catalog.ops import QuantSpec, gemm_op
from mindie_sim.optimization.dual_bound import (
    EventDagSpec, ResourceChoiceProgram, certify_lower_bound,
    check_optimality_certificate, inward_floor_ns,
)
from mindie_sim.prediction.api import RecipeBook
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operators import attention, linear_attention, paged_mla, sparse_indexer
from mindie_sim.prediction.operators.attention._fia import AttentionContract
from mindie_sim.prediction.operators.attention.common import C8_V1_ABI, QUANT_MODE_INT8_C8, QUANT_MODE_TQ4, SFA_V1_ABI
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.linear_attention._gdn_native import arch35_gdn_vf_vector_ops
from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.operators.paged_mla import indexed_attention, indexed_quant_attention
from mindie_sim.prediction.operators.paged_mla.common import PagedMlaContract
from mindie_sim.prediction.operators.sharedkv.split_core import (
    _p01_active_core_bps, p01_cold_hbm_group_cap, p01_cold_mte2_aic_cap,
)
from mindie_sim.prediction.operators.sparse_indexer.common import SparseIndexerContract
from mindie_sim.prediction.operators.vector.ring_update import build_ring_update, make_ring_update_op
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
from mindie_sim.prediction.streams import StreamOwners
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.primitives.schema import compute_envelope_sha256, parse_primitive_table
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from tools._fleet_400k_reconcile import (
    apply_sku_observed_clock, canonical_capture_sha256, load_verified_capture, score_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.tier0
BUNDLE_PATH = REPO_ROOT / "data" / "verification" / "all-skus-512-400k-cliff-heldout-20261121.json"
PRODUCER_PATH = REPO_ROOT / "data" / "verification" / "all-skus-512-400k-cliff-heldout-20261121.producer.py"
RECONCILE_PATH = REPO_ROOT / "tools" / "_fleet_400k_reconcile.py"
MANIFEST_PATH = REPO_ROOT / "data" / "primitives" / "manifest.json"

REACHABLE_SKUS = ("Ascend910B4", "Ascend910_9362", "Ascend910_9382", "Ascend950DT_9582")
EXPECTED_CLOCKS_MHZ = {"Ascend910B4": 1650.0, "Ascend910_9362": 1650.0, "Ascend910_9382": 1800.0, "Ascend950DT_9582": 1650.0}
EXPECTED_SEALED_TOTAL = 1768
EXPECTED_WORKER_DIAGS = 12
ANCHOR_GRID = [512, 2048, 8192, 32768, 131072, 262144, 409600]
HELDOUT_GRID = [768, 2049, 4097, 24576, 98304, 200000, 350000, 400000]
REL_TOL, ABS_FLOOR_NS, NS_PER_US, US_PER_MS, LSTAR_INV_EPS = 0.10, 2000.0, 1000.0, 1000.0, 1e-6
A5_C8_HELDOUT_400K_CANN93_US = 50.155851511893545
PRIVACY_PATTERNS = (
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("home_mac", re.compile("/" + "Users/[A-Za-z0-9_.-]+")),
    ("home_linux", re.compile("/" + "home/[A-Za-z0-9_.-]+")),
    ("home_root", re.compile("/" + "root/[A-Za-z0-9_.-]+")),
    ("user_at_host", re.compile(r"[^\s\"'@]+@[A-Za-z0-9][A-Za-z0-9._-]+")),
)


@pytest.fixture(scope="module")
def sealed_bundle() -> dict:
    assert BUNDLE_PATH.is_file(), f"missing sealed bundle: {BUNDLE_PATH}"
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def _ctx(cs, op, *, requests=(), stream_owners=None, memory_view=None) -> OperatorPlanningContext:
    return OperatorPlanningContext(
        occurrence=OpOccurrence(replica=0, stage=0, rank=0, op_id=op.op_id),
        geometry=Geometry.from_devices(cs.machine.dies),
        datapath=datapath_for_die(cs.machine.dies[0]),
        requests=tuple(requests),
        stream_owners=stream_owners,
        admission_memory_view=memory_view,
    )


def test_bundle_provenance_clocks_inner_digests_and_zero_privacy_leaks(sealed_bundle: dict, tmp_path: Path) -> None:
    assert sealed_bundle["schema_version"] == "mindie-sim-fleet-400k-verification/1"
    recorded_sha = sealed_bundle["bundle_sha256"]
    body = {k: v for k, v in sealed_bundle.items() if k != "bundle_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == recorded_sha

    manifest_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    recorded_manifest_sha = sealed_bundle["provenance"]["primitive_manifest_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", recorded_manifest_sha)
    manifest_tables = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["tables"]
    all_same_cann = True
    for sku in REACHABLE_SKUS:
        entry = sealed_bundle["skus"][sku]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["primitive_envelope_sha256"])
        if entry["cann_version"] == manifest_tables[sku]["cann_version"]:
            assert entry["primitive_envelope_sha256"] == manifest_tables[sku]["envelope_sha256"]
        else:
            all_same_cann = False
            assert manifest_tables[sku]["cann_version"] > entry["cann_version"]
        assert entry["observed_aicore_freq_mhz"] == EXPECTED_CLOCKS_MHZ[sku]
        assert all(cap.get("inner_digest_verified") is True for cap in entry["captures"])
    if all_same_cann:
        assert recorded_manifest_sha == manifest_sha

    # #3: Inner raw_sha256 verifier fails closed on tampered capture payload.
    cap_f = tmp_path / "tampered.json"
    doc = {"sku": "Ascend910B4", "records": []}
    doc["raw_sha256"] = canonical_capture_sha256(doc)
    cap_f.write_text(json.dumps(doc), encoding="utf-8")
    assert load_verified_capture(cap_f)[0]["sku"] == "Ascend910B4"
    doc["records"] = [{"tampered": True}]
    cap_f.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="capture_inner_digest_mismatch"):
        load_verified_capture(cap_f)

    grids = sealed_bundle["context_grids"]
    assert grids["anchor"] == ANCHOR_GRID and grids["off_grid_cliff_heldout"] == HELDOUT_GRID
    assert grids["overlap_count"] == 0 and not (set(grids["anchor"]) & set(grids["off_grid_cliff_heldout"]))

    inv = sealed_bundle["fleet_inventory"]
    assert [row["sku"] for row in inv["reachable_calibrated_skus"]] == list(REACHABLE_SKUS)
    assert [row["sku"] for row in inv["unmeasured_skus"]] == ["Ascend950PR_9579"]
    assert inv["unmeasured_skus"][0]["status"] == "unmeasured"

    for artifact in (BUNDLE_PATH, PRODUCER_PATH, RECONCILE_PATH):
        text = artifact.read_text(encoding="utf-8")
        for label, pat in PRIVACY_PATTERNS:
            match = pat.search(text)
            assert match is None, f"privacy leak ({label}) in {artifact.name}: {match.group(0)}"


def test_zero_drop_multiset_reconciliation_g4_warm_ragged_dcp_and_strict_lstar(sealed_bundle: dict) -> None:
    # #6: Reject P50 < L* <= P99 and reject L* == 0.
    v_inv = score_verdict("c_inv", "fam", "heldout", 100.0, 120.0, 105.0, 105.0, "capture://x")
    assert v_inv["status"] == "fail" and v_inv["inv_obs"] is True
    v_zero = score_verdict("c_z", "fam", "heldout", 100.0, 120.0, 100.0, 0.0, "capture://x")
    assert v_zero["status"] == "fail" and v_zero["inv_sim"] is True and v_zero["inv_obs"] is True

    overall = sealed_bundle["overall_counts"]
    assert overall["total"] == overall["pass"] + overall["fail"] + overall["blocked"] == EXPECTED_SEALED_TOTAL
    assert overall["l1_cliff_pass"] + overall["l1_cliff_fail"] == overall["pass"] + overall["fail"]
    assert overall["l1_cliff_pass"] >= overall["pass"]
    cq = sealed_bundle["capture_quality_summary"]
    assert cq["clean_scored"] + cq["flagged_rows"] == overall["pass"] + overall["fail"]
    t2 = sealed_bundle["acceptance_policy"]["l1_two_tier_policy"]
    assert (t2["median_ape_max_pct"], t2["p90_ape_max_pct"], t2["cliff_relative_tolerance"], t2["cliff_absolute_floor_ns"]) == (8.0, 12.0, 0.15, 3000.0)
    assert overall["worker_diagnostics"] == EXPECTED_WORKER_DIAGS
    assert overall["inv_sim"] == 0 and overall["inv_obs"] == 0

    sum_total = sum_pass = sum_fail = sum_blk = sum_diag = 0
    for sku in REACHABLE_SKUS:
        entry = sealed_bundle["skus"][sku]
        counts, cases, blocked, diags = entry["counts"], entry["cases"], entry["blocked_cases"], entry["worker_diagnostics"]
        rec = entry["reconciliation"]
        assert rec["expected_records"] == rec["reconciled_records"] == len(cases) + len(blocked) + len(diags)
        assert rec["anchor_heldout_geometry_overlap"] == 0
        assert counts["total"] == counts["pass"] + counts["fail"] + counts["blocked"]
        assert counts["inv_sim"] == 0 and counts["inv_obs"] == 0
        # #1: Both ragged acquisitions retained (18/SKU) and Gate 4 warm + cold both scored (16+16/SKU).
        assert sum(1 for r in cases if r["split"] == "ragged") == 18
        assert sum(1 for r in cases if r["family"] == "g4_moe_gmm_cold") == 16
        assert sum(1 for r in cases if r["family"] == "g4_moe_gmm_warm") == 16
        # #2: No *_ABORT records in blocked_cases; anchor/heldout strictly geometry-disjoint.
        assert not any(b["case_id"].endswith(("_ABORT", "_TIMEOUT")) for b in blocked)
        anc_g = {r["geometry_key"] for r in (*cases, *blocked) if r.get("split") == "anchor" and r.get("geometry_key")}
        hld_g = {r["geometry_key"] for r in (*cases, *blocked) if r.get("split") == "heldout" and r.get("geometry_key")}
        assert not (anc_g & hld_g)
        # #4: DCP/PCP total never passes if inner component failed; L>32768 unexecuted blocked.
        by_cid = {r["case_id"]: r for r in cases}
        for row in cases:
            if row["family"].startswith("dcp_pcp_") and row.get("boundary") == "total":
                inner_ok = all(by_cid[f"{row['case_id']}_{suf}"]["status"] == "pass" for suf in ("fia", "comm", "merge"))
                if not inner_ok:
                    assert row["status"] == "fail" and row.get("inner_component_failures")
        sum_total += counts["total"]
        sum_pass += counts["pass"]
        sum_fail += counts["fail"]
        sum_blk += counts["blocked"]
        sum_diag += len(diags)

        for row in cases:
            meas_us, meas_p99, pred_us, lstar_us = float(row["meas_us"]), float(row["meas_p99_us"]), float(row["pred_us"]), float(row["lstar_us"])
            assert math.isfinite(meas_us) and meas_us > 0.0 and meas_p99 + 1e-9 >= meas_us
            assert math.isfinite(pred_us) and pred_us > 0.0 and math.isfinite(lstar_us) and lstar_us > 0.0
            assert row["inv_sim"] is False and row["inv_obs"] is False
            assert lstar_us <= pred_us * (1.0 + LSTAR_INV_EPS) and lstar_us <= meas_us * (1.0 + LSTAR_INV_EPS)

    assert (sum_total, sum_pass, sum_fail, sum_blk, sum_diag) == (
        overall["total"], overall["pass"], overall["fail"], overall["blocked"], overall["worker_diagnostics"],
    )
    # #6: B4 KDA chunk L=400K counts full captured cold inputs (>= 1024 MB -> >= 1280 us floor).
    kda_b4 = next(r for r in sealed_bundle["skus"]["Ascend910B4"]["cases"] if r["case_id"] == "kda_chunk_heldout_L400000")
    assert kda_b4["lstar_us"] >= 1280.0
    # Verify Lean OptimalityCertificate on 9362 2R HCCS_SW 256 MiB all-reduce (196 GB/s channel)
    ns_comm = inward_floor_ns(Fraction(268_435_456) * Fraction(1_000_000_000) / Fraction(196_000_000_000))
    prog = ResourceChoiceProgram(1, (ns_comm,), ())
    dag = EventDagSpec((0,), ())
    cert, _ = certify_lower_bound(prog, dag)
    assert check_optimality_certificate(prog, dag, cert)
    hccs_row = next(r for r in sealed_bundle["skus"]["Ascend910_9362"]["cases"] if r["case_id"] == "g5_HCCS_SW_r2_all_reduce_268435456B")
    assert math.isclose(hccs_row["lstar_us"], cert.l_star_lean_ns / NS_PER_US, rel_tol=1e-6)


def test_closed_off_grid_cliff_heldout_and_live_replay_parity(sealed_bundle: dict) -> None:
    expected_closed_med: dict[str, tuple[tuple[str, str], ...]] = {
        "Ascend910B4": (
            ("heldout", "fia_gqa_decode"), ("heldout", "fia_gqa_prefill"), ("heldout", "c8_sfa_decode"),
            ("heldout", "sfa_dsa_decode"), ("gate3", "g3_chain"), ("gate4", "g4_moe_gmm_warm"), ("gate5", "g5_HCCS_MESH_steady"),
        ),
        "Ascend910_9362": (
            ("heldout", "fia_gqa_decode"), ("heldout", "fia_gqa_prefill"), ("heldout", "sfa_dsa_decode"),
            ("gate3", "g3_chain"), ("gate4", "g4_moe_gmm_cold"),
        ),
        "Ascend910_9382": (
            ("heldout", "fia_gqa_prefill"), ("heldout", "c8_sfa_decode"), ("heldout", "gdn_chunk"),
            ("gate3", "g3_chain"), ("gate4", "g4_moe_gmm_cold"),
        ),
        "Ascend950DT_9582": (
            ("heldout", "fia_gqa_prefill"), ("heldout", "sfa_dsa_decode"), ("heldout", "c8_sfa_decode"),
            ("heldout", "lightning_indexer_topk"), ("gate3", "g3_chain"),
        ),
    }
    for sku, pairs in expected_closed_med.items():
        summary = sealed_bundle["skus"][sku]["family_summary"]
        for split, family in pairs:
            stats = summary[split][family]
            assert stats["med_ape_pct"] < 10.0, f"{sku} {split}/{family} med={stats['med_ape_pct']}%"
            assert stats["inv_sim"] == 0 and stats["inv_obs"] == 0
            assert "p90_ape_pct" in stats and isinstance(stats["l1_two_tier_gate_pass"], bool)
            assert stats["l1_cliff_pass"] + stats["l1_cliff_fail"] == stats["count"]

    def _case_row(sku: str, case_id: str) -> dict:
        return next(r for r in sealed_bundle["skus"][sku]["cases"] if r["case_id"] == case_id)

    def _run_one(sku: str, op: OpSpec, rb: RecipeBook | None = None) -> float:
        raw = compose_case("qwen3.8-27b", hardware=sku, tp=1, ep=1, num_devices=1, layers=1, quant="none")
        cs = parse_case(apply_sku_observed_clock(raw, EXPECTED_CLOCKS_MHZ[sku]))
        pt = load_primitives(sku)
        ev = EvaluationContext.create(cs, rb or RecipeBook(), primitive_table=pt)
        rep = graph_costs(cs, {0: [op]}, recipes=rb, evaluation=ev, include_submission=True, isolated=True, compact=True, fast_span_only=True)
        return float(rep["duration_ms"]) * US_PER_MS

    # Live replay parity at observed board clocks across all 4 SKUs (L=400,000 heldout cliff)
    pt_b4 = load_primitives("Ascend910B4")
    geom_b4 = dict(query=1, context=400000, topk=2048, heads=128, latent=512, rope=64, query_dtype="bf16",
                   cache_dtype="bf16", window=0, compression_ratio=1, cp=1, split_kv=not pt_b4.cann_version.startswith("9.2"))
    op_b4 = OpSpec("sfadsa", "attention", "decode", 0, 0, (), (1, 128, 512), (1, 128, 512), (1, 128, 512), (1, 128, 512),
                   recipe="indexed_attention", dtype="bf16", metadata=(("indexed_attention_geometry", geom_b4),))
    assert math.isclose(
        _run_one("Ascend910B4", op_b4, RecipeBook(recipes={op_b4.recipe: PlannedRecipe(lambda o, g: indexed_attention.build(o, g, scratch_policy="offchip"))})),
        _case_row("Ascend910B4", "sfa_dsa_decode_heldout_L400000")["pred_us"], rel_tol=1e-5, abs_tol=1e-3,
    )
    op_9362 = sparse_indexer.make_indexer_op(SparseIndexerContract(query_tokens=1, scanned_rows=400000, index_heads=64, index_dim=128, topk_items=2048, batch_size=1, native_split_core=True), op_id="lidx")
    assert math.isclose(
        _run_one("Ascend910_9362", op_9362, RecipeBook(recipes={op_9362.recipe: PlannedRecipe(sparse_indexer.build)})),
        _case_row("Ascend910_9362", "lightning_indexer_topk_heldout_L400000")["pred_us"], rel_tol=1e-5, abs_tol=1e-3,
    )
    op_9382 = OpSpec("c8sfa", "attention", "decode", 0, 0, (), (1, 128, 512), (1, 128, 512), (1, 128, 512), (1, 128, 512),
                     recipe="indexed_quant_attention", dtype="bf16",
                     metadata=(("indexed_attention_geometry", dict(query=1, context=400000, topk=2048, heads=128, latent=512, rope=64, query_dtype="bf16", cache_dtype="int8", window=0, compression_ratio=1, cp=1)),))
    assert math.isclose(
        _run_one("Ascend910_9382", op_9382, RecipeBook(recipes={op_9382.recipe: PlannedRecipe(lambda o, g: indexed_quant_attention.build(o, g, scratch_policy="offchip"))})),
        _case_row("Ascend910_9382", "c8_sfa_decode_heldout_L400000")["pred_us"], rel_tol=1e-5, abs_tol=1e-3,
    )
    op_a5_c8 = attention.make_attention_op(AttentionContract(
        query_tokens=1, num_heads=128, num_kv_heads=1, head_dim=576, v_head_dim=512, selected_tokens=2048,
        phase="decode", abi=C8_V1_ABI, kv_layout="TND", quant_mode=QUANT_MODE_INT8_C8, batch_size=1, sequences=((1, 400000),),
    ), op_id="c8sfa")
    a5_row = _case_row("Ascend950DT_9582", "c8_sfa_decode_heldout_L400000")
    a5_live = _run_one("Ascend950DT_9582", op_a5_c8, RecipeBook(recipes={op_a5_c8.recipe: PlannedRecipe(attention.build)}))
    a5_same_cann = sealed_bundle["skus"]["Ascend950DT_9582"]["cann_version"] == load_primitives("Ascend950DT_9582").cann_version
    expected_a5_us = a5_row["pred_us"] if a5_same_cann else A5_C8_HELDOUT_400K_CANN93_US
    assert math.isclose(a5_live, expected_a5_us, rel_tol=1e-5, abs_tol=1e-3)
    assert abs(a5_live - a5_row["meas_us"]) <= abs(a5_row["pred_us"] - a5_row["meas_us"])
    assert abs(a5_live - a5_row["meas_us"]) <= max(REL_TOL * a5_row["meas_us"], ABS_FLOOR_NS / NS_PER_US)
    # Producer gdn_recurrent evaluator wiring returns positive non-inverting predictions on Arch22 & Arch35.
    import importlib.util
    spec = importlib.util.spec_from_file_location("_fleet_producer", PRODUCER_PATH)
    assert spec and spec.loader
    pmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmod)
    for gdn_sku in ("Ascend910_9362", "Ascend950DT_9582"):
        harness = pmod._SkuReplayHarness(gdn_sku, observed_clock_mhz=EXPECTED_CLOCKS_MHZ[gdn_sku])
        pred_gdn, lst_gdn = pmod._eval_single_seq_record(harness, "gdn_recurrent", {"B": 2, "H": 16}, harness.pt.cann_version)
        assert pred_gdn is not None and math.isfinite(pred_gdn) and pred_gdn > 0.0
        assert math.isfinite(lst_gdn) and 0.0 < lst_gdn <= pred_gdn * (1.0 + LSTAR_INV_EPS)


def test_whitebox_operator_physics_conservation_regressions_13_to_21() -> None:
    """Lock fixes #13-#21 across SteadyMatmul, FIA, PagedMLA, Indexer, SFA, GDN, KDA, and RingUpdate."""
    c9362 = parse_case(compose_case("qwen3.8-27b", hardware="Ascend910_9362", tp=1, ep=1, num_devices=1, layers=1, quant="none"))
    c9582 = parse_case(compose_case("qwen3.8-27b", hardware="Ascend950DT_9582", tp=1, ep=1, num_devices=1, layers=1, quant="none"))

    # #13: Cold BF16 64^3 (16^3 tiles, reuse disabled) services full 73,728 B HBM refills.
    op_mm = gemm_op(op_id="mm64", deps=(), stage=0, layer=0, requests=("r0",), m=64, n=64, k=64,
                    quant=QuantSpec(name="bf16"), category="gemm", phase="prefill", recipe="mm64_r")
    o_det = operator_costs(c9362, (op_mm,), recipes=RecipeBook(recipes={"mm64_r": MappedMatmul(candidates=((16, 16, 16),), compute_m=16, compute_n=16, reuse_inputs=False, reuse_weights=False)}))["operators"][0]
    o_std = operator_costs(c9362, (op_mm,), recipes=RecipeBook(recipes={"mm64_r": SteadyMatmul(candidates=((16, 16, 16),), compute_m=16, compute_n=16, reuse_inputs=False, reuse_weights=False)}))["operators"][0]
    assert o_std["serviced"]["memory@r0:s0"] == pytest.approx(73728.0) and o_std["serviced"]["memory@r0:s0"] == pytest.approx(o_det["serviced"]["memory@r0:s0"])

    # #14: Causal FIA never credits first-touch KV as L2 hits (Q=K=1024, H=1, D=128 >= 524,288 B compulsory).
    op_fia = attention.make_attention_op(AttentionContract(
        query_tokens=1024, num_heads=1, num_kv_heads=1, head_dim=128, v_head_dim=128,
        sequences=((1024, 1024),), phase="prefill", causal=True, kv_layout="BNSD", kv_tensor_id="kv0", batch_size=1,
    ), op_id="fia_causal")
    for cs in (c9362, c9582):
        v_fia = verify_operator(cs, op_fia, attention.build)
        assert v_fia["status"] == "passed" and v_fia["result"]["serviced"]["memory@r0:s0"] >= 524288.0

    # #15: RequestView preserves Arch35 PagedMLA FlashDecode splits (zero 8.7x collapse).
    op_mla = paged_mla.make_op(PagedMlaContract(sequences=((1, 8192),), logical_heads=128, physical_heads=128, latent_dim=512, rope_dim=64, page_size=128), op_id="mla_8k")
    rv_mla = RequestView(request_id="r0", q_len=1, context=8192, samples=True, phase="decode")
    v_flat = verify_operator(c9582, op_mla, paged_mla.build)
    v_req = verify_operator(c9582, op_mla, lambda o, _g: paged_mla.build(o, _ctx(c9582, o, requests=(rv_mla,))))
    assert v_flat["status"] == "passed" and v_req["status"] == "passed"
    assert abs(v_flat["result"]["duration_ms"] - v_req["result"]["duration_ms"]) / v_flat["result"]["duration_ms"] < 0.05

    # #16: SparseIndexer consumes canonical RequestView kv_lens (67.1M vs 134.2M) and charges 128 independent queries.
    op_idx1 = sparse_indexer.make_indexer_op(SparseIndexerContract(query_tokens=1, scanned_rows=8192, index_heads=64, index_dim=128, topk_items=2048, batch_size=1, native_split_core=True), op_id="idx1")
    rv_idx = RequestView(request_id="r0", q_len=1, context=4096, samples=True, phase="decode")
    assert sparse_indexer.build(op_idx1, _ctx(c9362, op_idx1, requests=(rv_idx,))).work.matrix["bf16"] == 67125248
    op_idx128 = sparse_indexer.make_indexer_op(SparseIndexerContract(query_tokens=128, scanned_rows=4096, index_heads=64, index_dim=128, topk_items=2048, batch_size=128, native_split_core=True), op_id="idx128")
    v_idx128 = verify_operator(c9362, op_idx128, sparse_indexer.build)
    assert v_idx128["status"] == "passed" and v_idx128["result"]["serviced"]["mte2_aic@r0:s0"] >= 134217728 + 2097152

    # #17: Arch22 split-KV SFA emits residual PV MMAD (838,860,800 BF16 FLOPs).
    op_sfa22 = OpSpec("sfa22", "attention", "decode", 0, 0, (), (1, 128, 512), (1, 128, 512), (1, 128, 512), (1, 128, 512),
                      recipe="indexed_attention", dtype="bf16",
                      metadata=(("indexed_attention_geometry", dict(query=1, context=32768, topk=2048, heads=128, latent=512, rope=64, query_dtype="bf16", cache_dtype="bf16", window=0, compression_ratio=1, cp=1, split_kv=True)),))
    assert indexed_attention.build(op_sfa22, Geometry.from_devices(c9362.machine.dies), scratch_policy="offchip").work.matrix == {"bf16": 838860800}

    # #18: Arch35 SFA rejects TQ4, binds selection ReuseKey L2 hits, guards Split-G <2 AICs, and conserves 1-AIC non-Split-G stream.
    op_tq4 = attention.make_kv_quant_sfa_c8_op(op_id="c8_tq4", query_tokens=1, num_heads=128, rope_dim=64, quant_mode=QUANT_MODE_TQ4, selected_tokens=2048)
    with pytest.raises(UnmodeledCapabilityError, match="attention_arch35_tq4_unsupported"):
        attention.build(op_tq4, Geometry.from_devices(c9582.machine.dies))
    sels = (tuple(range(2048)), tuple(range(2048)))
    op_sfa_a5 = attention.make_attention_op(AttentionContract(
        query_tokens=2, num_heads=128, num_kv_heads=1, head_dim=576, v_head_dim=512,
        sequences=((1, 4096), (1, 4096)), phase="decode", abi=SFA_V1_ABI, selected_tokens=2048,
        batch_size=2, kv_tensor_id="kv_sfa", selection_lists=sels,
    ), op_id="sfa_reuse")
    assert verify_operator(c9582, op_sfa_a5, attention.build)["status"] == "passed"
    assert (attention.build(op_sfa_a5, _ctx(c9582, op_sfa_a5, memory_view=AdmissionMemoryView({0: 128 * 1024 * 1024}))).work.resources["hbm"]
            < attention.build(op_sfa_a5, _ctx(c9582, op_sfa_a5, memory_view=AdmissionMemoryView({0: 0}))).work.resources["hbm"])
    so1 = StreamOwners(aic=(0,), aiv=(0, 1), aic_limit=1, aiv_limit=2)
    with pytest.raises(UnmodeledCapabilityError, match="attention_sfa_split_g_insufficient_aics"):
        attention.build(op_sfa_a5, _ctx(c9582, op_sfa_a5, stream_owners=so1))
    op_sfa_a5_h64 = attention.make_attention_op(AttentionContract(
        query_tokens=2, num_heads=64, num_kv_heads=1, head_dim=576, v_head_dim=512,
        sequences=((1, 4096), (1, 4096)), phase="decode", abi=SFA_V1_ABI, selected_tokens=2048,
        batch_size=2, kv_tensor_id="kv_sfa",
    ), op_id="sfa_h64_1aic")
    assert verify_operator(c9582, op_sfa_a5_h64, lambda o, _g: attention.build(o, _ctx(c9582, o, stream_owners=so1)))["status"] == "passed"

    # #19: Arch35 GDN 65th-token tail panel = 233,472 ops; recurrent state read+write unconditional.
    assert arch35_gdn_vf_vector_ops(65, 128, 128)[1] == 233472
    op_gdn_r = OpSpec("gdn_r", "gdn_recurrent", "decode", 0, 0, (), (1, 16, 128), (1, 16, 128), (1, 16, 128), (1, 16, 128),
                      recipe="gdn_recurrent", dtype="bf16", layout="TND",
                      metadata=(("native_kind", "aclnnRecurrentGatedDeltaRule"), ("abi", "aclnnRecurrentGatedDeltaRule"),
                                ("operation", "gdn_recurrent"), ("lengths", (1,)), ("hq", 16), ("hk", 16), ("hv", 16),
                                ("dk", 128), ("dv", 128), ("dtype", "bf16"), ("state_dtype", "bf16"),
                                ("use_initial_state", False), ("output_final_state", False)))
    v_gdn_r = verify_operator(c9582, op_gdn_r, linear_attention.build)
    assert v_gdn_r["status"] == "passed" and v_gdn_r["result"]["serviced"]["memory@r0:s0"] >= 1048576.0

    # #20: Triton KDA prep FP32 FLOPs (131,072 for seq=1, 917,504 for seq=64) and Arch35 rejection.
    for seq, expected_fp32 in ((1, 131072), (64, 917504)):
        op_k = OpSpec(f"kda_{seq}", "kda_prefill", "prefill", 0, 0, (), (seq, 1, 128), (seq, 1, 128), (seq, 1, 128), (seq, 1, 128),
                      recipe="kda_prefill", dtype="bf16", layout="TND",
                      metadata=(("native_kind", "vllm_ascend.ops.triton.kda.kda.chunk_kda"), ("abi", "vllm_ascend.ops.triton.kda.kda.chunk_kda"),
                                ("operation", "kda_prefill"), ("lengths", (seq,)), ("hq", 1), ("hk", 1), ("hv", 1),
                                ("dk", 128), ("dv", 128), ("bt", 64), ("dtype", "bf16"), ("state_dtype", "fp32")))
        pl_k = linear_attention.build(op_k, Geometry.from_devices(c9362.machine.dies))
        prep_fp32 = next(w.amount for p in pl_k.pipeline_spec.phases if p.phase_id == "triton_kda_prepare_alu"
                         for w in p.work if w.resource == "cube" and w.dtype == "fp32")
        assert prep_fp32 == expected_fp32
        with pytest.raises(UnmodeledCapabilityError):
            linear_attention.build(op_k, Geometry.from_devices(c9582.machine.dies))

    # #21: RingUpdate with all L2 reserved writes full output to HBM (zero L2 absorption).
    op_ru = make_ring_update_op(1024, 32, 128, op_id="ru")
    v_ru = verify_operator(c9362, op_ru, lambda o, _g: build_ring_update(o, _ctx(c9362, o, memory_view=AdmissionMemoryView({0: 0}))))
    assert v_ru["status"] == "passed" and v_ru["result"]["serviced"].get("l2@r0:s0", 0.0) == 0.0
    assert v_ru["result"]["serviced"]["memory@r0:s0"] >= 31457280.0 - LSTAR_INV_EPS


def test_active_core_calibration_and_scoped_p18_topology() -> None:
    """Lock P01 capacity/conflict guards and scoped P18 collective cap."""
    for sku in REACHABLE_SKUS:
        pt = load_primitives(sku)
        ctx = SimpleNamespace(
            primitive_table=pt,
            resource_names={"mte2_aic": "mte2_aic@r0:s0", "hbm": "memory@r0:s0", "l2": "l2@r0:s0"},
            rates={"memory@r0:s0": 4_000_000_000.0},
        )
        group_cap = p01_cold_hbm_group_cap(ctx, 4, include_l2=True)
        assert group_cap.get("memory@r0:s0", 0.0) > 0.0 and group_cap.get("l2@r0:s0", 0.0) > 0.0
        owner_cap = p01_cold_mte2_aic_cap(ctx, 8, include_hbm=True)
        assert owner_cap.get("mte2_aic@r0:s0", 0.0) > 0.0 and owner_cap.get("memory@r0:s0", 0.0) > 0.0
        cs4 = parse_case(compose_case("qwen3.8-27b", hardware=sku, tp=4, ep=1, num_devices=4, layers=1, quant="none"))
        topo_nom = Topology(cs4.machine, cs4.runtime.parallel)
        topo_cal = Topology(cs4.machine, cs4.runtime.parallel, primitive_table=pt)
        p18_ms = pt.cell("P18_hccl_collective_band").metrics["link_bytes_per_s"].p50 / 1000.0
        target_scope = "package" if int(getattr(cs4.machine, "dies_per_package", 1)) >= 2 else "node"
        for name, lk_cal in topo_cal.links.items():
            lk_nom, sc = topo_nom.links[name], topo_cal.link_scope(name)
            if sc == target_scope:
                assert lk_cal.capacity_bytes_per_ms <= min(p18_ms, lk_nom.capacity_bytes_per_ms) + 1e-6
            elif sc in {"package", "node"}:
                assert math.isclose(lk_cal.capacity_bytes_per_ms, lk_nom.capacity_bytes_per_ms, rel_tol=1e-9)

    raw_b4 = json.loads((REPO_ROOT / "data" / "primitives" / "arch22_a2_910b4.json").read_text(encoding="utf-8"))
    raw_b4["primitives"]["P01_datacopy"]["parameters"]["mte2_aic_cold_scaling_bps"] = {"4": 9e12}
    raw_b4["envelope_sha256"] = compute_envelope_sha256(raw_b4)
    assert _p01_active_core_bps(SimpleNamespace(primitive_table=parse_primitive_table(raw_b4)), 4) is None


def test_nearest_rank_p90_unrounded_two_tier_and_kda_sfu_conservation() -> None:
    """Lock nearest-rank P90, unrounded threshold comparison, and KDA gate exp() BK SFU count."""
    import importlib.util
    from tools._fleet_400k_reconcile import nearest_rank_p90

    assert nearest_rank_p90([1.0, 2.0, 3.0, 4.0, 14.56]) == pytest.approx(14.56)
    assert nearest_rank_p90([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 29.69]) == pytest.approx(29.69)

    spec = importlib.util.spec_from_file_location("fleet_400k_producer_check", PRODUCER_PATH)
    assert spec is not None and spec.loader is not None
    prod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prod)
    # 8.004% rounds to 8.00% at 2 decimals, yet MUST fail the <= 8.0% median gate unrounded.
    borderline = [{
        "split": "heldout", "family": "border", "case_id": f"c{i}", "status": "pass",
        "l1_cliff_status": "pass", "meas_us": 1000.0, "pred_us": 1080.04, "ape_pct": 8.004,
        "inv_sim": False, "inv_obs": False,
    } for i in range(5)]
    summary = prod._summarize_families(borderline)["heldout"]["border"]
    assert summary["med_ape_pct"] == 8.0 and summary["l1_two_tier_gate_pass"] is False

    c9362 = parse_case(compose_case("qwen3.8-27b", hardware="Ascend910_9362", tp=1, ep=1, num_devices=1, layers=1, quant="none"))
    op_rec = OpSpec(
        "kda_rec_sfu", "kda_recurrent", "decode", 0, 0, (), (2, 8, 128), (2, 8, 128), (2, 8, 128), (2, 8, 128),
        recipe="kda_recurrent", dtype="bf16",
        metadata=(("native_kind", "vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda"),
                  ("abi", "vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda"),
                  ("operation", "kda_recurrent"), ("lengths", (1, 1)), ("hq", 8), ("hk", 8), ("hv", 8),
                  ("dk", 128), ("dv", 128), ("dtype", "bf16"), ("state_dtype", "fp32"),
                  ("use_qk_l2norm_in_kernel", False)),
    )
    pl_rec = linear_attention.build(op_rec, Geometry.from_devices(c9362.machine.dies))
    rc_rec = SimpleNamespace(
        bind=lambda w: dict(w.resources), resource_names={"vector": "vector", "hbm": "hbm", "mte2_aiv": "mte2_aiv", "mte3": "mte3", "l2": "l2"},
        rates={"vector": 1e12}, resource_units={}, primitive_table=load_primitives("Ascend910_9362"), service_rate=lambda *_a, **_k: None,
        physical_role=lambda r: r, read_pipe=lambda aiv=False: "mte2_aiv" if aiv else "mte2_aic",
    )
    prog = pl_rec.program(rc_rec)
    def _sum_sfu(p) -> float:
        if hasattr(p, "tile") and hasattr(p, "count"):
            return sum(_sum_sfu(p.tile(i)) for i in range(p.count))
        tot = 0.0
        for st in (p.stages if hasattr(p, "stages") else p):
            if st.program is not None:
                tot += _sum_sfu(st.program)
            elif "rec_decay_sfu" in st.name:
                tot += float(st.work.get("vector", 0.0))
        return tot
    assert _sum_sfu(prog) == pytest.approx(2 * 8 * 16 * 128)
