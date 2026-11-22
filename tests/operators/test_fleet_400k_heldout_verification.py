"""Verification tests for all-SKU 512..400K Anchor + Off-Grid Cliff Heldout bundle.

[Role] Lock cryptographic provenance, zero-drop accounting, Lean L* non-inversion,
  privacy hygiene, P01 active-core & P18 collective calibration physics, and live
  white-box replay parity across all 4 reachable Ascend SKUs plus unmeasured 950PR.
[Invariants]
  * Disjoint Anchor (512..409600) and Off-Grid Cliff Heldout (768..400000) grids.
  * Inclusive full float64 tolerance: |pred_ns - meas_ns| <= max(0.10 * meas_ns, 2000.0).
  * Finite > 0 guards on every measured and predicted duration; 0 Lean L* inversions.
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
from mindie_sim.optimization.dual_bound import (
    EventDagSpec, ResourceChoiceProgram, certify_lower_bound,
    check_optimality_certificate, inward_floor_ns,
)
from mindie_sim.prediction.api import RecipeBook
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import attention, linear_attention, sparse_indexer
from mindie_sim.prediction.operators.attention._fia import AttentionContract
from mindie_sim.prediction.operators.attention.common import C8_V1_ABI, QUANT_MODE_INT8_C8
from mindie_sim.prediction.operators.paged_mla import indexed_attention, indexed_quant_attention
from mindie_sim.prediction.operators.sharedkv.split_core import (
    _p01_active_core_bps, p01_cold_hbm_group_cap, p01_cold_mte2_aic_cap,
)
from mindie_sim.prediction.operators.sparse_indexer.common import SparseIndexerContract
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.primitives.schema import compute_envelope_sha256, parse_primitive_table
from mindie_sim.schema.operators import OpSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.tier0
BUNDLE_PATH = REPO_ROOT / "data" / "verification" / "all-skus-512-400k-cliff-heldout-20261121.json"
PRODUCER_PATH = REPO_ROOT / "data" / "verification" / "all-skus-512-400k-cliff-heldout-20261121.producer.py"
MANIFEST_PATH = REPO_ROOT / "data" / "primitives" / "manifest.json"

REACHABLE_SKUS = ("Ascend910B4", "Ascend910_9362", "Ascend910_9382", "Ascend950DT_9582")
EXPECTED_SEALED_TOTAL = 1614
ANCHOR_GRID = [512, 2048, 8192, 32768, 131072, 262144, 409600]
HELDOUT_GRID = [768, 2049, 4097, 24576, 98304, 200000, 350000, 400000]
REL_TOL = 0.10
ABS_FLOOR_NS = 2000.0
NS_PER_US = 1000.0
US_PER_MS = 1000.0
LSTAR_INV_EPS = 1e-6
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


def test_bundle_provenance_integrity_and_zero_privacy_leaks(sealed_bundle: dict) -> None:
    assert sealed_bundle["schema_version"] == "mindie-sim-fleet-400k-verification/1"
    recorded_sha = sealed_bundle["bundle_sha256"]
    body = {k: v for k, v in sealed_bundle.items() if k != "bundle_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == recorded_sha

    manifest_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert sealed_bundle["provenance"]["primitive_manifest_sha256"] == manifest_sha
    manifest_tables = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["tables"]
    for sku in REACHABLE_SKUS:
        entry = sealed_bundle["skus"][sku]
        assert entry["primitive_envelope_sha256"] == manifest_tables[sku]["envelope_sha256"]
        assert entry["cann_version"] == manifest_tables[sku]["cann_version"]
        assert entry["cann_version"] and entry["cann_version"] != "unmeasured"

    grids = sealed_bundle["context_grids"]
    assert grids["anchor"] == ANCHOR_GRID
    assert grids["off_grid_cliff_heldout"] == HELDOUT_GRID
    assert grids["overlap_count"] == 0
    assert not (set(grids["anchor"]) & set(grids["off_grid_cliff_heldout"]))

    inv = sealed_bundle["fleet_inventory"]
    assert [row["sku"] for row in inv["reachable_calibrated_skus"]] == list(REACHABLE_SKUS)
    assert [row["sku"] for row in inv["unmeasured_skus"]] == ["Ascend950PR_9579"]
    assert inv["unmeasured_skus"][0]["status"] == "unmeasured"

    for artifact in (BUNDLE_PATH, PRODUCER_PATH):
        text = artifact.read_text(encoding="utf-8")
        for label, pat in PRIVACY_PATTERNS:
            match = pat.search(text)
            assert match is None, f"privacy leak ({label}) in {artifact.name}: {match.group(0)}"


def test_zero_drop_accounting_finite_bounds_and_zero_lean_inversions(sealed_bundle: dict) -> None:
    overall = sealed_bundle["overall_counts"]
    assert overall["total"] == overall["pass"] + overall["fail"] + overall["blocked"]
    assert overall["total"] == EXPECTED_SEALED_TOTAL
    assert overall["inv_sim"] == 0
    assert overall["inv_obs"] == 0

    sum_total = sum_pass = sum_fail = sum_blk = 0
    for sku in REACHABLE_SKUS:
        entry = sealed_bundle["skus"][sku]
        counts = entry["counts"]
        cases = entry["cases"]
        blocked = entry["blocked_cases"]
        assert counts["total"] == counts["pass"] + counts["fail"] + counts["blocked"]
        assert len(cases) == counts["pass"] + counts["fail"]
        assert len(blocked) == counts["blocked"]
        assert counts["inv_sim"] == 0
        assert counts["inv_obs"] == 0

        split_ids: dict[str, set[str]] = {}
        for row in cases:
            cid = row["case_id"]
            assert isinstance(cid, str) and cid
            split_ids.setdefault(row["split"], set()).add(cid)
        for brow in blocked:
            assert isinstance(brow.get("case_id"), str) and brow["case_id"]
        assert not (split_ids.get("anchor", set()) & split_ids.get("heldout", set()))
        assert not (split_ids.get("anchor", set()) & split_ids.get("ragged", set()))
        assert not (split_ids.get("heldout", set()) & split_ids.get("ragged", set()))

        sum_total += counts["total"]
        sum_pass += counts["pass"]
        sum_fail += counts["fail"]
        sum_blk += counts["blocked"]

        for row in cases:
            meas_us = float(row["meas_us"])
            meas_p99 = float(row["meas_p99_us"])
            pred_us = float(row["pred_us"])
            lstar_us = float(row["lstar_us"])
            assert math.isfinite(meas_us) and meas_us > 0.0
            assert math.isfinite(meas_p99) and meas_p99 + 1e-9 >= meas_us
            assert math.isfinite(pred_us) and pred_us > 0.0
            assert math.isfinite(lstar_us) and lstar_us >= 0.0

            err_ns = abs(pred_us * NS_PER_US - meas_us * NS_PER_US)
            tol_ns = max(REL_TOL * meas_us * NS_PER_US, ABS_FLOOR_NS)
            expected_status = "pass" if err_ns <= tol_ns else "fail"
            assert row["status"] == expected_status, f"status drift on {sku}:{row['case_id']}"
            assert row["inv_sim"] is False
            assert row["inv_obs"] is False
            assert lstar_us <= pred_us * (1.0 + LSTAR_INV_EPS)
            assert lstar_us <= meas_p99 * (1.0 + LSTAR_INV_EPS)

    assert (sum_total, sum_pass, sum_fail, sum_blk) == (
        overall["total"], overall["pass"], overall["fail"], overall["blocked"],
    )
    # Verify Lean OptimalityCertificate on 9362 2R HCCS_SW 256 MiB all-reduce (196 GB/s channel)
    ns_comm = inward_floor_ns(Fraction(268_435_456) * Fraction(1_000_000_000) / Fraction(196_000_000_000))
    prog = ResourceChoiceProgram(1, (ns_comm,), ())
    dag = EventDagSpec((0,), ())
    cert, _ = certify_lower_bound(prog, dag)
    assert check_optimality_certificate(prog, dag, cert)
    hccs_row = next(r for r in sealed_bundle["skus"]["Ascend910_9362"]["cases"] if r["case_id"] == "g5_HCCS_SW_r2_all_reduce_268435456B")
    assert math.isclose(hccs_row["lstar_us"], cert.l_star_lean_ns / NS_PER_US, rel_tol=1e-6)


def test_closed_off_grid_cliff_heldout_accuracy_across_all_4_skus(sealed_bundle: dict) -> None:
    expected_closed_med: dict[str, tuple[tuple[str, str], ...]] = {
        "Ascend910B4": (
            ("heldout", "fia_gqa_decode"), ("heldout", "fia_gqa_prefill"),
            ("heldout", "sfa_dsa_decode"), ("gate3", "g3_chain"), ("gate5", "g5_HCCS_MESH_steady"),
        ),
        "Ascend910_9362": (
            ("heldout", "fia_gqa_decode"), ("gate3", "g3_chain"), ("gate4", "g4_moe_gmm"),
        ),
        "Ascend910_9382": (
            ("heldout", "fia_gqa_prefill"), ("heldout", "c8_sfa_decode"), ("heldout", "sfa_dsa_decode"),
            ("heldout", "gdn_chunk"), ("gate3", "g3_chain"), ("gate4", "g4_moe_gmm"), ("gate5", "g5_SIO_steady"),
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
    assert sealed_bundle["skus"]["Ascend910B4"]["family_summary"]["heldout"]["sfa_dsa_decode"]["pass"] == 8


def test_live_whitebox_replay_parity_against_sealed_bundle(sealed_bundle: dict) -> None:
    def _case_row(sku: str, case_id: str) -> dict:
        for row in sealed_bundle["skus"][sku]["cases"]:
            if row["case_id"] == case_id:
                return row
        raise AssertionError(f"missing case {sku}:{case_id}")

    def _run_one(sku: str, op: OpSpec, rb: RecipeBook | None = None) -> float:
        cs = parse_case(compose_case("qwen3.8-27b", hardware=sku, tp=1, ep=1, num_devices=1, layers=1, quant="none"))
        pt = load_primitives(sku)
        ev = EvaluationContext.create(cs, rb or RecipeBook(), primitive_table=pt)
        rep = graph_costs(cs, {0: [op]}, recipes=rb, evaluation=ev, include_submission=True, isolated=True, compact=True, fast_span_only=True)
        return float(rep["duration_ms"]) * US_PER_MS

    # 1. Ascend910B4 BF16 SFA heldout L=400,000 driven by published pt.cann_version (9.2.1 -> split_kv=False)
    pt_b4 = load_primitives("Ascend910B4")
    geom_b4 = dict(query=1, context=400000, topk=2048, heads=128, latent=512, rope=64, query_dtype="bf16",
                   cache_dtype="bf16", window=0, compression_ratio=1, cp=1, split_kv=not pt_b4.cann_version.startswith("9.2"))
    op_b4 = OpSpec("sfadsa", "attention", "decode", 0, 0, (), (1, 128, 512), (1, 128, 512), (1, 128, 512), (1, 128, 512),
                   recipe="indexed_attention", dtype="bf16", metadata=(("indexed_attention_geometry", geom_b4),))
    pred_b4 = _run_one("Ascend910B4", op_b4, RecipeBook(recipes={op_b4.recipe: PlannedRecipe(
        lambda o, g: indexed_attention.build(o, g, scratch_policy="offchip"),
    )}))
    row_b4 = _case_row("Ascend910B4", "sfa_dsa_decode_heldout_L400000")
    assert math.isclose(pred_b4, row_b4["pred_us"], rel_tol=1e-5, abs_tol=1e-3)
    assert row_b4["status"] == "pass"

    # 2. Ascend910_9362 Lightning Indexer heldout L=400,000 + out-of-box ring_attention_update_v2
    c_9362 = SparseIndexerContract(query_tokens=1, scanned_rows=400000, index_heads=64, index_dim=128,
                                   topk_items=2048, batch_size=1, native_split_core=True)
    op_9362 = sparse_indexer.make_indexer_op(c_9362, op_id="lidx")
    pred_9362 = _run_one("Ascend910_9362", op_9362, RecipeBook(recipes={op_9362.recipe: PlannedRecipe(sparse_indexer.build)}))
    row_9362 = _case_row("Ascend910_9362", "lightning_indexer_topk_heldout_L400000")
    assert math.isclose(pred_9362, row_9362["pred_us"], rel_tol=1e-5, abs_tol=1e-3)
    assert row_9362["status"] == "pass"
    assert _run_one("Ascend910_9362", attention.make_ring_update_op(1, 32, 128)) > 0.0

    # 3. Ascend910_9382 C8 SFA heldout L=400,000
    geom_9382 = dict(query=1, context=400000, topk=2048, heads=128, latent=512, rope=64,
                     query_dtype="bf16", cache_dtype="int8", window=0, compression_ratio=1, cp=1)
    op_9382 = OpSpec("c8sfa", "attention", "decode", 0, 0, (), (1, 128, 512), (1, 128, 512), (1, 128, 512), (1, 128, 512),
                     recipe="indexed_quant_attention", dtype="bf16", metadata=(("indexed_attention_geometry", geom_9382),))
    pred_9382 = _run_one("Ascend910_9382", op_9382, RecipeBook(recipes={op_9382.recipe: PlannedRecipe(
        lambda o, g: indexed_quant_attention.build(o, g, scratch_policy="offchip"),
    )}))
    row_9382 = _case_row("Ascend910_9382", "c8_sfa_decode_heldout_L400000")
    assert math.isclose(pred_9382, row_9382["pred_us"], rel_tol=1e-5, abs_tol=1e-3)
    assert row_9382["status"] == "pass"

    # 4. Ascend950DT_9582 C8 SFA + Lightning Indexer heldout L=400,000
    c_a5_c8 = AttentionContract(
        query_tokens=1, num_heads=128, num_kv_heads=1, head_dim=576, v_head_dim=512, selected_tokens=2048,
        phase="decode", abi=C8_V1_ABI, kv_layout="TND", quant_mode=QUANT_MODE_INT8_C8,
        batch_size=1, sequences=((1, 400000),),
    )
    op_a5_c8 = attention.make_attention_op(c_a5_c8, op_id="c8sfa")
    pred_a5_c8 = _run_one("Ascend950DT_9582", op_a5_c8, RecipeBook(recipes={op_a5_c8.recipe: PlannedRecipe(attention.build)}))
    row_a5_c8 = _case_row("Ascend950DT_9582", "c8_sfa_decode_heldout_L400000")
    assert math.isclose(pred_a5_c8, row_a5_c8["pred_us"], rel_tol=1e-5, abs_tol=1e-3)
    assert row_a5_c8["status"] == "pass"

    c_a5_idx = SparseIndexerContract(query_tokens=1, scanned_rows=400000, index_heads=64, index_dim=128,
                                     topk_items=2048, batch_size=1, native_split_core=True)
    op_a5_idx = sparse_indexer.make_indexer_op(c_a5_idx, op_id="lidx")
    pred_a5_idx = _run_one("Ascend950DT_9582", op_a5_idx, RecipeBook(recipes={op_a5_idx.recipe: PlannedRecipe(sparse_indexer.build)}))
    row_a5_idx = _case_row("Ascend950DT_9582", "lightning_indexer_topk_heldout_L400000")
    assert math.isclose(pred_a5_idx, row_a5_idx["pred_us"], rel_tol=1e-5, abs_tol=1e-3)
    assert row_a5_idx["status"] == "pass"


def test_active_core_calibration_scoped_p18_and_work_conservation() -> None:
    """Lock P01 capacity/conflict guards, scoped P18 collective cap, and operator work conservation."""
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
            lk_nom = topo_nom.links[name]
            sc = topo_cal.link_scope(name)
            if sc == target_scope:
                assert lk_cal.capacity_bytes_per_ms <= min(p18_ms, lk_nom.capacity_bytes_per_ms) + 1e-6
            elif sc in {"package", "node"}:
                assert math.isclose(lk_cal.capacity_bytes_per_ms, lk_nom.capacity_bytes_per_ms, rel_tol=1e-9)

    # Reject forged scaling curve exceeding certified capacity
    raw_b4 = json.loads((REPO_ROOT / "data" / "primitives" / "arch22_a2_910b4.json").read_text(encoding="utf-8"))
    raw_b4["primitives"]["P01_datacopy"]["parameters"]["mte2_aic_cold_scaling_bps"] = {"4": 9e12}
    raw_b4["envelope_sha256"] = compute_envelope_sha256(raw_b4)
    assert _p01_active_core_bps(SimpleNamespace(primitive_table=parse_primitive_table(raw_b4)), 4) is None

    # Exact work conservation on Triton KDA (chunk, ragged, recurrent), native GDN recurrent, and ragged indexer
    cs9 = parse_case(compose_case("qwen3.8-27b", hardware="Ascend910_9362", tp=1, ep=1, num_devices=1, layers=1, quant="none"))
    for oid, op_name, abi, lens, hv in (
        ("kda_c", "kda_prefill", "triton_chunk_kda", (64, 192), 4),
        ("kda_r", "kda_recurrent", "triton_recurrent_kda", (1,), 8),
        ("gdn_r", "gdn_recurrent", "aclnnRecurrentGatedDeltaRule", (1,), 41),
    ):
        op = OpSpec(oid, "linear_attention", "prefill" if "prefill" in op_name else "decode", 0, 0, (),
                    (sum(lens), hv, 128), (sum(lens), hv, 128), (sum(lens), hv, 128), (sum(lens), hv, 128),
                    recipe=op_name, dtype="bf16",
                    metadata=(("operation", op_name), ("native_kind", abi), ("lengths", lens), ("hv", hv), ("dk", 128), ("dv", 128), ("bt", 64)))
        assert verify_operator(cs9, op, linear_attention.build)["work_mismatches"] == {}
    op_idx = replace(
        sparse_indexer.make_indexer_op(SparseIndexerContract(query_tokens=2, scanned_rows=513, topk_items=2048, index_heads=64, index_dim=128, batch_size=2, native_split_core=True), op_id="idx_rag"),
        metadata=tuple({**dict(sparse_indexer.make_indexer_op(SparseIndexerContract(query_tokens=2, scanned_rows=513, topk_items=2048, index_heads=64, index_dim=128, batch_size=2, native_split_core=True), op_id="idx_rag").metadata), "kv_lens": (1, 513)}.items()),
    )
    assert verify_operator(cs9, op_idx, sparse_indexer.build)["work_mismatches"] == {}
