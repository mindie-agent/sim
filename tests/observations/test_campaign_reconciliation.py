"""Wave-1 negative gates, golden v2 contract parity, nominal specs, and campaign reconciliation.

[Role] Enforce all 6 Wave-1 negative gates (anti-predicted_* spoofing, raw/campaign/shard
  SHA-256 tamper & traversal, missing planned case / split / OpOccurrence multiset mismatch,
  single-point strict tolerance breach, L* non-inversion + non-circular binding mismatch,
  train/heldout overlap + private provenance), un-derated datasheet HardwareNominalSpec
  invariants, dual-peak HBM admission, msprof outward-token isolation, DAG-replayed step
  deltas, v1/v2 bench_packets compilation, obs CLI routing, and dynamic campaign mounting.
[Invariants] Pure Tier-0 CPU behavioral tests (zero subprocesses, zero NPU context,
  zero host pollution).
[Timeouts & Locks] In-memory / tmp_path only (< 2s); no locks.
[Side-Effect Policy] Writes temporary test inputs exclusively under ``tmp_path``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mindie_sim.cli import _main as cli_main
from mindie_sim.cli.parser import resolve_route
from mindie_sim.observations.bench_packets import PacketError, build_packets
from mindie_sim.observations.campaign_reconcile import (
    CampaignIntegrityError,
    CasePredictionReference,
    diff_campaigns,
    enrich_operator_matrix,
    nominal_spec_for_sku,
    reconcile_campaign,
    replay_step_dag_us,
    verify_clock_in_band,
)
from mindie_sim.observations.comparison import apply_capture_observed_clock
from mindie_sim.schema.bench_matrix import (
    BENCH_CAMPAIGN_V1_SCHEMA,
    BENCH_ENV_V1_SCHEMA,
    BENCH_PLAN_V2_SCHEMA,
    BenchEnvFingerprint,
    BenchPlanV2,
    CampaignManifest,
    CanonicalKernelKey,
    HbmAdmissibilityReceipt,
    HardwareNominalSpec,
    MatrixSpec,
    OperatorMatrixManifest,
    OperatorMatrixRow,
    StepSummary,
    assert_no_outward_internal_tokens,
    canonical_case_id,
    canonical_json,
    canonical_key_digest,
    canonical_sha256,
    compute_campaign_sha256,
    compute_case_raw_sha256,
    compute_plan_sha256,
    resolve_fabric_collective_abi,
)
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0

TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_FIXTURE_PATH = TESTS_DIR / "fixtures" / "bench_matrix" / "v2_contract_golden.json"

#: Independent fixed reference predictions (NEVER computed from observed silicon p50_us).
INDEPENDENT_GOLDEN_PREDICTIONS: dict[str, CasePredictionReference] = {
    "k2-a9cdeca563731ad4": CasePredictionReference(
        case_id="k2-a9cdeca563731ad4",
        pred_us=43.0,
        l_star_us=36.0,
        timed_nodes=("n0",),
        participating_ranks=(0,),
        clock_operator="busy_union",
        normalized_clock_mhz=1650.0,
    ),
    "k2-5376db209f9e54dc": CasePredictionReference(
        case_id="k2-5376db209f9e54dc",
        pred_us=56.0,
        l_star_us=46.0,
        timed_nodes=("n0",),
        participating_ranks=(0,),
        clock_operator="elapsed_span",
        normalized_clock_mhz=1650.0,
    ),
}


def _load_golden() -> dict[str, Any]:
    assert GOLDEN_FIXTURE_PATH.is_file(), f"missing golden fixture: {GOLDEN_FIXTURE_PATH}"
    return json.loads(GOLDEN_FIXTURE_PATH.read_text(encoding="utf-8"))


def _seal_case(case: dict[str, Any]) -> dict[str, Any]:
    out = dict(case)
    out["raw_sha256"] = compute_case_raw_sha256(out)
    return out


def _seal_campaign(camp: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(camp)
    out["cases"] = [_seal_case(c) for c in out.get("cases", ())]
    out["campaign_sha256"] = compute_campaign_sha256(out)
    return out


def _write_shard(root: Path, rel_path: str, records: list[dict[str, Any]]) -> str:
    payload = ("\n".join(canonical_json(r) for r in records) + "\n").encode("utf-8")
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _make_matrix_row(
    entry: Any,
    plan: BenchPlanV2,
    cid: str,
    *,
    rank: int | None = None,
    stage: str | None = None,
    clock_operator: str = "elapsed_span",
    per_call_us: float = 43.0,
    l_star_us: float = 36.0,
    occurrence_count: int | None = None,
) -> OperatorMatrixRow:
    return OperatorMatrixRow(
        cell_id=entry.cell_id, model_id=entry.model_id, sku=plan.sku, soc_arch=plan.soc_arch,
        phase=entry.phase, seq_len=max(1, entry.seq_len), seq_point_kind="anchor",
        concurrency=entry.concurrency, parallel_arm=entry.parallel_arm,
        layer_archetype_id=entry.layer_archetype_id, native_layer=entry.native_layer,
        model_scope=entry.scope, replica=entry.replica,
        rank=entry.rank if rank is None else rank,
        op_id=entry.op_id,
        occurrence_count=entry.occurrence_count if occurrence_count is None else occurrence_count,
        family="attention", recipe="paged_mla", abi_backend="cann_aclnn",
        upstream_abi="aclnnFusedInferAttentionScoreV4", bench_builder="absorbed_mla",
        bench_case_id=cid, capture_mode="aclgraph",
        formula_symbolic="T_theo = HBM / B_hbm_nom", formula_terms={"kv_len": 32768},
        nominal_theo_us=30.0, per_call_stage0_us=per_call_us, model_exposed_stage0_us=per_call_us,
        step_share_pct=10.0, l_star_us=l_star_us, hbm_read_bytes=120_000_000, hbm_write_bytes=1_000_000,
        sim_mfu_pct=10.0, sim_mbu_rw_pct=72.0, sim_mbu_read_pct=71.4,
        stage=entry.stage if stage is None else stage,
        clock_operator=clock_operator,  # type: ignore[arg-type]
        normalized_clock_mhz=1650.0,
        participating_ranks=(0,),
        timed_nodes=("n0",),
    )


def test_negative_gate_rejects_caller_supplied_predicted_keys() -> None:
    golden = _load_golden()
    tampered = deepcopy(golden["campaign_manifest_v1"])
    tampered["cases"][0]["predicted_us"] = 41.0
    tampered["campaign_sha256"] = compute_campaign_sha256(tampered)
    with pytest.raises(CampaignIntegrityError, match="forbidden_prediction_key"):
        reconcile_campaign(tampered, plan=golden["bench_plan_v2"])

    tampered_ref = deepcopy(golden["campaign_manifest_v1"])
    tampered_ref["cases"][0]["sim_reference"] = {"sim_raw_us": 41.0}
    tampered_ref["campaign_sha256"] = compute_campaign_sha256(tampered_ref)
    with pytest.raises(CampaignIntegrityError, match="forbidden_prediction_key"):
        reconcile_campaign(tampered_ref, plan=golden["bench_plan_v2"])


def test_negative_gate_rejects_tampered_raw_campaign_and_shard_seals(tmp_path: Path) -> None:
    golden = _load_golden()
    # 1. Tamper p50_us without recomputing campaign_sha256.
    bad_camp = deepcopy(golden["campaign_manifest_v1"])
    bad_camp["cases"][0]["p50_us"] = 999.0
    with pytest.raises(CampaignIntegrityError, match="campaign_sha256_mismatch"):
        reconcile_campaign(bad_camp, plan=golden["bench_plan_v2"])

    # 2. Tamper p50_us and recompute campaign_sha256, leaving inner raw_sha256 stale.
    bad_raw = deepcopy(golden["campaign_manifest_v1"])
    bad_raw["cases"][0]["p50_us"] = 999.0
    bad_raw["campaign_sha256"] = compute_campaign_sha256(bad_raw)
    with pytest.raises(CampaignIntegrityError, match="case_raw_sha256_mismatch"):
        reconcile_campaign(bad_raw, plan=golden["bench_plan_v2"])

    # 3. Tamper immutable per-run shard file on disk.
    shard_rel = golden["campaign_manifest_v1"]["run_shards"][0]["path"]
    shard_file = tmp_path / shard_rel
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    shard_file.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(CampaignIntegrityError, match="run_shard_sha256_mismatch"):
        reconcile_campaign(golden["campaign_manifest_v1"], base_dir=tmp_path)

    # 4. Path traversal shard escaping campaign dir fails closed.
    escape = deepcopy(golden["campaign_manifest_v1"])
    escape["run_shards"][0]["path"] = "../outside.jsonl"
    escape = _seal_campaign(escape)
    with pytest.raises(CampaignIntegrityError, match="shard_path_escapes_campaign_dir"):
        reconcile_campaign(escape, base_dir=tmp_path)

    # 5. Case p50_us differing from its sealed shard record fails closed.
    forged_obs = deepcopy(golden["observations_v2"])
    forged_obs[0]["duration_ms"] = 0.090  # 90 us != 42.18 us in campaign manifest
    forged_sha = _write_shard(tmp_path, shard_rel, forged_obs)
    mismatch = deepcopy(golden["campaign_manifest_v1"])
    mismatch["run_shards"][0]["observations_sha256"] = forged_sha
    mismatch["observations_sha256"] = forged_sha
    mismatch = _seal_campaign(mismatch)
    with pytest.raises(CampaignIntegrityError, match="case_shard_p50_mismatch"):
        reconcile_campaign(mismatch, base_dir=tmp_path, predictions=INDEPENDENT_GOLDEN_PREDICTIONS)

    # 6. Omitted optional plan field on wire fails raw plan_sha256 verification.
    omitted_plan = deepcopy(golden["bench_plan_v2"])
    del omitted_plan["world_size"]
    with pytest.raises(ModelingError, match="bench_plan_sha256_mismatch"):
        BenchPlanV2.from_mapping(omitted_plan, verify_seal=True)

    # 7. Profiler top-level run_id survives CampaignManifest parse -> reconcile_campaign roundtrip.
    with_run_id = deepcopy(golden["campaign_manifest_v1"])
    with_run_id["run_id"] = "run-golden-0001"
    with_run_id["campaign_sha256"] = compute_campaign_sha256(with_run_id)
    parsed_camp = CampaignManifest.from_mapping(with_run_id, verify_seal=True)
    rep = reconcile_campaign(
        parsed_camp, plan=golden["bench_plan_v2"], predictions=INDEPENDENT_GOLDEN_PREDICTIONS, strict=True,
    )
    assert rep["passed"] is True
    assert rep["scored_count"] == 2


def test_negative_gate_fails_on_missing_case_split_swap_zero_coverage_and_occurrence_mismatch() -> None:
    golden = _load_golden()
    plan = BenchPlanV2.from_mapping(golden["bench_plan_v2"], verify_seal=True)
    spec = nominal_spec_for_sku(plan.sku)

    # 1. Missing 1 planned case fails closed.
    dropped = deepcopy(golden["campaign_manifest_v1"])
    dropped["cases"] = dropped["cases"][:-1]
    dropped = _seal_campaign(dropped)
    with pytest.raises(CampaignIntegrityError, match="planned_case_multiset_mismatch"):
        reconcile_campaign(dropped, plan=plan, predictions=INDEPENDENT_GOLDEN_PREDICTIONS)

    # 2. Relabeling heldout -> train in campaign fails planned case split conservation.
    relabeled = deepcopy(golden["campaign_manifest_v1"])
    relabeled["cases"][0]["split"] = "train"
    relabeled = _seal_campaign(relabeled)
    with pytest.raises(CampaignIntegrityError, match="planned_case_multiset_mismatch"):
        reconcile_campaign(relabeled, plan=plan, predictions=INDEPENDENT_GOLDEN_PREDICTIONS)

    # 3. Strict reconciliation without predictions refuses fake zero-coverage pass; plan=None is honest.
    with pytest.raises(CampaignIntegrityError, match="unscored="):
        reconcile_campaign(golden["campaign_manifest_v1"], plan=plan, strict=True)
    no_plan_rep = reconcile_campaign(
        golden["campaign_manifest_v1"], predictions=INDEPENDENT_GOLDEN_PREDICTIONS, strict=True,
    )
    assert no_plan_rep["conservation"]["plan_verified"] is False
    assert no_plan_rep["conservation"]["planned_cases"] is None

    # 4. status="ok", qualified=False and unexpected abi_rejection pass both fail closed.
    disqual = deepcopy(golden["campaign_manifest_v1"])
    disqual["cases"][0]["qualified"] = False
    disqual = _seal_campaign(disqual)
    with pytest.raises(CampaignIntegrityError, match="disqualified="):
        reconcile_campaign(disqual, plan=plan, predictions=INDEPENDENT_GOLDEN_PREDICTIONS, strict=True)

    abi_leak = deepcopy(golden["campaign_manifest_v1"])
    abi_leak["blocked_cases"][0]["abi_rejected"] = False
    abi_leak["blocked_cases"][0]["status"] = "failed"
    abi_leak = _seal_campaign(abi_leak)
    with pytest.raises(CampaignIntegrityError, match="abi_rejection_unexpected_pass"):
        reconcile_campaign(abi_leak, plan=plan, predictions=INDEPENDENT_GOLDEN_PREDICTIONS, strict=True)

    # 5. OpOccurrence conservation catches rank swap, bench_case_id swap, and stage mismatch.
    entries_list = [(cid, e) for cid, es in plan.reverse_index.items() for e in es]
    bad_rank_rows = tuple(_make_matrix_row(e, plan, cid, rank=7) for cid, e in entries_list)
    bad_matrix = OperatorMatrixManifest(
        spec=MatrixSpec(models=("synth-model",), skus=(plan.sku,)),
        nominal_specs={plan.sku: spec}, cliffs=(), hbm_receipts=(),
        rows=bad_rank_rows, step_summaries=(), reverse_index=plan.reverse_index,
        plan_sha256=plan.plan_sha256,
    )
    with pytest.raises(CampaignIntegrityError, match="occurrence_multiset_mismatch"):
        reconcile_campaign(golden["campaign_manifest_v1"], plan=plan, matrix=bad_matrix)

    swapped_cid_rows = (
        _make_matrix_row(entries_list[0][1], plan, entries_list[1][0]),
        _make_matrix_row(entries_list[1][1], plan, entries_list[0][0]),
    )
    swapped_matrix = OperatorMatrixManifest(
        spec=MatrixSpec(models=("synth-model",), skus=(plan.sku,)),
        nominal_specs={plan.sku: spec}, cliffs=(), hbm_receipts=(),
        rows=swapped_cid_rows, step_summaries=(), reverse_index=plan.reverse_index,
        plan_sha256=plan.plan_sha256,
    )
    with pytest.raises(CampaignIntegrityError, match="occurrence_multiset_mismatch"):
        reconcile_campaign(golden["campaign_manifest_v1"], plan=plan, matrix=swapped_matrix)

    bad_stage_rows = tuple(_make_matrix_row(e, plan, cid, stage="after") for cid, e in entries_list)
    bad_stage_matrix = OperatorMatrixManifest(
        spec=MatrixSpec(models=("synth-model",), skus=(plan.sku,)),
        nominal_specs={plan.sku: spec}, cliffs=(), hbm_receipts=(),
        rows=bad_stage_rows, step_summaries=(), reverse_index=plan.reverse_index,
        plan_sha256=plan.plan_sha256,
    )
    with pytest.raises(CampaignIntegrityError, match="occurrence_multiset_mismatch"):
        reconcile_campaign(golden["campaign_manifest_v1"], plan=plan, matrix=bad_stage_matrix)

    # 6. Non-circular matrix L* binding detects clock_operator mismatch against silicon campaign.
    wrong_clk_rows = tuple(_make_matrix_row(e, plan, cid, clock_operator="elapsed_span") for cid, e in entries_list)
    wrong_clk_matrix = OperatorMatrixManifest(
        spec=MatrixSpec(models=("synth-model",), skus=(plan.sku,)),
        nominal_specs={plan.sku: spec}, cliffs=(), hbm_receipts=(),
        rows=wrong_clk_rows, step_summaries=(), reverse_index=plan.reverse_index,
        plan_sha256=plan.plan_sha256,
    )
    with pytest.raises(CampaignIntegrityError, match="l_star_binding_clock_operator_mismatch"):
        reconcile_campaign(golden["campaign_manifest_v1"], plan=plan, matrix=wrong_clk_matrix)


def test_negative_gate_fails_on_single_point_strict_tolerance_breach() -> None:
    golden = _load_golden()
    camp = deepcopy(golden["campaign_manifest_v1"])
    preds = dict(INDEPENDENT_GOLDEN_PREDICTIONS)
    breach_cid = "k2-a9cdeca563731ad4"
    preds[breach_cid] = CasePredictionReference(
        case_id=breach_cid, pred_us=53.0, l_star_us=36.0,
        timed_nodes=("n0",), participating_ranks=(0,),
        clock_operator="busy_union", normalized_clock_mhz=1650.0,
    )
    with pytest.raises(CampaignIntegrityError, match="campaign_reconciliation_failed:pointwise_failures"):
        reconcile_campaign(camp, plan=golden["bench_plan_v2"], predictions=preds, strict=True)


def test_negative_gate_fails_on_l_star_inversion_and_binding_or_clock_violations() -> None:
    golden = _load_golden()
    camp = deepcopy(golden["campaign_manifest_v1"])
    cid0 = "k2-a9cdeca563731ad4"

    # 1. L* > meas_p50 (42.6 > 42.18 even though <= p99=44.05).
    inv_preds = dict(INDEPENDENT_GOLDEN_PREDICTIONS)
    inv_preds[cid0] = CasePredictionReference(
        case_id=cid0, pred_us=43.0, l_star_us=42.6,
        timed_nodes=("n0",), participating_ranks=(0,),
        clock_operator="busy_union", normalized_clock_mhz=1650.0,
    )
    with pytest.raises(CampaignIntegrityError, match="l_star_inversions"):
        reconcile_campaign(camp, plan=golden["bench_plan_v2"], predictions=inv_preds, strict=True)

    # 2. L* == 0 fails closed.
    zero_preds = dict(INDEPENDENT_GOLDEN_PREDICTIONS)
    zero_preds[cid0] = CasePredictionReference(
        case_id=cid0, pred_us=42.18, l_star_us=0.0,
        timed_nodes=("n0",), participating_ranks=(0,),
        clock_operator="busy_union", normalized_clock_mhz=1650.0,
    )
    with pytest.raises(CampaignIntegrityError, match="l_star_inversions"):
        reconcile_campaign(camp, plan=golden["bench_plan_v2"], predictions=zero_preds, strict=True)

    # 3. Binding mismatch on timed_nodes fails closed.
    bad_bind = dict(INDEPENDENT_GOLDEN_PREDICTIONS)
    bad_bind[cid0] = CasePredictionReference(
        case_id=cid0, pred_us=42.18, l_star_us=35.0,
        timed_nodes=("wrong_node",), participating_ranks=(0,),
        clock_operator="busy_union", normalized_clock_mhz=1650.0,
    )
    with pytest.raises(CampaignIntegrityError, match="l_star_binding_timed_nodes_mismatch"):
        reconcile_campaign(camp, plan=golden["bench_plan_v2"], predictions=bad_bind)

    # 4. Untraced busy_union fails closed (Astra #6).
    untraced = deepcopy(camp)
    untraced["cases"][0]["clock_operator"] = "busy_union"
    untraced["cases"][0]["traced_interval_verified"] = False
    untraced = _seal_campaign(untraced)
    with pytest.raises(CampaignIntegrityError, match="busy_union_requires_traced_intervals"):
        reconcile_campaign(untraced)

    # 5. Thermally throttled frequency rejected even if qualified=False (Astra #6).
    throttled = deepcopy(camp)
    throttled["cases"][0]["observed_clock_mhz"] = 900.0
    throttled["cases"][0]["qualified"] = False
    throttled = _seal_campaign(throttled)
    with pytest.raises(CampaignIntegrityError, match="clock_thermally_throttled"):
        reconcile_campaign(throttled)

    # 6. apply_capture_observed_clock wires observed_clock_mhz directly.
    case_doc = {"machine": {"device": {"compute_units": 32}}}
    aligned = apply_capture_observed_clock(case_doc, {"observed_clock_mhz": 1725.0})
    assert aligned["machine"]["device"]["observed_aicore_freq_mhz"] == pytest.approx(1725.0)
    aligned_nested = apply_capture_observed_clock(case_doc, {"version_fingerprint": {"observed_clock_mhz": 1680.0}})
    assert aligned_nested["machine"]["device"]["observed_aicore_freq_mhz"] == pytest.approx(1680.0)


def test_negative_gate_fails_on_train_heldout_overlap_and_private_provenance() -> None:
    golden = _load_golden()
    camp = deepcopy(golden["campaign_manifest_v1"])
    heldout_case = next(c for c in camp["cases"] if c["split"] == "heldout")

    # 1. Cross-plan training_manifest_sha256 geometry freeze blocks overlap in cases[] AND requires manifest.
    training_manifest = {"cases": [{"canonical_key_digest": heldout_case["canonical_key_digest"], "split": "train"}]}
    camp_with_tm = deepcopy(camp)
    camp_with_tm["training_manifest_sha256"] = canonical_sha256(training_manifest)
    camp_with_tm = _seal_campaign(camp_with_tm)
    with pytest.raises(CampaignIntegrityError, match="training_manifest_required_for_heldout_verification"):
        reconcile_campaign(camp_with_tm)
    with pytest.raises(CampaignIntegrityError, match="train_heldout_digest_overlap"):
        reconcile_campaign(camp_with_tm, training_manifest=training_manifest)

    # 2. Private provenance leak ($HOME / IPv4) rejected WITHOUT echoing private IP literal in error text,
    #    while valid 4-part CANN versions (e.g. 9.0.0.2) pass without false-positive IPv4 rejection.
    leaked = deepcopy(camp)
    leaked["cases"][0]["capture_uri"] = "capture://camp/10.20.30.40/op.json"
    leaked = _seal_campaign(leaked)
    with pytest.raises(CampaignIntegrityError, match="private_provenance_ipv4") as exc_info:
        reconcile_campaign(leaked)
    assert "10.20.30.40" not in str(exc_info.value)

    cann_four_part = deepcopy(camp)
    cann_four_part["version_fingerprint"]["cann_version"] = "9.0.0.2"
    cann_four_part["version_fingerprint"]["env_fingerprint_id"] = ""
    fp4 = BenchEnvFingerprint.from_mapping(cann_four_part["version_fingerprint"])
    cann_four_part["version_fingerprint"] = fp4.to_dict()
    cann_four_part["env_fingerprint_id"] = fp4.env_fingerprint_id
    cann_four_part = _seal_campaign(cann_four_part)
    rep4 = reconcile_campaign(cann_four_part, predictions=INDEPENDENT_GOLDEN_PREDICTIONS, strict=True)
    assert rep4["passed"] is True


def test_nominal_spec_underated_dual_peak_hbm_msprof_enrichment_and_dag_replay() -> None:
    # 1. Un-derated datasheet peaks + strict link-tier mapping (NO cross-rail fallback!).
    for sku, expected_hbm_tbps in (
        ("Ascend910B4", 0.8),
        ("Ascend910_9362", 1.6),
        ("Ascend950DT_9582", 4.0),
    ):
        spec = nominal_spec_for_sku(sku)
        assert spec.utilization == 1.0
        assert spec.launch_us == 0.0
        assert spec.hbm_peak_bytes_per_s == pytest.approx(expected_hbm_tbps * 1e12)
        assert verify_clock_in_band(sku, spec.nominal_clock_mhz) == pytest.approx(1.0)
        if sku == "Ascend910_9362":
            assert spec.link_peak_for_tier("d2d_intra_package") == pytest.approx(270.0 * 1e9)
            assert spec.link_peak_for_tier("d2d_intra_node") == pytest.approx(196.0 * 1e9)
            assert spec.link_peak_for_tier("inter_node_roce") == pytest.approx(25.0 * 1e9)
        if sku == "Ascend950DT_9582":
            assert spec.link_peak_for_tier("d2d_intra_node") == pytest.approx(56.0 * 1e9)
            assert spec.link_peak_for_tier("inter_node_roce") == pytest.approx(50.0 * 1e9)

    spec_a5 = nominal_spec_for_sku("Ascend950DT_9582")
    with pytest.raises(ModelingError, match="nominal_spec_missing_link_tier"):
        spec_a5.link_peak_for_tier("unknown_fabric_rail")
    with pytest.raises(ModelingError, match="nominal_spec_must_be_underated"):
        HardwareNominalSpec(
            sku=spec_a5.sku, soc_arch=spec_a5.soc_arch, aic_count=spec_a5.aic_count,
            aiv_count=spec_a5.aiv_count, hbm_capacity_bytes=spec_a5.hbm_capacity_bytes,
            l2_capacity_bytes=spec_a5.l2_capacity_bytes, ub_capacity_bytes=spec_a5.ub_capacity_bytes,
            nominal_clock_mhz=spec_a5.nominal_clock_mhz, cube_peaks_flops_per_s=spec_a5.cube_peaks_flops_per_s,
            vector_peak_ops_per_s=spec_a5.vector_peak_ops_per_s, hbm_peak_bytes_per_s=spec_a5.hbm_peak_bytes_per_s,
            raw_link_peaks_bytes_per_s=spec_a5.raw_link_peaks_bytes_per_s, utilization=0.80,
        )

    # 2. Dual-peak HBM receipt rejects die whose runtime fits (75 GiB) but load peak overflows (82 GiB).
    gib = 1024 ** 3
    with pytest.raises(ModelingError, match="hbm_admissibility_dual_peak_violation"):
        HbmAdmissibilityReceipt(
            admissible=True, reason="ok", usable_hbm_bytes_per_die=80 * gib,
            runtime_peak_hbm_bytes_by_die=(75 * gib,), load_peak_hbm_bytes_by_die=(82 * gib,),
        )

    # 3. DAG replay for overlapping streams + fail closed on missing DAG node duration.
    nodes = ("start", "branch_a", "branch_b", "join")
    edges = (("start", "branch_a"), ("start", "branch_b"), ("branch_a", "join"), ("branch_b", "join"))
    assert replay_step_dag_us(nodes, edges, {"start": 1.0, "branch_a": 10.0, "branch_b": 10.0, "join": 1.0}) == pytest.approx(12.0)
    assert replay_step_dag_us(nodes, edges, {"start": 1.0, "branch_a": 5.0, "branch_b": 10.0, "join": 1.0}) == pytest.approx(12.0)
    with pytest.raises(CampaignIntegrityError, match="dag_missing_node_duration:branch_b"):
        replay_step_dag_us(nodes, edges, {"start": 1.0, "branch_a": 5.0, "join": 1.0})

    # 4. Outward msprof token isolation enforced inside to_outward_dict() itself.
    row = OperatorMatrixRow(
        cell_id="c0", model_id="synth-llm", sku="Ascend950DT_9582", soc_arch="A5",
        phase="decode", seq_len=32768, seq_point_kind="anchor", concurrency=8,
        parallel_arm="tp8_ep8", layer_archetype_id="mla_layer", native_layer=0,
        model_scope="target", replica=0, rank=0, op_id="l0.attn", occurrence_count=2,
        family="attention", recipe="paged_mla", abi_backend="cann_aclnn",
        upstream_abi="aclnnFusedInferAttentionScoreV4", bench_builder="absorbed_mla",
        bench_case_id="k2-a9cdeca563731ad4", capture_mode="aclgraph",
        formula_symbolic="T_theo = (HBM_R + HBM_W) / B_hbm_nom",
        formula_terms={"kv_len": 32768}, nominal_theo_us=25.0, per_call_stage0_us=43.0,
        model_exposed_stage0_us=43.0, step_share_pct=20.0, l_star_us=36.0,
        hbm_read_bytes=90_000_000, hbm_write_bytes=10_000_000,
        sim_mfu_pct=15.0, sim_mbu_rw_pct=62.5, sim_mbu_read_pct=56.25,
        op_name="FusedInferAttentionScoreV4_0", op_type="FusedInferAttentionScoreV4",
        kernel_name="aclnnFusedInferAttentionScoreV4", accelerator_core="MIX_AIC",
        block_dim=32, mix_block_dim=64, stream_id=0,
        input_shapes="8,16,576;256,128,1,576", input_data_types="BF16;INT8",
        input_formats="TND;PA_BBND", output_shapes="8,16,512",
        output_data_types="BF16", output_formats="TND",
        arithmetic_intensity=12.5, bound_type="Memory Bound (HBM)",
        clock_operator="busy_union", normalized_clock_mhz=1650.0,
        participating_ranks=(0,), timed_nodes=("n0",),
    )
    assert row.hbm_total_bytes == 100_000_000
    assert row.sim_mbu_pct == pytest.approx(62.5)
    assert_no_outward_internal_tokens(json.dumps(row.to_outward_dict(), sort_keys=True))
    leaky_row = replace(row, formula_terms={"primitive": "P06_mmad"})
    with pytest.raises(ModelingError, match="outward_internal_token_leak"):
        leaky_row.to_outward_dict()

    # 5. Re-enriching clears stale measured values & replays repeated occurrence_count>1 DAG nodes.
    golden = _load_golden()
    camp = CampaignManifest.from_mapping(golden["campaign_manifest_v1"], verify_seal=True)
    step = StepSummary(
        cell_id="c0", model_id="synth-llm", sku="Ascend950DT_9582", soc_arch="A5",
        phase="decode", seq_len=32768, concurrency=8, parallel_arm="tp8_ep8",
        num_dies=8, admissible=True, sim_step_us=86.0, l_star_step_us=72.0,
        active_weight_bytes=8_000_000_000, kv_read_bytes=2_000_000_000,
        new_kv_write_bytes=100_000_000, total_step_hbm_rw_bytes=14_000_000_000,
        total_step_cube_flops=500_000_000_000, model_mbu_pct=31.5625,
        hbm_bus_util_pct=43.75, step_sim_mfu_pct=14.45,
    )
    mat = OperatorMatrixManifest(
        spec=MatrixSpec(models=("synth-llm",), skus=("Ascend950DT_9582",)),
        nominal_specs={"Ascend950DT_9582": spec_a5}, cliffs=(), hbm_receipts=(),
        rows=(row,), step_summaries=(step,), reverse_index={},
    )
    enriched = enrich_operator_matrix(
        mat, camp, step_dags={"c0": (("l0.attn#1", "l0.attn#2"), (("l0.attn#1", "l0.attn#2"),))},
    )
    assert enriched.rows[0].measured_us == pytest.approx(42.18)
    assert enriched.step_summaries[0].measured_step_us == pytest.approx(84.36)
    empty_camp = CampaignManifest.from_mapping(_seal_campaign({**golden["campaign_manifest_v1"], "cases": []}), verify_seal=True)
    re_enriched = enrich_operator_matrix(enriched, empty_camp)
    assert re_enriched.rows[0].measured_us is None
    assert re_enriched.rows[0].env_fingerprint_id is None
    assert re_enriched.step_summaries[0].measured_step_us is None
    assert re_enriched.step_summaries[0].dag_replayed is False


def test_golden_v2_contract_keys_bench_packets_diff_and_obs_cli(tmp_path: Path) -> None:
    golden = _load_golden()

    # 1. Exact byte-for-byte CanonicalKernelKey digests + nested timed_nodes + recursive forbidden keys.
    for sample in golden["canonical_key_samples"]:
        parsed = CanonicalKernelKey.from_mapping(sample["key"])
        assert canonical_json(parsed.to_canonical_dict()) == sample["canonical_json"]
        assert parsed.digest() == sample["canonical_key_digest"]
        assert parsed.case_id() == sample["case_id"]
        assert canonical_key_digest(sample["key"]) == sample["canonical_key_digest"]
        assert canonical_case_id(sample["key"]) == sample["case_id"]

    digests_by_name = {s["name"]: s["canonical_key_digest"] for s in golden["canonical_key_samples"]}
    assert digests_by_name["square_gemm_contiguous"] != digests_by_name["square_gemm_transposed"]

    bad_nested = deepcopy(golden["canonical_key_samples"][0]["key"])
    bad_nested["op_attrs"]["split"] = "heldout"
    with pytest.raises(ModelingError, match="canonical_key_forbidden_field"):
        CanonicalKernelKey.from_mapping(bad_nested)

    child_omitted = deepcopy(golden["canonical_key_samples"][1]["key"])
    del child_omitted["edges"]
    child_explicit = deepcopy(golden["canonical_key_samples"][1]["key"])
    sg_a = CanonicalKernelKey.from_mapping({**child_explicit, "kind": "subgraph", "timed_nodes": [child_omitted]})
    sg_b = CanonicalKernelKey.from_mapping({**child_explicit, "kind": "subgraph", "timed_nodes": [child_explicit]})
    assert sg_a.digest() == sg_b.digest()

    # 2. Verify BenchEnvFingerprint, BenchPlanV2, CampaignManifest seals.
    env = BenchEnvFingerprint.from_mapping(golden["env_fingerprint"])
    assert env.schema == BENCH_ENV_V1_SCHEMA
    assert env.env_fingerprint_id == golden["env_fingerprint"]["env_fingerprint_id"]
    plan = BenchPlanV2.from_mapping(golden["bench_plan_v2"], verify_seal=True)
    assert plan.schema == BENCH_PLAN_V2_SCHEMA
    assert plan.plan_sha256 == compute_plan_sha256(golden["bench_plan_v2"])
    camp = CampaignManifest.from_mapping(golden["campaign_manifest_v1"], verify_seal=True)
    assert camp.schema == BENCH_CAMPAIGN_V1_SCHEMA

    # 3. bench_packets handles v2 split="matrix", level="L1", residency separation, and rejects failed abi_rejection.
    assert resolve_fabric_collective_abi("all_reduce") == ("all_reduce", "hcclAllReduce")
    matrix_obs = deepcopy(golden["observations_v2"][0])
    matrix_obs["sample_id"] = "obs-matrix-0001"
    matrix_obs["split"] = "matrix"
    matrix_obs["identity"]["level"] = "L1"
    warm_a = deepcopy(golden["observations_v2"][1])
    warm_a["sample_id"], warm_a["residency"], warm_a["work_bytes"] = "obs-warm-1", "warm_l2", 1_048_576
    warm_b = deepcopy(golden["observations_v2"][1])
    warm_b["sample_id"], warm_b["residency"], warm_b["work_bytes"] = "obs-warm-2", "warm_l2", 2_097_152
    cold_b = deepcopy(golden["observations_v2"][1])
    cold_b["sample_id"], cold_b["work_bytes"] = "obs-cold-2", 2_097_152
    bundle = build_packets([*golden["observations_v2"], matrix_obs, warm_a, warm_b, cold_b])
    assert bundle["manifest"]["guarded_case_outcomes"] == 1
    assert bundle["manifest"]["matrix_observations"] == 1
    assert bundle["manifest"]["fit-rate_conditions"] == 2  # warm_l2 and cold_hbm kept separate!

    bad_abi_obs = deepcopy(golden["observations_v2"][2])
    bad_abi_obs["status"], bad_abi_obs["abi_rejected"] = "failed", False
    with pytest.raises(PacketError, match="abi_rejection_unexpected_pass"):
        build_packets([bad_abi_obs])

    # 4. diff_campaigns detects CANN delta, enforces shard integrity on paths, and separates clock_operator mismatches.
    cand = deepcopy(golden["campaign_manifest_v1"])
    cand["campaign_id"] = "camp-golden-cann94"
    cand["version_fingerprint"]["cann_version"] = "9.4.0"
    cand["version_fingerprint"]["env_fingerprint_id"] = ""
    cand_env = BenchEnvFingerprint.from_mapping(cand["version_fingerprint"])
    cand["version_fingerprint"] = cand_env.to_dict()
    cand["env_fingerprint_id"] = cand_env.env_fingerprint_id
    cand["cases"][0]["p50_us"] = float(cand["cases"][0]["p50_us"]) * 0.8
    cand["cases"][0]["p99_us"] = float(cand["cases"][0]["p99_us"]) * 0.8
    cand_obs = deepcopy(golden["observations_v2"])
    cand_obs[0]["duration_ms"] = cand["cases"][0]["p50_us"] / 1000.0
    cand_shard_rel = "runs/run-golden-cand.jsonl"
    cand_sha = _write_shard(tmp_path, cand_shard_rel, cand_obs)
    cand["observations_jsonl"] = cand_shard_rel
    cand["observations_sha256"] = cand_sha
    cand["run_shards"][0]["path"] = cand_shard_rel
    cand["run_shards"][0]["observations_sha256"] = cand_sha
    cand["cases"][0]["run_shard"] = cand_shard_rel
    cand["cases"][1]["run_shard"] = cand_shard_rel
    cand = _seal_campaign(cand)

    diff = diff_campaigns(golden["campaign_manifest_v1"], cand)
    assert diff["matched_kernels"] == len(camp.cases)
    assert diff["version_fingerprint_diff"]["cann_version"] == {"baseline": "9.3.RC1", "candidate": "9.4.0"}
    mla_delta = next(d for d in diff["kernel_deltas"] if d["case_id"] == cand["cases"][0]["case_id"])
    assert mla_delta["speedup_ratio"] == pytest.approx(1.25)

    mismatch_op = deepcopy(cand)
    mismatch_op["cases"][0]["clock_operator"] = "elapsed_span"
    mismatch_op = _seal_campaign(mismatch_op)
    diff_incompat = diff_campaigns(golden["campaign_manifest_v1"], mismatch_op)
    assert len(diff_incompat["incompatible_clock_operator_pairs"]) == 1
    assert diff_incompat["matched_kernels"] == 1

    # 5. obs CLI routing & execution (AG015-B production reachability).
    assert resolve_route(["obs", "bench-packets", "--receipts", "a.jsonl", "--out", "out"])[0].command == "bench-packets"
    assert resolve_route(["obs", "reconcile-campaign", "--campaign", "c.json"])[0].command == "reconcile-campaign"
    assert resolve_route(["obs", "diff-campaigns", "--baseline", "a.json", "--candidate", "b.json"])[0].command == "diff-campaigns"

    camp_a_path = tmp_path / "base_campaign.json"
    camp_b_path = tmp_path / "cand_campaign.json"
    plan_path = tmp_path / "plan.json"
    preds_path = tmp_path / "preds.json"
    diff_out = tmp_path / "diff.json"
    rec_out = tmp_path / "reconcile.json"
    obs_rel = golden["campaign_manifest_v1"]["run_shards"][0]["path"]
    _write_shard(tmp_path, obs_rel, golden["observations_v2"])
    obs_path = tmp_path / obs_rel
    pkt_out = tmp_path / "packets"
    camp_a_path.write_text(json.dumps(golden["campaign_manifest_v1"]), encoding="utf-8")
    camp_b_path.write_text(json.dumps(cand), encoding="utf-8")
    plan_path.write_text(json.dumps(golden["bench_plan_v2"]), encoding="utf-8")
    preds_path.write_text(json.dumps({
        cid: {
            "pred_us": ref.pred_us, "l_star_us": ref.l_star_us,
            "timed_nodes": list(ref.timed_nodes), "participating_ranks": list(ref.participating_ranks),
            "clock_operator": ref.clock_operator, "normalized_clock_mhz": ref.normalized_clock_mhz,
        }
        for cid, ref in INDEPENDENT_GOLDEN_PREDICTIONS.items()
    }), encoding="utf-8")

    assert cli_main(["obs", "bench-packets", "--receipts", str(obs_path), "--out", str(pkt_out)]) == 0
    assert (pkt_out / "manifest.json").is_file()
    assert cli_main([
        "obs", "reconcile-campaign", "--campaign", str(camp_a_path),
        "--plan", str(plan_path), "--predictions", str(preds_path), "--out", str(rec_out),
    ]) == 0
    assert json.loads(rec_out.read_text(encoding="utf-8"))["passed"] is True
    assert cli_main(["obs", "diff-campaigns", "--baseline", str(camp_a_path), "--candidate", str(camp_b_path), "--out", str(diff_out)]) == 0
    assert json.loads(diff_out.read_text(encoding="utf-8"))["matched_kernels"] == len(camp.cases)

    # 6. Cross-plan cache hit preserves original acquisition_uid & raw_sha256 while flipping cache_status="hit".
    cached_camp = deepcopy(golden["campaign_manifest_v1"])
    orig_raw_sha = cached_camp["cases"][0]["raw_sha256"]
    cached_camp["cases"][0]["cache_status"] = "hit"
    cached_camp["cases"][0]["acquisition_uid"] = "acq-prior-plan-0099"
    cached_camp["cases"][0]["measured_in_run_id"] = "run-prior-0099"
    cached_camp["cases"][0]["run_shard"] = "runs/run-prior-0099.jsonl"
    cached_camp["cases"][0]["raw_sha256"] = compute_case_raw_sha256(cached_camp["cases"][0])
    hit_only = deepcopy(golden["campaign_manifest_v1"])
    hit_only["cases"][0]["cache_status"] = "hit"
    assert compute_case_raw_sha256(hit_only["cases"][0]) == orig_raw_sha
    cached_camp["campaign_sha256"] = compute_campaign_sha256(cached_camp)
    hit_report = reconcile_campaign(
        cached_camp, plan=golden["bench_plan_v2"], predictions=INDEPENDENT_GOLDEN_PREDICTIONS,
        base_dir=tmp_path, strict=True,
    )
    assert hit_report["passed"] is True


def test_mounted_campaign_reconciliation(mounted_campaign_manifest: Path | None) -> None:
    """Dynamically reconcile mounted campaigns (CLI --campaign-* / env / golden fallback)."""
    if mounted_campaign_manifest is None:
        pytest.fail("explicit --campaign-manifest / --campaign-dir / --campaign-* filter matched zero campaigns")
    doc = json.loads(mounted_campaign_manifest.read_text(encoding="utf-8"))
    if "campaign_manifest_v1" in doc and "bench_plan_v2" in doc:
        camp = doc["campaign_manifest_v1"]
        plan = doc["bench_plan_v2"]
        report = reconcile_campaign(
            camp, plan=plan, predictions=INDEPENDENT_GOLDEN_PREDICTIONS, strict=True,
        )
    else:
        sibling_plan = mounted_campaign_manifest.parent / "bench_plan.json"
        sibling_preds = mounted_campaign_manifest.parent / "predictions.json"
        has_preds = sibling_preds.is_file()
        report = reconcile_campaign(
            doc,
            plan=sibling_plan if sibling_plan.is_file() else None,
            predictions=json.loads(sibling_preds.read_text(encoding="utf-8")) if has_preds else None,
            base_dir=mounted_campaign_manifest.parent,
            strict=True,
            integrity_only=not has_preds,
        )
    assert report["passed"] is True
    assert report["l_star_inversions"] == []
    assert report["pointwise_failures"] == []
