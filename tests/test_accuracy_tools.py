"""Tier-0 tests for the accuracy tooling: pure functions and synthetic artifacts only.

No subprocess, no network, no device, no real measurement file. Where a test needs a
filesystem artifact it writes a deliberately tiny synthetic one into ``tmp_path`` and drives
the same code path the CLI drives.

The tests are chosen for the places where a wrong answer would silently corrupt the round:
the WAPE denominator, the nearest-rank p90, the coverage denominator, bootstrap determinism,
the identity keys (including the declaration generator's exact geometry key), the declared
work model, protocol canonicalisation, the split enforcement / merge rules, and the
freeze/score integrity refusals.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.accuracy import assemble_dataset, freeze_predictions, metrics, score_frozen  # noqa: E402

A3_HARDWARE = "Ascend910_9362"
A2_HARDWARE = "Ascend910B4"

# One declared A3 test-set-2 point, as written by generate_splits.py.
DECLARED_GEMM_POINT = {
    "point_id": "a3:heldout:gemm:k=1536,m=1,n=7168",
    "family": "gemm", "shape": {"m": 1, "n": 7168, "k": 1536},
    "geometry_key": "geo:d87d2c56f9ee2edd0fb531af", "split": "heldout",
    "regime": "launch", "analytic_lower_bound_ms": 0.027547,
    "axis": "m", "cross_sku_matched": True, "reason": "fresh_heldout",
    "test_set": "test2_fresh_heldout", "acceptance_bearing": True,
}

A3_REGIME_RULE = {
    "bands": {"large": "analytic_lower_bound_ms >= 1.0",
              "launch": "analytic_lower_bound_ms < 0.05",
              "mid": "0.05 <= analytic_lower_bound_ms < 1.0"},
    "constants": {"hbm_bytes_per_s": 800e9, "peak_bf16_ops_per_s": 245760e9,
                  "vector_ops_per_s": 7680e9},
    "formula": "analytic_lower_bound_ms = 1000 * max(flops/peak, ops/vector, bytes/hbm)",
    "outcome_free": True,
    "work_models": {
        "gemm": "flops=2*m*n*k; bytes=2*(m*k+k*n+m*n)",
        "add": "ops=elements; bytes=6*elements",
        "copy": "bytes=2*rows*row_bytes",
        "rmsnorm": "ops=2*tokens*width; bytes=6*tokens*width",
        "moe_*": "flops=2*routed_tokens*n*k; bytes=2*(k*n+2*routed_tokens*k+routed_tokens*n)",
    },
}


# --------------------------------------------------------------------- helpers ---
def observation(condition_key="cond:one", *, prediction_rows=(), declared=1, **overrides):
    return {"condition_key": condition_key, **overrides}


def row(condition_id, prediction, observed, *, region="exact", status="ok", variant="analytic"):
    return {"condition_id": condition_id, "prediction_ms": prediction, "observed_ms": observed,
            "numeric": prediction is not None, "region": region, "status": status,
            "variant": variant,
            "ape_pct": None if prediction is None else metrics.ape_pct(prediction, observed),
            "signed_error_pct": None if prediction is None else
            metrics.signed_error_pct(prediction, observed)}


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def minimal_ir(**overrides):
    base = {
        "record_id": "src:case-1", "family": "gemm", "recipe": "torch_mm_preallocated",
        "shape": {"m": 4, "n": 8, "k": 16}, "hardware_key": A3_HARDWARE,
        "arch_family": "A3", "die": "phy1", "die_attested": True, "die_evidence": "test",
        "run_id": "run-1", "source_class": "measured", "status": "completed",
        "split_hint": "train", "split_origin": "test", "trials": [[1.0, 1.2, 0.8], [2.0, 2.2]],
        "protocol_raw": {"timer": "device_event", "timing_scope": "device_event_warm",
                         "warmup": 10, "iterations": 30, "trials": 2, "preallocated": True},
        "record_level": {}, "defaults": {}, "precision": {"input_dtype": "bf16"},
        "layout": "contiguous", "quantization": "none", "features": {},
        "identity_extra": {}, "flags": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------- metrics ---
def test_wape_divides_by_covered_observed_mass_only():
    pairs = [(1.2, 1.0), (1.1, 1.0), (9.0, 10.0)]
    # numerator 0.2 + 0.1 + 1.0 = 1.3 ; denominator 1+1+10 = 12 observed milliseconds
    assert metrics.wape_pct(pairs) == pytest.approx(100 * 1.3 / 12.0)
    assert metrics.wape_pct([]) is None


def test_wape_is_duration_weighted_not_count_weighted():
    small = [(1.1, 1.0)] * 10          # ten 10 % misses, 10 ms of mass
    large = [(1.0, 100.0)]             # one 99 % miss, 100 ms of mass
    # 90.9 % duration-weighted, while the median APE of the same rows is only 10 %.
    assert metrics.wape_pct(small + large) == pytest.approx(100 * (1.0 + 99.0) / 110.0)
    median_ape = metrics.median([metrics.ape_pct(p, o) for p, o in small + large])
    assert metrics.wape_pct(small + large) > median_ape
    assert metrics.wape_pct(large) == pytest.approx(99.0)


def test_p90_is_nearest_rank_on_the_ascending_list():
    eleven = [float(value) for value in range(11)]      # 0..10
    assert metrics.p90_nearest_rank(eleven) == 9.0      # index ceil(9.9)-1 = 9
    ten = [float(value) for value in range(10)]
    assert metrics.p90_nearest_rank(ten) == 8.0         # index ceil(9.0)-1 = 8
    assert metrics.p90_nearest_rank([]) is None


def test_median_ape_for_even_n_averages_the_two_central_values():
    rows = [row(f"c{i}", 1.0 + off, 1.0) for i, off in enumerate((0.1, 0.2, 0.3, 0.4))]
    assert metrics.summarize(rows, b=0)["median_ape_pct"] == pytest.approx(25.0)


def test_coverage_denominator_keeps_unsupported_and_unmeasured_conditions():
    rows = [row("a", 1.0, 1.0), row("b", None, 1.0, status="unsupported")]
    summary = metrics.summarize(rows, n_declared=5, b=0)
    assert summary["numeric"] == 1
    assert summary["coverage"] == pytest.approx(1 / 5)
    assert summary["coverage_denominator"] == (
        "all_declared_conditions_including_unsupported_and_transition_unresolved")
    assert summary["weighting"]["wape"] == "duration_weighted_over_covered_conditions"


def test_penalized_wape_charges_an_abstention_a_full_error():
    covered = [(1.5, 1.0)]
    uncovered = [3.0, 6.0]
    # (0.5 + 3 + 6) / (1 + 3 + 6)
    assert metrics.wape_penalized_pct(covered, uncovered) == pytest.approx(100 * 9.5 / 10.0)


def test_interval_containment_is_inclusive_and_ignores_missing_bounds():
    intervals = [(1.0, 2.0, 1.0), (1.0, 2.0, 2.0), (1.0, 2.0, 2.5)]
    result = metrics.interval_containment(intervals)
    assert (result["reported_intervals"], result["contained"]) == (3, 2)
    assert metrics.interval_containment([])["containment_rate"] is None


def test_bootstrap_is_deterministic_and_guarded_below_five_conditions():
    errors = [0.1, 0.2, 0.05, 0.3, 0.15, 0.25]
    observed = [1.0] * 6
    first = metrics.bootstrap_wape_ci(errors, observed, b=200, seed=7)
    second = metrics.bootstrap_wape_ci(errors, observed, b=200, seed=7)
    assert first == second
    assert first[0] <= 100 * sum(errors) / sum(observed) <= first[1]
    assert metrics.bootstrap_wape_ci(errors[:4], observed[:4], b=200, seed=7) is None
    assert metrics.summarize([row("c", 1.1, 1.0)], b=200)["wape_ci95"] is None


def test_paired_bootstrap_keeps_the_pairing_and_reports_a_delta():
    model = [0.05, 0.05, 0.05, 0.05, 0.05, 0.4]
    baseline = [0.4, 0.4, 0.4, 0.4, 0.4, 0.05]
    observed = [1.0] * 6
    result = metrics.paired_bootstrap_delta(model, baseline, observed, b=200, seed=11)
    assert result["delta_wape_pct"] == pytest.approx(
        100 * (sum(model) - sum(baseline)) / sum(observed))
    assert result["ci95"] is None or result["ci95"][0] <= result["ci95"][1]


def test_signed_bias_and_spearman_sanity():
    pairs = [(1.1, 1.0), (2.2, 2.0), (3.3, 3.0)]
    assert metrics.median_signed_bias_pct(pairs) == pytest.approx(10.0)
    assert metrics.spearman_rho([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert metrics.spearman_rho([1, 2], [2, 1]) is None      # needs n >= 3


def test_summarize_marks_small_cells_insufficient_and_histograms_statuses():
    rows = [row("a", 1.1, 1.0), row("b", None, 1.0, status="transition_unresolved")]
    summary = metrics.summarize(rows, b=0)
    assert summary["insufficient_n"] is True
    assert summary["statuses"] == {"ok": 1, "transition_unresolved": 1}


# --------------------------------------------------------------------- identity ---
def test_geometry_key_reproduces_the_declaration_generator():
    assert metrics.geometry_key(A3_HARDWARE, "gemm", {"m": 1, "n": 7168, "k": 1536}) == \
        DECLARED_GEMM_POINT["geometry_key"]
    # The key is hardware-, family- and shape-sensitive and protocol-blind.
    assert metrics.geometry_key(A2_HARDWARE, "gemm", {"m": 1, "n": 7168, "k": 1536}) != \
        DECLARED_GEMM_POINT["geometry_key"]
    declared_with_protocol = metrics.geometry_key(
        A3_HARDWARE, "gemm", {"m": 1, "n": 7168, "k": 1536}, precision={"input_dtype": "bf16"})
    assert declared_with_protocol != DECLARED_GEMM_POINT["geometry_key"]


def test_copy_payload_identity_unifies_rows_and_elements():
    by_rows = metrics.shape_identity("copy", {"rows": 128, "row_bytes": 1024})
    by_elements = metrics.shape_identity("copy_bytes", {"elements": 65536}, dtype="bf16")
    assert by_rows == by_elements == "copy_bytes:131072"
    assert metrics.shape_identity("copy", {"elements": 65537}, dtype="bf16") != by_elements


def test_shape_key_and_family_aliases_produce_matching_candidates():
    attention = metrics.identity_candidates("attention_qk",
                                            {"batch": 1, "q_tokens": 129, "kv_tokens": 129,
                                             "q_heads": 3, "kv_heads": 1, "qk_dim": 256})
    declared = metrics.shape_identity("attention_qk", {"batch": 1, "q": 129, "kv": 129,
                                                      "q_heads": 3, "kv_heads": 1,
                                                      "qk_dim": 256})
    assert declared in attention
    rope = metrics.identity_candidates("rope", {"batch": 1, "seq": 31, "heads": 3, "dim": 64})
    assert metrics.shape_identity("rope", {"batch": 1, "tokens": 31, "heads": 3,
                                           "width": 64}) in rope
    masked = metrics.identity_candidates("masked_softmax", {"heads": 2, "q": 96, "kv": 96})
    assert metrics.shape_identity("attention_softmax", {"heads": 2, "q": 96, "kv": 96}) in masked
    moe = metrics.identity_candidates("grouped_matmul",
                                      {"experts": 8, "routed_tokens": 264, "k": 8192, "n": 4096})
    assert metrics.shape_identity("moe_gate_up", {"experts": 8, "routed_tokens": 264,
                                                  "k": 8192, "n": 4096}) in moe


def test_condition_key_separates_dies_and_protocols_but_not_run_ids():
    kwargs = dict(hardware_key=A3_HARDWARE, family="gemm", shape={"m": 4, "n": 8, "k": 16},
                  recipe="torch_mm_preallocated", protocol_signature_value="proto:aaa")
    base = metrics.condition_key(die="phy1", **kwargs)
    assert base == metrics.condition_key(die="phy1", **kwargs)
    assert base != metrics.condition_key(die="phy0", **kwargs)
    assert base != metrics.condition_key(die="phy1", **{**kwargs,
                                                       "protocol_signature_value": "proto:bbb"})
    assert base != metrics.condition_key(die="phy1", **{**kwargs, "amortization": 64})


# -------------------------------------------------------------------- work model ---
def test_declared_work_model_reproduces_the_declared_lower_bound_and_band():
    model = metrics.WorkModel(A3_REGIME_RULE)
    demand = model.demand("gemm", {"m": 1, "n": 7168, "k": 1536})
    assert demand["flops"] == pytest.approx(2 * 1 * 7168 * 1536)
    lower = model.lower_bound_ms("gemm", {"m": 1, "n": 7168, "k": 1536})
    # the declaration rounds its own lower bound to 6 decimals
    assert lower == pytest.approx(DECLARED_GEMM_POINT["analytic_lower_bound_ms"], abs=1e-6)
    edges = metrics.band_edges(A3_REGIME_RULE["bands"])
    assert edges == (0.05, 1.0)
    assert metrics.band_of(lower, edges) == DECLARED_GEMM_POINT["regime"]
    assert metrics.band_of(0.05, edges) == "mid" and metrics.band_of(1.0, edges) == "large"


def test_work_model_prefix_match_and_missing_variable_are_explicit():
    model = metrics.WorkModel(A3_REGIME_RULE)
    assert model.demand("moe_gate_up", {"routed_tokens": 8, "experts": 2, "n": 4, "k": 8})[
        "flops"] == pytest.approx(2 * 8 * 4 * 8)
    with pytest.raises(metrics.WorkModelError):
        model.demand("gemm", {"m": 1, "n": 2})
    with pytest.raises(metrics.WorkModelError):
        model.demand("attention_qk", {"q": 1, "kv": 1})


def test_work_model_refuses_executable_expressions():
    with pytest.raises(metrics.WorkModelError):
        metrics.WorkModel({"work_models": {"gemm": "flops=__import__('os').getcwd()"},
                           "constants": {"peak_bf16_ops_per_s": 1.0}})
    with pytest.raises(metrics.WorkModelError):
        metrics.WorkModel({"work_models": {"gemm": "flops=open('/etc/passwd').read()"},
                           "constants": {"peak_bf16_ops_per_s": 1.0}})


def test_band_edges_fall_back_to_unbanded_instead_of_guessing():
    assert metrics.band_edges({}) == (None, None)
    assert metrics.band_of(0.001, (None, None)) is None


# -------------------------------------------------------------------- protocol ---
def test_protocol_signature_treats_absence_as_unknown_never_as_a_wildcard():
    complete = {"timer": "device_event", "timing_scope": "device_event_warm", "warmup": 10,
                "iterations": 30, "trials": 2, "preallocated": True, "cache_policy": "reuse"}
    silent = {key: value for key, value in complete.items() if key != "cache_policy"}
    assert metrics.protocol_core(silent)["cache_policy"] == metrics.UNKNOWN
    assert metrics.protocol_signature(metrics.protocol_core(complete)) != \
        metrics.protocol_signature(metrics.protocol_core(silent))
    assert metrics.protocol_matches(metrics.protocol_core(complete),
                                    metrics.protocol_core(silent)) is False


def test_protocol_lifts_and_strict_mode():
    record = {"warmup_iterations": 10, "samples_per_trial": 30, "trials": 2,
              "preallocated": False, "timer": "device_event",
              "timing_scope": "device_event_warm_operator"}
    core = metrics.protocol_core(record)
    assert core["warmup"] == 10 and core["iterations"] == 30 and core["trials"] == 2
    assert core["preallocated"] is False
    origins = metrics.protocol_origins(record, record_level={"timing_scope": "x"},
                                       defaults={"cache_policy": "reuse"})
    assert origins["warmup"] == "attested" and origins["cache_policy"] == "declared_default"
    assert origins["measurement_scope"] == "unknown"
    loose = metrics.protocol_signature(core, strict=False)
    strict = metrics.protocol_signature(core, strict=True)
    assert loose != strict
    strict_core = metrics.protocol_core(record, strict=True)
    # A strict comparison is only violated when one side *attests* a strict-only field.
    assert metrics.protocol_matches(strict_core, core, strict=True) is True
    assert metrics.protocol_matches({**strict_core, "warmup_scope": "before_each_trial"},
                                    {**strict_core, "warmup_scope": "once_per_condition"},
                                    strict=True) is False


def test_protocol_value_normalisation_refuses_garbage():
    core = metrics.protocol_core({"trials": "unknown", "warmup": "n/a", "preallocated": 0})
    assert core["trials"] == metrics.UNKNOWN and core["warmup"] == metrics.UNKNOWN
    assert core["preallocated"] is False


# ------------------------------------------------------------------- assembler ---
def test_normalize_record_uses_trial_medians_not_the_pooled_median():
    model = metrics.WorkModel(A3_REGIME_RULE)
    record = assemble_dataset.normalize_record(
        minimal_ir(), {"id": "test", "arch_family": "A3"}, model, strict_protocol=False)[0]
    assert record["duration_ms"] == pytest.approx(1.55)    # median(median(1.0,1.2,0.8), median(2.0,2.2))
    assert record["diagnostics"]["pooled_median_ms"] != record["duration_ms"]
    assert record["diagnostics"]["estimator"] == "median_of_per_trial_medians"
    assert record["flops"] == 2 * 4 * 8 * 16 and record["hbm_bytes"] > 0
    assert set(record) == set(metrics.OBSERVATION_KEYS)


def test_normalize_record_excludes_pending_and_synthetic_with_a_reason():
    model = metrics.WorkModel(A3_REGIME_RULE)
    source = {"id": "test", "arch_family": "A3"}
    for ir, reason in ((minimal_ir(status="pending"), "not_measured"),
                       (minimal_ir(flags=["not_a3_measured"]), "not_a3_measured"),
                       (minimal_ir(source_class="synthetic"),
                        "not_calibratable_source_class"),
                       (minimal_ir(trials=[]), "no_usable_samples")):
        observation, exclusion = assemble_dataset.normalize_record(
            ir, source, model, strict_protocol=False)
        assert observation is None and exclusion["reason"] == reason


def build_minimal_corpus(root: Path) -> tuple[Path, Path]:
    """A tiny corpus: one declared heldout point, one train shape, one duplicate row."""
    declarations = root / "declarations"
    write_json(declarations / "a3-split-declaration.json", {
        "schema": "mindie-sim-split-declaration/1",
        "declaration_id": "test-a3",
        "sku": {"hardware_key": A3_HARDWARE, "arch_family": "A3",
                "physical_form": "dual-die package"},
        "regime_rule": A3_REGIME_RULE,
        "new_heldout": {"test_set": "test2_fresh_heldout",
                        "points": [DECLARED_GEMM_POINT]},
        "families": [],
        "test_sets": {"test1_old_heldout": {"conditions": 0}},
    })
    data = root / "data"
    heldout_case = {
        "request": {"case_id": "heldout-1", "op_family": "gemm",
                    "shape": {"m": 1, "n": 7168, "k": 1536}},
        "campaign_split": "heldout",
        "production_result": {
            "status": "completed", "timing_scope": "device_event_warm",
            "raw_samples_ms": [0.03, 0.031, 0.029, 0.032],
            "protocol": {"timer": "device_event", "warmup": 10, "iterations": 4,
                         "preallocated": True, "layout": "contiguous",
                         "measurement_scope": "warm"},
            "run_id": "run-heldout", "details": {"kind": "mm"},
            "environment": {"device_name": A3_HARDWARE, "visible_devices": "1"},
        },
    }
    train_case = {
        "request": {"case_id": "train-1", "op_family": "gemm",
                    "shape": {"m": 4, "n": 8, "k": 16}},
        "campaign_split": "train",
        "production_result": {
            "status": "completed", "timing_scope": "device_event_warm",
            "raw_samples_ms": [0.02, 0.021, 0.019],
            "protocol": {"timer": "device_event", "warmup": 10, "iterations": 3,
                         "preallocated": True, "layout": "contiguous",
                         "measurement_scope": "warm"},
            "run_id": "run-train", "details": {"kind": "mm"},
            "environment": {"device_name": A3_HARDWARE, "visible_devices": "1"},
        },
    }
    # A second run of the same train condition (merge) and an exact duplicate (collapse).
    rerun_case = json.loads(json.dumps(train_case))
    rerun_case["production_result"]["run_id"] = "run-train-rerun"
    rerun_case["production_result"]["raw_samples_ms"] = [0.022, 0.02, 0.021]
    duplicate_case = json.loads(json.dumps(train_case))
    write_json(data / "nine-model-linearity-20261021/heldout-results.json",
               {"schema": "x", "cases": [heldout_case]})
    write_json(data / "nine-model-linearity-20261021/train-results.json",
               {"schema": "x", "cases": [train_case, rerun_case, duplicate_case]})
    return data, declarations


def test_assemble_enforces_the_split_merges_runs_and_collapses_duplicates(tmp_path):
    data, declarations = build_minimal_corpus(tmp_path)
    result = assemble_dataset.assemble(
        data_root=data, a2_root=tmp_path / "no-a2", a3_round2_root=tmp_path / "no-a3",
        declarations_dir=declarations, round1_evidence=tmp_path / "no-round1",
        strict_protocol=False, keep_excluded=False)
    observations = result["dataset"]["observations"]
    by_identity = {item["diagnostics"]["geometry_identity"]: item for item in observations}
    heldout = by_identity["gemm:k=1536,m=1,n=7168"]
    assert heldout["split"] == "heldout"
    assert heldout["diagnostics"]["declared_heldout"] is True
    assert heldout["diagnostics"]["heldout_point_id"] == DECLARED_GEMM_POINT["point_id"]
    train = by_identity["gemm:k=16,m=4,n=8"]
    assert train["split"] == "train"
    assert len(train["runs"]) == 2                      # re-run merged, duplicate collapsed
    assert result["manifest"]["merge"]["exact_duplicate_rows"] == 1
    assert result["manifest"]["counts"]["train_heldout_conflicts"] == 0
    assert result["manifest"]["counts"]["observations"] == 2


def test_assemble_relabels_a_train_row_that_hits_a_declared_heldout_key(tmp_path):
    data, declarations = build_minimal_corpus(tmp_path)
    # Give the train case the heldout geometry under a different protocol: it must be
    # relabelled (never trained on) and listed as a relabel, not silently dropped.
    path = data / "nine-model-linearity-20261021/train-results.json"
    document = json.loads(path.read_text())
    document["cases"][0]["request"]["shape"] = {"m": 1, "n": 7168, "k": 1536}
    document["cases"][0]["production_result"]["protocol"]["iterations"] = 4
    path.write_text(json.dumps(document))
    result = assemble_dataset.assemble(
        data_root=data, a2_root=tmp_path / "no-a2", a3_round2_root=tmp_path / "no-a3",
        declarations_dir=declarations, round1_evidence=tmp_path / "no-round1",
        strict_protocol=False, keep_excluded=False)
    relabelled = result["manifest"]["split_enforcement"]["relabelled_train_rows"]
    assert len(relabelled) == 1
    assert relabelled[0]["family"] == "gemm"
    assert result["manifest"]["counts"]["train_heldout_conflicts"] == 0


def test_outside_store_table_reports_the_estimator_mismatch(tmp_path):
    data, declarations = build_minimal_corpus(tmp_path)
    result = assemble_dataset.assemble(
        data_root=data, a2_root=tmp_path / "no-a2", a3_round2_root=tmp_path / "no-a3",
        declarations_dir=declarations, round1_evidence=tmp_path / "no-round1",
        strict_protocol=False, keep_excluded=False)
    table = assemble_dataset.build_outside_store_table(result["dataset"], result["manifest"],
                                                       store_identities=set())
    assert table["summary"]["rows"] >= 1
    first = table["rows"][0]
    assert first["estimator"] == "median_of_per_trial_medians"
    assert first["pooled_median_ms"] is not None


# --------------------------------------------------------------- freeze / score ---
def test_manifest_validation_refuses_outcome_fields_and_duplicate_ids():
    with pytest.raises(freeze_predictions.IntegrityError):
        freeze_predictions.normalize_manifest({"entries": [
            {"condition_id": "a", "family": "gemm", "shape": {"m": 1},
             "hardware_key": A3_HARDWARE, "observed_ms": 1.0}]})
    with pytest.raises(freeze_predictions.IntegrityError):
        freeze_predictions.normalize_manifest({"entries": [
            {"condition_id": "a", "family": "gemm", "shape": {"m": 1},
             "hardware_key": A3_HARDWARE},
            {"condition_id": "a", "family": "gemm", "shape": {"m": 2},
             "hardware_key": A3_HARDWARE}]})


def test_manifest_keys_match_the_declared_geometry_key():
    manifest = freeze_predictions.normalize_manifest({"entries": [
        {"condition_id": "p1", "kind": "curve", "family": "gemm",
         "shape": {"m": 1, "n": 7168, "k": 1536}, "hardware_key": A3_HARDWARE,
         "protocol": {"timer": "device_event", "warmup": 10, "iterations": 30,
                      "trials": 2, "preallocated": True}}]})
    entry = manifest["entries"][0]
    assert entry["geometry_key"] == DECLARED_GEMM_POINT["geometry_key"]
    assert entry["protocol_key"].startswith("proto:")


def build_frozen_corpus(root: Path, *, dirty=False, outcomes_read=False,
                        dataset_mutator=None, prediction_mutator=None):
    """A complete dataset + frozen predictions + receipt, written to ``root``."""
    dataset_dir = root / "dataset"
    declarations = root / "declarations"
    write_json(declarations / "a3-split-declaration.json", {
        "schema": "mindie-sim-split-declaration/1", "declaration_id": "test-a3",
        "sku": {"hardware_key": A3_HARDWARE, "arch_family": "A3"},
        "regime_rule": A3_REGIME_RULE,
        "new_heldout": {"test_set": "test2_fresh_heldout", "points": [
            {**DECLARED_GEMM_POINT, "point_id": "p1"},
            {**DECLARED_GEMM_POINT, "point_id": "p2",
             "shape": {"m": 2, "n": 7168, "k": 1536},
             "geometry_key": metrics.geometry_key(A3_HARDWARE, "gemm",
                                                  {"m": 2, "n": 7168, "k": 1536}),
             "regime": "launch"}]},
        "families": [], "test_sets": {"test1_old_heldout": {"conditions": 0}},
    })
    protocol_core = {"timer": "device_event", "timing_scope": "device_event_warm",
                     "warmup": 10, "iterations": 4, "trials": 1, "preallocated": True,
                     "cache_policy": "unknown", "measurement_scope": "warm",
                     "attention_scope": "unknown"}
    signature = metrics.protocol_signature(protocol_core)
    observations = []
    for point_id, shape, duration in (("p1", {"m": 1, "n": 7168, "k": 1536}, 0.03),
                                     ("p2", {"m": 2, "n": 7168, "k": 1536}, 0.04)):
        observations.append({
            "condition_key": f"cond:{point_id}",
            "hardware_key": A3_HARDWARE, "die": "phy1", "arch_family": "A3",
            "family": "gemm", "recipe": "torch_mm_preallocated",
            "protocol_signature": signature, "run_id": f"run-{point_id}",
            "split": "heldout", "shape": shape, "elements": None,
            "flops": 2 * shape["m"] * shape["n"] * shape["k"], "hbm_bytes": 1,
            "duration_ms": duration, "raw_samples_ms": [duration] * 4,
            "source_class": "measured", "timing_scope": "device_event_warm",
            "runs": [{"run_id": f"run-{point_id}", "trials": 1,
                      "trial_medians_ms": [duration], "raw_events": 4,
                      "raw_samples_sha256": "sha", "pooled_median_ms": duration,
                      "source_artifact": "synthetic", "source_sha256": "sha",
                      "source_id": "test", "record_id": point_id}],
            "diagnostics": {
                "geometry_key": metrics.geometry_key(A3_HARDWARE, "gemm", shape),
                "geometry_identity": metrics.shape_identity("gemm", shape),
                "geometry_identities": [metrics.shape_identity("gemm", shape)],
                "family_class": "gemm", "pooled_median_ms": duration,
                "pooled_minus_trial_median_ms": 0.0, "trial_medians_ms": [duration],
                "trials": 1, "raw_events": 4, "estimator": "median_of_per_trial_medians",
                "single_trial": True, "protocol": protocol_core,
                "protocol_origin": {name: "attested" for name in protocol_core},
                "die_attested": True, "die_evidence": "test", "precision": None,
                "layout": None, "quantization": None, "features": {},
                "execution_kind": None, "worker_recipe": None, "backend": None,
                "analytic_ms_artifact": None, "source_artifact": "synthetic",
                "source_sha256": "sha", "source_id": "test", "record_id": point_id,
                "split_origin": "test", "split_hint": "heldout",
                "declared_heldout": True, "heldout_point_id": point_id,
                "heldout_test_set": "test2_fresh_heldout", "flags": [],
                "alias_group": None, "alias_members": [],
            },
        })
    dataset = {"schema_version": metrics.OBSERVATION_SCHEMA, "observations": observations}
    if dataset_mutator:
        dataset = dataset_mutator(dataset)
    write_json(dataset_dir / "observations.json", dataset)
    dataset_hash = metrics.sha256_value(dataset["observations"])
    write_json(dataset_dir / "dataset-manifest.json",
               {"schema_version": "mindie-sim.dataset-manifest/1",
                "dataset_sha256": dataset_hash})

    predictions = {
        "schema_version": metrics.PREDICTIONS_SCHEMA, "frozen_at": "2026-01-01T00:00:00+00:00",
        "manifest": {"manifest_id": "test", "sha256": "0" * 64, "entries": 1},
        "source": {"commit": "0" * 40, "dirty": dirty, "dirty_entries": [],
                   "package_file_hashes_sha256": "0" * 64},
        "protocol": {"declaration_files": {
            "a3-split-declaration.json":
                metrics.sha256_file(declarations / "a3-split-declaration.json")}},
        "dataset": {"sha256": dataset_hash,
                    "file_sha256": metrics.sha256_file(dataset_dir / "observations.json"),
                    "path": str(dataset_dir), "training_conditions": 0},
        "scope": {"hardware_keys": [A3_HARDWARE], "test_sets": ["test2_fresh_heldout"],
                  "declared_heldout_conditions": 2, "undeclared_queries": []},
        "status": "completed", "expected_entries": 1, "completed_entries": 1,
        "evaluation_outcomes_read": outcomes_read, "outcomes_directory_listed": False,
        "strict_protocol": False, "automatic_fitting_performed": False,
        "scoring_performed": False,
        "predictions": [
            {"condition_id": "p1",
             "geometry_key": DECLARED_GEMM_POINT["geometry_key"],
             "protocol_key": signature, "protocol_core": protocol_core, "kind": "curve",
             "family": "gemm", "shape": {"m": 1, "n": 7168, "k": 1536},
             "hardware_key": A3_HARDWARE, "query_sha256": "sha", "status": "completed",
             "variants": {
                 "analytic": {"status": "ok", "prediction_ms": 0.036, "region": None},
                 "calibrated_db": {"status": "ok", "prediction_ms": 0.033, "region": None},
                 "curve": {"status": "ok", "prediction_ms": 0.0305,
                           "region": "interpolation"}}},
        ],
    }
    if prediction_mutator:
        predictions = prediction_mutator(predictions)
    predictions_path = write_json(root / "predictions.frozen.json", predictions)
    receipt = {
        "schema_version": metrics.FREEZE_SCHEMA, "frozen_at": "2026-01-01T00:00:00+00:00",
        "round_id": "round-test",
        "manifest": predictions["manifest"], "source": predictions["source"],
        "protocol": predictions["protocol"], "dataset": predictions["dataset"],
        "training_inventory_sha256": "0" * 64, "training_inventory_conditions": 0,
        "heldout_conditions": 2, "training_heldout_overlap": 0,
        "scope": predictions["scope"], "predictions_path": str(predictions_path),
        "predictions_sha256": metrics.sha256_file(predictions_path),
        "evaluation_outcomes_read": False, "outcomes_directory_listed": False,
        "expected_entries": 1, "completed_entries": 1, "complete": True,
        "strict_protocol": False, "dirty_tree_recorded": dirty,
        "imported_from": None, "command": [],
    }
    receipt_path = write_json(root / "freeze-receipt.json", receipt)
    return {"dataset": dataset_dir, "predictions": predictions_path, "receipt": receipt_path,
            "declarations": declarations, "root": root}


def test_score_verifies_hashes_and_refuses_a_tampered_or_incomplete_freeze(tmp_path):
    corpus = build_frozen_corpus(tmp_path / "good")
    base = dict(dataset_path=corpus["dataset"], predictions_path=corpus["predictions"],
                receipt_path=corpus["receipt"], declarations=corpus["declarations"],
                round1_evidence=tmp_path / "no-round1", out=tmp_path / "out",
                bootstrap_b=32, allow_dirty=True)
    # A tampered prediction file (one value changed, so the hash must differ).
    tampered_document = json.loads(corpus["predictions"].read_text())
    tampered_document["predictions"][0]["variants"]["curve"]["prediction_ms"] = 0.031
    tampered = write_json(tmp_path / "tampered.json", tampered_document)
    with pytest.raises(score_frozen.IntegrityError):
        score_frozen.score(**{**base, "predictions_path": tampered})
    # A dataset whose content no longer matches the receipt.
    other = build_frozen_corpus(tmp_path / "other", dataset_mutator=lambda doc: {
        **doc, "observations": [{**doc["observations"][0], "duration_ms": 9.0}]})
    with pytest.raises(score_frozen.IntegrityError):
        score_frozen.score(**{**base, "receipt_path": other["receipt"],
                              "dataset_path": other["dataset"]})
    # An incomplete freeze.
    incomplete_root = tmp_path / "incomplete"
    incomplete = build_frozen_corpus(incomplete_root)
    document = json.loads(incomplete["predictions"].read_text())
    document["completed_entries"] = 0
    write_json(incomplete["predictions"], document)
    receipt = json.loads(incomplete["receipt"].read_text())
    receipt["predictions_sha256"] = metrics.sha256_file(incomplete["predictions"])
    write_json(incomplete["receipt"], receipt)
    with pytest.raises(score_frozen.IntegrityError):
        score_frozen.score(**{**base, "predictions_path": incomplete["predictions"],
                              "receipt_path": incomplete["receipt"]})
    # A freeze that claims it read outcomes.
    dirty = build_frozen_corpus(tmp_path / "outcomes", outcomes_read=True)
    with pytest.raises(score_frozen.IntegrityError):
        score_frozen.score(**{**base, "predictions_path": dirty["predictions"],
                              "receipt_path": dirty["receipt"]})
    # A dirty tree without --allow-dirty.
    dirty_tree = build_frozen_corpus(tmp_path / "dirty", dirty=True)
    with pytest.raises(score_frozen.IntegrityError):
        score_frozen.score(**{**base, "predictions_path": dirty_tree["predictions"],
                              "receipt_path": dirty_tree["receipt"], "allow_dirty": False})


def test_score_picks_the_primary_path_and_divides_coverage_by_declared_conditions(tmp_path):
    corpus = build_frozen_corpus(tmp_path / "good")
    result = score_frozen.score(
        dataset_path=corpus["dataset"], predictions_path=corpus["predictions"],
        receipt_path=corpus["receipt"], declarations=corpus["declarations"],
        round1_evidence=tmp_path / "no-round1", out=tmp_path / "out",
        policy="curve_first", bootstrap_b=32, allow_dirty=True)
    acceptance = result["acceptance"]
    # Two conditions are declared, one is frozen and numeric -> coverage 1/2.
    assert acceptance["conditions"] == 2 and acceptance["numeric"] == 1
    assert acceptance["coverage"] == pytest.approx(0.5)
    assert acceptance["coverage_denominator"].startswith("all_declared_conditions")
    # curve_first picks the direct curve (0.0305) over the calibrated DB (0.033).
    assert acceptance["primary_path"]["policy"] == "curve_first"
    assert acceptance["primary_path"]["per_condition"]["p1"] == "curve"
    assert acceptance["primary_path"]["counts"] == {"curve": 1, "none": 1}
    assert acceptance["wape_pct"] == pytest.approx(100 * abs(0.0305 - 0.03) / 0.03)
    assert {item["reason"] for item in acceptance["excluded"]} == {"not_queried"}
    # The all-variant roll-up still reports the calibrated variant.
    assert acceptance["variants"]["calibrated_db"]["numeric"] == 1
    assert acceptance["variants"]["calibrated_db"]["wape_pct"] == pytest.approx(
        100 * abs(0.033 - 0.03) / 0.03)
    assert [gate["id"] for gate in acceptance["gates"]] == [
        f"G{index}" for index in range(1, 13)]
    assert acceptance["protocol_acceptance"]["schema_version"] == \
        metrics.PROTOCOL_ACCEPTANCE_SCHEMA


def test_calibrated_first_policy_reproduces_the_other_primary_chain(tmp_path):
    corpus = build_frozen_corpus(tmp_path / "good")
    result = score_frozen.score(
        dataset_path=corpus["dataset"], predictions_path=corpus["predictions"],
        receipt_path=corpus["receipt"], declarations=corpus["declarations"],
        round1_evidence=tmp_path / "no-round1", out=tmp_path / "out",
        policy="calibrated_first", bootstrap_b=32, allow_dirty=True)
    acceptance = result["acceptance"]
    assert acceptance["primary_path"]["per_condition"]["p1"] == "calibrated_db"
    assert acceptance["wape_pct"] == pytest.approx(100 * abs(0.033 - 0.03) / 0.03)


def test_cli_exit_codes_are_0_for_scoring_and_2_when_enforced_threshold_fails(tmp_path):
    corpus = build_frozen_corpus(tmp_path / "good")
    common = ["--dataset", str(corpus["dataset"]), "--predictions", str(corpus["predictions"]),
              "--receipt", str(corpus["receipt"]), "--declarations", str(corpus["declarations"]),
              "--round1-evidence", str(tmp_path / "no-round1"),
              "--bootstrap-b", "16", "--allow-dirty"]
    assert score_frozen.main([*common, "--out", str(tmp_path / "out1")]) == 0
    assert score_frozen.main([*common, "--out", str(tmp_path / "out2"), "--enforce",
                              "--max-wape", "0.1"]) == 2
    assert score_frozen.main([*common, "--out", str(tmp_path / "out3"), "--enforce",
                              "--max-wape", "500"]) == 0
    # Integrity failure path returns 3 rather than raising through the CLI.
    tampered_document = json.loads(corpus["predictions"].read_text())
    tampered_document["predictions"][0]["variants"]["curve"]["prediction_ms"] = 0.031
    tampered = write_json(tmp_path / "tampered.json", tampered_document)
    assert score_frozen.main(["--dataset", str(corpus["dataset"]),
                              "--predictions", str(tampered),
                              "--receipt", str(corpus["receipt"]),
                              "--declarations", str(corpus["declarations"]),
                              "--round1-evidence", str(tmp_path / "no-round1"),
                              "--out", str(tmp_path / "out4"), "--allow-dirty"]) == 3


def test_repeat_noise_floor_uses_merged_same_protocol_runs(tmp_path):
    dataset = {"observations": [
        {"family": "gemm", "split": "train",
         "runs": [{"trial_medians_ms": [0.100]}, {"trial_medians_ms": [0.120]},
                  {"trial_medians_ms": [0.080]}]},
        {"family": "rmsnorm", "split": "train", "runs": [{"trial_medians_ms": [0.05]}]},
    ]}
    noise = score_frozen.repeat_noise_floor(dataset, ["gemm", "rmsnorm"])
    assert noise["conditions_by_family"]["gemm"] == 1
    assert "rmsnorm" not in noise["conditions_by_family"]
    assert noise["repeat_noise_floor_pct"]["gemm"] == pytest.approx(100 * 0.02 / 0.11)


# ------------------------------------------------------------- heldout manifests ---
def test_heldout_manifests_are_deterministic_heldout_only_and_collision_checked(tmp_path):
    from tools.accuracy import heldout_manifests

    declarations = tmp_path / "declarations"
    write_json(declarations / "a3-split-declaration.json", {
        "declaration_id": "test-a3",
        "sku": {"hardware_key": A3_HARDWARE, "arch_family": "A3"},
        "regime_rule": A3_REGIME_RULE,
        "protocol_identity": {"timer": "device_event", "timing_scope": "device_event_warm",
                              "warmup": 10, "warmup_scope": "before_each_trial",
                              "iterations": 30, "trials": 3, "preallocated": True,
                              "cache_policy": "same_preallocated_buffers_reused_no_cache_flush"},
        "new_heldout": {"test_set": "test2_fresh_heldout", "points": [
            {**DECLARED_GEMM_POINT, "point_id": "p-gemm"},
            {"point_id": "p-fa", "family": "attention_qk", "regime": "launch",
             "shape": {"batch": 1, "q": 24, "kv": 24, "q_heads": 3, "kv_heads": 1,
                       "qk_dim": 256, "v_dim": 256},
             "geometry_key": "geo:fa", "acceptance_bearing": False,
             "test_set": "test2_fresh_heldout"},
            {"point_id": "p-rope", "family": "rope", "regime": "launch",
             "shape": {"batch": 1, "tokens": 24, "heads": 2, "width": 64},
             "geometry_key": "geo:rope", "acceptance_bearing": False,
             "test_set": "test2_fresh_heldout"}]},
        "families": [], "test_sets": {"test1_old_heldout": {"conditions": 0}},
    })
    dataset_dir = tmp_path / "dataset"
    write_json(dataset_dir / "observations.json",
               {"schema_version": metrics.OBSERVATION_SCHEMA, "observations": []})

    first = heldout_manifests.generate(
        declarations=declarations, dataset=dataset_dir, out=tmp_path / "out1",
        receipt=tmp_path / "no-receipt.json", skip_dry_run=True, run_prefix="ho")
    second = heldout_manifests.generate(
        declarations=declarations, dataset=dataset_dir, out=tmp_path / "out2",
        receipt=tmp_path / "no-receipt.json", skip_dry_run=True, run_prefix="ho")
    # Deterministic: the same declaration produces byte-identical manifests.
    for entry in first["manifests"]:
        assert entry["sha256"] == next(item["sha256"] for item in second["manifests"]
                                       if item["filename"] == entry["filename"])
    by_name = {entry["filename"]: entry for entry in first["manifests"]}
    campaign = next(entry for name, entry in by_name.items() if "gemm" in name)
    assert campaign["cases"] == 1 and campaign["split_counts"] == {"heldout": 1}
    attention = next(entry for name, entry in by_name.items() if "attention" in name)
    assert attention["cases"] == 2
    manifest = json.loads((tmp_path / "out1" / attention["path"]).read_text())
    assert manifest["kind"] == "a3_attention_rope_manifest"
    assert {case["split"] for case in manifest["cases"]} == {"heldout"}
    fusion = next(case for case in manifest["cases"]
                  if case["operator"] == "fusion_attention")
    assert fusion["mask"] == "causal_bottom_right"      # q == kv > 1 -> prefill-shaped
    assert fusion["scale"] == pytest.approx(1 / math.sqrt(256))
    assert fusion["shape"]["q_tokens"] == 24 and fusion["shape"]["kv_tokens"] == 24
    # The guard refuses without a receipt, and the collision check passes on an empty train set.
    assert first["guard"]["guard"] == "fail"
    assert first["guard"]["reasons"] == ["frozen_predictions_receipt_missing"]
    assert first["collisions"]["passed"] is True


def test_heldout_manifest_generation_refuses_a_heldout_condition_in_training(tmp_path):
    from tools.accuracy import heldout_manifests

    declarations = tmp_path / "declarations"
    write_json(declarations / "a3-split-declaration.json", {
        "declaration_id": "test-a3",
        "sku": {"hardware_key": A3_HARDWARE, "arch_family": "A3"},
        "regime_rule": A3_REGIME_RULE,
        "protocol_identity": {"timer": "device_event", "timing_scope": "device_event_warm",
                              "warmup": 10, "iterations": 30, "trials": 3,
                              "preallocated": True},
        "new_heldout": {"test_set": "test2_fresh_heldout",
                        "points": [DECLARED_GEMM_POINT]},
        "families": [], "test_sets": {"test1_old_heldout": {"conditions": 0}},
    })
    identity = metrics.shape_identity("gemm", {"m": 1, "n": 7168, "k": 1536})
    dataset_dir = tmp_path / "dataset"
    write_json(dataset_dir / "observations.json", {
        "schema_version": metrics.OBSERVATION_SCHEMA,
        "observations": [{"hardware_key": A3_HARDWARE, "split": "train",
                          "diagnostics": {"geometry_identity": identity,
                                          "geometry_identities": [identity]}}]})
    with pytest.raises(heldout_manifests.IntegrityError):
        heldout_manifests.generate(declarations=declarations, dataset=dataset_dir,
                                   out=tmp_path / "out", receipt=None, skip_dry_run=True,
                                   run_prefix="ho")


def test_guard_passes_on_a_current_receipt_and_fails_when_the_dataset_changed(tmp_path):
    from tools.accuracy import heldout_manifests

    corpus = build_frozen_corpus(tmp_path / "guard")
    passing = heldout_manifests.guard(corpus["receipt"], corpus["dataset"])
    assert passing["guard"] == "pass" and passing["reasons"] == []
    document = json.loads((corpus["dataset"] / "observations.json").read_text())
    document["observations"][0]["duration_ms"] += 0.005
    write_json(corpus["dataset"] / "observations.json", document)
    failing = heldout_manifests.guard(corpus["receipt"], corpus["dataset"])
    assert failing["guard"] == "fail"
    assert "training_dataset_hash_changed_since_freeze" in failing["reasons"]
    # A receipt that claims it read outcomes is refused even when the hash matches.
    write_json(corpus["dataset"] / "observations.json", document)
    claims = json.loads(Path(corpus["receipt"]).read_text())
    claims["evaluation_outcomes_read"] = True
    write_json(Path(corpus["receipt"]), claims)
    refused = heldout_manifests.guard(corpus["receipt"], corpus["dataset"])
    assert "frozen_predictions_receipt_claims_outcomes_were_read" in refused["reasons"]
