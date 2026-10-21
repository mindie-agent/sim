"""Tier 0 tests for the fitted physical calibration layer.

Fixtures are in-memory synthetic observations generated from declared parameters
through the same roofline the fitter inverts, so every assertion is about the
fitter / predictor contract, not about a recorded hardware outcome.  No file,
subprocess, network, device or sleep is used.
"""
from __future__ import annotations

import copy
import json
import math

import pytest

from mindie_sim import a3_physical_calibration as physical

AIC = 20
PEAK_FLOPS = 20 * 8192 * 1650e6
PEAK_BW = 1.2e12
L2_BYTES = 201326592

MACHINE = {
    "hardware_key": "Synthetic910_X1",
    "arch_family": "TEST",
    "aic_count": AIC,
    "aiv_count": 40,
    "clock_mhz": 1650.0,
    "ops_per_cycle": 8192.0,
    "peak_bw_bytes_per_s": PEAK_BW,
    "l2_bytes": L2_BYTES,
    "vec_elements_per_cycle_per_aiv": 128.0,
}

TRUTH = {
    "block_m": 128,
    "block_n": 256,
    "eta_compute": 0.55,
    "eta_memory": 0.90,
    "launch_ms": 0.018,
    "memory_model": "hbm",
}

# Both waves of the roofline bind here: (1, 8192, 8192) is memory-bound,
# (1024, 2048, 2048) is compute-bound, and 127/128/129 straddle the tile edge.
SHAPES = (
    (1, 8192, 8192),
    (1, 4096, 4096),
    (8, 4096, 2048),
    (127, 2560, 2048),
    (128, 2560, 2048),
    (129, 2560, 2048),
    (256, 2560, 2048),
    (257, 2560, 2048),
    (512, 2560, 2048),
    (1024, 2048, 2048),
    (2048, 1024, 2048),
    (4096, 2048, 1024),
    (128, 5120, 1024),
    (32, 8192, 1024),
    (16, 8192, 2048),
    (1024, 4096, 4096),
    (1, 16384, 8192),
    (4, 8192, 8192),
    (4096, 4096, 4096),
)


def roofline(truth, m, n, k):
    flops = 2.0 * m * n * k
    hbm_bytes = (m * k + k * n + m * n) * 2.0
    blocks = math.ceil(m / truth["block_m"]) * math.ceil(n / truth["block_n"])
    waves = math.ceil(blocks / AIC)
    wave_eff = blocks / (waves * AIC)
    compute_ms = flops / (PEAK_FLOPS * truth["eta_compute"] * wave_eff) * 1000.0
    if truth.get("memory_model") == "l2_blend":
        l2_part = min(hbm_bytes, truth["l2_usable_bytes"])
        hbm_part = hbm_bytes - l2_part
        memory_ms = (
            hbm_part / (PEAK_BW * truth["eta_memory"]) + l2_part / truth["l2_bw"]
        ) * 1000.0
    else:
        memory_ms = hbm_bytes / (PEAK_BW * truth["eta_memory"]) * 1000.0
    return truth["launch_ms"] + max(compute_ms, memory_ms), flops, hbm_bytes


def observations(truth=TRUTH, shapes=SHAPES, *, noise=0.01, **overrides):
    rows = []
    for index, (m, n, k) in enumerate(shapes):
        duration, flops, hbm_bytes = roofline(truth, m, n, k)
        row = {
            "condition_key": "cond-%03d" % index,
            "hardware_key": MACHINE["hardware_key"],
            "die": "phy0",
            "arch_family": "TEST",
            "family": "gemm",
            "recipe": "torch_mm_preallocated",
            "protocol_signature": "sha256:fixture-a",
            "run_id": "fixture-run",
            "split": "train",
            "shape": {"m": m, "n": n, "k": k},
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "duration_ms": duration * (1.0 + noise * math.sin(index * 2.399963)),
            "dtype": "bf16",
            "source_class": "measured",
            "timing_scope": "device_event_warm",
            "raw_event_count": 30,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def fit(rows, **kwargs):
    return physical.fit_calibration(
        rows,
        MACHINE,
        hardware_key=MACHINE["hardware_key"],
        arch_family="TEST",
        **kwargs,
    )


def only_family(calibration):
    assert len(calibration.families) == 1
    return next(iter(calibration.families.values()))


def op(m, n, k, **overrides):
    value = {"family": "gemm", "recipe": "torch_mm_preallocated", "shape": {"m": m, "n": n, "k": k}}
    value.update(overrides)
    return value


# ---------------------------------------------------------------------------
# Recovery and mechanism
# ---------------------------------------------------------------------------
def test_fit_recovers_declared_parameters_from_synthetic_roofline():
    calibration = fit(observations())
    parameters = only_family(calibration)
    assert parameters.fit_level == "family_recipe_protocol"
    assert parameters.samples == len(SHAPES)
    assert parameters.raw_event_count == 30 * len(SHAPES)
    assert (parameters.block_m, parameters.block_n) == (TRUTH["block_m"], TRUTH["block_n"])
    assert parameters.eta_compute == pytest.approx(TRUTH["eta_compute"], rel=0.08)
    assert parameters.eta_memory == pytest.approx(TRUTH["eta_memory"], rel=0.08)
    assert parameters.launch_ms == pytest.approx(TRUTH["launch_ms"], abs=0.003)
    assert parameters.memory_model == "hbm"
    # 1% deterministic noise: both train and train-only CV must stay small.
    assert parameters.train_wape_pct < 2.0
    assert parameters.cv_wape_pct is not None and parameters.cv_wape_pct < 3.0
    assert parameters.quality == "trusted"


def test_wave_quantization_step_at_the_tile_edge_is_represented():
    calibration = fit(observations())
    parameters = only_family(calibration)
    # 128 rows occupy exactly one 128x256 tile row (wave_eff 0.5); 129 rows need
    # two (wave_eff 1.0), so the wider call is faster per FLOP - the step is in
    # the generated data, so the fit must reproduce it rather than smooth it.
    at_128 = calibration.predict_op(op(128, 2560, 2048), MACHINE)
    at_129 = calibration.predict_op(op(129, 2560, 2048), MACHINE)
    assert at_128 is not None and at_129 is not None
    assert at_129 < 0.85 * at_128
    observed_128 = next(row for row in observations() if row["shape"]["m"] == 128.0 or row["shape"]["m"] == 128)
    observed_129 = next(row for row in observations() if row["shape"]["m"] == 129)
    assert at_128 == pytest.approx(observed_128["duration_ms"], rel=0.05)
    assert at_129 == pytest.approx(observed_129["duration_ms"], rel=0.05)
    assert parameters.block_m == 128


def test_l2_blend_is_selected_only_when_the_data_has_a_residency_step():
    hbm = fit(observations())
    assert only_family(hbm).memory_model == "hbm"

    blended_truth = dict(TRUTH, memory_model="l2_blend", l2_usable_bytes=50331648, l2_bw=2.2e12)
    blended = fit(observations(blended_truth))
    parameters = only_family(blended)
    assert parameters.memory_model == "l2_blend"
    assert parameters.l2_bw_bytes_per_s is not None
    assert parameters.l2_bw_bytes_per_s > PEAK_BW
    assert parameters.l2_bw_bytes_per_s == pytest.approx(blended_truth["l2_bw"], rel=0.15)
    # The plateau rule must not inflate the resident prefix: it is a trade-off
    # partner of eta_memory, and only the smaller end of the plateau is physical.
    assert 0.5 <= parameters.l2_usable_bytes / blended_truth["l2_usable_bytes"] <= 2.0
    assert parameters.eta_memory == pytest.approx(blended_truth["eta_memory"], rel=0.10)
    selection = parameters.memory_model_selection
    assert selection["compared"] is True
    assert selection["chosen"] == "l2_blend"
    assert selection["hbm"]["cv_wape_pct"] > selection["l2_blend"]["cv_wape_pct"]
    assert parameters.cv_wape_pct < 1.0


# ---------------------------------------------------------------------------
# Train-only discipline
# ---------------------------------------------------------------------------
def test_heldout_split_never_reaches_the_fit():
    train_rows = observations()
    heldout_rows = copy.deepcopy(train_rows[:5])
    for index, row in enumerate(heldout_rows):
        row["split"] = "heldout"
        row["condition_key"] = "heldout-%03d" % index
        row["duration_ms"] *= 4.0  # would visibly move every parameter if used
    clean = fit(train_rows)
    polluted = fit(train_rows + heldout_rows)
    assert only_family(clean).to_dict() == only_family(polluted).to_dict()
    assert polluted.provenance["rejections"]["by_reason"]["non_train_split:heldout"] == len(heldout_rows)
    assert polluted.provenance["condition_count"] == len(train_rows)
    assert polluted.provenance["split_policy"] == "train_only"


def test_declared_heldout_guard_blocks_a_physical_identity_collision():
    rows = observations()
    guard = physical.HeldoutGuard(
        condition_keys=frozenset({"declared-condition"}),
        identities=frozenset({physical.observation_identity("gemm", rows[3]["shape"])}),
        sources=("declaration",),
        declared_conditions=1,
    )
    with pytest.raises(physical.PhysicalCalibrationError, match="heldout_identity_in_training_set"):
        fit(rows, heldout_guard=guard)
    # A disjoint declaration stays clean and is recorded as such.
    clean_guard = physical.HeldoutGuard(
        condition_keys=frozenset({"declared-condition"}),
        identities=frozenset({physical.observation_identity("gemm", {"m": 7, "n": 11, "k": 13})}),
        sources=("declaration",),
        declared_conditions=1,
    )
    calibration = fit(rows, heldout_guard=clean_guard)
    assert calibration.provenance["heldout_guard"]["status"] == "clean"
    assert calibration.provenance["heldout_guard"]["overlap"] == []


def test_ambiguous_identity_is_rejected_not_merged():
    rows = observations()
    ambiguous = [
        dict(rows[0], condition_key="no-protocol", protocol_signature=""),
        dict(rows[1], condition_key="other-hw", hardware_key="Ascend910_OTHER"),
        dict(rows[2], condition_key="synthetic", source_class="synthetic"),
        dict(rows[3], condition_key="host-timed", timing_scope="host_wall"),
        dict(rows[4], condition_key="fp16", dtype="fp16"),
    ]
    calibration = fit(rows + ambiguous)
    reasons = calibration.provenance["rejections"]["by_reason"]
    assert reasons["ambiguous_protocol_signature"] == 1
    assert reasons["hardware_key_mismatch:Ascend910_OTHER"] == 1
    assert reasons["non_measured_source_class:synthetic"] == 1
    assert reasons["unsupported_timing_scope:host_wall"] == 1
    assert reasons["dtype_outside_fit:fp16"] == 1
    assert calibration.provenance["condition_count"] == len(rows)
    assert calibration.provenance["protocol_signature_count"] == 1

    with pytest.raises(physical.PhysicalCalibrationError, match="no_train_observations_after_rejection"):
        physical.fit_calibration(
            [dict(rows[0], protocol_signature="")],
            MACHINE,
            hardware_key=MACHINE["hardware_key"],
        )


def test_protocols_are_never_pooled_across_a_family():
    rows = observations()
    other = observations(shapes=SHAPES[:8])
    for index, row in enumerate(other):
        row["protocol_signature"] = "sha256:fixture-b"
        row["condition_key"] = "proto-b-%03d" % index
    calibration = fit(rows + other)
    assert len(calibration.families) == 2
    signatures = {item.protocol_signature for item in calibration.families.values()}
    assert signatures == {"sha256:fixture-a", "sha256:fixture-b"}
    separation = [item for item in calibration.unmodeled if item["reason"] == "protocol_groups_kept_separate"]
    assert separation and separation[0]["family"] == "gemm"


def test_small_group_falls_back_to_a_pooled_family_fit():
    rows = observations()
    small = copy.deepcopy(rows[:3])
    for index, row in enumerate(small):
        row["recipe"] = "torch_mm_out"
        row["condition_key"] = "out-%03d" % index
    calibration = fit(rows + small)
    pooled = calibration.parameters_for("gemm", "torch_mm_out")
    assert pooled is not None
    assert pooled.recipe == "*"
    assert pooled.fit_level == "family_protocol"
    assert pooled.samples == len(SHAPES) + len(small)


def test_group_below_minimum_samples_is_declared_unmodeled():
    rows = observations(shapes=SHAPES[:3])
    calibration = fit(rows, min_samples=6)
    assert calibration.families == {}
    reasons = {item["reason"] for item in calibration.unmodeled}
    assert "insufficient_samples" in reasons
    assert calibration.predict_op(op(128, 2560, 2048), MACHINE) is None


# ---------------------------------------------------------------------------
# Prediction contract
# ---------------------------------------------------------------------------
def test_prediction_matches_the_declared_roofline_and_reports_a_breakdown():
    calibration = fit(observations())
    explained = calibration.predict_op_explained(op(1024, 2048, 2048), MACHINE)
    assert explained.status == "predicted"
    assert explained.reason == "physical_calibration"
    truth_ms, flops, hbm_bytes = roofline(TRUTH, 1024, 2048, 2048)
    assert explained.duration_ms == pytest.approx(truth_ms, rel=0.05)
    assert explained.breakdown["blocks"] == 8 * 8
    assert explained.breakdown["waves"] == math.ceil(64 / AIC)
    assert explained.breakdown["compute_ms"] > explained.breakdown["memory_ms"]
    assert explained.parameters["samples"] == len(SHAPES)


def test_unmodeled_family_and_dtype_return_none_with_an_explicit_reason():
    calibration = fit(observations())
    unseen = calibration.predict_op_explained({"family": "attention_qk", "recipe": "torch_sdpa",
                                               "shape": {"m": 1, "n": 8192, "k": 128}}, MACHINE)
    assert unseen.duration_ms is None
    assert unseen.status == "unmodeled"
    assert unseen.reason == "unmodeled_family"
    wrong_recipe = calibration.predict_op_explained(
        {"family": "gemm", "recipe": "torch_baddbmm", "shape": {"m": 8, "n": 8, "k": 8}}, None)
    assert wrong_recipe.duration_ms is None
    assert wrong_recipe.reason == "unmodeled_recipe_or_protocol"
    wrong_dtype = calibration.predict_op_explained(op(128, 2560, 2048, dtype="fp32"), MACHINE)
    assert wrong_dtype.duration_ms is None
    assert wrong_dtype.reason.startswith("dtype_outside_fit")
    missing_shape = calibration.predict_op_explained(
        {"family": "gemm", "recipe": "torch_mm_preallocated", "shape": {"m": 128}}, MACHINE)
    assert missing_shape.reason == "missing_shape_m_n_k"


def test_machine_hardware_key_mismatch_is_refused():
    calibration = fit(observations())
    foreign = dict(MACHINE, hardware_key="Ascend910_9362")
    explained = calibration.predict_op_explained(op(128, 2560, 2048), foreign)
    assert explained.duration_ms is None
    assert explained.reason == "hardware_key_mismatch"
    assert calibration.predict_op(op(128, 2560, 2048), foreign) is None
    # An artifact that carries a contradictory machine block is rejected on load.
    document = calibration.to_dict()
    document["machine"]["hardware_key"] = "Ascend910_9362"
    with pytest.raises(physical.PhysicalCalibrationError, match="artifact_hardware_key_mismatch"):
        physical.PhysicalCalibration.from_mapping(document)


def test_fitted_parameters_stay_inside_the_declared_bounds():
    calibration = fit(observations())
    parameters = only_family(calibration)
    assert 0.0 <= parameters.launch_ms <= physical.LAUNCH_BOUNDS_MS[1]
    assert physical.ETA_BOUNDS[0] < parameters.eta_compute <= 1.0
    assert physical.ETA_BOUNDS[0] < parameters.eta_memory <= 1.0
    assert parameters.block_m in physical.BLOCK_M_CHOICES
    assert parameters.block_n in physical.BLOCK_N_CHOICES
    assert 0 <= parameters.l2_usable_bytes <= L2_BYTES
    assert parameters.identified["eta_compute"] is True
    assert parameters.identified["eta_memory"] is True


def test_provenance_and_artifact_round_trip():
    calibration = fit(observations(), git_head="0" * 40, source_artifacts=("fixture",))
    document = calibration.to_dict()
    reloaded = physical.PhysicalCalibration.from_mapping(document)
    assert reloaded.to_dict() == document
    assert reloaded.to_dict() == json.loads(json.dumps(document))
    provenance = document["provenance"]
    assert provenance["condition_count"] == len(SHAPES)
    assert provenance["raw_event_count"] == 30 * len(SHAPES)
    assert provenance["source_runs"] == ["fixture-run"]
    assert provenance["git_head"] == "0" * 40
    assert provenance["source_artifacts"] == ["fixture"]
    assert provenance["training_hash"].startswith(tuple("0123456789abcdef"))
    assert len(provenance["training_hash"]) == 64
    # Same observations, same parameters and the same training hash.
    assert fit(observations()).to_dict()["provenance"]["training_hash"] == provenance["training_hash"]
    # A heldout-only change to a single duration must change the hash.
    perturbed = observations()
    perturbed[0]["duration_ms"] *= 1.5
    assert fit(perturbed).provenance["training_hash"] != provenance["training_hash"]
    for key in ("schema_version", "hardware_key", "arch_family", "machine", "families", "unmodeled"):
        assert key in document
    assert document["machine"]["peak_flops"] == pytest.approx(PEAK_FLOPS)
    assert document["machine"]["aic_count"] == AIC
