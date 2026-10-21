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


# ---------------------------------------------------------------------------
# Fused attention: three physical families, never a GEMM
# ---------------------------------------------------------------------------
#: Declared truth of the synthetic attention kernel.  The tile geometry, the
#: efficiencies and the ragged-tile tail are all recovered from durations that
#: were generated by an *independent* implementation of the declared physics.
ATTENTION_TRUTH = {
    "q_tile": 128,
    "kv_tile": 512,
    "eta_compute": 0.55,
    "eta_memory": 0.90,
    "launch_ms": 0.020,
    "tail_tile_ms": 0.0015,
    "dim_tile": 128,
    "workspace_bytes_per_query_head": 64.0,
}
ATTENTION_SHAPES = (
    (1, 4096, 3, 256, True),
    (1, 4096, 2, 256, True),
    (1, 16384, 3, 256, True),
    (1, 16384, 2, 256, False),
    (1, 8192, 3, 256, True),
    (1, 32768, 2, 256, True),
    (1024, 1024, 3, 256, True),
    (4096, 4096, 3, 256, True),
    (1024, 4096, 3, 256, True),
    (4096, 16384, 2, 256, True),
    (128, 128, 3, 256, True),
    (8192, 8192, 3, 256, True),
    (2048, 2048, 2, 256, True),
    (2048, 2048, 2, 64, True),
    (512, 512, 3, 256, False),
    (256, 4096, 3, 256, True),
    (16384, 16384, 2, 128, True),
    (512, 2048, 2, 256, False),
    (1024, 2048, 3, 128, True),
    (128, 512, 2, 64, True),
)


def attention_roofline(truth, q, kv, heads, dim, causal):
    """Independent implementation of the declared attention physics.

    ``visible = kv - (q-1)/2`` for a causal mask (prior context fully visible,
    in-chunk triangular), ``kv`` otherwise; ``4*q*visible*heads*dim`` FLOPs for
    the fused QK+PV pair; bytes = Q read + KV read (K and V on the KV head
    count) + O write + workspace; tiles are q_tile x kv_tile for QK and
    q_tile x dim_tile for PV, wave-quantized on the machine's AIC count.
    """
    visible = kv if (not causal or q <= 1 or kv < q) else max(1.0, kv - (q - 1) / 2.0)
    q_blocks = math.ceil(q / truth["q_tile"])
    kv_blocks = math.ceil(visible / truth["kv_tile"])
    dim_blocks = math.ceil(dim / truth["dim_tile"])
    tiles = q_blocks * kv_blocks + q_blocks * dim_blocks
    blocks = tiles * heads
    waves = math.ceil(blocks / AIC)
    wave_eff = blocks / (waves * AIC)
    # The *declared* FLOPs are mask-agnostic (4*q*kv*heads*dim); the causal mask
    # only removes rows of the score matrix, which the model applies as a
    # visible-key factor.  Feeding an already-discounted count would double-count.
    flops = 4.0 * q * kv * heads * dim
    hbm_bytes = (
        q * heads * dim + kv * heads * (dim + dim) + q * heads * dim
    ) * 2.0 + truth["workspace_bytes_per_query_head"] * q * heads
    compute_ms = (
        4.0 * q * visible * heads * dim
        / (PEAK_FLOPS * truth["eta_compute"] * wave_eff) * 1000.0
    )
    memory_ms = hbm_bytes / (PEAK_BW * truth["eta_memory"]) * 1000.0
    tail_tiles = (
        (1 if int(round(q)) % truth["q_tile"] else 0)
        + (1 if int(math.ceil(visible)) % truth["kv_tile"] else 0)
    ) * heads
    return (
        truth["launch_ms"] + truth["tail_tile_ms"] * tail_tiles + max(compute_ms, memory_ms),
        flops,
        hbm_bytes,
    )


def attention_observations(truth=ATTENTION_TRUTH, shapes=ATTENTION_SHAPES, *, noise=0.01,
                           family="attention_fused", **overrides):
    rows = []
    for index, (q, kv, heads, dim, causal) in enumerate(shapes):
        duration, flops, hbm_bytes = attention_roofline(truth, q, kv, heads, dim, causal)
        row = {
            "condition_key": "attn-%03d" % index,
            "hardware_key": MACHINE["hardware_key"],
            "die": "phy0",
            "arch_family": "TEST",
            "family": family,
            "recipe": "npu_fusion_attention",
            "protocol_signature": "sha256:attention-fixture",
            "run_id": "fixture-attention",
            "split": "train",
            "shape": {"q": q, "kv": kv, "heads": heads, "dim": dim, "causal": causal},
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "duration_ms": duration * (1.0 + noise * math.sin(index * 2.399963)),
            "dtype": "bf16",
            "source_class": "measured",
            "timing_scope": "device_event_warm_operator",
            "raw_event_count": 30,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def attention_op(q, kv, heads, dim, *, causal=False, **overrides):
    value = {
        "family": "attention_fused",
        "recipe": "npu_fusion_attention",
        "shape": {"q": q, "kv": kv, "heads": heads, "dim": dim, "causal": causal},
    }
    value.update(overrides)
    return value


@pytest.fixture(scope="module")
def attention_calibration():
    return fit(attention_observations())


def test_attention_fit_recovers_tiles_efficiency_and_launch(attention_calibration):
    parameters = next(iter(attention_calibration.families.values()))
    assert parameters.model_kind == "attention"
    assert parameters.samples == len(ATTENTION_SHAPES)
    # The tile grid is q_tile x kv_tile, searched on the declared grids.  Which
    # grid point wins is only weakly identified while wave_eff is near 1 (the
    # same reason the dense-GEMM tile is weakly identified), so the assertion is
    # that the tiles come from the declared grids and that the physics recovers.
    assert parameters.block_m in physical.ATTENTION_Q_TILES
    assert parameters.block_n in physical.ATTENTION_KV_TILES
    assert parameters.model_form == "wave_max"
    assert parameters.eta_compute == pytest.approx(ATTENTION_TRUTH["eta_compute"], rel=0.05)
    assert parameters.eta_memory == pytest.approx(ATTENTION_TRUTH["eta_memory"], rel=0.05)
    assert parameters.launch_ms == pytest.approx(ATTENTION_TRUTH["launch_ms"], abs=0.005)
    assert parameters.train_wape_pct < 2.0
    assert parameters.cv_wape_pct is not None and parameters.cv_wape_pct < 3.0
    # Every in-corpus geometry is predicted from the physics, not from a table.
    # A single shape can be off by the wave-quantization factor of the tile grid
    # the fit selected (the grid is only weakly identified here); the aggregate
    # duration-weighted error is what the artifact reports.
    absolute = 0.0
    total = 0.0
    for q, kv, heads, dim, causal in ATTENTION_SHAPES:
        truth_ms, _flops, _bytes = attention_roofline(
            ATTENTION_TRUTH, q, kv, heads, dim, causal
        )
        predicted = attention_calibration.predict_op(
            attention_op(q, kv, heads, dim, causal=causal), MACHINE
        )
        assert predicted == pytest.approx(truth_ms, rel=0.15)
        absolute += abs(predicted - truth_ms)
        total += truth_ms
    assert 100.0 * absolute / total < 2.0


def test_attention_causal_mask_reduces_in_chunk_work(attention_calibration):
    full = attention_calibration.predict_op_explained(
        attention_op(4096, 4096, 3, 256, causal=False), MACHINE
    )
    causal = attention_calibration.predict_op_explained(
        attention_op(4096, 4096, 3, 256, causal=True), MACHINE
    )
    assert full.status == causal.status == "predicted"
    # kv == q: the prior context is empty and only the in-chunk triangle is
    # visible, so the causal work is (q+1)/2 keys per query, not q.
    assert causal.breakdown["visible_kv"] == pytest.approx((4096 + 1) / 2.0, rel=1e-6)
    assert full.breakdown["visible_kv"] == 4096.0
    assert causal.breakdown["flops_effective"] == pytest.approx(
        0.5 * full.breakdown["flops_effective"], rel=1e-3
    )
    # Decode sees the whole cache even under a causal mask.
    decode = attention_calibration.predict_op_explained(
        attention_op(1, 4096, 3, 256, causal=True), MACHINE
    )
    assert decode.breakdown["visible_kv"] == 4096.0


def test_attention_never_maps_onto_gemm(attention_calibration):
    # No declared alias turns an attention family into a GEMM.
    assert not (set(physical.FAMILY_ALIASES) & set(physical.ATTENTION_KINDS))
    assert not (set(physical.FAMILY_ALIASES.values()) & set(physical.ATTENTION_KINDS))
    # The attention fit is its own model kind and creates no gemm family.
    parameters = next(iter(attention_calibration.families.values()))
    assert parameters.model_kind == "attention"
    assert not any(item.model_kind == "gemm" for item in attention_calibration.families.values())
    assert "gemm" not in {item.family for item in attention_calibration.families.values()}
    # A GEMM-only artifact refuses every attention family by name; it does not
    # fall back to the GEMM row even though the shapes are rectangular.
    gemm_only = fit(observations())
    for family in physical.ATTENTION_KINDS:
        explained = gemm_only.predict_op_explained(
            {"family": family, "recipe": "npu_fusion_attention",
             "shape": {"q": 1024, "kv": 1024, "heads": 3, "dim": 256}}, MACHINE
        )
        assert explained.duration_ms is None
        assert explained.reason == "unmodeled_family"
    # An attention observation without its own q/kv axes is rejected, never
    # silently reinterpreted as an m/n/k GEMM.
    with pytest.raises(physical.PhysicalCalibrationError, match="attention_observation_missing_q_kv"):
        physical.Observation.from_mapping(
            dict(attention_observations()[0], shape={"m": 1, "n": 8192, "k": 256}),
            model_kind="attention",
        )
    # And an attention family fitted on q/kv shapes does not answer for a
    # GEMM-shaped query.
    missing = attention_calibration.predict_op_explained(
        {"family": "attention_fused", "recipe": "npu_fusion_attention",
         "shape": {"m": 1024, "n": 2048, "k": 2048}}, MACHINE
    )
    assert missing.duration_ms is None
    assert missing.reason == "missing_shape_q_kv"


def test_attention_ragged_tile_tail_is_selected_when_it_costs_real_time():
    # A per-tile tail is forward-selected only when the ragged q/kv edge costs
    # measurable time.  Make it dominant (20 us per partial tile) and check that
    # it is adopted, recovered and that it pays on train-only CV.
    truth = dict(ATTENTION_TRUTH, tail_tile_ms=0.020, launch_ms=0.0)
    shapes = tuple(
        (q, kv, 3, 256, True)
        for q, kv in ((33, 33), (65, 65), (97, 129), (129, 257), (257, 513),
                      (513, 1025), (1025, 2049), (2049, 4097), (33, 1025),
                      (65, 4097), (97, 513), (129, 2049))
    )
    calibration = fit(attention_observations(truth=truth, shapes=shapes))
    parameters = next(iter(calibration.families.values()))
    assert parameters.tail["tile"] is True
    assert parameters.tail_tile_ms == pytest.approx(truth["tail_tile_ms"], rel=0.10)
    trials = parameters.model_selection["tails"]["trials"]
    base = next(item for item in trials if not item["tail"]["tile"])
    adopted = next(item for item in trials if item["tail"]["tile"])
    assert adopted["adopted"] is True
    assert adopted["cv_wape_pct"] < base["cv_wape_pct"] - physical.MIN_TAIL_IMPROVEMENT_PCT


def test_attention_block_interval_and_device_event_keep_their_own_intercepts():
    device = attention_observations()
    block = attention_observations()
    for index, row in enumerate(block):
        row["condition_key"] = "block-%03d" % index
        row["timing_scope"] = physical.ATTENTION_BLOCK_SCOPE
        # A completion-clock block interval carries the host/allocation gap: the
        # same kernel is slower by that gap, and the gap belongs to the
        # intercept, not to the fitted efficiency.
        row["duration_ms"] = row["duration_ms"] + 0.025
    calibration = fit(device + block)
    records = [
        item for item in calibration.families.values() if item.model_kind == "attention"
    ]
    assert records
    # Either two independent per-scope rows or one shared-efficiency row with a
    # launch intercept per scope; both must keep the intercepts separate.
    launch_by_scope: dict[str, float] = {}
    eta_by_scope: dict[str, float] = {}
    for item in records:
        if item.fit_level == "family_scope_shared":
            launch_by_scope.update(item.launch_ms_by_scope)
            eta_by_scope.update({scope: item.eta_compute for scope in item.launch_ms_by_scope})
        else:
            launch_by_scope[item.timing_scope] = item.launch_ms
            eta_by_scope[item.timing_scope] = item.eta_compute
    assert set(launch_by_scope) == {
        "device_event_warm_operator", physical.ATTENTION_BLOCK_SCOPE,
    }
    # The intercepts differ by the completion-clock gap; the efficiencies are the
    # same physical kernel (shared by construction, or fitted separately).
    assert launch_by_scope[physical.ATTENTION_BLOCK_SCOPE] > (
        launch_by_scope["device_event_warm_operator"] + 0.015
    )
    assert eta_by_scope[physical.ATTENTION_BLOCK_SCOPE] == pytest.approx(
        eta_by_scope["device_event_warm_operator"], rel=0.15
    )
    for scope in launch_by_scope:
        explained = calibration.predict_op_explained(
            attention_op(1024, 4096, 3, 256, timing_scope=scope), MACHINE
        )
        assert explained.status == "predicted"
        assert explained.breakdown["launch_ms"] == pytest.approx(launch_by_scope[scope])
    # A scope the artifact did not fit is unmodeled, never answered from another
    # scope's intercept.
    unknown = calibration.predict_op_explained(
        attention_op(1024, 4096, 3, 256,
                     timing_scope="device_event_warm_operator_batched_amortized"),
        MACHINE,
    )
    assert unknown.duration_ms is None


def test_attention_below_minimum_samples_is_unmodeled():
    calibration = fit(attention_observations(shapes=ATTENTION_SHAPES[:4]))
    assert calibration.families == {}
    coverage = calibration.provenance["family_coverage"]["attention_fused"]
    assert coverage["status"] == "unmodeled_insufficient_samples"
    assert attention_op(1024, 1024, 3, 256) and calibration.predict_op(
        attention_op(1024, 1024, 3, 256), MACHINE
    ) is None


# ---------------------------------------------------------------------------
# Grouped matmul (GMM): per-expert batched GEMM, never the dense-GEMM fit
# ---------------------------------------------------------------------------
GROUPED_TRUTH = {
    "block_m": 64,
    "block_n": 128,
    "eta_compute": 0.60,
    "eta_memory": 0.90,
    "launch_ms": 0.020,
    "per_expert_ms": 0.0015,
    "expert_scaling": 1.0,
}
#: A mix that binds both branches: large-expert / few-token rows are weight-bound,
#: many-token / small-weight rows are compute-bound, so eta_compute and the
#: per-expert cost are separately identified.
GROUPED_SHAPES = (
    (256, 8, 2048, 1024), (1024, 8, 2048, 1024), (4096, 8, 2048, 1024),
    (16384, 8, 2048, 1024), (256, 32, 2048, 1024), (1024, 32, 2048, 1024),
    (4096, 32, 2048, 1024), (16384, 32, 2048, 1024), (256, 128, 2048, 1024),
    (1024, 128, 2048, 1024), (4096, 128, 2048, 1024), (16384, 128, 2048, 1024),
    (64, 8, 7168, 7168), (512, 8, 7168, 7168), (2048, 8, 7168, 7168),
    (64, 32, 7168, 7168), (512, 32, 7168, 7168), (2048, 32, 7168, 7168),
    (64, 128, 7168, 7168), (512, 128, 7168, 7168), (2048, 128, 7168, 7168),
    (64, 256, 7168, 7168), (512, 256, 7168, 7168), (2048, 256, 7168, 7168),
)


def grouped_roofline(truth, tokens, experts, n, k):
    """Independent implementation of the declared grouped-matmul physics.

    One GEMM per expert over ``tokens_per_expert x k`` against ``k x n``, with
    ``experts * ceil(tpe/m_tile) * ceil(n/n_tile)`` cube tiles, padded FLOPs,
    ``2*tokens*n*k`` declared FLOPs and the declared bytes convention.
    """
    tokens_per_expert = tokens / experts
    rows = math.ceil(tokens_per_expert / truth["block_m"])
    cols = math.ceil(n / truth["block_n"])
    tiles = experts * rows * cols
    waves = math.ceil(tiles / AIC)
    wave_eff = tiles / (waves * AIC)
    flops_eff = 2.0 * experts * rows * truth["block_m"] * n * k
    hbm_bytes = (experts * n * k + tokens * k + tokens * n) * 2.0
    compute_ms = flops_eff / (PEAK_FLOPS * truth["eta_compute"] * wave_eff) * 1000.0
    memory_ms = hbm_bytes / (PEAK_BW * truth["eta_memory"]) * 1000.0
    duration = (
        truth["launch_ms"]
        + truth["per_expert_ms"] * experts ** truth["expert_scaling"]
        + max(compute_ms, memory_ms)
    )
    return duration, 2.0 * tokens * n * k, hbm_bytes


def grouped_observations(truth=GROUPED_TRUTH, shapes=GROUPED_SHAPES, *, noise=0.01, **overrides):
    rows = []
    for index, (tokens, experts, n, k) in enumerate(shapes):
        duration, flops, hbm_bytes = grouped_roofline(truth, tokens, experts, n, k)
        row = {
            "condition_key": "gmm-%03d" % index,
            "hardware_key": MACHINE["hardware_key"],
            "die": "phy0",
            "arch_family": "TEST",
            "family": "moe_gate_up",
            "recipe": "npu_grouped_matmul_list_bf16",
            "protocol_signature": "sha256:grouped-fixture",
            "run_id": "fixture-grouped",
            "split": "train",
            "shape": {"tokens": tokens, "experts": experts, "n": n, "k": k},
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "duration_ms": duration * (1.0 + noise * math.sin(index * 2.399963)),
            "dtype": "bf16",
            "source_class": "measured",
            "timing_scope": "device_event_warm_operator",
            "raw_event_count": 30,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def grouped_op(tokens, experts, n, k, **overrides):
    value = {
        "family": "moe_gate_up",
        "recipe": "npu_grouped_matmul_list_bf16",
        "shape": {"tokens": tokens, "experts": experts, "n": n, "k": k},
    }
    value.update(overrides)
    return value


@pytest.fixture(scope="module")
def grouped_calibration():
    return fit(grouped_observations())


def test_grouped_fit_recovers_the_per_expert_cost_and_the_dense_baseline(grouped_calibration):
    parameters = next(iter(grouped_calibration.families.values()))
    assert parameters.model_kind == "grouped_gemm"
    assert parameters.eta_compute == pytest.approx(GROUPED_TRUTH["eta_compute"], rel=0.05)
    assert parameters.eta_memory == pytest.approx(GROUPED_TRUTH["eta_memory"], rel=0.05)
    assert parameters.launch_ms == pytest.approx(GROUPED_TRUTH["launch_ms"], abs=0.003)
    assert parameters.per_expert_ms == pytest.approx(GROUPED_TRUTH["per_expert_ms"], rel=0.10)
    assert parameters.expert_scaling == GROUPED_TRUTH["expert_scaling"]
    assert parameters.activation_read is True
    assert parameters.train_wape_pct < 2.0
    assert parameters.cv_wape_pct is not None and parameters.cv_wape_pct < 3.0
    for tokens, experts, n, k in GROUPED_SHAPES:
        truth_ms, _flops, _bytes = grouped_roofline(GROUPED_TRUTH, tokens, experts, n, k)
        predicted = grouped_calibration.predict_op(
            grouped_op(tokens, experts, n, k), MACHINE
        )
        assert predicted == pytest.approx(truth_ms, rel=0.05)
    # The grouped fit reports its comparison against the rejected dense-GEMM
    # transfer on the SAME rows and folds; it must win by a wide margin because
    # a dense GEMM reads the weight matrix once and has no per-expert dispatch.
    transfer = parameters.model_selection["dense_gemm_transfer"]
    assert transfer is not None
    assert transfer["beats_dense_transfer"] is True
    assert transfer["cv_wape_pct"] > 3.0 * parameters.cv_wape_pct
    assert "dense_gemm_transfer" in transfer["model"]


def test_grouped_never_reuses_the_dense_gemm_fit():
    # A GEMM-only artifact refuses the grouped family by name ...
    gemm_only = fit(observations())
    explained = gemm_only.predict_op_explained(
        grouped_op(2048, 32, 7168, 7168), MACHINE
    )
    assert explained.duration_ms is None
    assert explained.reason == "unmodeled_family"
    # ... and a grouped observation without its expert axes is rejected rather
    # than flattened into an m/n/k GEMM.
    with pytest.raises(physical.PhysicalCalibrationError, match="grouped_observation_missing"):
        physical.Observation.from_mapping(
            dict(grouped_observations()[0], shape={"m": 2048, "n": 7168, "k": 7168}),
            model_kind="grouped_gemm",
        )
    # A grouped family does not answer a GEMM-shaped query either.
    grouped = fit(grouped_observations())
    missing = grouped.predict_op_explained(
        {"family": "moe_gate_up", "recipe": "npu_grouped_matmul_list_bf16",
         "shape": {"m": 2048, "n": 7168, "k": 7168}}, MACHINE
    )
    assert missing.duration_ms is None
    assert missing.reason == "missing_shape_tokens_experts_n_k"


def test_grouped_expert_axis_is_reported_unmodeled_at_a_single_expert_count():
    # A corpus measured at one expert count cannot identify the expert axis: the
    # per-expert term collapses into the intercept, and the family is flagged.
    shapes = tuple((tokens, 8, n, k) for tokens, _e, n, k in GROUPED_SHAPES[:12])
    calibration = fit(grouped_observations(shapes=shapes))
    parameters = next(iter(calibration.families.values()))
    assert parameters.per_expert_ms is None
    assert parameters.expert_scaling is None
    assert parameters.identified["expert_scaling"] is False
    assert parameters.quality == "low_confidence"
    assert "grouped_expert_axis_unmodeled_low_confidence" in parameters.warnings
    # The per-expert term is genuinely absent, so the prediction is the launch +
    # max(compute, memory) model -- it does not borrow a per-expert cost.
    assert parameters.to_dict()["per_expert_ms"] is None


def test_grouped_and_attention_families_round_trip_through_the_artifact():
    attention = fit(attention_observations())
    grouped = fit(grouped_observations())
    merged = physical.PhysicalCalibration(
        hardware_key=attention.hardware_key,
        arch_family=attention.arch_family,
        machine=attention.machine,
        families=dict(attention.families, **grouped.families),
        fit_config=dict(attention.fit_config),
        provenance=dict(attention.provenance),
        unmodeled=attention.unmodeled,
    )
    document = merged.to_dict()
    reloaded = physical.PhysicalCalibration.from_mapping(json.loads(json.dumps(document)))
    assert reloaded.to_dict() == document
    families = {item.family for item in reloaded.families.values()}
    assert {"attention_fused", "moe_gate_up"} <= families
    # Each family keeps its own physics after the round trip.
    assert reloaded.predict_op(grouped_op(2048, 32, 7168, 7168), MACHINE) is not None
    assert reloaded.predict_op(attention_op(1024, 4096, 3, 256), MACHINE) is not None


# ---------------------------------------------------------------------------
# Absorbed MLA, expert-count launch cap, topology lookup
# ---------------------------------------------------------------------------
def test_absorbed_mla_uses_the_measured_v_dim_not_the_qk_dim():
    # MLA absorbs the value width: qk_dim 576, v_dim 512.  A model that assumed
    # one "dim" would invent 12% more PV work than the kernel executes.
    truth = dict(ATTENTION_TRUTH, eta_compute=0.55)
    shapes = (
        (4096, 4096, 64, 576), (16384, 16384, 64, 576), (1024, 1024, 128, 576),
        (4096, 4096, 128, 576), (1, 16384, 64, 576), (1, 65536, 64, 576),
        (4096, 4096, 96, 192), (1024, 1024, 96, 192),
        (4096, 4096, 8, 256), (1024, 1024, 8, 256),
        (2048, 2048, 16, 128), (512, 512, 16, 128),
    )
    rows = []
    for index, (q, kv, heads, qk_dim) in enumerate(shapes):
        v_dim = 512 if qk_dim == 576 else max(64, qk_dim - 64)
        # Independent physics: QK contracts qk_dim and PV contracts v_dim.
        visible = kv if not (q > 1 and kv >= q) else max(1.0, kv - (q - 1) / 2.0)
        flops_eff = 2.0 * q * visible * heads * (qk_dim + v_dim)
        hbm_bytes = (
            q * heads * qk_dim + kv * heads * (qk_dim + v_dim) + q * heads * v_dim
        ) * 2.0 + 64.0 * q * heads
        compute_ms = flops_eff / (PEAK_FLOPS * truth["eta_compute"]) * 1000.0
        memory_ms = hbm_bytes / (PEAK_BW * truth["eta_memory"]) * 1000.0
        duration = truth["launch_ms"] + max(compute_ms, memory_ms)
        rows.append(
            {
                "condition_key": "mla-%03d" % index,
                "hardware_key": MACHINE["hardware_key"],
                "die": "phy0",
                "arch_family": "TEST",
                "family": "attention_fused",
                "recipe": "npu_fusion_attention",
                "protocol_signature": "sha256:mla-fixture",
                "run_id": "fixture-mla",
                "split": "train",
                "shape": {"q": q, "kv": kv, "heads": heads, "kv_heads": 1,
                          "dim": qk_dim, "v_dim": v_dim, "causal": q > 1},
                "flops": 2.0 * q * kv * heads * (qk_dim + v_dim),
                "hbm_bytes": hbm_bytes,
                "duration_ms": duration,
                "dtype": "bf16",
                "source_class": "measured",
                "timing_scope": "device_event_warm_operator",
                "raw_event_count": 30,
            }
        )
    calibration = fit(rows)
    parameters = next(iter(calibration.families.values()))
    predicted = calibration.predict_op(
        {"family": "attention_fused", "recipe": "npu_fusion_attention",
         "timing_scope": "device_event_warm_operator",
         "shape": {"q": 4096, "kv": 4096, "heads": 64, "kv_heads": 1,
                   "dim": 576, "v_dim": 512, "causal": True}},
        MACHINE,
    )
    index = 0
    truth_ms = rows[index]["duration_ms"]
    assert predicted == pytest.approx(truth_ms, rel=0.05)
    # A qk_dim-only model would be off by the v_dim share (512/1088 = 47% of
    # the fused FLOPs); the prediction must not be that.
    only_qk = predicted * 576.0 / 1088.0
    assert abs(predicted - only_qk) > 0.1 * predicted


def test_grouped_expert_cap_turns_extra_experts_into_extra_launches():
    calibration = fit(grouped_observations())
    parameters = next(iter(calibration.families.values()))
    # The declared tensorsList cap is 128 experts per call.
    assert physical.GROUPED_LAUNCH_CAPACITY == 128
    within = calibration.predict_op_explained(grouped_op(2048, 128, 7168, 7168), MACHINE)
    over = calibration.predict_op_explained(grouped_op(2048, 256, 7168, 7168), MACHINE)
    assert within.breakdown["launches"] == 1.0
    assert over.breakdown["launches"] == 2.0
    three = calibration.predict_op_explained(grouped_op(2048, 384, 7168, 7168), MACHINE)
    assert three.breakdown["launches"] == 3.0
    # The per-launch cost is either fitted or pinned to the measured dispatch,
    # never silently zero when a corpus contains multi-launch rows.
    assert parameters.per_launch_ms is not None
    assert over.breakdown["per_launch_ms"] == pytest.approx(parameters.per_launch_ms)


def test_attention_topology_split_is_adopted_only_when_it_pays():
    # Two declared topologies with genuinely different cube efficiencies: the
    # pooled row cannot express both, so the split must win on train-only CV.
    shapes = (
        (1024, 1024, 8, 2), (2048, 2048, 8, 2), (4096, 4096, 8, 2),
        (1024, 4096, 8, 2), (2048, 8192, 8, 2), (4096, 16384, 8, 2),
        (1024, 1024, 4, 4), (2048, 2048, 4, 4), (4096, 4096, 4, 4),
        (1024, 4096, 4, 4), (2048, 8192, 4, 4), (4096, 16384, 4, 4),
    )
    rows = []
    for index, (q, kv, heads, kv_heads) in enumerate(shapes):
        topology = physical.attention_topology(heads, kv_heads)
        truth = dict(ATTENTION_TRUTH)
        truth["eta_compute"] = 0.50 if topology == "gqa" else 0.85
        duration, flops, hbm_bytes = attention_roofline(truth, q, kv, heads, 128, True)
        rows.append(
            {
                "condition_key": "topo-%03d" % index,
                "hardware_key": MACHINE["hardware_key"],
                "die": "phy0",
                "arch_family": "TEST",
                "family": "attention_fused",
                "recipe": "npu_fusion_attention",
                "protocol_signature": "sha256:topo-fixture",
                "run_id": "fixture-topo",
                "split": "train",
                "shape": {"q": q, "kv": kv, "heads": heads, "kv_heads": kv_heads,
                          "dim": 128, "causal": True, "topology": topology},
                "flops": flops,
                "hbm_bytes": hbm_bytes,
                "duration_ms": duration,
                "dtype": "bf16",
                "source_class": "measured",
                "timing_scope": "device_event_warm_operator",
                "raw_event_count": 30,
            }
        )
    calibration = fit(rows)
    attention_rows = [
        item for item in calibration.families.values() if item.model_kind == "attention"
    ]
    # The adoptable axis is the two-class one (GQA vs shared-KV/MLA), because a
    # 2-sample MHA class cannot carry its own fit.
    assert {item.topology for item in attention_rows} == {"gqa", "shared_kv_mla"}
    for item in attention_rows:
        comparison = item.model_selection["topology_split"]
        assert comparison["adopted"] is True
        assert comparison["cv_improvement_pct"] >= physical.MIN_CV_IMPROVEMENT_PCT
    # The split rows carry the topology-dependent efficiency they were fitted on.
    gqa = next(item for item in attention_rows if item.topology == "gqa")
    other = next(item for item in attention_rows if item.topology == "shared_kv_mla")
    assert other.eta_compute == pytest.approx(0.85, rel=0.10)
    assert gqa.eta_compute == pytest.approx(0.50, rel=0.10)
    # A query derives its topology from the head geometry, so the caller does not
    # have to name it ...
    explained = calibration.predict_op_explained(
        {"family": "attention_fused", "recipe": "npu_fusion_attention",
         "timing_scope": "device_event_warm_operator",
         "shape": {"q": 4096, "kv": 4096, "heads": 8, "kv_heads": 2, "dim": 128,
                   "causal": True}},
        MACHINE,
    )
    assert explained.status == "predicted"
    assert explained.parameters["topology"] == "gqa"
    # ... and a topology the artifact never fitted is refused, not borrowed.
    unknown = calibration.predict_op_explained(
        {"family": "attention_fused", "recipe": "npu_fusion_attention",
         "timing_scope": "device_event_warm_operator", "topology": "shared_kv",
         "shape": {"q": 4096, "kv": 4096, "heads": 8, "kv_heads": 2, "dim": 128}},
        MACHINE,
    )
    assert unknown.duration_ms is None
    assert unknown.reason.startswith("unmodeled_topology")


# ---------------------------------------------------------------------------
# Declared timing-scope families
# ---------------------------------------------------------------------------
def test_attention_is_found_from_a_case_declaring_the_generic_per_call_scope():
    # The attention row is fitted at the worker's spelling
    # ``device_event_warm_operator``; a case that declares the generic
    # ``device_event_warm`` must still be answered, and must say so.
    calibration = fit(attention_observations())
    explained = calibration.predict_op_explained(
        attention_op(1024, 4096, 3, 256, timing_scope="device_event_warm"), MACHINE
    )
    assert explained.status == "predicted"
    assert explained.parameters["scope_alias_from"] == "device_event_warm"
    assert explained.parameters["scope_alias_to"] == "device_event_warm_operator"
    assert "scope_alias_from=device_event_warm" in explained.reason
    exact = calibration.predict_op_explained(
        attention_op(1024, 4096, 3, 256, timing_scope="device_event_warm_operator"), MACHINE
    )
    assert exact.parameters["scope_alias_from"] is None
    assert exact.duration_ms == pytest.approx(explained.duration_ms)


def test_gemm_is_found_from_a_case_declaring_the_operator_scope():
    calibration = fit(observations())
    explained = calibration.predict_op_explained(
        op(1024, 2048, 2048, timing_scope="device_event_warm_operator"), MACHINE
    )
    assert explained.status == "predicted"
    assert explained.parameters["scope_alias_to"] == "device_event_warm"
    # A block-interval request prefers the device-event row of the same family.
    block = calibration.predict_op_explained(
        op(1024, 2048, 2048, timing_scope=physical.ATTENTION_BLOCK_SCOPE), MACHINE
    )
    assert block.parameters["scope_alias_to"] == "device_event_warm"


def test_scope_families_never_cross_batched_and_per_call():
    rows = observations()
    batched = copy.deepcopy(rows)
    for index, row in enumerate(batched):
        row["condition_key"] = "batched-%03d" % index
        row["timing_scope"] = "device_event_warm_operator_batched_amortized"
    calibration = fit(rows + batched)
    per_call = calibration.predict_op_explained(
        op(1024, 2048, 2048, timing_scope="device_event_warm_per_call"), MACHINE
    )
    batched_hit = calibration.predict_op_explained(
        op(1024, 2048, 2048,
           timing_scope="device_event_warm_operator_batched_amortized"), MACHINE
    )
    assert per_call.parameters["scope_alias_to"] == "device_event_warm"
    assert batched_hit.parameters["scope_alias_from"] is None
    # A batched request with only per-call rows fitted is unmodeled: the two
    # protocols do not amortise the same work.
    per_call_only = fit(rows)
    refused = per_call_only.predict_op_explained(
        op(1024, 2048, 2048,
           timing_scope="device_event_warm_operator_batched_amortized"), MACHINE
    )
    assert refused.duration_ms is None
    # And a scope whose family cannot be determined is not guessed.
    host = per_call_only.predict_op_explained(
        op(1024, 2048, 2048, timing_scope="host_wall"), MACHINE
    )
    assert host.duration_ms is None


# ---------------------------------------------------------------------------
# Multi-GB working sets: measured streaming rate, no L2 credit
# ---------------------------------------------------------------------------
def test_streaming_working_sets_use_the_measured_rate_not_the_warm_fit(monkeypatch):
    # The streaming rate is declared per hardware key; teach the synthetic part
    # the measured rate so the rule is exercised end to end.
    monkeypatch.setitem(physical.STREAMING_BW_BYTES_PER_S, MACHINE["hardware_key"], 1.01e12)
    # The training corpus never measured a multi-GB working set (its largest is
    # 0.7 GB, 3.5x the declared L2), so a 2 GB decode weight stream must be
    # priced at the declared measured streaming rate and must not be credited
    # with L2 residency.
    calibration = fit(observations())
    parameters = only_family(calibration)
    assert parameters.streaming_measured is False
    big = calibration.predict_op_explained(op(1, 145408, 7168), MACHINE)
    assert big.status == "predicted"
    assert big.breakdown["streaming_regime"] == 1.0
    measured_rate = physical.STREAMING_BW_BYTES_PER_S[MACHINE["hardware_key"]]
    assert big.breakdown["eta_memory_effective"] == pytest.approx(measured_rate / PEAK_BW)
    hbm_bytes = (1 * 7168 + 7168 * 145408 + 1 * 145408) * 2.0
    assert big.breakdown["memory_ms"] == pytest.approx(
        hbm_bytes / measured_rate * 1000.0, rel=1e-9
    )
    assert big.duration_ms == pytest.approx(
        big.breakdown["launch_ms"] + big.breakdown["additive_ms"]
        + max(big.breakdown["compute_ms"], big.breakdown["memory_ms"]),
        rel=1e-9,
    )
    # Below the streaming threshold the fitted (warm) model still applies.
    small = calibration.predict_op_explained(op(1, 8192, 7168), MACHINE)
    assert small.breakdown.get("streaming_regime") is None
    # A corpus that *did* measure the streaming regime keeps its own fitted
    # efficiency there instead of the declared rate.
    heavy_shapes = SHAPES + ((1, 145408, 7168), (2, 145408, 7168), (4, 145408, 7168))
    heavy = fit(observations(shapes=heavy_shapes))
    heavy_parameters = only_family(heavy)
    assert heavy_parameters.streaming_measured is True
    heavy_big = heavy.predict_op_explained(op(1, 145408, 7168), MACHINE)
    assert heavy_big.breakdown.get("streaming_regime") is None
    assert heavy_big.duration_ms == pytest.approx(
        heavy_parameters.launch_ms
        + max(heavy_big.breakdown["compute_ms"], heavy_big.breakdown["memory_ms"]),
        rel=1e-6,
    )
