"""Tier 0 tests for the fitted physical-calibration integration.

The artifact is a hand-declared *synthetic* calibration (a wave-quantized GEMM
family and a flat vector family) built in memory: every assertion here is about
the integration contract -- which duration wins, what the fallback reason is,
what provenance is reported -- never about a recorded hardware outcome.  No
subprocess, network, device, sleep or measurement file is used; the CLI is
driven in-process through ``mindie_sim.__main__.main``.

Two measurement protocols appear deliberately: ``device_event_warm`` (per-call
interval, A3 test set 1) and ``device_event_warm_batched`` (kernel service inside
a batched graph, A2's primary corpus).  They carry different launch intercepts,
so a scope the artifact does not model must fall back explicitly instead of
being answered from another scope's row.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim.a3_cost import (
    PHYSICAL_CALIBRATED, PHYSICAL_LOW_CONFIDENCE, PHYSICAL_TRANSFER,
    PHYSICAL_UNMODELED, PredictiveCostModel, physical_transfer_target,
)
from mindie_sim.a3_framework import predict
from mindie_sim.a3_ops import OpSpec
from mindie_sim.a3_physical_calibration import (
    ACCEPTED_TIMING_SCOPES, SCHEMA_VERSION, PhysicalCalibration,
)
from mindie_sim.a3_types import A3InputError, MachineSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CASE = ROOT / "examples" / "a3_dense_predict.json"
#: The declared A3 model the production loop predicts (hardware key
#: ``Ascend910_9362``) and the *published* artifact for that key.
PRODUCTION_CASE = ROOT / "examples" / "a3_real_hardware_predict.json"
PUBLISHED_A3_ARTIFACT = ROOT / ".workstreams" / "model-physical" / "a3-physical-calibration.json"
PER_CALL = "device_event_warm"
BATCHED = "device_event_warm_batched"
#: The other per-call spelling the artifact fits the attention/grouped rows at.
OPERATOR_SCOPE = "device_event_warm_operator"

#: Artifact machine block.  ``peak_flops`` is derived (aic x ops/cycle x clock),
#: and the integration only passes ``hardware_key``, so the fitted number comes
#: from this block and not from the simulated machine's declared peaks.
ARTIFACT_MACHINE = {
    "arch_family": "TEST",
    "aic_count": 20,
    "aiv_count": 40,
    "clock_mhz": 1650.0,
    "ops_per_cycle": 8192.0,
    "peak_bw_bytes_per_s": 1.2e12,
    "l2_bytes": 64 * 1024 * 1024,
    "vec_elements_per_cycle_per_aiv": 128.0,
}
AIC = ARTIFACT_MACHINE["aic_count"]
VEC_PEAK = ARTIFACT_MACHINE["aiv_count"] * ARTIFACT_MACHINE["vec_elements_per_cycle_per_aiv"] * 1.65e9
GEMM_LAUNCH_MS = 0.0177
GEMM_SAMPLES = 367
VECTOR_LAUNCH_MS = 0.0242
VECTOR_SAMPLES = 45


def _family_row(
    family: str,
    *,
    recipe: str,
    model_kind: str,
    timing_scope: str,
    launch_ms: float,
    quality: str,
    samples: int,
) -> dict:
    """One fitted family row; only the fields the predictor reads vary."""
    cube = model_kind in {"gemm", "attention", "grouped_gemm"}
    return {
        "family": family,
        "recipe": recipe,
        "protocol_signature": "",
        "timing_scope": timing_scope,
        "model_kind": model_kind,
        "model_form": "wave_max" if cube else "flat_additive",
        "fit_level": "family_protocol",
        "block_m": 32 if cube else None,
        "block_n": 256 if cube else None,
        "launch_ms": launch_ms,
        "eta_compute": 0.837 if cube else None,
        "eta_memory": 0.814 if cube else 1.0,
        "vec_ops_per_s": None if cube else VEC_PEAK,
        "memory_model": "hbm",
        "l2_usable_bytes": 0,
        "l2_bw_bytes_per_s": None,
        "compute_dtype": "bf16",
        "samples": samples,
        "raw_event_count": samples * 6,
        "train_wape_pct": 16.6 if cube else 17.6,
        "cv_wape_pct": 17.65 if cube else 20.3,
        "median_ape_pct": 20.3,
        "p90_ape_pct": 51.8,
        "max_ape_pct": 85.2,
        "binding_share": {"compute": 0.5, "memory": 0.5},
        "binding_counts": {"compute": samples // 2, "memory": samples // 2},
        "identified": {"launch": True, "tile_geometry": cube, "eta_compute": cube},
        "pinned_parameters": [] if cube else ["eta_memory", "vec_ops_per_s"],
        "intervals": {},
        "breakdown": {},
        "model_selection": {},
        "memory_model_selection": {},
        "quality": quality,
        "warnings": [] if quality == "trusted" else ["parameters_unidentified"],
    }


def physical_artifact(
    *,
    hardware_key: str = "Synthetic910_X1",
    timing_scope: str = PER_CALL,
    launch_shift_ms: float = 0.0,
) -> dict:
    """Two-family artifact: a trusted GEMM fit and an unidentified vector fit."""
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_key": hardware_key,
        "arch_family": ARTIFACT_MACHINE["arch_family"],
        "machine": {**ARTIFACT_MACHINE, "hardware_key": hardware_key},
        "fit_config": {"fit_policy": "synthetic_integration_fixture"},
        "families": {
            f"gemm/{PER_CALL}": _family_row(
                "gemm", recipe="*", model_kind="gemm", timing_scope=timing_scope,
                launch_ms=GEMM_LAUNCH_MS + launch_shift_ms,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
            f"add/{PER_CALL}": _family_row(
                "add", recipe="*", model_kind="vector", timing_scope=timing_scope,
                launch_ms=VECTOR_LAUNCH_MS + launch_shift_ms,
                quality="low_confidence", samples=VECTOR_SAMPLES,
            ),
        },
        "provenance": {
            "generator": "tests.test_a3_physical_integration",
            "condition_count": GEMM_SAMPLES + VECTOR_SAMPLES,
            "raw_event_count": (GEMM_SAMPLES + VECTOR_SAMPLES) * 6,
            "fitted_utc": "2026-10-21T00:00:00+00:00",
            "training_hash": "sha256:synthetic-integration-fixture",
            "heldout_guard": {"status": "clean"},
        },
        "unmodeled": [],
    }


def calibration(**kwargs) -> PhysicalCalibration:
    return PhysicalCalibration.from_mapping(physical_artifact(**kwargs))


def synthetic_machine(hardware_key: str = "Synthetic910_X1") -> MachineSpec:
    return MachineSpec.from_mapping({
        "name": hardware_key,
        "hardware_key": hardware_key,
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": 2,
        "device": {
            "hbm_gib": 64,
            "cube_flops_per_s": AIC * ARTIFACT_MACHINE["ops_per_cycle"] * 1.65e9,
            "vector_ops_per_s": VEC_PEAK,
            "hbm_bw_bytes_per_s": ARTIFACT_MACHINE["peak_bw_bytes_per_s"],
            "mte_bw_bytes_per_s": 2.4e12,
            "l2_bytes": ARTIFACT_MACHINE["l2_bytes"],
            "launch_us": 20.0,
            "aic_count": AIC,
            "aiv_count": ARTIFACT_MACHINE["aiv_count"],
        },
    })


def gemm_op(m: int = 4096, n: int = 4096, k: int = 4096, *, family: str = "gemm",
            recipe: str = "torch_mm_preallocated") -> OpSpec:
    return OpSpec(
        op_id=f"{family}.{m}x{n}x{k}", family=family, phase="prefill", stage=0,
        layer=0, requests=("r",), semantic_shape=(m, n, k), local_shape=(m, n, k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=2.0 * m * n * k, hbm_bytes=(m * k + k * n + m * n) * 2.0,
        dtype="bf16", layout="contiguous", recipe=recipe,
    )


def vector_op(
    elements: int = 1 << 20, *, family: str = "add",
    recipe: str = "elementwise_add_reference",
) -> OpSpec:
    return OpSpec(
        op_id=f"{family}.{elements}", family=family, phase="prefill", stage=0,
        layer=0, requests=("r",), semantic_shape=(elements,), local_shape=(elements,),
        padded_shape=(elements,), launch_shape=(elements,),
        vector_ops=float(elements), hbm_bytes=float(elements) * 2.0,
        dtype="bf16", layout="contiguous", recipe=recipe,
    )


def cost_model(artifact: PhysicalCalibration | None, **kwargs) -> PredictiveCostModel:
    return PredictiveCostModel(
        synthetic_machine(), physical_calibration=artifact, **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. The fitted layer owns the duration
# ---------------------------------------------------------------------------
def test_synthetic_artifact_drives_cost_with_the_fitted_duration():
    artifact = calibration()
    op = gemm_op()
    cost = cost_model(artifact).cost(op)

    predicted = artifact.predict_op_explained(
        {
            "family": "gemm", "recipe": "*", "timing_scope": PER_CALL,
            "shape": {"m": 4096, "n": 4096, "k": 4096},
            "flops": op.flops, "hbm_bytes": op.hbm_bytes,
        },
        {"hardware_key": "Synthetic910_X1"}, compute_dtype="bf16",
    )
    assert cost.calibration.status == PHYSICAL_CALIBRATED
    assert cost.estimated_ms == pytest.approx(predicted.duration_ms, rel=1e-9)
    # The analytic formula is untouched: it is the comparison baseline.
    assert cost.analytic_ms == pytest.approx(
        cost_model(None).cost(op).analytic_ms, rel=1e-12,
    )
    assert cost.estimated_ms != pytest.approx(cost.analytic_ms, rel=1e-6)
    # The fitted intercept is the launch term, and it is not added twice.
    assert cost.launch_ms == pytest.approx(GEMM_LAUNCH_MS, rel=1e-12)
    assert cost.service_ms == pytest.approx(
        predicted.duration_ms - GEMM_LAUNCH_MS, rel=1e-9,
    )
    assert cost.estimated_ms == pytest.approx(cost.launch_ms + cost.service_ms, rel=1e-12)

    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_CALIBRATED
    assert record["applied"] is True
    assert record["parameters"]["quality"] == "trusted"
    assert record["fit_level"] == "family_protocol"
    assert record["model_form"] == "wave_max"
    assert record["breakdown"]["launch_ms"] == pytest.approx(GEMM_LAUNCH_MS)
    assert record["breakdown"]["compute_ms"] > 0
    assert record["hardware_key"] == "Synthetic910_X1"
    assert record["query"]["timing_scope"] == PER_CALL
    # The engine schedules demands, so the physical duration must reach them.
    demands = cost.demand_map()
    assert demands["launch"] == pytest.approx(GEMM_LAUNCH_MS, rel=1e-12)
    assert cost.demand_map() != cost_model(None).cost(op).demand_map()


def test_a_different_fitted_intercept_moves_the_estimate_by_exactly_that_amount():
    # Proves the artifact owns the number (a ratio would scale with the analytic
    # time) and that no other intercept is added on top.
    base = cost_model(calibration()).cost(gemm_op())
    shifted = cost_model(calibration(launch_shift_ms=0.010)).cost(gemm_op())
    assert shifted.estimated_ms - base.estimated_ms == pytest.approx(0.010, rel=1e-9)
    assert shifted.analytic_ms == pytest.approx(base.analytic_ms, rel=1e-12)


def test_launch_is_not_double_counted_against_the_fitted_duration():
    # The declared dispatch term is 20 us and the fitted intercept is 17.7 us.
    # The fitted duration already *contains* its intercept, so the estimate must
    # be the artifact's number, not that number plus a dispatch term.
    artifact = calibration()
    op = gemm_op()
    cost = cost_model(artifact).cost(op)
    analytic = cost_model(None).cost(op)
    predicted = artifact.predict_op_explained(
        {
            "family": "gemm", "recipe": "*", "timing_scope": PER_CALL,
            "shape": {"m": 4096, "n": 4096, "k": 4096},
            "flops": op.flops, "hbm_bytes": op.hbm_bytes,
        },
        {"hardware_key": "Synthetic910_X1"}, compute_dtype="bf16",
    )
    assert analytic.launch_ms == pytest.approx(0.020, rel=1e-12)
    assert cost.launch_ms == pytest.approx(GEMM_LAUNCH_MS, rel=1e-12)
    assert cost.estimated_ms == pytest.approx(predicted.duration_ms, rel=1e-9)
    assert cost.estimated_ms == pytest.approx(cost.service_ms + GEMM_LAUNCH_MS, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Unmodeled identities fall back explicitly
# ---------------------------------------------------------------------------
def test_unknown_family_falls_back_with_the_artifact_reason():
    cost = cost_model(calibration()).cost(
        gemm_op(family="attention_qk", recipe="catalog_gqa_qk"),
    )
    assert cost.calibration.status != PHYSICAL_CALIBRATED
    assert cost.calibration.status.startswith("uncalibrated")
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert record["applied"] is False
    # Never mapped onto the fitted gemm family.
    assert record["reason"] == "unmodeled_family"
    assert cost.estimated_ms == pytest.approx(cost_model(None).cost(
        gemm_op(family="attention_qk", recipe="catalog_gqa_qk"),
    ).estimated_ms, rel=1e-12)


def test_declared_family_transfer_prices_a_projection_gemm_and_says_so():
    # The graph names a kernel recipe; the artifact keys the measured runner
    # recipe.  The declared transfer map is the adapter, and it is never silent:
    # the op records where the row came from.
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"gemm/{PER_CALL}": _family_row(
                "gemm", recipe="torch_npu_managed_runner", model_kind="gemm",
                timing_scope=PER_CALL, launch_ms=GEMM_LAUNCH_MS,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
        },
    })
    op = gemm_op(family="qkv_projection", recipe="prefill_causal:qkv")
    cost = cost_model(artifact).cost(op)
    record = cost.to_dict()["physical"]
    assert cost.calibration.status == PHYSICAL_CALIBRATED
    assert cost.calibration.source.endswith("_family_transfer")
    assert record["transfer_from"] == {
        "family": "qkv_projection", "recipe": "prefill_causal:qkv",
    }
    assert record["transfer_to"] == {
        "family": "gemm", "recipe": "torch_npu_managed_runner",
    }
    assert record["query"]["recipe"] == "torch_npu_managed_runner"
    assert "transfer_from=qkv_projection/prefill_causal:qkv" in cost.calibration.reason


def test_transfer_candidates_follow_the_declared_timing_scope():
    batched = physical_transfer_target("gate_up_projection", BATCHED)
    per_call = physical_transfer_target("gate_up_projection", PER_CALL)
    # The scope decides the order; both SKUs' declared spellings are candidates.
    # The op's own family is asked first (the artifact fits family-specific rows
    # such as ``qkv_projection``); the pooled ``gemm`` declaration follows.
    assert batched == (
        ("gate_up_projection", "torch_mm_preallocated"),
        ("gate_up_projection", "torch_npu_managed_runner"),
        ("gate_up_projection", ""),
        ("gemm", "torch_mm_preallocated"), ("gemm", "torch_npu_managed_runner"),
        ("gemm", ""),
    )
    assert per_call == (
        ("gate_up_projection", "torch_npu_managed_runner"),
        ("gate_up_projection", "torch_mm_preallocated"),
        ("gate_up_projection", ""),
        ("gemm", "torch_npu_managed_runner"), ("gemm", "torch_mm_preallocated"),
        ("gemm", ""),
    )
    assert physical_transfer_target("rmsnorm", PER_CALL) == (
        ("rmsnorm", "npu_rms_norm"), ("rmsnorm", ""),
    )
    assert physical_transfer_target("attention_softmax", PER_CALL) == (
        ("softmax", "torch_softmax"), ("softmax", ""),
    )
    assert physical_transfer_target("swiglu", PER_CALL) == (
        ("swiglu", "npu_swiglu"), ("swiglu", ""),
    )
    assert physical_transfer_target("quantize", PER_CALL) == (
        ("quantize", "npu_dynamic_quant"), ("quantize", ""),
    )
    # A2/A3 spell the same vector family differently: both are candidates.
    assert physical_transfer_target("add", PER_CALL) == (
        ("add", "elementwise_add_reference"),
        ("add", "torch_npu_managed_runner"),
        ("add", ""),
    )
    # No fitted identity exists for these at all.
    for family in ("embedding", "sampling", "allreduce", "dequant", "barrier"):
        assert physical_transfer_target(family, PER_CALL) is None


def test_graph_family_mapping_closes_the_named_gaps():
    """The real graph's family names resolve to the measured kernels.

    Discovered on the Kimi-K3 structural case: ``elementwise`` (the add family),
    ``router_softmax`` (a softmax), the gated/linear ``linear_*`` projections and
    the grouped expert GEMM all name a kernel the artifact already fits under
    another spelling.  The table is declared, not guessed per shape.
    """
    from mindie_sim.a3_cost import (
        GEMM_TRANSFER_FAMILIES, PHYSICAL_TRANSFER_ALIASES,
    )

    # 1. `elementwise` is the graph name for the add family.
    assert PHYSICAL_TRANSFER_ALIASES["elementwise"] == "add"
    assert physical_transfer_target("elementwise", PER_CALL) == (
        ("add", "elementwise_add_reference"),
        ("add", "torch_npu_managed_runner"),
        ("add", ""),
    )
    # 2. `router_softmax` is a softmax over the expert axis.
    assert PHYSICAL_TRANSFER_ALIASES["router_softmax"] == "attention_softmax"
    assert physical_transfer_target("router_softmax", PER_CALL) == (
        ("softmax", "torch_softmax"), ("softmax", ""),
    )
    # 3. The linear_* projections are the measured dense GEMM.
    for family in ("linear_q_projection", "linear_k_projection",
                   "linear_v_projection", "linear_o_projection",
                   "linear_gate", "linear_output_gate"):
        assert family in GEMM_TRANSFER_FAMILIES
        candidates = physical_transfer_target(family, PER_CALL)
        assert candidates
        # Its own name first, then the measured dense GEMM -- never another kernel.
        assert {row[0] for row in candidates} == {family, "gemm"}
    # 4. The grouped expert GEMM resolves to the grouped kernel first and only
    #    then to the dense GEMM (a declared, reported approximation).
    for family in ("moe_gate_up", "moe_down"):
        candidates = physical_transfer_target(family, PER_CALL)
        assert [row[0] for row in candidates] == [
            family, family, "grouped_matmul", "grouped_matmul", "gemm", "gemm",
            "gemm",
        ]
        assert candidates[0] == (family, "npu_grouped_matmul_list_bf16")
        assert candidates[2] == ("grouped_matmul", "npu_grouped_matmul_list_bf16")
    # An MC2 chunk's compute never takes the dense-GEMM fallback at all.
    assert physical_transfer_target(
        "moe_gate_up", PER_CALL, allow_dense_gemm=False,
    ) == (
        ("moe_gate_up", "npu_grouped_matmul_list_bf16"),
        ("moe_gate_up", ""),
        ("grouped_matmul", "npu_grouped_matmul_list_bf16"),
        ("grouped_matmul", ""),
    )
    # 5. Attention resolves to an attention family or to nothing -- never a GEMM.
    for family in ("attention_qk", "attention_pv", "attention_fused"):
        candidates = physical_transfer_target(family, PER_CALL)
        assert candidates == ((family, "npu_fusion_attention"), (family, ""))
        assert all(row[0] != "gemm" for row in candidates)
    for family in ("embedding", "sampling"):
        assert physical_transfer_target(family, PER_CALL) is None


def test_linear_gate_is_only_priced_as_a_gemm_when_it_is_one():
    """A vector gate kernel is not a projection.

    ``linear_gate``/``linear_output_gate`` are matrix projections in a gated
    attention block and vector gate kernels in a linear-attention block.  Only the
    GEMM-shaped op is transferred onto the measured dense GEMM; the 1-D op keeps
    its own analytic path even though a fitted GEMM row exists.
    """
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"gemm/{PER_CALL}": _family_row(
                "gemm", recipe="torch_npu_managed_runner", model_kind="gemm",
                timing_scope=PER_CALL, launch_ms=GEMM_LAUNCH_MS,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
        },
    })
    projection = gemm_op(family="linear_gate", recipe="catalog_gate_projection")
    applied = cost_model(artifact).cost(projection)
    assert applied.to_dict()["physical"]["transfer_to"] == {
        "family": "gemm", "recipe": "torch_npu_managed_runner",
    }
    assert applied.calibration.status == PHYSICAL_CALIBRATED

    gate = vector_op(family="linear_gate", recipe="catalog_kda_beta_gate")
    vector_cost = cost_model(artifact).cost(gate)
    record = vector_cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert "transfer_to" not in record
    assert vector_cost.estimated_ms == pytest.approx(
        cost_model(None).cost(gate).estimated_ms, rel=1e-12,
    )


def test_a2_add_resolves_through_its_own_recipe_and_a3_add_through_its_own():
    # A2 keys the vector family as elementwise_add_reference.
    a2_artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(hardware_key="Synthetic910_A2"),
        "families": {
            f"add/{PER_CALL}": _family_row(
                "add", recipe="elementwise_add_reference", model_kind="vector",
                timing_scope=PER_CALL, launch_ms=VECTOR_LAUNCH_MS + 0.002,
                quality="low_confidence", samples=VECTOR_SAMPLES,
            ),
        },
    })
    a2_machine = synthetic_machine("Synthetic910_A2")
    # The graph names a kernel recipe, not a measured runner recipe.
    op = vector_op(recipe="prefill_causal:act")
    a2_cost = PredictiveCostModel(
        a2_machine, physical_calibration=a2_artifact,
    ).cost(op)
    assert a2_cost.to_dict()["physical"]["transfer_to"] == {
        "family": "add", "recipe": "elementwise_add_reference",
    }

    # The same op against an A3-style artifact falls back to the A3 recipe.
    a3_artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"add/{PER_CALL}": _family_row(
                "add", recipe="torch_npu_managed_runner", model_kind="vector",
                timing_scope=PER_CALL, launch_ms=VECTOR_LAUNCH_MS,
                quality="low_confidence", samples=VECTOR_SAMPLES,
            ),
        },
    })
    a3_cost = cost_model(a3_artifact).cost(op)
    assert a3_cost.to_dict()["physical"]["transfer_to"] == {
        "family": "add", "recipe": "torch_npu_managed_runner",
    }


def test_swiglu_and_quantize_resolve_when_fitted_and_stay_unmodeled_when_absent():
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"swiglu/{PER_CALL}": _family_row(
                "swiglu", recipe="npu_swiglu", model_kind="vector",
                timing_scope=PER_CALL, launch_ms=0.020, quality="trusted",
                samples=24,
            ),
            f"quantize/{PER_CALL}": _family_row(
                "quantize", recipe="npu_dynamic_quant", model_kind="vector",
                timing_scope=PER_CALL, launch_ms=0.021, quality="trusted",
                samples=24,
            ),
        },
    })
    swiglu = cost_model(artifact).cost(vector_op(family="swiglu"))
    assert swiglu.to_dict()["physical"]["transfer_to"] == {
        "family": "swiglu", "recipe": "npu_swiglu",
    }
    quantize = cost_model(artifact).cost(vector_op(family="quantize"))
    assert quantize.to_dict()["physical"]["transfer_to"] == {
        "family": "quantize", "recipe": "npu_dynamic_quant",
    }

    # Absent from the artifact: the direct answer is kept, the attempts are named.
    bare = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"gemm/{PER_CALL}": _family_row(
                "gemm", recipe="torch_npu_managed_runner", model_kind="gemm",
                timing_scope=PER_CALL, launch_ms=GEMM_LAUNCH_MS,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
        },
    })
    for family in ("swiglu", "quantize"):
        cost = cost_model(bare).cost(vector_op(family=family))
        record = cost.to_dict()["physical"]
        assert record["status"] == PHYSICAL_UNMODELED
        assert "transfer_from" not in record
        assert {row["family"] for row in record["refused_transfer"]["attempts"]} == {
            family,
        }
        assert cost.estimated_ms == pytest.approx(
            cost_model(None).cost(vector_op(family=family)).estimated_ms, rel=1e-12,
        )


def test_exact_only_lookup_refuses_the_transfer_and_names_the_available_rows():
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"gemm/{PER_CALL}": _family_row(
                "gemm", recipe="torch_npu_managed_runner", model_kind="gemm",
                timing_scope=PER_CALL, launch_ms=GEMM_LAUNCH_MS,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
        },
    })
    op = gemm_op(recipe="torch_mm_preallocated")
    cost = cost_model(artifact, physical_allow_family_transfer=False).cost(op)
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert record["reason"] == "unmodeled_recipe_or_protocol"
    assert record["available_recipes_for_family"] == ["torch_npu_managed_runner"]
    assert "transfer_from" not in record
    assert cost.estimated_ms == pytest.approx(
        cost_model(None).cost(op).estimated_ms, rel=1e-12,
    )


def test_a_refused_transfer_names_the_target_it_tried():
    # The artifact fits only the vector family, so a projection GEMM's declared
    # gemm target does not exist: both attempts are reported.
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"add/{PER_CALL}": _family_row(
                "add", recipe="*", model_kind="vector", timing_scope=PER_CALL,
                launch_ms=VECTOR_LAUNCH_MS, quality="low_confidence",
                samples=VECTOR_SAMPLES,
            ),
        },
    })
    cost = cost_model(artifact).cost(
        gemm_op(family="qkv_projection", recipe="prefill_causal:qkv"),
    )
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert record["reason"] == "unmodeled_family"
    attempts = record["refused_transfer"]["attempts"]
    assert [(row["family"], row["recipe"]) for row in attempts] == [
        ("qkv_projection", "torch_npu_managed_runner"),
        ("qkv_projection", "torch_mm_preallocated"),
        ("qkv_projection", ""),
        ("gemm", "torch_npu_managed_runner"), ("gemm", "torch_mm_preallocated"),
        ("gemm", ""),
    ]
    assert all(row["reason"] == "unmodeled_family" for row in attempts)


def test_a_declared_physical_recipe_selects_the_fitted_row():
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"gemm/{PER_CALL}": _family_row(
                "gemm", recipe="torch_npu_managed_runner", model_kind="gemm",
                timing_scope=PER_CALL, launch_ms=GEMM_LAUNCH_MS,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
        },
    })
    op = OpSpec(
        **{**gemm_op().__dict__, "metadata": (
            ("physical_recipe", "torch_npu_managed_runner"),
        )},
    )
    cost = cost_model(artifact).cost(op)
    assert cost.calibration.status == PHYSICAL_CALIBRATED
    assert cost.to_dict()["physical"]["query"]["recipe"] == "torch_npu_managed_runner"


def test_unmodeled_timing_scope_falls_back_instead_of_borrowing_another_scope():
    artifact = calibration(timing_scope=PER_CALL)
    cost = cost_model(artifact, timing_scope=BATCHED).cost(gemm_op())
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert record["reason"] == "unmodeled_recipe_or_protocol"
    assert record["query"]["timing_scope"] == BATCHED
    assert record["timing_scope"] == BATCHED
    assert record["available_timing_scopes_for_family"] == [PER_CALL]
    assert cost.estimated_ms == pytest.approx(
        cost_model(None, timing_scope=BATCHED).cost(gemm_op()).estimated_ms,
        rel=1e-12,
    )
    # The same op under the scope the artifact does model is calibrated, and the
    # two intervals are different numbers (different intercepts).
    per_call = cost_model(artifact, timing_scope=PER_CALL).cost(gemm_op())
    assert per_call.calibration.status == PHYSICAL_CALIBRATED


def test_artifact_for_another_machine_is_an_explicit_refusal_in_the_cost_model():
    foreign = calibration(hardware_key="Ascend910B4")
    cost = cost_model(foreign).cost(gemm_op())
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert record["reason"] == "hardware_key_mismatch"
    assert cost.estimated_ms == pytest.approx(
        cost_model(None).cost(gemm_op()).estimated_ms, rel=1e-12,
    )


# ---------------------------------------------------------------------------
# 3. Unidentified parameters: GEMM applies, add/copy keep the analytic path
# ---------------------------------------------------------------------------
def test_low_confidence_gemm_row_is_applied_and_reports_its_quality():
    # The batched GEMM row is the only fitted GEMM model for that protocol and
    # is physically fitted (wave-quantized, refit-per-fold CV); its quality label
    # is reported, but the analytic formula is not preferred over it.
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(),
        "families": {
            f"gemm/{BATCHED}": _family_row(
                "gemm", recipe="torch_mm_preallocated", model_kind="gemm",
                timing_scope=BATCHED, launch_ms=GEMM_LAUNCH_MS,
                quality="low_confidence", samples=GEMM_SAMPLES,
            ),
        },
    })
    cost = cost_model(artifact, timing_scope=BATCHED).cost(gemm_op())
    assert cost.calibration.status == PHYSICAL_CALIBRATED
    assert "quality=low_confidence" in cost.calibration.reason
    assert "fitted_row_applied" in cost.calibration.reason
    assert cost.to_dict()["physical"]["parameters"]["quality"] == "low_confidence"
    assert cost.estimated_ms != pytest.approx(cost.analytic_ms, rel=1e-6)


def test_low_confidence_family_keeps_the_analytic_path_and_says_so():
    artifact = calibration()
    op = vector_op()
    cost = cost_model(artifact).cost(op)
    baseline = cost_model(None).cost(op)
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_LOW_CONFIDENCE
    assert record["applied"] is False
    assert record["parameters"]["quality"] == "low_confidence"
    # A number exists and is reported, but it does not replace the estimate.
    assert record["duration_ms"] is not None and record["duration_ms"] > 0
    assert cost.estimated_ms == pytest.approx(baseline.estimated_ms, rel=1e-12)
    assert cost.analytic_ms == pytest.approx(baseline.analytic_ms, rel=1e-12)
    assert cost.demand_map() == baseline.demand_map()


# ---------------------------------------------------------------------------
# 3b. The graph's large-operator families reach the fitted rows
# ---------------------------------------------------------------------------
def _large_operator_artifact(
    *, include: tuple[str, ...] = (), hardware_key: str = "Synthetic910_X1",
) -> PhysicalCalibration:
    """Artifact with the measured large-operator families the graph names.

    ``include`` selects which fitted families exist, so a test can show both
    that a row is reached when present and that nothing is borrowed when it is
    absent.
    """
    rows: dict[str, dict] = {}
    if "gemm" in include:
        rows[f"gemm/{PER_CALL}"] = _family_row(
            "gemm", recipe="torch_npu_managed_runner", model_kind="gemm",
            timing_scope=PER_CALL, launch_ms=GEMM_LAUNCH_MS,
            quality="trusted", samples=GEMM_SAMPLES,
        )
    if "add" in include:
        rows[f"add/{PER_CALL}"] = _family_row(
            "add", recipe="torch_npu_managed_runner", model_kind="vector",
            timing_scope=PER_CALL, launch_ms=VECTOR_LAUNCH_MS,
            quality="low_confidence", samples=VECTOR_SAMPLES,
        )
    if "softmax" in include:
        rows[f"softmax/{PER_CALL}"] = _family_row(
            "softmax", recipe="torch_softmax", model_kind="vector",
            timing_scope=PER_CALL, launch_ms=0.031, quality="trusted", samples=24,
        )
    if "grouped_matmul" in include:
        rows[f"grouped_matmul/{PER_CALL}"] = _family_row(
            "grouped_matmul", recipe="npu_grouped_matmul_list_bf16",
            model_kind="grouped_gemm", timing_scope=PER_CALL,
            launch_ms=0.019, quality="trusted", samples=13,
        )
    if "moe_mc2" in include:
        # The same physical family measured twice: the prefill grouped corpus and
        # the MC2 campaign's own compute legs.  Only the fitted intercept differs,
        # so a prediction difference is attributable to which row answered.
        rows[f"moe_gate_up/npu_grouped_matmul_list_bf16"] = _family_row(
            "moe_gate_up", recipe="npu_grouped_matmul_list_bf16",
            model_kind="grouped_gemm", timing_scope=PER_CALL,
            launch_ms=0.019, quality="trusted", samples=15,
        )
        rows[f"moe_gate_up/npu_grouped_matmul_list_bf16_mc2"] = _family_row(
            "moe_gate_up", recipe="npu_grouped_matmul_list_bf16_mc2",
            model_kind="grouped_gemm", timing_scope=PER_CALL,
            launch_ms=0.011, quality="trusted", samples=14,
        )
    if "attention_qk" in include:
        rows[f"attention_qk/{PER_CALL}"] = _family_row(
            "attention_qk", recipe="npu_fusion_attention",
            model_kind="attention", timing_scope=PER_CALL,
            launch_ms=0.023, quality="trusted", samples=37,
        )
    return PhysicalCalibration.from_mapping({
        **physical_artifact(hardware_key=hardware_key), "families": rows,
    })


def grouped_op(
    tokens: int = 2048, experts: int = 32, n: int = 7168, k: int = 7168,
    *, family: str = "moe_gate_up", recipe: str = "moe_mc2",
) -> OpSpec:
    return OpSpec(
        op_id=f"{family}.{tokens}x{n}x{k}", family=family, phase="prefill",
        stage=0, layer=0, requests=("r",),
        semantic_shape=(tokens, n, k), local_shape=(tokens, n, k),
        padded_shape=(tokens, n, k), launch_shape=(tokens, n, k),
        flops=2.0 * tokens * n * k,
        hbm_bytes=(experts * n * k + tokens * k + tokens * n) * 2.0,
        dtype="bf16", layout="contiguous", recipe=recipe,
        metadata=(("expert_tokens", tokens), ("group_size", experts)),
    )


def attention_op(
    q: int = 1, kv: int = 4096, heads: int = 8, dim: int = 128,
    *, family: str = "attention_qk", recipe: str = "decode:qk",
) -> OpSpec:
    return OpSpec(
        op_id=f"{family}.{q}x{kv}", family=family, phase="decode", stage=0,
        layer=0, requests=("r",), semantic_shape=(q, kv, heads, dim),
        local_shape=(q, kv, heads, dim), padded_shape=(q, kv, heads, dim),
        launch_shape=(q, kv, heads, dim),
        flops=4.0 * heads * dim * q * kv,
        hbm_bytes=(q * heads * dim + kv * heads * dim) * 2.0,
        dtype="bf16", layout="contiguous", recipe=recipe,
    )


def test_elementwise_resolves_to_the_add_row_and_says_it_is_low_confidence():
    # `elementwise` is the graph's name for the measured add kernel.  The add row
    # is reported as unidentified (its rates are noise-dominated), so the analytic
    # path stays primary -- but the wiring is now visible instead of "unmodeled".
    artifact = _large_operator_artifact(include=("add",))
    cost = cost_model(artifact).cost(vector_op(family="elementwise"))
    record = cost.to_dict()["physical"]
    assert record["transfer_to"] == {
        "family": "add", "recipe": "torch_npu_managed_runner",
    }
    assert record["status"] == PHYSICAL_LOW_CONFIDENCE
    assert record["wiring"]["status"] == PHYSICAL_UNMODELED
    assert record["wiring"]["physical_status"] == PHYSICAL_LOW_CONFIDENCE


def test_router_softmax_resolves_to_the_softmax_row():
    artifact = _large_operator_artifact(include=("softmax",))
    op = vector_op(family="router_softmax", recipe="catalog_moe_router_score")
    cost = cost_model(artifact).cost(op)
    record = cost.to_dict()["physical"]
    assert record["transfer_to"] == {"family": "softmax", "recipe": "torch_softmax"}
    assert record["status"] == PHYSICAL_CALIBRATED
    assert record["wiring"]["status"] == PHYSICAL_TRANSFER
    assert cost.estimated_ms == pytest.approx(
        artifact.predict_op_explained(
            {"family": "softmax", "recipe": "torch_softmax",
             "timing_scope": PER_CALL,
             "shape": {"elements": 1 << 20},
             "elements": 1 << 20, "hbm_bytes": float(1 << 21)},
            {"hardware_key": "Synthetic910_X1"}, compute_dtype="bf16",
        ).duration_ms,
        rel=1e-9,
    )


def test_attention_families_never_resolve_to_a_gemm_row():
    # The artifact fits both a dense GEMM and the QK half.  The QK op must reach
    # its own attention family; with no attention row it must stay unmodeled
    # rather than being priced as a rectangular GEMM.
    both = _large_operator_artifact(include=("gemm", "attention_qk"))
    cost = cost_model(both).cost(attention_op())
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_CALIBRATED
    assert record["transfer_to"] == {
        "family": "attention_qk", "recipe": "npu_fusion_attention",
    }
    assert record["wiring"]["status"] == PHYSICAL_TRANSFER

    gemm_only = _large_operator_artifact(include=("gemm",))
    for family in ("attention_qk", "attention_pv", "attention_fused"):
        op = attention_op(family=family)
        unmodeled = cost_model(gemm_only).cost(op)
        record = unmodeled.to_dict()["physical"]
        assert record["status"] == PHYSICAL_UNMODELED
        assert record["reason"] == "unmodeled_family"
        assert "transfer_to" not in record
        assert unmodeled.estimated_ms == pytest.approx(
            cost_model(None).cost(op).estimated_ms, rel=1e-12,
        )


def test_grouped_matmul_row_prices_the_expert_gemm_before_the_dense_gemm():
    artifact = _large_operator_artifact(include=("gemm", "grouped_matmul"))
    op = grouped_op(recipe="catalog_npu_grouped_matmul")
    cost = cost_model(artifact).cost(op)
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_CALIBRATED
    assert record["transfer_to"] == {
        "family": "grouped_matmul", "recipe": "npu_grouped_matmul_list_bf16",
    }
    # The fitted row is the grouped geometry (tokens/experts/n/k), which is a
    # different number from the dense GEMM transfer of the same total shape.
    assert record["query"]["shape"] == {
        "m": 2048.0, "n": 7168.0, "k": 7168.0,
        "tokens": 2048.0, "experts": 32.0,
    }
    flat = cost_model(
        _large_operator_artifact(include=("gemm",)),
    ).cost(grouped_op(recipe="catalog_npu_grouped_matmul"))
    assert flat.to_dict()["physical"]["transfer_to"]["family"] == "gemm"
    assert cost.estimated_ms != pytest.approx(flat.estimated_ms, rel=1e-6)


def test_mc2_compute_never_takes_the_dense_gemm_transfer():
    # An MC2 chunk's duration is the fused kernel; a second GEMM model on top of
    # it would double count the same cube work.  The grouped identity is still
    # attempted, so the dense GEMM is only skipped -- never silently replaced by
    # the analytic path while a grouped row exists.
    artifact = _large_operator_artifact(include=("gemm",))
    cost = cost_model(artifact).cost(grouped_op(recipe="moe_mc2"))
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert "transfer_to" not in record
    # The MC2-measured recipe is asked first (a fused chunk member launches the
    # split-item call the MC2 campaign timed), then the prefill grouped spelling,
    # then the pooled grouped family, then the recipe-agnostic row of each.
    assert [row["family"] for row in record["refused_transfer"]["attempts"]] == [
        "moe_gate_up", "moe_gate_up", "moe_gate_up", "grouped_matmul", "grouped_matmul",
    ]
    assert [row["recipe"] for row in record["refused_transfer"]["attempts"][:2]] == [
        "npu_grouped_matmul_list_bf16_mc2", "npu_grouped_matmul_list_bf16",
    ]
    assert "gemm" not in {
        row["family"] for row in record["refused_transfer"]["attempts"]
    }

    # The same op without the MC2 marker keeps the declared dense-GEMM fallback.
    non_mc2 = cost_model(artifact).cost(grouped_op(recipe="catalog_npu_grouped_matmul"))
    assert non_mc2.to_dict()["physical"]["transfer_to"]["family"] == "gemm"

    # With a grouped row present the MC2 op is physically priced.
    fitted = cost_model(
        _large_operator_artifact(include=("gemm", "grouped_matmul")),
    ).cost(grouped_op(recipe="moe_mc2"))
    assert fitted.to_dict()["physical"]["transfer_to"] == {
        "family": "grouped_matmul", "recipe": "npu_grouped_matmul_list_bf16",
    }


def test_mc2_compute_prefers_the_mc2_measured_recipe_row():
    """A fused chunk member is priced by the kernel the MC2 campaign timed.

    Both rows describe the same grouped family at the same geometry; the MC2 row
    was fitted on the split-item legs of the fused chunk, the other on the prefill
    grouped corpus.  The MC2 op must take the MC2 row, and an op that does not
    declare the MC2 marker must keep the prefill row.
    """
    artifact = _large_operator_artifact(include=("moe_mc2",))
    shape = {"tokens": 2048, "experts": 32, "n": 4096, "k": 7168}
    mc2 = cost_model(artifact).cost(grouped_op(**shape, recipe="moe_mc2"))
    record = mc2.to_dict()["physical"]
    assert record["status"] == PHYSICAL_CALIBRATED
    assert record["transfer_to"] == {
        "family": "moe_gate_up", "recipe": "npu_grouped_matmul_list_bf16_mc2",
    }
    prefill = cost_model(artifact).cost(
        grouped_op(**shape, recipe="catalog_npu_grouped_matmul")
    )
    prefill_record = prefill.to_dict()["physical"]
    assert prefill_record["transfer_to"]["recipe"] == "npu_grouped_matmul_list_bf16"
    # Identical efficiencies and geometry, different measured intercept: the whole
    # difference between the two estimates is the two fitted launches.
    assert mc2.estimated_ms - prefill.estimated_ms == pytest.approx(0.011 - 0.019, rel=1e-9)


def test_attention_resolves_at_the_per_call_scope_and_refuses_another_scope():
    """Eager serving measures attention per call, not amortized in a graph.

    The fused attention row is a per-call row: it answers at the scope it was
    measured at, and a case that declares a scope the artifact does not carry
    reports ``physical_unmodeled`` (with the scopes the artifact does model)
    instead of being answered from another protocol's intercept.
    """
    per_call_attention = PER_CALL
    artifact = PhysicalCalibration.from_mapping({
        **physical_artifact(timing_scope=per_call_attention),
        "families": {
            f"attention_fused/{per_call_attention}": _family_row(
                "attention_fused", recipe="npu_fusion_attention",
                model_kind="attention", timing_scope=per_call_attention,
                launch_ms=0.023, quality="trusted", samples=37,
            ),
        },
    })
    fused = attention_op(family="attention_fused", recipe="decode:fused")
    cost = cost_model(artifact, timing_scope=per_call_attention).cost(fused)
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_CALIBRATED
    assert record["transfer_to"] == {
        "family": "attention_fused", "recipe": "npu_fusion_attention",
    }
    # The fitted attention geometry is (q, kv, heads, dim), not a dense m/n/k.
    assert record["query"]["shape"] == {
        "q": 1.0, "kv": 4096.0, "heads": 8.0, "dim": 128.0, "causal": 0.0,
    }
    assert cost.estimated_ms == pytest.approx(record["duration_ms"], rel=1e-9)

    # The batched scope is a different protocol: the fused op is not answered.
    batched = cost_model(artifact, timing_scope=BATCHED).cost(fused)
    record = batched.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert record["reason"] == "unmodeled_recipe_or_protocol"
    assert record["query"]["timing_scope"] == BATCHED
    assert batched.estimated_ms == pytest.approx(
        cost_model(None, timing_scope=BATCHED).cost(fused).estimated_ms, rel=1e-12,
    )


def test_a_per_call_case_applies_gemm_and_attention_through_the_scope_family():
    """One declared scope must reach every per-call row of the artifact.

    A serving case measures kernels per call: the GEMM rows live at
    ``device_event_warm`` and the attention rows at ``device_event_warm_operator``.
    Both are the same *protocol family* (per-call device-event intervals), so a case
    that declares either one must reach both.  The alias is reported, never silent:
    the op records the declared scope, the fitted scope and the alias pair, and the
    payload aggregates the pairs.
    """
    per_call_family = PhysicalCalibration.from_mapping({
        **physical_artifact(timing_scope=PER_CALL),
        "families": {
            f"gemm/{PER_CALL}": _family_row(
                "gemm", recipe="torch_npu_managed_runner", model_kind="gemm",
                timing_scope=PER_CALL, launch_ms=GEMM_LAUNCH_MS,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
            f"attention_fused/{OPERATOR_SCOPE}": _family_row(
                "attention_fused", recipe="npu_fusion_attention",
                model_kind="attention", timing_scope=OPERATOR_SCOPE,
                launch_ms=0.023, quality="trusted", samples=37,
            ),
        },
    })
    projection = gemm_op(family="linear_o_projection", recipe="kda_o_projection")
    attention = attention_op(family="attention_fused", recipe="prefill:fused")
    for op, expected_family in (
        (projection, "gemm"), (attention, "attention_fused"),
    ):
        cost = cost_model(per_call_family, timing_scope=PER_CALL).cost(op)
        record = cost.to_dict()["physical"]
        assert record["status"] == PHYSICAL_CALIBRATED, record["reason"]
        assert record["transfer_to"]["family"] == expected_family
        # The declared scope is what the case asked for; the fitted scope is the
        # row that answered, and the alias pair names the hop.
        assert record["declared_timing_scope"] == PER_CALL
        if expected_family == "attention_fused":
            assert record["fitted_timing_scope"] == OPERATOR_SCOPE
            assert record["scope_alias"] == {
                "from": PER_CALL, "to": OPERATOR_SCOPE,
            }
            assert record["wiring"]["scope_alias_from"] == PER_CALL
            assert record["wiring"]["scope_alias_to"] == OPERATOR_SCOPE
        else:
            assert record["fitted_timing_scope"] == PER_CALL
            assert "scope_alias" not in record

    # The payload aggregates the alias so a reader can see how much of the
    # prediction rests on a protocol sibling.
    payload_in = _example_payload()
    payload = predict(
        payload_in,
        physical_calibration=PhysicalCalibration.from_mapping({
            **physical_artifact(
                hardware_key=payload_in["machine"]["name"], timing_scope=PER_CALL,
            ),
            "families": {
                f"qkv_projection/{OPERATOR_SCOPE}": _family_row(
                    "qkv_projection", recipe="torch_npu_managed_runner",
                    model_kind="gemm", timing_scope=OPERATOR_SCOPE,
                    launch_ms=GEMM_LAUNCH_MS, quality="trusted",
                    samples=GEMM_SAMPLES,
                ),
            },
        }),
        timing_scope=PER_CALL,
    )
    assert payload["physical_calibration"]["scope_alias_counts"] == {
        f"{PER_CALL}->{OPERATOR_SCOPE}": sum(
            1 for step in payload["traces"] for op in step["ops"]
            if op["family"] == "qkv_projection"
        ),
    }


def test_predict_reports_the_large_operator_wiring_counts():
    # The example graph names its projections with their own families, so a
    # GEMM-only artifact answers them through the declared transfer map.
    payload_in = _example_payload()
    artifact = _large_operator_artifact(
        include=("gemm", "softmax"),
        hardware_key=payload_in["machine"]["name"],
    )
    payload = predict(payload_in, physical_calibration=artifact)
    provenance = payload["physical_calibration"]
    counts = provenance["wiring_counts"]
    assert sum(counts.values()) == sum(len(step["ops"]) for step in payload["traces"])
    assert counts.get(PHYSICAL_TRANSFER, 0) > 0
    # The large-operator families are counted separately from the tail.
    assert provenance["large_operator_wiring_counts"]
    assert provenance["mc2_op_count"] == 0
    assert set(counts) <= {
        PHYSICAL_CALIBRATED, PHYSICAL_TRANSFER, PHYSICAL_UNMODELED,
        "mc2_overlap_unverified", "mc2_overlap_calibrated",
    }
    ops = [op for step in payload["traces"] for op in step["ops"]]
    wired = [op for op in ops if op["physical"] is not None]
    assert wired and all("wiring" in op["physical"] for op in wired)
    assert all(
        op["physical"]["wiring"]["status"] in {
            PHYSICAL_CALIBRATED, PHYSICAL_TRANSFER, PHYSICAL_UNMODELED,
        }
        for op in wired
    )


def test_predict_reports_the_analytic_path_wiring_when_no_artifact_is_supplied():
    payload = predict(_example_payload())
    assert payload["physical_calibration"]["wiring_counts"] == {
        PHYSICAL_UNMODELED: sum(len(step["ops"]) for step in payload["traces"]),
    }
    assert payload["physical_calibration"]["mc2_op_count"] == 0


# ---------------------------------------------------------------------------
# 4. Public predict(): provenance, scope reporting, hardware-key refusal
# ---------------------------------------------------------------------------
def _example_payload() -> dict:
    return json.loads(EXAMPLE_CASE.read_text(encoding="utf-8"))


def example_artifact(*, timing_scope: str = PER_CALL, hardware_key: str | None = None) -> PhysicalCalibration:
    payload = _example_payload()
    return PhysicalCalibration.from_mapping({
        **physical_artifact(
            hardware_key=hardware_key or payload["machine"]["name"],
            timing_scope=timing_scope,
        ),
        # The example model lowers projections with their own family names; a
        # pooled row models the GEMM roofline of those projections explicitly.
        "families": {
            f"qkv_projection/{PER_CALL}": _family_row(
                "qkv_projection", recipe="*", model_kind="gemm",
                timing_scope=timing_scope, launch_ms=GEMM_LAUNCH_MS,
                quality="trusted", samples=GEMM_SAMPLES,
            ),
            f"add/{PER_CALL}": _family_row(
                "add", recipe="*", model_kind="vector", timing_scope=timing_scope,
                launch_ms=VECTOR_LAUNCH_MS, quality="low_confidence",
                samples=VECTOR_SAMPLES,
            ),
        },
    })


def test_predict_reports_the_physical_provenance_and_the_timing_scope():
    payload = predict(_example_payload(), physical_calibration=example_artifact())
    provenance = payload["physical_calibration"]
    assert provenance["status"] == "configured"
    assert provenance["hardware_key"] == payload["plan"]["machine"]["hardware_key"]
    assert provenance["schema_version"] == SCHEMA_VERSION
    assert provenance["model_kind"] == "fitted_physical_roofline"
    assert provenance["training_sample_count"] == GEMM_SAMPLES + VECTOR_SAMPLES
    assert provenance["timing_scope"] == PER_CALL
    assert payload["timing_scope"] == PER_CALL
    assert provenance["applied_op_count"] > 0
    ops = [op for step in payload["traces"] for op in step["ops"]]
    applied = [op for op in ops if op["physical"]["status"] == PHYSICAL_CALIBRATED]
    assert applied and all(op["family"] == "qkv_projection" for op in applied)
    assert all(op["analytic_ms"] is not None for op in applied)
    # Everything the artifact does not model is reported, not silently absent.
    assert provenance["unmodeled_op_count"] > 0
    assert any(
        "unmodeled_family" in reason
        for reason in provenance["unmodeled_reasons"]
    )


def test_predict_reports_the_analytic_path_when_no_artifact_is_supplied():
    payload = predict(_example_payload())
    assert payload["physical_calibration"]["status"] == "not_configured"
    assert payload["timing_scope"] == PER_CALL
    assert all(
        op["physical"] is None for step in payload["traces"] for op in step["ops"]
    )


def test_predict_refuses_an_artifact_for_another_hardware_key():
    payload = {
        **_example_payload(),
        "machine": {**_example_payload()["machine"]},
    }
    foreign = example_artifact(hardware_key="Ascend910B4")
    with pytest.raises(A3InputError, match="physical_calibration_hardware_key_conflict"):
        predict(payload, physical_calibration=foreign)


def test_predict_scope_override_is_reported_and_falls_back_explicitly():
    payload = predict(
        _example_payload(), physical_calibration=example_artifact(timing_scope=BATCHED),
        timing_scope=PER_CALL,
    )
    assert payload["timing_scope"] == PER_CALL
    assert payload["physical_calibration"]["applied_op_count"] == 0
    assert payload["physical_calibration"]["unmodeled_op_count"] > 0
    assert all(
        op["physical"] is None or op["physical"]["status"] == PHYSICAL_UNMODELED
        for step in payload["traces"] for op in step["ops"]
    )
    # The declared scope must be a physical protocol identity, not free text:
    # the CLI refuses an unknown one instead of quietly using the default.
    from mindie_sim.__main__ import main as cli_main

    assert PER_CALL in ACCEPTED_TIMING_SCOPES and BATCHED in ACCEPTED_TIMING_SCOPES


def test_predict_cli_refuses_an_unknown_timing_scope(tmp_path: Path, capsys):
    from mindie_sim.__main__ import main as cli_main

    output = tmp_path / "unknown-scope.json"
    code = cli_main([
        "predict", "--input", str(EXAMPLE_CASE), "--timing-scope", "not_a_scope",
        "--output", str(output),
    ])
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["reason"].startswith("timing_scope_unknown:not_a_scope")


@pytest.mark.skipif(
    not PUBLISHED_A3_ARTIFACT.exists(),
    reason="published A3 physical-calibration artifact is not present",
)
def test_production_loop_case_reaches_the_published_artifact_through_the_transfer_map():
    """The published artifact must actually be used by the public predict path.

    Structure only -- which families are answered, and that a projection GEMM is
    the only thing ever transferred onto the fitted ``gemm`` family -- so a refit
    of the artifact does not invalidate the test.
    """
    artifact = PhysicalCalibration.load(PUBLISHED_A3_ARTIFACT)
    payload = predict(
        json.loads(PRODUCTION_CASE.read_text(encoding="utf-8")), physical_calibration=artifact,
    )
    provenance = payload["physical_calibration"]
    assert provenance["applied_op_count"] > 0
    assert provenance["applied_transfer_op_count"] == provenance["applied_op_count"]
    ops = [op for step in payload["traces"] for op in step["ops"]]
    applied = {
        op["family"] for op in ops
        if op["physical"] and op["physical"]["status"] == PHYSICAL_CALIBRATED
    }
    assert {"qkv_projection", "o_projection", "gate_up_projection",
            "down_projection", "logits"} <= applied
    for op in ops:
        record = op["physical"]
        if not record:
            continue
        transfer = record.get("transfer_to") or {}
        if op["family"] in {"embedding", "sampling", "swiglu"}:
            assert record["status"] == PHYSICAL_UNMODELED
        if op["family"] in {
            "attention_qk", "attention_pv", "attention_fused",
            "embedding", "sampling", "swiglu",
        }:
            # Never relabelled as a GEMM, never silently priced by another
            # family.  An attention op may now be calibrated through its own
            # fitted attention family (the artifact carries one), which the
            # structural checks in the unit suite cover.
            assert transfer.get("family") != "gemm"


# ---------------------------------------------------------------------------
# 5. CLI smoke (in-process, no device)
# ---------------------------------------------------------------------------
def test_predict_cli_with_physical_calibration_emits_analytic_and_physical_fields(
    tmp_path: Path, capsys,
):
    from mindie_sim.__main__ import main as cli_main

    artifact_path = tmp_path / "physical-calibration.json"
    artifact_path.write_text(
        json.dumps(example_artifact().to_dict()), encoding="utf-8",
    )
    output = tmp_path / "prediction.json"
    code = cli_main([
        "predict", "--input", str(EXAMPLE_CASE),
        "--physical-calibration", str(artifact_path),
        "--timing-scope", PER_CALL,
        "--output", str(output),
    ])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "modeled"
    assert payload["command"] == "predict"
    # Existing public fields survive the integration.
    assert set(payload["metrics"]) >= {"ttft_ms", "itl_ms", "tpot_ms", "e2e_ms"}
    assert payload["physical_calibration"]["status"] == "configured"
    assert payload["physical_calibration"]["training_sample_count"] > 0
    assert payload["timing_scope"] == PER_CALL
    ops = [op for step in payload["traces"] for op in step["ops"]]
    assert any(op["physical"]["status"] == PHYSICAL_CALIBRATED for op in ops)
    assert all(
        op["analytic_ms"] is not None and op["estimated_ms"] is not None
        for op in ops if op["physical"] is not None
    )


def test_predict_cli_refuses_a_foreign_artifact_and_reports_the_reason(
    tmp_path: Path, capsys,
):
    from mindie_sim.__main__ import main as cli_main

    artifact_path = tmp_path / "foreign.json"
    artifact_path.write_text(
        json.dumps(example_artifact(hardware_key="Ascend910B4").to_dict()),
        encoding="utf-8",
    )
    output = tmp_path / "refused.json"
    code = cli_main([
        "predict", "--input", str(EXAMPLE_CASE),
        "--physical-calibration", str(artifact_path),
        "--output", str(output),
    ])
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "physical_calibration_hardware_key_conflict" in payload["reason"]


def test_hardware_cli_without_a_sustained_band_reports_it_unmeasured(
    tmp_path: Path, capsys,
):
    from mindie_sim.__main__ import main as cli_main

    sku = tmp_path / "sku.json"
    sku.write_text(json.dumps({
        "soc_version": "Ascend950DT_9599",
        "arch_family": "A5",
        "cube_spec": {"aic_count_per_die": 36},
        "memory_hierarchy_spec": {
            "platform_config_hbm_mib": 196608.0, "hbm_peak_bw_gbps": 4800.0,
        },
    }), encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({
        "flops_bf16": 1e12, "target_us": 100.0,
    }), encoding="utf-8")

    unmeasured = tmp_path / "unmeasured.json"
    assert cli_main([
        "hardware", "--sku-profile", str(sku), "--workload", str(workload),
        "--output", str(unmeasured),
    ]) == 0
    thermal = json.loads(unmeasured.read_text(encoding="utf-8"))["thermal"]
    # No fabricated 1250-1300 MHz band: an omitted clock band stays unmeasured.
    assert thermal["sustained_mhz_range"] == []
    assert thermal["sustained_reference_us"] is None

    declared = tmp_path / "declared.json"
    assert cli_main([
        "hardware", "--sku-profile", str(sku), "--workload", str(workload),
        "--sustained-mhz", "1650:1650", "--output", str(declared),
    ]) == 0
    declared_thermal = json.loads(declared.read_text(encoding="utf-8"))["thermal"]
    assert declared_thermal["sustained_mhz_range"] == [1650.0, 1650.0]
    assert declared_thermal["sustained_reference_us"] is not None


# ---------------------------------------------------------------------------
# 6. Communication / CV identity is an exact join
# ---------------------------------------------------------------------------
def test_comm_and_cv_unbound_observations_are_refused_for_a_keyed_query():
    from mindie_sim.a3_communication import CommCondition, CommunicationCalibration
    from mindie_sim.a3_cv import CvCalibration, CvCondition

    machine = synthetic_machine()  # hardware_key "Synthetic910_X1"
    unbound = CommCondition("sio", "all_reduce_only", 1024, 100.0, 4)
    foreign = CommCondition(
        "sio", "all_reduce_only", 1024, 50.0, 4, hardware_key="Ascend910B4",
    )
    bound = CommCondition(
        "sio", "all_reduce_only", 1024, 0.5, 4, hardware_key="Synthetic910_X1",
    )
    refused_only = CommunicationCalibration([unbound, foreign]).predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        machine=machine, ranks=2, dtype="bf16",
    )
    assert refused_only["status"].startswith("uncalibrated")
    audit = refused_only["identity"]["refused_observations"]
    assert audit["unbound"] == 1 and audit["mismatched"] == 1
    assert {row["reason"] for row in audit["examples"]} == {
        "hardware_key_unbound", "hardware_key_mismatch",
    }
    assert refused_only["identity_reason"] == (
        "measured_conditions_exist_only_for_another_machine_key"
    )

    # A bound observation is used; the unbound one is never pooled into the fit.
    mixed = CommunicationCalibration([bound, unbound]).predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        machine=machine, ranks=2, dtype="bf16",
    )
    assert mixed["status"].startswith("calibrated")
    assert mixed["span_ms"] == pytest.approx(0.5, rel=1e-9)
    assert mixed["identity"]["refused_observations"]["count"] == 1

    # A `device:placement` key is the same machine plus a fabric, not a wildcard.
    compound = CommunicationCalibration([CommCondition(
        "sio", "all_reduce_only", 1024, 0.5, 4,
        hardware_key="Synthetic910_X1:sio",
    )]).predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        machine=machine, ranks=2, dtype="bf16",
    )
    assert compound["status"].startswith("calibrated")

    shape = (128, 256, 256)
    cv_refused = CvCalibration([
        CvCondition("c_only", 1024, 0.0, shape, "bf16", 1, 1,
                    1.0, 1.0, 1.0, 1.0, 1.0, 4),
    ]).predict(
        c_shape_mkn=shape, v_elements=1024, machine=machine, dtype="bf16",
        hardware_key=machine.hardware_key,
    )
    assert cv_refused["status"].startswith("uncalibrated")
    assert cv_refused["identity"]["refused_conditions"]["unbound"] == 1
    assert cv_refused["identity_reason"] == (
        "measured_conditions_exist_only_for_another_machine_key"
    )


def test_comm_unknown_fabric_protocol_is_not_mapped_onto_hccs():
    # a5-audit B-2: a UB-Mesh link must not resolve to the A2/A3 HCCS placement.
    from mindie_sim.a3_cost import communication_placement_for_link

    class _Link:
        def __init__(self, channel: str, protocol: str) -> None:
            self.channel = channel
            self.protocol = protocol

    assert communication_placement_for_link(_Link("d2d_intra_package", "SIO")) == "sio"
    assert communication_placement_for_link(_Link("d2d_intra_node", "HCCS")) == "hccs"
    assert communication_placement_for_link(
        _Link("d2d_intra_node", "HCCS_SW"),
    ) == "hccssw"
    assert communication_placement_for_link(_Link("d2d_inter_node", "RoCEv2_400G")) == "roce"
    assert communication_placement_for_link(_Link("d2d_intra_node", "UB_MESH")) == ""
