"""NR-S7 FIX2 regression witnesses (GLM gates, c8 planes, MLAPO, pool, ownership).

[Role]
  Encode the Round-2 counterexamples for F01..F06 and F14/F15: every executing
  GLM KDA layer, typed c8 planes and scale bytes, the exact MLAPO predicate,
  GLM grouping/block demand/stride/declared baselines, stage/PCP ownership,
  DSpark activation and scratch, and universal pre-coercion typing.

[Invariants]
  Each assertion is an exact source-derived identity or typed rejection; no
  test asserts a value copied from the implementation under test.

[Timeouts & Locks]
  Pure in-memory arithmetic; no device, subprocess or network access.

[Side-Effect Policy]
  Builds immutable records; creates nothing outside pytest's temporary root.
"""
from __future__ import annotations

import dataclasses as dc
import json
from typing import Any, Union

import pytest

from nr_s7_fix2_support import (
    _base_specs, _case, _config, _dcp, _draft, _ds41, _geometry, _glm,
    _identity, _kda_layers, _keys, _kimi, _layer, _spec, _target)

from mindie_sim.hybrid_kv_state_mcp import hybrid_kv_state_mcp_envelope
from mindie_sim.hybrid_kv_state_options import (
    HybridKvStateOptions, model_mlapo_eligible, _machine_limits)
from mindie_sim.hybrid_kv_state_pool import baseline_pricing
from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state
from mindie_sim.kv_nz_transfer import nd_to_nz_reformat_cost
from mindie_sim.kv_pool_geometry import (
    grouped_pool_layout, longest_run_recurrent_groups)
from mindie_sim.linear_state_physics import (
    LinearLayer, validate_glm_kda_execution)
from mindie_sim.linear_state_replay import precopy_cost, verify_commit_cost
from mindie_sim.memory_breakdown import (
    analyze_memory_breakdown, apply_state_pool_delta)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.dspark import DsparkSpec


# ------------------------------------------------------ F01 GLM enforcement


def test_f01_glm_standard_mode_rejects_fp32_activation():
    case = dc.replace(_case(), model=dc.replace(_case().model,
                                                activation_dtype="fp32"))
    report = evaluate_hybrid_kv_state(case, _glm(use_replayssm=False))
    assert report["status"] == "rejected"
    assert "glm_kda_execution:glm_kda_requires_bf16_activation" in report[
        "errors"]
    assert report["linear_state"]["glm_execution"]["ok"] is False


def test_f01_glm_validates_every_layer_in_both_modes():
    base = _case()
    layers = list(base.model.layer_plans)
    kda = [index for index, plan in enumerate(layers)
           if plan.attention == "kda"]
    plan = layers[kda[1]]
    layers[kda[1]] = dc.replace(
        plan, conv_kernel=3,
        conv_state_elements=plan.conv_width * (3 - 1))
    case = dc.replace(base, model=dc.replace(base.model,
                                             layer_plans=tuple(layers)))
    for recoverssm in (False, True):
        report = evaluate_hybrid_kv_state(case, _glm(
            use_replayssm=recoverssm))
        assert report["status"] == "rejected"
        assert "glm_kda_execution:glm_kda_speculation_requires_conv_kernel_4" \
            in report["errors"]


@pytest.mark.parametrize("value", [1.0, 0.0, -5.5])
def test_f01_glm_gate_lower_bound_must_be_in_range(value):
    base = _case()
    layers = list(base.model.layer_plans)
    kda = [index for index, plan in enumerate(layers)
           if plan.attention == "kda"]
    layers[kda[0]] = dc.replace(layers[kda[0]],
                                kda_gate_lower_bound=value)
    case = dc.replace(base, model=dc.replace(base.model,
                                             layer_plans=tuple(layers)))
    report = evaluate_hybrid_kv_state(case, _glm())
    assert report["status"] == "rejected"
    assert "glm_kda_execution:glm_kda_gate_lower_bound_out_of_range" in report[
        "errors"]


def test_f01_glm_align_block_must_hold_the_spec_window():
    valid = validate_glm_kda_execution(
        kind="kda", activation_dtype="bf16", key_dim=128, value_dim=128,
        conv_kernel=4, speculative_k=4, cache_mode="align", block_size=128,
        kda_gate_lower_bound=-5.0, mamba_block_size=128)
    assert valid.ok
    invalid = validate_glm_kda_execution(
        kind="kda", activation_dtype="bf16", key_dim=128, value_dim=128,
        conv_kernel=4, speculative_k=4, cache_mode="align", block_size=8,
        kda_gate_lower_bound=-5.0, mamba_block_size=4)
    assert "glm_kda_align_block_smaller_than_spec_window" in invalid.codes


# ------------------------------------------------ F02 typed c8 page planes


def test_f02_c8_rope_stays_bf16_and_scale_plane_is_priced():
    case = _case("kimi-k3")
    case = dc.replace(case, model=dc.replace(case.model, kv_dtype="int8"))
    report = evaluate_hybrid_kv_state(case, _kimi(
        enable_kv_nz=True, pd_disaggregated=True, is_kv_consumer=True,
        kv_connector="MooncakeConnectorV2", batch=1, sequence_tokens=128,
        prefix_hit_tokens=0))
    assert report["errors"] == []
    nz = report["layout"]["kv_nz"]["layout"]
    planes = {plane["name"]: plane for plane in nz["planes"]}
    assert planes["nope"]["c0_elements"] == 32
    assert planes["nope"]["dtype"] == "int8"
    assert planes["rope"]["c0_elements"] == 16
    assert planes["rope"]["dtype"] == "bf16"
    assert planes["nope"]["element_bytes"] == 1.0
    assert planes["rope"]["element_bytes"] == 2.0
    assert nz["total_bytes"] == 82_432
    assert nz["bytes_per_block"] == 82_432
    scale = nz["scale_plane"]
    assert scale["dtype"] == "fp32" and scale["bytes"] == 512
    assert report["layout"]["mixed_pool"]["stages"][0][
        "attn_page_bytes"] == 82_432
    pd = report["layout"]["component_major_pd"]
    assert pd["nope_dtype"] == "int8" and pd["rope_dtype"] == "bf16"
    assert pd["scale_bytes"] == 512
    assert pd["component_major"]["payload_bytes"] == 24 * 82_432
    assert pd["component_major"]["payload_bytes"] == 1_978_368
    assert pd["split"]["payload_bytes"] == 1_978_368


# ------------------------------------------------------ F03 MLAPO predicate


def test_f03_a3_unquantized_glm_is_not_a_mlapo_baseline():
    report = evaluate_hybrid_kv_state(_case(), _glm(
        enable_kv_nz=True, pd_disaggregated=True, is_kv_consumer=True,
        kv_connector="MooncakeConnectorV2"))
    assert report["model"]["mlapo_quant"] == "native"
    assert report["model"]["mlapo_eligible"] is False
    assert report["layout"]["kv_nz"]["mlapo_disabled_by_nz"] is False


def test_f03_a5_native_weights_are_a_real_mlapo_baseline():
    report = evaluate_hybrid_kv_state(
        _case(sku="Ascend950DT_9581", topology="a5_pod"), _glm(
            enable_kv_nz=True, pd_disaggregated=True, is_kv_consumer=True,
            kv_connector="MooncakeConnectorV2"))
    assert report["model"]["mlapo_eligible"] is True
    assert report["layout"]["kv_nz"]["mlapo_disabled_by_nz"] is True


def test_f03_disabled_flag_and_draft_role_remove_eligibility():
    model = _case().model
    assert model_mlapo_eligible(model, hardware_arch="A5",
                                enable_mlapo=False) is False
    assert model_mlapo_eligible(model, hardware_arch="A3",
                                decode_instance=True) is False
    assert model_mlapo_eligible(model, hardware_arch="A5",
                                is_draft=True) is False


# ------------------------------------- F04 grouping, demand, stride, batch


def test_f04_glm_groups_are_the_longest_consecutive_run():
    case = _case()
    groups = longest_run_recurrent_groups(
        [(plan.index, plan.attention) for plan in case.model.layer_plans])
    assert [len(group) for group in groups] == [12, 11, 11]
    report = evaluate_hybrid_kv_state(case, _glm())
    pool = report["layout"]["mixed_pool"]
    assert pool["global_recurrent_group_sizes"] == [12, 11, 11]
    assert pool["main_slot_count"] == 12
    assert pool["main_page_bytes"] * pool["main_slot_count"] == 7_299_072


def test_f04_pp_projects_global_groups_and_page_class():
    case = dc.replace(_case(pp=2), runtime=dc.replace(
        _case(pp=2).runtime, pp_layer_partition=(1, 44)))
    report = evaluate_hybrid_kv_state(case, _glm(
        batch=8, sequence_tokens=2048))
    pool = report["layout"]["mixed_pool"]
    assert pool["global_recurrent_group_sizes"] == [12, 11, 11]
    assert pool["projected_group_sizes_per_stage"] == [[1, 0, 0],
                                                       [11, 11, 11]]
    assert pool["global_main_page_bytes"] == 608_256
    assert [stage["main_page_bytes"] for stage in pool["stages"]] == [
        608_256, 608_256]
    assert pool["stages"][0]["attention_blocks"] == 0
    assert pool["stages"][0]["blocks"] == 8
    assert pool["stages"][1]["blocks"] == 128 + 24


def test_f04_pool_covers_every_reported_state_byte():
    report = evaluate_hybrid_kv_state(
        _case(tp=8), HybridKvStateOptions(
            use_replayssm=False, num_spec=4, batch=8,
            sequence_tokens=128, graph_analytical_bound=True))
    pool = report["layout"]["mixed_pool"]
    linear = report["linear_state"]["memory"]
    assert linear["total_bytes"] == 771_522_560
    assert pool["attention_blocks"] == 8
    assert pool["recurrent_blocks"] == 120
    # The compressor-tail group contributes one global block per request.
    assert pool["tail_blocks"] == 8
    assert pool["blocks"] == 136
    # Standard mode's canonical page is state 524,288 + conv 43,008.
    assert pool["main_page_bytes"] == 567_296
    assert pool["physical_pool_bytes"] == 136 * (12 * 567_296 + 11 * 131_072)
    assert pool["physical_pool_bytes"] >= linear["total_bytes"]


def test_f04_state_page_stride_reaches_pool_and_peak():
    plain = evaluate_hybrid_kv_state(
        _case(), _glm(batch=2, sequence_tokens=2048))
    strided = evaluate_hybrid_kv_state(
        _case(), _glm(batch=2, sequence_tokens=2048,
                      state_page_stride_bytes=1_000_000))
    assert plain["linear_state"]["memory"]["total_bytes"] == 41_361_408
    assert strided["linear_state"]["memory"]["total_bytes"] == 68_000_000
    assert plain["layout"]["mixed_pool"]["main_page_bytes"] == 608_256
    assert plain["layout"]["mixed_pool"]["raw_state_page_bytes"] == 608_256
    assert strided["layout"]["mixed_pool"]["main_page_bytes"] == 1_000_000
    assert strided["layout"]["mixed_pool"]["raw_state_page_bytes"] == 608_256
    assert strided["memory"]["adjusted_peak_hbm_bytes"] > plain["memory"][
        "adjusted_peak_hbm_bytes"]


def test_f04_align_state_groups_demand_block_aligned_checkpoint_pages():
    report = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192),
        _kimi(batch=2, sequence_tokens=4096))
    pool = report["layout"]["mixed_pool"]
    # 4096 / 128 = 32 checkpoint blocks per group per sequence, three groups.
    assert pool["recurrent_blocks"] == 3 * 32 * 2
    assert pool["attention_blocks"] == 64
    assert pool["blocks"] == 64 + 192
    assert pool["physical_pool_bytes"] >= report["linear_state"]["memory"][
        "total_bytes"]


def test_f04_declared_baseline_pages_are_batch_multiplied():
    case = _case(tp=16)
    kda = tuple(plan.index for plan in _kda_layers(case))
    groups = ({"layers": kda, "tokens_per_page": 128,
               "bytes_per_layer_page": 524_288, "fixed_pages": 1,
               "components": ("recurrent_state", "conv_state")},)
    declared = _case(tp=16, memory={"cache_page_groups": groups})
    pricing = baseline_pricing(declared, tokens=2048, active_sequences=8)
    # 34 layers * 524,288 * 8 sequences + 11 MLA layers over 128 blocks.
    assert pricing.declared_layer_bytes == 34 * 524_288 * 8
    assert pricing.declared_layer_bytes == 142_606_336
    assert pricing.replaced_per_stage == pytest.approx(
        (184_549_376 + 142_606_336,))
    assert pricing.replaced_per_stage == pytest.approx((327_155_712,))


def test_f04_declared_components_outside_the_replaced_set_are_ignored():
    case = _case(tp=16)
    kda = tuple(plan.index for plan in _kda_layers(case))
    unrelated = ({"layers": kda, "tokens_per_page": 128,
                  "bytes_per_layer_page": 524_288, "fixed_pages": 1,
                  "components": ("index_k",)},)
    declared = _case(tp=16, memory={"cache_page_groups": unrelated})
    pricing = baseline_pricing(declared, tokens=2048, active_sequences=8)
    plain = baseline_pricing(_case(tp=16), tokens=2048, active_sequences=8)
    assert pricing.declared_layer_bytes == 0.0
    assert pricing.declared_group_bytes == 0.0
    # The real state components remain uncovered and are priced per layer.
    assert pricing.replaced_per_stage == plain.replaced_per_stage


# ------------------------------------- F05 staging, requalification, PCP


def test_f05_nz_staging_is_charged_only_on_mla_stages():
    case = dc.replace(_case(pp=2), runtime=dc.replace(
        _case(pp=2).runtime, pp_layer_partition=(1, 44)))
    base = HybridKvStateOptions(num_spec=4, batch=2, sequence_tokens=2048,
                                graph_analytical_bound=True)
    nz = evaluate_hybrid_kv_state(case, dc.replace(
        base, enable_kv_nz=True, pd_disaggregated=True,
        is_kv_consumer=True, kv_connector="MooncakeConnectorV2"))
    assert nz["memory"]["nz_staging_per_stage"] == [0.0, 4_194_304.0]
    transient = next(
        component for component in nz["memory"]["baseline"]["components"]
        if component["name"] == "transient_workspace_peak_bytes")
    assert transient["per_stage_bytes"][0] == 956_920_832.0
    assert transient["per_stage_bytes"][1] == 3_158_930_432.0


def test_f05_signed_release_requalifies_and_positive_delta_disqualifies():
    base_case = _case(tp=16)
    kwargs = dict(sequence_tokens=128, sequences=1,
                  graph_analytical_bound=True)
    baseline = analyze_memory_breakdown(base_case, **kwargs)
    cap = int((baseline.peak_hbm_bytes - 1000)
              / base_case.runtime.gpu_memory_utilization)
    case = dc.replace(base_case, machine=dc.replace(
        base_case.machine, dies=tuple(
            dc.replace(device, memory_bytes=cap)
            for device in base_case.machine.dies)))
    over = analyze_memory_breakdown(case, **kwargs)
    assert over.status == "memory_infeasible" and not over.oom_feasible
    released = apply_state_pool_delta(over, per_stage_bytes=(-2000,),
                                      allow_negative=True)
    assert released.status == "qualified"
    assert released.oom_feasible and released.memory_qualified
    assert released.headroom_bytes == pytest.approx(1000.0)
    grown = apply_state_pool_delta(released, per_stage_bytes=(5000,),
                                   allow_negative=True)
    assert grown.status == "memory_infeasible" and not grown.oom_feasible
    assert grown.headroom_bytes == pytest.approx(-4000.0)


def test_f05_pcp_shards_kda_heads_when_tp_is_one():
    case = _case(tp=1, pcp=2)
    report = evaluate_hybrid_kv_state(case, _glm(
        batch=1, sequence_tokens=128, kv_connector="MooncakeConnectorV2",
        kv_role="kv_producer", pd_disaggregated=True))
    assert report["errors"] == []
    assert report["parallel"]["pcp"] == 2
    pages = report["linear_state"]["memory"]["layer_pages"]
    assert all(page["page_bytes"] == 2_433_024 for page in pages)
    # 32 local heads, not 64.
    assert report["linear_state"]["memory"]["layer_pages"][0][
        "recurrent_state_bytes"] == 32 * 128 * 128 * 4
    assert report["layout"]["mixed_pool"]["main_page_bytes"] == 2_433_024


# ------------------------------ F06 active SWA, draft weights, aux stages


def test_f06_bounded_swa_scales_with_batch():
    values = []
    for batch in (1, 2, 8):
        report = evaluate_hybrid_kv_state(
            _case("deepseek-v4.1-flash", max_model_len=8192),
            _ds41(batch=batch, extra_retained_tokens=8))
        values.append(report["dspark"]["modeled_draft_kv_bytes"])
    assert values == [884_736, 1_769_472, 7_077_888]


def test_f06_explicit_activation_always_prices_the_draft_weights():
    for key, expected in (("deepseek-v4.1-flash", 1_240_494_632),
                          ("kimi-k3", 709_314_560)):
        case = _case(key, max_model_len=8192)
        explicit = evaluate_hybrid_kv_state(case, _ds41(dspark_active=True)
                                            if "deepseek" in key
                                            else _kimi(dspark_active=True))
        runtime_active = dc.replace(case, runtime=dc.replace(
            case.runtime, dspark=DsparkSpec(3, (.8, .64, .512))))
        active = evaluate_hybrid_kv_state(
            runtime_active, _ds41(dspark_active=True)
            if "deepseek" in key else _kimi(dspark_active=True))
        assert explicit["dspark"]["active"] is True
        assert explicit["memory"]["baseline"]["draft_weight_bytes"] == expected
        assert active["memory"]["baseline"]["draft_weight_bytes"] == expected
        assert explicit["memory"]["adjusted_peak_hbm_bytes"] == active[
            "memory"]["adjusted_peak_hbm_bytes"]


def test_f06_aux_scratch_is_transient_on_producing_stages():
    case = dc.replace(_case("deepseek-v4.1-flash", pp=2,
                            max_model_len=8192),
                      runtime=dc.replace(
                          _case("deepseek-v4.1-flash", pp=2,
                                max_model_len=8192).runtime,
                          pp_layer_partition=(37, 3)))
    report = evaluate_hybrid_kv_state(case, _ds41(batch=2,
                                                  sequence_tokens=2048))
    per_stage = report["memory"]["aux_scratch_per_stage"]
    assert per_stage == [81_920.0, 245_760.0]
    assert report["memory"]["aux_scratch_bytes"] == 245_760.0
    from mindie_sim.memory_breakdown import (
        analyze_memory_breakdown, apply_state_pool_delta)
    ledger = analyze_memory_breakdown(
        case, sequence_tokens=2048, sequences=2,
        graph_analytical_bound=True)
    adjusted = apply_state_pool_delta(
        ledger,
        per_stage_bytes=tuple(report["memory"]["net_delta_per_stage"]),
        transient_per_stage_bytes=tuple(per_stage), allow_negative=True)
    scratch_detail = dict(adjusted.components[5].detail)
    assert dict(adjusted.detail)["hybrid_state_pool_bytes"] == pytest.approx(
        sum(report["memory"]["net_delta_per_stage"]))
    assert "aux_scratch_stage_peak_bytes" not in dict(
        adjusted.components[2].detail)
    assert scratch_detail["aux_scratch_stage_peak_bytes"] == 245_760.0
    assert scratch_detail["aux_scratch_per_stage_bytes"] == per_stage


# ------------------------------------------- F14 universal input typing


def test_f14_public_helpers_reject_raw_types_before_coercion():
    with pytest.raises(ModelingError, match="layer_index_invalid"):
        LinearLayer(name="l", index=True)
    with pytest.raises(ModelingError, match="layer_index_invalid"):
        LinearLayer(name="l", index=1.9)
    with pytest.raises(ModelingError, match="layer_index_invalid"):
        LinearLayer(name="l", index=-1)
    with pytest.raises(ModelingError, match="compute_flops_per_s_invalid"):
        verify_commit_cost(_config(), [_layer()], batch=1,
                           sequence_tokens=128, hbm_bytes_per_s=1e12,
                           compute_flops_per_s=0)
    with pytest.raises(ModelingError, match="hbm_bytes_per_s_invalid"):
        verify_commit_cost(_config(), [_layer()], batch=1,
                           sequence_tokens=128, hbm_bytes_per_s=float("nan"),
                           compute_flops_per_s=1e12)
    with pytest.raises(ModelingError, match="fused_op_invalid"):
        nd_to_nz_reformat_cost(pages=1, layers=1, nd_page_bytes=1024,
                               hbm_bytes_per_s=1e12, fused_op=0)
    with pytest.raises(ModelingError, match="baseline_bytes_invalid"):
        grouped_pool_layout(
            attn_group_size=1, attn_page_bytes=128,
            recurrent_groups=((64,),), attention_blocks=1,
            recurrent_blocks=(1,)).reconcile(float("nan"))
    with pytest.raises(ModelingError, match="baseline_bytes_invalid"):
        grouped_pool_layout(
            attn_group_size=1, attn_page_bytes=128,
            recurrent_groups=((64,),), attention_blocks=1,
            recurrent_blocks=(1,)).reconcile("100")
    with pytest.raises(ModelingError, match="sequence_start_invalid"):
        precopy_cost(_config(), [_layer()], sequence_start=-2,
                     sequence_end=-1, batch=1, hbm_bytes_per_s=1e12)


def test_f14_rates_and_aux_ids_are_validated_before_use():
    from mindie_sim.dspark_kv_physics import draft_row_bytes_per_token
    with pytest.raises(ModelingError, match="not_divisible_by_tp"):
        draft_row_bytes_per_token(kv_heads=10, head_dim=64, tp=3,
                                  dtype="bf16")
    with pytest.raises(ModelingError, match="aux_layer_ids_invalid"):
        HybridKvStateOptions(dspark_aux_layer_ids=(True,))
    case = _case()
    from mindie_sim.hybrid_kv_state_options import _machine_limits
    broken = dc.replace(case.machine, dies=tuple(
        dc.replace(device, rates=tuple(
            (name, "1000") if name == "hbm" else (name, value)
            for name, value in device.rates))
        for device in case.machine.dies))
    with pytest.raises(ModelingError, match="hbm_invalid"):
        _machine_limits(dc.replace(case, machine=broken))


# ------------------------------------------- F15 MCP ASCII + code priority


def test_f15_all_diagnostic_codes_survive_and_payload_is_ascii():
    report = evaluate_hybrid_kv_state(_case(), _glm())
    report = json.loads(json.dumps(report))
    report["errors"] = []
    report["diagnostics"] = [f"e{index}" for index in range(8)]
    envelope = hybrid_kv_state_mcp_envelope(report)
    assert envelope["diagnostics"] == [f"e{index}" for index in range(8)]
    assert "codes_truncated" not in envelope
    payload = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
    assert all(0x20 <= ord(char) <= 0x7E for char in payload)


def test_f15_numeric_fields_reject_unicode_digits_structurally():
    report = evaluate_hybrid_kv_state(_case(), _glm())
    report = json.loads(json.dumps(report))
    report["linear_state"]["memory"]["state_slots_per_sequence"] = "\u0661"
    with pytest.raises(ModelingError, match="numeric_invalid"):
        hybrid_kv_state_mcp_envelope(report)


def test_f15_code_overflow_is_explicit():
    report = evaluate_hybrid_kv_state(_case(), _glm())
    report = json.loads(json.dumps(report))
    report["errors"] = [f"very_long_diagnostic_code_{index:02d}"
                        for index in range(20)]
    report["diagnostics"] = list(report["errors"])
    envelope = hybrid_kv_state_mcp_envelope(report)
    assert envelope.get("codes_truncated") is True
    assert envelope["total_error_count"] == 20
    payload = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
    assert all(0x20 <= ord(char) <= 0x7E for char in payload)


FIX12_VARIANTS = ((int, -1), (int, 0), (int, 1), (float, -1.0),
                  (Any, -1), (object, -1), (Union[int, str], -1))


def fix12_extend(record, additions, depth=1):
    """Construct an ordinary subtype while leaving every inherited field intact."""
    cls = dc.make_dataclass("Fix12" + type(record).__name__,
        [(name, annotation, dc.field(kw_only=True)) for name, annotation, _ in additions],
        bases=(type(record),), frozen=type(record).__dataclass_params__.frozen)
    for level in range(depth - 1):
        cls = type(f"Fix12Level{level}", (cls,), {})
    original = {f.name: getattr(record, f.name) for f in dc.fields(record)}
    result = cls(**original, **{name: item for name, _, item in additions})
    assert all(getattr(result, name) == item for name, item in original.items())
    return result


@pytest.mark.parametrize("annotation,item", FIX12_VARIANTS)
@pytest.mark.parametrize("depth", (1, 4))
@pytest.mark.parametrize("capture", ("none", "full", "piecewise"))
def test_fix12_signed_extensions_across_six_families_and_ten_consumers(annotation, item, depth, capture):
    from mindie_sim.catalog.types import DraftSpec
    from mindie_sim.nr_s7_validate import _validate_nested
    from test_nr_s7_fix2_regressions_r2 import fix9_consumer_calls
    case = _case()
    records = (case.model, case.model.mla, case.model.layer_plans[0],
        DraftSpec("mtp", 1, 128, 256, 1, 1, 128, 128, 128), case.runtime, case.runtime.parallel)
    additions = []
    for record in records:
        extended = fix12_extend(record, [("diagnostic_delta", annotation, item)], depth)
        _validate_nested(extended, "extension")
        additions.append(("extra_" + type(record).__name__, type(record), extended))
    # One typed carrier exercises all six families in each real consumer call.
    workload = fix12_extend(case.workload, additions)
    case = dc.replace(case, workload=workload,
                      runtime=dc.replace(case.runtime, graph_capture_mode=capture))
    calls = fix9_consumer_calls(case)
    assert len(calls) == 10
    for call in calls.values():
        assert call() is not None


def test_fix12_unrelated_names_do_not_acquire_base_rules():
    from mindie_sim.nr_s7_validate import _validate_nested
    case = _case()
    names = ("rates", "compute_peaks", "request_id", "dies", "requests", "detail",
             "pp_layer_partition", "gpu_memory_utilization", "tp", "max_num_seqs")
    controls = ((case.model, [(name, int, -1) for name in names]),
                (case.runtime, [("catalog_facts", str, "diagnostic")]),
                (case.model.mla, [("attention", str, "diagnostic")]))
    for record, additions in controls:
        extended = fix12_extend(record, additions, 4)
        _validate_nested(extended, "unrelated")
        # Relaxing a special base annotation must not relax this declared type.
        for name, _, _ in additions:
            with pytest.raises(ModelingError, match=name):
                _validate_nested(dc.replace(extended, **{name: []}), "wrong_type")
    # A name owned by another registered base in the MRO keeps that base's domain.
    layer, parallel = case.model.layer_plans[0], case.runtime.parallel
    cls = dc.make_dataclass("Fix12Combined", [], bases=(type(parallel), type(layer)), frozen=True)
    combined = cls(**{f.name: getattr(r, f.name) for r in (layer, parallel) for f in dc.fields(r)})
    _validate_nested(combined, "combined")  # Layer index 0 and native ID -1 are legal.
    for changes in ({"tp": 0}, {"head_dim": -1}):
        with pytest.raises(ModelingError):
            _validate_nested(dc.replace(combined, **changes), "combined_invalid")


def test_fix12_overlapping_metadata_keeps_each_base_contract():
    from mindie_sim.nr_s7_validate import _validate_nested
    from mindie_sim.schema.model import ModelSpec, WorkloadSpec
    case = _case()
    for bases in ((ModelSpec, WorkloadSpec), (WorkloadSpec, ModelSpec)):
        cls = type("Fix12Overlapping", bases, {})
        combined = object.__new__(cls)
        for record in (case.model, case.workload):
            for field in dc.fields(record):
                object.__setattr__(combined, field.name, getattr(record, field.name))
        _validate_nested(combined, "overlapping")
        # ModelSpec permits arbitrary metadata values; WorkloadSpec requires str.
        object.__setattr__(combined, "metadata", (("diagnostic", []),))
        with pytest.raises(ModelingError, match="metadata"):
            _validate_nested(combined, "overlapping")


def test_fix12_any_object_payloads_and_typed_nested_extensions():
    from mindie_sim.nr_s7_validate import _validate_nested
    from mindie_sim.schema.model import ModelSpec
    from test_nr_s7_fix2_regressions_r2 import fix9_consumer_calls, fix10_reject_before_body
    case = _case()
    invalid = dc.replace(case.model, num_heads=0)
    box_type = dc.make_dataclass("Fix12Box", [("model", ModelSpec)], frozen=True)
    shapes = ((Union[ModelSpec, int], invalid), (tuple[ModelSpec, ...], (invalid,)),
              (list[ModelSpec], [invalid]), (dict[str, ModelSpec], {"bad": invalid}),
              (box_type, box_type(invalid)))
    for annotation in (Any, object):
        for _, item in shapes:
            _validate_nested(fix12_extend(case.model, [("payload", annotation, item)]), "opaque")
        extended = fix12_extend(case.model, [("payload", annotation, [item for _, item in shapes])])
        for call in fix9_consumer_calls(dc.replace(case, model=extended)).values():
            assert call() is not None
        del extended.__dict__["payload"]
        with pytest.raises(ModelingError, match="payload"):
            _validate_nested(extended, "missing")
    for annotation, item in shapes:
        extended = fix12_extend(case.model, [("payload", annotation, item)])
        with pytest.raises(ModelingError, match="num_heads"):
            _validate_nested(extended, "typed")
        fix10_reject_before_body(fix9_consumer_calls(dc.replace(case, model=extended)).values())
    _validate_nested(fix12_extend(case.model, [("payload", Union[ModelSpec, int], -1)]), "union_int")


@pytest.mark.parametrize("route,capture", [(r, "piecewise") for r in range(7)] + [(0, "none"), (0, "full")])
def test_fix13_typed_containers_reject_before_all_ten_consumers(route, capture):
    from collections import UserDict, UserList
    from collections.abc import Mapping, Sequence, Set as AbstractSet
    from mindie_sim.nr_s7_validate import _validate_nested
    from mindie_sim.schema.model import ModelSpec, MlaSpec
    from test_nr_s7_fix2_regressions_r2 import fix9_consumer_calls, fix10_reject_before_body
    case = _case()
    box = dc.make_dataclass("Fix13Box", [("model", ModelSpec)], frozen=True)
    annotation, wrap, valid, invalid = (
        (Sequence[ModelSpec], lambda x: UserList([x]), case.model, dc.replace(case.model, num_heads=0)),
        (dict[MlaSpec, int], lambda x: {x: 1}, case.model.mla, dc.replace(case.model.mla, kv_lora_rank=0)),
        (Mapping[MlaSpec, int], lambda x: UserDict({x: 1}), case.model.mla, dc.replace(case.model.mla, kv_lora_rank=0)),
        (set[MlaSpec], lambda x: {x}, case.model.mla, dc.replace(case.model.mla, kv_lora_rank=0)),
        (frozenset[MlaSpec], lambda x: frozenset([x]), case.model.mla, dc.replace(case.model.mla, kv_lora_rank=0)),
        (AbstractSet[MlaSpec], lambda x: {x: 1}.keys(), case.model.mla, dc.replace(case.model.mla, kv_lora_rank=0)),
        (box, box, case.model, dc.replace(case.model, num_heads=0)))[route]
    for depth in (1, 5):
        for item in (invalid, valid):
            payload = wrap(item)
            extended = fix12_extend(case.model, [("payload", annotation, payload)], depth)
            calls = fix9_consumer_calls(dc.replace(case, model=extended,
                runtime=dc.replace(case.runtime, graph_capture_mode=capture)))
            assert len(calls) == 10
            if item is invalid:
                with pytest.raises(ModelingError, match="payload"):
                    _validate_nested(extended, "typed")
                fix10_reject_before_body(calls.values())
            else:
                _validate_nested(extended, "typed")
                assert all(call() is not None for call in calls.values())
            assert extended.payload is payload


def test_fix13_set_element_types_and_container_cycles_preserve_opacity():
    from collections import UserDict, UserList
    from collections.abc import Sequence, Set as AbstractSet
    from mindie_sim.nr_s7_validate import _matches_type, _validate_nested, typed_inputs
    from mindie_sim.schema.model import ModelSpec
    from types import SimpleNamespace
    from mindie_sim.hybrid_kv_state_options import model_is_sparse_sfa
    case = _case()
    bad = dc.replace(case.model, num_heads=0)
    for origin, wrap in ((set, set), (frozenset, frozenset), (AbstractSet, lambda x: dict.fromkeys(x).keys())):
        for item in (True, "1", 1.5, float("inf"), 10**400):
            assert not _matches_type(wrap([item]), origin[int])
            extended = fix12_extend(case.model, [("payload", origin[int], wrap([item]))])
            with pytest.raises(ModelingError, match="payload"):
                _validate_nested(extended, "types")
        _validate_nested(fix12_extend(case.model, [("payload", origin[int], wrap([-1, 0, 1]))]), "signed")
    for value in ("text", b"bytes", bytearray(b"bytes"), {}):
        assert not _matches_type(value, Sequence[Any])
    sequence, mapping = UserList(), UserDict()
    sequence.append(mapping)
    mapping["cycle"] = sequence
    _validate_nested(sequence, "cycle")
    mapping["bad"] = bad
    with pytest.raises(ModelingError, match="num_heads"):
        _validate_nested(sequence, "cycle")
    @dc.dataclass
    class CyclicSet(AbstractSet):
        items: object = ()
        def __iter__(self):
            return iter(self.items)
        def __len__(self):
            return len(self.items)
        def __contains__(self, item):
            return any(item is value for value in self.items)
    cyclic_set = CyclicSet()
    cyclic_set.items = (cyclic_set,)
    _validate_nested(cyclic_set, "set_cycle")
    cyclic_set.items = (cyclic_set, bad)
    with pytest.raises(ModelingError, match="num_heads"):
        _validate_nested(cyclic_set, "set_cycle")
    for base, payload in ((UserList, [bad]), (UserDict, {dc.replace(case.model.mla, kv_lora_rank=0): 1})):
        container = dc.make_dataclass("Fix13Container", [("data", object)], bases=(base,))
        with pytest.raises(ModelingError, match="num_heads|kv_lora_rank"):
            _validate_nested(container(payload), "dataclass_container")
    opaque = {"sequence": sequence, "set": {dc.replace(case.model.mla, kv_lora_rank=0)}}
    box = dc.make_dataclass("Fix13Opaque", [("payload", Any), ("other", object)], frozen=True)
    wrapped = box(opaque, opaque)
    _validate_nested(fix12_extend(case.model, [("payload", box, wrapped)]), "box")
    assert wrapped.payload is wrapped.other is opaque
    for annotation in (Any, object):
        def identity(value):
            return value
        identity.__annotations__ = {"value": annotation}
        assert typed_inputs(identity)(wrapped) is wrapped
    for text in (None, SimpleNamespace(index_topk=0), SimpleNamespace(index_topk=8, compress_ratios=None)):
        hf = dc.make_dataclass("Fix13HF", [("hf_text_config", Any), ("hf_config", object), ("payload", Any)])
        wrapper = hf(text, SimpleNamespace(), opaque)
        assert model_is_sparse_sfa(wrapper) == (text is not None and not hasattr(text, "compress_ratios"))
        assert wrapper.payload is opaque
