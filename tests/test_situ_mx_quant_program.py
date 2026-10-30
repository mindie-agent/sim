"""Source-backed SituMxQuant complete-call program invariants."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.family_situ_mx import (
    Config, PROFILE, config_from_group, make_op, owner_rows, recipe_book, tiling_for,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import InvocationTiming


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROFILE = "situ_mx_quant_arch35_2a9bd1b0"


def _case(profile=None):
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant"]
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
        "rmsnorm_quant"] = "vllm_ascend_native_norm"
    case = parse_case(raw)
    quant = replace(
        case.model.quant, weight_format="fp8_e4m3", expert_weight_format="fp8_e4m3",
        weight_block=(32, 32), activation_format="fp8_e4m3",
        activation_group_size=32, activation_scale_format="fp8_e8m0",
        activation_scale_pack=2, exclusions=())
    case = replace(case, model=replace(case.model, quant=quant))
    if profile is None:
        return case
    kernels = {**case.runtime.kernels, "lowerings": {
        **case.runtime.kernels.get("lowerings", {}),
        "native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["situ_quant"],
                        "program_profiles": {"situ_quant": profile}},
        },
    }}
    return replace(case, runtime=replace(case.runtime, kernels=kernels))


def _group(case, width=768):
    ops = target_operators(case, phase="decode", tokens=1, context=16,
                           resource_fusion=False)
    return next((d, m) for d, m in _groups(ops)
                if d["role"] == "situ_quant" and d["tensor_view"]["width"] == width)


def _geometry(case):
    return Geometry.from_devices(case.machine.dies[:8])


def _prepared(case, op):
    return RecipeBook.from_mapping(case.runtime.kernels).prepare(op, _geometry(case))[0]


def _external(members):
    ids = {op.op_id for op in members}
    return {dep: 0 for op in members for dep in op.deps if dep not in ids}


def test_t1_h768_runs_three_owned_owner_programs_and_split_read_pipe():
    case = _case(PROFILE)
    declaration, members = _group(case)
    fused = lower_native_fusions(members, case.runtime)[0]
    prepared = _prepared(case, fused)
    frontend = target_operators(case, phase="decode", tokens=1, context=16)
    selected = next(op for op in frontend
                    if op.family == "native_fusion" and op.local_shape == (1, 768)
                    and dict(op.metadata).get("fusion_role") == "situ_quant")
    assert _prepared(case, selected).hbm_bytes == prepared.hbm_bytes

    # Independent BF16/FP8/E8M0 payload: one 2H input, H payload, 2 scales/64.
    assert prepared.local_shape == (1, 768)
    assert prepared.hbm_bytes == 3072 + 792
    assert prepared.mte3_bytes == 792
    assert dict(prepared.metadata)["resource_work"]["mte2_aiv"] == 3072
    assert prepared.vector_ops > sum(op.vector_ops for op in members)

    report = graph_costs(case, (fused,), external_inputs=_external(members),
                         isolated=False, trace_limit=600)
    loads = [event for event in report["events"] if ":tile0:load" in event["task"]]
    assert len(loads) == 3
    assert {tuple(event["placement"]["mte2_aiv@r0:s0"]) for event in loads} == {
        (0,), (1,), (2,)}
    assert {tuple(event["placement"]["vector@r0:s0"]) for event in loads} == {
        (0,), (1,), (2,)}
    assert all(event["serviced"]["mte2_aiv@r0:s0"] == 1024 for event in loads)
    assert report["execution"]["serviced"]["mte2_aiv@r0:s0"] == 3072
    assert report["execution"]["serviced"]["mte2@r0:s0"] == 0
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0


def test_tail_widths_keep_actual_payload_and_256_wide_fp8_conversion():
    case = _case()
    geometry = _geometry(case)
    for width, payload, write, owner_count, minimum_lanes in (
            (65, 2 * 65 * 2, 69, 1, 6 * 256),
            (257, 2 * 257 * 2, 267, 2, 6 * 512)):
        config = Config(rows=1, width=width, output_dtype="fp8_e4m3",
                        beta=4.0, linear_beta=None)
        op, _ = recipe_book().prepare(make_op(config), geometry)
        assert op.hbm_bytes == payload + write
        assert op.mte3_bytes == write
        assert op.vector_ops >= minimum_lanes
        plan = tiling_for(config, geometry)
        assert plan.used_cores == owner_count
        assert owner_rows(config, plan) == (1,) * owner_count


def test_linear_beta_branch_changes_only_declared_situ_lanes():
    case = _case()
    geometry = _geometry(case)
    plain = Config(rows=1, width=65, output_dtype="fp8_e4m3",
                   beta=4.0, linear_beta=None)
    linear = replace(plain, linear_beta=25.0)
    a, _ = recipe_book().prepare(make_op(plain), geometry)
    b, _ = recipe_book().prepare(make_op(linear), geometry)
    assert (b.hbm_bytes, b.mte3_bytes) == (a.hbm_bytes, a.mte3_bytes)
    assert b.vector_ops - a.vector_ops == 22 * 65


def test_flattened_rows_above_aiv_use_unequal_owner_rows():
    case = _case()
    config = Config(rows=130, width=768, output_dtype="fp8_e4m3",
                    beta=4.0, linear_beta=None)
    plan = tiling_for(config, _geometry(case))
    rows = owner_rows(config, plan)
    assert plan.used_cores == 64
    assert len(rows) == 64 and sum(rows) == 130
    assert rows.count(3) == 2 and rows.count(2) == 62


def test_late_input_readiness_and_fixed_ub_lifetimes_release_after_call():
    case = _case()
    config = Config(rows=1, width=768, output_dtype="fp8_e4m3",
                    beta=4.0, linear_beta=25.0)
    result = operator_costs(case, (make_op(config),), recipes=recipe_book(),
                            input_ready_ms=3.0)
    row = result["operators"][0]
    assert row["input_ready_ms"] == 3.0
    assert row["service_before_input"] == {}
    assert row["live_tasks"] == 0
    assert row["remaining_buffers"]["ub@r0:s0"] == 0
    peak = row["peak_buffers_per_unit"]
    assert peak == {f"ub@r0:s0:unit{owner}": 252864 for owner in range(3)}
    assert sum(peak.values()) <= 3 * dict(case.machine.dies[0].buffers)["ub"]


def test_ordinary_group_selection_is_required_and_member_work_is_bypassed():
    selected = _case(PROFILE)
    declaration, members = _group(selected)
    baseline_kernels = {key: value for key, value in selected.runtime.kernels.items()
                        if key != "lowerings"}
    baseline_kernels = {**baseline_kernels, "lowerings": {
        key: value for key, value in selected.runtime.kernels["lowerings"].items()
        if key != "native_fusion"}}
    baseline = replace(selected, runtime=replace(selected.runtime, kernels=baseline_kernels))
    assert lower_native_fusions(members, baseline.runtime) == tuple(members)

    fused = lower_native_fusions(members, selected.runtime)[0]
    prepared = _prepared(selected, fused)
    inflated = tuple(replace(op, vector_ops=op.vector_ops + 10**9,
                             hbm_bytes=op.hbm_bytes + 10**9,
                             mte3_bytes=op.mte3_bytes + 10**9)
                     for op in members)
    inflated_fused = lower_native_fusions(inflated, selected.runtime)[0]
    inflated_prepared = _prepared(selected, inflated_fused)
    assert (inflated_prepared.hbm_bytes, inflated_prepared.mte3_bytes,
            inflated_prepared.vector_ops) == (
                prepared.hbm_bytes, prepared.mte3_bytes, prepared.vector_ops)

    quantification = next(op for op in members if op.family == "quantize")
    with pytest.raises(ModelingError, match="external_activation_intermediate"):
        config_from_group(declaration, members,
                          (members[0].op_id, quantification.op_id))
    changed = {**declaration["tensor_view"], "axis": 0}
    with pytest.raises(ModelingError, match="axis_minus_one"):
        config_from_group({**declaration, "tensor_view": changed}, members,
                          (quantification.op_id,))


def test_explicit_program_profile_does_not_apply_the_legacy_traffic_shortcut():
    selected = _case(PROFILE)
    _, members = _group(selected)
    fused = lower_native_fusions(members, selected.runtime)[0]
    boundary = dict(fused.metadata)["fusion_boundary"]
    assert boundary["program_profile"] == PROFILE
    assert boundary["traffic_profile"] is None
    assert boundary["evidence"]["capture_binary_identity"] == "not_established"
    assert boundary["fidelity"]["pipeline_overlap"].startswith("two_queue_slots")
    assert "observed_65us_remains_unexplained" in " ".join(
        boundary["fidelity"]["limitations"])

    legacy = _case(LEGACY_PROFILE)
    lowerings = {
        **legacy.runtime.kernels["lowerings"],
        "native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["situ_quant"], "traffic_profile": LEGACY_PROFILE},
        },
    }
    legacy = replace(legacy, runtime=replace(
        legacy.runtime, kernels={**legacy.runtime.kernels, "lowerings": lowerings}))
    declaration, legacy_members = _group(legacy)
    legacy_op = lower_native_fusions(legacy_members, legacy.runtime)[0]
    activation = next(op for op in legacy_members if op.family == "situ")
    assert legacy_op.vector_ops == sum(op.vector_ops for op in legacy_members)
    assert legacy_op.hbm_bytes == (
        sum(op.hbm_bytes for op in legacy_members) - 2 * activation.mte3_bytes)
    assert dict(legacy_op.metadata)["fusion_boundary"]["traffic_profile"] == LEGACY_PROFILE


@pytest.mark.parametrize("linear_beta,situ_per_element", ((None, 32), (25.0, 54)))
def test_complete_lane_work_matches_independent_source_arithmetic(linear_beta, situ_per_element):
    case = _case()
    config = Config(1, 768, "fp8_e4m3", 4.0, linear_beta)
    operation = make_op(config)
    prepared, _ = recipe_book().prepare(operation, _geometry(case))
    # Eight 32-value groups per owner: bit-mask32, pair-max16, reduce-max15.
    # Conversion effective lanes: multiply256, casts512, packing adds768.
    exponent = 3 * 8 * (32 + 16 + 15)
    quantize = 3 * (256 + 512 + 768)
    scale = 24 * 13
    expected = 768 * situ_per_element + exponent + scale + quantize
    report = graph_costs(case, (operation,), recipes=recipe_book(),
                         isolated=False, trace_limit=200)
    assert prepared.vector_ops == expected
    assert report["execution"]["serviced"]["vector@r0:s0"] == expected
    exponent_events = [e for e in report["events"] if e["task"].endswith(":maxexp")]
    assert sum(e["serviced"]["vector@r0:s0"] for e in exponent_events) == exponent


def test_single_scratch_buffers_block_next_situ_until_previous_quantization_finishes():
    case = _case()
    device = case.machine.dies[0]
    units = {**dict(device.resource_units), "vector": 1, "mte2_aiv": 1, "mte3": 1}
    device = replace(device, resource_units=tuple(units.items()),
                     buffers=tuple({**dict(device.buffers), "ub": 6656}.items()),
                     buffer_units=tuple({**dict(device.buffer_units), "ub": 1}.items()))
    case = replace(case, machine=replace(case.machine, dies=tuple(
        replace(device, rank=die.rank) for die in case.machine.dies)))
    config = Config(3, 256, "fp8_e4m3", 4.0, None)
    plan = tiling_for(config, Geometry.from_devices((device,)))
    assert (plan.factor_dim0, plan.factor_dim1, plan.used_cores) == (2, 1, 1)
    result = graph_costs(case, (make_op(config),), recipes=recipe_book(),
                         isolated=False, trace_limit=200)
    events = {e["task"].split(":tile")[-1]: e for e in result["events"] if ":tile" in e["task"]}
    assert events["1:situ"]["start_ms"] >= events["0:quantize"]["end_ms"]
    assert result["execution"]["serviced"]["vector@r0:s0"] == 31008
    assert result["execution"]["serviced"]["memory@r0:s0"] == 3864
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert all(value == 0 for value in result["execution"]["live_allocated"].values())


def test_declared_device_minimum_is_once_per_complete_kernel_and_excludes_host():
    case = _case()
    operation = make_op(Config(1, 768, "fp8_e4m3", 4.0, 25.0))
    plain = graph_costs(case, (operation,), recipes=recipe_book(), isolated=False)
    measured = json.loads((ROOT / "examples/invocations/a5-measured-programs.json").read_text())
    classes = measured["machine"]["device"]["kernel_invocations"]
    device = case.machine.dies[0]
    declared = {**dict(device.kernel_invocations), **{
        key: InvocationTiming.from_mapping(value) for key, value in classes.items()}}
    named = replace(device, kernel_invocations=tuple(declared.items()))
    unchanged = replace(case, machine=replace(case.machine, dies=tuple(
        replace(named, rank=die.rank) for die in case.machine.dies)))
    assert graph_costs(unchanged, (operation,), recipes=recipe_book(),
                       isolated=False)["duration_ms"] == plain["duration_ms"]
    # This 10us minimum is an independent causal control, not Situ calibration.
    timed = replace(device, launch_us=7, kernel_invocations=(
        ("vector", InvocationTiming(minimum_ms=.01,
                                    provenance={"basis": "independent_causal_control"})),))
    case = replace(case, machine=replace(case.machine, dies=tuple(
        replace(timed, rank=die.rank) for die in case.machine.dies)))
    device_only = graph_costs(case, (operation,), recipes=recipe_book(),
                              isolated=False, trace_limit=200)
    submitted = graph_costs(case, (operation,), recipes=recipe_book(),
                            isolated=False, include_submission=True, trace_limit=200)
    assert device_only["duration_ms"] == pytest.approx(.01)
    assert submitted["duration_ms"] == pytest.approx(.017)
    assert sum(e["task"].endswith(":minimum") for e in device_only["events"]) == 1
    assert device_only["execution"]["serviced"] == plain["execution"]["serviced"]
    assert submitted["execution"]["serviced"]["host@r0:s0"] == .007


def test_service_alias_pins_each_owner_to_its_physical_dma_partition():
    case = _case()
    device = case.machine.dies[0]
    removed = {"mte2", "mte2_aic", "mte2_aiv", "mte3"}
    rates = {key: value for key, value in device.rates if key not in removed}
    rates["dma"] = 4e12
    units = {key: value for key, value in device.resource_units if key not in removed}
    units["dma"] = 128
    modes = {**dict(device.service_modes), "mte2": ("dma", 4e12),
             "mte3": ("dma", 4e12)}
    device = replace(device, rates=tuple(rates.items()), resource_units=tuple(units.items()),
                     service_modes=tuple(modes.items()))
    case = replace(case, machine=replace(case.machine, dies=tuple(
        replace(device, rank=die.rank) for die in case.machine.dies)))
    result = graph_costs(case, (make_op(Config(1, 768, "fp8_e4m3", 4.0, None)),),
                         recipes=recipe_book(), isolated=False, trace_limit=200)
    transfers = [e for e in result["events"] if e["task"].endswith((":load", ":store"))]
    assert {tuple(e["placement"]["dma@r0:s0"]) for e in transfers} == {
        (0, 1), (2, 3), (4, 5)}
    # Two of 128 DMA units per AIV: each payload transfer takes bytes/(4TB/s/64).
    for event in transfers:
        expected = event["serviced"]["dma@r0:s0"] / (4e9 / 64)
        assert event["end_ms"] - event["start_ms"] == pytest.approx(expected)
    assert result["execution"]["serviced"]["dma@r0:s0"] == 3864
