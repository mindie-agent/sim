"""CPU-only, independent arithmetic checks for opt-in Vector execution plans."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_sim.a3_ops import OpSpec
from mindie_sim.a3_types import MachineSpec
from mindie_sim.operator_plan import EstimateContext, evaluate_kernel_plan
from mindie_sim.operator_vector import (
    CANNBOT_REVISION, CannbotVectorProvider, VectorInstruction,
    VectorInstructionProvider, VectorProfile,
)


def machine(*, arch="TEST", metadata=None, cores=2, bandwidth=1e30):
    return MachineSpec.from_mapping({
        "name": "synthetic-vector-fixture", "arch_family": arch,
        "num_devices": 1, "dies_per_package": 1, "ranks_per_node": 1,
        "device": {"hbm_bytes": 1024 ** 3, "cube_flops_per_s": 1e30,
                   "vector_ops_per_s": 1e30, "hbm_bw_bytes_per_s": bandwidth,
                   "mte_bw_bytes_per_s": bandwidth, "launch_us": 0,
                   "aiv_count": cores, "ub_bytes": 4096},
        # The plan's compute dtype is fp32; a real profile declares its peaks
        # and peak_flops_for no longer guesses an undeclared dtype.
        "compute_peaks": {"fp32": 1e30, "bf16": 1e30, "fp16": 1e30},
        "metadata": metadata or {},
    })


def profile(**changes):
    return VectorProfile(
        name="unit-test-vector", source="assumed:test-only rates",
        architectures=("TEST",), frequency_hz=1e6, simd_bytes=16,
        vector_cores=2, ub_bytes_per_core=4096, tile_elements=4,
        instructions=(VectorInstruction("Add"), VectorInstruction("Mul"),
                      VectorInstruction("Neg"), VectorInstruction("Cast", dtypes=("fp16", "bf16", "fp32")),
                      VectorInstruction("Exp", pipe="transcendental", throughput_cycles=4, latency_cycles=4),
                      VectorInstruction("Div", pipe="transcendental", throughput_cycles=4, latency_cycles=4)),
        **changes,
    )


def op(elements=8, *, instructions=None, extra=None, family="custom", dtype="fp32"):
    flow = {"elements": elements,
            "instructions": instructions or [{"op": "Add", "dtype": "fp32"}],
            "input_bytes_per_element": 4, "output_bytes_per_element": 4,
            "ub_bytes_per_element": 12, **(extra or {})}
    return OpSpec("v", family, "decode", 0, 0, (), (elements,), (elements,),
                  (elements,), (elements,), dtype=dtype,
                  metadata=(("vector_flow", flow),))


def estimate(operation, *, vector_profile=None, target=None):
    result = VectorInstructionProvider(vector_profile or profile()).estimate(
        operation, target or machine(), EstimateContext(compute_dtype="fp32"))
    assert result.plan is not None, result.unsupported_reason
    return result.plan


def compute_stages(plan):
    return [stage for stage in plan.stages if stage.name.endswith(".vector")]


def test_shared_issue_port_can_dominate_both_individual_pipes():
    plan = estimate(op(instructions=[{"op": "Add", "dtype": "fp32", "count": 6},
                                     {"op": "Exp", "dtype": "fp32"}]))
    # Four elements/core = one 16-byte SIMD bundle each. Six ALUs + one Exp
    # share two issue slots -> 7/2 cycles; ALU pipe 6/2=3, Exp pipe 4/2=2.
    stage, = compute_stages(plan)
    assert stage.minimum_duration_ms == pytest.approx(3.5 / 1e6 * 1000)
    detail = plan.provenance["tile_breakdown"][0]
    assert detail["issue_cycles"]["alu"] == 3.5
    assert detail["pipe_cycles"] == {"alu/alu": 3.0, "alu/transcendental": 2.0}
    assert evaluate_kernel_plan(plan, machine(), EstimateContext("fp32"))["elapsed_ms"] == pytest.approx(0.0035)


def test_independent_subpipes_overlap_and_dependent_recurrence_is_explicit():
    independent = estimate(op(instructions=[{"op": "Add", "dtype": "fp32"},
                                            {"op": "Exp", "dtype": "fp32"}]))
    dependent = estimate(op(instructions=[{"op": "Exp", "dtype": "fp32", "dependency": "dependent"}]))
    assert compute_stages(independent)[0].minimum_duration_ms == pytest.approx(0.002)
    assert compute_stages(dependent)[0].minimum_duration_ms == pytest.approx(0.004)


def test_declared_full_instruction_dependency_has_a_critical_path():
    plan = estimate(op(instructions=[{"name": "a", "op": "Add", "dtype": "fp32"},
                                     {"name": "b", "op": "Exp", "dtype": "fp32", "deps": ["a"]}]))
    # A .5-cycle full-instruction stage followed by a 2-cycle full-instruction
    # stage. This explicit ordering is more conservative than element ILP.
    assert compute_stages(plan)[0].minimum_duration_ms == pytest.approx(0.0025)


def test_core_tail_simd_tail_and_bytes_are_accounted_independently():
    plan = estimate(op(17, instructions=[{"op": "Exp", "dtype": "fp32"}]))
    # Balanced 9/8 elements across two cores. Per-core tiles 4,4,1 produce
    # whole SIMD issues in all three waves, but memory only transfers 17 values.
    assert [item["elements"] for item in plan.provenance["tile_breakdown"]] == [8, 8, 1]
    assert [item["heaviest_core_elements"] for item in plan.provenance["tile_breakdown"]] == [4, 4, 1]
    assert [item.minimum_duration_ms for item in compute_stages(plan)] == pytest.approx([0.002] * 3)
    assert plan.resource_totals() == {"hbm": 136.0, "mte2": 68.0, "vector": 17.0, "mte3": 68.0}
    assert set(plan.stages[6].deps) == {"t1.read", "t0.write"}
    assert evaluate_kernel_plan(plan, machine(), EstimateContext("fp32"))["elapsed_ms"] == pytest.approx(0.006)


def test_buffers_control_finite_pipeline_instead_of_applying_a_magic_efficiency():
    # All stages take .032 ms: read/write 32 bytes per wave at 1 MB/s;
    # Exp 2 cycles at 62.5 kHz is .032 ms. Eight elements/wave, three waves.
    target = machine(bandwidth=1e6)
    test_profile = replace(profile(), frequency_hz=62500)
    buffered = estimate(op(24, instructions=[{"op": "Exp", "dtype": "fp32"}]), vector_profile=test_profile, target=target)
    serial = estimate(op(24, instructions=[{"op": "Exp", "dtype": "fp32"}], extra={"buffers": 1}), vector_profile=test_profile, target=target)
    buffered_result = evaluate_kernel_plan(buffered, target, EstimateContext("fp32"))
    serial_result = evaluate_kernel_plan(serial, target, EstimateContext("fp32"))
    assert serial_result["elapsed_ms"] == pytest.approx(9 * .032)
    # Shared HBM read/write competes, so this is between ideal 5 stages and
    # fully serial 9 stages. Explicit stage work, not a posthoc multiplier.
    assert 5 * .032 <= buffered_result["elapsed_ms"] < serial_result["elapsed_ms"]
    assert buffered_result["resource_serviced"]["hbm@s0"] == pytest.approx(192)


def test_auto_swiglu_uses_family_even_when_activation_metadata_says_silu():
    operation = replace(op(8, family="swiglu", dtype="bf16"), metadata=(("activation", "silu"),))
    plan = estimate(operation)
    assert plan.provenance["mapping"] == "assumed_swiglu_fp32_intermediates"
    # 8 bf16 output elements; packed gate/up read 32 bytes, output 16 bytes.
    assert plan.resource_totals()["mte2"] == 32
    assert plan.resource_totals()["mte3"] == 16
    assert [row["op"] for row in plan.provenance["tile_breakdown"][0]["instructions"]] == [
        "Cast", "Neg", "Exp", "Add", "Div", "Mul", "Cast"]
    assert plan.provenance["calibration"] == "assumed"
    assert plan.provenance["tile_breakdown"][0]["instructions"][0]["bundles"] == 2
    assert plan.resource_totals()["vector"] == 8 * 8


def test_optional_provider_import_is_lazy_and_architecture_is_not_guessed(tmp_path):
    provider = VectorInstructionProvider()
    assert provider.estimate(op(), machine(arch="A3"), EstimateContext()).unsupported_reason == "vector_profile:architecture_mismatch"
    native = CannbotVectorProvider(tmp_path / "not-installed")
    assert native.estimate(op(), machine(), EstimateContext()).unsupported_reason == "cannbot:case_not_declared"


@pytest.mark.parametrize("changes,reason", [
    ({"instructions": [{"op": "Unknown", "dtype": "fp32"}]}, "unsupported:Unknown"),
    ({"instructions": [{"op": "Exp", "dtype": "bf16"}]}, "unsupported_dtype:Exp:bf16"),
    ({"instructions": [{"op": "Add", "dtype": "fp32", "deps": ["future"]}]}, "dependency_must_precede"),
    ({"ub_bytes_per_element": 4096}, "ub_capacity_exceeded"),
    ({"active_cores": 3}, "active_cores_exceed"),
    ({"elements": 0}, "elements:invalid"),
    ({"elements": True}, "numeric_required"),
])
def test_invalid_or_unsupported_flows_return_explanations(changes, reason):
    result = VectorInstructionProvider(profile()).estimate(op(extra=changes), machine(), EstimateContext())
    assert result.plan is None
    assert reason in result.unsupported_reason


def test_native_cli_matches_independent_replay_when_external_checkout_is_available(tmp_path):
    repo_text = os.getenv("CANNBOT_REPO")
    if not repo_text:
        pytest.skip("CANNBOT_REPO is optional; no external native engine is bundled")
    repo = Path(repo_text).resolve()
    case = {"case_id": "native-add", "hardware_variant": "ascend950pr_9589",
            "variables": {"N": 4096}, "active_vector_cores": 64,
            "memory": {"gm_read_bytes": "N*4", "gm_write_bytes": "N*4"},
            "vector": {"instructions": [{"op": "Add", "dtype": "fp32", "elements": "N"}]}}
    operation = replace(op(4096), metadata=(("cannbot_case", case),))
    target = machine(arch="A5", metadata={"cannbot_hardware_variant": case["hardware_variant"],
                                          "vector_frequency_hz": 1.65e9}, cores=64, bandwidth=1.6e12)
    estimate_result = CannbotVectorProvider(repo).estimate(operation, target, EstimateContext("fp32"))
    assert estimate_result.plan is not None, estimate_result.unsupported_reason
    plan = estimate_result.plan
    source = tmp_path / "native.jsonl"
    source.write_text(json.dumps(case) + "\n", encoding="utf-8")
    skill = repo / "ops/ops-theoretical-performance-estimation"
    replay = subprocess.run([sys.executable, str(skill / "scripts/evaluate_a5_vector_roofline.py"),
                             "--hardware-json", str(skill / "references/hardware/ascend950pr.json"),
                             "--instruction-json", str(skill / "references/vector_instruction_types.json"),
                             "--cases", str(source)], capture_output=True, text=True, check=True)
    expected = json.loads(replay.stdout)["results"][0]
    assert plan.provenance["revision"] == CANNBOT_REVISION
    assert len(plan.stages) == 1
    assert plan.stages[0].minimum_duration_ms == expected["theoretical_time_us"] / 1000
    assert plan.provenance["native_result"] == expected
    assert not plan.includes_dispatch
    assert plan.provenance["hardware_contract"]["hbm_bw_bytes_per_s"] == 1.6e12
    wrong_shape = replace(op(8), metadata=(("cannbot_case", case),))
    wrong_shape_result = CannbotVectorProvider(repo).estimate(wrong_shape, target, EstimateContext("fp32"))
    assert wrong_shape_result.unsupported_reason == "cannbot:case_operator_element_count_mismatch"


def test_native_refuses_unmatched_machine_and_revision(tmp_path):
    case = {"hardware_variant": "ascend950pr_9589"}
    operation = replace(op(), metadata=(("cannbot_case", case),))
    target = machine(arch="A5")
    result = CannbotVectorProvider(tmp_path).estimate(operation, target, EstimateContext())
    assert result.unsupported_reason == "cannbot:hardware_variant_not_explicitly_matched"
    with pytest.raises(ValueError, match="full_revision_required"):
        CannbotVectorProvider(tmp_path, revision="main")


def test_native_revision_guard_runs_before_any_external_engine(monkeypatch, tmp_path):
    from types import SimpleNamespace
    case = {"hardware_variant": "ascend950pr_9589"}
    operation = replace(op(), metadata=(("cannbot_case", case),))
    target = machine(arch="A5", metadata={"cannbot_hardware_variant": case["hardware_variant"]})
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="f" * 40 + "\n")

    monkeypatch.setattr("mindie_sim.operator_vector.subprocess.run", run)
    result = CannbotVectorProvider(tmp_path).estimate(operation, target, EstimateContext())
    assert result.unsupported_reason == "cannbot:revision_mismatch"
    assert len(calls) == 1
    assert calls[0][-2:] == ["rev-parse", "HEAD"]


def test_native_refuses_changes_to_pinned_engine_or_calibration(monkeypatch, tmp_path):
    from types import SimpleNamespace
    case = {"hardware_variant": "ascend950pr_9589"}
    operation = replace(op(), metadata=(("cannbot_case", case),))
    target = machine(arch="A5", metadata={"cannbot_hardware_variant": case["hardware_variant"]})
    source = tmp_path / "ops/ops-theoretical-performance-estimation/scripts/evaluate_a5_vector_roofline.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"changed_model_constant = 123\n")

    def run(command, **kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=CANNBOT_REVISION + "\n")
        assert command[3] == "show"
        return SimpleNamespace(stdout=b"original_model_constant = 7\n")

    monkeypatch.setattr("mindie_sim.operator_vector.subprocess.run", run)
    result = CannbotVectorProvider(tmp_path).estimate(operation, target, EstimateContext())
    assert result.unsupported_reason == "cannbot:modified_engine_or_calibration"


def test_native_refuses_cache_aware_mapping_without_fabricating_hbm_work(tmp_path):
    operation = replace(op(), metadata=(("cannbot_case", {
        "gm_bandwidth_model": "l2_capacity", "hardware_variant": "ascend950pr_9589",
    }),))
    result = CannbotVectorProvider(tmp_path).estimate(operation, machine(), EstimateContext())
    assert result.unsupported_reason == "cannbot:cache_aware_memory_requires_l2_resource_contract"


def test_heterogeneous_pp_stage_uses_its_own_vector_core_and_ub_limits():
    target = machine()
    second = replace(target.dies[0], rank=1, device_id=1, aiv_count=1, ub_bytes=48)
    target = replace(target, dies=(target.dies[0], second))
    stage0 = estimate(op(8), target=target)
    stage1 = estimate(replace(op(8), stage=1), target=target)
    assert stage0.provenance["vector_cores"] == 2
    assert stage1.provenance["vector_cores"] == 1
    assert stage0.buffer_certificate["tiles_per_heaviest_core"] == 1
    # 48B UB / (12B per element x two buffers) = two elements per tile.
    assert stage1.buffer_certificate["tile_elements"] == 2
    assert stage1.buffer_certificate["tiles_per_heaviest_core"] == 4
    assert stage1.buffer_certificate["ub_bytes_used"] == 48
    result = VectorInstructionProvider(profile()).estimate(replace(op(), stage=-1), target, EstimateContext())
    assert "operator.stage:invalid" in result.unsupported_reason


def test_native_profile_match_checks_physical_fields_not_only_variant_alias():
    from mindie_sim.operator_vector import _native_hardware_contract
    hardware = {"variants": {"test-native": {
        "core_counts": {"vector_core": 2}, "frequency": {"vector_freq_mhz": 1},
        "peak": {"gm_bandwidth_bytes_per_s": 1e12}}}}
    case = {"hardware_variant": "test-native"}
    target = machine(arch="A5", metadata={"vector_frequency_hz": 1e6}, bandwidth=1e12)
    binding = _native_hardware_contract(case, target, op(), hardware)
    assert binding["vector_cores"] == 2
    assert binding["vector_frequency_hz"] == 1e6
    wrong_cores = replace(target, dies=(replace(target.dies[0], aiv_count=1),))
    with pytest.raises(ValueError, match="hardware_mismatch:vector_core_count"):
        _native_hardware_contract(case, wrong_cores, op(), hardware)
    wrong_bandwidth = replace(target, dies=(replace(target.dies[0], hbm_bw_bytes_per_s=4e12),))
    with pytest.raises(ValueError, match="hardware_mismatch:hbm_bandwidth"):
        _native_hardware_contract(case, wrong_bandwidth, op(), hardware)
    with pytest.raises(ValueError, match="explicit_vector_frequency_hz_required"):
        _native_hardware_contract(case, replace(target, metadata=()), op(), hardware)
    wrong_mte = replace(target, dies=(replace(target.dies[0], mte_bw_bytes_per_s=1e10),))
    with pytest.raises(ValueError, match="mte_bandwidth_below_native_gm_rate"):
        _native_hardware_contract(case, wrong_mte, op(), hardware)
