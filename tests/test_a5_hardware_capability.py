"""A5 capability, directional paths and per-placement SKU contracts.

[Role]
  Establish the hardware-layer facts T05 introduces: representation, cast,
  native compute and runtime implementation are separate; data paths are
  directional with representation/owner/buffer/sync; every heterogeneous
  placement keeps its own SKU and resources.

[Invariants]
  * A declared peak is pricing supply, not a capability. An unsupported native
    format cannot be enabled by adding a peak or a calibration multiplier.
  * A2/A3 and A5 path sets differ in the sourced direction: A5 adds L0C->UB and
    UB->L1 and removes GM->L0A/L0B and L1->GM.
  * Storage/transfer of an 8-bit payload is not native FP8 compute and does not
    imply a cast rule.
  * Per-die resources and capacities are never broadcast across mixed SKUs.

[Timeouts & Locks]
  Tier 0 in-memory metadata tests; no device access, weights or locks.

[Side-Effect Policy]
  Pure construction and queries; no files written.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.custom_hardware import (
    build_custom_hardware_machine,
    parse_hardware,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import ComputeArchSpec, RuntimeCapability
from mindie_sim.schema.hardware import Hardware, Link


DT_9581 = "Ascend950DT_9581"
DT_9582 = "Ascend950DT_9582"
PR_9579 = "Ascend950PR_9579"
A2 = "Ascend910B1"
A3 = "Ascend910_9392"
A5_SKUS = (DT_9581, DT_9582, PR_9579)


def machine(sku: str, **overrides):
    return Hardware.from_mapping({"backend": "ascend", "sku": sku,
                                  "num_devices": 1, **overrides})


# ---------------------------------------------------------------------------
# Concrete SKU resources, owners and accounting scope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sku", A5_SKUS)
def test_a5_sku_instantiates_its_own_owners_capacity_and_scope(sku):
    hardware = machine(sku)
    die = hardware.dies[0]
    assert die.sku == sku
    assert die.arch_family == "A5"
    assert die.capability.preset == "npu_arch_3510"
    assert die.compute_units == 32
    assert die.owner_count("vector") == 64
    assert die.total_rate("vector") == pytest.approx(6758400000000.0)
    assert die.owner_rate("vector") == pytest.approx(105600000000.0)
    assert die.owner_rate("mte2_aiv") == pytest.approx(
        die.total_rate("mte2_aiv") / 64)
    assert die.pool_capacity("ub") == 253952
    assert die.total_buffer_bytes("ub") == 253952 * 64
    assert die.quantity_scope("memory") == "per_device"
    assert die.quantity_scope("buffers") == "per_owner"
    assert die.quantity_scope("compute_peaks") == "device_total"
    assert die.quantity_scope("resource_units") == "owners_per_device"
    assert hardware.dies_per_package == 1
    assert hardware.package_memory_bytes(0) == die.memory_bytes


def test_a5_dt_and_pr_share_arch_capability_but_not_capacity():
    dt = machine(DT_9581).dies[0]
    dt2 = machine(DT_9582).dies[0]
    pr = machine(PR_9579).dies[0]
    assert dict(dt.compute_peaks) == dict(pr.compute_peaks) == dict(dt2.compute_peaks)
    assert dt.memory_bytes == 154_618_822_656
    assert dt2.memory_bytes == 103_079_215_104
    assert pr.memory_bytes == 68_719_476_736
    assert len({dt.memory_bytes, dt2.memory_bytes, pr.memory_bytes}) == 3
    # The same architecture capability is reused; SKU capacity is not.
    assert dt.capability.representations == pr.capability.representations
    assert dt.capability.native_modes == pr.capability.native_modes
    assert dt.capability.paths == pr.capability.paths


def test_a2_a3_keep_native_modes_and_package_scope():
    a2 = machine(A2).dies[0]
    a3_raw = {"backend": "ascend", "sku": A3, "num_devices": 2}
    a3 = Hardware.from_mapping(a3_raw)
    assert a2.capability.preset == "npu_arch_2201"
    assert a2.capability.supports_native("matrix", "bf16", "fp32")
    assert a2.capability.supports_native("matrix", "int8", "int32")
    assert a2.peak("int4") > 0
    assert a3.dies_per_package == 2
    assert a3.package_memory_bytes(0) == 2 * a3.dies[0].memory_bytes
    assert a3.dies[0].capability.preset == "npu_arch_2201"


# ---------------------------------------------------------------------------
# Peak supply does not enable a native mode
# ---------------------------------------------------------------------------
def test_a2_fp8_storage_is_not_native_matrix_and_peak_cannot_enable_it():
    die = machine(A2).dies[0]
    capability = die.capability
    assert capability.representation_supports("fp8_e4m3", "storage")
    assert capability.representation_supports("fp8_e4m3", "transfer")
    assert not capability.representation_supports("fp8_e4m3", "operand")
    assert not capability.supports_native("matrix", "fp8_e4m3", "fp32")
    with pytest.raises(ModelingError, match="native_compute_mode_unsupported"):
        capability.native_mode("matrix", "fp8_e4m3", "fp32")
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        machine(A2, device={"compute_peaks": {"fp8_e4m3": 1e15}})
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e5m2"):
        machine(A2, device={"compute_peaks": {"fp8_e5m2": 2e15}})


def test_calibration_multiplier_cannot_enable_an_unsupported_native_dtype():
    die = machine(A2).dies[0]
    calibrated = machine(A2, device={
        "compute_peaks": {"bf16": dict(die.compute_peaks)["bf16"] * 1.25},
        "rates": {"hbm": dict(die.rates)["hbm"] * 1.10},
    }).dies[0]
    assert calibrated.peak("bf16") == pytest.approx(die.peak("bf16") * 1.25)
    assert dict(calibrated.rates)["hbm"] == pytest.approx(
        dict(die.rates)["hbm"] * 1.10)
    assert calibrated.capability.native_modes == die.capability.native_modes
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        machine(A2, device={
            "compute_peaks": {"fp8_e4m3": dict(die.compute_peaks)["bf16"] * 2},
            "rates": {"hbm": dict(die.rates)["hbm"] * 1.10},
        })


def test_device_peak_query_rejects_a_rate_added_after_capability_parse():
    die = machine(A2).dies[0]
    forged = replace(die, compute_peaks=tuple(sorted({
        **dict(die.compute_peaks), "fp8": 9e15}.items())))
    with pytest.raises(ModelingError, match="native_compute_mode_unsupported"):
        forged.peak("fp8")


def test_a5_declares_native_fp8_and_runtime_implementation_separately():
    capability = machine(DT_9581).dies[0].capability
    mode = capability.native_mode("matrix", "fp8_e4m3", "fp32")
    assert mode.unit == "cube"
    assert "fp8_e4m3" in mode.operands
    # One payload format may have multiple declared native replacements.
    assert {item.name for item in capability.native_modes_for("matrix", "fp8")} == {
        "fp8_e4m3_x_fp8_e4m3_acc_fp32", "fp8_e5m2_x_fp8_e5m2_acc_fp32"}
    assert "accumulator" in mode.provenance["basis"].lower()
    runtime = RuntimeCapability("declared-test-runtime", "1",
                                provides=("fp8_kernel",))
    with pytest.raises(ModelingError,
                       match="native_mode_implementation_undeclared"):
        capability.implemented_native_mode("matrix", "fp8_e4m3", "fp32", runtime)
    declared = ComputeArchSpec.from_mapping({
        "preset": "npu_arch_3510",
        "native_compute": {
            "fp8_e4m3_x_fp8_e4m3_acc_fp32": {
                "kind": "matrix", "operands": ["fp8_e4m3"],
                "accumulators": ["fp32"], "unit": "cube",
                "implementations": ["fp8_kernel"],
            },
        },
    })
    assert declared.implemented_native_mode(
        "matrix", "fp8_e4m3", "fp32", runtime).name == \
        "fp8_e4m3_x_fp8_e4m3_acc_fp32"
    other = RuntimeCapability("other-runtime", "1", provides=("other_kernel",))
    with pytest.raises(ModelingError, match="native_mode_not_implemented"):
        declared.implemented_native_mode("matrix", "fp8_e4m3", "fp32", other)


def test_runtime_implementation_selects_a_nonfirst_native_candidate():
    capability = ComputeArchSpec.from_mapping({
        "representations": {
            "bf16": {"bits": 16, "roles": [
                "storage", "transfer", "operand", "accumulator"]},
        },
        "native_compute": {
            "mode_a": {
                "kind": "matrix", "operands": ["bf16"],
                "accumulators": ["bf16"], "unit": "cube",
                "implementations": ["kernel_a"],
            },
            "mode_b": {
                "kind": "matrix", "operands": ["bf16"],
                "accumulators": ["bf16"], "unit": "cube",
                "implementations": ["kernel_b"],
            },
        },
    })
    assert capability.native_mode("matrix", "bf16", "bf16").name == "mode_a"
    runtime_b = RuntimeCapability("cann", "declared", provides=("kernel_b",))
    assert capability.implemented_native_mode(
        "matrix", "bf16", "bf16", runtime_b).name == "mode_b"
    with pytest.raises(ModelingError, match="native_mode_not_implemented"):
        capability.implemented_native_mode(
            "matrix", "bf16", "bf16",
            RuntimeCapability("other", "1", provides=("kernel_c",)))
    undeclared = ComputeArchSpec.from_mapping({
        "representations": {
            "bf16": {"bits": 16, "roles": [
                "storage", "transfer", "operand", "accumulator"]},
        },
        "native_compute": {
            "mode_a": {
                "kind": "matrix", "operands": ["bf16"],
                "accumulators": ["bf16"], "unit": "cube",
            },
        },
    })
    with pytest.raises(ModelingError,
                       match="native_mode_implementation_undeclared"):
        undeclared.implemented_native_mode(
            "matrix", "bf16", "bf16", runtime_b)


def test_cast_rule_does_not_become_a_native_mode():
    capability = ComputeArchSpec.from_mapping({
        "preset": "npu_arch_2201",
        "casts": {
            "fp8_e4m3->bf16": {
                "source": "fp8_e4m3", "target": "bf16",
                "implementation": "declared", "lossless": False,
            },
        },
    })
    assert capability.can_cast("fp8_e4m3", "bf16")
    assert not capability.supports_native("matrix", "fp8_e4m3", "fp32")
    assert capability.cast_rule("fp8_e4m3", "bf16").implementation == "declared"


def test_a5_int4_is_storage_only_and_peak_cannot_enable_native_cube():
    capability = ComputeArchSpec.from_mapping({"preset": "npu_arch_3510"})
    assert capability.representation_supports("int4", "storage")
    assert capability.representation_supports("int4", "transfer")
    assert not capability.representation_supports("int4", "operand")
    assert capability.native_modes_for("matrix", "int4", "int32") == ()
    assert not capability.supports_native("matrix", "int4", "int32")
    with pytest.raises(ModelingError, match="native_compute_mode_unsupported"):
        capability.native_mode("matrix", "int4", "int32")
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:int4"):
        machine(DT_9581, device={"compute_peaks": {"int4": 1e15}})
    # A legal explicit dequant/cast remains usable without becoming native.
    dequant = ComputeArchSpec.from_mapping({
        "preset": "npu_arch_3510",
        "casts": {"int4->bf16": {
            "source": "int4", "target": "bf16",
            "implementation": "declared_dequant", "lossless": False,
        }},
    })
    assert dequant.can_cast("int4", "bf16")
    assert not dequant.supports_native("matrix", "int4", "int32")
    # A2/A3 still declares its sourced arch2201 INT4 Cube mode.
    a2 = machine(A2).dies[0]
    assert a2.capability.supports_native("matrix", "int4", "int32")
    assert a2.peak("int4") > 0
    with pytest.raises(ModelingError, match="native_compute_operand_role_missing"):
        ComputeArchSpec.from_mapping({
            "representations": {"int4": {"bits": 4, "roles": ["storage"]}},
            "native_compute": {"int4_cube": {
                "kind": "matrix", "operands": ["int4"], "unit": "cube",
            }},
        })


def test_legacy_instruction_services_survive_capability_completion():
    base = machine(DT_9581).to_dict()
    base["devices"][0]["compute_arch"] = {
        "instruction_services": {
            "legacy_rate": {
                "unit": "calls", "owner_axis": "vector",
                "per_owner_rate": 1e9,
                "provenance": {"kind": "legacy_serializer_control"},
            },
            "legacy_pipe": {
                "unit": "calls", "owner_axis": "vector",
                "per_owner_rate": 2e9, "resource": "vector",
                "provenance": {"kind": "legacy_serializer_control"},
            },
        },
    }
    restored = Hardware.from_mapping(base)
    die = restored.dies[0]
    assert die.capability.preset == "npu_arch_3510"
    assert [item.name for item in die.capability.instruction_services] == [
        "legacy_pipe", "legacy_rate"]
    assert dict(die.rates)["legacy_rate"] == pytest.approx(1e9 * 64)
    assert dict(die.service_modes)["legacy_pipe"] == (
        "vector", pytest.approx(2e9 * 64))
    assert Hardware.from_mapping(restored.to_dict()) == restored


def test_backend_capability_adapter_is_neutral_and_unknown_safe():
    from mindie_sim.backends import registered_capability

    assert registered_capability(DT_9582, "A5") == {"preset": "npu_arch_3510"}
    assert registered_capability(A3, "A3") == {"preset": "npu_arch_2201"}
    assert registered_capability(DT_9582, "csa_attention") is None
    assert registered_capability("UNREGISTERED_A5", "A5") is None
    assert registered_capability("", "A5") is None


def test_pre_capability_registered_a5_case_resolves_arch_capability_only():
    materialized = machine(DT_9582).to_dict()
    device = materialized["devices"][0]
    device.pop("compute_arch")
    device.pop("arch_family")
    legacy = {**materialized, "name": "legacy_case", "architecture": "A5"}
    legacy.pop("hardware_key", None)
    restored = Hardware.from_mapping(legacy)
    die = restored.dies[0]
    assert die.sku == DT_9582
    assert die.capability.preset == "npu_arch_3510"
    # Legacy explicit rates and peaks stay exactly what the case declared.
    assert die.peak("fp8_e4m3") == dict(machine(DT_9582).dies[0].compute_peaks)[
        "fp8_e4m3"]
    assert die.capability.path("l0c", "ub") is not None


def test_legacy_pre_capability_a5_case_runs_through_production_entry():
    """Minimal production counterexample: parse/predict a pre-capability case."""
    from mindie_sim import predict
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.structure import parse_case

    case = compose_case(
        model="deepseek-v3.2-prune-3l", hardware="a5-server-dt",
        tp=8, layers=1, prompt_tokens=64, output_tokens=4,
        max_num_seqs=1, max_num_batched_tokens=64, max_model_len=128,
    )
    # The legacy profiler case corpus predates device.compute_arch. Remove the
    # new declaration and exercise the public parse/predict entry unchanged.
    case["machine"]["device"].pop("compute_arch", None)
    parsed = parse_case(case)
    assert parsed.machine.dies[0].capability.preset == "npu_arch_3510"
    prediction = predict(case, scope="body", phase="decode", tokens=1,
                         context=64)
    assert prediction["duration_ms"] > 0


def test_pre_capability_unknown_sku_still_requires_explicit_capability():
    materialized = machine(DT_9582).to_dict()
    materialized["devices"][0].pop("compute_arch")
    materialized["devices"][0]["sku"] = "UNREGISTERED_A5"
    materialized.pop("hardware_key", None)
    with pytest.raises(ModelingError, match="hardware_native_capability_required"):
        Hardware.from_mapping(materialized)


def test_pre_capability_resolution_cannot_hide_an_unsupported_peak():
    a2 = machine(A2).to_dict()
    a2["devices"][0].pop("compute_arch")
    a2["devices"][0]["compute_peaks"]["fp8_e4m3"] = 1e15
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        Hardware.from_mapping(a2)
    a5 = machine(DT_9582).to_dict()
    a5["devices"][0].pop("compute_arch")
    a5["devices"][0]["compute_peaks"]["fp4"] = 1e15
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp4"):
        Hardware.from_mapping(a5)


# ---------------------------------------------------------------------------
# Directional data paths, representation gating and local constraints
# ---------------------------------------------------------------------------
def test_directional_paths_distinguish_2201_and_3510():
    old = machine(A2).dies[0].capability
    new = machine(DT_9581).dies[0].capability
    assert old.path("gm", "l0a") is not None
    assert old.path("gm", "l0b") is not None
    assert old.path("l1", "gm") is not None
    assert old.path("l0c", "ub") is None
    assert old.path("ub", "l1") is None
    assert new.path("gm", "l0a") is None
    assert new.path("gm", "l0b") is None
    assert new.path("l1", "gm") is None
    assert new.path("l0c", "ub") is not None
    assert new.path("ub", "l1") is not None
    # Every path is directional.
    assert new.path("gm", "ub") is not None
    assert new.path("l0a", "l1") is None
    assert new.path("ub", "gm") is not None
    assert new.path("gm", "ub") is not None


def test_path_representation_roles_separate_payload_from_native_operand():
    old = machine(A2).dies[0].capability
    new = machine(DT_9581).dies[0].capability
    assert old.path("gm", "l1", "fp8_e4m3") is not None
    assert old.path("l1", "l0a", "fp8_e4m3") is None
    assert new.path("l1", "l0a", "fp8_e4m3") is not None
    assert old.path("l1", "l0a", "bf16") is not None
    assert new.path("l1", "l0a", "int4") is None
    with pytest.raises(ModelingError, match="data_path_unsupported"):
        new.require_path("gm", "l0a", "bf16")


def test_a5_declares_sourced_l1_to_ub_without_inventing_a_rate():
    a5 = machine(DT_9581).dies[0].capability
    path = a5.path("l1", "ub")
    assert path is not None
    assert path.buffer == "ub"
    assert path.executor == "aic"
    assert path.alignment_bytes == 32
    assert path.unit == "mte1_l1_ub"
    dut = machine(DT_9581).dies[0]
    assert dict(dut.service_modes)["mte1_l1_ub"] == (
        "mte1", pytest.approx(6.7584e12))
    assert dict(dut.rates)["mte1"] == pytest.approx(13.5168e12)
    assert a5.path("l1", "ub", "int4") is not None
    assert a5.path("l1", "ub", "fp8_e4m3") is not None
    assert a5.path("ub", "l1") is not None
    assert machine(A2).dies[0].capability.path("l1", "ub") is None


def test_path_keeps_owner_reach_buffer_alignment_and_sync():
    capability = machine(DT_9581).dies[0].capability
    l0c_ub = capability.require_path("l0c", "ub", "fp32")
    assert l0c_ub.unit == "fixpipe"
    assert l0c_ub.buffer == "ub"
    assert l0c_ub.sync == "event"
    assert "aic" in l0c_ub.reach and "aiv" in l0c_ub.reach
    assert l0c_ub.alignment_bytes == 64
    ub_l1 = capability.require_path("ub", "l1", "bf16")
    assert ub_l1.unit == "mte3"
    assert ub_l1.buffer == "l1"
    assert ub_l1.sync == "event"
    assert ub_l1.alignment_bytes == 32
    l1_l0a = capability.require_path("l1", "l0a", "bf16")
    assert l1_l0a.unit == "mte1"
    assert l1_l0a.executor == "aic"
    assert l1_l0a.buffer == "l0a"
    assert l1_l0a.alignment_bytes == 512


def test_per_device_capability_override_merges_with_its_sku_preset():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": A2, "num_devices": 2,
        "devices": [
            {},
            {"compute_arch": {"casts": {
                "fp8_e4m3->bf16": {
                    "source": "fp8_e4m3", "target": "bf16",
                    "implementation": "declared", "lossless": False,
                },
            }}},
        ],
    })
    first, second = hardware.dies
    assert first.capability.preset == "npu_arch_2201"
    assert second.capability.preset == "npu_arch_2201"
    assert not first.capability.can_cast("fp8_e4m3", "bf16")
    assert second.capability.can_cast("fp8_e4m3", "bf16")
    # The partial override added a cast without discarding native modes.
    assert second.capability.supports_native("matrix", "bf16", "fp32")
    assert second.capability.path("gm", "l1") is not None


# ---------------------------------------------------------------------------
# Heterogeneous placement: each rank's own SKU and endpoint resources
# ---------------------------------------------------------------------------
def test_mixed_dt_pr_does_not_leak_dt_register_or_pipe_services():
    base_pr = machine(PR_9579).dies[0]
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "device": {"launch_us": 7.0},
        "devices": [{"sku": DT_9581}, {"sku": PR_9579}],
    })
    dt, pr = hardware.dies
    # PR declares the same register/pipe families with its own HBM supply:
    # 1.6 TB/s / 64 AIV / 128 B = 195312500 cachelines/s/AIV, not DT's rate.
    for role in ("pipe_v_lanes", "pipe_v_issue", "pipe_s_issue",
                 "ub_read_bytes", "ub_write_bytes", "gm_hbm_cacheline_128b"):
        assert role in dict(dt.rates)
        assert role in dict(pr.rates)
    assert "get_sff_value_u64_i64" in dict(pr.service_modes)
    service = next(item for item in pr.capability.instruction_services
                   if item.name == "gm_hbm_cacheline_128b")
    assert service.per_owner_rate == pytest.approx(195312500.0)
    assert dict(pr.rates)["gm_hbm_cacheline_128b"] == pytest.approx(12.5e9)
    assert dict(pr.rates) == dict(base_pr.rates)
    assert dict(pr.service_modes) == dict(base_pr.service_modes)
    assert pr.capability.instruction_services
    assert pr.launch_us == pytest.approx(7.0)


def test_mixed_dt_a2_does_not_inherit_a5_fp8_peaks():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "devices": [{"sku": DT_9581}, {"sku": A2}],
    })
    dt, a2 = hardware.dies
    assert "fp8_e4m3" in dict(dt.compute_peaks)
    assert "fp8_e4m3" not in dict(a2.compute_peaks)
    assert a2.capability.preset == "npu_arch_2201"
    assert a2.peak("int8") > 0
    # A2 keeps its own profile resources, not DT's register pipes.
    assert "pipe_v_lanes" not in dict(a2.rates)
    assert "gm_hbm_cacheline_128b" not in dict(a2.rates)


def _mixed_sort32(base_sku, other_sku, *, clock=1.6e9,
                  presets=("ascend_sort32_spec",), service=None):
    arch = {"presets": list(presets)}
    if clock is not None:
        arch["clock_hz"] = clock
    if service is not None:
        arch["instruction_services"] = service
    return Hardware.from_mapping({
        "backend": "ascend", "sku": base_sku, "num_devices": 2,
        "device": {"compute_arch": arch},
        "devices": [{}, {"sku": other_sku}],
    })


def _alone_with_sort32(sku, *, clock=1.6e9,
                       presets=("ascend_sort32_spec",), service=None):
    arch = {"presets": list(presets)}
    if clock is not None:
        arch["clock_hz"] = clock
    if service is not None:
        arch["instruction_services"] = service
    return Hardware.from_mapping({
        "backend": "ascend", "sku": sku, "num_devices": 1,
        "device": {"compute_arch": arch},
    }).dies[0]


def _sorted_service(device):
    return dict(device.service_modes).get("vbitsort32")


def _execute_sort32(device, vector_work=0.0):
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.instructions import (
        CONTROL_INSTRUCTIONS, InstructionCount, InstructionInventory,
        PrimitiveInventory,
    )
    from mindie_sim.prediction.mechanism import Geometry, Work
    from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage

    rate = dict(device.rates)["vector"] / 1000
    units = device.units("vector")
    engine = Engine([Resource("vector", rate, units=units)])
    context = RecipeContext(
        None, {}, {"vector": "vector"}, {"vector": rate}, device.compute_units,
        resource_units={"vector": units},
        service_modes=dict(device.service_modes))
    counts = InstructionInventory({"VBitSort32": InstructionCount(1600, 51200)})
    work = counts.capacity_work(CONTROL_INSTRUCTIONS)
    priced, unpriced = PrimitiveInventory(work.resources).partition(
        Geometry.from_devices((device,)).resource_units)
    assert not unpriced
    demand = context.bind(priced)
    stages = [Stage("sort", demand,
                    placement=context.owned(0, units, ("vbitsort32",)))]
    if vector_work:
        stages.append(Stage(
            "vector", context.bind(Work(resources={"vector": vector_work})),
            placement=context.owned(0, units, ("vector",))))
    ready = engine.submit("input", latency=7)
    done = KernelProgram(tuple(stages)).start(engine, "sort", ready)
    engine.advance(until=(done,))
    assert engine.serviced["vector"] == pytest.approx(
        demand["vector"] + vector_work)
    return done.time - ready.time


def _review_nested_aiv_hardware(base=DT_9581, other=PR_9579):
    return Hardware.from_mapping({
        "backend": "ascend", "sku": base, "num_devices": 2,
        "device": {"compute_arch": {
            "presets": ["ascend_sort32_spec"], "clock_hz": 1.6e9}},
        "devices": [
            {},
            {"sku": other, "device": {"aiv_count": 32}},
        ],
    })


def test_nested_aiv_override_is_not_scaled_twice():
    hardware = _review_nested_aiv_hardware()
    dt, pr = hardware.dies
    # PR bare is 64 AIV; the nested explicit 32 is final, not 32 * 32/64.
    assert dt.units("vector") == 64
    assert pr.units("vector") == 32
    assert dict(pr.rates)["vector"] == pytest.approx(3.3792e12)
    assert pr.owner_rate("vector") == pytest.approx(1.056e11)
    assert _sorted_service(pr) == ("vector", pytest.approx(3.2e9))
    assert dict(pr.work_units)["vbitsort32"] == "blocks32"
    assert dict(dt.rates)["vector"] == pytest.approx(6.7584e12)
    assert _sorted_service(dt) == ("vector", pytest.approx(6.4e9))
    assert dict(pr.service_modes)["ub_copy"][1] == pytest.approx(
        27.0336e12 * 0.5)
    # Inherited AIV-owned facts scale once; explicit vector owner stays 32.
    assert pr.units("mte2_aiv") == 32
    assert pr.pool_units("ub") == 32
    assert Hardware.from_mapping(hardware.to_dict()) == hardware


def test_nested_aiv_override_control_reverse_sku_and_flattening():
    reverse = _review_nested_aiv_hardware(PR_9579, DT_9581)
    pr, dt = reverse.dies
    assert pr.units("vector") == 64
    assert dt.units("vector") == 32
    assert dt.owner_rate("vector") == pytest.approx(1.056e11)
    nested = _review_nested_aiv_hardware().dies[1]
    flat = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "device": {"compute_arch": {
            "presets": ["ascend_sort32_spec"], "clock_hz": 1.6e9}},
        "devices": [
            {},
            {"sku": PR_9579, "aiv_count": 32},
        ],
    }).dies[1]
    assert dict(nested.rates) == dict(flat.rates)
    assert dict(nested.resource_units) == dict(flat.resource_units)
    assert dict(nested.buffer_units) == dict(flat.buffer_units)
    assert dict(nested.service_modes) == dict(flat.service_modes)
    assert dict(nested.work_units) == dict(flat.work_units)


def test_local_explicit_peak_rate_pool_and_service_are_not_scaled():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "devices": [
            {},
            {"sku": PR_9579, "device": {
                "aiv_count": 32,
                "compute_peaks": {"bf16": 1e12},
                "rates": {"vector": 1e12},
                "buffer_units": {"ub": 17},
                "buffers": {"ub": 1000},
                "service_modes": {
                    "explicit_mode": {"resource": "vector", "rate": 123.0}},
            }},
        ],
    })
    pr = hardware.dies[1]
    assert pr.units("vector") == 32
    assert pr.peak("bf16") == pytest.approx(1e12)
    assert dict(pr.rates)["vector"] == pytest.approx(1e12)
    assert pr.pool_units("ub") == 17
    assert pr.pool_capacity("ub") == 1000
    assert dict(pr.service_modes)["explicit_mode"] == ("vector", 123.0)
    # The explicit mode keeps its declared rate; explicit vector rate is not
    # re-scaled by the owner count.
    assert dict(pr.service_modes)["explicit_mode"] == ("vector", 123.0)
    assert Hardware.from_mapping(hardware.to_dict()) == hardware


def test_heterogeneous_placement_expands_explicit_instruction_presets():
    for base, other in ((DT_9581, PR_9579), (PR_9579, DT_9581)):
        hardware = _mixed_sort32(base, other)
        assert hardware.sku == "mixed"
        for die, sku in zip(hardware.dies, (base, other)):
            assert die.sku == sku
            alone = _alone_with_sort32(sku)
            assert _sorted_service(die) == _sorted_service(alone)
            assert _sorted_service(die) == ("vector", pytest.approx(6.4e9))
            assert dict(die.rates) == dict(alone.rates)
            assert dict(die.resource_units) == dict(alone.resource_units)
            assert dict(die.work_units)["vbitsort32"] == "blocks32"
            assert [item.name for item in die.capability.instruction_services] == [
                item.name for item in alone.capability.instruction_services]
        assert Hardware.from_mapping(hardware.to_dict()) == hardware


def test_heterogeneous_preset_clock_disable_explicit_rate_and_roundtrip():
    clocked = _mixed_sort32(DT_9581, PR_9579, clock=3.2e9)
    for die in clocked.dies:
        assert _sorted_service(die) == ("vector", pytest.approx(12.8e9))
    disabled = _mixed_sort32(DT_9581, PR_9579, presets=())
    for die in disabled.dies:
        assert _sorted_service(die) is None
        assert "vbitsort32" not in dict(die.work_units)
    explicit = _mixed_sort32(
        DT_9581, PR_9579,
        service={"vbitsort32": {"per_owner_rate": 2e8}})
    for die in explicit.dies:
        assert _sorted_service(die) == ("vector", pytest.approx(12.8e9))
    restored = Hardware.from_mapping(explicit.to_dict())
    assert restored == explicit
    assert [_sorted_service(die) for die in restored.dies] == [
        _sorted_service(die) for die in explicit.dies]


def test_heterogeneous_local_clock_and_preset_disable_are_per_placement():
    clocked = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "device": {"compute_arch": {
            "presets": ["ascend_sort32_spec"], "clock_hz": 1.6e9}},
        "devices": [
            {},
            {"sku": PR_9579, "device": {"compute_arch": {"clock_hz": 3.2e9}}},
        ],
    })
    dt, pr = clocked.dies
    assert _sorted_service(dt) == ("vector", pytest.approx(6.4e9))
    assert _sorted_service(pr) == ("vector", pytest.approx(12.8e9))
    disabled = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "device": {"compute_arch": {
            "presets": ["ascend_sort32_spec"], "clock_hz": 1.6e9}},
        "devices": [
            {},
            {"sku": PR_9579,
             "device": {"compute_arch": {"presets": []}}},
        ],
    })
    dt, pr = disabled.dies
    assert _sorted_service(dt) == ("vector", pytest.approx(6.4e9))
    assert _sorted_service(pr) is None
    assert "vbitsort32" not in dict(pr.work_units)


def test_nested_aiv32_sort32_work_uses_correct_owner_capacity():
    hardware = _review_nested_aiv_hardware()
    pr = hardware.dies[1]
    expected = 1600 * 16 / 1.6e9 * 1000
    assert pr.units("vector") == 32
    assert _sorted_service(pr) == ("vector", pytest.approx(3.2e9))
    assert _execute_sort32(pr) == pytest.approx(expected)
    equal_vector = dict(pr.rates)["vector"] / pr.units("vector") * expected / 1000
    assert _execute_sort32(pr, vector_work=equal_vector) == pytest.approx(
        2 * expected)


def _neutral_a2(peaks=None):
    return {
        "sku": "Ascend910B1", "arch_family": "A2",
        "device": {
            "memory_bytes": 68_719_476_736, "compute_units": 24,
            "compute_peaks": peaks or {"bf16": 363_724_800_000_000.0},
            "rates": {"hbm": 1_800_000_000_000.0,
                      "vector": 2_841_600_000_000.0},
        },
    }


def test_neutral_arch_identity_and_fp8_admission_cover_public_entries():
    from mindie_sim.custom_hardware import parse_hardware as custom_parse
    from mindie_sim.schema.hardware import parse_hardware

    with_fp8 = _neutral_a2({
        "bf16": 363_724_800_000_000.0,
        "fp8_e4m3": 1_000_000_000_000_000.0,
    })
    for entry in (Hardware.from_mapping, parse_hardware, custom_parse):
        with pytest.raises(ModelingError,
                           match="compute_peak_without_native_compute_mode:fp8_e4m3"):
            entry(with_fp8)
    for spelling in ("arch_family", "architecture"):
        raw = _neutral_a2()
        if spelling == "architecture":
            raw = {**raw, "architecture": raw.pop("arch_family")}
        parsed_machine = Hardware.from_mapping(raw)
        die = parsed_machine.dies[0]
        assert parsed_machine.arch_family == "A2"
        assert die.arch_family == "A2"
        assert die.capability.preset == "npu_arch_2201"
        assert die.peak("bf16") == pytest.approx(363_724_800_000_000.0)
        assert Hardware.from_mapping(parsed_machine.to_dict()) == parsed_machine
    a3 = _neutral_a2({"bf16": 1.0, "fp8_e4m3": 2.0})
    a3["sku"] = "Ascend910_9392"
    a3["arch_family"] = "A3"
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        Hardware.from_mapping(a3)
    a5 = machine(DT_9581).to_dict()["devices"][0]
    a5.pop("compute_arch")
    a5.pop("arch_family")
    a5["compute_peaks"]["fp8_e4m3"] = 8.650752e14
    positive = Hardware.from_mapping({
        "sku": DT_9581, "arch_family": "A5", "device": a5})
    assert positive.dies[0].capability.preset == "npu_arch_3510"
    assert positive.dies[0].peak("fp8_e4m3") == pytest.approx(8.650752e14)


def _neutral_a2_conflict():
    raw = _neutral_a2()
    raw["architecture"] = "A2"
    raw["arch_family"] = "generic"
    return raw


def test_same_layer_architecture_precedes_arch_family_across_entries():
    from mindie_sim.custom_hardware import parse_hardware as custom_parse
    from mindie_sim.schema.hardware import parse_hardware

    with_fp8 = _neutral_a2_conflict()
    with_fp8["device"]["compute_peaks"]["fp8_e4m3"] = 1e15
    for entry in (Hardware.from_mapping, parse_hardware, custom_parse):
        with pytest.raises(ModelingError,
                           match="compute_peak_without_native_compute_mode:fp8_e4m3"):
            entry(with_fp8)
    positive = _neutral_a2_conflict()
    for entry in (Hardware.from_mapping, parse_hardware, custom_parse):
        machine = entry(positive)
        assert machine.arch_family == "A2"
        assert machine.dies[0].arch_family == "A2"
        assert machine.dies[0].capability.preset == "npu_arch_2201"
        assert machine.dies[0].peak("bf16") == pytest.approx(3.637248e14)


def test_cross_layer_arch_alias_record_overrides_public_device():
    raw = _neutral_a2()
    raw["arch_family"] = "generic"
    raw["device"]["arch_family"] = "generic"
    raw["devices"] = [{"architecture": "A2"}]
    raw["device"]["compute_peaks"]["fp8_e4m3"] = 1e15
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        Hardware.from_mapping(raw)
    raw["device"]["compute_peaks"].pop("fp8_e4m3")
    machine = Hardware.from_mapping(raw)
    assert machine.arch_family == "A2"
    assert machine.dies[0].capability.preset == "npu_arch_2201"


def test_four_layer_arch_normalization_and_same_layer_priority():
    # machine public architecture=A2 / arch_family=generic -> A2.
    raw = _neutral_a2_conflict()
    raw["device"].pop("arch_family", None)
    raw["devices"] = [{}]
    raw["device"]["compute_peaks"]["fp8_e4m3"] = 1e15
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        Hardware.from_mapping(raw)
    # Later layers can replace the canonical identity only when they declare
    # the canonical/alias fields at their own level.
    layered = _neutral_a2()
    layered["arch_family"] = "generic"
    layered["device"]["arch_family"] = "generic"
    layered["devices"] = [{
        "arch_family": "generic",
        "device": {"architecture": "A2", "arch_family": "generic"},
    }]
    layered["device"]["compute_peaks"]["fp8_e4m3"] = 1e15
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        Hardware.from_mapping(layered)
    layered["device"]["compute_peaks"].pop("fp8_e4m3")
    assert Hardware.from_mapping(layered).dies[0].capability.preset == "npu_arch_2201"


def test_same_layer_conflict_in_legacy_parse_case_positive_and_negative():
    from mindie_sim import predict
    from mindie_sim.prediction.structure import parse_case
    from test_mechanism_contract import hardware_case

    negative = hardware_case()
    negative["machine"] = _neutral_a2_conflict()
    negative["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 1e15
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        parse_case(negative)
    positive = hardware_case()
    positive["machine"] = _neutral_a2_conflict()
    parsed = parse_case(positive)
    assert parsed.machine.arch_family == "A2"
    assert parsed.machine.dies[0].capability.preset == "npu_arch_2201"
    assert predict(positive, scope="body", phase="decode", tokens=1,
                   context=16)["duration_ms"] > 0


def test_legacy_neutral_parse_case_rejects_a2_fp8_peak():
    from mindie_sim.prediction.structure import parse_case
    from test_mechanism_contract import hardware_case

    raw = hardware_case()
    raw["machine"] = _neutral_a2({
        "bf16": 363_724_800_000_000.0,
        "fp8_e4m3": 1_000_000_000_000_000.0,
    })
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        parse_case(raw)


def test_legacy_neutral_parse_case_and_predict_keep_a2_identity():
    from mindie_sim import predict
    from mindie_sim.prediction.structure import parse_case
    from test_mechanism_contract import hardware_case

    raw = hardware_case()
    raw["machine"] = _neutral_a2()
    parsed = parse_case(raw)
    assert parsed.machine.arch_family == "A2"
    assert parsed.machine.dies[0].capability.preset == "npu_arch_2201"
    assert predict(raw, scope="body", phase="decode", tokens=1,
                   context=16)["duration_ms"] > 0


@pytest.mark.parametrize("alias", ["num_cores", "cores", "compute_units",
                                   "aic_count"])
def test_neutral_core_aliases_override_public_24_in_nested_device(alias):
    raw = _neutral_a2()
    raw["architecture"] = raw.pop("arch_family")
    raw["devices"] = [{"device": {alias: 12}}]
    die = Hardware.from_mapping(raw).dies[0]
    assert die.compute_units == 12
    assert die.units("cube") == 12
    assert die.peak("bf16") == pytest.approx(363_724_800_000_000.0)
    assert dict(die.rates)["vector"] == pytest.approx(2_841_600_000_000.0)


def test_neutral_core_alias_same_layer_priority_four_layer_and_equivalence():
    raw = _neutral_a2()
    raw["architecture"] = raw.pop("arch_family")
    raw["devices"] = [{"device": {
        "num_cores": 20, "cores": 18, "compute_units": 16, "aic_count": 14,
    }}]
    assert Hardware.from_mapping(raw).dies[0].compute_units == 16
    raw["devices"] = [{"device": {"num_cores": 8, "cores": 6}}]
    assert Hardware.from_mapping(raw).dies[0].compute_units == 8
    raw["devices"] = [{"device": {"aic_count": 12, "num_cores": 8}}]
    assert Hardware.from_mapping(raw).dies[0].compute_units == 12
    # Machine public -> machine device -> record -> record.device, normalized
    # within each layer; the later placement level wins.
    layered = _neutral_a2()
    layered["architecture"] = layered.pop("arch_family")
    layered["num_cores"] = 20
    layered["device"]["cores"] = 18
    layered["devices"] = [{
        "num_cores": 16,
        "device": {"cores": 14},
    }]
    assert Hardware.from_mapping(layered).dies[0].compute_units == 14


def test_dt_pr_local_aic16_aiv32_sort32_work_and_vector_competition():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "device": {"compute_arch": {
            "presets": ["ascend_sort32_spec"], "clock_hz": 1.6e9}},
        "devices": [
            {},
            {"sku": PR_9579, "device": {
                "compute_units": 16, "aiv_count": 32,
                "compute_peaks": {"bf16": 1e12},
                "rates": {"vector": 1e12},
                "buffer_units": {"ub": 17},
                "service_modes": {
                    "explicit_mode": {"resource": "vector", "rate": 123.0}},
            }},
        ],
    })
    pr = hardware.dies[1]
    expected = 1600 * 16 / 1.6e9 * 1000
    assert pr.compute_units == 16 and pr.units("vector") == 32
    assert _sorted_service(pr) == ("vector", pytest.approx(3.2e9))
    assert _execute_sort32(pr) == pytest.approx(expected)
    equal_vector = dict(pr.rates)["vector"] / pr.units("vector") * expected / 1000
    assert _execute_sort32(pr, vector_work=equal_vector) == pytest.approx(
        2 * expected)


def test_dt_pr_local_aic_aiv_peak_rate_pool_service_and_mte1_basis():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "device": {"compute_arch": {
            "presets": ["ascend_sort32_spec"], "clock_hz": 1.6e9}},
        "devices": [
            {},
            {"sku": PR_9579, "device": {
                "compute_units": 16, "aiv_count": 32,
                "compute_peaks": {"bf16": 1e12},
                "rates": {"vector": 1e12},
                "buffer_units": {"ub": 17},
                "service_modes": {
                    "explicit_mode": {"resource": "vector", "rate": 123.0}},
            }},
        ],
    })
    dt, pr = hardware.dies
    assert dt.compute_units == 32 and dt.units("vector") == 64
    assert pr.compute_units == 16 and pr.units("vector") == 32
    assert pr.peak("bf16") == pytest.approx(1e12)
    assert dict(pr.rates)["vector"] == pytest.approx(1e12)
    assert pr.pool_units("ub") == 17
    assert dict(pr.service_modes)["explicit_mode"] == ("vector", 123.0)
    assert pr.units("mte1") == 16
    assert dict(pr.rates)["mte1"] == pytest.approx(6.7584e12)
    assert dict(pr.service_modes)["mte1_l1_ub"] == (
        "mte1", pytest.approx(3.3792e12))
    assert _sorted_service(pr) == ("vector", pytest.approx(3.2e9))
    assert dict(dt.rates)["mte1"] == pytest.approx(13.5168e12)
    assert dict(dt.service_modes)["mte1_l1_ub"] == (
        "mte1", pytest.approx(6.7584e12))
    assert Hardware.from_mapping(hardware.to_dict()) == hardware


def test_heterogeneous_sort32_work_is_priced_and_competes_with_vector():
    hardware = _mixed_sort32(DT_9581, PR_9579)
    pr = hardware.dies[1]
    expected = 1600 * 16 / 1.6e9 * 1000
    assert _execute_sort32(pr) == pytest.approx(expected)
    equal_vector = dict(pr.rates)["vector"] / pr.units("vector") * expected / 1000
    assert _execute_sort32(pr, vector_work=equal_vector) == pytest.approx(
        2 * expected)


def test_mixed_a5_placements_keep_their_own_sku_and_capacity():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "device": {"launch_us": 7.0},
        "devices": [{"sku": DT_9581}, {"sku": PR_9579}],
    })
    assert hardware.arch_family == "A5"
    assert hardware.sku == "mixed"
    assert not hardware.homogeneous
    dt, pr = hardware.dies
    assert (dt.sku, pr.sku) == (DT_9581, PR_9579)
    assert (dt.memory_bytes, pr.memory_bytes) == (
        154_618_822_656, 68_719_476_736)
    assert hardware.placement_sku(0) == DT_9581
    assert hardware.placement_sku(1) == PR_9579
    # The machine-level explicit override reaches both placements; the local
    # SKU defaults still differ and are not broadcast.
    assert dt.launch_us == pr.launch_us == pytest.approx(7.0)
    assert hardware.per_placement_peaks("bf16") == {
        0: dt.peak("bf16"), 1: pr.peak("bf16")}
    with pytest.raises(ModelingError,
                       match="hardware_resources_are_per_placement"):
        hardware.peak_flops_for("bf16")
    assert Hardware.from_mapping(hardware.to_dict()) == hardware


def test_launch_timing_is_part_of_the_per_die_homogeneous_signature():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": PR_9579, "num_devices": 2,
        "devices": [{}, {"launch_us": 1.0}],
    })
    assert [die.launch_us for die in hardware.dies] == [0.0, 1.0]
    assert not hardware.homogeneous


def test_hardware_context_exposes_one_placement_without_model_inputs():
    mixed = Hardware.from_mapping({
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "devices": [{"sku": DT_9581}, {"sku": PR_9579}],
    })
    dt = mixed.hardware_context(0)
    pr = mixed.hardware_context(1)
    assert dt["schema"] == "mindie-sim.hardware-context/1"
    assert dt["device_sku"] == DT_9581
    assert pr["device_sku"] == PR_9579
    assert dt["homogeneous"] is False
    assert dt["placement"]["rank"] == 0
    assert dt["capability"]["preset"] == "npu_arch_3510"
    assert dt["memory_bytes"] != pr["memory_bytes"]
    assert "model" not in dt and "workload" not in dt and "duration_ms" not in dt
    assert "gm_hbm_cacheline_128b" in dt["rates"]
    # PR declares the same line service; the aggregate rate is HBM/128B.
    assert pr["rates"]["gm_hbm_cacheline_128b"] == pytest.approx(12.5e9)
    with pytest.raises(ModelingError, match="hardware_placement_unknown"):
        mixed.hardware_context(9)


def test_transfer_channels_and_explicit_link_peak_roundtrip():
    raw = {
        "backend": "ascend", "sku": DT_9581, "num_devices": 2,
        "links": [{
            "name": "fabric", "capacity_bytes_per_ms": 25e6,
            "peak_capacity_bytes_per_ms": 25e6, "members": [0, 1],
        }],
        "transfer_channels": {
            "d2d_inter_node": {"effective_uni_bw_gbps": 50.0}},
    }
    hardware = Hardware.from_mapping(raw)
    link = hardware.links[0]
    assert link.peak_capacity_bytes_per_ms == 25e6
    restored = Hardware.from_mapping(hardware.to_dict())
    assert restored == hardware
    assert restored.transfer_channels == hardware.transfer_channels
    assert restored.links[0].peak_capacity_bytes_per_ms == 25e6
    assert Link.from_mapping(link.to_dict()) == link


def test_endpoint_rates_bind_one_placement_without_broadcasting():
    link = Link.from_mapping({
        "name": "shared_forward", "capacity_bytes_per_ms": 60.0,
        "members": [0, 1], "endpoint_rates": {1: 30.0},
    })
    assert link.endpoint_capacity_bytes_per_ms(0) is None
    assert link.capacity_at(0) == 60.0
    assert link.endpoint_capacity_bytes_per_ms(1) == 30.0
    assert link.capacity_at(1) == 30.0
    assert Link.from_mapping(link.to_dict()) == link
    with pytest.raises(ModelingError, match="link_endpoint_member_unknown"):
        Link.from_mapping({
            "name": "bad", "capacity_bytes_per_ms": 60.0,
            "members": [0, 1], "endpoint_rates": {2: 30.0},
        })


# ---------------------------------------------------------------------------
# Custom neutral SKU: capability is explicit, no invented path set
# ---------------------------------------------------------------------------
def test_custom_template_declares_its_synthetic_native_modes_only():
    spec = {"sku": "CUSTOM_A5_LIKE", "hbm_gb": 96.0,
            "hbm_bw_gbps": 4000.0, "cube_fp16_tflops": 600.0}
    hardware = parse_hardware(build_custom_hardware_machine(spec, num_devices=1))
    die = hardware.dies[0]
    assert die.capability.has_native_modes()
    assert die.peak("fp8") > 0
    assert die.capability.path("gm", "l1") is None
    assert die.capability.runtimes == ()
    assert Hardware.from_mapping(hardware.to_dict()) == hardware


def test_custom_template_rejects_explicit_capability_that_lacks_a_peak_mode():
    base = {"sku": "CUSTOM_EXPLICIT", "hbm_gb": 96.0,
            "hbm_bw_gbps": 4000.0, "cube_fp16_tflops": 600.0}
    with pytest.raises(ModelingError,
                       match="compute_peak_without_native_compute_mode"):
        parse_hardware(build_custom_hardware_machine(
            {**base, "capability": {"preset": "npu_arch_2201"}},
            num_devices=1))
    declared = parse_hardware(build_custom_hardware_machine(
        {**base, "capability": {"preset": "npu_arch_3510"}},
        num_devices=1))
    capability = declared.dies[0].capability
    assert capability.native_mode("matrix", "fp8_e4m3", "fp32")
    assert capability.path("l0c", "ub") is not None
    assert capability.path("gm", "l0a") is None


def test_capability_declaration_roundtrips_with_explicit_overrides():
    declared = ComputeArchSpec.from_mapping({
        "preset": "npu_arch_3510",
        "casts": {
            "fp8_e4m3->bf16": {
                "source": "fp8_e4m3", "target": "bf16",
                "implementation": "test_only", "lossless": False,
            },
        },
    })
    restored = ComputeArchSpec.from_mapping(declared.to_dict())
    assert restored == declared
    assert restored.preset == "npu_arch_3510"
    assert restored.can_cast("fp8_e4m3", "bf16")
    assert restored.path("l0c", "ub") is not None
