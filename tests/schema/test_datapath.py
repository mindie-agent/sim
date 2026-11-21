"""Datapath capability, directional paths, per-placement overrides and split read axes.

[Role] Verify ``ascend.json`` datapath declarations, ``DatapathCapabilitySpec``
  derivation, homogeneity checks in ``Geometry.from_devices``, propagation into
  ``ExecutionBindings`` / ``RecipeContext``, capability vs peak/cast/runtime
  separation, heterogeneous per-placement SKU/preset/override isolation, MTE1
  and vector direction-mode scaling, and split AIC/AIV matrix read axes.
[Invariants] Every ``npu_arch_2201`` SKU forbids ``l0c_to_ub`` / ``ub_to_l1``
  and native FP8; every ``npu_arch_3510`` SKU exposes ``l0c_to_ub`` /
  ``ub_to_l1`` / ``l1_to_ub`` and native FP8 while keeping INT4 storage-only.
  A peak or cast never enables a native mode; per-placement overrides never
  broadcast or double-scale; split matrix read axes bypass legacy ``mte2``.
[Timeouts & Locks] Pure Tier-0 in-memory tests (<5s); no I/O writes or locks.
[Side-Effect Policy] Read-only over repository profile JSON.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

import mindie_sim.api as sdk_api
from mindie_sim import predict
from mindie_sim.backends import registered_capability
from mindie_sim.backends.ascend import profiles
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.custom_hardware import build_custom_hardware_machine, parse_hardware as custom_parse
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.evaluation import ExecutionBindings
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.instructions import (
    CONTROL_INSTRUCTIONS, InstructionCount, InstructionInventory, PrimitiveInventory,
)
from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul, mapping_bound
from mindie_sim.prediction.operators.matrix.matrix_phase import PhaseMatmul
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, RecipeContext, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import ComputeArchSpec, RuntimeCapability
from mindie_sim.schema.datapath import datapath_for_devices, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.hardware import Hardware, Link, parse_hardware as schema_parse

from _support.axes import ALL_SKUS, REPRESENTATIVE_SKUS, SKUS
from _support.hardware import (
    A5 as REGBASE_SKUS, MODEL, WORKLOAD, OneMode, ReadMoves, declared_supply,
    hardware_case, isa_of, machine, op, one_owner_ms, run, sku_case,
)

pytestmark = pytest.mark.tier0

MEM_SKU, DUAL_DIE_SKU, REG_SKU = REPRESENTATIVE_SKUS["a2"], REPRESENTATIVE_SKUS["a3"], REPRESENTATIVE_SKUS["a5"]
REG_PR_SKU, MEM_WIDE_SKU, AMOUNT = "Ascend950PR_9579", "Ascend910B2", 1_000_000_000
SPLIT_RECIPES = (TiledMatmul(1, 1, 1), HierarchicalMatmul(1, 1, 1, 1, 1),
                 MappedMatmul(((1, 1, 1),), 1, 1), PhaseMatmul(((1, 1, 1),), 1, 1),
                 SteadyMatmul(((1, 1, 1),), 1, 1))


def _compose_tiny(sku: str, **options):
    return sdk_api.compose_case("qwen3-32b", hardware=sku, options={"dp": 1, "tp": 1, "pp": 1, "layers": 1, **options})


def _neutral_mem_machine(peaks=None):
    return {"sku": "Ascend910B1", "arch_family": "A2", "device": {
        "memory_bytes": 68_719_476_736, "compute_units": 24,
        "compute_peaks": peaks or {"bf16": 363_724_800_000_000.0},
        "rates": {"hbm": 1_800_000_000_000.0, "vector": 2_841_600_000_000.0}}}


def _mixed_preset(base_sku, other_sku, *, clock=1.6e9, presets=("ascend_sort32_spec",), service=None, other_extra=None):
    arch = {"presets": list(presets)}
    if clock is not None:
        arch["clock_hz"] = clock
    if service is not None:
        arch["instruction_services"] = service
    return Hardware.from_mapping({"backend": "ascend", "sku": base_sku, "num_devices": 2,
                                  "device": {"compute_arch": arch},
                                  "devices": [{}, {"sku": other_sku, **(other_extra or {})}]})


def _execute_sort32(device, vector_work=0.0):
    rate, units = dict(device.rates)["vector"] / 1000, device.units("vector")
    engine = Engine([Resource("vector", rate, units=units)])
    ctx = RecipeContext(None, {}, {"vector": "vector"}, {"vector": rate}, device.compute_units,
                        resource_units={"vector": units}, service_modes=dict(device.service_modes))
    work = InstructionInventory({"VBitSort32": InstructionCount(1600, 51200)}).capacity_work(CONTROL_INSTRUCTIONS)
    priced, unpriced = PrimitiveInventory(work.resources).partition(Geometry.from_devices((device,)).resource_units)
    assert not unpriced
    demand = ctx.bind(priced)
    stages = [Stage("sort", demand, placement=ctx.owned(0, units, ("vbitsort32",)))]
    if vector_work:
        stages.append(Stage("vector", ctx.bind(Work(resources={"vector": vector_work})),
                            placement=ctx.owned(0, units, ("vector",))))
    ready = engine.submit("input", latency=7)
    done = KernelProgram(tuple(stages)).start(engine, "sort", ready)
    engine.advance(until=(done,))
    assert engine.serviced["vector"] == pytest.approx(demand["vector"] + vector_work)
    return done.time - ready.time


# --- Stage-1 datapath derivation, plumbing, homogeneity and UB peer edges --------------
def test_ascend_profiles_l0c_to_ub_datapath_contract():
    raw_profiles = profiles()
    assert len(raw_profiles) == len(ALL_SKUS) == 16
    mem_skus = [s for s in raw_profiles if isa_of(s) == "npu_arch_2201"]
    reg_skus = [s for s in raw_profiles if isa_of(s) == "npu_arch_3510"]
    assert (len(mem_skus), len(reg_skus)) == (12, 4)
    for sku in mem_skus:
        assert "l0c_to_ub" not in raw_profiles[sku]["device"]["physical_mechanisms"]["matrix"]["supported_output_paths"]
        dp = datapath_for_die(_compose_tiny(sku).machine.dies[0])
        assert (dp.arch_id, dp.has_l0c_to_ub, dp.has_ub_to_l1, dp.has_path("gm", "ub")) == (
            "npu_arch_2201", False, False, True)
    for sku in reg_skus:
        assert "l0c_to_ub" in raw_profiles[sku]["device"]["physical_mechanisms"]["matrix"]["supported_output_paths"]
        dp = datapath_for_die(_compose_tiny(sku).machine.dies[0])
        assert (dp.arch_id, dp.has_l0c_to_ub, dp.has_ub_to_l1) == ("npu_arch_3510", True, True)


@pytest.mark.parametrize("sku", SKUS)
def test_compute_arch_and_datapath_plumbing_through_bindings_and_context(sku):
    arch_id, has_direct_ub = isa_of(sku), isa_of(sku) == "npu_arch_3510"
    case = _compose_tiny(sku)
    die, geom = case.machine.dies[0], Geometry.from_devices(case.machine.dies)
    assert geom.compute_arch == die.compute_arch and geom.datapath is not None
    assert (geom.datapath.arch_id, geom.datapath.has_l0c_to_ub, geom.datapath.has_ub_to_l1) == (
        arch_id, has_direct_ub, has_direct_ub)
    bindings = ExecutionBindings.create(case)
    assert bindings.compute_arch == die.compute_arch
    assert bindings.datapath_for_rank(0).arch_id == bindings.datapath_for_stage(0, 0).arch_id == arch_id
    ctx = Execution(case, CostModel(case), bindings=bindings).stage_contexts[(0, 0)]
    assert (ctx.compute_arch, ctx.datapath) == (die.compute_arch, bindings.datapath_for_stage(0, 0))
    assert (ctx.effective_aic_cores, ctx.effective_aiv_cores) == (die.aic_count, die.units("vector"))


def test_geometry_rejects_heterogeneous_compute_arch_stage():
    die22, die35 = _compose_tiny(MEM_WIDE_SKU).machine.dies[0], _compose_tiny(REG_SKU).machine.dies[0]
    with pytest.raises(ModelingError, match="stage_devices_heterogeneous_compute_arch"):
        Geometry.from_devices((die22, die35))
    with pytest.raises(ModelingError, match="stage_devices_heterogeneous_datapath"):
        Geometry.from_devices((die22, _compose_tiny(MEM_SKU).machine.dies[0]))  # 24/48 vs 20/40
    die35_mod = replace(die35, compute_arch=replace(die35.compute_arch, instruction_services=()))
    with pytest.raises(ModelingError, match="stage_devices_heterogeneous_compute_arch"):
        Geometry.from_devices((die35, die35_mod))


def test_ub_peer_edges_derived_from_ub_mesh_links():
    reg_tp8 = _compose_tiny(REG_SKU, tp=8, num_devices=8)
    bindings = ExecutionBindings.create(reg_tp8)
    dp0, dp7 = bindings.datapath_for_rank(0), bindings.datapath_for_rank(7)
    assert dp0.ub_peer_edges == tuple((0, p) for p in range(1, 8))
    assert dp7.ub_peer_edges == tuple((7, p) for p in range(7))
    assert bindings.datapath_for_stage(0, 0).ub_peer_edges == dp0.ub_peer_edges
    assert ExecutionBindings.create(_compose_tiny(MEM_WIDE_SKU, tp=8, num_devices=8)).datapath_for_rank(0).ub_peer_edges == ()
    non_ub = Link(name="hccs_pair:0", capacity_bytes_per_ms=40e6, members=(0, 1),
                  channel="d2d_intra_node", semantics="per_pair")
    assert ub_peer_edges_for_rank(0, (non_ub,), ()) == ()
    asym = (Link(name="ub_pair:0", capacity_bytes_per_ms=40e6, members=(0, 1, 2), channel="ub_mesh", semantics="per_pair"),
            Link(name="ub_pair:1", capacity_bytes_per_ms=40e6, members=(0, 3), channel="ub_mesh", semantics="per_pair"))
    with pytest.raises(ModelingError, match="stage_devices_heterogeneous_datapath"):
        datapath_for_devices(reg_tp8.machine.dies[:2], links=asym)


# --- concrete SKU accounting, native modes, casts, runtime impls and directional paths -
@pytest.mark.parametrize("sku", SKUS)
def test_concrete_sku_owners_scope_and_package_capacity(sku):
    _, aic, aiv, memory, _ = declared_supply(sku)
    hw = machine(sku, num_devices=profiles()[sku]["dies_per_package"])
    die = hw.dies[0]
    assert (die.sku, die.capability.preset, die.compute_units, die.owner_count("vector")) == (sku, isa_of(sku), aic, aiv)
    for role in ("vector", "mte2_aiv"):
        assert die.owner_rate(role) == pytest.approx(die.total_rate(role) / aiv)
    assert die.total_buffer_bytes("ub") == die.pool_capacity("ub") * aiv
    for field, scope in (("memory", "per_device"), ("buffers", "per_owner"),
                         ("compute_peaks", "device_total"), ("resource_units", "owners_per_device")):
        assert die.quantity_scope(field) == scope
    assert hw.package_memory_bytes(0) == hw.dies_per_package * memory


def test_regbase_skus_share_arch_capability_with_distinct_memory_capacities():
    dies = [machine(sku).dies[0] for sku in REGBASE_SKUS]
    ref = dies[0]
    for die in dies[1:]:
        assert dict(die.compute_peaks) == pytest.approx({
            dtype: rate * die.aic_count / ref.aic_count for dtype, rate in ref.compute_peaks})
        assert (die.capability.representations, die.capability.native_modes, die.capability.paths) == (
            ref.capability.representations, ref.capability.native_modes, ref.capability.paths)
    assert len({die.memory_bytes for die in dies}) == len(dies)


def test_peak_calibration_forged_peak_and_cast_never_enable_unsupported_native_mode():
    mem_die, reg_die = machine(MEM_SKU).dies[0], machine(REG_SKU).dies[0]
    mem_cap, reg_cap = mem_die.capability, reg_die.capability
    assert mem_cap.supports_native("matrix", "bf16", "fp32") and mem_cap.supports_native("matrix", "int8", "int32")
    assert mem_cap.supports_native("matrix", "int4", "int32") and mem_die.peak("int4") > 0
    for cap, sku, dtype, acc in ((mem_cap, MEM_SKU, "fp8_e4m3", "fp32"), (reg_cap, REG_SKU, "int4", "int32")):
        assert cap.representation_supports(dtype, "storage") and cap.representation_supports(dtype, "transfer")
        assert not cap.representation_supports(dtype, "operand") and not cap.supports_native("matrix", dtype, acc)
        assert cap.native_modes_for("matrix", dtype, acc) == ()
        with pytest.raises(ModelingError, match="native_compute_mode_unsupported"):
            cap.native_mode("matrix", dtype, acc)
        with pytest.raises(ModelingError, match=f"compute_peak_without_native_compute_mode:{dtype}"):
            machine(sku, device={"compute_peaks": {dtype: 1e15}})
    with pytest.raises(ModelingError, match="compute_peak_without_native_compute_mode:fp8_e5m2"):
        machine(MEM_SKU, device={"compute_peaks": {"fp8_e5m2": 2e15}})
    cal = machine(MEM_SKU, device={"compute_peaks": {"bf16": dict(mem_die.compute_peaks)["bf16"] * 1.25},
                                   "rates": {"hbm": dict(mem_die.rates)["hbm"] * 1.10}}).dies[0]
    assert cal.peak("bf16") == pytest.approx(mem_die.peak("bf16") * 1.25)
    assert dict(cal.rates)["hbm"] == pytest.approx(dict(mem_die.rates)["hbm"] * 1.10)
    assert cal.capability.native_modes == mem_cap.native_modes
    forged = replace(mem_die, compute_peaks=tuple(sorted({**dict(mem_die.compute_peaks), "fp8": 9e15}.items())))
    with pytest.raises(ModelingError, match="native_compute_mode_unsupported"):
        forged.peak("fp8")
    for preset, src, acc in (("npu_arch_2201", "fp8_e4m3", "fp32"), ("npu_arch_3510", "int4", "int32")):
        casted = ComputeArchSpec.from_mapping({"preset": preset, "casts": {
            f"{src}->bf16": {"source": src, "target": "bf16", "implementation": "declared", "lossless": False}}})
        assert casted.can_cast(src, "bf16") and casted.cast_rule(src, "bf16").implementation == "declared"
        assert not casted.supports_native("matrix", src, acc)
    with pytest.raises(ModelingError, match="native_compute_operand_role_missing"):
        ComputeArchSpec.from_mapping({"representations": {"int4": {"bits": 4, "roles": ["storage"]}},
                                      "native_compute": {"int4_cube": {"kind": "matrix", "operands": ["int4"], "unit": "cube"}}})


def test_native_mode_and_runtime_implementation_selection_are_separate():
    cap = machine(REG_SKU).dies[0].capability
    mode = cap.native_mode("matrix", "fp8_e4m3", "fp32")
    assert mode.unit == "cube" and "fp8_e4m3" in mode.operands and "accumulator" in mode.provenance["basis"].lower()
    assert {m.name for m in cap.native_modes_for("matrix", "fp8")} == {
        "fp8_e4m3_x_fp8_e4m3_acc_fp32", "fp8_e5m2_x_fp8_e5m2_acc_fp32"}
    rt = RuntimeCapability("declared-rt", "1", provides=("kernel_b",))
    with pytest.raises(ModelingError, match="native_mode_implementation_undeclared"):
        cap.implemented_native_mode("matrix", "fp8_e4m3", "fp32", rt)
    multi = ComputeArchSpec.from_mapping({
        "representations": {"bf16": {"bits": 16, "roles": ["storage", "transfer", "operand", "accumulator"]}},
        "native_compute": {
            "mode_a": {"kind": "matrix", "operands": ["bf16"], "accumulators": ["bf16"], "unit": "cube", "implementations": ["kernel_a"]},
            "mode_b": {"kind": "matrix", "operands": ["bf16"], "accumulators": ["bf16"], "unit": "cube", "implementations": ["kernel_b"]},
        },
    })
    assert multi.native_mode("matrix", "bf16", "bf16").name == "mode_a"
    assert multi.implemented_native_mode("matrix", "bf16", "bf16", rt).name == "mode_b"
    with pytest.raises(ModelingError, match="native_mode_not_implemented"):
        multi.implemented_native_mode("matrix", "bf16", "bf16", RuntimeCapability("other", "1", provides=("kernel_c",)))


def test_directional_paths_representation_gating_metadata_and_per_device_merge():
    old, new = machine(MEM_SKU).dies[0].capability, machine(REG_SKU).dies[0].capability
    for src, dst in (("gm", "l0a"), ("gm", "l0b"), ("l1", "gm")):
        assert old.path(src, dst) is not None and new.path(src, dst) is None
    for src, dst in (("l0c", "ub"), ("ub", "l1"), ("l1", "ub")):
        assert old.path(src, dst) is None and new.path(src, dst) is not None
    assert new.path("gm", "ub") is not None and new.path("ub", "gm") is not None and new.path("l0a", "l1") is None
    assert old.path("gm", "l1", "fp8_e4m3") is not None and old.path("l1", "l0a", "fp8_e4m3") is None
    assert new.path("l1", "l0a", "fp8_e4m3") is not None and new.path("l1", "l0a", "int4") is None
    assert old.path("l1", "l0a", "bf16") is not None and new.path("l1", "ub", "int4") is not None
    with pytest.raises(ModelingError, match="data_path_unsupported"):
        new.require_path("gm", "l0a", "bf16")
    l0c_ub = new.require_path("l0c", "ub", "fp32")
    assert (l0c_ub.unit, l0c_ub.buffer, l0c_ub.sync, l0c_ub.alignment_bytes) == ("fixpipe", "ub", "event", 64)
    assert {"aic", "aiv"} <= set(l0c_ub.reach)
    ub_l1 = new.require_path("ub", "l1", "bf16")
    assert (ub_l1.unit, ub_l1.buffer, ub_l1.sync, ub_l1.alignment_bytes) == ("mte3", "l1", "event", 32)
    l1_l0a = new.require_path("l1", "l0a", "bf16")
    assert (l1_l0a.unit, l1_l0a.executor, l1_l0a.buffer, l1_l0a.alignment_bytes) == ("mte1", "aic", "l0a", 512)
    l1_ub = new.require_path("l1", "ub", "fp8_e4m3")
    assert (l1_ub.unit, l1_ub.executor, l1_ub.buffer, l1_ub.alignment_bytes) == ("mte1_l1_ub", "aic", "ub", 32)
    reg_die = machine(REG_SKU).dies[0]
    assert dict(reg_die.service_modes)["mte1_l1_ub"] == ("mte1", pytest.approx(dict(reg_die.rates)["mte1"] / 2))
    first, second = Hardware.from_mapping({
        "backend": "ascend", "sku": MEM_SKU, "num_devices": 2,
        "devices": [{}, {"compute_arch": {"casts": {"fp8_e4m3->bf16": {
            "source": "fp8_e4m3", "target": "bf16", "implementation": "declared", "lossless": False}}}}],
    }).dies
    assert not first.capability.can_cast("fp8_e4m3", "bf16") and second.capability.can_cast("fp8_e4m3", "bf16")
    assert second.capability.supports_native("matrix", "bf16", "fp32") and second.capability.path("gm", "l1") is not None


# --- legacy/pre-capability completion, neutral arch/core aliases & custom capability ---
def test_legacy_instruction_services_and_pre_capability_completion():
    assert registered_capability(REG_SKU, "A5") == {"preset": "npu_arch_3510"}
    assert registered_capability(DUAL_DIE_SKU, "A3") == {"preset": "npu_arch_2201"}
    for args in ((REG_SKU, "csa_attention"), ("UNREGISTERED_SKU", "A5"), ("", "A5")):
        assert registered_capability(*args) is None
    base, aiv = machine(REG_SKU).to_dict(), declared_supply(REG_SKU)[2]
    base["devices"][0]["compute_arch"] = {"instruction_services": {
        "legacy_rate": {"unit": "calls", "owner_axis": "vector", "per_owner_rate": 1e9,
                        "provenance": {"kind": "legacy_serializer_control"}},
        "legacy_pipe": {"unit": "calls", "owner_axis": "vector", "per_owner_rate": 2e9,
                        "resource": "vector", "provenance": {"kind": "legacy_serializer_control"}}}}
    restored = Hardware.from_mapping(base)
    die = restored.dies[0]
    assert die.capability.preset == "npu_arch_3510" and Hardware.from_mapping(restored.to_dict()) == restored
    assert [s.name for s in die.capability.instruction_services] == ["legacy_pipe", "legacy_rate"]
    assert dict(die.rates)["legacy_rate"] == pytest.approx(1e9 * aiv)
    assert dict(die.service_modes)["legacy_pipe"] == ("vector", pytest.approx(2e9 * aiv))
    stripped = machine(REG_SKU).to_dict()
    stripped["devices"][0].pop("compute_arch")
    stripped["devices"][0].pop("arch_family")
    legacy = {**stripped, "name": "legacy_case", "architecture": "A5"}
    legacy.pop("hardware_key", None)
    leg_die = Hardware.from_mapping(legacy).dies[0]
    assert (leg_die.sku, leg_die.capability.preset) == (REG_SKU, "npu_arch_3510")
    assert leg_die.peak("fp8_e4m3") == dict(machine(REG_SKU).dies[0].compute_peaks)["fp8_e4m3"]
    assert leg_die.capability.path("l0c", "ub") is not None
    case = compose_case(model="deepseek-v3.2-prune-3l", hardware="a5-server-dt", tp=8, layers=1,
                        prompt_tokens=64, output_tokens=4, max_num_seqs=1, max_num_batched_tokens=64, max_model_len=128)
    case["machine"]["device"].pop("compute_arch", None)
    assert parse_case(case).machine.dies[0].capability.preset == "npu_arch_3510"
    assert predict(case, scope="body", phase="decode", tokens=1, context=64)["duration_ms"] > 0
    unknown = machine(REG_SKU).to_dict()
    unknown["devices"][0].pop("compute_arch")
    unknown["devices"][0]["sku"] = "UNREGISTERED_SKU"
    unknown.pop("hardware_key", None)
    with pytest.raises(ModelingError, match="hardware_native_capability_required"):
        Hardware.from_mapping(unknown)
    for sku, bad_dtype in ((MEM_SKU, "fp8_e4m3"), (REG_SKU, "fp4")):
        bad = machine(sku).to_dict()
        bad["devices"][0].pop("compute_arch")
        bad["devices"][0]["compute_peaks"][bad_dtype] = 1e15
        with pytest.raises(ModelingError, match=f"compute_peak_without_native_compute_mode:{bad_dtype}"):
            Hardware.from_mapping(bad)


def test_neutral_arch_normalization_and_core_alias_precedence_across_entries():
    for variant in (
        _neutral_mem_machine(),
        {**_neutral_mem_machine(), "architecture": "A2", "arch_family": "generic"},
        {**_neutral_mem_machine(), "arch_family": "generic",
         "device": {**_neutral_mem_machine()["device"], "arch_family": "generic"}, "devices": [{"architecture": "A2"}]},
        {**_neutral_mem_machine(), "arch_family": "generic",
         "device": {**_neutral_mem_machine()["device"], "arch_family": "generic"},
         "devices": [{"arch_family": "generic", "device": {"architecture": "A2", "arch_family": "generic"}}]},
    ):
        for entry in (Hardware.from_mapping, schema_parse, custom_parse):
            hw = entry(variant)
            assert hw.arch_family == "A2" and hw.dies[0].capability.preset == "npu_arch_2201"
        bad = {**variant, "device": {**variant["device"], "compute_peaks": {**variant["device"]["compute_peaks"], "fp8_e4m3": 1e15}}}
        for entry in (Hardware.from_mapping, schema_parse, custom_parse):
            with pytest.raises(ModelingError, match="compute_peak_without_native_compute_mode:fp8_e4m3"):
                entry(bad)
    raw = {**hardware_case(), "machine": {**_neutral_mem_machine(), "architecture": "A2", "arch_family": "generic"}}
    assert parse_case(raw).machine.dies[0].capability.preset == "npu_arch_2201"
    assert predict(raw, scope="body", phase="decode", tokens=1, context=16)["duration_ms"] > 0
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 1e15
    with pytest.raises(ModelingError, match="compute_peak_without_native_compute_mode:fp8_e4m3"):
        parse_case(raw)
    for alias in ("num_cores", "cores", "compute_units", "aic_count"):
        die = Hardware.from_mapping({**_neutral_mem_machine(), "architecture": "A2", "devices": [{"device": {alias: 12}}]}).dies[0]
        assert (die.compute_units, die.units("cube")) == (12, 12)
    for override, expected in (({"num_cores": 20, "cores": 18, "compute_units": 16, "aic_count": 14}, 16),
                               ({"num_cores": 8, "cores": 6}, 8), ({"aic_count": 12, "num_cores": 8}, 12)):
        assert Hardware.from_mapping({**_neutral_mem_machine(), "architecture": "A2",
                                      "devices": [{"device": override}]}).dies[0].compute_units == expected
    layered = {**_neutral_mem_machine(), "architecture": "A2", "num_cores": 20,
               "device": {**_neutral_mem_machine()["device"], "cores": 18},
               "devices": [{"num_cores": 16, "device": {"cores": 14}}]}
    assert Hardware.from_mapping(layered).dies[0].compute_units == 14


def test_custom_template_capability_and_compute_arch_roundtrip():
    base = {"sku": "CUSTOM_EXPLICIT", "hbm_gb": 96.0, "hbm_bw_gbps": 4000.0, "cube_fp16_tflops": 600.0}
    synth = custom_parse(build_custom_hardware_machine(base, num_devices=1))
    die = synth.dies[0]
    assert die.capability.has_native_modes() and die.peak("fp8") > 0 and die.capability.path("gm", "l1") is None
    assert die.capability.runtimes == () and Hardware.from_mapping(synth.to_dict()) == synth
    with pytest.raises(ModelingError, match="compute_peak_without_native_compute_mode"):
        custom_parse(build_custom_hardware_machine({**base, "capability": {"preset": "npu_arch_2201"}}, num_devices=1))
    cap = custom_parse(build_custom_hardware_machine({**base, "capability": {"preset": "npu_arch_3510"}}, num_devices=1)).dies[0].capability
    assert cap.native_mode("matrix", "fp8_e4m3", "fp32") and cap.path("l0c", "ub") is not None and cap.path("gm", "l0a") is None
    arch = ComputeArchSpec.from_mapping({"preset": "npu_arch_3510", "casts": {"fp8_e4m3->bf16": {
        "source": "fp8_e4m3", "target": "bf16", "implementation": "test_only", "lossless": False}}})
    assert ComputeArchSpec.from_mapping(arch.to_dict()) == arch


# --- heterogeneous per-placement isolation, instruction presets, context & link peaks --
def test_heterogeneous_placements_isolate_sku_services_context_and_homogeneity():
    base_pr = machine(REG_PR_SKU).dies[0]
    mixed = Hardware.from_mapping({"backend": "ascend", "sku": REG_SKU, "num_devices": 2,
                                   "device": {"launch_us": 7.0}, "devices": [{"sku": REG_SKU}, {"sku": REG_PR_SKU}]})
    dt, pr = mixed.dies
    assert (mixed.sku, mixed.homogeneous, mixed.placement_sku(0), mixed.placement_sku(1)) == ("mixed", False, REG_SKU, REG_PR_SKU)
    assert dt.launch_us == pr.launch_us == pytest.approx(7.0)
    assert mixed.per_placement_peaks("bf16") == {0: dt.peak("bf16"), 1: pr.peak("bf16")}
    with pytest.raises(ModelingError, match="hardware_resources_are_per_placement"):
        mixed.peak_flops_for("bf16")
    pr_cl = next(s for s in pr.capability.instruction_services if s.name == "gm_hbm_cacheline_128b")
    assert pr_cl.per_owner_rate == pytest.approx(dict(pr.rates)["hbm"] / pr.owner_count("vector") / 128)
    assert dict(pr.rates)["gm_hbm_cacheline_128b"] == pytest.approx(dict(pr.rates)["hbm"] / 128)
    assert dict(pr.rates) == dict(base_pr.rates) and dict(pr.service_modes) == dict(base_pr.service_modes)
    assert "get_sff_value_u64_i64" in dict(pr.service_modes) and Hardware.from_mapping(mixed.to_dict()) == mixed
    cross = Hardware.from_mapping({"backend": "ascend", "sku": REG_SKU, "num_devices": 2,
                                   "devices": [{"sku": REG_SKU}, {"sku": MEM_SKU}]})
    assert "fp8_e4m3" not in dict(cross.dies[1].compute_peaks) and "pipe_v_lanes" not in dict(cross.dies[1].rates)
    assert "gm_hbm_cacheline_128b" not in dict(cross.dies[1].rates)
    assert not Hardware.from_mapping({"backend": "ascend", "sku": REG_PR_SKU, "num_devices": 2,
                                      "devices": [{}, {"launch_us": 1.0}]}).homogeneous
    ctx0, ctx1 = mixed.hardware_context(0), mixed.hardware_context(1)
    assert (ctx0["schema"], ctx0["device_sku"], ctx1["device_sku"], ctx0["homogeneous"]) == (
        "mindie-sim.hardware-context/1", REG_SKU, REG_PR_SKU, False)
    assert ctx0["memory_bytes"] != ctx1["memory_bytes"] and ctx0["capability"]["preset"] == "npu_arch_3510"
    assert not ({"model", "workload", "duration_ms"} & set(ctx0))
    assert ctx1["rates"]["gm_hbm_cacheline_128b"] == pytest.approx(dict(pr.rates)["hbm"] / 128)
    with pytest.raises(ModelingError, match="hardware_placement_unknown"):
        mixed.hardware_context(9)


def test_heterogeneous_instruction_preset_clock_disable_and_nested_owner_scaling():
    for base, other in ((REG_SKU, REG_PR_SKU), (REG_PR_SKU, REG_SKU)):
        hw = _mixed_preset(base, other)
        for die, sku in zip(hw.dies, (base, other)):
            alone = machine(sku, device={"compute_arch": {"presets": ["ascend_sort32_spec"], "clock_hz": 1.6e9}}).dies[0]
            assert dict(die.service_modes)["vbitsort32"] == dict(alone.service_modes)["vbitsort32"]
            assert dict(die.work_units)["vbitsort32"] == "blocks32" and dict(die.rates) == dict(alone.rates)
        assert Hardware.from_mapping(hw.to_dict()) == hw
    base_sort = 1.6e9 / 16
    assert all(dict(d.service_modes)["vbitsort32"] == ("vector", pytest.approx(2 * base_sort * d.units("vector")))
               for d in _mixed_preset(REG_SKU, REG_PR_SKU, clock=3.2e9).dies)
    assert all("vbitsort32" not in dict(d.service_modes) for d in _mixed_preset(REG_SKU, REG_PR_SKU, presets=()).dies)
    explicit = _mixed_preset(REG_SKU, REG_PR_SKU, service={"vbitsort32": {"per_owner_rate": 2e8}})
    assert Hardware.from_mapping(explicit.to_dict()) == explicit
    assert dict(explicit.dies[1].service_modes)["vbitsort32"] == ("vector", pytest.approx(2e8 * explicit.dies[1].units("vector")))
    local_clk = _mixed_preset(REG_SKU, REG_PR_SKU, other_extra={"device": {"compute_arch": {"clock_hz": 3.2e9}}})
    assert dict(local_clk.dies[1].service_modes)["vbitsort32"][1] == pytest.approx(2 * base_sort * local_clk.dies[1].units("vector"))
    local_off = _mixed_preset(REG_SKU, REG_PR_SKU, other_extra={"device": {"compute_arch": {"presets": []}}})
    assert "vbitsort32" in dict(local_off.dies[0].service_modes) and "vbitsort32" not in dict(local_off.dies[1].service_modes)
    base_pr = machine(REG_PR_SKU).dies[0]
    half_aiv = base_pr.units("vector") // 2
    pr_n = _mixed_preset(REG_SKU, REG_PR_SKU, other_extra={"device": {"aiv_count": half_aiv}}).dies[1]
    pr_f = _mixed_preset(REG_SKU, REG_PR_SKU, other_extra={"aiv_count": half_aiv}).dies[1]
    assert (dict(pr_n.rates), dict(pr_n.resource_units), dict(pr_n.buffer_units), dict(pr_n.service_modes)) == (
        dict(pr_f.rates), dict(pr_f.resource_units), dict(pr_f.buffer_units), dict(pr_f.service_modes))
    assert pr_n.units("vector") == pr_n.units("mte2_aiv") == pr_n.pool_units("ub") == half_aiv
    assert pr_n.owner_rate("vector") == pytest.approx(base_pr.owner_rate("vector"))
    assert dict(pr_n.service_modes)["vbitsort32"][1] == pytest.approx(base_sort * half_aiv)
    tuned = _mixed_preset(REG_SKU, REG_PR_SKU, other_extra={"device": {
        "compute_units": 16, "aiv_count": half_aiv, "compute_peaks": {"bf16": 1e12}, "rates": {"vector": 1e12},
        "buffer_units": {"ub": 17}, "buffers": {"ub": 1000},
        "service_modes": {"explicit_mode": {"resource": "vector", "rate": 123.0}}}})
    pr_t = tuned.dies[1]
    assert (pr_t.compute_units, pr_t.units("vector"), pr_t.pool_units("ub"), pr_t.pool_capacity("ub")) == (16, half_aiv, 17, 1000)
    assert (pr_t.peak("bf16"), dict(pr_t.rates)["vector"], dict(pr_t.service_modes)["explicit_mode"]) == (
        pytest.approx(1e12), pytest.approx(1e12), ("vector", 123.0))
    assert pr_t.units("mte1") == 16 and dict(pr_t.rates)["mte1"] == pytest.approx(base_pr.owner_rate("mte1") * pr_t.aic_count)
    assert dict(pr_t.service_modes)["mte1_l1_ub"] == ("mte1", pytest.approx(dict(pr_t.rates)["mte1"] / 2))
    assert Hardware.from_mapping(tuned.to_dict()) == tuned
    expected = 1600 * 16 / 1.6e9 * 1000
    assert _execute_sort32(pr_t) == pytest.approx(expected)
    assert _execute_sort32(pr_t, vector_work=dict(pr_t.rates)["vector"] / pr_t.units("vector") * expected / 1000) == pytest.approx(2 * expected)


def test_link_explicit_peak_and_per_endpoint_rates_roundtrip():
    hw = Hardware.from_mapping({
        "backend": "ascend", "sku": REG_SKU, "num_devices": 2,
        "links": [{"name": "fabric", "capacity_bytes_per_ms": 25e6, "peak_capacity_bytes_per_ms": 25e6, "members": [0, 1]}],
        "transfer_channels": {"d2d_inter_node": {"effective_uni_bw_gbps": 50.0}},
    })
    assert hw.links[0].peak_capacity_bytes_per_ms == 25e6
    restored = Hardware.from_mapping(hw.to_dict())
    assert restored == hw and restored.transfer_channels == hw.transfer_channels
    link = Link.from_mapping({"name": "shared_fwd", "capacity_bytes_per_ms": 60.0, "members": [0, 1], "endpoint_rates": {1: 30.0}})
    assert (link.endpoint_capacity_bytes_per_ms(0), link.capacity_at(0),
            link.endpoint_capacity_bytes_per_ms(1), link.capacity_at(1)) == (None, 60.0, 30.0, 30.0)
    assert Link.from_mapping(link.to_dict()) == link
    with pytest.raises(ModelingError, match="link_endpoint_member_unknown"):
        Link.from_mapping({"name": "bad", "capacity_bytes_per_ms": 60.0, "members": [0, 1], "endpoint_rates": {2: 30.0}})


# --- MTE1 / vector direction modes and machine/device/rank rate overrides --------------
def test_mte1_rate_overrides_rescale_inherited_modes_and_honor_explicit_and_rank_layers():
    base = sku_case(DUAL_DIE_SKU)
    die0 = base.machine.dies[0]
    mte1, aic = dict(die0.rates)["mte1"], die0.units("mte1")
    l0a, l0b = dict(die0.service_modes)["mte1_l0a"][1], dict(die0.service_modes)["mte1_l0b"][1]
    half = sku_case(DUAL_DIE_SKU, {"mte1_bw_bytes_per_s": mte1 / 2})
    neutral = sku_case(DUAL_DIE_SKU, {"mte1_bw_bytes_per_s": 2 * mte1, "rates": {"mte1": mte1 / 2}})
    assert dict(half.machine.dies[0].service_modes) == dict(neutral.machine.dies[0].service_modes)
    assert dict(half.machine.dies[0].service_modes)["mte1_l0a"] == ("mte1", pytest.approx(l0a / 2))
    for mode, rate in (("mte1_l0a", l0a), ("mte1_l0b", l0b)):
        before, after = one_owner_ms(base, mode, AMOUNT), one_owner_ms(half, mode, AMOUNT)
        assert before == pytest.approx(AMOUNT * aic / rate * 1000) and after == pytest.approx(2 * before)
    explicit = sku_case(DUAL_DIE_SKU, {"mte1_bw_bytes_per_s": mte1 / 2, "service_modes": {"mte1_l0a": {"rate": l0a}}})
    assert dict(explicit.machine.dies[0].service_modes)["mte1_l0a"] == ("mte1", pytest.approx(l0a))
    comp = run(explicit, (op("a", "a", "sa"), op("b", "b", "sb")),
               {"a": OneMode("mte1_l0a", AMOUNT, 0), "b": OneMode("mte1_l0b", AMOUNT, 0)})
    assert comp["duration_ms"] == pytest.approx((AMOUNT * aic / l0a + AMOUNT * aic / (l0b / 2)) * 1000)
    assert comp["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(
        AMOUNT * (mte1 / 2) / l0a + AMOUNT * (mte1 / 2) / (l0b / 2))
    case = Case.from_mapping({
        "machine": {"backend": "ascend", "sku": DUAL_DIE_SKU, "num_devices": 2, "ranks_per_node": 2,
                    "mte1_bw_bytes_per_s": mte1 / 4, "rates": {"vector": 1e12},
                    "device": {"rates": {"mte1": mte1 / 2}}, "devices": [{"mte1_bw_bytes_per_s": mte1 / 4}, {}]},
        "model": MODEL, "workload": WORKLOAD, "runtime": {"dp": 2}})
    assert Case.from_mapping(case.to_dict()) == case
    for replica, expected_mte1 in enumerate((mte1 / 4, mte1 / 2)):
        die = case.machine.dies[replica]
        mode_rate = dict(die.service_modes)["mte1_l0a"][1]
        assert dict(die.rates)["mte1"] == pytest.approx(expected_mte1) and dict(die.rates)["vector"] == 1e12
        assert mode_rate == pytest.approx(l0a * (expected_mte1 / mte1))
        report = graph_costs(case, {replica: (op("m", "m"),)},
                             recipes=RecipeBook({"m": OneMode("mte1_l0a", AMOUNT, 0)}), isolated=False)
        assert report["duration_ms"] == pytest.approx(AMOUNT * aic / mode_rate * 1000)


def test_vector_modes_follow_device_and_machine_overrides_without_moving_mte1():
    base = sku_case(REG_SKU)
    die0 = base.machine.dies[0]
    vec, aiv, ub_copy_rate = dict(die0.rates)["vector"], die0.units("vector"), dict(die0.service_modes)["ub_copy"][1]
    half = sku_case(REG_SKU, {"vector_ops_per_s": vec / 2})
    explicit_dev = sku_case(REG_SKU, {"rates": {"vector": vec / 2}, "service_modes": {"ub_copy": {"rate": ub_copy_rate}}})
    explicit_mach = sku_case(REG_SKU, vector_ops_per_s=vec / 2, service_modes={"ub_copy": {"rate": ub_copy_rate}})
    assert Case.from_mapping(explicit_mach.to_dict()) == explicit_mach
    for role, amt in (("ub_copy", 1e9), ("reg_issue_slots", 1e6)):
        r0 = dict(die0.service_modes)[role][1]
        before, after = one_owner_ms(base, role, amt), one_owner_ms(half, role, amt)
        assert before == pytest.approx(amt * aiv / r0 * 1000) and after == pytest.approx(2 * before)
        for case in (explicit_dev, explicit_mach):
            assert one_owner_ms(case, role, amt) == pytest.approx(before if role == "ub_copy" else after)
    assert one_owner_ms(base, "mte1", AMOUNT) == one_owner_ms(half, "mte1", AMOUNT)


def test_split_mte_convenience_explicit_roles_and_neutral_pipe_aliases_reach_engine():
    base_die = sku_case(DUAL_DIE_SKU).machine.dies[0]
    aic, aiv, hbm = base_die.units("mte2_aic"), base_die.units("mte2_aiv"), dict(base_die.rates)["hbm"]
    convenience = sku_case(DUAL_DIE_SKU, {"mte_bw_bytes_per_s": 1e12})
    assert [dict(convenience.machine.dies[0].rates)[r] for r in ("mte2", "mte2_aic", "mte2_aiv", "mte3")] == [1e12] * 4
    assert one_owner_ms(convenience, "mte2_aic", AMOUNT) == pytest.approx(AMOUNT * aic / 1e12 * 1000)
    override = sku_case(DUAL_DIE_SKU, {"mte_bw_bytes_per_s": 1e12, "rates": {"mte2_aic": 2e12, "mte2_aiv": 4e12, "mte3": 8e12}})
    for role, units, rate in (("mte2_aic", aic, 2e12), ("mte2_aiv", aiv, 4e12), ("mte3", aiv, 8e12)):
        assert one_owner_ms(override, role, AMOUNT) == pytest.approx(AMOUNT * units / rate * 1000)
    shared = run(override, (op("aic", "aic", "a"), op("aiv", "aiv", "b")), {
        "aic": ReadMoves("mte2_aic", aic * AMOUNT, aic), "aiv": ReadMoves("mte2_aiv", aiv * AMOUNT, aiv)})
    assert shared["duration_ms"] == pytest.approx((aic + aiv) * AMOUNT / hbm * 1000)
    narrow = sku_case(DUAL_DIE_SKU, {"vector_ops_per_s": 5e12, "hbm_bw_bytes_per_s": 2e12}).machine.dies[0]
    assert (dict(narrow.rates)["vector"], dict(narrow.rates)["hbm"], narrow.units("vector")) == (5e12, 2e12, aiv)
    assert dict(narrow.rates)["mte1"] == dict(base_die.rates)["mte1"]
    neutral = Case.from_mapping({
        "machine": {"memory_bytes": 1024 ** 3, "compute_peaks": {"bf16": 1e12},
                    "rates": {"hbm": 1e12, "vector": 1e12}, "mte_bw_bytes_per_s": 1e12,
                    "mte2_aic_bw_bytes_per_s": 2e12, "mte2_aiv_bw_bytes_per_s": 4e12,
                    "mte3_bw_bytes_per_s": 8e12, "device": {"rates": {"mte2_aic": 3e12}}},
        "model": MODEL, "workload": WORKLOAD, "runtime": {}})
    assert Case.from_mapping(neutral.to_dict()) == neutral
    for role, rate in (("mte2", 1e12), ("mte2_aic", 3e12), ("mte2_aiv", 4e12), ("mte3", 8e12)):
        assert one_owner_ms(neutral, role, AMOUNT) == pytest.approx(AMOUNT / rate * 1000)


# --- matrix split read axes (AIC operands vs AIV scales bypass legacy generic mte2) ----
def _run_split_matrix(recipe, *, aic=1, aiv=2, generic=1e-6):
    op_spec = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(), m=1, n=1, k=1,
                      quant=QuantSpec("bf16"), category="dense_mlp", phase="prefill")
    metadata = {**dict(op_spec.metadata), "compute_format": "int8", "output_format": "bf16",
                "matrix_groups": 1, "packed_weight_bytes": 5, "activation_scale_bytes": 4,
                "matrix_epilogue": {"kind": "row_column_scale", "operations_per_output": 4,
                                    "row_scale_bytes": 4, "column_scale_bytes": 4}}
    op_spec = replace(op_spec, dtype="int8", vector_ops=4, hbm_bytes=12, mte2_bytes=10,
                      mte3_bytes=2, metadata=tuple(metadata.items()))
    rates = {**dict.fromkeys(("cube", "vector", "hbm", "mte1", "mte3"), 1e12),
             "mte2": generic, "mte2_aic": aic, "mte2_aiv": aiv}
    units = {name: 2 if name in ("vector", "mte2_aiv", "mte3") else 1 for name in rates}
    pools = {"l1": 256, "l0a": 32, "l0b": 32, "l0c": 256}
    ctx = RecipeContext(op_spec, {"cube": op_spec.flops, "vector": op_spec.vector_ops}, {n: n for n in rates},
                        rates, 1, pool_names={n: n for n in pools}, pool_capacities=pools, resource_units=units)
    engine = Engine([Resource(n, r, units=units[n]) for n, r in rates.items()], pools=pools, pool_units={n: 1 for n in pools})
    done = recipe.lower(ctx).start(engine, "matrix", engine.submit("ready", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values()) and engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return done.time - 7, engine, ctx


@pytest.mark.parametrize("recipe", SPLIT_RECIPES, ids=lambda r: type(r).__name__)
def test_split_matrix_read_axes_bypass_legacy_mte2_and_scale_independently(recipe):
    duration, engine, ctx = _run_split_matrix(recipe)
    assert duration == pytest.approx(6, abs=1e-8)
    assert (engine.serviced["mte2"], engine.serviced["mte2_aic"], engine.serviced["mte2_aiv"],
            engine.serviced["hbm"], engine.serviced["cube"], engine.serviced["vector"]) == (0, 2, 8, 12, 2, 4)
    assert mapping_bound(ctx, HierarchicalMatmul(1, 1, 1, 1, 1)) == 4
    fast_generic, other, _ = _run_split_matrix(recipe, generic=1e12)
    assert fast_generic == duration and other.serviced == engine.serviced
    for axis, expected in (("aic", (8, 6, 5)), ("aiv", (10, 6, 4))):
        scans = [_run_split_matrix(recipe, **{axis: (1 if axis == "aic" else 2) * s}) for s in (.5, 1, 2)]
        assert [r[0] for r in scans] == pytest.approx(expected, abs=1e-8)
        assert all(r[1].serviced == scans[1][1].serviced for r in scans)
