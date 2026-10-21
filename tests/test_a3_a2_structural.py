"""Structural A2/A3 corrections in the predictive path (A2/A3 accuracy wave).

Every assertion here is a *derivation* check, not an accuracy number:

* the wave term uses the machine's own AIC count (20 vs 24 really differ);
* launch is additive dispatch + service, not a max-competitive resource;
* attention PV contracts ``kv`` while QK contracts ``dim``;
* a link is resolved from the op's declared fabric and cannot fall back to
  ``links[0]``;
* an A2 package has an HCCS mesh and no SIO; an A3 package keeps SIO;
* a measured residual requires hardware identity (no empty-key wildcard);
* the measurement machine is derived from the resolved device profile plus the
  runtime clock snapshot, not from an A3 literal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim.a3_cost import (
    CalibrationStore, PredictiveCostModel, _gemm_efficiency, _shape_efficiency,
)
from mindie_sim.a3_ops import IterationWork, OpSpec
from mindie_sim.a3_real_data import machine_from_environment
from mindie_sim.a3_resources import plan_replica_links, select_link_for_ranks
from mindie_sim.a3_types import A3InputError, MachineSpec, ParallelSpec
from mindie_sim.a3_workload import IterationPlan
from mindie_sim.hardware import resolve_device_capability_profile

FIXTURES = Path(__file__).parent / "fixtures" / "mindie_sim"


def _device(**overrides) -> dict:
    device = {
        "hbm_gib": 8.0, "cube_flops_per_s": 1e14, "vector_ops_per_s": 1e13,
        "hbm_bw_bytes_per_s": 1e12, "mte_bw_bytes_per_s": 2e12,
        "launch_us": 20.0, "l2_bytes": 1024 * 1024,
    }
    device.update(overrides)
    return device


def _metadata(machine: MachineSpec) -> dict:
    """Machine metadata is a string map; structured values are JSON-encoded."""
    raw = dict(machine.metadata)
    parsed = {}
    for key, value in raw.items():
        try:
            parsed[key] = json.loads(value)
        except (TypeError, ValueError):
            parsed[key] = value
    return parsed


def _gemm_op(m: int, n: int, k: int, **overrides) -> OpSpec:
    payload = dict(
        op_id="gemm", family="gemm", phase="operator", stage=0, layer=0,
        requests=(), semantic_shape=(m, n, k), local_shape=(m, n, k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=2.0 * m * n * k, hbm_bytes=0.0, dtype="bf16",
        layout="contiguous", recipe="r",
    )
    payload.update(overrides)
    return OpSpec(**payload)


# ---------------------------------------------------------------------------
# D1/B1 — AIC count from the machine
# ---------------------------------------------------------------------------


def test_wave_quantization_uses_declared_aic_count_not_a_literal():
    op = _gemm_op(32, 5120, 5120)  # 40 tiles at the declared 64x128 geometry
    eff20, detail20 = _gemm_efficiency(op, 20)
    eff24, detail24 = _gemm_efficiency(op, 24)
    assert detail20["tiles"] == detail24["tiles"] == 40.0
    assert detail20["waves"] == 2.0 and detail24["waves"] == 2.0
    # 40 tiles on 20 cores is exactly two full waves; on 24 cores the second
    # wave is 16/24 occupied, so the kernel cannot retire earlier.
    assert eff20 == pytest.approx(1.0)
    assert eff24 == pytest.approx(40.0 / 48.0)

    machine20 = MachineSpec.from_mapping({
        "name": "a2-decode", "num_devices": 1,
        "device": _device(aic_count=20),
    })
    machine24 = MachineSpec.from_mapping({
        "name": "a1-decode", "num_devices": 1,
        "device": _device(aic_count=24),
    })
    cost20 = PredictiveCostModel(machine20).cost(op).demand_map()
    cost24 = PredictiveCostModel(machine24).cost(op).demand_map()
    assert cost20["cube"] / cost24["cube"] == pytest.approx(40.0 / 48.0)
    assert _shape_efficiency(op, machine20)[2]["aic_count_source"] == "die.aic_count"

    # A machine that declares no AIC count keeps the documented legacy
    # reference and says so in the cost detail.
    undeclared = MachineSpec.from_mapping({
        "name": "no-aic", "num_devices": 1, "device": _device(),
    })
    _, _, detail_legacy = _shape_efficiency(op, undeclared)
    assert detail_legacy["aic_count_source"] == "legacy_fallback_24"
    assert detail_legacy["wave_efficiency"] == pytest.approx(40.0 / 48.0)


# ---------------------------------------------------------------------------
# W8 — launch is additive
# ---------------------------------------------------------------------------


def test_launch_is_additive_dispatch_plus_service():
    machine = MachineSpec.from_mapping({
        "name": "launch-additive", "num_devices": 1,
        "device": _device(launch_us=20.0),
    })
    # 1e6 FLOPs at 1e14 flops/s is 10 ns of cube service -- far below dispatch.
    op = _gemm_op(1000, 1000, 1)
    op = OpSpec(
        op_id="tiny", family="gemm", phase="operator", stage=0, layer=0,
        requests=(), semantic_shape=(1000, 1, 1), local_shape=(1000, 1, 1),
        padded_shape=(1000, 1, 1), launch_shape=(1000, 1, 1),
        flops=1e6, dtype="bf16", layout="contiguous", recipe="r",
    )
    cost = PredictiveCostModel(machine).cost(op)
    service = cost.demand_map()["cube"] / 1e11  # flops / (flops per ms)
    assert cost.service_ms == pytest.approx(service)
    assert cost.launch_ms == pytest.approx(0.020)
    assert cost.estimated_ms == pytest.approx(0.020 + service)
    # The old model reported max(launch, service) == launch and hid the service
    # time entirely; additive composition must strictly exceed the launch floor.
    assert cost.estimated_ms > cost.launch_ms

    eager = cost.estimated_ms
    graph = PredictiveCostModel(machine, launch_amortization=0.0).cost(op)
    assert graph.launch_ms == 0.0
    assert graph.estimated_ms == pytest.approx(service)
    assert eager - graph.estimated_ms == pytest.approx(0.020)


# ---------------------------------------------------------------------------
# W3/W4/W5/W6 — no invented saturating floors
# ---------------------------------------------------------------------------


def test_no_saturating_efficiency_floors_remain_in_demands():
    machine = MachineSpec.from_mapping({
        "name": "raw-demands", "num_devices": 1,
        "device": _device(aic_count=64, launch_us=0.0),
    })
    op = OpSpec(
        op_id="vec", family="rmsnorm", phase="operator", stage=0, layer=0,
        requests=(), semantic_shape=(100,), local_shape=(100,),
        padded_shape=(100,), launch_shape=(100,), vector_ops=100.0,
        hbm_bytes=1024.0, mte2_bytes=512.0, mte3_bytes=512.0,
        dtype="bf16", layout="contiguous", recipe="r",
    )
    demands = PredictiveCostModel(machine).cost(op).demand_map()
    # Raw work: the resource ceiling is the only bound.  0.5+0.5*ops/4096 and
    # 0.65+0.35*bytes/2MiB / 0.5+0.5*bytes/1MiB previously inflated small work.
    assert demands["vector"] == pytest.approx(100.0)
    assert demands["hbm"] == pytest.approx(1024.0)
    assert demands["mte2"] == pytest.approx(512.0)
    assert demands["mte3"] == pytest.approx(512.0)

    # The deleted k/(k+128) factor: k no longer changes the wave term.
    eff_k_small, detail_small = _gemm_efficiency(_gemm_op(64, 128, 8), 64)
    eff_k_large, detail_large = _gemm_efficiency(_gemm_op(64, 128, 8192), 64)
    assert detail_small["k_pipeline_efficiency"] == 1.0
    assert detail_large["k_pipeline_efficiency"] == 1.0
    assert eff_k_small == eff_k_large == pytest.approx(1.0 / 64.0)


# ---------------------------------------------------------------------------
# W10 — PV attention mapping
# ---------------------------------------------------------------------------


def test_attention_pv_contracts_kv_while_qk_contracts_dim():
    q, kv, heads, dim = 8, 1024, 4, 64
    common = dict(
        phase="attention", stage=0, layer=0, requests=(),
        semantic_shape=(q, kv, heads, dim), local_shape=(q, kv, heads, dim),
        padded_shape=(q, kv, heads, dim), launch_shape=(q, kv, heads, dim),
        flops=4.0 * q * kv * heads * dim, dtype="bf16",
        layout="contiguous", recipe="r",
    )
    qk = OpSpec(op_id="qk", family="attention_qk", **common)
    pv = OpSpec(op_id="pv", family="attention_pv", **common)
    fused = OpSpec(op_id="fused", family="attention_fused", **common)

    _, qk_detail = _gemm_efficiency(qk, 20)
    _, pv_detail = _gemm_efficiency(pv, 20)
    _, fused_detail = _gemm_efficiency(fused, 20)

    # QK: (m,n,k) = (q, kv, dim) -> ceil(8/64) * ceil(1024/128) = 8 tiles.
    assert qk_detail["tiles"] == 8.0
    assert qk_detail["contraction_mapping"] == "attention_qk:m=q,n=kv,k=dim"
    # PV: (m,n,k) = (q, dim, kv) -> ceil(8/64) * ceil(64/128) = 1 tile.
    # The asymmetric kv=1024 / dim=64 case is the discriminator: under the old
    # QK mapping PV would also count 8 tiles and look 8x more efficient.
    assert pv_detail["tiles"] == 1.0
    assert pv_detail["contraction_mapping"] == "attention_pv:m=q,n=dim,k=kv"
    assert pv_detail["wave_efficiency"] == pytest.approx(1.0 / 20.0)
    assert fused_detail["tiles"] == 9.0  # both matmuls in one fused kernel
    assert fused_detail["contraction_mapping"] == "attention_fused:qk_tiles+pv_tiles"

    # Demand-level: PV is tile-starved by dim, not by kv.
    machine = MachineSpec.from_mapping({
        "name": "attn", "num_devices": 1, "device": _device(aic_count=20),
    })
    pv_cost = PredictiveCostModel(machine).cost(pv)
    qk_cost = PredictiveCostModel(machine).cost(qk)
    assert pv_cost.demand_map()["cube"] > qk_cost.demand_map()["cube"]


# ---------------------------------------------------------------------------
# D3/D4/B4/B5 — one link source of truth and per-op fabric
# ---------------------------------------------------------------------------


def test_default_a2_links_are_hccs_mesh_and_have_no_sio():
    a2 = MachineSpec.from_mapping({"sku": "Ascend910B4", "num_devices": 8})
    channels = {link.channel for link in a2.links}
    assert channels == {"d2d_intra_node"}
    node = a2.link_for_scope("d2d_intra_node")
    assert node is not None
    assert node.protocol == "HCCS"
    assert node.semantics == "mesh"
    assert node.capacity_bytes_per_ms == pytest.approx(168e6)
    assert node.latency_us == pytest.approx(2.5)
    assert a2.dies_per_package == 1
    # A single-die package has no SIO wire: asking for one is an explicit error.
    with pytest.raises(A3InputError, match="link_scope_unresolved:sio"):
        a2.link_for_scope("sio")
    assert a2.channel("d2d_intra_package").effective_uni_bw_gbps is None


def test_default_a3_links_keep_sio_and_hccs_sw():
    a3 = MachineSpec.from_mapping({"sku": "Ascend910_9362", "num_devices": 8})
    by_channel = {}
    for link in a3.links:
        by_channel.setdefault(link.channel, []).append(link)
    assert set(by_channel) == {"d2d_intra_package", "d2d_intra_node"}
    # Four dual-die packages => four dedicated SIO pair wires.
    assert sorted(link.members for link in by_channel["d2d_intra_package"]) == [
        (0, 1), (2, 3), (4, 5), (6, 7),
    ]
    sio = a3.link_for_scope("sio")
    assert sio is not None and sio.protocol == "SIO"
    assert sio.semantics == "direct"
    assert sio.latency_us == pytest.approx(0.8)
    switch = a3.link_for_scope("hccs")
    assert switch is not None and switch.protocol == "HCCS_SW"
    assert switch.semantics == "switched"


def test_rank_membership_prefers_the_most_specific_fabric():
    a3 = MachineSpec.from_mapping({"sku": "Ascend910_9362", "num_devices": 8})
    a2 = MachineSpec.from_mapping({"sku": "Ascend910B4", "num_devices": 8})
    # A two-die TP group inside one package runs on its SIO pair; a pair that
    # spans packages has no SIO wire and must use the intra-node switch.
    assert select_link_for_ranks(a3, (0, 1)).channel == "d2d_intra_package"
    assert select_link_for_ranks(a3, (0, 2)).channel == "d2d_intra_node"
    # A2 has one full mesh: every pair is an independent HCCS wire.
    for ranks in ((0, 1), (0, 7), (3, 4)):
        assert select_link_for_ranks(a2, ranks).channel == "d2d_intra_node"
    # A one-package A3 model has both wires covering ranks 0..1; the dedicated
    # SIO pair is the faster specific path, the switch is for cross-package.
    a3_package = MachineSpec.from_mapping({
        "sku": "Ascend910_9362", "num_devices": 2,
    })
    assert select_link_for_ranks(a3_package, (0, 1)).channel == "d2d_intra_package"


def test_op_link_scope_selects_the_fabric_and_rejects_an_absent_one():
    a3 = MachineSpec.from_mapping({"sku": "Ascend910_9362", "num_devices": 8})
    a2 = MachineSpec.from_mapping({"sku": "Ascend910B4", "num_devices": 8})
    kwargs = dict(
        op_id="comm", family="allreduce", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1,), local_shape=(1,),
        padded_shape=(1,), launch_shape=(1,), link_bytes=448_000_000.0,
        dtype="bf16", layout="contiguous", recipe="r",
    )
    op = OpSpec(**kwargs)
    on_sio = PredictiveCostModel(
        a3, communication_placement="sio",
    ).cost(op)
    on_hccs = PredictiveCostModel(
        a3, communication_placement="hccs",
    ).cost(op)
    # SIO 448 Gbps carries 448 MB in 1 ms; HCCS_SW 196 Gbps needs ~2.29 ms.
    # 448 MB on a 448 Gbps SIO wire = 1 ms service; the dispatch term is the
    # 20 us kernel intercept plus the fabric's fixed setup latency.
    assert on_sio.estimated_ms == pytest.approx(1.0 + 0.020 + 0.0008, rel=1e-6)
    assert on_hccs.estimated_ms == pytest.approx(
        448_000_000.0 / 196e6 + 0.020 + 0.0025, rel=1e-6,
    )
    assert on_sio.estimated_ms < on_hccs.estimated_ms

    # The same declared SIO placement on A2 (no SIO wire) is an explicit error,
    # not a silent price on the HCCS link.
    with pytest.raises(A3InputError, match="link_scope_unresolved:sio"):
        PredictiveCostModel(a2, communication_placement="sio").cost(op)


def test_measured_whole_span_envelope_is_not_double_charged_launch():
    class SpanCalibration:
        def predict(self, **kwargs):
            return {
                "status": "calibrated_typed_whole_span",
                "span_ms": 0.5,
                "stages": (),
                "observed_execution": "unknown",
                "calibrated_scope": "typed_whole_span",
            }

    a3 = MachineSpec.from_mapping({"sku": "Ascend910_9362", "num_devices": 2})
    op = OpSpec(
        op_id="comm", family="allreduce", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1024,), local_shape=(1024,),
        padded_shape=(1024,), launch_shape=(1024,), link_bytes=4096.0,
        dtype="bf16", layout="contiguous", recipe="r",
    )
    cost = PredictiveCostModel(
        a3, communication_calibration=SpanCalibration(),
        communication_placement="sio",
    ).cost(op)
    assert cost.demand_map()["completion_envelope_ms"] == pytest.approx(0.5)
    # The envelope already contains dispatch + service; adding the 20 us
    # intercept on top would report 0.52 ms for a measured 0.5 ms kernel.
    assert cost.estimated_ms == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# D5/B9 — declared mesh vs switched sharing
# ---------------------------------------------------------------------------


def _link_machine(semantics: str) -> MachineSpec:
    return MachineSpec.from_mapping({
        "name": f"{semantics}-fabric", "num_devices": 4,
        "device": _device(),
        "links": [{
            "name": "fabric", "channel": "d2d_intra_node",
            "semantics": semantics, "capacity_gbps": 100,
            "latency_us": 1.0, "members": [0, 1, 2, 3],
        }],
    })


def test_mesh_scales_by_concurrent_pairs_and_switched_by_replicas():
    parallel = ParallelSpec(dp=4, tp=1)
    mesh = plan_replica_links(_link_machine("mesh"), parallel)
    assert {group["capacity_scale"] for group in mesh} == {1.0}
    assert {group["sharing_rule"] for group in mesh} == {"mesh_pair_wire"}
    assert {group["transport_semantics"] for group in mesh} == {"mesh"}

    switched = plan_replica_links(_link_machine("switched"), parallel)
    assert {group["capacity_scale"] for group in switched} == {0.25}
    assert {group["sharing_rule"] for group in switched} == {"switched_equal_share"}
    assert {group["share_count"] for group in switched} == {4}


# ---------------------------------------------------------------------------
# B7/D6 — identity
# ---------------------------------------------------------------------------


def _observation(**overrides) -> dict:
    payload = {
        "op_family": "gemm", "phase": "operator", "recipe": "r",
        "shape": {"m": 8, "n": 128, "k": 128},
        "duration_ms": 2.0, "analytic_ms": 1.0,
        "source_class": "measured", "source": "toy", "dtype": "bf16",
        "layout": "contiguous", "features": {}, "timing_scope": "device_event_warm",
    }
    payload.update(overrides)
    return payload


def _factor(store: CalibrationStore, machine_key: str):
    return store.factor_for(
        op_family="gemm", phase="operator", recipe="r",
        shape={"m": 8, "n": 128, "k": 128}, machine_key=machine_key,
        dtype="bf16", layout="contiguous", features={},
        timing_scope="device_event_warm",
    )


def test_empty_observation_machine_key_never_matches_and_a3_does_not_leak_to_a2():
    empty_key = CalibrationStore.from_mapping({"observations": [_observation()]})
    result = _factor(empty_key, "Ascend910B4")
    assert result.status == "uncalibrated_analytic"
    assert result.factor == 1.0
    assert result.reason == "no_compatible_measured_observation"

    a3_rows = CalibrationStore.from_mapping({
        "observations": [_observation(machine_key="Ascend910_9362")],
    })
    assert _factor(a3_rows, "Ascend910_9362").status == "calibrated_exact"
    assert _factor(a3_rows, "Ascend910B4").status == "uncalibrated_analytic"

    # A query without identity is refused instead of wildcarding all machines.
    missing = _factor(a3_rows, "")
    assert missing.status == "uncalibrated_machine_key_missing"
    assert missing.reason == "machine_key_required_for_calibration"


# ---------------------------------------------------------------------------
# B11 — roofline and cost share the compute peak
# ---------------------------------------------------------------------------


def test_roofline_and_cost_use_the_same_declared_compute_peak():
    machine = MachineSpec.from_mapping({
        "name": "declared-fp8", "num_devices": 1,
        "device": _device(cube_flops_per_s=1e14),
        "compute_peaks": {"bf16": 1e14, "fp8_e4m3": 4e14},
    })
    op = _gemm_op(1024, 1024, 1024)
    work = IterationWork(
        (op,), IterationPlan(()), 1, 1024, 1024, None, "prefill", "fixture",
    )
    fp8 = PredictiveCostModel(machine, compute_dtype="fp8_e4m3")
    bf16 = PredictiveCostModel(machine, compute_dtype="bf16")
    flops = 2 * 1024 ** 3
    # The roofline honours the declared fp8 peak: 2*1024^3 FLOPs at 4e14
    # flops/s = 5.37e-3 ms, 4x faster than the bf16 declarations.  Reading
    # ``die.cube_flops_per_s`` here (the old behaviour) would make both equal.
    assert fp8.hardware_lower_bound_ms(work) == pytest.approx(
        flops / 4e14 * 1000.0, rel=1e-12,
    )
    assert bf16.hardware_lower_bound_ms(work) == pytest.approx(
        flops / 1e14 * 1000.0, rel=1e-12,
    )
    # cost() uses the same declared peak: identical shape and demands, so the
    # kernel service ratio is exactly the peak ratio (the additive launch term
    # is common to both and is not scaled).
    assert bf16.cost(op).service_ms == pytest.approx(
        4.0 * fp8.cost(op).service_ms, rel=1e-12,
    )


# ---------------------------------------------------------------------------
# D2/B2 — fixture from profile + clock snapshot
# ---------------------------------------------------------------------------


def _a3_environment() -> dict:
    return {
        "device_name": "Ascend910_9362",
        "visible_devices": "0,1",
        "clock_common_before": (
            FIXTURES / "npu-smi-a3-common.txt"
        ).read_text(encoding="utf-8"),
    }


def test_a3_measurement_machine_is_derived_from_profile_and_runtime_clock():
    environment = _a3_environment()
    machine = machine_from_environment(environment)
    profile = resolve_device_capability_profile(
        soc="Ascend910_9362",
        snapshot=None,
    )
    die = machine.dies[0]
    assert machine.hardware_key == "Ascend910_9362"
    assert machine.arch_family == "A3"
    assert machine.dies_per_package == 2
    assert die.aic_count == 20
    assert die.aiv_count == 40
    # Peaks are the declared per-cycle capability at the *runtime* clock, not a
    # literal: 20 AIC x 8192 ops/cycle x 1650 MHz.
    runtime_mhz = 1650.0
    assert die.cube_flops_per_s == pytest.approx(
        profile.cube_spec.ops_per_cycle_per_die("bf16") * runtime_mhz * 1e6,
    )
    assert die.vector_ops_per_s > 0
    assert die.hbm_bw_bytes_per_s == pytest.approx(1.2e12)
    assert die.l2_bytes == int(
        round(float(profile.memory_hierarchy_spec.l2_capacity_mib) * 1024 ** 2)
    )
    # No A3 literal: the links and their rates come from the declared channels.
    assert {link.channel for link in machine.links} == {
        "d2d_intra_package", "d2d_intra_node",
    }
    assert machine.link_for_scope("hccs").protocol == "HCCS_SW"
    metadata = _metadata(machine)
    assert float(metadata["model_clock_mhz"]) == runtime_mhz
    assert float(metadata["runtime_clock_mhz"]) == runtime_mhz
    assert float(metadata["configured_clock_mhz"]) == 1650.0
    assert metadata["clock_source"] == "npu_smi_runtime_aicore_freq"
    assert metadata["vector_capability_source"].startswith(
        "legacy_measurement_nominal"
    )


def test_a2_measurement_machine_preserves_configured_vs_runtime_discrepancy():
    environment = {
        "device_name": "Ascend910B4",
        "visible_devices": "0,1,2,3,4,5,6,7",
        "clock_common_before": (
            FIXTURES / "npu-smi-a2-common.txt"
        ).read_text(encoding="utf-8"),
    }
    machine = machine_from_environment(environment)
    assert machine.hardware_key == "Ascend910B4"
    assert machine.arch_family == "A2"
    assert machine.dies_per_package == 1
    assert len(machine.dies) == 8
    assert machine.dies[0].aic_count == 20
    assert machine.dies[0].cube_flops_per_s == pytest.approx(
        20 * 8192 * 1650e6,
    )
    metadata = _metadata(machine)
    # The .ini says 1500 MHz; the npu-smi rated clock under load is 1650.  Both
    # stay readable; the model uses the runtime value and records the source.
    assert float(metadata["configured_clock_mhz"]) == 1500.0
    assert float(metadata["runtime_clock_mhz"]) == 1650.0
    assert float(metadata["model_clock_mhz"]) == 1650.0
    assert metadata["clock_source"] == "npu_smi_runtime_aicore_freq"
    assert any(
        item["field"] == "configured_clock_mhz"
        for item in metadata["discrepancies"]
    )
    # A2 is a single-die package: HCCS mesh only, no fabricated SIO wire.
    assert [link.channel for link in machine.links] == ["d2d_intra_node"]
    assert machine.links[0].semantics == "mesh"
    assert machine.links[0].capacity_bytes_per_ms == pytest.approx(168e6)
    assert machine.links[0].latency_us == pytest.approx(2.5)


def test_machine_from_environment_requires_identity_and_labels_configured_clock():
    with pytest.raises(A3InputError, match="machine_identity_missing"):
        machine_from_environment({"visible_devices": "0"})
    machine = machine_from_environment({"device_name": "Ascend910_9362"})
    metadata = _metadata(machine)
    assert metadata["runtime_clock_mhz"] is None
    assert metadata["clock_source"] == "configured_platform_ini_or_sku_catalog"
