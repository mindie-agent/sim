"""Tier 0 tests for the replayed-graph launch charge.

A replayed bucket submits its kernels once, so it pays a declared device
dispatch per kernel, not the per-op host intercept; eager decode and prefill keep
the host-dispatch charge they measurably pay.  Fixtures are in-memory synthetic
ops/cost model, and every expected number is derived from the module's own
declared constants -- no recorded E2E outcome is encoded here.
"""
from __future__ import annotations

import pytest

from mindie_sim.a3_cost import (
    GRAPH_KERNEL_LAUNCH_US,
    GRAPH_REPLAY_COMPOSITION,
    PHYSICAL_CALIBRATED,
    PredictiveCostModel,
)
from mindie_sim.a3_ops import IterationWork, OpSpec
from mindie_sim.a3_types import MachineSpec
from mindie_sim.a3_physical_calibration import PhysicalCalibration
from mindie_sim.a3_workload import IterationPlan

AIC = 20
HOST_LAUNCH_US = 20.0
PEAK_BW = 1.2e12
L2_BYTES = 201326592
GEMM_LAUNCH_MS = 0.0206
PER_CALL = "device_event_warm_operator"


def machine() -> MachineSpec:
    return MachineSpec.from_mapping({
        "name": "Synthetic910_X1",
        "hardware_key": "Synthetic910_X1",
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": 2,
        "device": {
            "hbm_gib": 64,
            "cube_flops_per_s": AIC * 8192 * 1.65e9,
            "vector_ops_per_s": 8.448e12,
            "hbm_bw_bytes_per_s": PEAK_BW,
            "mte_bw_bytes_per_s": 2.4e12,
            "l2_bytes": L2_BYTES,
            "launch_us": HOST_LAUNCH_US,
            "aic_count": AIC,
            "aiv_count": 40,
        },
    })


ARTIFACT_MACHINE = {
    "hardware_key": "Synthetic910_X1",
    "arch_family": "A3",
    "aic_count": AIC,
    "aiv_count": 40,
    "clock_mhz": 1650.0,
    "ops_per_cycle": 8192.0,
    "peak_flops": AIC * 8192 * 1.65e9,
    "peak_bw_bytes_per_s": PEAK_BW,
    "l2_bytes": L2_BYTES,
    "vec_peak_ops_per_s": 8.448e12,
}


def gemm_op(m: int = 4096, n: int = 4096, k: int = 4096) -> OpSpec:
    return OpSpec(
        op_id=f"gemm.{m}x{n}x{k}", family="gemm", phase="decode", stage=0, layer=0,
        requests=("r0",), semantic_shape=(m, n, k), local_shape=(m, n, k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=2.0 * m * n * k, hbm_bytes=(m * k + k * n + m * n) * 2.0,
        dtype="bf16", recipe="torch_mm_preallocated",
    )


def vector_op(elements: int = 1 << 20) -> OpSpec:
    return OpSpec(
        op_id=f"add.{elements}", family="add", phase="decode", stage=0, layer=0,
        requests=("r0",), semantic_shape=(elements,), local_shape=(elements,),
        padded_shape=(elements,), launch_shape=(elements,),
        vector_ops=float(elements), hbm_bytes=float(elements) * 2.0,
        dtype="bf16", recipe="elementwise_add_reference",
    )


def grouped_op(experts: int) -> OpSpec:
    return OpSpec(
        op_id=f"gate_up.{experts}", family="moe_gate_up", phase="decode", stage=0,
        layer=0, requests=("r0",), semantic_shape=(2048, experts, 4096, 7168),
        local_shape=(2048, experts, 4096, 7168),
        padded_shape=(2048, experts, 4096, 7168),
        launch_shape=(2048, experts, 4096, 7168),
        flops=2.0 * 2048 * 4096 * 7168,
        hbm_bytes=(experts * 4096 * 7168 + 2048 * 7168 + 2048 * 4096) * 2.0,
        dtype="bf16", recipe="npu_grouped_matmul_list_bf16",
        metadata=(("group_size", experts),),
    )


def work(ops, *, mode: str, graph_bucket: int | None) -> IterationWork:
    return IterationWork(tuple(ops), IterationPlan(()), 1, 1, 1, graph_bucket,
                         mode, "fixture")


def cost_model(**kwargs) -> PredictiveCostModel:
    return PredictiveCostModel(machine(), **kwargs)


def artifact() -> PhysicalCalibration:
    """A fitted gemm row whose host intercept is what a replayed bucket drops."""
    row = {
        "family": "gemm", "recipe": "*", "protocol_signature": "",
        "timing_scope": PER_CALL, "model_kind": "gemm", "model_form": "flat_additive",
        "fit_level": "family_protocol", "block_m": None, "block_n": None,
        "launch_ms": GEMM_LAUNCH_MS, "eta_compute": 0.9, "eta_memory": 0.9,
        "vec_ops_per_s": None, "memory_model": "hbm", "l2_usable_bytes": 0,
        "l2_bw_bytes_per_s": None, "compute_dtype": "bf16", "samples": 40,
        "raw_event_count": 240, "train_wape_pct": 5.0, "cv_wape_pct": 6.0,
        "median_ape_pct": 5.0, "p90_ape_pct": 9.0, "max_ape_pct": 12.0,
        "binding_share": {"compute": 0.5, "memory": 0.5},
        "binding_counts": {"compute": 20, "memory": 20},
        "identified": {"launch": True}, "pinned_parameters": [], "intervals": {},
        "breakdown": {}, "model_selection": {}, "memory_model_selection": {},
        "quality": "trusted", "warnings": [],
    }
    return PhysicalCalibration.from_mapping({
        "schema_version": "mindie-sim.physical-calibration/1",
        "hardware_key": "Synthetic910_X1",
        "arch_family": "A3",
        "machine": ARTIFACT_MACHINE,
        "fit_config": {"fit_policy": "synthetic_graph_replay_fixture"},
        "families": {f"gemm/{PER_CALL}": row},
        "provenance": {
            "generator": "tests.test_a3_graph_replay_launch",
            "condition_count": 40, "raw_event_count": 240,
            "fitted_utc": "2026-10-21T00:00:00+00:00",
            "training_hash": "sha256:synthetic-graph-replay-fixture",
            "heldout_guard": {"status": "clean"},
        },
        "unmodeled": [],
    })


DEVICE_LAUNCH_MS = GRAPH_KERNEL_LAUNCH_US / 1000.0


def test_a_replayed_decode_bucket_pays_the_device_dispatch_per_kernel():
    ops = (gemm_op(), gemm_op(1, 4096, 4096), vector_op())
    costed = cost_model().cost_iteration(work(ops, mode="decode", graph_bucket=1))
    assert len(costed) == 3
    for cost in costed:
        assert cost.launch_ms == pytest.approx(DEVICE_LAUNCH_MS, rel=1e-12)
        assert cost.launch_detail["composition"] == GRAPH_REPLAY_COMPOSITION
        assert cost.launch_detail["per_launch_ms"] == pytest.approx(DEVICE_LAUNCH_MS)
        assert cost.launch_detail["reason"] == "graph_replay_device_dispatch"
        # The intercept the bucket no longer pays per kernel is reported, not
        # silently dropped: it is the host submit the eager path still charges.
        assert cost.launch_detail["host_intercept_ms"] == pytest.approx(
            HOST_LAUNCH_US / 1000.0, rel=1e-9
        )
        # Only the launch term changed: the service part still adds up.
        assert cost.estimated_ms == pytest.approx(
            cost.service_ms + DEVICE_LAUNCH_MS, rel=1e-9
        )


def test_eager_decode_keeps_the_serial_host_intercept():
    ops = (gemm_op(), gemm_op(1, 4096, 4096), vector_op())
    costed = cost_model().cost_iteration(work(ops, mode="decode", graph_bucket=None))
    for cost in costed:
        assert cost.launch_ms == pytest.approx(HOST_LAUNCH_US / 1000.0, rel=1e-12)
        assert cost.launch_detail["composition"] == "serial"
        assert cost.estimated_ms == pytest.approx(
            cost.service_ms + cost.launch_ms, rel=1e-9
        )
    # A replayed bucket differs from eager by exactly the per-kernel intercept
    # difference: nothing else in the two costs is touched.
    graph = cost_model().cost_iteration(work(ops, mode="decode", graph_bucket=1))
    for eager_cost, graph_cost in zip(costed, graph):
        assert eager_cost.service_ms == pytest.approx(graph_cost.service_ms, rel=1e-12)
        assert eager_cost.estimated_ms - graph_cost.estimated_ms == pytest.approx(
            HOST_LAUNCH_US / 1000.0 - DEVICE_LAUNCH_MS, rel=1e-9
        )


def test_prefill_keeps_the_host_intercept_and_its_concurrent_composition():
    ops = (gemm_op(), vector_op())
    costed = cost_model().cost_iteration(work(ops, mode="prefill", graph_bucket=None))
    for cost in costed:
        assert cost.launch_ms == pytest.approx(HOST_LAUNCH_US / 1000.0, rel=1e-12)
        assert cost.launch_detail is None or cost.launch_detail.get("composition") != (
            GRAPH_REPLAY_COMPOSITION
        )
        assert cost.estimated_ms == pytest.approx(
            cost.service_ms + cost.launch_ms, rel=1e-9
        )


def test_a_grouped_kernel_pays_the_device_dispatch_per_tensor_list_launch():
    # 384 experts is three sequential npu_grouped_matmul launches (128 per list);
    # the device dispatch repeats per launch, never once per op.
    cost = cost_model().cost_iteration(
        work((grouped_op(384),), mode="decode", graph_bucket=1)
    )[0]
    assert cost.launch_detail["launches"] == 3
    assert cost.launch_ms == pytest.approx(3 * DEVICE_LAUNCH_MS, rel=1e-12)
    assert cost.launch_detail["per_launch_ms"] == pytest.approx(DEVICE_LAUNCH_MS)
    assert cost.launch_detail["host_intercept_ms"] == pytest.approx(
        3 * HOST_LAUNCH_US / 1000.0, rel=1e-9
    )


def mc2_machine() -> MachineSpec:
    """Two-fabric synthetic part: an MC2 chunk needs a real comm lane to fuse."""
    return MachineSpec.from_mapping({
        "name": "synthetic-mc2-8-die",
        "hardware_key": "Synthetic910_MC2",
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": 8,
        "ranks_per_node": 8,
        "device": {
            "hbm_gib": 64,
            "cube_flops_per_s": AIC * 8192 * 1.65e9,
            "vector_ops_per_s": 8.448e12,
            "hbm_bw_bytes_per_s": PEAK_BW,
            "mte_bw_bytes_per_s": 2.0 * PEAK_BW,
            "launch_us": HOST_LAUNCH_US,
            "aic_count": AIC,
            "aiv_count": 40,
        },
        "compute_peaks": {"bf16": AIC * 8192 * 1.65e9},
        "links": [{
            "name": "d2d_intra", "channel": "d2d_intra_node",
            "capacity_gbps": 196.0, "latency_us": 2.5,
            "members": list(range(8)),
        }],
    })


def mc2_chunk(experts: int = 4) -> tuple[OpSpec, ...]:
    """One MC2 chunk as the lowering emits it: dispatch, grouped GEMMs, combine."""
    tokens, top_k, hidden, width, ep_size = 4096, 8, 128, 256, 2
    rows = tokens // experts
    payload = tokens * top_k * hidden * 2.0
    link = payload * (ep_size - 1) / ep_size
    chunk_meta = (("mc2_chunk", 0), ("mc2_chunks", 1), ("experts", tuple(range(experts))))
    dispatch = OpSpec(
        op_id="s0.l0.mc2_dispatch", family="ep_dispatch", phase="decode", stage=0,
        layer=0, requests=("r0",), semantic_shape=(rows, top_k, hidden),
        local_shape=(rows, top_k, hidden), padded_shape=(rows, top_k, hidden),
        launch_shape=(rows, top_k, hidden), link_bytes=link,
        recipe="mc2_dispatch", metadata=chunk_meta,
    )
    gate_up = OpSpec(
        op_id="s0.l0.moe_gate_up_e0", family="moe_gate_up", phase="decode", stage=0,
        layer=0, requests=("r0",), semantic_shape=(rows, 2 * width, hidden),
        local_shape=(rows, 2 * width, hidden), padded_shape=(rows, 2 * width, hidden),
        launch_shape=(rows, 2 * width, hidden),
        flops=2.0 * rows * 2 * width * hidden,
        hbm_bytes=(2 * width * hidden + rows * hidden) * 2.0,
        dtype="bf16", recipe="moe_mc2", deps=(dispatch.op_id,),
        metadata=(("expert", 0), ("expert_tokens", rows), ("mc2_chunk", 0)),
    )
    down = OpSpec(
        op_id="s0.l0.moe_down_e0", family="moe_down", phase="decode", stage=0,
        layer=0, requests=("r0",), semantic_shape=(rows, hidden, width),
        local_shape=(rows, hidden, width), padded_shape=(rows, hidden, width),
        launch_shape=(rows, hidden, width),
        flops=2.0 * rows * hidden * width,
        hbm_bytes=(hidden * width + rows * width) * 2.0,
        dtype="bf16", recipe="moe_mc2", deps=(gate_up.op_id,),
        metadata=(("expert", 0), ("expert_tokens", rows), ("mc2_chunk", 0)),
    )
    combine = OpSpec(
        op_id="s0.l0.mc2_combine", family="ep_combine", phase="decode", stage=0,
        layer=0, requests=("r0",), semantic_shape=(rows, hidden),
        local_shape=(rows, hidden), padded_shape=(rows, hidden),
        launch_shape=(rows, hidden), link_bytes=link, recipe="mc2_combine",
        deps=(down.op_id,),
    )
    return (dispatch, gate_up, down, combine)


def mc2_cost_model() -> PredictiveCostModel:
    from mindie_sim.a3_communication import CommCondition, CommunicationCalibration

    small, large = 1_000_000, 5_000_000
    calibration = CommunicationCalibration([
        CommCondition("hccssw", "p2p_only", size,
                      0.03 + 5e-8 * size, 8, hardware_key="Synthetic910_MC2",
                      ranks=2, dtype="bf16")
        for size in (small, large)
    ])
    return PredictiveCostModel(
        mc2_machine(), compute_dtype="bf16", timing_scope=PER_CALL,
        communication_calibration=calibration,
    )


def test_a_replayed_mc2_chunk_pays_the_device_dispatch_too():
    """The fused kernel is launched by the graph, so it is not exempt.

    Its grouped matmul is still one tensor-list launch per
    ``GROUPED_LAUNCH_EXPERT_CAP`` experts: the device dispatch repeats per launch.
    Eager keeps the host intercept on the same chunk.
    """
    ops = mc2_chunk(experts=4)
    replayed = mc2_cost_model().cost_iteration(work(ops, mode="decode", graph_bucket=1))
    eager = mc2_cost_model().cost_iteration(work(ops, mode="decode", graph_bucket=None))
    graph_record = next(item for item in replayed if item.mc2 is not None)
    eager_record = next(item for item in eager if item.mc2 is not None)
    assert graph_record.mc2["bounds"]["launch_composition"] == GRAPH_REPLAY_COMPOSITION
    assert graph_record.mc2["bounds"]["launch_source"] == "device_dispatch"
    assert graph_record.mc2["bounds"]["per_launch_ms"] == pytest.approx(DEVICE_LAUNCH_MS)
    assert graph_record.mc2["bounds"]["launch_ms"] == pytest.approx(
        graph_record.mc2["bounds"]["launch_count"] * DEVICE_LAUNCH_MS
    )
    assert eager_record.mc2["bounds"]["launch_source"] == "host_dispatch"
    assert eager_record.mc2["bounds"]["per_launch_ms"] == pytest.approx(
        HOST_LAUNCH_US / 1000.0
    )
    # The fused kernel stays cheaper than its own op decomposition either way.
    assert sum(item.estimated_ms for item in replayed) < sum(
        item.estimated_ms for item in eager
    )


def test_the_device_dispatch_replaces_the_fitted_intercept_not_the_service():
    """The fitted duration's service survives; only its launch term is re-priced."""
    fitted = artifact()
    model = cost_model(physical_calibration=fitted)
    op = gemm_op()
    explained = fitted.predict_op_explained(
        {
            "family": "gemm", "recipe": "torch_mm_preallocated",
            "timing_scope": PER_CALL, "shape": {"m": 4096, "n": 4096, "k": 4096},
            "flops": op.flops, "hbm_bytes": op.hbm_bytes,
        },
        {"hardware_key": "Synthetic910_X1"}, compute_dtype="bf16",
    )
    fitted_service = explained.duration_ms - GEMM_LAUNCH_MS
    replayed = model.cost_iteration(work((op,), mode="decode", graph_bucket=1))[0]
    assert replayed.calibration.status == PHYSICAL_CALIBRATED
    assert replayed.launch_ms == pytest.approx(DEVICE_LAUNCH_MS, rel=1e-12)
    assert replayed.service_ms == pytest.approx(fitted_service, rel=1e-9)
    assert replayed.estimated_ms == pytest.approx(
        fitted_service + DEVICE_LAUNCH_MS, rel=1e-9
    )
    assert replayed.launch_detail["host_intercept_ms"] == pytest.approx(
        GEMM_LAUNCH_MS, rel=1e-12
    )
    # Eager keeps the fitted intercept as the fitted duration, unchanged.
    eager = model.cost_iteration(work((op,), mode="decode", graph_bucket=None))[0]
    assert eager.estimated_ms == pytest.approx(explained.duration_ms, rel=1e-9)
    assert eager.launch_ms == pytest.approx(GEMM_LAUNCH_MS, rel=1e-9)
