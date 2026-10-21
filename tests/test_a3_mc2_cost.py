"""Tier 0 tests for the MC2 fused-kernel cost model.

MC2 runs dispatch, the grouped expert GEMMs and combine inside one kernel, so its
duration is ``launch + max(compute, comm)`` rather than the sum of the op
decomposition the graph has to keep for scheduling.  These tests pin the model's
*contract*, not a hardware outcome:

* the byte formula ``tokens*top_k*hidden*dtype_bytes`` with the ``(ep-1)/ep``
  group factor, and where each of those inputs came from;
* ``alpha``/``beta`` from the measured communication lane when one exists;
* both bounds reported, an unverified overlap labelled rather than averaged;
* the chunk's members summing to the fused duration (no double count) -- checked
  both arithmetically and through the actual resource engine;
* the dense-GEMM transfer is never applied on top of an MC2 chunk.

The machine is a declared synthetic part (two fabrics, a known 20-AIC cube); no
network, device, sleep or measurement file is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim.a3_communication import CommCondition, CommunicationCalibration
from mindie_sim.a3_cost import (
    MC2_LANE_RANKS, MC2_NO_COMMUNICATION, MC2_OVERLAP_CALIBRATED,
    MC2_OVERLAP_UNVERIFIED, MC2_RECIPES, PHYSICAL_UNMODELED,
    PredictiveCostModel, is_mc2_op, mc2_chunk_key, mc2_comm_bytes,
    mc2_time_bounds,
)
from mindie_sim.a3_framework import predict
from mindie_sim.a3_ops import OpSpec
from mindie_sim.a3_physical_calibration import (
    SCHEMA_VERSION, PhysicalCalibration,
)
from mindie_sim.a3_resources import (
    ResourceEngine, add_costed_ops, build_resource_specs,
)

ROOT = Path(__file__).resolve().parents[1]
PER_CALL = "device_event_warm"
#: Synthetic part: 20 AIC at 1650 MHz, one HCCS switch fabric and one RoCE
#: fabric.  The values are declared, not measured; only their ratios matter here.
AIC = 20
CUBE_PEAK = AIC * 8192 * 1.65e9
VEC_PEAK = 40 * 128 * 1.65e9
HBM_PEAK = 1.2e12
HCCS_BYTES_PER_MS = 196e6
ROCE_BYTES_PER_MS = 50e6
LAUNCH_US = 20.0
HCCS_LATENCY_US = 2.5
ROCE_LATENCY_US = 5.0
BF16 = 2.0
#: Measured-lane fixture: a two-point affine fit with alpha=0.03 ms, beta=5e-8
#: ms/byte (the calibration's own fit, not a model-side assumption).
ALPHA_MS = 0.03
BETA_MS_PER_BYTE = 5e-8


def machine():
    from mindie_sim.a3_types import MachineSpec

    return MachineSpec.from_mapping({
        "name": "synthetic-mc2-16-die",
        "hardware_key": "Synthetic910_MC2",
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": 16,
        "ranks_per_node": 8,
        "device": {
            "hbm_gib": 64,
            "cube_flops_per_s": CUBE_PEAK,
            "vector_ops_per_s": VEC_PEAK,
            "hbm_bw_bytes_per_s": HBM_PEAK,
            "mte_bw_bytes_per_s": 2.0 * HBM_PEAK,
            "launch_us": LAUNCH_US,
            "aic_count": AIC,
            "aiv_count": 40,
        },
        "compute_peaks": {"bf16": CUBE_PEAK},
        "links": [
            {
                "name": "d2d_intra", "channel": "d2d_intra_node",
                "capacity_gbps": HCCS_BYTES_PER_MS / 1e6,
                "latency_us": HCCS_LATENCY_US,
                "members": list(range(8)),
            },
            {
                "name": "d2d_inter", "channel": "d2d_inter_node",
                "capacity_gbps": ROCE_BYTES_PER_MS / 1e6,
                "latency_us": ROCE_LATENCY_US,
                "members": list(range(16)),
            },
        ],
    })


def comm_calibration(*, hardware_key: str = "Synthetic910_MC2") -> CommunicationCalibration:
    """Two measured lane points -> alpha=0.03 ms, beta=5e-8 ms/byte."""
    small, large = 1_000_000, 5_000_000
    return CommunicationCalibration([
        CommCondition(
            "hccssw", "p2p_only", small,
            ALPHA_MS + BETA_MS_PER_BYTE * small, 8,
            hardware_key=hardware_key, ranks=2, dtype="bf16",
        ),
        CommCondition(
            "hccssw", "p2p_only", large,
            ALPHA_MS + BETA_MS_PER_BYTE * large, 8,
            hardware_key=hardware_key, ranks=2, dtype="bf16",
        ),
    ])


def model(*, with_comm: bool = False, artifact=None, **kwargs) -> PredictiveCostModel:
    return PredictiveCostModel(
        machine(), compute_dtype="bf16", timing_scope=PER_CALL,
        physical_calibration=artifact,
        communication_calibration=comm_calibration() if with_comm else None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# One MC2 chunk as the lowering emits it: barrier (layer token carrier), dispatch,
# the grouped expert GEMMs, combine.
# ---------------------------------------------------------------------------
def chunk_layer(
    *,
    tokens: int = 4096, top_k: int = 8, hidden: int = 128, experts: int = 4,
    chunks: int = 1, chunk: int = 0, ep_size: int = 2,
    width: int = 256, recipe: str = "moe_mc2",
    declared_link_bytes: float | None = None,
    metadata: tuple[tuple[str, object], ...] = (),
):
    """The ops of one MC2 chunk plus the barrier that declares the batch size.

    Every expert has ``tokens // experts`` routed rows in this chunk, so the
    dispatch payload the lowering declares is exactly
    ``tokens*top_k*hidden*BF16*(ep-1)/ep/chunks``.  The MC2 members keep the
    dispatch -> GEMM -> combine dependencies the lowering emits; the barrier is a
    layer-level op (it is only the layer's ``(tokens, hidden)`` declaration here).
    """
    rows = tokens // experts
    payload = tokens * top_k * hidden * BF16 / chunks
    link = payload * (ep_size - 1) / ep_size
    if declared_link_bytes is not None:
        link = declared_link_bytes
    chunk_metadata = (
        ("mc2_chunk", chunk), ("mc2_chunks", chunks),
        # The fused kernel's grouped matmul covers every local expert of the
        # chunk; the expert count is what the tensor-list launch cap applies to.
        ("experts", tuple(range(experts))),
    ) + metadata
    barrier = OpSpec(
        op_id=f"s0.l0.mc2_barrier{chunk}", family="barrier", phase="prefill",
        stage=0, layer=0, requests=("r",),
        semantic_shape=(tokens, hidden), local_shape=(tokens, hidden),
        padded_shape=(tokens, hidden), launch_shape=(tokens, hidden),
        recipe="mc2_barrier",
    )
    dispatch = OpSpec(
        op_id=f"s0.l0.mc2_dispatch{chunk}", family="ep_dispatch",
        phase="prefill", stage=0, layer=0, requests=("r",),
        semantic_shape=(rows, top_k, hidden), local_shape=(rows, top_k, hidden),
        padded_shape=(rows, top_k, hidden), launch_shape=(rows, top_k, hidden),
        link_bytes=link, recipe="mc2_dispatch", metadata=chunk_metadata,
        deps=(),
    )
    gate_up = OpSpec(
        op_id=f"s0.l0.moe_gate_up_e0_{chunk}", family="moe_gate_up",
        phase="prefill", stage=0, layer=0, requests=("r",),
        semantic_shape=(rows, 2 * width, hidden),
        local_shape=(rows, 2 * width, hidden),
        padded_shape=(rows, 2 * width, hidden),
        launch_shape=(rows, 2 * width, hidden),
        flops=2.0 * rows * 2 * width * hidden,
        hbm_bytes=(2 * width * hidden + rows * hidden) * BF16,
        dtype="bf16", recipe=recipe, deps=(dispatch.op_id,),
        metadata=(("expert", 0), ("expert_tokens", rows), ("mc2_chunk", chunk)),
    )
    down = OpSpec(
        op_id=f"s0.l0.moe_down_e0_{chunk}", family="moe_down",
        phase="prefill", stage=0, layer=0, requests=("r",),
        semantic_shape=(rows, hidden, width), local_shape=(rows, hidden, width),
        padded_shape=(rows, hidden, width), launch_shape=(rows, hidden, width),
        flops=2.0 * rows * hidden * width,
        hbm_bytes=(hidden * width + rows * width) * BF16,
        dtype="bf16", recipe=recipe, deps=(gate_up.op_id,),
        metadata=(("expert", 0), ("expert_tokens", rows), ("mc2_chunk", chunk)),
    )
    combine = OpSpec(
        op_id=f"s0.l0.mc2_combine{chunk}", family="ep_combine", phase="prefill",
        stage=0, layer=0, requests=("r",),
        semantic_shape=(rows, hidden), local_shape=(rows, hidden),
        padded_shape=(rows, hidden), launch_shape=(rows, hidden),
        link_bytes=link, recipe="mc2_combine", metadata=chunk_metadata,
        deps=(down.op_id,),
    )
    return [barrier, dispatch, gate_up, down, combine]


def grouped_op(
    *,
    tokens: int = 2048, experts: int = 8, n: int = 7168, k: int = 7168,
    family: str = "moe_gate_up", recipe: str = "catalog_npu_grouped_matmul",
    group_size: int | None = None,
) -> OpSpec:
    """One grouped expert GEMM with its routed geometry declared."""
    declared_group = experts if group_size is None else group_size
    return OpSpec(
        op_id=f"{family}.{tokens}x{experts}x{n}x{k}", family=family,
        phase="prefill", stage=0, layer=0, requests=("r",),
        semantic_shape=(tokens, n, k), local_shape=(tokens, n, k),
        padded_shape=(tokens, n, k), launch_shape=(tokens, n, k),
        flops=2.0 * tokens * n * k,
        hbm_bytes=(experts * n * k + tokens * k + tokens * n) * BF16,
        dtype="bf16", recipe=recipe,
        metadata=(("expert_tokens", tokens), ("group_size", declared_group)),
    )


class Work:
    """Minimal ``IterationWork`` shape the cost model reads."""

    def __init__(self, ops, graph_bucket=None):
        self.ops = tuple(ops)
        # A non-None bucket means the workload replayed this iteration, which
        # selects the serial launch composition.
        self.graph_bucket = graph_bucket


def compose(ops, cost_model: PredictiveCostModel):
    return cost_model.cost_iteration(Work(ops))


def mc2_records(costed):
    return [item for item in costed if item.mc2 is not None]


def chunk_estimate(ops, cost_model: PredictiveCostModel) -> dict:
    costed = compose(ops, cost_model)
    records = mc2_records(costed)
    assert records, "the chunk must be recognised as MC2"
    return records[0].mc2


# ---------------------------------------------------------------------------
# 1. The byte formula and its declared inputs
# ---------------------------------------------------------------------------
def test_mc2_comm_bytes_is_the_routed_volume_times_the_ep_group_factor():
    payload, link = mc2_comm_bytes(
        tokens=4096, top_k=8, hidden=7168, dtype_bytes_value=BF16, ep_size=1,
    )
    assert payload == 4096 * 8 * 7168 * BF16
    # An EP group of one has no peer traffic.
    assert link == 0.0
    _, half = mc2_comm_bytes(
        tokens=4096, top_k=8, hidden=7168, dtype_bytes_value=BF16, ep_size=2,
    )
    assert half == pytest.approx(payload * 1 / 2, rel=1e-12)
    _, group = mc2_comm_bytes(
        tokens=4096, top_k=8, hidden=7168, dtype_bytes_value=BF16, ep_size=16,
    )
    assert group == pytest.approx(payload * 15 / 16, rel=1e-12)


def test_mc2_geometry_recovers_every_input_and_says_where_it_came_from():
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    geometry = model().mc2_geometry(ops, layer_ops=ops)
    assert geometry["bytes_formula"] == "tokens*top_k*hidden*dtype_bytes"
    # tokens: the layer's (tokens, hidden) carrier op.
    assert geometry["tokens"] == 4096 and geometry["tokens_source"] == "layer_token_carrier"
    # top_k/hidden: the dispatch op's declared routed shape.
    assert geometry["top_k"] == 8 and geometry["top_k_source"] == "dispatch_shape"
    assert geometry["hidden"] == 128 and geometry["hidden_source"] == "dispatch_shape"
    # The EP group is inverted from the routed volume the graph already declares.
    assert geometry["ep_size"] == 2
    assert geometry["ep_source"] == "inverted_from_declared_link_bytes"
    assert geometry["ep_group_factor"] == 0.5
    assert geometry["payload_bytes"] == 4096 * 8 * 128 * BF16
    assert geometry["payload_bytes_per_chunk"] == geometry["payload_bytes"]
    assert geometry["link_bytes"] == pytest.approx(geometry["payload_bytes"] / 2)
    assert geometry["link_bytes_source"] == "formula"
    # The formula and the graph's own declared volume agree exactly.
    assert geometry["declared_over_formula"] == pytest.approx(1.0, rel=1e-12)
    assert geometry["link_scope"] == "d2d_intra_node"
    assert geometry["link_protocol"] == "HCCS_SW"
    assert geometry["link_note"] == ""


def test_mc2_geometry_separates_the_chunks_and_reads_declared_metadata():
    declared = (
        ("mc2_tokens", 4096), ("mc2_top_k", 8), ("mc2_hidden", 128),
        ("mc2_ep_size", 3), ("mc2_chunks", 2),
    )
    ops = chunk_layer(
        tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2, chunks=2,
        declared_link_bytes=1.0, metadata=declared,
    )
    geometry = model().mc2_geometry(ops, layer_ops=ops)
    assert geometry["tokens_source"] == "declared_metadata"
    assert geometry["ep_size"] == 3 and geometry["ep_source"] == "declared"
    # The formula wins; the graph's declared volume is reported next to it.
    assert geometry["link_bytes"] == pytest.approx(
        4096 * 8 * 128 * BF16 / 2 * 2 / 3, rel=1e-12,
    )
    assert geometry["declared_link_bytes"] == 1.0
    assert geometry["declared_over_formula"] < 1e-6


def test_mc2_picks_the_inter_node_fabric_when_the_group_spans_nodes():
    # EP=16 on an 8-ranks-per-node part leaves the node: the declared RoCE fabric
    # is priced, not the intra-node switch.
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=16)
    geometry = model().mc2_geometry(ops, layer_ops=ops)
    assert geometry["ep_size"] == 16
    assert geometry["link_scope"] == "d2d_inter_node"
    assert geometry["link_protocol"].startswith("RoCE")
    # EP=2 is an intra-node group.
    intra = model().mc2_geometry(
        chunk_layer(ep_size=2), layer_ops=chunk_layer(ep_size=2),
    )
    assert intra["link_scope"] == "d2d_intra_node"


# ---------------------------------------------------------------------------
# 2. Bounds: serial, overlapped, and an explicit overlap parameter
# ---------------------------------------------------------------------------
def test_mc2_bounds_follow_the_measured_chain_contract():
    # The campaign measured the fused chain at 1.0-1.2x the sum of its legs, so the
    # legs bound is the default estimate and the full-overlap max is refuted.
    legs = mc2_time_bounds(launch_ms=0.02, compute_ms=0.05, comm_ms=0.2)
    assert legs["estimated_ms"] == pytest.approx(0.02 + 0.05 + 0.2)
    assert legs["legs_ms"] == pytest.approx(legs["estimated_ms"])
    assert legs["serial_ms"] == pytest.approx(legs["legs_ms"])
    assert legs["chain_factor"] == 1.0
    assert legs["upper_ms"] == pytest.approx(0.02 + (0.05 + 0.2) * 1.2)
    assert legs["max_bound_ms"] == pytest.approx(0.02 + 0.2)
    assert legs["max_bound_refuted_by_measurement"] is True
    # The chain factor is monotone and bounded by the measured range: it can never
    # claim an overlap (factor < 1) nor a penalty beyond the campaign ceiling.
    factors = [0.0, 0.5, 0.9, 1.0, 1.1, 1.2, 1.5, 3.0]
    estimates = [
        mc2_time_bounds(
            launch_ms=0.02, compute_ms=0.05, comm_ms=0.2, chain_factor=f,
        )["estimated_ms"]
        for f in factors
    ]
    assert estimates == sorted(estimates)
    assert estimates[0] == pytest.approx(0.02 + (0.05 + 0.2) * 1.0)
    assert estimates[-1] == pytest.approx(0.02 + (0.05 + 0.2) * 1.2)
    # A factor above the measured ceiling is clamped, not extrapolated.
    assert mc2_time_bounds(
        launch_ms=0.02, compute_ms=0.05, comm_ms=0.2, chain_factor=5.0,
    )["chain_factor"] == 1.2
    # Degenerate inputs collapse the bounds.
    flat = mc2_time_bounds(launch_ms=0.02, compute_ms=0.0, comm_ms=0.2)
    assert flat["estimated_ms"] == pytest.approx(flat["legs_ms"])


def test_mc2_chunk_reports_both_bounds_and_labels_the_unverified_overlap():
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    estimate = chunk_estimate(ops, model())
    assert estimate["status"] == MC2_OVERLAP_UNVERIFIED
    assert estimate["wiring_status"] == MC2_OVERLAP_UNVERIFIED
    assert estimate["reason"].startswith("chain_unverified:missing=")
    assert "measured_comm_lane" in estimate["reason"]
    assert "fitted_chain_factor" in estimate["reason"]
    assert "1.0-1.2x_the_leg_sum" in estimate["reason"]
    bounds = estimate["bounds"]
    assert bounds["upper_ms"] > bounds["estimated_ms"] > bounds["launch_ms"]
    assert bounds["max_bound_ms"] < bounds["legs_ms"]
    # The declared default is the legs sum: the measured A3 fused chain is
    # 1.0-1.2x the legs and never the full-overlap ``max``.
    assert bounds["estimated_ms"] == pytest.approx(bounds["legs_ms"])
    assert bounds["primary"] == "legs"
    assert bounds["launch_count"] == 1
    assert estimate["chain"] == {
        "factor": 1.0, "bounds": [1.0, 1.2], "source": "declared",
        "verified": False, "measured_lane": False,
        "max_bound_refuted_by_measurement": True,
    }
    # The comm term is the analytic lane: two directions over the declared wire.
    comm = estimate["comm"]
    assert comm["directions"] == 2
    assert comm["lane"]["source"] == "analytic_link_capacity"
    assert comm["lane"]["alpha_ms"] == 0.0
    assert comm["comm_ms"] == pytest.approx(
        2 * estimate["geometry"]["link_bytes"] / HCCS_BYTES_PER_MS, rel=1e-12,
    )
    # No measured intercept means the declared fabric setup is the launch term.
    assert bounds["launch_ms"] == pytest.approx(
        LAUNCH_US / 1000.0 + 2 * HCCS_LATENCY_US / 1000.0, rel=1e-12,
    )


def test_mc2_chain_factor_moves_the_estimate_between_the_campaign_bounds():
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    legs = chunk_estimate(ops, model(mc2_chain_factor=1.0))
    ceiling = chunk_estimate(ops, model(mc2_chain_factor=1.2))
    assert legs["status"] == MC2_OVERLAP_UNVERIFIED
    assert legs["bounds"]["estimated_ms"] == pytest.approx(
        legs["bounds"]["legs_ms"]
    )
    assert legs["bounds"]["primary"] == "legs"
    assert ceiling["bounds"]["estimated_ms"] == pytest.approx(
        ceiling["bounds"]["upper_ms"]
    )
    assert ceiling["bounds"]["estimated_ms"] > legs["bounds"]["estimated_ms"]
    assert ceiling["bounds"]["primary"] == "chain"
    # Never below the legs sum: no measured condition shows an overlap.
    assert ceiling["bounds"]["max_bound_ms"] < legs["bounds"]["estimated_ms"]


def test_mc2_without_any_communication_is_declared_not_overlapped():
    # EP of one has no peer traffic at all: no chunk fusion is claimed, and the
    # analytic decomposition keeps its own launches.
    ops = chunk_layer(tokens=512, top_k=2, hidden=128, experts=2, ep_size=1)
    costed = compose(ops, model())
    estimate = mc2_records(costed)[0].mc2
    assert estimate["status"] == MC2_NO_COMMUNICATION
    assert estimate["comm"]["comm_ms"] == 0.0
    assert estimate["geometry"]["link_bytes"] == 0.0
    # Nothing was absorbed: the graph's decomposition is the estimate.
    dispatch = next(item for item in costed if item.op.family == "ep_dispatch")
    assert dispatch.estimated_ms > 0
    assert dispatch.mc2["role"] == "no_communication"


# ---------------------------------------------------------------------------
# 3. Measured lane: alpha/beta come from the communication calibration
# ---------------------------------------------------------------------------
def test_mc2_measured_lane_prices_alpha_and_beta_from_the_calibration():
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    estimate = chunk_estimate(ops, model(with_comm=True))
    lane = estimate["comm"]["lane"]
    assert lane["source"] == "measured_comm_curve"
    assert lane["placement"] == "hccssw"
    assert lane["alpha_ms"] == pytest.approx(ALPHA_MS, rel=1e-9)
    assert lane["beta_ms_per_byte"] == pytest.approx(BETA_MS_PER_BYTE, rel=1e-9)
    link_bytes = estimate["geometry"]["link_bytes"]
    assert estimate["comm"]["dispatch_ms"] == pytest.approx(
        ALPHA_MS + BETA_MS_PER_BYTE * link_bytes, rel=1e-9,
    )
    assert estimate["comm"]["comm_ms"] == pytest.approx(
        2 * (ALPHA_MS + BETA_MS_PER_BYTE * link_bytes), rel=1e-9,
    )
    # A measured intercept already contains the fabric setup: the launch term is
    # the kernel dispatch only, and the latency is not paid twice.
    assert estimate["bounds"]["launch_ms"] == pytest.approx(LAUNCH_US / 1000.0)
    # The chain factor is still declared, so it is still reported unverified.
    assert estimate["status"] == MC2_OVERLAP_UNVERIFIED
    assert estimate["chain"]["measured_lane"] is True
    assert "fitted_chain_factor" in estimate["reason"]


def test_mc2_with_a_fitted_chain_factor_on_a_measured_lane_is_verified():
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    estimate = chunk_estimate(
        ops, model(
            with_comm=True, mc2_chain_factor=1.1,
            mc2_chain_factor_source="fitted",
        ),
    )
    assert estimate["status"] == MC2_OVERLAP_CALIBRATED
    assert estimate["chain"]["verified"] is True
    assert estimate["chain"]["factor"] == pytest.approx(1.1)
    legs = estimate["bounds"]["legs_ms"]
    upper = estimate["bounds"]["upper_ms"]
    assert legs < estimate["bounds"]["estimated_ms"] < upper


def test_mc2_measured_lane_for_another_machine_is_not_borrowed():
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    foreign = model()
    foreign.communication_calibration = comm_calibration(hardware_key="Ascend910B4")
    estimate = chunk_estimate(ops, foreign)
    assert estimate["comm"]["lane"]["source"] == "analytic_link_capacity"
    assert "no_measured_lane_curve_for_fabric" in estimate["comm"]["lane"]["note"]


# ---------------------------------------------------------------------------
# 4. Chunk composition: one launch, compute overlapped with comm, no double count
# ---------------------------------------------------------------------------
def test_mc2_chunk_members_sum_to_the_fused_duration():
    ops = chunk_layer(
        tokens=4096, top_k=8, hidden=256, experts=8, ep_size=2, width=512,
    )
    cost_model = model()
    serial = [cost_model.cost(op) for op in ops]
    costed = compose(ops, cost_model)
    estimate = mc2_records(costed)[0].mc2
    by_id = {item.op.op_id: item for item in costed}

    compute_ids = [
        item.op.op_id for item in costed
        if item.op.family in {"moe_gate_up", "moe_down"}
    ]
    comm_ids = [
        item.op.op_id for item in costed
        if item.op.family in {"ep_dispatch", "ep_combine"}
    ]
    fused_total = sum(by_id[op_id].estimated_ms for op_id in compute_ids)
    assert fused_total == pytest.approx(estimate["bounds"]["estimated_ms"], rel=1e-9)

    # The dispatch and combine traffic lives inside the fused kernel: their
    # marginal duration is zero and they say so.
    for op_id in comm_ids:
        assert by_id[op_id].estimated_ms == 0.0
        assert by_id[op_id].demand_map() == {}
        assert by_id[op_id].mc2["role"] == "absorbed"
        assert by_id[op_id].mc2["note"].startswith("dispatch/combine traffic")

    # One launch for the whole chunk, not one per member.
    launches = sum(by_id[op_id].launch_ms for op_id in compute_ids)
    assert launches == pytest.approx(estimate["bounds"]["launch_ms"], rel=1e-9)

    # Every member's duration is reproducible from its own demands: the engine
    # sees exactly what the fused estimate claims.
    capacity = machine().link_for_scope(estimate["geometry"]["link_scope"])
    capacities = cost_model.capacity_map(capacity)
    for op_id in compute_ids:
        item = by_id[op_id]
        demands = item.demand_map()
        service = max(
            work / capacities[name] for name, work in demands.items()
            if name != "launch"
        )
        assert demands["launch"] + service == pytest.approx(
            item.estimated_ms, rel=1e-9,
        )

    # And the whole chunk is cheaper than keeping the decomposition's launches.
    serial_total = sum(item.estimated_ms for item in serial if item.op.op_id in {
        *compute_ids, *comm_ids,
    })
    assert fused_total < serial_total


def test_mc2_resource_engine_schedule_matches_the_fused_bound():
    # The engine is the consumer of ``demands``: if the fused estimate were not
    # reachable from them, the payload metrics would silently disagree with the
    # trace.  Run the actual engine on the chunk.
    ops = chunk_layer(
        tokens=4096, top_k=8, hidden=256, experts=8, ep_size=2, width=512,
    )
    cost_model = model()
    costed = compose(ops, cost_model)
    estimate = mc2_records(costed)[0].mc2
    assert estimate["status"] == MC2_OVERLAP_UNVERIFIED

    resources, stage_map = build_resource_specs(
        machine(), 1, compute_dtype="bf16",
        link=machine().link_for_scope(estimate["geometry"]["link_scope"]),
        ranks=[0, 1],
    )
    engine = ResourceEngine(resources)
    members = [
        item for item in costed
        if item.op.family in {"ep_dispatch", "ep_combine", "moe_gate_up", "moe_down"}
    ]
    id_map = add_costed_ops(
        engine, members, stage_map, prefix="p", start_ms=0.0,
    )
    result = engine.run(start_ms=0.0, until=tuple(id_map.values()))
    makespan = max(result["completed_at_ms"][task] for task in id_map.values())
    # The launch is composed serially (folded into the service demand), so the
    # engine realizes the chunk estimate itself: one fused launch, not one per
    # expert, and no max() shortcut.
    bounds = estimate["bounds"]
    assert makespan == pytest.approx(bounds["estimated_ms"], rel=1e-6)


def test_mc2_single_chunk_launch_is_not_charged_per_expert():
    # Two experts in the chunk: the fused kernel still pays one launch.
    ops = chunk_layer(tokens=2048, top_k=4, hidden=128, experts=2, ep_size=2)
    cost_model = model()
    costed = compose(ops, cost_model)
    estimate = mc2_records(costed)[0].mc2
    compute = [
        item for item in costed if item.op.family in {"moe_gate_up", "moe_down"}
    ]
    assert sum(item.launch_ms for item in compute) == pytest.approx(
        estimate["bounds"]["launch_ms"], rel=1e-12,
    )
    assert [item.mc2["role"] for item in compute] == ["compute", "compute"]
    assert sum(item.mc2["share"] for item in compute) == pytest.approx(1.0)


def test_mc2_is_only_recognised_from_its_own_recipes():
    assert MC2_RECIPES == {"moe_mc2", "mc2", "mc2_dispatch", "mc2_combine"}
    mc2_ops = chunk_layer()
    assert all(is_mc2_op(op) for op in mc2_ops[1:])
    assert not is_mc2_op(mc2_ops[0])  # the layer barrier is not an MC2 op
    assert mc2_chunk_key(mc2_ops[1]) == (0, 0, 0)
    # The activation kernel is its own vector kernel, not part of the window.
    swiglu = OpSpec(
        op_id="s0.l0.moe_act_e0", family="swiglu", phase="prefill", stage=0,
        layer=0, requests=("r",), semantic_shape=(512, 256),
        local_shape=(512, 256), padded_shape=(512, 256), launch_shape=(512, 256),
        vector_ops=512 * 256 * 4.0, recipe="moe_mc2_swiglu",
        metadata=(("mc2_chunk", 0),),
    )
    assert not is_mc2_op(swiglu)
    costed = compose(chunk_layer() + [swiglu], model())
    act = next(item for item in costed if item.op.family == "swiglu")
    assert act.mc2 is None
    assert act.estimated_ms == pytest.approx(model().cost(swiglu).estimated_ms)


# ---------------------------------------------------------------------------
# 5. No dense-GEMM transfer on top of an MC2 chunk
# ---------------------------------------------------------------------------
def artifact(*, families: tuple[str, ...], hardware_key: str = "Synthetic910_MC2"):
    row = {
        "family": "gemm", "recipe": "torch_npu_managed_runner",
        "protocol_signature": "", "timing_scope": PER_CALL,
        "model_kind": "gemm", "model_form": "wave_max",
        "fit_level": "family_protocol", "block_m": 32, "block_n": 256,
        "launch_ms": 0.0177, "eta_compute": 0.837, "eta_memory": 0.814,
        "vec_ops_per_s": None, "memory_model": "hbm", "l2_usable_bytes": 0,
        "l2_bw_bytes_per_s": None, "compute_dtype": "bf16", "samples": 367,
        "raw_event_count": 2202, "train_wape_pct": 16.6, "cv_wape_pct": 17.65,
        "median_ape_pct": 20.3, "p90_ape_pct": 51.8, "max_ape_pct": 85.2,
        "binding_share": {"compute": 0.5, "memory": 0.5},
        "binding_counts": {"compute": 180, "memory": 187},
        "identified": {"launch": True, "tile_geometry": True, "eta_compute": True},
        "pinned_parameters": [], "intervals": {}, "breakdown": {},
        "model_selection": {}, "memory_model_selection": {},
        "quality": "trusted", "warnings": [],
    }
    rows: dict[str, dict] = {}
    if "gemm" in families:
        rows[f"gemm/{PER_CALL}"] = isinstance(row, dict) and dict(row)
    if "grouped_matmul" in families:
        grouped = dict(row)
        grouped.update({
            "family": "grouped_matmul",
            "recipe": "npu_grouped_matmul_list_bf16",
            "model_kind": "grouped_gemm", "samples": 13,
        })
        rows[f"grouped_matmul/{PER_CALL}"] = grouped
    return PhysicalCalibration.from_mapping({
        "schema_version": SCHEMA_VERSION,
        "hardware_key": hardware_key,
        "arch_family": "A3",
        "machine": {
            "hardware_key": hardware_key, "arch_family": "A3",
            "aic_count": AIC, "aiv_count": 40, "clock_mhz": 1650.0,
            "ops_per_cycle": 8192.0, "peak_bw_bytes_per_s": HBM_PEAK,
            "l2_bytes": 64 * 1024 * 1024,
            "vec_elements_per_cycle_per_aiv": 128.0,
        },
        "fit_config": {"fit_policy": "synthetic_mc2_fixture"},
        "families": rows,
        "provenance": {
            "generator": "tests.test_a3_mc2_cost", "condition_count": 380,
            "raw_event_count": 2280, "fitted_utc": "2026-10-21T00:00:00+00:00",
            "training_hash": "sha256:synthetic-mc2-fixture",
            "heldout_guard": {"status": "clean"},
        },
        "unmodeled": [],
    })


def test_mc2_compute_uses_the_grouped_row_and_never_the_dense_gemm():
    grouped = artifact(families=("gemm", "grouped_matmul"))
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    estimate = chunk_estimate(ops, model(artifact=grouped))
    compute = estimate["compute"]
    assert compute["source"] == "fitted_physical"
    assert compute["fitted_artifact_families"] == ["grouped_matmul"]
    assert compute["dense_gemm_transfer"] == "suppressed_for_mc2"

    # A GEMM-only artifact leaves the compute analytic: the dense transfer is not
    # applied on top of the fused kernel.
    gemm_only = artifact(families=("gemm",))
    costed = compose(ops, model(artifact=gemm_only))
    estimate = mc2_records(costed)[0].mc2
    assert estimate["compute"]["source"] == "analytic_roofline"
    assert estimate["compute"]["fitted_members"] == 0
    for item in costed:
        record = item.physical
        if record is None:
            continue
        assert str(item.op.family) != "moe_gate_up" or not record.get("applied")
        if item.op.family in {"moe_gate_up", "moe_down"}:
            assert record["status"] == PHYSICAL_UNMODELED
            assert "transfer_to" not in record
            assert all(
                row["family"] != "gemm"
                for row in (record.get("refused_transfer") or {}).get("attempts", ())
            )


def test_mc2_chunk_estimate_never_adds_a_second_gemm_price():
    # The fused duration is ``launch + compute + comm - overlap``.  If a GEMM
    # price were added on top, the estimate would exceed the serial bound.
    grouped = artifact(families=("gemm", "grouped_matmul"))
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=4, ep_size=2)
    estimate = chunk_estimate(ops, model(artifact=grouped))
    bounds = estimate["bounds"]
    compute = estimate["compute"]["compute_ms"]
    comm = estimate["comm"]["comm_ms"]
    assert bounds["max_bound_ms"] == pytest.approx(
        bounds["launch_ms"] + max(compute, comm), rel=1e-9,
    )
    assert bounds["legs_ms"] == pytest.approx(
        bounds["launch_ms"] + compute + comm, rel=1e-9,
    )
    assert bounds["estimated_ms"] == pytest.approx(bounds["legs_ms"])
    # The campaign ceiling is 1.2x the legs and is where a fitted chain lands.
    ceiling = chunk_estimate(ops, model(artifact=grouped, mc2_chain_factor=1.2))
    assert ceiling["bounds"]["estimated_ms"] == pytest.approx(
        ceiling["bounds"]["upper_ms"], rel=1e-9,
    )
    assert ceiling["bounds"]["estimated_ms"] > bounds["estimated_ms"]


# ---------------------------------------------------------------------------
# 6. Public predict(): a MoE model shows the MC2 path
# ---------------------------------------------------------------------------
MOE_CASE = ROOT / "examples" / "a3_mla_moe_predict.json"


def test_predict_shows_the_mc2_path_for_a_moe_model():
    payload = json.loads(MOE_CASE.read_text(encoding="utf-8"))
    base = payload["runtime"]
    without_mc2 = predict({**payload, "runtime": {**base, "moe_recipe": "alltoall"}})
    with_mc2 = predict({**payload, "runtime": {**base, "moe_recipe": "mc2"}})
    provenance = with_mc2["physical_calibration"]
    assert provenance["mc2_op_count"] > 0
    assert provenance["wiring_counts"].get(MC2_OVERLAP_UNVERIFIED, 0) > 0
    # The all-to-all recipe keeps the op decomposition: no MC2 record at all.
    assert without_mc2["physical_calibration"]["mc2_op_count"] == 0

    ops = [op for step in with_mc2["traces"] for op in step["ops"]]
    mc2_ops = [op for op in ops if op["mc2"]]
    assert mc2_ops
    for op in mc2_ops:
        assert op["mc2"]["wiring_status"] == MC2_OVERLAP_UNVERIFIED
        bounds = op["mc2"]["bounds"]
        assert bounds["legs_ms"] == pytest.approx(bounds["estimated_ms"])
        assert bounds["upper_ms"] >= bounds["estimated_ms"]
        assert bounds["max_bound_ms"] < bounds["legs_ms"]
        assert op["mc2"]["geometry"]["link_bytes"] >= 0
        assert op["mc2"]["chain"]["verified"] is False
        if op["mc2"]["role"] == "absorbed":
            assert op["estimated_ms"] == 0.0
        else:
            assert op["estimated_ms"] > 0
    assert sum(1 for op in mc2_ops if op["mc2"]["role"] == "absorbed") > 0
    # The fused kernel is cheaper than the all-to-all decomposition: a prediction
    # change, not an accuracy claim.
    assert with_mc2["metrics"]["ttft_ms"]["mean"] < without_mc2["metrics"]["ttft_ms"]["mean"]
    assert with_mc2["metrics"]["tpot_ms"]["mean"] <= without_mc2["metrics"]["tpot_ms"]["mean"]


def test_predict_models_an_explicit_alltoall_moe_without_mc2_markers():
    # ``moe_recipe: alltoall`` must not produce MC2 records: the fused-kernel
    # estimate is tied to the declared recipe, never inferred from the family.
    payload = json.loads(MOE_CASE.read_text(encoding="utf-8"))
    payload = {
        **payload,
        "runtime": {**payload["runtime"], "moe_recipe": "alltoall"},
    }
    result = predict(payload)
    assert result["physical_calibration"]["mc2_op_count"] == 0
    assert all(
        op["mc2"] is None for step in result["traces"] for op in step["ops"]
    )


# ---------------------------------------------------------------------------
# 7. Grouped-matmul launch count: the tensor-list expert cap (128)
# ---------------------------------------------------------------------------
def test_grouped_matmul_launch_count_follows_the_tensor_list_expert_cap():
    from mindie_sim.a3_cost import (
        GROUPED_LAUNCH_EXPERT_CAP, grouped_expert_count,
        grouped_matmul_launch_count, mc2_chunk_expert_count,
    )

    assert GROUPED_LAUNCH_EXPERT_CAP == 128

    # One launch up to the cap; a 256- or 384-expert layer is two/three launches.
    for experts, launches in ((1, 1), (8, 1), (128, 1), (129, 2), (256, 2),
                              (384, 3)):
        op = grouped_op(tokens=2048, experts=experts, n=1024, k=1024)
        assert grouped_expert_count(op) == experts
        assert grouped_matmul_launch_count(op) == launches
    # A family that is not a grouped matmul is always one launch.
    assert grouped_matmul_launch_count(gemm_op_for_launch()) == 1
    # The MC2 chunk's expert list is declared on its dispatch op.
    assert mc2_chunk_expert_count(chunk_layer(experts=4, ep_size=2)) == 4
    assert mc2_chunk_expert_count(chunk_layer(experts=256, ep_size=2)) == 256


def gemm_op_for_launch() -> OpSpec:
    return OpSpec(
        op_id="gemm.launch", family="gemm", phase="prefill", stage=0, layer=0,
        requests=("r",), semantic_shape=(512, 1024, 1024),
        local_shape=(512, 1024, 1024), padded_shape=(512, 1024, 1024),
        launch_shape=(512, 1024, 1024), flops=2.0 * 512 * 1024 * 1024,
        dtype="bf16", recipe="torch_mm_preallocated",
    )


def test_grouped_matmul_pays_one_dispatch_intercept_per_launch():
    one = grouped_op(tokens=2048, experts=8, n=1024, k=1024)
    wide = grouped_op(tokens=2048, experts=256, n=1024, k=1024)
    cost_model = model()
    narrow_cost = cost_model.cost(one)
    wide_cost = cost_model.cost(wide)
    per_launch = LAUNCH_US / 1000.0
    assert narrow_cost.launch_ms == pytest.approx(per_launch)
    assert wide_cost.launch_ms == pytest.approx(2 * per_launch)
    # Launch stays additive: each cost is its own launch plus its own service.
    for costed in (narrow_cost, wide_cost):
        assert costed.estimated_ms == pytest.approx(
            costed.launch_ms + costed.service_ms, rel=1e-9,
        )
    detail = dict(wide_cost.launch_detail or {})
    assert detail["launches"] == 2
    assert detail["experts"] == 256
    assert detail["expert_cap_per_launch"] == 128
    assert detail["reason"] == "npu_grouped_matmul_tensorlist_expert_cap"
    assert narrow_cost.launch_detail["reason"] == "single_launch"
    # A dense GEMM has no grouped cap, so it carries no grouped launch detail.
    assert cost_model.cost(gemm_op_for_launch()).launch_detail is None


def test_a_fitted_grouped_row_pays_the_extra_launch_intercepts():
    # The fitted intercept is one tensor-list launch; the extra launches repeat
    # exactly that intercept, so a 256-expert op pays it twice.
    fitted = artifact(families=("grouped_matmul",))

    narrow = model(artifact=fitted).cost(
        grouped_op(experts=8, n=1024, k=1024, family="grouped_matmul"),
    )
    wide = model(artifact=fitted).cost(
        grouped_op(experts=256, n=1024, k=1024, family="grouped_matmul"),
    )
    assert narrow.calibration.status.startswith("physical")
    assert wide.calibration.status.startswith("physical")
    assert wide.to_dict()["physical"]["transfer_to"] == {
        "family": "grouped_matmul", "recipe": "npu_grouped_matmul_list_bf16",
    }
    # Both are fitted; the wide op pays exactly two launch intercepts.
    assert wide.launch_ms == pytest.approx(2 * narrow.launch_ms, rel=1e-9)
    assert wide.to_dict()["launch_detail"]["launches"] == 2


def test_mc2_chunk_launch_count_uses_the_widest_grouped_member():
    # 256 local experts in the chunk -> the grouped matmul inside the fused
    # kernel is two tensor-list launches, and the chunk pays two intercepts.
    ops = chunk_layer(tokens=2048, top_k=8, hidden=128, experts=256,
                      ep_size=2, width=256)
    estimate = chunk_estimate(ops, model())
    assert estimate["bounds"]["launch_count"] == 2
    # Two dispatch intercepts for the two tensor-list launches; the fabric
    # setup latency is per direction, not per launch.
    assert estimate["bounds"]["launch_ms"] == pytest.approx(
        2 * LAUNCH_US / 1000.0 + 2 * HCCS_LATENCY_US / 1000.0,
    )
    narrow = chunk_estimate(
        chunk_layer(tokens=2048, top_k=8, hidden=128, experts=8, ep_size=2),
        model(),
    )
    assert narrow["bounds"]["launch_count"] == 1


# ---------------------------------------------------------------------------
# 8. Measured lanes per EP group, and kernels the model refuses to fake
# ---------------------------------------------------------------------------
def campaign_lane_conditions() -> "CommunicationCalibration":
    """Campaign-shaped lane points (A3, hidden 7168, top-8), per EP group.

    Transcribed from the completed campaign's analysis tables: per-direction lane
    ms at the per-rank payload of each measured condition.
    """
    from mindie_sim.a3_communication import CommCondition, CommunicationCalibration

    points = (
        # ep, per-direction payload bytes, (dispatch + combine) / 2 ms
        (2, 29_360_128, 0.488),
        (4, 88_080_384, 0.462),
        (8, 205_520_896, 0.656),
    )
    return CommunicationCalibration([
        CommCondition(
            "hccssw", "p2p_only", payload, lane, 10,
            hardware_key="Synthetic910_MC2", timing_scope="communication_lane",
            ranks=ep, dtype="bf16", algorithm="p2p_only",
            collective="send_recv", group_size=ep,
        )
        for ep, payload, lane in points
    ])


def test_mc2_lane_uses_the_ep_groups_own_measured_point():
    ops = chunk_layer(tokens=4096, top_k=8, hidden=128, experts=8, ep_size=4)
    cost_model = model()
    cost_model.communication_calibration = campaign_lane_conditions()
    estimate = chunk_estimate(ops, cost_model)
    lane = estimate["comm"]["lane"]
    assert lane["source"] == "measured_comm_curve"
    assert lane["measured_ranks"] == 4, "the EP-4 point must answer an EP-4 group"
    assert estimate["comm"]["lane"]["alpha_ms"] + (
        estimate["comm"]["lane"]["beta_ms_per_byte"]
        * estimate["geometry"]["link_bytes"]
    ) > 0
    # A group with no measured point falls back to the two-rank lane (declared)
    # or to the analytic wire, never to another group's curve silently.
    three = chunk_estimate(
        chunk_layer(tokens=4096, top_k=8, hidden=128, experts=8, ep_size=3),
        cost_model,
    )
    assert three["comm"]["lane"]["measured_ranks"] == MC2_LANE_RANKS
    assert three["comm"]["lane"]["requested_ranks"] == 3


def test_a_fused_grouped_matmul_swiglu_quant_is_refused_not_faked():
    # The campaign measures a fused activation+quant epilogue inside the grouped
    # matmul.  Pricing it as a plain grouped GEMM (plus a separate activation)
    # would double count, so the identity is refused with an explicit reason.
    from mindie_sim.a3_cost import PHYSICAL_UNEXPRESSIBLE_RECIPES

    assert "npu_grouped_matmul_swiglu_quant" in PHYSICAL_UNEXPRESSIBLE_RECIPES
    fused = grouped_op(
        family="grouped_matmul", recipe="npu_grouped_matmul_swiglu_quant",
        group_size=256,
    )
    cost = model(artifact=artifact(families=("gemm", "grouped_matmul"))).cost(fused)
    record = cost.to_dict()["physical"]
    assert record["status"] == PHYSICAL_UNMODELED
    assert record["reason"] == (
        "fused_grouped_matmul_swiglu_quant_epilogue_not_expressible"
    )
    assert "transfer_to" not in record
    assert record["wiring"]["status"] == PHYSICAL_UNMODELED
    assert record["wiring"]["reason"].startswith("fused_grouped_matmul")
