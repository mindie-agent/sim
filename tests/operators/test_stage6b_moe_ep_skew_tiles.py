"""MoE EP straggler rank skew and per-expert Cube M=16 tile quantization.

[Role] Verify probability-path busiest-EP-rank order-statistic moments,
  explicit-owner conservation, EPLB variance damping, physical-slot weight
  floors, and per-expert Cube FRACTAL_M=16 tile quantization with dispersion.
[Invariants] At B=1..N and EP=16..64, critical-path active banks stay >= 1.0
  and rows stay >= ceil(T*k/EP); explicit owner sums conserve exact T*k;
  EPLB damping reduces skew monotonically while keeping integer floors;
  ExpectedGroupedMatmul pads each bank independently to FRACTAL_M=16 while
  conserving exact logical op.flops.
[Timeouts & Locks] Pure analytical arithmetic and bounded in-memory Engine runs.
[Side-Effect Policy] Read-only; no device access, network, or filesystem writes.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.catalog.expert_slots import expert_slot_plan
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.prediction.feedforward import moe
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.operators.grouped_moe.expected_grouped import (
    FRACTAL_M,
    ExpectedGroupedMatmul,
    _dispersed_group_rows,
)
from mindie_sim.prediction.operators.matrix.matrix import TiledMatmul
from mindie_sim.prediction.operators.matrix.matrix_mapping import (
    MappedMatmul,
    simulate_matrix_operator,
)
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.operators.moe_routing.expert_slot_work import (
    EPLB_VARIANCE_DAMPING as SLOT_EPLB_DAMPING,
    physical_slot_work,
)
from mindie_sim.prediction.operators.moe_transport.moe_callers import (
    EPLB_VARIANCE_DAMPING,
    ORDER_STAT_M2_Z,
    LocalExpertReduce,
    RoutingPolicy,
    _normal_max_order_z,
)
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.parallel import ParallelSpec

pytestmark = pytest.mark.tier0

DAMPING_EXPECTED = 0.35
FRACTAL_EXPECTED = 16
CUBE_N = 2048
CUBE_K = 4096
EXPERT_WEIGHT_UNIT = 1024.0


def _stub_model_and_layer(
    *, num_experts: int, top_k: int, routing: tuple[float, ...] | None = None,
):
    model = SimpleNamespace(expert_routing=routing)
    layer = SimpleNamespace(num_experts=num_experts, top_k=top_k, index=0)
    return model, layer


def _recipe_context(op, *, cores: int = 8) -> RecipeContext:
    work = {
        "cube": float(op.flops),
        "hbm": float(op.hbm_bytes),
        "mte2": float(op.mte2_bytes),
        "mte3": float(op.mte3_bytes),
        "mte1": float(op.mte2_bytes),
        "vector": float(op.vector_ops),
    }
    rates = {
        "cube": 1e12,
        "hbm": 1e10,
        "mte2": 1e10,
        "mte3": 1e10,
        "mte1": 2e10,
        "vector": 1e11,
    }
    names = {role: role for role in work}
    return RecipeContext(
        op=op,
        work=work,
        resource_names=names,
        rates=rates,
        core_count=cores,
        compute_peaks={"bf16": 1e12},
        pool_names={"l1": "l1", "l0": "l0"},
        pool_capacities={"l1": 1024 * 1024, "l0": 512 * 1024},
        pool_units={"l1": cores, "l0": cores},
    )


def test_named_physical_constants_and_order_statistic_monotonicity():
    assert EPLB_VARIANCE_DAMPING == pytest.approx(DAMPING_EXPECTED)
    assert SLOT_EPLB_DAMPING == pytest.approx(DAMPING_EXPECTED)
    assert FRACTAL_M == FRACTAL_EXPECTED
    assert ORDER_STAT_M2_Z == pytest.approx(math.sqrt(2.0 / math.pi))
    assert _normal_max_order_z(0) == 0.0
    assert _normal_max_order_z(1) == 0.0
    assert _normal_max_order_z(2) == pytest.approx(ORDER_STAT_M2_Z)
    values = [_normal_max_order_z(m) for m in (2, 3, 4, 8, 16, 32, 64, 128)]
    assert values == sorted(values)
    assert values[-1] > values[0]


@pytest.mark.parametrize("ep", (16, 32, 64))
@pytest.mark.parametrize("tokens,experts,top_k,zero_experts", [
    (1, 256, 8, 0),
    (1, 256, 10, 0),
    (6, 256, 8, 0),
    (6, 384, 8, 0),
    (1, 256, 12, 128),
    (6, 256, 12, 128),
])
def test_busiest_ep_rank_straggler_enforces_integer_floor_and_no_fractional_banks(
    ep, tokens, experts, top_k, zero_experts,
):
    policy = RoutingPolicy()
    model, layer = _stub_model_and_layer(num_experts=experts, top_k=top_k)
    parallel = ParallelSpec(tp=ep, ep=ep)
    r_max, a_max = policy.local(
        model, layer, parallel, tokens, owner=None, zero_experts=zero_experts,
    )
    e_loc = experts // ep
    total_real = float(tokens * top_k) * (float(experts) / float(experts + zero_experts))
    assert r_max == float(int(r_max))
    assert r_max >= math.ceil(total_real / ep - 1e-12)
    assert r_max <= tokens * min(top_k, e_loc)
    assert a_max >= 1.0
    assert a_max <= min(float(e_loc), r_max)
    # Verify verify-step E=256, EP=64, top-8, T=6 exceeds the naive 0.75 / 0.694 mean.
    if (ep, tokens, experts, top_k, zero_experts) == (64, 6, 256, 8, 0):
        assert r_max >= 2.0 and a_max >= 2.0


def test_replicated_ep_and_local_expert_reduce_enforce_active_bank_floor():
    policy = RoutingPolicy()
    # EP > num_experts (replicated experts) still never bills < 1 active bank.
    model, layer = _stub_model_and_layer(num_experts=8, top_k=2)
    r_max, a_max = policy.local(model, layer, ParallelSpec(tp=16, ep=16), tokens=1, owner=None)
    assert r_max >= 1.0 and a_max == pytest.approx(1.0)
    # LocalExpertReduce probability path also enforces >= 1.0 active bank at B=1, EP=16.
    big_model, big_layer = _stub_model_and_layer(num_experts=256, top_k=8)
    caller = LocalExpertReduce(tokens=1, top_k=8, experts=256, ep=16)
    prepared = caller.prepare(
        policy, big_model, big_layer, ParallelSpec(tp=16, ep=16), tokens=1, realization=None,
    )
    assert prepared.local_rows >= 1.0
    assert prepared.active >= 1.0
    # Zero tokens still yields zero rows and zero active banks.
    assert policy.local(big_model, big_layer, ParallelSpec(tp=16, ep=16), tokens=0) == (0.0, 0.0)


def test_end_to_end_feedforward_never_reads_fractional_expert_weight_bank():
    raw = compose_case(
        "deepseek-v3", REPRESENTATIVE_SKUS["a5"],
        tp=16, dp=1, ep=16, layers=4, prompt_tokens=128, output_tokens=8,
        exclude_optimizations=("cann_megamoe_fusion",),
    )
    case = parse_case(raw)
    moe_layer = next(plan for plan in case.model.layer_plans if plan.ffn == "moe")
    ops, _ = moe(
        prefix="s0.l3", plan=moe_layer, deps="in", stage=0, requests=("r0",),
        tokens=1, global_tokens=1, model=case.model, parallel=case.runtime.parallel,
        mode="decode", runtime=case.runtime,
    )
    gate = next(op for op in ops if op.recipe == "grouped_gate_up")
    meta = dict(gate.metadata)
    single_bank_bytes = case.model.quant.matrix_bytes(
        2 * moe_layer.moe_intermediate_size, case.model.hidden_size, "routed_expert",
    )
    assert meta["expected_rows"] >= 1.0
    assert meta["expected_active_experts"] >= 1.0
    assert meta["routed_weight_read_bytes"] >= single_bank_bytes


@pytest.mark.parametrize("ep,experts,top_k,tokens,skewed,zero_experts", [
    (4, 32, 4, 1, False, 0),
    (16, 256, 8, 1, False, 0),
    (64, 256, 8, 6, False, 0),
    (8, 128, 8, 12, True, 0),
    (16, 8, 2, 5, False, 0),
    (32, 256, 12, 6, False, 128),
])
def test_explicit_owner_summation_conserves_exact_total_assignments(
    ep, experts, top_k, tokens, skewed, zero_experts,
):
    probs = (
        tuple(float((i % 7) + 1) for i in range(experts))
        if skewed else None
    )
    model, layer = _stub_model_and_layer(num_experts=experts, top_k=top_k, routing=probs)
    parallel = ParallelSpec(tp=ep, ep=ep)
    policy = RoutingPolicy()
    owner_results = [
        policy.local(model, layer, parallel, tokens, owner=r, zero_experts=zero_experts)
        for r in range(ep)
    ]
    expected_total = float(tokens * top_k) * (float(experts) / float(experts + zero_experts))
    assert math.fsum(rows for rows, _ in owner_results) == pytest.approx(expected_total, rel=1e-12)
    straggler_rows, straggler_active = policy.local(
        model, layer, parallel, tokens, owner=None, zero_experts=zero_experts,
    )
    assert straggler_rows >= max(rows for rows, _ in owner_results)
    assert straggler_active >= max(act for _, act in owner_results)
    if ep <= experts and zero_experts == 0:
        _, expert_active = policy.expert_moments(model, layer, tokens)
        assert math.fsum(act for _, act in owner_results) == pytest.approx(
            math.fsum(expert_active), rel=1e-12,
        )


def test_eplb_variance_damping_reduces_skew_cleanly_while_preserving_floors():
    policy = RoutingPolicy()
    model, layer = _stub_model_and_layer(num_experts=256, top_k=8)
    plain_parallel = ParallelSpec(tp=16, ep=16, redundant_experts_per_rank=0)
    redundant_parallel = ParallelSpec(tp=16, ep=16, redundant_experts_per_rank=1)

    r_plain, a_plain = policy.local(model, layer, plain_parallel, tokens=6, owner=None)
    r_explicit, a_explicit = policy.local(
        model, layer, plain_parallel, tokens=6, owner=None, eplb_damped=True,
    )
    r_redundant, a_redundant = policy.local(model, layer, redundant_parallel, tokens=6, owner=None)
    rt_force = SimpleNamespace(kernels={"force_eplb_source": {"kind": "vllm_ascend_sp_mc2"}},
                               routing={}, metadata=())
    r_force, a_force = policy.local(
        model, layer, plain_parallel, tokens=6, owner=None, runtime=rt_force,
    )
    rt_meta = SimpleNamespace(
        kernels={}, routing={},
        metadata=(("additional_config", {"eplb_config": {"dynamic_eplb": True}}),),
    )
    r_meta, a_meta = policy.local(
        model, layer, plain_parallel, tokens=6, owner=None, runtime=rt_meta,
    )

    assert (r_explicit, a_explicit) == pytest.approx((r_redundant, a_redundant))
    assert (r_explicit, a_explicit) == pytest.approx((r_force, a_force))
    assert (r_explicit, a_explicit) == pytest.approx((r_meta, a_meta))
    assert r_explicit < r_plain and a_explicit < a_plain
    assert r_explicit >= math.ceil((6 * 8) / 16) and a_explicit >= 1.0


def test_physical_slot_work_enforces_bank_floor_at_small_batch_and_conserves_rows():
    layer = SimpleNamespace(num_experts=256, index=0)
    parallel = ParallelSpec(tp=64, ep=64, redundant_experts_per_rank=1)
    plan = expert_slot_plan(None, layer, parallel)
    inclusions = tuple([8.0 / 256.0] * 256)
    work = physical_slot_work(
        slot_plan=plan, logical_inclusions=inclusions, global_tokens=1,
        packed_weight_bytes=EXPERT_WEIGHT_UNIT,
    )
    assert math.fsum(work.expected_rows_by_rank) == pytest.approx(8.0)
    assert all(slots >= 1 for slots in work.active_slots_by_rank)
    assert all(work.weight_read_bytes(rank) >= EXPERT_WEIGHT_UNIT for rank in range(64))

    sat = physical_slot_work(
        slot_plan=plan, logical_inclusions=inclusions, global_tokens=10**9,
        packed_weight_bytes=EXPERT_WEIGHT_UNIT,
    )
    assert sat.weight_read_bytes(0) == pytest.approx(5.0 * EXPERT_WEIGHT_UNIT)


def test_expected_grouped_per_expert_fractal_padding_and_binomial_dispersion():
    # Case 1: Small batch B=1, EP=64 verify (active=2.26, rows=3) -> 3 banks * 16 = 48.
    op_small = gemm_op(
        op_id="g_small", deps=(), stage=0, layer=0, requests=(),
        m=3, n=CUBE_N, k=CUBE_K, quant=QuantSpec("bf16"), category="routed_expert",
        phase="decode", family="grouped_gemm", recipe="grouped_gate_up",
        metadata=(
            ("routing_basis", "probability_moments"),
            ("matrix_mapping_domain", "aggregate_experts"),
            ("matrix_weight_bank_groups", 4),
            ("expected_rows", 3.0),
            ("expected_active_experts", 2.26),
        ),
    )
    captured: list = []

    class _CaptureSteady(SteadyMatmul):
        def lower(self, context, **kwargs):
            captured.append((context.op, kwargs.get("group_rows")))
            return super().lower(context, **kwargs)

    ExpectedGroupedMatmul(_CaptureSteady()).lower(_recipe_context(op_small))
    mapped_small, groups_small = captured[-1]
    assert groups_small == (FRACTAL_M, FRACTAL_M, FRACTAL_M)
    assert mapped_small.padded_shape[0] == 3 * FRACTAL_M
    assert mapped_small.local_shape[0] == 3 * FRACTAL_M
    assert mapped_small.flops == op_small.flops

    # Case 2: Audit counterexample (32 active banks, 480 rows -> mean_m = 15):
    # dispersion splits into 16 banks of 11 (16 padded) + 16 banks of 19 (32 padded) = 768.
    dispersed = _dispersed_group_rows(480.0, 32)
    assert dispersed == (11,) * 16 + (19,) * 16
    op_disp = gemm_op(
        op_id="g_disp", deps=(), stage=0, layer=0, requests=(),
        m=480, n=CUBE_N, k=CUBE_K, quant=QuantSpec("bf16"), category="routed_expert",
        phase="decode", family="grouped_gemm", recipe="grouped_gate_up",
        metadata=(
            ("routing_basis", "probability_moments"),
            ("matrix_mapping_domain", "aggregate_experts"),
            ("matrix_weight_bank_groups", 32),
            ("expected_rows", 480.0),
            ("expected_active_experts", 32.0),
        ),
    )
    ExpectedGroupedMatmul(_CaptureSteady()).lower(_recipe_context(op_disp))
    mapped_disp, groups_disp = captured[-1]
    assert groups_disp == (FRACTAL_M,) * 16 + (2 * FRACTAL_M,) * 16
    assert mapped_disp.padded_shape[0] == 768
    assert mapped_disp.local_shape[0] == 768
    assert mapped_disp.flops == op_disp.flops == 2 * 480 * CUBE_N * CUBE_K

    # Case 3: Engine replay conserves exact logical op.flops in serviced cube work.
    result = simulate_matrix_operator(
        _recipe_context(op_disp), recipe=ExpectedGroupedMatmul(SteadyMatmul()),
    )
    assert result.serviced["cube"] == pytest.approx(float(op_disp.flops), rel=1e-9)

    # Case 4: Odd bank count with unequal padded groups lowers cleanly on both
    # SteadyMatmul AND equal-group recipes (TiledMatmul / MappedMatmul).
    op_odd = replace(
        op_disp,
        op_id="g_odd",
        local_shape=(45, CUBE_N, CUBE_K),
        metadata=tuple(
            (k, 45.0 if k == "expected_rows" else 3.0 if k == "expected_active_experts" else v)
            for k, v in op_disp.metadata
        ),
    )
    for matmul_cls in (SteadyMatmul, MappedMatmul, TiledMatmul):
        prog = ExpectedGroupedMatmul(matmul_cls()).lower(_recipe_context(op_odd))
        assert prog is not None


def test_large_batch_skew_vanishes_asymptotically_and_saturates_banks():
    policy = RoutingPolicy()
    model, layer = _stub_model_and_layer(num_experts=256, top_k=8)
    parallel = ParallelSpec(tp=8, ep=8)
    r_small, _ = policy.local(model, layer, parallel, tokens=6, owner=None)
    r_large, a_large = policy.local(model, layer, parallel, tokens=4096, owner=None)
    rel_small = (r_small - 6.0) / 6.0
    rel_large = (r_large - 4096.0) / 4096.0
    assert rel_large < rel_small
    assert rel_large < 0.05
    assert a_large == pytest.approx(32.0)


def test_catalog_high_ep_graph_produces_integer_straggler_and_valid_banks():
    raw = compose_case(
        "qwen3.5-35b-a3b", REPRESENTATIVE_SKUS["a5"],
        tp=1, dp=16, ep=16, layers=2, prompt_tokens=128, output_tokens=8,
        exclude_optimizations=("cann_megamoe_fusion",),
    )
    case = parse_case(raw)
    graph = target_graphs(case, phase="decode", tokens=1, context=128, sequences=1, scope="body")[0]
    gate = next(op for op in graph if op.recipe == "grouped_gate_up")
    meta = dict(gate.metadata)
    assert meta["expected_rows"] == float(gate.local_shape[0])
    assert meta["expected_active_experts"] >= 1.0
