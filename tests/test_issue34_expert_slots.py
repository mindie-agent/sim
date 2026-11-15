"""Physical redundant-expert slots, exact packed bytes and active-copy reads."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.expert_slots import (
    expert_slot_plan, routed_expert_weight_terms,
)
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes
from mindie_sim.prediction.expert_slot_work import (
    bind_expert_slot_work, physical_slot_work,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from issue34_helpers import dense_layer, quant


def layer(experts=8):
    return dense_layer(0, ffn="moe", num_experts=experts, top_k=2,
                       moe_intermediate_size=32, num_shared_experts=0)


def terms(experts=8):
    return (
        WeightTerm("expert_gate", 64, 32, "routed_expert", "expert_gate_up",
                   count=experts, quant=quant()),
        WeightTerm("expert_down", 32, 64, "routed_expert", "expert_down",
                   count=experts, quant=quant()),
    )


def test_uniform_slot_plan_and_distinct_extra_slots():
    parallel = ParallelSpec(ep=4, redundant_experts_per_rank=1)
    plan = expert_slot_plan(None, layer(), parallel)
    assert plan.base_slots_by_rank == (2, 2, 2, 2)
    assert plan.redundant_slots_by_rank == (1, 1, 1, 1)
    for rank, ids in enumerate(plan.expert_ids_by_rank):
        assert len(ids) == 3
        assert set(plan.expert_ids_by_rank[rank]) == set(ids)


def test_invalid_redundant_counts_and_explicit_maps_fail():
    with pytest.raises(ModelingError, match="exceed_distinct_local_experts"):
        expert_slot_plan(None, layer(8), ParallelSpec(
            ep=4, redundant_experts_per_rank=7))
    with pytest.raises(ModelingError, match="exceed_distinct_local_experts"):
        expert_slot_plan(None, layer(8), ParallelSpec(
            ep=1, redundant_experts_per_rank=1))
    with pytest.raises(ModelingError, match="redundant_count_mismatch"):
        expert_slot_plan(None, layer(8), ParallelSpec(
            ep=4, redundant_experts_per_rank=1), routing={
                "expert_slot_policy": "explicit",
                "expert_slot_map": {0: [[0, 1], [2, 3], [4, 5], [6, 7]]}})


def test_extra_terms_are_exact_runtime_copies_not_logical_experts():
    parallel = ParallelSpec(ep=4, redundant_experts_per_rank=1)
    plan = expert_slot_plan(None, layer(), parallel)
    routed = routed_expert_weight_terms(terms(), plan, ep_rank=0)
    gate = [term for term in routed if term.name.startswith("expert_gate")]
    extra = next(term for term in gate if term.name == "expert_gate_redundant")
    assert extra.count == 1 and extra.resident_count == 1
    assert extra.source_owned is False
    base = next(term for term in gate if term.name == "expert_gate")
    assert base.resident_count == 2
    assert base.logical_parameters == 64 * 32 * 8
    local = 64 * 32 * 2
    assert term_bytes(base, quant(), parallel) == local * 2
    assert term_bytes(extra, quant(), parallel) == local


def test_etp_and_uneven_owners_use_actual_shapes():
    parallel = ParallelSpec(tp=8, ep=2, expert_tp=4,
                            redundant_experts_per_rank=1)
    plan = expert_slot_plan(None, layer(), parallel)
    assert plan.base_slots_by_rank == (4, 4)
    routed = routed_expert_weight_terms(terms(), plan, ep_rank=1)
    base = next(term for term in routed if term.name == "expert_gate")
    assert term_bytes(base, quant(), parallel) == (64 // 4) * 32 * 4 * 2


def test_balanced_active_slots_conserve_rows_and_saturate():
    parallel = ParallelSpec(ep=4, redundant_experts_per_rank=1)
    plan = expert_slot_plan(None, layer(), parallel)
    work = physical_slot_work(
        slot_plan=plan, logical_inclusions={expert: 2 / 8 for expert in range(8)},
        global_tokens=1000, packed_weight_bytes=128)
    assert sum(work.expected_rows_by_rank) == pytest.approx(1000 * 2)
    assert all(value >= 0 for row in work.per_rank_slot_rows for value in row)
    saturated = physical_slot_work(
        slot_plan=plan, logical_inclusions={expert: 2 / 8 for expert in range(8)},
        global_tokens=10 ** 9, packed_weight_bytes=128)
    base = physical_slot_work(
        slot_plan=expert_slot_plan(None, layer(), ParallelSpec(ep=4)),
        logical_inclusions={expert: 2 / 8 for expert in range(8)},
        global_tokens=10 ** 9, packed_weight_bytes=128)
    ratio = (sum(saturated.weight_read_bytes(rank) for rank in range(4))
             / sum(base.weight_read_bytes(rank) for rank in range(4)))
    assert ratio == pytest.approx(3 / 2)


def test_exact_realization_is_integer_and_never_duplicates_tokens():
    parallel = ParallelSpec(ep=4, redundant_experts_per_rank=1)
    plan = expert_slot_plan(None, layer(), parallel)
    realization = {0: 5, 1: 3, 2: 7, 3: 1}
    work = physical_slot_work(
        slot_plan=plan, logical_inclusions={}, global_tokens=16,
        routing_realization=realization, packed_weight_bytes=128)
    assert sum(sum(row) for row in work.per_rank_slot_rows) == 16
    for rank, row in enumerate(work.per_rank_slot_rows):
        assert all(value == int(value) for value in row)
    active = sum(work.active_slots_by_rank)
    assert active <= sum(len(ids) for ids in plan.expert_ids_by_rank)


def test_bind_expert_slot_work_updates_only_declared_reads():
    parallel = ParallelSpec(ep=2, redundant_experts_per_rank=1)
    plan = expert_slot_plan(None, layer(), parallel)
    work = physical_slot_work(
        slot_plan=plan, logical_inclusions={expert: 2 / 8 for expert in range(8)},
        global_tokens=1000, packed_weight_bytes=128)
    op = OpSpec("grouped", "grouped_gemm", "decode", 0, 0, (), (1,), (1,), (1,), (1,),
                hbm_bytes=1000.0,
                metadata=(("ep_rank", 0), ("routed_weight_read_bytes", 250.0),))
    bound = bind_expert_slot_work((op,), slot_work=work)[0]
    metadata = dict(bound.metadata)
    assert metadata["physical_slot_weight_bytes"] == pytest.approx(
        work.weight_read_bytes(0))
    assert bound.hbm_bytes == pytest.approx(
        1000.0 - 250.0 + work.weight_read_bytes(0))
