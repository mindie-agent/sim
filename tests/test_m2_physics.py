"""M2 physical mechanism, traffic, roofline and zero-envelope invariants."""
from __future__ import annotations

import math

import pytest

from mindie_sim.backends.ascend import materialize
from mindie_sim.prediction.attention_pipeline import TiledFlashAndPagedAttentionRecipe
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.matrix_traffic import (MatrixGeometry, MatrixSchedule,
                                                  plan_matrix_traffic)
from mindie_sim.prediction.memory_traffic import (AttentionAccessSpec,
                                                  AttentionSchedule, IndexAccessSpec,
                                                  IndexSchedule, plan_attention_traffic,
                                                  plan_index_key_traffic)
from mindie_sim.prediction.operators.kda_recurrent import (Geometry, Params,
                                                           summarize_work)
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.resource_attribution import estimate_exposed_hbm
from mindie_sim.prediction.roofline_graph import (PhysicalWorkGraph,
                                                  PhysicalWorkNode,
                                                  RooflineTargets,
                                                  critical_path_roofline,
                                                  link_utilization)
from mindie_sim.prediction.vector_pipeline import AIVTiledVectorAndGatherRecipe
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.physical_mechanisms import (MemoryHierarchySpec,
                                                   parse_physical_mechanisms)


def hardware():
    raw = materialize({"backend": "ascend", "sku": "Ascend950DT_9582",
                       "num_devices": 8, "ranks_per_node": 8})
    return Hardware.from_mapping(raw)


def op(family="elementwise", recipe="residual_add", **kwargs):
    shape = kwargs.pop("shape", (8, 7168))
    return OpSpec("op", family, "decode", 0, 0, ("r",), shape, shape, shape,
                  shape, recipe=recipe, dtype="bf16", **kwargs)


def test_ascend_profile_declares_typed_mechanisms_and_per_pair_mesh():
    machine = hardware()
    die = machine.dies[0]
    assert die.physical_mechanisms.memory_hierarchy.l2_capacity_bytes == 134217728
    assert (256, 256) in die.physical_mechanisms.matrix.base_tiles
    pair = [link for link in machine.links if "d2d_intra_node" in link.channel][0]
    assert pair.semantics == "per_pair"
    assert pair.capacity_bytes_per_ms == 47_500_000
    assert pair.peak_capacity_bytes_per_ms == 56_000_000


def test_unknown_physical_mechanism_raises_exact_error():
    with pytest.raises(ModelingError) as caught:
        RecipeBook().select(op(family="unmodeled_test", recipe="unknown"))
    assert str(caught.value) == "unmodeled_operator_mechanism:unmodeled_test:unknown"


def test_dense_resident_kv_is_read_once_but_refills_remain():
    hierarchy = MemoryHierarchySpec(134217728, 0.75, 128, True, {})
    plan = plan_attention_traffic(
        AttentionAccessSpec(
            q_lengths=(128,), context_lengths=(4096,),
            unique_rows=4224, refill_rows=8192, row_bytes=576,
            phase="prefill"),
        hierarchy, AttentionSchedule(query_tile=128))
    assert plan.hbm_read_bytes == 4224 * 576
    assert plan.private_refill_bytes == 8192 * 576
    assert plan.l2_hit_bytes == (8192 - 4224) * 576


def test_unknown_sparse_decode_topk_is_not_collapsed_by_capacity():
    hierarchy = MemoryHierarchySpec(134217728, 0.75, 128, True, {})
    access = AttentionAccessSpec(
        q_lengths=(1,), context_lengths=(32768,), unique_rows=32768,
        refill_rows=2048, selected_rows=2048, row_bytes=576,
        phase="decode", topk=2048)
    plan = plan_attention_traffic(access, hierarchy, AttentionSchedule())
    assert plan.hbm_read_bytes == 2048 * 576
    assert plan.residency_basis == "unknown_dynamic_topk_per_query_gather"


def test_shared_dense_index_prefix_reads_once():
    hierarchy = MemoryHierarchySpec(134217728, 0.75, 128, True, {})
    plan = plan_index_key_traffic(
        IndexAccessSpec(query_rows=4, prefix_tokens=4096,
                        private_tokens=(128, 128, 128, 128), row_bytes=576),
        hierarchy, IndexSchedule(batch_tile=True))
    assert plan.hbm_read_bytes == (4096 + 512) * 576
    assert plan.l2_hit_bytes > 0


def test_serial_critical_path_sums_and_shared_resource_cuts_bound():
    graph = PhysicalWorkGraph((
        PhysicalWorkNode("load", hbm_ms=10.0),
        PhysicalWorkNode("compute", cube_ms=20.0, predecessors=("load",)),
    ))
    result = critical_path_roofline(
        graph, RooflineTargets(cube=0.8, vector=0.8, hbm=0.8, link=0.8))
    assert math.isclose(result.critical_path_ideal_ms, 30.0)
    assert result.physical_ideal_ms >= 30.0
    assert result.target_duration_ms >= result.physical_ideal_ms


def test_link_utilization_is_raw_and_reports_violations():
    result = link_utilization({"link:a": 120.0}, {"link:a": 100.0}, 1.0)
    assert math.isclose(result.maximum, 1.2)
    assert result.violations == ("link:a",)


def test_exposed_hbm_formula_is_not_mfu():
    assert estimate_exposed_hbm(
        10.0, 8.0, 4.0, cube_cover_fraction=0.5,
        vector_cover_fraction=0.5) == 4.0


def test_matrix_traffic_separates_cold_and_refill_and_split_partials():
    hierarchy = MemoryHierarchySpec(134217728, 0.75, 128, True, {})
    geometry = MatrixGeometry(32, 32, 7168)
    schedule = MatrixSchedule(32, 32, 256, 16, 16, 16, 28, 1, 1, 2, 2, "gm")
    traffic = plan_matrix_traffic(op(shape=(32, 32, 7168)), geometry, schedule, hierarchy)
    assert traffic.hbm_read_bytes == 32 * 7168 * 2 + 7168 * 32 * 2
    assert traffic.partial_bytes > 0
    assert traffic.private_refill_bytes >= traffic.hbm_read_bytes


def test_kda_summary_publishes_state_io_and_vector_work():
    params = Params.parse({
        "lengths": (1,), "heads": 4, "key_heads": 2,
        "dk": 128, "dv": 128, "dtype": "bf16", "state_dtype": "fp32",
    })
    from mindie_sim.prediction.mechanism import Geometry as PlanGeometry
    geometry = PlanGeometry(32, {"vector": 64, "mte2_aiv": 64,
                                "mte2_aic": 32, "mte3": 64},
                            {"ub": 253952}, {"ub": 64})
    work = summarize_work(params, geometry)
    assert work.resources["hbm"] > 0
    assert work.resources["vector"] > 0
    assert work.resources["mte2_aiv"] > 0
