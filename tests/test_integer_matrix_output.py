"""Ordinary dynamic INT8 output work, independent of source-call adapters."""
from dataclasses import replace

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul, HierarchicalMatmul
from mindie_sim.prediction.matrix_mapping import MappedMatmul
from mindie_sim.prediction.matrix_phase import PhaseMatmul
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from mindie_sim.prediction.program import RecipeContext
from test_matrix_mapping import setup


@pytest.mark.parametrize("recipe", [
    TiledMatmul(4, 4, 3),
    HierarchicalMatmul(4, 4, 3, 2, 2),
    MappedMatmul(((4, 4, 3),), 2, 2),
    PhaseMatmul(((4, 4, 3),), 2, 2),
    SteadyMatmul(((4, 4, 3),), 2, 2),
])
def test_integer_output_casts_and_scales_follow_complete_k_reduction(recipe):
    op = gemm_op(
        op_id="w8a8", deps=(), stage=0, layer=0, requests=(),
        m=2, n=3, k=7, quant=QuantSpec("dynamic", weight_format="int8"),
        category="dense_mlp", phase="prefill")
    rates = {"cube": 100, "vector": 8, "hbm": 1000, "mte1": 1000,
             "mte2": 1000, "mte3": 1000}
    context = RecipeContext(op, {"cube": op.flops, "vector": op.vector_ops},
                            {key: key for key in rates}, rates, 1)
    engine = Engine([Resource(key, value) for key, value in rates.items()], trace_limit=1000)
    done = recipe.lower(context).start(engine, "w8a8", engine.submit("ready", release=5))
    engine.advance(until=(done,))
    assert engine.serviced["vector"] == 4 * 2 * 3
    assert engine.serviced["cube"] == 2 * 2 * 3 * 7
    assert engine.serviced["hbm"] == pytest.approx(op.hbm_bytes)
    vector = [row for row in engine.trace if row["demand"].get("vector", 0)]
    matrix = [row for row in engine.trace if row["demand"].get("cube", 0)]
    assert min(row["start_ms"] for row in vector) >= max(row["end_ms"] for row in matrix) - 1e-10
    assert done.time >= 5 + op.vector_ops / rates["vector"]
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_explicit_narrow_mapping_distributes_real_output_extents_over_more_owners():
    _, context = setup(m=32, n=24, k=256, cores=16)
    context = replace(context, rates={key: (1 if key == "cube" else 1e12)
                                     for key in context.rates},
                      pool_capacities={"l1": 512 * 1024, "l0a": 64 * 1024,
                                       "l0b": 64 * 1024, "l0c": 256 * 1024})
    narrow, new = MappedMatmul(narrow_candidates=((16, 16, 64),)).select(context)
    # Disable both small-grid domains to retain the original 128-tile control.
    _, old = MappedMatmul(narrow_candidates=(), underfilled_candidates=()).select(context)
    assert narrow.tile_m == narrow.compute_m == 16
    assert narrow.tile_n == narrow.compute_n == 16
    assert min(row["resource_bound_ms"] for row in new["candidates"]
               if row["resource_bound_ms"] is not None) < min(
        row["resource_bound_ms"] for row in old["candidates"] if row["resource_bound_ms"] is not None)
    # An explicitly provided candidate domain remains exact, even for narrow N.
    _, explicit = MappedMatmul(((32, 32, 64),), 16, 16).select(context)
    assert [row["tile"] for row in explicit["candidates"]] == [[32, 32, 64]]


def test_one_partial_quantization_group_still_has_one_scale_per_output_row():
    common = dict(op_id="tail", deps=(), stage=0, layer=0, requests=(),
                  m=2, n=3, k=64, category="dense_mlp", phase="decode")
    exact = gemm_op(**common, quant=QuantSpec("dynamic", weight_format="int8"))
    tail = gemm_op(**common, quant=QuantSpec(
        "dynamic", weight_format="int8", activation_group_size=128))
    assert dict(tail.metadata)["activation_scale_shape"] == (2,)
    assert dict(tail.metadata)["matrix_epilogue"] == dict(exact.metadata)["matrix_epilogue"]
    assert tail.vector_ops == exact.vector_ops == 24
