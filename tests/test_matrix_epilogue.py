"""Independent fused scaling arithmetic and output-dependency controls."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul
from mindie_sim.prediction.matrix_mapping import mapping_bound
from test_matrix_mapping import setup


def quantized_context(*, m=6, n=5, k=6, groups=2, cores=3, rates=None):
    _, context = setup(m=m, n=n, k=k, cores=cores)
    weights, inputs, output = groups * n * k, m * k, m * n * 2
    row_scales, column_scales = 4 * m, 4 * groups * n
    metadata = dict(context.op.metadata)
    metadata.update(
        compute_format="int8", output_format="bf16", matrix_groups=groups,
        packed_weight_bytes=weights + column_scales,
        activation_scale_bytes=row_scales,
        matrix_epilogue={
            "kind": "row_column_scale", "operations_per_output": 4,
            "row_scale_bytes": row_scales, "column_scale_bytes": column_scales,
        })
    reads = inputs + weights + row_scales + column_scales
    op = replace(context.op, dtype="int8", vector_ops=4 * m * n,
                 hbm_bytes=reads + output, mte2_bytes=reads, mte3_bytes=output,
                 metadata=tuple(metadata.items()))
    rates = rates or {"cube": 128, "vector": 32, "hbm": 1024,
                      "mte2": 64, "mte1": 128, "mte3": 128}
    context = replace(context, op=op, work={"cube": op.flops, "vector": op.vector_ops},
                      rates=rates, resource_names={name: name for name in rates},
                      resource_units={name: 1 if name == "hbm" else cores for name in rates})
    engine = Engine([Resource(name, rate, units=context.resource_units[name])
                     for name, rate in rates.items()], pools=context.pool_capacities,
                    pool_units={name: cores for name in context.pool_capacities})
    return engine, context


@pytest.mark.parametrize("tile_k", [2, 6])
def test_scale_traffic_is_output_work_not_a_k_operand(tile_k):
    engine, context = quantized_context()
    recipe = HierarchicalMatmul(4, 4, tile_k, 2, 2)
    done = recipe.lower(context).start(engine, "quantized", engine.submit("ready", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    # Two independent [3,6]@[6,5] matrices: 36B A, 60B B,
    # 24B row scales, 40B column scales and 60B BF16 output.
    # The N tail duplicates A/row-scales on MTE2 but never over K.
    assert engine.serviced == pytest.approx({
        "cube": 360, "vector": 120, "hbm": 220,
        "mte2": 220, "mte1": 228, "mte3": 60,
    })
    assert mapping_bound(context, recipe) <= done.time - 7
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_epilogue_waits_for_all_k_contributions_and_precedes_store():
    rates = {"cube": 32, "vector": 8, "hbm": 1e12,
             "mte2": 1e12, "mte1": 1e12, "mte3": 1e12}
    engine, context = quantized_context(m=2, n=2, k=4, groups=1, cores=1, rates=rates)
    done = HierarchicalMatmul(2, 2, 2, 2, 2).lower(context).start(
        engine, "quantized", engine.submit("ready"))
    engine.advance(horizon=0.99)
    assert engine.serviced["vector"] == engine.serviced["mte3"] == 0
    engine.advance(horizon=2.9)
    assert engine.serviced["mte3"] == 0
    engine.advance(until=(done,))
    assert done.time == pytest.approx(3, abs=1e-8)
    assert engine.serviced["hbm"] == 40  # No INT32 accumulator spill to HBM.


@pytest.mark.parametrize("tile_k", [2, 6])
def test_split_copy_owners_serve_operands_and_final_scales_on_existing_pipes(tile_k):
    rates = {"cube": 128, "vector": 32, "hbm": 1024, "mte2_aic": 64,
             "mte2_aiv": 32, "mte1": 128, "mte3": 128}
    _, context = quantized_context(rates=rates)
    units = {**context.resource_units, "mte2_aiv": 6}
    context = replace(context, resource_units=units)
    engine = Engine([Resource(name, rate, units=units[name]) for name, rate in rates.items()],
                    pools=context.pool_capacities,
                    pool_units={name: 3 for name in context.pool_capacities})
    recipe = HierarchicalMatmul(4, 4, tile_k, 2, 2)
    done = recipe.lower(context).start(engine, "split", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced == pytest.approx({
        "cube": 360, "vector": 120, "hbm": 220, "mte2_aic": 132,
        "mte2_aiv": 88, "mte1": 228, "mte3": 60,
    })
    assert set(engine.resources) == set(rates)
    assert mapping_bound(context, recipe) <= done.time
    assert not any(engine.allocated.values())


def test_shared_operand_and_scale_pipe_sums_both_demands_in_mapping_bound():
    rates = {"cube": 1e12, "vector": 1e12, "hbm": 1e12,
             "mte2": 1, "mte1": 1e12, "mte3": 1e12}
    engine, context = quantized_context(m=1, n=1, k=1, groups=1, cores=1, rates=rates)
    recipe = HierarchicalMatmul(1, 1, 1, 1, 1)
    done = recipe.lower(context).start(engine, "shared", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced["mte2"] == 10  # A+B=2B, two FP32 scales=8B.
    assert mapping_bound(context, recipe) == 10
    assert done.time == pytest.approx(10, abs=1e-8)
