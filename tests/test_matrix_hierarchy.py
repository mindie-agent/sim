"""Independent byte and bottleneck controls for shared versus private traffic."""
import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.program import RecipeContext


def execute(m, n, k, *, tile=(1, 1, 1), reuse=True, mte2_rate=100,
            hbm_rate=100, quant="bf16", cores=2):
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=m, n=n, k=k, quant=QuantSpec("bf16") if quant == "bf16" else quant_preset(quant),
                 category="dense_mlp", phase="prefill")
    rates = {"cube": 100, "hbm": hbm_rate, "mte2": mte2_rate, "mte3": 100}
    if op.vector_ops:
        rates["vector"] = 100
    names = {name: name for name in rates}
    engine = Engine([Resource(name, rate, units=1 if name == "hbm" else cores)
                     for name, rate in rates.items()])
    context = RecipeContext(
        op, {"cube": op.flops, **({"vector": op.vector_ops} if op.vector_ops else {})},
        names, rates, cores,
        resource_units={"cube": cores, "hbm": 1, "mte2": cores, "mte3": cores, "vector": cores},
    )
    done = TiledMatmul(*tile, slots=2, reuse_inputs=reuse, reuse_weights=reuse).lower(
        context).start(engine, "matrix", engine.submit("ready"))
    engine.advance(until=(done,))
    return engine, done.time


def test_shared_hbm_hits_preserve_each_core_private_operand_copy():
    shared, _ = execute(1, 2, 1)
    uncached, _ = execute(1, 2, 1, reuse=False)
    # BF16 A is2B and B is4B. Each of two owners needs its own2B A
    # and2B B, even if the shared cache fetches A from HBM only once.
    assert shared.serviced == {"cube": 4, "hbm": 10, "mte2": 8, "mte3": 4}
    assert uncached.serviced == {"cube": 4, "hbm": 12, "mte2": 8, "mte3": 4}


def test_shared_cache_cannot_hide_a_private_copy_bandwidth_bottleneck():
    shared, shared_ms = execute(1, 2, 1, mte2_rate=2)
    uncached, uncached_ms = execute(1, 2, 1, reuse=False, mte2_rate=2)
    # Four local bytes per owner at1B/ms, then4FLOP/100 and4B/100.
    assert shared_ms == uncached_ms == pytest.approx(4.08)
    assert shared.serviced["hbm"] < uncached.serviced["hbm"]
    _, memory_limited_shared = execute(1, 2, 1, hbm_rate=2)
    _, memory_limited_uncached = execute(1, 2, 1, reuse=False, hbm_rate=2)
    assert memory_limited_shared == pytest.approx(5.04)
    assert memory_limited_uncached == pytest.approx(6.04)


def test_tail_tiles_duplicate_operands_only_on_the_tiled_axis():
    shared, _ = execute(3, 5, 7, tile=(2, 3, 4))
    uncached, _ = execute(3, 5, 7, tile=(2, 3, 4), reuse=False)
    # Two M tiles and two N tiles, including both tails. Splitting K into
    # panels adds no whole-operand duplicate in this declared tile model.
    activation, weight, output = 3 * 7 * 2, 5 * 7 * 2, 3 * 5 * 2
    local_reads = 2 * activation + 2 * weight
    assert [shared.serviced["mte2"], uncached.serviced["mte2"]] == pytest.approx([local_reads] * 2)
    assert shared.serviced["hbm"] == pytest.approx(activation + weight + output)
    assert uncached.serviced["hbm"] == pytest.approx(local_reads + output)
    assert [shared.serviced["cube"], uncached.serviced["cube"]] == pytest.approx([2 * 3 * 5 * 7] * 2)


def test_packed_weights_and_activation_scales_remain_in_both_memory_boundaries():
    shared, _ = execute(2, 4, 3, tile=(1, 2, 3), quant="w8a8_int8")
    uncached, _ = execute(2, 4, 3, tile=(1, 2, 3), reuse=False, quant="w8a8_int8")
    # INT8 A6B + two FP32 activation scales8B; B12B plus two FP32
    # metadata entries per output channel32B; BF16 output16B.
    activation, packed_weight, output = 14, 44, 16
    assert shared.serviced["hbm"] == activation + packed_weight + output
    assert shared.serviced["mte2"] == 2 * activation + 2 * packed_weight
    assert uncached.serviced["hbm"] == 2 * activation + 2 * packed_weight + output
    assert shared.serviced["mte2"] == uncached.serviced["mte2"]
    assert shared.serviced["vector"] == uncached.serviced["vector"] == 4 * 2 * 4
