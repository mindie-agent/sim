"""Finite matrix tail, traffic and live-buffer arithmetic oracles."""
import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.program import RecipeContext


def run(*, reuse, slots, capacity=4096):
    op = gemm_op(op_id="m", deps=(), stage=0, layer=0, requests=(), m=9, n=7, k=5,
                 quant=QuantSpec("bf16"), category="attn", phase="prefill")
    names = {name: name for name in ("cube", "hbm", "mte2", "mte3")}
    rates = {"cube": 100, "hbm": 1000, "mte2": 1000, "mte3": 1000}
    engine = Engine([Resource(key, rate, units=2) for key, rate in rates.items()],
                    pools={"l1": capacity, "l0c": 256}, pool_units={"l1": 2, "l0c": 2})
    context = RecipeContext(op, {"cube": op.flops, "hbm": op.hbm_bytes,
                                "mte2": op.mte2_bytes, "mte3": op.mte3_bytes},
                            names, rates, 2, pool_names={"l1": "l1", "l0c": "l0c"})
    complete = TiledMatmul(4, 4, 3, slots, reuse, reuse).lower(context).start(
        engine, "m", engine.submit("start"))
    engine.advance(until=(complete,))
    return engine, complete.time


def test_matrix_tails_conserve_flops_and_explicit_operand_reuse_changes_only_traffic():
    reused, _ = run(reuse=True, slots=2)
    reloaded, _ = run(reuse=False, slots=2)
    assert reused.serviced["cube"] == reloaded.serviced["cube"] == 2 * 9 * 7 * 5
    assert reused.serviced["hbm"] == 2 * (9 * 5 + 7 * 5 + 9 * 7)
    assert reloaded.serviced["hbm"] == 2 * (9 * 5 * 2 + 7 * 5 * 3 + 9 * 7)
    assert reused.allocated == {"l1": 0, "l0c": 0}
    assert reused.peak_allocated["l0c"] > 0


def test_matrix_panel_capacity_serializes_loads_without_changing_work():
    wide, wide_ms = run(reuse=False, slots=2)
    narrow, narrow_ms = run(reuse=False, slots=2, capacity=48)
    assert narrow.serviced == pytest.approx(wide.serviced)
    assert narrow_ms > wide_ms
    assert narrow.peak_allocated["l1"] <= 48
    assert narrow.summary()["live_programs"] == 0
