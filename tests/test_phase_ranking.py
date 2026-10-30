"""A real same-work counterexample to treating phase-only ranking as detailed."""
import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul
from mindie_sim.prediction.matrix_phase import PhaseMatmul
from mindie_sim.prediction.program import RecipeContext


def test_serial_phase_and_detailed_overlap_can_rank_the_same_hardware_tradeoff_differently():
    op = gemm_op(
        op_id="matrix", deps=(), stage=0, layer=0, requests=(),
        m=4, n=2, k=40, quant=QuantSpec("bf16"),
        category="dense_mlp", phase="prefill")
    pools = {"l1": 48, "l0a": 16, "l0b": 16, "l0c": 32}
    results = {}
    for candidate, cube_rate, load_rate in (
        ("balanced", 32, 24), ("faster_copy_slower_matrix", 32 / 1.5, 240),
    ):
        rates = {"cube": cube_rate, "hbm": 1e12, "mte2": load_rate,
                 "mte1": 1e12, "mte3": 1e12}
        context = RecipeContext(
            op, {"cube": op.flops}, {name: name for name in rates}, rates, 1,
            pool_names={name: name for name in pools}, pool_capacities=pools,
            resource_units={name: 1 for name in rates})
        for fidelity, recipe in (
            ("detailed", HierarchicalMatmul(4, 2, 2, 2, 2)),
            ("phase", PhaseMatmul(((4, 2, 2),), 2, 2)),
        ):
            engine = Engine([Resource(name, rate) for name, rate in rates.items()],
                            pools=pools)
            done = recipe.lower(context).start(engine, "matrix", engine.submit("ready"))
            engine.advance(until=(done,))
            results[candidate, fidelity] = done.time
            assert engine.serviced == pytest.approx(
                {"cube": 640, "hbm": 496, "mte2": 480, "mte1": 640, "mte3": 16})
            assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    # Twenty panels: A overlaps 1ms refill/1ms compute; B overlaps
    # 0.1ms refill/1.5ms compute. Serial phases instead add all service.
    assert results == pytest.approx({
        ("balanced", "detailed"): 21,
        ("balanced", "phase"): 40,
        ("faster_copy_slower_matrix", "detailed"): 30.1,
        ("faster_copy_slower_matrix", "phase"): 32,
    }, abs=1e-8)
    assert results["balanced", "detailed"] < results["faster_copy_slower_matrix", "detailed"]
    assert results["balanced", "phase"] > results["faster_copy_slower_matrix", "phase"]
