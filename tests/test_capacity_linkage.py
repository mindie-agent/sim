"""Per-owner hardware buffer capacities drive regular and grouped tile plans."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
from mindie_sim.prediction.piecewise.families_moe_collective import (
    DEFAULT_L0C_BYTES,
)
from mindie_sim.prediction.subgraphs.state import MoeRoutingState, StateView
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature,
)

RATES = (
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
    ("link_hccs", 2000.0), ("link_roce", 500.0),
)
ACTUAL = PipeSupply(
    RATES,
    buffers=(("l0a", 65536), ("l0b", 65536), ("l0c", 262144),
             ("l1", 524288), ("ub", 253952)),
    pool_units=(("l0a", 32), ("l0b", 32), ("l0c", 32), ("l1", 32), ("ub", 64)),
)
DEFAULT = PipeSupply(RATES)
POLICY = ImplementationPolicy(mt=128, nt=128, kt=128, cores=32,
                              moe_row_tile=128, split_target=10 ** 9,
                              max_splits=1, pipeline_slots=2)


def _grouped(nt: int):
    signature = StaticSignature(
        role="expert_gate_up",
        matrix=(("K", 6144), ("N", 4096)),
        experts=(("E", 8), ("local", 8), ("k", 2)),
        quant=(("act_bytes", 2.0), ("act_format", "bf16"),
               ("weight_format", "bf16"), ("weight_total", 4096 * 6144 * 2.0)),
        tiles=(("cores", 32), ("nt", nt), ("moe_rows", 128)),
        extras=(("row_tile", 128),),
    )
    return BoundOperator(
        OperatorTemplate(name="expert_gate_up", family="grouped_gemm",
                         role="gate_up"), signature)


def _cost(supply: PipeSupply, nt: int):
    geometry = __import__(
        "mindie_sim.prediction.piecewise", fromlist=["WorkloadGeometry"]
    ).WorkloadGeometry.homogeneous(1, 8, 9)
    state = StateView(routing=((0, MoeRoutingState((2,) * 8, 2, 8)),))
    compiled = OperatorPiecewiseCompiler(supply, POLICY).compile_operator(
        _grouped(nt))
    return compiled, compiled.evaluate(geometry, state)


def test_grouped_tile_plan_uses_actual_per_owner_l0c_not_default():
    _, old = _cost(DEFAULT, 256)
    _, actual = _cost(ACTUAL, 256)
    # n_tile is the grouped n-tile count (n_pad / n_tile); the 128 KiB
    # default cannot hold n_tile=256, the actual per-owner 256 KiB can.
    assert old.get("n_tile") == 32
    assert actual.get("n_tile") == 16
    assert old.get("tile_load_bytes") != actual.get("tile_load_bytes")
    assert old.get("flops") == actual.get("flops")
    assert dict(ACTUAL.rates) == dict(DEFAULT.rates)
    # The source nt=128 plan is nonbinding and stays identical.
    _, old_source = _cost(DEFAULT, 128)
    _, actual_source = _cost(ACTUAL, 128)
    assert old_source.get("n_tile") == actual_source.get("n_tile") == 32
    # Constrained actual owners (8192 B each) shrink the grouped plan too.
    constrained = replace(ACTUAL, buffers=tuple(
        (name, 8192) for name, _ in ACTUAL.buffers))
    _, tight_actual = _cost(constrained, 256)
    assert tight_actual.get("n_tile") == 256
    assert tight_actual.get("tile_load_bytes") < old.get("tile_load_bytes")
    assert tight_actual.get("flops") == old.get("flops")

    # A physically impossible owner capacity is reported, not zero-guarded.
    tight = replace(ACTUAL, buffers=tuple(
        (name, 1 if name == "l0c" else value) for name, value in ACTUAL.buffers))
    with pytest.raises(ModelingError, match="tile_plan_capacity_exceeded"):
        _cost(tight, 256)
