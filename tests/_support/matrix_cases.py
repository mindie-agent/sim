"""Shared matrix mapping and quantized-epilogue Engine contexts for operator tests.

[Role] Provide single-source Engine + RecipeContext builders for dense/grouped
  matrix tiling, ownership, and row-column scale epilogue tests without
  cross-test imports.
[Invariants] ``matrix_run`` verifies zero work before external release at t=7 ms
  and zero live tasks, programs, or allocations after completion.
[Timeouts & Locks] Pure in-memory Engine execution; no I/O, sleeps, or locks.
[Side-Effect Policy] Mutates only caller-supplied local Engine instances.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.program import RecipeContext

__all__ = ("matrix_run", "matrix_setup", "quantized_context")


def matrix_setup(
    m: int = 8,
    n: int = 8,
    k: int = 6,
    *,
    cores: int = 2,
    pools: dict[str, int] | None = None,
    copy_rate: float = 16,
    records: list[Any] | None = None,
) -> tuple[Engine, RecipeContext]:
    """Construct a controlled multi-core Cube/MTE/HBM Engine and GEMM context."""
    op = gemm_op(
        op_id="matrix",
        deps=(),
        stage=0,
        layer=0,
        requests=(),
        m=m,
        n=n,
        k=k,
        quant=QuantSpec("bf16"),
        category="dense_mlp",
        phase="prefill",
    )
    rates = {"cube": 128, "hbm": 1024, "mte2": copy_rate, "mte1": 1024, "mte3": 1024}
    pools = pools or {"l1": 256, "l0a": 32, "l0b": 32, "l0c": 256}
    engine = Engine(
        [Resource(name, rate, units=1 if name == "hbm" else cores) for name, rate in rates.items()],
        pools=pools,
        pool_units={name: cores for name in pools},
    )
    context = RecipeContext(
        op,
        {"cube": op.flops},
        {name: name for name in rates},
        rates,
        cores,
        pool_names={name: name for name in pools},
        pool_capacities=pools,
        resource_units={name: 1 if name == "hbm" else cores for name in rates},
        record_mapping=records.append if records is not None else None,
    )
    return engine, context


def matrix_run(recipe: Any, **kwargs: Any) -> tuple[Engine, float]:
    """Execute ``recipe`` released at t=7 ms and assert complete resource drain."""
    engine, context = matrix_setup(**kwargs)
    done = recipe.lower(context).start(engine, "matrix", engine.submit("ready", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return engine, done.time - 7


def quantized_context(
    *,
    m: int = 6,
    n: int = 5,
    k: int = 6,
    groups: int = 2,
    cores: int = 3,
    rates: dict[str, float] | None = None,
) -> tuple[Engine, RecipeContext]:
    """Construct an INT8->BF16 row-column-scaled matrix Engine and RecipeContext."""
    _, context = matrix_setup(m=m, n=n, k=k, cores=cores)
    weights, inputs, output = groups * n * k, m * k, m * n * 2
    row_scales, column_scales = 4 * m, 4 * groups * n
    metadata = dict(context.op.metadata)
    metadata.update(
        compute_format="int8",
        output_format="bf16",
        matrix_groups=groups,
        packed_weight_bytes=weights + column_scales,
        activation_scale_bytes=row_scales,
        matrix_epilogue={
            "kind": "row_column_scale",
            "operations_per_output": 4,
            "row_scale_bytes": row_scales,
            "column_scale_bytes": column_scales,
        },
    )
    reads = inputs + weights + row_scales + column_scales
    op = replace(
        context.op,
        dtype="int8",
        vector_ops=4 * m * n,
        hbm_bytes=reads + output,
        mte2_bytes=reads,
        mte3_bytes=output,
        metadata=tuple(metadata.items()),
    )
    rates = rates or {"cube": 128, "vector": 32, "hbm": 1024, "mte2": 64, "mte1": 128, "mte3": 128}
    context = replace(
        context,
        op=op,
        work={"cube": op.flops, "vector": op.vector_ops},
        rates=rates,
        resource_names={name: name for name in rates},
        resource_units={name: 1 if name == "hbm" else cores for name in rates},
    )
    engine = Engine(
        [Resource(name, rate, units=context.resource_units[name]) for name, rate in rates.items()],
        pools=context.pool_capacities,
        pool_units={name: cores for name in context.pool_capacities},
    )
    return engine, context
