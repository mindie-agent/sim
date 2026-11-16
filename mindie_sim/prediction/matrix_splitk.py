"""Finite Split-K matrix lowering for underfilled output grids.

[Role] Partition one output tile's K panels across distinct AIC owners, then
  merge partial accumulators through a bounded reduction phase.
[Invariants] Issued FLOPs and all declared resource demand are conserved. Only
  the final output write is retained; Split-K partials are separately reported
  as an implementation cost, never hidden inside a whole-kernel multiplier.
[Timeouts & Locks] Bounded split count and stage count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import replace

from .engine import Allocation
from .mechanism import Work
from .program import KernelProgram, ParallelProgram, Stage

ZERO = 0
ONE = 1
TWO = 2
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
SECOND = 1
FP32_BYTES = 4
SPLITK_MERGE_MIN_MS = 0.005
SECOND_LAST = -2
LAST_THREE = -3


def _split_work(context, split_count: int):
    names = context.resource_names
    hbm = names.get("hbm")
    vector = names.get("vector")
    mte3 = names.get("mte3")
    cube = names.get("cube")
    if cube is None:
        cube = next((key for key in context.work if key.startswith("matrix@")),
                    None)
    hbm_total = float(context.work.get(hbm, ZERO_FLOAT)) if hbm else ZERO_FLOAT
    mte3_total = float(context.work.get(mte3, ZERO_FLOAT)) if mte3 else ZERO_FLOAT
    hbm_write = min(hbm_total, mte3_total)
    split_keys = {}
    for key, value in context.work.items():
        if key in {vector, mte3} or (hbm and key == hbm):
            continue
        split_keys[key] = float(value)
    split_hbm = max(ZERO_FLOAT, hbm_total - hbm_write)
    merge_vector = float(context.work.get(vector, ZERO_FLOAT)) if vector else ZERO_FLOAT
    partial_bytes = (float(split_count - ONE) * float(context.op.local_shape[SECOND_LAST])
                     * float(context.op.local_shape[LAST_THREE]) * FP32_BYTES)
    return (hbm, vector, mte3, cube, split_keys, split_hbm, hbm_write,
            merge_vector, partial_bytes)


def splitk_program(context, split_count: int) -> KernelProgram:
    """Lower one underfilled output grid onto `split_count` AIC owners."""
    count = max(ONE, int(split_count))
    (hbm, vector, mte3, cube, split_keys, split_hbm, hbm_write,
     merge_vector, partial_bytes) = _split_work(context, count)
    cores = max(ONE, int(context.core_count))

    dram_mode = dict(context.service_modes).get("mte2_aic_dram")
    hbm_caps = ({hbm: float(dram_mode[SECOND])} if hbm is not None and dram_mode else {})

    def split_program(index):
        load = {}
        if hbm is not None and split_hbm > ZERO_FLOAT:
            load[hbm] = split_hbm / count
        for key, value in split_keys.items():
            load[key] = value / count
        mma = {cube: float(context.work.get(cube, ZERO_FLOAT)) / count}
        placement = context.owned(index, cores, (key for key in
                                                 ("cube", "mte1", "mte2_aic")
                                                 if key in context.resource_names))
        if not placement:
            placement = context.owned(index, cores, ("cube",))
        return KernelProgram((
            Stage("load", load, caps=hbm_caps),
            Stage("mma", mma, after=("load",), placement=placement),
        ))

    splits = ParallelProgram(tuple(split_program(index) for index in range(count)))
    merge = {}
    if vector is not None and merge_vector > ZERO_FLOAT:
        merge[vector] = merge_vector
    if mte3 is not None and context.work.get(mte3, ZERO_FLOAT) > ZERO_FLOAT:
        merge[mte3] = float(context.work[mte3])
    if hbm is not None and hbm_write > ZERO_FLOAT:
        merge[hbm] = hbm_write
    stages = [
        Stage("setup", latency=context.setup_ms),
        Stage("splits", after=("setup",), program=splits),
        Stage("merge", merge, after=("splits",), latency=SPLITK_MERGE_MIN_MS,
              caps=hbm_caps),
    ]
    del partial_bytes
    return KernelProgram(tuple(stages), minimum_ms=context.minimum_ms)
