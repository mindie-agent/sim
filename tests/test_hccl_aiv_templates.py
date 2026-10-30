"""Repeated AIV copy templates retain real atomic work and per-call tails."""
from dataclasses import asdict

from mindie_sim.prediction.hccl_aiv import build, UB_BATCH, UB_RESIDENCY
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.operators import OpSpec


def test_large_reduce_reuses_full_batches_without_merging_atomic_and_copy_tails():
    full_batches, tail, ranks, units = 1_000_000, 32, 4, 64
    payload = full_batches * UB_BATCH + tail
    selection = select({'expansion': 'AIV', 'template': 'mesh_oneshot', 'block_limit': units},
                       collective='allreduce', ranks=ranks, payload_bytes=payload)
    raw = {'selection': asdict(selection), 'rank': 0, 'phase': 'reduce_output'}
    op = OpSpec('reduce', 'hccl_local', 'prefill', 0, 0, (), (), (), (), (),
                metadata=(('hccl_phase', raw),))
    roles = {'hbm': 1, 'mte2_aiv': units, 'mte3': units, 'vector': units}
    geometry = Geometry(units, roles, buffers={'ub': 256 * 1024})
    plan = build(op, geometry)
    context = RecipeContext(op, {}, {role: role for role in roles},
                            {role: 1000 * count for role, count in roles.items()}, units,
                            resource_units=roles, pool_names={'ub': 'ub'})
    owner, = plan.program(context).stages
    pipeline = owner.program
    assert pipeline.count == ranks * (full_batches + 1)
    assert pipeline.slots == 1
    assert owner.acquire == owner.free
    assert owner.acquire[0].units == (ranks,)
    assert owner.acquire[0].size == UB_RESIDENCY
    copy, copy_tail = pipeline.tile(0), pipeline.tile(full_batches)
    atomic = pipeline.tile(full_batches + 1)
    atomic_tail = pipeline.tile(2 * (full_batches + 1) - 1)
    assert pipeline.tile(full_batches // 2) is copy
    assert pipeline.tile(2 * (full_batches + 1)) is atomic
    assert pipeline.tile(pipeline.count - 1) is atomic_tail
    for stages, amount, factor in ((copy, UB_BATCH, 1), (copy_tail, tail, 1),
                                   (atomic, UB_BATCH, 2), (atomic_tail, tail, 2)):
        read, write = stages
        assert read.work == {'hbm': amount, 'mte2_aiv': amount}
        assert write.work == {'hbm': factor * amount, 'mte3': amount}
        assert write.after == ('read',)
        assert read.placement['mte2_aiv'] == write.placement['mte3'] == (ranks,)
    # One initial copy followed by R-1 read-modify-write copies. UB framing
    # never adds payload bytes; it only determines the preserved tile count.
    assert plan.work.resources == {
        'hbm': payload * (2 + 3 * (ranks - 1)),
        'mte2_aiv': ranks * payload, 'mte3': ranks * payload}
