"""Source traffic, finite residency and actual Engine pipeline controls."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.indexed_attention import build
from mindie_sim.prediction.sparse_attention import selected_query_owners
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def run(query=3, context=511, *, policy="resident", compact=True, competing=False,
        memory_rate=1e8, vector_rate=1e8):
    shape = dict(query=query, context=context, topk=2048, window=0,
                 compression_ratio=1, cp=1, heads=8, latent=512, rope=64,
                 query_dtype="bf16", cache_dtype="bf16")
    op = OpSpec("sfa", "attention", "prefill", 0, 0, (), (), (), (), (),
                metadata=(("indexed_attention_geometry", shape),))
    rates = dict(cube=1, vector=vector_rate, hbm=memory_rate, mte2_aic=1e8,
                 mte2_aiv=1e8, mte1=1e8, mte3=1e8, fixpipe=1e8, l2=1e9)
    units = {role: (1 if role in {"hbm", "l2"} else 2) for role in rates}
    units.update(vector=4, mte2_aiv=4, mte3=4)
    pools, pool_units = {"l2": 32 * 1024**2}, {"l2": 1}
    geometry = Geometry(2, units, pools, pool_units)
    plan = build(op, geometry, scratch_policy=policy, compact=compact)
    engine = Engine([Resource(role, rate, units=units[role])
                     for role, rate in rates.items()], pools=pools, pool_units=pool_units)
    ctx = RecipeContext(
        plan.work.apply(op), {}, {role: role for role in rates}, rates, 2,
        resource_units=units, compute_peaks={"bf16": 1e13},
        pool_names={"l2": "l2"}, pool_capacities=pools, pool_units=pool_units,
        service_modes={**{role + "_cached": (role, 2e11)
                         for role in ("mte2_aic", "mte2_aiv")},
                       "l2_read": ("l2", 1e12), "l2_write": ("l2", 1e12)})
    ready = engine.submit("ready", release=7)
    if competing:
        engine.submit("competitor", {"hbm": 2e9}, after=(ready,))
    done = plan.lower(ctx).start(engine, "indexed", ready)
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    return plan.work, engine, done.time - 7


def test_resident_scratch_consumes_pipes_but_not_duplicate_offchip_bytes():
    work, engine, _ = run(query=1, context=0)
    # One selected row: latent+rope BF16, two INT32 indices, Q, output.
    assert work.resources["hbm"] == 1152 + 8 + 8 * 576 * 2 + 8 * 512 * 2
    assert work.resources["mte2_aic_cached"] > 1152
    assert work.resources["mte2_aiv_cached"] > 0
    assert work.resources["fixpipe"] > 0
    assert 0 < engine.peak_allocated["l2"] < 32 * 1024**2
    # The source computes a 16-row matrix even for eight real query heads.
    assert work.matrix["bf16"] == 2 * 16 * 16 * (576 + 512)


def test_offchip_scratch_changes_only_placement_and_external_traffic():
    resident, re, _ = run(query=3, context=511)
    offchip, oe, _ = run(query=3, context=511, policy="offchip")
    assert resident.matrix == offchip.matrix
    assert resident.resources["vector"] == offchip.resources["vector"]
    assert offchip.resources["hbm"] > resident.resources["hbm"]
    assert not oe.peak_allocated["l2"]
    assert re.summary()["live_tasks"] == oe.summary()["live_tasks"] == 0


def test_compact_and_expanded_engine_conserve_physical_service_on_ragged_tiles():
    compact, ce, compact_ms = run(query=5, context=1022)
    expanded, ee, expanded_ms = run(query=5, context=1022, compact=False)
    assert compact == expanded
    assert ce.serviced == pytest.approx(ee.serviced)
    # This is a declared fluid approximation, not asserted exact equivalence.
    assert compact_ms > 0 and expanded_ms > 0
    assert ce.summary()["live_tasks"] == ee.summary()["live_tasks"] == 0


def test_shared_memory_contention_remains_on_the_common_engine():
    _, _, baseline = run(memory_rate=1e6)
    _, _, loaded = run(competing=True, memory_rate=1e6)
    assert loaded > baseline


def test_narrow_heads_cannot_borrow_second_vector_core_or_previous_cached_rates():
    work, _, fast = run(query=1, context=1023)
    _, slow_engine, slow = run(query=1, context=1023, vector_rate=1)
    # Only AIV0 computes both tiles; the four-AIV chip rate is not its supply.
    assert slow >= work.resources["vector"] * 4 * (1 - 1e-12)
    assert slow > fast
    _, _, fast_again = run(query=1, context=1023)
    assert fast_again == pytest.approx(fast)
    assert slow_engine.summary()["live_tasks"] == 0


def test_source_output_is_atomic_until_single_final_vector_read():
    from mindie_sim.prediction.indexed_attention import _phases

    stages = dict(_phases(
        512, 8, 512, 64, first=False, last=False, scratch_policy="resident"))
    # Four independent 128-column products reread the private Q and P
    # operands. Their common GM/L1 reads do not acquire the same factor.
    assert stages["qk_load"].resources["mte1"] == (4 * 16 + 512) * 576 * 2
    assert stages["pv_load"].resources["mte1"] == (4 * 16 + 512) * 512 * 2
    assert stages["softmax"].resources["mte2_aiv_cached"] == 8 * 512 * 4 + 1024
    assert not stages["update"].resources.get("mte2_aiv_cached", 0)
    assert stages["partial"].resources["l2_read"] == 16 * 512 * 4
    assert stages["nupdate"].resources["l2_read"] == 16 * 512 * 4
    final = dict(_phases(
        512, 8, 512, 64, first=False, last=True, scratch_policy="resident"))
    assert final["update"].resources["mte2_aiv_cached"] == 8 * 512 * 4


def _decode_run(query=1, context=4096):
    # Reviewed conditional per-core copy/cache supplies from the GLM
    # indexed-pipeline alternative.  They are an explicit scenario, not a
    # calibration selected from the 90.7 us decode observation.
    rates = dict(hbm=1.2e12, vector=2.112e12, mte2_aic=1.056e12,
                 mte2_aiv=1.056e12, mte1=8.448e12, mte3=2.112e12,
                 fixpipe=8.448e12, l2=3.63e12, cube=2.7e14)
    units = dict(hbm=1, l2=1, cube=20, vector=40, mte2_aic=20,
                 mte2_aiv=40, mte1=20, mte3=40, fixpipe=20)
    pools, pool_units = {"l2": 176160768}, {"l2": 1}
    service = {"mte2_aic_cached": ("mte2_aic", 3.63e12),
               "mte2_aiv_cached": ("mte2_aiv", 3.63e12),
               "l2_read": ("l2", 3.63e12), "l2_write": ("l2", 2.838e12)}
    shape = dict(query=query, context=context, topk=2048, window=0,
                 compression_ratio=1, cp=1, heads=8, latent=512, rope=64,
                 query_dtype="bf16", cache_dtype="bf16")
    op = OpSpec("sfa", "attention", "decode", 0, 0, (), (), (), (), (),
                metadata=(("indexed_attention_geometry", shape),))
    geometry = Geometry(20, units, pools, pool_units)
    plan = build(op, geometry, scratch_policy="resident")
    resources = [Resource(role, rate, units=units[role])
                 for role, rate in rates.items()]
    engine = Engine(resources, pools=pools, pool_units=pool_units)
    ctx = RecipeContext(
        plan.work.apply(op), {}, {role: role for role in rates}, rates, 20,
        resource_units=units, compute_peaks={"bf16": 2.7e14},
        pool_names={"l2": "l2"}, pool_capacities=pools, pool_units=pool_units,
        service_modes=service)
    ready = engine.submit("ready", release=0)
    done = plan.lower(ctx).start(engine, "decode", ready)
    engine.advance(until=(done,))
    return plan.work, done.time


def test_one_row_decode_cannot_use_full_chip_bandwidth():
    work, elapsed = _decode_run()
    # A full-chip envelope would charge only the 2.36 MB gather at 1.2 TB/s.
    # One AIC and its two AIVs serve the task, so the finite copy phase is
    # several times the chip-aggregate bound even after overlap.
    chip_bound_ms = work.resources["hbm"] / 1.2e12 * 1000
    assert selected_query_owners(1, 20) == 1
    assert chip_bound_ms < 0.003
    assert elapsed > 5 * chip_bound_ms


def test_block_sized_topk_is_not_reinterpreted_as_row_count():
    shape = dict(query=1, context=4096, topk=2048, window=0,
                 compression_ratio=1, cp=1, heads=8, latent=512, rope=64,
                 query_dtype="bf16", cache_dtype="bf16", sparse_block_size=8)
    op = OpSpec("sfa", "attention", "decode", 0, 0, (), (), (), (), (),
                metadata=(("indexed_attention_geometry", shape),))
    geometry = Geometry(20, {role: 20 for role in ("cube", "mte1", "fixpipe")},
                        {"l2": 1 << 30}, {"l2": 1})
    with pytest.raises(ModelingError, match="source_geometry_outside_reviewed_domain"):
        build(op, geometry, scratch_policy="resident")
