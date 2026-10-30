"""Independent SCFA source phases, hand ledgers and ordinary Engine checks."""
from dataclasses import replace
from itertools import islice
import tracemalloc

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.sharedkv_attention import (
    _phases, build, source_tiles,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def _shape(query=32, context=0, window=128, ratio=4, topk=512, **changes):
    shape = dict(query=query, context=context, window=window,
                 cmp_ratio=ratio, topk=topk, heads=8, width=512, cp=1, pcp=1,
                 branch="SCFA", query_layout="TND", cache_layout="PA_ND",
                 sequences=1, views=1, kv_heads=1, query_dtype="bf16",
                 cache_dtype="bf16", output_dtype="bf16", sinks_dtype="fp32",
                 sparse_block_size=1, owner_schedule="metadata_cost_balanced",
                 metadata_s2_split=False, return_softmax_lse=False,
                 selected_rows="valid_prefix")
    return {**shape, **changes}


def _op(**kwargs):
    return OpSpec("sharedkv", "attention", "prefill", 0, 0, (), (), (), (), (),
                  metadata=(("sharedkv_attention_geometry", _shape(**kwargs)),))


def _run(op, *, ready_at=0, cores=4, policy="resident", rate_scale=1,
         busy_store_unit=None, trace_limit=0, store_rate=None):
    rates = {"hbm": 1e8, "vector": 1e8, "mte2_aic": 1e8,
             "mte2_aiv": 1e8, "mte1": 1e8, "mte3": 1e8,
             "fixpipe": 1e8, "l2": 1e9, "cube": 1e12}
    rates = {role: value * rate_scale for role, value in rates.items()}
    if store_rate is not None:
        rates["mte3"] = store_rate
    units = {"hbm": 1, "l2": 1, "cube": cores, "vector": 2 * cores,
             "mte2_aic": cores, "mte2_aiv": 2 * cores, "mte1": cores,
             "mte3": 2 * cores, "fixpipe": cores}
    pools, pool_units = {"l2": 64 * 1024**2}, {"l2": 1}
    service = {role + "_cached": (role, 2e8 * rate_scale)
               for role in ("mte2_aic", "mte2_aiv")}
    service.update(l2_read=("l2", 2e9 * rate_scale),
                   l2_write=("l2", 1e9 * rate_scale))
    geometry = Geometry(cores, units, pools, pool_units)
    plan = build(op, geometry, scratch_policy=policy)
    engine = Engine([Resource(role, rate, units=units[role])
                     for role, rate in rates.items()],
                    pools=pools, pool_units=pool_units, trace_limit=trace_limit)
    context = RecipeContext(
        plan.work.apply(op), {}, {role: role for role in rates}, rates, cores,
        resource_units=units, compute_peaks={"bf16": 1e13},
        pool_names={"l2": "l2"}, pool_capacities=pools,
        pool_units=pool_units, service_modes=service)
    ready = engine.submit("ready", release=ready_at)
    if busy_store_unit is not None:
        engine.submit("unrelated_mte3", {"mte3": 1e12},
                      placement={"mte3": (busy_store_unit,)})
    done = plan.lower(context).start(engine, "sharedkv", ready)
    return plan, engine, context, done, ready


def test_one_query_has_one_combined_first_and_last_tile():
    tiles = tuple(source_tiles(1, 4096, 128, 4, 512, 0, 1))
    assert tiles == (("ori", 128, True, False), ("cmp", 512, False, True))
    # One final output, written only by the last combined tile.
    phases = dict(_phases("cmp", 512, 8, 512, first=False, last=True,
                          scratch_policy="resident"))
    assert phases["accum_store"].resources["mte3"] == 8 * 512 * 2
    assert "output" not in phases and "update" not in phases


def test_window_and_selected_rows_are_two_typed_data_regions():
    ori = dict(_phases("ori", 128, 8, 512, first=True, last=False,
                       scratch_policy="resident"))
    cmp = dict(_phases("cmp", 512, 8, 512, first=False, last=True,
                       scratch_policy="resident"))
    assert not any(name.startswith(("gather", "publish")) for name in ori)
    assert ori["qk_load"].resources["hbm"] == (128 + 8) * 512 * 2
    assert ori["pv_load"].resources["hbm"] == 128 * 512 * 2
    # Compressed rows use top-k and page-table scalar entries. These do not
    # become private AIV DMA bytes or an invented scalar-response delay.
    for aiv in range(2):
        assert cmp[f"gather{aiv}"].resources == {
            "hbm": 256 * 512 * 2 + 256 * 8, "mte2_aiv": 256 * 512 * 2}
        assert cmp[f"publish{aiv}"].resources["mte3"] == 256 * 512 * 2


def test_scfa_vector_accumulator_uses_one_previous_region_and_one_output():
    first = dict(_phases("ori", 128, 8, 512, first=True, last=False,
                        scratch_policy="resident"))
    middle = dict(_phases("cmp", 512, 8, 512, first=False, last=False,
                          scratch_policy="resident"))
    last = dict(_phases("cmp", 512, 8, 512, first=False, last=True,
                        scratch_policy="resident"))
    # First tile reads only the current PV workspace and writes the vector
    # accumulator. Middle reads current plus prior accumulator. Last reads
    # current plus prior and writes only the final BF16 output.
    assert first["accum_read"].resources["mte2_aiv_cached"] == 8 * 512 * 4
    assert middle["accum_read"].resources["mte2_aiv_cached"] == 2 * 8 * 512 * 4
    assert last["accum_read"].resources["mte2_aiv_cached"] == 2 * 8 * 512 * 4
    assert middle["accum_store"].resources["mte3"] == 8 * 512 * 4
    assert last["accum_store"].resources["mte3"] == 8 * 512 * 2
    assert last["accum_store"].resources["hbm"] == 8 * 512 * 2
    assert "l2_write" not in last["accum_store"].resources


def test_softmax_has_one_aiv_private_read_not_aic_plus_aiv():
    phases = dict(_phases("cmp", 512, 8, 512, first=True, last=False,
                          scratch_policy="resident"))
    softmax = phases["score_read"].resources
    assert softmax["mte2_aiv_cached"] == 8 * 512 * 4
    assert "mte2_aic" not in softmax and "mte2_aic_cached" not in softmax
    assert "mte2_aiv" not in softmax


def test_offchip_minus_resident_is_each_scratch_byte_exactly_once():
    resident, _, _, _, _ = _run(_op(query=8))
    offchip, _, _, _, _ = _run(_op(query=8), policy="offchip")
    assert resident.work.matrix == offchip.work.matrix
    assert resident.work.resources["mte2_aic_cached"] > 0
    assert resident.work.resources["mte2_aiv_cached"] > 0
    assert "mte2_aic_cached" not in offchip.work.resources
    assert "mte2_aiv_cached" not in offchip.work.resources
    scratch = (resident.work.resources.get("l2_read", 0)
               + resident.work.resources.get("l2_write", 0))
    assert (offchip.work.resources["hbm"]
            - resident.work.resources["hbm"]) == scratch


def test_branch_dispatch_is_scfa_only_and_requires_explicit_owner_scope():
    geometry = Geometry(4, {role: 4 for role in
                            ("cube", "mte1", "fixpipe", "mte2_aic", "mte2_aiv")},
                        {"l2": 1 << 30}, {"l2": 1})
    for kwargs in (dict(ratio=0, topk=0), dict(ratio=4, topk=0),
                   dict(owner_schedule=None), dict(sparse_block_size=8),
                   dict(branch="SWA"), dict(branch="CFA"),
                   dict(metadata_s2_split=True), dict(return_softmax_lse=True),
                   dict(views=2), dict(sequences=2), dict(kv_heads=2),
                   dict(sinks_dtype=None), dict(heads=2), dict(topk=2048)):
        with pytest.raises(ModelingError, match="source_geometry_outside_reviewed_domain"):
            build(_op(**kwargs), geometry)
    with pytest.raises(ModelingError, match="source_geometry_outside_reviewed_domain"):
        build(replace(_op(), phase="decode"), geometry)


def test_engine_waits_for_inputs_and_releases_finite_per_owner_state():
    plan, engine, _, done, _ = _run(_op(query=8), ready_at=7)
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert done.time >= 7
    assert plan.work.resources["hbm"] > 0
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())


def test_reusing_plan_does_not_consume_its_iterator_or_freeze_resource_rates():
    op = _op(query=8)
    times = []
    for scale in (1, 2, 1):
        _, engine, _, done, _ = _run(op, rate_scale=scale)
        engine.advance(until=(done,))
        times.append(done.time)
    assert times[1] < times[0]
    assert times[2] == pytest.approx(times[0])


def test_source_ownership_has_two_gather_aivs_and_one_vector_aiv():
    _, engine, _, done, _ = _run(_op(query=1, context=4096), cores=1, trace_limit=200)
    engine.advance(until=(done,))
    by_name = {row["task"].split(":")[-1]: row for row in engine.trace}
    assert by_name["gather0"]["placement"]["mte2_aiv"] == (0,)
    assert by_name["gather1"]["placement"]["mte2_aiv"] == (1,)
    assert by_name["publish0"]["placement"]["mte3"] == (0,)
    assert by_name["publish1"]["placement"]["mte3"] == (1,)
    assert by_name["accum_store"]["placement"]["mte3"] == (0,)
    assert by_name["softmax"]["placement"]["vector"] == (0,)


def test_one_query_combined_hand_ledger_is_not_two_output_calls():
    plan, _, _, _, _ = _run(_op(query=1, context=4096, window=128,
                                ratio=4, topk=512), cores=1)
    window = 2 * 128 * 512 * 2         # source QK and PV each read original KV
    selected = 512 * 512 * 2 + 512 * 8
    query_loads = 2 * 8 * 512 * 2       # one per combined S2 tile
    output = 8 * 512 * 2
    sinks = 2 * 8 * 4                 # each paired AIV initializes once
    assert plan.work.resources["hbm"] == window + selected + query_loads + output + sinks


def test_idle_odd_aiv_cannot_supply_even_only_store_work():
    times = []
    for unit in (None, 0, 1):
        # Only 128 compressed rows: all valid rows are on AIV0. AIV1 has no
        # store work, so an unrelated odd-unit store cannot slow this call.
        _, engine, _, done, _ = _run(_op(query=1, context=512), cores=1,
                                    busy_store_unit=unit, store_rate=1e3)
        engine.advance(until=(done,))
        times.append(done.time)
    assert times[1] > times[0]
    assert times[2] == pytest.approx(times[0])


def test_large_logical_query_stream_does_not_materialize_all_tiles():
    tracemalloc.start()
    try:
        stream = source_tiles(10**12, 4096, 128, 4, 512, 0, 32)
        sample = tuple(islice(stream, 3))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert sample == (("ori", 128, True, False), ("cmp", 512, False, True),
                      ("ori", 128, True, False))
    assert peak < 64 * 1024


def test_scratch_storage_and_live_tasks_are_bounded_by_owners_and_slots():
    peaks = []
    for query in (8, 128):
        plan, engine, context, done, _ = _run(_op(query=query))
        engine.advance(until=(done,))
        assert engine.serviced == pytest.approx(context.bind(plan.work))
        assert not any(engine.allocated.values())
        assert engine.summary()["live_programs"] == engine.summary()["live_tasks"] == 0
        peaks.append((engine.peak_tasks, engine.peak_allocated["l2"]))
    assert peaks[0][1] == peaks[1][1]
    assert peaks[1][0] <= 4 * 2 * 22 + 64


def test_compressed_publish_precedes_aic_read_and_final_store_follows_pv():
    _, engine, _, done, _ = _run(_op(query=1, context=4096), cores=1,
                                ready_at=7, trace_limit=200)
    engine.advance(until=(done,))
    rows = list(engine.trace)
    selected = {row["task"].split(":")[-1]: row for row in rows if ":tile1:" in row["task"]}
    original = {row["task"].split(":")[-1]: row for row in rows if ":tile0:" in row["task"]}
    assert selected["qk_load"]["start_ms"] >= max(
        selected[f"publish{aiv}"]["end_ms"] for aiv in range(2))
    for before, after in (("qk_load", "qk_operands"), ("qk_operands", "qk"),
                          ("score_read", "softmax"), ("pv_load", "pv_operands"),
                          ("pv_operands", "pv"), ("pv_store", "accum_read"),
                          ("accum_read", "accum"), ("accum", "accum_store")):
        assert selected[after]["start_ms"] >= selected[before]["end_ms"]
    assert selected["accum_read"]["start_ms"] >= original["accum_store"]["end_ms"]
    assert all(row["start_ms"] >= 7 for row in rows if row["demand"])


def test_early_scfa_query_without_valid_compressed_rows_keeps_one_output():
    plan, engine, context, done, _ = _run(_op(query=3, context=0, ratio=128),
                                        cores=1, trace_limit=200)
    engine.advance(until=(done,))
    rows = list(engine.trace)
    assert not any(":gather" in row["task"] or ":publish" in row["task"] for row in rows)
    # Three independent causal windows contain 1+2+3 rows. Each window is read
    # once for QK and again for PV; Q and final output are three complete tensors.
    expected = 2 * (1 + 2 + 3) * 512 * 2 + 2 * 3 * 8 * 512 * 2 + 2 * 8 * 4
    assert plan.work.resources["hbm"] == expected
    stores = [row for row in rows if row["task"].endswith(":accum_store")]
    assert len(stores) == 3
    assert sum(row["demand"]["hbm"] for row in stores) == 3 * 8 * 512 * 2
    assert engine.serviced == pytest.approx(context.bind(plan.work))


def test_1024_selected_rows_have_one_output_and_bounded_two_slots():
    plan, engine, context, done, _ = _run(_op(query=1, context=8192, topk=1024),
                                        cores=1, trace_limit=200)
    engine.advance(until=(done,))
    stores = [row for row in engine.trace if row["task"].endswith(":accum_store")]
    assert len(stores) == 3
    assert ["hbm" in row["demand"] for row in stores] == [False, False, True]
    # Logical reads remain independent of repeated physical stages.
    expected = (2 * 128 * 512 * 2 + 1024 * (512 * 2 + 8)
                + 3 * 8 * 512 * 2 + 8 * 512 * 2 + 2 * 8 * 4)
    assert plan.work.resources["hbm"] == expected
    assert engine.serviced == pytest.approx(context.bind(plan.work))


def test_metadata_query_ranges_drive_private_owners_without_changing_query_work():
    from collections import Counter

    # Source costs are 16 for q0..2 and 32 for q3..7. The contiguous source
    # partition is [0,3), [3,4), [4,6), [6,8), rather than two queries per core.
    plan, engine, context, done, _ = _run(_op(query=8), cores=4, trace_limit=1000)
    engine.advance(until=(done,))
    final_stores = [row for row in engine.trace
                    if row["task"].endswith(":accum_store") and row["demand"].get("hbm")]
    assert Counter(row["placement"]["mte3"][0] for row in final_stores) == {0: 3, 2: 1, 4: 2, 6: 2}
    serial, _, _, _, _ = _run(_op(query=8), cores=1)
    assert plan.work.matrix == serial.work.matrix
    expected = dict(serial.work.resources)
    # Only initialization changes with the enabled core count. All query
    # phases, including both external KV reads and scratch copies, are equal.
    for role in ("hbm", "mte2_aiv"):
        expected[role] += (4 - 1) * 2 * 8 * 4
    assert plan.work.resources == expected
    assert engine.serviced == pytest.approx(context.bind(plan.work))
