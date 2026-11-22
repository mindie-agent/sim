"""Paged latent attention work, storage, and pipeline contracts.

[Role] Exercise the governed tensor and physical execution boundary.
[Invariants] Preserve active tensor work, padding, and directional traffic.
[Timeouts & Locks] Bounded local engine runs. No locks.
[Side-Effect Policy] Read retained cases and fixtures. No device access.
"""
import json
import pickle
from collections import Counter
from dataclasses import replace
from math import ceil
from pathlib import Path

import numpy as np
import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.hardware import sku_case
from _support.operator_cases import paged_mla_attention, paged_mla_case
from mindie_sim import target_graphs
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import paged_mla as governed
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.matrix.refill_steady import PeriodicPipelineSpec, compact_periodic_peer_group
from mindie_sim.prediction.operators.paged_mla import DSL_MQS_ABI, FIA_ABI, FLASH_ABI, MQS_ABI, paged_mla
from mindie_sim.prediction.program import PipelineProgram, PipelineTemplateProgram, RecipeContext, Stage
from mindie_sim.prediction.results import FusedPipelineDescriptor, OpOccurrence
from mindie_sim.prediction.streams import StreamOwners
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0

GM_SKU = REPRESENTATIVE_SKUS["a3"]
DIRECT_SKU = REPRESENTATIVE_SKUS["a5"]
LATENT, ROPE, DQK, RECORD, PAGE = 512, 64, 576, 1152, 128
READY = 7
CAPTURES = Path(__file__).resolve().parents[2] / "examples/cases/high-concurrency-20261030"


def _execute(operator, plan, geometry, *, hardware_rates=False, copy_bytes=0, ready_ms=READY):
    units, pools = geometry.resource_units, geometry.buffers
    rates = dict.fromkeys(units, 1e8) | {"cube": 1}
    if hardware_rates:
        rates.update((role, rate / 1000) for role, rate in geometry.rates.items())
    peaks = geometry.compute_peaks if hardware_rates else {"bf16": 1e12}
    ctx = RecipeContext(plan.work.apply(operator), {}, {r: r for r in units}, rates, geometry.compute_units,
                        resource_units=units, compute_peaks=peaks, pool_names={p: p for p in pools},
                        pool_capacities=pools, pool_units=geometry.buffer_units)
    engine = Engine([Resource(r, rates[r], units=n) for r, n in units.items()],
                    pools=pools, pool_units=geometry.buffer_units, trace_limit=16000)
    program = plan.lower(ctx)
    done = program.start(engine, "mla", engine.submit("input", release=ready_ms))
    expected = ctx.bind(plan.work)
    if copy_bytes:
        engine.submit("hbm_copy", {"hbm": copy_bytes}, release=ready_ms + 0.020)
        expected["hbm"] += copy_bytes
    engine.advance(horizon=ready_ms)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert {r: n for r, n in engine.serviced.items() if n} == pytest.approx(expected)
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return program.stages[1].program, engine

def _governed_geom(sku):
    return Geometry.from_devices(sku_case(sku).machine.dies)

def _tiles(query_group, split=0):
    split_group = query_group.stages[0].program
    for wave in range(split_group.count):
        for stage in split_group.tile(wave):
            if stage.metadata["split"] == split:
                return [task for clock in range(stage.program.count) for task in stage.program.tile(clock)]
    raise AssertionError(f"split {split} was not dispatched")

def _phase(stages, name):
    return sorted((stage for stage in stages if stage.name.partition(":")[0] == name),
                  key=lambda stage: stage.metadata["kv_tile"])

def _one_owner(geometry):
    aic, aivs = geometry.datapath.aic_aiv_pairs[0]
    return replace(geometry, stream_owners=StreamOwners((aic,), aivs, 1, len(aivs)))

@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("s2", [96, 160])
def test_governed_dense_work_survives_serialization(hardware, s2):
    geometry = _governed_geom(hardware)
    direct = hardware == DIRECT_SKU
    abi = FLASH_ABI if direct else FIA_ABI
    keys, logical, physical = 2 * s2 + 3, 3, 4
    contract = governed.PagedMlaContract(((1, keys),), logical, physical, abi=abi, s2_base_size=s2)
    operator = governed.make_op(contract)
    assert (contract.dqk, contract.dv, governed.latent_record_bytes(contract)) == (DQK, LATENT, RECORD)
    assert operator.flops == 2 * keys * logical * (DQK + LATENT)
    assert operator.local_shape == (1, logical, DQK) and operator.padded_shape == (1, physical, DQK)
    with pytest.raises(ModelingError, match="physical_heads_below_logical"):
        governed.make_op(replace(contract, physical_heads=2))
    occurrence = OpOccurrence(0, 0, 0, operator.op_id)
    ctx = OperatorPlanningContext(occurrence, geometry, geometry.datapath)
    result = governed.register_family().resolve(operator, ctx)
    plan, descriptor = result.recipe, result.descriptor
    assert isinstance(descriptor, FusedPipelineDescriptor)
    assert (descriptor.owner, descriptor.member_op_ids, descriptor.implementation) == (occurrence, (operator.op_id,), abi)
    restored, restored_desc = pickle.loads(pickle.dumps((plan, descriptor)))
    assert restored_desc == descriptor and descriptor.sources and restored.work == plan.work
    totals = Counter()
    for item in (item for phase in descriptor.phases for item in phase.work):
        totals[item.resource] += item.amount
    assert dict(totals) == plan.work.resources | {"cube": 2 * keys * physical * (DQK + LATENT)}
    program, engine = _execute(operator, restored, geometry)
    key_tile = 112 if direct else s2
    tiles = ceil(keys / key_tile)
    stages = _tiles(program.tile(0)[0].program)
    assert [s.metadata["key_logical_bytes"] for s in _phase(stages, "kv_load")] == [
        min(key_tile, keys - i * key_tile) * RECORD for i in range(tiles)]
    assert sum(s.metadata.get("value_logical_bytes", 0) for s in _phase(stages, "pv_load")) == (
        0 if direct else keys * LATENT * 2)
    if direct:
        assert plan.metadata["buffer_depths"] == (1, 1, 2) and plan.metadata["resident_latent"]
    else:
        assert (plan.metadata["task_record_count"], plan.metadata["workspace_depth"]) == (3, 2)
        for name, delay in (("qk", 0), ("softmax", 1), ("pv", 1), ("output_update", 2)):
            assert [(s.metadata["clock"], s.metadata["task_record_idx"], s.metadata["workspace_buf_idx"])
                    for s in _phase(stages, name)] == [(i + delay, i % 3, i % 2) for i in range(tiles)]
        assert all(u == (0,) for e in engine.trace for role, u in e["placement"].items() if role in {"vector", "mte2_aiv", "mte3"})

@pytest.mark.parametrize("abi", [FLASH_ABI, MQS_ABI])
def test_workspace_strategy_rejects_direct_only_abis(abi):
    operator = governed.make_op(sequences=((1, 1),), logical_heads=1, physical_heads=1,
                                abi=abi, n_ori=int(abi == MQS_ABI))
    with pytest.raises(UnmodeledCapabilityError, match="paged_mla_unsupported_abi"):
        governed.build(operator, _governed_geom(GM_SKU))

@pytest.mark.parametrize("abi", [FIA_ABI, FLASH_ABI])
def test_direct_routes_are_required_and_serviced(abi):
    geometry = _governed_geom(DIRECT_SKU)
    operator = governed.make_op(sequences=((1, 17),), logical_heads=3, physical_heads=4, abi=abi)
    plan = governed.build(operator, geometry)
    for src, dst, key in (("l0c", "ub", "matrix_output_path"), ("ub", "l1", "probability_path")):
        route = geometry.datapath.route(src, dst).path
        assert route.name == plan.metadata[key] and plan.work.resources[route.unit] > 0
        datapath = replace(geometry.datapath, usable_paths=tuple(
            p for p in geometry.datapath.usable_paths if (p.source, p.target) != (src, dst)))
        with pytest.raises(UnmodeledCapabilityError, match=f"route_unsupported:{src}->{dst}"):
            governed.build(operator, replace(geometry, datapath=datapath))
    assert geometry.datapath.route("l0c", "ub").path.unit == "fixpipe"

@pytest.mark.parametrize("keys,splits", [(0, 8), (3, 8), (225, 2), (337, 4)])
def test_split_softmax_state_bytes_and_stable_merge(keys, splits):
    operator = governed.make_op(sequences=((1, keys),), logical_heads=2, physical_heads=4,
                                return_lse=True, abi=FLASH_ABI, s2_base_size=3, splits=splits)
    geometry = _governed_geom(DIRECT_SKU)
    plan = governed.build(operator, geometry)
    program, _ = _execute(operator, plan, geometry)
    finish = {s.name: s for s in program.tile(0)[0].program.stages}
    partials = min(splits, ceil(keys / 112))
    if partials < 2:
        assert set(finish) == {"splits", "normalize", "store"}
        assert bool(finish["normalize"].work) == bool(keys)
        assert finish["splits"].program.count == partials
        assert bool(plan.work.matrix) == bool(keys)
        return
    assert finish["merge_read"].metadata["vec2_split_state_logical_bytes"] == partials * 2 * (LATENT + 2) * 4
    assert [finish[k].after for k in ("merge", "normalize", "store")] == [("merge_read",), ("merge",), ("normalize",)]
    assert finish["store"].metadata["lse_logical_bytes"] == 2 * 4
    # Use the split-max representation and include an empty partial in the merge.
    values = np.random.default_rng(412).normal(size=(keys, LATENT))
    scores = np.linspace(-800, 800, keys)
    states = [(-np.inf, 0, np.zeros(LATENT))]
    for indices in np.array_split(np.arange(keys), partials):
        maximum = scores[indices].max()
        weights = np.exp(scores[indices] - maximum)
        states.append((maximum, weights.sum(), weights @ values[indices]))
    m = max(state[0] for state in states)
    z, v = (sum(np.exp(peak - m) * payload[i] for peak, *payload in states) for i in (0, 1))
    weights = np.exp(scores - scores.max())
    np.testing.assert_allclose(v / z, weights @ values / weights.sum(), atol=1e-12)
    assert m + np.log(z) == pytest.approx(scores.max() + np.log(weights.sum()))

@pytest.mark.parametrize("mode,row_bytes,encoding", [(1, 608, "bf16_g64"), (2, 584, "e8m0_g64")])
def test_mixed_quant_payload_and_branch_local_validation(mode, row_bytes, encoding):
    contract = governed.PagedMlaContract(((1, 5),), 1, 1, abi=MQS_ABI, page_size=2,
        n_ori=(2,), n_cmp=(3,), ori_indices=((3, 0),), cmp_indices=((0, 4, 2),),
        ori_block_table=((8, 4),), cmp_block_table=((1, 7, 9),), quant_mode=mode,
        kv_layout="PA_BBND", scale_encoding=encoding)
    operator = governed.make_op(contract)
    geometry = _governed_geom(DIRECT_SKU)
    plan = governed.build(operator, geometry)
    assert plan.metadata["scale_encoding"] == encoding
    program, _ = _execute(operator, plan, geometry)
    load, = _phase(_tiles(program.tile(0)[0].program), "kv_load")
    assert (load.metadata["ori_logical_bytes"], load.metadata["cmp_logical_bytes"]) == (2 * row_bytes, 3 * row_bytes)
    assert load.metadata["logical_bytes"] == 5 * row_bytes + (2 + 3 + 2 + 3) * 4
    metadata_sectors = (5 + 2) * 32  # Five page records and two index spans.
    if mode == 2:
        for branch, rows in (("ori", 2), ("cmp", 3)):
            assert load.metadata[f"{branch}_data_logical_bytes"] == rows * 576
            assert load.metadata[f"{branch}_scale_logical_bytes"] == rows * 8
        # Even physical pages align data. Odd pages start 16 B into a sector.
        assert load.work["hbm"] == 2 * 576 + 3 * (576 + 32) + 5 * 32 + metadata_sectors
        assert load.work["mte2_aiv"] == 5 * (576 + 32) + metadata_sectors
    else:
        assert load.work["hbm"] == load.work["mte2_aiv"] == 5 * 608 + metadata_sectors
    for branch in ("ori", "cmp"):
        for field, value, error in ((f"n_{branch}", (-1,), f"invalid_n_{branch}"),
                                    (f"{branch}_indices", ((0,),), f"{branch}_index_count"),
                                    (f"{branch}_block_table", ((0,),), f"{branch}_block_table_coverage")):
            with pytest.raises(ModelingError, match=error):
                governed.make_op(replace(contract, **{field: value}))
    with pytest.raises(ModelingError, match="invalid_scale_encoding"):
        governed.make_op(replace(contract, scale_encoding=("fp32", "fp16_g16")))

@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_contiguous_inline_scale_kv_reads_records_without_page_tables(layout):
    geometry = _governed_geom(DIRECT_SKU)
    op = governed.make_op(sequences=((1, 2),), logical_heads=4, physical_heads=4, abi=MQS_ABI,
                          kv_layout=layout, n_ori=1, n_cmp=1, scale_encoding=("bf16_g64", "bf16_g64"))
    program, _ = _execute(op, governed.build(op, geometry), geometry)
    load, = _phase(_tiles(program.tile(0)[0].program), "kv_load")
    assert (load.metadata["ori_logical_bytes"], load.metadata["cmp_logical_bytes"]) == (608, 608)
    assert load.work == {"mte2_aiv": 2 * 608, "hbm": 2 * 608}

@pytest.mark.parametrize("layout,mode,row_bytes,repeated_hbm,contiguous_hbm", [
    ("PA_BBND", 1, 608, 1280, 1280),
    ("PA_BBND", 2, 584, 1280, 1248),
    ("TND", 1, 608, 1248, 1248),
])
@pytest.mark.parametrize("second_index", [0, 1])
def test_sparse_gather_charges_repeats_and_coalesces_contiguous_rows(
    layout, mode, row_bytes, repeated_hbm, contiguous_hbm, second_index,
):
    geom = _governed_geom(DIRECT_SKU)
    op = governed.make_op(sequences=((1, 2),), logical_heads=1, physical_heads=1, abi=MQS_ABI,
                          kv_layout=layout, page_size=2, n_ori=2, quant_mode=mode, ori_indices=((0, second_index),))
    program, engine = _execute(op, governed.build(op, geom), geom)
    load, = _phase(_tiles(program.tile(0)[0].program), "kv_load")
    expected_hbm = contiguous_hbm if second_index else repeated_hbm
    # Contiguous scales can share a GM sector while each scale still takes a padded UB row.
    assert load.work == {"hbm": expected_hbm, "mte2_aiv": repeated_hbm}
    assert load.metadata["ori_logical_bytes"] == 2 * row_bytes
    assert load.metadata["dram_sector_bytes"] == expected_hbm
    assert sum(row["demand"]["hbm"] for row in engine.trace if ":kv_load:" in row["task"]) == expected_hbm

@pytest.mark.parametrize("abi,mode,attributes,error", [
    (MQS_ABI, 1, {"scale_encoding": "bf16_g32"}, "invalid_scale_encoding"),
    (MQS_ABI, 1, {"scale_encoding": "e8m0_g64"}, "invalid_scale_encoding"),
    (MQS_ABI, 1, {"kv_layout": "PA_BLK_NZ"}, "invalid_kv_layout"),
    (MQS_ABI, 2, {"scale_encoding": "bf16_g64"}, "invalid_scale_encoding"),
    (MQS_ABI, 2, {"scale_encoding": ("e8m0_g64", "e8m0_g32")}, "invalid_scale_encoding"),
    (MQS_ABI, 2, {"kv_layout": "BSND"}, "invalid_kv_layout"),
    (MQS_ABI, 2, {"kv_layout": "TND"}, "invalid_kv_layout"),
    (MQS_ABI, 2, {"page_size": 1025}, "invalid_page_size"),
    (MQS_ABI, 2, {"rope_dim": 32}, "requires_rope_dim"),
    (DSL_MQS_ABI, 1, {"kv_layout": "TND"}, "invalid_kv_layout"),
    (DSL_MQS_ABI, 2, {"scale_encoding": "e8m0_g64"}, "invalid_quant_mode"),
])
def test_sparse_cache_packing_rejects_cross_abi_or_cross_mode_attributes(abi, mode, attributes, error):
    with pytest.raises(ModelingError, match=error):
        governed.make_op(sequences=((1, 2),), logical_heads=64, physical_heads=64,
                         abi=abi, quant_mode=mode, n_ori=1, n_cmp=1, **attributes)

def test_block_scale_plane_uses_physical_page_boundary():
    geometry = _governed_geom(DIRECT_SKU)
    op = governed.make_op(sequences=((1, 2),), logical_heads=4, physical_heads=4, abi=MQS_ABI,
                          quant_mode=2, n_ori=2, page_size=5, ori_block_table=((3,),))
    program, _ = _execute(op, governed.build(op, geometry), geometry)
    load, = _phase(_tiles(program.tile(0)[0].program), "kv_load")
    # 3 * (5 * 584) + 5 * 576 starts eight bytes before the next sector.
    # The adjacent 8 B scales cross it, as does the 1152 B data run.
    assert load.work["hbm"] == (2 * 576 + 32) + 2 * 32 + 32
    assert load.work["mte2_aiv"] == 2 * (576 + 32) + 32

@pytest.mark.parametrize("splits,intervals", [(1, ((0, 1, 2),)), (2, ((0, 1), (2,)))])
def test_workspace_pv_accumulates_within_each_owner_interval(splits, intervals):
    geometry = _one_owner(_governed_geom(GM_SKU))
    operator = governed.make_op(sequences=((1, 1025),), logical_heads=4, physical_heads=4,
                                s2_base_size=512, splits=splits)
    plan = governed.build(operator, geometry)
    program, engine = _execute(operator, plan, geometry)
    query = program.tile(0)[0].program
    pv_bytes = 4 * LATENT * 4
    for slot, tiles in enumerate(intervals):
        stages = _tiles(query, slot)
        writes = _phase(stages, "partial")
        assert [s.metadata["kv_tile"] for s in writes] == list(tiles)
        assert {s.metadata["bn2_idx_in_cur_core"] for s in writes} == {slot}
        assert {s.metadata["mm2_slot"] for s in writes} == {slot}
        assert [s.metadata.get("atomic_read_bytes", 0) for s in writes] == [0] + [pv_bytes] * (len(tiles) - 1)
        assert [s.work["hbm"] for s in writes] == [pv_bytes] + [2 * pv_bytes] * (len(tiles) - 1)
        for rescale, update, previous, current in zip(
            _phase(stages, "atomic_rescale")[1:], _phase(stages, "atomic_write")[1:], writes, writes[1:]
        ):
            assert rescale.work["vector"] > 0 and previous.name in rescale.previous
            assert rescale.name in update.after and update.name in current.after
            assert update.metadata["mm2_offset_bytes"] == current.metadata["mm2_offset_bytes"]
            assert update.work == {"mte3": pv_bytes, "hbm": 2 * pv_bytes}
        reads = [s for s in _phase(stages, "output_read") if s.work]
        assert len(reads) == 1 and reads[0].metadata["kv_tile"] == tiles[-1]
        assert reads[0].metadata["mm2_offset_bytes"] == writes[-1].metadata["mm2_offset_bytes"]
    assert sum(e["demand"].get("fixpipe", 0) for e in engine.trace) == plan.work.resources["fixpipe"]

def test_workspace_accumulator_rotation_skips_empty_intervals():
    geometry = _one_owner(_governed_geom(GM_SKU))
    operator = governed.make_op(sequences=((1, 513), (1, 0), (1, 1025), (1, 17)),
                                logical_heads=4, physical_heads=4)
    program, _ = _execute(operator, governed.build(operator, geometry), geometry)
    for wave, (interval, tiles) in {0: (0, 2), 2: (1, 3), 3: (2, 1)}.items():
        writes = _phase(_tiles(program.tile(wave)[0].program), "partial")
        assert len(writes) == tiles
        assert {s.metadata["bn2_idx_in_cur_core"] for s in writes} == {interval}
        assert {s.metadata["mm2_slot"] for s in writes} == {interval % 2}
        assert len({s.metadata["mm2_offset_bytes"] for s in writes}) == 1

@pytest.mark.parametrize("heads,keys", [(8, 17), (3, 241), (1, 241)])
def test_direct_fixpipe_rounds_rows_and_publishes_each_panel(heads, keys):
    geometry = _one_owner(_governed_geom(DIRECT_SKU))
    operator = governed.make_op(sequences=((1, keys),), logical_heads=heads, physical_heads=heads, abi=FLASH_ABI)
    plan = governed.build(operator, geometry)
    program, engine = _execute(operator, plan, geometry)
    stages = _tiles(program.tile(0)[0].program)
    score_stages = _phase(stages, "score_store")
    padded_rows = 2 * ceil(heads / 2)
    expected = [padded_rows * 32 * ceil(min(112, keys - start) / 8) for start in range(0, keys, 112)]
    if (heads, keys) == (8, 17):
        assert expected == [768]
    assert [s.work["fixpipe"] for s in score_stages] == expected
    fixpipe_bytes = sum(expected)
    by_task = {event["task"]: event for event in engine.trace}
    for panel in range(2):
        publications = _phase(stages, f"partial{panel}")
        releases = _phase(stages, f"release{panel}")
        for tile, (publish, release) in enumerate(zip(publications, releases, strict=True)):
            assert (publish.metadata["mm2_slot"], publish.metadata["mm2_generation"]) == (panel, tile)
            assert (publish.metadata["column_start"], publish.metadata["columns"]) == (panel * 256, 256)
            assert publish.work["fixpipe"] == padded_rows * 256 * 4
            assert publish.name in release.previous
            acknowledgements = release.program.stages
            assert tuple(s.name for s in acknowledgements) == tuple(f"aiv{aiv}" for aiv in (0, 1))
            assert all(s.keep_empty for s in acknowledgements)
            fixpipe_bytes += padded_rows * 256 * 4
            if tile:
                assert releases[tile - 1].name in publish.after
                ready = next(e for name, e in by_task.items()
                             if name.endswith(f":tile{tile + 1}:{publish.name}"))
                for ack in acknowledgements:
                    prior = next(e for name, e in by_task.items()
                                 if name.endswith(f":tile{tile + 1}:{releases[tile - 1].name}:program:{ack.name}"))
                    assert ready["start_ms"] >= prior["end_ms"] - 1e-12
                    if tile > 1 and ack.work:
                        sibling = int(ack.name.removeprefix("aiv"))
                        assert ack.placement["vector"] == (sibling,)
    assert plan.work.resources["fixpipe"] == fixpipe_bytes
    assert plan.work.matrix == {"bf16": 2 * heads * keys * (DQK + LATENT)}

@pytest.mark.parametrize("keys,return_lse", [(17, True), (0, True), (0, False)])
def test_dsl_mixed_quant_binds_fp32_sinks_and_seeds_even_empty_selection(keys, return_lse):
    geometry = _governed_geom(DIRECT_SKU)
    heads = 64
    oris = min(keys, 3)
    contract = governed.PagedMlaContract(((1, keys),), heads, heads, abi=DSL_MQS_ABI,
                                        n_ori=oris, n_cmp=keys - oris, return_lse=return_lse)
    operator = governed.make_op(contract)
    assert dict(operator.metadata)["paged_mla_contract"]["sinks"] == ("fp32", (heads,))
    plan = governed.build(operator, geometry)
    program, engine = _execute(operator, plan, geometry)
    load, attention = program.stages
    assert load.metadata["sinks_logical_bytes"] == 256
    assert load.work == {"mte2_aiv": 256, "hbm": 256}
    assert attention.after == (load.name,)
    group = attention.program.tile(0)[0].program
    phases = _tiles(group) if keys else group.stages
    seed, = (s for s in phases if s.name.partition(":")[0] == "sink_init" and s.work)
    assert seed.work == {"vector": heads * (LATENT + 2)}
    if keys:
        softmax = _phase(phases, "softmax")[0]
        assert seed.name in softmax.after
        ori, cmp = _phase(phases, "kv_load")
        assert ori.metadata["ori_logical_bytes"] == oris * 544
        assert cmp.metadata["cmp_logical_bytes"] == (keys - oris) * 320
        assert governed.mqsmla_payload_bytes(oris, keys - oris) == oris * 544 + (keys - oris) * 320
    else:
        assert not plan.work.matrix
        assert not any("qk:" in e["task"] or "pv:" in e["task"] for e in engine.trace)
        assert plan.work.resources["hbm"] == 256 + heads * (2 * LATENT + 4 * return_lse)

@pytest.mark.parametrize("encoding", ["e8m0_g32", ("e8m0_g32", "bf16_g16"), ("bf16_g64", "bf16_g64"),
                                      ("bf16_g32", "fp16_g16"), ("bf16_g16", "bf16_g32")])
def test_dsl_mixed_quant_rejects_incompatible_bank_scales(encoding):
    with pytest.raises(ModelingError, match="invalid_scale_encoding"):
        governed.make_op(sequences=((1, 2),), logical_heads=64, physical_heads=64, abi=DSL_MQS_ABI,
                         n_ori=1, n_cmp=1, scale_encoding=encoding)

@pytest.mark.parametrize("sinks", [("bf16", (64,)), ("fp32", (1, 64)), ("fp32", (32,))])
def test_dsl_mixed_quant_rejects_incompatible_sink_tensors(sinks):
    with pytest.raises(ModelingError, match="invalid_sinks"):
        governed.make_op(sequences=((1, 0),), logical_heads=64, physical_heads=64,
                         abi=DSL_MQS_ABI, sinks=sinks)

def test_dsl_mixed_quant_cost_partition_reduces_only_the_split_row():
    geometry = _governed_geom(DIRECT_SKU)
    heads = 64
    operator = governed.make_op(sequences=((1, 8 * PAGE + 1), (1, 0)), logical_heads=heads, physical_heads=heads,
                                abi=DSL_MQS_ABI, n_ori=(PAGE, 0), n_cmp=(7 * PAGE + 1, 0), splits=2)
    plan = governed.build(operator, geometry)
    program, _ = _execute(operator, plan, geometry)
    *loads, attention = program.stages
    assert all(load.metadata["sinks_logical_bytes"] == 256 for load in loads)
    first, empty = (group.program for group in attention.program.tile(0))
    # The cheaper tail belongs to the second core. The native cut is 4/5 tiles.
    slot_bytes = heads * (LATENT + 2) * 4
    for split, offsets in enumerate((range(4), range(4, 9))):
        stages = _tiles(first, split)
        assert [s.metadata["kv_tile"] for s in _phase(stages, "kv_load")] == list(offsets)
        assert sum(s.metadata.get("vec2_split_state_logical_bytes", 0)
                   for s in _phase(stages, "split_write")) == slot_bytes
        assert sum(bool(s.work) for s in _phase(stages, "sink_init")) == int(split == 0)
    finish = {stage.name: stage for stage in first.stages}
    assert finish["merge_read"].metadata["vec2_split_state_logical_bytes"] == 2 * slot_bytes
    assert finish["merge_read"].after == ("splits",)
    assert finish["merge"].after == ("merge_read",) and finish["merge"].work["vector"] > 0
    assert next(s for s in empty.stages if s.name == "sink_init").work["vector"] == 64 * (LATENT + 2)

def test_dsl_mixed_quant_repeated_rows_keep_native_selective_boundaries():
    geometry = _governed_geom(DIRECT_SKU)
    heads, rows = 64, 72
    operator = governed.make_op(sequences=((rows, 5 * PAGE),), logical_heads=heads, physical_heads=heads,
                                abi=DSL_MQS_ABI, n_ori=PAGE, n_cmp=4 * PAGE, splits=32)
    plan = governed.build(operator, geometry)
    program, _ = _execute(operator, plan, geometry)
    attention = program.stages[-1].program
    queries = (stage.program for wave in range(attention.count) for stage in attention.tile(wave))
    for index, query in enumerate(queries):
        cut = {2: 1, 4: 2, 6: 3}.get(index % 9)
        expected = (range(5),) if cut is None else (range(cut), range(cut, 5))
        assert query.stages[-1].metadata["nonempty_splits"] == len(expected)
        for split, tiles in enumerate(expected):
            assert [s.metadata["kv_tile"] for s in _phase(_tiles(query, split), "kv_load")] == list(tiles)

@pytest.mark.parametrize("nested", [False, True])
def test_short_sparse_queries_use_their_own_banks_without_split_state(nested):
    geometry = _governed_geom(DIRECT_SKU)
    heads = 64
    selections = dict(n_ori=[1, 0], n_cmp=[2, 1], ori_indices=[[1], []], cmp_indices=[[0, 1], [2]])
    if nested:
        selections = {name: [rows] for name, rows in selections.items()}
    inputs = dict(sequences=[[2, 3]], logical_heads=heads, physical_heads=heads, abi=DSL_MQS_ABI,
                  page_size=2, ori_block_table=[[3]], cmp_block_table=[[4, 7]], splits=2, **selections)
    operator = governed.make_op(inputs)
    plan = governed.build(operator, geometry)
    slot_bytes = heads * (LATENT + 2) * 4
    assert plan.metadata["workspace_slots_per_core"] == 2
    assert plan.metadata["workspace_bytes_per_core"] == 2 * slot_bytes
    assert plan.metadata["workspace_bytes"] == len(geometry.datapath.aic_aiv_pairs) * 2 * slot_bytes
    assert operator.flops == plan.work.matrix["bf16"] == 2 * 4 * heads * (LATENT + LATENT)
    program, engine = _execute(operator, plan, geometry)
    attention = program.stages[-1].program
    first, last = (group.program for group in attention.tile(0))
    first_tiles = _tiles(first)
    ori, cmp = (s.metadata for s in _phase(first_tiles, "kv_load"))
    tail = _phase(_tiles(last), "kv_load")[0].metadata
    assert (ori["ori_logical_bytes"], cmp["cmp_logical_bytes"], tail["cmp_logical_bytes"]) == (544, 2 * 320, 320)
    assert (ori["ori_indices_logical_bytes"], cmp["cmp_indices_logical_bytes"], tail["cmp_indices_logical_bytes"]) == (4, 8, 4)
    for group in (first, last):
        assert [stage.name for stage in group.stages] == ["splits", "normalize", "store"]
        assert group.stages[-1].metadata["nonempty_splits"] == 1
        assert all(not s.work for s in _phase(_tiles(group), "split_write"))
    assert not any(event["demand"] for event in engine.trace
                   if "split_write" in event["task"] or "merge_read" in event["task"])
    assert plan.work == governed.build(governed.make_op(inputs | {"splits": 1}), geometry).work
    with pytest.raises(ModelingError, match="cmp_block_table_coverage:0:1"):
        governed.make_op(inputs | {"cmp_block_table": [[4]]})

@pytest.mark.parametrize("layout", ["PA_BLK_BSND", "PA_BLK_NZ", "NZ_LATENT"])
def test_latent_alias_avoids_value_refill_with_exact_tail_bytes(layout):
    geometry = _one_owner(_governed_geom(DIRECT_SKU))
    keys = 2 * PAGE + 1
    contract = governed.PagedMlaContract(((1, keys),), 8, 8, abi=FIA_ABI, kv_layout=layout,
                                        block_table=((5, 1, 11),), splits=2)
    for alias in (True, False):
        operator = governed.make_op(replace(contract, latent_alias=alias))
        plan = governed.build(operator, geometry)
        program, _ = _execute(operator, plan, geometry)
        query = program.tile(0)[0].program
        stages = _tiles(query, 0) + _tiles(query, 1)
        keys_read = sum(s.metadata["key_logical_bytes"] for s in _phase(stages, "kv_load"))
        values_read = sum(s.metadata.get("value_logical_bytes", 0) for s in _phase(stages, "pv_load"))
        assert keys_read == keys * RECORD
        assert values_read == (0 if alias else keys * LATENT * 2)
        assert governed.latent_record_bytes(replace(contract, latent_alias=alias)) == RECORD + (
            0 if alias else LATENT * 2)

def test_sparse_page_aliases_preserve_each_request_gather():
    geometry = _governed_geom(DIRECT_SKU)
    operator = governed.make_op(sequences=((1, 3), (1, 2)), logical_heads=4, physical_heads=4,
                                abi=MQS_ABI, page_size=2, n_ori=(2, 1), n_cmp=(1, 1),
                                ori_indices=((0, 2), (2,)), cmp_indices=((1,), (1,)),
                                ori_block_table=((1, 1), (1, 1)), cmp_block_table=((7,), (7,)))
    program, _ = _execute(operator, governed.build(operator, geometry), geometry)
    for query, n_ori in zip(program.tile(0), (2, 1), strict=True):
        gather, = _phase(_tiles(query.program), "kv_load")
        assert gather.metadata["ori_logical_bytes"] == n_ori * 608
        assert gather.metadata["cmp_logical_bytes"] == 608
        assert gather.metadata["ori_pages_logical_bytes"] == 4 * n_ori
        assert gather.work["hbm"] > 0

@pytest.mark.parametrize("rows,captured", [(12, False), (48, False), (12, True)])
def test_capture_rows_and_caller_allocations_remain_distinct(rows, captured):
    suffix = "88_c8" if rows == 48 else "30_c2"
    raw = json.loads((CAPTURES / "glm78-capture-rows" / f"a3_glm53_w4a8_tp16_tpot{suffix}.json").read_text())
    if captured:
        raw["runtime"]["graph_capture_keys"] = [16]
    active = 16 if captured else rows
    geometry = _governed_geom(GM_SKU)
    op = governed.make_op(sequences=((6, 4096),) * (rows // 6), logical_heads=4, physical_heads=4)
    capacity = (active, *op.launch_shape[1:])
    captured_op = replace(op, padded_shape=capacity, launch_shape=capacity)
    assert governed.build(captured_op, geometry).work == governed.build(op, geometry).work
    graph = target_graphs(raw, phase="decode", tokens=6, sequences=rows // 6, context=4096,
                          scope="body", apply_body_defaults=False)[0]
    prologs = [dict(o.metadata)["fusion_tensor_view"] for o in graph if dict(o.metadata).get("fusion_role") == "mla_prolog"]
    assert Counter(p["query_norm_output"] for p in prologs) == {False: 57, True: 21}
    assert all((p["tokens"], p["token_scale_shape"]) == (active, (active, 1)) for p in prologs)
    ledgers = [dict(o.metadata)["routing_realization"] for o in graph if o.recipe == "balanced_grouped_gate_up"]
    assert len(ledgers) == 75 and all(sum(d["global_expert_rows"]) == active * 8 for d in ledgers)
    assert all(o.local_shape[0] == active for o in graph if o.family == "router" or ".shared.gate_up" in o.op_id)
    assert [o.local_shape[0] for o in graph if o.family == "allgather"] == [-(-active // 16) * 16] * 75
    concats = [dict(o.metadata)["input_shapes"] for o in graph if dict(o.metadata).get("native_kind") == "ConcatD"]
    padded = active == 12
    assert Counter(map(tuple, concats)) == ({((12, 6144), (4, 6144)): 75, ((12, 256), (4, 256)): 75} if padded else {})
    assert sum(o.recipe == "mc2_tp_output_view" for o in graph) == 75 * padded
    for o in graph:
        m = dict(o.metadata)
        if padded and o.family == "alltoallv" and o.op_id.endswith(".dispatch") and not m.get("source_token_mask", {}).get("basis", "").startswith("source_"):
            assert not any(map(any, m["peer_traffic"]["payload_bytes"][active:]))

@pytest.mark.parametrize("dtype,width", [("bf16", 2), ("fp32", 4)])
def test_router_payload_width_is_independent_of_projection_compute(dtype, width):
    raw = json.loads((CAPTURES / "kimi-variants/a5_kimi_tp8_e16_bf16_c32_p4k.json").read_text())
    raw["model"]["layers"] = 4
    raw["runtime"]["kernels"].pop("lowerings", None)
    raw["model"]["layer_overrides"]["3"]["router_output_dtype"] = dtype
    graph = target_graphs(raw, phase="decode", tokens=1, sequences=32, context=4096)[0]
    ops = {o.op_id: o for o in graph}
    router, scores = ops["s0.l3.router"], ops["s0.l3.router_scores"]
    meta = dict(router.metadata)
    assert {meta[k] for k in ("weight_format", "activation_format", "compute_format")} == {router.dtype} == {"bf16"}
    assert meta["output_format"] == dtype and router.mte3_bytes == scores.mte2_bytes == 4 * 16 * width

@pytest.mark.parametrize("demands,arrival,expected", [
    (((0.24, 0.30), (0.60, 3.25)), None, (22.780, 209.2)),
    (((0.25, 0.50), (0.25, 1.00)), 16, (37.75, 64.500)),
])
def test_asymmetric_refill_peers_stay_literal(demands, arrival, expected):
    ctx = RecipeContext(None, {}, {"hbm": "hbm", "cube": "cube"}, {"hbm": 1, "cube": 2}, 2,
                        resource_units={"cube": 2})
    def tile(peer, hbm, cube):
        return lambda _: (Stage("load", {"hbm": hbm}),
                          Stage("compute", {"cube": cube}, after=("load",), placement={"cube": (peer,)}))
    def run(programs):
        engine = Engine((Resource("hbm", 1), Resource("cube", 2, units=2)), backend="python")
        if arrival is not None:
            engine.submit("external", {"hbm": 96}, release=arrival)
        ready = engine.submit("ready")
        ends = tuple(pipe.start(engine, f"peer{peer}", ready) for peer, pipe in enumerate(programs))
        engine.advance(until=ends, first=False)
        return tuple(end.time for end in ends)
    specs = tuple(PeriodicPipelineSpec(64, 2, tile(peer, *demand)) for peer, demand in enumerate(demands))
    selected = compact_periodic_peer_group(ctx, specs)
    assert all(isinstance(pipe, PipelineProgram) for pipe in selected)
    times = run(selected)
    assert times == run(spec.literal() for spec in specs)
    assert times == pytest.approx(expected, rel=1e-12)

@pytest.mark.parametrize("context,splits,copy_bytes", [(409_600, 32, 0), (16_384, 2, 13_700_000)])
def test_paged_pipeline_matches_literal_timing(monkeypatch, context, splits, copy_bytes):
    raw = paged_mla_case(splits=splits)
    raw["runtime"]["max_model_len"] = context + PAGE
    op, _ = paged_mla_attention(raw, context=context)
    geometry = _governed_geom(DIRECT_SKU)
    plan = paged_mla.build(op, geometry)
    options = {"hardware_rates": True, "copy_bytes": copy_bytes, "ready_ms": 0}
    program, fast = _execute(op, plan, geometry, **options)
    for producer in program.stages[1].program.programs:
        pipe = producer.tile(0)[0].program.stages[1].program
        assert isinstance(pipe, PipelineTemplateProgram)
        for phase in pipe.template.phases:
            assert not any(stage.program for group in (phase.stages, *phase.residue_stages) for stage in group)
    monkeypatch.setattr(paged_mla, "_template_pipeline", lambda _, pipe: pipe)
    monkeypatch.setattr(paged_mla, "_flat_cycle_stage", lambda stage: (stage,))
    _, literal = _execute(op, plan, geometry, **options)
    assert fast.now == pytest.approx(literal.now, rel=1e-12)
    if copy_bytes:
        fast_copy, literal_copy = (next(row["end_ms"] for row in engine.trace if row["task"] == "hbm_copy")
                                   for engine in (fast, literal))
        assert fast_copy == pytest.approx(literal_copy, rel=1e-12)
    else:
        assert fast.advance_steps < 500 < literal.advance_steps
