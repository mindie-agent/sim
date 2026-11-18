"""Wave 4B `#62` Items 3 & 6: pooled indexer row shard, KeyPool ring,
arch35 quantized lightning indexer and the causal-conv UPDATE ABI.

[Role] Prove the four calibrated programs on the paired `Ascend950PR_958b`
  (1.4 TB/s) and `Ascend950DT_9582` (4.0 TB/s) device profiles, and that the
  caller-side `AscendIndexerKPoolRowShard` byte contract is emitted.
[Invariants] Every expected latency is the paired-SKU measurement quoted by the
  Wave 4 specification; every program is the declared finite `Stage` /
  `PipelineProgram` / `ParallelProgram` graph whose resource demands sum to the
  plan work. No assertion reads a measurement inside a planner.
[Timeouts & Locks] Tier 1: in-memory engines only, no device or file locks.
[Side-Effect Policy] Read-only over the packaged device profiles and
  builder-local records.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from mindie_sim.backends.ascend import profiles
from mindie_sim.catalog.index_pool import pool_selection
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction import family_quant_indexer as fq
from mindie_sim.prediction import gdn_decode_programs as gdn
from mindie_sim.prediction import index_pool
from mindie_sim.prediction import index_pool_native as ipn
from mindie_sim.prediction import sparse_indexer_pipeline as sp
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelineProgram, RecipeBook, RecipeContext,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView


PR = "Ascend950PR_958b"
DT = "Ascend950DT_9582"
TOLERANCE = 0.12
CONV_TOLERANCE = 0.15
HIDDEN = 4096
DECODE_CONTEXT = 131072
INDEX_HEADS = 32
INDEX_WIDTH = 128
INDEX_TOPK = 2048
POOL_ROW_BYTES = 132
INDEX_ROW_BYTES = 136
LOCAL_TP = 8
BATCH = 88
KIMI_BATCH = 96
STEPS = 6
QUERY_ROWS = BATCH * STEPS
LOCAL_QUERY_ROWS = QUERY_ROWS // LOCAL_TP
LOCAL_PAYLOAD = LOCAL_QUERY_ROWS * INDEX_TOPK * 4
AIV_OWNERS = 64

DEVICE = {
    "model": {"name": "wave4b", "layers": 1, "hidden_size": HIDDEN,
              "num_heads": 8, "num_kv_heads": 1, "head_dim": 576,
              "intermediate_size": 1024, "vocab_size": 128},
    "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                               "output_tokens": 1}]},
    "runtime": {"tp": 1, "dp": 1, "ep": 1, "pp": 1,
                "max_num_batched_tokens": 4096, "max_num_seqs": 256},
}


def _machine(sku, devices=1):
    device = deepcopy(profiles()[sku]["device"])
    # The packaged profile already carries every resolved rate, mode and
    # invocation floor; only the instruction preset handle is re-derived.
    device.pop("compute_arch", None)
    links = ([] if devices == 1 else
             [{"name": "ub_mesh", "capacity_gbps": 392, "latency_us": 2.5,
               "members": list(range(devices)), "semantics": "per_pair"}])
    return {"name": sku, "sku": sku, "hardware_key": sku, "arch_family": "A5",
            "num_devices": devices, "ranks_per_node": devices,
            "dies_per_package": 1, "device": device,
            "dies": [{"rank": index, "node": 0, "package": 0}
                     for index in range(devices)],
            "links": links}


def _case(sku, devices=1, *, tp=1):
    raw = deepcopy(DEVICE)
    raw["machine"] = _machine(sku, devices)
    raw["runtime"]["tp"] = tp
    return Case.from_mapping(raw)


def _run(op, build, sku, *, devices=1, tp=1):
    case = _case(sku, devices, tp=tp)
    recipes = RecipeBook({} if build is None
                         else {op.recipe: PlannedRecipe(build)})
    execution = Execution(case, CostModel(case), recipes=recipes,
                          include_submission=False, trace_limit=500000)
    ready = execution.engine.submit("ready", release=0.0)
    step = execution.add(
        IterationWork((op,), IterationPlan(()), 1, 0, 0, None, "decode", "wave4b"),
        0, 0, external_inputs={"ready": ready})
    execution.engine.advance(until=(step.completion,))
    assert not execution.engine.summary()["live_tasks"]
    return execution


def _latency(op, build, sku, **kwargs):
    return _run(op, build, sku, **kwargs).engine.now * 1000.0


def _op(recipe, shape, ledger_key=None, ledger=None, *, family="indexer",
        dtype="fp8_e4m3", metadata=()):
    declared = tuple(metadata)
    if ledger_key is not None:
        declared = ((ledger_key, ledger),) + declared
    return OpSpec(
        op_id="wave4b.op", family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape,
        launch_shape=shape, dtype=dtype, recipe=recipe, metadata=declared)


def _stages(program):
    """Every finite Stage a program graph issues, including pipeline windows."""
    if isinstance(program, KernelProgram):
        for stage in program.stages:
            yield stage
            if stage.program is not None:
                yield from _stages(stage.program)
    elif isinstance(program, PipelineProgram):
        for index in range(program.count):
            for stage in program.tile(index):
                yield stage
                if stage.program is not None:
                    yield from _stages(stage.program)
    elif isinstance(program, ParallelProgram):
        for branch in program.programs:
            yield from _stages(branch)
    else:  # pragma: no cover - a non-program recipe is a modelling regression
        raise AssertionError(f"wave4b_program_not_finite:{type(program)!r}")


def _context(sku):
    device = _case(sku).machine.dies[0]
    rates = dict(device.rates)
    names = {role: f"{role}@r0:s0" for role in rates}
    names["cube"] = "cube@r0:s0"
    units = {role: device.units(role) for role in rates}
    units["cube"] = device.compute_units
    context_rates = {names[role]: rate for role, rate in rates.items()}
    return device, RecipeContext(
        None, {}, names, context_rates, device.compute_units,
        resource_units=units,
        compute_peaks=dict(device.compute_peaks),
        service_modes={mode: (role, rate)
                       for mode, (role, rate) in dict(device.service_modes).items()},
        pool_names={pool: f"{pool}@r0:s0" for pool in dict(device.buffers)},
        pool_units={pool: device.pool_units(pool) for pool in dict(device.buffers)})


def _conserved(plan, op, sku):
    """Every bound unit of plan work appears exactly once in issued stages."""
    device, context = _context(sku)
    prepared = plan.work.apply(op)
    bound = replace(context, op=prepared)
    program = plan.program(bound)
    expected = bound.bind(plan.work)
    issued = {}
    for stage in _stages(program):
        for name, amount in stage.work.items():
            issued[name] = issued.get(name, 0.0) + float(amount)
    assert set(issued) <= set(expected) | {"cube"}
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)
    return program


# --- Item (3): `cann_pool_key_indexer` `quant_mode=0` ----------------------
def _pool_ledger(rows, heads, pools, pool, *, topk=INDEX_TOPK):
    blocks = sp.key_blocks(pools)
    return {
        "query_rows": rows, "score_dtype": "fp8_e4m3",
        "wave_rows": sp.INDEXER_WAVE_ROWS,
        "waves": sp.indexer_waves(rows), "blocks": blocks,
        "heads": heads, "width": INDEX_WIDTH, "pool_size": pool,
        "output_width": topk, "pool_row_bytes": POOL_ROW_BYTES,
        "score_pairs": float(rows * pools * heads),
        "wave_cache_bytes": float(blocks * sp.POOL_KEY_BLOCK * POOL_ROW_BYTES),
        "wave_descale_bytes": float(blocks * heads * sp.POOL_KEY_BLOCK * 4),
        "weight_bytes": float(heads * INDEX_WIDTH + 4 * heads),
        "output_bytes": float(rows * topk * 4),
    }


POOL_CASES = (
    # rows, pool, measured PR, measured DT
    (QUERY_ROWS, 16, 189.00, 152.21),
    (LOCAL_QUERY_ROWS, 16, 45.42, 38.484),
    (LOCAL_QUERY_ROWS, 4, 121.40, 96.730),
)


def _pool_op(rows, pool):
    ledger = _pool_ledger(rows, INDEX_HEADS, DECODE_CONTEXT // pool, pool)
    return _op(ipn.POOL_KEY_RECIPE, (rows, INDEX_HEADS, INDEX_WIDTH),
               sp.POOL_LEDGER, ledger), ledger


def test_pool_key_indexer_wave_program_matches_the_paired_silicon():
    recipe = sp.PoolKeyIndexerRecipe()
    for rows, pool, measured_pr, measured_dt in POOL_CASES:
        op, _ = _pool_op(rows, pool)
        for sku, measured in ((PR, measured_pr), (DT, measured_dt)):
            predicted = _latency(op, recipe.build, sku)
            assert predicted == pytest.approx(measured, rel=TOLERANCE), (
                rows, pool, sku, predicted, measured)


def test_pool_key_indexer_partitions_query_rows_into_three_aic_waves():
    recipe = sp.PoolKeyIndexerRecipe()
    replicated, mirrored = _pool_op(QUERY_ROWS, 16)
    sharded, shard_ledger = _pool_op(LOCAL_QUERY_ROWS, 16)
    assert mirrored["waves"] == 17
    assert shard_ledger["waves"] == 3
    # Wave quantization, not 1/8 scaling: three of seventeen windows remain.
    assert shard_ledger["waves"] / mirrored["waves"] == pytest.approx(3 / 17)
    sharded_dt = _latency(sharded, recipe.build, DT)
    replicated_dt = _latency(replicated, recipe.build, DT)
    assert sharded_dt < replicated_dt * 0.3


def test_pool_key_indexer_is_one_finite_conserving_engine_program():
    op, ledger = _pool_op(LOCAL_QUERY_ROWS, 16)
    plan = sp.PoolKeyIndexerRecipe().build(op, None)
    program = _conserved(plan, op, DT)
    names = [stage.name for stage in _stages(program)]
    for required in ("stream", "tiles", "handoff", "selection"):
        assert required in names
    # One handoff per AIC wave; no per-row envelope and no fitted multiplier.
    assert names.count("handoff") == ledger["waves"]
    assert plan.work.matrix["fp8_e4m3"] == pytest.approx(
        2.0 * LOCAL_QUERY_ROWS * INDEX_HEADS * ledger["blocks"]
        * sp.POOL_KEY_BLOCK * INDEX_WIDTH)


# --- Item (3): `cann_key_pool` two-stage ring -----------------------------
def _key_pool_ledger(pool, completed):
    return {
        "batch": BATCH, "steps": STEPS, "pool_size": pool,
        "width": INDEX_WIDTH, "completed_pools": completed,
        "projection_flops": float(4 * QUERY_ROWS * INDEX_WIDTH * HIDDEN),
        "hidden_bytes": float(QUERY_ROWS * HIDDEN * 2),
        "weight_bytes": float(2 * INDEX_WIDTH * HIDDEN * 2),
    }


def test_key_pool_ring_uses_the_declared_completion_pools_and_silicon():
    for pool, completed in ((16, 0), (4, 1)):
        assert pool_selection(STEPS, 0, INDEX_TOPK, pool,
                              always_select_tail=False).completed_pools == completed
    recipe = sp.KeyPoolRingRecipe()
    for pool, completed, measured_pr, measured_dt in (
            (16, 0, 98.71, 87.849), (4, BATCH, 115.11, 106.301)):
        ledger = _key_pool_ledger(pool, completed)
        op = _op(ipn.KEY_POOL_RECIPE, (QUERY_ROWS, INDEX_WIDTH, HIDDEN),
                 sp.KEYPOOL_LEDGER, ledger, family="key_pool", dtype="bf16")
        plan = recipe.build(op, None)
        assert plan.metadata["completed_pools"] == completed
        assert plan.metadata["ring_steps"] == STEPS
        for sku, measured in ((PR, measured_pr), (DT, measured_dt)):
            predicted = _latency(op, recipe.build, sku)
            assert predicted == pytest.approx(measured, rel=0.10), (
                pool, sku, predicted, measured)
        program = _conserved(plan, op, DT)
        names = {stage.name for stage in _stages(program)}
        assert {"projection", "workspace_join", "ring_read", "ape_norm",
                "ring_write", "ring_sync"} <= names


def test_key_pool_owner_ring_is_token_serial_per_sequence():
    ledger = _key_pool_ledger(16, 0)
    op = _op(ipn.KEY_POOL_RECIPE, (QUERY_ROWS, INDEX_WIDTH, HIDDEN),
             sp.KEYPOOL_LEDGER, ledger, family="key_pool", dtype="bf16")
    plan = sp.KeyPoolRingRecipe().build(op, None)
    program = plan.program(replace(_context(DT)[1], op=plan.work.apply(op)))
    ring = next(stage for stage in _stages(program) if stage.name == "ring_read")
    assert ring.placement
    # One AIV owner per sequence ring; the token step is strictly serial.
    syncs = [stage for stage in _stages(program) if stage.name == "ring_sync"]
    assert len(syncs) == BATCH * STEPS
    assert all(stage.latency > 0 for stage in syncs)


# --- Item (3): `AscendIndexerKPoolRowShard` caller ------------------------
def _case_model(*, pool=16, index_dtype="fp8_e4m3", topk=INDEX_TOPK,
                compress_ratio=0):
    plan = LayerPlan(
        index=0, attention="mla", architecture="mla", ffn="dense",
        intermediate_size=1024, num_heads=8, num_kv_heads=1, head_dim=576,
        q_lora_rank=512, kv_lora_rank=512, qk_nope_head_dim=512,
        qk_rope_head_dim=0, v_head_dim=128, mla_variant="latent_mla",
        cache_kind="dense", compress_ratio=compress_ratio, indexer=True,
        indexer_type="full",
        index_heads=INDEX_HEADS, index_dim=INDEX_WIDTH, index_topk=topk,
        index_kpool=pool, index_kpool_compress=True, index_input="hidden",
        owns_index_k=True, owns_topk=True)
    quant = QuantSpec("bf16_fp8_index", weight_format="bf16",
                      activation_format="bf16", native_format="bf16",
                      expert_weight_format="bf16", kv_dtype="int8",
                      index_dtype=index_dtype, index_scale_dtype="fp16")
    representations = {} if index_dtype == "int8" else {
        "index_k": {"dtype": index_dtype, "group_size": 32,
                    "scale_dtype": "fp16"}}
    model = ModelSpec(
        name="wave4b", layers=1, hidden_size=HIDDEN, num_heads=8,
        head_dim=576, intermediate_size=1024, vocab_size=1024,
        attention="mla", ffn="dense", layer_plans=(plan,), quant=quant,
        metadata=(
            ("cache_representations", representations),
            ("native_keypool_caller", {
                "implementation": ipn.IMPLEMENTATION, "tail_capacity": pool,
                "block_table_columns": 8195,
                "block_table_columns_explicit": True, "row_shard_tp": 1,
                "source": ipn.SOURCE})))
    return plan, model


class _Runtime:
    def __init__(self, *, o_proj_tp, linear_proj_tp=1, metadata=()):
        self.metadata = dict(metadata)
        self.prefix_cache_mode = "private"
        self.prefix_read_mode = "auto"
        self.parallel = type("Parallel", (), {
            "o_proj_tp": o_proj_tp, "linear_proj_tp": linear_proj_tp,
            "tp": LOCAL_TP})()


class _ArchParallel:
    tp = LOCAL_TP
    cp = 1
    pcp = 1


def _views(count):
    return tuple(RequestView(f"r{index}", STEPS, 131072, False, "decode")
                 for index in range(count))


def _selection_ops(*, o_proj_tp, metadata=(), pool=16):
    plan, model = _case_model(pool=pool)
    # The native caller declares the FP32 pool scale layout (132 B rows).
    model = replace(model, quant=replace(model.quant,
                                         index_scale_dtype="fp32"))
    views = _views(BATCH)
    builder = Builder("s0.l0", plan, 0, views, model, "decode")
    ready = ipn.selection(
        builder, query="q", weights="w", cache="cache", views=views, plan=plan,
        quant=builder.quant, padded_tokens=QUERY_ROWS,
        runtime=_Runtime(o_proj_tp=o_proj_tp, metadata=metadata))
    return {op.recipe: op for op in builder.ops}, ready


def test_row_shard_width_follows_local_tp_and_stays_replicated_at_one():
    views = _views(BATCH)
    assert index_pool.row_shard_tp(options={}, runtime=_Runtime(o_proj_tp=8)) == 8
    assert index_pool.row_shard_tp(options={}, runtime=_Runtime(o_proj_tp=1)) == 1
    assert index_pool.row_shard_tp(
        options={index_pool.ROW_SHARD_OPTION: 1}, runtime=_Runtime(o_proj_tp=8)) == 1
    assert index_pool.row_shard_tp(
        options={}, runtime=_Runtime(o_proj_tp=1, metadata=(
            (index_pool.ROW_SHARD_OPTION, 4),))) == 4
    owned = index_pool.row_shard_partition(views, LOCAL_TP)
    assert len(owned) == BATCH // LOCAL_TP
    assert sum(view.q_len for view in owned) == LOCAL_QUERY_ROWS
    assert index_pool.row_shard_partition(views, 1) == views


def test_row_shard_lowering_emits_the_declared_index_allgather():
    ops, ready = _selection_ops(o_proj_tp=LOCAL_TP)
    indexer = ops[ipn.POOL_KEY_RECIPE]
    gather = ops[ipn.ROW_SHARD_GATHER]
    ledger = dict(indexer.metadata)[sp.POOL_LEDGER]
    assert ledger["query_rows"] == LOCAL_QUERY_ROWS + 1
    assert ledger["waves"] == 3
    assert ledger["blocks"] == (DECODE_CONTEXT // 16) // sp.POOL_KEY_BLOCK
    meta = dict(gather.metadata)
    assert gather.family == "allgather"
    assert gather.link_bytes == 7 * LOCAL_PAYLOAD == 3784704
    assert meta["local_rows"] == LOCAL_QUERY_ROWS
    assert meta["local_payload_bytes"] == LOCAL_PAYLOAD == 540672
    assert meta["peer_shard_bytes"] == 3784704
    assert meta["assembled_bytes"] == 8 * LOCAL_PAYLOAD == 4325376
    assert meta["assembled_shape"] == (QUERY_ROWS, INDEX_TOPK)
    assert meta["row_shard_tp"] == LOCAL_TP
    assert meta["gather_position"] == "before_sfa_consumes_index_topk"
    assert gather.local_shape == (QUERY_ROWS, INDEX_TOPK)
    topk = ops["index_topk"]
    assert topk.local_shape == (QUERY_ROWS, INDEX_TOPK)
    assert gather.op_id in topk.deps
    assert ready == topk.op_id
    declaration = dict(topk.metadata)["structure_subgraphs"][-1]
    assert declaration["implementation"]["conditions"]["row_shard"] == (
        "ascend_indexer_kpool_row_shard")
    assert declaration["tensor_view"]["row_shard_tp"] == LOCAL_TP


def test_row_shard_allgather_is_priced_by_the_declared_topology():
    ops, _ = _selection_ops(o_proj_tp=LOCAL_TP)
    gather = ops[ipn.ROW_SHARD_GATHER]
    indexer = replace(ops[ipn.POOL_KEY_RECIPE], deps=())
    lone = replace(gather, deps=())
    sharded_pr = _latency(lone, None, PR, devices=LOCAL_TP, tp=LOCAL_TP)
    sharded_dt = _latency(lone, None, DT, devices=LOCAL_TP, tp=LOCAL_TP)
    assert sharded_pr > 0.0
    # Wire work is fabric-only: both SKUs pay the same link service.
    assert sharded_pr == pytest.approx(sharded_dt, rel=1e-9)
    index_time = _latency(indexer, sp.PoolKeyIndexerRecipe().build, DT)
    assert sharded_dt < index_time * 0.5


def test_replicated_mode_keeps_the_whole_query_plane_and_no_gather():
    ops, ready = _selection_ops(o_proj_tp=1)
    assert ipn.ROW_SHARD_GATHER not in ops
    ledger = dict(ops[ipn.POOL_KEY_RECIPE].metadata)[sp.POOL_LEDGER]
    assert ledger["query_rows"] == QUERY_ROWS + 1
    assert ledger["waves"] == 17
    assert ops["index_topk"].local_shape == (QUERY_ROWS, INDEX_TOPK)
    assert ready == ops["index_topk"].op_id


# --- Item (6a): arch35 FP8 `quant_mode=1` --------------------------------
ARCH35_CASES = (
    # tokens, heads, keys, measured PR, measured DT, tolerance
    (960, 32, 32768, 607.98, 617.953, 0.08),
    (960, 32, 65536, 1224.33, 1211.549, 0.08),
    (480, 64, 32768, 447.78, 484.514, 0.08),
    (576, 32, 16384, 195.53, 207.349, 0.10),
    (576, 32, 131072, 1535.40, 1562.024, 0.10),
)


def _arch35_ledger(tokens, heads, keys):
    blocks = sp.key_blocks(keys)
    return {
        "query_rows": tokens, "wave_rows": sp.INDEXER_WAVE_ROWS,
        "waves": sp.indexer_waves(tokens), "blocks": blocks,
        "heads": heads, "width": INDEX_WIDTH, "keys": blocks * sp.POOL_KEY_BLOCK,
        "key_row_bytes": INDEX_ROW_BYTES,
        "output_bytes": float(tokens * INDEX_TOPK * 4)}


def _arch35_op(tokens, heads, keys):
    return _op(fq.ARCH35_RECIPE, (tokens, heads, INDEX_WIDTH),
               sp.QUANT_LEDGER, _arch35_ledger(tokens, heads, keys))


def test_arch35_quant_indexer_matches_the_paired_silicon():
    recipe = sp.Arch35QuantIndexerRecipe()
    for tokens, heads, keys, measured_pr, measured_dt, tolerance in ARCH35_CASES:
        op = _arch35_op(tokens, heads, keys)
        for sku, measured in ((PR, measured_pr), (DT, measured_dt)):
            predicted = _latency(op, recipe.build, sku)
            assert predicted == pytest.approx(measured, rel=tolerance), (
                tokens, heads, keys, sku, predicted, measured)


def test_arch35_efficiency_emerges_from_half_the_query_waves():
    ledger = _arch35_ledger(480, 64, 32768)
    op = _arch35_op(480, 64, 32768)
    predicted = _latency(op, sp.Arch35QuantIndexerRecipe().build, DT)
    flops = 2.0 * 480 * 64 * 32768 * INDEX_WIDTH
    achieved = flops / (predicted * 1e-6) / profiles()[DT]["device"][
        "compute_peaks"]["fp8_e4m3"]
    assert 0.55 < achieved < 0.70
    assert ledger["waves"] == 15
    assert _arch35_ledger(960, 32, 32768)["waves"] == 30


def test_arch35_program_is_one_conserving_block_pipeline():
    ledger = _arch35_ledger(576, 32, 16384)
    op = _arch35_op(576, 32, 16384)
    plan = sp.Arch35QuantIndexerRecipe().build(op, None)
    program = _conserved(plan, op, DT)
    names = [stage.name for stage in _stages(program)]
    assert names.count("topk_merge") == ledger["waves"] * ledger["blocks"]
    assert "indices" in names
    assert plan.metadata["blocks"] == ledger["blocks"]


def test_arch35_and_arch22_recipes_coexist_and_route_by_representation():
    assert set(fq.recipes()) == {fq.RECIPE, fq.ARCH35_RECIPE}
    plan, fp8_model = _case_model(pool=1, index_dtype="fp8_e4m3",
                                  compress_ratio=4)
    views = tuple(RequestView(f"r{index}", STEPS, 131072, False, "decode")
                  for index in range(160))
    builder = Builder("s0.l0", plan, 0, views, fp8_model, "decode")
    fq.declare(builder, deps=("q", "w", "cache"), views=views, plan=plan,
               quant=builder.quant, parallel=_ArchParallel())
    op = builder.ops[-1]
    ledger = dict(op.metadata)[sp.QUANT_LEDGER]
    assert op.recipe == fq.ARCH35_RECIPE
    assert ledger == {"query_rows": 960, "wave_rows": sp.INDEXER_WAVE_ROWS,
                      "waves": 30, "blocks": 32, "heads": INDEX_HEADS,
                      "width": INDEX_WIDTH, "keys": 32768,
                      "completed_keys": 32768,
                      # Every private request owns its own compressed cache.
                      "payload_keys": 160 * 32768,
                      "key_row_bytes": INDEX_ROW_BYTES,
                      "output_bytes": float(960 * INDEX_TOPK * 4)}
    assert dict(op.matrix_flops) == {
        "fp8_e4m3": 2.0 * 960 * INDEX_HEADS * 32768 * INDEX_WIDTH}
    view = dict(op.metadata)["native_tensor_view"]
    assert view["quant_mode"] == 1
    assert view["kind"] == "aclnn_quant_lightning_indexer_v2_arch35"
    assert view["output_shape"] == (960, 1, INDEX_TOPK)
    # The arch22 INT8/FP16 contract keeps its own native domain gate.
    _, int8_model = _case_model(pool=1, index_dtype="int8",
                                compress_ratio=4)
    arch22_quant = int8_model.quant
    fq._check_domain(
        type("Plan", (), {"index_heads": 64, "index_dim": 128,
                          "index_topk": 512, "compress_ratio": 4,
                          "candidate_filter": False, "index_kpool": 1})(),
        arch22_quant, type("P", (), {"cp": 1, "pcp": 1})())
    assert fq.NATIVE_HEAD_COUNT == 64
    assert fq.ARCH35_HEAD_COUNTS == (32, 64)


# --- Item (6b): `causal_conv1d_update` UPDATE ABI ------------------------
CONV_CASES = (
    # batch, steps, channels, geometry, measured PR, measured DT, tolerance
    (BATCH, STEPS, 3072, (1024, 3, AIV_OWNERS), 23.86, 29.892, CONV_TOLERANCE),
    (KIMI_BATCH, STEPS, 4608, (512, 9, AIV_OWNERS), 50.99, 56.767, TOLERANCE),
    (BATCH, STEPS, 24576, (4096, 6, AIV_OWNERS), 82.28, 79.78, CONV_TOLERANCE),
)


def _conv_op(batch, steps, channels):
    lengths = (steps,) * batch
    shape = (batch * steps, channels)
    return _op(gdn.CONV, shape, family="short_convolution", dtype="bf16", metadata=(
        ("sequence_tokens", lengths), ("physical_sequence_tokens", lengths),
        ("kernel_width", 4), ("speculative", True), ("bias", False),
        ("state_history_slots", 2 + steps), ("padding_slots", ())))


def test_conv_update_binds_the_live_geometry_tiler():
    for batch, steps, channels, geometry, *_ in CONV_CASES:
        assert gdn.update_geometry(batch, channels, AIV_OWNERS) == geometry
    waves = [int(math.ceil(case[0] * case[3][1] / AIV_OWNERS))
             for case in CONV_CASES]
    assert waves == [5, 14, 9]


def test_conv_update_matches_the_paired_silicon():
    for batch, steps, channels, _geometry, pr, dt, tolerance in CONV_CASES:
        op = _conv_op(batch, steps, channels)
        for sku, measured in ((PR, pr), (DT, dt)):
            predicted = _latency(op, gdn.conv_plan, sku)
            assert predicted == pytest.approx(measured, rel=tolerance), (
                channels, sku, predicted, measured)


def test_conv_update_has_no_fabricated_prefill_sync_barrier():
    op = _conv_op(BATCH, STEPS, 24576)
    geometry = Geometry.from_devices((_case(DT).machine.dies[0],))
    plan = gdn.conv_plan(op, geometry)
    program = _conserved(plan, op, DT)
    names = {stage.name for stage in _stages(program)}
    assert not {"sync_all", "prefill_sync", "multi_column_sync",
                "state_reload_sync"} & names
    assert {"state_resolve", "weights_history", "weights_cast", "convolution",
            "output", "external_stream", "state_write"} <= names
    # One streamed envelope per physical AIV owner; `4096 x 6` keeps 9 owner
    # wave repetitions without an extra barrier round.
    streams = [stage for stage in _stages(program) if stage.name == "external_stream"]
    assert len(streams) == AIV_OWNERS
    assert plan.work.resources["hbm"] == pytest.approx(
        sum(2 * (STEPS + 3) * 24576 * 2 for _ in range(BATCH)))


# --- Final-gate rows 6-8 and 13-14: indexer dispatch, arch35 conservation,
# KeyPool row shard and the MLA prolog operand contracts.
ROOT = Path(__file__).resolve().parents[1]


def _arch35_op_for(ratio, views, *, topk=INDEX_TOPK, heads=INDEX_HEADS):
    plan, model = _case_model(pool=1, index_dtype="fp8_e4m3",
                              compress_ratio=ratio, topk=topk)
    plan = replace(plan, index_heads=heads)
    builder = Builder("s0.l0", plan, 0, views, model, "decode")
    fq.declare(builder, deps=("q", "w", "cache"), views=views, plan=plan,
               quant=builder.quant, parallel=_ArchParallel())
    return builder.ops[-1]


def test_bf16_and_fp16_index_operands_never_select_the_quantized_fusion():
    from mindie_sim.compose import compose_case
    for qmode in ("bf16", "fp16"):
        dsv32 = parse_case(compose_case(
            "deepseek-v3.2", "a2", quant=qmode, tp=8, dp=2, ep=16, concurrency=2))
        dsv32_ops = target_operators(dsv32, phase="decode", tokens=1, context=2048)
        assert not any(dict(op.metadata).get("native_kind")
                       == "aclnnQuantLightningIndexerV2" for op in dsv32_ops)
    raw = json.loads((ROOT / "examples/cases/glm53-a3-w4a8.json").read_text())
    raw["model"]["quantization"] = "bf16"
    bf16 = parse_case(raw)
    assert bf16.model.quant.index_dtype == "bf16"
    assert bf16.model.quant.activation_format == "bf16"
    fp16 = replace(bf16, model=replace(
        bf16.model, quant=replace(bf16.model.quant, index_dtype="fp16")))
    for case in (bf16, fp16):
        recipes = {op.recipe for op in target_operators(
            case, phase="decode", tokens=1, context=4096)}
        assert "quant_lightning_indexer" not in recipes
        assert {"index_scores", "index_topk"} <= recipes
    # W4A8 keeps a quantized activation, so the fused CANN kernel stays.
    fused = target_operators(parse_case(json.loads(
        (ROOT / "examples/cases/glm53-a3-w4a8.json").read_text())),
        phase="decode", tokens=1, context=4096)
    assert any(dict(op.metadata).get("native_kind")
               == "aclnnQuantLightningIndexerV2" for op in fused)


def test_arch35_zero_compression_and_completed_key_flops():
    # A dense non-compressed indexer must not divide by compress_ratio == 0.
    dense = _arch35_op_for(0, (RequestView("r", 1, 1, False, "decode"),))
    assert dict(dense.metadata)[sp.QUANT_LEDGER]["completed_keys"] == 1
    assert dense.flops == 2 * 1 * INDEX_HEADS * 1 * INDEX_WIDTH == 8192
    # Ratio 4 / context 4 completes one key, not the padded 1,024-key tile.
    sparse = _arch35_op_for(4, (RequestView("r", 1, 4, False, "decode"),))
    ledger = dict(sparse.metadata)[sp.QUANT_LEDGER]
    assert ledger["completed_keys"] == 1
    assert ledger["keys"] == sp.POOL_KEY_BLOCK
    assert sparse.flops == 8192
    assert dict(sparse.metadata)["native_tensor_view"]["visible_keys"] == 1


def test_arch35_private_cache_payload_is_charged_once_on_hbm_and_mte2():
    views = (RequestView("a", 1, 1024, False, "decode"),
             RequestView("b", 1, 1024, False, "decode"))
    op = _arch35_op_for(1, views)
    ledger = dict(op.metadata)[sp.QUANT_LEDGER]
    assert ledger["completed_keys"] == 1024
    assert ledger["payload_keys"] == 2048
    payload = 2 * 1024 * INDEX_ROW_BYTES
    assert payload >= 262144
    assert op.hbm_bytes >= payload
    assert op.mte2_bytes >= payload
    assert op.mte2_bytes > 0
    partial = (RequestView("a", 1, 2048, False, "decode", prefix_tokens=1024,
                           prefix_cache_key="shared_a"),
               RequestView("b", 1, 2048, False, "decode", prefix_tokens=1024,
                           prefix_cache_key="shared_a"))
    partial_ledger = dict(_arch35_op_for(1, partial).metadata)[sp.QUANT_LEDGER]
    assert partial_ledger["payload_keys"] == 1024 + 2 * 1024 == 3072


def test_keypool_row_shard_has_no_negative_dummy_rows():
    plan, model = _case_model(pool=16)
    model = replace(model, quant=replace(model.quant, index_scale_dtype="fp32"))
    views = _views(1)
    builder = Builder("s0.l0", plan, 0, views, model, "decode")
    ipn.selection(builder, query="q", weights="w", cache="cache", views=views,
                  plan=plan, quant=builder.quant, padded_tokens=STEPS,
                  runtime=_Runtime(o_proj_tp=LOCAL_TP))
    meta = dict(next(op for op in builder.ops
                     if op.recipe == ipn.POOL_KEY_RECIPE).metadata)
    assert (meta["valid_query_rows"], meta["padded_query_rows"],
            meta["native_query_rows"]) == (6, 6, 7)
    assert meta["dummy_request_queries"] == 1


def test_keypool_gathered_selection_keeps_each_request_row_count():
    from mindie_sim.prediction.mla_cann_epilogs import _selection_reference
    plan, model = _case_model(pool=4)
    plan = replace(plan, index_kpool_always_select_tail=True)
    model = replace(model, quant=replace(model.quant, index_scale_dtype="fp32"))
    views = (RequestView("a", 1, 4, False, "decode"),
             RequestView("b", 1, 100, False, "decode"))
    builder = Builder("s0.l0", plan, 0, views, model, "decode")
    ipn.selection(builder, query="q", weights="w", cache="cache", views=views,
                  plan=plan, quant=builder.quant, padded_tokens=2,
                  runtime=_Runtime(o_proj_tp=LOCAL_TP))
    packed = next(op for op in builder.ops if op.recipe == "index_topk")
    meta = dict(packed.metadata)
    assert meta["selected_request_ids"] == ("a", "b")
    assert meta["selected_tokens_by_request"] == (5, 101)
    assert meta["selected_token_rows"] == 106
    reference = _selection_reference(builder, packed.op_id)
    assert reference.rows_for("a") == 5
    assert reference.rows_for("b") == 101
    # A partial one-entry map must never broadcast request 0's rows; only a
    # producer that owns the complete selection may be replayed.
    from mindie_sim.prediction.mla_cann_epilogs import SharedSelectionRef
    partial = SharedSelectionRef("partial", selected_token_rows=106,
                                 selected_by_request={"a": 5})
    assert partial.rows_for("b") is None
    single = SharedSelectionRef("single", selected_token_rows=5,
                                selected_by_request={"a": 5})
    assert single.rows_for("b") == 5


def test_mla_prolog_v4_scales_are_group32_e8m0_and_v3_rejects_fp8_weights():
    from mindie_sim.schema.common import ModelingError
    from test_wave3a_cann921_mla_keypool_prolog import (
        HIDDEN, HEAD_WIDTH, KV_RANK, LAYOUT, PROLOG_QUERY_LORA_RANK,
        ParallelSpec, _declaration, _op, _prolog_case,
    )
    from mindie_sim.prediction import family_mla_prolog as prolog
    plan, model = _prolog_case()
    ops = [
        _op("s0.l0.q_a", "gemm", (8, PROLOG_QUERY_LORA_RANK, HIDDEN)),
        _op("s0.l0.q_norm", "rmsnorm", (8, PROLOG_QUERY_LORA_RANK)),
        _op("s0.l0.q_b", "gemm", (8, 8 * HEAD_WIDTH, KV_RANK)),
        _op("s0.l0.kv_a", "gemm", (8, HEAD_WIDTH, HIDDEN)),
        _op("s0.l0.kv_norm", "rmsnorm", (8, KV_RANK)),
        _op("s0.l0.q_absorb", "gemm", (64, KV_RANK, KV_RANK)),
    ]
    prolog.declare(
        ops, prefix="s0.l0", plan=plan, model=model,
        parallel=ParallelSpec(tp=8, cp=8, pcp=1), quant=model.quant, tokens=8,
        members=tuple(op.op_id for op in ops), q_a="s0.l0.q_a",
        q_b="s0.l0.q_b", kv_a="s0.l0.kv_a", q_absorb="s0.l0.q_absorb",
        cache_layout=LAYOUT)
    view = _declaration(ops, "s0.l0.q_absorb")["tensor_view"]
    assert view["query_norm_scale_dtype"] == "fp8_e8m0"
    assert view["query_norm_scale_shape"] == (8, PROLOG_QUERY_LORA_RANK // 32)
    assert 8 * PROLOG_QUERY_LORA_RANK // 32 == 384
    # FP8 projection weights without the V4 packed contract are V3-illegal.
    fp8_quant = replace(model.quant, kv_dtype="bf16")
    with pytest.raises(ModelingError, match="mla_prolog_v3_rejects_fp8"):
        prolog.validate(plan=plan, parallel=ParallelSpec(tp=8, cp=8, pcp=1),
                        model=model, quant=fp8_quant,
                        lowering_passes=("activation_quantization",))
