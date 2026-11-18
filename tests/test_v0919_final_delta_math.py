"""v0.9.19 final fact-and-math conservation regressions.

[Role]
  Pin the thirteen final audit findings at their exact reproduced numbers:
  one inverse-index write, one DBO cohort ledger, causal window tails, the
  CP-sharded index-K handoff, packed MQSMLA rows, compressed prefix unions,
  arch35 completed keys, mixed-prefix private rows, packed C8 geometry and
  statistics, local-TP row shards, the selected router tile, and the V4 query
  scale shape.

[Invariants]
  Every assertion is derived from declared geometry or an operator graph; no
  model name, captured duration or device context selects a result.

[Timeouts & Locks]
  In-memory graphs only; no sleep, network, NPU context or file writes.

[Side-Effect Policy]
  Read-only over packaged cases and builder-local records.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import compose_case, predict_pd
from mindie_sim.catalog.cache import (
    layer_cache_components, owned_token_rows, packed_region_row_bytes,
)
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.prediction import family_quant_indexer as quant_indexer
from mindie_sim.prediction import indexed_quant_attention as c8
from mindie_sim.prediction import mqsmla
from mindie_sim.prediction.cache_path import CacheGeometry, CacheState
from mindie_sim.prediction.cache_representation import (
    cache_identity, store,
)
from mindie_sim.prediction.index_pool import row_shard_partition
from mindie_sim.prediction.latent_dcp import packed_sfa_geometry
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.matrix_splitk import build_router_splitk
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView

import test_v0919_deep_audit_track_b as track_b
import test_wave3a_cann921_mla_keypool_prolog as wave3a
import test_wave4a_c8sfa_mqsmla_calibration as wave4a
import test_wave4b_indexer_conv1d_calibration as wave4b
import test_wave4c_gmm_router_calibration as wave4c
from test_prediction_system import linear_work


ROOT = Path(__file__).resolve().parents[1]
PACKED_MLA_CASE = (ROOT / "examples/cases/longctx-decode-20261030/"
                   "v41-a5-tp1-dp8-ep8.json")
ROUTER_OPERAND_BYTES = 102760448
MODE1_ROW_BYTES = 608
MODE2_ROW_BYTES = 584


# ---------------------------------------------------------------------------
# 1. One inverse-index write on the absorbed init-routing pack.
# ---------------------------------------------------------------------------
def test_absorbed_init_routing_writes_the_inverse_index_once():
    graph = track_b._moe_inverse_index_graph()
    pack = next(op for op in graph if op.recipe == "moe_init_routing_v3")
    unpack = next(op for op in graph if op.recipe == "route_unpack")
    assignments = int(pack.local_shape[0]) * int(pack.local_shape[1])
    inverse_bytes = assignments * 4

    assert assignments == 64
    assert inverse_bytes == 256
    assert dict(pack.metadata)["inverse_index_bytes"] == inverse_bytes
    assert dict(pack.metadata)["moe_init_routing"]["write_bytes"] == (
        pytest.approx(pack.mte3_bytes))
    # The quantized payload is 270,336 B; adding the inverse map a second time
    # would produce the audited 270,848 B.
    assert pack.mte3_bytes == pytest.approx(270336 + inverse_bytes)
    assert pack.mte3_bytes == 270592
    assert pack.hbm_bytes == pytest.approx(pack.mte2_bytes + pack.mte3_bytes)
    assert dict(unpack.metadata)["inverse_index_bytes"] == inverse_bytes
    assert unpack.mte2_bytes == pytest.approx(524544 + inverse_bytes)


# ---------------------------------------------------------------------------
# 2. DBO allocates the exact cohort rows to each microbatch.
# ---------------------------------------------------------------------------
def _dbo_graphs(sequences):
    raw = compose_case(
        "qwen3-30b-a3b", hardware="a5", tp=1, dp=2, ep=2, num_devices=2,
        layers=1, prompt_tokens=8, output_tokens=2, max_num_seqs=4,
        enable_dbo=True)
    return target_graphs(
        parse_case(raw), phase="decode", tokens=1, context=32,
        sequences=sequences, scope="body")


def test_dbo_microbatches_use_the_cohort_row_ledger():
    graphs = _dbo_graphs([4, 3])
    per_replica = []
    for replica in (0, 1):
        gates = [op for op in graphs[replica] if op.recipe == "grouped_gate_up"]
        assert [op.local_shape[0] for op in gates] == [16, 12]
        per_replica.append([op.local_shape[0] for op in gates])
    # Cohort assignments conserve exactly: microbatch zero owns 2 + 2 rows
    # (32 assignments) and microbatch one owns 2 + 1 rows (24 assignments).
    assert [sum(row[index] for row in per_replica) for index in (0, 1)] == [32, 24]


# ---------------------------------------------------------------------------
# 3. Sliding-window KV owns its live tail, not the prefix rows.
# ---------------------------------------------------------------------------
def _window_case():
    raw = compose_case(
        "qwen3-32b", hardware="a5", tp=1, ep=1, num_devices=1, layers=1,
        prompt_tokens=10, output_tokens=2)
    raw["model"].update(
        hidden_size=64, num_heads=4, num_kv_heads=1, head_dim=16,
        intermediate_size=64, vocab_size=128)
    raw["model"]["layer_plans"] = [{
        "index": 0, "attention": "gqa", "ffn": "dense", "window": 4,
        "num_heads": 4, "num_kv_heads": 1, "head_dim": 16,
        "intermediate_size": 64}]
    return parse_case(raw)


def test_window_kv_intersects_the_live_tail_rows():
    case = _window_case()
    plan = case.model.layer_plans[0]
    parallel = replace(case.runtime.parallel, tp=2, cp=2, dcp=2,
                       cp_plan="dcp")
    owners = ((0, 5), (5, 10))
    width = 2 * plan.num_kv_heads * plan.head_dim
    row_bytes = cache_format(case.model, "gqa_kv").row_bytes(width)
    assert (width, row_bytes, 4 * row_bytes) == (32, 64, 256)

    assert owned_token_rows(4, owners, 0, start=6) == 0
    assert owned_token_rows(4, owners, 1, start=6) == 4
    for rank, expected in ((0, 0.0), (1, 256.0)):
        values = layer_cache_components(
            case.model, plan, parallel, 10, capacity_padded=False,
            rank=rank, owner_spans=owners)
        assert values["gqa_kv"] == pytest.approx(expected)


def test_pd_handoff_clips_the_window_tail_before_charging():
    raw = track_b._pd_model_case(10, 1)
    raw["model"]["layer_plans"] = [{
        "index": 0, "attention": "gqa", "ffn": "dense", "window": 4,
        "num_heads": 4, "num_kv_heads": 1, "head_dim": 16,
        "intermediate_size": 64}]
    track_b._bind_pd_pool(raw, "prefill", (2,), tp=1, cp=1)
    track_b._bind_pd_pool(raw, "decode", (0, 1), tp=2, cp=2)
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]

    assert len(transfers) == 1
    row = transfers[0]
    assert (row["component"], row["source_rank"], row["destination_rank"]) == (
        "gqa_kv", 2, 1)
    assert row["payload_bytes"] == pytest.approx(256.0)
    assert tuple(row["source_token_span"]) == (6, 10)
    assert tuple(row["destination_token_span"]) == (6, 10)


# ---------------------------------------------------------------------------
# 4. dsa_cp with TP > 1 and CP == 1 maps each stage rank to shard zero.
# ---------------------------------------------------------------------------
def _deepseek_pd_case():
    raw = compose_case(
        "deepseek-v3.2", hardware="a5", tp=1, dp=1, ep=1, num_devices=3,
        layers=1, prompt_tokens=4, output_tokens=2)
    raw["pools"] = {
        "prefill": {"machine": copy.deepcopy(raw["machine"]),
                    "runtime": copy.deepcopy(raw["runtime"])},
        "decode": {"machine": copy.deepcopy(raw["machine"]),
                   "runtime": copy.deepcopy(raw["runtime"])},
    }
    for pool, ranks, tp, cp, cp_plan in (
            ("prefill", (0, 1), 2, 1, "dsa_cp"),
            ("decode", (2,), 1, 1, "none")):
        machine = raw["pools"][pool]["machine"]
        machine["num_devices"] = len(ranks)
        machine["devices"] = [{"rank": rank} for rank in ranks]
        machine["links"][0]["members"] = list(ranks)
        raw["pools"][pool]["runtime"]["parallel"].update(
            tp=tp, cp=cp, dcp=cp, cp_plan=cp_plan)
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0, "semantics": "per_pair"},
        "chunk_bytes": 64}
    return raw


def test_dsa_cp_tp2_cp1_handoff_reads_the_single_cp_shard():
    result = predict_pd(_deepseek_pd_case(), lower=linear_work,
                        event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]
    rows = {(row["component"], row["source_rank"], row["destination_rank"]):
            row["payload_bytes"] for row in transfers}
    assert rows == {("index_k", 0, 2): pytest.approx(1024.0),
                    ("mla_kv", 0, 2): pytest.approx(4608.0)}


# ---------------------------------------------------------------------------
# 5. MQSMLA Mode-1/2 writers use the packed physical row.
# ---------------------------------------------------------------------------
def _packed_mode_model(scale_dtype):
    case = parse_case(json.loads(PACKED_MLA_CASE.read_text()))
    metadata = dict(case.model.metadata)
    regions = dict(metadata.get("cache_representations", {}))
    region = {"dtype": "fp8_e4m3", "group_size": 32,
              "scale_dtype": scale_dtype}
    regions["window"] = dict(region)
    regions["compressed_kv"] = dict(region)
    return replace(case.model, metadata=tuple({
        **metadata, "cache_representations": regions}.items()))


@pytest.mark.parametrize(("scale_dtype", "row_bytes"), [
    ("bf16", MODE1_ROW_BYTES),
    ("uint8", MODE2_ROW_BYTES),
])
def test_mqsmla_mode_writers_use_the_packed_row_bytes(scale_dtype, row_bytes):
    model = _packed_mode_model(scale_dtype)
    plan = model.layer_plans[0]
    width = plan.cache_row_elements
    assert width == 512
    assert packed_region_row_bytes(model, "window", width) == row_bytes
    views = (RequestView("r0", 1, 100, False, "decode"),)
    builder = Builder("l0", plan, 0, views, model, "decode")
    geometry = CacheGeometry(capacity_bytes=1 << 20, line_bytes=1,
                             transfer_bytes=1, staging_lines=1)
    state = CacheState(geometry)
    store(builder, "window", ("src",), 3, width, "window", state=state,
          geometry=geometry,
          identity=cache_identity("window", layer=0, request="r0"))
    op = builder.ops[-1]
    assert op.mte3_bytes == pytest.approx(3 * row_bytes)
    assert dict(op.metadata)["cache_row_bytes"] == row_bytes
    assert dict(op.metadata)["cache_representation"]["row_bytes"] == row_bytes


def test_mqsmla_epilog_prefers_the_declared_cache_row_width():
    from mindie_sim.prediction.mla_cann_epilogs import (
        KV_COMPRESS_EPILOG_FAMILY, kv_cache_epilog,
    )
    model = _packed_mode_model("bf16")
    fmt = cache_format(model, "window")
    op = replace(
        wave4a._c8_op(((1, 128),), 128, 64, 0),
        dtype="bf16", local_shape=(3, 512),
        metadata=(("cache_row_bytes", MODE1_ROW_BYTES),))
    epilog = kv_cache_epilog(op, fmt, "window", native=True)
    assert epilog.family == KV_COMPRESS_EPILOG_FAMILY
    assert dict(epilog.metadata)["cache_epilog"]["row_bytes"] == MODE1_ROW_BYTES


# ---------------------------------------------------------------------------
# 6. MQSMLA causal window union and dense HCA compressed prefix rows.
# ---------------------------------------------------------------------------
def test_mqsmla_window_union_advances_across_query_rows():
    shape = {"sequences": ((6, 131072),), "window": 128, "ratio": 0,
             "topk": 0}
    external, shared = mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8)
    assert shared == 0
    assert external == 133 * mqsmla.quant_row_bytes(mqsmla.KV_QUANT_FP8)
    assert external == 80864


def test_mqsmla_hca_short_prefix_keeps_every_private_row():
    shape = {"sequences": ((1, 191), (1, 191)), "window": 128, "ratio": 128,
             "topk": 0, "prefix_cache_mode": "shared",
             "prefix_tokens": (64, 64), "prefix_cache_keys": ("k", "k")}
    external, shared = mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8)
    assert shared == 0
    assert external == 258 * 608 == 156864


# ---------------------------------------------------------------------------
# 7. Arch35 indexer counts completed causal keys, floor for compressed rows.
# ---------------------------------------------------------------------------
def test_quant_indexer_counts_the_causal_query_tokens():
    dense = SimpleNamespace(context=1023, q_len=1)
    assert quant_indexer._view_completed_keys(dense, 1) == 1024
    assert quant_indexer._candidate_keys((dense,), 1)[0] == 1024

    short = SimpleNamespace(context=5, q_len=1)
    assert quant_indexer._view_completed_keys(short, 4) == 1
    assert quant_indexer._candidate_keys((short,), 4)[0] == 1


def _arch35_flops(keys):
    ledger = {
        "query_rows": 1, "wave_rows": 32, "waves": 1, "blocks": 1,
        "heads": 32, "width": 128, "keys": 1024, "completed_keys": keys,
        "payload_keys": keys, "key_row_bytes": 256, "output_bytes": 0.0,
    }
    from mindie_sim.prediction import sparse_indexer_pipeline as pipeline
    return pipeline.arch35_indexer_work(ledger).matrix["fp8_e4m3"]


def test_quant_indexer_flops_follow_the_completed_key_count():
    assert _arch35_flops(1024) == 2 * 1024 * 32 * 128 == 8388608
    assert _arch35_flops(1) == 2 * 1 * 32 * 128 == 8192


# ---------------------------------------------------------------------------
# 8. C8 mixed-prefix batches keep private rows on their own tiles.
# ---------------------------------------------------------------------------
def test_c8_distinct_prefix_keys_keep_private_rows_on_their_tiles():
    sequences = ((1, 128), (1, 256))
    op = wave4a._c8_op(sequences, 128, 64, 0)
    base = dict(op.metadata)["indexed_attention_geometry"]
    shape = {**base, "prefix_cache_mode": "shared",
             "prefix_tokens": (128, 128),
             "prefix_cache_keys": ("k0", "k1")}
    shared, private, selected, private_keys = c8.prefix_partition(
        shape, sequences, 128, sequence_keys=True)
    assert (shared, private, selected) == (0, 256, 0)
    assert private_keys == (128, 128)

    _owners, streams = c8.tile_stream(sequences, 128, 32, 128, 1, False)
    kv_hbm = 0
    for parts in streams:
        for sequence, tiles in parts:
            for key in tiles:
                for name, work, _after in c8._tile_phases(
                        key, shape["heads"], shape["latent"], shape["rope"],
                        private_keys[sequence], 128):
                    if name == "kv_stage":
                        kv_hbm += work.resources["hbm"]
    assert kv_hbm == 256 * 656 == 167936


# ---------------------------------------------------------------------------
# 9. Plain INT8 MLA must not select the 656 B/row C8 kernel.
# ---------------------------------------------------------------------------
def test_plain_int8_mla_cache_does_not_select_packed_c8():
    plan = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64,
                           cache_row_elements=576, index_topk=2048,
                           indexer=True)
    views = (SimpleNamespace(q_len=1, context=256, prefix_tokens=0,
                             prefix_cache_key=""),)
    meta = {"attention_tensor_view": {"cache_dtype": "int8"}}
    plain = CacheFormat("int8")
    assert plain.group_size == 0 and plain.row_bytes(576) == 576
    assert packed_sfa_geometry(
        meta, 8, plan, views, 8, cache_dtype="int8",
        cache_fmt=plain) is None

    packed = CacheFormat("int8", 128, "fp32", 64, "bf16")
    geometry = packed_sfa_geometry(
        meta, 8, plan, views, 8, cache_dtype="int8", cache_fmt=packed)
    assert geometry is not None
    assert (geometry["latent"], geometry["rope"]) == (512, 64)


# ---------------------------------------------------------------------------
# 10. Paired C8 writes the FP32 LSE statistics in plan and stages.
# ---------------------------------------------------------------------------
def _recipe_context(op, geometry):
    names = {role: f"{role}@r0:s0" for role in geometry.resource_units}
    names["cube"] = "cube@r0:s0"
    rates = {names[role]: float(rate) / 1000.0
             for role, rate in geometry.rates.items()}
    return RecipeContext(
        op, {}, names, rates, geometry.compute_units,
        resource_units=dict(geometry.resource_units),
        compute_peaks=dict(geometry.compute_peaks),
        pool_names={pool: f"{pool}@r0:s0" for pool in geometry.buffers},
        pool_units=dict(geometry.buffer_units),
        pool_capacities=dict(geometry.buffers),
        physical_mechanisms=geometry.physical_mechanisms)


def test_paired_c8_statistics_are_written_by_plan_and_stage():
    op = wave4a._c8_op(((1, 128),), 128, 64, 0)
    shape = dict(op.metadata)["indexed_attention_geometry"]
    hardware = Geometry.from_devices(
        parse_case(wave4a._c8_case(wave4a.SKUS[0][1])).machine.dies)
    base = c8.arch35_plan(shape, hardware)
    statistics = 96 * 2 * 4
    assert statistics == 768
    with_stats = c8.arch35_plan({**shape, "statistics_bytes": statistics},
                                hardware)
    assert with_stats.work.resources["mte3"] == (
        base.work.resources["mte3"] + statistics)
    assert with_stats.work.resources["hbm"] == (
        base.work.resources["hbm"] + statistics)

    program = with_stats.program(_recipe_context(op, hardware))
    stage = next(item for item in program.stages
                 if item.name == "lse_statistics")
    assert stage.after == ("call0",)
    assert sorted(stage.work.values()) == [statistics, statistics]


# ---------------------------------------------------------------------------
# 11. Local-TP row sharding splits one multi-row request at the row boundary.
# ---------------------------------------------------------------------------
def test_keypool_row_shard_splits_one_multi_row_request():
    eight_decode = tuple(
        RequestView(f"r{index}", 1, 131072, False, "decode")
        for index in range(8))
    single_long = (RequestView("r0", 8, 131072, False, "decode"),)

    assert [view.q_len for view in row_shard_partition(eight_decode, 8)] == [1]
    owned = row_shard_partition(single_long, 8)
    assert len(owned) == 1 and owned[0].q_len == 1

    plan, model = wave4b._case_model(pool=16)
    model = replace(model, quant=replace(model.quant,
                                         index_scale_dtype="fp32"))
    builder = Builder("s0.l0", plan, 0, single_long, model, "decode")
    from mindie_sim.prediction import index_pool_native as native
    native.selection(
        builder, query="q", weights="w", cache="cache", views=single_long,
        plan=plan, quant=builder.quant, padded_tokens=8,
        runtime=wave4b._Runtime(o_proj_tp=8))
    gather = next(op for op in builder.ops
                  if op.recipe == native.ROW_SHARD_GATHER)
    meta = dict(gather.metadata)
    assert meta["local_rows"] == 1
    assert meta["local_payload_bytes"] == 2048 * 4 == 8192
    assert gather.link_bytes == 57344
    assert meta["assembled_bytes"] == 65536
    assert meta["assembled_shape"] == (8, 2048)


# ---------------------------------------------------------------------------
# 12. Router operand traffic follows the mapping the executor selects.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("shape", "expected"), [
    ((528, 288, 4096), 42860544),
    ((240, 384, 7168), 42663936),
    ((512, 896, 7168), ROUTER_OPERAND_BYTES),
])
def test_router_operand_traffic_uses_the_executed_tile(shape, expected):
    raw = wave4c._case(wave4c.V41, wave4c.DT)
    op = wave4c._router(*shape)
    plan = build_router_splitk(op, wave4c._geometry(raw))
    context = wave4c._bound_context(raw, plan, op)
    expected_work = context.bind(plan.work)
    issued = wave4c._stage_work(plan.program(context))

    assert plan.work.resources["mte2"] == expected
    assert expected_work[context.resource_names["mte2_aic"]] == expected
    assert issued[context.resource_names["mte2_aic"]] == expected


# ---------------------------------------------------------------------------
# 13. V4 query-normalized scale shape follows query_norm_output.
# ---------------------------------------------------------------------------
def test_mla_prolog_v4_without_query_norm_output_has_no_scale_shape():
    from mindie_sim.prediction import family_mla_prolog as prolog
    plan, model = wave3a._prolog_case()
    plan = replace(plan, indexer=False, owns_topk=False,
                   index_query_per_layer=False)
    ops = [
        wave3a._op("s0.l0.q_a", "gemm",
                   (8, wave3a.PROLOG_QUERY_LORA_RANK, wave3a.HIDDEN)),
        wave3a._op("s0.l0.q_norm", "rmsnorm",
                   (8, wave3a.PROLOG_QUERY_LORA_RANK)),
        wave3a._op("s0.l0.q_b", "gemm",
                   (8, 8 * wave3a.HEAD_WIDTH, wave3a.KV_RANK)),
        wave3a._op("s0.l0.kv_a", "gemm",
                   (8, wave3a.HEAD_WIDTH, wave3a.HIDDEN)),
        wave3a._op("s0.l0.kv_norm", "rmsnorm", (8, wave3a.KV_RANK)),
        wave3a._op("s0.l0.q_absorb", "gemm",
                   (64, wave3a.KV_RANK, wave3a.KV_RANK)),
    ]
    prolog.declare(
        ops, prefix="s0.l0", plan=plan, model=model,
        parallel=wave3a.ParallelSpec(tp=8, cp=8, pcp=1),
        quant=model.quant, tokens=8,
        members=tuple(op.op_id for op in ops),
        q_a="s0.l0.q_a", q_b="s0.l0.q_b", kv_a="s0.l0.kv_a",
        q_absorb="s0.l0.q_absorb", cache_layout=wave3a.LAYOUT)
    view = wave3a._declaration(ops, "s0.l0.q_absorb")["tensor_view"]

    assert view["query_norm_output"] is False
    assert view["query_norm_scale_dtype"] == "fp8_e8m0"
    assert view["query_norm_scale_shape"] == ()
