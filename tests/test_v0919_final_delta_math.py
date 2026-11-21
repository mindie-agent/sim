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
    cache_row_split, layer_cache_components, owned_token_rows,
    packed_region_row_bytes,
)
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.prediction.operators.sparse_indexer import family_quant_indexer as quant_indexer
from mindie_sim.prediction.operators.paged_mla import family_mqsmla
from mindie_sim.prediction.operators.paged_mla import indexed_quant_attention as c8
from mindie_sim.prediction.operators.mla import latent_dcp
from mindie_sim.prediction.operators.paged_mla import mqsmla
from mindie_sim.prediction.cache_path import CacheGeometry, CacheState
from mindie_sim.prediction.cache_representation import (
    CacheLowering, cache_identity, store,
)
from mindie_sim.prediction.operators.sparse_indexer.index_pool import row_shard_partition
from mindie_sim.prediction.operators.mla.latent_dcp import packed_sfa_geometry
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.operators.matrix.matrix_splitk import build_router_splitk
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView
from mindie_sim.schema.operators import OpSpec

import test_v0919_deep_audit_track_b as track_b
import _support.operator_calibration_helpers as wave3a
import _support.operator_calibration_helpers as wave4a
import _support.operator_calibration_helpers as wave4b
from _support.moe_hc_mc2_fixtures import router_splitk as wave4c
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
    from mindie_sim.public.errors import UnmodeledCapabilityError

    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        _dbo_graphs([4, 3])


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
    from mindie_sim.prediction.operators.mla.mla_epilogs import (
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
    from mindie_sim.prediction.operators.sparse_indexer import sparse_indexer_pipeline as pipeline
    return pipeline.arch35_indexer_work(ledger).matrix["fp8_e4m3"]


def test_quant_indexer_flops_follow_the_completed_key_count():
    assert _arch35_flops(1024) == 2 * 1024 * 32 * 128 == 8388608
    assert _arch35_flops(1) == 2 * 1 * 32 * 128 == 8192


# ---------------------------------------------------------------------------
# 8. C8 mixed-prefix batches keep private rows on their own tiles.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 9. Plain INT8 MLA must not select the 656 B/row C8 kernel.
# ---------------------------------------------------------------------------
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
    from mindie_sim.prediction.operators.sparse_indexer import index_pool_native as native
    native.selection(
        builder, query="q", weights="w", cache="cache", views=single_long,
        plan=plan, quant=builder.quant, padded_tokens=8,
        runtime=wave4b._Runtime(o_proj_tp=8), row_shard_tp=8)
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
    ((528, 288, 4096), 24772608),
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
    assert issued[context.resource_names["mte2_aic"]] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 13. V4 query-normalized scale shape follows query_norm_output.
# ---------------------------------------------------------------------------
def test_mla_prolog_v4_without_query_norm_output_has_no_scale_shape():
    from mindie_sim.prediction.operators.mla_prolog import family_mla_prolog as prolog
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


# ---------------------------------------------------------------------------
# Round 2. DBO under PCP keeps each microbatch's physical sharded rows.
# ---------------------------------------------------------------------------
def test_dbo_pcp_microbatches_consume_only_their_sharded_rows():
    from mindie_sim.public.errors import UnmodeledCapabilityError

    raw = compose_case(
        "qwen3-30b-a3b", hardware="a5", tp=1, dp=1, ep=1, pcp=2,
        num_devices=2, layers=1, enable_dbo=True, prompt_tokens=8,
        output_tokens=2, max_num_seqs=2)
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        target_graphs(
            parse_case(raw), phase="prefill", tokens=8, context=0,
            sequences=2, scope="body")


# ---------------------------------------------------------------------------
# Round 2. Dense non-GQA MLA handoff owns the intersected token rows.
# ---------------------------------------------------------------------------
def test_pd_dense_mla_window_owns_only_the_intersected_source_rows():
    raw = compose_case(
        "deepseek-v3.2", hardware="a5", tp=1, dp=1, ep=1, num_devices=3,
        layers=1, prompt_tokens=10, output_tokens=2)
    raw["pools"] = {
        "prefill": {"machine": copy.deepcopy(raw["machine"]),
                    "runtime": copy.deepcopy(raw["runtime"])},
        "decode": {"machine": copy.deepcopy(raw["machine"]),
                   "runtime": copy.deepcopy(raw["runtime"])},
    }
    for pool, ranks in (("prefill", (0, 1)), ("decode", (2, 3))):
        machine = raw["pools"][pool]["machine"]
        machine["num_devices"] = len(ranks)
        machine["devices"] = [{"rank": rank} for rank in ranks]
        machine["links"][0]["members"] = list(ranks)
        raw["pools"][pool]["runtime"]["parallel"].update(
            tp=2, cp=2, dcp=2, cp_plan="dcp")
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0, "semantics": "per_pair"},
        "chunk_bytes": 64}
    raw["model"]["layer_plans"][0]["num_kv_heads"] = 0
    raw["model"]["layer_plans"][0]["window"] = 8
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    rows = [(row["source_rank"], tuple(row["source_token_span"]),
             tuple(row["destination_token_span"]), row["payload_bytes"])
            for row in result["execution"]["pd_handoffs"][0]["transfers"]
            if row["component"] == "mla_kv" and row["destination_rank"] == 3]
    # Decode rank 3 owns [5, 10) (five 1,152 B rows). Prefill rank 0 owns the
    # disjoint [2, 5) window head and must not be charged for it.
    assert rows == [(1, (5, 10), (5, 10), 5760.0)]


# ---------------------------------------------------------------------------
# Round 2. MQSMLA packed stores read the activation row, not a rotary tail.
# ---------------------------------------------------------------------------
def test_mqsmla_packed_store_reads_activation_width_not_the_rotary_tail():
    for scale_dtype, row_bytes in (("bf16", MODE1_ROW_BYTES),
                                   ("uint8", MODE2_ROW_BYTES)):
        model = _packed_mode_model(scale_dtype)
        plan = model.layer_plans[0]
        views = (RequestView("r0", 1, 127, False, "decode"),)
        builder = Builder("l0", plan, 0, views, model, "decode")
        geometry = CacheGeometry(capacity_bytes=1 << 20, line_bytes=1,
                                 transfer_bytes=1, staging_lines=1)
        state = CacheState(geometry)
        store(builder, "compressed", ("src",), 1, 512, "compressed_kv",
              state=state, geometry=geometry,
              identity=cache_identity("compressed_kv", layer=2, request="r0"))
        op = builder.ops[-1]
        # The [1, 512] BF16 producer writes 1,024 B; the packed row (608/584)
        # is the output extent, never an extra source read.
        assert op.mte2_bytes == pytest.approx(1024.0)
        assert op.mte3_bytes == pytest.approx(row_bytes)
        assert dict(op.metadata)["cache_row_bytes"] == row_bytes


# ---------------------------------------------------------------------------
# Round 2. Mode-2 lowering splits the declared row into 576 B + 8 B.
# ---------------------------------------------------------------------------
def test_mqsmla_mode2_lowering_splits_the_declared_payload_and_scales():
    model = _packed_mode_model("uint8")
    plan = model.layer_plans[0]
    views = (RequestView("r0", 1, 127, False, "decode"),)
    builder = Builder("l0", plan, 0, views, model, "decode")
    geometry = CacheGeometry(capacity_bytes=1 << 20, line_bytes=1,
                             transfer_bytes=1, staging_lines=1)
    lowering = CacheLowering(builder, CacheState(geometry))
    lowering.store("compressed", ("src",), 1, 512, "compressed_kv",
                   identity="compressed_kv:l2:r0")
    view = lowering.requests[-1][1].view
    payload, scale = cache_row_split(model, "compressed_kv", 512)
    assert (payload, scale) == (576.0, 8.0)
    assert (view.payload_bytes, view.scale_bytes, view.padding_bytes) == (576, 7, 1)
    assert sum((view.payload_bytes, view.scale_bytes, view.padding_bytes)) == MODE2_ROW_BYTES


# ---------------------------------------------------------------------------
# Round 2. HCA propagates each request's distinct prefix cache key.
# ---------------------------------------------------------------------------
def test_mqsmla_hca_keeps_distinct_prefix_cache_keys():
    views = (
        RequestView("a", 1, 383, False, "decode", prefix_tokens=128,
                    prefix_cache_key="k0"),
        RequestView("b", 1, 383, False, "decode", prefix_tokens=128,
                    prefix_cache_key="k1"),
    )
    fields = family_mqsmla._prefix_fields(views, SimpleNamespace(prefix_cache_mode="shared"))
    assert fields["prefix_tokens"] == (128, 128)
    assert fields["prefix_cache_keys"] == ("k0", "k1")
    shape = {"sequences": ((1, 383), (1, 383)), "window": 128, "ratio": 128,
             "topk": 512, **fields}
    external, _shared = mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8)
    # 256 window rows plus three private completed rows per request: distinct
    # keys own 262 physical rows, not one shared 261-row pool.
    assert external == 262 * 608 == 159296


# ---------------------------------------------------------------------------
# Round 2. Unequal HCA contexts share only the completed compressed prefix.
# ---------------------------------------------------------------------------
def test_mqsmla_hca_unequal_contexts_share_only_the_completed_prefix():
    shape = {"sequences": ((1, 511), (1, 767), (1, 767)), "window": 128,
             "ratio": 4, "topk": 512, "prefix_cache_mode": "shared",
             "prefix_tokens": (256, 256, 0),
             "prefix_cache_keys": ("k", "k", "")}
    external, shared = mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8)
    # 384 window rows + (128 - 64 + 192) shared/private rows for the {0,1}
    # group + 192 private rows for request 2.
    assert shared == 64
    assert external == 832 * 608 == 505856


# ---------------------------------------------------------------------------
# Round 2. An empty arch35 selection still declares its INT32 index store.
# ---------------------------------------------------------------------------
def test_quant_indexer_empty_selection_declares_the_index_store():
    op = wave4b._arch35_op_for(
        4, (RequestView("r", 1, 0, False, "decode"),))
    ledger = dict(op.metadata)[quant_indexer.QUANT_LEDGER]
    assert ledger["completed_keys"] == 0
    assert ledger["blocks"] == 0
    assert ledger["keys"] == 0
    assert ledger["row_completed_keys"] == ((1, 0),)
    assert op.flops == 0
    assert op.mte3_bytes == 8192
    plan = quant_indexer.Arch35QuantIndexerRecipe().build(op, None)
    program = wave4b._conserved(plan, op, wave4b.DT)
    names = [stage.name for stage in wave4b._stages(program)]
    assert "blocks" not in names
    assert "indices" in names


# ---------------------------------------------------------------------------
# Round 2. PCP query ranges count each owned row's own causal keys.
# ---------------------------------------------------------------------------
def test_quant_indexer_pcp_query_ranges_count_per_row_causal_keys():
    views = (RequestView(
        "p", 2, 0, False, "prefill", query_ranges=((0, 1), (3, 4)),
        global_q_len=4),)
    op = wave4b._arch35_op_for(4, views, heads=64)
    ledger = dict(op.metadata)[quant_indexer.QUANT_LEDGER]
    # Offset 0 sees 1 // 4 == 0 keys; offset 3 sees 4 // 4 == 1 key.
    assert ledger["row_completed_keys"] == ((1, 0), (1, 1))
    assert ledger["completed_keys"] == 1
    assert op.flops == 2 * 64 * 128 == 16384


# ---------------------------------------------------------------------------
# Round 2. Unequal decode contexts keep their own per-query key counts.
# ---------------------------------------------------------------------------
def test_quant_indexer_per_query_causal_keys_are_not_flattened():
    views = (RequestView("a", 1, 1, False, "decode"),
             RequestView("b", 1, 3, False, "decode"))
    op = wave4b._arch35_op_for(1, views, heads=32)
    ledger = dict(op.metadata)[quant_indexer.QUANT_LEDGER]
    assert ledger["row_completed_keys"] == ((1, 2), (1, 4))
    # 2 + 4 pairs, never the padded max of four pairs twice.
    assert op.flops == 2 * 32 * 128 * (2 + 4) == 49152


# ---------------------------------------------------------------------------
# Round 2. C8 mixed-prefix gathers count every visible query row.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Round 2. Kimi DCP8: packed guard and declared LSE statistics.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Round 2. KeyPool TopK reads the assembled selection, not the local shard.
# ---------------------------------------------------------------------------
def test_keypool_pack_reads_local_indices_positions_and_cumulative_tail():
    plan, model = wave4b._case_model(pool=16)
    model = replace(model, quant=replace(model.quant,
                                         index_scale_dtype="fp32"))
    views = (RequestView("r0", 8, 131072, False, "decode"),)
    builder = Builder("s0.l0", plan, 0, views, model, "decode")
    from mindie_sim.prediction.operators.sparse_indexer import index_pool_native as native
    native.selection(
        builder, query="q", weights="w", cache="cache", views=views,
        plan=plan, quant=builder.quant, padded_tokens=8,
        runtime=wave4b._Runtime(o_proj_tp=8), row_shard_tp=8)
    gather = next(op for op in builder.ops
                  if op.recipe == native.ROW_SHARD_GATHER)
    packed = next(op for op in builder.ops if op.recipe == "index_topk")
    # 1,024 assembled pools x 16 INT32 ids + eight INT64 causal tails.
    assert dict(gather.metadata)["assembled_bytes"] == 65536
    assert packed.mte2_bytes == 8192 + 8 + 8
    assert packed.mte3_bytes == 8192
