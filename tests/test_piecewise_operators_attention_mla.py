"""Physics invariants for the attention, MLA, indexer and recurrent lane.

[Role]
  Independently verify the piecewise attention/MLA/indexer/KPool/GDN/KDA
  models against structural formulas: head-group steps, page/burst traffic,
  split-K partial/LSE bytes, core-wave tails, latent absorption/expansion,
  sparse gather transactions, telescope compression and FP32 state matrices.

[Invariants]
  Every assertion is derived from declared heads, dimensions, tiles, request
  lengths, state or supply ratios.  No test reads a measured latency, a
  training table, a model name or a documentation string; names appear only to
  select the declared canonical operator under test.

[Timeouts & Locks]
  Small in-process geometry only; no device work, I/O, waits or locks.

[Side-Effect Policy]
  Read-only public registry and compiler construction; no file is written and no
  test patches the production family registry.
"""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from mindie_sim.catalog.index_pool import completed_prefix
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply, RequestGeometry,
    WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.families import FAMILY_COMPILERS
from mindie_sim.prediction.piecewise.families_attention import (
    _compile_attention, _compile_mla,
)
from mindie_sim.prediction.piecewise.families_recurrent_indexer import (
    BURST_BYTES, align, compile_compression, compile_indexer, k1_new_rows,
    range_burst_bytes, selected_token_rows,
)
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_model,
)
from mindie_sim.prediction.subgraphs.state import AcceptanceState, StateView
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature,
)


SUPPLY = PipeSupply((
    ("cube", 1_000_000.0), ("vector", 1_000_000.0), ("hbm", 1_000_000.0),
    ("mte1", 2_000_000.0), ("mte2", 2_000_000.0), ("mte3", 2_000_000.0),
    ("fixpipe", 2_000_000.0), ("link", 2_000_000.0),
))
RATIO_SUPPLY = PipeSupply((
    ("cube", 7_760_000.0), ("vector", 400_000.0), ("hbm", 1_000_000.0),
    ("mte1", 1_000_000.0), ("mte2", 1_000_000.0), ("mte3", 2_000_000.0),
    ("fixpipe", 2_000_000.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=4,
                              split_target=10_000, max_splits=4)


def make_policy(mt=16, nt=16, cores=4, split_target=10_000, max_splits=4):
    return ImplementationPolicy(mt=mt, nt=nt, cores=cores,
                                split_target=split_target, max_splits=max_splits)


def build_model(layer, *, layers=1, hidden_size=256, heads=8, kv_heads=2,
                head_dim=64):
    plan = dict(index=0, attention="gqa", ffn="dense", num_heads=heads,
                num_kv_heads=kv_heads, head_dim=head_dim,
                intermediate_size=128)
    plan.update(layer)
    raw = dict(name="lane2", layers=layers, hidden_size=hidden_size,
               num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
               intermediate_size=128, vocab_size=128, layer_plans=[plan])
    return ModelSpec.from_mapping(raw)


def compile_instance(instance, *, supply=SUPPLY, policy=POLICY):
    return OperatorPiecewiseCompiler(supply, policy).compile(instance)


def compile_attention(heads, kv_heads, head_dim, *, tp=1, mt=16, nt=16,
                      cores=4, window=0, supply=SUPPLY, split_target=10_000):
    model = build_model(dict(num_heads=heads, num_kv_heads=kv_heads,
                             head_dim=head_dim, window=window),
                        heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                        hidden_size=max(128, heads * head_dim))
    policy = make_policy(mt=mt, nt=nt, cores=cores, split_target=split_target)
    features = FeatureConfig.from_model(
        model, tp=tp, token_tile=mt, kv_tile=nt, cores=cores,
        split_target=split_target, max_splits=4)
    instantiated = instantiate_model(model, features)
    instance = instantiated.find("A0", layer=0, role="attention")
    return compile_instance(instance, supply=supply, policy=policy)


def attention_cost(compiled, geometry, state=None):
    return compiled.operator("attention").evaluate(geometry, state)


def decode(requests, *, phase="decode"):
    return WorkloadGeometry(tuple(RequestGeometry(q, context, phase=phase)
                                  for q, context in requests), phase=phase)


def prefill(requests):
    return decode(requests, phase="prefill")


# 1. Arbitrary H_q/H_kv grouping, including MHA, MQA and KV replication under TP.


def test_head_group_generalisation_and_tp_kv_replication():
    cases = ((1, 1, 1), (4, 4, 1), (8, 1, 8), (16, 2, 8), (12, 1, 12),
             (32, 1, 32), (64, 1, 64), (32, 4, 8), (32, 2, 16), (96, 3, 32),
             (16, 8, 2), (10, 3, 4), (13, 4, 4))
    for heads, kv_heads, expected in cases:
        compiled = compile_attention(heads, kv_heads, 64)
        cost = attention_cost(compiled, decode(((1, 32),)))
        assert cost.get("group_heads") == expected
        assert cost.get("query_heads") == heads
        assert cost.get("kv_heads") == kv_heads
        # One query-tile step when q*g fits M_t, otherwise ceil(qg/M_t).
        assert cost.get("query_tile_step") == max(1, math.ceil(expected / 16))

    # TP=8 over H_kv=2: two logical KV heads are replicated to one local head
    # per rank, so each rank serves 8 local query heads with g=8.
    replicated = compile_attention(64, 2, 64, tp=8)
    cost = attention_cost(replicated, decode(((1, 32),)))
    assert cost.get("query_heads") == 8
    assert cost.get("kv_heads") == 1
    assert cost.get("group_heads") == 8
    # TP=16 over H_kv=4: local 2 query heads over one replicated KV head.
    replicated = compile_attention(32, 4, 64, tp=16)
    cost = attention_cost(replicated, decode(((1, 32),)))
    assert (cost.get("query_heads"), cost.get("kv_heads"),
            cost.get("group_heads")) == (2, 1, 2)


# 2. Cube fractal 16 and UB 32-byte alignment for arbitrary head widths.


def test_head_dim_fractal_alignment_and_ub_padding():
    for dim in (64, 80, 96, 128, 192, 256, 512, 576):
        compiled = compile_attention(4, 1, dim)
        cost = attention_cost(compiled, decode(((1, 4),)))
        assert cost.get("dk_cube") == align(dim, 16) == dim
        assert cost.get("dv_cube") == align(dim, 16) == dim
        assert cost.get("dv_ub") == align(dim, 16) == dim
    # Non-16 widths are padded only for Cube/UB work, while HBM keeps the true
    # row width.
    for dim, padded in ((100, 112), (120, 128), (130, 144), (8, 16)):
        compiled = compile_attention(4, 1, dim)
        cost = attention_cost(compiled, decode(((1, 4),)))
        assert cost.get("dk_cube") == padded
        assert cost.get("dv_ub") == align(dim, 16)
        assert cost.get("kv_row_bytes") == 2 * dim * 2
        assert cost.get("q_row_width") == dim


# 3. Query-tile step j_q = ceil(q*g/M_t) repeats the KV sweep.


def test_query_tile_step_scales_kv_reads_only_above_mt():
    costs = {}
    for heads in (8, 32, 64):
        compiled = compile_attention(heads, 1, 64, mt=16, nt=16)
        costs[heads] = attention_cost(compiled, decode(((1, 32),)))
    assert costs[8].get("query_tile_steps") == 1
    assert costs[32].get("query_tile_steps") == 2
    assert costs[64].get("query_tile_steps") == 4
    # Decode q=1 keeps the same horizon per query tile, so KV burst bytes step
    # exactly with j_q instead of being amortized.
    one = costs[8].get("kv_burst_bytes")
    assert costs[32].get("kv_burst_bytes") == 2 * one
    assert costs[64].get("kv_burst_bytes") == 4 * one


# 4. KV tile steps and page/burst amplification.


def test_kv_tile_steps_and_page_burst_granularity():
    compiled = compile_attention(4, 1, 64, mt=128, nt=16)
    steps = []
    for context in (0, 15, 16, 17):
        cost = attention_cost(compiled, decode(((1, context),)))
        steps.append((context, cost.get("kv_tile_steps"),
                      cost.get("kv_burst_bytes")))
    assert [item[1] for item in steps] == [1, 1, 2, 2]
    assert steps[0][2] == BURST_BYTES + 8  # one 128-row page, minimum burst
    assert steps[2][2] == 17 * 256 + 8     # 17 valid rows inside one page

    # A one-row cache page costs one minimum burst per page instead of the
    # logical payload: small pages amplify MTE2 work.
    model = build_model(dict(num_heads=4, num_kv_heads=1, head_dim=8))
    policy = make_policy(mt=128, nt=16)
    features = FeatureConfig.from_model(model, token_tile=128, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    instance = instantiate_model(model, features).find("A0", layer=0,
                                                       role="attention")
    base = next(item for item in instance.operators
                if item.name == "attention")
    big = OperatorPiecewiseCompiler(SUPPLY, policy).compile_operator(
        BoundOperator(base.template, base.theta.relocated(page=128)))
    small = OperatorPiecewiseCompiler(SUPPLY, policy).compile_operator(
        BoundOperator(base.template, base.theta.relocated(page=16)))
    geometry = decode(((1, 16),))
    big_cost = big.evaluate(geometry)
    small_cost = small.evaluate(geometry)
    assert small_cost.get("kv_burst_bytes") > big_cost.get("kv_burst_bytes")
    assert small_cost.get("page_transactions") > big_cost.get("page_transactions")
    expected, pages = range_burst_bytes(0, 17, Fraction(32), page=16)
    assert pages == 2 and expected == 1040


# 5. FIA long-sequence pipe switch: Vector -> MTE2 -> Cube.


def test_fia_long_sequence_vector_mte2_cube_switching():
    compiled = compile_attention(4, 1, 576, mt=128, nt=128, cores=2,
                                 supply=RATIO_SUPPLY, split_target=10_000_000)
    pipes = {}
    for context in (1, 8, 1024):
        cost = attention_cost(compiled, decode(((1, context),)))
        pipes[context] = cost.dominant_pipe
        assert cost.duration_ms == sum(cost.stage_times)
    assert pipes[1] == "vector"   # per-query softmax/accumulator overhead
    assert pipes[8] == "mte2"     # paged K/V plus Q load
    assert pipes[1024] == "cube"  # QK^T + PV dominate the long sweep
    short = attention_cost(compiled, decode(((1, 8),)))
    long = attention_cost(compiled, decode(((1, 1024),)))
    assert long.pipe_time("mte2") > short.pipe_time("mte2")
    assert long.pipe_time("cube") > short.pipe_time("cube")
    assert long.pipe_time("cube") > long.pipe_time("mte2")


# 6. Continuous crossover thresholds are exact declared-supply ratios.


def test_pipe_crossover_thresholds_match_declared_supply_ratios():
    compiled = compile_attention(4, 1, 576, mt=128, nt=128, cores=2,
                                 supply=RATIO_SUPPLY, split_target=10_000_000)
    cost = attention_cost(compiled, decode(((1, 2048),)))
    head_row = Fraction(2304)
    cube_row = Fraction(2 * (576 + 576))
    vec_pair = 7 + Fraction(3 * 576 + 10, 128)
    cube_rate, mte2_rate, vec_rate = (Fraction(7_760_000), Fraction(1_000_000),
                                      Fraction(400_000))
    expected_cube = head_row * cube_rate / (cube_row * mte2_rate)
    expected_vec = head_row * vec_rate / (vec_pair * mte2_rate)
    assert cost.get("cube_mte2_balance_rows") == expected_cube
    assert cost.get("cube_mte2_wave_rows") == expected_cube / 2
    assert cost.get("vector_mte2_balance_rows") == expected_vec
    assert cost.get("cube_vector_ratio") == (
        cube_row * vec_rate / (vec_pair * cube_rate))
    # With a full wave of two tasks, g=4 is below the cube/MTE2 threshold and
    # g=8 is above it: the dominant pipe flips without a fitted lookup.
    geometry = decode(((1, 2048), (1, 2048)))
    small = compile_attention(4, 1, 576, mt=128, nt=128, cores=2,
                              supply=RATIO_SUPPLY, split_target=10_000_000)
    large = compile_attention(8, 1, 576, mt=128, nt=128, cores=2,
                              supply=RATIO_SUPPLY, split_target=10_000_000)
    assert attention_cost(small, geometry).dominant_pipe == "mte2"
    assert attention_cost(large, geometry).dominant_pipe == "cube"


# 7. Split-K P(x) partial output/LSE traffic and vector merge.


def test_split_k_partial_lse_bytes_and_vector_merge():
    policy = make_policy(split_target=64, cores=4)
    model = build_model(dict(num_heads=8, num_kv_heads=2, head_dim=64))
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=64, max_splits=4)
    compiled = compile_instance(
        instantiate_model(model, features).find("A0", layer=0,
                                                role="attention"),
        policy=policy).operator("attention")
    geometry = decode(((1, 512), (1, 512)))
    cost = compiled.evaluate(geometry)
    splits = int(cost.get("kv_splits"))
    rows = int(cost.get("query_rows"))
    assert 2 <= splits <= 4
    expected = 2 * (splits - 1) * rows * (4 * 64 + 8)
    assert cost.get("split_partial_bytes") == expected
    merge = dict(cost.stage_work[1])
    assert merge["mte2"] == Fraction((splits - 1) * rows * (4 * 64 + 8))
    assert merge["mte3"] == Fraction(rows * 64 * 2)
    assert cost.get("split_merge_vector") == (
        (splits - 1) * rows * (3 * 64 + 9) + rows * (64 + 1))
    # A single split owns the final output and writes no partial.
    single = compiled.evaluate(decode(((1, 4),)))
    assert single.get("kv_splits") == 1
    assert single.get("split_partial_bytes") == 0
    assert dict(single.stage_work[1])["mte2"] == 0


# 8. Core-wave tail w = ceil(J/C) cannot borrow idle cores.


def test_core_wave_tail_charges_a_full_wave():
    model = build_model(dict(num_heads=8, num_kv_heads=2, head_dim=64))
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    compiled = compile_instance(
        instantiate_model(model, features).find("A0", layer=0,
                                                role="attention")).operator(
        "attention")
    three = compiled.evaluate(decode(((1, 32), (1, 32), (1, 32))))
    assert three.get("query_tile_steps") == 6  # 3 requests x 2 KV heads
    assert three.get("waves") == 2 and three.get("tail_cores") == 2
    assert three.get("active_cores") == 4
    assert three.pipe_work("cube") == three.get("physical_flops") * Fraction(4, 3)
    four = compiled.evaluate(decode(((1, 32),) * 4))
    assert four.get("waves") == 2 and four.get("tail_cores") == 4
    assert four.pipe_work("cube") == four.get("physical_flops")


# 9. MLA absorption/expansion, latent width and cache row bytes.


def test_mla_absorption_expansion_and_latent_width():
    layer = dict(attention="mla", num_heads=8, num_kv_heads=1, head_dim=576,
                 q_lora_rank=128, kv_lora_rank=512, qk_nope_head_dim=512,
                 qk_rope_head_dim=64, v_head_dim=128)
    model = build_model(layer, hidden_size=1024, heads=8, kv_heads=1,
                        head_dim=576)
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    instance = instantiate_model(model, features).find("A1", layer=0,
                                                       role="attention")
    compiled = compile_instance(instance)
    cost = compiled.operator("mla_attention").evaluate(decode(((1, 0),)))
    rows = int(cost.get("query_rows"))
    assert rows == 8
    assert cost.get("latent_width") == 576
    assert cost.get("dk_cube") == 576 and cost.get("dv_cube") == 512
    assert cost.get("kv_row_bytes") == (512 + 64) * 2
    absorb = 2 * rows * align(512, 16) * align(512, 16)
    expand = 2 * rows * align(512, 16) * align(128, 16)
    assert cost.get("absorb_flops") == absorb
    assert cost.get("v_expand_flops") == expand
    assert cost.get("logical_flops") == 2 * 1 * 8 * (576 + 512) + absorb + expand
    assert cost.get("absorb_weight_bytes") == 8 * 512 * 512 * 2
    assert cost.get("expand_weight_bytes") == 8 * 512 * 128 * 2
    # Grouped output projection ledger is available when the declared operator
    # carries o_a/o_b geometry; the canonical A1 signature leaves it separate.
    grouped_op = next(item for item in instance.operators
                      if item.name == "mla_attention")
    grouped = BoundOperator(grouped_op.template,
                            grouped_op.theta.relocated(o_groups=4,
                                                       o_lora_rank=64))
    grouped_cost = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        grouped).evaluate(decode(((1, 0),)))
    assert grouped_cost.get("grouped_o_a_flops") > 0
    assert grouped_cost.get("grouped_o_b_flops") > 0
    # The latent cache row is (r_kv + d_r), not two GQA K/V rows.
    assert cost.get("kv_burst_amplification") == Fraction(145, 144)


def test_mla_window_bounds_the_kv_sweep():
    layer = dict(attention="mla", num_heads=4, num_kv_heads=1, head_dim=576,
                 q_lora_rank=64, kv_lora_rank=512, qk_nope_head_dim=512,
                 qk_rope_head_dim=64, v_head_dim=128, window=128)
    model = build_model(layer, hidden_size=1024, heads=4, kv_heads=1,
                        head_dim=576)
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    compiled = compile_instance(
        instantiate_model(model, features).find("A1", layer=0,
                                                role="attention"))
    cost = compiled.operator("mla_attention").evaluate(decode(((1, 4096),)))
    assert cost.get("kv_positions") == 128
    open_cost = compiled.operator("mla_attention").evaluate(decode(((1, 64),)))
    assert open_cost.get("kv_positions") == 65


# 10. A2 shared-KV window/compressed attention and K1 telescope rows.


def _a2_model(**overrides):
    layer = dict(attention="gqa", num_heads=8, num_kv_heads=2, head_dim=64,
                 cache_kind="window_compressed", window=16, compress_ratio=4,
                 compressor_kind="weighted_pool", indexer=True, index_heads=4,
                 index_dim=32, index_topk=16, index_kpool=1, owns_long_kv=True)
    layer.update(overrides)
    model = build_model(layer)
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    return model, instantiate_model(model, features)


def test_shared_kv_window_and_sparse_compressed_attention():
    model, instantiated = _a2_model()
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    del features
    compiled = compile_instance(instantiated.find("A2", layer=0,
                                                  role="attention"))
    geometry = decode(((4, 28), (4, 28)))
    window = compiled.operator("window_attention").evaluate(geometry)
    # The union of four 16-token windows is 16 + 3 positions per request.
    assert window.get("kv_positions") == 2 * 2 * (16 + 3)
    sparse = compiled.operator("compressed_attention").evaluate(geometry)
    expected_rows = 4 * sum(min((28 + token + 1) // 4, 16)
                            for token in range(4))
    assert sparse.get("selected_rows") == expected_rows
    # One new compressed row per completed group over each request step.
    assert sparse.get("compressed_rows") == 2 * (32 // 4 - 28 // 4)
    # The compressed cache is gathered row by row; a 256-byte row still costs a
    # minimum MTE2 burst, so transaction bytes exceed the logical payload.
    assert sparse.get("kv_burst_amplification") > Fraction(19, 10)
    assert sparse.get("causal_pairs") == 8 * (expected_rows // 2)


def test_selected_token_rows_matches_pool_geometry():
    assert selected_token_rows(10, 4, 16) == 2
    assert selected_token_rows(10, 4, 4) == 2
    assert selected_token_rows(17, 4, 4, kpool=2) == 4
    assert selected_token_rows(17, 1, 8, kpool=4,
                               always_select_tail=True) == 9
    assert selected_token_rows(5, 1, 4) == 4


# 11. LightningIndexer gather transactions and Cube->Fixpipe->Vector chain.


def test_indexer_gather_and_cube_fixpipe_vector_dependency():
    model, instantiated = _a2_model()
    compiled = compile_instance(instantiated.find("I1", layer=0,
                                                  role="index_select"))
    geometry = decode(((4, 28), (4, 28)))
    score = compiled.operator("index_score").evaluate(geometry)
    rows = sum(completed_prefix(32, 1) - completed_prefix(28, 1)
               for _ in range(2))
    assert score.get("index_rows") == rows
    assert score.get("qk_int8_flops") == 2 * rows * 4 * 32
    assert score.get("head_reduce_flops") == 2 * rows * 16
    assert score.get("fixpipe_bytes") == 2 * rows * 4 + 4 * rows
    assert score.get("gather_transactions") == rows
    assert score.get("gather_amplification") == Fraction(65, 4)
    assert len(score.stage_work) == 6
    assert score.pipe_work("cube") == 2 * rows * 4 * 32
    assert score.pipe_work("fixpipe") == 2 * rows * 4 + 4 * rows
    assert score.pipe_work("vector") == 2 * rows * 16
    assert score.pipe_work("mte3") == rows * 4
    assert any(dict(score.stage_work[5]).values())
    assert dict(score.materialization)["index_scores"] == rows * 4
    assert score.duration_ms == sum(score.stage_times)
    selection = compiled.operator("topk").evaluate(geometry)
    assert selection.get("selected_tokens") == 2 * 4 * 16
    assert selection.pipe_work("mte3") == 2 * 4 * 16 * 4
    assert selection.pipe_work("mte2") == dict(
        score.materialization)["index_scores"]
    single = compiled.operator("index_score").evaluate(decode(((1, 0),)))
    assert len(single.stage_work) == 6
    assert not any(dict(single.stage_work[5]).values())
    assert all(single.stage_times[i] > 0 for i in range(5))
    early = decode(((4, 0),))
    early_score = compiled.operator("index_score").evaluate(early)
    early_select = compiled.operator("topk").evaluate(early)
    assert early_select.pipe_work("mte2") == dict(
        early_score.materialization)["index_scores"] == 40

    # Arbitrary index heads/dim/top-k/pool: scans pool rows, not token rows.
    pooled_layer = dict(index_heads=8, index_dim=80, index_topk=32,
                        index_kpool=4)
    pooled_model, pooled = _a2_model(**pooled_layer)
    features = FeatureConfig.from_model(pooled_model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    del features
    pooled_compiled = compile_instance(pooled.find("I1", layer=0,
                                                   role="index_select"))
    pooled_cost = pooled_compiled.operator("index_score").evaluate(geometry)
    pooled_rows = sum(completed_prefix(32, 4) - completed_prefix(28, 4)
                      for _ in range(2))
    assert pooled_cost.get("index_rows") == pooled_rows
    assert pooled_cost.get("qk_int8_flops") == 2 * pooled_rows * 8 * 80
    assert pooled_cost.get("gather_transactions") == pooled_rows
    zero = decode(((1, 0),))
    assert pooled_compiled.operator("index_score").evaluate(zero).materialization == (("index_scores", 0),)
    assert pooled_compiled.operator("topk").evaluate(zero).pipe_work("mte2") == 0


# 12. K1 telescope and KPool row production.


def test_k1_compression_telescopes_across_chunks():
    model, instantiated = _a2_model()
    compiled = compile_instance(instantiated.find("K1", layer=0,
                                                  role="compress"))
    compress = compiled.operator("compress")
    chunks = (3, 5, 2, 7)
    state = StateView()
    context = 0
    produced = []
    for query in chunks:
        geometry = WorkloadGeometry.homogeneous(1, query, context + query)
        evaluation = compiled.evaluate(geometry, state)
        cost = next(item for item in evaluation.operators
                    if item.name == "compress")
        expected = k1_new_rows(context + query, query, 4)
        assert expected == (context + query) // 4 - context // 4
        assert cost.get("new_rows") == expected
        produced.append(expected)
        state = evaluation.state_out
        context += query
    assert produced == [0, 2, 0, 2]
    assert sum(produced) == 17 // 4
    assert sum(math.ceil(query / 4) for query in chunks) == 6
    assert dict(state.compression)[0].remainder == 1
    assert compress is not None


def test_kpool_rows_and_tail_retention():
    model, instantiated = _a2_model(index_kpool=4,
                                    index_kpool_always_select_tail=True)
    compiled = compile_instance(instantiated.find("K1", layer=0,
                                                  role="compress"))
    kpool = compiled.operator("kpool")
    geometry = WorkloadGeometry((RequestGeometry(7, 10),))
    cost = kpool.evaluate(geometry)
    pools = k1_new_rows(17, 7, 4)
    assert pools == 2
    assert cost.get("pool_rows") == 2
    assert cost.get("pool_out_rows") == 3
    assert cost.get("remainder") == 1
    row_bytes = 2 * 2 * 64 * 2
    assert cost.pipe_work("mte2") == 2 * 4 * row_bytes
    assert cost.pipe_work("mte3") == 3 * row_bytes


# 13. GDN CausalConv1d state, chunked recurrence and FP32 state matrix.


def test_gdn_conv_state_and_chunked_recurrence():
    layer = dict(attention="gdn", linear_heads=2, linear_key_heads=2,
                 linear_value_heads=2, linear_key_head_dim=8,
                 linear_head_dim=8, conv_kernel=4,
                 recurrent_state_elements=64, conv_state_elements=16,
                 state_bytes=256)
    model = build_model(layer)
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    instance = instantiate_model(model, features).find("A3", layer=0,
                                                       role="attention")
    compiled = compile_instance(instance)
    geometry = prefill(((8, 24), (8, 24)))
    conv = compiled.operator("conv").evaluate(geometry)
    qkv_width = 2 * 2 * 8 + 2 * 8
    assert conv.get("conv_kernel") == 4
    assert conv.get("conv_state_bytes") == 4 * qkv_width * 2
    assert conv.pipe_work("mte2") == 16 * qkv_width * 2 + 2 * 4 * qkv_width * 2
    assert conv.get("conv_ops") > 0

    delta = compiled.operator("gated_delta")
    cost = delta.evaluate(geometry)
    matrix = 2 * 8 * 8 * 4
    assert cost.get("state_matrix_bytes") == matrix
    assert cost.get("state_bytes") == 2 * matrix
    assert cost.get("commit_rows") == 2
    assert cost.get("chunk_size") == 8
    assert cost.get("chunks") == 2  # one chunk per request
    assert cost.get("flops") == 16384
    assert cost.get("input_bytes") == 16 * (2 * 2 * 8 + 2 * 8 + 2 * 8) * 2
    assert cost.get("output_bytes") == 16 * 2 * 8 * 2
    assert cost.pipe_work("mte2") == 2 * matrix + cost.get("input_bytes")
    assert cost.pipe_work("mte3") == 2 * matrix + cost.get("output_bytes")
    for act_bytes in (2, 4):
        datum = BoundOperator(
            delta.operator.template,
            delta.operator.theta.relocated(act_bytes=act_bytes))
        widened = OperatorPiecewiseCompiler(
            SUPPLY, POLICY).compile_operator(datum).evaluate(geometry)
        assert widened.get("act_bytes") == act_bytes
        assert widened.get("input_bytes") == 16 * 64 * act_bytes
        assert widened.get("output_bytes") == 16 * 2 * 8 * act_bytes
        assert widened.get("flops") == cost.get("flops")

    narrow = BoundOperator(delta.operator.template,
                           delta.operator.theta.relocated(chunk=4))
    narrow_delta = OperatorPiecewiseCompiler(
        SUPPLY, POLICY).compile_operator(narrow).evaluate(geometry)
    assert narrow_delta.get("chunks") == 4
    assert narrow_delta.get("flops") < cost.get("flops")


def test_gdn_generalises_key_value_heads_dims_and_conv_kernel():
    layer = dict(attention="gdn", linear_heads=2, linear_key_heads=2,
                 linear_value_heads=4, linear_key_head_dim=16,
                 linear_head_dim=8, conv_kernel=6,
                 recurrent_state_elements=64, conv_state_elements=16,
                 state_bytes=256)
    model = build_model(layer)
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    instance = instantiate_model(model, features).find("A3", layer=0,
                                                       role="attention")
    compiled = compile_instance(instance)
    geometry = decode(((1, 8),) * 3)
    conv = compiled.operator("conv").evaluate(geometry)
    qkv_width = 2 * 2 * 16 + 4 * 8
    assert conv.get("conv_kernel") == 6
    assert conv.get("conv_state_bytes") == 6 * qkv_width * 2
    delta = compiled.operator("gated_delta").evaluate(geometry)
    matrix = 4 * 16 * 8 * 4  # H_v * d_k * d_v * 4
    assert (delta.get("key_heads"), delta.get("value_heads"),
            delta.get("dk"), delta.get("dv")) == (2, 4, 16, 8)
    assert delta.get("state_matrix_bytes") == matrix
    assert delta.get("state_bytes") == 3 * matrix
    assert delta.get("commit_rows") == 3
    per_request = 2 * (16 * 2 + 8 * 4) + 4 * 16 * 8 * 4
    assert delta.get("flops") == 3 * per_request


# 14. KDA accepted-slot state commit and exact FP32 state bytes.


def test_kda_accepted_slots_commit_exact_state_matrix():
    layer = dict(attention="kda", linear_heads=2, linear_key_heads=2,
                 linear_value_heads=2, linear_key_head_dim=8,
                 linear_head_dim=8, conv_kernel=4, gate_rank=4,
                 recurrent_state_elements=64, conv_state_elements=16,
                 state_bytes=256)
    model = build_model(layer)
    features = FeatureConfig.from_model(model, token_tile=16, kv_tile=16,
                                        cores=4, split_target=10_000,
                                        max_splits=4)
    instance = instantiate_model(model, features).find("A4", layer=0,
                                                       role="attention")
    compiled = compile_instance(instance)
    op = compiled.operator("kda_recurrent")
    geometry = decode(((1, 8),) * 4)
    matrix = 2 * 8 * 8 * 4
    values = {}
    for accepted in (0, 2, 8):
        state = StateView(acceptance=AcceptanceState(accepted=accepted))
        cost = op.evaluate(geometry, state)
        values[accepted] = (cost.get("accepted_slots"),
                            cost.get("commit_rows"), cost.pipe_work("mte3"))
        assert cost.get("state_matrix_bytes") == matrix
        assert cost.get("state_bytes") == 4 * matrix
    assert values[0][:2] == (0, 4)
    assert values[2][:2] == (2, 4)   # accepted slots below batch still commit once
    assert values[8][:2] == (8, 8)   # accepted slots above batch add commits
    assert values[8][2] == 8 * matrix + 4 * 2 * 8 * 2
    assert values[0][2] == 4 * matrix + 4 * 2 * 8 * 2


# 15. Recurrent algebra helpers are exact and reject invalid geometry.


def test_compression_helper_boundaries_and_invalid_geometry():
    assert k1_new_rows(17, 7, 4) == 2
    assert k1_new_rows(16, 4, 4) == 1
    assert k1_new_rows(3, 3, 4) == 0
    assert sum(k1_new_rows(total, query, 8)
               for total, query in ((8, 8), (17, 9), (32, 15))) == 32 // 8
    with pytest.raises(ModelingError):
        k1_new_rows(4, 5, 4)
    with pytest.raises(ModelingError):
        k1_new_rows(4, 1, 0)
    assert align(0, 16) == 0 and align(17, 16) == 32
    payload, pages = range_burst_bytes(0, 0, Fraction(128))
    assert payload == 0 and pages == 0


# 16. The production registry owns the refined indexer/compression compilers.


def test_public_registry_uses_refined_indexer_and_compression():
    assert FAMILY_COMPILERS["indexer"] is compile_indexer
    assert FAMILY_COMPILERS["compression"] is compile_compression
    model, instantiated = _a2_model()
    compiled = compile_instance(instantiated.find("K1", layer=0,
                                                  role="compress"))
    geometry = WorkloadGeometry((RequestGeometry(4, 0),))
    compress = compiled.operator("compress").evaluate(geometry)
    row_bytes = 2 * 2 * 64 * 2
    assert compress.get("new_rows") == 1
    assert compress.get("source_rows") == 4
    assert compress.pipe_work("mte2") >= 4 * row_bytes
    assert compress.pipe_work("mte3") == row_bytes


# 17. Ragged attention waves obey the longest pinned task lower bound.


def test_ragged_attention_wave_obeys_longest_task_bound():
    supply = PipeSupply((
        ("cube", 1000.0), ("vector", 1e12), ("hbm", 1e12),
        ("mte1", 1e12), ("mte2", 1e12), ("mte3", 1e12),
        ("fixpipe", 1e12), ("link", 1e12),
    ))
    compiled = compile_attention(1, 1, 64, cores=4, supply=supply)
    cost = attention_cost(compiled, decode(((1, 0), (1, 1023))))
    long_task = 2 * 1 * 1024 * (64 + 64)
    assert cost.get("cube_task_max") == long_task
    assert cost.pipe_work("cube") == 4 * long_task
    assert cost.duration_ms >= Fraction(4 * long_task, 1000)
    balanced = attention_cost(compiled, decode(((1, 1023), (1, 1023))))
    assert balanced.get("cube_task_max") == long_task
    assert balanced.pipe_work("cube") == 4 * long_task


# 18. Query ranges select physical rows, Q loads and output rows.


def test_query_ranges_select_physical_attention_rows():
    compiled = compile_attention(4, 1, 64, cores=4)
    partial = attention_cost(compiled, WorkloadGeometry((
        RequestGeometry(4, 0, query_ranges=((0, 1),)),)))
    assert partial.get("query_rows") == 4
    assert partial.get("q_bytes") == 1 * 4 * 64 * 2
    assert partial.get("output_bytes") == 1 * 4 * 64 * 2
    assert partial.get("query_tile_steps") == 1
    full = attention_cost(compiled, decode(((4, 0),)))
    assert full.get("query_rows") == 16
    assert full.get("q_bytes") == 4 * 4 * 64 * 2
    assert full.get("output_bytes") == 4 * 4 * 64 * 2


def test_unsorted_query_ranges_keep_the_physical_tile_bound():
    compiled = compile_attention(4, 1, 64, cores=4)
    costs = []
    for ranges in (((0, 1), (3, 4)), ((3, 4), (0, 1))):
        geometry = WorkloadGeometry((RequestGeometry(4, 0,
                                                     query_ranges=ranges),))
        cost = attention_cost(compiled, geometry)
        assert cost.get("physical_flops") >= cost.get("logical_flops")
        costs.append(cost)
    assert costs[0].get("physical_pairs") == costs[1].get("physical_pairs") == 32
    assert costs[0].get("physical_flops") == costs[1].get("physical_flops")


# 19. Vision stage work uses media tokens, never tile counts.


def test_vision_stage_work_uses_media_tokens_not_tiles():
    template = OperatorTemplate(name="vision_encoder", family="vision",
                                role="vision")
    theta = StaticSignature(
        role="vision", matrix=(("width", 64),), tiles=(("mt", 128),),
        extras=(("patch_width", 64), ("layers", 1), ("heads", 4)))
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        BoundOperator(template, theta))
    geometry = WorkloadGeometry((RequestGeometry(1, 0),), media_tokens=256)
    cost = compiled.evaluate(geometry)
    assert cost.get("media_tokens") == 256
    assert cost.get("flops") == 2 * 256 * 64 * 64 + 2 * 256 * 64 * 64
    assert cost.pipe_work("mte3") == 256 * 64 * 2
    assert dict(cost.materialization)["media_hidden"] == 256 * 64 * 2


# 20. Ragged GQA loads only the projected query heads from HBM.


def test_ragged_gqa_q_bytes_uses_true_query_heads():
    compiled = compile_attention(10, 3, 64, cores=4)
    cost = attention_cost(compiled, decode(((1, 32),)))
    assert cost.get("group_heads") == 4
    assert cost.get("query_rows") == 10
    assert cost.get("q_bytes") == 1 * 10 * 64 * 2
    assert cost.get("kv_row_bytes") == 3 * (64 + 64) * 2


# 21. Ordinary attention KV rows support independent K and V widths.


def test_default_kv_row_supports_unequal_key_value_widths():
    template = OperatorTemplate(name="attention", family="attention",
                                role="attention")
    theta = StaticSignature(
        role="attention",
        heads=(("q", 4), ("kv", 1), ("dk", 80), ("dv", 128)),
        quant=(("act_bytes", 2.0),),
        tiles=(("mt", 16), ("nt", 16), ("cores", 4)),
        extras=(("causal", 1), ("window", 0)))
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        BoundOperator(template, theta))
    cost = compiled.evaluate(decode(((1, 4),)))
    assert cost.get("kv_row_bytes") == 1 * (80 + 128) * 2
    assert cost.get("q_row_width") == 80
