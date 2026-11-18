"""v0.9.19 physical routing, cache and caller conservation.

[Role] Verify the eleven R3 results on the public graph and recipe boundaries.
[Invariants] Expected values follow from row identities, placement and byte widths.
[Timeouts & Locks] Finite pure-Python checks, no locks.
[Side-Effect Policy] No accelerator, weights or persistent state.
"""
from dataclasses import replace
from itertools import combinations, product
from types import SimpleNamespace

import pytest

import test_v0919_final_delta_math as prior
import test_wave4a_c8sfa_mqsmla_calibration as a
import test_wave4b_indexer_conv1d_calibration as b
from mindie_sim.catalog.mqsmla_layout import packed_row_parts
from mindie_sim.prediction import family_mqsmla, index_pool_native, latent_dcp, mqsmla
from mindie_sim.prediction.batch_prefix_attention import nested_prefix_union
from mindie_sim.prediction.dbo_ledger import _capacities
from mindie_sim.prediction.index_pool import native_shard_sizes
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.pd_ranges import transfers_for_runs
from mindie_sim.schema.workload import RequestView


@pytest.mark.parametrize("dp,ep,pcp", [(4, 4, 2), (1, 2, 2), (1, 1, 2)])
def test_dbo_matches_unsplit_physical_assignments(dp, ep, pcp):
    raw = prior.compose_case(
        "qwen3-30b-a3b", hardware="a5", tp=1, dp=dp, ep=ep, pcp=pcp,
        num_devices=dp * pcp, layers=1, enable_dbo=True,
        prompt_tokens=8, output_tokens=2, max_num_seqs=2)
    if ep > dp:
        raw["runtime"]["parallel"]["rank_table"] = [[0, 1]]
    def graph():
        return prior.target_graphs(prior.parse_case(raw), phase="prefill",
                                   tokens=8, context=0, sequences=2, scope="body")
    split = graph()
    raw["runtime"]["enable_dbo"] = False
    whole = graph()
    for replica, ops in split.items():
        packs = [op.local_shape[0] * op.local_shape[1] for op in ops if op.recipe == "route_pack"]
        gates = [op.local_shape[0] for op in ops if op.recipe == "grouped_gate_up"]
        unpack = [op.local_shape[0] * op.local_shape[1] for op in ops if op.recipe == "route_unpack"]
        assert packs == gates == unpack == [32, 32]
        assert sum(gates) == sum(op.local_shape[0] for op in whole[replica]
                                 if op.recipe == "grouped_gate_up")


def test_capture_capacity_is_distributed_once():
    assert _capacities([2, 1], 8, 2) == [2, 6]
    assert _capacities([1, 1], 2, 2) == ()
    assert _capacities([5, 3], 8, 1) == [5, 3]


def test_two_receivers_read_one_named_source_extent():
    raw = prior._deepseek_pd_case()
    raw["workload"]["requests"][0]["prompt_tokens"] = 10
    for side, ranks in (("prefill", [0, 1]), ("decode", [2, 3, 4, 5])):
        cfg = raw["pools"][side]
        cfg["machine"].update(num_devices=len(ranks), devices=[{"rank": r} for r in ranks])
        cfg["machine"]["links"][0]["members"] = ranks
        cfg["runtime"]["parallel"].update(tp=len(ranks), dcp=2, cp=2, cp_plan="dcp")
    raw["model"]["layer_plans"][0].update(num_kv_heads=0, window=8, owns_index_k=False)
    result = prior.predict_pd(raw, lower=prior.linear_work, event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]
    rows = [t for t in transfers if t["component"] == "mla_kv" and t["source_rank"] == 1]
    assert {t["destination_rank"] for t in rows} == {3, 5}
    assert len({(t["source_storage"], t["source_version"]) for t in rows}) == 1
    assert all(tuple(t["source_byte_range"]) == (0, 5760) for t in rows)
    assert all(tuple(t["source_token_span"]) == (5, 10) for t in rows)
    assert len({(t["destination_rank"], t["destination_storage"]) for t in rows}) == 2
    assert sum(t["payload_bytes"] for t in rows) == 11520


def test_interleaved_rows_map_to_offsets_in_the_packed_owner():
    sources = [(9, ((0, 2), (4, 6)), 40), (7, ((2, 4), (6, 8)), 40)]
    actual = list(transfers_for_runs(sources, ((1, 7),)))
    assert [(rank, lo, hi, offset) for rank, lo, hi, size, offset, unit in actual] == [
        (9, 1, 2, 10), (7, 2, 4, 0), (9, 4, 6, 20), (7, 6, 7, 20)]
    assert sum((hi - lo) * unit for _, lo, hi, _, _, unit in actual) == 60


@pytest.mark.parametrize("dtype,parts", [("bf16", (576, 14, 18)), ("uint8", (576, 7, 1)),
                                       ("fp8_e8m0", (576, 7, 1))])
def test_arithmetic_scale_bytes_and_allocator_extent(dtype, parts):
    from mindie_sim.catalog.cache import cache_row_split
    model = prior._packed_mode_model(dtype)
    assert packed_row_parts(model, "compressed_kv", 512) == parts
    assert cache_row_split(model, "compressed_kv", 512) == (parts[0], sum(parts[1:]))


def test_hca_runtime_decides_aliasing():
    views = tuple(RequestView(key, 1, 383, False, "decode", prefix_tokens=128,
                              prefix_cache_key="A") for key in ("a", "b"))
    for mode, total, union in (("private", 159296, 0), ("shared", 158688, 1)):
        fields = family_mqsmla._prefix_fields(views, SimpleNamespace(prefix_cache_mode=mode))
        shape = dict(sequences=((1, 383),) * 2, window=128, ratio=128, topk=0, **fields)
        assert mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8) == (total, union)


def test_nested_upper_bound_is_attainable_by_enumeration():
    draws = [(extent, count) for extent in range(4) for count in range(extent + 1)]
    for spec in product(draws, repeat=3):
        options = [list(combinations(range(n), k)) for n, k in spec]
        exact = max(len(set().union(*chosen)) for chosen in product(*options))
        assert nested_prefix_union(spec) == exact


def test_csa_nested_prefix_and_private_window():
    contexts = (4, 4, 4096)
    shape = dict(sequences=tuple((1, c) for c in contexts), prefix_tokens=contexts,
                 prefix_cache_keys=("A",) * 3, prefix_cache_mode="shared",
                 ratio=4, window=128, topk=512)
    assert mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8) == (395808, 513)


@pytest.mark.parametrize("contexts,keys,topk,union,local", [
    ((1, 1, 1024), ("A", "A", "A"), 512, 513, 2),
    ((63, 63), ("A", "A"), 128, 63, 2),
    ((127, 127, 127), ("A", "A", "B"), 128, 127, 130),
])
def test_c8_prefix_plus_tile_rows(contexts, keys, topk, union, local):
    sequences = tuple((1, c) for c in contexts)
    op = a._c8_op(sequences, topk, 64, 0)
    shape = dict(dict(op.metadata)["indexed_attention_geometry"],
                 prefix_tokens=contexts, prefix_cache_keys=keys, prefix_cache_mode="shared")
    op = replace(op, metadata=(("indexed_attention_geometry", shape),))
    geom = a.Geometry.from_devices(prior.parse_case(a._c8_case(a.SKUS[0][1])).machine.dies)
    plan = a.build_packed(op, geom, scratch_policy="offchip", schedule=a.SCHEDULE_ARCH35)
    stats = plan.metadata["c8_schedule"]
    assert stats["external_fill_bytes"] == union * 656
    assert stats["private_rows"] == local
    from collections import Counter
    from mindie_sim.prediction.indexed_quant_attention import bind_sustained_dram
    query_output = len(contexts) * 64 * (576 + 512) * 2
    assert plan.work.resources["hbm"] == query_output + (union + local) * 656
    _, context = b._context(b.DT)
    context = bind_sustained_dram(replace(context, op=plan.work.apply(op)))
    issued = Counter()
    for stage in b._stages(plan.program(context)):
        issued.update(stage.work)
    assert dict(issued) == pytest.approx(context.bind(plan.work))


def test_arch35_distinct_operands_and_cold_fill():
    views = (RequestView("a", 1, 1, False, "decode"), RequestView("b", 1, 3, False, "decode"))
    op = b._arch35_op_for(1, views, heads=32)
    plan = b.sp.Arch35QuantIndexerRecipe().build(op, None)
    assert op.flops == 49152
    assert plan.work.resources["mte2_aic"] == 2 * 32 * 128 + 6 * 128
    assert plan.work.resources["mte2_aiv"] == 2 * 32 * 8 + 6 * 4
    assert op.hbm_bytes == 8704 + 792 + 2 * 8192
    b._conserved(plan, op, b.DT)


def test_kimi_padded_producer_writes_the_entire_pack_input():
    case = prior.parse_case(prior.compose_case(
        "kimi-k3.1", hardware="a5", dp=1, tp=8, dcp=8, ep=8, num_devices=8,
        layers=4, quant="int8", prompt_tokens=8, output_tokens=2))
    model = replace(case.model, quant=replace(case.model.quant, kv_dtype="int8"))
    layer = model.layer_plans[3]
    views = (RequestView("r", 1, 127, False, "decode"),)
    builder = Builder("p", layer, 0, views, model, "decode")
    member = replace(a._c8_op(((1, 127),), 128, 96, 0), recipe="paired_mla_attention_analytic",
                     mte3_bytes=98304, hbm_bytes=98304, metadata=())
    builder.add(member)
    latent_dcp.attention_batch(builder, [member.op_id], views, plan=layer,
                               heads=96, local_heads=12, cp=8, rows=8)
    ops = {op.op_id.split(".")[-1]: op for op in builder.ops}
    assert ops["latent_attention_batch"].mte3_bytes == ops["dcp_output_pack"].mte2_bytes == 792576
    assert ops["dcp_output_pack"].mte3_bytes == ops["dcp_lse_merge"].mte2_bytes == 792576
    assert ops["dcp_output_alltoall"].link_bytes == 693504
    assert ops["dcp_lse_merge"].mte3_bytes == 98304


def test_keypool_native_gates_pack_before_exchange_and_counts_indices_once():
    plan, model = b._case_model(pool=16)
    model = replace(model, quant=replace(model.quant, index_scale_dtype="fp32"))
    runtime = b._Runtime(o_proj_tp=8)
    views = tuple(RequestView(str(i), 1, 131072, False, "decode") for i in range(8192))
    assert native_shard_sizes(views[:8], 16, 8) == (8,)
    shallow = (replace(views[0], q_len=8192, context=0),)
    assert native_shard_sizes(shallow, 16, 8) == (8192,)
    sizes = native_shard_sizes(views, 16, 8)
    assert sizes == (1023, 1024, 1024, 1024, 1024, 1024, 1024, 1025)
    count = 0
    for rank, owned in enumerate(sizes):
        builder = Builder("l", plan, 0, views, model, "decode")
        index_pool_native.selection(builder, query="q", weights="w", cache="k", views=views,
                                    plan=plan, quant=builder.quant, padded_tokens=8192,
                                    runtime=runtime, row_shard_rank=rank)
        ops = {o.op_id.split(".")[-1]: o for o in builder.ops}
        pack = ops["index_topk"]
        assert pack.deps == (ops["cann_pool_key_indexer"].op_id,)
        assert ops["index_exchange_stage"].deps == (pack.op_id,)
        wire = ops["index_kpool_row_shard_gather"]
        assert wire.deps == (ops["index_exchange_stage"].op_id,)
        assert ops["index_exchange_restore"].deps == (wire.op_id,)
        assert wire.link_bytes == 7 * 1025 * 2063 * 4
        assert pack.mte3_bytes == owned * 2063 * 4
        assert pack.mte2_bytes == owned * (2048 * 4 + 2 * 8)
        count += pack.mte2_bytes - owned * 2 * 8
    assert count == 67108864



def test_rank_table_membership_can_cross_pcp_inside_each_dp_replica():
    raw = prior.compose_case("qwen3-30b-a3b", hardware="a5", tp=1, dp=4, ep=2,
                             pcp=2, num_devices=8, layers=1, enable_dbo=True,
                             prompt_tokens=8, output_tokens=2, max_num_seqs=2)
    table = [[3, 2], [1, 0], [7, 6], [5, 4]]
    raw["runtime"]["parallel"]["rank_table"] = table
    case = prior.parse_case(raw)
    from mindie_sim.prediction.dispatch import cohorts
    from mindie_sim.prediction.topology import Topology
    assert cohorts(case.runtime.parallel, case.machine) == [(0,), (1,), (2,), (3,)]
    assert Topology(case.machine, case.runtime.parallel).groups("ep", 3, 0) == ((7, 6),)
    g = prior.target_graphs(case, phase="prefill", tokens=8, context=0, sequences=2,
                            selected_replica=3, physical_rank=7, scope="body")
    assert [op.local_shape[0] for op in g[3] if op.recipe == "grouped_gate_up"] == [32, 32]


def test_real_sp_bucket_preserves_every_rank_and_valid_row():
    import json
    from mindie_sim.prediction.dbo_ledger import plan
    from mindie_sim.prediction.frontend import Frontend
    raw = json.loads((prior.ROOT / "examples/cases/kimi-k3-a5-sp-right-pad-conditional.json").read_text())
    case = prior.parse_case(raw)
    counts = (10, 12, 14, 16)
    peers = {d: tuple(RequestView(f"{d}-{i}", 1, 128, False, "decode")
                      for i in range(n)) for d, n in enumerate(counts)}
    chunks = plan(Frontend(), case.model, case.runtime, peers[0], 16,
                  peers, 0, case.machine, 0, 2)
    assert [chunk.capacity for chunk in chunks] == [8, 8]
    for part in chunks:
        assert part.allocated == (1,) * 32 and part.physical == 32
        assert part.valid == tuple(int(i < n // 2) for n in counts for i in range(8))
    assert sum(map(lambda x: sum(x.valid), chunks)) == sum(counts)


def test_forced_mask_marginals_match_inverse_and_expert_banks():
    from mindie_sim.prediction.source_force_eplb import ledger
    valid, allocated = (2, 2, 2, 1), (2, 2, 3, 3)
    result = ledger(32, 4, 8, 2, 7, 2, valid, allocated, 1)
    assert tuple(map(sum, result.peer_assignments)) == (16, 16, 16, 8)
    assert tuple(map(sum, zip(*result.peer_assignments))) == tuple(map(sum, result.rank_expert_rows))
    assert sum(result.expert_rows) == 56


def test_native_small_batch_has_no_gather_and_override_is_explicit():
    plan, model = b._case_model(pool=16)
    model = replace(model, quant=replace(model.quant, index_scale_dtype="fp32"))
    views = (RequestView("r", 8, 131072, False, "decode"),)
    for opts, expected in (({}, 0), ({"row_shard_tp": 8}, 1)):
        builder = Builder("l", plan, 0, views, model, "decode")
        index_pool_native.selection(builder, query="q", weights="w", cache="k", views=views,
                                    plan=plan, quant=builder.quant, padded_tokens=8,
                                    runtime=b._Runtime(o_proj_tp=8), **opts)
        wires = [op for op in builder.ops if op.recipe == index_pool_native.ROW_SHARD_GATHER]
        assert len(wires) == expected
        assert sum(op.link_bytes for op in wires) == expected * 57344



@pytest.mark.parametrize("groups", [((0, 2), (1, 3)), ((3, 0), (2, 1)), ((1, 0), (3, 2))])
def test_sp_source_is_the_tp_coordinate_even_if_ep_groups_cross_dp(groups):
    from mindie_sim.prediction.dbo_ledger import plan
    p = SimpleNamespace(tp=2, dp=2, pcp=1, pp=1, ep=2, rank_table=groups,
                        pcp_lane_policy="parked")
    runtime = SimpleNamespace(parallel=p, routing={}, kernels={"sequence_parallel": {"padding": "right_to_tp"}})
    frontend = SimpleNamespace(graph=SimpleNamespace(select=lambda views, rt: (None, "none")))
    peers = {d: tuple(RequestView(f"{d}-{i}", 1, 128, False, "decode")
                      for i in range(n)) for d, n in enumerate((4, 8))}
    for group in groups:
        rank = group[0]
        d = rank // 2
        chunks = plan(frontend, None, runtime, peers[d], len(peers[d]), peers, d, None, rank, 2)
        expected = tuple((1, 2)[r // 2] for r in group)
        assert len(chunks) == 2
        assert all(x.valid == x.allocated == expected and x.physical == sum(expected) for x in chunks)



def test_glm_multi_request_chunk_kda_folds_in_kernel_qk_l2norm():
    raw = prior.compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
        layers=4, prompt_tokens=64, output_tokens=2, max_num_seqs=2)
    raw["runtime"]["lowering_passes"] = ["kda_in_kernel_l2norm"]
    case = prior.parse_case(raw)
    ops = prior.target_graphs(case, phase="prefill", tokens=64, context=0,
                              sequences=2, scope="body")[0]
    assert not [op for op in ops if op.recipe == "qk_l2norm"]
    assert [op for op in ops if dict(op.metadata).get("in_kernel_qk_l2norm")]


def test_legacy_dsl_mqsmla_contract_rejects_cann_only_row_widths():
    import test_v0919_deep_audit_track_a as track_a
    from mindie_sim.observed_contracts.attention_contracts import (
        observed_attention_contract,
    )

    query = ((4, 64, 512), "bf16", "ND")
    empty = ((), None, None)
    for row_width in (608, 584):
        inputs = [query, empty, ((16, 1, 1, row_width), "uint8", "ND"),
                  ((1, 16), "int32", "ND"), empty, empty, empty]
        outputs = [query, empty]
        assert observed_attention_contract(
            track_a._observed_row("MqsmlaKernel___call__", inputs, outputs)) is None
        assert observed_attention_contract(
            track_a._observed_row("aclnnMixedQuantSparseFlashMla", inputs, outputs)) is not None


def test_compose_case_pp_pcp_ep_rank_table_covers_every_stage():
    from mindie_sim.prediction.topology import Topology

    raw = prior.compose_case(
        "deepseek-v4-pro", "a3", tp=4, dp=2, pp=2, pcp=2, ep=16, layers=4,
        prompt_tokens=128, output_tokens=8, sp=True, va_recommended=True,
        quant="w4a8")
    case = prior.parse_case(raw)
    assert case.runtime.parallel.ep_domain == 16
    assert case.runtime.parallel.rank_table == (
        tuple(range(0, 8)) + tuple(range(16, 24)),
        tuple(range(8, 16)) + tuple(range(24, 32)),
    )
    topo = Topology(case.machine, case.runtime.parallel)
    for stage, expected in enumerate(case.runtime.parallel.rank_table):
        for replica in range(case.runtime.parallel.dp):
            assert topo.groups("ep", replica, stage) == (expected,)


def test_glm_mixed_prefill_decode_kda_eliminates_standalone_qk_l2norm():
    from mindie_sim.prediction.attention import AttentionPolicy
    from mindie_sim.prediction.implementation_selection import resolve
    from mindie_sim.prediction.recurrent import linear
    from mindie_sim.prediction.transforms import lower_implementations

    raw = prior.compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
        layers=4, prompt_tokens=64, output_tokens=2, max_num_seqs=2)
    raw["runtime"]["lowering_passes"] = ["kda_in_kernel_l2norm"]
    case = prior.parse_case(raw)
    plan = next(lp for lp in case.model.layer_plans if lp.attention == "kda")
    views = (RequestView("p", 64, 0, True, "prefill"),
             RequestView("d", 1, 128, True, "decode"))
    resolved_rt, _ = resolve(case.runtime, views, padded_tokens=65, replay_scope="none")
    res = linear(
        prefix="s0.l0", plan=plan, views=views, stage=0, deps="in",
        model=case.model, parallel=resolved_rt.parallel,
        quant=case.model.quant, mode="prefill",
        policy=AttentionPolicy(recurrent_chunk=64), runtime=resolved_rt,
        padded_tokens=65)
    lowered = lower_implementations(res.ops, resolved_rt, case.model.activation_dtype)
    assert not [op for op in lowered if op.recipe == "qk_l2norm"]
    folded = [op for op in lowered if dict(op.metadata).get("in_kernel_qk_l2norm")]
    assert len(folded) == 2


def test_dspark_draft_scoped_prolog_and_contiguous_draft_cache_predicate():
    import test_v0919_deep_audit_track_b as track_b

    raw = track_b._dspark_case()
    target_binding = raw["runtime"]["kernels"]["lowerings"]["mla_prolog"]
    raw["runtime"]["kernels"]["lowerings"]["mla_prolog"] = {
        "cases": [
            {"domain": {"model_scopes": ["target"]}, "binding": target_binding},
            {"domain": {"model_scopes": ["draft"]}, "binding": "vllm_ascend_mla_prolog_v3"},
        ],
        "otherwise": "analytical",
    }
    graph = prior.target_graphs(
        prior.parse_case(raw), phase="decode", tokens=6, context=4,
        sequences=1, scope="model")[0]
    body = [op for op in graph if op.phase == "draft"
            and dict(op.metadata).get("fusion_role") == "mla_prolog"]
    context = [op for op in graph if op.recipe == "draft_context_mla_prolog_v3"]
    assert len(body) == 5 and len(context) == 5

    contiguous = track_b._dspark_case()
    contiguous["model"].setdefault("metadata", {})["draft_cache_representations"] = {
        "mla_nope": {"dtype": "bf16", "contiguous": True},
        "mla_rope": {"dtype": "bf16", "contiguous": True},
    }
    contiguous["runtime"]["kernels"]["lowerings"]["mla_prolog"] = {
        "cases": [
            {"domain": {"model_scopes": ["draft"]}, "binding": "vllm_ascend_mla_prolog_v3"},
        ],
        "otherwise": target_binding,
    }
    graph_contig = prior.target_graphs(
        prior.parse_case(contiguous), phase="decode", tokens=6, context=4,
        sequences=1, scope="model")[0]
    body_contig = [op for op in graph_contig if op.phase == "draft"
                   and dict(op.metadata).get("fusion_role") == "mla_prolog"]
    assert len(body_contig) == 5
    assert not [op for op in graph_contig if op.recipe == "draft_context_mla_prolog_v3"]
    assert len([op for op in graph_contig if op.recipe == "draft_context_qkv_a"]) == 5


def test_legacy_dsl_mqsmla_lowering_rejects_cann_quant_modes():
    from dataclasses import replace
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.schema.common import ModelingError

    for scale_dtype in ("bf16", "fp8_e8m0"):
        raw = prior.compose_case(
            "deepseek-v4.1-flash", hardware="a5", tp=4, dp=2, ep=8,
            num_devices=8, layers=4, quant="w8a8", prompt_tokens=16,
            output_tokens=2)
        packed = {"dtype": "fp8_e4m3", "group_size": 64, "scale_dtype": scale_dtype}
        raw["model"].setdefault("metadata", {})["cache_representations"] = {
            "window": dict(packed),
            "compressed_kv": dict(packed),
        }
        parsed = prior.parse_case(raw)
        cann_graph = prior.target_graphs(
            parsed, phase="decode", tokens=1, context=128, sequences=1, scope="body")[0]
        cann_op = next(op for op in cann_graph if op.recipe == "aclnn_mixed_quant_sparse_flash_mla")
        meta = dict(cann_op.metadata)
        geom = dict(meta["mqsmla_geometry"])
        geom.pop("implementation", None)
        meta["mqsmla_geometry"] = geom
        meta.pop("native_kind", None)
        crafted = replace(cann_op, recipe="cannbot_mixed_quant_sparse_flash_mla",
                          metadata=tuple(meta.items()))
        with pytest.raises(ModelingError):
            operator_costs(parsed, (crafted,), input_ready_ms=0)
        raw["runtime"].setdefault("metadata", {}).pop("automatic_lowerings", None)
        raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
            "mqsmla_attention"] = "cannbot_mixed_quant_sparse_flash_mla"
        with pytest.raises(ModelingError):
            prior.target_graphs(prior.parse_case(raw), phase="decode",
                                tokens=1, context=128, sequences=1,
                                scope="body", apply_body_defaults=False)


