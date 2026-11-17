"""CPU-only regressions for draft, attention traffic and recurrent ownership."""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import pytest

from mindie_sim import target_graphs
from mindie_sim.catalog.cache import stage_cache_components
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.draft import optional_draft_spec
from mindie_sim.catalog.weights import stage_weight_terms
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.body_defaults import (
    effective_body_declaration, service_program_defaults,
)
from mindie_sim.prediction.family_mqsmla import FORMATS, RECIPE, lower_attention, validate
from mindie_sim.prediction.generic_latent_attention import (
    index_key_window, unique_index_key_rows,
)
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction import mqsmla
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.physical_key import physical_key
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView


@lru_cache(maxsize=1)
def mq_case():
    from test_mqsmla_source import case, attention
    c = case()
    return c, attention(target_graphs(c, phase="decode", tokens=1, context=2048)[0])


@lru_cache(maxsize=1)
def kimi_case():
    from test_kimi_k31_alignment import case
    return case()


def test_dspark_aliases_reach_weight_and_cache_ledgers():
    for name in ("deepseek-v4-pro-0813", "deepseek-v4-10t", "deepseek-v4.1-flash"):
        model = load_model(name)
        draft = optional_draft_spec(model, "dspark")
        assert draft is model.auxiliary.mtp and draft.kind == "dspark"
        parallel = ParallelSpec(tp=8)
        main = stage_weight_terms(model, parallel)
        full = stage_weight_terms(model, parallel, include_draft=True, draft_kind="dspark")
        assert sum(term.logical_parameters for stage in full for term in stage) > sum(
            term.logical_parameters for stage in main for term in stage)
        cache = stage_cache_components(model, parallel, 1024, draft_kind="dspark")
        assert cache[-1]["draft_kv"] > 0


def test_model_scope_keeps_service_matrix_defaults_and_explicit_choices():
    from test_prediction_system import case
    raw = case()
    for explicit in ({}, {"programs": {"matmul": {"kind": "tiled_matmul"}}},
                     {"recipes": {"router": {"kind": "tiled_matmul"}},
                      "families": {"grouped_gemm": {"kind": "steady_matmul"}}}):
        raw["runtime"]["kernels"] = deepcopy(explicit)
        parsed = parse_case(raw)
        original = deepcopy(parsed.runtime.kernels)
        effective, _ = effective_body_declaration(parsed, "decode", tokens=6, scope="model")
        service = service_program_defaults(parsed)
        assert effective.runtime.kernels["programs"] == service.runtime.kernels["programs"]
        for section, entries in explicit.items():
            for key, value in entries.items():
                assert effective.runtime.kernels[section][key] == value
        assert parsed.runtime.kernels == original
        disabled, _ = effective_body_declaration(parsed, "decode", scope="model", enabled=False)
        assert disabled is parsed


def test_hca_swa_and_local_head_groups_conserve_source_work():
    c, original = mq_case()
    model = load_model("deepseek-v4-pro-0813")
    model = replace(model, metadata=tuple({
        **dict(model.metadata), "cache_representations": FORMATS}.items()))
    hca = next(p for p in model.layer_plans if p.compress_ratio == 128)
    view = RequestView("r", 1, 131584, True, "decode")
    geometry = Geometry.from_devices(c.machine.dies)
    for tp in (1, 2, 4, 8):
        parallel = ParallelSpec(tp=tp)
        validate(views=(view,), plan=hca, parallel=parallel, model=model,
                 padded_tokens=1, topk_present=False, cores=32)
        ops = [original]
        lower_attention(ops, [original.op_id], selection=SimpleNamespace(name=RECIPE),
                        views=(view,), plan=hca, topk_present=False, prefix="s0.l0",
                        parallel=parallel)
        shape = dict(ops[0].metadata)["mqsmla_geometry"]
        assert shape["heads"] == 128 // tp and shape["ratio"] == 128
        assert tuple(mqsmla.query_rows(shape)) == ((128, 1028),)
        expected = Work()
        for start in range(0, shape["heads"], 64):
            heads = min(64, shape["heads"] - start)
            stream = mqsmla.OwnerTiles.create(((128, 1028),))
            for index in range(stream.count):
                for _, work in mqsmla.tile_phases(*stream.key(index), heads=heads):
                    expected += work
        assert mqsmla.build(ops[0], geometry).work == expected
    from mindie_sim.prediction.family_mqsmla import query_shape
    swa = next(p for p in c.model.layer_plans if not p.owns_long_kv and not p.owns_topk)
    assert query_shape((view,), swa)["ratio"] == 0
    for heads in (8, 16, 32, 48, 64, 80, 96, 128):
        assert mqsmla.supported_heads(heads)
    assert not mqsmla.supported_heads(4)


def test_stage_memoization_keeps_engine_duration_bit_exact():
    c, op = mq_case()
    cached = operator_costs(c, (op,))["operators"][0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mqsmla, "lru_cache", lambda **_: lambda function: function)
        uncached = operator_costs(c, (op,))["operators"][0]
    assert cached["duration_ms"] == uncached["duration_ms"]
    assert cached["serviced"] == uncached["serviced"]
    assert cached["live_tasks"] == uncached["live_tasks"] == 0


def test_fused_draft_identity_preserves_offsets_and_namespaces():
    _, base = mq_case()
    def op(stage, layer, draft="draft0.", offset=-1):
        origins = ({"stage": stage, "layer": layer + offset,
                    "prefix": f"{draft}s{stage}.l{layer + offset}",
                    "op_id": f"{draft}s{stage}.l{layer + offset}.norm"},)
        return replace(base, stage=stage, layer=layer,
                       metadata=(("fused_origin_scopes", origins),))
    assert physical_key(op(1, 2)) == physical_key(op(3, 9))
    assert physical_key(op(1, 2)) != physical_key(op(1, 2, offset=0))
    assert physical_key(op(1, 2)) != physical_key(op(1, 2, draft="draft1."))
    assert physical_key(op(1, 2)) != physical_key(op(1, 2, draft=""))
    assert physical_key(op(1, 2, draft="draft_prefill.")) == physical_key(
        op(3, 9, draft="draft_prefill."))


def test_compressed_cp_index_reads_are_per_query_tile():
    for compression in (1, 2, 4, 128):
        for cp in (1, 8):
            for query in (1, 6, 128, 129):
                for noncausal in (False, True):
                    ends = range(128, query, 128)
                    horizons = [4096 + (query if noncausal else end)
                                for end in (*ends, query)]
                    owned = []
                    for horizon in horizons:
                        counts = [0] * cp
                        for row in range(horizon // compression):
                            counts[(row // max(1, 128 // compression)) % cp] += 1
                        owned.append(max(counts))
                    assert index_key_window(
                        ((query, 4096),), cp=cp, compression=compression,
                        noncausal=noncausal, kv_len=4096 + query) == (max(owned), sum(owned))
    views = tuple(RequestView(str(i), 6, 256, True, "decode",
                              prefix_tokens=128, prefix_cache_key="shared") for i in range(2))
    assert unique_index_key_rows(views, compression=4) == (98, 130)
    assert unique_index_key_rows(views, compression=4, share_prefix=False) == (130, 130)
    different = (views[0], replace(views[1], prefix_cache_key="other"))
    assert unique_index_key_rows(different, compression=4) == (130, 130)



def test_generic_indexer_precharge_is_not_deduplicated_twice():
    from mindie_sim.catalog.cache import index_row_bytes
    from mindie_sim.prediction.generic_latent_attention import semantic_index_selection
    from mindie_sim.prediction.lowering import Builder
    c, _ = mq_case()
    model = load_model("deepseek-v4-pro-0813")
    plan = next(p for p in model.layer_plans if p.indexer)
    views = tuple(RequestView(str(i), 6, 256, True, "decode",
                              prefix_tokens=128, prefix_cache_key="same") for i in range(2))
    parallel = ParallelSpec()
    row = index_row_bytes(model, plan.index_dim)
    score_rows = 2 * sum((256 + i) // 4 for i in range(1, 7))
    for mode, key_rows in (("batch_shared", 98), ("per_sequence", 130)):
        runtime = replace(c.runtime, parallel=parallel, kernels={},
                          prefix_cache_mode="shared", prefix_read_mode=mode, kv_block_size=128)
        b = Builder("s0.l1", plan, 0, views, model, "decode")
        semantic_index_selection(
            b, deps=("input",), views=views, plan=plan, model=model, quant=model.quant,
            parallel=parallel, shard=1, prefix="s0.l1", padded_tokens=12, runtime=runtime)
        scores = next(op for op in b.ops if op.family == "indexer")
        assert scores.flops == 2 * score_rows * plan.index_heads * plan.index_dim
        assert scores.mte2_bytes == 130 * row
        assert scores.hbm_bytes == key_rows * row + score_rows * 4


def test_draft_cache_uses_bf16_rows_and_context_block_ownership():
    c = kimi_case()
    draft = optional_draft_spec(c.model, "dspark")
    assert cache_format(c.model, "mla_kv").row_bytes(576) == 656
    for tokens in (0, 1, 1536 * 8, 1536 * 8 + 1, 131588):
        for padded in (False, True):
            parts = stage_cache_components(
                c.model, c.runtime.parallel, tokens, draft_kind="dspark",
                capacity_padded=padded, interleave=1536, runtime=c.runtime)
            rows = ((tokens + 1536 * 8 - 1) // (1536 * 8) * 1536
                    if padded else tokens / 8)
            assert parts[-1]["draft_kv"] == rows * 1152 * draft.layers


def draft_regions(c, *, read_mode="batch_shared"):
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.speculation import draft_program
    runtime = replace(c.runtime, prefix_cache_mode="shared", prefix_read_mode=read_mode,
                      metadata=tuple({**dict(c.runtime.metadata), "draft_mode": "dspark"}.items()))
    views = tuple(RequestView(str(i), 6, 131584, True, "decode", prefix_tokens=124518,
                              prefix_cache_key="same", draft_proposal_tokens=5) for i in range(4))
    return draft_program(Frontend(resource_fusion=False), c.model, runtime,
                         views, "ready", machine=c.machine)[0]


def test_gathered_draft_prefix_reuse_preserves_compute_and_outputs():
    c = kimi_case()
    batches = []
    for mode in ("per_sequence", "batch_shared"):
        ops = [op for region in draft_regions(c, read_mode=mode) for op in region.ops]
        batches.append([op for op in ops if op.op_id.endswith(".latent_attention_batch")])
    private, shared = batches
    assert len(private) == len(shared) == 5
    for old, new in zip(private, shared):
        assert new.hbm_bytes < old.hbm_bytes
        assert new.flops == old.flops and new.mte3_bytes == old.mte3_bytes
        assert dict(new.metadata)["prefix_read_reuse_applied"]
        assert new.recipe == "batch_prefix_attention"
        assert "cp_per_rank_pairs" in dict(new.metadata)


def test_real_dspark_graph_exports_decode_and_draft_subgraphs():
    from mindie_sim import compose_case
    from mindie_sim.structure import export_op_structure
    c = parse_case(compose_case("deepseek-v4-pro-0813", "a5", tp=8, ep=8,
                               max_model_len=140000))
    ops = tuple(op for region in draft_regions(c) for op in region.ops)
    helper = next(op for op in ops if op.recipe == "hc_pre_mix")
    ops = tuple(replace(op, phase="decode") if op is helper else op for op in ops)
    assert {op.phase for op in ops} == {"decode", "draft"}
    report = export_op_structure(
        ops, model={}, execution={"phase": "decode", "scope": "model"},
        kernels=c.runtime.kernels)
    assert report["nodes"] and report["subgraphs"]
    with pytest.raises(ValueError, match="structure_phase_mismatch"):
        export_op_structure(ops, model={}, execution={"phase": "prefill", "scope": "model"})


def test_recover_ssm_keeps_native_kda_and_accounts_replay_separately():
    from mindie_sim.feature_mechanisms import _apply_recover_ssm
    from mindie_sim.prediction.capacity import Memory
    c = kimi_case()
    rt = {"metadata": {"draft_proposal_tokens": 3}}
    _apply_recover_ssm(rt, SimpleNamespace(caller_explicit=set(), model=c.model))
    assert rt["recover_ssm"]
    recovered = replace(c, runtime=replace(c.runtime, recover_ssm=True))
    before, after = Memory(c), Memory(recovered)
    records = 69 * 12 * (128 * 4 + 2 * 128 * 2) * 4
    page = before.pages.groups[0].bytes_per_group_page
    assert before.cache_bytes(131584)[0] - after.cache_bytes(131584)[0] == 3 * 3 * page - records
    assert all("replay_records" not in names for names in after.pages.covered.values())
    ops = target_graphs(recovered, phase="decode", tokens=4, context=4096,
                        scope="body", resource_fusion=False)[0]
    native = [op for op in ops if op.recipe == "vllm_ascend_recurrent_kda_arch35"]
    assert len(native) == 69
    assert all(dict(op.metadata)["recover_ssm"] for op in native)
