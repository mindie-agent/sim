"""Causal pool conservation and real GLM q6/131201 tensor geometry controls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.cache import index_selection_cost, layer_cache_components
from mindie_sim.catalog.index_pool import pool_selection
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction import index_pool
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.execution_state import bind_op_accesses
from mindie_sim.prediction.index_pool import _tail_state_accesses
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.offload import (OffloadPolicy, _accesses_from_ops,
                                           _as_accesses, project_state_tiers)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView


ROOT = Path(__file__).resolve().parents[1]


def case():
    raw = json.loads((ROOT / 'examples/cases/high-concurrency-20261030/matrix-extension/'
                     'a3_glm53flash_tp16_aclgraph_c32.json').read_text())
    raw['model']['layers'] = 4
    raw['runtime']['kernels']['programs']['indexed_attention'] = {
        'kind': 'indexed_attention_packed_int8', 'scratch_policy': 'offchip'}
    return raw


def layer(raw, batch=1, query=6, context=131201):
    return [op for op in target_graphs(raw, scope='body', phase='decode', tokens=query,
                                     context=context, sequences=batch)[0] if op.layer == 3]


def _producer_builder(views, *, versions=None, owners=None, readers=None,
                      last_uses=None, raw=None):
    parsed = parse_case(raw if raw is not None else case())
    plan = parsed.model.layer_plans[3]
    builder = Builder("s0.l3", plan, 0, tuple(views), parsed.model, "decode")
    index_pool.producer(
        builder, source="hidden", views=tuple(views), plan=plan,
        model=parsed.model, quant=builder.quant,
        padded_tokens=sum(view.q_len for view in views),
        state_versions=versions, state_owners=owners,
        state_readers=readers, state_last_uses=last_uses)
    tail = next(op for op in builder.ops
                if op.op_id.endswith("index_pool_tail_store"))
    return builder, tail


def _producer(views, *, versions=None, owners=None, readers=None, last_uses=None):
    return _producer_builder(
        views, versions=versions, owners=owners, readers=readers,
        last_uses=last_uses)[1]


def _packed_raw():
    raw = case()
    raw["model"].setdefault("quant", {}).update(
        index_dtype="fp8_e4m3", index_scale_dtype="fp16")
    raw["model"].setdefault("metadata", {})["cache_representations"] = {
        "index_k": {"dtype": "fp8_e4m3", "group_size": 32,
                     "scale_dtype": "fp16"}}
    return raw


def _work(views, op):
    return SimpleNamespace(
        plan=IterationPlan(tuple(views), iteration=0), ops=(op,))


def _offload_case():
    raw = case()
    raw["runtime"].setdefault("metadata", {})["offload"] = {
        "kvc_ddr_ratio": 0.5, "kvc_ssd_ratio": 0.25}
    return parse_case(raw)


def test_pool_scan_and_expansion_match_independent_causal_sets():
    for context in (0, 1, 3, 7, 2046, 4096, 131201):
        for query in (0, 1, 3, 6, 9):
            result = pool_selection(query, context, 2048, 4, always_select_tail=True)
            visible = [n for n in range(context + 1, context + query + 1)]
            complete = [n // 4 for n in visible]
            selected = [min(n, 512) for n in complete]
            tails = [n % 4 for n in visible]
            assert result.scanned_pools == sum(complete)
            assert result.selected_pools == sum(selected)
            assert result.selected_tokens == sum(n * 4 + t for n, t in zip(selected, tails))
            assert result.causal_tail_tokens == sum(tails)
            assert result.completed_pools == len([n for n in visible if n % 4 == 0])


def test_catalog_pool_cache_keeps_full_token_kv_and_real_gate_parameters():
    parsed = parse_case(case()); plan = parsed.model.layer_plans[3]
    assert (plan.index_kpool, plan.index_kpool_compress, plan.index_kpool_always_select_tail) == (4, True, True)
    cache = layer_cache_components(parsed.model, plan, parsed.runtime.parallel, 131201,
                                   capacity_padded=False)
    assert cache['index_k'] == (131201 // 4) * 128 * 2
    assert cache['index_pool_tail'] == 4 * 2 * 128 * 4
    assert cache['mla_kv'] > cache['index_k'] * 4
    selection = index_selection_cost(parsed.model, plan, 131201)
    assert selection.scored_rows == 131201 // 4
    terms = {term.name: term for term in layer_weight_terms(plan, parsed.model)}
    assert terms['indexer_pool_gate'].rows == 128
    assert terms['indexer_pool_gate'].cols == parsed.model.hidden_size
    assert terms['indexer_pool_ape'].rows == 4
    # Head weights keep the checkpoint's BF16 representation; only the retained
    # analytic K/gate joint projection prepares FP32 runtime copies.
    for name, width in (('indexer_wk', 128), ('indexer_pool_gate', 128)):
        original, prepared = terms[name], terms[name + '_prepared_fp32']
        assert term_bytes(original, parsed.model.quant, parsed.runtime.parallel) == width * 4096 * 2
        assert term_bytes(prepared, parsed.model.quant, parsed.runtime.parallel) == width * 4096 * 4
        assert term_bytes(prepared, parsed.model.quant, parsed.runtime.parallel, source=True) == 0
    assert 'indexer_weights_prepared_fp32' not in terms
    assert term_bytes(terms['indexer_weights'], parsed.model.quant,
                      parsed.runtime.parallel) == 32 * 4096 * 2


@pytest.mark.parametrize('raw_eager', [False, True])
@pytest.mark.parametrize('batch', [1, 4, 8])
def test_real_q6_buckets_scan_all_pools_and_keep_2051_capacity(batch, raw_eager):
    raw = case()
    raw['runtime'].setdefault('metadata', {})['raw_eager'] = raw_eager
    ops = {op.recipe: op for op in layer(raw, batch)}
    scanned = batch * sum((131201 + i) // 4 for i in range(1, 7))
    scores = ops['index_pool_scores']; meta = dict(scores.metadata)
    assert meta['scanned_pools'] == scanned
    # FP32 contraction work is unchanged. Only raw eager execution sends
    # the score plane and intermediate pool indices through HBM.
    assert scores.flops == 0
    assert scores.matrix_flops == ()
    assert scores.vector_ops == 2 * scanned * 128
    assert meta['score_mode'] == 'linear_weighted_query'
    assert meta['score_pool_row_bytes'] == 256
    unique = batch * ((131201 + 6) // 4)
    assert meta['score_unique_pool_rows'] == unique
    assert meta['score_logical_pool_read_bytes'] == scanned * 256
    assert meta['score_l2_hit_bytes'] == (scanned - unique) * 256
    assert meta['score_materialization'] == (
        'separate_fp32_score_then_topk' if raw_eager else 'ub_online_topk')
    weighted_read = scores.mte2_bytes - scanned * 256
    assert weighted_read > 0
    assert scores.mte2_bytes == scanned * 256 + weighted_read
    score_bytes = 4 * scanned if raw_eager else 0
    assert scores.hbm_bytes == unique * 256 + weighted_read + score_bytes
    assert scores.mte3_bytes == score_bytes
    topk = ops['index_pool_topk']
    assert topk.mte2_bytes == score_bytes
    assert topk.hbm_bytes == score_bytes + topk.mte3_bytes
    assert topk.mte3_bytes == (topk.local_shape[0] * 512 * 4 if raw_eager else 0)
    packed = ops['index_topk']
    assert packed.mte2_bytes == topk.mte3_bytes
    assert packed.mte3_bytes == packed.local_shape[0] * 2051 * 4
    assert ops['index_pool_topk'].local_shape == (batch * 6, 512)
    assert ops['index_topk'].local_shape == (batch * 6, 2051)
    assert dict(ops['index_topk'].metadata)['selected_token_rows'] == batch * (6 * 2048 + 11)
    attention = next(op for op in ops.values() if op.family == 'attention')
    attn = dict(attention.metadata)
    assert attn['indexed_gather_rows'] == batch * (6 * 2048 + 11)
    assert attn['logical_pairs'] == attn['indexed_gather_rows']
    assert ops['index_pool_tail_store'].deps[0].endswith('.index_pool_compress')


def test_4k_to_131201_changes_scan_work_but_not_the_2048_token_history_budget():
    values = []
    for context in (4096, 131201):
        ops = {op.recipe: op for op in layer(case(), context=context)}
        values.append(ops)
    lo, hi = values
    assert hi['index_pool_scores'].vector_ops / lo['index_pool_scores'].vector_ops == pytest.approx(
        sum((131201+i)//4 for i in range(1,7)) / sum((4096+i)//4 for i in range(1,7)))
    assert hi['index_topk'].local_shape == lo['index_topk'].local_shape == (6, 2051)
    assert hi['index_pool_scores'].vector_ops > 31 * lo['index_pool_scores'].vector_ops


def test_bf16_head_weights_keep_the_checkpoint_projection_boundary():
    ops = {op.recipe: op for op in layer(case())}
    # ``indexer.weights`` stays BF16: the analytic fallback no longer fabricates
    # an FP32 head-weight copy, so the joint K/weights projection splits.
    assert 'index_k_weights' not in ops
    key, gate, weights = ops['index_k'], ops['index_pool_gate'], ops['index_weights']
    assert key.local_shape == (6, 128, 4096) and key.dtype == 'fp32'
    assert gate.local_shape == (6, 128, 4096) and gate.dtype == 'fp32'
    assert weights.local_shape == (6, 32, 4096) and weights.dtype == 'bf16'
    assert key.deps == gate.deps == (ops['index_pool_hidden_fp32'].op_id,)
    # Only the FP32 K/gate operands consume the shared hidden cast.
    assert weights.deps[0] != key.deps[0]
    assert dict(key.metadata)['packed_weight_bytes'] == 128 * 4096 * 4
    assert ops['index_norm'].deps == (key.op_id,)
    assert ops['index_norm'].mte2_bytes == (6 + 2) * 128 * 4
    assert 'index_norm_input_fp32' not in ops
    assert 'index_weights_input_cast' not in ops and 'index_weights_cast' not in ops
    # The native indexer consumes the BF16 checkpoint head weights directly;
    # no FP16/FP32 scale producer remains on this boundary.
    assert 'index_weights_scale' not in ops
    assert ops['index_pool_weighted_query'].deps == (
        ops['index_pool_query_fp32'].op_id, ops['index_pool_weights_fp32'].op_id)


def test_declared_per_head_alternative_keeps_pool_scan_and_token_selection():
    raw = case(); reference = {op.recipe: op for op in layer(raw)}
    raw['runtime']['kernels'].setdefault('lowerings', {})['pool_key_indexer'] = {
        'implementation': 'causal_pool_indexer_analytic',
        'options': {'score_mode': 'per_head_relu'}}
    other = {op.recipe: op for op in layer(raw)}
    scanned = dict(reference['index_pool_scores'].metadata)['scanned_pools']
    # A declared per-head (non-MMA) score path keeps the pool scan on Vector and
    # does not fabricate Cube work; the explicit per_head_mma implementation is
    # the only path allowed to charge the per-head contraction.
    assert other['index_pool_scores'].flops == 0
    assert other['index_pool_scores'].matrix_flops == ()
    assert other['index_pool_scores'].vector_ops == scanned * 32 * 2
    assert reference['index_pool_weighted_query'].vector_ops == 6 * 32 * 128 * 2
    assert 'index_pool_weighted_query' not in other
    assert other['index_topk'].local_shape == reference['index_topk'].local_shape


def test_graph_padding_preserves_projection_capacity_without_fake_pool_writes():
    ops = {op.recipe: op for op in layer(case(), batch=3)}
    # The retained graph buckets pad 18 valid target queries to 24 rows.
    for name in ('index_pool_hidden_fp32', 'index_k', 'index_pool_gate',
                 'index_norm', 'index_weights',
                 'index_pool_query_fp32', 'index_pool_weighted_query',
                 'index_topk'):
        assert ops[name].local_shape[0] == 24
    metadata = dict(ops['index_pool_scores'].metadata)
    assert metadata['valid_query_rows'] == 18
    assert metadata['padded_query_rows'] == 24
    assert metadata['scanned_pools'] == 3 * sum((131201 + i) // 4 for i in range(1, 7))
    assert dict(ops['index_pool_compress'].metadata)['completed_pools'] == 3
    assert dict(ops['index_pool_tail_store'].metadata)['retained_rows'] == 12
    assert dict(ops['index_topk'].metadata)['selected_token_rows'] == 3 * (6 * 2048 + 11)


def test_compact_and_detailed_sparse_pipelines_conserve_pool_tail_work_and_readiness():
    reports = []
    for compact in (False, True):
        raw = case(); raw['runtime']['kernels']['programs']['indexed_attention']['compact'] = compact
        ops = [op for op in layer(raw, 2, 3, 2047) if op.family == 'attention']
        result = graph_costs(raw, ops, input_ready_ms=5, isolated=False,
                             external_inputs={dep: 7 for op in ops for dep in op.deps}, trace_limit=1000)
        assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
        assert all(row['start_ms'] >= 7 for row in result['events'] if row['demand'])
        reports.append(result)
    assert reports[0]['execution']['serviced'] == pytest.approx(reports[1]['execution']['serviced'])


def test_pool_tail_exposes_actual_state_access_volume_and_span():
    raw = case()
    ops = target_graphs(raw, scope='body', phase='decode', tokens=1,
                        context=4, sequences=1)[0]
    tail = next(op for op in ops if op.op_id.endswith('index_pool_tail_store'))
    assert dict(tail.metadata)['retained_rows'] == 1
    assert dict(tail.metadata)['index_width'] == 128
    assert dict(tail.metadata)['write_bytes'] == 1024
    accesses = _accesses_from_ops(type('Work', (), {'ops': (tail,), 'plan': None})())
    assert len(accesses) == 1
    access = accesses[0]
    assert access.bytes == 1024
    assert access.direction == "write"
    assert access.output == access.storage == "index_pool_tail"
    assert access.span == ((0, 1),)
    assert access.occurrence == tail.op_id


def test_pool_tail_exposes_per_request_state_accesses():
    raw = case()
    ops = target_graphs(raw, scope='body', phase='decode', tokens=1,
                        context=4, sequences=2)[0]
    tail = next(op for op in ops if op.op_id.endswith('index_pool_tail_store'))
    meta = dict(tail.metadata)
    assert meta['state_access_total_bytes'] == 2048
    assert len(meta['state_accesses']) == 2
    assert [item['bytes'] for item in meta['state_accesses']] == [1024, 1024]
    assert len({item['storage'] for item in meta['state_accesses']}) == 2
    assert all(item['span'] == ((0, 1),) for item in meta['state_accesses'])
    accesses = _as_accesses(meta['state_accesses'])
    assert len(accesses) == 2
    assert {access.bytes for access in accesses} == {1024}
    assert {access.request for access in accesses} == {'analysis0:0', 'analysis0:1'}
    assert all(access.direction == "write" and access.position == "tail"
               for access in accesses)
    assert {access.state_version for access in accesses} == {None}


def test_tail_ring_spans_context_131201_and_consecutive_versions():
    plan = SimpleNamespace(index_kpool=4)
    view = SimpleNamespace(request_id="a", q_len=6, context=131201,
                           query_ranges=(), state_version=7)
    geometry = SimpleNamespace(retained_tail_rows=4)
    accesses = _tail_state_accesses(
        (view,), (geometry,), plan, 128, "s0.l3.index_pool_tail_store")
    assert accesses[0]["bytes"] == 4096
    assert accesses[0]["span"] == ((0, 4),)
    assert accesses[0]["state_version"] == 7

    views = (SimpleNamespace(request_id="a", q_len=1, context=4,
                             query_ranges=(), state_version=7),
             SimpleNamespace(request_id="b", q_len=1, context=5,
                             query_ranges=(), state_version=8))
    geometry = (SimpleNamespace(retained_tail_rows=1),
                SimpleNamespace(retained_tail_rows=1))
    accesses = _tail_state_accesses(views, geometry, plan, 128, "tail",
                                    versions={"a": 7, "b": 8})
    assert [item["bytes"] for item in accesses] == [1024, 1024]
    assert [item["span"] for item in accesses] == [((0, 1),), ((1, 2),)]
    assert [item["state_version"] for item in accesses] == [7, 8]

    reversed_accesses = _tail_state_accesses(
        tuple(reversed(views)), tuple(reversed(geometry)), plan, 128, "tail")
    assert [item["request"] for item in reversed_accesses] == ["b", "a"]
    assert [item["span"] for item in reversed_accesses] == [((1, 2),), ((0, 1),)]

    uneven = (SimpleNamespace(request_id="a", q_len=1, context=4,
                              query_ranges=(), state_version=7),
              SimpleNamespace(request_id="b", q_len=2, context=4,
                              query_ranges=(), state_version=8))
    geometry = (SimpleNamespace(retained_tail_rows=1),
                SimpleNamespace(retained_tail_rows=2))
    accesses = _tail_state_accesses(uneven, geometry, plan, 128, "tail")
    assert [item["bytes"] for item in accesses] == [1024, 2048]
    assert [item["span"] for item in accesses] == [((0, 1),), ((0, 2),)]


def test_pool_tail_actual_geometry_touches_only_four_ring_slots():
    raw = case()
    ops = target_graphs(raw, scope='body', phase='decode', tokens=6,
                        context=131201, sequences=1)[0]
    tail = next(op for op in ops if op.op_id.endswith('index_pool_tail_store'))
    meta = dict(tail.metadata)
    assert meta['state_access_total_bytes'] == 4096
    assert meta['state_accesses'][0]['span'] == ((0, 4),)
    assert meta['state_accesses'][0]['bytes'] == 4096


def test_pool_tail_same_access_projects_tier_bytes_once():
    raw = case()
    ops = target_graphs(raw, scope='body', phase='decode', tokens=1,
                        context=4, sequences=2)[0]
    tail = next(op for op in ops if op.op_id.endswith('index_pool_tail_store'))
    accesses = _as_accesses(dict(tail.metadata)['state_accesses'])
    policy = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25)
    records = project_state_tiers(accesses, policy, expected_bytes=2048)
    assert len(records) == 2
    assert sum(row["ddr_bytes"] for row in records) == 1024
    assert sum(row["ssd_bytes"] for row in records) == 512
    assert sum(row["hbm_bytes"] for row in records) == 1536
    assert len({row["storage"] for row in records}) == 2


def test_disjoint_query_ranges_union_physical_write_slots():
    views = (RequestView("a", 2, 4, False, "decode",
                         query_ranges=((0, 1), (2, 3))),)
    tail = _producer(views, versions={"a": 7}, owners={"a": 3},
                     readers={"a": "s0.l3.attention.read"},
                     last_uses={"a": "s0.l3.attention.last_use"})
    meta = dict(tail.metadata)
    access = meta["state_accesses"][0]
    assert access["span"] == ((0, 1), (2, 3))
    assert access["bytes"] == 2048
    assert access["storage"] == "index_pool_tail:s0.l3:a"
    assert access["state_version"] == 7
    assert access["rank"] == 3
    assert access["grant"] == access["storage"]
    assert access["reader"] == "s0.l3.attention.read"
    assert access["last_use"] == "s0.l3.attention.last_use"
    assert access["occurrence"].endswith(":a:v7")
    assert meta["state_access_total_bytes"] == 2048
    bound = bind_op_accesses(_work(views, tail), tail)
    assert len(bound) == 1
    projected, = project_state_tiers(
        (bound[0].access,), OffloadPolicy(kvc_ddr_ratio=0.5,
                                          kvc_ssd_ratio=0.25))
    assert projected["span"] == ((0, 1), (2, 3))
    assert (projected["ddr_bytes"], projected["ssd_bytes"],
            projected["hbm_bytes"]) == (1024.0, 512.0, 1536.0)
    # The same physical slot union is order-independent.
    reversed_views = (RequestView("a", 2, 4, False, "decode",
                                  query_ranges=((2, 3), (0, 1))),)
    reversed_tail = _producer(reversed_views, versions={"a": 7})
    assert dict(reversed_tail.metadata)["state_accesses"][0]["span"] == \
        ((0, 1), (2, 3))


def test_same_request_consecutive_materializations_keep_storage_and_versions():
    views7 = (RequestView("a", 1, 4, False, "decode"),)
    views8 = (RequestView("a", 1, 5, False, "decode"),)
    tail7 = _producer(views7, versions={"a": 7})
    tail8 = _producer(views8, versions={"a": 8})
    first = dict(tail7.metadata)["state_accesses"][0]
    second = dict(tail8.metadata)["state_accesses"][0]
    assert first["storage"] == second["storage"] == "index_pool_tail:s0.l3:a"
    assert first["producer"] == second["producer"] == tail7.op_id
    assert first["span"] == ((0, 1),)
    assert second["span"] == ((1, 2),)
    assert first["state_version"] == 7 and second["state_version"] == 8
    assert first["occurrence"] != second["occurrence"]
    assert first["occurrence"].endswith(":a:v7")
    assert second["occurrence"].endswith(":a:v8")
    # Normal binding keeps both materializations on the same persistent
    # storage as separate occurrences with their actual versions.
    bound7 = bind_op_accesses(_work(views7, tail7), tail7)[0].access
    bound8 = bind_op_accesses(_work(views8, tail8), tail8)[0].access
    assert bound7.storage == bound8.storage
    assert (bound7.span, bound7.state_version) == (((0, 1),), 7)
    assert (bound8.span, bound8.state_version) == (((1, 2),), 8)
    assert bound7.key != bound8.key


def test_tail_ring_context3_wraps_to_disjoint_slot_union():
    views = (RequestView("a", 2, 3, False, "decode"),)
    tail = _producer(views)
    access = dict(tail.metadata)["state_accesses"][0]
    assert access["span"] == ((0, 1), (3, 4))
    assert access["bytes"] == 2048


def test_empty_tail_producer_has_zero_transfer():
    assert _tail_state_accesses((), (), SimpleNamespace(index_kpool=4), 128,
                                "s0.l3.index_pool_tail_store") == []
    no_views = _producer(())
    assert dict(no_views.metadata)["state_accesses"] == ()
    assert dict(no_views.metadata)["state_access_total_bytes"] == 0


def test_normal_execution_consumes_the_same_tail_access_once():
    views = (RequestView("a", 1, 4, False, "decode"),
             RequestView("b", 1, 4, False, "decode"))
    tail = _producer(views, versions={"a": 7, "b": 8}, owners={"a": 0, "b": 1})
    work = IterationWork(
        ops=(tail,), plan=IterationPlan(views, iteration=0), stage_count=1,
        valid_tokens=2, padded_tokens=2, graph_bucket=None, mode="decode",
        recipe="test")
    case_spec = _offload_case()
    execution = Execution(case_spec, CostModel(case_spec),
                          include_serving_io=False)
    external = {dep: execution.engine.submit("ext:" + dep)
                for dep in tail.deps}
    step = execution.add(work, 0, 0.0, external_inputs=external)
    execution.engine.advance(until=(step.completion,))
    serviced = execution.engine.serviced
    assert serviced["offload_ddr@r0:s0"] == pytest.approx(1024.0)
    assert serviced["offload_ssd@r0:s0"] == pytest.approx(512.0)
    # The fallback path shares the same issued-access registry; no second
    # writeback is charged for the same occurrence/storage/version/span.
    bound = bind_op_accesses(work, tail)
    assert len(bound) == 2
    projected = project_state_tiers(
        tuple(item.access for item in bound),
        OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25),
        expected_bytes=2048)
    assert sum(row["ddr_bytes"] for row in projected) == 1024
    assert sum(row["ssd_bytes"] for row in projected) == 512
    assert sum(row["hbm_bytes"] for row in projected) == 1536
    assert {row["storage"] for row in projected} == {
        "index_pool_tail:s0.l3:a", "index_pool_tail:s0.l3:b"}
    assert index_pool.state_accesses(tail) == tuple(dict(tail.metadata)["state_accesses"])


def test_packed_parent_exposes_real_k_scale_and_fp32_tail_presence():
    views = (RequestView("a", 1, 7, False, "decode"),)
    builder, tail = _producer_builder(views, raw=_packed_raw())
    store = next(op for op in builder.ops if op.op_id.endswith("index_store"))
    parent = index_pool.packed_parent(store)
    assert parent["parent_occurrence"] == store.op_id
    assert parent["value"]["producer"] == "s0.l3.index_key_quantize"
    assert parent["value"]["dtype"] == "fp8_e4m3"
    assert parent["value"]["packed_elements"] == 1
    assert parent["value"]["physical_row_bytes"] == 136
    assert parent["value"]["data_row_bytes"] == 128
    assert parent["value"]["presence"] is True
    assert parent["value"]["materialization"] == "packed_cache_row"
    assert parent["scale"]["producer"] == "s0.l3.index_key_quantize"
    assert parent["scale"]["output"] == "index_scales"
    assert parent["scale"]["dtype"] == "fp16"
    assert parent["scale"]["shape"] == (1, 4)
    assert parent["scale"]["count"] == 4
    assert parent["scale"]["row_bytes"] == 8
    assert parent["scale"]["row_offset"] == 128
    assert parent["scale"]["presence"] is True
    assert parent["scale"]["materialization"] == "native_scatter_scale_slot"
    assert parent["row_materialization"] == "packed_k_plus_scale_suffix"
    assert parent["tail"]["producer"] == tail.op_id
    assert parent["tail"]["dtype"] == "fp32"
    assert parent["tail"]["shape"] == (1, 2, 128)
    assert parent["presence"] == {"value": True, "scale": True, "tail": True}
    assert parent["native_scatter"]["parent"] == store.op_id
    assert parent["native_scatter"]["scale"] == "s0.l3.index_key_quantize"
    assert parent["alias"] == {"value": "s0.l3.index_key_quantize",
                               "tail": tail.op_id,
                               "scale": "s0.l3.index_key_quantize"}


def test_bf16_packed_parent_states_absent_scale_without_fabricating_one():
    views = (RequestView("a", 1, 4, False, "decode"),)
    builder, tail = _producer_builder(views)
    store = next(op for op in builder.ops if op.op_id.endswith("index_store"))
    parent = index_pool.packed_parent(store)
    assert parent["value"]["dtype"] == "bf16"
    assert parent["value"]["packed_elements"] == 1
    assert parent["value"]["materialization"] == "logical_value"
    assert parent["scale"] == {
        "producer": None, "output": None, "dtype": "none", "shape": (),
        "count": 0, "row_bytes": 0, "row_offset": parent["value"]["data_row_bytes"],
        "presence": False, "materialization": "absent"}
    assert parent["row_materialization"] == "k_only"
    assert parent["tail"]["presence"] is True
    assert parent["presence"] == {"value": True, "scale": False, "tail": True}
    assert parent["native_scatter"]["scale"] is None
    assert "one packed K write otherwise" in \
        parent["native_scatter"]["child_contract"]
