"""Declared CP plan reaches piecewise subgraphs and parent geometry."""
from __future__ import annotations
from dataclasses import replace
from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import (ModelPipeline,
    _pcp_rank_geometry, assemble_model, selected_geometry)
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.schema.model import ModelSpec

SUPPLY = PipeSupply((("cube", 1_000_000.0), ("vector", 1_000_000.0),
                     ("hbm", 1_000_000.0), ("mte1", 2_000_000.0),
                     ("mte2", 2_000_000.0), ("mte3", 2_000_000.0),
                     ("fixpipe", 2_000_000.0), ("link", 1_000_000.0)))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)





def _model(indexer=False):
    plan = dict(index=0, attention="mla" if indexer else "gqa", ffn="dense",
                num_heads=4, num_kv_heads=2, head_dim=16,
                intermediate_size=64, qk_norm=True)
    if indexer:
        plan.update(indexer=True, index_heads=2, index_dim=16, index_topk=8,
                    owns_index_k=True, index_k_owner_layer_id=0,
                    topk_owner_layer_id=0, owns_topk=True)
    raw = dict(name="toy", layers=1, hidden_size=64, num_heads=4,
               num_kv_heads=2, head_dim=16, intermediate_size=64,
               vocab_size=128, layer_plans=[plan])
    if indexer:
        raw["mla"] = dict(q_lora_rank=16, kv_lora_rank=16,
                          qk_nope_head_dim=8, qk_rope_head_dim=8,
                          v_head_dim=8, index_topk=8)
    return ModelSpec.from_mapping(raw)


def _features(indexer=False, **values):
    base = dict(token_tile=16, kv_tile=16, cores=2, moe_row_tile=4,
                split_target=1000, max_splits=4)
    base.update(values)
    return FeatureConfig.from_model(_model(indexer), **base)


def test_cp_plan_is_declared_projected_and_preserved():
    f = _features(cp=2, cp_plan="dcp")
    assert f.cp_plan == "dcp"
    assert f.project(tp=1).cp_plan == "dcp"
    dsa = _features(tp=4, cp=1, cp_plan="dsa_cp")
    assert dsa.cp_plan == "dsa_cp" and dsa.tp == 4 and dsa.cp == 1
    assert dsa.project(tp=4).cp_plan == "dsa_cp"
    assert _features(pcp=2, cp_plan="pcp").world_size == 2


def test_dcp_replicates_indexer_and_dsa_uses_query_token_sp_gather():
    dcp = _features(indexer=True, cp=2, cp_plan="dcp")
    p_dcp = assemble_model(_model(indexer=True), supply=SUPPLY,
                           policy=POLICY, features=dcp)
    i1_dcp = p_dcp.instantiated.find("I1", layer=0)
    a1_dcp = p_dcp.instantiated.find("A1", layer=0)
    assert "cp_gather" not in {op.name for op in i1_dcp.operators}
    assert any(op.name == "dcp_allgather" for op in a1_dcp.operators)
    dsa = _features(indexer=True, tp=4, cp=1, cp_plan="dsa_cp")
    p_dsa = assemble_model(_model(indexer=True), supply=SUPPLY,
                           policy=POLICY, features=dsa)
    i1_dsa = p_dsa.instantiated.find("I1", layer=0)
    a1_dsa = p_dsa.instantiated.find("A1", layer=0)
    assert "cp_gather" not in {op.name for op in i1_dsa.operators}
    assert "dsa_cp_output_all_gather" in {op.name for op in a1_dsa.operators}
    assert not any(op.name in {"dcp_allgather", "dcp_lse_merge"}
                   for op in a1_dsa.operators)
    assert dict(p_dsa._subgraph(i1_dsa).feature_metadata) == dict(dsa.metadata)


def test_pcp_rank_geometry_and_gather_are_phase_and_parent_visible():
    f = _features(pcp=2, cp_plan="pcp", cp=1)
    pipe = assemble_model(_model(), supply=SUPPLY, policy=POLICY, features=f)
    prefill = WorkloadGeometry((RequestGeometry(8, 0, phase="prefill"),),
                               phase="prefill")
    from mindie_sim.catalog.context_parallel import (
        pcp_rank_chunk_ranges, split_pcp_pairs,
    )
    local = _pcp_rank_geometry(prefill, 2)
    assert sum(r.q_len for r in local.requests) == 4
    split = split_pcp_pairs(8, 0, 2)
    rank = split.per_rank.index(split.max_rank)
    assert tuple((r.q_len, r.context) for r in local.requests) == tuple(
        (end - start, start) for start, end in pcp_rank_chunk_ranges(8, 2, rank))
    on = pipe.forward(prefill, fusion=False)
    off = assemble_model(_model(), supply=SUPPLY, policy=POLICY,
                         features=_features(pcp=1, cp=1)).forward(prefill, fusion=False)
    assert any(row.cost.kind == "A0" and row.cost.name == "attention"
               for row in on.rows)
    gather = [row for row in on.rows if row.cost.family == "collective"
              and row.cost.name == "pcp_kv_gather"]
    assert len(gather) == 1
    assert gather[0].duration_ms > 0
    assert not any(row.cost.name == "dcp_allgather" for row in on.rows)
    assert on.total_ms != off.total_ms
    decode = WorkloadGeometry((RequestGeometry(1, 8, phase="decode"),),
                              phase="decode")
    dec = pipe.forward(decode, fusion=False)
    dg = [row for row in dec.rows if row.cost.name == "pcp_kv_gather"]
    assert not dg or all(row.duration_ms == 0 for row in dg)


def _eval(pipe, kind, geometry, state=None, layer=0, role=None):
    from mindie_sim.prediction.subgraphs.state import StateView
    subgraph = pipe.instantiated.find(kind, layer=layer, role=role)
    state = state or StateView()
    if kind in {"I1", "A1", "A2"}:
        state = state.mark_kv(layer).mark_index(layer).mark_selection(layer)
    return pipe._subgraph(subgraph).evaluate(geometry, state)


def test_dcp_attention_is_sharded_while_dsa_uses_local_query_sp():
    geometry = WorkloadGeometry((RequestGeometry(256, 0, phase="prefill"),),
                                phase="prefill")
    base = assemble_model(_model(), supply=SUPPLY, policy=POLICY,
                          features=_features(cp=1, cp_plan="none"))
    dcp = assemble_model(_model(), supply=SUPPLY, policy=POLICY,
                         features=_features(cp=2, cp_plan="dcp"))
    base_a0, dcp_a0 = _eval(base, "A0", geometry), _eval(dcp, "A0", geometry)
    base_attn = next(c for c in base_a0.operators if c.name == "attention")
    dcp_attn = next(c for c in dcp_a0.operators if c.name == "attention")
    assert dcp_attn.duration_ms < base_attn.duration_ms
    assert any(row.name in {"dcp_allgather", "dcp_lse_merge"}
               for row in dcp_a0.operators)

    from mindie_sim.catalog.context_parallel import dsa_cp_rank_geometry
    idx = _model(indexer=True)
    p_dsa = assemble_model(idx, supply=SUPPLY, policy=POLICY,
                           features=_features(indexer=True, tp=4, cp=1,
                                              cp_plan="dsa_cp"))
    local = dsa_cp_rank_geometry(geometry, 4)
    dsa_i1 = _eval(p_dsa, "I1", local, layer=0)
    score = next(c for c in dsa_i1.operators if c.name == "index_score")
    rows = dict(score.derived)["index_rows"]
    assert rows == sum(request.pairs(causal=True) for request in local.requests)
    assert not any(c.name == "cp_gather" for c in dsa_i1.operators)
    dsa_a1 = _eval(p_dsa, "A1", local, layer=0)
    assert any(c.name == "dsa_cp_output_all_gather"
               for c in dsa_a1.operators)
    assert not any(c.name in {"dcp_allgather", "dcp_lse_merge"}
                   for c in dsa_a1.operators)


def test_pcp_actual_source_model_uses_local_rank_chunks():
    import json
    from pathlib import Path
    from mindie_sim.prediction.structure import parse_case
    case_path = (Path(__file__).parents[1] / "examples/cases/"
                 "longctx-decode-20261030/a3-qwen35-frozen-ffn/"
                 "qwen35-a3-tp4-frozen-ffn.json")
    case = parse_case(json.loads(case_path.read_text()))
    cores = max(1, case.machine.dies[0].compute_units)
    f = FeatureConfig.from_model(
        case.model, tp=4, dp=1, ep=1, cp=1, pcp=2, cp_plan="pcp",
        cores=cores, graph_buckets=(), token_tile=16, kv_tile=16,
        moe_row_tile=4, split_target=1000, max_splits=4)
    supply = PipeSupply.from_hardware(case.machine, dtype="bf16")
    pipe = ModelPipeline(case.model, f, supply, POLICY)
    geometry = WorkloadGeometry((RequestGeometry(8, 0, phase="prefill"),),
                                phase="prefill")
    local = _pcp_rank_geometry(geometry, 2)
    assert len(local.requests) == 2
    assert sum(r.q_len for r in local.requests) == 4
    prediction = pipe.forward(geometry, fusion=False)
    assert any(row.cost.name == "pcp_kv_gather" for row in prediction.rows)
    assert not any(row.cost.name == "dcp_allgather" for row in prediction.rows)


def test_actual_v41_dsa_query_token_sp_local_rank_and_output_gather():
    import json
    from dataclasses import replace
    from pathlib import Path
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.catalog.context_parallel import dsa_cp_rank_geometry
    case_path = (Path(__file__).parents[1] / "examples/cases/"
                 "longctx-decode-20261030/v41-a3-tp4-dp4-ep16.json")
    case = parse_case(json.loads(case_path.read_text()))
    cores = max(1, case.machine.dies[0].compute_units)
    base = FeatureConfig.from_runtime(case.runtime, case.model).project(
        tp=4, dp=1, ep=16, cores=cores, graph_buckets=())
    supply = PipeSupply.from_hardware(case.machine, dtype="bf16")
    geometry = WorkloadGeometry((RequestGeometry(256, 0, phase="prefill"),),
                                phase="prefill")
    layer = next(l.index for l in case.model.layer_plans if l.indexer)
    dcp = assemble_model(case.model, supply=supply, policy=POLICY,
                         features=replace(base, cp=2, cp_plan="dcp"))
    dsa = assemble_model(case.model, supply=supply, policy=POLICY,
                         features=replace(base, cp=1, cp_plan="dsa_cp"))
    dcp_i1 = _eval(dcp, "I1", geometry, layer=layer)
    local = dsa_cp_rank_geometry(geometry, 4)
    dsa_i1 = _eval(dsa, "I1", local, layer=layer)
    dcp_score = next(c for c in dcp_i1.operators if c.name == "index_score")
    dsa_score = next(c for c in dsa_i1.operators if c.name == "index_score")
    dcp_rows = dict(dcp_score.derived)["index_rows"]
    dsa_rows = dict(dsa_score.derived)["index_rows"]
    assert dsa_rows < dcp_rows
    assert dsa_rows == sum(r.pairs(causal=True) for r in local.requests)
    assert not any(c.name == "cp_gather" for c in dsa_i1.operators)
    dsa_a2 = _eval(dsa, "A2", local, layer=layer, role="attention")
    assert any(c.name == "dsa_cp_output_all_gather"
               for c in dsa_a2.operators)
    assert not any(c.name in {"dcp_allgather", "dcp_lse_merge"}
                   for c in dsa_a2.operators)
    dcp_a2 = _eval(dcp, "A2", geometry, layer=layer, role="attention")
    assert any(c.name in {"dcp_allgather", "dcp_lse_merge"}
               for c in dcp_a2.operators)


def test_runtime_cp_plan_is_not_lost_by_feature_projection():
    import json
    from dataclasses import replace
    from pathlib import Path
    from mindie_sim.prediction.structure import parse_case
    case_path = (Path(__file__).parents[1] / "examples/cases/"
                 "longctx-decode-20261030/a3-qwen35-frozen-ffn/"
                 "qwen35-a3-tp4-frozen-ffn.json")
    case = parse_case(json.loads(case_path.read_text()))
    runtime = replace(case.runtime, parallel=replace(
        case.runtime.parallel, cp=1, cp_plan="dsa_cp"))
    f = FeatureConfig.from_runtime(runtime, case.model)
    assert f.cp == 1 and f.tp == 4 and f.cp_plan == "dsa_cp"
    assert f.project(tp=4).cp_plan == "dsa_cp"
    assert dict(f.metadata).get("cp_kv_cache_interleave_size") == (
        runtime.kv_block_size or 128)


def test_raw_enable_dsa_cp_resolves_to_query_token_sp_plan():
    import json
    from dataclasses import asdict, replace
    from pathlib import Path
    from mindie_sim.prediction.structure import parse_case
    case_path = (Path(__file__).parents[1] / "examples/cases/"
                 "longctx-decode-20261030/v41-a3-tp4-dp4-ep16.json")
    case = parse_case(json.loads(case_path.read_text()))
    runtime = replace(case.runtime, parallel=replace(
        case.runtime.parallel, enable_dsa_cp=True))
    assert runtime.parallel.cp == 1
    assert runtime.parallel.tp == 4
    assert runtime.parallel.cp_plan == "dsa_cp"
    assert runtime.parallel.enable_dsa_cp is True
    base = FeatureConfig.from_runtime(case.runtime, case.model)
    dsa = FeatureConfig.from_runtime(runtime, case.model)
    diff = {key for key, value in asdict(base).items()
            if value != asdict(dsa)[key]}
    assert diff == {"cp_plan"}
    assert dsa.cp == 1 and dsa.tp == 4 and dsa.cp_plan == "dsa_cp"


def test_pcp_two_logical_requests_keep_one_output_row_each_and_emit_mask():
    features = _features(pcp=2, cp_plan="pcp", cp=1)
    pipe = assemble_model(_model(), supply=SUPPLY, policy=POLICY,
                          features=features)
    geometry = WorkloadGeometry(
        (RequestGeometry(8, 0, phase="prefill"),
         RequestGeometry(8, 0, phase="prefill")), phase="prefill")
    local = _pcp_rank_geometry(geometry, 2)
    assert local.local_batch == 2
    assert sum(r.emit for r in local.requests) == 2
    prediction = pipe.forward(geometry, request_ids=("r0", "r1"), fusion=False)
    outputs = [row for row in prediction.rows
               if row.cost.name in {"lm_head", "sampling"}]
    assert len(outputs) == 2
    assert all(row.padded_tokens == 2 for row in outputs)
    one = pipe.forward(geometry, request_ids=("r0", "r1"),
                       emit_mask=(True, False), fusion=False)
    one_out = [row for row in one.rows
               if row.cost.name in {"lm_head", "sampling"}]
    assert len(one_out) == 2
    assert all(row.padded_tokens == 1 for row in one_out)
    none = pipe.forward(geometry, request_ids=("r0", "r1"),
                        emit_mask=(False, False), fusion=False)
    assert not [row for row in none.rows
                if row.cost.name in {"lm_head", "sampling"}]


def test_pcp_nonfinal_rank_has_no_logit_row_and_preserves_parent_ranges():
    features = _features(pcp=2, cp_plan="pcp", cp=1)
    pipe = assemble_model(_model(), supply=SUPPLY, policy=POLICY,
                          features=features)
    geometry = WorkloadGeometry((
        RequestGeometry(8, 0, phase="prefill",
                        query_ranges=((1, 3), (6, 8))),), phase="prefill")
    rank1 = _pcp_rank_geometry(geometry, 2, rank=1)
    assert not any(request.emit for request in rank1.requests)
    assert all(request.parent_query_ranges == ((1, 3), (6, 8))
               for request in rank1.requests)
    rank0 = _pcp_rank_geometry(geometry, 2, rank=0)
    selected_emit = [request for request in rank0.requests if request.emit]
    assert len(selected_emit) == 1
    assert selected_emit[0].emit_position == 7
    prediction = pipe.forward(geometry, request_ids=("r0",), fusion=False)
    assert [row for row in prediction.rows
            if row.cost.name in {"lm_head", "sampling"}]


def test_owner_group_expansion_preserves_chunks_and_ragged_guard():
    from mindie_sim.schema.common import ModelingError
    requests = (
        RequestGeometry(2, 0, phase="prefill", owner_index=0, emit=False),
        RequestGeometry(2, 6, phase="prefill", owner_index=0, emit=True,
                        emit_position=7),
    )
    exact = WorkloadGeometry(requests, phase="prefill", cohort_batch=1,
                             local_batch=1)
    assert exact.for_subgraph("F3").requests == exact.requests
    # physical chunks == target logical batch must still expand the one owner
    # group; the old early batch==self.batch return would hide this.
    same_count = WorkloadGeometry(requests, phase="prefill", cohort_batch=2,
                                  local_batch=1).for_subgraph("F3")
    assert len(same_count.requests) == 4
    assert [r.owner_index for r in same_count.requests] == [0, 0, 1, 1]
    expanded = WorkloadGeometry(requests, phase="prefill", cohort_batch=4,
                                local_batch=1).for_subgraph("F3")
    assert len(expanded.requests) == 8
    assert [r.owner_index for r in expanded.requests] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [(r.q_len, r.context) for r in expanded.requests] == [
        (2, 0), (2, 6)] * 4
    assert [r.emit for r in expanded.requests] == [False, True] * 4
    ordinary = WorkloadGeometry(
        (RequestGeometry(2, 0, phase="prefill"),
         RequestGeometry(2, 6, phase="prefill")), phase="prefill",
        cohort_batch=1, local_batch=2)
    try:
        ordinary.for_subgraph("F3")
    except ModelingError as error:
        assert "workload_ragged_cohort_expansion_ambiguous" in str(error)
    else:
        raise AssertionError("ordinary ragged guard was bypassed")
