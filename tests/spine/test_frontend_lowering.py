"""Frontend lowering: target graphs, draft/media frontends and body declarations.

[Role] Exercise the public frontend and graph-analysis boundary: stream and
  memory contention, expert-cohort transport, draft and multimodal/media
  lowering, effective body-declaration carriers, and partial public paths.
[Invariants] Independent analytical byte/FLOP/wire totals are conserved; every
  invalid input raises its own exact error code; declared ranks keep identity.
[Timeouts & Locks] Tier-0 in-memory simulation over compact two-layer catalogs.
[Side-Effect Policy] Read-only case loads and in-memory execution; no devices.
"""
from copy import copy, deepcopy
from dataclasses import replace
from functools import lru_cache
import json

import pytest
from _support.captures import REPO_ROOT, captured_report, norm_case
from _support.hf_configs import config, run_case
from _support.memory_cases import case, draft_case
from _support.operator_cases import mqsmla_attention, mqsmla_case
from _support.serving_cases import STUDY_PROBES, pd_case
from mindie_sim import Case, graph_costs, operator_costs, predict, study, target_graphs, target_operators
from mindie_sim.__main__ import parser, run
from mindie_sim.catalog.cache import cache_bytes_per_token_estimate, cache_format, request_cache_breakdown, stage_cache_components
from mindie_sim.catalog.draft import optional_draft_spec
from mindie_sim.catalog.draft_weights import draft_terms
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import AuxiliarySpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import stage_weight_terms
from mindie_sim.cohort_accuracy import compare_cohorts
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import CONFIG_DIR, load_model
from mindie_sim.prediction.operators.attention.attention_pipeline import TiledFlashAndPagedAttentionRecipe
from mindie_sim.prediction.body_defaults import effective_body_declaration, service_program_defaults
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.program.physical_key import physical_key
from mindie_sim.prediction.program import RecipeBook, TransferRecipe
from mindie_sim.prediction.scheduler import DraftPolicy, Progress
from mindie_sim.prediction.frontend.speculation import draft_program
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure import export_op_structure, export_structure

pytestmark = pytest.mark.tier0
GEOMETRY = dict(phase="decode", tokens=1, context=128, sequences=2, scope="body")
MEDIA_CASE = REPO_ROOT / "examples/cases/qwen3.8-27b-media.json"
KIMI_CASE = REPO_ROOT / "examples/cases/kimi-k3.json"
FLASH_DRAFT_CASE = REPO_ROOT / "tests/fixtures/verification/v4flash-mtp-case.json"
def op(name, *, flops=0, memory=0, deps=(), stream="main", stage=0):
    """One decorated prefill ``OpSpec`` on an isolated private stream."""
    return OpSpec(name, "test", "prefill", stage, 0, (), (1,), (1,), (1,), (1,), flops=flops, hbm_bytes=memory, deps=deps, dtype="bf16", stream=stream)
def _gdn_case(sku):
    return {"machine": {"backend": "ascend", "sku": sku, "num_devices": 8}, "model": {"preset": "qwen3.6-27b", "quantization": "w8a8", "layers": 1}, "runtime": {"tp": 8}, "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]}}
def _gqa_case():
    raw = case(output=1); raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}; return raw
def _graph(raw, graphs, **kwargs):
    return graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False, **kwargs)
def _attention(result):
    return [row for row in result["timeline"] if row["family"] == "attention"]
def _assert_timeline_close(left, right, *, rel=1e-9):
    """Identical structure and ids; engine float sums may differ by one ulp."""
    assert [row["id"] for row in left] == [row["id"] for row in right]
    def close(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            assert set(a) == set(b), path
            for key in a: close(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            assert len(a) == len(b), path
            for index, (x, y) in enumerate(zip(a, b)): close(x, y, f"{path}[{index}]")
        elif isinstance(a, bool) or isinstance(b, bool): assert a == b, path
        elif isinstance(a, float) and isinstance(b, float): assert a == pytest.approx(b, rel=rel, abs=1e-15), path
        else: assert a == b, path
    for x, y in zip(left, right): close(x, y, x["id"])
@lru_cache(maxsize=1)
def _kimi_case():
    raw = json.loads((REPO_ROOT / "mindie_sim/model_configs/kimi-k3.json").read_text()); text = raw["text_config"]
    text.update(model_type="kimi_k31", dspark_attention_kind="mla", dspark_block_size=5, dsa_config={"dsa_layers": text["linear_attn_config"]["full_attn_layers"], "index_num_heads": 32, "index_head_dim": 128, "index_topk": 2048, "index_rope_dim": 0})
    quant = {"name": "k31_fp8_cache", "weight_format": "bf16", "activation_format": "bf16", "expert_weight_format": "bf16", "kv_dtype": "fp8_e4m3", "index_dtype": "fp8_e4m3", "index_scale_dtype": "fp16"}
    return parse_case(compose_case(raw, hardware="a5", tp=8, dp=4, ep=32, dcp=8, num_devices=32, block_size=1536, max_model_len=140000, mtp_proposal_tokens=3, mtp_acceptance_length=2, graph_capture_mode="none", prompt_tokens=131072, output_tokens=1024, quant=quant))
@lru_cache(maxsize=1)
def _mq_case():
    parsed = mqsmla_case(); return parsed, mqsmla_attention(target_graphs(parsed, phase="decode", tokens=1, context=2048)[0])
def _draft_regions(parsed, *, read_mode="batch_shared"):
    runtime = replace(parsed.runtime, prefix_cache_mode="shared", prefix_read_mode=read_mode, metadata=tuple({**dict(parsed.runtime.metadata), "draft_mode": "dspark"}.items()))
    views = tuple(RequestView(str(index), 6, 131584, True, "decode", prefix_tokens=124518, prefix_cache_key="same", draft_proposal_tokens=5) for index in range(4))
    return draft_program(Frontend(resource_fusion=False), parsed.model, runtime, views, "ready", machine=parsed.machine)[0]
def _draft_kv_delta(model, tp: int, tokens: int = 4096) -> float:
    parallel = ParallelSpec(tp=tp, dp=1, ep=1, pp=1)
    base, _ = cache_bytes_per_token_estimate(model, parallel, tokens)
    drafted, _ = cache_bytes_per_token_estimate(model, parallel, tokens, draft_kind="dspark")
    return drafted - base
def test_graph_costs_conserves_streams_dependencies_and_submission():
    compute, memory = op("compute", flops=4e6), op("copy", memory=3e6, stream="copy")
    parallel = graph_costs(case(), (compute, memory)); serial = graph_costs(case(), (compute, replace(memory, stream="main")))
    assert parallel["duration_ms"] == 4 and serial["duration_ms"] == 7
    assert parallel["isolated"]["operator_sum_ms"] == serial["isolated"]["operator_sum_ms"] == 7
    assert parallel["bounds"]["precedence"]["duration_ms"] == 4
    assert serial["bounds"]["precedence"]["duration_ms"] == 7
    assert parallel["execution"]["serviced"] == serial["execution"]["serviced"]
    shared = graph_costs(case(), (memory, op("other_copy", memory=2e6, stream="other")))
    assert shared["duration_ms"] == 5 and shared["bounds"]["precedence"]["duration_ms"] == 3
    assert shared["bounds"]["resource_capacity_ms"] == 5
    assert shared["execution"]["serviced"]["memory@r0:s0"] == 5e6
    first = op("producer", flops=4e6, deps=("upstream",)); second = op("consumer", memory=3e6, deps=("producer",), stream="copy")
    report = graph_costs(case(), (first, second), input_ready_ms=10, external_inputs={"upstream": 12}, trace_limit=100)
    assert report["duration_ms"] == 9 and report["complete_ms"] == 19
    assert report["bounds"]["precedence"]["duration_ms"] == 9
    assert [row["ready_ms"] for row in report["timeline"]] == [12, 16]
    assert all(event["start_ms"] >= 12 for event in report["events"] if event["demand"])
    with pytest.raises(ValueError, match="graph_dependency_unavailable:producer:upstream"): graph_costs(case(), (first, second))
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    raw = case(); raw["machine"]["device"]["launch_us"] = 1000; raw["runtime"]["submission"] = {"scheduler_ms": 2, "completion_ms": .5}
    operators = (op("a", flops=4e6), op("b", flops=2e6, deps=("a",)))
    pure, submitted = graph_costs(raw, operators), graph_costs(raw, operators, include_submission=True)
    assert pure["duration_ms"] == 6 and submitted["duration_ms"] == 9.5
    parsed = parse_case(raw); execution = Execution(parsed, CostModel(parsed))
    step = execution.add(IterationWork(operators, IterationPlan(()), 1, 0, 0, None, "prefill", "test"), 0, 0)
    execution.engine.advance(until=(step.completion,))
    assert submitted["duration_ms"] == step.completion.time
    assert submitted["execution"]["serviced"] == execution.engine.serviced
    assert pure["execution"]["serviced"]["host@r0:s0"] == 0
    assert submitted["execution"]["serviced"]["host@r0:s0"] == 2
    assert submitted["execution"]["serviced"]["scheduler@r0"] == 2
@pytest.mark.parametrize("scope", ("body", "model"))
def test_real_frontend_preserves_join_barriers_and_model_boundary(scope):
    raw = case(pp=2, tp=2); graphs = target_graphs(raw, phase="prefill", tokens=3, scope=scope)
    report = graph_costs(raw, graphs, input_ready_ms=5, external_inputs={"analysis_input": 5} if scope == "body" else None)
    assert report["duration_ms"] > 0
    assert report["bounds"]["combined_ms"] <= report["duration_ms"] + 1e-9
    assert not any(row["family"] in {"sampling", "token_commit"} for row in report["timeline"])
    if scope == "model":
        assert any(row["op_id"] == "iteration_input" for row in report["timeline"])
        assert any(row["family"] == "embedding" for row in report["timeline"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    summed = {}
    for row in report["isolated"]["operators"]:
        for resource, amount in row["serviced"].items(): summed[resource] = summed.get(resource, 0) + amount * row["occurrences"]
    actual = report["execution"]["serviced"]
    assert {key: amount for key, amount in actual.items() if amount} == pytest.approx({key: amount for key, amount in summed.items() if amount})
def test_expert_cohort_conserves_transport_startup_and_cohort_completeness():
    raw = case(dp=2, ep=2); raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    exchange = replace(op("exchange"), family="alltoall", link_bytes=1000, metadata=(("group_axis", "ep"),))
    graphs = {0: (op("local", flops=4e6), exchange), 1: (op("local", flops=1e6), exchange)}
    report = graph_costs(raw, graphs)
    assert report["duration_ms"] == 6 and report["execution"]["serviced"]["link:fabric"] == 2000
    exchanges = [row for row in report["timeline"] if row["op_id"] == "exchange"]
    assert [row["complete_ms"] for row in exchanges] == [6, 6]
    assert [row["ready_ms"] for row in exchanges] == [4, 4]
    assert [row["local_ready_ms"] for row in exchanges] == [4, 1]
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    with pytest.raises(ValueError, match="graph_expert_collective_requires_complete_cohort"): graph_costs(raw, graphs[0])
    model_graphs = target_graphs(raw, phase="decode", tokens=1, context=8, scope="model")
    assert set(model_graphs) == {0, 1}
    model_report = graph_costs(raw, model_graphs)
    assert model_report["duration_ms"] > 0
    assert model_report["bounds"]["combined_ms"] <= model_report["duration_ms"] + 1e-9
    assert model_report["execution"]["live_tasks"] == model_report["execution"]["live_programs"] == 0
    assert model_report["execution"]["serviced"]["link:fabric"] > 0
    raw["machine"]["links"][0]["latency_us"] = 10000; peer_graphs = {0: (exchange,), 1: (exchange,)}
    startup = graph_costs(raw, peer_graphs)
    assert startup["duration_ms"] == 12 and startup["bounds"]["combined_ms"] <= startup["duration_ms"]
    raw["machine"]["links"][0]["latency_us"] = 0
    recipe = TransferRecipe(transaction_bytes=10, outstanding=1, response_ms=1)
    limited = graph_costs(raw, peer_graphs, recipes=RecipeBook(families={"alltoall": recipe}))
    # Two senders share 1000B/ms; each of 100 transactions uses .02ms wire service plus a 1ms response.
    assert limited["duration_ms"] == pytest.approx(102) and limited["execution"]["serviced"]["link:fabric"] == 2000
    assert limited["bounds"]["combined_ms"] <= limited["duration_ms"]
def test_finite_transfer_uses_link_frames_without_reframing_wire_payload():
    raw = case(tp=2); raw["machine"]["links"][0].update(transaction_bytes=16, protocol_overhead_bytes=4)
    exchange = replace(op("exchange"), family="alltoall", link_bytes=1)
    recipes = RecipeBook(families={"alltoall": TransferRecipe(transaction_bytes=16, outstanding=1, response_ms=1)})
    result = graph_costs(raw, (exchange,), recipes=recipes)
    # Two one-byte messages become two 20-byte frames; each returns one credit.
    assert result["execution"]["serviced"]["link:fabric"] == 40 and result["duration_ms"] == pytest.approx(2.04)
    default_granule = graph_costs(raw, (exchange,), recipes=RecipeBook(families={"alltoall": TransferRecipe(outstanding=1, response_ms=1)}))
    assert default_granule["duration_ms"] == result["duration_ms"]
    with pytest.raises(ValueError, match="transfer_granule_conflicts_with_link"):
        graph_costs(raw, (exchange,), recipes=RecipeBook(families={"alltoall": TransferRecipe(transaction_bytes=8, outstanding=1, response_ms=1)}))
def test_graph_cli_runs_public_model_path_without_sampling_or_host_work():
    args = parser().parse_args(["graph", "examples/quickstart.json", "--phase", "decode", "--tokens", "1", "--context", "16", "--skip-isolated"])
    report = run(args)
    assert report["duration_ms"] > 0
    assert any(row["family"] == "embedding" for row in report["timeline"])
    assert not any(row["family"] == "sampling" for row in report["timeline"])
    assert all(amount == 0 for name, amount in report["execution"]["serviced"].items() if name.startswith(("host@", "scheduler@")))
def test_target_graph_retains_frontend_capture_scope_and_padding():
    raw = case(); raw["machine"]["device"]["launch_us"] = 1000; raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full")
    graphs = target_graphs(raw, phase="decode", tokens=1)
    captured = graph_costs(raw, graphs, include_submission=True); eager = graph_costs(raw, graphs, include_submission=True, replay_scope="none")
    assert captured["replay_scope"] == "full"
    assert captured["execution"]["serviced"]["host@r0:s0"] == 1
    assert eager["execution"]["serviced"]["host@r0:s0"] > 1
    assert captured["execution"]["work"] == eager["execution"]["work"]
    assert captured["duration_ms"] < eager["duration_ms"]
def test_media_geometry_is_explicit_and_preserves_actual_frontend_tower_work():
    raw = json.loads(MEDIA_CASE.read_text())
    with pytest.raises(ValueError, match="analysis_requires_explicit_media_tokens_per_sequence"): target_graphs(raw, phase="prefill", tokens=5)
    graphs = target_graphs(raw, phase="prefill", tokens=5, media_tokens_per_sequence=4)
    language = target_graphs(raw, phase="prefill", tokens=5, media_tokens_per_sequence=0)
    parsed = parse_case(raw)
    actual = Frontend()(IterationPlan((RequestView("r0", 5, 0, True, "prefill", draft_prepare=False),)), parsed.model, parsed.runtime, parsed.machine)
    expected = [item for item in actual.ops if item.family not in {"sampling", "token_commit"}]
    assert sum(item.flops for item in graphs[0]) == sum(item.flops for item in expected)
    assert sum(item.hbm_bytes for item in graphs[0]) == sum(item.hbm_bytes for item in expected)
    assert sum(item.flops for item in graphs[0]) > sum(item.flops for item in language[0])
def test_inactive_model_layers_do_not_expand_per_core_programs():
    raw, cores, layers = case(), 64, 24; raw["machine"]["device"]["compute_units"] = cores
    parsed, shape = parse_case(raw), (128, 128 * cores, 128)
    matrix = replace(op("matrix", flops=2 * shape[0] * shape[1] * shape[2]), local_shape=shape, semantic_shape=shape, padded_shape=shape, launch_shape=shape, metadata=(("program_kind", "matmul"),))
    operators = tuple(replace(matrix, op_id=f"matrix{index}") for index in range(layers))
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    step = execution.add(IterationWork(operators, IterationPlan(()), 1, 0, 0, None, "prefill", "test"), 0, 7)
    assert execution.engine.summary()["live_tasks"] < 4 * layers + 10  # no eager per-core expansion before input
    execution.engine.advance(horizon=6)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(step.completion,))
    assert step.completion.time == pytest.approx(7 + layers * matrix.flops / 1e6)
    assert execution.engine.serviced["matrix@r0:s0"] == pytest.approx(layers * matrix.flops / 1e6)
    assert execution.engine.peak_tasks < 12 * cores + 4 * layers
    assert execution.engine.summary()["live_tasks"] == 0 and execution.engine.summary()["live_programs"] == 0
def test_draft_prefill_populates_cache_and_skips_unused_one_token_draft():
    parsed = draft_case(); baseline = predict(replace(parsed, runtime=replace(parsed.runtime, metadata=()))); result = predict(parsed, event_limit=2000)
    assert result["requests"][0]["generated_tokens"] == 6
    assert result["requests"][0]["speculative"]["proposed_tokens"] > 0
    assert result["requests"][0]["ttft_ms"] == baseline["requests"][0]["ttft_ms"]
    commits = [event for event in result["events"] if "token_commit" in event["task"] and not event["task"].endswith(":submit")]
    prepared = [event for event in result["events"] if "draft_prefill." in event["task"] and not event["task"].endswith(":submit")]
    assert prepared and min(event["start_ms"] for event in prepared) >= commits[0]["end_ms"]
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    view = RequestView("r", 1, 4, True, "decode", draft_proposal_tokens=2)
    work = Frontend()(IterationPlan((view,)), parsed.model, parsed.runtime, parsed.machine)
    target = [item for item in work.ops if item.recipe == "qkv_projection" and not item.op_id.startswith("draft")]
    assert len(target) == 2 and all(item.local_shape[0] == 3 for item in target)
    assert len([item for item in work.ops if item.recipe == "draft_features"]) == 2
    single = replace(parsed, workload=replace(parsed.workload, requests=(replace(parsed.workload.requests[0], output_tokens=1),)))
    single_result = predict(single, event_limit=2000)
    assert not any("draft_prefill." in event["task"] for event in single_result["events"])
    assert single_result["metrics"]["makespan_ms"] == single_result["requests"][0]["ttft_ms"]
def test_prefix_marginals_match_categorical_counts_and_samples_are_reproducible():
    spec = SpeculativeSpec(3, (.8, .4, .1))
    assert spec.expected_committed_tokens() == 2.3
    conditional = SpeculativeSpec(3, (.8, .5, .25), acceptance_mode="conditional")
    assert spec.acceptance_distribution() == conditional.acceptance_distribution()
    parsed = draft_case(); parsed = replace(parsed, runtime=replace(parsed.runtime, metadata=(("draft_mode", "mtp"), ("speculative", {"draft_len": 2, "accepted_distribution": [.5, 0, .5]}))))
    request, sequence = parsed.workload.requests[0], []; policy, a, b = DraftPolicy.for_request(parsed, request), Progress(request), Progress(request)
    for step in range(100):
        a.decode_steps = b.decode_steps = step; value = policy.committed(a, 2)
        assert value == policy.committed(b, 2); sequence.append(value)
    assert set(sequence) == {1, 3}
    assert DsparkSpec(2, (.5, .2), confidence=(.1, .8), confidence_threshold=.5).step_spec().draft_len == 0
def test_encoded_media_keeps_sequence_positions_and_skips_only_the_tower():
    raw = json.loads(MEDIA_CASE.read_text()); raw["workload"] = {"requests": [{"id": "media", "text_tokens": 4, "media_tokens": 4, "output_tokens": 1}]}
    parsed = Case.from_mapping(raw); fresh = RequestView("media", 8, 0, True, "prefill")
    a = Frontend()(IterationPlan((fresh,)), parsed.model, parsed.runtime, parsed.machine)
    b = Frontend()(IterationPlan((replace(fresh, media_encoded=True),),), parsed.model, parsed.runtime, parsed.machine)
    assert a.valid_tokens == b.valid_tokens == 8
    assert any(item.op_id.startswith("media0:") for item in a.ops)
    assert not any(item.op_id.startswith("media0:") for item in b.ops)
    assert sum(item.flops for item in a.ops) > sum(item.flops for item in b.ops)
    assert parsed.workload.requests[0].prompt_tokens == 8
    patches = parsed.model.vision.patch_tokens_for_media(4)
    qk = next(item for item in a.ops if item.op_id == "media0:vision.l0.qk"); pv = next(item for item in a.ops if item.op_id == "media0:vision.l0.pv")
    assert qk.flops == pv.flops == 2 * patches ** 2 * parsed.model.vision.query_width
    assert pv.hbm_bytes == pv.mte2_bytes + pv.mte3_bytes
def test_draft_quantization_and_projection_shards_match_resident_weights():
    parsed, quant = draft_case(), quant_preset("w8a8_dynamic"); draft = replace(parsed.model.auxiliary.mtp, weight_category="mtp")
    model = replace(parsed.model, quant=quant, auxiliary=AuxiliarySpec(mtp=draft))
    work = Frontend()(IterationPlan((RequestView("r", 1, 4, True, "decode", draft_proposal_tokens=2),)), model, parsed.runtime, parsed.machine)
    draft_matrices = [item for item in work.ops if item.op_id.startswith("draft") and item.flops]
    assert all(item.dtype == "bf16" for item in draft_matrices)
    body = [item for item in work.ops if not item.op_id.startswith("draft") and item.recipe == "qkv_projection"]
    assert all(item.dtype == "int8" for item in body)
    term = next(term for term in draft_terms(draft, model) if term.name == "draft_q")
    assert term_bytes(term, quant, parsed.runtime.parallel) == draft.q_width * draft.hidden_size * 2
def test_dspark_kv_cache_footprint_and_block_draft_subgraph():
    parsed = Case.from_mapping(json.loads(KIMI_CASE.read_text())); model = parsed.model
    assert model.auxiliary is not None and model.auxiliary.dspark is not None
    p_tp8 = ParallelSpec(tp=8, dp=4, ep=32, pp=1); base_tp8 = request_cache_breakdown(model, p_tp8, 4096); dspark_tp8 = request_cache_breakdown(model, p_tp8, 4096, draft_kind="dspark")
    assert "draft_kv" not in base_tp8["by_component"]
    # 16 MLA layers: 576 elements * 2B = 1152 B/token/layer -> 18,432 B/token; TP=8 -> 5 * 2 local kv_heads * 2 * 64 * 2B = 2,560 B/token.
    assert dspark_tp8["by_component"]["draft_kv"] == 4096 * 2560.0 and dspark_tp8["total_bytes"] - base_tp8["total_bytes"] == 4096 * 2560.0
    p_tp1 = ParallelSpec(tp=1, dp=32, ep=32, pp=1); base_tp1 = request_cache_breakdown(model, p_tp1, 4096); dspark_tp1 = request_cache_breakdown(model, p_tp1, 4096, draft_kind="dspark")
    assert dspark_tp1["by_component"]["draft_kv"] == 4096 * 20480.0  # 16 local kv_heads -> 5 * 2 * 16 * 64 * 2B/token
    assert base_tp1["by_component"]["mla_kv"] == 4096 * 24 * 1152.0
    bpt_base, _ = cache_bytes_per_token_estimate(model, p_tp1, 4096); bpt_dspark, _ = cache_bytes_per_token_estimate(model, p_tp1, 4096, draft_kind="dspark")
    assert bpt_dspark - bpt_base == 20480.0
    rt_base = replace(parsed.runtime, parallel=p_tp8, pp_layer_partition=(), metadata=()); rt_dspark = replace(parsed.runtime, parallel=p_tp8, pp_layer_partition=(), dspark=DsparkSpec(7, (0.9,) * 7), metadata=(("draft_mode", "dspark"),))
    mem_base, mem_dspark = Memory(replace(parsed, runtime=rt_base)), Memory(replace(parsed, runtime=rt_dspark))
    assert mem_dspark.weights[0] > mem_base.weights[0] and mem_dspark.workspace[0] > mem_base.workspace[0]
    assert mem_dspark.cache_bytes(4096, query_tokens=1)[0] - mem_base.cache_bytes(4096)[0] == 4096 * 2560.0
    bd_base, bd_dspark = mem_base.cache_pool_breakdown(4096), mem_dspark.cache_pool_breakdown(4096)
    assert bd_dspark["draft_kv_bytes"] == 4096 * 2560.0 and bd_dspark["query_tokens"] == 8
    assert bd_dspark["by_component"]["recurrent_state"] == 8 * bd_base["by_component"]["recurrent_state"]
    assert mem_dspark.cache_bytes(4096)[0] > mem_base.cache_bytes(4096)[0] + 4096 * 2560.0
@pytest.mark.parametrize("loader", ("catalog", "config"))
def test_dspark_block5_mla_catalog_and_draft_subgraph(loader):
    if loader == "catalog":
        model = load_model("kimi-k3-dspark-block5"); metadata = dict(model.metadata)
        assert metadata["model_coverage"] == "complete" and metadata["unrecognized_config_fields"] == {}
        assert model.attention == "mla" and model.layers == 5
        assert all(plan.attention == "mla" and plan.kv_cache_elements == 576 and plan.noncausal_attention is False for plan in model.layer_plans)
        assert model.auxiliary is not None and model.auxiliary.standalone_draft and model.auxiliary.dspark.block_size == 5
    else:
        raw = json.loads((REPO_ROOT / "mindie_sim/model_configs/kimi-k3.json").read_text()); raw["text_config"]["dspark_block_size"] = 5; model = ModelSpec.from_mapping(raw)
        assert model.auxiliary is not None and model.auxiliary.dspark is not None
        assert model.auxiliary.dspark.kind == "dspark" and model.auxiliary.dspark.block_size == 5
    for tp in (1, 2, 4, 8): assert _draft_kv_delta(model, tp) == 5760.0
def test_target_plus_draft_prefill_preserves_dependencies_without_commit(tmp_path):
    parsed = draft_case(); target = target_graphs(parsed, phase="prefill", tokens=4); prepared = target_graphs(parsed, phase="prefill", tokens=4, draft_prepare=True)
    assert not any(item.op_id.startswith("draft_prefill.") for item in target[0])
    assert len([item for item in prepared[0] if item.recipe == "qkv_projection"]) == 3
    assert len([item for item in target[0] if item.recipe == "qkv_projection"]) == 2
    assert not any(item.family in {"sampling", "token_commit"} for item in prepared[0])
    result = graph_costs(parsed, prepared); assert result["duration_ms"] > graph_costs(parsed, target)["duration_ms"]
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert any(item.op_id.startswith("draft_prefill.") for item in target_operators(parsed, phase="prefill", tokens=4, scope="model", draft_prepare=True))
    source = tmp_path / "declared-mtp.json"; source.write_text(json.dumps(parsed.to_dict()))
    cli = run(parser().parse_args(["graph", str(source), "--phase", "prefill", "--tokens", "4", "--draft-prepare"]))
    assert cli["duration_ms"] == result["duration_ms"] and cli["draft_prepare"] is True
    raw = json.loads(FLASH_DRAFT_CASE.read_text()); ops = target_graphs(raw, phase="prefill", tokens=4, draft_prepare=True)[0]
    attention = [item for item in ops if item.family == "attention"]
    assert len(attention) == 44 and sum(item.op_id.startswith("draft_prefill.") for item in attention) == 1
    assert not any(dep in {"token_commit", "sampling"} for item in ops for dep in item.deps)
def test_gdn_body_graph_operator_and_structure_share_actual_caller():
    raw = _gdn_case("Ascend950DT_9582"); before = deepcopy(raw); body, graphs = predict(raw, **GEOMETRY), target_graphs(raw, **GEOMETRY)
    direct = _graph(raw, graphs); isolated = operator_costs(raw, target_operators(raw, **GEOMETRY)); structure = export_structure(raw, **GEOMETRY)
    _assert_timeline_close(direct["timeline"], body["timeline"])
    assert direct["duration_ms"] == pytest.approx(body["body_forward_ms"], rel=1e-9, abs=1e-15)
    assert direct["execution"]["serviced"] == pytest.approx(body["execution"]["serviced"], rel=1e-9, abs=1e-9)
    for recipe in ("ascend_causal_conv1d_update", "ascend_gdn_decode_gating", "ascend_recurrent_gdn_fp32_state"):
        assert sum(row["recipe"] == recipe for row in structure["nodes"]) == 1
        assert sum(row["operator"]["recipe"] == recipe for row in isolated["operators"]) == 1
    selected = structure["execution"]["software"]["kernels"]["lowerings"]["gdn_decode"]
    assert selected["implementation"] == "vllm_ascend_gdn_decode" and selected["options"]["gating_beta_store"]["from_hardware"]
    assert structure["execution"]["default_mechanisms"] == body["default_mechanisms"] and raw == before
def test_graph_carrier_keeps_current_hardware_and_explicit_book_or_recipe_overrides():
    raw = _gqa_case(); graphs = target_graphs(raw, **GEOMETRY); automatic = _graph(raw, graphs)
    assert all(row["program"] == "StagedGQADecode" for row in _attention(automatic))
    single_rank = _graph(raw, graphs[0]); _assert_timeline_close(single_rank["timeline"], automatic["timeline"])
    assert single_rank["execution"]["serviced"] == pytest.approx(automatic["execution"]["serviced"], rel=1e-9, abs=1e-9)
    for candidate in (dict(graphs), {0: graphs[0]}): _assert_timeline_close(_graph(raw, candidate)["timeline"], automatic["timeline"])
    cloned = deepcopy(graphs); assert cloned[0].effective_kernels is cloned.effective_kernels
    _assert_timeline_close(_graph(raw, cloned)["timeline"], automatic["timeline"])
    rank_costs = operator_costs(raw, graphs[0])
    assert all(row["program"] == "StagedGQADecode" for row in rank_costs["operators"] if row["operator"]["family"] == "attention")
    isolated = operator_costs(raw, target_operators(raw, **GEOMETRY))
    assert all(row["program"] == "StagedGQADecode" for row in isolated["operators"] if row["operator"]["family"] == "attention")
    assert operator_costs(raw, target_operators(raw, **GEOMETRY)[:])["operators"] == isolated["operators"]
    for clone in (copy, deepcopy): assert operator_costs(raw, clone(graphs[0]))["operators"] == rank_costs["operators"]
    slow = deepcopy(raw); slow["machine"]["hbm_bw_bytes_per_s"] = 1e8; slowed = _graph(slow, graphs)
    assert slowed["duration_ms"] > automatic["duration_ms"] * 10
    # M2 caps small-M cold weight refills at the declared mte2_aic_dram supply, so a slower external HBM rate changes the serviced byte split.
    fast_service, slow_service = automatic["execution"]["serviced"], slowed["execution"]["serviced"]
    assert fast_service["mte1@r0:s0"] == pytest.approx(78880) and fast_service["mte2_aic@r0:s0"] == pytest.approx(74752)
    assert slow_service["mte1@r0:s0"] == pytest.approx(76576) and slow_service["mte2_aic@r0:s0"] == pytest.approx(72448)
    for key in set(fast_service) | set(slow_service):
        if key not in {"mte1@r0:s0", "mte2_aic@r0:s0", "memory@r0:s0"}: assert slow_service.get(key) == pytest.approx(fast_service.get(key))
    # DRAM service modes normalize byte service to the raw bus rate; native byte work is conserved.
    assert fast_service["memory@r0:s0"] > slow_service["memory@r0:s0"] and slowed["execution"]["work"] == automatic["execution"]["work"]
    override = _graph(raw, graphs, recipes=RecipeBook(families={"attention": TiledFlashAndPagedAttentionRecipe()}))
    assert all(row["program"] == "TiledFlashAndPagedAttentionRecipe" for row in _attention(override))
    changed = deepcopy(raw); changed["runtime"]["kernels"] = {"recipes": {"flash_attention_decode": {"kind": "staged_gqa_decode"}}}
    assert all(row["program"] == "StagedGQADecode" for row in _attention(_graph(changed, graphs)))
def test_explicit_book_does_not_install_structural_caller_and_model_scope_is_unchanged():
    raw = _gdn_case("Ascend950DT_9582"); explicit = RecipeBook(); body = predict(raw, recipes=explicit, **GEOMETRY); graphs = target_graphs(raw, apply_body_defaults=False, **GEOMETRY)
    _assert_timeline_close(body["timeline"], _graph(raw, graphs, recipes=explicit)["timeline"])
    assert not any(row["recipe"] == "ascend_gdn_decode_gating" for row in body["timeline"])
    model = {**GEOMETRY, "scope": "model"}; resolved_model = target_graphs(raw, **model)
    # Assembly scope is not implementation scope: model assembly consumes the same shared resolver and keeps its actual scope.
    assert resolved_model.scope == "model" and resolved_model.effective_kernels == target_graphs(raw, **GEOMETRY).effective_kernels
    raw_model = target_graphs(raw, apply_body_defaults=False, **model)
    assert raw_model.effective_kernels != resolved_model.effective_kernels
    assert export_structure(raw, **model)["graph_id"] != export_structure(raw, apply_body_defaults=False, **model)["graph_id"]
def test_plain_source_subset_and_rank_map_require_matching_effective_book():
    raw = _gdn_case("Ascend950DT_9582"); graphs = target_graphs(raw, **GEOMETRY); native = [item for item in graphs[0] if item.recipe == "ascend_gdn_decode_gating"]
    assert len(native) == 1
    with pytest.raises(ModelingError, match="source_plan_binding_required:ascend_gdn_decode_gating"): operator_costs(raw, native)
    extracted = operator_costs(raw, native, recipes=RecipeBook.from_mapping(graphs.effective_kernels))["operators"]
    carried = [row for row in operator_costs(raw, graphs[0])["operators"] if row["operator"]["recipe"] == "ascend_gdn_decode_gating"]
    assert extracted == carried
    rank_raw = _gqa_case(); selected = target_graphs(rank_raw, **GEOMETRY)[0]; control = target_graphs(rank_raw, apply_body_defaults=False, **GEOMETRY)[0]
    for other in (control, tuple(selected)):
        with pytest.raises(ValueError, match="body_target_kernel_declarations_differ_between_ranks"): operator_costs(rank_raw, {0: selected, 1: other})
@pytest.mark.requires_profiler
def test_profiler_cohort_replays_effective_body_and_rejects_undeclared_manifest():
    raw = norm_case(); raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}
    mapping, _ = captured_report(raw, "a3_glm53_decode.csv", "RmsNorm"); compared = compare_cohorts(raw, mapping)
    assert compared["windows"][0]["cohorts"][0]["status"] == "compared"
    assert compared["declared_execution"]["software"]["kernels"]["recipes"]["flash_attention_decode"]["kind"] == "staged_gqa_decode"
    structure = export_structure(raw, phase="decode", tokens=1, scope="body", apply_body_defaults=False)
    undeclared, _ = captured_report(raw, "a3_glm53_decode.csv", "RmsNorm", structure=structure)
    assert compare_cohorts(raw, undeclared)["windows"][0]["cohorts"][0]["status"] == "compared"  # explicit generic control still replays
    undeclared["declared_execution"].pop("apply_body_defaults"); undeclared["declared_execution"].pop("default_mechanisms")  # historical manifests must rebuild
    with pytest.raises(ValueError, match="cohort_case_graph_id_mismatch"): compare_cohorts(raw, undeclared)
def test_body_study_and_cli_graph_share_selected_programs(tmp_path):
    raw = _gqa_case(); body = predict(raw, **GEOMETRY); result = study([{"id": "base", "case": raw}], [{"id": "body", **GEOMETRY}], level="graph")
    assert result["evaluations"][0]["duration_ms"] == body["body_forward_ms"]
    path = tmp_path / "case.json"; path.write_text(json.dumps(raw))
    args = ["graph", str(path), "--phase", "decode", "--tokens", "1", "--context", "128", "--sequences", "2", "--scope", "body", "--skip-isolated"]
    assert run(parser().parse_args(args))["duration_ms"] == body["body_forward_ms"]
    control = run(parser().parse_args([*args, "--no-body-defaults"]))
    assert all(row["program"] == "TiledFlashAndPagedAttentionRecipe" for row in control["timeline"] if row["family"] == "attention")
def test_hf_overrides_preserve_layers_and_unknown_semantics_in_rankings():
    raw = run_case(config(custom_module={"order": 1})); original = deepcopy(raw)
    result = study([
        {"id": "base", "case": raw},
        {"id": "memory", "case": raw, "overrides": {"machine": {"device": {"rates": {"memory": 2e9}}}}},
        {"id": "other-module", "case": raw, "overrides": {"model": {"custom_module": {"order": 2}}}},
        {"id": "deeper", "case": raw, "overrides": {"model": {"num_hidden_layers": 3}}},
    ], [STUDY_PROBES[0]], level="graph", serving=True)
    assert result["status"] == "completed" and raw == original
    base, faster, custom, deeper = result["evaluations"]
    assert base["operators"]["work"] == faster["operators"]["work"] and base["workload_key"] == faster["workload_key"]
    assert faster["duration_ms"] < base["duration_ms"] and custom["workload_key"] != base["workload_key"]
    assert deeper["operators"]["work"]["flops"] > base["operators"]["work"]["flops"] and len(result["rankings"]) == 3
    assert result["coverage_summary"] == {"complete_candidates": 0, "partial_candidates": 4}
    for candidate, evaluation in zip(result["candidates"], result["evaluations"]):
        for scope in (candidate, evaluation, candidate["serving"]):
            assert scope["model_coverage"] == "partial" and "unrecognized_config_field:custom_module" in scope["unmodeled_features"]
def test_disaggregated_prediction_keeps_partial_model_scope():
    raw = pd_case(); raw["model"] = config(custom_module={"order": 1}); result = predict(raw)
    assert result["status"] == "modeled" and result["requests"][0]["status"] == "served"
    assert result["model_coverage"] == "partial" and "unrecognized_config_field:custom_module" in result["unmodeled_features"]
    assert result["execution"]["live_tasks"] == 0
@pytest.mark.parametrize("wrapper", [None, "hf_config", "text_config"])
def test_hf_layer_alias_override_changes_real_work_through_wrappers(wrapper):
    model = config()
    if wrapper: model = {wrapper: model}
    result = study([{"id": "base", "case": run_case(model)}, {"id": "deeper", "case": run_case(model), "overrides": {"model": {"num_layers": 3}}}], [STUDY_PROBES[0]])
    base, deeper = result["evaluations"]
    assert result["status"] == "completed" and deeper["operators"]["work"]["flops"] > base["operators"]["work"]["flops"]
    assert deeper["workload_key"] != base["workload_key"]
def test_native_hybrid_adapter_retains_custom_fields_and_unknown_quant_proxy():
    raw = json.loads((CONFIG_DIR / "qwen3.5-35b-a3b.json").read_text()); text = raw.get("text_config", raw)
    text.update(custom_module={"order": 1}, attention_bias=True, quantization_config={"quant_method": "future-packing"})
    model = ModelSpec.from_mapping(raw); metadata = dict(model.metadata)
    assert model.layer_plans and model.quant.weight_format == "bf16"
    assert metadata["unrecognized_config_fields"]["custom_module"] == {"order": 1}
    assert any(gap.startswith("projection_bias_not_in_native_layer_plan") for gap in metadata["unmodeled_features"])
    assert any("future-packing" in gap for gap in metadata["degraded_submodules"]) and ModelSpec.from_mapping(model.to_dict()) == model
def test_dspark_aliases_reach_weight_and_cache_ledgers():
    for name in ("deepseek-v4-pro-0813", "deepseek-v4-10t", "deepseek-v4.1-flash"):
        model = load_model(name); draft = optional_draft_spec(model, "dspark"); assert draft is model.auxiliary.mtp and draft.kind == "dspark"
        parallel = ParallelSpec(tp=8); main = stage_weight_terms(model, parallel); full = stage_weight_terms(model, parallel, include_draft=True, draft_kind="dspark")
        assert sum(term.logical_parameters for stage in full for term in stage) > sum(term.logical_parameters for stage in main for term in stage)
        assert stage_cache_components(model, parallel, 1024, draft_kind="dspark")[-1]["draft_kv"] > 0
def test_model_scope_keeps_service_matrix_defaults_and_explicit_choices():
    raw = case()
    for explicit in ({}, {"programs": {"matmul": {"kind": "tiled_matmul"}}}, {"recipes": {"router": {"kind": "tiled_matmul"}}, "families": {"grouped_gemm": {"kind": "steady_matmul"}}}):
        raw["runtime"]["kernels"] = deepcopy(explicit); parsed = parse_case(raw); original = deepcopy(parsed.runtime.kernels)
        effective, _ = effective_body_declaration(parsed, "decode", tokens=6, scope="model"); service = service_program_defaults(parsed)
        assert effective.runtime.kernels["programs"] == service.runtime.kernels["programs"]
        for section, entries in explicit.items():
            for key, value in entries.items(): assert effective.runtime.kernels[section][key] == value
        assert parsed.runtime.kernels == original
        disabled, _ = effective_body_declaration(parsed, "decode", scope="model", enabled=False); assert disabled is parsed
def test_stage_memoization_keeps_engine_duration_bit_exact():
    parsed, attention = _mq_case(); from mindie_sim.prediction.operators.paged_mla import mqsmla
    cached = operator_costs(parsed, (attention,))["operators"][0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mqsmla, "lru_cache", lambda **_: lambda function: function); uncached = operator_costs(parsed, (attention,))["operators"][0]
    assert cached["duration_ms"] == uncached["duration_ms"] and cached["serviced"] == uncached["serviced"]
    assert cached["live_tasks"] == uncached["live_tasks"] == 0
def test_fused_draft_identity_preserves_offsets_and_namespaces():
    _, base = _mq_case()
    def scoped(stage, layer, draft="draft0.", offset=-1):
        origins = ({"stage": stage, "layer": layer + offset, "prefix": f"{draft}s{stage}.l{layer + offset}", "op_id": f"{draft}s{stage}.l{layer + offset}.norm"},)
        return replace(base, stage=stage, layer=layer, metadata=(("fused_origin_scopes", origins),))
    assert physical_key(scoped(1, 2)) == physical_key(scoped(3, 9))
    assert physical_key(scoped(1, 2)) != physical_key(scoped(1, 2, offset=0))
    assert physical_key(scoped(1, 2)) != physical_key(scoped(1, 2, draft="draft1."))
    assert physical_key(scoped(1, 2)) != physical_key(scoped(1, 2, draft=""))
    assert physical_key(scoped(1, 2, draft="draft_prefill.")) == physical_key(scoped(3, 9, draft="draft_prefill."))
def test_draft_cache_uses_bf16_rows_and_context_block_ownership():
    parsed = _kimi_case(); draft = optional_draft_spec(parsed.model, "dspark"); assert cache_format(parsed.model, "mla_kv").row_bytes(576) == 656
    for tokens in (0, 1, 1536 * 8, 1536 * 8 + 1, 131588):
        for padded in (False, True):
            parts = stage_cache_components(parsed.model, parsed.runtime.parallel, tokens, draft_kind="dspark", capacity_padded=padded, interleave=1536, runtime=parsed.runtime)
            rows = ((tokens + 1536 * 8 - 1) // (1536 * 8) * 1536 if padded else tokens / 8)
            assert parts[-1]["draft_kv"] == rows * 1152 * draft.layers
def test_gathered_draft_prefix_reuse_preserves_compute_and_outputs():
    parsed, batches = _kimi_case(), []
    for mode in ("per_sequence", "batch_shared"):
        ops = [item for region in _draft_regions(parsed, read_mode=mode) for item in region.ops]
        batches.append([item for item in ops if item.op_id.endswith(".latent_attention_batch")])
    private, shared = batches; assert len(private) == len(shared) == 5
    for old, new in zip(private, shared):
        assert new.hbm_bytes < old.hbm_bytes and new.flops == old.flops and new.mte3_bytes == old.mte3_bytes
        assert dict(new.metadata)["prefix_read_reuse_applied"] and new.recipe == "batch_prefix_attention" and "cp_per_rank_pairs" in dict(new.metadata)
def test_real_dspark_graph_exports_decode_and_draft_subgraphs():
    parsed = parse_case(compose_case("deepseek-v4-pro-0813", "a5", tp=8, ep=8, max_model_len=140000))
    ops = tuple(item for region in _draft_regions(parsed) for item in region.ops); helper = next(item for item in ops if item.recipe == "hc_pre_mix")
    ops = tuple(replace(item, phase="decode") if item is helper else item for item in ops); assert {item.phase for item in ops} == {"decode", "draft"}
    report = export_op_structure(ops, model={}, execution={"phase": "decode", "scope": "model"}, kernels=parsed.runtime.kernels)
    assert report["nodes"] and report["subgraphs"]
    with pytest.raises(ValueError, match="structure_phase_mismatch"): export_op_structure(ops, model={}, execution={"phase": "prefill", "scope": "model"})


def test_situ_eager_chain_activation_params_and_router_contract_unshadowed():
    from types import SimpleNamespace
    from mindie_sim.observations.communication.calibration import evaluate_a5_mc2_chains
    from mindie_sim.observed_contracts.matrix import observed_router_contract
    from mindie_sim.schema.common import ModelingError

    eager_case = parse_case(compose_case("kimi-k3", "a5", tp=2, dp=1, ep=2, layers=2, raw_eager=True))
    ops = target_graphs(eager_case, phase="decode", tokens=1, context=128, sequences=1, scope="body")[0]
    situ_casts = [op for op in ops if op.recipe == "situ_cast_fp32"]
    assert situ_casts, "raw_eager SiTU FFN must lower via situ_eager_chain"
    routed_cast = next(op for op in situ_casts if "expert_act.cast_in" in op.op_id)
    assert "memory_accesses" in dict(routed_cast.metadata)
    situ_outs = [op for op in ops if op.recipe == "situ_cast_out"]
    assert situ_outs and all("activation_params" in dict(op.metadata) for op in situ_outs)
    assert dict(situ_outs[0].metadata)["activation_params"]["beta"] == pytest.approx(4.0)

    router_row = SimpleNamespace(
        kind="MoeGatingTopK",
        input_shapes=((8, 64), ()), input_dtypes=("bf16", None), input_formats=("ND", None),
        output_shapes=((8, 4), (8, 4), (8, 64)), output_dtypes=("bf16", "int32", "fp32"),
        output_formats=("ND", "ND", "ND"),
    )
    contract = observed_router_contract(router_row)
    assert contract is not None and contract[0] == "router_selection_active" and contract[1] == "MoeGatingTopK"

    with pytest.raises(ModelingError, match="a5_mc2_chain_nonpositive_sum"):
        evaluate_a5_mc2_chains({
            "schema": "a5-campaign-ingestion/1",
            "mc2_chains": [{"condition": "bad", "dtype": "bf16", "chain_p50_ms": 1.0,
                            "sum_leg_p50_ms": 0.0, "legs_ms": {}}],
        })


def test_feedforward_and_collective_lowering_enforce_mutex_and_architecture_guards():
    from _support.axes import REPRESENTATIVE_SKUS

    base = compose_case(
        "deepseek-v3.2", REPRESENTATIVE_SKUS["a5"],
        tp=1, dp=8, ep=8, layers=4, feature_track="shipping",
    )
    # Shipping Before track activates MLAPO native fusion on eligible MLA layers.
    merged_ops = target_graphs(parse_case(base), phase="decode", tokens=1, context=128, sequences=1, scope="body")[0]
    assert any(
        item.op_id.endswith(".mla_prolog.native")
        and dict(item.metadata).get("fusion_role") == "mla_prolog"
        for item in merged_ops
    )

    # Raw uncomposed runtime with BOTH enable_fused_mc2 > 0 and MSO enabled:
    # feedforward.alltoall must enforce mutex BEFORE assigning stream="shared_stream".
    conflicting = deepcopy(base)
    conflicting["runtime"]["multistream_overlap_shared_expert"] = True
    conflicting["runtime"]["metadata"]["additional_config"] = {"enable_fused_mc2": 2}
    ops_fused = target_graphs(parse_case(conflicting), phase="decode", tokens=1, context=128, sequences=1, scope="body")[0]
    assert not any(item.stream == "shared_stream" for item in ops_fused)

    # Disabling fused MC2 (enable_fused_mc2=0) preserves MSO shared_stream overlap.
    mso_only = deepcopy(base)
    mso_only["runtime"]["multistream_overlap_shared_expert"] = True
    mso_only["runtime"]["metadata"]["additional_config"] = {"enable_fused_mc2": 0}
    ops_mso = target_graphs(parse_case(mso_only), phase="decode", tokens=1, context=128, sequences=1, scope="body")[0]
    assert any(item.stream == "shared_stream" for item in ops_mso)

    # Explicit hardware CCU collective bindings are rejected on non-CCU generations and accepted on CCU generation.
    ccu_binding = {"allreduce": {"expansion": "CCU_MS", "template": "mesh_oneshot_v1"}}
    tiny_model = {
        "model_type": "qwen3", "hidden_size": 512, "intermediate_size": 1024,
        "num_hidden_layers": 1, "num_attention_heads": 8, "num_key_value_heads": 4,
        "head_dim": 64, "vocab_size": 1024, "max_position_embeddings": 2048,
    }
    for family in ("a2", "a3"):
        bad = compose_case(tiny_model, REPRESENTATIVE_SKUS[family], tp=2, layers=1, raw_eager=True)
        bad["runtime"]["kernels"] = {"hccl": ccu_binding}
        with pytest.raises(ModelingError, match="hccl_ccu_unsupported_on_arch"):
            target_graphs(parse_case(bad), phase="decode", tokens=1, context=64, sequences=1, scope="body")
    good = compose_case(tiny_model, REPRESENTATIVE_SKUS["a5"], tp=2, layers=1, raw_eager=True)
    good["runtime"]["kernels"] = {"hccl": ccu_binding}
    assert target_graphs(parse_case(good), phase="decode", tokens=1, context=64, sequences=1, scope="body")[0]
