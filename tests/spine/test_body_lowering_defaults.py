"""Body lowering defaults: attention, cohort, compact output, events and matrices.

[Role] Prove that ordinary body execution installs one effective declaration:
  attention/GDN/matrix defaults, cohort geometry, compact emission and the
  event schedule all share the resolved physical plan.
[Invariants] Default installation preserves explicit user choices and exact
  work; compact output is an emission reduction of the same finite graph.
[Timeouts & Locks] Tier-0 in-memory simulation over small one/two-rank graphs.
[Side-Effect Policy] Read-only case loads and in-memory execution; no devices.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest
from _support.memory_cases import case
from _support.moe_hc_mc2_fixtures import balanced_case, grouped_case
from mindie_sim import graph_costs, operator_costs, predict, predict_pd, target_graphs, target_operators
from mindie_sim.__main__ import parser, run
from mindie_sim.compose import compose_case
from mindie_sim.prediction.frontend.body_spec import body_forward
from mindie_sim.prediction.body_defaults import attention_defaults
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operators.matrix.matrix import TiledMatmul
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_structure

from _support.axes import BODY_DEFAULT_SKUS as SKUS, GEN5_DT_SKU, KDA_ONLY_SKUS, KDA_SKUS, WIDE_SKUS  # noqa: E402

pytestmark = pytest.mark.tier0
RESIDUAL_RECIPES = ("residual_values_fp32", "attention_residual_scores", "attention_residual_softmax", "attention_residual_mix", "attention_residual_output_cast")
MATRIX_PROGRAMS = (("tiled_matmul", "TiledMatmul"), ("mapped_matmul", "MappedMatmul"))
MATRIX_FAMILIES = {"qkv_projection", "output_projection", "gate_up_projection", "down_projection"}
IDENTITY_KEYS = ("family", "layer", "stream", "phase", "program", "recipe", "source_boundary")
GEOMETRY = dict(scope="body", phase="decode", tokens=4, context=128, sequences=2)
def _gdn_case(sku):
    return {"machine": {"backend": "ascend", "sku": sku, "num_devices": 8}, "model": {"preset": "qwen3.6-27b", "quantization": "w8a8", "layers": 1}, "runtime": {"tp": 8}, "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]}}
def _op(name, *, flops=0, memory=0, deps=(), stream="main", stage=0):
    return OpSpec(name, "test", "prefill", stage, 0, (), (1,), (1,), (1,), (1,), flops=flops, hbm_bytes=memory, deps=deps, dtype="bf16", stream=stream)
def _engine():
    return Engine([Resource("compute", 10.0, units=2), Resource("queue", 1.0, policy="fifo")], pools={"hbm": 100.0}, pool_units={"hbm": 1})
def _physical_case():
    raw = case(output=1); raw["model"].update(layers=1, hidden_size=256, head_dim=64, intermediate_size=512)
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}; return raw
def _matrix_rows(report):
    return [row for row in report["timeline"] if row["family"] in MATRIX_FAMILIES]
def _attention_rows(report):
    return [row for row in report["timeline"] if row["family"] == "attention"]
def _cohort_case(*, tp=1, dp=8, ep=8):
    raw = balanced_case(tp=tp, dp=dp, ep=ep, experts=48, tokens=1); raw["model"]["top_k"] = 6
    raw["runtime"].update(graph_mode="auto", graph_capture_mode="full", graph_capture_keys=[1, 4, 8]); return raw
def _routing(ops):
    return next(dict(op.metadata)["routing_realization"] for op in ops if "routing_realization" in dict(op.metadata))
def _wire(report):
    return sum(value for key, value in report["execution"]["serviced"].items() if key.startswith("link:"))
def _groups(full):
    grouped = {}
    for row in full["timeline"]: grouped.setdefault(f"s{row['stage']}:{row['op_id']}", {}).setdefault(row["replica"], []).append(row)
    return grouped
def _assert_compact_equals_full(full, compact):
    assert compact["result_mode"] == "compact" and "timeline" not in compact and "events" not in compact
    assert compact["timeline_omitted"] is True and compact["events_omitted"] is True
    assert "by_window" not in compact["profile"] and "window_ms" not in compact["profile"]
    for key in ("duration_ms", "complete_ms", "input_ready_ms", "operator_occurrences", "replicas", "replay_scope", "replay_scope_by_replica", "resources", "execution", "parameters"):
        assert compact[key] == full[key], key
    expected = _groups(full); assert set(compact["profile"]["by_operator"]) == set(expected)
    for identity, by_rank in expected.items():
        item = compact["profile"]["by_operator"][identity]; rows = [row for values in by_rank.values() for row in values]
        assert item["operator_events"] == len(rows)
        assert item["ready_ms"] == min(row["ready_ms"] for row in rows) and item["complete_ms"] == max(row["complete_ms"] for row in rows)
        assert item["elapsed_ms"] == max(row["elapsed_ms"] for row in rows)
        assert item["elapsed_sum_ms"] == pytest.approx(sum(row["elapsed_ms"] for row in rows), rel=1e-12, abs=1e-15)
        assert set(item["replicas"]) == set(by_rank)
        for rank, rank_rows in by_rank.items():
            actual = item["replicas"][rank]
            assert actual["operator_events"] == len(rank_rows)
            assert actual["ready_ms"] == min(row["ready_ms"] for row in rank_rows) and actual["complete_ms"] == max(row["complete_ms"] for row in rank_rows)
            assert actual["elapsed_ms"] == pytest.approx(sum(row["elapsed_ms"] for row in rank_rows), rel=1e-12, abs=1e-15)
            assert tuple(actual[key] for key in IDENTITY_KEYS) == tuple(rank_rows[0][key] for key in IDENTITY_KEYS)
            assert all((row["layer"], row["stream"]) == (actual["layer"], actual["stream"]) for row in rank_rows)
    for family, family_rows in {family: [row for row in full["timeline"] if row["family"] == family] for family in {row["family"] for row in full["timeline"]}}.items():
        summary = compact["profile"]["by_family"][family]
        assert summary["operator_events"] == len(family_rows)
        assert summary["elapsed_ms"] == pytest.approx(sum(row["elapsed_ms"] for row in family_rows), rel=1e-12, abs=1e-15)
        for rank in {row["replica"] for row in family_rows}:
            rank_rows = [row for row in family_rows if row["replica"] == rank]
            assert summary["replicas"][rank]["operator_events"] == len(rank_rows)
            assert summary["replicas"][rank]["elapsed_ms"] == pytest.approx(sum(row["elapsed_ms"] for row in rank_rows), rel=1e-12, abs=1e-15)
    assert compact["event_profile"]["completed_physical_events"] == full["execution"]["completed_tasks"]
    assert compact["event_profile"]["trace_retained"] == len(full["events"])
    assert compact["event_profile"]["trace_dropped"] == full["execution"]["trace_dropped"]
    assert compact["event_profile"]["trace_events"] == len(full["events"]) + full["execution"]["trace_dropped"]
@pytest.mark.parametrize("sku", SKUS)
def test_gqa_body_default_executes_private_pipes_and_preserves_explicit_recipe(sku):
    raw = case(output=1); raw["machine"] = {"backend": "ascend", "sku": sku, "num_devices": 1}
    report = predict(raw, scope="body", tokens=4, context=128, sequences=2); attention = _attention_rows(report)
    assert attention and all(row["program"] == "StagedGQADecode" for row in attention)
    assert report["execution"]["serviced"]["mte1@r0:s0"] > 0
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    explicit = deepcopy(raw); explicit["runtime"]["kernels"] = {"recipes": {"flash_attention": dict(report["default_mechanisms"]["gqa_tiles"])}}
    prior = predict(explicit, scope="body", tokens=4, context=128, sequences=2)
    assert all(row["program"] == "StagedGQADecode" for row in prior["timeline"] if row["family"] == "attention")
    for key in ("flops", "hbm_bytes", "link_bytes"): assert prior["execution"]["work"][key] == report["execution"]["work"][key]
    assert report["execution"]["work"]["vector_ops"] >= prior["execution"]["work"]["vector_ops"]
    explicit["runtime"]["kernels"]["recipes"] = {"flash_attention_decode": {"kind": "staged_gqa_decode", "kv_splits": 2}}
    split = predict(explicit, scope="body", tokens=4, context=128, sequences=2)
    assert split["default_mechanisms"]["selections"][0]["reason"] == "explicit_implementation_preserved" and split["body_forward_ms"] > 0
@pytest.mark.parametrize("sku", SKUS)
def test_indexed_mla_defaults_keep_per_sequence_queries_and_native_mte1_work(sku):
    raw = {"machine": {"backend": "ascend", "sku": sku, "num_devices": 8}, "model": {"preset": "glm-5.3", "quantization": "w8a8", "layers": 1}, "runtime": {"tp": 8}, "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]}}
    single = predict(raw, scope="body", context=2048); batch = predict(raw, scope="body", context=2048, sequences=2)
    a = [row for row in single["timeline"] if row["family"] == "attention"]; b = _attention_rows(batch)
    assert len(a) == len(b) == 1 and b[0]["recipe"] == "latent_attention_batch"
    assert batch["geometry"]["valid_query_tokens"] == 2 and batch["execution"]["serviced"]["mte1@r0:s0"] > 0
    assert batch["default_mechanisms"]["selections"][0]["selection"] == "default_installed"
    assert batch["execution"]["live_tasks"] == batch["execution"]["live_programs"] == 0
def test_unavailable_pipe_domain_is_visible_and_keeps_analytic_execution():
    report = predict(case(output=1), scope="body", context=64)
    assert report["default_mechanisms"]["selections"][0]["reason"] == "independent_copy_pipes_not_declared"
    assert report["body_forward_ms"] > 0
@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
def test_wide_head_defaults_fit_resident_and_overlapping_tile_buffers(dtype, sku):
    raw = case(output=1); raw["machine"] = {"backend": "ascend", "sku": sku, "num_devices": 1}
    raw["model"].update(layers=1, hidden_size=2048, head_dim=512, activation_dtype=dtype, kv_dtype=dtype)
    report = predict(raw, scope="body", tokens=4, context=256, sequences=2); rows = _attention_rows(report)
    assert rows and all(row["program"] == "StagedGQADecode" for row in rows)
    assert report["default_mechanisms"]["gqa_tiles"]["key_tile"] < 128
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0 and report["body_forward_ms"] > 0
    raw["runtime"]["kernels"] = {"recipes": {"flash_attention": dict(report["default_mechanisms"]["gqa_tiles"])}}
    control = predict(raw, scope="body", tokens=4, context=256, sequences=2)
    for key in ("flops", "hbm_bytes", "link_bytes"): assert control["execution"]["work"][key] == report["execution"]["work"][key]
    assert report["execution"]["work"]["vector_ops"] >= control["execution"]["work"]["vector_ops"]
@pytest.mark.parametrize("kv_dtype", ("bf16", "int8"))
@pytest.mark.parametrize("sku", SKUS)
def test_rope_free_indexed_body_defaults_execute_declared_cache_representation(sku, kv_dtype):
    from _support.captures import REPO_ROOT
    raw = json.loads((REPO_ROOT / "examples/source-families/packed-int8-indexed-c32.json").read_text())
    raw["model"]["layers"] = 4; raw["model"]["quant"]["kv_dtype"] = kv_dtype; raw["machine"]["sku"] = sku
    if sku == "Ascend950DT_9582": raw["machine"]["ranks_per_node"] = 8
    raw["runtime"]["kernels"]["programs"].pop("indexed_attention")
    result = predict(raw, scope="body", tokens=4, context=128, sequences=3)
    assert result["execution"]["serviced"]["mte2_aiv@r0:s0"] > 0
    selected = next(row for row in result["default_mechanisms"]["selections"] if row["family"] == "indexed_mla")
    effective, _ = attention_defaults(parse_case(raw), "decode", tokens=4)
    if sku == GEN5_DT_SKU and kv_dtype == "int8":
        assert selected["selection"] == "preserved"
        assert selected["reason"] == "packed_int8_indexed_attention_arch35_requires_rope64"
        assert "indexed_attention" not in effective.runtime.kernels.get("programs", {})
    else:
        assert selected["selection"] == "default_installed"
        assert effective.runtime.kernels["programs"]["indexed_attention"]["kind"] == (
            "indexed_attention_packed_int8" if kv_dtype == "int8" else "indexed_attention_pipeline"
        )
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
@pytest.mark.parametrize("model", ("bailing-v3", "deepseek-v2-lite", "longcat-flash-lite", "kimi-k3"))
@pytest.mark.parametrize("sku", WIDE_SKUS)
def test_dense_mla_body_default_respects_kernel_head_domain(model, sku):
    parsed = parse_case(compose_case(model, hardware=sku, tp=8, dp=1, num_devices=8, layers=4))
    report = operator_costs(parsed, target_graphs(parsed, phase="decode", tokens=1, context=4096, sequences=1, scope="body"))
    assert report["operator_sum_ms"] > 0
@pytest.mark.parametrize("sku,expected_kda", KDA_SKUS)
def test_kda_body_defaults_install_scan_and_attention_residual(sku, expected_kda):
    raw = compose_case("kimi-k3", hardware=sku, tp=8, dp=1, ep=8, num_devices=8, quant="w4a8", layers=8)
    for name in ("kda", "attention_residual"): raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {}).pop(name, None)
    effective, report = attention_defaults(parse_case(raw), "decode", tokens=1); lowerings = effective.runtime.kernels.get("lowerings", {}); kda_entry = lowerings.get("kda")
    assert (kda_entry["implementation"] if isinstance(kda_entry, dict) else kda_entry) == expected_kda
    by_fam = {row["family"]: row for row in report["selections"]}
    assert by_fam["kda_decode"]["selection"] == "default_installed"
    if sku in {"Ascend950DT_9582", "Ascend910_9392"}:
        assert lowerings.get("attention_residual") == "vllm_ascend_attn_res_prefix" and by_fam["attention_residual"]["selection"] == "default_installed"
    else:
        assert "attention_residual" not in lowerings and by_fam["attention_residual"]["selection"] == "preserved"
@pytest.mark.parametrize("recipe", RESIDUAL_RECIPES)
def test_attention_residual_default_preserves_explicit_member_programs(recipe):
    raw = compose_case("kimi-k3", hardware="a3", tp=8, ep=8, layers=4)
    raw["runtime"]["kernels"] = {"recipes": {recipe: {"kind": "tiled_pointwise", "input_bytes_per_element": 2, "outputs": []}}}
    ops = target_operators(raw, phase="decode", tokens=1, context=4)
    assert any(op.recipe == recipe for op in ops) and not any(op.recipe == "attn_res_fwd" for op in ops)
    decision, = [row for row in ops.default_selection["selections"] if row["family"] == "attention_residual"]
    assert decision["reason"] == "explicit_implementation_preserved"
@pytest.mark.parametrize("sku", KDA_ONLY_SKUS)
def test_explicit_kda_scan_family_keeps_the_requested_execution(sku):
    raw = compose_case("kimi-k3", hardware=sku, tp=8, dp=1, ep=1, num_devices=8, quant="w4a8", layers=1)
    raw["runtime"]["kernels"] = {"families": {"recurrent_scan": {"kind": "recurrent_scan"}}}
    ops = target_operators(raw, phase="decode", tokens=1, context=128)
    assert next(row for row in ops.default_selection["selections"] if row["family"] == "kda_decode")["reason"] == "explicit_implementation_preserved"
    scans = [row for row in operator_costs(raw, ops)["operators"] if row["operator"]["family"] == "recurrent_scan"]
    assert scans and all(row["program"] == "RecurrentScan" for row in scans)
    assert all(row["duration_ms"] > 0 and not row["live_tasks"] for row in scans)
def test_dp_query_counts_preserve_expert_rows_and_idle_collective_participants():
    raw = _cohort_case()
    graphs = target_graphs(raw, phase="decode", tokens=1, context=32, sequences=[1, 0, 0, 0, 0, 0, 0, 0], scope="body")
    ledger = [_routing(ops) for ops in graphs.values()]
    assert [sum(item["local_expert_rows"]) for item in ledger] == [1] * 6 + [0, 0]
    assert sum(sum(item["local_expert_rows"]) for item in ledger) == 6 and [item["representative_rank"] for item in ledger] == list(range(8))
    for rank, ops in graphs.items():
        exchanges = [op for op in ops if op.family == "alltoall"]
        assert len(exchanges) == 2
        if rank: assert not any(op.family in ("attention", "norm", "router") for op in ops) and ledger[rank]["padded_local_tokens"] == 0
        if rank >= 6: assert not any(op.family == "grouped_gemm" for op in ops)
    report = graph_costs(raw, graphs, input_ready_ms=7, external_inputs={"analysis_input": 7}, trace_limit=100000)
    assert report["replay_scope_by_replica"] == {0: "full", **{r: "none" for r in range(1, 8)}} and report["replay_scope"] == "mixed"
    assert all(row["ready_ms"] >= 7 for row in report["timeline"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    # Uniform-peer transport remains a separate declared expectation; no per-destination integer router realization is invented.
    assert _wire(report) == pytest.approx(2 * 6 * 32 * 2 * 7 / 8)
    result = operator_costs(raw, graphs)
    assert result["replicas"] == list(range(8)) and all(not row["service_before_input"] for row in result["operators"])
    scalar = predict(raw, scope="body", context=32, sequences=1); listed = predict(raw, scope="body", context=32, sequences=[1] * 8)
    assert scalar["body_forward_ms"] == listed["body_forward_ms"] and scalar["execution"]["serviced"] == listed["execution"]["serviced"]
    assert listed["geometry"]["valid_query_tokens"] == 8
    counts = [2, 0, 1, 0, 0, 0, 0, 0]; padded = predict(raw, scope="body", context=32, sequences=counts)
    assert list(padded["geometry"]["padded_query_tokens_by_replica"].values()) == [4, 0, 1, 0, 0, 0, 0, 0]
    assert padded["geometry"]["valid_query_tokens"] == 3
    counts_graphs = target_graphs(raw, phase="decode", tokens=1, context=32, sequences=counts, scope="body")
    assert sum(sum(_routing(ops)["local_expert_rows"]) for ops in counts_graphs.values()) == 5 * 6
def test_tp_representative_is_local_max_not_global_rank_zero():
    raw = _cohort_case(tp=4, dp=4, ep=16)
    graphs = target_graphs(raw, phase="decode", tokens=1, sequences=[1, 0, 0, 0], scope="body")
    values = [_routing(ops) for ops in graphs.values()]
    assert [v["representative_rank"] for v in values] == [0, 4, 8, 12]
    assert [sum(v["local_expert_rows"]) for v in values] == [1, 1, 0, 0]
    assert [v["represented_ep_ranks"] for v in values] == [tuple(range(i, i + 4)) for i in (0, 4, 8, 12)]
    assert all(v["rank_timing_assumption"] == "heaviest_local_tp_shard_upper_bound" for v in values)
    assert sum(values[0]["rank_assignment_rows"]) == 6
    # Multiplying representative work by TP is an upper bound, not an exact shard sum; the global histogram keeps the exact six.
    assert sum(sum(v["local_expert_rows"]) * 4 for v in values) == 8
def test_unequal_source_rows_preserve_reverse_peer_matrix():
    from mindie_sim.prediction.collectives.peer_traffic import PeerTraffic
    raw = _cohort_case(dp=4, ep=4); raw["runtime"]["graph_mode"] = "eager"
    graphs = target_graphs(raw, phase="decode", tokens=1, context=16, sequences=[1, 0, 3, 2], scope="body")
    forward = next(op for op in graphs[0] if op.recipe == "expert_dispatch"); backward = next(op for op in graphs[0] if op.recipe == "expert_combine")
    f = PeerTraffic.from_mapping(dict(forward.metadata)["peer_traffic"], 4); b = PeerTraffic.from_mapping(dict(backward.metadata)["peer_traffic"], 4)
    assert all(b.payload(i, j) == f.payload(j, i) for i in range(4) for j in range(4))
    for rank, ops in graphs.items(): assert next(op for op in ops if op.recipe == "expert_combine").link_bytes == b.row_total(rank)
    report = graph_costs(raw, graphs, external_inputs={"analysis_input": 2}, input_ready_ms=2)
    assert _wire(report) == pytest.approx(2 * 6 * 6 * 32 * 2 * 3 / 4)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
def test_pd_body_retains_full_cohort_counts_in_both_pools():
    raw = _cohort_case(); machine, runtime = raw.pop("machine"), raw.pop("runtime")
    raw["pools"] = {name: {"machine": deepcopy(machine), "runtime": deepcopy(runtime)} for name in ("prefill", "decode")}
    raw["connector"] = {"bytes_per_s": 1e9}
    result = predict_pd(raw, scope="body", sequences=[1] + [0] * 7, context=16)
    for pool in result["pools"].values():
        assert pool["geometry"]["valid_query_tokens"] == 1 and pool["geometry"]["padded_query_tokens"] == 1
        assert pool["execution"]["live_tasks"] == pool["execution"]["live_programs"] == 0
def test_serving_single_request_uses_actual_dp_expert_shards():
    raw = _cohort_case(); raw["runtime"]["graph_mode"] = "eager"; raw["workload"]["requests"][0].update(prompt_tokens=1, output_tokens=1)
    result = predict(raw); graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=1, sequences=[1] + [0] * 7, scope="model"))
    # Serving additionally samples/commits; expert matrix and wire service keep the same physical cohort, including zero-query replicas.
    grouped_matrix = lambda out: sum(v for k, v in out["execution"]["serviced"].items() if k.startswith("matrix@"))
    assert grouped_matrix(result) == pytest.approx(grouped_matrix(graph)) and _wire(result) == pytest.approx(_wire(graph))
    assert result["execution"]["live_tasks"] == 0
def test_model_scope_moe_mtp_preserves_replica_rows_and_target_completion():
    from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
    from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.workload import IterationPlan, RequestView
    parsed = parse_case(_cohort_case())
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=32, intermediate_size=64, num_heads=4, num_kv_heads=2, head_dim=8, q_width=32, kv_width=16, vocab_size=128, num_experts=48, top_k=6, expert_intermediate_size=64, target_shared_embedding=True, target_shared_lm_head=True)
    parsed = replace(parsed, model=replace(parsed.model, auxiliary=AuxiliarySpec(mtp=replace(draft, layer_plans=native_draft_layer_plans(parsed.model, draft)))), runtime=replace(parsed.runtime, metadata=(("draft_mode", "mtp"),)))
    graphs = target_graphs(parsed, phase="prefill", tokens=1, sequences=[1] + [0] * 7, scope="model", draft_prepare=True); totals = {"target": 0, "draft": 0}
    for rank, ops in graphs.items():
        for op in ops:
            if op.recipe == "balanced_grouped_gate_up":
                md = dict(op.metadata)["routing_realization"]; assert md["representative_rank"] == rank
                totals["draft" if op.op_id.startswith("draft") else "target"] += sum(md["local_expert_rows"])
                assert op.phase == ("draft_prefill" if op.op_id.startswith("draft") else "prefill")
        assert not any(op.family in ("sampling", "token_commit") for op in ops) and any(op.family == "logits" for op in ops) == (rank == 0)
    assert totals == {"target": 6, "draft": 6}
    report = graph_costs(parsed, graphs, input_ready_ms=3, isolated=False)
    target_head = next(row for row in report["timeline"] if row["replica"] == 0 and row["family"] == "logits")
    assert next(row for row in report["timeline"] if row["replica"] == 0 and row["recipe"] == "draft_features")["ready_ms"] >= target_head["complete_ms"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    # Ascend V2 MTP step 0 reuses target verification shape (1 + K = 3 -> 18 assignments); step 1 uses 1 token (6).
    view = RequestView("verify", 1, 32, True, "decode", draft_proposal_tokens=2); assignments = {}
    for rank in range(8):
        work = Frontend()(IterationPlan((view,) if rank == 0 else ()), parsed.model, parsed.runtime, parsed.machine, global_views=(view,), global_tokens=3, replica=rank)
        for op in work.ops:
            if op.recipe == "balanced_grouped_gate_up":
                key = op.op_id.split(".")[0]; assignments[key] = assignments.get(key, 0) + sum(dict(op.metadata)["routing_realization"]["local_expert_rows"])
    assert assignments == {"draft0": 18, "draft1": 6, "s0": 18}
def test_idle_expert_program_uses_actual_mixed_cohort_phase():
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.workload import IterationPlan, RequestView
    parsed = parse_case(_cohort_case(dp=2, ep=2))
    views = (RequestView("prefill", 1, 0, True, "prefill"), RequestView("decode", 1, 32, True, "decode"))
    idle = Frontend()(IterationPlan(()), parsed.model, parsed.runtime, parsed.machine, global_views=views, global_tokens=2, replica=1)
    grouped = [op for op in idle.ops if op.family == "grouped_gemm"]
    assert grouped and all(op.phase == "mixed" for op in grouped)
    assert all(dict(op.metadata)["routing_realization"]["representative_rank"] == 1 for op in grouped)
    assert not any(op.family == "logits" for op in idle.ops)
@pytest.mark.parametrize("raw", [pytest.param(compose_case("glm-5.1", "a3", tp=1, dp=1, ep=1, pcp=1, layers=1, prompt_tokens=8, output_tokens=1, max_num_batched_tokens=16, max_num_seqs=2, max_model_len=256), id="body"), pytest.param("fixture", id="checked-in")])
def test_compact_equals_full_per_family_operator_rank_and_source_window(raw):
    from _support.captures import REPO_ROOT
    if raw == "fixture":
        raw = json.loads((REPO_ROOT / "examples/cases/glm-5.3.json").read_text()); raw["model"] = {**raw["model"], "layers": 2, "quantization": "w8a8"}
    graphs = target_graphs(raw, phase="decode", tokens=8, context=0, sequences=1, scope="body")
    full = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False, trace_limit=4096)
    compact = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False, trace_limit=4096, compact=True)
    _assert_compact_equals_full(full, compact)
def test_compact_keeps_heterogeneous_rank_identity_and_has_no_histogram():
    raw = case(dp=2); first = op = _op("same_id", memory=64, stream="computeA"); second = replace(op, stream="computeB", layer=7)
    options = {"external_inputs": {}, "isolated": False, "trace_limit": 10}
    full = graph_costs(raw, {0: [first], 1: [second]}, **options); compact = graph_costs(raw, {0: [first], 1: [second]}, compact=True, **options)
    _assert_compact_equals_full(full, compact)
    item = compact["profile"]["by_operator"]["s0:same_id"]
    assert (item["replicas"][0]["layer"], item["replicas"][0]["stream"]) == (0, "computeA")
    assert (item["replicas"][1]["layer"], item["replicas"][1]["stream"]) == (7, "computeB")
    assert "layer" not in item and "stream" not in item and "by_window" not in compact["profile"]
    assert compact["event_profile"]["completed_physical_events"] > 0
    zero = graph_costs(raw, {0: [first], 1: [second]}, compact=True, external_inputs={}, isolated=False, trace_limit=0)
    assert zero["event_profile"]["completed_physical_events"] > 0 and zero["event_profile"]["trace_events"] == 0
    with pytest.raises(TypeError): graph_costs(raw, {0: [first], 1: [second]}, compact=True, compact_window_ms=1.0, **options)
def test_compact_multi_rank_keeps_each_replica_count_and_time():
    raw = compose_case("glm-5.1", "a3", tp=1, dp=2, ep=2, pcp=1, layers=4, prompt_tokens=8, output_tokens=1, max_num_batched_tokens=16, max_num_seqs=2, max_model_len=256)
    graphs = target_graphs(raw, phase="decode", tokens=8, context=0, sequences=1, scope="body")
    full = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False, trace_limit=1024)
    compact = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False, trace_limit=1024, compact=True)
    _assert_compact_equals_full(full, compact); assert set(compact["profile"]["by_rank"]) == {0, 1}
    for item in compact["profile"]["by_operator"].values(): assert sum(rank["operator_events"] for rank in item["replicas"].values()) == item["operator_events"]
def test_body_forward_compact_keeps_same_body_span_and_profile():
    raw = compose_case("glm-5.1", "a3", tp=1, dp=1, ep=1, pcp=1, layers=1, prompt_tokens=8, output_tokens=1, max_num_batched_tokens=16, max_num_seqs=2, max_model_len=256)
    default = body_forward(raw, phase="decode", tokens=8, context=0, sequences=1, event_limit=512)
    compact = body_forward(raw, phase="decode", tokens=8, context=0, sequences=1, event_limit=512, compact=True)
    assert compact["body_forward_ms"] == default["body_forward_ms"] and compact["geometry"] == default["geometry"]
    assert compact["capacity"] == default["capacity"] and compact["execution"] == default["execution"] and compact["duration_ms"] == default["duration_ms"]
    assert compact["profile"]["operator_events"] == len(default["timeline"])
    assert compact["profile"]["by_family"] == {family: {"operator_events": len(rows), "elapsed_ms": pytest.approx(sum(row["elapsed_ms"] for row in rows), rel=1e-12, abs=1e-15), "replicas": compact["profile"]["by_family"][family]["replicas"]} for family, rows in {family: [row for row in default["timeline"] if row["family"] == family] for family in {row["family"] for row in default["timeline"]}}.items()}
def test_compact_isolated_reduction_preserves_weighted_sum_and_bound():
    from _support.captures import REPO_ROOT
    raw = json.loads((REPO_ROOT / "examples/cases/glm-5.3.json").read_text()); raw["model"] = {**raw["model"], "layers": 2, "quantization": "w8a8"}
    graphs = target_graphs(raw, phase="decode", tokens=8, context=0, sequences=1, scope="body")
    full = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, trace_limit=0); compact = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, trace_limit=0, compact=True)
    assert compact["isolated"]["operator_sum_ms"] == pytest.approx(full["isolated"]["operator_sum_ms"], rel=1e-12, abs=1e-15)
    assert compact["isolated"]["operators_omitted"] is True and "operators" not in compact["isolated"]
    assert compact["isolated"]["by_family"] == {family: {"occurrences": sum(row["occurrences"] for row in full["isolated"]["operators"] if row["operator"]["family"] == family), "total_ms": pytest.approx(sum(row["total_ms"] for row in full["isolated"]["operators"] if row["operator"]["family"] == family), rel=1e-12, abs=1e-15)} for family in {row["operator"]["family"] for row in full["isolated"]["operators"]}}
    assert compact["bounds"] == full["bounds"] and compact["duration_ms"] == full["duration_ms"]
def test_compact_publishes_selected_matrix_wrapper_kind_without_second_op():
    raw = case(); wrapper = {"native_kind": "GroupedMatmul", "source_role": "single_group_dense_wrapper", "source": {"basis": "declared_source_wrapper_not_capture_dispatch_proof"}}
    selected = replace(_op("latent", flops=4e6), family="projection", recipe="latent_projection", semantic_shape=(1, 1, 1), local_shape=(1, 1, 1), padded_shape=(1, 1, 1), launch_shape=(1, 1, 1), metadata=(("matrix_wrapper", wrapper),))
    compact = graph_costs(raw, {0: [selected]}, compact=True, isolated=False, trace_limit=0); item = compact["profile"]["by_operator"]["s0:latent"]; rank = item["replicas"][0]
    assert item["operator_events"] == 1 and rank["native_kind"] == "GroupedMatmul" and rank["source_role"] == "single_group_dense_wrapper"
    assert rank["source_basis"] == "declared_source_wrapper_not_capture_dispatch_proof" and compact["execution"]["completed_tasks"] >= 1
    plain_rank = graph_costs(raw, {0: [replace(selected, metadata=())]}, compact=True, isolated=False, trace_limit=0)["profile"]["by_operator"]["s0:latent"]["replicas"][0]
    assert "native_kind" not in plain_rank and "source_role" not in plain_rank and "source_basis" not in plain_rank
def test_zero_work_chain_matches_serial_latency_and_same_timestamp_order():
    engine = _engine(); previous = engine.submit("origin", latency=1.5)
    for index in range(4000): previous = engine.submit(f"hop{index}", after=(previous,))
    tail = engine.submit("tail", {"compute": 20.0}, after=(previous,)); engine.advance(until=(tail,))
    assert previous.time == 1.5 and tail.time == 1.5 + 2.0 and engine.summary()["live_tasks"] == 0 and engine.serviced["compute"] == 20.0
    ordered = _engine(); seen = []
    first = ordered.submit("first", latency=1.0, on_complete=lambda event: seen.append(event.name)); second = ordered.submit("second", latency=1.0, on_complete=lambda event: seen.append(event.name))
    ordered.advance(until=(first, second), first=False)
    assert seen == ["first", "second"] and first.time == second.time == 1.0
def test_fifo_and_pool_admission_keep_lower_sequence_priority():
    engine = _engine(); earlier = engine.submit("earlier", {"queue": 4.0}); later = engine.submit("later", {"queue": 1.0}); engine.advance(until=(later,), first=False)
    assert earlier.time == 4.0 and later.time == 5.0
    pooled = _engine()
    holder = pooled.submit("holder", acquire=(Allocation("live", "hbm", 80.0),), latency=2.0, free=(Allocation("live", "hbm", 80.0),))
    waiting = pooled.submit("waiting", acquire=(Allocation("next", "hbm", 80.0),), latency=1.0, free=(Allocation("next", "hbm", 80.0),))
    pooled.advance(until=(holder, waiting), first=False)
    assert holder.time == 2.0 and waiting.time == 3.0 and math.isclose(pooled.peak_allocated["hbm"], 80.0)
def test_fair_staggered_work_matches_closed_form():
    engine = Engine([Resource("compute", 100.0)]); amounts = [1.0 + index * 0.01 for index in range(32)]
    events = [engine.submit(f"t{index}", {"compute": amount}) for index, amount in enumerate(amounts)]; engine.advance(until=tuple(events), first=False)
    remaining, now = list(amounts), 0.0
    while any(remaining):
        alive = [value for value in remaining if value > 0]; rate = 100.0 / len(alive); step = min(value / rate for value in alive); now += step
        remaining = [value - step * rate if value > 0 else 0.0 for value in remaining]
    assert events[-1].time == pytest.approx(now) and engine.serviced["compute"] == pytest.approx(sum(amounts))
def test_body_forward_detail_preserves_timeline_and_is_deterministic():
    raw = case(output=1); options = dict(phase="decode", tokens=1, context=32, sequences=1)
    first, second = body_forward(raw, **options), body_forward(raw, **options)
    assert first["body_forward_ms"] == second["body_forward_ms"] and first["timeline"] == second["timeline"]
    assert first["operator_occurrences"] == sum(row["family"] != "barrier" for row in first["timeline"])
    changed = body_forward(raw, phase="decode", tokens=2, context=32, sequences=1)
    assert changed["geometry"]["tokens_per_sequence"] == 2 and changed["body_forward_ms"] != first["body_forward_ms"]
def test_body_excludes_host_and_step_wait_but_retains_slow_memory():
    raw = case(output=1); options = dict(scope="body", tokens=4, context=64, sequences=3); base = predict(raw, **options)
    raw["machine"]["device"]["launch_us"] = 20000; raw["runtime"]["submission"] = {"scheduler_ms": 10, "completion_ms": 10}; raw["workload"]["requests"][0]["arrival_ms"] = 1000
    assert predict(raw, **options)["body_forward_ms"] == base["body_forward_ms"]
    raw["machine"]["device"]["rates"]["memory"] /= 100; slower = predict(raw, **options)
    assert slower["body_forward_ms"] > base["body_forward_ms"] and slower["execution"]["work"] == base["execution"]["work"]
    assert all(row["family"] not in {"embedding", "logits", "sampling", "token_commit"} for row in base["timeline"])
    assert base["execution"]["submission"]["host_service_ms"] == 0
def test_body_retains_full_capture_padding_and_balanced_domain():
    raw = case(); raw["runtime"].update(graph_mode="auto", graph_capture_keys=[8], graph_capture_mode="full")
    body = target_graphs(raw, phase="decode", tokens=1, scope="body")[0]; model = target_graphs(raw, phase="decode", tokens=1, scope="model")[0]
    a = next(op for op in body if op.op_id.endswith("norm1")); b = next(op for op in model if op.op_id == a.op_id)
    assert a.local_shape == b.local_shape == (8, 32) and a.vector_ops == b.vector_ops == 1280
    report = predict(raw, scope="body")
    assert report["replay_scope"] == "full" and report["execution"]["submission"]["graph_replays"] == 0
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32); raw["runtime"]["routing"] = {"policy": "balanced"}
    assert predict(raw, scope="body")["body_forward_ms"] > 0
    expert = next(op for op in target_graphs(raw, phase="decode", tokens=1, scope="body")[0] if op.family == "grouped_gemm"); counts = dict(expert.metadata)["routing_realization"]
    assert sum(counts["global_expert_rows"]) == 8 * 2 and (counts["valid_local_tokens"], counts["padded_local_tokens"]) == (1, 8)
def test_body_complete_ep_cohort_retains_network_service_and_pp_dependencies():
    raw = case(dp=2, tp=1, ep=2, pp=2); raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32); raw["runtime"]["routing"] = {"policy": "balanced"}
    options = dict(scope="body", sequences=4, context=32); fast = predict(raw, **options)
    raw["machine"]["links"][0]["capacity_bytes_per_ms"] /= 2; slow = predict(raw, **options)
    assert fast["geometry"]["participating_replicas"] == [0, 1] and fast["geometry"]["valid_query_tokens"] == 8
    assert fast["execution"]["serviced"]["link:fabric"] > 0 and slow["body_forward_ms"] > fast["body_forward_ms"] and slow["execution"]["serviced"] == fast["execution"]["serviced"]
    assert any(row["family"] == "pp_p2p" for row in fast["timeline"]) and fast["execution"]["live_tasks"] == fast["execution"]["live_programs"] == 0
def test_pd_body_has_independent_pool_spans_and_predict_cli_exposes_geometry():
    from _support.serving_cases import pd_case
    raw = pd_case(); raw["pools"]["decode"]["machine"]["device"]["rates"]["memory"] /= 100
    result = predict_pd(raw, scope="body", context=32, tokens=4, sequences=2)
    assert result == predict(raw, scope="body", context=32, tokens=4, sequences=2)
    assert result["body_forward_ms"]["decode"] > result["body_forward_ms"]["prefill"]
    modified = deepcopy(raw); modified["connector"]["bytes_per_s"] /= 1000
    assert predict_pd(modified, scope="body", context=32, tokens=4, sequences=2)["body_forward_ms"] == result["body_forward_ms"]
    for pool in result["pools"].values():
        assert pool["geometry"]["tokens_per_sequence"] == 4 and pool["execution"]["live_tasks"] == pool["execution"]["live_programs"] == 0
    args = parser().parse_args(["predict", "examples/quickstart.json", "--scope", "body", "--phase", "decode", "--tokens", "4", "--sequences", "2", "--context", "2048", "--event-limit", "0"])
    cli = run(args)
    assert cli["body_forward_ms"] > 0 and cli["geometry"]["valid_query_tokens"] == 8 and cli["execution"]["submission"]["included"] is False
def test_standalone_draft_is_not_reported_as_a_target_body():
    raw = case(); raw["model"] = {"preset": "kimi-k3-dspark"}
    with pytest.raises(ModelingError, match="draft_requires_target_model"): predict(raw, scope="body")
@pytest.mark.parametrize("query", (1, 4))
@pytest.mark.parametrize("sku", SKUS)
def test_plain_body_executes_caller_copies_and_separate_source_programs(sku, query):
    raw = _gdn_case(sku); before = deepcopy(raw)
    result = predict(raw, scope="body", tokens=query, context=128, sequences=2); recipes = [row["recipe"] for row in result["timeline"]]
    assert recipes.count("ascend_causal_conv1d_update") == 1 and recipes.count("ascend_gdn_decode_gating") == 1
    assert recipes.count("decode_q_l2norm") == recipes.count("decode_k_l2norm") == 1
    # Live A3/A5 captures confirm all 5 fake GDN decode TensorMove ops are absent.
    assert sum(row["family"] == "tensor_copy" for row in result["timeline"]) == 0
    assert result["execution"]["serviced"]["mte2_aiv@r0:s0"] > 0 and result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["default_mechanisms"]["selections"][0]["selection"] == "default_installed" and raw == before
def test_explicit_program_and_unsupported_precision_keep_their_existing_execution():
    raw = _gdn_case("Ascend950DT_9582"); raw["runtime"]["kernels"] = {"families": {"gdn_gating": {"kind": "tiled_pointwise", "input_bytes_per_element": 2, "outputs": []}}}
    explicit = predict(raw, scope="body", tokens=4, context=128)
    assert not any(row["recipe"] == "ascend_gdn_decode_gating" for row in explicit["timeline"])
    assert explicit["default_mechanisms"]["selections"][0]["reason"] == "explicit_implementation_preserved"
    raw = _gdn_case("Ascend950DT_9582"); raw["model"]["activation_dtype"] = "fp16"; other = predict(raw, scope="body", tokens=4, context=128)
    assert other["body_forward_ms"] > 0 and other["default_mechanisms"]["selections"][0]["reason"] == "gdn_requires_bf16_inputs_fp32_state"
def test_prefill_does_not_install_a_decode_caller_and_explicit_options_retain_priority():
    raw = _gdn_case("Ascend950DT_9582"); parsed, report = attention_defaults(parse_case(raw), "prefill", tokens=4)
    assert "gdn_decode" not in parsed.runtime.kernels.get("lowerings", {}) and report["selections"][0]["reason"] == "decode_profile_only"
    binding = {"implementation": "vllm_ascend_gdn_decode", "options": {"gating_programs": 8}}; raw["runtime"]["kernels"] = {"lowerings": {"gdn_decode": binding}}
    selected, report = attention_defaults(parse_case(raw), "decode", tokens=4); options = selected.runtime.kernels["lowerings"]["gdn_decode"]["options"]
    assert options["gating_programs"] == 8 and "recurrent_implementation" not in options and options["gating_beta_store"]["from_hardware"] is True
    assert binding == {"implementation": "vllm_ascend_gdn_decode", "options": {"gating_programs": 8}} and report["selections"][0]["reason"] == "explicit_implementation_preserved"
def test_explicit_per_request_recurrence_survives_body_caller_selection():
    raw = _gdn_case("Ascend950DT_9582"); raw["runtime"]["kernels"] = {"recipes": {"recurrent_scan0": {"kind": "recurrent_vector"}}}
    result = predict(raw, scope="body", tokens=4, context=128, sequences=2); recipes = [row["recipe"] for row in result["timeline"]]
    assert "recurrent_scan0" in recipes and "ascend_gdn_decode_gating" not in recipes
    assert result["default_mechanisms"]["selections"][0]["reason"] == "explicit_implementation_preserved" and result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
def test_source_recurrence_body_defaults_execute_identified_store_queue():
    raw = _gdn_case("Ascend950DT_9582"); ops = target_operators(raw, scope="body", phase="decode", tokens=4, sequences=12, context=128)
    gated, = [op for op in ops if op.recipe == "ascend_gdn_decode_gating"]; recurrent, = [op for op in ops if op.recipe == "ascend_recurrent_gdn_fp32_state"]
    assert dict(recurrent.metadata)["recurrence_selection"] == "arch35_fp32"
    baseline, = [row for row in operator_costs(raw, ops)["operators"] if row["operator"]["op_id"] == gated.op_id]
    slower = deepcopy(raw); rate = dict(parse_case(raw).machine.dies[0].rates)["scalar_store_issue"]; slower["machine"]["device"] = {"rates": {"scalar_store_issue": rate / 100}}
    slow, = [row for row in operator_costs(slower, ops)["operators"] if row["operator"]["op_id"] == gated.op_id]
    assert slow["duration_ms"] > baseline["duration_ms"] and slow["serviced"] == pytest.approx(baseline["serviced"], rel=1e-12)
    report = predict(raw, scope="body", tokens=4, sequences=12, context=128)
    assert report["execution"]["serviced"]["scalar_store_issue@r0:s0"] > 0 and sum(row["family"] == "allreduce" for row in report["timeline"]) >= 2
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
def test_source_recurrence_domain_and_primitive_do_not_transfer_to_unmeasured_hardware():
    from mindie_sim.schema.hardware import Hardware
    a3, _ = attention_defaults(parse_case(_gdn_case("Ascend910_9362")), "decode", tokens=4)
    assert not a3.runtime.kernels["lowerings"]["gdn_decode"]["options"]
    wider, _ = attention_defaults(parse_case(_gdn_case("Ascend950DT_9582")), "decode", tokens=9)
    assert "recurrent_implementation" not in wider.runtime.kernels["lowerings"]["gdn_decode"]["options"]
    hardware = parse_case(_gdn_case("Ascend950DT_9582")).machine
    assert Hardware.from_mapping(hardware.to_dict()).dies == hardware.dies
def test_reused_target_uses_current_hardware_store_response_without_changing_work():
    raw = _gdn_case("Ascend950DT_9582"); ops = target_operators(raw, scope="body", phase="decode", tokens=4, sequences=2, context=128); changed = deepcopy(raw)
    changed["machine"]["device"] = {"store_queues": {"scalar_store_issue": {"response_ms": .001}}}
    fresh = target_operators(changed, scope="body", phase="decode", tokens=4, sequences=2, context=128)
    def gating_cost(hardware, target):
        return next(row for row in operator_costs(hardware, target)["operators"] if row["operator"]["recipe"] == "ascend_gdn_decode_gating")
    prior, reused, rebuilt = gating_cost(raw, ops), gating_cost(changed, ops), gating_cost(changed, fresh)
    assert reused["duration_ms"] == rebuilt["duration_ms"] > prior["duration_ms"] and reused["serviced"] == pytest.approx(prior["serviced"], rel=1e-12)
    assert dict(dict(parse_case(changed).machine.dies[0].store_queues)["scalar_store_issue"])["outstanding"] == 8
    changed["machine"]["device"]["work_units"] = {"scalar_store_issue": "bytes"}
    with pytest.raises(ValueError, match="native_work_unit_conflict:scalar_store_issue"): parse_case(changed)
def test_body_matrix_default_is_shared_by_predict_graph_operators_and_structure():
    raw = _physical_case(); before = deepcopy(raw); graphs = target_graphs(raw, **GEOMETRY); body = predict(raw, **GEOMETRY)
    graph = graph_costs(raw, graphs, isolated=False, external_inputs={"analysis_input": 0}); isolated = operator_costs(raw, target_operators(raw, **GEOMETRY)); structure = export_structure(raw, **GEOMETRY)
    assert _matrix_rows(body) and all(row["program"] == "SteadyMatmul" for row in _matrix_rows(body))
    assert graph["timeline"] == body["timeline"] and graph["duration_ms"] == body["body_forward_ms"] and graph["execution"]["serviced"] == pytest.approx(body["execution"]["serviced"], rel=1e-12, abs=1e-12)
    assert all(row["program"] == "SteadyMatmul" for row in isolated["operators"] if row["operator"]["family"] in MATRIX_FAMILIES)
    program = structure["execution"]["software"]["kernels"]["programs"]["matmul"]
    assert program["kind"] == "steady_matmul" and program["owner_cohorts"] is True
    # M2 declares the full ordered candidate grid; the historical coarse pair remains part of that same default.
    assert [128, 128, 128] in program["candidates"] and [128, 128, 64] in program["candidates"]
    assert structure["execution"]["default_mechanisms"] == body["default_mechanisms"] and body["execution"]["live_tasks"] == body["execution"]["live_programs"] == 0 and raw == before
@pytest.mark.parametrize("program,expected", MATRIX_PROGRAMS)
def test_explicit_matrix_program_is_preserved_in_both_body_costing_paths(program, expected):
    raw = _physical_case(); raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": program}}}; report = predict(raw, **GEOMETRY)
    assert all(row["program"] == expected for row in _matrix_rows(report))
    isolated = operator_costs(raw, target_graphs(raw, **GEOMETRY)[0])
    assert all(row["program"] == expected for row in isolated["operators"] if row["operator"]["family"] in MATRIX_FAMILIES)
    assert next(row for row in report["default_mechanisms"]["selections"] if row["family"] == "matmul")["selection"] == "preserved"
def test_named_family_source_and_explicit_book_keep_priority():
    raw = _physical_case(); raw["runtime"]["kernels"] = {"families": {family: {"kind": "mapped_matmul"} for family in MATRIX_FAMILIES}, "recipes": {"dense_down": {"kind": "tiled_matmul"}}}
    rows = _matrix_rows(predict(raw, **GEOMETRY))
    assert any(row["recipe"] == "dense_down" and row["program"] == "TiledMatmul" for row in rows) and all(row["program"] == "MappedMatmul" for row in rows if row["recipe"] != "dense_down")
    source_body = predict(grouped_case(source=True), scope="body", phase="decode", tokens=1)
    assert all(row["program"] == "OperatorPlan" for row in source_body["timeline"] if row["family"] == "grouped_gemm")
    report = predict(_physical_case(), recipes=RecipeBook(programs={"matmul": TiledMatmul(16, 16)}), **GEOMETRY)
    assert all(row["program"] == "TiledMatmul" for row in _matrix_rows(report)) and report["default_mechanisms"] == {"profile": "explicit_recipe_book_preserved"}
def test_model_scope_and_body_optout_retain_detailed_control_and_semantic_work():
    raw = _physical_case(); options = {**GEOMETRY, "scope": "model"}; model = target_graphs(raw, **options)
    assert model.effective_kernels["programs"]["matmul"]["kind"] == "steady_matmul"
    assert all(row["program"] == "SteadyMatmul" for row in operator_costs(raw, model)["operators"] if row["operator"]["family"] in MATRIX_FAMILIES)
    control = deepcopy(raw); control["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "tiled_matmul"}}}
    raw_model = target_graphs(control, apply_body_defaults=False, **options)
    assert raw_model.effective_kernels["programs"]["matmul"]["kind"] == "tiled_matmul"
    assert all(row["program"] == "TiledMatmul" for row in operator_costs(control, raw_model)["operators"] if row["operator"]["family"] in MATRIX_FAMILIES)
    generic, compact = target_graphs(control, apply_body_defaults=False, **GEOMETRY), target_graphs(raw, **GEOMETRY)
    assert {r: tuple(op for op in v if op.family in MATRIX_FAMILIES) for r, v in compact.items()} == {r: tuple(op for op in v if op.family in MATRIX_FAMILIES) for r, v in generic.items()}
    graph = graph_costs(control, generic, isolated=False, external_inputs={"analysis_input": 0})
    assert all(row["program"] == "TiledMatmul" for row in _matrix_rows(graph))
def test_serving_default_remains_the_explicit_tiled_control():
    from mindie_sim.prediction.body_defaults import DEFAULT_MATMUL_CANDIDATES
    raw = _physical_case(); automatic = predict(raw); steady = deepcopy(raw)
    steady["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul", "owner_cohorts": True, "candidates": [list(tile) for tile in DEFAULT_MATMUL_CANDIDATES]}}, "families": {"grouped_gemm": {"kind": "steady_matmul", "owner_cohorts": True}}}
    steady_report = predict(steady)
    assert automatic["metrics"] == steady_report["metrics"]
    assert {k: v for k, v in automatic["execution"].items() if k != "runtime_uid"} == {k: v for k, v in steady_report["execution"].items() if k != "runtime_uid"}
    explicit = deepcopy(raw); explicit["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "tiled_matmul"}}}; control = predict(explicit)
    assert control["execution"]["work"] == automatic["execution"]["work"] and automatic["execution"]["allocation_events"] > control["execution"]["allocation_events"]
def test_default_keeps_tier4_router_owner_grid_and_grouped_mapping_domain():
    from _support.captures import REPO_ROOT
    raw = json.loads((REPO_ROOT / "examples/cases/glm53-a3-w4a8.json").read_text()); raw["model"]["layers"] = 4
    graph = target_graphs(raw, scope="body", phase="decode", tokens=1, context=4096)[0]
    router_index = next(i for i, op in enumerate(graph) if op.layer == 3 and op.family == "router")
    router = operator_costs(raw, graph[router_index:router_index + 1])["operators"][0]; mapping = next(row for row in router["basis"]["matrix_mappings"] if "selected" in row)
    assert router["operator"]["local_shape"] == [1, 256, 6144]
    # moe_router_dtype=float32 runs on AscendGateLinear Tier 4: BF16 Cube operands accumulate into FP32, so the matrix dtype is BF16.
    assert router["operator"]["dtype"] == "bf16"
    assert (mapping["selected"]["tile_m"], mapping["selected"]["tile_n"], mapping["selected"]["tile_k"]) == (16, 128, 128)
    assert mapping["default_128_grid_tiles"] == 2 and router["peak_buffers"]["l0b@r0:s0"] <= 65536
    explicit = deepcopy(raw); explicit["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul"}}}
    for i, op in enumerate(graph):
        if op.layer == 3 and op.family == "grouped_gemm":
            inherited = operator_costs(raw, graph[i:i + 1])["operators"][0]; previous = operator_costs(explicit, graph[i:i + 1])["operators"][0]
            assert inherited["serviced"] == previous["serviced"] and inherited["duration_ms"] == previous["duration_ms"]
            assert next(row["selected"] for row in inherited["basis"]["matrix_mappings"] if "selected" in row)["tile_n"] == 256


def test_homogeneous_layer_dedup_isolates_irregular_quant_exclusions(monkeypatch):
    """Per-layer quant exclusions must split archetypes so homogeneous dedup has 0.0 drift."""
    from mindie_sim.prediction.frontend.repeat import _FrontendRepeat, _layer_quant_signature

    raw = compose_case("qwen3-32b", hardware="a3", layers=3, tp=1, dp=1, quant="w8a8_dynamic")
    raw["model"]["quant"]["exclusions"] = list(raw["model"]["quant"]["exclusions"]) + ["model.layers.1.self_attn"]
    case = parse_case(raw)
    assert _layer_quant_signature(case.model, 1) != _layer_quant_signature(case.model, 0)
    assert _layer_quant_signature(case.model, 0) == _layer_quant_signature(case.model, 2)
    ded = target_graphs(case, scope="body", phase="decode", tokens=1, context=256)[0]
    monkeypatch.setattr(_FrontendRepeat, "_try_lower_homogeneous", lambda *a, **k: None)
    ref = target_graphs(case, scope="body", phase="decode", tokens=1, context=256)[0]
    assert [(o.layer, o.family, o.dtype, o.flops, o.hbm_bytes) for o in ded] == [
        (o.layer, o.family, o.dtype, o.flops, o.hbm_bytes) for o in ref
    ]


def test_stars_retired_interval_split_and_raw_eager_host_submit(monkeypatch):
    """StarsDescriptorQueue splits straddling intervals cleanly; raw_eager charges P15 host submit."""
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.execution import runner as exec_runner
    from mindie_sim.prediction.execution.submission import StarsDescriptorQueue, StarsSpec
    from mindie_sim.primitives.loader import load_primitives

    q = StarsDescriptorQueue(StarsSpec("die", 64, 1, 1.0, 2.0, "descriptor_ready", "test"))
    q._retired_comp_ns = [(0, 5000)]
    q._retired_desc_ns = [(1000, 2000), (2000, 4000)]
    q._fold_retired_up_to(2500)
    assert q._retired_desc_ns == [(2500, 4000)] and q._retired_comp_ns == [(2500, 5000)]
    q._fold_retired_up_to(None)
    assert q._folded_uncovered_ns == 0 and q._folded_desc_ns == 3000

    raw = compose_case("qwen3-32b", hardware="Ascend910_9362", layers=1, tp=1, dp=1)
    raw["runtime"]["graph_capture_mode"] = "full"
    case_full = parse_case(raw)
    ev_full = EvaluationContext.create(case_full, RecipeBook())
    ex_full = Execution(case_full, ev_full.cost, recipes=RecipeBook(), bindings=ev_full.bindings, evaluation=ev_full)
    seen_modes = []
    orig_build = exec_runner.build_stars_queues
    monkeypatch.setattr(exec_runner, "build_stars_queues", lambda *a, **k: (seen_modes.append(k.get("submission_mode")), orig_build(*a, **k))[1])
    exec_runner.attach_stars_queues(ex_full)
    assert seen_modes == ["aclgraph"]

    raw_eager = compose_case("qwen3-32b", hardware="Ascend910_9362", layers=1, tp=1, dp=1)
    raw_eager["runtime"].update({"graph_mode": "none", "graph_capture_mode": "none", "compilation_config": {"cudagraph_mode": "NONE"}, "metadata": {"feature_track": "raw_eager"}})
    case_eager = parse_case(raw_eager)
    ev_eager = EvaluationContext.create(case_eager, RecipeBook(), primitive_table=load_primitives("Ascend910_9362", require_calibrated=True))
    op = target_graphs(case_eager, scope="body", phase="decode", tokens=1, context=128)[0][0]
    with_sub = graph_costs(case_eager, {0: [op]}, evaluation=ev_eager, include_submission=True, isolated=True, external_inputs={"analysis_input": 0})
    no_sub = graph_costs(case_eager, {0: [op]}, evaluation=ev_eager, include_submission=False, isolated=True, external_inputs={"analysis_input": 0})
    assert with_sub["execution"]["serviced"].get("host@r0:s0", 0.0) > 0.0 and with_sub["duration_ms"] > no_sub["duration_ms"]
