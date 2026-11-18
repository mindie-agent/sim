"""Source-geometry invariants exercised through ordinary compose/serving paths.

[Role] Verify K3.1 cache ownership, communication and draft context execution.
[Invariants] Expectations derive from byte arithmetic and explicit source shapes.
[Timeouts & Locks] Bounded CPU-only graphs; no locks or external services.
[Side-Effect Policy] Uses isolated pytest state; no NPU contexts or host config.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import compose_case, target_graphs
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.draft import draft_quant
from mindie_sim.catalog.draft_geometry import draft_layer_plans
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]


def native_config():
    config = json.loads((ROOT / "mindie_sim/model_configs/kimi-k3.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    text["model_type"] = "kimi_k31"
    text["dsa_config"] = {
        "dsa_layers": text["linear_attn_config"]["full_attn_layers"],
        "index_num_heads": 32, "index_head_dim": 128,
        "index_topk": 2048, "index_rope_dim": 0,
    }
    text.update(dspark_attention_kind="mla", dspark_block_size=5)
    return config


def case(cp=8, proposals=3):
    return parse_case(compose_case(
        native_config(), hardware="a5", tp=8, dp=4, ep=32, dcp=cp,
        num_devices=32, block_size=1536, max_model_len=140000,
        quant={"name": "k31_fp8_cache", "weight_format": "bf16", "activation_format": "bf16",
               "expert_weight_format": "bf16", "kv_dtype": "fp8_e4m3",
               "index_dtype": "fp8_e4m3", "index_scale_dtype": "fp16"},
        mtp_proposal_tokens=proposals, mtp_acceptance_length=min(2, proposals + 1),
        graph_capture_mode="none", prompt_tokens=131072, output_tokens=1024,
    ))


def test_native_sparse_configuration_and_independent_draft_formats():
    c = case()
    sparse = [p for p in c.model.layer_plans if p.indexer]
    assert len(sparse) == 24
    assert sum(p.attention == "kda" for p in c.model.layer_plans) == 69
    assert sparse[0].index == 3 and sparse[-1].index == 92
    assert all((p.index_heads, p.index_dim, p.index_topk) == (32, 128, 2048) for p in sparse)
    assert all(p.owns_index_k and p.index_query_per_layer for p in sparse)
    assert cache_format(c.model, "mla_kv").row_bytes(576) == 656
    assert cache_format(c.model, "index_k").row_bytes(128) == 130
    draft = c.model.auxiliary.dspark
    assert draft_quant(c.model, draft).kv_dtype == "bf16"
    assert all(not p.indexer and not p.mla_output_gate for p in draft_layer_plans(c.model, "dspark"))


@pytest.mark.parametrize("cp", [4, 8])
@pytest.mark.parametrize("proposals", [0, 3])
def test_shared_pool_quote_and_allocation_report_conserve_bytes(cp, proposals):
    c = case(cp, proposals)
    memory = Memory(c)
    pool_unit = 1536 * (24 * 656 + 24 * 130 + (5 * 1152 if proposals else 0))
    assert memory.pages.groups[0].bytes_per_group_page == pool_unit
    states = 3 * (2 + proposals)
    for tokens in (1, 1536 * cp, 1536 * cp + 1, 131588):
        history = (tokens + 1536 * cp - 1) // (1536 * cp)
        expected = (history + states) * pool_unit
        assert memory.cache_bytes(tokens) == (expected,)
        report = memory.cache_allocation(tokens)
        assert report["allocated"]["total_bytes_per_stage_rank"] == [expected]
        assert report["allocated"]["reserve_bytes_per_stage_rank"][0] >= 0
    assert memory.cache_bytes(0) == (0,)


def test_index_history_is_sharded_but_recurrent_heads_are_not():
    c = case()
    sparse = next(p for p in c.model.layer_plans if p.indexer)
    components = layer_cache_components(c.model, sparse, c.runtime.parallel,
                                       131072, capacity_padded=False)
    assert components["index_k"] == 131072 * 130 / 8
    graph = target_graphs(c, phase="decode", tokens=4, context=131072,
                          sequences=2, scope="body", resource_fusion=False)[0]
    qsfa = [op for op in graph if op.recipe == "paired_fp8_qsfa_analytic"]
    assert len(qsfa) == 24
    assert all(dict(op.metadata)["gathered_q_heads"] == 96 for op in qsfa)
    assert len([op for op in graph if op.recipe == "dcp_output_alltoall"]) == 24
    assert not any(op.family == "dcp_allgather" for op in graph)
    scan = next(op for op in graph if op.recipe == "quant_lightning_indexer")
    assert not any(op.recipe in {"index_scores", "index_topk"} for op in graph)
    assert dict(scan.metadata)["native_kind"] == "aclnnQuantLightningIndexerV2"
    assert dict(scan.metadata)["selection_output_dtype"] == "int32"
    # Exact rank-owned visible rows, not a balanced total/CP approximation.
    from mindie_sim.catalog.context_parallel import owned_positions
    owned = [sum(owned_positions(0, 131072 + offset, 8, 1536, rank)
                 for _ in range(2) for offset in range(1, 5))
             for rank in range(8)]
    # Score arithmetic is per token; the four queries occupy one tile.  The
    # one fused call reads Q and the head weights once and materializes no
    # FP32 score plane.
    assert scan.flops == 2 * max(owned) * 32 * 128
    tile_rows = [2 * owned_positions(0, 131072 + 4 - 1, 8, 1536, rank)
                 for rank in range(8)]
    by_id = {op.op_id: op for op in graph}
    operand_bytes = sum(by_id[dep].mte3_bytes for dep in scan.deps[:2])
    assert scan.mte2_bytes == operand_bytes + max(tile_rows) * 130
    assert scan.mte3_bytes == 4 * 2 * 2048 * 4  # int32 indices, 4 query rows, 2 seq
    candidate = next(op for op in graph if op.recipe == "candidate_gather")
    assert candidate.link_bytes == 8 * 2048 * 8 * 7
    kda = [op for op in graph if op.family in {"recurrent_vector", "recurrent_scan"}]
    assert kda
    assert all(dict(op.metadata)["heads"] == 12 for op in kda)


def test_invalid_native_sparse_layer_is_visible():
    config = native_config()
    config["text_config"]["dsa_config"]["dsa_layers"] = [1]
    with pytest.raises(ValueError, match="dsa_layers"):
        compose_case(config, hardware="a5")


def test_external_draft_context_rows_are_distinct_from_query_rows():
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.speculation import draft_program
    from mindie_sim.schema.workload import RequestView
    c = case()
    views = (RequestView("request", 4, 131072, True, "decode",
                         draft_proposal_tokens=3),)
    regions, _ = draft_program(Frontend(resource_fusion=False), c.model, c.runtime,
                               views, "ready")
    ops = [op for region in regions for op in region.ops]
    features = next(op for op in ops if op.recipe == "draft_features")
    assert features.local_shape[0] == 4
    context = [op for op in ops if op.recipe == "draft_context_mla_prolog_v3"]
    assert len(context) == 5
    assert all(op.local_shape == (4, 1536 + 576) for op in context)
    assert all(dict(op.metadata)["native_kind"] == "aclnnMlaPrologV3WeightNz"
               for op in context)
    stores = [op for op in ops if op.recipe == "draft_context_kv_scatter"]
    from mindie_sim.catalog.context_parallel import owned_positions
    cp = max(1, c.runtime.parallel.cp)
    block = max(1, c.runtime.kv_block_size)
    start = 131072
    owned = max(owned_positions(start, start + 3, cp, block, rank)
                for rank in range(cp))
    assert len(stores) == 5
    assert all(op.local_shape[0] == owned for op in stores)
    assert all(op.mte3_bytes == owned * 1152 for op in stores)
    assert all(dict(op.metadata)["native_kind"] == "aclnnScatterPaKvCache"
               for op in stores)
    embedding = next(op for op in ops if op.recipe == "draft_query_embedding")
    assert embedding.mte3_bytes == 3 * 7168 * 2
    assert not any(op.family == "indexer" for op in ops)
    assert all(op.dtype == "bf16" for op in context)


def test_paired_pool_is_used_by_live_admission_and_release():
    from mindie_sim.prediction.capacity import MemoryPolicy
    c = case()
    quote = Memory(c).cache_bytes(131588)[0]
    # Choose a hardware budget admitting exactly two requests after weights.
    probe = Memory(c, MemoryPolicy(derive_workspace=False))
    budget = probe.fixed_per_rank[0] + 2 * quote
    machine = replace(c.machine, dies=tuple(
        replace(device, memory_bytes=budget) for device in c.machine.dies))
    runtime = replace(c.runtime, gpu_memory_utilization=1)
    memory = Memory(replace(c, machine=machine, runtime=runtime),
                    MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "first", 131588)
    assert memory.resize(0, "second", 131588)
    assert not memory.resize(0, "third", 131588)
    memory.release(0, "first")
    assert memory.resize(0, "third", 131588)


def test_fp8_call_has_an_executable_analytic_recipe_in_the_shared_engine():
    from mindie_sim import predict
    c = case().to_dict()
    plan = c["model"]["layer_plans"][3]
    plan.update(index=0, kv_owner_layer_id=0, index_k_owner_layer_id=0,
                topk_owner_layer_id=0, ffn="dense", intermediate_size=128)
    c["model"].update(layers=1, body_layers=1, layer_plans=[plan])
    c["runtime"]["pp_layer_partition"] = [1]
    c["runtime"]["metadata"].update(draft_proposal_tokens=0)
    c["runtime"]["metadata"].pop("draft_mode", None)
    c["runtime"]["mtp_acceptance_length"] = 1
    result = predict(c, scope="body", phase="decode", tokens=4,
                     context=4096, sequences=1)
    assert result["duration_ms"] > 0
    assert result["operator_occurrences"] > 0


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
def test_query_head_gather_and_output_collective_scale_with_dcp(cp):
    graph = target_graphs(case(cp), phase="decode", tokens=1, context=4096,
                          scope="body", resource_fusion=False)[0]
    attention = [op for op in graph if op.recipe == "paired_fp8_qsfa_analytic"]
    assert len(attention) == 24
    assert all(dict(op.metadata)["gathered_q_heads"] == 12 * cp for op in attention)
    assert len([op for op in graph if op.recipe == "dcp_output_alltoall"]) == (24 if cp > 1 else 0)


def test_explicit_int8_template_is_not_silently_rebound_to_fp8_dcp():
    c = case()
    runtime = replace(c.runtime, kernels={
        "programs": {"indexed_attention": {"kind": "indexed_attention_packed_int8"}}})
    with pytest.raises(ValueError, match="conflicts_with"):
        target_graphs(replace(c, runtime=runtime), phase="decode", tokens=1,
                      context=4096, scope="body")


def test_indexer_precision_is_independent_of_main_kv_precision():
    c = case()
    quant = replace(c.model.quant, index_dtype="bf16", index_scale_dtype="none")
    metadata = dict(c.model.metadata)
    regions = dict(metadata["cache_representations"])
    regions.pop("index_k")
    metadata["cache_representations"] = regions
    model = replace(c.model, quant=quant, metadata=tuple(metadata.items()))
    memory = Memory(replace(c, model=model))
    assert memory.pages.groups[0].bytes_per_group_page == 1536 * (
        24 * 656 + 24 * 128 * 2 + 5 * 1152)
