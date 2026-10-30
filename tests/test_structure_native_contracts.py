"""Declared model graphs compared with retained A3/A5 native descriptors."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.observations.kernels import kernel_rows
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure
from mindie_sim.structure_contracts import comparison_scope, predicted_contract
from test_mla_family_selection import mla_case


ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return parse_case(json.loads((ROOT / "examples/cases" / f"{name}.json").read_text()))


def _capture(group, name, profiler_api):
    import mindie_profiler

    path = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures" / group / name
    if not path.is_file():
        pytest.skip(f"retained profiler source fixture is unavailable: {path}")
    return tuple(kernel_rows(path))


@pytest.mark.parametrize("case_name,capture,tokens", [
    ("qwen35-a3-tp4", "a3_qwen35.csv", 8192),
    ("qwen36-a3-tp4", "a3_qwen36.csv", 8192),
    ("qwen38-a5", "a5_qwen38.csv", 16384),
])
def test_real_gated_norm_abi_matches_independent_a3_a5_model_frontends(
    case_name, capture, tokens, profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract

    actual = next(row for row in _capture("body_fused", capture, profiler_api)
                  if row.kind == "_layer_norm_fwd_1pass_kernel_npu")
    case = _case(case_name)
    operators = target_operators(case, phase="prefill", tokens=tokens)
    gates = [op for op in operators if op.recipe == "linear_output_norm_gate"]
    assert len(gates) > 1
    assert len({op.op_id for op in gates}) == len(gates)
    assert all(predicted_contract(op.to_dict()) == observed_contract(actual) for op in gates)
    assert all(dict(op.metadata)["fused_from"] == ("rmsnorm", "gate") for op in gates)
    assert all(op.source_boundary == "" for op in gates)
    assert "norm_order_not_observed" in comparison_scope(gates[0].to_dict())
    # The captured dtype is evidence; a declared alternate precision cannot
    # inherit a match just because it preserves row and channel counts.
    altered = replace(gates[0], dtype="fp16")
    assert predicted_contract(altered.to_dict()) != observed_contract(actual)


@pytest.mark.parametrize("case_name,capture", [
    ("qwen35-a3-tp4", "a3_qwen35.csv"),
    ("qwen36-a3-tp4", "a3_qwen36.csv"),
])
def test_gemma_weight_offset_is_declared_before_matching_real_native_norm(
    case_name, capture, profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract
    from mindie_sim.prediction.transforms import residual_norm

    rows = _capture("body_fused", capture, profiler_api)
    actual = next(row for row in rows if row.kind == "GemmaRmsNorm")
    actual_add = next(row for row in rows if row.kind == "AddRmsNormBias")
    case = _case(case_name)
    operators = target_operators(case, phase="prefill", tokens=8192)
    norms = [op for op in operators if op.family == "rmsnorm"
             and dict(op.metadata).get("norm_weight_offset") == 1.0]
    assert norms
    assert all(predicted_contract(op.to_dict()) == observed_contract(actual) for op in norms)
    ordinary = replace(norms[0], metadata=tuple(
        (key, 0.0 if key == "norm_weight_offset" else value)
        for key, value in norms[0].metadata))
    assert predicted_contract(ordinary.to_dict()) != observed_contract(actual)
    # These A3 cases do not select residual fusion. Even a controlled opt-in
    # lowering must retain Gemma's weight shift and avoid inventing AddBias.
    fused = [op for op in residual_norm(operators, None) if op.family == "add_rmsnorm"]
    assert fused
    assert all(predicted_contract(op.to_dict()) is None for op in fused)
    assert observed_contract(actual_add)[1] == "AddRmsNormBias"
    # Shape alone cannot choose a native add-norm implementation or reveal
    # whether its caller supplied weight or weight+1.
    assert all(predicted_contract(op.to_dict()) != observed_contract(actual_add) for op in fused)


def test_real_kimi_kda_norm_matches_declared_chunk_without_filling_missing_rows(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    rows = _capture("body_fused", "kimi_a5_norm_gate.csv", profiler_api)
    actual, missing = rows
    case = _case("kimi-k3-a5")
    # The retained serving recipe sets max_num_batched_tokens=8192 for the
    # 32768-token request. This declares one chunk, not the full request.
    operators = target_operators(case, phase="prefill", tokens=8192)
    norms = [op for op in operators if op.recipe == "linear_output_norm_gate"]
    assert norms
    assert all(dict(op.metadata)["recurrent_attention_kind"] == "kda" for op in norms)
    assert all(predicted_contract(op.to_dict()) == observed_contract(actual) for op in norms)
    assert observed_contract(missing) is None
    assert all("auxiliary_stats_not_compared" in comparison_scope(op.to_dict()) for op in norms)
    # The retained Kimi dense FIA uses unabsorbed Q/K/V. The current MLA graph
    # absorbs Q into latent width; it cannot claim that different boundary.
    assert all(predicted_contract(op.to_dict()) is None
               for op in operators if op.family == "attention")


def test_prefill_attention_projection_keeps_kv_heads_and_rejects_paged_assumptions():
    case = _case("qwen38-a5")
    prefill = target_operators(case, phase="prefill", tokens=16384)
    attention = next(op for op in prefill if op.family == "attention")
    contract = predicted_contract(attention.to_dict())
    assert contract[0] == "attention_active_qkv"
    assert contract[2] == (
        ((16384, 24, 256), "bf16", "ND"),
        ((16384, 4, 256), "bf16", "ND"),
        ((16384, 4, 256), "bf16", "ND"),
    )
    assert contract[3] == (((16384, 24, 256), "bf16", "ND"),)
    prefix = target_operators(case, phase="prefill", tokens=64, context=1024)
    decode = target_operators(case, phase="decode", tokens=1, context=1024)
    assert all(predicted_contract(op.to_dict()) is None
               for op in (*prefix, *decode) if op.family == "attention")


@pytest.mark.parametrize("case_name,capture,tokens", [
    ("qwen35-a3-tp4", "a3_qwen35.csv", 8192),
    ("qwen36-a3-tp4", "a3_qwen36.csv", 8192),
    ("qwen38-a5", "a5_qwen38.csv", 16384),
])
def test_real_attention_cache_projections_match_independent_model_frontends(
    case_name, capture, tokens, profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract

    rows = _capture("attention_cache_contracts", capture, profiler_api)
    actual_attention = next(row for row in rows if row.kind == "FusedInferAttentionScore")
    actual_cache = next(row for row in rows if row.kind == "ScatterPaKvCache")
    operators = target_operators(_case(case_name), phase="prefill", tokens=tokens)
    attention = [op for op in operators if op.family == "attention"]
    stores = [op for op in operators if op.family == "cache_store"
              and dict(op.metadata).get("cache_kind") == "gqa"]
    assert len(attention) == len(stores) > 1
    assert all(predicted_contract(op.to_dict()) == observed_contract(actual_attention)
               for op in attention)
    assert all(predicted_contract(op.to_dict()) == observed_contract(actual_cache)
               for op in stores)
    assert len({op.layer for op in stores}) == len(stores)
    assert all("native_storage_layout_cache_capacity_slot_dtype_values" in
               comparison_scope(op.to_dict()) for op in stores)


def test_real_cache_capacity_and_paged_attention_preserve_comparison_boundary(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    rows = _capture("attention_cache_contracts", "a5_qwen38.csv", profiler_api)
    caches = [row for row in rows if row.kind == "ScatterPaKvCache"]
    attention = [row for row in rows if row.kind == "FusedInferAttentionScore"]
    assert len(caches) == len(attention) == 2
    assert caches[0].input_shapes[1] != caches[1].input_shapes[1]
    assert observed_contract(caches[0]) == observed_contract(caches[1])
    assert observed_contract(attention[0])[0] == "attention_active_qkv"
    assert observed_contract(attention[1])[0] == "attention_paged_active_qkv"
    assert observed_contract(attention[0]) != observed_contract(attention[1])
    paged = attention[1]
    changed = replace(paged, input_shapes=(
        paged.input_shapes[0], (1000, *paged.input_shapes[1][1:]),
        (1000, *paged.input_shapes[2][1:]), *paged.input_shapes[3:]))
    assert observed_contract(changed) == observed_contract(paged)


def test_a3_model_identity_needs_attention_or_norm_beyond_shared_cache_projection(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    q35 = _capture("attention_cache_contracts", "a3_qwen35.csv", profiler_api)
    q36 = _capture("attention_cache_contracts", "a3_qwen36.csv", profiler_api)
    cache = lambda rows: observed_contract(next(
        row for row in rows if row.kind == "ScatterPaKvCache"))
    attention = lambda rows: observed_contract(next(
        row for row in rows if row.kind == "FusedInferAttentionScore"))
    assert cache(q35) == cache(q36)
    assert attention(q35) != attention(q36)
    declared = target_operators(_case("qwen35-a3-tp4"), phase="prefill", tokens=8192)
    assert all(predicted_contract(op.to_dict()) != attention(q36)
               for op in declared if op.family == "attention")


def test_cache_declaration_keeps_active_tokens_separate_from_prefix_capacity():
    case = _case("qwen38-a5")
    prefix = target_operators(case, phase="prefill", tokens=64, context=1024)
    decode = target_operators(case, phase="decode", tokens=1, context=1024)
    for operators, tokens in ((prefix, 64), (decode, 1)):
        stores = [op for op in operators if op.family == "cache_store"
                  and dict(op.metadata).get("cache_kind") == "gqa"]
        assert stores
        assert all(predicted_contract(op.to_dict()) == (
            "kv_cache_store_active_elements", "ScatterPaKvCache",
            ((tokens * 4 * 256, "bf16"), (tokens * 4 * 256, "bf16")), (tokens,),
        ) for op in stores)


def test_real_glm_sparse_attention_matches_declared_latent_and_rope_geometry(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    rows = _capture("attention_cache_contracts", "a3_glm53.csv", profiler_api)
    actual = next(row for row in rows if row.kind == "SparseFlashAttention")
    operators = target_operators(_case("glm53-a3-w4a8"), phase="prefill", tokens=4096)
    attention = [op for op in operators if op.family == "attention"]
    assert len(attention) == 78
    assert all(predicted_contract(op.to_dict()) == observed_contract(actual)
               for op in attention)
    assert all("cache_capacity_page_size_sequence_and_index_tensors" in
               comparison_scope(op.to_dict()) for op in attention)
    # LightningIndexer performs score and top-k together. The existing DAG
    # keeps those two operations separate, so neither claims that fused ABI.
    assert all(predicted_contract(op.to_dict()) is None
               for op in operators if op.family in {"indexer", "topk"})


def test_real_shared_kv_attention_distinguishes_compressed_cache_presence(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    rows = _capture("attention_cache_contracts", "a3_deepseek_v4f.csv", profiler_api)
    actual = next(row for row in rows if row.kind == "SparseAttnSharedkv")
    case = _case("deepseek-v4-flash-0731")
    operators = target_operators(case, phase="prefill", tokens=4096)
    attention = [op for op in operators if op.family == "attention"]
    uncompressed = [op for op in attention
                    if case.model.layer_plans[op.layer].compress_ratio == 0]
    compressed = [op for op in attention
                  if case.model.layer_plans[op.layer].compress_ratio > 0]
    assert len(uncompressed) == 2 and len(compressed) == 41
    assert all(predicted_contract(op.to_dict()) == observed_contract(actual)
               for op in uncompressed)
    assert all(predicted_contract(op.to_dict()) != observed_contract(actual)
               for op in compressed)
    compressed_rows = _capture(
        "attention_cache_contracts", "a3_deepseek_v4f_compressed.csv", profiler_api)
    assert len(compressed_rows) == 2
    for row in compressed_rows:
        assert all(predicted_contract(op.to_dict()) == observed_contract(row)
                   for op in compressed)
        assert all(predicted_contract(op.to_dict()) != observed_contract(row)
                   for op in uncompressed)
    glm = target_operators(_case("glm53-a3-w4a8"), phase="prefill", tokens=4096)
    assert all(predicted_contract(op.to_dict()) != observed_contract(actual)
               for op in glm if op.family == "attention")


def test_source_attention_call_keeps_children_instead_of_becoming_one_kernel_contract():
    raw = mla_case()
    unexpanded = export_structure(raw, phase="decode", tokens=1, context=16,
                                 expand_calls=False)
    expanded = export_structure(raw, phase="decode", tokens=1, context=16)
    parent_id = "rank:0/s0.l0.latent_attention_decode"
    parent = next(node for node in unexpanded["nodes"] if node["node_id"] == parent_id)
    assert parent["contract"] is None
    assert parent["contract_scope"] == (
        "source_call; expand_to_kernel_boundaries_before_tensor_comparison")
    call = next(row for row in expanded["source_calls"] if row["node_id"] == parent_id)
    children = [node for node in expanded["nodes"] if node["node_id"] in call["children"]]
    assert [node["operator"]["metadata"]["kernel_phase"] for node in children] == [
        "stage1", "stage2",
    ]
    assert set(children[1]["deps"]) == {children[0]["node_id"], *parent["deps"]}
    assert all(node["lineage"]["source_call_id"] == parent_id for node in children)
