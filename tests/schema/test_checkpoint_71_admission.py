"""Admission contracts for the 71-checkpoint curated dataset and streaming reader.

[Role] Verify lossless loading and QuantSpec lowering across all 71 curated
  on-disk checkpoints (66 ModelSlim + 5 HF quantization_config), bounded
  chunked streaming of ModelSlim descriptors, non-ModelSlim HF formats
  (AWQ, GPTQ, Quark, compressed-tensors, unquantized native), exact
  TensorPrecisionABI emission on GEMM ops, and strict unknown-tag failure.
[Invariants] Tier-0 pure in-memory and tmp_path execution only; zero network
  I/O, zero cache writes on analytical reads, zero runtime-adapter imports,
  and zero host home directory paths in packaged data or provenance.
[Timeouts & Locks] Pure CPU tests completing in <2s; no external locks.
[Side-Effect Policy] Writes only inside pytest tmp_path fixtures.
"""
from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
import re
import urllib.request

import pytest

from mindie_sim.catalog.checkpoint_reader import (
    CURATED_CHECKPOINTS_SCHEMA,
    classify_tensor_categories,
    curated_weight_pattern_map,
    deserialize_curated_entry,
    iter_json_object_items,
    list_curated_checkpoint_ids,
    lookup_curated_checkpoint,
    quant_spec_from_checkpoint_map,
    read_checkpoint_quant_map,
    read_hf_quantization_config,
    read_modelslim_description_file,
    read_modelslim_description_mapping,
    resolve_checkpoint_tensor_tag,
    serialize_curated_entry,
)
from mindie_sim.catalog.npu_checkpoints import (
    DEFAULT_NPU_CHECKPOINTS,
    SCANNED_QUANT_PATTERNS,
    classify_weight_tensor_quant,
    resolve_default_checkpoint,
)
from mindie_sim.catalog.ops import gemm_op, precision_abi_for_gemm
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import TensorPrecisionABI
from mindie_sim.schema.parallel import ParallelSpec

pytestmark = pytest.mark.tier0

EXPECTED_TOTAL_CURATED = 71
EXPECTED_MODELSLIM_CURATED = 66
EXPECTED_HF_CURATED = 5
EXPECTED_EXCEPTION_REPOS = 12
MAX_PACKAGED_JSON_BYTES = 250 * 1024

REQUIRED_15_MODELSLIM_TAGS = frozenset({
    "FLOAT",
    "W8A8",
    "W8A8_DYNAMIC",
    "W4A8_DYNAMIC",
    "W8A8_MXFP8",
    "W4A8_MXFP",
    "W4A4_MXFP4",
    "W8A8_MIX",
    "FAQuant",
    "C8",
    "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",
    "INT8_DYNAMIC",
    "FP8_DYNAMIC",
    "MXFP8_DYNAMIC",
    "INT8_SYM",
})

PRIVATE_IPV4_RE = re.compile(
    r"(?:^|[^0-9.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?:[^0-9.]|$)"
)


def _home_path_regex() -> re.Pattern[str]:
    roots = "|".join(("Users", "home", "root"))
    return re.compile(rf"(?:/(?:{roots})/|[A-Za-z]:\\Users\\)")


def test_packaged_curated_dataset_all_71_entries_and_hygiene() -> None:
    """All 71 packaged entries load losslessly within size/privacy budgets."""
    data_ref = resources.files("mindie_sim.catalog").joinpath("data", "curated_checkpoints.json")
    raw_bytes = data_ref.read_bytes()
    assert len(raw_bytes) < MAX_PACKAGED_JSON_BYTES
    raw_text = raw_bytes.decode("utf-8")
    assert not _home_path_regex().search(raw_text)
    assert not PRIVATE_IPV4_RE.search(raw_text)
    assert "vllm-ascend" not in raw_text.lower()
    assert "vllm_ascend" not in raw_text.lower()

    reader_path = Path(resources.files("mindie_sim.catalog").joinpath("checkpoint_reader.py"))
    reader_src = reader_path.read_text(encoding="utf-8").lower()
    assert "vllm-ascend" not in reader_src
    assert "vllm_ascend" not in reader_src

    envelope = json.loads(raw_text)
    assert envelope["schema_version"] == CURATED_CHECKPOINTS_SCHEMA
    assert envelope["count"] == EXPECTED_TOTAL_CURATED

    ids = list_curated_checkpoint_ids()
    assert len(ids) == EXPECTED_TOTAL_CURATED

    modelslim_count = 0
    hf_count = 0
    exception_repos: list[str] = []
    observed_tags: set[str] = set()
    observed_cache_tags: set[str] = set()

    for repo_id in ids:
        qmap = lookup_curated_checkpoint(repo_id)
        assert qmap is not None, repo_id
        assert qmap.checkpoint_id == repo_id
        assert not _home_path_regex().search(qmap.provenance.checkpoint_id)
        assert not _home_path_regex().search(qmap.provenance.descriptor_path)

        rebuilt, _non_weight = deserialize_curated_entry(repo_id, serialize_curated_entry(qmap))
        assert rebuilt.digest() == qmap.digest(), repo_id

        if qmap.source_format == "modelslim_description":
            modelslim_count += 1
        elif qmap.source_format == "hf_quantization_config":
            hf_count += 1

        if qmap.layer_exceptions:
            exception_repos.append(repo_id)

        for rule in qmap.rules:
            observed_tags.add(rule.raw_tag)
        for exc in qmap.layer_exceptions:
            observed_tags.add(exc.raw_tag)
        if qmap.cache_rule.kv_tag:
            observed_cache_tags.add(qmap.cache_rule.kv_tag)
        if qmap.cache_rule.index_tag:
            observed_cache_tags.add(qmap.cache_rule.index_tag)
        for _layer_id, kv_layer_tag in qmap.cache_rule.layer_scoped_kv:
            observed_tags.add(kv_layer_tag)
        for _layer_id, idx_layer_tag in qmap.cache_rule.layer_scoped_index:
            observed_tags.add(idx_layer_tag)

        qspec = quant_spec_from_checkpoint_map(qmap)
        assert qspec.name
        assert not (
            qspec.weight_format == "bf16"
            and qspec.expert_weight_format == "bf16"
            and not qspec.category_overrides
        ), f"silent_bf16_degradation:{repo_id}"

    assert modelslim_count == EXPECTED_MODELSLIM_CURATED
    assert hf_count == EXPECTED_HF_CURATED
    assert len(exception_repos) == EXPECTED_EXCEPTION_REPOS
    all_seen = observed_tags | observed_cache_tags
    missing_tags = REQUIRED_15_MODELSLIM_TAGS - all_seen
    assert not missing_tags, f"missing_tags={sorted(missing_tags)}"
    assert "FAKQuant" in observed_cache_tags


def test_hf_quantization_config_minimax_step5_and_deepseek_v41_never_degrade() -> None:
    """MiniMax-M3 MXFP8, Step-5 FP8, and DS-V4.1 hybrid parse cleanly in one pass."""
    mm_map = lookup_curated_checkpoint("MiniMax/MiniMax-M3-MXFP8")
    assert mm_map is not None
    assert mm_map.source_format == "hf_quantization_config"
    assert len(mm_map.ignored_modules) == 62
    mm_spec = quant_spec_from_checkpoint_map(mm_map)
    assert mm_spec.weight_format == "fp8_e4m3"
    assert mm_spec.activation_format == "fp8_e4m3"
    assert mm_spec.weight_block == (1, 32)
    assert mm_spec.scale_format == "fp8_e8m0"
    assert mm_spec.linear_method("attn") == "mx_fp8"
    assert resolve_checkpoint_tensor_tag(mm_map, mm_map.ignored_modules[0]) == "FLOAT"
    assert resolve_checkpoint_tensor_tag(mm_map, "model.layers.5.self_attn.q_proj.weight") == "W8A8_MXFP8"

    s5_map = lookup_curated_checkpoint("stepfun-ai/Step-5-Preview")
    assert s5_map is not None
    assert s5_map.source_format == "hf_quantization_config"
    assert len(s5_map.ignored_modules) == 1406
    s5_spec = quant_spec_from_checkpoint_map(s5_map)
    assert s5_spec.weight_format == "fp8_e4m3"
    assert s5_spec.weight_block == (128, 128)
    assert s5_spec.linear_method("routed_expert") == "fp8"
    assert resolve_checkpoint_tensor_tag(s5_map, s5_map.ignored_modules[0]) == "FLOAT"
    assert resolve_checkpoint_tensor_tag(s5_map, "model.layers.10.moe.gate_proj.weight") == "FP8_DYNAMIC"

    ds41_map = lookup_curated_checkpoint("deepseek-ai/DeepSeek-V4.1-Flash-1017")
    assert ds41_map is not None
    assert ds41_map.source_format == "hf_quantization_config"
    ds41_spec = quant_spec_from_checkpoint_map(ds41_map)
    assert ds41_spec.weight_format == "fp8_e4m3"
    assert ds41_spec.weight_block == (32, 32)
    assert ds41_spec.scale_format == "fp8_e8m0"
    assert ds41_spec.expert_weight_format == "mxfp4"
    assert ds41_spec.expert_block == 32
    assert ds41_spec.linear_method("mla") == "mx_fp8"
    assert ds41_spec.linear_method("routed_expert") == "mx_fp4"

    for v4_repo in ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro-0813"):
        v4_map = lookup_curated_checkpoint(v4_repo)
        assert v4_map is not None
        v4_spec = quant_spec_from_checkpoint_map(v4_map)
        assert v4_spec.weight_format == "fp8_e4m3"
        assert v4_spec.weight_block == (128, 128)
        assert v4_spec.scale_format == "fp8_e8m0"


def test_flagship_kimi_qwen38_and_component_matching_contracts() -> None:
    """Kimi-K3.1 AttnRes/latent_moe bytes, Qwen3.8 exclusions, and exact component matching."""
    kimi_id = "Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp"
    kimi_pats = SCANNED_QUANT_PATTERNS[kimi_id]
    assert len(kimi_pats["W8A8_MXFP8"]) == 15
    assert len(kimi_pats["W4A8_MXFP"]) == 3
    assert len(kimi_pats["FLOAT"]) == 39

    kimi_map = lookup_curated_checkpoint(kimi_id)
    assert kimi_map is not None
    kimi_spec = quant_spec_from_checkpoint_map(kimi_map)
    assert kimi_spec.format_for("attn_res_proj") == "bf16"
    assert kimi_spec.linear_method("attn_res_proj") == "dense"
    assert kimi_spec.format_for("latent_moe") == "fp8_e4m3"
    assert kimi_spec.linear_method("latent_moe") == "mx_fp8"
    assert kimi_spec.format_for("routed_expert") == "mxfp4"
    assert kimi_spec.linear_method("routed_expert") == "mx_fp4"

    assert classify_tensor_categories("language_model.model.layers.0.self_attention_res_proj.weight", has_mla=True, has_kda=True) == ("attn_res_proj",)
    assert classify_tensor_categories("language_model.model.layers.0.mlp_res_proj.weight", has_mla=True, has_kda=True) == ("attn_res_proj",)
    assert classify_tensor_categories("language_model.model.layers.0.self_attn.b_proj.weight", has_mla=True, has_kda=True) == ("kda.b",)
    assert classify_tensor_categories("language_model.model.layers.0.self_attn.kv_b_proj.weight", has_mla=True, has_kda=True) == ("mla_absorb",)

    parallel = ParallelSpec(dp=1, tp=1, ep=1)
    attn_res_term = WeightTerm(name="attn_res_proj", category="attn_res_proj", rows=1, cols=7168, shard="replicated")
    latent_term = WeightTerm(name="latent_down", category="latent_moe", rows=3584, cols=7168, shard="replicated")
    assert term_bytes(attn_res_term, kimi_spec, parallel) == 14336
    assert term_bytes(latent_term, kimi_spec, parallel) == 26492928

    proj_rule = next(r for r in kimi_map.rules if r.pattern == "mm_projector.proj.*")
    assert proj_rule.presence.scope == "projector"
    assert proj_rule.presence.layer_indices == (0, 2)

    q38_map = lookup_curated_checkpoint("Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp")
    assert q38_map is not None
    q38_spec = quant_spec_from_checkpoint_map(q38_map)
    assert q38_spec.format_for("shared_expert") == "bf16"
    assert q38_spec.linear_method("shared_expert") == "dense"
    assert q38_spec.format_for("routed_expert") == "mxfp4"
    assert q38_spec.linear_method("routed_expert") == "mx_fp4"
    assert any(r.presence.scope == "draft" for r in q38_map.rules)
    q38_pats = curated_weight_pattern_map("Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp")
    assert any("shard_*" in pat for pat in q38_pats["FLOAT"])


def test_per_layer_exceptions_and_cache_rules_across_modelslim_flagships() -> None:
    """Verify GLM-5.3, DeepSeek-R1/V3, Qwen3-32B, Gemma-4, Step-5, and cache tags."""
    glm_mxfp8 = lookup_curated_checkpoint("Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8")
    assert glm_mxfp8 is not None
    mla_layers = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45)
    assert glm_mxfp8.layer_exceptions[0].layer_indices == mla_layers
    assert resolve_checkpoint_tensor_tag(glm_mxfp8, "model.language_model.layers.3.self_attn.o_proj.weight") == "W8A8_MXFP8"
    assert resolve_checkpoint_tensor_tag(glm_mxfp8, "model.language_model.layers.0.self_attn.o_proj.weight") == "FLOAT"

    glm_w8a8 = lookup_curated_checkpoint("Eco-Tech/GLM-5.3-Flash-w8a8")
    assert glm_w8a8 is not None
    assert resolve_checkpoint_tensor_tag(glm_w8a8, "model.language_model.layers.7.self_attn.o_proj.weight") == "W8A8_DYNAMIC"
    assert resolve_checkpoint_tensor_tag(glm_w8a8, "model.language_model.layers.2.self_attn.o_proj.weight") == "FLOAT"

    ds_r1 = lookup_curated_checkpoint("Eco-Tech/DeepSeek-R1-0528-w4a8c8-mtp-QuaRot")
    assert ds_r1 is not None
    assert resolve_checkpoint_tensor_tag(ds_r1, "model.layers.61.mlp.experts.0.down_proj.weight") == "W8A8_DYNAMIC"
    assert resolve_checkpoint_tensor_tag(ds_r1, "model.layers.3.mlp.experts.0.down_proj.weight") == "W4A8_DYNAMIC"
    assert ds_r1.cache_rule.kv_tag == "FAKQuant"
    assert len(ds_r1.cache_rule.layer_scoped_kv) == 48
    assert {"fa_q.scale", "fa_k.scale", "fa_v.scale", "deq_scale"} <= set(ds_r1.auxiliary_buffers)

    q32 = lookup_curated_checkpoint("Eco-Tech/Qwen3-32B-w8a8c8")
    assert q32 is not None
    assert q32.layer_exceptions[0].layer_indices == (1, 2, 6, 7, 11, 43, 44, 45, 62)
    assert resolve_checkpoint_tensor_tag(q32, "model.layers.1.mlp.down_proj.weight") == "FLOAT"
    assert resolve_checkpoint_tensor_tag(q32, "model.layers.0.mlp.down_proj.weight") == "W8A8_MIX"
    assert q32.cache_rule.kv_tag == "C8"
    assert {"kv_cache_scale", "kv_cache_offset", "weight_scale", "weight_offset", "input_scale", "input_offset", "deq_scale"} <= set(q32.auxiliary_buffers)

    g4 = lookup_curated_checkpoint("Eco-Tech/gemma-4-31B-it-w4a4-mxfp4")
    assert g4 is not None
    assert resolve_checkpoint_tensor_tag(g4, "model.language_model.layers.9.mlp.down_proj.weight") == "FLOAT"
    assert resolve_checkpoint_tensor_tag(g4, "model.language_model.layers.0.mlp.down_proj.weight") == "W4A4_MXFP4"
    g4_spec = quant_spec_from_checkpoint_map(g4)
    assert g4_spec.format_for("dense_mlp") == "mxfp4"
    assert g4_spec.for_category("dense_mlp").activation_format == "fp4_e2m1"
    assert g4_spec.linear_method("dense_mlp") == "mx_fp4"

    s5_mxfp8 = lookup_curated_checkpoint("Eco-Tech/Step-5-Preview-w8a8-mxfp8")
    assert s5_mxfp8 is not None
    assert resolve_checkpoint_tensor_tag(s5_mxfp8, "model.layers.88.moe.down_proj.weight") == "FLOAT"
    assert resolve_checkpoint_tensor_tag(s5_mxfp8, "model.layers.5.moe.down_proj.weight") == "W8A8_MXFP8"
    s5_mxfp8_spec = quant_spec_from_checkpoint_map(s5_mxfp8)
    assert s5_mxfp8_spec.format_for("routed_expert") == "fp8_e4m3"
    assert s5_mxfp8_spec.linear_method("routed_expert") == "mx_fp8"
    assert s5_mxfp8_spec.format_for("attn") == "bf16"
    assert s5_mxfp8_spec.linear_method("attn") == "dense"
    assert s5_mxfp8_spec.format_for("shared_expert") == "bf16"

    q38_27b = lookup_curated_checkpoint("Eco-Tech/Qwen3.8-27B-w8a8c8-mxfp8")
    assert q38_27b is not None
    assert q38_27b.cache_rule.kv_tag == "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"

    glm53_c8 = lookup_curated_checkpoint("Eco-Tech/GLM-5.3-W8A8C8-mxfp8")
    assert glm53_c8 is not None
    assert glm53_c8.cache_rule.kv_tag == "MXFP8_DYNAMIC"
    assert glm53_c8.cache_rule.index_tag == "FP8_DYNAMIC"

    glm52_c8 = lookup_curated_checkpoint("Eco-Tech/GLM-5.2-w8a8c8")
    assert glm52_c8 is not None
    assert glm52_c8.cache_rule.kv_tag == "INT8_DYNAMIC"
    assert glm52_c8.cache_rule.index_tag == "INT8_DYNAMIC"

    ds41_ms = lookup_curated_checkpoint("Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8")
    assert ds41_ms is not None
    engram_rule = next(r for r in ds41_ms.rules if r.raw_tag == "INT8_SYM")
    assert engram_rule.presence.layer_indices == (1, 14)


def test_streaming_chunked_modelslim_reader_preserves_aux_exceptions_and_sha256(tmp_path: Path) -> None:
    """Chunked streaming decoder parses multi-thousand-entry files with tiny chunk size."""
    payload: dict[str, object] = {
        "model_quant_type": "W4A8_DYNAMIC",
        "version": "1.0.0",
        "group_size": 0,
        "kv_cache_type": "C8",
        "fa_quant_type": "FAKQuant",
        "metadata": {"nested": {"note": "escaped \"quote\" \\ slash"}, "list": [1, 2, 3]},
        "optional": {"rotation": True, "threshold": 0.125},
        "model.embed_tokens.weight": "FLOAT",
        "lm_head.weight": "FLOAT",
    }
    for layer in range(24):
        payload[f"model.layers.{layer}.self_attn.q_proj.weight"] = "W8A8_DYNAMIC"
        payload[f"model.layers.{layer}.self_attn.q_proj.weight_scale"] = "W8A8_DYNAMIC"
        payload[f"model.layers.{layer}.self_attn.q_proj.weight_offset"] = "W8A8_DYNAMIC"
        payload[f"model.layers.{layer}.self_attn.q_proj.input_scale"] = "W8A8_DYNAMIC"
        payload[f"model.layers.{layer}.self_attn.q_proj.input_offset"] = "W8A8_DYNAMIC"
        payload[f"model.layers.{layer}.self_attn.q_proj.deq_scale"] = "W8A8_DYNAMIC"
        payload[f"model.layers.{layer}.self_attn.quant_type"] = ""
        payload[f"model.layers.{layer}.self_attn.kv_cache_scale"] = "C8"
        payload[f"model.layers.{layer}.self_attn.fa_q.scale"] = "FAQuant"
        for exp in range(8):
            tag = "W8A8_DYNAMIC" if layer == 23 else "W4A8_DYNAMIC"
            payload[f"model.layers.{layer}.mlp.experts.{exp}.down_proj.weight"] = tag
            payload[f"model.layers.{layer}.mlp.experts.{exp}.down_proj.weight_scale"] = tag

    desc_file = tmp_path / "quant_model_description.json"
    desc_file.write_text(json.dumps(payload), encoding="utf-8")

    items_iter, hasher = iter_json_object_items(desc_file, chunk_size=128, force_stdlib=True)
    streamed = dict(items_iter)
    assert len(streamed) == len(payload)
    assert streamed["metadata"] == payload["metadata"]
    assert streamed["optional"] == payload["optional"]

    qmap_chunked = read_modelslim_description_file(desc_file, chunk_size=128, force_stdlib=True)
    qmap_via_dir = read_checkpoint_quant_map(tmp_path)
    assert qmap_chunked.digest() == qmap_via_dir.digest()
    assert qmap_chunked.provenance.source_sha256 == hasher.hexdigest()
    assert not _home_path_regex().search(qmap_chunked.provenance.checkpoint_id)
    assert not _home_path_regex().search(qmap_chunked.provenance.descriptor_path)
    assert resolve_checkpoint_tensor_tag(qmap_chunked, "model.layers.23.mlp.experts.5.down_proj.weight") == "W8A8_DYNAMIC"
    assert resolve_checkpoint_tensor_tag(qmap_chunked, "model.layers.5.mlp.experts.5.down_proj.weight") == "W4A8_DYNAMIC"
    assert {"weight", "weight_scale", "weight_offset", "input_scale", "input_offset", "deq_scale", "kv_cache_scale", "fa_q.scale"} <= set(qmap_chunked.auxiliary_buffers)


def test_hf_formats_awq_gptq_quark_compressed_tensors_and_unquantized_native(tmp_path: Path) -> None:
    """All standard HF quantization configs and native unquantized checkpoints parse accurately."""
    awq_map = read_hf_quantization_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "awq", "bits": 4, "group_size": 128,
                "modules_to_not_convert": ["lm_head"],
            },
        },
        checkpoint_id="fixture/awq-4b",
    )
    awq_spec = quant_spec_from_checkpoint_map(awq_map)
    assert awq_map.source_format == "hf_quantization_config"
    assert awq_spec.weight_format == "int4"
    assert awq_spec.activation_format == "bf16"
    assert awq_spec.weight_block == (1, 128)
    assert awq_map.ignored_modules == ("lm_head",)

    gptq_map = read_hf_quantization_config(
        {
            "torch_dtype": "float16",
            "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True},
        },
        checkpoint_id="fixture/gptq-4b",
    )
    gptq_spec = quant_spec_from_checkpoint_map(gptq_map)
    assert gptq_spec.weight_format == "int4"
    assert gptq_spec.activation_format == "fp16"
    assert gptq_spec.native_format == "fp16"

    quark_map = read_hf_quantization_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": "fp8_e8m0"},
                    "input_tensors": {"dtype": "fp8_e4m3"},
                },
                "exclude": ["lm_head"],
            },
        },
        checkpoint_id="fixture/quark-mxfp8",
    )
    quark_spec = quant_spec_from_checkpoint_map(quark_map)
    assert quark_spec.weight_format == "fp8_e4m3"
    assert quark_spec.scale_format == "fp8_e8m0"
    assert quark_spec.linear_method("attn") == "mx_fp8"

    ct_fp4 = read_hf_quantization_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "ignore": ["lm_head"],
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "group_size": 32, "scale_dtype": "fp8_e8m0"},
                        "input_activations": {"num_bits": 8, "type": "float"},
                    }
                },
            },
        },
        checkpoint_id="fixture/ct-mxfp4",
    )
    ct_spec = quant_spec_from_checkpoint_map(ct_fp4)
    assert ct_spec.weight_format == "mxfp4"
    assert ct_spec.activation_format == "fp8_e4m3"
    assert ct_spec.weight_block == (1, 32)
    assert ct_spec.linear_method("attn") == "mx_fp4"

    native_dir = tmp_path / "native-fp16"
    native_dir.mkdir()
    (native_dir / "config.json").write_text(json.dumps({"torch_dtype": "float16", "hidden_size": 4096}), encoding="utf-8")
    native_map = read_checkpoint_quant_map(native_dir)
    assert native_map.source_format == "unquantized_native"
    assert native_map.native_float_dtype == "fp16"
    native_spec = quant_spec_from_checkpoint_map(native_map)
    assert native_spec.weight_format == "fp16"
    assert native_spec.linear_method("attn") == "dense"


def test_gemm_op_emits_tensor_precision_abi_and_fusion_compatibility() -> None:
    """GEMM ops attach one canonical TensorPrecisionABI matching QuantSpec truth."""
    q_mx = quant_preset("mxfp8_mxfp4")
    op_exp = gemm_op(
        op_id="moe_down", deps=(), stage=0, layer=0, requests=("r0",),
        m=32, n=7168, k=2048, category="routed_expert", quant=q_mx, phase="decode",
    )
    op_lat = gemm_op(
        op_id="latent_down", deps=(), stage=0, layer=0, requests=("r0",),
        m=32, n=3584, k=7168, category="latent_moe", quant=q_mx, phase="decode",
    )
    op_res = gemm_op(
        op_id="attn_res", deps=(), stage=0, layer=0, requests=("r0",),
        m=32, n=1, k=7168, category="attn_res_proj", quant=q_mx, phase="decode",
    )

    assert isinstance(op_exp.precision_abi, TensorPrecisionABI)
    assert op_exp.precision_abi.linear_method == "mx_fp4"
    assert op_exp.precision_abi.weight_dtype == "mxfp4"
    assert op_exp.precision_abi.input_dtype == "fp8_e4m3"
    assert op_exp.precision_abi.weight_scale_dtype == "fp8_e8m0"
    assert op_exp.precision_abi.act_group_size == 32
    assert dict(op_exp.metadata)["linear_method"] == "mx_fp4"
    assert dict(op_exp.metadata)["precision_abi_signature"] == op_exp.precision_abi.abi_signature

    assert op_lat.precision_abi is not None
    assert op_lat.precision_abi.linear_method == "mx_fp8"
    assert op_lat.precision_abi.weight_dtype == "fp8_e4m3"
    assert op_lat.precision_abi.input_dtype == "fp8_e4m3"

    assert op_res.precision_abi is not None
    assert op_res.precision_abi.linear_method == "dense"
    assert op_res.precision_abi.weight_dtype == "bf16"
    assert op_res.precision_abi.weight_scale_dtype == ""

    assert not op_exp.precision_abi.is_fusion_compatible(op_res.precision_abi)
    op_lat_up = gemm_op(
        op_id="latent_up", deps=(), stage=0, layer=0, requests=("r0",),
        m=32, n=7168, k=3584, category="latent_moe", quant=q_mx, phase="decode",
    )
    assert op_lat_up.precision_abi is not None
    assert op_lat.precision_abi.is_fusion_compatible(op_lat_up.precision_abi)

    abi_w8 = precision_abi_for_gemm(quant_preset("w8a8"), category="attn")
    assert abi_w8.linear_method == "w8a8"
    assert abi_w8.accum_dtype == "int32"
    abi_w4a4 = precision_abi_for_gemm(quant_preset("w4a4_mxfp4"), category="routed_expert")
    assert abi_w4a4.linear_method == "mx_fp4"
    assert abi_w4a4.input_dtype == "fp4_e2m1"


def test_unknown_tags_fail_hard_analytical_reads_are_pure_and_a3_defaults_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown quant tags raise ModelingError; analytical reads touch neither network nor cache."""
    with pytest.raises(ModelingError, match="unknown_checkpoint_quant_tag:MYSTERY_QUANT_9000"):
        read_modelslim_description_mapping({"model.layers.0.self_attn.q_proj.weight": "MYSTERY_QUANT_9000"})

    with pytest.raises(ModelingError, match="unknown_checkpoint_quant_tag:MYSTERY_QUANT_9000"):
        classify_weight_tensor_quant(
            "model.layers.0.self_attn.q_proj.weight",
            {"model.layers.0.self_attn.q_proj.weight": "MYSTERY_QUANT_9000"},
        )

    with pytest.raises(ModelingError, match="unknown_checkpoint_quant_tag:alien_quant_v9"):
        read_hf_quantization_config(
            {"quantization_config": {"quant_method": "alien_quant_v9"}},
            checkpoint_id="fixture/alien",
        )

    def _forbid_urlopen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("analytical checkpoint read must never invoke network I/O")

    cache_root = tmp_path / "isolated_cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_urlopen)

    qmap = read_checkpoint_quant_map("Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp")
    assert qmap.checkpoint_id == "Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp"
    assert not cache_root.exists()

    # Calibrated unscanned A3 checkpoints remain intact.
    for model_key in ("deepseek-v4-flash", "qwen3.5-397b-a17b"):
        ckpt = resolve_default_checkpoint(model_key, hw_family="a3")
        assert ckpt is not None
        assert ckpt.build_quant_spec().weight_format in {"int8", "int4"}
        assert ckpt.build_checkpoint_quant_map().checkpoint_id == ckpt.repo_id

    for (_model, _hw), spec in DEFAULT_NPU_CHECKPOINTS.items():
        built_map = spec.build_checkpoint_quant_map()
        built_spec = spec.build_quant_spec()
        assert built_map.checkpoint_id == spec.repo_id
        assert built_spec.name


def test_nested_text_config_quantization_and_per_layer_native_overrides() -> None:
    """Nested VLM text_config.quantization_config parses cleanly and preserves native_layers delta."""
    from dataclasses import replace
    from mindie_sim.catalog.weights import stage_weight_bytes
    from mindie_sim.model_catalog import load_model
    from mindie_sim.schema.parallel import ParallelSpec

    nested_cfg = {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "text_config": {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "mxfp4",
                "modules_to_not_convert": ["lm_head"],
            },
        },
    }
    nested_map = read_hf_quantization_config(nested_cfg, checkpoint_id="fixture/nested-mxfp4")
    assert nested_map.source_format == "hf_quantization_config"
    nested_spec = quant_spec_from_checkpoint_map(nested_map)
    assert nested_spec.weight_format == "mxfp4"

    s5_map = lookup_curated_checkpoint("Eco-Tech/Step-5-Preview-w8a8-mxfp8")
    assert s5_map is not None
    s5_spec = quant_spec_from_checkpoint_map(s5_map)
    overrides = s5_spec.layer_quant_overrides_dict()
    assert set((overrides.get("native_layers") or {}).keys()) == {"88", "89", "90"}

    s5_model = load_model("step-5-preview")
    with_overrides = sum(stage_weight_bytes(replace(s5_model, quant=s5_spec), ParallelSpec(tp=1, ep=1)))
    without_overrides = sum(stage_weight_bytes(
        replace(s5_model, quant=replace(s5_spec, layer_quant_overrides=())),
        ParallelSpec(tp=1, ep=1),
    ))
    delta_gib = (with_overrides - without_overrides) / (1024**3)
    assert delta_gib == pytest.approx(18.5625, rel=0.05)
