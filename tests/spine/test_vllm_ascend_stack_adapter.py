"""Tier-0 tests for the vLLM-Ascend runtime stack plugin.

[Role] Verify ``VllmAscendStackAdapter``, ``VllmAscendLaunchCodec``, 1:1
  ``vllm-ascend`` quantization/packing/prepared-weights, per-native-layer typed
  ``CacheStateABI`` across five model geometries, speculative/RecoverSSM rules,
  domain/ISA guards, ``simulation`` vs ``modeling`` rewrite tracks, and live
  one-step ``plan_iteration`` scheduling.
[Invariants] Pure in-memory Tier-0 tests; zero subprocesses, network, or host
  mutations; zero literal SKU strings (imported from ``_support.axes``).
[Timeouts & Locks] Sub-second execution; no locks.
[Side-Effect Policy] Read-only over repository assets; zero persistent writes.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest

from _support.axes import GEN3_SKU, GEN5_DT_SKU
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.checkpoint import (
    CacheQuantRule,
    CheckpointProvenance,
    CheckpointQuantMap,
    LayerQuantException,
    TensorQuantRule,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.stack import (
    IterationSchedulerState,
    LaunchConfigCodec,
    RuntimeStackAdapter,
    SchedulerDone,
    WaitForEvent,
)
from mindie_sim.schema.workload import IterationPlan
from mindie_sim.stacks.vllm_ascend import (
    VllmAscendLaunchCodec,
    VllmAscendStackAdapter,
    clamp_fia_speculative_tokens,
    compute_aclgraph_capture_sizes,
)

pytestmark = pytest.mark.tier0

STACK_PKG_DIR = Path(__file__).resolve().parents[2] / "mindie_sim" / "stacks" / "vllm_ascend"


@lru_cache(maxsize=4)
def _hw(sku: str) -> Hardware:
    return parse_case(
        compose_case(
            "qwen3-32b", hardware=sku, tp=1, layers=1,
            prompt_tokens=16, output_tokens=2, concurrency=1,
        )
    ).machine


def _port_map(plan) -> dict[tuple[str, int, str], object]:
    return {coord: abi for coord, abi in plan.port_abis}


def test_adapter_protocol_capabilities_defaults_and_layer_boundary() -> None:
    adapter = VllmAscendStackAdapter()
    assert isinstance(adapter, RuntimeStackAdapter)
    assert isinstance(adapter.launch_codec, LaunchConfigCodec)

    cap = adapter.capabilities()
    assert cap.stack_id == "vllm-ascend"
    assert cap.supported_domains == ("llm", "vlm")
    assert cap.supports_domain("llm") and cap.supports_domain("vlm")
    assert not cap.supports_domain("diffusion")
    assert cap.has_launch_codec is True
    assert "vllm_ascend_mla_prolog_v3" in cap.supported_recipe_ids

    a5_hw, a3_hw = _hw(GEN5_DT_SKU), _hw(GEN3_SKU)
    assert adapter.default_checkpoint_for("kimi-k3.1", a5_hw) == "Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp"
    assert adapter.default_checkpoint_for("kimi-k3.1", a3_hw) == "Eco-Tech/Kimi-K3.1-Pro-w4a8"
    assert adapter.default_checkpoint_for("qwen3.8-flash-next", a5_hw) == "Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp"
    assert adapter.default_checkpoint_for("qwen3.8-flash-next", a3_hw) == "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp"
    assert adapter.default_checkpoint_for("hunyuan-video-dit", a5_hw) is None

    # AG010: zero export/compose imports anywhere inside stacks/vllm_ascend/.
    for py_file in sorted(STACK_PKG_DIR.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("mindie_sim.export", "mindie_sim.compose"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("mindie_sim.export", "mindie_sim.compose"))


def test_diffusion_dit_model_rejected_with_domain_mismatch() -> None:
    adapter = VllmAscendStackAdapter()
    dit = load_model("hunyuan-video-dit")
    a5_hw = _hw(GEN5_DT_SKU)
    par = ParallelSpec(tp=1)
    with pytest.raises(ModelingError, match="stack_domain_mismatch"):
        adapter.resolve_options(
            dit, a5_hw, par, role="colocated", track="shipping",
            requested_options={}, field_origins={},
        )


def test_kimi_k31_a5_mxfp_and_a3_w4a8_quant_packing_cache_and_recoverssm() -> None:
    adapter = VllmAscendStackAdapter()
    model = load_model("kimi-k3.1")
    a5_hw, a3_hw = _hw(GEN5_DT_SKU), _hw(GEN3_SKU)
    par_tp4 = ParallelSpec(tp=4, ep=4)

    ckpt_a5 = CheckpointQuantMap(
        provenance=CheckpointProvenance("Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp", "modelslim_description"),
        native_float_dtype="bf16",
        model_quant_type="W4A8_MXFP",
        rules=(
            TensorQuantRule("model.layers.*.self_attn.q_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.k_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.v_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.b_proj.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.self_attn.f_a_proj.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.self_attn.f_b_proj.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.self_attn.g_proj.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.self_attn.o_proj.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.self_attn.q_a_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.q_b_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.kv_a_proj_with_mqa.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.kv_b_proj.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.self_attn.indexer.wk.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.self_attn.indexer.weights_proj.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.mlp.latent_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.mlp.experts.0.gate_proj.weight", "W4A8_MXFP"),
            TensorQuantRule("model.layers.*.mlp.experts.0.down_proj.weight", "W4A8_MXFP"),
            TensorQuantRule("model.layers.*.mlp.shared_experts.gate_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.mlp.shared_experts.up_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.mlp.shared_experts.down_proj.weight", "W8A8_MXFP8"),
        ),
        cache_rule=CacheQuantRule(
            kv_tag="FP8", index_tag="FP8", kv_scale_dtype="fp32", index_scale_dtype="fp32", provenance="checkpoint",
        ),
    )

    opts = adapter.resolve_options(
        model, a5_hw, par_tp4, role="pd_decode", track="shipping",
        requested_options={"num_speculative_tokens": 5, "spec_method": "dspark"},
        field_origins={"num_speculative_tokens": "cli_flag"},
    )
    qplan = adapter.resolve_quant_instantiation(
        model, ckpt_a5, a5_hw, par_tp4, opts, role="pd_decode", phase="decode",
    )
    ports = _port_map(qplan)

    # KDA layer 1 (native_layer_id=1): MXFP8 q/k/v split from BF16 b/f_a/g & f_b.
    assert ports[("target", 1, "q_proj")].linear_method == "mx_fp8"
    assert ports[("target", 1, "b_proj")].linear_method == "dense"
    assert ports[("target", 1, "f_b_proj")].linear_method == "dense"
    assert ports[("target", 1, "self_attention_res_proj")].linear_method == "dense"
    assert ports[("target", 0, "output_attn_res_proj")].linear_method == "dense"

    packed_names = {name: shards for name, shards, _ in qplan.packed_groups}
    assert packed_names["target.L1.in_proj_qkv"] == ("q_proj", "k_proj", "v_proj")
    assert packed_names["target.L1.fused_bfg_proj"] == ("b_proj", "f_a_proj", "g_proj")

    # MLA+DSA layer 4 (index=3, native_layer_id=4): BF16 mla.{g,o,kv_b}, indexer.{wk,weights}; MXFP8 latent_moe; MXFP4 experts.
    assert ports[("target", 4, "mla.g_proj")].linear_method == "dense"
    assert ports[("target", 4, "mla.o_proj")].linear_method == "dense"
    assert ports[("target", 4, "mla.kv_b_proj")].linear_method == "dense"
    assert ports[("target", 4, "indexer.wk")].linear_method == "dense"
    assert ports[("target", 4, "indexer.weights_proj")].linear_method == "dense"
    assert packed_names["target.L4.wk_weights_proj"] == ("indexer.wk", "indexer.weights_proj")
    assert ports[("target", 4, "mlp.latent_moe")].linear_method == "mx_fp8"
    assert ports[("target", 4, "mlp.latent_moe")].weight_block == (1, 32)
    assert ports[("target", 4, "mlp.experts.w13")].linear_method == "mx_fp4"

    # Prepared runtime-only BF16 F = F_b @ F_a at TP4: (3072, 7168) * 2 B = 44,040,192 B.
    prep_f = next(p for p in qplan.prepared_weights if p.allocation_id == "kimi_k3.kda.prepared_f_matmul")
    assert prep_f.local_shape == (3072, 7168)
    assert prep_f.dtype == "bf16"
    assert prep_f.local_shape[0] * prep_f.local_shape[1] * 2 == 44040192
    assert len(prep_f.layer_indices) == 69
    assert prep_f.retains_source_tensors is True and prep_f.lifetime == "persistent"

    # Exact CacheStateABI counts & byte widths: 69 KDA + 24 MLA (656 B) + 24 DSA Indexer K (no phantom weights row).
    assert len(qplan.recurrent_state_abi) == 69
    assert len(qplan.kv_cache_abi) == 24
    assert len(qplan.index_cache_abi) == 24
    assert qplan.kv_cache_abi[0].total_plane_bytes() == 656
    assert {p.plane_id for p in qplan.index_cache_abi[0].planes} == {"index_k", "index_k_scale"}

    # Shipping merged + DSpark RecoverSSM active with all 69 owners & batched launch.
    lplan = adapter.select_lowering_plan(model, qplan, a5_hw, par_tp4, opts)
    assert lplan.scheduling.speculative_proposal_count == 5
    assert lplan.scheduling.draft_invocation_count == 1
    assert len(lplan.scheduling.post_accept_commits) == 1
    commit = lplan.scheduling.post_accept_commits[0]
    assert commit.owner_count == 69
    assert commit.batched_kernel_launches == 1
    assert commit.published_version == commit.base_version + 1
    assert all(k % 6 == 0 for k in opts.graph_capture_keys)

    # PD matching-K rule on prefill keeps K=5 for Kimi-K3.1 DSpark.
    opts_p = adapter.resolve_options(
        model, a5_hw, ParallelSpec(tp=1, pcp=4, ep=4), role="pd_prefill", track="shipping",
        requested_options={"num_speculative_tokens": 5, "spec_method": "dspark"},
        field_origins={},
    )
    assert opts_p.speculative is not None and opts_p.speculative.draft_len == 5

    # A3 W4A8 + C8 check (FRACTAL_NZ + W4A8 post-load scale prepared weight).
    ckpt_a3 = CheckpointQuantMap(
        provenance=CheckpointProvenance("Eco-Tech/Kimi-K3.1-Pro-w4a8", "modelslim_description"),
        native_float_dtype="bf16",
        model_quant_type="W4A8_DYNAMIC",
        rules=(
            TensorQuantRule("model.layers.*.self_attn.q_proj.weight", "W8A8_DYNAMIC"),
            TensorQuantRule("model.layers.*.mlp.experts.0.gate_proj.weight", "W4A8_DYNAMIC"),
            TensorQuantRule("model.layers.*.mlp.experts.0.down_proj.weight", "W4A8_DYNAMIC"),
        ),
        cache_rule=CacheQuantRule(kv_tag="INT8", index_tag="INT8", kv_scale_dtype="fp32", provenance="checkpoint"),
    )
    opts_a3 = adapter.resolve_options(
        model, a3_hw, par_tp4, role="pd_decode", track="shipping", requested_options={}, field_origins={},
    )
    qplan_a3 = adapter.resolve_quant_instantiation(
        model, ckpt_a3, a3_hw, par_tp4, opts_a3, role="pd_decode", phase="decode",
    )
    ports_a3 = _port_map(qplan_a3)
    assert ports_a3[("target", 4, "mlp.experts.w13")].linear_method == "w4a8"
    assert ports_a3[("target", 4, "mlp.experts.w13")].weight_layout == "FRACTAL_NZ"
    assert any(p.allocation_id == "w4a8.post_load_int64_scale_and_bias" for p in qplan_a3.prepared_weights)
    assert qplan_a3.kv_cache_abi[0].total_plane_bytes() == 656


def test_qwen38_flash_next_mxfp_keeps_linear_ple_hc_and_mtp_bf16() -> None:
    adapter = VllmAscendStackAdapter()
    model = load_model("qwen3.8-flash-next")
    a5_hw = _hw(GEN5_DT_SKU)
    par = ParallelSpec(tp=8, ep=8)

    ckpt = CheckpointQuantMap(
        provenance=CheckpointProvenance("Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp", "modelslim_description"),
        native_float_dtype="bf16",
        model_quant_type="W4A8_MXFP",
        rules=(
            TensorQuantRule("model.layers.*.self_attn.q_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.k_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.v_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.o_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.index_qk_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.linear_attn.in_proj_qkv.weight", "FLOAT"),
            TensorQuantRule("model.layers.*.mlp.experts.0.gate_proj.weight", "W4A8_MXFP"),
            TensorQuantRule("model.layers.*.mlp.experts.0.down_proj.weight", "W4A8_MXFP"),
            TensorQuantRule("model.layers.*.mlp.shared_experts.gate_proj.weight", "FLOAT"),
            TensorQuantRule("mtp.layers.*.mlp.experts.0.gate_proj.weight", "FLOAT"),
            TensorQuantRule("mtp.layers.*.mlp.experts.0.down_proj.weight", "FLOAT"),
        ),
        cache_rule=CacheQuantRule(kv_tag="FP8", index_tag="FP8", provenance="checkpoint"),
    )
    opts = adapter.resolve_options(
        model, a5_hw, par, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    qplan = adapter.resolve_quant_instantiation(
        model, ckpt, a5_hw, par, opts, role="colocated", phase="decode",
    )
    ports = _port_map(qplan)

    # Layer 0 (GDN): linear_attn, hyper_connection, shared_expert stay BF16; routed experts MXFP4.
    assert ports[("target", 0, "linear_attn.in_proj_qkv")].linear_method == "dense"
    assert ports[("target", 0, "hyper_connection.pre_proj")].linear_method == "dense"
    assert ports[("target", 1, "ple.embed_proj")].linear_method == "dense"
    assert ports[("target", 0, "mlp.experts.w13")].linear_method == "mx_fp4"

    # Layer 3 (Full-Attn): q/k/v/o + index_qk MXFP8 & packed.
    assert ports[("target", 3, "q_proj")].linear_method == "mx_fp8"
    assert ports[("target", 3, "o_proj")].linear_method == "mx_fp8"
    assert ports[("target", 3, "index_q_proj")].linear_method == "mx_fp8"
    packed = {name: shards for name, shards, _ in qplan.packed_groups}
    assert packed["target.L3.qkv_proj"] == ("q_proj", "k_proj", "v_proj")
    assert packed["target.L3.index_qk_proj"] == ("index_q_proj", "index_k_proj")

    # Draft MTP (all 512 draft experts) stays strictly BF16.
    assert ports[("draft", 0, "mlp.experts.w13")].linear_method == "dense"
    assert ports[("draft", 0, "mlp.experts.w2")].linear_method == "dense"

    # Exact CacheStateABI counts: 36 GDN + 1 PLE conv (layer 1) + 12 GQA + 12 KPool (factor=4, tail=3).
    gdn_abis = [a for a in qplan.recurrent_state_abi if a.component_role == "recurrent_state"]
    ple_abis = [a for a in qplan.recurrent_state_abi if a.component_role == "ple_conv_state"]
    assert len(gdn_abis) == 36
    assert len(ple_abis) == 1 and ple_abis[0].native_layer == 1
    assert len(qplan.kv_cache_abi) == 12
    assert len(qplan.index_cache_abi) == 12
    assert qplan.index_cache_abi[0].pooling_ratio == 4
    assert qplan.index_cache_abi[0].tail_slots == 3
    assert qplan.index_cache_abi[0].compressed_append_units(5, 4) == 1


def test_glm53_flash_packed_qkvbfg_a_layer_exceptions_and_nope_mla_bytes() -> None:
    adapter = VllmAscendStackAdapter()
    model = load_model("glm-5.3-flash")
    a5_hw = _hw(GEN5_DT_SKU)
    par = ParallelSpec(tp=8, ep=8)
    mla_layers = tuple(range(3, 44, 4))

    ckpt_fp16_scale = CheckpointQuantMap(
        provenance=CheckpointProvenance("Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8", "modelslim_description"),
        native_float_dtype="bf16",
        model_quant_type="W8A8_MXFP8",
        rules=(
            TensorQuantRule("model.layers.*.linear_attn.q_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.linear_attn.k_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.linear_attn.v_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.linear_attn.b_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.linear_attn.f_a_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.linear_attn.g_a_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.linear_attn.o_proj.weight", "W8A8_MXFP8"),
            TensorQuantRule("model.layers.*.self_attn.o_proj.weight", "W8A8_MXFP8"),
        ),
        layer_exceptions=(
            LayerQuantException("model.layers.*.self_attn.o_proj.weight", "target", mla_layers, "FLOAT"),
        ),
        cache_rule=CacheQuantRule(kv_tag="FP8", index_tag="FP8", kv_scale_dtype="fp16", provenance="checkpoint"),
    )

    opts = adapter.resolve_options(
        model, a5_hw, par, role="colocated", track="shipping",
        requested_options={"use_replayssm": True}, field_origins={},
    )
    qplan = adapter.resolve_quant_instantiation(
        model, ckpt_fp16_scale, a5_hw, par, opts, role="colocated", phase="decode",
    )
    packed = {name: shards for name, shards, _ in qplan.packed_groups}
    assert packed["target.L0.fused_qkvbfg_a_proj"] == (
        "q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj",
    )
    ports = _port_map(qplan)
    assert ports[("target", 0, "o_proj")].linear_method == "mx_fp8"
    for mla_idx in mla_layers:
        assert ports[("target", mla_idx, "mla.o_proj")].linear_method == "dense"

    # 34 KDA + 11 NoPE MLA (520 B with FP16 scales, zero rotary) + 11 Indexer K.
    assert len(qplan.recurrent_state_abi) == 34
    assert len(qplan.kv_cache_abi) == 11
    assert len(qplan.index_cache_abi) == 11
    assert qplan.kv_cache_abi[0].total_plane_bytes() == 520
    assert all(p.plane_id != "mla_rotary" for p in qplan.kv_cache_abi[0].planes)

    # With FP32 scales on NoPE MLA -> 528 B.
    ckpt_fp32_scale = CheckpointQuantMap(
        provenance=ckpt_fp16_scale.provenance,
        native_float_dtype="bf16",
        rules=ckpt_fp16_scale.rules,
        cache_rule=CacheQuantRule(kv_tag="FP8", index_tag="FP8", kv_scale_dtype="fp32", provenance="checkpoint"),
    )
    qplan_fp32 = adapter.resolve_quant_instantiation(
        model, ckpt_fp32_scale, a5_hw, par, opts, role="colocated", phase="decode",
    )
    assert qplan_fp32.kv_cache_abi[0].total_plane_bytes() == 528


def test_deepseek_v41_hf_quant_and_shared_cache_owners() -> None:
    adapter = VllmAscendStackAdapter()
    model = load_model("deepseek-v4.1-flash")
    a5_hw = _hw(GEN5_DT_SKU)
    par = ParallelSpec(dp=8, tp=1, ep=8)

    ckpt = CheckpointQuantMap(
        provenance=CheckpointProvenance(
            "deepseek-ai/DeepSeek-V4.1-Flash-1017", "hf_quantization_config", origin="hf_config_quantization",
        ),
        native_float_dtype="bf16",
        hf_quant_config=(("expert_dtype", "fp4"), ("quant_method", "fp8"), ("scale_fmt", "ue8m0")),
        ignored_modules=("lm_head",),
        cache_rule=CacheQuantRule(kv_tag="FP8", index_tag="FP8", kv_scale_dtype="fp32", provenance="checkpoint"),
    )
    opts = adapter.resolve_options(
        model, a5_hw, par, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    qplan = adapter.resolve_quant_instantiation(
        model, ckpt, a5_hw, par, opts, role="colocated", phase="decode",
    )
    ports = _port_map(qplan)
    assert ports[("target", 0, "mla.q_a_proj")].linear_method == "mx_fp8"
    assert ports[("target", 0, "mlp.experts.w13")].linear_method == "mx_fp4"
    assert ports[("target", 0, "lm_head")].linear_method == "dense"

    window_abis = [a for a in qplan.kv_cache_abi if a.component_role == "window_kv"]
    long_abis = [a for a in qplan.kv_cache_abi if a.component_role == "compressed_kv"]
    engram_abis = [a for a in qplan.recurrent_state_abi if a.component_role == "engram_table"]
    assert len(window_abis) == 40
    assert tuple(a.native_layer for a in long_abis) == (2, 8, 14, 20)
    assert tuple(a.native_layer for a in qplan.index_cache_abi) == (2, 8, 14, 20)
    assert any(a.consumer_keys for a in qplan.index_cache_abi)
    assert tuple(a.native_layer for a in engram_abis) == (1, 14)
    assert engram_abis[0].placement.memory_tier == "dram"


def test_qwen3_32b_c8_producer_faquant_pdmix_and_mxfp8_kv_page_layout() -> None:
    adapter = VllmAscendStackAdapter()
    model = load_model("qwen3-32b")
    a5_hw, a3_hw = _hw(GEN5_DT_SKU), _hw(GEN3_SKU)
    par_tp8 = ParallelSpec(tp=8)

    # 1. C8 producer (pd_prefill) keeps BF16 while consumer (pd_decode) uses INT8 + scales/offsets.
    ckpt_c8 = CheckpointQuantMap(
        provenance=CheckpointProvenance("Eco-Tech/Qwen3-32B-w8a8c8", "modelslim_description"),
        native_float_dtype="bf16",
        rules=(TensorQuantRule("model.layers.*.self_attn.q_proj.weight", "W8A8_MIX"),),
        auxiliary_buffers=("model.layers.0.self_attn.fa_k_scale",),
        cache_rule=CacheQuantRule(kv_tag="INT8", provenance="checkpoint"),
    )
    opts_p = adapter.resolve_options(
        model, a3_hw, par_tp8, role="pd_prefill", track="shipping", requested_options={}, field_origins={},
    )
    qplan_p = adapter.resolve_quant_instantiation(
        model, ckpt_c8, a3_hw, par_tp8, opts_p, role="pd_prefill", phase="prefill",
    )
    assert qplan_p.kv_cache_abi[0].planes[0].dtype == "bf16"
    assert any(k == "kv_cache_c8_producer" for k, _, _ in qplan_p.dispositions)
    assert dict(_port_map(qplan_p)[("target", 0, "q_proj")].weight_metadata)["dynamic_act"] is True

    opts_d = adapter.resolve_options(
        model, a3_hw, par_tp8, role="pd_decode", track="shipping", requested_options={}, field_origins={},
    )
    qplan_d = adapter.resolve_quant_instantiation(
        model, ckpt_c8, a3_hw, par_tp8, opts_d, role="pd_decode", phase="decode",
    )
    assert len(qplan_d.kv_cache_abi) == 64
    assert qplan_d.kv_cache_abi[0].planes[0].dtype == "int8"
    assert dict(_port_map(qplan_d)[("target", 0, "q_proj")].weight_metadata)["dynamic_act"] is False
    assert any(p.allocation_id == "faquant.fak_descale_float" for p in qplan_d.prepared_weights)

    # 2. QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL: K | K-scale | V | V-scale | padding, 33,792 B, stands FAQuant down.
    ckpt_mxfp = CheckpointQuantMap(
        provenance=CheckpointProvenance("Eco-Tech/Qwen3-32B-mxfp8-kv", "modelslim_description"),
        native_float_dtype="bf16",
        rules=(TensorQuantRule("model.layers.*.self_attn.q_proj.weight", "W8A8_MXFP8"),),
        auxiliary_buffers=("model.layers.0.self_attn.fa_k_scale", "model.layers.0.self_attn.v_cache_scale"),
        cache_rule=CacheQuantRule(kv_tag="QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL", provenance="checkpoint"),
    )
    opts_a5 = adapter.resolve_options(
        model, a5_hw, par_tp8, role="colocated", track="shipping",
        requested_options={"kv_block_size": 128}, field_origins={},
    )
    qplan_mxfp = adapter.resolve_quant_instantiation(
        model, ckpt_mxfp, a5_hw, par_tp8, opts_a5, role="colocated", phase="decode",
    )
    kv0 = qplan_mxfp.kv_cache_abi[0]
    assert tuple(p.plane_id for p in kv0.planes) == (
        "k_payload", "k_scale", "v_payload", "v_scale", "page_padding",
    )
    assert kv0.total_plane_bytes(include_padding=False) == 33792
    assert kv0.planes[1].write_trigger == "append"
    assert kv0.planes[3].write_trigger == "page_reassign"
    assert any(k == "fa_quant" and status == "downgraded" for k, status, _ in qplan_mxfp.dispositions)
    assert any(p.allocation_id == "mxfp_c8.v_cache_scale_float_reciprocal" for p in qplan_mxfp.prepared_weights)


def test_blind_box_unquantized_projection_provenance_and_unknown_tag() -> None:
    adapter = VllmAscendStackAdapter()
    model = load_model("kimi-k3.1")
    a5_hw, a3_hw = _hw(GEN5_DT_SKU), _hw(GEN3_SKU)
    par = ParallelSpec(tp=4, ep=4)

    unquant = CheckpointQuantMap(
        provenance=CheckpointProvenance(
            "blindbox/kimi-k3.1-bf16", "unquantized_native", origin="unquantized_native",
        ),
        native_float_dtype="bf16",
    )
    opts_a5 = adapter.resolve_options(
        model, a5_hw, par, role="colocated", track="shipping",
        requested_options={"target_quant_projection": "mxfp8_mxfp4"}, field_origins={},
    )
    qplan = adapter.resolve_quant_instantiation(
        model, unquant, a5_hw, par, opts_a5, role="colocated", phase="decode",
    )
    ports = _port_map(qplan)
    assert qplan.provenance == "stack_default_quant_projection"
    assert ports[("target", 1, "q_proj")].linear_method == "mx_fp8"
    assert ports[("target", 4, "mlp.experts.w13")].linear_method == "mx_fp4"
    # quant_config=None modules strictly stay native BF16!
    assert ports[("target", 1, "self_attention_res_proj")].linear_method == "dense"
    assert ports[("target", 4, "mlp.gate")].linear_method == "dense"
    assert ports[("target", 4, "mla.kv_b_proj")].linear_method == "dense"
    assert ports[("target", 0, "embed_tokens")].linear_method == "dense"
    assert ports[("target", 0, "lm_head")].linear_method == "dense"

    # A3 rejects MXFP projection.
    opts_a3 = adapter.resolve_options(
        model, a3_hw, par, role="colocated", track="shipping",
        requested_options={"target_quant_projection": "mxfp8_mxfp4"}, field_origins={},
    )
    with pytest.raises(ModelingError, match="hw_isa_incompatible"):
        adapter.resolve_quant_instantiation(
            model, unquant, a3_hw, par, opts_a3, role="colocated", phase="decode",
        )

    # Unknown checkpoint tag fails loudly.
    bad_ckpt = CheckpointQuantMap(
        provenance=CheckpointProvenance("bad/ckpt", "modelslim_description"),
        rules=(TensorQuantRule("model.layers.*.self_attn.q_proj.weight", "W3A7_MYSTERY"),),
    )
    with pytest.raises(ModelingError, match="unknown_checkpoint_quant_tag"):
        adapter.resolve_quant_instantiation(
            model, bad_ckpt, a5_hw, par, opts_a5, role="colocated", phase="decode",
        )


def test_launch_codec_round_trip_deployment_mode_and_audit() -> None:
    codec = VllmAscendLaunchCodec()

    # Ambiguous script leaves deployment_mode_hint=None (never silently guesses).
    ambig_script = """
    export HCCL_INTRA_PCIE_ENABLE=1
    vllm serve /weights/Qwen3-32B \\
      --tensor-parallel-size 8 \\
      --max-model-len 65536 \\
      --enforce-eager \\
      --additional-config '{"enable_fused_mc2": 2, "multistream_overlap_shared_expert": true}'
    """
    ingress = codec.parse_launch_config(ambig_script)
    assert ingress.deployment_mode_hint is None
    assert ingress.parallel.tp == 8

    audit = codec.audit_launch_config(
        ingress,
        before_facts={"fits_hbm": False, "peak_hbm_bytes": 70 * (1024**3), "step_span_us": 50000.0},
        tuned_facts={
            "model_name": "qwen3-32b", "hw_family": "a5", "phase": "decode",
            "weight_format": "MXFP8", "kv_cache_dtype": "fp8_e4m3",
            "tp": 4, "dp": 2, "fits_hbm": True, "step_span_us": 20000.0,
        },
    )
    issue_ids = {item["issue_id"] for item in audit["issues_found"]}
    assert {
        "missing_quantization_bf16", "missing_kv_quant", "missing_aclgraph_full_decode",
        "mc2_mso_mutex_conflict", "a5_legacy_transport_override", "hbm_oom_in_original_script",
    }.issubset(issue_ids)

    # Render + round-trip with explicit pd_decode kv-transfer-config.
    rendered = codec.render_launch_config({
        "model_name": "kimi-k3.1", "model_ref": "Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp",
        "hw_family": "a5", "role": "pd_decode", "phase": "decode",
        "tp": 8, "dp": 1, "ep": 8, "dcp": 8,
        "weight_format": "MXFP8_MXFP4", "kv_cache_dtype": "fp8_e4m3",
        "spec_proposals": 5, "spec_method": "dspark", "has_mla": True,
        "multistream_overlap_shared_expert": True,
    })
    assert "HCCL_OP_EXPANSION_MODE=\"CCU_SCHED\"" in rendered
    assert "HCCL_INTRA_PCIE_ENABLE" not in rendered
    assert "enable_weight_nz_layout" not in rendered

    parsed_back = codec.parse_launch_config(rendered)
    assert parsed_back.deployment_mode_hint == "pd_decode"
    assert parsed_back.parallel.tp == 8
    assert parsed_back.parallel.cp == 8
    assert parsed_back.parallel.ep == 8

    # Prefill producer hint check.
    prefill_sh = codec.render_launch_config({
        "model_name": "kimi-k3.1", "hw_family": "a3", "role": "pd_prefill", "phase": "prefill",
        "tp": 1, "dp": 1, "pp": 4, "pcp": 4, "ep": 4, "weight_format": "W4A8", "kv_cache_dtype": "int8",
    })
    assert "CCU" not in prefill_sh
    assert codec.parse_launch_config(prefill_sh).deployment_mode_hint == "pd_prefill"


def test_simulation_vs_modeling_tracks_full_and_piecewise_and_plan_iteration() -> None:
    adapter = VllmAscendStackAdapter()
    model = load_model("qwen3-32b")
    a5_hw = _hw(GEN5_DT_SKU)
    par = ParallelSpec(tp=4)
    ckpt = CheckpointQuantMap(
        provenance=CheckpointProvenance("Eco-Tech/Qwen3-32B-w8a8c8", "modelslim_description"),
        native_float_dtype="bf16",
    )

    # FIA ceiling clamp (K+1 <= 16) and n*(K+1) capture sizes.
    clamped_k, warn = clamp_fia_speculative_tokens(20)
    assert clamped_k == 15 and warn is not None
    assert compute_aclgraph_capture_sizes([1, 2, 4], num_speculative_tokens=3) == (4, 8, 16)

    # Simulation mode: strictly ("config",), zero synthesized_target, FULL_AND_PIECEWISE preserved.
    opts_sim = adapter.resolve_options(
        model, a5_hw, par, role="pd_disaggregated", track="shipping",
        requested_options={
            "optimization_mode": "simulation",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "num_speculative_tokens": 3,
            "pd_kv_transfer_mode": "layerwise_overlap",
        },
        field_origins={},
    )
    assert opts_sim.graph_capture_mode == "full_and_piecewise"
    assert opts_sim.capture_rule is not None
    assert opts_sim.capture_rule.select_bucket(phase="decode", active_tokens=8) == (8, "full")
    assert opts_sim.capture_rule.select_bucket(phase="prefill", active_tokens=100)[1] == "piecewise"

    qplan = adapter.resolve_quant_instantiation(
        model, ckpt, a5_hw, par, opts_sim, role="pd_disaggregated", phase="decode",
    )
    lplan_sim = adapter.select_lowering_plan(model, qplan, a5_hw, par, opts_sim)
    assert lplan_sim.allowed_rewrite_tracks == ("config",)
    assert lplan_sim.allowed_rewrite_classes == frozenset({"shipping"})
    assert "synthesized_target" not in lplan_sim.allowed_rewrite_classes
    assert lplan_sim.scheduling.piecewise_split_points
    assert lplan_sim.scheduling.pd_kv_transfer is not None
    assert lplan_sim.scheduling.pd_kv_transfer.mode == "layerwise_overlap"
    assert len(lplan_sim.scheduling.pd_kv_transfer.layer_edges) == 64

    # Modeling mode unlocks all 4 rewrite tracks & all 4 kernel readiness classes.
    opts_mod = adapter.resolve_options(
        model, a5_hw, par, role="colocated", track="target",
        requested_options={"optimization_mode": "modeling"}, field_origins={},
    )
    lplan_mod = adapter.select_lowering_plan(model, qplan, a5_hw, par, opts_mod)
    assert lplan_mod.allowed_rewrite_tracks == ("config", "framework", "kernel_new", "kernel_tune")
    assert "synthesized_target" in lplan_mod.allowed_rewrite_classes
    assert "tiler_extension" in lplan_mod.allowed_rewrite_classes

    # Live one-step plan_iteration: IterationPlan, WaitForEvent, SchedulerDone.
    step = adapter.plan_iteration(
        model,
        IterationSchedulerState(
            now_ms=10.0,
            ready_request_ids=("req_low", "req_high"),
            request_priorities=(("req_low", 5), ("req_high", 1)),
            request_progress=(("req_low", 128, 0), ("req_high", 256, 0)),
            max_num_seqs=4,
            max_num_batched_tokens=64,
        ),
        opts_sim,
        phase="decode",
    )
    assert isinstance(step, IterationPlan)
    assert step.requests[0].request_id == "req_high"
    assert step.valid_tokens == 8

    wait = adapter.plan_iteration(
        model,
        IterationSchedulerState(now_ms=10.0, ready_request_ids=(), in_flight_completions=(("req_a", 15.5),)),
        opts_sim,
        phase="decode",
    )
    assert isinstance(wait, WaitForEvent) and wait.until_ms == 15.5

    done = adapter.plan_iteration(
        model,
        IterationSchedulerState(now_ms=20.0, ready_request_ids=()),
        opts_sim,
        phase="decode",
    )
    assert isinstance(done, SchedulerDone)


def test_h2a_to_h2b_curated_hf_and_unquantized_checkpoint_seam() -> None:
    """Real cross-module H2-A -> H2-B seam: HF quant, Step-5 ignored_modules, and blind-box projection."""
    from mindie_sim.catalog.checkpoint_reader import (
        lookup_curated_checkpoint,
        read_hf_quantization_config,
    )

    adapter = VllmAscendStackAdapter()
    a5_hw = _hw(GEN5_DT_SKU)
    par = ParallelSpec(tp=8, ep=8)

    # 1. DeepSeek-V4.1-Flash-1017 (hf_quantization_config): CONSTRUCTOR_UNQUANTIZED_ROLES stay FLOAT,
    #    W8A8_MXFP8 + group_size=32 + routed expert MXFP4 survive despite catch-all "*" rule.
    ds_model = load_model("deepseek-v4.1-flash")
    ds_ckpt = lookup_curated_checkpoint("deepseek-ai/DeepSeek-V4.1-Flash-1017")
    assert ds_ckpt is not None and ds_ckpt.source_format == "hf_quantization_config"
    ds_opts = adapter.resolve_options(
        ds_model, a5_hw, par, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    ds_plan = adapter.resolve_quant_instantiation(
        ds_model, ds_ckpt, a5_hw, par, ds_opts, role="colocated", phase="decode",
    )
    ds_ports = _port_map(ds_plan)
    assert ds_ports[("target", 0, "embed_tokens")].linear_method == "dense"
    assert ds_ports[("target", 0, "lm_head")].linear_method == "dense"
    assert ds_ports[("target", 3, "mla.kv_b_proj")].linear_method == "dense"
    assert ds_ports[("target", 3, "mlp.gate")].linear_method == "dense"
    assert ds_ports[("target", 3, "mla.q_a_proj")].linear_method == "mx_fp8"
    assert ds_ports[("target", 3, "mla.q_a_proj")].act_group_size == 32
    assert ds_ports[("target", 3, "mla.q_a_proj")].weight_block == (32, 32)
    assert ds_ports[("target", 3, "mlp.experts.w13")].linear_method == "mx_fp4"
    assert ds_ports[("target", 3, "mlp.experts.w2")].linear_method == "mx_fp4"

    # 2. Step-5-Preview (hf_quantization_config + modules_to_not_convert): L88..90 moe.experts stay FLOAT.
    step5_model = load_model("step-5-preview")
    step5_ckpt = lookup_curated_checkpoint("stepfun-ai/Step-5-Preview")
    assert step5_ckpt is not None and step5_ckpt.source_format == "hf_quantization_config"
    step5_opts = adapter.resolve_options(
        step5_model, a5_hw, par, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    step5_plan = adapter.resolve_quant_instantiation(
        step5_model, step5_ckpt, a5_hw, par, step5_opts, role="colocated", phase="decode",
    )
    step5_ports = _port_map(step5_plan)
    assert step5_ports[("target", 4, "mlp.experts.w13")].linear_method == "fp8"
    assert step5_ports[("target", 4, "mlp.gate")].linear_method == "dense"
    for tail_layer in (88, 89, 90):
        assert step5_ports[("target", tail_layer, "mlp.experts.w13")].linear_method == "dense"
        assert step5_ports[("target", tail_layer, "mlp.experts.w2")].linear_method == "dense"

    # 3. Unquantized native read via read_hf_quantization_config activates target_quant_projection
    #    with provenance="stack_default_quant_projection" while keeping CONSTRUCTOR_UNQUANTIZED_ROLES FLOAT.
    qwen_model = load_model("qwen3-32b")
    unq_ckpt = read_hf_quantization_config({"model_type": "qwen3"}, checkpoint_id="blindbox/qwen3-bf16")
    assert unq_ckpt.source_format == "unquantized_native"
    qwen_opts = adapter.resolve_options(
        qwen_model, a5_hw, ParallelSpec(tp=4), role="colocated", track="shipping",
        requested_options={"target_quant_projection": "mxfp8"}, field_origins={},
    )
    qwen_plan = adapter.resolve_quant_instantiation(
        qwen_model, unq_ckpt, a5_hw, ParallelSpec(tp=4), qwen_opts, role="colocated", phase="decode",
    )
    qwen_ports = _port_map(qwen_plan)
    assert qwen_plan.provenance == "stack_default_quant_projection"
    assert qwen_ports[("target", 0, "embed_tokens")].linear_method == "dense"
    assert qwen_ports[("target", 0, "lm_head")].linear_method == "dense"
    assert qwen_ports[("target", 0, "q_proj")].linear_method == "mx_fp8"
    assert qwen_ports[("target", 0, "mlp.gate_proj")].linear_method == "mx_fp8"


def test_hf_awq_gptq_compressed_quark_and_all_modelslim_weight_tags() -> None:
    """Zero silent BF16 degradation on AWQ/GPTQ/Compressed-Tensors/Quark and 100% weight tag coverage."""
    from mindie_sim.catalog.checkpoint_reader import (
        KNOWN_MODELSLIM_WEIGHT_TAGS,
        read_hf_quantization_config,
    )
    from mindie_sim.stacks.vllm_ascend.quant_policy import resolve_raw_tag_to_abi

    # Every declared weight tag in checkpoint_reader resolves cleanly on A5 without unknown_checkpoint_quant_tag.
    for raw_tag in sorted(KNOWN_MODELSLIM_WEIGHT_TAGS):
        abi = resolve_raw_tag_to_abi(raw_tag, hw_family="a5")
        if raw_tag not in {"FLOAT", "BF16", "FP16", "NONE"}:
            assert abi.linear_method != "dense", f"unexpected dense fallback for {raw_tag}"

    adapter = VllmAscendStackAdapter()
    model = load_model("qwen3-32b")
    a5_hw = _hw(GEN5_DT_SKU)
    par = ParallelSpec(tp=4)
    opts = adapter.resolve_options(
        model, a5_hw, par, role="colocated", track="shipping", requested_options={}, field_origins={},
    )

    hf_cases = (
        ("awq_w4", {"quant_method": "awq", "bits": 4, "group_size": 128}, "w4a16", "int4", (1, 128)),
        ("gptq_w4", {"quant_method": "gptq", "bits": 4, "group_size": 128}, "w4a16", "int4", (1, 128)),
        ("gptq_w8", {"quant_method": "gptq", "bits": 8, "group_size": 128}, "w4a16", "int8", (1, 128)),
        (
            "ct_w4a16",
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {"group_0": {"weights": {"type": "int", "num_bits": 4, "group_size": 128}}},
            },
            "w4a16", "int4", (1, 128),
        ),
        (
            "ct_w4a8",
            {
                "quant_method": "compressed-tensors",
                "format": "int-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {"type": "int", "num_bits": 4, "group_size": 128},
                        "input_activations": {"type": "int", "num_bits": 8},
                    },
                },
            },
            "w4a8", "int4", (1, 128),
        ),
        (
            "quark_fp8",
            {"quant_method": "quark", "global_quant_config": {"weight": {"dtype": "fp8_e4m3", "group_size": 128}}},
            "fp8", "fp8_e4m3", (1, 128),
        ),
        (
            "quark_mxfp4",
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"dtype": "fp4_e2m1", "group_size": 32},
                    "input_tensors": {"dtype": "fp8_e4m3"},
                },
            },
            "mx_fp4", "fp4_e2m1", (1, 32),
        ),
        (
            "quark_int8",
            {"quant_method": "quark", "global_quant_config": {"weight": {"dtype": "int8"}}},
            "w8a8", "int8", (),
        ),
    )
    for label, qcfg, exp_method, exp_wdtype, exp_blk in hf_cases:
        ckpt = read_hf_quantization_config(
            {"model_type": "qwen3", "torch_dtype": "bfloat16", "quantization_config": qcfg},
            checkpoint_id=f"org/{label}",
        )
        qplan = adapter.resolve_quant_instantiation(
            model, ckpt, a5_hw, par, opts, role="colocated", phase="decode",
        )
        ports = _port_map(qplan)
        q_abi = ports[("target", 0, "q_proj")]
        assert q_abi.linear_method == exp_method, label
        assert q_abi.weight_dtype == exp_wdtype, label
        if exp_blk:
            assert q_abi.weight_block == exp_blk, label
        # Constructor quant_config=None modules remain native BF16 across all 4 formats.
        assert ports[("target", 0, "embed_tokens")].linear_method == "dense", label
        assert ports[("target", 0, "lm_head")].linear_method == "dense", label


def test_curated_expert_kv_engram_and_speculative_conv_abi_fidelity() -> None:
    """Lock curated expert path aliases, dynamic KV/index tags, Engram tables, and speculative conv rollback."""
    from mindie_sim.catalog.checkpoint_reader import lookup_curated_checkpoint

    adapter = VllmAscendStackAdapter()
    a5_hw, a3_hw = _hw(GEN5_DT_SKU), _hw(GEN3_SKU)
    par8 = ParallelSpec(tp=8, ep=8)

    # 1. DeepSeek-V4.1-Flash A3 w4a8 (ffn.experts.*.w1/w2/w3) & HF FP4 (1, 32) expert block + 196.61 GB Engram tables.
    ds_model = load_model("deepseek-v4.1-flash")
    ds_w4a8_ckpt = lookup_curated_checkpoint("Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8")
    assert ds_w4a8_ckpt is not None
    ds_a3_opts = adapter.resolve_options(
        ds_model, a3_hw, par8, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    ds_a3_plan = adapter.resolve_quant_instantiation(
        ds_model, ds_w4a8_ckpt, a3_hw, par8, ds_a3_opts, role="colocated", phase="decode",
    )
    ds_a3_ports = _port_map(ds_a3_plan)
    assert ds_a3_ports[("target", 0, "mlp.experts.w13")].linear_method == "w4a8"
    assert ds_a3_ports[("target", 0, "mlp.experts.w2")].linear_method == "w4a8"
    assert ds_a3_ports[("target", 0, "mlp.shared_expert.gate_proj")].linear_method == "w8a8"
    assert ds_a3_ports[("target", 0, "mla.q_b_proj")].linear_method == "w8a8"
    engram_abis = [a for a in ds_a3_plan.recurrent_state_abi if a.component_role == "engram_table"]
    assert tuple(a.native_layer for a in engram_abis) == (1, 14)
    assert engram_abis[0].total_plane_bytes() == 196_611_158_016
    assert engram_abis[1].total_plane_bytes() == 196_616_541_184

    ds_hf_ckpt = lookup_curated_checkpoint("deepseek-ai/DeepSeek-V4.1-Flash-1017")
    assert ds_hf_ckpt is not None
    ds_a5_opts = adapter.resolve_options(
        ds_model, a5_hw, par8, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    ds_a5_ports = _port_map(adapter.resolve_quant_instantiation(
        ds_model, ds_hf_ckpt, a5_hw, par8, ds_a5_opts, role="colocated", phase="decode",
    ))
    assert ds_a5_ports[("target", 3, "mla.q_a_proj")].weight_block == (32, 32)
    assert ds_a5_ports[("target", 3, "mlp.experts.w13")].weight_block == (1, 32)

    # 2. Step-5 ModelSlim moe.{gate,up,down}_proj resolves MXFP8 on L3..87 and FLOAT on L88..90.
    step5_model = load_model("step-5-preview")
    step5_ms_ckpt = lookup_curated_checkpoint("Eco-Tech/Step-5-Preview-w8a8-mxfp8")
    assert step5_ms_ckpt is not None
    step5_opts = adapter.resolve_options(
        step5_model, a5_hw, par8, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    step5_ports = _port_map(adapter.resolve_quant_instantiation(
        step5_model, step5_ms_ckpt, a5_hw, par8, step5_opts, role="colocated", phase="decode",
    ))
    assert step5_ports[("target", 10, "mlp.experts.w13")].linear_method == "mx_fp8"
    assert step5_ports[("target", 10, "mlp.experts.w2")].linear_method == "mx_fp8"
    for tail_l in (88, 89, 90):
        assert step5_ports[("target", tail_l, "mlp.experts.w13")].linear_method == "dense"
        assert step5_ports[("target", tail_l, "mlp.experts.w2")].linear_method == "dense"

    # 3. GLM-5.3 MXFP8_DYNAMIC/INT8_DYNAMIC KV+index & DeepSeek-R1 QuaRot FAKQuant KV + appended L61 MTP.
    glm_model = load_model("glm-5.3-flash")
    glm_mx_ckpt = lookup_curated_checkpoint("Eco-Tech/GLM-5.3-W8A8C8-mxfp8")
    assert glm_mx_ckpt is not None
    glm_a5_opts = adapter.resolve_options(
        glm_model, a5_hw, par8, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    glm_mx_plan = adapter.resolve_quant_instantiation(
        glm_model, glm_mx_ckpt, a5_hw, par8, glm_a5_opts, role="colocated", phase="decode",
    )
    assert glm_mx_plan.kv_cache_abi[0].planes[0].dtype == "fp8_e4m3"
    assert glm_mx_plan.kv_cache_abi[0].planes[1].dtype == "e8m0"
    assert glm_mx_plan.index_cache_abi[0].planes[0].dtype == "fp8_e4m3"

    glm_c8_ckpt = lookup_curated_checkpoint("Eco-Tech/GLM-5.3-w8a8c8")
    assert glm_c8_ckpt is not None
    glm_a3_opts = adapter.resolve_options(
        glm_model, a3_hw, par8, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    glm_c8_plan = adapter.resolve_quant_instantiation(
        glm_model, glm_c8_ckpt, a3_hw, par8, glm_a3_opts, role="colocated", phase="decode",
    )
    assert glm_c8_plan.kv_cache_abi[0].planes[0].dtype == "int8"
    assert glm_c8_plan.index_cache_abi[0].planes[0].dtype == "int8"

    from mindie_sim.compose.inputs import ensure_native_mla_draft

    r1_model = ensure_native_mla_draft(load_model("deepseek-r1"), 1)
    r1_ckpt = lookup_curated_checkpoint("Eco-Tech/DeepSeek-R1-0528-w4a8c8-mtp-QuaRot")
    assert r1_ckpt is not None
    r1_opts = adapter.resolve_options(
        r1_model, a3_hw, par8, role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    r1_plan = adapter.resolve_quant_instantiation(
        r1_model, r1_ckpt, a3_hw, par8, r1_opts, role="colocated", phase="decode",
    )
    assert r1_plan.kv_cache_abi[0].planes[0].dtype == "int8"
    r1_ports = _port_map(r1_plan)
    assert r1_ports[("draft", 0, "mlp.experts.w13")].linear_method == "w8a8"
    assert r1_ports[("draft", 0, "mlp.experts.w2")].linear_method == "w8a8"

    # 4. Qwen3.8 GDN speculative conv rollback residency (122,880 B at K=3 vs 61,440 B at K=0) + replay_records.
    qwen_model = load_model("qwen3.8-flash-next")
    qwen_ckpt = lookup_curated_checkpoint("Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp")
    assert qwen_ckpt is not None
    par1 = ParallelSpec(tp=1, ep=1)
    qwen_k3_opts = adapter.resolve_options(
        qwen_model, a5_hw, par1, role="pd_decode", track="proposed",
        requested_options={"num_speculative_tokens": 3},
        field_origins={"num_speculative_tokens": "cli_flag"},
    )
    qwen_k3_plan = adapter.resolve_quant_instantiation(
        qwen_model, qwen_ckpt, a5_hw, par1, qwen_k3_opts, role="pd_decode", phase="decode",
    )
    gdn0 = next(a for a in qwen_k3_plan.recurrent_state_abi if a.component_role == "recurrent_state")
    plane_bytes = {p.plane_id: p.bytes_per_unit for p in gdn0.planes}
    assert plane_bytes["conv_history"] == 122_880
    assert plane_bytes["replay_records"] == 197_376
