"""Unit tests for xLLM custom_xllm_math operator lowerings in mindie_sim."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.diffusion_ops import (
    MINDIE_SD_CUSTOM_OPS,
    compute_mindie_sd_effective_denoising_steps,
    lower_mindie_sd_custom_op,
)
from mindie_sim.prediction.diffusion_ops import lower_op as sd_lower_op
from mindie_sim.prediction.custom_framework_ops import XLLM_CUSTOM_OPS, XLLM_VENDOR, lower_op
from mindie_sim.prediction.implementations import IMPLEMENTATIONS, lower_operators, select
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def _make_op(
    family: str,
    *,
    recipe: str = "",
    local_shape: tuple[int, ...] = (16, 4096),
    flops: float = 1_000_000.0,
    mte2_bytes: float = 65536.0,
    mte3_bytes: float = 32768.0,
    vector_ops: float = 1024.0,
    dtype: str = "bf16",
) -> OpSpec:
    return OpSpec(
        op_id=f"test.{family}",
        family=family,
        phase="decode",
        stage=0,
        layer=0,
        requests=("req-0",),
        semantic_shape=local_shape,
        local_shape=local_shape,
        padded_shape=local_shape,
        launch_shape=local_shape,
        flops=flops,
        hbm_bytes=mte2_bytes + mte3_bytes,
        mte2_bytes=mte2_bytes,
        mte3_bytes=mte3_bytes,
        vector_ops=vector_ops,
        recipe=recipe or family,
        dtype=dtype,
    )


def test_all_six_custom_xllm_math_ops_registered_and_lowered():
    expected_ops = {
        "smla_va",
        "sparse_flash_attention_lse",
        "moe_grouped_matmul_swiglu_quant",
        "mega_gdn_decode",
        "onerec_final_beam_select",
        "reshape_and_cache_a5",
    }
    assert set(XLLM_CUSTOM_OPS) == expected_ops
    assert "xllm_custom_xllm_math" in IMPLEMENTATIONS

    # 1. smla_va
    smla = lower_op(_make_op("mla_attention", local_shape=(8, 512)), op_name="smla_va", soc_family="Ascend910_93")
    smla_meta = dict(smla.metadata)
    assert smla.recipe == "smla_va"
    assert smla_meta["native_vendor"] == XLLM_VENDOR
    assert smla_meta["native_kind"] == "SmlaVa"
    assert "attn_out" in smla.returns

    # 2. sparse_flash_attention_lse
    base_sfa = _make_op("indexed_attention", local_shape=(16, 128), mte3_bytes=4096.0)
    sfa = lower_op(base_sfa, op_name="sparse_flash_attention_lse", return_lse=True)
    sfa_meta = dict(sfa.metadata)
    assert sfa.recipe == "sparse_flash_attention_lse"
    assert sfa_meta["native_kind"] == "SparseFlashAttentionLse"
    assert "softmax_lse" in sfa.returns
    assert sfa.mte3_bytes == base_sfa.mte3_bytes + 2 * 16 * 4
    assert sfa_meta["xllm_attention_geometry"]["softmax_stats_bytes"] == 128
    assert {"softmax_max", "softmax_sum"} <= set(sfa.returns)

    # 3. moe_grouped_matmul_swiglu_quant. The source operator is a complete
    # fused occurrence; an ordinary grouped gate/up matrix is not reinterpreted.
    base_moe = replace(
        _make_op("grouped_gemm", recipe="moe_grouped_matmul_swiglu_quant",
                 local_shape=(32, 256, 128), dtype="int8"),
        metadata=(("matrix_group_rows", (16, 0, 16)),
                  ("weight_format", "int8"),
                  ("weight_scale_format", "fp32"),
                  ("activation_scale_format", "fp32"),
                  ("output_format", "int8"),
                  ("output_scale_format", "fp32")))
    moe = lower_op(base_moe, op_name="moe_grouped_matmul_swiglu_quant")
    moe_meta = dict(moe.metadata)
    assert moe.recipe == "moe_grouped_matmul_swiglu_quant"
    assert moe_meta["native_kind"] == "MoeGroupedMatmulSwigluQuant"
    assert moe_meta["xllm_swiglu_quant_fused"] is True
    assert moe.dtype == "int8" and moe.local_shape == (32, 128)
    assert moe.returns == ("y", "y_scale")
    assert moe.matrix_flops == (("int8", 2 * 32 * 256 * 128),)
    assert moe.mte3_bytes == 32 * 128 + 32 * 4 + 32 * 256 * 4
    assert moe.mte2_bytes == (32 * 128 + 2 * 256 * 128 + 2 * 256 * 4
                              + 32 * 4 + 3 * 8 + 32 * 256 * 4)
    with pytest.raises(ModelingError, match="requires_declared_fusion_region"):
        lower_op(_make_op("grouped_gemm", local_shape=(32, 256, 128), dtype="int8"),
                 op_name="moe_grouped_matmul_swiglu_quant")

    # 4. mega_gdn_decode
    gdn_op = replace(
        _make_op("gdn_decode", local_shape=(4, 768)),
        metadata=(("qkv_shape", (4, 768)), ("z_shape", (4, 2, 128)),
                  ("num_k_heads", 2), ("conv_state_shape", (4, 3, 768)),
                  ("ssm_state_shape", (4, 2, 128, 128)),
                  ("conv_weight_shape", (4, 768)),
                  ("b_shape", (4, 2)), ("a_shape", (4, 2)),
                  ("a_log_shape", (2,)), ("dt_bias_shape", (2,)),
                  ("read_state_indices", (0, -1, 2, 3)),
                  ("write_state_indices", (0, 1, 2, 3)),
                  ("read_versions", (1, 1, 1, 1)),
                  ("write_versions", (1, 1, 1, 1)),
                  ("state_layout", "V,K")))
    gdn = lower_op(gdn_op, op_name="mega_gdn_decode")
    gdn_meta = dict(gdn.metadata)
    assert gdn.recipe == "mega_gdn_decode"
    assert gdn_meta["native_kind"] == "MegaGdnDecode"
    assert gdn.returns == ("convOut", "convStateOut", "ssmStateOut", "out")
    assert gdn.local_shape == (4, 2, 128)
    assert gdn_meta["xllm_mega_gdn_config"]["v_heads"] == 2
    assert gdn_meta["xllm_mega_gdn_config"]["read_state_indices"] == (0, -1, 2, 3)

    # 5. onerec_final_beam_select
    beam = lower_op(_make_op("beam_select", local_shape=(4, 1024), mte3_bytes=128.0), beam_width=64)
    beam_meta = dict(beam.metadata)
    assert beam.recipe == "onerec_final_beam_select"
    assert beam_meta["native_kind"] == "OnerecFinalBeamSelect"
    assert beam_meta["beam_width"] == 64
    assert {"out_token_ids", "out_token_index", "out_log_probs",
            "out_sequence"} <= set(beam.returns)
    assert beam.mte3_bytes == 3 * 64 * 4
    assert {"out_token_ids", "out_token_index", "out_log_probs",
            "out_sequence"} <= set(beam.returns)

    # 6. reshape_and_cache_a5
    cache_a5 = lower_op(
        _make_op("kv_cache_write", local_shape=(20, 128), mte3_bytes=1024.0),
        soc_family="Ascend950",
        kv_page_size=128,
    )
    cache_meta = dict(cache_a5.metadata)
    assert cache_a5.recipe == "reshape_and_cache_a5"
    assert cache_meta["native_kind"] == "ReshapeAndCacheA5"
    assert cache_meta["aligned_page_rows"] == 128
    assert "xtensor_kv_page" in cache_a5.returns


def test_lower_operators_applies_xllm_bindings_and_rejects_invalid_soc():
    ops = (
        _make_op("smla_va", local_shape=(8, 512)),
        _make_op("reshape_and_cache_a5", local_shape=(16, 128)),
    )
    bindings = {
        "smla_va": "xllm_smla_va",
        "reshape_and_cache_a5": {
            "implementation": "xllm_reshape_and_cache_a5",
            "options": {"soc_family": "Ascend950", "kv_page_size": 128},
        },
    }
    lowered = lower_operators(ops, bindings)
    assert len(lowered) == 2
    assert dict(lowered[0].metadata)["implementation_binding"] == {
        "selector": "smla_va",
        "implementation": "xllm_smla_va",
    }
    assert dict(lowered[1].metadata)["implementation_binding"] == {
        "selector": "reshape_and_cache_a5",
        "implementation": "xllm_reshape_and_cache_a5",
    }

    # reshape_and_cache_a5 is A5-only when soc_family is specified
    with pytest.raises(ModelingError, match="xllm_custom_op_soc_unsupported"):
        lower_op(
            _make_op("reshape_and_cache_a5", local_shape=(16, 128)),
            soc_family="Ascend910B",
        )

    sel = select({"smla_va": "xllm_custom_xllm_math"}, "smla_va")
    assert sel is not None
    assert sel.name == "xllm_custom_xllm_math"

    # Specific xllm_<op> implementation aliases auto-inject op_name even on shared families
    shared_ops = (
        _make_op("mla_attention", local_shape=(8, 512)),
        _make_op("elementwise", local_shape=(4, 1024)),
    )
    shared_lowered = lower_operators(
        shared_ops,
        {
            "mla_attention": "xllm_sparse_flash_attention_lse",
            "elementwise": {"implementation": "xllm_onerec_final_beam_select",
                            "options": {"beam_width": 32}},
        },
    )
    assert shared_lowered[0].recipe == "sparse_flash_attention_lse"
    assert dict(shared_lowered[0].metadata)["native_kind"] == "SparseFlashAttentionLse"
    assert shared_lowered[1].recipe == "onerec_final_beam_select"
    assert dict(shared_lowered[1].metadata)["native_kind"] == "OnerecFinalBeamSelect"


def test_mindie_sd_custom_ops_lowering_and_denoising_step_cache():
    assert set(MINDIE_SD_CUSTOM_OPS) == {
        "rainfusion",
        "ada_bsa",
        "eagle_bsa",
        "eagle_qbsa",
        "adalayernorm",
        "norm_rope_concat",
        "laurel",
        "quant_four_over_six_a5",
        "eagle_ffn",
    }

    attn_op = _make_op("indexed_attention", local_shape=(256, 128), flops=10_000_000.0, mte2_bytes=65536.0)
    rf = lower_mindie_sd_custom_op(attn_op, op_name="rainfusion",
                                   sparsity_ratio=0.75, block_size=64)
    assert rf["op_name"] == "mindie_sd_rainfusion"
    assert rf["bound"] == "cube"
    assert rf["cube_flops"] == pytest.approx(2_500_000.0)
    assert rf["sparsity_ratio"] == pytest.approx(0.75)

    role_meta = (("query_shape", (1, 256, 1, 128)),
                 ("key_shape", (1, 256, 1, 128)),
                 ("value_shape", (1, 256, 1, 128)),
                 ("return_softmax_lse", True))
    qbsa_op = replace(attn_op, metadata=role_meta + (("k_dtype", "fp8_e4m3"),
                                                       ("v_dtype", "fp8_e4m3")))
    bsa_op = replace(attn_op, metadata=role_meta + (("k_dtype", "bf16"),
                                                     ("v_dtype", "bf16")))
    qbsa = lower_mindie_sd_custom_op(qbsa_op, op_name="eagle_qbsa",
                                     sparsity_ratio=0.5, block_size=64)
    bsa = lower_mindie_sd_custom_op(bsa_op, op_name="eagle_bsa",
                                    sparsity_ratio=0.5, block_size=64)
    assert qbsa["memory_bytes"] < bsa["memory_bytes"]

    laurel_summary = lower_mindie_sd_custom_op(_make_op("gemm", local_shape=(128, 1024)), op_name="laurel", low_rank=32)
    assert laurel_summary["op_name"] == "mindie_sd_laurel"
    assert laurel_summary["cube_flops"] == pytest.approx(4 * 128 * 1024 * 32)

    q46 = lower_mindie_sd_custom_op(
        _make_op("quantize", local_shape=(64, 128)),
        op_name="quant_four_over_six_a5",
        soc_family="Ascend950",
    )
    assert q46["op_name"] == "mindie_sd_quant_four_over_six_a5"
    assert "mxfp4_scale" in q46["lowered_op"].returns

    with pytest.raises(ModelingError, match="mindie_sd_custom_op_soc_unsupported"):
        lower_mindie_sd_custom_op(
            _make_op("quantize", local_shape=(64, 128)),
            op_name="quant_four_over_six_a5",
            soc_family="Ascend910B",
        )

    # Auto-injection of op_name via lower_operators for specific mindie_sd_<op> aliases
    sd_lowered = lower_operators(
        (
            _make_op("indexed_attention", local_shape=(128, 128)),
            _make_op("rmsnorm", local_shape=(128, 1024)),
            _make_op("gemm", local_shape=(128, 1024)),
        ),
        {
            "indexed_attention": {"implementation": "mindie_sd_ada_bsa",
                                  "options": {"sparsity_ratio": 0.5, "block_size": 64}},
            "rmsnorm": "mindie_sd_adalayernorm",
            "gemm": "mindie_sd_eagle_ffn",
        },
    )
    assert [op.recipe for op in sd_lowered] == [
        "mindie_sd_ada_bsa",
        "mindie_sd_adalayernorm",
        "mindie_sd_eagle_ffn",
    ]

    # Regression [P2-1]: OpSpec(family="ada_bsa", recipe="indexed_attention") resolves to "ada_bsa"
    ada_implicit = lower_mindie_sd_custom_op(
        _make_op("ada_bsa", recipe="indexed_attention", local_shape=(128, 128)),
        sparsity_ratio=0.5, block_size=64)
    assert ada_implicit["op_name"] == "mindie_sd_ada_bsa"

    # DiTCache / TeaCache / TaylorSeer step calculation
    steps = compute_mindie_sd_effective_denoising_steps(infer_steps=50, warmup_steps=2, cache_interval=2)
    assert steps["infer_steps"] == 50
    assert steps["computed_steps"] == 26
    assert steps["cached_steps"] == 24
    assert steps["speedup_factor"] == pytest.approx(50.0 / 26.0)




def test_xllm_capability_registry_matches_registered_selectors_and_gates_generic_families():
    from mindie_sim.prediction.custom_framework_ops import capabilities
    from mindie_sim.prediction.implementations import IMPLEMENTATIONS

    caps = capabilities()
    assert set(caps["operators"]) == set(XLLM_CUSTOM_OPS)
    registered = {name for name, item in IMPLEMENTATIONS.items()
                  if item.module == "mindie_sim.prediction.custom_framework_ops"}
    assert set(caps["implementations"]) == registered
    for name, spec in caps["operators"].items():
        assert spec["native_kind"] == XLLM_CUSTOM_OPS[name]["native_kind"]
        assert spec["supported_socs"] and spec["input_dtypes"]
        assert set(spec["families"]) == (
            set(spec["selector_families"]) | set(spec["direct_only_families"]))
        for family in spec["selector_families"]:
            assert family in IMPLEMENTATIONS["xllm_custom_xllm_math"].operators
    with pytest.raises(ModelingError, match="requires_explicit_name"):
        lower_op(_make_op("grouped_gemm", local_shape=(32, 8192, 4096)))
    with pytest.raises(ModelingError, match="input_dtype_unsupported"):
        lower_op(_make_op("smla_va", dtype="int8"), op_name="smla_va")
    with pytest.raises(ModelingError, match="requires_positive_beam_width"):
        lower_op(_make_op("beam_select", local_shape=(4, 1024)), op_name="onerec_final_beam_select")
    with pytest.raises(ModelingError, match="requires_latent_sparse_or_mla_attention"):
        lower_op(_make_op("attention", recipe="flash_attention_decode",
                          local_shape=(8, 16392, 4, 256)), op_name="smla_va")
    with pytest.raises(ModelingError, match="gdn_decode_requires_declared_fusion_region"):
        lower_op(_make_op("recurrent_vector", recipe="recurrent_scan0",
                          local_shape=(8, 48, 128, 128)),
                 op_name="mega_gdn_decode")


def test_real_a5_paged_cache_write_uses_the_declared_occurrence():
    from mindie_sim.catalog.ops import cache_store_op

    view = {"key_shape": (8, 1, 256), "value_shape": (8, 1, 256),
            "key_dtype": "bf16", "value_dtype": "bf16",
            "slot_shape": (8,)}
    op = cache_store_op(
        op_id="s0.l3.cache", deps=(), stage=0, layer=3, requests=(),
        phase="decode", recipe="kv_store", rows=8, row_bytes=1024,
        dtype="bf16", kind="gqa",
        metadata=(("cache_tensor_view", view),
                  ("slot_mapping", tuple(range(8))),
                  ("cache_capacity_slots", 8)))
    lowered = lower_op(op, op_name="reshape_and_cache_a5",
                       soc_family="Ascend950", kv_page_size=128)
    meta = dict(lowered.metadata)
    assert lowered.recipe == "reshape_and_cache_a5"
    assert lowered.dtype == "bf16"
    assert meta["cache_row_basis"] == "cache_tensor_view_key_value_bytes"
    assert meta["cache_row_bytes"] == 1024
    assert meta["cache_valid_write_rows"] == 8
    assert meta["cache_page_capacity_bytes"] == 8 * 1024
    assert lowered.mte3_bytes == 8 * 1024
    assert "xtensor_kv_page" in lowered.returns
    assert lowered.deps == op.deps and lowered.requests == op.requests


def test_mindie_sd_capability_registry_and_explicit_declarations():
    from mindie_sim.prediction.diffusion_ops import capabilities
    from mindie_sim.prediction.implementations import IMPLEMENTATIONS

    caps = capabilities()
    assert set(caps["operators"]) == set(MINDIE_SD_CUSTOM_OPS)
    registered = {name for name, item in IMPLEMENTATIONS.items()
                  if item.module == "mindie_sim.prediction.diffusion_ops"}
    assert set(caps["implementations"]) == registered
    for name, spec in caps["operators"].items():
        assert spec["native_kind"] == MINDIE_SD_CUSTOM_OPS[name]["native_kind"]
        assert set(spec["families"]) == (
            set(spec["selector_families"]) | set(spec["direct_only_families"]))
        assert spec["input_dtypes"]
        for family in spec["selector_families"]:
            assert family in IMPLEMENTATIONS["mindie_sd_custom_ops"].operators

    with pytest.raises(ModelingError, match="requires_explicit_name"):
        sd_lower_op(_make_op("rmsnorm", local_shape=(128, 1024)))
    sparse = _make_op("indexed_attention", local_shape=(128, 128), flops=1_000_000.0,
                      mte2_bytes=65536.0)
    with pytest.raises(ModelingError, match="requires_declared_sparsity_ratio"):
        sd_lower_op(sparse, op_name="rainfusion")
    with pytest.raises(ModelingError, match="requires_declared_cache_dtype"):
        sd_lower_op(sparse, op_name="eagle_qbsa", sparsity_ratio=0.5, block_size=64)
    with pytest.raises(ModelingError, match="sparse_requires_declared_logical_flops"):
        sd_lower_op(_make_op("indexed_attention", local_shape=(128, 128), flops=0.0,
                             mte2_bytes=65536.0),
                    op_name="rainfusion", sparsity_ratio=0.5, block_size=64)


def test_mindie_sd_summary_does_not_invent_hardware_rates():
    op = _make_op("indexed_attention", local_shape=(256, 128),
                  flops=10_000_000.0, mte2_bytes=65536.0)
    summary = lower_mindie_sd_custom_op(
        op, op_name="rainfusion", sparsity_ratio=0.25, block_size=64)
    assert summary["sparsity_ratio"] == pytest.approx(0.25)
    assert summary["estimated_us"] is None
    assert summary["estimate_basis"] == "requires_declared_hardware_rates"
    declared = lower_mindie_sd_custom_op(
        op, hardware={"cube_tflops": 432.5, "vector_tflops": 27.0,
                      "hbm_bandwidth_gbps": 4000.0},
        op_name="rainfusion", sparsity_ratio=0.25, block_size=64)
    assert declared["estimated_us"] > 0
    assert declared["estimate_basis"] == "declared_hardware_roofline"
    incomplete = lower_mindie_sd_custom_op(
        op, hardware={"cube_tflops": 432.5}, op_name="rainfusion",
        sparsity_ratio=0.25, block_size=64)
    assert incomplete["estimated_us"] is None
    assert incomplete["estimate_basis"] == "declared_hardware_rates_incomplete"


def test_vendor_cache_write_uses_full_token_width_and_page_alignment():
    op = replace(_make_op("kv_cache_write", local_shape=(8, 2, 128),
                          mte3_bytes=0.0),
                 metadata=(("slot_mapping", tuple(range(8))),
                           ("cache_capacity_slots", 8)))
    lowered = lower_op(op, op_name="reshape_and_cache_a5",
                       soc_family="Ascend950", kv_page_size=128)
    meta = dict(lowered.metadata)
    assert meta["cache_row_basis"] == "vendor_shape_elements_times_dtype"
    assert meta["cache_row_bytes"] == 2 * 128 * 2
    assert lowered.mte3_bytes == 8 * 2 * 128 * 2


def test_cache_write_uses_valid_rows_not_page_capacity_or_invalid_slots():
    op = replace(_make_op("kv_cache_write", local_shape=(8, 2, 128),
                          mte3_bytes=0.0),
                 metadata=(("slot_mapping", tuple(range(8))),
                           ("cache_capacity_slots", 8)))
    page4 = lower_op(op, op_name="reshape_and_cache_a5",
                     soc_family="Ascend950", kv_page_size=4)
    page128 = lower_op(op, op_name="reshape_and_cache_a5",
                       soc_family="Ascend950", kv_page_size=128)
    assert page4.mte3_bytes == page128.mte3_bytes == 8 * 2 * 128 * 2
    assert (dict(page4.metadata)["cache_page_capacity_bytes"]
            == dict(page128.metadata)["cache_page_capacity_bytes"])

    slots = replace(
        _make_op("kv_cache_write", local_shape=(3, 2, 256), mte3_bytes=0.0),
        metadata=(("slot_mapping", (0, -1, 2)),
                  ("cache_capacity_slots", 2)))
    invalid = lower_op(slots, op_name="reshape_and_cache_a5",
                       soc_family="Ascend950", kv_page_size=128)
    meta = dict(invalid.metadata)
    assert meta["cache_valid_write_rows"] == 1
    assert meta["cache_invalid_slot_rows"] == 2
    assert invalid.mte3_bytes == 1 * 2 * 256 * 2
    assert meta["cache_slot_index_read_bytes"] == 3 * 4
    assert invalid.mte2_bytes == 1 * 2 * 256 * 2 + 3 * 4
    unknown = lower_op(_make_op("kv_cache_write", local_shape=(3, 2, 128),
                                mte3_bytes=0.0),
                       op_name="reshape_and_cache_a5", soc_family="Ascend950",
                       kv_page_size=128)
    unknown_meta = dict(unknown.metadata)
    assert unknown_meta["cache_valid_write_rows"] is None
    assert unknown_meta["cache_work_basis"] == "deferred_slot_binding"
    assert unknown_meta["cache_slot_index_read_bytes"] == 3 * 4
    assert unknown_meta["cache_conditional_write_upper_bound"] == 3 * 2 * 128 * 2
    assert unknown.mte3_bytes == 0.0
    assert unknown.mte2_bytes == 3 * 4


def _fused_members():
    matrix = replace(
        _make_op("grouped_gemm", recipe="grouped_gate_up",
                 local_shape=(32, 256, 128), dtype="int8"),
        metadata=(("matrix_group_rows", (16, 0, 16)),
                  ("weight_format", "int8"),
                  ("weight_scale_format", "fp32"),
                  ("activation_scale_format", "fp32"),
                  ("output_format", "int8"),
                  ("output_scale_format", "fp32")))
    activation = _make_op("swiglu", local_shape=(32, 128), dtype="bf16")
    activation = replace(activation, deps=(matrix.op_id,))
    quant = _make_op("quantize", local_shape=(32, 128), dtype="bf16")
    quant = replace(quant, deps=(activation.op_id,))
    return matrix, activation, quant


def test_fused_grouped_plan_and_region_share_int8_contract_with_zero_group():
    from mindie_sim.prediction.custom_framework_ops import (
        fused_region_contract, recipes)
    from mindie_sim.prediction.mechanism import Geometry

    matrix, activation, quant = _fused_members()
    complete = replace(matrix, recipe="moe_grouped_matmul_swiglu_quant")
    direct = lower_op(complete, op_name="moe_grouped_matmul_swiglu_quant")
    config = dict(direct.metadata)["xllm_fused_config"]
    assert config["group_rows"] == (16, 0, 16)
    assert config["active_groups"] == 2
    assert config["matrix_flops"] == 2 * 32 * 256 * 128
    assert config["weight_bytes"] == 2 * 256 * 128
    assert config["output_bytes"] == 32 * 128
    assert config["workspace_bytes"] == 32 * 256 * 4
    assert direct.matrix_flops == (("int8", config["matrix_flops"]),)
    assert direct.mte3_bytes == 32 * 128 + 32 * 4 + 32 * 256 * 4

    geometry = Geometry(64, {
        "vector": 64, "mte2": 64, "mte3": 64,
        "reg_alu_lanes": 64, "reg_issue_slots": 64, "reg_data_writes": 64,
        "reg_special_lanes": 64, "reg_reduce_lanes": 64,
        "reg_ub_accesses": 64, "reg_ub_stores": 64,
    }, {"ub": 196608}, {"ub": 64})
    prepared, plan = recipes()["moe_grouped_matmul_swiglu_quant"].prepare(
        direct, geometry)
    execution = dict(prepared.metadata)["program_execution"]
    assert execution["intermediate_hbm_bytes"] == 32 * 256 * 4
    assert plan.work.resources["mte3"] == direct.mte3_bytes
    assert plan.work.resources["mte2"] == direct.mte2_bytes
    assert execution["epilogue_work_basis"]["basis"] == (
        "arch35_dequant_swiglu_quant_capacity")
    assert plan.work.matrix["int8"] == config["matrix_flops"]
    assert plan.work.resources["reg_alu_lanes"] > 0
    assert plan.work.resources["reg_reduce_lanes"] > 0
    assert plan.work.resources["reg_special_lanes"] > 0
    assert plan.work.resources["hbm"] == direct.hbm_bytes
    from mindie_sim.prediction.program import RecipeBook, RecipeContext
    ctx = RecipeContext(op=prepared, work={},
                        resource_names={
                            "hbm": "hbm", "mte2": "mte2", "mte3": "mte3",
                            "vector": "vector", "cube": "cube",
                            "reg_alu_lanes": "reg_alu_lanes",
                            "reg_issue_slots": "reg_issue_slots",
                            "reg_data_writes": "reg_data_writes",
                            "reg_special_lanes": "reg_special_lanes",
                            "reg_reduce_lanes": "reg_reduce_lanes",
                            "reg_ub_accesses": "reg_ub_accesses",
                            "reg_ub_stores": "reg_ub_stores"},
                        rates={"hbm": 1e9, "mte2": 1e9, "mte3": 1e9,
                               "vector": 1e9, "cube": 1e9,
                               "reg_alu_lanes": 1e9, "reg_issue_slots": 1e9,
                               "reg_data_writes": 1e9, "reg_special_lanes": 1e9,
                               "reg_reduce_lanes": 1e9, "reg_ub_accesses": 1e9,
                               "reg_ub_stores": 1e9},
                        core_count=1, compute_peaks={"int8": 1e9},
                        pool_names={"gm": "gm"}, pool_units={"gm": 1})
    fused_stages = {stage.name: stage for stage in plan.program(ctx).stages}
    assert fused_stages["matrix_0"].acquire
    assert fused_stages["workspace_read_0"].free
    assert fused_stages["matrix_0"].acquire[0].size == (
        min(config["workspace_limit_rows"], config["rows"]) * 256 * 4)
    book = RecipeBook.from_mapping({"lowerings": {
        "grouped_gemm": "xllm_moe_grouped_matmul_swiglu_quant"}})
    assert type(book.select(direct)).__name__ == "PlannedRecipe"

    class _Selection:
        name = "declared_native_fusion"
    region = fused_region_contract(
        (matrix, activation, quant), {quant.op_id}, _Selection())
    assert region["config"] == config
    with pytest.raises(ModelingError, match="exclusive_member_chain"):
        fused_region_contract(
            (matrix, activation, quant), {quant.op_id, activation.op_id},
            _Selection())
    with pytest.raises(ModelingError, match="fp8_requires_owner_recipe"):
        lower_op(replace(complete, dtype="fp8_e4m3"),
                 op_name="moe_grouped_matmul_swiglu_quant")


def test_norm_rope_concat_keeps_all_three_materialized_outputs():
    from mindie_sim.prediction.diffusion_ops import recipes as sd_recipes
    from mindie_sim.prediction.mechanism import Geometry

    op = replace(
        _make_op("norm_rope_concat", local_shape=(1, 64, 2, 128)),
        metadata=(("query_shape", (1, 64, 2, 128)),
                  ("key_shape", (1, 64, 2, 128)),
                  ("value_shape", (1, 64, 2, 128)),
                  ("norm_type", 0), ("rope_type", 0), ("concat_order", 0)))
    lowered = sd_lower_op(op, op_name="norm_rope_concat")
    contract = dict(lowered.metadata)["mindie_sd_contract"]
    assert {item["name"] for item in contract["writes"]} == {
        "query_output", "key_output", "value_output"}
    assert lowered.mte3_bytes == 3 * 1 * 64 * 2 * 128 * 2
    assert lowered.vector_ops == 3 * 1 * 64 * 2 * 128
    assert set(lowered.returns) >= {"query_output", "key_output", "value_output"}

    enabled = replace(
        op, metadata=(("query_shape", (1, 64, 2, 128)),
                      ("key_shape", (1, 64, 2, 128)),
                      ("value_shape", (1, 64, 2, 128)),
                      ("encoder_query_shape", (1, 8, 2, 128)),
                      ("encoder_key_shape", (1, 8, 2, 128)),
                      ("encoder_value_shape", (1, 8, 2, 128)),
                      ("norm_type", 2), ("norm_added_type", 2),
                      ("rope_type", 1), ("rope_len", 8), ("is_training", True)))
    rich = sd_lower_op(enabled, op_name="norm_rope_concat")
    rich_contract = dict(rich.metadata)["mindie_sd_contract"]
    q_out = next(item for item in rich_contract["writes"]
                 if item["name"] == "query_output")
    assert q_out["shape"] == (1, 2, 72, 128)
    assert rich.mte3_bytes > lowered.mte3_bytes
    assert rich.vector_ops > lowered.vector_ops
    assert sum(item["name"] == "norm_query_mean" for item in rich_contract["writes"]) == 1

    aliased = replace(
        op, metadata=tuple(dict(op.metadata).items())
        + (("output_aliases", {"value": "value"}),))
    alias_lowered = sd_lower_op(aliased, op_name="norm_rope_concat")
    alias_contract = dict(alias_lowered.metadata)["mindie_sd_contract"]
    assert next(item for item in alias_contract["writes"]
                if item["name"] == "value_output")["materialized"] is False
    assert alias_lowered.mte3_bytes == 2 * 1 * 64 * 2 * 128 * 2

    geometry = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                        {"ub": 196608}, {"ub": 64})
    prepared, plan = sd_recipes()["mindie_sd_norm_rope_concat"].prepare(
        lowered, geometry)
    assert prepared.mte3_bytes == lowered.mte3_bytes
    assert plan.work.resources["mte3"] == lowered.mte3_bytes
    assert plan.work.resources["vector"] == lowered.vector_ops


def test_eagle_ffn_has_two_matrix_products_and_no_intermediate_hbm():
    from mindie_sim.prediction.diffusion_ops import recipes as sd_recipes
    from mindie_sim.prediction.mechanism import Geometry

    op = replace(
        _make_op("ffn", local_shape=(1, 128), flops=0.0, mte2_bytes=0.0,
                 mte3_bytes=0.0),
        metadata=(("x_shape", (1, 128)), ("weight1_shape", (256, 128)),
                  ("weight2_shape", (128, 256)), ("activation", "silu")))
    lowered = sd_lower_op(op, op_name="eagle_ffn")
    contract = dict(lowered.metadata)["mindie_sd_contract"]
    matrices = [item for item in contract["compute"]
                if item["kind"] == "matrix"]
    assert len(matrices) == 2
    assert sum(dict(item["matrix_flops"])["bf16"] for item in matrices) == 131072
    assert lowered.flops == 131072
    assert lowered.local_shape == (1, 128)
    assert lowered.padded_shape == lowered.launch_shape == (1, 128)
    # 512 B BF16 hidden GM write + 256 B output write; hidden is read again.
    assert lowered.mte3_bytes == 1 * 256 * 2 + 1 * 128 * 2
    assert contract["public_outputs"] == ("y",)
    hidden = next(item for item in contract["writes"]
                  if item["name"] == "hidden_store")
    assert hidden["bytes"] == 1 * 256 * 2
    assert any(item["name"] == "hidden" and item["bytes"] == 512
               for item in contract["reads"])

    geometry = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                        {"ub": 196608}, {"ub": 64})
    prepared, plan = sd_recipes()["mindie_sd_eagle_ffn"].prepare(lowered, geometry)
    assert prepared.flops == 131072
    assert plan.work.matrix["bf16"] == 131072


def test_sparse_roles_keep_mask_scale_and_output_separate():
    op = replace(
        _make_op("indexed_attention", local_shape=(1, 32, 4, 64)),
        metadata=(("query_shape", (1, 32, 4, 64)),
                  ("key_shape", (1, 64, 2, 64)),
                  ("value_shape", (1, 64, 2, 64)),
                  ("block_mask", (1, 0, 1, 0, 1, 1, 0, 1)),
                  ("kv_dtype", "fp8_e4m3")))
    lowered = sd_lower_op(op, op_name="rainfusion", sparsity_ratio=0.5,
                          block_size=16, kv_dtype="fp8_e4m3")
    contract = dict(lowered.metadata)["mindie_sd_contract"]
    reads = {item["name"]: item for item in contract["reads"]}
    assert reads["query"]["bytes"] == 1 * 32 * 4 * 64 * 2
    assert reads["key"]["bytes"] == 0.0
    assert reads["kv_load"]["bytes"] == 4 * 16 * 2 * (64 + 64) * 1
    assert contract["attributes"]["active_block_pairs"] == 5
    assert lowered.flops == 2 * 5 * 16 * 16 * 4 * (64 + 64)
    assert any(item["name"] == "attention_out" for item in contract["writes"])
    changed = sd_lower_op(op, op_name="rainfusion", sparsity_ratio=0.25,
                          block_size=16, kv_dtype="fp8_e4m3")
    # An explicit block mask is authoritative over the ratio.
    assert changed.flops == lowered.flops
    assert changed.mte2_bytes == lowered.mte2_bytes


def test_mega_gdn_decode_consumes_f_state_recurrence_and_two_states():
    from mindie_sim.prediction.custom_framework_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry

    op = replace(
        _make_op("gdn_decode", local_shape=(4, 768)),
        metadata=(("qkv_shape", (4, 768)), ("z_shape", (4, 2, 128)),
                  ("num_k_heads", 2), ("conv_state_shape", (4, 3, 768)),
                  ("ssm_state_shape", (4, 2, 128, 128)),
                  ("conv_weight_shape", (4, 768)),
                  ("b_shape", (4, 2)), ("a_shape", (4, 2)),
                  ("a_log_shape", (2,)), ("dt_bias_shape", (2,)),
                  ("read_state_indices", (0, -1, 2, 3)),
                  ("write_state_indices", (0, 1, 2, 3)),
                  ("read_versions", (1, 1, 1, 1)),
                  ("write_versions", (1, 1, 1, 1)),
                  ("state_layout", "V,K")))
    lowered = lower_op(op, op_name="mega_gdn_decode")
    geometry = Geometry(64, {
        "vector": 64, "mte2": 64, "mte2_aiv": 64, "mte2_aic": 64,
        "mte3": 64, "hbm": 1,
    }, {"ub": 196608}, {"ub": 64})
    prepared, plan = recipes()["mega_gdn_decode"].prepare(lowered, geometry)
    assert prepared.returns == ("convOut", "convStateOut", "ssmStateOut", "out")
    assert plan.metadata["state_mechanism"] == "gdn_recurrent.build_program"
    assert plan.metadata["two_states"] == ("conv_state_out", "ssm_state_out")
    assert plan.metadata["state_indices"]["read"] == (0, -1, 2, 3)
    assert plan.metadata["readout_onchip"] is True
    effects = plan.metadata["gdn_state_effects"]
    assert any(item["target"] == "conv_state" and item["slot"] == 0
               and item["action"] == "read" for item in effects)
    assert any(item["target"] == "ssm_state" and item["slot"] == 3
               and item["action"] == "write" for item in effects)
    assert not any(item["slot"] == 2 and item["action"] == "read"
                   and item["occurrence"] == 1 for item in effects)
    assert plan.work.resources["vector"] > 0
    assert plan.work.resources["hbm"] > 0
    assert prepared.local_shape == (4, 2, 128)


def test_mindie_sd_exact_name_is_normalized_without_generic_default():
    op = replace(_make_op("rainfusion", recipe="mindie_sd_rainfusion",
                          local_shape=(128, 128), flops=1_000_000.0,
                          mte2_bytes=65536.0),
                 metadata=(("sparsity_ratio", 0.5), ("block_size", 64)))
    lowered = sd_lower_op(op)
    assert lowered.recipe == "mindie_sd_rainfusion"
    with pytest.raises(ModelingError, match="requires_explicit_name"):
        sd_lower_op(_make_op("indexed_attention", local_shape=(128, 128),
                             flops=1_000_000.0, mte2_bytes=65536.0))


def test_cache_write_completion_orders_the_next_attention_consumer():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _support.moe_hc_mc2_fixtures import attnres_case as _case
    from mindie_sim import graph_costs

    case = _case(8, hidden=128)
    cache = lower_op(replace(
                         _make_op("kv_cache_write", local_shape=(8, 2, 128),
                                  mte3_bytes=0.0),
                         metadata=(("slot_mapping", tuple(range(8))),
                                   ("cache_capacity_slots", 8))),
                     op_name="reshape_and_cache_a5", soc_family="Ascend950",
                     kv_page_size=128)
    consumer = replace(_make_op("elementwise", local_shape=(8, 128),
                                vector_ops=1024.0, dtype="bf16"),
                       op_id="after_cache", deps=(cache.op_id,))
    result = graph_costs(case, {0: (cache, consumer)}, input_ready_ms=1,
                         external_inputs={"analysis_input": 0})
    timeline = {row["op_id"]: row for row in result["timeline"]}
    assert timeline["after_cache"]["ready_ms"] >= timeline[cache.op_id]["complete_ms"]
    assert result["execution"]["live_tasks"] == 0


def test_fused_grouped_epilogue_competes_on_the_shared_vector_resource():
    from mindie_sim.prediction.custom_framework_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeContext
    from mindie_sim.prediction.engine import Engine, Resource

    op = replace(
        _make_op("grouped_gemm", recipe="moe_grouped_matmul_swiglu_quant",
                 local_shape=(32, 256, 128), dtype="int8"),
        metadata=(("matrix_group_rows", (16, 0, 16)),
                  ("weight_format", "int8"),
                  ("weight_scale_format", "fp32"),
                  ("activation_scale_format", "fp32"),
                  ("output_format", "int8"),
                  ("output_scale_format", "fp32")))
    lowered = lower_op(op, op_name="moe_grouped_matmul_swiglu_quant")
    geometry = Geometry(1, {"vector": 1, "mte2": 1, "mte3": 1},
                        {"ub": 196608}, {"ub": 1})
    prepared, plan = recipes()["moe_grouped_matmul_swiglu_quant"].prepare(
        lowered, geometry)
    names = {"hbm": "hbm", "mte2": "mte2", "mte3": "mte3",
             "vector": "vector", "cube": "cube"}
    context = RecipeContext(
        op=prepared, work={}, resource_names=names,
        rates={"hbm": 1e9, "mte2": 1e9, "mte3": 1e9,
               "vector": 1.0, "cube": 1e9},
        core_count=1, compute_peaks={"int8": 1e6})
    engine = Engine((Resource("hbm", 1e9), Resource("mte2", 1e9),
                     Resource("mte3", 1e9), Resource("vector", 1.0),
                     Resource("cube", 1e9)))
    ready = engine.submit("ready")
    fused = plan.program(context).start(engine, "fused", ready)
    external = engine.submit("external", {"vector": 50.0})
    engine.advance(until=(fused, external), first=False)
    assert engine.serviced["vector"] == plan.work.resources["vector"] + 50.0
    assert engine.serviced["mte3"] == plan.work.resources["mte3"]
    assert fused.time > external.time


def test_sparse_batch_tail_tnd_and_independent_scales():
    meta = (("query_shape", (2, 32, 4, 64)),
            ("key_shape", (2, 64, 2, 64)),
            ("value_shape", (2, 64, 2, 64)),
            ("block_mask", (1, 0, 1, 0, 1, 1, 0, 1)),
            ("q_scale_shape", (2, 32, 4)),
            ("k_scale_shape", (2, 8, 2)),
            ("v_scale_shape", (2, 8, 2)),
            ("lse_shape", (2, 32, 4)))
    op = replace(_make_op("indexed_attention", local_shape=(2, 32, 4, 64)),
                 metadata=meta)
    lowered = sd_lower_op(op, op_name="rainfusion", sparsity_ratio=0.5,
                          block_size=16)
    contract = dict(lowered.metadata)["mindie_sd_contract"]
    assert contract["attributes"]["batch"] == 2
    assert lowered.flops == 2 * 1_310_720
    assert {item["name"] for item in contract["reads"]} >= {
        "q_scale", "k_scale", "v_scale"}
    qk = next(item for item in contract["compute"] if item["name"] == "qk")
    assert {"q_scale", "k_scale", "v_scale"} <= set(qk["inputs"])
    assert any(item["name"] == "softmax_lse" for item in contract["writes"])
    assert "softmax_lse" in lowered.returns

    tail = replace(_make_op("indexed_attention", local_shape=(1, 20, 4, 64)),
                   metadata=(("query_shape", (1, 20, 4, 64)),
                             ("key_shape", (1, 40, 2, 64)),
                             ("value_shape", (1, 40, 2, 64)),
                             ("block_mask", (1, 1, 1, 1, 1, 1))))
    tail_low = sd_lower_op(tail, op_name="rainfusion", sparsity_ratio=0.5,
                           block_size=16)
    # q blocks 16/4 over kv blocks 16/16/8: sum(qrows*kvrows)=800.
    assert tail_low.flops == 2 * 800 * 4 * (64 + 64)

    tnd = replace(_make_op("indexed_attention", local_shape=(32, 4, 64)),
                  metadata=(("query_shape", (32, 4, 64)),
                            ("key_shape", (64, 2, 64)),
                            ("value_shape", (64, 2, 64)),
                            ("query_lengths", (32,)),
                            ("key_lengths", (64,)),
                            ("value_lengths", (64,)),
                            ("block_mask", (1, 1, 1, 1, 1, 1, 1, 1))))
    tnd_low = sd_lower_op(tnd, op_name="rainfusion", sparsity_ratio=0.5,
                          block_size=16)
    assert dict(tnd_low.metadata)["layout"] == "TND"
    assert tnd_low.local_shape == (32, 4, 64) or tnd_low.flops > 0


def test_norm_rope_and_ffn_contracts_build_producer_consumer_edges():
    from mindie_sim.prediction.diffusion_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeContext

    norm = replace(
        _make_op("norm_rope_concat", local_shape=(1, 64, 2, 128)),
        metadata=(("query_shape", (1, 64, 2, 128)),
                  ("key_shape", (1, 64, 2, 128)),
                  ("value_shape", (1, 64, 2, 128)),
                  ("norm_type", 2), ("rope_type", 1), ("rope_len", 8)))
    lowered = sd_lower_op(norm, op_name="norm_rope_concat")
    geometry = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                        {"ub": 196608}, {"ub": 64})
    prepared, plan = recipes()["mindie_sd_norm_rope_concat"].prepare(
        lowered, geometry)
    context = RecipeContext(op=prepared, work={},
                            resource_names={"hbm": "hbm", "mte2": "mte2",
                                            "mte3": "mte3", "vector": "vector",
                                            "cube": "cube"},
                            rates={"hbm": 1e9, "mte2": 1e9, "mte3": 1e9,
                                   "vector": 1e9, "cube": 1e9},
                            core_count=1, compute_peaks={"bf16": 1e9})
    program = plan.program(context)
    stages = {stage.name: stage for stage in program.stages}
    norm_stage = stages["compute_query_norm"]
    rope_stage = stages["compute_query_rope"]
    concat_stage = stages["compute_query_concat_transpose"]
    assert {"read_query", "read_norm_query_weight",
            "read_norm_query_bias"} <= set(norm_stage.after)
    assert "compute_query_norm" in rope_stage.after
    assert "compute_query_rope" in concat_stage.after

    ffn = replace(_make_op("ffn", local_shape=(1, 128), flops=0.0,
                           mte2_bytes=0.0, mte3_bytes=0.0),
                  metadata=(("x_shape", (1, 128)),
                            ("weight1_shape", (128, 256)),
                            ("weight2_shape", (256, 128)),
                            ("bias1_shape", (256,)),
                            ("bias2_shape", (128,)),
                            ("bias_dtype", "bf16"),
                            ("activation", "silu")))
    ffn_low = sd_lower_op(ffn, op_name="eagle_ffn")
    assert ffn_low.flops == 131072
    ffn_contract = dict(ffn_low.metadata)["mindie_sd_contract"]
    assert ffn_contract["attributes"]["weight1_layout"] == "canonical_KxH"
    assert ffn_contract["attributes"]["weight2_layout"] == "canonical_HxN"
    assert ffn_low.mte3_bytes == 512 + 256
    ffn_prepared, ffn_plan = recipes()["mindie_sd_eagle_ffn"].prepare(
        ffn_low, geometry)
    ffn_context = RecipeContext(op=ffn_prepared, work={},
                                resource_names=context.resource_names,
                                rates=context.rates, core_count=1,
                                compute_peaks={"bf16": 1e9},
                                pool_names={"gm": "gm"},
                                pool_units={"gm": 1})
    stages = {stage.name: stage for stage in ffn_plan.program(ffn_context).stages}
    assert set(stages["compute_linear1"].after) >= {
        "read_x", "read_weight1", "read_bias1"}
    assert "compute_linear1" in stages["compute_activation"].after
    assert "compute_activation" in stages["write_hidden_store"].after
    assert "write_hidden_store" in stages["read_hidden"].after
    assert set(stages["compute_linear2"].after) >= {
        "read_hidden", "read_weight2", "read_bias2"}
    assert stages["write_hidden_store"].acquire
    assert stages["read_hidden"].free
    assert stages["write_hidden_store"].acquire[0].size == 512


def test_fused_workspace_split_reuse_boundary_is_explicit():
    matrix, activation, quant = _fused_members()
    complete = replace(matrix, recipe="moe_grouped_matmul_swiglu_quant",
                       metadata=tuple(dict(matrix.metadata).items())
                       + (("workspace_limit_rows", 8),))
    lowered = lower_op(complete, op_name="moe_grouped_matmul_swiglu_quant")
    config = dict(lowered.metadata)["xllm_fused_config"]
    assert config["workspace_bytes"] == 32 * 256 * 4
    assert config["split_workspace"] is True
    assert config["workspace_limit_rows"] == 8
    # Two live chunks of 8 rows each are reused across the 32-row call.
    assert config["workspace_live_bytes"] == 16 * 256 * 4


def test_adalayernorm_laurel_and_mxfp4_contracts_have_values_and_sources():
    from mindie_sim.prediction.diffusion_ops import recipes as sd_recipes
    from mindie_sim.prediction.mechanism import Geometry

    adaln = replace(
        _make_op("rmsnorm", local_shape=(2, 3, 128)),
        metadata=(("x_shape", (2, 3, 128)), ("scale_shape", (128,)),
                  ("shift_shape", (128,)), ("weight_shape", (128,)),
                  ("bias_shape", (128,)), ("return_stats", True)))
    adaln_low = sd_lower_op(adaln, op_name="adalayernorm")
    adaln_contract = dict(adaln_low.metadata)["mindie_sd_contract"]
    assert adaln_low.mte2_bytes == (2 * 3 * 128 * 2 + 4 * 128 * 2)
    assert adaln_low.mte3_bytes == (2 * 3 * 128 * 2 + 2 * 2 * 3 * 1 * 2)
    assert {"y", "mean", "rstd"} <= set(adaln_low.returns)

    laurel = replace(_make_op("gemm", local_shape=(128, 1024), flops=0.0),
                     metadata=(("x_shape", (128, 1024)), ("low_rank", 32),
                               ("weight_a_shape", (1024, 32)),
                               ("weight_b_shape", (32, 1024))))
    laurel_low = sd_lower_op(laurel, op_name="laurel", low_rank=32)
    assert laurel_low.flops == 4 * 128 * 1024 * 32
    assert "y" in laurel_low.returns

    quant = replace(
        _make_op("quantize", local_shape=(64, 128)),
        metadata=(("x_shape", (64, 128)),))
    quant_low = sd_lower_op(quant, op_name="quant_four_over_six_a5",
                            soc_family="Ascend950")
    assert {"mxfp4_payload", "mxfp4_scale"} <= set(quant_low.returns)
    quant_contract = dict(quant_low.metadata)["mindie_sd_contract"]
    payload = next(item for item in quant_contract["writes"]
                   if item["name"] == "mxfp4_payload")
    scale = next(item for item in quant_contract["writes"]
                 if item["name"] == "mxfp4_scale")
    assert quant_contract["public_outputs"] == ("mxfp4_payload", "mxfp4_scale")
    assert payload["shape"] == (64, 64) and scale["shape"] == (64, 2, 2)
    assert payload["bytes"] == 64 * 64 and scale["bytes"] == 64 * 4

    q96 = replace(_make_op("quantize", local_shape=(64, 96)),
                  metadata=(("x_shape", (64, 96)),))
    q96_low = sd_lower_op(q96, op_name="quant_four_over_six_a5",
                          soc_family="Ascend950")
    q96_contract = dict(q96_low.metadata)["mindie_sd_contract"]
    q96_payload = next(item for item in q96_contract["writes"]
                       if item["name"] == "mxfp4_payload")
    q96_scale = next(item for item in q96_contract["writes"]
                     if item["name"] == "mxfp4_scale")
    assert q96_payload["shape"] == (64, 48) and q96_payload["bytes"] == 3072
    assert q96_scale["shape"] == (64, 2, 2) and q96_scale["bytes"] == 256

    adaln_batch = replace(
        _make_op("rmsnorm", local_shape=(2, 3, 128)),
        metadata=(("x_shape", (2, 3, 128)),
                  ("scale_shape", (2, 128)), ("shift_shape", (2, 128))))
    adaln_batch_low = sd_lower_op(adaln_batch, op_name="adalayernorm")
    assert adaln_batch_low.mte2_bytes == 2 * 3 * 128 * 2 + 2 * 2 * 128 * 2
    assert adaln_batch_low.mte3_bytes == 2 * 3 * 128 * 2
    geometry = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                        {"ub": 196608}, {"ub": 64})
    for name, lowered in (("adalayernorm", adaln_low), ("laurel", laurel_low),
                          ("quant_four_over_six_a5", quant_low)):
        prepared, plan = sd_recipes()[f"mindie_sd_{name}"].prepare(
            lowered, geometry)
        assert prepared.mte2_bytes == lowered.mte2_bytes
        assert plan.work.resources["mte3"] == lowered.mte3_bytes


def test_smla_sparse_lse_and_onerec_have_typed_ports_and_work():
    smla = replace(
        _make_op("mla_attention", local_shape=(2, 8, 4, 64), dtype="bf16",
                 mte2_bytes=0.0, mte3_bytes=0.0),
        metadata=(("q_shape", (2, 8, 4, 64)),
                  ("ori_kv_shape", (2, 64, 2, 64)),
                  ("cmp_kv_shape", (2, 32, 2, 64)),
                  ("sparse_index_count", 8),
                  ("return_softmax_lse", True)))
    smla_low = lower_op(smla, op_name="smla_va", return_lse=True)
    meta = dict(smla_low.metadata)
    assert set(meta["xllm_ports"]["outputs"]) >= {"attn_out", "softmax_lse"}
    assert smla_low.mte2_bytes == (
        2 * 8 * 4 * (64 + 64) * 2 + 2 * 64 * 2 * 64 * 2
        + 2 * 32 * 2 * 64 * 2 + 8 * 4)
    assert smla_low.mte3_bytes == 2 * 8 * 4 * 64 * 2 + 2 * 8 * 4 * 4
    assert {"attn_out", "softmax_lse"} <= set(smla_low.returns)

    sparse = replace(
        _make_op("indexed_attention", local_shape=(1, 32, 4, 64), dtype="bf16"),
        metadata=(("q_shape", (1, 32, 4, 64)),
                  ("key_shape", (1, 64, 2, 64)),
                  ("value_shape", (1, 64, 2, 64)),
                  ("sparse_index_count", 16),
                  ("return_lse", True)))
    sparse_low = lower_op(sparse, op_name="sparse_flash_attention_lse",
                          return_lse=True)
    sparse_meta = dict(sparse_low.metadata)
    assert set(sparse_meta["xllm_ports"]["inputs"]) >= {
        "query", "key", "value", "sparse_indices"}
    assert "softmax_max" in sparse_low.returns
    assert "softmax_sum" in sparse_low.returns
    assert sparse_low.mte3_bytes >= 2 * 1 * 32 * 4 * 4

    beam = lower_op(_make_op("beam_select", local_shape=(4, 1024),
                             mte3_bytes=0.0),
                    op_name="onerec_final_beam_select", beam_width=32)
    assert dict(beam.metadata)["xllm_ports"]["outputs"] == (
        "out_token_ids", "out_token_index", "out_log_probs", "out_sequence")
    assert beam.mte3_bytes == 3 * 32 * 4


def test_fused_workspace_windows_reuse_two_buffers_at_crossing_geometry():
    from mindie_sim.prediction.custom_framework_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry

    op = replace(
        _make_op("grouped_gemm", recipe="moe_grouped_matmul_swiglu_quant",
                 local_shape=(65537, 256, 128), dtype="int8"),
        metadata=(("matrix_group_rows", (65537, 0)),
                  ("weight_format", "int8"),
                  ("weight_scale_format", "fp32"),
                  ("activation_scale_format", "fp32"),
                  ("output_format", "int8"),
                  ("output_scale_format", "fp32")))
    lowered = lower_op(op, op_name="moe_grouped_matmul_swiglu_quant")
    geometry = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                        {"ub": 196608}, {"ub": 64})
    prepared, plan = recipes()["moe_grouped_matmul_swiglu_quant"].prepare(
        lowered, geometry)
    assert plan.metadata["workspace_window_rows"] == (32768, 32768, 1)
    assert plan.metadata["workspace_windows"] == 3
    assert plan.metadata["workspace_live_bytes"] == 2 * 32768 * 256 * 4
    assert plan.metadata["intermediate_hbm_bytes"] == 65537 * 256 * 4
    assert plan.work.resources["hbm"] == prepared.hbm_bytes
    from mindie_sim.prediction.program import RecipeContext
    ctx = RecipeContext(
        op=prepared, work={},
        resource_names={name: name for name in (
            "hbm", "mte2", "mte3", "vector", "cube", "reg_alu_lanes",
            "reg_issue_slots", "reg_data_writes", "reg_special_lanes",
            "reg_reduce_lanes", "reg_ub_accesses", "reg_ub_stores")},
        rates={name: 1e9 for name in (
            "hbm", "mte2", "mte3", "vector", "cube", "reg_alu_lanes",
            "reg_issue_slots", "reg_data_writes", "reg_special_lanes",
            "reg_reduce_lanes", "reg_ub_accesses", "reg_ub_stores")},
        core_count=1, compute_peaks={"int8": 1e9},
        pool_names={"gm": "gm"}, pool_units={"gm": 1})
    stages = {stage.name: stage for stage in plan.program(ctx).stages}
    assert stages["matrix_0"].acquire[0] is stages["matrix_2"].acquire[0]
    assert stages["workspace_read_0"].free[0] is stages["matrix_0"].acquire[0]
    assert stages["matrix_2"].acquire[0].size == 32768 * 256 * 4


def test_sparse_real_bnsd_axes_and_lse_shape():
    op = replace(
        _make_op("indexed_attention", local_shape=(1, 4, 128, 128)),
        metadata=(("layout", "BNSD"),
                  ("query_shape", (1, 4, 128, 128)),
                  ("key_shape", (1, 2, 256, 128)),
                  ("value_shape", (1, 2, 256, 128)),
                  ("block_mask", (1, 1)),
                  ("return_softmax_lse", True)))
    lowered = sd_lower_op(op, op_name="rainfusion", sparsity_ratio=0.0,
                          block_size=128)
    contract = dict(lowered.metadata)["mindie_sd_contract"]
    assert lowered.flops == 67_108_864
    assert lowered.local_shape == (1, 4, 128, 128)
    assert lowered.mte3_bytes == 1 * 4 * 128 * 128 * 2 + 1 * 4 * 128 * 1 * 4
    lse = next(item for item in contract["writes"] if item["name"] == "softmax_lse")
    assert lse["shape"] == (1, 4, 128, 1) and lse["bytes"] == 2048


def test_onerec_result_width_and_sequence_length_are_external_values():
    def rec(step, batch=1):
        return lower_op(
            replace(_make_op("beam_select", local_shape=(batch, 128),
                             mte3_bytes=0.0),
                    metadata=(("beam_width", 16), ("topk", 128),
                              ("result_width", 64), ("batch", batch),
                              ("sequence_capacity", 3),
                              ("current_step", step))),
            op_name="onerec_final_beam_select")

    step1, step2, batch2 = rec(1), rec(2), rec(2, batch=2)
    assert step1.mte3_bytes == 3 * 64 * 4 + 1 * 64 * 2 * 4 == 1280
    assert step2.mte3_bytes == 3 * 64 * 4 + 1 * 64 * 3 * 4 == 1536
    assert batch2.mte3_bytes == 3 * 64 * 4 + 2 * 64 * 3 * 4 == 2304
    meta = dict(step2.metadata)["xllm_beam_selection"]
    assert meta["sequence_shape"] == (1, 64, 3)
    assert meta["active_beam"] == 16 and meta["topk"] == 128
    assert meta["sequence_capacity"] == 3 and meta["current_step"] == 2


def test_sparse_tnd_is_equivalent_to_bnsd_control():
    op = replace(
        _make_op("indexed_attention", local_shape=(128, 4, 128)),
        metadata=(("layout", "TND"),
                  ("query_shape", (128, 4, 128)),
                  ("key_shape", (256, 2, 128)),
                  ("value_shape", (256, 2, 128)),
                  ("query_lengths", (128,)),
                  ("key_lengths", (256,)),
                  ("value_lengths", (256,)),
                  ("block_mask", (1, 1)),
                  ("return_softmax_lse", True)))
    lowered = sd_lower_op(op, op_name="rainfusion", sparsity_ratio=0.0,
                          block_size=128)
    contract = dict(lowered.metadata)["mindie_sd_contract"]
    assert lowered.flops == 67_108_864
    assert lowered.local_shape == (128, 4, 128)
    assert lowered.mte3_bytes == 128 * 4 * 128 * 2 + 128 * 4 * 4
    lse = next(item for item in contract["writes"] if item["name"] == "softmax_lse")
    assert lse["shape"] == (128, 4) and lse["bytes"] == 2048


def test_mega_gdn_read7_write2_state_versions_are_actual_slots():
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.custom_framework_ops import recipes
    op = replace(
        _make_op("mega_gdn_decode", local_shape=(1, 768), dtype="bf16"),
        metadata=(("qkv_shape", (1, 768)), ("z_shape", (1, 2, 128)),
                  ("num_k_heads", 2),
                  ("conv_state_shape", (9, 3, 768)),
                  ("ssm_state_shape", (9, 2, 128, 128)),
                  ("conv_weight_shape", (4, 768)),
                  ("b_shape", (1, 2)), ("a_shape", (1, 2)),
                  ("a_log_shape", (2,)), ("dt_bias_shape", (2,)),
                  ("read_state_indices", (7,)),
                  ("write_state_indices", (2,)),
                  ("read_versions", (1,)), ("write_versions", (2,)),
                  ("state_capacity", 9), ("state_layout", "V,K")))
    lowered = lower_op(op, op_name="mega_gdn_decode")
    meta = dict(lowered.metadata)["xllm_mega_gdn_config"]
    assert meta["read_state_indices"] == (7,)
    assert meta["write_state_indices"] == (2,)
    assert meta["conv_state_shape"] == (9, 3, 768)
    assert meta["ssm_state_shape"] == (9, 2, 128, 128)
    geometry = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                        {"ub": 196608}, {"ub": 64})
    prepared, plan = recipes()["mega_gdn_decode"].prepare(lowered, geometry)
    effects = plan.metadata["gdn_state_effects"]
    assert any(item["target"] == "conv_state" and item["slot"] == 7
               and item["action"] == "read" and item["version"] == 1
               for item in effects)
    assert any(item["target"] == "ssm_state" and item["slot"] == 2
               and item["action"] == "write" and item["version"] == 2
               and item["span_bytes"] == 2 * 128 * 128 * 4
               and item["commit_event"] for item in effects)
    assert plan.metadata["state_program"] == "gdn_recurrent.program"
    assert plan.metadata["final_output_write_bytes"] == 1 * 2 * 128 * 2
    from mindie_sim.prediction.program import RecipeContext
    names = {name: name for name in ("hbm", "mte2", "mte3", "vector", "cube")}
    context = RecipeContext(op=prepared, work={}, resource_names=names,
                            rates={name: 1e9 for name in names},
                            core_count=1, compute_peaks={"bf16": 1e9},
                            pool_names={"ub": "ub"}, pool_units={"ub": 64})
    stages = {stage.name: stage for stage in plan.program(context).stages}
    assert tuple(stages) == ("conv", "qk_norm_gate", "recurrent_state",
                             "readout_rmsnorm_zgate")
    assert stages["recurrent_state"].program is not None
    traffic = plan.metadata["state_traffic_bytes"]
    assert traffic["conv_read"] == traffic["conv_write"] == 3 * 768 * 2
    assert traffic["ssm_read"] == traffic["ssm_write"] == 2 * 128 * 128 * 4


def test_ffn_hidden_grant_is_capacity_checked_and_released_after_read():
    from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
    from mindie_sim.prediction.diffusion_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeContext

    op = replace(_make_op("ffn", local_shape=(1, 128), flops=0.0,
                          mte2_bytes=0.0, mte3_bytes=0.0),
                 metadata=(("x_shape", (1, 128)),
                           ("weight1_shape", (128, 256)),
                           ("weight2_shape", (256, 128)),
                           ("activation", "silu")))
    lowered = sd_lower_op(op, op_name="eagle_ffn")
    prepared, plan = recipes()["mindie_sd_eagle_ffn"].prepare(
        lowered, Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                          {"ub": 196608}, {"ub": 64}))
    names = ("hbm", "mte2", "mte3", "vector", "cube")
    context = RecipeContext(op=prepared, work={},
                            resource_names={name: name for name in names},
                            rates={name: 1e9 for name in names},
                            core_count=1, compute_peaks={"bf16": 1e9},
                            pool_names={"gm": "gm"}, pool_units={"gm": 1})
    program = plan.program(context)
    engine = Engine(tuple(Resource(name, 1e9) for name in names),
                    pools={"gm": 511}, pool_units={"gm": 1})
    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        program.start(engine, "ffn", engine.submit("ready"))
    granted_op = replace(prepared, metadata=tuple(
        (k, v) for k, v in dict(prepared.metadata).items()
        if k != "workspace_grant") + (("workspace_grant", "caller-held"),))
    granted_plan = plan
    granted_context = replace(context, op=granted_op)
    granted_stages = {stage.name: stage
                      for stage in granted_plan.program(granted_context).stages}
    assert not granted_stages["write_hidden_store"].acquire
    engine = Engine(tuple(Resource(name, 1e9) for name in names),
                    pools={"gm": 512}, pool_units={"gm": 1})
    done = program.start(engine, "ffn", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.peak_allocated["gm"] == 512
    assert engine.allocated["gm"] == 0


def test_ffn_operand_ready_events_gate_both_matrix_stages():
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.diffusion_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeContext

    op = replace(_make_op("ffn", local_shape=(1, 128), flops=0.0,
                          mte2_bytes=0.0, mte3_bytes=0.0),
                 metadata=(("x_shape", (1, 128)),
                           ("weight1_shape", (256, 128)),
                           ("weight2_shape", (128, 256)),
                           ("activation", "silu")))
    lowered = sd_lower_op(op, op_name="eagle_ffn")
    prepared, plan = recipes()["mindie_sd_eagle_ffn"].prepare(
        lowered, Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                          {"ub": 196608}, {"ub": 64}))
    names = ("hbm", "mte2", "mte3", "vector", "cube")
    context = RecipeContext(op=prepared, work={},
                            resource_names={name: name for name in names},
                            rates={name: 1e9 for name in names},
                            core_count=1, compute_peaks={"bf16": 1e9})
    engine = Engine(tuple(Resource(name, 1e9) for name in names),
                    trace_limit=100)
    ready = engine.submit("ready")
    done = plan.program(context).start(engine, "ffn", ready)
    engine.advance(until=(done,))
    rows = {row["task"].removeprefix("ffn:"): row for row in engine.trace}
    assert rows["compute_linear1"]["start_ms"] >= rows["read_weight1"]["end_ms"]
    assert rows["compute_linear2"]["start_ms"] >= rows["read_hidden"]["end_ms"]
    assert rows["write_hidden_store"]["start_ms"] >= rows["compute_activation"]["end_ms"]


def test_fused_workspace_window_program_reuses_two_buffers_in_engine():
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.custom_framework_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeContext

    matrix, _, _ = _fused_members()
    limited = replace(matrix, recipe="moe_grouped_matmul_swiglu_quant",
                      metadata=tuple(dict(matrix.metadata).items())
                      + (("workspace_limit_rows", 8),))
    lowered = lower_op(limited, op_name="moe_grouped_matmul_swiglu_quant")
    prepared, plan = recipes()["moe_grouped_matmul_swiglu_quant"].prepare(
        lowered, Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64,
                              "reg_alu_lanes": 64, "reg_issue_slots": 64,
                              "reg_data_writes": 64, "reg_special_lanes": 64,
                              "reg_reduce_lanes": 64, "reg_ub_accesses": 64,
                              "reg_ub_stores": 64},
                          {"ub": 196608}, {"ub": 64}))
    names = ("hbm", "mte2", "mte3", "vector", "cube", "reg_alu_lanes",
             "reg_issue_slots", "reg_data_writes", "reg_special_lanes",
             "reg_reduce_lanes", "reg_ub_accesses", "reg_ub_stores")
    context = RecipeContext(op=prepared, work={},
                            resource_names={name: name for name in names},
                            rates={name: 1e9 for name in names},
                            core_count=1, compute_peaks={"int8": 1e9},
                            pool_names={"gm": "gm"}, pool_units={"gm": 1})
    engine = Engine(tuple(Resource(name, 1e9) for name in names),
                    pools={"gm": 2 * 8 * 256 * 4}, pool_units={"gm": 1},
                    trace_limit=50)
    done = plan.program(context).start(engine, "fused", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.peak_allocated["gm"] == 2 * 8 * 256 * 4
    assert engine.allocated["gm"] == 0
    assert [row["task"] for row in engine.trace].count("fused:matrix_2") == 1


def test_sparse_per_head_mask_uses_head_arithmetic_but_shared_kv_load():
    op = replace(
        _make_op("indexed_attention", local_shape=(1, 2, 128, 128)),
        metadata=(("layout", "BNSD"),
                  ("query_shape", (1, 2, 128, 128)),
                  ("key_shape", (1, 2, 256, 128)),
                  ("value_shape", (1, 2, 256, 128)),
                  ("block_mask", (((1, 0), (0, 1)),)),
                  ("mask_dtype", "int8")))
    lowered = sd_lower_op(op, op_name="rainfusion", sparsity_ratio=0.0,
                          block_size=128)
    contract = dict(lowered.metadata)["mindie_sd_contract"]
    assert lowered.flops == 16_777_216
    assert lowered.local_shape == (1, 2, 128, 128)
    assert lowered.mte3_bytes == 1 * 2 * 128 * 128 * 2
    reads = {item["name"]: item for item in contract["reads"]}
    # Two heads have disjoint KV blocks; the union loads each block once.
    assert reads["kv_load"]["bytes"] == 2 * 128 * 2 * (128 + 128) * 2
    assert contract["attributes"]["mask_layout"] == "B,H,Qblock,KVblock"
    assert contract["attributes"]["active_block_pairs"] == 2


def test_ffn_hidden_waits_for_non_gm_memory_holder_until_10us():
    from mindie_sim.prediction.engine import Allocation, Engine, Resource
    from mindie_sim.prediction.diffusion_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeContext

    op = replace(_make_op("ffn", local_shape=(1, 128), flops=0.0,
                          mte2_bytes=0.0, mte3_bytes=0.0),
                 metadata=(("x_shape", (1, 128)),
                           ("weight1_shape", (128, 256)),
                           ("weight2_shape", (256, 128)),
                           ("activation", "silu"),
                           ("workspace_pool", "workspace")))
    lowered = sd_lower_op(op, op_name="eagle_ffn")
    prepared, plan = recipes()["mindie_sd_eagle_ffn"].prepare(
        lowered, Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                          {"ub": 196608}, {"ub": 64}))
    names = ("hbm", "mte2", "mte3", "vector", "cube")
    context = RecipeContext(op=prepared, work={},
                            resource_names={name: name for name in names},
                            rates={name: 1e9 for name in names},
                            core_count=1, compute_peaks={"bf16": 1e9},
                            pool_names={"workspace": "workspace"},
                            pool_units={"workspace": 1})
    engine = Engine(tuple(Resource(name, 1e9) for name in names),
                    pools={"workspace": 512}, pool_units={"workspace": 1},
                    trace_limit=50)
    holder = Allocation("caller_holder", "workspace", 1, (0,))
    engine.submit("holder", latency=10.0, acquire=(holder,), free=(holder,))
    ready = engine.submit("ready")
    done = plan.program(context).start(engine, "ffn", ready)
    engine.advance(until=(done,))
    rows = {row["task"].removeprefix("ffn:"): row for row in engine.trace}
    assert rows["write_hidden_store"]["start_ms"] >= 10.0
    assert engine.peak_allocated["workspace"] == 512
    assert engine.allocated["workspace"] == 0


def test_cache_recipe_reads_index_once_then_masked_write_and_publish():
    from mindie_sim.prediction.custom_framework_ops import recipes
    from mindie_sim.prediction.mechanism import Geometry

    known = replace(
        _make_op("kv_cache_write", local_shape=(3, 2, 256), mte3_bytes=0.0),
        metadata=(("slot_mapping", (0, -1, 2)),
                  ("cache_capacity_slots", 2)))
    known_low = lower_op(known, op_name="reshape_and_cache_a5",
                         soc_family="Ascend950", kv_page_size=128)
    geometry = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64},
                        {"ub": 196608}, {"ub": 64})
    prepared, plan = recipes()["reshape_and_cache_a5_plan"].prepare(
        known_low, geometry)
    stages = [stage.name for stage in plan.program(
        __import__("mindie_sim.prediction.program", fromlist=["RecipeContext"])
        .RecipeContext(op=prepared, work={},
                       resource_names={"hbm": "hbm", "mte2": "mte2",
                                       "mte3": "mte3", "vector": "vector"},
                       rates={"hbm": 1e9, "mte2": 1e9, "mte3": 1e9,
                              "vector": 1e9},
                       core_count=1)).stages]
    assert stages[:2] == ["slot_external_ready", "slot_index_read"]
    assert stages[2:] == ["masked_cache_write", "cache_publish"]
    assert plan.work.resources["mte2"] == 1024 + 12
    assert plan.work.resources["mte3"] == 1024

    unknown = lower_op(
        _make_op("kv_cache_write", local_shape=(3, 2, 256), mte3_bytes=0.0),
        op_name="reshape_and_cache_a5", soc_family="Ascend950",
        kv_page_size=128)
    prepared_u, plan_u = recipes()["reshape_and_cache_a5_plan"].prepare(
        unknown, geometry)
    stages_u = [stage.name for stage in plan_u.program(
        __import__("mindie_sim.prediction.program", fromlist=["RecipeContext"])
        .RecipeContext(op=prepared_u, work={},
                       resource_names={"hbm": "hbm", "mte2": "mte2",
                                       "mte3": "mte3", "vector": "vector"},
                       rates={"hbm": 1e9, "mte2": 1e9, "mte3": 1e9,
                              "vector": 1e9},
                       core_count=1)).stages]
    assert stages_u == ["slot_external_ready", "slot_index_read"]
    assert plan_u.work.resources["mte3"] == 0
    assert plan_u.metadata["cache_conditional_write_upper_bound"] == 3072
