"""Unit tests for xLLM custom_xllm_math operator lowerings in mindie_sim."""
from __future__ import annotations

import pytest

from mindie_sim.prediction.family_mindie_sd_ops import (
    MINDIE_SD_CUSTOM_OPS,
    compute_mindie_sd_effective_denoising_steps,
    lower_mindie_sd_custom_op,
)
from mindie_sim.prediction.family_xllm_ops import XLLM_CUSTOM_OPS, XLLM_VENDOR, lower_op
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
    assert sfa.mte3_bytes == base_sfa.mte3_bytes + 16 * 4
    assert sfa_meta["xllm_lse_bytes"] == 64

    # 3. moe_grouped_matmul_swiglu_quant
    base_moe = _make_op("grouped_gemm", local_shape=(32, 4096), mte3_bytes=32 * 4096 * 2)
    moe = lower_op(base_moe, op_name="moe_grouped_matmul_swiglu_quant")
    moe_meta = dict(moe.metadata)
    assert moe.recipe == "moe_grouped_matmul_swiglu_quant"
    assert moe_meta["native_kind"] == "MoeGroupedMatmulSwigluQuant"
    assert moe_meta["xllm_swiglu_quant_fused"] is True
    assert "dynamic_scale" in moe.returns
    assert moe.mte3_bytes == 32 * 2048 + 32 * 4

    # 4. mega_gdn_decode
    gdn = lower_op(_make_op("gdn_decode", local_shape=(4, 2048)), op_name="mega_gdn_decode")
    gdn_meta = dict(gdn.metadata)
    assert gdn.recipe == "mega_gdn_decode"
    assert gdn_meta["native_kind"] == "MegaGdnDecode"
    assert "recurrent_state" in gdn.returns

    # 5. onerec_final_beam_select
    beam = lower_op(_make_op("beam_select", local_shape=(4, 1024), mte3_bytes=128.0), beam_width=64)
    beam_meta = dict(beam.metadata)
    assert beam.recipe == "onerec_final_beam_select"
    assert beam_meta["native_kind"] == "OnerecFinalBeamSelect"
    assert beam_meta["beam_width"] == 64
    assert {"beam_ids", "beam_scores"} <= set(beam.returns)
    assert beam.mte3_bytes == 4 * 64 * (8 + 4)

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
            "elementwise": "xllm_onerec_final_beam_select",
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
    rf = lower_mindie_sd_custom_op(attn_op, op_name="rainfusion", sparsity_ratio=0.75)
    assert rf["op_name"] == "mindie_sd_rainfusion"
    assert rf["bound"] == "cube"
    assert rf["cube_flops"] == pytest.approx(2_500_000.0)
    assert rf["sparsity_ratio"] == pytest.approx(0.75)

    qbsa = lower_mindie_sd_custom_op(attn_op, op_name="eagle_qbsa", sparsity_ratio=0.5)
    bsa = lower_mindie_sd_custom_op(attn_op, op_name="eagle_bsa", sparsity_ratio=0.5)
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
            "indexed_attention": "mindie_sd_ada_bsa",
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
    ada_implicit = lower_mindie_sd_custom_op(_make_op("ada_bsa", recipe="indexed_attention", local_shape=(128, 128)))
    assert ada_implicit["op_name"] == "mindie_sd_ada_bsa"

    # DiTCache / TeaCache / TaylorSeer step calculation
    steps = compute_mindie_sd_effective_denoising_steps(infer_steps=50, warmup_steps=2, cache_interval=2)
    assert steps["infer_steps"] == 50
    assert steps["computed_steps"] == 26
    assert steps["cached_steps"] == 24
    assert steps["speedup_factor"] == pytest.approx(50.0 / 26.0)


