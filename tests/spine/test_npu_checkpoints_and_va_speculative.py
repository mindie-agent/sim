"""Behavioral and invariant tests for NPU checkpoints and vLLM Ascend serving rules."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import urllib.error

import pytest

from mindie_sim.catalog.npu_checkpoints import (
    A2_A3_INT_ONLY_FAMILIES,
    DEFAULT_NPU_CHECKPOINTS,
    ECO_TECH_REPOS,
    MAX_REMOTE_METADATA_BYTES,
    SCANNED_QUANT_PATTERNS,
    checkpoint_cache_dir,
    classify_weight_tensor_quant,
    list_model_checkpoints,
    lookup_eco_tech_repo,
    msmodelslim_quarot_guidance,
    normalize_hw_family,
    parse_quant_model_description,
    rank_checkpoint_candidates,
    resolve_checkpoint_metadata,
    resolve_default_checkpoint,
    select_checkpoint_for_capacity,
    validate_quant_for_hardware,
)
from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.catalog.va_speculative_rules import (
    ASCEND_FIA_MAX_DECODE_TOKENS,
    EP_WEIGHT_FILTER_MIN_EP,
    MAX_SPECULATIVE_TOKENS,
    SPECULATIVE_COMPANIONS,
    build_va_serving_rules,
    clamp_fia_speculative_tokens,
    compute_aclgraph_capture_sizes,
    is_hybrid_recurrent_model,
    is_multimodal_model,
    list_speculative_companions,
    resolve_speculative_companion,
    validate_aclgraph_capture_sizes,
)
from mindie_sim.model_catalog import list_models
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0


@pytest.fixture
def isolated_cache_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate HOME, XDG_CACHE_HOME, and MINDIE_DATA_HOME for every metadata test."""
    home = tmp_path / "home"
    xdg = tmp_path / "xdg_cache"
    data = tmp_path / "mindie_data"
    home.mkdir()
    xdg.mkdir()
    data.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(data))
    return xdg / "mindie-sim" / "checkpoints"


class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def read(self, amt: int = -1) -> bytes:
        return self._buf.read(amt)

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_eco_tech_snapshot_and_full_catalog_coverage() -> None:
    """All 165 Eco-Tech repos are indexed and every catalog model resolves on a2/a3/a5."""
    assert len(ECO_TECH_REPOS) == 165
    assert len(set(ECO_TECH_REPOS)) == 165
    assert lookup_eco_tech_repo("kimi-k3.1-pro-w4a8-mxfp") == "Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp"
    assert lookup_eco_tech_repo("eco-tech/glm-5-w4a8-mtp-quarot") == "Eco-Tech/GLM-5-w4a8-mtp-QuaRot"

    catalog_keys = [m["key"] for m in list_models()]
    assert len(catalog_keys) >= 39
    for key in catalog_keys:
        by_hw = list_model_checkpoints(key)
        assert set(by_hw) == {"a2", "a3", "a5"}, f"missing hw family for {key}: {by_hw}"
        for hw, ckpt in by_hw.items():
            assert ckpt.repo_id
            assert ckpt.modelscope_url.startswith("https://")
            qspec = ckpt.build_quant_spec()
            assert qspec.name
            if hw in A2_A3_INT_ONLY_FAMILIES:
                # Strict hardware-ISA separation: zero MXFP/FP8 on A2/A3.
                assert validate_quant_for_hardware(
                    ckpt.quant_preset, hw, kv_cache_dtype=ckpt.kv_cache_dtype, strict=True,
                ) == ()
                assert "mxfp" not in ckpt.repo_id.lower()
                assert "fp8" not in ckpt.kv_cache_dtype.lower()


def test_hw_family_normalization_and_strict_isa_guards() -> None:
    """SKU/topology aliases normalize cleanly and A2/A3 strictly reject MXFP/FP8."""
    assert normalize_hw_family("Ascend950DT_9582") == "a5"
    assert normalize_hw_family("950pr") == "a5"
    assert normalize_hw_family("Ascend910_9392") == "a3"
    assert normalize_hw_family("910c") == "a3"
    assert normalize_hw_family("Ascend910B4") == "a2"
    assert normalize_hw_family("800i-a2") == "a2"

    for hw in ("a2", "a3"):
        warns = validate_quant_for_hardware("w4a8_mxfp", hw, kv_cache_dtype="fp8")
        assert len(warns) == 2
        with pytest.raises(ModelingError, match="hw_isa_incompatible"):
            validate_quant_for_hardware("w8a8_mxfp8", hw, strict=True)
        with pytest.raises(ModelingError, match="hw_isa_incompatible"):
            validate_quant_for_hardware("w8a8_dynamic", hw, kv_cache_dtype="fp8", strict=True)

    assert validate_quant_for_hardware("w4a8_mxfp", "a5", kv_cache_dtype="fp8", strict=True) == ()


def test_quarot_priority_and_msmodelslim_guidance() -> None:
    """QuaRot checkpoints strictly rank ahead of non-QuaRot peers at equal bitwidth."""
    ranked = rank_checkpoint_candidates(
        ("Eco-Tech/GLM-5-w4a8", "Eco-Tech/GLM-5-w4a8-mtp-QuaRot", "Eco-Tech/GLM-5-w8a8"),
        hw_family="a3",
        preferred_quant="w4a8",
    )
    assert ranked[0] == "Eco-Tech/GLM-5-w4a8-mtp-QuaRot"

    glm5 = resolve_default_checkpoint("glm-5", hw_family="a3")
    assert glm5 is not None
    assert glm5.repo_id == "Eco-Tech/GLM-5-w4a8-mtp-QuaRot"
    assert glm5.uses_quarot is True

    ds_v32 = resolve_default_checkpoint("deepseek-v3.2", hw_family="a3")
    assert ds_v32 is not None and ds_v32.uses_quarot is True
    qwen_235b = resolve_default_checkpoint("qwen3-235b-a22b", hw_family="a3")
    assert qwen_235b is not None and qwen_235b.uses_quarot is True

    guide_a3 = msmodelslim_quarot_guidance("a3")
    guide_a5 = msmodelslim_quarot_guidance("a5")
    assert guide_a3["prefer_quarot"] is True and "C8" in guide_a3["recommended_algorithm"]
    assert guide_a5["prefer_quarot"] is True and "MXFP" in guide_a5["recommended_algorithm"]


def test_capacity_driven_int8_to_int4_and_c8_kv_upgrade() -> None:
    """Tight HBM on INT-only hardware upgrades W8A8 -> W4A8 + C8 KV when repo exists."""
    base = resolve_default_checkpoint("qwen3.8-flash-next", hw_family="a3")
    assert base is not None
    assert base.repo_id == "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp"
    assert base.w4a8_repo_id == "Eco-Tech/Qwen3.8-Flash-Next-w4a8"

    upgraded, notes = select_checkpoint_for_capacity(
        "qwen3.8-flash-next", "a3", w8a8_fits_hbm=False, long_context=True,
    )
    assert upgraded is not None
    assert upgraded.repo_id == "Eco-Tech/Qwen3.8-Flash-Next-w4a8"
    assert upgraded.quant_preset == "w4a8_dynamic"
    assert upgraded.kv_cache_dtype == "int8"
    assert any("w8a8_to_w4a8" in n for n in notes)
    assert any("c8_int8" in n for n in notes)

    qspec = upgraded.build_quant_spec()
    assert qspec.weight_format == "int8"
    assert qspec.expert_weight_format == "int4"
    assert qspec.kv_dtype == "int8"


def test_scanned_pattern_maps_parse_bit_exact_per_tensor_quant() -> None:
    """Exact scanned patterns yield bit-exact QuantSpec overrides and tensor classifications."""
    mxfp_patterns = SCANNED_QUANT_PATTERNS["Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp"]
    assert len(mxfp_patterns["W8A8_MXFP8"]) == 15
    assert len(mxfp_patterns["W4A8_MXFP"]) == 3
    assert len(mxfp_patterns["FLOAT"]) == 39

    qspec = parse_quant_model_description(
        mxfp_patterns, checkpoint_repo="Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp",
    )
    assert qspec.weight_format == "fp8_e4m3"
    assert qspec.expert_weight_format == "mxfp4"
    assert qspec.format_for("routed_expert") == "mxfp4"
    assert qspec.format_for("shared_expert") == "fp8_e4m3"
    assert qspec.format_for("latent_moe") == "fp8_e4m3"
    assert qspec.format_for("dense_mlp") == "fp8_e4m3"
    assert qspec.format_for("mla.q_b") == "fp8_e4m3"
    assert qspec.format_for("mla_absorb") == "bf16"
    assert qspec.format_for("mla.o") == "bf16"
    assert qspec.format_for("kda.g") == "bf16"
    assert qspec.format_for("kda.q") == "fp8_e4m3"
    assert qspec.format_for("indexer.wk") == "bf16"
    assert qspec.format_for("indexer.weights") == "bf16"
    assert qspec.format_for("indexer_fp8") == "fp8_e4m3"
    assert compute_dtype_for(qspec, "mla.q_b") == ("fp8_e4m3", "fp8_e4m3")
    assert compute_dtype_for(qspec, "mla_absorb") == ("bf16", "bf16")

    # Concrete numbered tensor keys match wildcard patterns exactly.
    t_exp = classify_weight_tensor_quant(
        "language_model.model.layers.17.block_sparse_moe.experts.42.w1.weight",
        mxfp_patterns,
    )
    assert t_exp.raw_tag == "W4A8_MXFP"
    assert t_exp.weight_format == "mxfp4"
    assert t_exp.ledger_format == "MXFP4"
    assert t_exp.group_size == 32

    t_kvb = classify_weight_tensor_quant(
        "language_model.model.layers.17.self_attn.kv_b_proj.weight", mxfp_patterns,
    )
    assert t_kvb.raw_tag == "FLOAT" and t_kvb.is_quantized is False and t_kvb.ledger_format == "BF16"

    t_lat = classify_weight_tensor_quant(
        "language_model.model.layers.3.block_sparse_moe.routed_expert_down_proj.weight",
        mxfp_patterns,
    )
    assert t_lat.raw_tag == "W8A8_MXFP8" and t_lat.ledger_format == "MXFP8"

    # Mixed W4A8_DYNAMIC (INT8 ordinary linears + INT4 routed experts + FLOAT kv_b/o_proj).
    dyn_ckpt = DEFAULT_NPU_CHECKPOINTS[("kimi-k3.1", "a3")]
    dyn_qspec = dyn_ckpt.build_quant_spec()
    assert dyn_qspec.weight_format == "int8"
    assert dyn_qspec.expert_weight_format == "int4"
    assert dyn_qspec.format_for("mla.q_a") == "int8"
    assert dyn_qspec.format_for("routed_expert") == "int4"
    assert dyn_qspec.format_for("mla_absorb") == "bf16"
    assert dyn_qspec.kv_dtype == "int8"

    # Qwen3.8-Flash-Next MXFP: self_attn W8A8_MXFP8, experts W4A8_MXFP, GDN/PLE/MTP/mHC FLOAT.
    qf_pats = SCANNED_QUANT_PATTERNS["Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp"]
    qf_qspec = parse_quant_model_description(qf_pats)
    assert qf_qspec.format_for("attn.qkv") == "fp8_e4m3"
    assert qf_qspec.format_for("attn.o") == "fp8_e4m3"
    assert qf_qspec.format_for("routed_expert") == "mxfp4"
    assert qf_qspec.format_for("gdn") == "bf16"
    assert qf_qspec.format_for("shared_expert") == "bf16"
    assert qf_qspec.format_for("mtp") == "bf16"
    t_ple = classify_weight_tensor_quant(
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_99.weight",
        qf_pats,
    )
    assert t_ple.raw_tag == "FLOAT" and t_ple.is_quantized is False


def test_raw_msmodelslim_quant_description_with_c8_kv() -> None:
    """Raw flat {tensor: tag} msmodelslim JSON with C8 KV parses cleanly."""
    raw_desc = {
        "model_quant_type": "W4A8_DYNAMIC",
        "kv_cache_type": "C8",
        "model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
        "model.layers.0.self_attn.o_proj.weight": "W8A8_DYNAMIC",
        "model.layers.0.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
        "model.layers.0.mlp.gate.weight": "FLOAT",
        "lm_head.weight": "FLOAT",
    }
    qspec = parse_quant_model_description(raw_desc, base_config={"torch_dtype": "bfloat16"})
    assert qspec.weight_format == "int8"
    assert qspec.expert_weight_format == "int4"
    assert qspec.kv_dtype == "int8"
    assert qspec.format_for("attn.qkv") == "int8"
    assert qspec.format_for("routed_expert") == "int4"
    assert qspec.format_for("router") == "bf16"


def test_local_checkpoint_metadata_reader(tmp_path: Path) -> None:
    """Local checkpoint directory and config.json file load without network I/O."""
    ckpt_dir = tmp_path / "GLM-5.3-w8a8c8"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(
        json.dumps({"model_type": "glm5", "torch_dtype": "bfloat16", "hidden_size": 6144}),
        encoding="utf-8",
    )
    (ckpt_dir / "quant_model_description.json").write_text(
        json.dumps({
            "kv_cache_type": "C8",
            "model.layers.0.self_attn.q_a_proj.weight": "W8A8_DYNAMIC",
            "model.layers.0.mlp.experts.0.w1.weight": "W8A8_DYNAMIC",
            "lm_head.weight": "FLOAT",
        }),
        encoding="utf-8",
    )
    res_dir = resolve_checkpoint_metadata(ckpt_dir, hw_family="a3", allow_network=False)
    assert res_dir.source_kind == "local_dir"
    assert res_dir.config["hidden_size"] == 6144
    assert res_dir.quant_spec is not None
    assert res_dir.quant_spec.weight_format == "int8"
    assert res_dir.quant_spec.kv_dtype == "int8"
    assert res_dir.warnings == ()

    res_file = resolve_checkpoint_metadata(ckpt_dir / "config.json", hw_family="a3", allow_network=False)
    assert res_file.source_kind == "local_file"
    assert res_file.quant_spec is not None and res_file.quant_spec.kv_dtype == "int8"


def test_remote_metadata_fetch_cache_cap_timeout_and_offline_fallback(
    isolated_cache_env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked remote fetches verify URL routing, atomic caching, 256 KiB cap, TimeoutError, and offline fallback."""
    assert checkpoint_cache_dir() == isolated_cache_env

    seen_urls: list[str] = []

    def _mock_urlopen_ok(req: Any, timeout: float = 0.0) -> _FakeHttpResponse:
        assert timeout == 5.0
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/config.json"):
            return _FakeHttpResponse(json.dumps({"torch_dtype": "bfloat16", "num_hidden_layers": 93}).encode())
        if req.full_url.endswith("/quant_model_description.json"):
            return _FakeHttpResponse(json.dumps({
                "language_model.model.layers.0.self_attn.q_a_proj.weight": "W8A8_MXFP8",
                "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight": "W4A8_MXFP",
                "language_model.lm_head.weight": "FLOAT",
            }).encode())
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen_ok)
    res = resolve_checkpoint_metadata(
        "https://modelers.cn/models/Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp", hw_family="a5",
    )
    assert res.source_kind == "modelers"
    assert res.from_cache is False
    assert any("modelers.cn/coderepo/web/v1/file/Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp/main/media/config.json" in u for u in seen_urls)
    assert res.quant_spec is not None and res.quant_spec.expert_weight_format == "mxfp4"

    # Second call serves from isolated cache without invoking urlopen.
    def _fail_if_called(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("network must not be touched on cache hit")

    monkeypatch.setattr("urllib.request.urlopen", _fail_if_called)
    res_cached = resolve_checkpoint_metadata(
        "https://modelers.cn/models/Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp", hw_family="a5",
    )
    assert res_cached.from_cache is True and res_cached.source_kind == "cache"

    # Oversize (> 256 KiB) response is rejected cleanly and falls back to baked defaults.
    def _mock_oversize(req: Any, timeout: float = 0.0) -> _FakeHttpResponse:
        del req, timeout
        return _FakeHttpResponse(b"{" + b" " * (MAX_REMOTE_METADATA_BYTES + 16) + b"}")

    monkeypatch.setattr("urllib.request.urlopen", _mock_oversize)
    res_big = resolve_checkpoint_metadata("Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp", hw_family="a5")
    assert res_big.offline_fallback is True
    assert any("exceeds_256KiB" in w for w in res_big.warnings)
    assert res_big.quant_spec is not None and res_big.quant_spec.expert_weight_format == "mxfp4"

    # Offline URL error degrades cleanly to built-in registry.
    def _mock_offline(req: Any, timeout: float = 0.0) -> Any:
        del req, timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", _mock_offline)
    res_off = resolve_checkpoint_metadata(
        "https://www.modelscope.cn/models/Eco-Tech/Kimi-K3.1-Pro-w4a8", hw_family="a3",
    )
    assert res_off.offline_fallback is True
    assert res_off.quant_spec is not None and res_off.quant_spec.expert_weight_format == "int4"

    # TimeoutError strictly propagates (never masked as OSError).
    def _mock_timeout(req: Any, timeout: float = 0.0) -> Any:
        del req, timeout
        raise TimeoutError("simulated socket timeout")

    monkeypatch.setattr("urllib.request.urlopen", _mock_timeout)
    with pytest.raises(TimeoutError, match="simulated socket timeout"):
        resolve_checkpoint_metadata("https://huggingface.co/Qwen/Qwen3-32B", hw_family="a5")


def test_speculative_companions_draft_tp_and_kv_dtype_rules() -> None:
    """Non-MTP and MTP models resolve documented companions, draft TP, and BF16 draft KV overrides."""
    # Kimi-K3.1: DSpark external draft matches target TP and forces bfloat16 KV when target KV is fp8.
    k31 = resolve_speculative_companion(
        "kimi-k3.1", target_tp=8, target_kv_dtype="fp8", hw_family="a5",
    )
    assert k31.method == "dspark"
    assert k31.draft_model == "Eco-Tech/Kimi-K3-DSpark-MLA-NPU"
    assert k31.num_speculative_tokens == 5
    assert k31.draft_tensor_parallel_size == 8
    assert k31.draft_kv_cache_dtype == "bfloat16"
    assert k31.speculative_config["kv_cache_dtype"] == "bfloat16"
    assert k31.speculative_config["draft_tensor_parallel_size"] == 8

    # Kimi-K2.6: EAGLE3 primary (draft_tp=1) + DFlash alternative (K=15).
    k26_e3 = resolve_speculative_companion("kimi-k2.6", target_tp=16)
    assert k26_e3.method == "eagle3"
    assert k26_e3.draft_model == "lightseekorg/kimi-k2.6-eagle3"
    assert k26_e3.draft_tensor_parallel_size == 1
    assert k26_e3.num_speculative_tokens == 3
    k26_df = resolve_speculative_companion("kimi-k2.6", target_tp=16, method="dflash")
    assert k26_df.method == "dflash"
    assert k26_df.draft_model == "z-lab/Kimi-K2.5-DFlash"
    assert k26_df.num_speculative_tokens == 15

    # MiniMax-M3: EAGLE3 (draft_tp=1, BF16 draft KV on FP8 target).
    mm3 = resolve_speculative_companion("minimax-m3", target_tp=8, target_kv_dtype="fp8")
    assert mm3.method == "eagle3" and mm3.draft_tensor_parallel_size == 1
    assert mm3.draft_kv_cache_dtype == "bfloat16"

    # Qwen3-8B / Qwen3-30B-A3B / Qwen3.8-27B / Llama-3* companions.
    assert resolve_speculative_companion("qwen3-8b", method="dflash").draft_model == "z-lab/Qwen3-8B-DFlash-b16"
    assert resolve_speculative_companion("qwen3-8b", method="dspark").draft_model == "deepseek-ai/dspark_qwen3_8b_block7"
    assert resolve_speculative_companion("qwen3-30b-a3b", target_tp=4).draft_tensor_parallel_size == 1
    q27_opts = list_speculative_companions("qwen3.8-27b")
    assert q27_opts[0].method == "qwen3_5_mtp"
    assert any(c.draft_model == "z-lab/Qwen3.8-27B-DFlash2" and c.num_speculative_tokens == 8 for c in q27_opts)
    assert any(c.draft_model == "niannianxu/Qwen3.8-27B-LiLiCorr-block8-epoch0" and c.enforce_eager_draft for c in q27_opts)

    llama = resolve_speculative_companion("meta-llama/Meta-Llama-3.1-8B-Instruct", target_tp=4)
    assert llama.method == "eagle3" and llama.draft_tensor_parallel_size == 1
    assert "yuhuili/" in llama.draft_model
    assert any(c.method == "ngram" for c in llama.alternatives)

    # Built-in DeepSeek-V4 / V3 / GLM-5 companions exist.
    assert "deepseek-v4-pro-0813" in SPECULATIVE_COMPANIONS
    assert resolve_speculative_companion("deepseek-v4-pro-0813").method == "dspark"
    assert resolve_speculative_companion("deepseek-v3.2").method == "deepseek_mtp"
    assert resolve_speculative_companion("glm-5.3").method == "deepseek_mtp"


def test_fia_ceiling_and_aclgraph_capture_multiples() -> None:
    """Ascend FIA 16-token ceiling clamps K > 15 and ACLGraph emits exact n*(K+1) multiples."""
    clamped_k, warn = clamp_fia_speculative_tokens(20)
    assert clamped_k == MAX_SPECULATIVE_TOKENS == ASCEND_FIA_MAX_DECODE_TOKENS - 1
    assert warn is not None and "fia_16_token_limit_clamped" in warn

    ok_k, ok_warn = clamp_fia_speculative_tokens(15)
    assert ok_k == 15 and ok_warn is None

    sizes = compute_aclgraph_capture_sizes(
        [1, 2, 3, 4], num_speculative_tokens=4, cudagraph_mode="FULL_DECODE_ONLY",
    )
    assert sizes == (5, 10, 15, 20)
    assert validate_aclgraph_capture_sizes(sizes, num_speculative_tokens=4) == ()
    bad_warns = validate_aclgraph_capture_sizes([1, 4, 8], num_speculative_tokens=4)
    assert len(bad_warns) == 1 and "not_multiples_of_5" in bad_warns[0]


def test_pd_disaggregation_rules_and_serving_cli_guards() -> None:
    """PD rules enforce hybrid matching-K vs pure-attention P=1/D=K, plus host-RAM & ISA guards."""
    assert is_hybrid_recurrent_model("qwen3.8-flash-next") is True
    assert is_hybrid_recurrent_model("kimi-k3.1") is True
    assert is_hybrid_recurrent_model("glm-5.3") is False
    assert is_multimodal_model("kimi-k3.1") is True

    # Hybrid recurrent model under PD disaggregation: P and D share identical K=3.
    hybrid_pd = build_va_serving_rules(
        "qwen3.8-flash-next", hw_family="a5", tp=4, ep=16,
        pd_disaggregated=True, pd_role="prefill", text_only_serving=True,
        batch_sizes=[1, 2, 4],
    )
    assert hybrid_pd.prefill_speculative_config["num_speculative_tokens"] == 3
    assert hybrid_pd.decode_speculative_config["num_speculative_tokens"] == 3
    assert hybrid_pd.compilation_config["cudagraph_capture_sizes"] == [4, 8, 16]
    assert hybrid_pd.enable_ep_weight_filter is True
    assert "--enable-ep-weight-filter" in hybrid_pd.cli_flags
    assert "--mm-processor-cache-gb 0" in hybrid_pd.cli_flags
    assert "--safetensors-load-strategy prefetch" in hybrid_pd.cli_flags
    assert hybrid_pd.additional_config["enable_cpu_binding"] is True

    # Pure-attention model under PD disaggregation: P=1, D=K.
    pure_pd = build_va_serving_rules(
        "glm-5.2", hw_family="a3", tp=8, ep=EP_WEIGHT_FILTER_MIN_EP,
        target_kv_dtype="fp8", pd_disaggregated=True, pd_role="decode",
        num_speculative_tokens=5,
    )
    assert pure_pd.prefill_speculative_config["num_speculative_tokens"] == 1
    assert pure_pd.decode_speculative_config["num_speculative_tokens"] == 5
    # A3 rejects FP8 KV and normalizes to C8 int8 + enables weight NZ layout + dual-Die HCCS guidance.
    assert pure_pd.effective_kv_cache_dtype == "int8"
    assert any("a2_a3_fp8_kv_unsupported_using_int8_c8" in w for w in pure_pd.warnings)
    assert "--kv-cache-dtype int8" in pure_pd.cli_flags
    assert pure_pd.additional_config["enable_weight_nz_layout"] is True
    assert "HCCS" in pure_pd.collective_guidance and "SIO" in pure_pd.collective_guidance

    # A2 cross-node RoCE guidance + FIA clamp warning.
    a2_rules = build_va_serving_rules(
        "kimi-k2.6", hw_family="a2", tp=16, ep=4, spec_method="dflash",
        num_speculative_tokens=20,
    )
    assert a2_rules.speculative_config["num_speculative_tokens"] == 15
    assert any("fia_16_token_limit_clamped" in w for w in a2_rules.warnings)
    assert "RoCE" in a2_rules.collective_guidance
    assert "--no-enable-prefix-caching" in a2_rules.cli_flags
    assert a2_rules.enable_ep_weight_filter is False
    assert a2_rules.additional_config["enable_weight_nz_layout"] is True
