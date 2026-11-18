"""End-to-end compose/entry-point parity gate across A2, A3 and A5.

[Role] Prove that the composed catalog declaration, the exported structure, the
  independent operator graph and body prediction select one operator stack, and
  that the A5 V4.1 packed-cache ABI selects the mixed-quant sparse MLA kernel
  while V4.1 on A2/A3 and V4 on A5 never claim the legacy shared-KV kernel.
[Invariants] Every assertion reads the real export schema (``report["nodes"]``
  and ``node["implementation"]["native_kinds"]``). Catalog composition alone is
  not serving validity: standalone draft entries are attached to their target.
  Hardware/quantization geometry selects a kernel; a model name never does.
[Timeouts & Locks] In-memory compose/export/body prediction only; no device.
[Side-Effect Policy] Read-only over the catalog; writes nothing.
"""
from __future__ import annotations

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import list_models
from mindie_sim.prediction import predict
from mindie_sim.prediction.body_defaults import effective_body_declaration
from mindie_sim.prediction.family_mqsmla import FORMATS as MQSMLA_FORMATS
from mindie_sim.prediction.family_mqsmla import RECIPE as MQSMLA_RECIPE
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure

SKUS = ("a2", "a3", "a5")
#: CANN 9.2.1 official mixed-quant sparse MLA native kind.
MQSMLA_NATIVE = "aclnnMixedQuantSparseFlashMla"
LEGACY_MQSMLA_NATIVE = "MqsmlaKernel___call__"
SHARED_KV_NATIVE = "SparseAttnSharedkv"
SPARSE_MLA_NATIVE = "SparseFlashMla"
MIXED_QUANT_NATIVE = "MixedQuantSparseFlashMla"
V41_CACHE_KEYS = ("window", "compressed_kv", "index_k")
PACKED_ROW_BYTES = {"window": 544, "compressed_kv": 320, "index_k": 68}
# One cache entry is the 512-wide absorbed MLA row; the index region is the
# 128-element index head.
REGION_WIDTH = {"window": 512, "compressed_kv": 512, "index_k": 128}
DEFAULT_CONTEXT = 2048
DEFAULT_MAX_MODEL_LEN = 32768
CATALOG_MODELS = tuple(entry["key"] for entry in list_models())


def attention_natives(raw_case, *, phase="decode", tokens=1,
                      context=DEFAULT_CONTEXT, sequences=1, scope="body"):
    """Native attention kinds from the real structure export schema."""
    report = export_structure(
        parse_case(raw_case), phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope)
    return {
        native
        for node in report["nodes"]
        if node["family"] == "attention"
        for native in node["implementation"].get("native_kinds", ())
    }


def attention_recipes(raw_case, *, context=DEFAULT_CONTEXT, sequences=1):
    graphs = target_graphs(
        parse_case(raw_case), phase="decode", tokens=1, context=context,
        sequences=sequences, scope="body")
    return {op.recipe for op in graphs[0] if op.family == "attention"}


def selection_summary(raw_case):
    return dict(parse_case(raw_case).runtime.metadata)["operator_selection_summary"]


def row_bytes(model, region):
    from mindie_sim.catalog.cache_format import cache_format
    return cache_format(model, region).row_bytes(REGION_WIDTH[region])


@pytest.mark.parametrize("tp", (1, 2, 4, 8))
def test_dsv41_a5_packed_cache_and_mqsmla_across_tp(tp: int):
    dp = max(1, 8 // tp)
    raw = compose_case(
        "deepseek-v4.1-flash", "a5", tp=tp, dp=dp, ep=8, concurrency=dp,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    declared = raw["model"]["metadata"]["cache_representations"]
    assert tuple(declared) == V41_CACHE_KEYS
    for region, key in enumerate(V41_CACHE_KEYS):
        assert declared[key] == (MQSMLA_FORMATS[key] if key in MQSMLA_FORMATS
                                 else {"dtype": "uint8", "packed_elements": 2,
                                       "group_size": 32, "scale_dtype": "uint8"}), region
    parsed = parse_case(raw)
    assert {row_bytes(parsed.model, region) for region in V41_CACHE_KEYS} == {
        PACKED_ROW_BYTES["window"], PACKED_ROW_BYTES["compressed_kv"],
        PACKED_ROW_BYTES["index_k"]}
    assert row_bytes(parsed.model, "index_k") < REGION_WIDTH["index_k"]
    assert attention_recipes(raw, sequences=dp) == {MQSMLA_RECIPE}
    assert attention_natives(raw, sequences=dp) == {MQSMLA_NATIVE}
    assert SHARED_KV_NATIVE not in attention_natives(raw, sequences=dp)


def test_dsv41_a5_aclgraph_padding_idle_replicas_and_mtp_verify():
    raw = compose_case(
        "deepseek-v4.1-flash", "a5", tp=2, dp=4, ep=8, concurrency=3,
        max_model_len=DEFAULT_MAX_MODEL_LEN, mtp_proposal_tokens=2,
        mtp_acceptance_length=2.0)
    parsed = parse_case(raw)
    # Uneven cohort: one idle replica and one ACLGraph-padded bucket.
    graphs = target_graphs(
        parsed, phase="decode", tokens=1, context=DEFAULT_CONTEXT,
        sequences=(3, 1, 0, 0), scope="body", padded_idle=True)
    assert {op.recipe for op in graphs[0] if op.family == "attention"} == {MQSMLA_RECIPE}
    # MTP verify rows: q_len = 1 + proposal, still a contiguous decode row set.
    verify = target_graphs(
        parsed, phase="decode", tokens=1 + 2, context=DEFAULT_CONTEXT,
        sequences=(1, 1, 1, 0), scope="body")
    assert {op.recipe for op in verify[0] if op.family == "attention"} == {MQSMLA_RECIPE}
    result = predict(raw, phase="decode", tokens=1, context=DEFAULT_CONTEXT,
                     sequences=(1, 1, 1, 0), scope="body")
    assert result["duration_ms"] > 0


def test_dsv41_a5_fp8_quantization_uses_the_same_packed_stack():
    fp8 = compose_case(
        "deepseek-v4.1-flash", "a5", quant="fp8", tp=2, dp=4, ep=8,
        concurrency=4, max_model_len=DEFAULT_MAX_MODEL_LEN)
    default = compose_case(
        "deepseek-v4.1-flash", "a5", tp=2, dp=4, ep=8, concurrency=4,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    assert (fp8["model"]["metadata"]["cache_representations"]
            == default["model"]["metadata"]["cache_representations"])
    assert fp8["model"]["quant"]["expert_weight_format"] != "mxfp4"
    assert attention_natives(fp8, sequences=4) == {MQSMLA_NATIVE}
    assert attention_recipes(fp8, sequences=4) == {MQSMLA_RECIPE}


def test_dsv41_a5_tp16_falls_back_with_an_explicit_reason():
    raw = compose_case(
        "deepseek-v4.1-flash", "a5", tp=16, dp=1, ep=1, concurrency=1, sp=False,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    summary = selection_summary(raw)
    reasons = {note["reason"] for note in summary["degraded_fallbacks"]}
    assert "mqsmla_local_heads_outside_source_tp_tiling" in reasons
    assert all(note["family"] != "mqsmla_decode" for note in summary["selected_native"])
    assert any("mqsmla_decode->analytical" in warning
               for warning in summary["selection_warnings"])
    # Composition and lowering stay clean on analytical execution: the packed
    # cache ABI is a hardware declaration and is retained, while the kernel
    # binding is withdrawn, so the producer is the mixed-quant sparse-MLA
    # family rather than a legacy shared-KV operator.
    natives = attention_natives(raw)
    assert MQSMLA_NATIVE not in natives
    assert SHARED_KV_NATIVE not in natives
    assert natives == {MIXED_QUANT_NATIVE}


def test_dsv41_explicit_bf16_cache_override_is_not_relabelled():
    from mindie_sim.compose_operator_defaults import (
        CACHE_OVERRIDE_REASON, apply_compose_v41_a5_defaults,
    )
    from mindie_sim.model_catalog import load_model

    model = load_model("deepseek-v4.1-flash")
    model_dict = {"metadata": {"cache_representations": {
        "window": {"dtype": "bf16"},
        "compressed_kv": {"dtype": "bf16"},
    }}}
    lowerings: dict = {}
    notes = apply_compose_v41_a5_defaults(
        model_dict=model_dict, lowerings=lowerings, backend="ascend", arch="a5",
        layer_plans=model.layer_plans,
        activation_dtype=model.activation_dtype, tp=2, cp=1,
        runtime_metadata={})
    assert CACHE_OVERRIDE_REASON in {note.get("reason") for note in notes}
    assert "mqsmla_attention" not in lowerings
    assert model_dict["metadata"]["cache_representations"]["window"] == {"dtype": "bf16"}

    # The same override on a composed case keeps analytical execution instead
    # of relabelling the declared BF16 regions as the packed native stack.
    raw = compose_case(
        "deepseek-v4.1-flash", "a5", tp=2, dp=4, ep=8, concurrency=4,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    raw["model"]["metadata"]["cache_representations"] = {
        "window": {"dtype": "bf16"},
        "compressed_kv": {"dtype": "bf16"},
    }
    parsed = parse_case(raw)
    _effective, selection = effective_body_declaration(parsed, "decode", tokens=1)
    reasons = {note["reason"] for note
               in selection["operator_selection_summary"]["degraded_fallbacks"]}
    assert any("cache_representation_contract_required" in reason for reason in reasons)
    natives = attention_natives(parsed)
    assert MQSMLA_NATIVE not in natives
    assert natives == {SPARSE_MLA_NATIVE}


@pytest.mark.parametrize("hardware", ("a2", "a3"))
def test_dsv41_on_a2_a3_is_sparse_flash_mla_only(hardware: str):
    raw = compose_case(
        "deepseek-v4.1-flash", hardware, tp=4, dp=2, ep=8, concurrency=2,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    natives = attention_natives(raw, sequences=2)
    assert SHARED_KV_NATIVE not in natives
    assert natives == {SPARSE_MLA_NATIVE}
    assert MQSMLA_NATIVE not in natives


def test_dsv4_shared_kv_is_restricted_to_a2_a3():
    a3 = compose_case(
        "deepseek-v4-flash-0731", "a3", tp=8, dp=1, ep=8, concurrency=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    assert attention_natives(a3) == {SHARED_KV_NATIVE}
    a5 = compose_case(
        "deepseek-v4-flash-0731", "a5", tp=8, dp=1, ep=8, concurrency=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    a5_natives = attention_natives(a5)
    assert SHARED_KV_NATIVE not in a5_natives
    assert a5_natives <= {SPARSE_MLA_NATIVE, MIXED_QUANT_NATIVE}
    assert a5_natives


def test_dsv4_a5_quantized_cache_binds_mixed_quant_sparse_flash_mla():
    raw = compose_case(
        "deepseek-v4-flash-0731", "a5", quant="w8a8", tp=8, dp=1, ep=8,
        concurrency=1, max_model_len=DEFAULT_MAX_MODEL_LEN)
    parsed = parse_case(raw)
    cache_dtype = parsed.model.quant.kv_dtype
    natives = attention_natives(parsed)
    expected = (MIXED_QUANT_NATIVE if cache_dtype not in {"bf16", "fp16"}
                else SPARSE_MLA_NATIVE)
    assert natives == {expected}


def test_paged_mla_source_kernels_cover_both_generations():
    kimi_a5 = compose_case(
        "kimi-k3", "a5", tp=8, dp=1, ep=8, concurrency=4,
        max_model_len=65536)
    assert attention_recipes(kimi_a5, context=32768, sequences=4) == {
        "ascend_paged_mla_decode_arch35"}
    # The same kernel now spans several sequences and a 128k context.
    assert attention_recipes(kimi_a5, context=126000, sequences=4) == {
        "ascend_paged_mla_decode_arch35"}

    dsr1_a2 = compose_case(
        "deepseek-r1", "a2", tp=8, dp=1, ep=8, concurrency=4,
        max_model_len=8192)
    assert attention_recipes(dsr1_a2, context=1024, sequences=4) == {
        "ascend_paged_mla_decode_arch22"}
    # Below the declared page tile the arch22 kernel stays analytical.
    assert "ascend_paged_mla_decode_arch22" not in attention_recipes(
        dsr1_a2, context=256, sequences=4)


def test_windowed_gqa_layers_keep_analytical_execution_with_a_reason():
    """The staged decode source rejects a non-zero window.

    A hybrid sliding-window + full-GQA model keeps analytical execution there,
    and the exact cause is exported instead of being silently hidden.
    """
    for key in ("gemma-4-26b-a4b", "gemma-4-31b"):
        raw = compose_case(
            key, "a5", tp=8, dp=1, ep=1, concurrency=4,
            max_model_len=DEFAULT_MAX_MODEL_LEN)
        assert "flash_attention_decode" not in attention_recipes(
            raw, context=DEFAULT_CONTEXT, sequences=4), key
        _effective, selection = effective_body_declaration(
            parse_case(raw), "decode", tokens=1)
        reasons = {note["reason"] for note
                   in selection["operator_selection_summary"]["degraded_fallbacks"]}
        assert "windowed_noncausal_gqa_domain" in reasons, key
    # A dense causal GQA model still binds the staged decoder in both scopes.
    dense = compose_case(
        "qwen3-32b", "a5", tp=8, dp=1, ep=1, concurrency=4,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    assert attention_recipes(dense, context=DEFAULT_CONTEXT, sequences=4) == {
        "flash_attention_decode"}


def test_qwen38_flash_next_low_rank_hc_subgraphs_do_not_crash():
    raw = compose_case(
        "qwen3.8-flash-next", "a5", tp=4, dp=1, ep=4, concurrency=2,
        max_model_len=8192)
    report = export_structure(parse_case(raw), phase="decode", tokens=1,
                              context=1024)
    roles = {group["role"] for group in report["subgraphs"]}
    assert {"hc_pre", "hc_post"} <= roles


def test_residual_norm_is_not_gated_by_qk_norm():
    raw = compose_case(
        "qwen3-32b", "a5", tp=8, dp=1, ep=1, concurrency=4,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    lowerings = dict(parse_case(raw).runtime.kernels.get("lowerings") or {})
    assert lowerings.get("add_rmsnorm") == "vllm_ascend_native_norm"


@pytest.mark.parametrize("hardware", SKUS)
def test_all_catalog_models_compose_and_lower_on_every_sku(hardware: str):
    """Every catalog entry composes and lowers; drafts attach to a target."""
    from mindie_sim.catalog.draft import optional_draft_spec

    checked = 0
    drafts = 0
    for key in CATALOG_MODELS:
        raw = None
        for experts in (8, 1):
            try:
                raw = compose_case(
                    key, hardware, tp=8, dp=1, ep=experts, concurrency=1,
                    max_model_len=DEFAULT_MAX_MODEL_LEN)
                break
            except Exception as exc:  # dense models reject ep > 1
                if experts == 1:
                    raise AssertionError(f"{key}/{hardware}: {exc}") from exc
        parsed = parse_case(raw)
        if parsed.model.is_standalone_draft:
            # A standalone draft entry is a real declaration, but serving
            # validity requires its supported target (checked below).
            auxiliary = parsed.model.auxiliary
            assert auxiliary.dspark is not None or auxiliary.mtp is not None
            kind = "dspark" if auxiliary.dspark is not None else "mtp"
            assert optional_draft_spec(parsed.model, kind) is not None
            drafts += 1
            continue
        graphs = target_graphs(
            parsed, phase="decode", tokens=1, context=DEFAULT_CONTEXT,
            sequences=1, scope="body")
        assert graphs, (key, hardware)
        report = export_structure(
            parsed, phase="decode", tokens=1, context=DEFAULT_CONTEXT,
            scope="body")
        assert report["nodes"], (key, hardware)
        checked += 1
    assert checked + drafts == len(CATALOG_MODELS)
    assert checked > 0


def test_entry_points_agree_on_the_selected_operator_stack():
    raw = compose_case(
        "deepseek-v4.1-flash", "a5", tp=2, dp=4, ep=8, concurrency=4,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    parsed = parse_case(raw)
    geometry = dict(phase="decode", tokens=1, context=DEFAULT_CONTEXT,
                    sequences=4)

    declared = dict(parsed.runtime.metadata)["operator_selection_summary"]
    assert {note["binding"] for note in declared["selected_native"]} >= {MQSMLA_RECIPE}

    effective, selection = effective_body_declaration(parsed, "decode", tokens=1)
    body_selected = {note["binding"] for note
                     in selection["operator_selection_summary"]["selected_native"]}
    assert MQSMLA_RECIPE in body_selected

    bodies = target_graphs(parsed, scope="body", **geometry)
    models = target_graphs(parsed, scope="model", **geometry)
    body_recipes = {op.recipe for op in bodies[0] if op.family == "attention"}
    model_recipes = {op.recipe for op in models[0] if op.family == "attention"}
    assert body_recipes == model_recipes == {MQSMLA_RECIPE}

    structure = export_structure(parsed, scope="body", **geometry)
    exported = {
        native
        for node in structure["nodes"]
        if node["family"] == "attention"
        for native in node["implementation"].get("native_kinds", ())
    }
    assert exported == {MQSMLA_NATIVE}
    assert attention_natives(parsed, **geometry) == exported

    # The independent operator path and the resolved declaration name the same
    # implementation for the selected attention family.
    resolved = {record["family"]: record["selection"]
                for record in models.default_selection["selections"]}
    assert resolved["mqsmla_decode"] == "default_installed"
    assert any(note["binding"] == MQSMLA_RECIPE
               for note in selection["operator_selection_summary"]["selected_native"])
    assert models.scope == "model" and bodies.scope == "body"
    assert models[0].scope == "model" and bodies[0].scope == "body"
