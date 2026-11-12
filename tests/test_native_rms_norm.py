"""Bounded focused behavior for the normalized Gemma initial RMS leaf.

[Role] Prove the exact archived R31 host/ACLNN call, the generic normalized
  AscendGemmaRMSNorm class on another source identity, wrong-override
  rejection, unchanged adjacent non-Gemma cases, and the zero-metadata
  negative on the real layer-0 N0 only.
[Invariants] Generic class semantics require explicit class/role/residual/
  weight-offset/block; exact R31 provenance is used only when fully archived
  and matching.  Generic source does not assert Qwen lines/commit/image/ELF.
  Captured rows and the host summary live only in this test.
[Timeouts & Locks] One layer-0 instantiation, pure metadata, two small reads.
[Side-Effect Policy] Read-only repository case and retained raw capture.
"""
from __future__ import annotations

import csv
import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.native_calls import (
    validate_native_occurrence,
)
from mindie_sim.prediction.piecewise.native_rms_norm import (
    API_STATISTIC_PATH, R31_IMAGE_ID, R31_IMAGE_TAG, RAW_GAMMA_SHIFT_ROW,
    RAW_RMS_NORM_ROW, SOURCE_RAW_PATH, VLLM_ASCEND_DEVICE_OP_SHA,
    VLLM_ASCEND_LAYERNORM_SHA, VLLM_ASCEND_REVISION, VLLM_QWEN3_5_SHA,
    VLLM_REVISION, binding_spec, dynamic_call,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_layer,
)

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"
DOMAIN = AnalysisDomain(max_tokens=10 ** 9, max_moe_rows=10 ** 9,
                        max_query_tiles=10 ** 9, max_kv_tiles=10 ** 9,
                        max_index_rows=10 ** 12, max_work=10 ** 15,
                        max_q=10 ** 9, max_s_kv=10 ** 9, max_seq=10 ** 9,
                        max_batch=10 ** 6)
EXACT = {
    "image_id": R31_IMAGE_ID,
    "image": R31_IMAGE_TAG,
    "vllm_ascend_commit": VLLM_ASCEND_REVISION,
    "vllm_ascend_package": "0.30.0rc2.dev200+g8fd0bc488",
    "vllm_ascend_layernorm_sha256": VLLM_ASCEND_LAYERNORM_SHA,
    "vllm_ascend_device_op_sha256": VLLM_ASCEND_DEVICE_OP_SHA,
    "vllm_commit": VLLM_REVISION,
    "vllm_qwen3_5_sha256": VLLM_QWEN3_5_SHA,
    "rms_norm_source_class": "AscendGemmaRMSNorm",
    "rms_norm_source_weight_offset": 1.0,
    "rms_norm_source_role": "input_layernorm.initial",
    "rms_norm_source_residual": False,
    "rms_norm_source_block": "initial",
    "source_model_type": "qwen3_5_text",
    "arch_family": "A5",
}
GENERIC = {
    "rms_norm_source_class": "AscendGemmaRMSNorm",
    "rms_norm_source_weight_offset": 1.0,
    "rms_norm_source_role": "input_layernorm.initial",
    "rms_norm_source_residual": False,
    "rms_norm_source_block": "initial",
    "source_model_type": "llama3_text",
    "arch_family": "A3",
}


def _features(extra: dict | None):
    case = parse_case(json.loads(QWEN.read_text()))
    features = FeatureConfig.from_model(
        case.model, tp=2, dp=4, ep=1, pp=1, cores=64,
        token_tile=16, kv_tile=16, moe_row_tile=4,
        split_target=1000, max_splits=4)
    meta = dict(features.metadata)
    meta["additional_config"] = {"enable_flashcomm1": "1"}
    if extra:
        meta.update(extra)
    return case, replace(features, metadata=tuple(sorted(meta.items())))


def _compiled(case, features, role: str):
    layer = case.model.layer_plans[0]
    instantiated = instantiate_layer(case.model, layer, features)
    instance = next(
        item for item in instantiated.subgraphs
        if item.kind == "N0" and dict(item.metadata).get("role") == role)
    compiler = OperatorPiecewiseCompiler(
        PipeSupply.from_hardware(case.machine, dtype="bf16"),
        ImplementationPolicy(cores=64), domain=DOMAIN)
    return compiler.compile(
        instance, feature_metadata=features.metadata).operator("rmsnorm")


def _prefill(tokens: int):
    return WorkloadGeometry(
        (RequestGeometry(tokens, tokens, phase="prefill"),),
        phase="prefill", cohort_batch=1, local_batch=1)


def _binding(spec):
    return SimpleNamespace(
        api=spec.api, typed=spec.typed, call_id="r31-gemma-normalized",
        source=spec.source, kernels=spec.kernels, unresolved=spec.unresolved)


def test_exact_r31_still_bound_with_archived_provenance():
    case, features = _features(EXACT)
    node = _compiled(case, features, "attn_norm")
    spec = binding_spec(node, phase="prefill")
    assert spec is not None
    assert spec.api["entry"] == "torch_npu.npu_rms_norm"
    assert spec.api["aclnn_symbol"] == "aclnnRmsNorm"
    assert spec.api["acl_symbol_status"] == "affirmed_by_original_host_summary"
    assert spec.source["sha256"] == VLLM_ASCEND_LAYERNORM_SHA
    assert spec.typed["attrs"]["source_class"] == "AscendGemmaRMSNorm"
    assert spec.typed["attrs"]["source_weight_offset"] == 1.0
    envelope = dynamic_call(
        _binding(spec), node, _prefill(4096),
        validate=validate_native_occurrence)
    assert envelope is not None
    prepare = envelope["prepare"]
    assert prepare["source_exactness"] is True
    assert prepare["selected_acl_symbol"] == "aclnnRmsNorm"
    assert prepare["complete"] is True
    assert prepare["source_trace"]["host_summary"]["sha256"] == (
        "e19308dea200ead700b417475da5983576ec8884646e3b03e833be6b06e0808e")
    assert prepare["preceding_gamma_shift"]["raw_row"] == RAW_GAMMA_SHIFT_ROW
    assert prepare["physical_work"]["observed_core"] == "AI_VECTOR_CORE"


def test_generic_normalized_gemma_binds_general_semantics():
    case, features = _features(GENERIC)
    node = _compiled(case, features, "attn_norm")
    spec = binding_spec(node, phase="prefill")
    assert spec is not None
    assert spec.api["entry"] == "torch_npu.npu_rms_norm"
    assert spec.api["aclnn_symbol"] is None
    assert spec.api["candidates"] == []
    assert spec.source is None
    assert spec.typed["attrs"]["source_class"] == "AscendGemmaRMSNorm"
    assert spec.typed["attrs"]["source_weight_offset"] == 1.0
    assert spec.typed["attrs"]["source_residual"] == "absent"
    assert spec.typed["attrs"]["source_block"] == "initial"
    envelope = dynamic_call(
        _binding(spec), node, _prefill(4096),
        validate=validate_native_occurrence)
    assert envelope is not None
    prepare = envelope["prepare"]
    assert prepare["source_tier"] == "generic_normalized_gemma"
    assert prepare["source_exactness"] is False
    assert prepare["general_semantics"] is True
    assert prepare["host_variant_uncertain"] is True
    assert prepare["abi_binding_reality"] is False
    assert prepare["selected_acl_symbol"] is None
    assert prepare["complete"] is True
    assert prepare["partial"] is False
    assert prepare["shapes"]["x"] == [2048, 5120]
    assert prepare["shapes"]["rstd"] == [2048, 1]
    trace = prepare["source_trace"]
    assert trace["normalized_source"]["class"] == "AscendGemmaRMSNorm"
    assert trace["normalized_source"]["role"] == "input_layernorm.initial"
    assert trace["normalized_source"]["residual"] == "absent"
    assert trace["normalized_source"]["weight_offset"] == 1.0
    assert trace["normalized_source"]["block"] == "initial"
    assert trace["qwen_source"] is None
    assert trace["host_summary"] is None
    assert "vllm" not in trace and "vllm_ascend" not in trace
    assert prepare["match"]["source_image"] is None
    assert prepare["preceding_gamma_shift"]["evidence"] == "class_semantics"
    assert prepare["preceding_gamma_shift"]["raw_row"] is None
    assert prepare["physical_work"]["observed_core"] is None
    assert prepare["physical_work"]["kernel_evidence"] is None


def test_generic_exact_r31_wrapper_provenance_does_not_claim_qwen_model():
    case, features = _features({**EXACT, "source_model_type": "llama3_text",
                                "arch_family": "A3"})
    node = _compiled(case, features, "attn_norm")
    spec = binding_spec(node, phase="prefill")
    assert spec is not None
    assert spec.source is None
    assert spec.api["aclnn_symbol"] is None
    envelope = dynamic_call(
        _binding(spec), node, _prefill(4096),
        validate=validate_native_occurrence)
    assert envelope is not None
    prepare = envelope["prepare"]
    assert prepare["source_exactness"] is False
    assert prepare["source_tier"] == "generic_normalized_gemma"
    wrapper = prepare["source_trace"]["wrapper_provenance"]
    assert wrapper["layernorm_sha256"] == VLLM_ASCEND_LAYERNORM_SHA
    assert wrapper["device_op_sha256"] == VLLM_ASCEND_DEVICE_OP_SHA
    assert prepare["source_trace"]["qwen_source"] is None
    assert "vllm" not in prepare["source_trace"]
    assert prepare["match"]["source_image"] is None


def test_empty_zero_metadata_and_partial_are_negative_or_generic():
    for extra in (None, {"arch_family": "A5", "source_model_type": "qwen3_5_text",
                         "vllm_ascend_commit": VLLM_ASCEND_REVISION}):
        case, features = _features(extra)
        assert binding_spec(_compiled(case, features, "attn_norm"),
                            phase="prefill") is None
    for key in ("rms_norm_source_class", "rms_norm_source_weight_offset",
                "rms_norm_source_role"):
        partial = {k: v for k, v in GENERIC.items() if k != key}
        case, features = _features(partial)
        assert binding_spec(_compiled(case, features, "attn_norm"),
                            phase="prefill") is None
    no_provenance = {k: v for k, v in EXACT.items()
                     if k not in {"image_id", "image", "vllm_ascend_commit",
                                  "vllm_ascend_package",
                                  "vllm_ascend_layernorm_sha256",
                                  "vllm_ascend_device_op_sha256",
                                  "vllm_commit", "vllm_qwen3_5_sha256"}}
    case, features = _features(no_provenance)
    node = _compiled(case, features, "attn_norm")
    spec = binding_spec(node, phase="prefill")
    assert spec is not None and spec.api["aclnn_symbol"] is None
    envelope = dynamic_call(
        _binding(spec), node, _prefill(4096),
        validate=validate_native_occurrence)
    assert envelope is not None
    assert envelope["prepare"]["source_exactness"] is False


def test_wrong_override_and_adjacent_non_gemma_are_unchanged():
    for patch in ({"rms_norm_source_class": "AscendRMSNorm"},
                  {"rms_norm_source_weight_offset": 0.0},
                  {"rms_norm_source_role": "post_attention_layernorm"},
                  {"rms_norm_source_residual": True},
                  {"rms_norm_source_block": "final"}):
        case, features = _features({**GENERIC, **patch})
        assert binding_spec(_compiled(case, features, "attn_norm"),
                            phase="prefill") is None
    case, features = _features(EXACT)
    assert binding_spec(_compiled(case, features, "ffn_norm"),
                        phase="prefill") is None
    case, features = _features(None)
    assert binding_spec(_compiled(case, features, "attn_norm"),
                        phase="prefill") is None


def test_source_exact_without_image_keeps_exact_call():
    no_image = {k: v for k, v in EXACT.items()
                if k not in {"image_id", "image"}}
    case, features = _features(no_image)
    node = _compiled(case, features, "attn_norm")
    spec = binding_spec(node, phase="prefill")
    assert spec is not None
    envelope = dynamic_call(
        _binding(spec), node, _prefill(4096),
        validate=validate_native_occurrence)
    assert envelope is not None
    prepare = envelope["prepare"]
    assert prepare["source_exactness"] is True
    assert prepare["match"]["source_image"] is None
    assert prepare["source_trace"]["original_runtime_manifest"]["preflight_image"] == R31_IMAGE_ID


def test_raw_host_summary_and_exact_r31_kernel_rows():
    api_path = Path(API_STATISTIC_PATH)
    raw_path = Path(SOURCE_RAW_PATH)
    if not api_path.exists() or not raw_path.exists():
        pytest.skip("retained raw host summary unavailable")
    import hashlib
    assert hashlib.sha256(api_path.read_bytes()).hexdigest() == (
        "e19308dea200ead700b417475da5983576ec8884646e3b03e833be6b06e0808e")
    with api_path.open(newline="") as stream:
        api_rows = list(csv.DictReader(stream))
    counts = {row["API Name"]: int(row["Count"]) for row in api_rows
              if row["API Name"] in {
                  "aclnnRmsNorm", "aclnnRmsNormGetWorkspaceSize",
                  "aclnnRmsNormTiling", "aclnnAddRmsNorm"}}
    assert counts == {
        "aclnnRmsNorm": 1,
        "aclnnRmsNormGetWorkspaceSize": 1,
        "aclnnRmsNormTiling": 1,
        "aclnnAddRmsNorm": 128,
    }
    with gzip.open(raw_path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    shift = rows[RAW_GAMMA_SHIFT_ROW]
    assert shift["Name"] == "aclnnAdds_AddAiCore_Add"
    assert shift["Type"] == "Add"
    assert int(shift["Block Num"]) == 10
    assert shift["Input Shapes"] == '"5120;"'
    assert shift["Output Shapes"] == '"5120"'
    norm = rows[RAW_RMS_NORM_ROW]
    assert norm["Name"] == "RmsNorm"
    assert norm["Type"] == "RmsNorm"
    assert int(norm["Block Num"]) == 64
    assert norm["Input Shapes"] == '"2048,5120;5120"'
    assert norm["Output Shapes"] == '"2048,5120;2048,1"'
    assert norm["Output Data Types"] == "DT_BF16;FLOAT"
