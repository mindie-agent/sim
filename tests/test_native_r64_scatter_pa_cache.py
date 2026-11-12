"""Source-bound R64 ScatterPaCache indexer cache-write adapter invariants.

[Role] Exercise the adapter against the existing ``I0:kv_access:
  index_cache_store`` compiled operator and check the real R64 fused
  ``indexer_compress_epilog_v2`` / two-kernel ``ScatterPaCache`` occurrence.

[Invariants] The selected source branch is the single-stream
  ``A5DeviceAdaptor.indexer_quant_scatter`` call; the multi-stream part1
  branch, a non-R64 image, a BF16/absent index cache and the separate
  ``I1:indexer:index_score`` read stay unbound.  The token lane is an
  occurrence field; the captured cache capacity is evidence only and is never
  fabricated as a predictive dimension.  The envelope stays prepare-only and
  explicitly incomplete where the fused C++ source, ACLNN symbol, packed row
  bytes and cache capacity are not retained.

[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local construction only; no product state mutation.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    OperatorPiecewiseCompiler, PipeSupply)
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r64_scatter_pa_cache import (
    CAPTURE_CASE, ENTRY, KERNEL_NAME, R64_IMAGE_SHA, SOURCE, SOURCE_LINES,
    binding_spec, dynamic_call)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.builders_attention import _build_i0
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.schema.subgraphs import BoundOperator

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/glm53-a5.json"
R35_IMAGE_SHA = (
    "5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")


def _compiled(*, image_id=R64_IMAGE_SHA, indexer_kv_dtype="fp8_e4m3",
              multistream=False, kpool=1):
    case = parse_case(json.loads(CASE.read_text(encoding="utf-8")))
    layer = next(plan for plan in case.model.layer_plans if plan.indexer)
    layer = replace(layer, index_kpool=kpool, index_dim=128, index_heads=32,
                    owns_index_k=True, index_input="hidden")
    features = FeatureConfig.from_runtime(
        case.runtime, case.model, machine=case.machine)
    metadata = dict(features.metadata)
    metadata.update({
        "image_id": image_id,
        "deployment_image_id": image_id,
        "oci_image_digest": image_id,
        "indexer_kv_dtype": indexer_kv_dtype,
        "indexer_query_scale_present": True,
        "indexer_query_scale_dtype": "fp32",
        "mx_quant_source_revision": "r64",
        "kv_block_size": 128,
    })
    if multistream:
        metadata["multistream_attention_indexer"] = True
    features = replace(
        features,
        index_dtype=("fp8_e4m3" if indexer_kv_dtype == "fp8_e4m3"
                     else "bf16"),
        metadata=tuple(sorted(metadata.items())))
    ctx = _Ctx(case.model, layer, features, stage=0, layers=case.model.layers)
    built = _build_i0(ctx)
    template = next(operator for operator in built.template.operators
                    if operator.name == "index_cache_store")
    operator = BoundOperator(template, built.signature("index_cache_store"))
    return OperatorPiecewiseCompiler(
        PipeSupply.from_hardware(case.machine, dtype="int8")
    ).compile_operator(operator, kind="I0",
                       feature_metadata=features.metadata)


def _binding(compiled):
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    surface = _surface(compiled, spec, source_boundary="kernel")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return spec, bind_native_call(declared)


def _geometry(tokens, phase="prefill"):
    return WorkloadGeometry((RequestGeometry(tokens, 0, phase=phase),),
                            phase=phase)


def test_r64_index_cache_store_prepare_only():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    assert spec.api["entry"] == ENTRY
    assert spec.api["aclnn_symbol"] is None
    assert spec.source["sha256"] == SOURCE["sha256"]
    assert spec.source["symbol"] == "A5DeviceAdaptor.indexer_quant_scatter"
    assert spec.owner == "I0:kv_access:index_cache_store"
    assert binding.timing_owner == {
        "op_id": "I0:kv_access:index_cache_store", "scope": "call"}
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["evidence"] == "measured"
    assert spec.kernels[0]["kernel_source_status"] == (
        "kernel_source_not_retained")
    assert spec.typed["attrs"]["selected_branch"] == (
        "single_stream_indexer_quant_scatter")
    assert spec.typed["attrs"]["source_lines"] == list(SOURCE_LINES)
    assert "exact_native_entry_unresolved" not in binding.unresolved
    assert "attributable_source_reference_unresolved" not in binding.unresolved
    assert "scatter_pa_cache_kernel_source_not_retained" in binding.unresolved
    assert binding.typed["signature_hash"]

    first = dynamic_call(binding, compiled, _geometry(1024),
                         validate=validate_native_occurrence)
    assert first is not None
    assert first["actual"]["shapes"] == {
        "update": [1024, 1, 128],
        "cache": [None, 1, 1, 128],
        "scale_update": [1024, 1, 1],
        "scale_cache": [None, 1, 1, 1],
        "output": [None, 1, 1, 128],
        "scale_output": [None, 1, 1, 1],
    }
    prepare = first["prepare"]
    assert prepare["call"] == ENTRY
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["selected_branch"] == (
        "single_stream_indexer_quant_scatter")
    assert prepare["rejected_branch"] == (
        "multistream_indexer_quant_scatter_part1")
    assert prepare["source_lines"] == list(SOURCE_LINES)
    assert prepare["call_lines"] == [1547, 1552]
    assert prepare["wrapper_arguments"]["layout"] == 2
    assert prepare["kernel_arguments"] == {
        "update": "update", "cache": "cache", "indices": "slot_mapping"}
    k_cache = prepare["kernel_occurrences"]["indexer_k_cache"]
    scale_cache = prepare["kernel_occurrences"]["indexer_scale_cache"]
    assert k_cache["input_dtypes"] == ["fp8_e4m3", "fp8_e4m3", "int32"]
    assert k_cache["input_shapes"] == {
        "update": [1024, 1, 128], "cache": [None, 1, 1, 128],
        "indices": [1024]}
    assert scale_cache["input_dtypes"] == ["fp32", "fp32", "int32"]
    assert scale_cache["input_shapes"] == {
        "update": [1024, 1, 1], "cache": [None, 1, 1, 1],
        "indices": [1024]}
    assert prepare["captured_example"]["case"] == CAPTURE_CASE
    assert prepare["captured_example"]["captured_cache_rows"] == 522240
    assert prepare["captured_example"]["note"].endswith(
        "not_a_predictive_dimension")
    assert prepare["mapping"]["status"] == "captured_not_source_linked"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert "indexer_cache_capacity_not_declared" in prepare["partial_reasons"]


def test_r64_index_cache_store_rejects_other_sources_and_branches():
    compiled = _compiled()
    assert binding_spec(compiled, phase="decode") is None
    assert binding_spec(_compiled(image_id=R35_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(_compiled(indexer_kv_dtype="bf16"),
                        phase="prefill") is None
    assert binding_spec(_compiled(multistream=True),
                        phase="prefill") is None
    assert binding_spec(_compiled(kpool=4), phase="prefill") is None
    _, binding = _binding(compiled)
    assert dynamic_call(binding, compiled, _geometry(1024, phase="decode"),
                        validate=validate_native_occurrence) is None
    assert dynamic_call(binding, compiled, _geometry(0),
                        validate=validate_native_occurrence) is None
    # The separate I1 read is not this call.
    assert binding.api["entry"] != "torch_npu.npu_quant_lightning_indexer"


def test_r64_index_cache_store_token_lane_is_occurrence_only():
    compiled = _compiled()
    _, binding = _binding(compiled)
    small = dynamic_call(binding, compiled, _geometry(1024),
                         validate=validate_native_occurrence)
    large = dynamic_call(binding, compiled, _geometry(2048),
                         validate=validate_native_occurrence)
    assert small is not None and large is not None
    assert small["prepare"]["call_id"] == large["prepare"]["call_id"]
    assert large["actual"]["shapes"]["update"] == [2048, 1, 128]
    assert large["actual"]["shapes"]["scale_update"] == [2048, 1, 1]
    # Capacity is not declared by the normal case; the captured 522240 rows
    # never become a predictive shape.
    assert large["actual"]["shapes"]["cache"] == [None, 1, 1, 128]
    assert large["actual"]["shapes"]["scale_cache"] == [None, 1, 1, 1]
    assert large["prepare"]["captured_example"]["captured_cache_rows"] == (
        522240)
    assert binding.call_id == _binding(_compiled())[1].call_id
