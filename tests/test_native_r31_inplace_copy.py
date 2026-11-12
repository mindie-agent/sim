"""Focused R31 A3 ``aclnnInplaceCopy`` z-materialization adapter checks.

[Role] Prove the real R31 Qwen3.6 A3 ``z_slice`` row selects one native
  ``aclnnInplaceCopy`` call identity from the actual R31 selected
  ``gdn.py`` blob ``5e3b0fb0...`` at lines 396-397/443/444, prepares the
  exact packed-view ``[T,64,128] -> [T,24,128]`` typed occurrence and keeps
  the missing Python/ATen/ACLNN-header/kernel/backend evidence explicit.

[Invariants] The R31 deployment image and actual R31 selected source blob are
  the only source selectors; the 64-source-head / 24-kept-head / 128-wide
  view, packed extent 8192, offset 5120 and packed-view stride are static;
  only T changes.  The non-contiguous copy is owned once by
  ``A3:copy:z_slice``.  Other images (including the different c3d00a
  ``e9ed210b`` blob), phases, copy rows and decode geometries do not
  validate.

[Timeouts & Locks] Pure bounded metadata arithmetic; no device or locks.

[Side-Effect Policy] Reads the retained task R31 source archive for its
  member hash/line check only; no product state, tensor, allocation or
  device launch.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tarfile

import pytest

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r31_inplace_copy import (
    ACLNN_API, CANN_VERSION, COPY_OWNER, COPY_ROLE, KERNEL_NAME, PROFILER_NAME,
    R31_IMAGE_SHA, R31_SOURCE_BLOB_SHA, SOURCE, SOURCE_ARCHIVE_PATH,
    SOURCE_ARCHIVE_SHA256, SOURCE_LINES, SOURCE_MANIFEST_SHA256,
    SOURCE_MEMBER_PATH, binding_spec, dynamic_call)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_model)

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"
SCRATCH = Path(os.environ.get(
    "MINDIE_TASK_SCRATCH", "/private/tmp/mindie-7337-20261111-1457"))
SOURCE_ARCHIVE = SCRATCH / SOURCE_ARCHIVE_PATH
R31 = (("image_id", "sha256:" + R31_IMAGE_SHA),)
C3D00_ONLY = ((
    "source_image",
    "quay.io/ascend/vllm-ascend:v0.30.0rc2-a5 (c3d00a196905)"),)
R64 = (("image_id",
        "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"),)
UNKNOWN = (("image_id", "sha256:" + "f" * 64),)
DOMAIN = AnalysisDomain(max_tokens=10 ** 9, max_moe_rows=10 ** 9,
                        max_query_tiles=10 ** 9, max_kv_tiles=10 ** 9,
                        max_index_rows=10 ** 12, max_work=10 ** 15,
                        max_q=10 ** 9, max_s_kv=10 ** 9, max_seq=10 ** 9,
                        max_batch=10 ** 6)


def _case():
    case = parse_case(json.loads(QWEN.read_text()))
    features = FeatureConfig.from_model(
        case.model, tp=2, dp=4, ep=1, pp=1, cores=64, token_tile=16,
        kv_tile=16, moe_row_tile=4, split_target=1000, max_splits=4)
    return case, instantiate_model(case.model, features)


def _compiler(case):
    return OperatorPiecewiseCompiler(
        PipeSupply.from_hardware(case.machine, dtype="bf16"),
        ImplementationPolicy(cores=64), domain=DOMAIN)


def _subgraph(inst, kind, name):
    return next(sub for layer in inst.layers for sub in layer.subgraphs
                if sub.kind == kind
                and any(operator.name == name for operator in sub.operators))


def _geometry(tokens, phase="prefill"):
    return WorkloadGeometry(
        (RequestGeometry(tokens, tokens, phase=phase),),
        phase=phase, cohort_batch=1, local_batch=1)


def _binding(compiled, spec):
    surface = _surface(compiled, spec, source_boundary="kernel")
    return bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved))


def test_r31_z_slice_keeps_static_view_and_dynamic_tokens():
    case, inst = _case()
    compiler = _compiler(case)
    sub = _subgraph(inst, "A3", "z_slice")
    compiled = compiler.compile(sub, feature_metadata=R31)
    operator = compiled.operator("z_slice")
    spec = binding_spec(operator, phase="prefill")
    assert spec is not None
    assert spec.api["namespace"] == "aclnn"
    assert spec.api["entry"] == ACLNN_API
    assert spec.api["aclnn_symbol"] == ACLNN_API
    assert spec.api["version"] == CANN_VERSION
    assert spec.source["path"] == "vllm_ascend/ops/gdn.py"
    assert spec.source["sha256"] == R31_SOURCE_BLOB_SHA
    assert spec.source["sha256"].startswith("5e3b0fb0")
    assert spec.source["symbol"] == SOURCE["symbol"]
    assert spec.source["revision"] == "r31/9e869e814846"
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["evidence"] == "measured"
    assert spec.kernels[0]["acl_api"] == ACLNN_API
    assert spec.execution_core == "aiv"
    for reason in ("python_aten_dispatch_unresolved",
                   "acl_api_header_not_retained",
                   "slice_bounds_values_not_retained",
                   "kernel_source_runtime_binary_only",
                   "acl_binary_selection_trace_only",
                   "chosen_backend_unresolved"):
        assert reason in spec.unresolved
    assert spec.owner == "z_slice"
    attrs = spec.typed["attrs"]
    assert attrs["copy_role"] == COPY_ROLE
    assert attrs["copy_owner"] == COPY_OWNER
    assert attrs["copy_kind"] == "noncontiguous_view_materialization"
    assert attrs["source_view_noncontiguous"] is True
    assert attrs["source_view_layout"] == "packed_qkvz_strided"
    assert attrs["source_stride_elements"] == [8192, 128, 1]
    assert attrs["output_stride_elements"] == [3072, 128, 1]
    assert attrs["source_heads"] == 64
    assert attrs["kept_heads"] == 24
    assert attrs["head_dim"] == 128
    assert attrs["source_extent"] == 8192
    assert attrs["offset_elements"] == 5120
    assert attrs["source_offset_heads"] == 40
    assert attrs["source_lines"] == SOURCE_LINES
    assert attrs["source_blob_sha256"] == R31_SOURCE_BLOB_SHA
    assert spec.typed["shapes"] == {
        "source": [64, 128], "output": [24, 128]}

    binding = _binding(operator, spec)
    assert binding.op_id == "A3:copy:z_slice"
    assert binding.timing_owner == {
        "op_id": "A3:copy:z_slice", "scope": "call"}
    first = dynamic_call(binding, operator, _geometry(4096),
                         validate=validate_native_occurrence)
    assert first is not None
    actual = first["actual"]
    assert actual["dtype"] == "bf16"
    assert actual["layout"] == "ND"
    assert actual["shapes"] == {
        "source": [4096, 64, 128], "output": [4096, 24, 128]}
    assert actual["attrs"]["tokens"] == 4096
    assert actual["attrs"]["source_view_offset_elements"] == 5120
    assert actual["attrs"]["slice_offset_heads"] == 40
    assert actual["attrs"]["slice_kept_heads"] == 24
    assert actual["attrs"]["source_view_noncontiguous"] is True
    assert actual["attrs"]["copy_owner"] == COPY_OWNER
    prepare = first["prepare"]
    assert prepare["call"] == ACLNN_API
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["source_blob_sha256"] == R31_SOURCE_BLOB_SHA
    assert prepare["shapes"] == {
        "source": [4096, 64, 128], "output": [4096, 24, 128]}
    assert prepare["descriptors"]["source"]["view_stride_elements"] == [
        8192, 128, 1]
    assert prepare["descriptors"]["source"]["noncontiguous"] is True
    assert prepare["descriptors"]["source"]["copy_owner"] == COPY_OWNER
    assert prepare["descriptors"]["output"]["view_stride_elements"] == [
        3072, 128, 1]
    assert prepare["descriptors"]["output"]["contiguous"] is True
    assert prepare["slice_view"]["source_view_noncontiguous"] is True
    assert prepare["slice_view"]["source_offset_heads"] == 40
    assert prepare["slice_view"]["copy_owner"] == COPY_OWNER
    assert prepare["chosen_backend"]["status"] == "unresolved"
    assert prepare["observed"]["sample_tokens"] == 4096
    assert prepare["observed"]["raw_stream_sha256"] == (
        "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a")
    assert prepare["observed"]["csv_gzip_sha256"] == (
        "a523936e982e4fd45b80071e906276d17d18fac03c5bfaca903e2f1158f778d1")
    assert prepare["observed"]["input_shapes"] == [
        [4096, 64, 128], [3], [3]]
    assert prepare["observed"]["output_shapes"] == [[4096, 24, 128]]
    assert prepare["observed"]["sequence_counts"] == {
        "immediate_previous_tensor_move": 47,
        "immediate_next_tensor_move": 47,
        "layer_norm_within_next_3": 48,
    }
    assert prepare["binary_selection"]["status"] == "uncertain"
    assert prepare["binary_selection"]["chosen_backend_status"] == "unresolved"
    assert prepare["deployment"]["source_sha256"] == R31_SOURCE_BLOB_SHA
    assert prepare["deployment"]["source_revision"] == "r31/9e869e814846"
    assert prepare["deployment"]["source_archive_sha256"] == (
        SOURCE_ARCHIVE_SHA256)
    assert prepare["deployment"]["source_manifest_sha256"] == (
        SOURCE_MANIFEST_SHA256)
    assert prepare["match"]["source_sha256"] == R31_SOURCE_BLOB_SHA
    assert prepare["match"]["source_view_noncontiguous"] is True
    assert prepare["match"]["copy_owner"] == COPY_OWNER

    held = dynamic_call(binding, operator, _geometry(2048),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"] == {
        "source": [2048, 64, 128], "output": [2048, 24, 128]}
    assert held["prepare"]["observed"]["sample_matches_current_geometry"] is False
    # A new T is a new occurrence, not a new selected call identity.
    again = _binding(operator, binding_spec(operator, phase="prefill"))
    assert again.call_id == binding.call_id


def test_r31_z_slice_rejects_other_images_phases_and_rows():
    case, inst = _case()
    compiler = _compiler(case)
    sub = _subgraph(inst, "A3", "z_slice")
    compiled = compiler.compile(sub, feature_metadata=R31)
    operator = compiled.operator("z_slice")
    assert binding_spec(operator, phase="decode") is None
    assert binding_spec(operator, phase="") is None
    assert binding_spec(compiled.operator("beta_slice"), phase="prefill") is None
    assert binding_spec(compiled.operator("core_zeros"), phase="prefill") is None
    for metadata in (C3D00_ONLY, R64, UNKNOWN):
        other = compiler.compile(sub, feature_metadata=metadata)
        assert binding_spec(other.operator("z_slice"), phase="prefill") is None
    spec = binding_spec(operator, phase="prefill")
    binding = _binding(operator, spec)
    assert dynamic_call(binding, operator, _geometry(4096, phase="decode"),
                        validate=validate_native_occurrence) is None
    assert dynamic_call(binding, operator, _geometry(0),
                        validate=validate_native_occurrence) is None


@pytest.mark.skipif(not SOURCE_ARCHIVE.is_file(),
                    reason="R31 selected source archive is not retained")
def test_r31_z_slice_source_member_line_and_hash_are_real():
    assert sha256(SOURCE_ARCHIVE.read_bytes()).hexdigest() == (
        SOURCE_ARCHIVE_SHA256)
    with tarfile.open(SOURCE_ARCHIVE) as archive:
        member = next(
            item for item in archive.getmembers()
            if item.name.endswith(SOURCE_MEMBER_PATH))
        raw = archive.extractfile(member).read()
    assert sha256(raw).hexdigest() == R31_SOURCE_BLOB_SHA
    lines = raw.decode("utf-8").splitlines()
    assert lines[SOURCE_LINES["producer_call"][0] - 1].strip().startswith(
        "mixed_qkvz, _ = self.in_proj_qkvz")
    assert lines[SOURCE_LINES["split_view"][0] - 1].strip().startswith(
        "mixed_qkv, z = mixed_qkvz.split")
    assert lines[SOURCE_LINES["materialize"][0] - 1].strip() == (
        "z = z.reshape(-1, z.shape[-1])")
    assert lines[SOURCE_LINES["consumer"][0] - 1].strip() == (
        "core_attn_out = self.norm(core_attn_out, z)")
    assert R31_SOURCE_BLOB_SHA.startswith("5e3b0fb0")
    assert PROFILER_NAME == "aclnnInplaceCopy_SliceAiCore_Slice"
