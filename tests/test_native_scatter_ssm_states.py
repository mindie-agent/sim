"""Source-bound R31 recurrent-state scatter adapter invariants.

[Role] Exercise the operator adapter against the real R31 GDN caller and
  ``state_index.py`` kernel source, the true R31 original integration source,
  and the existing compiled A3 GDN lanes.

[Invariants] The selected call is ``scatter_ssm_states_`` /
  ``_scatter_ssm_states_kernel`` with the exact source argument order
  ``(state, indices, source, stride_state_batch, stride_indices,
  stride_source_batch, row_size, BLOCK_SIZE)``, block size 1024, eight warps
  and grid ``(cdiv(row_size,1024), batch)``.  Both genuine caller branches
  are admitted as separate selected variants: ``fla_gdn_prefill`` at
  ``gdn.py:806-810`` (source cast to state dtype, caller-contiguous) and
  ``use_fused_chunk`` at ``gdn.py:833-837`` (direct chunk final-state operand,
  dtype unresolved unless declared).  A missing cache capacity, stride, state
  layout or branch stays unbound/unresolved; a descriptor default is never
  invented.  The scatter is a state write distinct from
  ``_gather_ssm_states_kernel`` and from the signed chunk child scopes.

[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local construction only; no product state mutation.
"""
from __future__ import annotations

from types import SimpleNamespace

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply, RequestGeometry,
    WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_scatter_ssm_states import (
    ENTRY, KERNEL_LINES, KERNEL_NAME, QWEN36_IMAGE_SHA, R31_IMAGE_SHA,
    SOURCE, STATE_INDEX_SHA, WRAPPER_LINES, WRAPPER_SYMBOL,
    binding_spec, dynamic_call)
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)

SUPPLY = PipeSupply((
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
    ("link_hccs", 2000.0), ("link_roce", 500.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=1)
R35_IMAGE_SHA = (
    "5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
ROW_SIZE = 24 * 128 * 128
LANES = (("key_heads", 8), ("value_heads", 24), ("dk", 128), ("dv", 128))
BASE_EXTRAS = LANES + (
    ("state_layout", "V,K"), ("state_dtype", "fp32"),
    ("index_dtype", "int32"), ("native_kind", "state_scatter"),
    ("source_kernel", "_scatter_ssm_states_kernel"),
)
FLA_EXTRAS = BASE_EXTRAS + (("scatter_branch", "fla_gdn_prefill"),)
FUSED_EXTRAS = BASE_EXTRAS + (("scatter_branch", "use_fused_chunk"),)


def _gated_delta():
    signature = StaticSignature(
        role="delta", matrix=(("rows", 1), ("row_bytes", ROW_SIZE * 4)),
        extras=LANES + (("chunk", 64),))
    template = OperatorTemplate(name="gated_delta", family="recurrent",
                                role="delta")
    return OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        BoundOperator(template, signature), kind="A3")


def _compiled(*, image_id=None, kind="A3", name="scatter_recurrent_state",
              family="state_scatter", extras=FLA_EXTRAS, siblings=None):
    signature = StaticSignature(
        role="scatter",
        matrix=(("rows", 1), ("row_bytes", ROW_SIZE * 4)),
        extras=extras)
    template = OperatorTemplate(name=name, family=family, role="scatter")
    operator = BoundOperator(template, signature)
    metadata = (("image_id", "sha256:" + image_id),) if image_id else ()
    return SimpleNamespace(
        kind=kind, operator=operator,
        siblings=(_gated_delta(),) if siblings is None else siblings,
        feature_metadata=metadata)


def _binding(compiled):
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    surface = _surface(compiled, spec, source_boundary="kernel")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return spec, bind_native_call(declared)


def _geometry(lengths=(4096,), phase="prefill"):
    return WorkloadGeometry(
        tuple(RequestGeometry(length, 0, phase=phase) for length in lengths),
        phase=phase)


def test_r31_fla_state_scatter_source_and_prepare_only_occurrence():
    compiled = _compiled(image_id=R31_IMAGE_SHA)
    spec, binding = _binding(compiled)
    assert spec.api["namespace"] == "triton"
    assert spec.api["entry"] == ENTRY
    assert spec.source["path"] == "vllm_ascend/ops/triton/mamba/state_index.py"
    assert spec.source["sha256"] == STATE_INDEX_SHA
    assert spec.source["symbol"] == WRAPPER_SYMBOL
    assert spec.source["revision"] == SOURCE["revision"]
    assert spec.owner == "scatter_recurrent_state"
    assert binding.timing_owner == {
        "op_id": "A3:state_scatter:scatter_recurrent_state", "scope": "call"}
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["evidence"] == "source_declared"
    assert spec.kernels[0]["lines"] == list(KERNEL_LINES)
    assert spec.kernels[0]["block_size"] == 1024
    assert spec.kernels[0]["num_warps"] == 8
    attrs = spec.typed["attrs"]
    assert attrs["scatter_branch"] == "fla_gdn_prefill"
    assert attrs["source_lines"] == [806, 810]
    assert attrs["alternate_source_lines"] == [833, 837]
    assert attrs["source_dtype"] == "fp32"
    assert attrs["source_dtype_status"] == "state_dtype_from_source_cast"
    assert attrs["source_contiguity"] == "caller_contiguous"
    assert attrs["source_stride0"] is None
    assert attrs["source_stride0_status"] == "caller_contiguous_unobserved"
    assert attrs["wrapper_lines"] == list(WRAPPER_LINES)
    assert attrs["gather_argument_present"] is False
    assert attrs["gather_distinct_from_gather"] is True
    assert attrs["row_size"] == ROW_SIZE
    assert attrs["grid_blocks"] == ROW_SIZE // 1024
    assert "source_dtype_unresolved" not in spec.unresolved

    result = dynamic_call(binding, compiled, _geometry(),
                          validate=validate_native_occurrence)
    assert result is not None
    assert result["actual"]["dtype"] == "fp32"
    assert result["actual"]["shapes"] == {
        "indices": [1], "source": [1, 24, 128, 128],
        "state": [None, 24, 128, 128]}
    prepare = result["prepare"]
    assert prepare["arguments"] == {
        "state": "state", "indices": "indices", "source": "source"}
    assert prepare["argument_order"] == ["state", "indices", "source"]
    assert prepare["kernel_argument_order"] == [
        "state_ptr", "indices_ptr", "source_ptr", "stride_state_batch",
        "stride_indices", "stride_source_batch", "row_size", "BLOCK_SIZE"]
    assert prepare["kernel_arguments"] == {
        "state_ptr": "state", "indices_ptr": "indices",
        "source_ptr": "source", "stride_state_batch": None,
        "stride_indices": None, "stride_source_batch": None,
        "row_size": ROW_SIZE, "BLOCK_SIZE": 1024}
    assert prepare["launch_options"] == {"num_warps": 8}
    assert prepare["grid"] == [ROW_SIZE // 1024, 1]
    assert prepare["state_cache"]["rows"] is None
    assert prepare["state_cache"]["row_size"] == ROW_SIZE
    assert prepare["state_cache"]["status"] == "unresolved"
    assert prepare["state_cache"]["stride0"] is None
    assert prepare["source_operand"] == {
        "expression": "last_recurrent_state.to(ssm_state.dtype).contiguous()",
        "lines": [806, 810], "dtype": "fp32",
        "dtype_status": "state_dtype_from_source_cast",
        "contiguity": "caller_contiguous", "stride0": None,
        "stride0_status": "caller_contiguous_unobserved",
        "cast_to_state_dtype": True,
    }
    assert prepare["descriptors"]["state"]["shape"] == [
        None, 24, 128, 128]
    assert prepare["descriptors"]["state"]["state_layout"] == "V,K"
    assert prepare["descriptors"]["indices"]["dtype"] == "int32"
    assert prepare["descriptors"]["source"]["dtype"] == "fp32"
    assert prepare["descriptors"]["source"]["shape"] == [
        1, 24, 128, 128]
    assert "source = last_recurrent_state.to(ssm_state.dtype).contiguous()" in (
        prepare["source_validation"])
    capture = prepare["observed_capture"]
    assert capture["path"].endswith("integration-r31-a3-input-source.json")
    assert capture["geometry"]["value_heads"] == 24
    assert capture["geometry"]["state_rows"] == 2375
    assert capture["scatter_example"]["input_shapes"] == [
        [1], [1, 24, 128, 128]]
    assert capture["scatter_example"]["output_shapes"] == [
        [2375, 24, 128, 128]]
    assert capture["capture_matches_selected_branch"] is True
    assert capture["predictive_input"] is False
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert "state_cache_rows_unresolved" in prepare["unresolved"]
    assert "state_stride0_unresolved" in prepare["unresolved"]
    assert "source_stride0_unresolved" in prepare["unresolved"]
    assert "source_dtype_unresolved" not in prepare["unresolved"]
    assert "no_guessed_state_layout" in prepare["non_claims"]


def test_state_scatter_is_distinct_from_gather_and_chunk():
    compiled = _compiled(image_id=R31_IMAGE_SHA)
    spec, binding = _binding(compiled)
    assert spec.owner != "gather_recurrent_state"
    assert spec.owner != "gated_delta"
    assert spec.members == ()
    assert spec.kernels[0]["name"] == "_scatter_ssm_states_kernel"
    assert "gather" not in spec.kernels[0]["name"]
    assert "has_initial_state" not in spec.typed["shapes"]
    assert "has_initial_state" not in spec.typed["attrs"]
    assert spec.typed["attrs"]["native_kind"] == "state_scatter"
    assert spec.typed["attrs"]["state_layout"] == "V,K"
    result = dynamic_call(binding, compiled, _geometry((2048, 2048)),
                          validate=validate_native_occurrence)
    assert result is not None
    assert result["actual"]["shapes"]["indices"] == [2]
    assert result["actual"]["shapes"]["source"] == [2, 24, 128, 128]
    assert result["prepare"]["grid"] == [ROW_SIZE // 1024, 2]
    assert result["prepare"]["state_cache"]["mutated_rows"] == 2
    assert result["prepare"]["match"]["branch"] == "fla_gdn_prefill"


def test_state_scatter_explicit_rows_and_strides_are_carried():
    explicit = FLA_EXTRAS + (
        ("state_rows", 9216), ("state_stride0", 400000),
        ("indices_stride0", 2), ("source_stride0", ROW_SIZE))
    compiled = _compiled(image_id=R31_IMAGE_SHA, extras=explicit)
    _, binding = _binding(compiled)
    result = dynamic_call(binding, compiled, _geometry(),
                          validate=validate_native_occurrence)
    assert result is not None
    assert result["actual"]["shapes"]["state"] == [9216, 24, 128, 128]
    assert result["actual"]["attrs"]["state_rows"] == 9216
    assert result["actual"]["attrs"]["state_rows_status"] == "explicit"
    cache = result["prepare"]["state_cache"]
    assert cache["shape"] == [9216, 24, 128, 128]
    assert cache["rows"] == 9216
    assert cache["stride0"] == 400000
    assert cache["status"] == "explicit"
    arguments = result["prepare"]["kernel_arguments"]
    assert arguments["stride_state_batch"] == 400000
    assert arguments["stride_indices"] == 2
    assert arguments["stride_source_batch"] == ROW_SIZE
    assert result["prepare"]["source_operand"]["stride0_status"] == "explicit"
    assert result["prepare"]["partial_reasons"] == [
        "runtime_execution_not_performed"]


def test_state_scatter_fused_branch_describes_direct_operand():
    compiled = _compiled(image_id=R31_IMAGE_SHA, extras=FUSED_EXTRAS)
    spec, binding = _binding(compiled)
    attrs = spec.typed["attrs"]
    assert attrs["scatter_branch"] == "use_fused_chunk"
    assert attrs["source_lines"] == [833, 837]
    assert attrs["alternate_source_lines"] == [806, 810]
    assert attrs["source_dtype"] is None
    assert attrs["source_dtype_status"] == "unresolved_chunk_final_state_dtype"
    assert attrs["source_contiguity"] == "unretained_branch_operand"
    assert attrs["source_stride0_status"] == "unretained_branch_operand"
    assert "source_dtype_unresolved" in spec.unresolved
    result = dynamic_call(binding, compiled, _geometry(),
                          validate=validate_native_occurrence)
    assert result is not None
    assert result["prepare"]["source_operand"]["expression"] == (
        "last_recurrent_state")
    assert result["prepare"]["source_operand"]["dtype"] is None
    assert result["prepare"]["descriptors"]["source"]["dtype"] is None
    assert result["prepare"]["observed_capture"][
        "capture_matches_selected_branch"] is False
    assert "source = last_recurrent_state" in (
        result["prepare"]["source_validation"])
    assert "source_dtype_unresolved" in result["prepare"]["unresolved"]


def test_state_scatter_fused_branch_can_declare_operand_dtype():
    explicit = FUSED_EXTRAS + (("source_dtype", "bf16"),)
    compiled = _compiled(image_id=R31_IMAGE_SHA, extras=explicit)
    spec, binding = _binding(compiled)
    assert spec.typed["attrs"]["source_dtype"] == "bf16"
    assert spec.typed["attrs"]["source_dtype_status"] == "explicit"
    assert "source_dtype_unresolved" not in spec.unresolved
    result = dynamic_call(binding, compiled, _geometry(),
                          validate=validate_native_occurrence)
    assert result is not None
    assert result["actual"]["dtype"] == "fp32"
    assert result["prepare"]["descriptors"]["source"]["dtype"] == "bf16"
    assert result["prepare"]["source_operand"]["dtype"] == "bf16"
    assert result["prepare"]["source_operand"]["cast_to_state_dtype"] is False


def test_state_scatter_rejects_other_sources_phases_and_families():
    assert binding_spec(_compiled(image_id=R35_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(_compiled(image_id=None), phase="prefill") is None
    assert binding_spec(_compiled(image_id=QWEN36_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA),
                        phase="decode") is None
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA),
                        phase="") is None
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA, kind="F1"),
                        phase="prefill") is None
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  name="gather_recurrent_state",
                                  family="state_gather"),
                        phase="prefill") is None
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  name="tp_reduce_scatter",
                                  family="collective"),
                        phase="prefill") is None
    recurrent_unmarked = tuple(
        item for item in FLA_EXTRAS
        if item[0] not in {"native_kind", "source_kernel"})
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  family="recurrent",
                                  extras=recurrent_unmarked),
                        phase="prefill") is None
    missing_state_dtype = tuple(
        item for item in FLA_EXTRAS if item[0] != "state_dtype")
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  extras=missing_state_dtype),
                        phase="prefill") is None
    missing_layout = tuple(
        item for item in FLA_EXTRAS if item[0] != "state_layout")
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  extras=missing_layout),
                        phase="prefill") is None
    missing_branch = tuple(
        item for item in FLA_EXTRAS if item[0] != "scatter_branch")
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  extras=missing_branch),
                        phase="prefill") is None
    invalid_branch = FLA_EXTRAS[:-1] + (("scatter_branch", "decode"),)
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  extras=invalid_branch),
                        phase="prefill") is None
    missing_lanes = (("state_layout", "V,K"), ("state_dtype", "fp32"),
                     ("index_dtype", "int32"),
                     ("native_kind", "state_scatter"),
                     ("scatter_branch", "fla_gdn_prefill"))
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  extras=missing_lanes, siblings=()),
                        phase="prefill") is None
    bad_fla_stride = FLA_EXTRAS + (("source_stride0", ROW_SIZE + 1),)
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  extras=bad_fla_stride),
                        phase="prefill") is None
    bad_fused_stride = FUSED_EXTRAS + (("source_stride0", ROW_SIZE - 1),)
    assert binding_spec(_compiled(image_id=R31_IMAGE_SHA,
                                  extras=bad_fused_stride),
                        phase="prefill") is None
    _, binding = _binding(_compiled(image_id=R31_IMAGE_SHA))
    assert dynamic_call(binding, _compiled(image_id=R31_IMAGE_SHA),
                        _geometry(phase="decode"),
                        validate=validate_native_occurrence) is None
    empty = SimpleNamespace(phase_name="prefill", requests=())
    assert dynamic_call(binding, _compiled(image_id=R31_IMAGE_SHA), empty,
                        validate=validate_native_occurrence) is None
    assert dynamic_call(binding, _compiled(image_id=R31_IMAGE_SHA),
                        _geometry((0,)),
                        validate=validate_native_occurrence) is None


def test_original_qwen36_caller_is_independent_evidence_not_an_admission():
    compiled = _compiled(image_id=R31_IMAGE_SHA)
    spec, _ = _binding(compiled)
    assert spec.typed["attrs"]["independent_caller_sha256"] == (
        "e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98")
    assert spec.typed["attrs"]["independent_caller_lines"] == [677, 681]
    assert spec.typed["attrs"]["deployment_image_id"] == R31_IMAGE_SHA
    assert spec.typed["attrs"]["source_lines"] == [806, 810]
    assert binding_spec(_compiled(image_id=QWEN36_IMAGE_SHA),
                        phase="prefill") is None
