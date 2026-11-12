"""Focused R31 O0 selected-hidden ``aclnnIndex`` adapter contract tests.

The adapter is metadata-only: these tests exercise the real shared
``native_calls._CallSpec``/``NativeCallBinding`` path, the declared source
runner policy, the producing+selected geometry expansion, and the negative
runner/row gates.  They do not allocate tensors, import torch or launch a
kernel.
"""
from types import SimpleNamespace

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import native_calls
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    validate_native_occurrence)
from mindie_sim.prediction.piecewise.model_pipeline import (
    producing_selected_geometry, selected_geometry)
from mindie_sim.prediction.piecewise.native_r31_aclnn_index import (
    ACLNN_API, CALLER, CALLER_LINE, INDICES_LINE, KERNEL_NAME,
    LIBOPAPI_SHA256, PROFILER_NAME, R31_SOURCE_VERSION, RUNNER_INDICES,
    binding_spec, dynamic_call, observed_rows)
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)


def _extras(**overrides):
    values = {
        "row_binding": "selected",
        "source_runner_class": "NPUModelRunner",
        "source_runner_invocation": "execute_model",
        "source_runner_indices": RUNNER_INDICES,
        "source_runner_output_selection": "hidden_states[logits_indices]",
        "source_runner_index_class": "advanced_tensor_index",
        "source_runner_revision": "r31/9e869e8148",
        "source_runner_indices_attached": 0,
        "source_runner_policy_source": "retained_r31_runner_policy",
        "source_runner_native_acl_api": ACLNN_API,
        "source_runner_observed_kernel": KERNEL_NAME,
        "source_runner_observed_profiler": PROFILER_NAME,
        "source_runner_observed_core": "aiv",
    }
    values.update(overrides)
    return tuple((key, value) for key, value in values.items()
                 if value is not None)


def _compiled(*, extras=None, variant="all", kind="O0",
              name="hidden_select", family="copy", role="select",
              act_bytes=2.0, hidden=5120, indices_attached=0):
    extras = (_extras(source_runner_indices_attached=indices_attached)
              if extras is None else tuple(extras))
    theta = StaticSignature(
        role, matrix=(("elements", hidden),),
        quant=(("act_bytes", act_bytes),), extras=extras)
    template = OperatorTemplate(name=name, family=family, role=role)
    return SimpleNamespace(
        kind=kind, variant=variant,
        operator=BoundOperator(template, theta),
        feature_metadata=(), siblings=())


def _binding(compiled=None):
    compiled = compiled or _compiled()
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    surface = native_calls._surface(
        compiled, spec, source_boundary="kernel")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return spec, bind_native_call(declared)


def _geometry(*pairs):
    requests = tuple(
        RequestGeometry(q_len, context, phase="prefill",
                        query_ranges=((q_len - 1, q_len),))
        for q_len, context in pairs)
    return WorkloadGeometry(requests=requests, phase="prefill")


def test_binding_spec_is_the_declared_source_runner_identity():
    spec, binding = _binding()
    assert type(spec).__name__ == "_CallSpec"
    assert spec.api == {
        "namespace": "aclnn", "entry": ACLNN_API,
        "aclnn_symbol": ACLNN_API, "version": R31_SOURCE_VERSION}
    assert spec.source["path"] == CALLER["path"]
    assert spec.source["sha256"] == CALLER["sha256"]
    assert spec.source["symbol"] == CALLER["symbol"]
    assert spec.source["revision"] == "r31/9e869e8148"
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["profiler_name"] == PROFILER_NAME
    assert spec.kernels[0]["source_sha256"] == LIBOPAPI_SHA256
    assert spec.typed["dtype"] == "bf16"
    assert spec.typed["layout"] == "ND"
    assert spec.typed["shapes"] == {"hidden_states": [5120]}
    attrs = spec.typed["attrs"]
    assert attrs["acl_api"] == ACLNN_API
    assert attrs["op_type"] == KERNEL_NAME
    assert attrs["row_binding"] == "selected"
    assert attrs["source_runner_class"] == "NPUModelRunner"
    assert attrs["source_runner_invocation"] == "execute_model"
    assert attrs["source_runner_output_selection"] == (
        "hidden_states[logits_indices]")
    assert attrs["source_runner_index_class"] == "advanced_tensor_index"
    assert attrs["source_runner_revision"] == "r31/9e869e8148"
    assert attrs["source_runner_indices_attached"] is False
    assert attrs["source_lines"] == [INDICES_LINE, CALLER_LINE]
    assert attrs["source_sha256"] == CALLER["sha256"]
    assert spec.owner == "hidden_select"
    assert set(spec.unresolved) >= {
        "python_wrapper_not_retained", "acl_api_header_not_retained",
        "acl_parameter_order_unresolved", "index_kernel_source_not_retained",
        "execution_rank_not_selected"}
    assert binding.timing_owner == {
        "op_id": "O0:copy:hidden_select", "scope": "call"}
    assert binding.execution_core == "aiv"
    assert binding.call_id == (
        "5aef65940702177adbc8f881bf36b25ce693590fc1294e9b43f72f240931ffc7")
    # The same declared runner is phase-independent: no qwen/image gate.
    assert binding_spec(_compiled(), phase="decode") is not None
    assert binding_spec(_compiled(), phase="") is not None


def test_dynamic_call_uses_producing_and_selected_geometry():
    compiled = _compiled()
    _, binding = _binding(compiled)
    env = dynamic_call(
        binding, compiled, _geometry((4096, 0)),
        validate=validate_native_occurrence)
    assert env is not None
    actual = env["actual"]
    assert actual["dtype"] == "bf16" and actual["layout"] == "ND"
    assert actual["shapes"] == {
        "hidden_states": [4096, 5120],
        "logits_indices": [1],
        "sample_hidden_states": [1, 5120],
    }
    assert actual["attrs"]["producing_rows"] == 4096
    assert actual["attrs"]["selected_rows"] == 1
    assert actual["attrs"]["source_rows"] == 4096
    assert actual["attrs"]["index_values_status"] == "not_captured"
    prepare = env["prepare"]
    assert prepare["shapes"] == actual["shapes"]
    assert prepare["arguments"] == {
        "self": "hidden_states", "index": "logits_indices",
        "out": "sample_hidden_states"}
    assert prepare["argument_order"] == ["self", "index", "out"]
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False and prepare["partial"] is True
    assert prepare["descriptors"]["hidden_states"]["shape"] == [4096, 5120]
    assert prepare["descriptors"]["logits_indices"] == {
        "shape": [1], "dtype": "int64", "layout": "ND", "role": "input",
        "pointer": None, "allocation": "external"}
    assert prepare["source_runner"]["class"] == "NPUModelRunner"
    assert prepare["source_runner"]["output_selection"] == (
        "hidden_states[logits_indices]")
    assert prepare["source_runner"]["indices_attached"] is False
    assert prepare["source_runner"]["index_values_status"] == "not_captured"
    assert prepare["source_runner"]["indices"] == RUNNER_INDICES
    assert prepare["lowering"]["index_value_source_line"] == INDICES_LINE
    assert "source_runner_indices_not_attached" in prepare["unresolved"]
    assert prepare["match"]["native_type"] == KERNEL_NAME
    assert prepare["match"]["producing_rows"] == 4096
    assert prepare["match"]["selected_rows"] == 1
    assert prepare["match"]["source_lines"] == [INDICES_LINE, CALLER_LINE]
    observed = prepare["observed"]
    assert observed["input_shapes"] == [[4096, 5120], [1], [2], [1]]
    assert observed["input_dtypes"] == [
        "DT_BF16", "INT64", "INT64", "INT64"]
    assert observed["output_shapes"] == [[1, 5120]]
    assert observed["sample_matches_current_geometry"] is True
    assert observed["index_values_status"] == "not_captured"
    assert observed["csv_sha256"] == (
        "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a")


def test_attached_indices_remove_the_partial_indices_reason():
    compiled = _compiled(indices_attached=1)
    spec, binding = _binding(compiled)
    assert spec.typed["attrs"]["source_runner_indices_attached"] is True
    env = dynamic_call(
        binding, compiled, _geometry((4096, 0)),
        validate=validate_native_occurrence)
    assert env is not None
    assert env["prepare"]["source_runner"]["indices_attached"] is True
    assert "source_runner_indices_not_attached" not in (
        env["prepare"]["unresolved"])
    assert env["prepare"]["complete"] is False


def test_multi_request_uses_each_producing_row_count():
    compiled = _compiled()
    _, binding = _binding(compiled)
    env = dynamic_call(
        binding, compiled, _geometry((2048, 0), (1024, 0)),
        validate=validate_native_occurrence)
    assert env is not None
    assert env["actual"]["shapes"] == {
        "hidden_states": [3072, 5120],
        "logits_indices": [2],
        "sample_hidden_states": [2, 5120],
    }
    assert env["actual"]["attrs"]["producing_rows"] == 3072
    assert env["actual"]["attrs"]["selected_rows"] == 2
    assert env["prepare"]["observed"]["sample_matches_current_geometry"] is False


def test_rejects_other_runners_rows_and_dtypes():
    good = _compiled()
    for other in (
            _compiled(extras=_extras(row_binding="")),
            _compiled(extras=_extras(source_runner_class="OtherRunner")),
            _compiled(extras=_extras(
                source_runner_invocation="other_invocation")),
            _compiled(extras=_extras(
                source_runner_output_selection="other[logits]")),
            _compiled(extras=_extras(
                source_runner_index_class="basic_index")),
            _compiled(extras=_extras(
                source_runner_revision="r35/5461a1a8")),
            _compiled(variant="selected"),
            _compiled(kind="F1"),
            _compiled(name="final_norm"),
            _compiled(family="gemm"),
            _compiled(act_bytes=1.0),
            _compiled(hidden=0)):
        assert binding_spec(other, phase="prefill") is None
    assert binding_spec(good, phase="prefill") is not None


def test_producing_selected_geometry_keeps_all_active_rows():
    active = WorkloadGeometry(
        (RequestGeometry(4096, 0, phase="prefill"),
         RequestGeometry(2048, 0, phase="prefill")), phase="prefill")
    selected = selected_geometry(active, (True, False))
    produced = producing_selected_geometry(active, selected)
    assert produced is not None
    assert produced.tokens == 6144
    assert produced.selected_tokens == 1
    assert produced.producing_lm_head_rows() == 1
    assert selected_geometry(active, (False, False)) is None
    assert producing_selected_geometry(active, None) is None


def test_retained_observed_sample_matches_the_source_row():
    rows = observed_rows()
    row = rows["O0:copy:hidden_select"]
    assert row["source"] == [4096, 5120]
    assert row["index"] == [1]
    assert row["output"] == [1, 5120]
    assert row["source_dtype"] == "bf16"
    assert row["index_dtype"] == "int64"
    assert row["output_dtype"] == "bf16"
    assert row["profiler_name"] == PROFILER_NAME
    assert row["acl_api"] == ACLNN_API
    assert row["kernel"] == KERNEL_NAME
    assert row["caller"] == CALLER["symbol"]
    assert row["caller_line"] == CALLER_LINE
    assert row["indices_line"] == INDICES_LINE
    assert row["complete"] is False
