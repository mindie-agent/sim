"""Source-bound R64 SFA indexer ``_triton_rope_siso`` adapter invariants.

[Role] Exercise the query and key launch adapters against the exact retained
  R64 caller/kernel identity and check typed occurrences for the enrolled
  prefill geometry and an alternate geometry/phase/dtype.

[Invariants] The adapter derives head count, head width, rotary width, style,
  dtype and phase from the producer-declared callable facts; it does not gate
  on the enrolled 32/128/64 prefill sample.  Query and key remain separate
  in-place launches; each row owns one native launch; the static call identity
  does not change with ``T``.  Other images, missing source implementation,
  missing geometry and unsupported dtypes are rejected.  Prepare claims no
  allocation, launch, cubin or measured duration.

[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local construction only; no product state mutation.
"""
from __future__ import annotations

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
    CALLER_LINES, CALLER_SOURCE, ENTRY, KERNEL_LINES, KERNEL_NAME,
    LAUNCH_LINES, R64_IMAGE_SHA, R64_SOURCE_COMMIT, ROLE_KEY, ROLE_QUERY,
    SOURCE_IMPLEMENTATION, WRAPPER_LINES, WRAPPER_SOURCE,
    binding_spec, dynamic_call)
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)

SUPPLY = PipeSupply((
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
    ("link_hccs", 2000.0), ("link_roce", 500.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=1, moe_row_tile=16,
                              split_target=10 ** 9, max_splits=8)
# Enrolled sample, retained only as one test fixture.
SAMPLE_HEADS = 32
SAMPLE_DIM = 128
SAMPLE_ROPE = 64
BASE_EXTRAS = (
    ("rope_implementation", SOURCE_IMPLEMENTATION),
    ("index_dim", SAMPLE_DIM), ("qk_rope_head_dim", SAMPLE_ROPE),
    ("is_neox_style", False), ("rope_dtype", "bf16"),
    ("index_query_heads", SAMPLE_HEADS), ("index_key_heads", 1),
)
QUERY_EXTRAS = BASE_EXTRAS + (
    ("index_rope_role", ROLE_QUERY), ("index_heads", SAMPLE_HEADS))
KEY_EXTRAS = BASE_EXTRAS + (
    ("index_rope_role", ROLE_KEY), ("index_heads", 1))
META = (
    ("image_id", f"sha256:{R64_IMAGE_SHA}"),
    ("image", "nightly-ci-releases-v0.32.0rc-a5-openeuler-cann9.2.1-20261101-r64"),
)


def _compiled(*, name="index_rope", elements=SAMPLE_HEADS * SAMPLE_DIM,
              extras=QUERY_EXTRAS, image_id=f"sha256:{R64_IMAGE_SHA}",
              kind="I0"):
    template = OperatorTemplate(name=name, family="elementwise", role="rope")
    signature = StaticSignature(
        role="rope", matrix=(("elements", int(elements)),),
        quant=(("act_bytes", 2),),
        extras=(("ops_per_element", 6),
                ("read_bytes", float(elements * 2)),
                ("write_bytes", float(elements * 2)), *extras))
    operator = BoundOperator(template, signature)
    metadata = (("image_id", image_id), ("image", "r64-test"))
    return OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        operator, kind=kind, feature_metadata=metadata)


def _binding(compiled, phase="prefill"):
    spec = binding_spec(compiled, phase=phase)
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


def test_r64_indexer_query_and_key_are_separate_one_launch_calls():
    query = _compiled()
    query_spec, query_binding = _binding(query)
    assert query_spec.api["namespace"] == "triton"
    assert query_spec.api["entry"] == ENTRY
    assert query_spec.source["sha256"] == CALLER_SOURCE["sha256"]
    assert query_spec.source["symbol"] == "AscendSFAIndexerBackend.forward"
    assert query_spec.kernels[0]["name"] == KERNEL_NAME
    assert query_spec.kernels[0]["source_sha256"] == WRAPPER_SOURCE["sha256"]
    assert query_spec.kernels[0]["kernel_lines"] == KERNEL_LINES
    assert query_spec.kernels[0]["wrapper_lines"] == WRAPPER_LINES
    assert query_spec.kernels[0]["launch_lines"] == LAUNCH_LINES
    assert query_spec.kernels[0]["caller_lines"] == CALLER_LINES[ROLE_QUERY]
    assert query_spec.typed["attrs"]["index_rope_role"] == ROLE_QUERY
    assert query_spec.typed["attrs"]["n_heads"] == SAMPLE_HEADS
    assert query_spec.typed["attrs"]["is_neox_style"] is False
    assert query_binding.timing_owner == {
        "op_id": "I0:elementwise:index_rope", "scope": "call"}
    assert query_binding.member_op_ids == ("I0:elementwise:index_rope",)

    first = dynamic_call(query_binding, query, _geometry(1024),
                         validate=validate_native_occurrence)
    assert first is not None
    actual = first["actual"]
    assert actual["dtype"] == "bf16"
    assert actual["shapes"] == {
        "qk": [1024, 32, 128], "cos": [1024, 64],
        "sin": [1024, 64], "output": [1024, 32, 128]}
    assert actual["attrs"]["qk_row_stride"] == 32 * 128
    assert actual["attrs"]["pad_n_h"] == 32
    assert actual["attrs"]["pad_rope_dim"] == 64
    prepare = first["prepare"]
    assert prepare["call"] == ENTRY
    assert prepare["arguments"] == {
        "qk": "qk", "cos": "cos", "sin": "sin",
        "rope_dim": 64, "is_neox_style": False}
    assert prepare["descriptors"]["qk"]["in_place"] is True
    assert prepare["descriptors"]["output"]["alias_of"] == "qk"
    assert prepare["descriptors"]["output"]["shape"] == [1024, 32, 128]
    assert prepare["kernel_arguments"][-2] == {
        "name": "IS_NEOX_STYLE", "value": False}
    assert prepare["kernel_arguments"][-1] == {
        "name": "USE_COS_SIN", "value": False}
    assert prepare["indexer_rope_pair"]["query"]["this_row"] is True
    assert prepare["indexer_rope_pair"]["key"]["shape"] == [1024, 1, 128]
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert "compiled_triton_cubin_not_retained" in prepare["partial_reasons"]
    assert prepare["match"]["native_type"] == KERNEL_NAME
    assert prepare["match"]["source_lines"] == CALLER_LINES[ROLE_QUERY]
    assert prepare["match"]["kernel_source_sha256"] == WRAPPER_SOURCE["sha256"]
    assert prepare["match"]["kernel_lines"] == KERNEL_LINES
    assert prepare["match"]["launch_lines"] == LAUNCH_LINES
    assert prepare["match"]["source_commit"] == R64_SOURCE_COMMIT

    key = _compiled(name="index_k_rope", elements=SAMPLE_DIM,
                    extras=KEY_EXTRAS)
    key_spec, key_binding = _binding(key)
    assert key_spec.typed["attrs"]["index_rope_role"] == ROLE_KEY
    assert key_spec.typed["attrs"]["n_heads"] == 1
    assert key_spec.source["symbol"] == "AscendSFAIndexerBackend.forward_k"
    assert key_binding.timing_owner == {
        "op_id": "I0:elementwise:index_k_rope", "scope": "call"}
    assert key_binding.call_id != query_binding.call_id
    key_out = dynamic_call(key_binding, key, _geometry(1024),
                           validate=validate_native_occurrence)
    assert key_out is not None
    assert key_out["actual"]["shapes"] == {
        "qk": [1024, 1, 128], "cos": [1024, 64],
        "sin": [1024, 64], "output": [1024, 1, 128]}
    assert key_out["actual"]["attrs"]["qk_row_stride"] == 128
    assert key_out["actual"]["attrs"]["pad_n_h"] == 1
    assert key_out["prepare"]["match"]["source_lines"] == CALLER_LINES[ROLE_KEY]
    assert key_out["prepare"]["indexer_rope_pair"]["key"]["this_row"] is True

    held = dynamic_call(query_binding, query, _geometry(2048),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"]["qk"] == [2048, 32, 128]
    assert held["actual"]["shapes"]["cos"] == [2048, 64]
    # A new token count is a new occurrence, not a new selected call.
    assert query_binding.call_id == _binding(_compiled())[1].call_id


def test_geometry_phase_and_dtype_are_derived_not_the_sample():
    # Alternate valid callable geometry: 64-wide head, 32-wide rotary,
    # NeoX style, FP16, decode phase.  None of these are the sample values.
    alt_base = (
        ("rope_implementation", SOURCE_IMPLEMENTATION),
        ("index_dim", 64), ("qk_rope_head_dim", 32),
        ("is_neox_style", True), ("rope_dtype", "fp16"),
        ("index_query_heads", 16), ("index_key_heads", 1),
    )
    query = _compiled(
        elements=16 * 64,
        extras=alt_base + (("index_rope_role", ROLE_QUERY),
                           ("index_heads", 16)))
    query_spec, query_binding = _binding(query, phase="decode")
    assert query_spec.typed["attrs"]["head_dim"] == 64
    assert query_spec.typed["attrs"]["rope_dim"] == 32
    assert query_spec.typed["attrs"]["is_neox_style"] is True
    assert query_spec.typed["attrs"]["rope_dtype"] == "fp16"
    query_out = dynamic_call(query_binding, query, _geometry(8, phase="decode"),
                             validate=validate_native_occurrence)
    assert query_out is not None
    assert query_out["actual"]["dtype"] == "fp16"
    assert query_out["actual"]["shapes"] == {
        "qk": [8, 16, 64], "cos": [8, 32],
        "sin": [8, 32], "output": [8, 16, 64]}
    assert query_out["actual"]["attrs"]["pad_n_h"] == 16
    assert query_out["actual"]["attrs"]["pad_rope_dim"] == 32
    assert query_out["prepare"]["arguments"]["is_neox_style"] is True
    assert query_out["prepare"]["indexer_rope_pair"]["query"]["shape"] == [
        8, 16, 64]
    assert query_out["prepare"]["indexer_rope_pair"]["key"]["shape"] == [
        8, 1, 64]

    key = _compiled(name="index_k_rope", elements=64,
                    extras=alt_base + (("index_rope_role", ROLE_KEY),
                                       ("index_heads", 1)))
    _, key_binding = _binding(key, phase="decode")
    key_out = dynamic_call(key_binding, key, _geometry(8, phase="decode"),
                           validate=validate_native_occurrence)
    assert key_out is not None
    assert key_out["actual"]["shapes"]["qk"] == [8, 1, 64]
    assert key_out["actual"]["attrs"]["qk_row_stride"] == 64


def test_missing_source_or_invalid_callable_facts_are_rejected():
    query = _compiled()
    assert binding_spec(query, phase="") is None
    assert binding_spec(query, phase="other_phase") is None
    assert binding_spec(_compiled(image_id="sha256:" + "0" * 64),
                        phase="prefill") is None

    def without(*keys):
        return tuple(item for item in QUERY_EXTRAS if item[0] not in keys)

    assert binding_spec(_compiled(extras=without("rope_implementation")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=without("index_dim")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=without("qk_rope_head_dim")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=without("is_neox_style")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=without("rope_dtype")),
                        phase="prefill") is None

    def replaced(key, value):
        return tuple((name, value if name == key else item)
                     for name, item in QUERY_EXTRAS)

    assert binding_spec(_compiled(extras=replaced("qk_rope_head_dim", 130)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=replaced("qk_rope_head_dim", 63)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=replaced("rope_dtype", "fp8_e4m3")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=replaced("is_neox_style", "maybe")),
                        phase="prefill") is None

    _, binding = _binding(query)
    assert dynamic_call(binding, query, _geometry(1024, phase="decode"),
                        validate=validate_native_occurrence) is None
    assert dynamic_call(binding, query, _geometry(0),
                        validate=validate_native_occurrence) is None

    key_missing_query_heads = _compiled(name="index_k_rope", elements=SAMPLE_DIM,
                                        extras=(
        ("rope_implementation", SOURCE_IMPLEMENTATION),
        ("index_heads", 1), ("index_key_heads", 1),
        ("index_dim", 128), ("qk_rope_head_dim", 64),
        ("is_neox_style", False), ("rope_dtype", "bf16"),
        ("index_rope_role", ROLE_KEY)))
    assert binding_spec(key_missing_query_heads, phase="prefill") is None
