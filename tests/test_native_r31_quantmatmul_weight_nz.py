"""Focused R31 W8A8-dynamic QuantMatmulWeightNz adapter checks.

[Role] Prove the real R31 PF4kC1 WeightNz rows (A3/F0/A0) select one
  operator-specific call identity and prepare their exact typed tensors,
  while other deployments, dense projections and wrong typed arguments do not.
[Invariants] The R31 image identity is the only source selector; K/N and both
  scale formats are static; only M changes across independent geometries.
[Side-Effect Policy] Read-only, CPU metadata only; no device, weights or I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _CallSpec, _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r31_quantmatmul_weight_nz import (
    R31_IMAGE_SHA, binding_spec, dynamic_call)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_model)
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"
R31 = (("image_id", "sha256:" + R31_IMAGE_SHA),)
R64 = (("image_id", "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"),)
R35 = (("image_id", "sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06"),)
UNKNOWN = (("image_id", "sha256:" + "f" * 64),)
DOMAIN = AnalysisDomain(max_tokens=10 ** 9, max_moe_rows=10 ** 9,
                        max_query_tiles=10 ** 9, max_kv_tiles=10 ** 9,
                        max_index_rows=10 ** 12, max_work=10 ** 15,
                        max_q=10 ** 9, max_s_kv=10 ** 9, max_seq=10 ** 9,
                        max_batch=10 ** 6)

# Original R31 PF4kC1 raw2476b865 kernel rows (one repetition, M=4096).
OBSERVED = {
    ("A3", "qkvz_projection"): {
        "x": [4096, 5120], "weight_nz": [256, 320, 16, 32],
        "weight_scale": [8192], "pertoken_scale": [4096],
        "output": [4096, 8192],
    },
    ("F0", "gate_up"): {
        "x": [4096, 5120], "weight_nz": [544, 320, 16, 32],
        "weight_scale": [17408], "pertoken_scale": [4096],
        "output": [4096, 17408],
    },
    ("F0", "down"): {
        "x": [4096, 8704], "weight_nz": [160, 544, 16, 32],
        "weight_scale": [5120], "pertoken_scale": [4096],
        "output": [4096, 5120],
    },
    ("A0", "qkv_projection"): {
        "x": [4096, 5120], "weight_nz": [224, 320, 16, 32],
        "weight_scale": [7168], "pertoken_scale": [4096],
        "output": [4096, 7168],
    },
    ("A0", "o_projection"): {
        "x": [4096, 3072], "weight_nz": [160, 192, 16, 32],
        "weight_scale": [5120], "pertoken_scale": [4096],
        "output": [4096, 5120],
    },
}


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


def _geometry(tokens):
    return WorkloadGeometry(
        (RequestGeometry(tokens, tokens, phase="prefill"),),
        phase="prefill", cohort_batch=1, local_batch=1)


def _binding(compiled, spec):
    surface = _surface(compiled, spec, source_boundary="kernel")
    return bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved))


@pytest.fixture(scope="module")
def qwen():
    return _case()


def _subgraph(inst, kind, name):
    return next(sub for layer in inst.layers for sub in layer.subgraphs
                if sub.kind == kind
                and any(operator.name == name for operator in sub.operators))


def test_r31_weight_nz_rows_keep_exact_static_and_dynamic_shapes(qwen):
    case, inst = qwen
    compiler = _compiler(case)
    seen = {}
    for (kind, name), observed in OBSERVED.items():
        compiled = compiler.compile(
            _subgraph(inst, kind, name), feature_metadata=R31)
        operator = compiled.operator(name)
        spec = binding_spec(operator)
        assert isinstance(spec, _CallSpec)
        assert spec.api["entry"] == "aclnnQuantMatmulWeightNz"
        assert spec.api["aclnn_symbol"] == "aclnnQuantMatmulWeightNz"
        assert spec.source["sha256"] == (
            "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456")
        assert spec.source["revision"] == "cann-9.2.0-r31"
        assert spec.kernels[0]["name"] == "QuantBatchMatmulV3"
        assert spec.kernels[0]["evidence"] == "measured"
        assert spec.execution_core == "mix"
        assert "python_caller_source_unresolved" in spec.unresolved
        assert "attributable_source_reference_unresolved" in spec.unresolved
        assert spec.typed["dtype"] == "int8"
        assert spec.typed["layout"] == "ND"
        k, n = spec.typed["shapes"]["weight"]
        assert [k, n] == [observed["x"][1], observed["output"][1]]
        assert spec.typed["shapes"]["x"] == [k]
        assert spec.typed["shapes"]["weight_scale"] == [n]
        assert spec.typed["shapes"]["output"] == [n]
        attrs = spec.typed["attrs"]
        assert attrs["weight_device_shape"] == observed["weight_nz"]
        assert attrs["weight_logical_shape"] == [k, n]
        assert attrs["weight_allocation_shape"] is None
        assert attrs["weight_allocation_status"] == (
            "python_caller_source_unresolved")
        assert attrs["weight_pre_nz_layout"] == "unresolved"
        assert attrs["weight_transform"] == "maybe_trans_nz"
        assert attrs["weight_scale_storage_format"] == "fp32"
        assert attrs["weight_scale_runtime_format"] == "bf16"
        assert attrs["pertoken_scale_format"] == "fp32"
        assert attrs["output_dtype"] == "bf16"
        assert attrs["n_padding"] == "active_32"
        assert attrs["padded_n"] == n
        assert attrs["deployment_image_id"] == R31_IMAGE_SHA
        binding = _binding(operator, spec)
        env = dynamic_call(binding, operator, _geometry(4096),
                           validate=validate_native_occurrence)
        assert env is not None
        actual = env["actual"]["shapes"]
        assert actual["x"] == observed["x"]
        assert actual["weight"] == [k, n]
        assert actual["weight_scale"] == observed["weight_scale"]
        assert actual["pertoken_scale"] == observed["pertoken_scale"]
        assert actual["output"] == observed["output"]
        prepare = env["prepare"]
        assert prepare["complete"] is False
        assert prepare["partial"] is True
        assert prepare["runtime_executed"] is False
        assert prepare["allocation"] == "not_performed"
        assert prepare["descriptors"]["weight_nz"]["shape"] == observed["weight_nz"]
        assert prepare["descriptors"]["weight_nz"]["layout"] == "FRACTAL_NZ"
        assert prepare["descriptors"]["weight"]["allocation_shape"] is None
        assert prepare["descriptors"]["weight_scale"]["dtype"] == "bf16"
        assert prepare["descriptors"]["weight_scale"]["storage_format"] == "fp32"
        assert prepare["descriptors"]["weight_scale"]["allocation_shape"] is None
        assert prepare["descriptors"]["pertoken_scale"]["dtype"] == "fp32"
        assert prepare["descriptors"]["pertoken_scale"]["producer"] == "unresolved"
        assert prepare["weight_storage"]["allocation_shape"] is None
        assert prepare["weight_storage"]["transform"] == "maybe_trans_nz"
        assert prepare["descriptors"]["output"]["dtype"] == "bf16"
        assert prepare["observed_acl_tensor_order"] == [
            "x", "weight", "weight_scale", "pertoken_scale"]
        assert prepare["observed_acl_tensor_dtypes"] == [
            "INT8", "INT8", "DT_BF16", "FLOAT"]
        assert prepare["observed_acl_tensor_formats"] == [
            "ND", "FRACTAL_NZ", "ND", "ND"]
        assert prepare["match"]["complete"] is False
        assert prepare["binary_selection"]["status"] == "runtime_binary_only"
        assert prepare["deployment"]["caller_sha256"] is None
        assert prepare["deployment"]["kernel_source_sha256"] is None
        seen[(kind, name)] = binding.call_id
    assert len(set(seen.values())) == len(OBSERVED)
    assert len(seen) == 5


def test_r31_static_identity_survives_independent_m(qwen):
    case, inst = qwen
    compiled = _compiler(case).compile(
        _subgraph(inst, "F0", "gate_up"), feature_metadata=R31)
    operator = compiled.operator("gate_up")
    spec = binding_spec(operator)
    binding = _binding(operator, spec)
    values = {}
    for tokens in (4096, 2048, 2, 1):
        env = dynamic_call(binding, operator, _geometry(tokens),
                           validate=validate_native_occurrence)
        assert env is not None
        shapes = env["actual"]["shapes"]
        assert shapes["x"] == [tokens, 5120]
        assert shapes["output"] == [tokens, 17408]
        assert shapes["pertoken_scale"] == [tokens]
        assert shapes["weight"] == [5120, 17408]
        assert shapes["weight_scale"] == [17408]
        assert env["prepare"]["descriptors"]["weight_nz"]["shape"] == [
            544, 320, 16, 32]
        assert env["prepare"]["match"]["k"] == 5120
        assert env["prepare"]["match"]["n"] == 17408
        values[tokens] = binding.call_id
    assert len(set(values.values())) == 1


def test_other_sources_and_dense_roles_stay_unbound(qwen):
    case, inst = qwen
    compiler = _compiler(case)
    gate_up = _subgraph(inst, "F0", "gate_up")
    for metadata in ((), R64, R35, UNKNOWN):
        compiled = compiler.compile(gate_up, feature_metadata=metadata)
        assert binding_spec(compiled.operator("gate_up")) is None
    dense = compiler.compile(
        _subgraph(inst, "A3", "o_projection"), feature_metadata=R31)
    assert binding_spec(dense.operator("o_projection")) is None
    compiled = compiler.compile(gate_up, feature_metadata=R31)
    operator = compiled.operator("gate_up")
    spec = binding_spec(operator)
    binding = _binding(operator, spec)
    wrong = {
        "dtype": "bf16", "layout": "ND",
        "shapes": {"x": [4096, 5120], "weight": [5120, 17408],
                   "weight_scale": [17408], "pertoken_scale": [4096],
                   "output": [4096, 17408]},
        "attrs": {**spec.typed["attrs"], "m": 4096, "tokens": 4096},
    }
    with pytest.raises(ModelingError, match="native_call_signature_dtype_mismatch"):
        validate_native_occurrence(binding, wrong)


def test_dynamic_call_rejects_another_api_even_with_same_m(qwen):
    case, inst = qwen
    compiled = _compiler(case).compile(
        _subgraph(inst, "F0", "gate_up"), feature_metadata=R31)
    operator = compiled.operator("gate_up")
    spec = binding_spec(operator)
    binding = _binding(operator, spec)
    other = type("Binding", (), {})()
    other.api = {"entry": "torch_npu.npu_quant_matmul",
                 "aclnn_symbol": None}
    other.typed = spec.typed
    other.source = spec.source
    other.call_id = binding.call_id
    assert dynamic_call(other, operator, _geometry(4096),
                        validate=validate_native_occurrence) is None
