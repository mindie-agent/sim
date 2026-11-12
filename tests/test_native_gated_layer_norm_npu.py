"""Real A3 Qwen3.6 GDN gated RMSNorm native-call preparation."""
import csv
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.native_binding import declare_native_call, bind_native_call
from mindie_sim.prediction.piecewise import native_calls
from mindie_sim.prediction.piecewise import native_gated_layer_norm_npu as gn
from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply)
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.subgraphs.builders_attention import _build_a3
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.schema.subgraphs import BoundOperator, SubgraphInstance
from mindie_sim.prediction.subgraphs.state import StateView

_SUPPLY = PipeSupply(
    (("cube", 1e9), ("vector", 1e9), ("hbm", 1e9), ("mte1", 1e9),
     ("mte2", 1e9), ("mte2_aiv", 1e9), ("mte2_aic", 1e9), ("mte3", 1e9),
     ("fixpipe", 1e9), ("link", 1e9), ("scalar_store_issue", 1e9),
     ("reg_data_writes", 1e9), ("reg_ub_accesses", 1e9),
     ("reg_ub_stores", 1e9), ("reg_alu_lanes", 1e9),
     ("reg_special_lanes", 1e9), ("reg_reduce_lanes", 1e9),
     ("reg_shuffle_lanes", 1e9)),
    resource_units=tuple((name, 64) for name in (
        "vector", "mte2_aiv", "mte2_aic", "mte3", "hbm",
        "scalar_store_issue", "reg_data_writes", "reg_ub_accesses",
        "reg_ub_stores", "reg_alu_lanes", "reg_special_lanes",
        "reg_reduce_lanes", "reg_shuffle_lanes", "mte1", "fixpipe")),
    service_modes=(("reg_issue_slots", ("vector", 211200000000.0)),
                   ("ub_copy", ("vector", 27033600000000.0))),
    compute_units=32, buffers=(("ub", 253952),), pool_units=(("ub", 64),),
    compute_peaks=(("bf16", 432537600000000.0),
                   ("fp32", 27033600000000.0)))
_POLICY = ImplementationPolicy(mt=128, nt=16, cores=32, moe_row_tile=4,
                               split_target=1000, max_splits=4)
_PF4K_METADATA = (
    ("arch_family", "A5"), ("model_identifier", "qwen3_5"),
    ("hardware_key", "Ascend950DT_9582"), ("device", "Ascend950DT_9582"),
    ("image_id", ("sha256:9e869e8148467296f56d1001b0dac4fe"
                  "a893c841ac7d0a03d4e824a233c2ffea")),
    ("image", ("swr.cn-north-12.myhuaweicloud.com/atlas_inference/"
               "vllm-ascend:nightly-ci-releases-v0.30.0-a5-cann9.2.0_"
               "20261022115200-20261023-r31")))
_FEATURES = (("arch_family", "A3"), ("model_identifier", "qwen3.6-27b"),
             ("hardware_key", "Ascend910_9362"),
             ("device", "Ascend910_9362"),
             ("image_id", ("sha256:9e869e8148467296f56d1001b0dac4fe"
                           "a893c841ac7d0a03d4e824a233c2ffea")),
             ("image", ("swr.cn-north-12.myhuaweicloud.com/atlas_inference/"
                        "vllm-ascend:nightly-ci-releases-v0.30.0-a5-cann9.2.0_"
                        "20261022115200-20261023-r31")))


def _compiled_a3(*, model_key="qwen3.6-27b", tp=4,
                 activation_format="int8", metadata=_FEATURES):
    model = replace(load_model(model_key), quant=QuantSpec(
        name="w8a8", weight_format="int8",
        activation_format=activation_format))
    layer = next(plan for plan in model.layer_plans if plan.linear_heads)
    features = FeatureConfig.from_model(
        model, tp=tp, dp=1, ep=1, pp=1, token_tile=128, kv_tile=16,
        cores=32, moe_row_tile=4, split_target=1000, max_splits=4,
        metadata=metadata)
    built = _build_a3(_Ctx(model, layer, features, stage=0,
                           layers=len(model.layer_plans)))
    operators = tuple(
        BoundOperator(template=template, theta=built.signature(template.name))
        for template in built.template.operators)
    instance = SubgraphInstance(
        instance_id="A3:recurrent_decode:L0:attention",
        template=built.template, layer=0, stage=0, operators=operators,
        metadata=(("role", "attention"), ("attention_kind", "gdn")))
    compiled = OperatorPiecewiseCompiler(
        _SUPPLY, _POLICY).compile(instance, feature_metadata=metadata)
    return next(operator for operator in compiled.operators
                if operator.operator.name == "gated_norm")


def _binding(compiled, phase="decode"):
    spec = gn.binding_spec(compiled, phase=phase)
    assert spec is not None
    surface = native_calls._surface(compiled, spec, source_boundary="kernel")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    binding = bind_native_call(declared)
    assert binding is not None
    return spec, replace(binding, timing_owner={
        "op_id": spec.owner, "scope": "call"})


def _decode_geometry():
    return WorkloadGeometry(
        tuple(RequestGeometry(4, 4232, phase="decode") for _ in range(64)),
        graph_bucket=0, cohort_batch=64, phase="decode", local_batch=64)


def test_real_a3_qwen36_decode_occurrence_and_unchanged_cost():
    compiled = _compiled_a3()
    theta = compiled.operator.theta
    geometry = _decode_geometry()
    cost_before = compiled.evaluate(geometry, StateView()).duration_ms
    spec, binding = _binding(compiled, "decode")
    assert spec.typed["dtype"] == "bf16"
    assert spec.typed["attrs"]["local_value_heads"] == 12
    assert spec.typed["attrs"]["epsilon"] == 1e-6
    assert spec.typed["attrs"]["group_width"] == 128
    assert spec.typed["attrs"]["activation"] == "swish"
    assert spec.typed["attrs"]["norm_before_gate"] is None
    dynamic = gn.dynamic_call(
        binding, compiled, geometry,
        validate=native_calls.validate_native_occurrence)
    assert dynamic is not None
    actual = dynamic["actual"]
    assert actual["dtype"] == "bf16" and actual["layout"] == "ND"
    assert actual["shapes"] == {
        "x": [3072, 128], "weight": [128], "gate": [3072, 128],
        "output": [3072, 128], "rstd": [3072]}
    assert actual["attrs"]["rows"] == 3072
    assert actual["attrs"]["ordinary_compiled_act_bytes"] == 1.0
    assert actual["attrs"]["ordinary_compiled_act_bytes_mismatch"] is True
    descriptors = dynamic["prepare"]["descriptors"]
    assert {name: (item["shape"], item["dtype"]) for name, item in
            descriptors.items() if item["present"]} == {
        "x": ([3072, 128], "bf16"), "weight": ([128], "bf16"),
        "gate": ([3072, 128], "bf16"), "output": ([3072, 128], "bf16"),
        "rstd": ([3072], "fp32")}
    assert compiled.operator.theta == theta
    assert compiled.evaluate(geometry, StateView()).duration_ms == cost_before


def test_real_a3_qwen36_prefill_abi_keeps_captured_row_shape():
    compiled = _compiled_a3()
    _spec, binding = _binding(compiled, "prefill")
    prefill = WorkloadGeometry.homogeneous(
        1, 8192, 8192, phase="prefill", graph_bucket=1,
        cohort_batch=1, local_batch=1)
    dynamic = gn.dynamic_call(
        binding, compiled, prefill,
        validate=native_calls.validate_native_occurrence)
    assert dynamic is not None
    assert dynamic["actual"]["shapes"]["x"] == [98304, 128]
    assert dynamic["actual"]["shapes"]["gate"] == [98304, 128]
    assert dynamic["actual"]["shapes"]["rstd"] == [98304]
    fixture = (Path(__file__).resolve().parents[2] / "profiler/tests/"
               "fixtures/body_fused/a3_qwen36.csv")
    if not fixture.is_file():
        pytest.skip("retained A3 Qwen3.6 ABI fixture is unavailable")
    with fixture.open(newline="") as handle:
        row = next(item for item in csv.DictReader(handle)
                   if item.get("Type") == "_layer_norm_fwd_1pass_kernel_npu")

    def _shape(text):
        return [tuple(int(value) for value in item.split(","))
                for item in text.strip('"').split(";")]

    assert _shape(row["Input Shapes"]) == [(98304, 128), (128,),
                                           (98304, 128)]
    assert _shape(row["Output Shapes"]) == [(98304, 128), (98304,)]
    assert row["Input Data Types"] == "DT_BF16;DT_BF16;DT_BF16"
    assert row["Output Data Types"] == "DT_BF16;FLOAT"


def test_original_pf4k_a5_tp2_local24_bf16_and_q1():
    compiled = _compiled_a3(tp=2, activation_format="bf16",
                            metadata=_PF4K_METADATA)
    _spec, binding = _binding(compiled, "prefill")
    assert _spec.typed["attrs"]["local_value_heads"] == 24
    assert dict(compiled.operator.theta.extras)["act_bytes"] == 2.0
    prefill = WorkloadGeometry.homogeneous(
        1, 4096, 4096, phase="prefill", graph_bucket=1,
        cohort_batch=1, local_batch=1)
    dynamic = gn.dynamic_call(
        binding, compiled, prefill,
        validate=native_calls.validate_native_occurrence)
    assert dynamic is not None
    assert dynamic["actual"]["attrs"]["rows"] == 98304
    assert dynamic["actual"]["shapes"] == {
        "x": [98304, 128], "weight": [128], "gate": [98304, 128],
        "output": [98304, 128], "rstd": [98304]}
    q1 = WorkloadGeometry(
        (RequestGeometry(1, 0, phase="decode"),), graph_bucket=1,
        phase="decode", local_batch=1)
    decode = gn.dynamic_call(
        binding, compiled, q1,
        validate=native_calls.validate_native_occurrence)
    assert decode is not None
    assert decode["actual"]["attrs"]["rows"] == 24
    assert decode["actual"]["shapes"]["x"] == [24, 128]
    assert decode["actual"]["shapes"]["rstd"] == [24]
    assert binding.call_id == (
        "7e600cd5c60f1de2df9e6f56c8ccd9664900a0559d78c86b95c88ec6bbba6fbf")


def test_equal_elements_different_head_widths_do_not_collapse():
    real = _compiled_a3(tp=2, activation_format="bf16",
                        metadata=_PF4K_METADATA)
    fake_theta = SimpleNamespace(
        matrix=(("elements", 3072),),
        extras=(("act_bytes", 2.0), ("result_bytes", 2.0),
                ("value_heads", 48), ("dv", 64), ("head_dim", 64),
                ("output_dtype", "bf16"), ("compute_dtype", "bf16")),
        quant=(("act_bytes", 2.0),))
    fake = SimpleNamespace(
        operator=SimpleNamespace(name="gated_norm", family="elementwise",
                                 role="norm", theta=fake_theta,
                                 template=real.operator.template),
        kind=real.kind, siblings=real.siblings,
        feature_metadata=real.feature_metadata)
    real_spec, real_binding = _binding(real, "decode")
    fake_spec, fake_binding = _binding(fake, "decode")
    assert real_spec.typed["attrs"]["local_value_heads"] == 24
    assert real_spec.typed["attrs"]["head_dim"] == 128
    assert fake_spec.typed["attrs"]["local_value_heads"] == 48
    assert fake_spec.typed["attrs"]["head_dim"] == 64
    geometry = WorkloadGeometry(
        (RequestGeometry(4, 0, phase="decode"),), graph_bucket=1,
        phase="decode", local_batch=1)
    real_dynamic = gn.dynamic_call(
        real_binding, real, geometry,
        validate=native_calls.validate_native_occurrence)
    fake_dynamic = gn.dynamic_call(
        fake_binding, fake, geometry,
        validate=native_calls.validate_native_occurrence)
    assert real_dynamic["actual"]["shapes"]["x"] == [96, 128]
    assert real_dynamic["actual"]["shapes"]["rstd"] == [96]
    assert fake_dynamic["actual"]["shapes"]["x"] == [192, 64]
    assert fake_dynamic["actual"]["shapes"]["weight"] == [64]
    assert fake_dynamic["actual"]["shapes"]["rstd"] == [192]
    assert 96 * 128 == 192 * 64


def test_unknown_or_int8_native_dtype_stays_unbound():
    real = _compiled_a3(tp=2, activation_format="bf16",
                        metadata=_PF4K_METADATA)
    theta = SimpleNamespace(
        matrix=(("elements", 3072),),
        extras=(("act_bytes", 1.0), ("result_bytes", 1.0),
                ("value_heads", 24), ("dv", 128)),
        quant=(("act_bytes", 1.0),))
    compiled = SimpleNamespace(
        operator=SimpleNamespace(name="gated_norm", family="elementwise",
                                 role="norm", theta=theta,
                                 template=real.operator.template),
        kind=real.kind, siblings=real.siblings,
        feature_metadata=real.feature_metadata)
    assert gn.binding_spec(compiled, phase="decode") is None


def test_binding_is_deterministic_and_standalone():
    compiled = _compiled_a3()
    first = _binding(compiled)[1]
    second = _binding(compiled)[1]
    assert first.call_id == second.call_id
    assert first.member_op_ids == (first.op_id,)
    assert first.timing_owner["op_id"] == first.op_id
    assert first.execution_core == "aiv"
    assert first.source["sha256"] == gn.R31_CALLER["sha256"]
    assert first.source["lines"] == gn.R31_CALLER["lines"]
    assert first.kernels[0]["source_sha256"] == gn.R31_KERNEL["sha256"]
    assert first.typed["attrs"]["source_wrapper_sha256"] == gn.R31_WRAPPER["sha256"]
    assert "norm_before_gate_not_observed" in first.unresolved


def test_r31_identity_admits_a5_and_model_label_variation():
    compiled = _compiled_a3()

    def variant(metadata):
        return SimpleNamespace(operator=compiled.operator, kind=compiled.kind,
                               siblings=compiled.siblings,
                               feature_metadata=tuple(metadata.items()))

    base = dict(_FEATURES)
    a5 = variant({**base, "arch_family": "A5",
                  "hardware_key": "Ascend950DT_9582",
                  "device": "Ascend950DT_9582",
                  "model_identifier": "qwen3_5"})
    qwen35 = variant({**base, "model_identifier": "qwen3.5-35b-a3b"})
    same_revision = variant({**base, "image_id": "",
                             "image": ("vllm-ascend:0.30.0rc2.dev200+"
                                       "g8fd0bc488")})
    r31_digest = variant({**base, "image": "r31",
                          "image_id": ("sha256:9e869e8148467296f56d1001"
                                       "b0dac4fea893c841ac7d0a03d4e824a233"
                                       "c2ffea")})
    for candidate in (a5, qwen35, same_revision, r31_digest):
        assert gn.binding_spec(candidate, phase="decode") is not None


def test_genuinely_unsupported_paths_stay_unbound():
    compiled = _compiled_a3()
    base = dict(_FEATURES)
    rmsnorm = SimpleNamespace(
        operator=SimpleNamespace(name="rmsnorm", family="rmsnorm",
                                 role="norm", theta=compiled.operator.theta),
        kind=compiled.kind, siblings=compiled.siblings,
        feature_metadata=compiled.feature_metadata)
    add_rmsnorm = SimpleNamespace(
        operator=SimpleNamespace(name="add_rmsnorm", family="add_rmsnorm",
                                 role="norm", theta=compiled.operator.theta),
        kind=compiled.kind, siblings=compiled.siblings,
        feature_metadata=compiled.feature_metadata)
    kda = SimpleNamespace(operator=compiled.operator, kind=compiled.kind,
                          siblings=compiled.siblings + (SimpleNamespace(
                              name="kda_conv"),),
                          feature_metadata=compiled.feature_metadata)
    bad_width = SimpleNamespace(
        operator=SimpleNamespace(
            name="gated_norm", family="elementwise", role="norm",
            theta=SimpleNamespace(matrix=(("elements", 64),),
                                  extras=(("act_bytes", 1.0),),
                                  quant=(("act_bytes", 1.0),))),
        kind=compiled.kind, siblings=compiled.siblings,
        feature_metadata=compiled.feature_metadata)
    wrong_kind = SimpleNamespace(operator=compiled.operator, kind="A0",
                                 siblings=compiled.siblings,
                                 feature_metadata=compiled.feature_metadata)
    non_gdn = SimpleNamespace(
        operator=compiled.operator, kind=compiled.kind,
        siblings=(SimpleNamespace(name="gated_norm"),),
        feature_metadata=compiled.feature_metadata)
    no_metadata = SimpleNamespace(
        operator=compiled.operator, kind=compiled.kind,
        siblings=compiled.siblings, feature_metadata=())
    tag_only_r31 = SimpleNamespace(
        operator=compiled.operator, kind=compiled.kind,
        siblings=compiled.siblings,
        feature_metadata=tuple({**base, "image_id": "",
                                "image": ("...20261022115200-20261023-r31")
                                }.items()))
    foreign_image = SimpleNamespace(
        operator=compiled.operator, kind=compiled.kind,
        siblings=compiled.siblings,
        feature_metadata=tuple({**base,
                                "image_id": "sha256:c3d00a196905",
                                "image": "sha256:c3d00a196905"}.items()))
    for candidate in (rmsnorm, add_rmsnorm, kda, bad_width, wrong_kind,
                      non_gdn, no_metadata, tag_only_r31, foreign_image):
        assert gn.binding_spec(candidate, phase="decode") is None


def test_consumer_gate_rejects_wrong_width_dtype_and_group():
    compiled = _compiled_a3()
    _spec, binding = _binding(compiled)
    dynamic = gn.dynamic_call(
        binding, compiled, _decode_geometry(),
        validate=native_calls.validate_native_occurrence)
    actual = deepcopy(dynamic["actual"])
    for key, value in (("x", [3072, 256]), ("gate", [3072, 64])):
        bad = deepcopy(actual)
        bad["shapes"][key] = value
        with pytest.raises(Exception, match="shape_mismatch"):
            native_calls.validate_native_occurrence(binding, bad)
    bad = deepcopy(actual)
    bad["dtype"] = "fp16"
    with pytest.raises(Exception, match="dtype_mismatch"):
        native_calls.validate_native_occurrence(binding, bad)
    bad = deepcopy(actual)
    bad["attrs"]["group_width"] = 64
    with pytest.raises(Exception, match="attr_mismatch"):
        native_calls.validate_native_occurrence(binding, bad)
