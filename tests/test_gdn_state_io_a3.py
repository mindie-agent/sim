"""Focused A3 GDN state-IO producer and family-baseline tests.

The tests drive the actual shared ``_build_a3`` builder and the registered
``state_gather``/``state_scatter`` family compilers with the real
``recurrent_case`` layer/supply.  No tensor, device, native_calls or public
prediction path is invoked.
"""
from dataclasses import replace

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler)
from mindie_sim.prediction.piecewise import native_calls
from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_gather_ssm_states import (
    binding_spec as gather_binding_spec,
    dynamic_call as gather_dynamic_call)
from mindie_sim.prediction.subgraphs.builders_attention import _build_a3
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.subgraphs import BoundOperator, SubgraphInstance
from test_operator_mechanisms import recurrent_case

STATE_NAMES = ("gather_recurrent_state", "scatter_recurrent_state")
R31_IMAGE = (
    "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
OTHER_IMAGE = "sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4"


def _case(*, state_dtype="fp32", source_lowerings=("gdn_decode",), pcp=1,
          image=R31_IMAGE, arch="A5", tp=1, value_heads=3):
    case = recurrent_case()
    layer = replace(
        case.model.layer_plans[0], linear_key_heads=1,
        linear_value_heads=value_heads, linear_heads=value_heads,
        linear_key_head_dim=128, linear_head_dim=128,
        conv_kernel=4, recurrent_state_dtype=state_dtype)
    case = replace(case, model=replace(case.model, layer_plans=(layer,)))
    features = FeatureConfig.from_runtime(
        case.runtime, case.model, machine=case.machine).project(
            tp=tp, dp=1, ep=1, pp=1,
            cores=max(1, case.machine.dies[0].compute_units), graph_buckets=())
    meta = dict(features.metadata)
    if source_lowerings is None:
        meta.pop("source_lowerings", None)
    else:
        meta["source_lowerings"] = source_lowerings
    meta["image_id"] = image
    meta["arch_family"] = arch
    features = replace(features, pcp=pcp, metadata=tuple(sorted(meta.items())))
    return case, layer, features


def _compiled(*, state_dtype="fp32", source_lowerings=("gdn_decode",),
              pcp=1, image=R31_IMAGE, arch="A5", tp=1, value_heads=3):
    case, layer, features = _case(
        state_dtype=state_dtype, source_lowerings=source_lowerings, pcp=pcp,
        image=image, arch=arch, tp=tp, value_heads=value_heads)
    ctx = _Ctx(model=case.model, layer=layer, features=features, stage=0,
               role="attention", variant="", layers=1)
    built = _build_a3(ctx)
    bound = tuple(BoundOperator(operator, built.signature(operator.name))
                  for operator in built.template.operators)
    instance = SubgraphInstance(
        instance_id="A3:test:L0:attention", template=built.template,
        layer=0, stage=0, operators=bound,
        state_in=built.template.state_in, state_out=built.template.state_out)
    supply = PipeSupply.from_hardware(
        case.machine, dtype=case.model.activation_dtype)
    compiler = OperatorPiecewiseCompiler(
        supply, ImplementationPolicy(
            cores=max(1, case.machine.dies[0].compute_units)),
        domain=AnalysisDomain())
    return built, compiler.compile(
        instance, feature_metadata=features.metadata)


def _geometry(*contexts, phase="prefill"):
    requests = tuple(
        RequestGeometry(q_len=64, context=context, phase=phase)
        for context in contexts)
    return WorkloadGeometry(requests=requests, phase=phase)


def _costs(compiled, geometry):
    return {cost.name: cost for cost in compiled.evaluate(
        geometry, StateView()).operators}


def test_a3_places_separate_gather_and_scatter_around_the_chunk():
    built, _ = _compiled()
    names = tuple(operator.name for operator in built.template.operators)
    assert names.index("gather_recurrent_state") < names.index("prepare")
    assert names.index("prepare") < names.index("gated_delta")
    assert names.index("gated_delta") < names.index("scatter_recurrent_state")
    gather = built.signature("gather_recurrent_state")
    scatter = built.signature("scatter_recurrent_state")
    assert gather.find("row_bytes") == 3 * 128 * 128 * 4
    assert scatter.find("row_bytes") == 3 * 128 * 128 * 4
    for theta in (gather, scatter):
        extras = dict(theta.extras)
        assert extras["state_dtype"] == "fp32"
        assert extras["state_layout"] == "V,K"
        assert extras["index_dtype"] == "int32"
        assert extras["value_heads"] == 3
        assert extras["state_io_condition"] == "confirmed_r31_fla"
    assert dict(gather.extras)["gather_branch"] == "fla_gdn_prefill"
    assert dict(scatter.extras)["scatter_branch"] == "fla_gdn_prefill"


def test_prefill_gather_reads_only_masked_rows_and_scatter_writes_batch():
    _, compiled = _compiled()
    state_bytes = 3 * 128 * 128 * 4
    fresh = _costs(compiled, _geometry(0))
    continued = _costs(compiled, _geometry(512))
    assert dict(fresh["gather_recurrent_state"].derived)["state_gather_selected"] == 0
    assert dict(fresh["gather_recurrent_state"].derived)[
        "state_gather_read_bytes"] == 5
    assert dict(fresh["gather_recurrent_state"].derived)[
        "state_gather_write_bytes"] == state_bytes
    assert dict(continued["gather_recurrent_state"].derived)[
        "state_gather_selected"] == 1
    assert dict(continued["gather_recurrent_state"].derived)[
        "state_gather_read_bytes"] == state_bytes + 5
    assert dict(continued["scatter_recurrent_state"].derived)[
        "state_scatter_read_bytes"] == state_bytes + 4
    assert dict(continued["scatter_recurrent_state"].derived)[
        "state_scatter_write_bytes"] == state_bytes
    assert continued["gather_recurrent_state"].duration_ms > 0
    assert continued["scatter_recurrent_state"].duration_ms > 0


def test_batch_scaling_uses_sequences_not_query_tokens():
    _, compiled = _compiled()
    state_bytes = 3 * 128 * 128 * 4
    costs = _costs(compiled, _geometry(0, 512, 1024))
    assert dict(costs["gather_recurrent_state"].derived)[
        "state_gather_rows"] == 3
    assert dict(costs["gather_recurrent_state"].derived)[
        "state_gather_selected"] == 2
    assert dict(costs["gather_recurrent_state"].derived)[
        "state_gather_read_bytes"] == 2 * state_bytes + 15
    assert dict(costs["scatter_recurrent_state"].derived)[
        "state_scatter_rows"] == 3
    assert dict(costs["scatter_recurrent_state"].derived)[
        "state_scatter_write_bytes"] == 3 * state_bytes


def test_decode_is_inactive_and_state_io_gates_match_source_capability():
    _, compiled = _compiled()
    decode = _costs(compiled, _geometry(512, phase="decode"))
    assert decode["gather_recurrent_state"].duration_ms == 0
    assert decode["scatter_recurrent_state"].duration_ms == 0
    assert decode["gather_recurrent_state"].fidelity == "inactive"
    assert decode["scatter_recurrent_state"].fidelity == "inactive"

    # R31 image + A5 + gdn_decode-only is still a confirmed PF FLA source.
    built, _ = _compiled(source_lowerings=("gdn_decode",))
    for name in STATE_NAMES:
        extras = dict(built.signature(name).extras)
        assert extras["state_io_condition"] == "confirmed_r31_fla"
        assert extras["gather_branch" if "gather" in name
                      else "scatter_branch"] == "fla_gdn_prefill"

    # No R31 identity and no declared gdn_prefill: conditional, no native claim.
    built, _ = _compiled(image=OTHER_IMAGE,
                         source_lowerings=("gdn_decode",))
    for name in STATE_NAMES:
        extras = dict(built.signature(name).extras)
        assert extras["state_io_condition"] == "conditional_hardware_fla"
        assert extras["gather_branch" if "gather" in name
                      else "scatter_branch"] == "conditional"

    # An explicitly declared prefill lowering is confirmed on another image.
    built, _ = _compiled(image=OTHER_IMAGE,
                         source_lowerings=("gdn_prefill",))
    for name in STATE_NAMES:
        extras = dict(built.signature(name).extras)
        assert extras["state_io_condition"] == "declared_gdn_prefill"
        assert extras["gather_branch" if "gather" in name
                      else "scatter_branch"] == "fla_gdn_prefill"

    for kwargs in (
            {"pcp": 2},
            {"state_dtype": "bf16"},
            {"arch": "310P"},
            {"image": R31_IMAGE, "arch": "310P"}):
        built, _ = _compiled(**kwargs)
        names = {operator.name for operator in built.template.operators}
        assert not (set(STATE_NAMES) & names), kwargs

def _bind(spec, operator):
    surface = native_calls._Surface(
        op_id="A3:test:" + operator.name, family=operator.family,
        recipe=operator.name, phase="prefill", source_boundary="kernel",
        dtype=spec.typed.get("dtype"), layout=spec.typed.get("layout"))
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return bind_native_call(declared)


def test_genuine_r31_tp2_24_head_gather_binds_at_adapter_level():
    _, compiled = _compiled(tp=2, value_heads=48)
    gather = next(operator for operator in compiled.operators
                  if operator.name == "gather_recurrent_state")
    spec = gather_binding_spec(gather, phase="prefill")
    assert spec is not None
    binding = _bind(spec, gather)
    env = gather_dynamic_call(
        binding, gather, _geometry(0),
        validate=validate_native_occurrence)
    assert env is not None
    prepare = env["prepare"]
    assert env["actual"]["shapes"] == {
        "state": [None, 24, 128, 128],
        "indices": [1],
        "has_initial_state": [1],
        "output": [1, 24, 128, 128],
    }
    assert prepare["arguments"]["row_size"] == 24 * 128 * 128
    assert prepare["grid"] == [384, 1]
    assert prepare["match"]["source_sha256"].startswith("03debd38")
    assert prepare["match"]["branch"] == "fla_gdn_prefill"
    assert prepare["partial"] is True
    assert gather_binding_spec(gather, phase="decode") is None
