from mindie_sim.prediction.moe_callers import project_expert_receipt
from mindie_sim.prediction.subgraphs.state import MoeRoutingState

"""Source-adapter contracts for the R64 W4A4 grouped gate/up composite."""
from types import SimpleNamespace

from mindie_sim.prediction.native_binding import bind_native_call, declare_native_call
from mindie_sim.prediction.piecewise import (
    native_grouped_matmul_swiglu_quant_v2 as adapter)
from mindie_sim.prediction.piecewise import native_calls

K, N, E, LOCAL, TOP_K = 6144, 4096, 2048, 16, 8
IMAGE = "sha256:" + adapter.R64_IMAGE_SHA
QUANT = {
    "act_format": "mxfp4", "output_format": "mxfp4",
    "weight_format": "mxfp4", "scale_group": 32,
    "linear_method": "mx_fp4", "weight_scale_format": "fp8_e8m0",
}
EXTRAS = {
    "E": 256, "E_local": LOCAL, "k": TOP_K, "latent": K,
    "expert_width": E, "fused_activation_quant": 1,
    "activation_typed_outputs": 2,
    "activation_source_operation": "GroupedMatmulSwigluQuantV2",
}


class _Theta:
    def __init__(self):
        self.quant = dict(QUANT)
        self.extras = dict(EXTRAS)
        self.matrix = {"K": K, "N": N}

    def find(self, key, default=None):
        for source in (self.extras, self.quant, self.matrix):
            if key in source:
                return source[key]
        return default


def _compiled(*, kind="F3", name="expert_gate_up", image=IMAGE,
              revision="r64", extras=None, meta=None):
    theta = _Theta()
    if extras:
        theta.extras.update(extras)
    operator = SimpleNamespace(name=name, family="grouped_gemm", theta=theta)
    feature_metadata = (("image_id", image),
                        ("mx_quant_source_revision", revision))
    if meta:
        feature_metadata += tuple(meta.items())
    return SimpleNamespace(
        kind=kind, layer=3, operator=operator,
        feature_metadata=feature_metadata,
        program=SimpleNamespace(binder=None))


def _binding(compiled):
    spec = adapter.binding_spec(compiled)
    assert spec is not None
    surface = native_calls._Surface(
        op_id="F3:grouped_gemm:expert_gate_up", family="grouped_gemm",
        recipe="gate_up", phase="", source_boundary="call",
        dtype=spec.typed["dtype"], layout=spec.typed["layout"],
        metadata=tuple(sorted(spec.typed["attrs"].items())))
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved)
    return bind_native_call(declared)


def _geometry(counts=None):
    return SimpleNamespace(
        phase_name="decode", tokens=2, effective_tokens=2,
        routing_for=(lambda layer: counts) if counts is not None else
        (lambda layer: None))


def _dynamic(compiled, geometry=None):
    return adapter.dynamic_call(
        _binding(compiled), compiled, geometry or _geometry(None),
        validate=native_calls.validate_native_occurrence)


_DEFAULT_IDS = object()


def _scope_meta(*, ids=_DEFAULT_IDS, rows=32, capacity=64, rank=2,
                group_type=1, scope="grouped_gate_up", input_group_list=None):
    meta = {
        "moe_rank_index": rank,
        "moe_received_rows": {3: rows},
        "moe_capacity_rows": {3: capacity},
        "moe_group_list_type": group_type,
        "moe_route_scope": scope,
    }
    if ids is not None:
        meta["moe_local_expert_ids"] = {
            3: tuple(range(LOCAL)) if ids is _DEFAULT_IDS else ids}
    if input_group_list is not None:
        meta["moe_input_group_list"] = {3: tuple(input_group_list)}
    return meta


def _receipt(compiled, geometry=None, *, state=None):
    return project_expert_receipt(
        compiled, geometry or _geometry(None), state=state,
        activation_dtype="fp4_e2m1", scale_dtype="fp8_e8m0",
        expected_scope="grouped_gate_up")


def test_only_verified_r64_w4a4_grouped_gate_up_is_admitted():
    assert adapter.SOURCE_PATH.endswith("w4a4/w4a4_mxfp4.py")
    assert adapter.SOURCE_SHA.startswith("cbf45467019febc")
    assert adapter.SOURCE_LINES == [383, 408]
    assert adapter.SOURCE_CALL_LINES == [394, 408]
    assert adapter.binding_spec(_compiled()) is not None
    assert adapter.binding_spec(_compiled(name="expert_down")) is None
    assert adapter.binding_spec(_compiled(image="sha256:" + "35" * 32,
                                          revision="unpinned")) is None
    compiled = _compiled()
    compiled.operator.theta.quant = {**QUANT, "weight_format": "fp8_e4m3"}
    assert adapter.binding_spec(compiled) is None


def test_absent_geometry_and_state_is_symbolic_none():
    prepare = _dynamic(_compiled())["prepare"]
    assert prepare["group_list"]["values"] is None
    assert prepare["capacity"]["rows"] is None
    assert prepare["capacity"]["valid_rows"] is None
    assert prepare["descriptors"]["output"]["shape"] == [None, E]
    assert prepare["complete"] is False
    assert prepare["shape_contract_complete"] is False
    assert prepare["partial"] is True
    assert {"received_rows", "group_list", "local_expert_ids",
            "upper_bound"}.issubset(prepare["partial_fields"])


def test_uniform_geometry_counts_without_local_ids_stay_symbolic():
    prepare = _dynamic(_compiled(meta=_scope_meta(ids=None)),
                       _geometry(tuple(1 for _ in range(LOCAL))))["prepare"]
    assert prepare["group_list"]["values"] is None
    assert prepare["capacity"]["valid_rows"] is None
    assert prepare["capacity"]["rows"] == 32
    assert prepare["shape_contract_complete"] is False
    assert "geometry_local_counts_without_local_expert_ids" in \
        prepare["group_list"]["basis"]


def test_shared_receipt_is_identical_for_gmm1_and_gmm2_inputs():
    compiled = _compiled(meta=_scope_meta())
    geometry = _geometry(tuple(1 for _ in range(LOCAL)))
    receipt = _receipt(compiled, geometry)
    assert receipt["received_rows"] == 32
    assert receipt["capacity_rows"] == 64
    assert receipt["valid_rows"] == 16
    assert receipt["group_list"] == list(range(1, LOCAL + 1))
    assert receipt["local_expert_ids"] == list(range(LOCAL))
    assert receipt["activation_dtype"] == "fp4_e2m1"
    assert receipt["scale_dtype"] == "fp8_e8m0"
    assert receipt["shape_complete"] is True

    prepare = _dynamic(compiled, geometry)["prepare"]
    assert prepare["receipt"]["received_rows"] == receipt["received_rows"]
    assert prepare["receipt"]["group_list"] == receipt["group_list"]
    assert prepare["capacity"]["upper_bound"] == receipt["capacity_rows"]
    assert prepare["shape_contract_complete"] is True
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["partial_fields"] == [
        "sixth_float_input", "compiled_kernel_identity"]

    # GMM2R2 consumes the same public dictionary; no second slice.
    gmm2_input = project_expert_receipt(
        compiled, geometry, activation_dtype="fp4_e2m1",
        scale_dtype="fp8_e8m0", expected_scope="grouped_gate_up")
    assert gmm2_input == receipt


def test_global_counts_need_explicit_current_rank_ids():
    counts = tuple(range(256))
    ids = tuple(range(0, 32, 2))
    compiled = _compiled(meta=_scope_meta(ids=ids, rows=256, capacity=256))
    geometry = _geometry(counts)
    receipt = _receipt(compiled, geometry)
    assert receipt["group_list"][-1] == sum(ids)
    assert receipt["valid_rows"] == sum(ids)
    assert receipt["local_expert_ids"] == list(ids)
    assert receipt["shape_complete"] is True

    missing = _compiled(meta=_scope_meta(ids=None, rows=256, capacity=256))
    missing_prepare = _dynamic(missing, geometry)["prepare"]
    assert missing_prepare["group_list"]["values"] is None
    assert missing_prepare["capacity"]["valid_rows"] is None
    assert missing_prepare["capacity"]["rows"] == 256
    assert missing_prepare["receipt"]["basis"] == \
        "global_counts_without_local_expert_ids"
    assert missing_prepare["shape_contract_complete"] is False


def test_state_routing_receipt_is_used_when_present():
    state = MoeRoutingState((1,) * LOCAL, 8, 2, tuple(range(LOCAL)))
    compiled = _compiled(meta=_scope_meta(ids=tuple(range(LOCAL))))
    compiled.state = SimpleNamespace(routing_state=lambda layer: state)
    geometry = _geometry(None)
    receipt = _receipt(compiled, geometry, state=compiled.state)
    assert receipt["basis"] == "state_routing"
    assert receipt["group_list"] == list(range(1, LOCAL + 1))
    assert receipt["received_rows"] == 32
    assert receipt["shape_complete"] is True
    prepare = _dynamic(compiled, geometry)["prepare"]
    assert prepare["group_list"]["basis"] == "state_routing"
    assert prepare["capacity"]["valid_rows"] == 16


def test_state_geometry_disagreement_stays_symbolic():
    state = MoeRoutingState((1,) * LOCAL, 8, 2, tuple(range(LOCAL)))
    compiled = _compiled(meta=_scope_meta())
    compiled.state = SimpleNamespace(routing_state=lambda layer: state)
    prepare = _dynamic(compiled, _geometry(tuple(2 for _ in range(LOCAL))))["prepare"]
    assert prepare["group_list"]["values"] is None
    assert prepare["capacity"]["valid_rows"] is None
    assert "state_geometry_counts_disagree" in prepare["receipt"]["unresolved"]
    assert prepare["shape_contract_complete"] is False


def test_received_rows_below_valid_and_bad_capacity_are_conflicts():
    compiled = _compiled(meta=_scope_meta(rows=2, capacity=64))
    prepare = _dynamic(compiled, _geometry(tuple(1 for _ in range(LOCAL))))["prepare"]
    assert prepare["capacity"]["rows"] is None
    assert prepare["capacity"]["valid_rows"] == 16
    assert "received_rows_below_valid_rows" in prepare["receipt"]["unresolved"]
    assert prepare["shape_contract_complete"] is False

    compiled = _compiled(meta=_scope_meta(rows=32, capacity=2))
    prepare = _dynamic(compiled, _geometry(tuple(1 for _ in range(LOCAL))))["prepare"]
    assert prepare["capacity"]["rows"] == 32
    assert prepare["capacity"]["upper_bound"] is None
    assert "capacity_below_received_or_valid_rows" in prepare["receipt"]["unresolved"]
    assert "upper_bound" in prepare["partial_fields"]


def test_cumulative_source_selection_is_retained_for_gmm2():
    cumulative = tuple(range(1, LOCAL + 1))
    compiled = _compiled(meta=_scope_meta(
        rows=16, capacity=16, group_type=0,
        input_group_list=cumulative))
    prepare = _dynamic(compiled, _geometry(tuple(1 for _ in range(LOCAL))))["prepare"]
    assert prepare["group_list"]["gmm1_group_list"] == list(cumulative)
    assert prepare["group_list"]["gmm2_group_list"] == list(cumulative)
    assert prepare["group_list"]["gmm2_group_list_type"] == 0
    assert prepare["capacity"]["valid_rows"] == 16
    assert prepare["capacity"]["rows"] == 16


def test_private_group_list_metadata_is_not_required_or_consumed():
    compiled = _compiled(meta={
        "moe_group_list": {3: tuple(1 for _ in range(LOCAL))},
        "moe_group_list_type": 1,
        "moe_received_rows": {3: 16},
    })
    prepare = _dynamic(compiled, _geometry(None))["prepare"]
    assert prepare["group_list"]["values"] is None
    assert prepare["capacity"]["rows"] == 16
    assert prepare["capacity"]["valid_rows"] is None
    assert "no_private_route_ledger_schema" in prepare["non_claims"]


def test_type1_dual_views_type0_and_unknown_source_type():
    base = {
        "moe_local_expert_ids": {3: (0, 1, 2)},
        "moe_rank_index": 0,
        "moe_received_rows": {3: 5},
        "moe_capacity_rows": {3: 5},
        "moe_route_scope": "grouped_gate_up",
    }
    compiled = _compiled(extras={"E_local": 3, "E": 3},
                         meta={**base, "moe_group_list_type": 1})
    geometry = _geometry((2, 0, 3))
    receipt = _receipt(compiled, geometry)
    assert receipt["canonical_counts"] == [2, 0, 3]
    assert receipt["input_group_list"] == [2, 0, 3]
    assert receipt["input_group_list_type"] == 1
    assert receipt["gmm1_group_list"] == [2, 2, 5]
    assert receipt["gmm1_group_list_type"] == 0
    assert receipt["gmm2_group_list"] == [2, 0, 3]
    assert receipt["gmm2_group_list_type"] == 1
    assert receipt["valid_rows"] == 5
    assert receipt["gmm1_shape_complete"] is True
    assert receipt["gmm2_shape_complete"] is True
    assert receipt["shape_complete"] is True

    prepare = _dynamic(compiled, geometry)["prepare"]
    assert prepare["group_list"]["values"] == [2, 2, 5]
    assert prepare["group_list"]["gmm1_group_list_type"] == 0
    assert prepare["group_list"]["gmm2_group_list"] == [2, 0, 3]
    assert prepare["group_list"]["gmm2_group_list_type"] == 1
    assert prepare["capacity"]["rows"] == 5
    assert prepare["capacity"]["valid_rows"] == 5
    assert prepare["shape_contract_complete"] is True
    assert prepare["gmm2_shape_complete"] is True
    assert prepare["pair_shape_complete"] is True
    assert prepare["complete"] is False

    # Type 0 source: original cumulative vector is retained for GMM2.
    compiled0 = _compiled(extras={"E_local": 3, "E": 3}, meta={
        **base, "moe_group_list_type": 0,
        "moe_input_group_list": {3: (2, 2, 5)}})
    receipt0 = _receipt(compiled0, geometry)
    assert receipt0["input_group_list"] == [2, 2, 5]
    assert receipt0["input_group_list_type"] == 0
    assert receipt0["gmm1_group_list"] == [2, 2, 5]
    assert receipt0["gmm2_group_list"] == [2, 2, 5]
    assert receipt0["gmm2_group_list_type"] == 0
    assert receipt0["shape_complete"] is True

    # Unknown source type: GMM1 conversion is known, GMM2 selection is not.
    compiled_unknown = _compiled(extras={"E_local": 3, "E": 3}, meta=base)
    unknown = _receipt(compiled_unknown, geometry)
    assert unknown["gmm1_group_list"] == [2, 2, 5]
    assert unknown["gmm1_group_list_type"] == 0
    assert unknown["gmm2_group_list"] is None
    assert unknown["gmm2_group_list_type"] is None
    assert unknown["gmm1_shape_complete"] is True
    assert unknown["gmm2_shape_complete"] is False
    assert unknown["shape_complete"] is False
    assert "input_group_list_type_not_declared" in unknown["unresolved"]
