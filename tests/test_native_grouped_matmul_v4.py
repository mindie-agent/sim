"""Source-adapter contracts for the R64 W4A4 routed-expert down call."""
from types import SimpleNamespace

from mindie_sim.prediction.native_binding import bind_native_call, declare_native_call
from mindie_sim.prediction.piecewise import native_calls
from mindie_sim.prediction.piecewise import native_grouped_matmul_v4 as adapter
from mindie_sim.prediction.subgraphs.state import MoeRoutingState, StateView

K, N, E, LOCAL, TOP_K = 2048, 6144, 256, 16, 8
IMAGE = "sha256:" + adapter.R64_IMAGE_SHA
QUANT = {
    "act_format": "mxfp4", "output_format": "bf16",
    "weight_format": "mxfp4", "scale_group": 32,
    "linear_method": "mx_fp4", "weight_scale_format": "fp8_e8m0",
    "prequant_format": "bf16",
}
EXTRAS = {
    "E": E, "E_local": LOCAL, "k": TOP_K, "latent": N, "expert_width": K,
    "activation_source": "grouped_gate_up_activation_quant",
    "activation_payload_row_bytes": 1024.0, "activation_scale_bytes": 64.0,
    "moe_comm": (("world_size", 16), ("top_k", TOP_K),
                 ("max_num_reqs", 64), ("max_num_batched_tokens", 8192),
                 ("enable_fused_mc2", 1), ("enable_prefill_mc2", 0)),
}
COUNTS = (2, 0, 3) + (0,) * (LOCAL - 3)
IDS = (3, 7) + (0,) * (LOCAL - 2)


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


def _compiled(*, meta_extra=(), state=None, kind="F3", name="expert_down",
              image=IMAGE, revision="r64"):
    operator = SimpleNamespace(name=name, family="grouped_gemm",
                               theta=_Theta())
    meta = [("image_id", image), ("mx_quant_source_revision", revision)]
    meta.extend(meta_extra)
    return SimpleNamespace(
        kind=kind, layer=0, operator=operator, feature_metadata=tuple(meta),
        state=state, program=SimpleNamespace(binder=None))


def _routing_state():
    return MoeRoutingState(COUNTS, 5, 1, local_expert_ids=IDS)


def _explicit_compiled(*, rank=0, received_rank=0, capacity_rank=0):
    return _compiled(
        state=StateView(routing=((0, _routing_state()),)),
        meta_extra=(
            ("moe_rank_index", rank),
            ("moe_received_rows", {received_rank: 5}),
            ("moe_capacity_rows", {capacity_rank: 128}),
            ("moe_group_list_type", 1),
        ))


def _binding(compiled):
    spec = adapter.binding_spec(compiled)
    assert spec is not None
    surface = native_calls._Surface(
        op_id="F3:grouped_gemm:expert_down", family="grouped_gemm",
        recipe="down", phase="", source_boundary="kernel",
        dtype=spec.typed["dtype"], layout=spec.typed["layout"],
        metadata=tuple(sorted(spec.typed["attrs"].items())))
    return bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved))


def _geometry(*, phase="prefill", tokens=1024):
    return SimpleNamespace(
        phase_name=phase, tokens=tokens, effective_tokens=tokens,
        routing_for=lambda layer: None)


def _dynamic(compiled, geometry):
    return adapter.dynamic_call(
        _binding(compiled), compiled, geometry,
        validate=native_calls.validate_native_occurrence)


def test_only_r64_w4a4_routed_down_is_admitted():
    compiled = _compiled()
    spec = adapter.binding_spec(compiled)
    assert spec is not None
    assert spec.api["entry"] == adapter.ENTRY
    assert spec.source["sha256"] == adapter.SOURCE_SHA
    assert spec.source["symbol"] == adapter.SOURCE_SYMBOL
    assert spec.typed["shapes"]["weight"] == [LOCAL, N, K]
    assert spec.typed["shapes"]["scale"] == [LOCAL, N, 32, 2]
    assert spec.typed["attrs"]["output_format"] == "bf16"
    assert spec.typed["attrs"]["x_dtype"] == "fp4_e2m1"
    assert adapter.binding_spec(_compiled(name="expert_gate_up")) is None
    assert adapter.binding_spec(_compiled(kind="F1")) is None
    assert adapter.binding_spec(
        _compiled(image="sha256:" + "35" * 32, revision="unpinned")) is None
    wrong = _compiled()
    wrong.operator.theta.quant = {**QUANT, "weight_format": "fp8_e4m3"}
    assert adapter.binding_spec(wrong) is None
    wrong = _compiled()
    wrong.operator.theta.quant = {**QUANT, "output_format": "mxfp4"}
    assert adapter.binding_spec(wrong) is None


def test_without_receipt_rows_stay_symbolic_and_partial():
    envelope = _dynamic(_compiled(), _geometry())
    prepare = envelope["prepare"]
    assert prepare["shapes"]["x"] == [None, K]
    assert prepare["shapes"]["output"] == [None, N]
    assert prepare["shapes"]["per_token_scale"] == [None, 32, 2]
    assert prepare["shapes"]["weight"] == [LOCAL, N, K]
    assert prepare["descriptors"]["x"]["allocation_shape"] is None
    assert prepare["capacity"]["rows"] is None
    assert prepare["capacity"]["input_rows"] is None
    assert prepare["group_list"]["values"] is None
    assert prepare["expected_cost_view"]["used_as_native_shape"] is False
    assert prepare["complete"] is False
    assert prepare["match"]["shape_contract_complete"] is False
    assert "received_rows_not_declared" in prepare["partial_reasons"]
    assert "source_group_list_counts_not_declared" in prepare["partial_reasons"]
    assert "local_expert_ids_not_declared" in prepare["partial_reasons"]
    assert "upper_bound_not_declared" in prepare["partial_reasons"]
    assert "no_uniform_moe_counts_as_native_rows" in prepare["non_claims"]
    assert "no_private_route_ledger_schema" in prepare["non_claims"]


def test_true_receipt_preserves_real_rank_ids_and_capacity():
    envelope = _dynamic(_explicit_compiled(), _geometry())
    prepare = envelope["prepare"]
    assert prepare["shapes"]["x"] == [5, K]
    assert prepare["shapes"]["output"] == [5, N]
    assert prepare["shapes"]["per_token_scale"] == [5, 32, 2]
    assert prepare["capacity"]["rows"] == 128
    assert prepare["capacity"]["input_rows"] == 5
    assert prepare["capacity"]["valid_rows"] == 5
    assert prepare["group_list"]["values"] == [2, 0, 3] + [0] * (LOCAL - 3)
    assert prepare["group_list"]["layout"] == "count"
    assert prepare["group_list"]["group_list_type"] == 1
    assert prepare["group_list"]["canonical_cumulative"] == [2, 2, 5] + [5] * (LOCAL - 3)
    assert prepare["group_list"]["canonical_group_list_type"] == 0
    assert prepare["group_list"]["local_expert_ids"] == list(IDS)
    assert prepare["group_list"]["counts"] == list(COUNTS)
    assert prepare["arguments"]["group_list_type"] == 1
    assert prepare["receipt"]["rank"] == 0
    assert prepare["receipt"]["group_list"] == [2, 2, 5] + [5] * (LOCAL - 3)
    assert prepare["receipt"]["shape_complete"] is True
    assert prepare["complete"] is False
    assert "received_rows_not_declared" not in prepare["partial_reasons"]
    assert "upper_bound_not_declared" not in prepare["partial_reasons"]
    assert "source_group_list_counts_not_declared" not in prepare["partial_reasons"]
    assert "no_gmm2_normalization_to_cumulative" in prepare["non_claims"]


def test_mc2_without_declared_type_keeps_encoding_unknown():
    compiled = _compiled(
        state=StateView(routing=((0, _routing_state()),)),
        meta_extra=(
            ("moe_rank_index", 0),
            ("moe_received_rows", {0: 5}),
            ("moe_capacity_rows", {0: 128}),
        ))
    envelope = _dynamic(compiled, _geometry(phase="decode", tokens=16))
    prepare = envelope["prepare"]
    assert prepare["shapes"]["x"] == [5, K]
    assert prepare["shapes"]["output"] == [5, N]
    assert prepare["capacity"]["rows"] == 128
    assert prepare["capacity"]["input_rows"] == 5
    assert prepare["group_list"]["values"] is None
    assert prepare["group_list"]["group_list_type"] is None
    assert prepare["group_list"]["source_encoding"] == \
        "mc2_producer_group_list_type_unknown"
    assert prepare["group_list"]["mc2_producer_unknown"] is True
    assert prepare["group_list"]["canonical_cumulative"] == \
        [2, 2, 5] + [5] * (LOCAL - 3)
    assert prepare["group_list"]["canonical_group_list_type"] == 0
    assert prepare["arguments"]["group_list_type"] is None
    assert prepare["match"]["mc2_group_list_type_unknown"] is True
    assert "mc2_incoming_group_list_type_not_declared" in \
        prepare["partial_reasons"]
    assert "original_group_list_type_not_declared" in \
        prepare["partial_reasons"]


def test_mismatched_actual_rank_is_symbolic(monkeypatch):
    receipt = {
        "layer": 0, "rank": 0, "ep": 16, "received_rows": 2,
        "input_rows": 2, "valid_rows": 2,
        "group_list": [2, 2, 5] + [5] * (LOCAL - 3),
        "group_list_layout": "cumulative", "group_list_type": 0,
        "local_expert_ids": list(IDS), "expert_rows": list(COUNTS),
        "capacity_rows": 128, "capacity_basis": "state_routing",
        "activation_dtype": "fp4_e2m1", "scale_dtype": "fp8_e8m0",
        "basis": "state_routing", "symbolic": False,
        "ownership_known": True, "shape_complete": True, "complete": False,
        "unresolved": [],
    }
    monkeypatch.setattr(adapter, "_expert_receipt",
                        lambda compiled, geometry: receipt)
    compiled = _compiled(meta_extra=(("moe_rank_index", 1),))
    envelope = _dynamic(compiled, _geometry())
    prepare = envelope["prepare"]
    assert prepare["shapes"]["x"] == [None, K]
    assert prepare["group_list"]["values"] is None
    assert prepare["capacity"]["rows"] is None
    assert "shared_expert_receipt_identity_mismatch" in prepare["partial_reasons"]
    assert "receipt_rank_mismatch" in prepare["partial_reasons"]
