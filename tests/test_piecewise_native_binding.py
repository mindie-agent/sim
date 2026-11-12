"""Native-call binding on the ordinary bound-operator piecewise path.

[Role] Prove that the normal subgraph prediction path carries one immutable
  selected implementation identity and one evaluated occurrence, including a
  real GDN 1:N call and a fused MLA prolog call.
[Invariants] Occurrence layer/index never enters call_id; a fused call is
  charged once; wrong dtype/layout/shape/attr never validate; unbound rows
  stay in the coverage denominator.
[Timeouts & Locks] Pure CPU arithmetic on repository cases; no device or I/O.
[Side-Effect Policy] Read-only test code.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.composition import evaluate_serial_split
from mindie_sim.prediction.piecewise.native_calls import (
    bind_fusion_call, forecast_native_coverage, validate_native_occurrence,
)
from mindie_sim.prediction.piecewise.native_invocation import (
    NativeInvocationUnavailable, invoke_qbmm, prepared_qbmm_plan,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, instantiate_model
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"
GLM = ROOT / "examples/cases/pooled-score-migration-20261030/a5_glm53flash_tp8_c4_p8k.json"
QWEN_GQA = ROOT / "examples/cases/qwen3.8-27b.json"
R64_METADATA = (
    ("image_id", "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"),
    ("mx_quant_source_revision", "r64"),
)
R35_METADATA = (
    ("image_id", "sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06"),
    ("mx_quant_source_revision", "unpinned"),
)
UNKNOWN_METADATA = (
    ("image_id", "sha256:" + "f" * 64),
    ("mx_quant_source_revision", "unpinned"),
)
DOMAIN = AnalysisDomain(max_tokens=10 ** 9, max_moe_rows=10 ** 9,
                        max_query_tiles=10 ** 9, max_kv_tiles=10 ** 9,
                        max_index_rows=10 ** 12, max_work=10 ** 15,
                        max_q=10 ** 9, max_s_kv=10 ** 9, max_seq=10 ** 9,
                        max_batch=10 ** 6)


def _case(path: Path, **features):
    case = parse_case(json.loads(path.read_text()))
    return case, FeatureConfig.from_model(case.model, **features)


def _compiler(case, cores=64):
    return OperatorPiecewiseCompiler(
        PipeSupply.from_hardware(case.machine, dtype="bf16"),
        ImplementationPolicy(cores=cores), domain=DOMAIN)


def _prefill(tokens=1024, batch=1):
    return WorkloadGeometry((RequestGeometry(tokens, tokens, phase="prefill"),),
                            phase="prefill", cohort_batch=batch, local_batch=batch)


def test_gdn_chunk_call_is_one_identity_with_occurrence_and_runtime_children():
    case, features = _case(QWEN, tp=2, dp=4, ep=1, pp=1, cores=64,
                           token_tile=16, kv_tile=16, moe_row_tile=4,
                           split_target=1000, max_splits=4)
    instantiated = instantiate_model(case.model, features)
    gdn = [item for layer in instantiated.layers for item in layer.subgraphs
           if item.kind == "A3"]
    assert len(gdn) >= 2
    compiler, geometry = _compiler(case), _prefill(4096)
    evaluated = []
    for instance in gdn[:2]:
        costs = {cost.name: cost for cost in compiler.compile(instance).evaluate(geometry).operators}
        evaluated.append(costs)
    first, second = evaluated
    assert first["gated_delta"].native_call is not None
    assert first["prepare"].native_call is not None
    assert first["gated_delta"].native_call["call_id"] == first["prepare"].native_call["call_id"]
    assert first["gated_delta"].native_call["call_id"] == second["gated_delta"].native_call["call_id"]
    assert first["gated_delta"].native_occurrence["layer"] != second["gated_delta"].native_occurrence["layer"]
    assert first["gated_delta"].native_occurrence["charge"] == "call"
    assert first["prepare"].native_occurrence["charge"] == "member"
    assert first["prepare"].native_call["timing_owner"]["op_id"] == "A3:recurrent:gated_delta"
    assert first["gated_delta"].native_call["timing_owner"]["op_id"] == "A3:recurrent:gated_delta"
    call = first["gated_delta"].native_call
    assert call["api"]["aclnn_symbol"] == "aclnnChunkGatedDeltaRuleFwd"
    assert call["evidence"]["aclnn_symbol"] == "source_declared"
    assert "aclnn_symbol_unresolved" not in call["unresolved"]
    assert [kernel["name"] for kernel in call["kernels"]] == [
        "Cast", "Transpose", "Prepare32", "H24", "O32"]
    assert next(kernel for kernel in call["kernels"]
                if kernel["name"] == "Transpose")["count"] == 3
    assert set(call["members"]["op_ids"]) == {"prepare", "gated_delta"}
    typed = first["gated_delta"].native_occurrence["typed"]["shapes"]
    assert call["typed"]["shapes"]["q"] == [8, 128]
    assert typed["q"] == [1, 4096, 8, 128]
    assert typed["v"] == [1, 4096, 24, 128]
    assert typed["g"] == [1, 4096, 24]
    assert typed["o"] == [1, 4096, 24, 128]
    assert typed["final_state"] == [1, 24, 128, 128]
    prepare = first["gated_delta"].native_occurrence["prepare"]
    assert prepare["schema"] == "mindie-sim/gdn-native-prepare/1"
    assert prepare["arguments"]["cu_seqlens"] == [0, 4096]
    assert prepare["arguments"]["chunk_size"] == 64
    assert prepare["arguments"]["chunks"] == 64
    assert prepare["arguments"]["chunk_indices"][:2] == [0, 0]
    assert prepare["arguments"]["chunk_indices"][-2:] == [0, 63]
    assert prepare["named_arguments"]["layout"] == "BSND"
    assert prepare["named_arguments"]["scale"] == 128 ** -0.5
    assert prepare["descriptors"]["initial_state"]["role"] == "input"
    assert prepare["descriptors"]["o"]["role"] == "output"
    assert typed["g_cumsum"] == [1, 4096, 24]
    assert typed["A"] == [1, 24, 4096, 64]
    assert typed["q_hat"] == [1, 4096, 8, 128]
    assert typed["k_hat"] == [1, 4096, 8, 128]
    assert typed["q_rstd"] == [1, 8, 4096]
    assert typed["k_rstd"] == [1, 8, 4096]
    assert set(prepare["outputs"]) == {"o", "final_state", "g_cumsum", "A",
                                       "q_hat", "k_hat", "q_rstd", "k_rstd"}
    assert set(prepare["disabled_outputs"]) == {"beta_eff", "h"}
    assert prepare["output_slots"]["A"]["acl_slot"] == "aOutOptional"
    assert sum(item["count"] for item in prepare["physical_presentations"]) == 7
    third = prepare["physical_presentations"][1]["third_transpose"]
    assert third["tensor"] == "g_cumsum"
    assert third["source_storage_shape"] == [1, 24, 4096]
    assert third["target_storage_shape"] == [1, 4096, 24]
    assert third["evidence"] == "GDNCall.device_program.materializations:g_cumsum_export"
    assert prepare["g_cumsum_materialization"]["recipe"] == "g_cumsum_export"
    assert prepare["allocation"] == "not_performed"
    assert first["gated_delta"].native_occurrence["prepare_args"]["shapes"]["q"] == [1, 4096, 8, 128]
    independent = {cost.name: cost for cost in compiler.compile(gdn[0]).evaluate(
        _prefill(2048)).operators}
    independent_call = independent["gated_delta"]
    assert independent_call.native_call["call_id"] == call["call_id"]
    assert independent_call.native_occurrence["typed"]["shapes"]["q"] == [1, 2048, 8, 128]
    assert independent_call.native_occurrence["prepare"]["arguments"]["cu_seqlens"] == [0, 2048]
    batch = WorkloadGeometry(
        (RequestGeometry(4096, 4096, phase="prefill"),
         RequestGeometry(4096, 4096, phase="prefill")),
        phase="prefill", cohort_batch=2, local_batch=2)
    batched = {cost.name: cost for cost in compiler.compile(gdn[0]).evaluate(batch).operators}
    batched_call = batched["gated_delta"]
    assert batched_call.native_call["call_id"] == call["call_id"]
    assert batched_call.native_occurrence["typed"]["shapes"]["q"] == [1, 8192, 8, 128]
    assert batched_call.native_occurrence["prepare"]["arguments"]["cu_seqlens"] == [0, 4096, 8192]
    assert batched_call.native_occurrence["prepare"]["descriptors"]["initial_state"]["shape"] == [2, 24, 128, 128]
    batched_third = batched_call.native_occurrence["prepare"]["physical_presentations"][1]["third_transpose"]
    assert batched_third["source_storage_shape"] == [1, 24, 8192]
    assert batched_third["target_storage_shape"] == [1, 8192, 24]
    decode = WorkloadGeometry((RequestGeometry(1, 1024, phase="decode"),),
                              phase="decode", cohort_batch=1, local_batch=1)
    decoded = {cost.name: cost for cost in compiler.compile(gdn[0]).evaluate(decode).operators}
    assert decoded["gated_delta"].native_call["api"]["entry"] == "RecurrentGatedDeltaRule"
    assert "prepare" not in decoded["gated_delta"].native_occurrence


def test_ordinary_gqa_fia_accepts_one_token_occurrence_shape():
    case, features = _case(QWEN_GQA, tp=4, dp=1, ep=1, pp=1, cores=64)
    instantiated = instantiate_model(case.model, features)
    instance = next(item for layer in instantiated.layers for item in layer.subgraphs
                    if item.kind == "A0")
    geometry = WorkloadGeometry((RequestGeometry(1, 1024, phase="decode"),),
                                phase="decode", cohort_batch=1, local_batch=1)
    cost = next(cost for cost in _compiler(case).compile(instance).evaluate(geometry).operators
                if cost.name == "attention")
    assert cost.native_call["api"]["entry"] == "torch_npu.npu_fused_infer_attention_score"
    assert cost.native_occurrence["typed"]["shapes"]["q"][0] == 1
    assert cost.native_occurrence["typed"]["shapes"]["q"][-2:] == cost.native_call["typed"]["shapes"]["q"]


def test_fused_mla_prolog_has_one_call_identity_and_one_charge():
    case, features = _case(GLM, tp=1, dp=16, ep=16, pp=1, cores=64)
    instantiated = instantiate_model(case.model, features)
    attention = [item for layer in instantiated.layers for item in layer.subgraphs
                 if item.kind == "A1"][0]
    contract = next(item for item in attention.template.fusions if item.name == "mla_prolog")
    from mindie_sim.prediction.piecewise.sfa_prolog import port_spec
    theta = {operator.name: operator.theta for operator in attention.operators}
    heads = dict(theta["mla_attention"].heads)
    q_lora = int(dict(theta["q_down"].matrix)["N"])
    kv_lora = int(heads["rkv"])
    # SFA C8 MX is a rope64 port contract; this case's flash variant declares
    # rope0, so the synthetic contract supplies the selected rope64 ABI.
    rope = max(64, int(heads["dr"]))
    synthetic = replace(attention, template=replace(
        attention.template,
        contract=(("sfa_prolog_v3", port_spec(
            heads=int(heads["q"]), q_lora=q_lora, kv_lora=kv_lora, rope=rope,
            store_row=kv_lora + rope)),)))
    binding = bind_fusion_call(synthetic, contract, ())
    assert binding is not None
    assert binding.api["entry"] == "torch_npu.npu_mla_prolog_v3"
    assert binding.source["path"] == "vllm_ascend/attention/sfa_v1.py"
    assert len(binding.source["sha256"]) == 64
    assert [kernel["name"] for kernel in binding.kernels] == ["MlaPrologV3"]
    assert set(binding.member_op_ids) == set(contract.members)
    assert binding.timing_owner["op_id"] == binding.op_id
    assert "aclnn_symbol_unresolved" in binding.unresolved


def test_generic_fused_prolog_is_explicitly_unbound():
    case, features = _case(GLM, tp=1, dp=16, ep=16, pp=1, cores=64)
    instantiated = instantiate_model(case.model, features)
    attention = [item for layer in instantiated.layers for item in layer.subgraphs
                 if item.kind == "A1"][0]
    contract = next(item for item in attention.template.fusions if item.name == "mla_prolog")
    state = StateView()
    for layer in range(len(case.model.layer_plans)):
        state = state.mark_kv(layer).mark_index(layer).mark_selection(layer)
    result = evaluate_serial_split(_compiler(case).compile(attention), _prefill(),
                                   state, fusion=("mla_prolog",))
    fused = next(item.fused for item in result.fusion_results)
    assert fused.native_call is None
    assert fused.native_occurrence["status"] == "unbound"
    assert "fused_source_not_selected" in fused.native_occurrence["unresolved"]
    assert [cost.name for cost in result.costs].count("fused:mla_prolog") == 1
    assert not {cost.name for cost in result.costs} & set(contract.members)


def test_negative_invocation_rejects_wrong_dtype_layout_shape_and_attr():
    case, features = _case(QWEN, tp=2, dp=4, ep=1, pp=1, cores=64,
                           token_tile=16, kv_tile=16, moe_row_tile=4,
                           split_target=1000, max_splits=4)
    instantiated = instantiate_model(case.model, features)
    instance = next(item for layer in instantiated.layers for item in layer.subgraphs
                    if item.kind == "A3")
    cost = next(cost for cost in _compiler(case).compile(instance).evaluate(_prefill()).operators
                if cost.name == "gated_delta")
    typed = cost.native_call["typed"]
    binding = type("Binding", (), {"typed": typed, "call_id": cost.native_call["call_id"]})()
    good = {"dtype": typed["dtype"], "layout": typed["layout"],
            "shapes": typed["shapes"], "attrs": typed["attrs"]}
    assert validate_native_occurrence(binding, good)["shapes"]
    for key, value in (("dtype", "fp32"), ("layout", "TND")):
        with pytest.raises(ModelingError, match=f"native_call_signature_{key}_mismatch"):
            validate_native_occurrence(binding, {**good, key: value})
    shapes = {**typed["shapes"], "q": [typed["shapes"]["q"][0] + 1,
                                       typed["shapes"]["q"][1]]}
    with pytest.raises(ModelingError, match="native_call_signature_shape_mismatch:q"):
        validate_native_occurrence(binding, {**good, "shapes": shapes})


@dataclass(frozen=True)
class _Row:
    name: str
    family: str
    count: int
    native_call: dict | None
    native_occurrence: dict | None = None


def test_forecast_coverage_counts_actual_occurrences_and_unbound_rows():
    call = {"call_id": "a" * 64, "family": "recurrent", "source_boundary": "call",
            "recipe": "gated_delta", "api": {"namespace": "aclnn",
                                             "entry": "aclnnChunkGatedDeltaRuleFwd",
                                             "aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd",
                                             "candidates": []},
            "source": {"path": "sources.tar.gz", "sha256": "b" * 64},
            "typed": {"signature_hash": "c" * 64}, "kernels": [{"name": "H24"}],
            "members": {"op_ids": ["prepare", "gated_delta"]}, "unresolved": []}
    rows = (_Row("gated_delta", "recurrent", 40, call),
            _Row("residual_add", "elementwise", 40, None))
    coverage = forecast_native_coverage(rows)
    assert coverage["denominator"]["emitted_operators"] == 80
    assert coverage["totals"]["bound_occurrences"] == 40
    assert coverage["totals"]["unresolved_occurrences"] == 40
    assert coverage["totals"]["unique_calls"] == 1
    assert coverage["denominator"]["unrecorded_emitted"] == 40


def test_forecast_coverage_separates_fusion_attribution_and_unresolved_status():
    call = {"call_id": "a" * 64, "family": "recurrent", "source_boundary": "call",
            "recipe": "gated_delta", "api": {"namespace": "aclnn",
                                             "entry": "aclnnChunkGatedDeltaRuleFwd",
                                             "aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd",
                                             "candidates": []},
            "source": {"path": "sources.tar.gz", "sha256": "b" * 64},
            "typed": {"signature_hash": "c" * 64}, "kernels": [{"name": "H24"}],
            "members": {"op_ids": ["prepare", "gated_delta"]}, "unresolved": []}
    unknown = {"status": "unresolved", "source": None,
               "api": {"namespace": "torch", "entry": "torch.example"},
               "typed": {"signature_hash": "d" * 64}, "unresolved": []}
    rows = (_Row("gated_delta", "recurrent", 2, call),
            _Row("residual_add", "norm_residual", 3, None,
                 {"charge": "none", "membership": "consumed_by_selected_fusion",
                  "timing_owner": "fused:add_rmsnorm_quant"}),
            _Row("unknown", "gemm", 4, unknown))
    coverage = forecast_native_coverage(rows)
    assert coverage["denominator"]["physical_operators"] == 6
    assert coverage["denominator"]["attribution_operators"] == 3
    assert coverage["totals"]["bound_occurrences"] == 2
    assert coverage["totals"]["unresolved_occurrences"] == 4
    assert coverage["totals"]["attribution_occurrences"] == 3
    assert coverage["attribution_by_family"] == {"norm_residual": 3}
    assert coverage["attribution_members"][0]["timing_owner"] == "fused:add_rmsnorm_quant"


def test_forecast_coverage_counts_gdn_phases_once():
    call = {"call_id": "e" * 64, "family": "recurrent", "source_boundary": "call",
            "recipe": "gated_delta",
            "api": {"namespace": "aclnn", "entry": "aclnnChunkGatedDeltaRuleFwd",
                    "aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd", "candidates": []},
            "source": {"path": "sources.tar.gz", "sha256": "f" * 64},
            "typed": {"signature_hash": "1" * 64}, "kernels": [{"name": "H24"}],
            "timing_owner": {"op_id": "A3:recurrent:gated_delta", "scope": "call"},
            "unresolved": []}
    owner = _Row("gated_delta", "recurrent", 48, call,
                 {"charge": "call", "call_owner": "A3:recurrent:gated_delta",
                  "timing_owner": {"op_id": "A3:recurrent:gated_delta"}})
    member = _Row("prepare", "recurrent", 48, call,
                  {"charge": "member", "phase": "prefill",
                   "call_owner": "A3:recurrent:gated_delta",
                   "timing_owner": {"op_id": "A3:recurrent:gated_delta"}})
    coverage = forecast_native_coverage((owner, member))
    assert coverage["totals"]["emitted_occurrences"] == 48
    assert coverage["totals"]["bound_occurrences"] == 48
    assert coverage["totals"]["phase_attribution_occurrences"] == 48
    assert coverage["totals"]["total_occurrences"] == 96
    assert coverage["phased_calls"][0]["phases"] == {"prefill": 48}


def test_mxfp8_qbmm_o_projection_has_prepare_arguments_and_independent_m():
    case, features = _case(GLM, tp=1, dp=32, ep=32, pp=1, cores=64)
    instantiated = instantiate_model(case.model, features)
    instance = next(item for layer in instantiated.layers for item in layer.subgraphs
                    if item.kind == "A1")
    compiled = _compiler(case).compile(instance, feature_metadata=R64_METADATA)
    evaluated = {}
    for m in (1024, 2048, 2):
        costs = {cost.name: cost for cost in compiled.evaluate(_prefill(m)).operators}
        evaluated[m] = costs["o_projection"]
    first = evaluated[1024]
    call = first.native_call
    assert call["api"]["entry"] == "torch_npu.npu_quant_matmul"
    assert call["api"]["version"] == "r64/8d422528"
    assert call["source"]["path"] == "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py"
    assert call["source"]["sha256"] == "38de2275cf33ced653b8398e98a07f483489dc8e7353b2b3c238a5ff00782f79"
    assert first.native_occurrence["charge"] == "call"
    assert call["typed"]["shapes"]["weight"] == [16384, 4096]
    assert call["typed"]["shapes"]["x"] == [16384]
    assert call["typed"]["shapes"]["output"] == [4096]
    assert call["typed"]["shapes"]["weight_scale"] == [256, 4096, 2]
    prepared = first.native_occurrence["prepare"]
    assert prepared["implementation"] == "W8A8_MXFP8"
    assert prepared["stage"] == "interface_prepared"
    assert prepared["shapes"]["x"] == [1024, 16384]
    assert prepared["shapes"]["output"] == [1024, 4096]
    assert prepared["shapes"]["pertoken_scale"] == [1024, 256, 2]
    assert prepared["descriptors"]["weight_nz"]["shape"] == [128, 1024, 16, 32]
    assert prepared["descriptors"]["weight_nz"]["layout"] == "FRACTAL_NZ"
    assert prepared["descriptors"]["weight_scale"]["dtype"] == "uint8"
    assert prepared["descriptors"]["weight_scale"]["storage_dtype"] == "uint8"
    assert prepared["descriptors"]["weight_scale"]["interpret_dtype"] == "fp8_e8m0"
    assert prepared["descriptors"]["weight_scale"]["allocation_shape"] == [4096, 512]
    assert prepared["descriptors"]["weight_scale"]["device_shape"] == [256, 4096, 2]
    assert prepared["scale_storage"]["weight_scale"]["allocation_shape"] == [4096, 512]
    assert prepared["scale_storage"]["weight_scale"]["allocation_dtype"] == "uint8"
    assert prepared["weight_storage"]["allocation_shape"] == [4096, 16384]
    assert prepared["weight_storage"]["device_shape"] == [128, 1024, 16, 32]
    assert prepared["deployment"]["image_id"].startswith("8d422528")
    assert prepared["match"]["k"] == 16384 and prepared["match"]["n"] == 4096
    assert prepared["arguments"]["group_sizes"] == [1, 1, 32]
    assert prepared["arguments"]["scale_dtype"] == "fp8_e8m0"
    assert prepared["arguments"]["bias"] is None
    assert prepared["arguments"]["offset"] is None
    assert prepared["optional_slots"]["bias"] == "not_present"
    assert prepared["optional_slots"]["offset"] == "not_passed"
    assert prepared["argument_order"][:3] == ["x1", "x2", "scale"]
    assert prepared["positional_arguments"] == {"x1": "x", "x2": "weight", "scale": "weight_scale"}
    assert prepared["keywords"]["offset"] is None
    assert prepared["keywords"]["x1_dtype"] is None and prepared["keywords"]["x2_dtype"] is None
    assert prepared["keywords"]["y_scale"] is None
    assert prepared["producer_inputs"]["x_original"]["dtype"] == "bf16"
    assert prepared["binary_selection"]["status"] == "uncertain"
    for m, expected in ((2048, 2048), (2, 2)):
        occurrence = evaluated[m].native_occurrence
        assert evaluated[m].native_call["call_id"] == call["call_id"]
        assert occurrence["typed"]["shapes"]["x"] == [expected, 16384]
        assert occurrence["typed"]["shapes"]["output"] == [expected, 4096]
        assert occurrence["typed"]["shapes"]["pertoken_scale"] == [expected, 256, 2]
    r35 = _compiler(case).compile(instance, feature_metadata=R35_METADATA)
    r35_values = {m: {cost.name: cost for cost in
                      r35.evaluate(_prefill(m)).operators}["o_projection"]
                  for m in (1024, 1, 2)}
    r35_first = r35_values[1024]
    r35_call = r35_first.native_call
    r35_prepared = r35_first.native_occurrence["prepare"]
    assert r35_call["api"]["version"] == "r35/5461a1a8"
    assert r35_call["source"]["sha256"] == "51ece928e77420a7f6600e2d87c46e9397a5140b6acee324aa75fdb97266d009"
    assert r35_call["source"]["revision"] == "r35/5461a1a8"
    assert r35_call["call_id"] != call["call_id"]
    assert "kernel_source_runtime_binary_only" in r35_call["unresolved"]
    assert r35_prepared["source_lines"]["qmm_call"] == [150, 160]
    assert r35_prepared["source_lines"]["scale_allocation"] == [89, 89]
    assert r35_prepared["selected_host_variant"] == "apply"
    assert r35_prepared["rejected_host_variants"] == ["apply_single_group_gmm:GroupedMatmulV3"]
    assert r35_prepared["deployment"]["image_id"].startswith("5461a1a8")
    assert r35_prepared["shapes"]["x"] == [1024, 16384]
    assert r35_prepared["shapes"]["output"] == [1024, 4096]
    assert r35_prepared["descriptors"]["weight_scale"]["allocation_shape"] == [4096, 512]
    assert r35_prepared["descriptors"]["weight_scale"]["device_shape"] == [256, 4096, 2]
    assert r35_prepared["match"]["source_version"] == "r35/5461a1a8"
    for m, expected in ((1, 1), (2, 2)):
        occurrence = r35_values[m].native_occurrence
        assert r35_values[m].native_call["call_id"] == r35_call["call_id"]
        assert occurrence["typed"]["shapes"]["x"] == [expected, 16384]
        assert occurrence["typed"]["shapes"]["output"] == [expected, 4096]
        assert occurrence["prepare"]["shapes"]["pertoken_scale"] == [expected, 256, 2]
    unknown = _compiler(case).compile(instance, feature_metadata=UNKNOWN_METADATA)
    unbound = {cost.name: cost for cost in unknown.evaluate(_prefill(1024)).operators}["o_projection"]
    assert unbound.native_call["api"]["entry"] is None
    assert unbound.native_call["source"] is None
    assert unbound.native_call["kernels"] == []
    assert unbound.native_occurrence["status"] == "unresolved"
    assert "prepare" not in unbound.native_occurrence
    assert "deployment_source_unpinned" in unbound.native_call["unresolved"]
    assert "source_revision_has_no_authentic_quant_wrapper" in unbound.native_call["unresolved"]
    assert unbound.native_call["call_id"] != call["call_id"]


def test_prepared_qbmm_invocation_plan_validates_r64_occurrence_without_npu():
    case, features = _case(GLM, tp=1, dp=32, ep=32, pp=1, cores=64)
    instantiated = instantiate_model(case.model, features)
    instance = next(item for layer in instantiated.layers for item in layer.subgraphs
                    if item.kind == "A1")
    compiled = _compiler(case).compile(instance, feature_metadata=R64_METADATA)
    cost = {item.name: item for item in compiled.evaluate(_prefill(1024)).operators}["o_projection"]
    occurrence = {**cost.native_occurrence, "native_call": cost.native_call}
    plan = prepared_qbmm_plan(occurrence)
    assert plan["entry"] == "torch_npu.npu_quant_matmul"
    assert plan["input_names"] == ["x", "weight", "weight_scale", "pertoken_scale"]
    assert plan["group_sizes"] == [1, 1, 32]
    assert plan["output_dtype"] == "bf16"
    bad = copy.deepcopy(occurrence)
    bad["native_call"]["source"]["sha256"] = "0" * 64
    with pytest.raises(NativeInvocationUnavailable, match="prepared_source_not_supported"):
        prepared_qbmm_plan(bad)
    with pytest.raises(NativeInvocationUnavailable, match="torch_unavailable"):
        invoke_qbmm(occurrence, {}, device="cpu", dispatcher=lambda **kwargs: None)
