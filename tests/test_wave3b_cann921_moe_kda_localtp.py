"""CANN 9.2.1 MoE gating, KDA placement and physical identity contracts.

[Role] Verify the Wave 3B alignments against declared structure and work.
[Invariants] Fused calls conserve the declared Vector Ops and elimination is
  limited to intermediate HBM round-trips; local placement bytes equal the
  capacity ledger; a routed quantizer is billed once on its owning kernel.
[Timeouts & Locks] Small in-memory graphs from retained catalog cases.
[Side-Effect Policy] No device, network or filesystem writes.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.quant import projection_quant
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.compose import compose_case
from mindie_sim.prediction import local_projection_tp
from mindie_sim.prediction.kda_source import options as kda_options
from mindie_sim.prediction.moe_cann_fusion import (
    HASH_NATIVE_KIND, INIT_ROUTING_NATIVE_KIND, MX_SCALE_GROUP,
    SOFTMAX_NATIVE_KIND, absorb_init_routing, build_selection,
    cann_native_active, dispatch_quantizes_internally,
)
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.passes_ffn_fusion import raw_selection
from mindie_sim.prediction.physical_key import (
    REFERENCE_KEYS, normalize_reference, physical_key, physical_signature,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec

ROOT = Path(__file__).resolve().parents[1]
INT8 = {"weight_format": "int8", "activation_format": "int8"}
MXFP8 = {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3"}


def compose(model, *, layers=4, dp=8, ep=8, concurrency=16, quant=None,
            local_tp=None, **extra):
    kwargs = dict(model=model, hardware="a5", tp=1, dp=dp, ep=ep,
                  layers=layers, concurrency=concurrency, **extra)
    if quant is not None:
        kwargs["quant"] = quant
    if local_tp is not None:
        kwargs.update(o_proj_tp=local_tp, linear_proj_tp=local_tp,
                      lm_head_tp=local_tp)
    return parse_case(compose_case(**kwargs))


GATING = "vllm_ascend_cann_gating"


def gating_runtime(runtime):
    """Declare the CANN gating overlay on an already parsed runtime."""
    kernels = dict(runtime.kernels)
    kernels["moe_gating"] = {"implementation": GATING}
    return replace(runtime, kernels=kernels)


def gating(case):
    return replace(case, runtime=gating_runtime(case.runtime))


def graph(case, *, tokens=8, context=128, sequences=1, fused=True):
    target = gating(case) if fused else case
    return target_operators(target, phase="decode", tokens=tokens,
                            context=context, sequences=sequences, scope="body")


def layer_ops(ops, layer):
    return [op for op in ops if op.layer == layer]


def mx_case():
    """GLM-5.3-Flash with one MXFP8 operand representation on every KDA role."""
    case = compose("glm-5.3-flash", quant="w8a8_mxfp8", local_tp=8)
    quant = case.model.quant
    override = tuple(quant.category_overrides) + (("kda", (
        ("weight_format", "fp8_e4m3"), ("activation_format", "fp8_e4m3"))),)
    raw = compose_case(model="glm-5.3-flash", hardware="a5", tp=1, dp=8, ep=8,
                       o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, layers=4,
                       concurrency=16, quant="w8a8_mxfp8")
    raw["model"]["quant"] = replace(quant, category_overrides=override).to_dict()
    return parse_case(raw)


def _op(ops, op_id):
    return next(op for op in ops if op.op_id == op_id)


def test_fp32_gate_declaration_uses_bf16_cube_weights_with_fp32_logits():
    case = compose("glm-5.3-flash", layers=4, dp=1, ep=1, concurrency=1)
    plan = next(p for p in case.model.layer_plans if p.ffn == "moe")
    assert (plan.router_dtype, plan.router_output_dtype,
            plan.router_precast_weight_copy) == ("", "fp32", False)
    ops = graph(case, tokens=4)
    assert not any(op.recipe == "router_input_cast" for op in ops)
    router = next(op for op in ops if op.family == "router" and op.layer == plan.index)
    metadata = dict(router.metadata)
    assert metadata["weight_format"] == metadata["activation_format"] == "bf16"
    assert metadata["output_format"] == "fp32"
    assert metadata["packed_weight_bytes"] == (
        2 * plan.num_experts * plan.router_input_size)
    terms = {term.name: term for term in layer_weight_terms(plan, case.model)}
    assert "router_precast" not in terms


def test_explicit_fp32_router_plan_keeps_its_declared_input_cast():
    """A direct ``LayerPlan(router_dtype="fp32")`` remains a declared copy."""
    case = compose("glm-5.3-flash", layers=4, dp=1, ep=1, concurrency=1)
    plans = tuple(replace(
        plan, router_dtype="fp32", router_output_dtype="",
        router_precast_weight_copy=False) if plan.ffn == "moe" else plan
        for plan in case.model.layer_plans)
    case = replace(case, model=replace(case.model, layer_plans=plans))
    ops = graph(case, tokens=4)
    cast = next(op for op in ops if op.recipe == "router_input_cast")
    router = next(op for op in ops if op.family == "router")
    assert router.deps == (cast.op_id,)
    assert (router.dtype, dict(router.metadata)["weight_format"]) == ("fp32", "fp32")
    assert cast.mte3_bytes == 4 * cast.local_shape[0] * cast.local_shape[1]


def test_hash_layer_emits_one_gating_call_without_renorm_or_cast():
    case = compose("deepseek-v4-flash-0731", layers=3, concurrency=1)
    ops = graph(case, tokens=4)
    hash_layers = [plan.index for plan in case.model.layer_plans
                   if plan.router_mode == "hash"]
    assert hash_layers
    for layer in hash_layers:
        plan = case.model.layer_plans[layer]
        selected = [op for op in layer_ops(ops, layer)
                    if op.op_id.endswith(".moe_gating_topk")]
        assert len(selected) == 1
        gating = selected[0]
        metadata = dict(gating.metadata)
        assert metadata["native_kind"] == HASH_NATIVE_KIND
        assert metadata["fused_from"] == ("hash_routing",)
        assert metadata["fused_member_count"] == 1
        assert metadata["renorm"] is False
        # The hash kernel has no in-kernel renorm: a checkpoint that declares
        # one keeps that single normalization as an external consumer.
        if plan.router_norm_topk:
            normalize = next(op for op in layer_ops(ops, layer)
                             if op.recipe == "route_normalize")
            assert normalize.deps == (gating.op_id,)


def _raw_counterpart(case, plan, *, tokens, weight_dtype, weight_bytes):
    ops, select, ids, weights = raw_selection(
        prefix=f"s{plan.scope}.l{plan.index}", plan=plan,
        logits_input="logits", weight_dtype=weight_dtype,
        weight_bytes=weight_bytes, stage=0, requests=(), tokens=tokens,
        mode="decode")
    assert select is not None and ids and weights
    return ops


def test_sqrtsoftplus_gating_fuses_renorm_and_conserves_vector_work():
    case = compose("deepseek-v4-flash-0731", layers=4, concurrency=1)
    plan = next(p for p in case.model.layer_plans
                if p.router_mode == "sqrtsoftplus" and p.ffn == "moe")
    tokens = 8
    selection = build_selection(
        prefix="s0.l0", plan=plan, logits_input="logits", weight_dtype="fp32",
        weight_bytes=4, stage=0, requests=(), tokens=tokens, mode="decode",
        runtime=gating_runtime(case.runtime))
    assert selection.fused is True
    assert selection.cann_kind == SOFTMAX_NATIVE_KIND
    gating, = [op for op in selection.ops if op.recipe == "moe_gating_topk"]
    metadata = dict(gating.metadata)
    assert metadata["native_kind"] == SOFTMAX_NATIVE_KIND
    assert metadata["renorm"] is True
    assert metadata["fused_member_count"] == 2
    raw = _raw_counterpart(case, plan, tokens=tokens, weight_dtype="fp32",
                           weight_bytes=4)
    assert [op.recipe for op in raw] == ["router_topk", "route_normalize"]
    assert gating.vector_ops == pytest.approx(sum(op.vector_ops for op in raw))
    # Only the logits are read once; only the selected ids and weights leave.
    assert gating.mte2_bytes == tokens * plan.num_experts * 4
    assert gating.mte3_bytes == tokens * plan.top_k * 8
    assert sum(op.mte2_bytes for op in raw) > gating.mte2_bytes
    # The declared BF16 Cube router GEMM stays a separate operator.
    assert all(op.recipe != "router" for op in selection.ops)


def test_noaux_tc_fuses_the_whole_chain_including_scale_and_bias_read():
    case = compose("deepseek-v3", layers=4, concurrency=1)
    plan = next(p for p in case.model.layer_plans
                if p.ffn == "moe" and p.router_noaux_tc)
    tokens = 8
    selection = build_selection(
        prefix="s0.l0", plan=plan, logits_input="logits", weight_dtype="fp32",
        weight_bytes=4, stage=0, requests=(), tokens=tokens, mode="decode",
        runtime=gating_runtime(case.runtime))
    gating = next(op for op in selection.ops if op.recipe == "moe_gating_topk")
    metadata = dict(gating.metadata)
    assert metadata["noaux_tc"] is True
    assert metadata["fused_member_count"] == 9
    assert gating.mte2_bytes == tokens * plan.num_experts * 4 + plan.num_experts * 4
    assert metadata["correction_bias_bytes"] == plan.num_experts * 4
    raw = _raw_counterpart(case, plan, tokens=tokens, weight_dtype="fp32",
                           weight_bytes=4)
    assert len(raw) == 9
    assert gating.vector_ops == pytest.approx(sum(op.vector_ops for op in raw))
    group_score = next(op for op in raw if op.recipe == "router_group_score_top2")
    assert group_score.vector_ops == 3 * tokens * (
        plan.num_experts - plan.router_groups)
    assert next(op for op in raw if op.recipe == "router_routed_scale")
    assert not any(op.recipe == "router_routed_scale" for op in selection.ops)


def test_raw_eager_keeps_the_declared_decomposition_and_gating_is_off():
    case = compose("deepseek-v3", layers=4, concurrency=1, raw_eager=True)
    assert cann_native_active(gating_runtime(case.runtime)) is False
    ops = graph(case, tokens=4)
    assert not any(op.recipe == "moe_gating_topk" for op in ops)
    assert any(op.recipe == "router_sigmoid_scores" for op in ops)


def test_explicit_generic_router_recipe_disables_the_fused_call():
    case = compose("deepseek-v3", layers=4, concurrency=1)
    kernels = dict(gating_runtime(case.runtime).kernels)
    kernels["recipes"] = {**dict(kernels.get("recipes") or {}), "router_topk": "declared"}
    runtime = replace(case.runtime, kernels=kernels)
    assert cann_native_active(runtime) is False
    # The composed CANN 9.2.1 stack declares the gating overlay; an explicit
    # generic router recipe above still keeps the per-operator decomposition.
    assert cann_native_active(case.runtime) is True


def test_ep_dispatch_quantizer_is_absorbed_into_init_routing_once():
    case = compose("glm-5.3-flash", quant="w8a8_mxfp8", layers=4)
    ops = graph(case)
    assert not any(op.op_id.endswith(".pack.quantize") for op in ops)
    pack = next(op for op in ops if op.op_id.endswith(".pack"))
    metadata = dict(pack.metadata)
    routing = metadata["moe_init_routing"]
    assert pack.recipe == "moe_init_routing_v3"
    assert metadata["native_kind"] == routing["native_kind"] == INIT_ROUTING_NATIVE_KIND
    assert routing["quant_mode"] == 1
    assert routing["payload_format"] == "fp8_e4m3"
    assert routing["scale_format"] == "fp8_e8m0"
    assert metadata["activation_scale_bytes"] == routing["scale_bytes"] > 0.0
    assert "activation_scales" in pack.returns
    # The quantization Vector Ops are billed on the routing pack, once.
    assert pack.vector_ops >= routing["vector_ops"] > 0.0
    # The quantized payload replaces the BF16 payload; the INT32 inverse
    # permutation remains an auxiliary output of the same pack.
    routed_slots = pack.local_shape[0] * pack.local_shape[1]
    assert routing["inverse_index_bytes"] == pytest.approx(routed_slots * 4)
    assert pack.mte3_bytes == pytest.approx(
        routing["write_bytes"] + routing["inverse_index_bytes"])
    gate = next(op for op in ops if op.family == "grouped_gemm")
    gate_metadata = dict(gate.metadata)
    # The pack publishes the quantized payload plus scales; the dispatch that
    # consumes it republishes the expert input rows.
    dispatch = next(op for op in ops if op.family == "alltoall"
                    and op.recipe == "expert_dispatch")
    assert gate_metadata["activation_payload_producer"] == dispatch.op_id
    assert gate_metadata["activation_scale_producer"] == pack.op_id


def test_fused_mc2_dispatch_owns_the_absorbed_init_routing_call():
    case = compose("glm-5.3-flash", quant="w8a8_mxfp8", layers=4,
                   moe_comm_mode="dynamic")
    ops = graph(case)
    assert not any(op.op_id.endswith(".pack.quantize") for op in ops)
    dispatch = next(op for op in ops if op.family == "mc2_protocol"
                    and dict(op.metadata).get("native_kind") == "MoeDistributeDispatchV3")
    metadata = dict(dispatch.metadata)
    routing = metadata["moe_init_routing"]
    assert routing["native_kind"] == INIT_ROUTING_NATIVE_KIND
    assert routing["quant_mode"] == 1
    assert metadata["fused_native_kinds"] == ("MoeDistributeDispatchV3",
                                              INIT_ROUTING_NATIVE_KIND)
    # The native dispatch prices the sender quantization in its phase ledger.
    assert "send_read_quantize" in metadata["mc2_phases"]
    assert metadata["mc2_phase"]["dispatch_quant"] == "mx"
    gate = next(op for op in ops if op.family == "grouped_gemm")
    assert dict(gate.metadata)["activation_scale_producer"] == dispatch.op_id


def test_absorb_requires_a_portable_payload_and_ep_greater_than_one():
    pack = OpSpec("p.pack", "routing", "decode", 0, 0, (), (1, 6), (1, 6),
                  (1, 6), (1, 6), vector_ops=10.0, mte2_bytes=100.0,
                  mte3_bytes=200.0, metadata=(("top_k", 6),))
    quantizer = OpSpec(
        "p.pack.quantize", "quantize", "decode", 0, 0, (), (8, 128), (8, 128),
        (8, 128), (8, 128), vector_ops=50.0, mte2_bytes=2048.0,
        mte3_bytes=1040.0, recipe="dynamic_quantize",
        returns=("activation_scales",),
        metadata=(("quantization_scope", "moe_dispatch_expert_input"),
                  ("input_format", "bf16"), ("output_format", "fp8_e4m3"),
                  ("activation_group_size", MX_SCALE_GROUP),
                  ("activation_scale_format", "fp8_e8m0"),
                  ("activation_scale_shape", (8, 4, 2)),
                  ("activation_scale_bytes", 64.0), ("source_rows", 8.0)))
    absorbed, retired = absorb_init_routing(pack=pack, quantizer=quantizer, ep=8)
    assert retired == quantizer.op_id
    assert absorbed.vector_ops == 60.0
    assert absorbed.mte2_bytes == 100.0
    # Six routed slots keep their INT32 inverse-permutation write.
    assert absorbed.mte3_bytes == 1040.0 + 6 * 4
    assert dict(absorbed.metadata)["moe_init_routing"]["quant_mode"] == 1
    coarse = replace(quantizer, metadata=tuple(
        (key, 128 if key == "activation_group_size" else value)
        for key, value in quantizer.metadata))
    assert absorb_init_routing(pack=pack, quantizer=coarse, ep=8) == (pack, None)
    assert absorb_init_routing(pack=pack, quantizer=quantizer, ep=1) == (pack, None)
    assert absorb_init_routing(
        pack=pack, quantizer=quantizer, ep=8, dispatch_owns_quant=True) == (pack, None)
    plain = compose("glm-5.3-flash", layers=4)
    assert dispatch_quantizes_internally(plain.runtime) is False
    arch22 = replace(plain.runtime, kernels={"native_moe_protocol": {
        "kind": "arch22_v2_caller", "dispatch_quant": "per_token_int8"}})
    assert dispatch_quantizes_internally(arch22) is True


def test_kda_source_declares_in_kernel_sigmoid_beta_with_activation_dtype():
    case = parse_case(json.loads(
        (ROOT / "examples/cases/kimi-k3-a5.json").read_text()))
    plan = next(p for p in case.model.layer_plans if p.attention == "kda")
    declared = kda_options(plan, activation_dtype=case.model.activation_dtype)
    assert declared["use_beta_sigmoid"] is True
    assert declared["beta_dtype"] == case.model.activation_dtype
    recipes = [op.recipe for op in graph(case, tokens=1)]
    assert "linear_beta_sigmoid" not in recipes
    assert recipes.count("vllm_ascend_recurrent_kda_arch35") == 69


def test_recover_ssm_verify_kernel_keeps_the_fused_beta_sigmoid():
    case = parse_case(json.loads(
        (ROOT / "examples/cases/kimi-k3-a5.json").read_text()))
    case = replace(case, runtime=replace(case.runtime, recover_ssm=True))
    ops = graph(case, tokens=2)
    assert not any(op.recipe == "linear_beta_sigmoid" for op in ops)
    kernel = next(op for op in ops if op.recipe == "vllm_ascend_recurrent_kda_arch35")
    parameters = dict(kernel.metadata)["kda_recurrent_params"]
    assert parameters["recover_ssm"] is True
    assert parameters["use_beta_sigmoid"] is True
    assert parameters["beta_dtype"] == case.model.activation_dtype


def test_local_tp8_kda_input_projection_matches_the_capacity_ledger():
    case = mx_case()
    ops = graph(case, tokens=1)
    plan = next(p for p in case.model.layer_plans if p.attention == "kda")
    layer = plan.index
    gemm = _op(ops, f"s{0}.l{layer}.linear_qkvbfg_a:local_tp:gemm")
    assert gemm.local_shape == (8, 3336, 4096)
    assert gemm.flops == pytest.approx(2 * 8 * 3336 * 4096)
    slices = dict(gemm.metadata)["packed_projection_slices"]
    assert [(item["role"], item["shape"][1], item["replicated"]) for item in slices] == [
        ("q", 1024, False), ("k", 1024, False), ("v", 1024, False),
        ("b", 8, False), ("f_a", 128, True), ("g_a", 128, True)]
    terms = {term.name: term for term in layer_weight_terms(plan, case.model)}
    quant, parallel = projection_quant(case.model.quant, plan), case.runtime.parallel
    ledger = sum(term_bytes(terms[name], quant, parallel)
                 for name in ("kda_q", "kda_k", "kda_v", "kda_b",
                              "kda_f_a", "kda_g_a"))
    assert dict(gemm.metadata)["packed_weight_bytes"] == ledger
    # Q/K/V/B are 24,640 logical output features; f_a/g_a never cross the wire.
    assert sum(item["shape"][1] for item in slices
               if not item["replicated"]) * 8 == 24640
    restore = _op(ops, f"s{0}.l{layer}.linear_qkvbfg_a")
    assert restore.family == "alltoall"
    assert restore.link_bytes == pytest.approx(7 * 1 * 3080 * 2)


def test_local_gate_matrices_execute_locally_without_a_collective():
    case = mx_case()
    runtime = case.runtime
    assert local_projection_tp._width(runtime, "linear_gate") == 1
    assert local_projection_tp._width(runtime, "linear_input") == 8
    ops = graph(case, tokens=1)
    plan = next(p for p in case.model.layer_plans if p.attention == "kda")
    for recipe in ("linear_gate_b", "linear_decay_b"):
        op = _op(ops, f"s0.l{plan.index}.{recipe}")
        assert op.local_shape == (1, 8192, 128)
        assert dict(op.metadata).get("local_tp_applied") is None
        assert not any(collective in dependency
                       for dependency in op.deps
                       for collective in ("local_tp:exchange", "local_tp:gather"))
    terms = {term.name: term for term in layer_weight_terms(plan, case.model)}
    quant, parallel = projection_quant(case.model.quant, plan), runtime.parallel
    for recipe, name in (("linear_gate_b", "kda_g_b"),
                         ("linear_decay_b", "kda_f_b")):
        op = _op(ops, f"s0.l{plan.index}.{recipe}")
        assert dict(op.metadata)["packed_weight_bytes"] == term_bytes(
            terms[name], quant, parallel)


def test_row_projection_exchanges_bf16_before_the_local_quantizer():
    case = mx_case()
    ops = graph(case, tokens=1)
    plan = next(p for p in case.model.layer_plans if p.attention == "kda")
    prefix = f"s0.l{plan.index}.linear_output"
    exchange = _op(ops, prefix + ":local_tp:exchange")
    quantizer = _op(ops, prefix + ".quantize")
    gemm = _op(ops, prefix + ":local_tp:gemm")
    restore = _op(ops, prefix)
    assert exchange.family == "alltoall" and exchange.dtype == "bf16"
    assert exchange.local_shape == (8, 1024)
    assert exchange.link_bytes == pytest.approx(7 * 1 * 1024 * 2)
    assert quantizer.deps == (exchange.op_id,)
    assert quantizer.local_shape == (8, 1024)
    quant_metadata = dict(quantizer.metadata)
    assert quant_metadata["activation_scale_shape"] == (8, 16, 2)
    assert quant_metadata["activation_scale_bytes"] == 8 * 16 * 2
    assert gemm.deps == (quantizer.op_id,)
    assert gemm.local_shape == (8, 4096, 1024)
    assert dict(gemm.metadata)["activation_scale_bytes"] == 8 * 16 * 2
    assert restore.family == "reduce_scatter" and restore.deps == (gemm.op_id,)
    # The retired full-axis quantizer is not executed twice.
    assert [index for index, op in enumerate(ops) if op.op_id == prefix + ".quantize"] == [
        index for index, op in enumerate(ops) if op.op_id == quantizer.op_id]


def test_kimi_kda_mixed_precision_packs_by_operand_representation():
    case = parse_case(json.loads((ROOT / "examples/cases/"
        "kimi-k3-a5-current-checkpoint-conditional.json").read_text()))
    plan = next(p for p in case.model.layer_plans
                if p.attention == "kda" and p.ffn == "moe")
    assert plan.packed_input_projection is True
    ops = graph(case, tokens=4)
    packed = {op.recipe: op for op in layer_ops(ops, plan.index)
              if "linear_qkvbfg_a" in op.recipe and op.family == "gemm"}
    assert len(packed) == 2
    metadata = [dict(op.metadata) for op in packed.values()]
    assert all(item["projection_packing"] ==
               "compatible_group_shared_input_output_slices" for item in metadata)
    groups = {}
    for item in metadata:
        for role_list in item["packed_projection_group_roles"]:
            groups[tuple(role_list.split("+"))] = item["activation_format"]
    assert groups == {("q", "k", "v"): "fp8_e4m3", ("b", "f_a", "g"): "bf16"}
    assert not any(op.recipe in {"linear_q", "linear_k", "linear_v", "linear_beta",
                                 "linear_decay_a", "linear_gate"}
                   for op in layer_ops(ops, plan.index))
    assert not any(dict(op.metadata).get("native_kind") == "Slice"
                   for op in layer_ops(ops, plan.index))


def test_physical_key_deduplicates_context_kv_and_fused_reference_fields():
    for key in ("typed_outputs", "fusion_external_outputs", "audit_trail",
                "fusion_boundary", "input_ports"):
        assert key in REFERENCE_KEYS
    assert normalize_reference("d:context_kv5.qkv_a", 0, 5) == "d:context_kv0.qkv_a"
    assert normalize_reference("draft1.context_kv9.store", 0, 5) == \
        "draft1.context_kv4.store"
    assert normalize_reference("s0.l5.context_kv5.store", 0, 5) == \
        "s0.l0.context_kv0.store"
    assert normalize_reference("kv_context_kv9", 0, 5) == "kv_context_kv9"

    def occurrence(layer):
        return OpSpec(
            f"d:context_kv{layer}.qkv_a", "gemm", "draft", 0, layer,
            (), (4, 128), (4, 128), (4, 128), (4, 128), flops=1.0,
            metadata=(
                ("typed_outputs", ({"producer": f"s0.l{layer}.context_kv{layer}.qkv_a",
                                    "output": "payload"},)),
                ("fusion_external_outputs", (f"d:context_kv{layer}.norm",)),
                ("input_ports", ({"op": f"d:context_kv{layer}.qkv_a",
                                  "output": "payload"},)),
                ("audit_trail", ({"op_id": f"d:context_kv{layer}.qkv_a",
                                  "kind": "transform_boundary"},)),
                ("fusion_boundary", {"events": {
                    "outputs": (f"d:context_kv{layer}.norm",)}}),
            ))

    first, second = occurrence(3), occurrence(7)
    assert physical_signature(first) == physical_signature(second)
    assert physical_key(first) == physical_key(second)
    other = replace(first, metadata=tuple(
        (key, ("different",) if key == "fusion_external_outputs" else value)
        for key, value in first.metadata))
    assert physical_signature(other) != physical_signature(first)


def test_gating_fusion_is_not_claimed_for_a_non_ascend_runtime():
    case = compose("deepseek-v3", layers=4, concurrency=1)
    declared = gating_runtime(case.runtime)
    metadata = dict(declared.metadata)
    metadata["execution_policy"] = {**dict(metadata["execution_policy"]),
                                    "backend": "nvidia"}
    runtime = replace(declared, metadata=tuple(metadata.items()))
    assert cann_native_active(runtime) is False
    assert cann_native_active(None) is False
