"""Physical FFN and MoE fusion rules.

[Role] Rewrite gated activation and quantization, on-chip grouped GEMM
  epilogues, and shared, routed, and residual output combines. Declare the
  router selection and expert gate-up interfaces used before rewriting.
[Invariants] Keep Cube FLOPs, scale epilogues, and vector work. Fold only the
  intermediate HBM round-trips and launches owned by the fusion. Match tensor
  geometry and dependency edges. Keep per-expert geometry and preserve padded
  capacity when it exceeds the activated rows.
[Timeouts & Locks] Linear walks over one layer graph with no I/O or locks.
[Side-Effect Policy] Return immutable operators and extend caller-owned
  group metadata.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.expert_slots import expert_slot_plan
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.runtime_features import split_routing_features
from mindie_sim.prediction.operators.moe_routing.expert_slot_work import bind_expert_slot_work, physical_slot_work
from mindie_sim.prediction.operators.vector.gated_activation import SITU_SOURCE
from mindie_sim.prediction.implementations import (
    append_audit, attach_quantized_ports, rewire_replacements,
)
from mindie_sim.prediction.operators.mx_quant_matrix.mxfp8_grouped import (SWIGLU_MX_AIV_UNITS, SWIGLU_MX_GROUP,
                            SWIGLU_MX_STAGES)
from mindie_sim.prediction.rewrite_mhc_rules import merge_meta, users_map
from mindie_sim.prediction.structure_groups import declare_group, source_reference

NO_PORTS: dict = {}

#: ``noaux_tc`` routing expression work, per the reference chain:
#: sigmoid scores -> correction bias -> per-group Top2 sum -> group TopK ->
#: mask -> expert TopK -> unbiased weight gather -> routed scale.
NOAUX_SIGMOID_OPS_PER_ELEMENT = 4
NOAUX_BIAS_ADD_OPS_PER_ELEMENT = 1
NOAUX_GROUP_TOP2_SCAN_OPS = 3
NOAUX_GROUP_TOPK_OPS_PER_GROUP = 2
NOAUX_GROUP_MASK_OPS_PER_ELEMENT = 1
NOAUX_WEIGHT_GATHER_OPS_PER_SELECTED = 1
NOAUX_ROUTED_SCALE_OPS_PER_SELECTED = 1
ROUTE_NORMALIZE_OPS_PER_SELECTED = 3
#: MXFP8 scale pairs: `activation_group_size=32` published as `(groups/2, 2)`.
MX32_SCALE_GROUP = 32
MX32_SCALE_PAIR = 64
#: Retained Situ overlay adapter lines for the fused RunV2Core entry.
SITU_OVERLAY_ADAPTER_LINES = (148, 184)
NOAUX_BIAS_BYTES = 4
#: The collapsed score+topk placeholder declares one scoring expression plus
#: the declared scoring scan per expert element; the per-function split is a
#: separate calibration axis, not part of this fusion.
ROUTER_SCORE_AND_TOPK_OPS = 5
#: A hash layer reads one token id plus its ``top_k`` table entries per row.
HASH_ROW_ID_BYTES = 8
SELECTED_ID_BYTES = 4
#: Narrow expert grids whose output tile count leaves fewer than one AIC wave
#: declare Split-K across the free AICs; the registered planner owns the split.
ROUTER_SPLITK_MAX_N = 1024
ROUTED_SCALE_RECIPE = "router_routed_scale"
ROUTE_NORMALIZE_RECIPE = "route_normalize"
CONTRACT_HASH = "MoeGatingTopKHash"
CONTRACT_SCORE = "MoeGatingTopK"
HASH_SELECTION_MODE = "hash_table"
SCORE_SELECTION_MODE = "score_topk"
ROUTER_TORCH_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "path": "vllm_ascend/ops/fused_moe/router/fused_topk_router.py",
    "sha256": "4a4e8cb559bda606457022ad627e787228b24bb6e514f1e162b323e3de6898d4",
    "source_state": "local_worktree", "symbol": "AscendFusedTopKRouter._compute_routing",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def router_gemm(*, prefix, plan, deps, stage, requests, tokens, model, hidden, mode):
    """Emit the ``AscendGateLinear`` router: BF16 Cube operands, FP32 logits.

    The narrow-N Tier-4 form consumes the checkpoint's BF16 gate matrix and
    requests an FP32 accumulator/store, so it keeps BF16 Cube operands even
    when the surrounding projection is MXFP8/FP8.  A narrow expert grid that
    leaves fewer output tiles than AICs declares Split-K.  An explicit FP32
    plan (``router_dtype == "fp32"``) keeps its declared cast unchanged.
    """
    cast = None
    router_input, router_quant, router_category = deps, model.quant, "router"
    output_dtype = plan.router_output_dtype or None
    if plan.router_dtype:
        if plan.router_dtype not in {"fp32", "bf16", "fp16"}:
            raise ModelingError(f"unsupported_router_compute_dtype:{plan.router_dtype}")
        router_quant = QuantSpec(
            name="declared_router", weight_format=plan.router_dtype,
            activation_format=plan.router_dtype)
        router_category = "router_runtime"
        width = plan.router_input_size or hidden
        if tokens and plan.router_dtype != model.activation_dtype:
            cast = vector_op(
                op_id=prefix + ".router_cast", deps=(deps,), stage=stage,
                layer=plan.index, requests=requests, elements=tokens * width,
                ops_per_element=1, dtype=plan.router_dtype, family="cast",
                phase=mode, recipe="router_input_cast", shape=(tokens, width),
                read_bytes_override=tokens * width * dtype_bytes(model.activation_dtype),
                write_bytes_override=tokens * width * dtype_bytes(plan.router_dtype),
                metadata=(("input_format", model.activation_dtype),
                          ("output_format", plan.router_dtype)),
            )
            router_input = cast.op_id
    elif output_dtype == "fp32":
        router_quant = QuantSpec(name="declared_router_bf16", weight_format="bf16",
                                 activation_format="bf16")
        router_category = "router_runtime"
    split_k = (not plan.router_dtype and output_dtype == "fp32"
               and int(plan.num_experts) <= ROUTER_SPLITK_MAX_N)
    router = gemm_op(
        op_id=prefix + ".router", deps=(router_input,), stage=stage,
        layer=plan.index, requests=requests, m=tokens, n=plan.num_experts,
        k=plan.router_input_size or hidden, quant=router_quant,
        category=router_category, phase=mode, family="router", recipe="router",
        output_dtype=output_dtype,
        metadata=((("router_operand_form", "bf16_cube_fp32_output"),
                   ("split_k_required", True)) if split_k else ()),
    )
    weight_dtype = dict(router.metadata)["output_format"]
    return ([] if cast is None else [cast]), router, weight_dtype, dtype_bytes(weight_dtype)


def _noaux_tc_chain(*, prefix, logits_input, weight_dtype, weight_bytes, stage,
                    layer, requests, plan, tokens, mode):
    """Exact ``noaux_tc`` grouped-topk routing chain (raw eager decomposition).

    Biased scores select experts; the emitted weights are gathered from the
    *unbiased* sigmoid scores.  The FP32 correction bias is one resident
    parameter per MoE layer, loaded once per invocation.
    """
    experts = int(plan.num_experts)
    groups = max(1, int(plan.router_groups))
    group_topk = max(1, int(plan.router_group_topk))
    selected = int(plan.top_k)
    scores = vector_op(
        op_id=prefix + ".router_scores", deps=(logits_input,), stage=stage,
        layer=layer, requests=requests, elements=tokens * experts,
        ops_per_element=NOAUX_SIGMOID_OPS_PER_ELEMENT, dtype="fp32",
        family="sigmoid", phase=mode, recipe="router_sigmoid_scores",
        shape=(tokens, experts),
        read_bytes_override=tokens * experts * weight_bytes,
        write_bytes_override=tokens * experts * NOAUX_BIAS_BYTES,
        metadata=(("input_format", weight_dtype), ("output_format", "fp32"),
                  ("score_function", "sigmoid")),
    )
    biased = vector_op(
        op_id=prefix + ".router_bias", deps=(scores.op_id,), stage=stage,
        layer=layer, requests=requests, elements=tokens * experts,
        ops_per_element=NOAUX_BIAS_ADD_OPS_PER_ELEMENT, dtype="fp32",
        family="elementwise", phase=mode, recipe="router_correction_bias",
        shape=(tokens, experts),
        read_bytes_override=(tokens + 1) * experts * NOAUX_BIAS_BYTES,
        write_bytes_override=tokens * experts * NOAUX_BIAS_BYTES,
        metadata=(("input_format", "fp32"), ("output_format", "fp32"),
                  ("correction_bias_elements", experts),
                  ("correction_bias_bytes", experts * NOAUX_BIAS_BYTES),
                  ("correction_bias_residency", "ukernel_argument_once_per_layer")),
    )
    group_scores = vector_op(
        op_id=prefix + ".router_group_score", deps=(biased.op_id,), stage=stage,
        layer=layer, requests=requests, elements=tokens * experts,
        ops_per_element=NOAUX_GROUP_TOP2_SCAN_OPS, dtype="fp32",
        family="topk", phase=mode, recipe="router_group_score_top2",
        shape=(tokens, groups),
        vector_ops_override=NOAUX_GROUP_TOP2_SCAN_OPS * tokens * max(0, experts - groups),
        write_bytes_override=tokens * groups * NOAUX_BIAS_BYTES,
    )
    group_select = vector_op(
        op_id=prefix + ".router_group_topk", deps=(group_scores.op_id,),
        stage=stage, layer=layer, requests=requests, elements=tokens * groups,
        ops_per_element=NOAUX_GROUP_TOPK_OPS_PER_GROUP, dtype="fp32",
        family="topk", phase=mode, recipe="router_group_topk",
        shape=(tokens, group_topk),
        metadata=(("group_topk", group_topk), ("groups", groups)),
    )
    mask = vector_op(
        op_id=prefix + ".router_group_mask", deps=(group_select.op_id, biased.op_id),
        stage=stage, layer=layer, requests=requests, elements=tokens * experts,
        ops_per_element=NOAUX_GROUP_MASK_OPS_PER_ELEMENT, dtype="fp32",
        family="elementwise", phase=mode, recipe="router_group_mask",
        shape=(tokens, experts),
    )
    expert_topk = vector_op(
        op_id=prefix + ".router_expert_topk", deps=(mask.op_id,), stage=stage,
        layer=layer, requests=requests, elements=tokens * experts,
        ops_per_element=max(1.0, math.log2(max(1, selected))), dtype="fp32",
        family="topk", phase=mode, recipe="router_expert_topk",
        shape=(tokens, selected),
        write_bytes_override=tokens * selected * NOAUX_BIAS_BYTES,
        metadata=(("topk_ids_dtype", "int32"), ("top_k", selected),
                  ("selection_scores", "corrected_sigmoid")),
    )
    weights = vector_op(
        op_id=prefix + ".router_weight_gather", deps=(expert_topk.op_id, scores.op_id),
        stage=stage, layer=layer, requests=requests, elements=tokens * selected,
        ops_per_element=NOAUX_WEIGHT_GATHER_OPS_PER_SELECTED, dtype="fp32",
        family="routing", phase=mode, recipe="router_weight_gather",
        shape=(tokens, selected),
        read_bytes_override=tokens * selected * NOAUX_BIAS_BYTES,
        write_bytes_override=tokens * selected * NOAUX_BIAS_BYTES,
        returns=("weights", "ids"),
        metadata=(("input_format", "fp32"), ("output_format", "fp32"),
                  ("weight_source", "unbiased_sigmoid_scores"),
                  ("topk_ids_producer", expert_topk.op_id),
                  ("router_weight_dtype", weight_dtype)),
    )
    return (scores, biased, group_scores, group_select, mask, expert_topk, weights)


def raw_selection(*, prefix, plan, logits_input, weight_dtype, weight_bytes,
                  stage, requests, tokens, mode):
    """The declared per-operator selection chain (retained for ``raw_eager``)."""
    hash_table = plan.router_mode == "hash"
    noaux_tc = bool(plan.router_noaux_tc) and not hash_table
    # An idle cohort representative still declares its selection kernels with
    # zero active rows; it never becomes a missing producer.
    if noaux_tc:
        ops = list(_noaux_tc_chain(
            prefix=prefix, logits_input=logits_input, weight_dtype=weight_dtype,
            weight_bytes=weight_bytes, stage=stage, layer=plan.index,
            requests=requests, plan=plan, tokens=tokens, mode=mode))
        select = ops[-1]
        ids_producer = dict(select.metadata)["topk_ids_producer"]
    else:
        select = vector_op(
            op_id=prefix + ".topk", deps=(logits_input,), stage=stage,
            layer=plan.index, requests=requests, elements=tokens * plan.num_experts,
            ops_per_element=ROUTER_SCORE_AND_TOPK_OPS + (
                plan.top_k / plan.num_experts if hash_table
                else math.log2(max(1, plan.top_k))),
            dtype="fp32", family="topk", phase=mode,
            recipe="hash_routing" if hash_table else "router_topk",
            shape=(tokens, plan.num_experts),
            read_bytes_override=tokens * plan.num_experts * weight_bytes
                                + (tokens * (HASH_ROW_ID_BYTES + plan.top_k * SELECTED_ID_BYTES)
                                   if hash_table else 0),
            write_bytes_override=tokens * plan.top_k * (weight_bytes + NOAUX_BIAS_BYTES),
            returns=("weights", "ids"),
            metadata=(("input_format", weight_dtype), ("output_format", weight_dtype),
                      ("topk_ids_dtype", "int32"), ("router_weight_dtype", weight_dtype),
                      ("score_function", plan.router_mode),
                      ("num_experts", int(plan.num_experts)),
                      ("top_k", int(plan.top_k)),
                      ("hash_table_active_rows", tokens if hash_table else 0)),
        )
        ops = [select]
        ids_producer = select.op_id
    if plan.router_groups > 1 and not hash_table and not noaux_tc:
        group_select = vector_op(
            op_id=prefix + ".group_select", deps=(logits_input,), stage=stage,
            layer=plan.index, requests=requests,
            elements=tokens * plan.num_experts, ops_per_element=2,
            dtype="fp32", family="topk", phase=mode, recipe="router_group_topk",
            shape=(tokens, plan.router_groups, plan.router_group_topk))
        select = replace(select, deps=(group_select.op_id,))
        ops = [group_select, select]
        ids_producer = select.op_id
    if plan.router_norm_topk:
        normalized = vector_op(
            op_id=prefix + ".route_normalize", deps=(select.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=tokens * plan.top_k,
            ops_per_element=ROUTE_NORMALIZE_OPS_PER_SELECTED, dtype="fp32",
            family="elementwise", phase=mode, recipe=ROUTE_NORMALIZE_RECIPE,
            read_bytes_override=tokens * plan.top_k * weight_bytes,
            write_bytes_override=tokens * plan.top_k * weight_bytes,
            metadata=(("input_format", weight_dtype), ("output_format", weight_dtype)))
        ops.append(normalized)
        select = normalized
    if not hash_table and float(plan.routed_scaling_factor) != 1.0:
        # The declared routed scale multiplies the selected weights *after*
        # the optional renorm; biased scores never leave the selection path.
        scaled = vector_op(
            op_id=prefix + ".route_scale", deps=(select.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=tokens * plan.top_k,
            ops_per_element=NOAUX_ROUTED_SCALE_OPS_PER_SELECTED, dtype="fp32",
            family="elementwise", phase=mode, recipe=ROUTED_SCALE_RECIPE,
            read_bytes_override=tokens * plan.top_k * weight_bytes,
            write_bytes_override=tokens * plan.top_k * weight_bytes,
            metadata=(("input_format", weight_dtype), ("output_format", weight_dtype),
                      ("routed_scaling_factor", float(plan.routed_scaling_factor))),
        )
        ops.append(scaled)
        select = scaled
    return ops, select, ids_producer, select.op_id


def declare_router_selection(*, ops, key, plan, tokens, weight_dtype, contract_kind,
                             cann_kind, fused, members, entries, anchor,
                             selection_mode, noaux_tc):
    """Declare the router selection subgraph on its anchor operator."""
    view = {
        "tokens": tokens, "experts": plan.num_experts, "topk": plan.top_k,
        "dtype": weight_dtype, "native_kind": contract_kind,
        "cann_native_kind": cann_kind, "fused": fused,
        "selection_mode": selection_mode,
        "score_function": plan.router_mode,
        # The hash call has no in-kernel renormalization: its declared top-k
        # normalization stays an external consumer.  Every score-based call
        # (softmax / sigmoid / sqrtsoftplus) owns `renorm=1`.
        "renorm": bool(plan.router_norm_topk
                       and selection_mode == SCORE_SELECTION_MODE),
        "external_renorm": bool(plan.router_norm_topk
                                and selection_mode == HASH_SELECTION_MODE),
        "noaux_tc": noaux_tc,
        "routed_scaling_factor": float(plan.routed_scaling_factor),
        "groups": plan.router_groups, "group_topk": plan.router_group_topk,
    }
    parameters = {
        "score_function": plan.router_mode,
        "renorm": view["renorm"], "external_renorm": view["external_renorm"],
        "noaux_tc": noaux_tc,
        "routed_scaling_factor": float(plan.routed_scaling_factor),
        "groups": plan.router_groups, "group_topk": plan.router_group_topk,
        "fused_native_kind": cann_kind if fused else None,
    }
    declare_group(
        ops, key=key, role="router_selection", family="moe",
        members=members, entries=entries, anchor=anchor, block="ffn",
        scope_role="body", view=view,
        implementation={"native_kind": cann_kind if fused else contract_kind,
                        "contract_native_kind": contract_kind,
                        "kind": ("declared_fused_cann_gating" if fused else
                                 "declared_native_correspondence"),
                        "fused": fused, "declared_parameters": parameters},
        source=ROUTER_TORCH_SOURCE,
    )


def declare_gate_up_fusion_group(*, ops, runtime, prefix, activation, gate_ops,
                                 gate, down, local_experts, latent, expert_width,
                                 routing_metadata, model):
    """Declare the on-chip grouped gate-up + activation/quantization interface."""
    gate_meta, down_meta = dict(gate.metadata), dict(down.metadata)
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    raw_eager = bool(
        rt_meta.get("raw_eager")
        or (rt_meta.get("execution_policy") or {}).get("raw_eager")
    )
    has_quant_pass = bool(runtime and "activation_quantization" in runtime.lowering_passes)
    has_routed_mx_scales = bool(
        down_meta.get("activation_scale_bytes")
        and down_meta.get("activation_scale_format") == "fp8_e8m0"
        and not raw_eager
    )
    if not (down_meta.get("activation_scale_bytes") and (has_quant_pass or has_routed_mx_scales)):
        group_role = "grouped_gate_up_act"
        group_key = prefix + "." + group_role
        reason = ("unquantized_expert_activation"
                  if has_quant_pass else "activation_quantization_pass_disabled")
        declare_group(
            ops, key=group_key, role=group_role, family="moe",
            members=(*(g.op_id for g in gate_ops), activation.op_id),
            entries=(gate_ops[0].op_id,),
            anchor=activation.op_id, block="ffn", scope_role="body",
            view={"input_width": latent, "intermediate_width": expert_width,
                  "local_experts": local_experts,
                  "routing_basis": dict(routing_metadata)["routing_basis"],
                  "input_dtype": gate_meta.get("activation_format", model.activation_dtype),
                  "output_dtype": down_meta.get("activation_format", model.activation_dtype),
                  "weight_dtype": gate_meta.get("weight_format", model.quant.expert_weight_format),
                  "intermediate_dtype": gate_meta.get("output_format", model.activation_dtype),
                  "weight_scale_group_size": model.quant.expert_block,
                  "activation": dict(activation.metadata)},
            implementation={"native_kinds": (
                ["GroupedMatmulSitu"] if activation.family == "situ" else
                ["GroupedMatmulSwiglu", "GroupedMatmul"]),
                "conditions": {"scope": "expert_gate_up_activation_unquantized",
                               "down_projection_external": True}},
            source=(SITU_SOURCE if activation.family == "situ" else source_reference(
                "vllm_ascend/ops/fused_moe/moe_mlp.py", "grouped_matmul_swiglu_quant_v2")),
            coverage={"kind": "complete_subgraph",
                      "limitations": ["expected_expert_counts_are_not_observed_routing"]},
            fusion_status="unfused_decomposed",
            fusion_reason=reason,
            native_fusion_eligible=False,
        )
        return False
    group_role = ("grouped_gate_up_situ_quant" if activation.family == "situ"
                  else "grouped_gate_up_activation_quant")
    group_key = prefix + "." + group_role
    # The output quantizer is inserted by the ordinary lowering pass, and
    # belongs to the fused gate-up interface rather than the down GEMM.
    if has_quant_pass:
        ops[-1] = replace(down, metadata=down.metadata + (
            ("input_conversion_subgraphs", (group_key,)),))
    situ_mx_domain = (
        activation.family == "situ" and activation.dtype == gate_meta["output_format"] == "bf16"
        and gate_meta["activation_format"] == down_meta["activation_format"] == "fp8_e4m3"
        and gate_meta["weight_format"] == "mxfp4"
        and model.quant.expert_block == MX32_SCALE_GROUP
        and gate_meta.get("activation_group_size") == \
            down_meta.get("activation_group_size") == MX32_SCALE_GROUP
        and gate_meta.get("activation_scale_format") ==
        down_meta.get("activation_scale_format") == "fp8_e8m0"
        and tuple(gate_meta.get("activation_scale_shape", ()))[1:] == \
            (latent // MX32_SCALE_PAIR, 2)
        and tuple(down_meta.get("activation_scale_shape", ()))[1:] == \
            (expert_width // MX32_SCALE_PAIR, 2)
        and latent % MX32_SCALE_PAIR == expert_width % MX32_SCALE_PAIR == 0
        and bool(dict(activation.metadata).get("activation_linear_beta")))
    situ_native_source = {
        "repository": "vllm-ascend",
        "base_revision": "75f52c1d7df8df8548d854a01b64935727a70e88",
        "source_state": "independent_local_delivery_overlay",
        "archive": "gmmsituquant-latest-0916/gmmsituquant-latest-overlay.zip",
        "archive_sha256": "fd5ded62862cabd7b686feceadcd8f39b692bdbb1989eb93c72eea29f4fc11da",
        "path": "csrc/moe/grouped_matmul_situ_quant/grouped_matmul_situ_quant_torch_adpt.h",
        "sha256": "d1bd2db4fedebcdd075cce5a0ce0022d145719bfdad7d5c2729308471a8771d3",
        "lines": list(SITU_OVERLAY_ADAPTER_LINES), "symbol": "RunV2Core",
        "scope": "independent_source_semantics; not_capture_binary_identity",
    }
    declare_group(
        ops, key=group_key, role=group_role, family="moe",
        members=(*(g.op_id for g in gate_ops), activation.op_id),
        entries=(gate_ops[0].op_id,),
        anchor=activation.op_id, block="ffn", scope_role="body",
        view={"input_width": latent, "intermediate_width": expert_width,
              "local_experts": local_experts,
              "routing_basis": dict(routing_metadata)["routing_basis"],
              "input_dtype": gate_meta["activation_format"],
              "output_dtype": down_meta["activation_format"],
              "weight_dtype": gate_meta["weight_format"],
              "intermediate_dtype": gate_meta["output_format"],
              "weight_scale_group_size": model.quant.expert_block,
              "input_scale_dtype": gate_meta.get("activation_scale_format"),
              "input_scale_group_size": gate_meta.get("activation_group_size"),
              "input_scale_tail": tuple(gate_meta.get("activation_scale_shape", ()))[1:],
              "output_scale_dtype": down_meta.get("activation_scale_format"),
              "output_scale_group_size": down_meta.get("activation_group_size"),
              "output_scale_tail": tuple(down_meta.get("activation_scale_shape", ()))[1:],
              "activation": dict(activation.metadata)},
        implementation={"native_kinds": (
            ["GroupedMatmulSituQuant"] if activation.family == "situ" else
            ["GroupedMatmulSwigluQuant", "GroupedMatmulSwigluQuantV2"]),
            "conditions": {"scope": "expert_gate_up_activation_output_quantization",
                           "down_projection_external": True}},
        source=((situ_native_source if situ_mx_domain else SITU_SOURCE)
                if activation.family == "situ" else source_reference(
            "vllm_ascend/ops/fused_moe/moe_mlp.py", "grouped_matmul_swiglu_quant_v2")),
        coverage={"kind": ("native_source_domain_mismatch"
                           if activation.family == "situ" and not situ_mx_domain
                           else "complete_subgraph"),
                  "limitations": ["native_binary_and_scalar_flags_not_established",
                                  "expected_expert_counts_are_not_observed_routing"]},
        fusion_status=("fused_new_kernel_required"
                       if activation.family == "situ" else "fused_framework_enable"),
        fusion_reason=("situ_mx_grouped_gate_up_target_fusion"
                       if activation.family == "situ"
                       else "grouped_matmul_swiglu_quant_eligible"),
    )
    return True


def routed_expert_work(op, *, n, k, weights, weight_kind, local_rows,
                       active_experts, local_experts, local_ep_rank,
                       routing_realization, routing_metadata, balanced):
    """Scale one aggregated expert matrix to the activated group-list rows."""
    fraction = local_rows / op.local_shape[0]
    read = (op.mte2_bytes - weights) * fraction + weights
    write = op.mte3_bytes * fraction
    metadata = dict(op.metadata)
    # Aggregate expert counts do not declare per-expert tile geometry.  Keep
    # this analytical program distinct from dense/batched mapping; an explicit
    # source grouped planner still owns its actual rows.
    metadata["matrix_mapping_domain"] = "aggregate_experts"
    metadata["matrix_weight_bank_groups"] = local_experts
    metadata["routed_weight_read_bytes"] = weights
    metadata["expert_slot_weight_kind"] = weight_kind
    metadata["ep_rank"] = local_ep_rank
    if routing_realization is not None:
        metadata["matrix_mapping_domain"] = "equal_expert_groups"
        metadata["matrix_groups"] = active_experts
        metadata["matrix_group_rows"] = routing_realization.local_expert_rows
        if len(set(routing_realization.local_expert_rows) - {0}) > 1:
            metadata["matrix_mapping_domain"] = "integer_expert_groups"
    if "activation_scale_bytes" in metadata:
        metadata["activation_scale_bytes"] *= fraction
    if "matrix_epilogue" in metadata:
        epilogue = dict(metadata["matrix_epilogue"])
        epilogue["row_scale_bytes"] *= fraction
        epilogue["operations_per_output"] *= fraction
        metadata["matrix_epilogue"] = epilogue
    capacity = dict(routing_metadata).get("matrix_capacity_rows")
    if isinstance(capacity, int) and capacity > local_rows:
        # The native eager allocation is larger than the activated group-list
        # rows. Keep active matrix_m in local_shape and the allocation in
        # padded/capacity geometry; the matrix work above is charged only on
        # the activated rows.
        metadata["matrix_input_capacity_shape"] = (capacity, k)
        metadata["matrix_output_capacity_shape"] = (capacity, n)
        padded_shape = (capacity, *op.padded_shape[1:])
    else:
        padded_shape = op.padded_shape
    # Packed-weight dequantization follows the active weight bytes already
    # selected for this occurrence; only row-dependent vector work scales with
    # the activated row fraction.
    weight_dequant_ops = (float(metadata.get("dequant_ops_per_weight", 0.0))
                          * float(metadata.get("dequant_weight_elements", 0.0)))
    row_ops = max(0.0, float(op.vector_ops) - weight_dequant_ops)
    return replace(op, flops=2 * local_rows * n * k, hbm_bytes=read + write,
                   vector_ops=row_ops * fraction + weight_dequant_ops,
                   mte2_bytes=read, mte3_bytes=write,
                   padded_shape=padded_shape,
                   recipe=("balanced_" + op.recipe) if balanced else op.recipe,
                   metadata=tuple(metadata.items()) + routing_metadata)


def activation_quant_fusion(ops, source_dtype):
    """Fuse ``swiglu``/``situ``/``geglu`` + ``dynamic_quantize`` into one vector op."""
    del source_dtype
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements, removed, aliases = {}, set(), {}
    for quant in ops:
        if quant.plan_required or quant.family != "quantize" or len(quant.deps) != 1:
            continue
        act = by_id.get(quant.deps[0])
        if (act is None or act.plan_required or act.family not in {"swiglu", "situ", "geglu"}
                or float(dict(act.metadata).get("activation_limit", 0.0) or 0.0) > 0.0
                or act.stage != quant.stage or act.stream != quant.stream):
            continue
        if users[act.op_id] != [quant.op_id]:
            continue
        read = float(act.mte2_bytes)
        write = float(quant.mte3_bytes)
        meta = dict(merge_meta(act, quant, extra=(
            ("fused_from", (act.family, "quantize")),
            ("primary_output", "payload"),
            ("fused_kernel", "npu_swiglu_mx_quant"),
            ("row_program", {"aiv_units": SWIGLU_MX_AIV_UNITS,
                             "group_size": SWIGLU_MX_GROUP,
                             "stages": SWIGLU_MX_STAGES}),
        )))
        append_audit(
            meta, kind="transform_boundary", op_id=quant.op_id,
            boundary={
                "kind": "activation_quant",
                "replaces_work": True, "replaces_allocation": True,
                "replaced_occurrences": (act.op_id, quant.op_id),
                "replacement_occurrence": quant.op_id,
                "work": {"mte2_bytes": read, "mte3_bytes": write,
                         "hbm_bytes": read + write,
                         "vector_ops": float(act.vector_ops) + float(quant.vector_ops),
                         "flops": 0.0},
            },
            layer=quant.layer, stage=quant.stage,
        )
        replacements[quant.op_id] = replace(
            quant,
            deps=act.deps,
            family="quantize",
            recipe=f"{act.family}_quantize",
            vector_ops=float(act.vector_ops) + float(quant.vector_ops),
            mte2_bytes=read,
            mte3_bytes=write,
            hbm_bytes=read + write,
            metadata=tuple(meta.items()),
        )
        removed.add(act.op_id)
        aliases[act.op_id] = quant.op_id
    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    rewired = rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=NO_PORTS,
    )
    return attach_quantized_ports(rewired, replacements)


def gmm_swiglu_quant_fusion(ops, source_dtype):
    """Keep the ``grouped_gemm`` activation payload on chip for the fused quant epilogue.

    The gate/up matrix phase stops writing its ``[M, 2*I]`` payload to HBM and the
    activation+quantize phase stops reading it back, while every Cube FLOP,
    scale epilogue and Vector OP is preserved.  A single submitted native kernel
    additionally requires a source-verified ``declared_native_fusion`` plan
    binding; this pass never fabricates that binding, so it models the traffic
    elimination without claiming an unverified launch reduction.
    """
    ops = activation_quant_fusion(ops, source_dtype)
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements: dict = {}
    for tail in ops:
        if (tail.plan_required or not str(tail.recipe).endswith("_quantize")
                or tail.family != "quantize" or len(tail.deps) != 1):
            continue
        gate = by_id.get(tail.deps[0])
        if (gate is None or gate.plan_required or gate.family != "grouped_gemm"
                or gate.stage != tail.stage or gate.stream != tail.stream):
            continue
        if users[gate.op_id] != [tail.op_id]:
            continue
        if gate.op_id in replacements or tail.op_id in replacements:
            continue
        gate_meta = dict(gate.metadata)
        act_dtype = str(gate_meta.get("output_format") or tail.dtype or gate.dtype)
        if len(gate.local_shape) >= 2:
            intermediate_bytes = float(
                gate.local_shape[0] * gate.local_shape[1] * dtype_bytes(act_dtype)
            )
        else:
            intermediate_bytes = min(float(gate.mte3_bytes), float(tail.mte2_bytes))
        eliminated_write = min(float(gate.mte3_bytes), intermediate_bytes)
        eliminated_read = min(float(tail.mte2_bytes), intermediate_bytes)
        new_gate_mte3 = max(0.0, float(gate.mte3_bytes) - eliminated_write)
        new_gate_hbm = max(0.0, float(gate.hbm_bytes) - eliminated_write)
        new_tail_mte2 = max(0.0, float(tail.mte2_bytes) - eliminated_read)
        new_tail_hbm = max(0.0, float(tail.hbm_bytes) - eliminated_read)
        replacements[gate.op_id] = replace(
            gate,
            mte3_bytes=new_gate_mte3,
            hbm_bytes=new_gate_hbm,
            metadata=merge_meta(gate, extra=(
                ("onchip_handoff_to", tail.op_id),
                ("onchip_handoff_boundary", {
                    "implementation": "declared_onchip_handoff",
                    "native": "GroupedMatmulSwigluQuantOnChip",
                    "submitted_kernels": 2,
                    "semantic_phase_count": 2,
                    "eliminated_activation_write_bytes": eliminated_write,
                    "timing_basis": "resource_DAG_not_sum_of_kernel_durations",
                    "launch_reduction_claimed": False,
                }),
            )),
        )
        replacements[tail.op_id] = replace(
            tail,
            mte2_bytes=new_tail_mte2,
            hbm_bytes=new_tail_hbm,
            metadata=merge_meta(tail, extra=(
                ("onchip_handoff_from", gate.op_id),
                ("eliminated_activation_read_bytes", eliminated_read),
                ("fused_from", ("grouped_gemm", tail.recipe)),
            )),
        )
    if not replacements:
        return list(ops)
    return [replacements.get(op.op_id, op) for op in ops]


def moe_tail_residual_fusion(ops, source_dtype):
    """Fold MoE ``shared + routed`` elementwise combine into post-MoE ``residual_add`` (PR #21999)."""
    del source_dtype
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements, removed, aliases = {}, set(), {}
    for res in ops:
        if res.recipe != "residual_add" or len(res.deps) != 2:
            continue
        comb = by_id.get(res.deps[0])
        if (comb is None or comb.plan_required or comb.family not in {"elementwise", "add"}
                or comb.recipe in {"residual_add"} or users[comb.op_id] != [res.op_id]
                or comb.stage != res.stage or comb.stream != res.stream
                or math.prod(comb.local_shape) != math.prod(res.local_shape)):
            continue
        # 3-way add (shared + routed + saved_residual): eliminates intermediate [T, H] write+read
        fused_read = max(0.0, float(comb.mte2_bytes) + float(res.mte2_bytes) - float(comb.mte3_bytes))
        fused_write = float(res.mte3_bytes)
        replacements[res.op_id] = replace(
            res,
            deps=tuple(dict.fromkeys((*comb.deps, res.deps[1]))),
            vector_ops=float(comb.vector_ops) + float(res.vector_ops),
            mte2_bytes=fused_read,
            mte3_bytes=fused_write,
            hbm_bytes=fused_read + fused_write,
            metadata=merge_meta(comb, res, extra=(("fused_moe_tail_add", True),)),
        )
        removed.add(comb.op_id)
        aliases[comb.op_id] = res.op_id
    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    return rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=NO_PORTS,
    )


def bind_transport_gate(gate, down, *, transport, latent, local_rows, payload, scale):
    """Bind one routed-expert matrix to its declared transport representation.

    The receive-side conversion must publish exactly the payload and scale
    representation the expert gate-up matrix reads, and the first expert
    branch must return BF16 rows for the declared combine.
    """
    gate_meta = dict(gate.metadata)
    expected_rows = math.ceil(local_rows)
    shape = tuple(gate_meta.get("activation_scale_shape", ()))
    if transport.mxfp8:
        tail = 1
        for extent in shape[1:]:
            tail *= int(extent)
        needed = math.ceil(latent / max(1, transport.scale_group))
        payload_ok = gate_meta.get("activation_format") == transport.payload_dtype
    elif transport.fp8:
        payload_ok = (
            gate_meta.get("activation_format") == transport.payload_dtype
            and (shape == () or (
                shape == (expected_rows,)
                and gate_meta.get("activation_scale_format", "fp32") in {"fp32", "float32"}
                and gate_meta.get("activation_group_size", latent) >= latent)))
    elif transport.quantized:
        payload_ok = (
            gate_meta.get("activation_format") == transport.payload_dtype
            and gate_meta.get("activation_scale_format", "fp32") in {"fp32", "float32"}
            and gate_meta.get("activation_group_size", latent) >= latent
            and shape == (expected_rows,))
    else:
        payload_ok = gate_meta.get("activation_format") == transport.payload_dtype
    if transport.mxfp8:
        payload_ok = bool(
            payload_ok and shape[:1] == (expected_rows,)
            and gate_meta.get("activation_scale_format") in {"fp8_e8m0", "e8m0"}
            and gate_meta.get("activation_group_size") == transport.scale_group
            and tail >= needed)
    if not payload_ok:
        raise ModelingError("moe_transport_gate_requires_matching_quantized_payload")
    if dict(down.metadata).get("output_format") != "bf16":
        raise ModelingError("moe_transport_initial_branch_requires_bf16_expert_output")
    return replace(gate, metadata=gate.metadata + (
        ("activation_producer", "explicit_transport_quantization"),
        ("activation_payload_producer", payload),
        ("activation_scale_producer", scale),
        ("moe_transport_kind", transport.kind),
        ("moe_transport_dispatch_dtype", transport.dispatch_dtype),
    ))


def bind_expert_slots(ops, *, model, plan, parallel, routing, runtime_routing,
                      routing_realization, group_tokens, expert_width, latent):
    """Bind each routed matrix's active weight read to its physical slot plan."""
    _, slot_routing = split_routing_features(runtime_routing)
    slot_plan = expert_slot_plan(model, plan, parallel, routing=slot_routing)
    declared = tuple(getattr(routing_realization, "expert_rows", ()) or ())
    if declared and len(declared) == int(plan.num_experts):
        assignments = sum(float(value) for value in declared)
        denominator = assignments / max(1, int(plan.top_k))
        inclusions = tuple(
            float(value) / denominator if denominator > 0.0 else 0.0
            for value in declared)
    else:
        rows, _active = routing.expert_moments(model, plan, int(group_tokens))
        denominator = float(group_tokens)
        inclusions = tuple(
            float(value) / denominator if denominator > 0.0 else 0.0
            for value in rows)
    experts = max(1, int(plan.num_experts))
    gate_unit = float(model.quant.matrix_bytes(
        2 * expert_width, latent, "routed_expert"))
    down_unit = float(model.quant.matrix_bytes(
        latent, expert_width, "routed_expert"))
    gate_work = physical_slot_work(
        slot_plan=slot_plan, logical_inclusions=inclusions,
        global_tokens=int(group_tokens),
        packed_weight_bytes=tuple([gate_unit] * experts))
    down_work = physical_slot_work(
        slot_plan=slot_plan, logical_inclusions=inclusions,
        global_tokens=int(group_tokens),
        packed_weight_bytes=tuple([down_unit] * experts))
    kinds = {
        op.op_id: str(dict(op.metadata).get("expert_slot_weight_kind", ""))
        for op in ops
    }
    gate_bound = {
        op.op_id: op for op in bind_expert_slot_work(
            tuple(op for op in ops if kinds[op.op_id] == "gate_up"),
            slot_work=gate_work)
    }
    down_bound = {
        op.op_id: op for op in bind_expert_slot_work(
            tuple(op for op in ops if kinds[op.op_id] == "down"),
            slot_work=down_work)
    }
    return [gate_bound.get(op.op_id, down_bound.get(op.op_id, op)) for op in ops]
