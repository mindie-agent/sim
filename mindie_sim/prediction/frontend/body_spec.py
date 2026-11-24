"""Model body block lowering, boundary descriptors and padding specs.

[Role] Own layer block lowering, stage boundaries, and body-level graph framing.
[Invariants] Conserves layer ordering and exact token padding contracts.
[Timeouts & Locks] Pure graph construction; no I/O or locks.
[Side-Effect Policy] Pure computation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from mindie_sim.schema.operators import OpSpec


def expand_call(op, kernels):
    children = tuple(kernels(op))
    ids = {child.op_id: op.op_id + "/" + child.op_id for child in children}
    meta = dict(op.metadata)
    semantic = {**meta.get("semantic_metadata", {}), **meta}
    scope = {key: semantic[key] for key in ("structure_block", "structure_role")
             if key in semantic}
    inherited = {key: value for key, value in op.metadata
                 if key in {"graph_captured", "graph_segment"}}
    # A fused source call owns one actual invocation.  Propagate the shared
    # membership and timing owner into every child so a later binding of a
    # child cannot mint its own call/charge.  Identity itself stays derived.
    declared = meta.get("native_call")
    shared_call = None
    if isinstance(declared, dict) and len(declared.get("kernels") or ()) > 1:
        shared_call = {
            **declared,
            "members": {
                "op_ids": [ids[child.op_id] for child in children],
                "charge": "call",
                "charge_note": "priced once at timing_owner.op_id",
            },
            "timing_owner": {"op_id": op.op_id, "scope": "call"},
        }

    def child_metadata(child):
        metadata = {**inherited, **dict(child.metadata), **scope}
        if shared_call is not None:
            metadata["native_call"] = shared_call
        return tuple(metadata.items())

    expanded = tuple(replace(
        child, op_id=ids[child.op_id],
        deps=tuple(dict.fromkeys((*op.deps, *(ids[dep] for dep in child.deps)))),
        stage=op.stage, layer=op.layer, requests=op.requests, phase=op.phase,
        stream=child.stream or op.stream,
        metadata=child_metadata(child),
    ) for child in children)
    consumed = {dep for child in children for dep in child.deps}
    terminal = tuple(ids[child.op_id] for child in children if child.op_id not in consumed)
    join = OpSpec(
        op.op_id, "barrier", op.phase, op.stage, op.layer, op.requests,
        op.semantic_shape, op.local_shape, op.padded_shape, op.launch_shape,
        deps=terminal or op.deps, stream=op.stream,
        metadata=(("expanded_implementation", op.recipe), *scope.items()),
    )
    return (*expanded, join)


from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import norm_role
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.prediction.structure_groups import declare_group, source_reference

_RMSNORM_OPS_PER_ELEMENT = 5
_HC_AFFINE_OPS_PER_ELEMENT = 5
_HC_INVERSE_RMS_FINALIZE_OPS = 3
_HC_AFFINE_SCALAR_PARAMETERS = 3
_HC_PROJECTION_ALIGNMENT = 16
_HC_SILU_OPS_PER_ELEMENT = 4
_HC_GATE_MIX_OPS_PER_STREAM = 4
_HC_COMBINE_OPS_PER_ELEMENT = 4
_RESIDUAL_SOFTMAX_OPS_PER_ELEMENT = 5
_FP32_BYTES = 4
FINAL_LAYER = -1

# Qwen4Exp's Ascend HC kernels are not yet in structure_groups' source
# registry; this release consumes the local worktree source directly.
_QWEN4_HC_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "local_worktree",
    "sha256": "28c6c8c6f26ff6c92023334eaed86d3c8606200dd27a67446cef1d624a0c1f91",
    "path": "vllm_ascend/ops/triton/qwen4_exp/hc.py",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def _qwen4_hc_source(symbol):
    return {**_QWEN4_HC_SOURCE, "symbol": symbol}


def norm(*, name, deps, stage, layer, requests, tokens, width, model, mode):
    semantics = dict(model.catalog_facts).get("normalization", {})
    act_bytes = dtype_bytes(model.activation_dtype)
    role = norm_role(name)
    quant = getattr(model, "quant", None)
    if quant is not None:
        gamma_runtime = quant.norm_gamma_bytes(width, role)
        gamma_storage = quant.norm_gamma_bytes(width, role, source=True)
        gamma_format = quant.norm_gamma_format(role)
        gamma_storage_format = quant.norm_gamma_format(role, source=True)
    else:
        gamma_runtime = gamma_storage = width * act_bytes
        gamma_format = gamma_storage_format = model.activation_dtype
    return vector_op(
        op_id=name, deps=tuple(deps), stage=stage, layer=layer, requests=requests,
        elements=tokens * width, ops_per_element=_RMSNORM_OPS_PER_ELEMENT, dtype=model.activation_dtype,
        family="rmsnorm", phase=mode, recipe="rmsnorm", shape=(tokens, width),
        read_bytes_override=tokens * width * act_bytes + gamma_runtime,
        write_bytes_override=tokens * width * act_bytes,
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("norm_eps", model.norm_eps),
                  ("norm_gamma_role", role),
                  ("norm_gamma_runtime_bytes", gamma_runtime),
                  ("norm_gamma_storage_bytes", gamma_storage),
                  ("norm_gamma_runtime_format", gamma_format),
                  ("norm_gamma_storage_format", gamma_storage_format)) + tuple(
                      (key, semantics[key]) for key in
                      ("norm_weight_offset", "norm_multiply_before_output_cast")
                      if key in semantics),
    )


class Residual:
    def before(self, context, ready):
        return [], ready, ready

    def after(self, context, output, saved):
        c = context
        act_bytes = dtype_bytes(c["model"].activation_dtype)
        op = vector_op(
            op_id=c["prefix"] + ".residual", deps=(output, saved), stage=c["stage"],
            layer=c["plan"].index, requests=c["requests"], elements=c["tokens"] * c["model"].hidden_size,
            ops_per_element=1, dtype=c["model"].activation_dtype,
            family="elementwise", phase=c["mode"], recipe="residual_add",
            read_bytes_override=c["tokens"] * c["model"].hidden_size * 2 * act_bytes,
            write_bytes_override=c["tokens"] * c["model"].hidden_size * act_bytes,
        )
        return [op], op.op_id


class HyperConnection:
    def before(self, context, ready):
        c = context
        plan, model = c["plan"], c["model"]
        h, hc, tokens = model.hidden_size, plan.hc_mult, c["tokens"]
        mixed = (2 + hc) * hc
        common = dict(stage=c["stage"], layer=plan.index, requests=c["requests"], phase=c["mode"])
        act_bytes = dtype_bytes(model.activation_dtype)
        values = vector_op(
            op_id=c["prefix"] + ".hc_values_fp32", deps=(ready,),
            elements=tokens * hc * h, ops_per_element=1, dtype="fp32",
            family="cast", recipe="hc_values_fp32", shape=(tokens, hc * h),
            read_bytes_override=tokens * hc * h * act_bytes,
            write_bytes_override=tokens * hc * h * _FP32_BYTES, **common,
        )
        project = gemm_op(
            op_id=c["prefix"] + ".hc_project", deps=(values.op_id,), m=tokens, n=mixed, k=hc * h,
            quant=model.quant, category="mhc", recipe="hc_projection", family="gemm",
            activation_dtype="fp32", output_dtype="fp32", **common,
        )
        inverse_rms = vector_op(
            op_id=c["prefix"] + ".hc_inv_rms", deps=(values.op_id,),
            elements=tokens * hc * h, ops_per_element=2, dtype="fp32",
            vector_ops_override=tokens * (2 * hc * h + _HC_INVERSE_RMS_FINALIZE_OPS),
            family="mhc", recipe="hc_inverse_rms", shape=(tokens,),
            read_bytes_override=tokens * hc * h * _FP32_BYTES,
            write_bytes_override=tokens * _FP32_BYTES, **common,
        )
        affine = vector_op(
            op_id=c["prefix"] + ".hc_affine", deps=(project.op_id, inverse_rms.op_id),
            elements=tokens * mixed, ops_per_element=_HC_AFFINE_OPS_PER_ELEMENT, dtype="fp32", family="mhc",
            recipe="hc_normalize_scale_bias_gate", shape=(tokens, mixed),
            read_bytes_override=(tokens * (mixed + 1) + mixed + _HC_AFFINE_SCALAR_PARAMETERS) * _FP32_BYTES,
            write_bytes_override=tokens * mixed * _FP32_BYTES, **common,
        )
        sinkhorn = vector_op(
            op_id=c["prefix"] + ".hc_sinkhorn", deps=(affine.op_id,), elements=tokens * mixed,
            ops_per_element=4 * plan.hc_sinkhorn_iters, dtype="fp32", family="mhc",
            recipe="hc_sinkhorn", returns=("post", "comb"),
            read_bytes_override=tokens * mixed * _FP32_BYTES, **common,
        )
        mix = vector_op(
            op_id=c["prefix"] + ".hc_mix", deps=(ready, sinkhorn.op_id), elements=tokens * h,
            ops_per_element=2 * hc, dtype=model.activation_dtype, family="mhc",
            recipe="hc_pre_mix", read_bytes_override=tokens * hc * (h * act_bytes + _FP32_BYTES),
            write_bytes_override=tokens * h * act_bytes, **common,
        )
        ops = [values, project, inverse_rms, affine, sinkhorn, mix]
        declare_group(
            ops, key=c["prefix"] + ".hc_pre", role="hc_pre", family="mhc",
            members=tuple(op.op_id for op in ops), entries=(values.op_id,),
            anchor=mix.op_id, block=c.get("structure_block"), scope_role="residual",
            view={"tokens": tokens, "hidden": h, "hc": hc, "mixed": mixed,
                  "dtype": model.activation_dtype, "parameter_dtype": "fp32",
                  "sinkhorn_iters": plan.hc_sinkhorn_iters},
            implementation={"native_kind": "HcPre",
                            "conditions": {"scope": "projection_inverse_rms_affine_sinkhorn_mix"}},
            source=source_reference("vllm_ascend/models/deepseek_v4/model.py", "DeepseekV4DecoderLayer.hc_pre"),
            coverage={"kind": "complete_subgraph",
                      "limitations": ["epsilon_and_sinkhorn_scalar_not_observed"]},
        )
        return ops, mix.op_id, (ready, sinkhorn.op_id)

    def after(self, context, output, saved):
        c = context
        h, hc, tokens = c["model"].hidden_size, c["plan"].hc_mult, c["tokens"]
        act_bytes = dtype_bytes(c["model"].activation_dtype)
        op = vector_op(
            op_id=c["prefix"] + ".hc_post", deps=(output, *saved), stage=c["stage"],
            layer=c["plan"].index, requests=c["requests"], elements=tokens * hc * h,
            ops_per_element=2 * hc + 1, dtype=c["model"].activation_dtype,
            family="mhc", phase=c["mode"], recipe="hc_post_mix",
            read_bytes_override=tokens * (h * act_bytes + hc * h * act_bytes + hc * _FP32_BYTES + hc * hc * _FP32_BYTES),
            write_bytes_override=tokens * hc * h * act_bytes,
        )
        ops = [op]
        declare_group(
            ops, key=c["prefix"] + ".hc_post_group", role="hc_post", family="mhc",
            members=(op.op_id,), entries=(op.op_id,), anchor=op.op_id,
            block=c.get("structure_block"), scope_role="residual",
            view={"tokens": tokens, "hidden": h, "hc": hc,
                  "dtype": c["model"].activation_dtype, "coefficient_dtype": "fp32"},
            implementation={"native_kind": "HcPost"},
            source=source_reference("vllm_ascend/models/deepseek_v4/model.py", "DeepseekV4DecoderLayer.hc_post"),
            coverage={"kind": "complete_subgraph", "limitations": []},
        )
        return ops, op.op_id


class LowRankHyperConnection:
    """Qwen4Exp GatedResidual: grouped Gemma norm, low-rank injections."""

    def before(self, context, ready):
        c = context
        plan, model = c["plan"], c["model"]
        h = model.hidden_size
        hc = max(1, int(plan.hc_mult or 1))
        lr = int(getattr(plan, "hc_lowrank", 0) or 0)
        tokens = c["tokens"]
        hc_dim = hc * h
        act_bytes = dtype_bytes(model.activation_dtype)
        down_out = ((lr + hc + _HC_PROJECTION_ALIGNMENT - 1)
                    // _HC_PROJECTION_ALIGNMENT) * _HC_PROJECTION_ALIGNMENT
        common = dict(stage=c["stage"], layer=plan.index,
                      requests=c["requests"], phase=c["mode"])
        hc_norm = vector_op(
            op_id=c["prefix"] + ".hc_norm", deps=(ready,),
            elements=tokens * hc_dim, ops_per_element=_RMSNORM_OPS_PER_ELEMENT,
            dtype=model.activation_dtype, family="rmsnorm",
            recipe="grouped_gemma_rmsnorm", shape=(tokens, hc_dim),
            read_bytes_override=(tokens * hc_dim + hc_dim) * act_bytes,
            write_bytes_override=tokens * hc_dim * act_bytes, **common)
        hc_down_inject = gemm_op(
            op_id=c["prefix"] + ".hc_down_inject", deps=(hc_norm.op_id,),
            m=tokens, n=down_out, k=hc_dim, quant=model.quant, category="linear_param_bf16",
            recipe="hc_lowrank_down_inject", family="gemm", **common)
        hc_silu = vector_op(
            op_id=c["prefix"] + ".hc_silu", deps=(hc_down_inject.op_id,),
            elements=tokens * lr, ops_per_element=_HC_SILU_OPS_PER_ELEMENT,
            dtype=model.activation_dtype, family="activation",
            recipe="hc_silu", shape=(tokens, lr),
            read_bytes_override=tokens * lr * act_bytes,
            write_bytes_override=tokens * lr * act_bytes, **common)
        hc_up = gemm_op(
            op_id=c["prefix"] + ".hc_up", deps=(hc_silu.op_id,),
            m=tokens, n=hc_dim, k=lr, quant=model.quant, category="linear_param_bf16",
            recipe="hc_lowrank_up", family="gemm", **common)
        hc_gate_mix = vector_op(
            op_id=c["prefix"] + ".hc_gate_mix", deps=(hc_norm.op_id, hc_up.op_id),
            elements=tokens * h, ops_per_element=_HC_GATE_MIX_OPS_PER_STREAM * hc,
            dtype=model.activation_dtype, family="mhc",
            recipe="hc_gate_mix", shape=(tokens, h),
            read_bytes_override=2 * tokens * hc_dim * act_bytes,
            write_bytes_override=tokens * h * act_bytes, **common)
        ops = [hc_norm, hc_down_inject, hc_silu, hc_up, hc_gate_mix]
        declare_group(
            ops, key=c["prefix"] + ".hc_pre", role="hc_pre", family="mhc",
            members=tuple(op.op_id for op in ops), entries=(hc_norm.op_id,),
            anchor=hc_gate_mix.op_id, block=c.get("structure_block"),
            scope_role="residual",
            view={"tokens": tokens, "hidden": h, "hc": hc,
                  "hc_lowrank": lr, "dtype": model.activation_dtype},
            implementation={"native_kind": "Qwen4ExpGatedResidualMix"},
            source=_qwen4_hc_source("hc_gate_mix"),
            coverage={"kind": "complete_subgraph", "limitations": []},
        )
        return ops, hc_gate_mix.op_id, (ready, hc_down_inject.op_id)

    def after(self, context, output, saved):
        c = context
        h = c["model"].hidden_size
        hc = max(1, int(c["plan"].hc_mult or 1))
        tokens = c["tokens"]
        hc_dim = hc * h
        act_bytes = dtype_bytes(c["model"].activation_dtype)
        op = vector_op(
            op_id=c["prefix"] + ".hc_post", deps=(output, *saved),
            stage=c["stage"], layer=c["plan"].index, requests=c["requests"],
            elements=tokens * hc_dim, ops_per_element=_HC_COMBINE_OPS_PER_ELEMENT,
            dtype=c["model"].activation_dtype, family="mhc", phase=c["mode"],
            recipe="hc_combine", shape=(tokens, hc_dim),
            read_bytes_override=tokens * (hc_dim + h + hc) * act_bytes,
            write_bytes_override=tokens * hc_dim * act_bytes)
        ops = [op]
        declare_group(
            ops, key=c["prefix"] + ".hc_post_group", role="hc_post",
            family="mhc", members=(op.op_id,), entries=(op.op_id,),
            anchor=op.op_id, block=c.get("structure_block"),
            scope_role="residual",
            view={"tokens": tokens, "hidden": h, "hc": hc,
                  "dtype": c["model"].activation_dtype},
            implementation={"native_kind": "Qwen4ExpHcCombine"},
            source=_qwen4_hc_source("hc_combine"),
            coverage={"kind": "complete_subgraph", "limitations": []},
        )
        return ops, op.op_id


def lowrank_hyper_final_mixer(*, prefix, deps, stage, requests, tokens, model, mode):
    """Collapse the multi-stream residual to hidden before ``lm_head``."""
    hc = max((p.hc_mult for p in model.layer_plans), default=1)
    lr = max((getattr(p, "hc_lowrank", 0) for p in model.layer_plans),
             default=1)
    h = model.hidden_size
    hc_dim = hc * h
    act_bytes = dtype_bytes(model.activation_dtype)
    ready = deps if isinstance(deps, tuple) else (deps,)
    common = dict(stage=stage, layer=FINAL_LAYER, requests=requests, phase=mode)
    hc_head_norm = vector_op(
        op_id=prefix + ".hc_head_norm", deps=ready,
        elements=tokens * hc_dim, ops_per_element=_RMSNORM_OPS_PER_ELEMENT,
        dtype=model.activation_dtype, family="rmsnorm",
        recipe="grouped_gemma_rmsnorm", shape=(tokens, hc_dim),
        read_bytes_override=(tokens * hc_dim + hc_dim) * act_bytes,
        write_bytes_override=tokens * hc_dim * act_bytes, **common)
    hc_head_down = gemm_op(
        op_id=prefix + ".hc_head_down", deps=(hc_head_norm.op_id,),
        m=tokens, n=lr, k=hc_dim, quant=model.quant, category="linear_param_bf16",
        recipe="hc_lowrank_head_down", family="gemm", **common)
    hc_head_silu = vector_op(
        op_id=prefix + ".hc_head_silu", deps=(hc_head_down.op_id,),
        elements=tokens * lr, ops_per_element=_HC_SILU_OPS_PER_ELEMENT,
        dtype=model.activation_dtype, family="activation", recipe="hc_silu",
        shape=(tokens, lr), read_bytes_override=tokens * lr * act_bytes,
        write_bytes_override=tokens * lr * act_bytes, **common)
    hc_head_up = gemm_op(
        op_id=prefix + ".hc_head_up", deps=(hc_head_silu.op_id,),
        m=tokens, n=hc_dim, k=lr, quant=model.quant, category="linear_param_bf16",
        recipe="hc_lowrank_head_up", family="gemm", **common)
    hc_head_gate_mix = vector_op(
        op_id=prefix + ".hc_head_gate_mix", deps=(hc_head_norm.op_id, hc_head_up.op_id),
        elements=tokens * h, ops_per_element=_HC_GATE_MIX_OPS_PER_STREAM * hc,
        dtype=model.activation_dtype, family="mhc", recipe="hc_gate_mix",
        shape=(tokens, h), read_bytes_override=2 * tokens * hc_dim * act_bytes,
        write_bytes_override=tokens * h * act_bytes, **common)
    ops = [hc_head_norm, hc_head_down, hc_head_silu, hc_head_up, hc_head_gate_mix]
    return ops, hc_head_gate_mix.op_id


class AttentionResidual(Residual):
    def before(self, context, ready):
        c = context
        model, tokens = c["model"], c["tokens"]
        sources = tuple(dict.fromkeys((*c.get("residual_sources", ()), ready)))
        common = dict(stage=c["stage"], layer=c["plan"].index,
                      requests=c["requests"], phase=c["mode"])
        elements = tokens * len(sources) * model.hidden_size
        act_bytes = dtype_bytes(model.activation_dtype)
        concat = vector_op(
            op_id=c["prefix"] + ".residual_concat", deps=sources,
            elements=elements, ops_per_element=1, dtype=model.activation_dtype,
            family="tensor_copy", recipe="attention_residual_concat",
            shape=(tokens, len(sources), model.hidden_size),
            read_bytes_override=elements * act_bytes,
            write_bytes_override=elements * act_bytes, **common)
        values = vector_op(
            op_id=c["prefix"] + ".residual_values_fp32", deps=(concat.op_id,),
            elements=elements, ops_per_element=1, dtype="fp32", family="cast",
            recipe="residual_values_fp32", shape=(tokens, len(sources), model.hidden_size),
            read_bytes_override=elements * act_bytes,
            write_bytes_override=elements * _FP32_BYTES, **common)
        normalized = norm(
            name=c["prefix"] + ".residual_norm", deps=(values.op_id,), stage=c["stage"],
            layer=c["plan"].index, requests=c["requests"], tokens=tokens * len(sources),
            width=model.hidden_size, model=replace(model, activation_dtype="fp32"), mode=c["mode"])
        scores = gemm_op(
            op_id=c["prefix"] + ".residual_scores", deps=(normalized.op_id,),
            stage=c["stage"], layer=c["plan"].index, requests=c["requests"],
            m=tokens * len(sources), n=1, k=model.hidden_size, quant=model.quant,
            category="param_fp32", phase=c["mode"], recipe="attention_residual_scores", family="gemm")
        probabilities = vector_op(
            op_id=c["prefix"] + ".residual_probabilities", deps=(scores.op_id,),
            elements=tokens * len(sources), ops_per_element=_RESIDUAL_SOFTMAX_OPS_PER_ELEMENT, dtype="fp32",
            family="softmax", recipe="attention_residual_softmax",
            shape=(tokens, len(sources)), **common)
        mix = vector_op(
            op_id=c["prefix"] + ".residual_mix", deps=(probabilities.op_id, values.op_id),
            stage=c["stage"], layer=c["plan"].index, requests=c["requests"],
            elements=tokens * model.hidden_size * len(sources), ops_per_element=2,
            dtype="fp32", family="elementwise", phase=c["mode"],
            recipe="attention_residual_mix",
            read_bytes_override=tokens * len(sources) * (model.hidden_size + 1) * _FP32_BYTES,
            write_bytes_override=tokens * model.hidden_size * _FP32_BYTES)
        cast = vector_op(
            op_id=c["prefix"] + ".residual_output_cast", deps=(mix.op_id,),
            elements=tokens * model.hidden_size, ops_per_element=1,
            dtype=model.activation_dtype, family="cast", recipe="attention_residual_output_cast",
            read_bytes_override=tokens * model.hidden_size * _FP32_BYTES,
            write_bytes_override=tokens * model.hidden_size * act_bytes,
            **common)
        ops = [concat, values, normalized, scores, probabilities, mix, cast]
        features = c.get("features")
        rt_meta = dict(getattr(features, "metadata", ()) or ()) if features is not None else {}
        spine = rt_meta.get("single_spine") if isinstance(rt_meta.get("single_spine"), Mapping) else {}
        track_spec = spine.get("track_spec") if isinstance(spine.get("track_spec"), Mapping) else {}
        track = str(track_spec.get("track") or "").strip().lower()
        is_eager = track == "eager" or bool(rt_meta.get("raw_eager"))
        arch_fam = str(rt_meta.get("arch_family") or "").strip().upper()
        attn_res_fused = (not is_eager) and arch_fam in {"A3", "A5"}
        declare_group(
            ops, key=c["prefix"] + ".attention_residual_group",
            role="attn_res_prefix_point", family="attn_res_prefix",
            members=tuple(op.op_id for op in ops), entries=(concat.op_id,),
            anchor=cast.op_id, block=c.get("structure_block"), scope_role="residual",
            view={"tokens": tokens, "hidden": model.hidden_size, "sources": len(sources),
                  "dtype": model.activation_dtype, "compute_dtype": "fp32"},
            implementation={"native_kind": "AttnResFwd",
                            "conditions": {"scope": "normalized_learned_residual_mixture"}},
            source=source_reference("vllm_ascend/models/kimi_k3.py", "_apply_attention_residual"),
            coverage={"kind": "complete_subgraph",
                      "limitations": ["native_residual_slot_order_requires_descriptor_evidence"]},
            fusion_status=("fused_existing" if attn_res_fused else "unfused_decomposed"),
            fusion_reason=(
                "shipping_attn_res_fwd_kernel"
                if attn_res_fused else "decomposed_attention_residual_fp32_chain"
            ),
            native_fusion_eligible=not is_eager,
        )
        return ops, cast.op_id, ready


RESIDUAL_BUILDERS = {"plain": Residual(), "hyper": HyperConnection(),
                     "lowrank_hyper": LowRankHyperConnection(),
                     "attention": AttentionResidual()}


import hashlib
import json
from collections.abc import Mapping

from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


#: Structural catalog reuse. The declared target graph is a pure function of
#: the case declaration and the probe geometry: no clock, request history or
#: measured value enters it. One probe of the same declaration re-executes every
#: operator through the Engine; only repeated frontend construction is skipped.
_GRAPH_CACHE: dict = {}
_GRAPH_CACHE_LIMIT = 2


def _freeze_key(value):
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key(v) for v in value)
    return value


def _declaration_key(case, geometry):
    if isinstance(case, Mapping):
        payload = json.dumps(case, sort_keys=True, default=repr,
                             ensure_ascii=False)
    else:
        payload = repr(case)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest, _freeze_key(geometry)


def _declared_graphs(case, *, phase, tokens, context, sequences, scope,
                     apply_body_defaults, padded_idle):
    key = _declaration_key(case, {
        "phase": phase, "tokens": tokens, "context": context,
        "sequences": sequences, "scope": scope,
        "defaults": apply_body_defaults, "padded_idle": padded_idle})
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    parsed = parse_case(case)
    from mindie_sim.prediction.operator_analysis import target_graphs
    graphs = target_graphs(
        parsed, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope,
        apply_body_defaults=apply_body_defaults, padded_idle=padded_idle)
    while len(_GRAPH_CACHE) >= _GRAPH_CACHE_LIMIT:
        _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
    _GRAPH_CACHE[key] = (parsed, graphs)
    return parsed, graphs


def physical_global_ranks(runtime, replicas):
    """Global rank scope for DP replicas under the declared DP-outer/TP-inner layout.

    The value is a scope, not an observed-owner substitution: W18F/W19F select
    the actual TP member (for example TP0 or TP2) from source provenance.
    """
    tp = max(1, int(runtime.parallel.tp))
    dp = max(1, int(runtime.parallel.dp))
    out = {}
    for replica in replicas:
        replica = int(replica)
        if replica < 0 or replica >= dp:
            raise ModelingError(f"body_replica_outside_declared_dp:{replica}:{dp}")
        out[replica] = tuple(replica * tp + tp_rank for tp_rank in range(tp))
    return out


def describe_included_components(case, *, scope: str = "body") -> dict[str, object]:
    """Return explicit component provenance so callers see body vs model/draft coverage."""
    parsed = parse_case(case)
    is_model_scope = str(scope) == "model"
    aux = getattr(parsed.model, "auxiliary", None)
    meta = dict(getattr(parsed.runtime, "metadata", ()) or ())
    proposals = int(meta.get("draft_proposal_tokens") or getattr(parsed.runtime, "mtp_proposal_tokens", 0) or 0)
    draft_layers = 0
    draft_kind: str | None = None
    if is_model_scope and proposals > 0 and aux is not None:
        draft_mode = str(meta.get("draft_mode") or "").strip().lower()
        candidates = (draft_mode, "dspark", "mtp", "eagle3") if draft_mode else ("dspark", "mtp", "eagle3")
        for candidate in dict.fromkeys(candidates):
            spec = getattr(aux, candidate, None)
            if spec is not None:
                draft_layers = int(getattr(spec, "layers", 0) or 0)
                draft_kind = candidate
                break
    return {
        "scope": str(scope),
        "target_layers": int(parsed.model.layers),
        "draft_layers": draft_layers,
        "draft_kind": draft_kind,
        "embedding_lm_head": is_model_scope,
    }


def body_forward(case, *, phase="decode", tokens=1, context=0, sequences=1,
                 scope="body", recipes=None, primitives=None, event_limit=0, compact=False,
                 padded_idle=False, primitive_table=None):
    if scope not in {"body", "model"}:
        raise ModelingError(f"invalid_forward_scope:{scope}")
    case, graphs = _declared_graphs(
        case, phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope=scope, apply_body_defaults=recipes is None,
        padded_idle=padded_idle)
    if case.model.is_standalone_draft:
        raise ModelingError("draft_requires_target_model:attach_auxiliary_to_target;catalog_inspection_available")
    default_selection = (dict(graphs.default_selection) if recipes is None else
                         {"profile": "explicit_recipe_book_preserved"})
    ext_in = {"analysis_input": 0} if scope == "body" else None
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.graph_analysis import graph_costs
    from mindie_sim.prediction.operator_analysis import cohort_sequence_counts
    evaluation = (
        EvaluationContext.create(case, recipes, primitives, primitive_table=primitive_table)
        if primitive_table is not None else None
    )
    result = graph_costs(case, graphs, recipes=recipes, primitives=primitives,
                         external_inputs=ext_in,
                         include_submission=False, isolated=False,
                         trace_limit=event_limit, compact=compact,
                         evaluation=evaluation)
    counts = cohort_sequence_counts(sequences, tuple(graphs))
    from mindie_sim.prediction.execution.decode_probe import probe_capacity
    capacity = probe_capacity(case, context=context, tokens=tokens,
                              sequences=sequences, phase=phase,
                              padded_idle=padded_idle)
    padded = capacity["padded_query_tokens_by_replica"]
    graph_rows = capacity["graph_query_tokens_by_replica"]
    included = describe_included_components(case, scope=scope)
    return {
        **result, "schema": "mindie-sim.body-forward/1", "status": "modeled",
        "scope": scope, "body_forward_ms": result["duration_ms"],
        "metric_boundary": (
            "target_body_device_graph" if scope == "body" else "full_model_step_device_graph"
        ),
        "included_components": included,
        "geometry": {"phase": phase, "tokens_per_sequence": tokens,
                     "context_tokens": context, "sequences_per_replica": sequences,
                     "participating_replicas": list(graphs),
                     "sequences_by_replica": counts,
                     "valid_query_tokens_by_replica": {r: tokens * count for r, count in counts.items()},
                     "graph_query_tokens_by_replica": graph_rows,
                     "graph_query_tokens": sum(graph_rows.values()),
                     "padded_query_tokens_by_replica": padded,
                     "physical_global_rank_basis": "dp_replica * tp + tp_rank",
                     "physical_global_ranks_by_replica": physical_global_ranks(
                         case.runtime, tuple(graphs)),
                     "padding_basis": "graph_bucket_then_declared_sequence_parallel_padding",
                     "valid_query_tokens": tokens * sum(counts.values()),
                     "padded_query_tokens": sum(padded.values())},
        "routing": case.runtime.routing or {"policy": "probability_moments"},
        "default_mechanisms": default_selection,
        "capacity": capacity,
        "boundary_details": {
            "start": "input hidden states and KV cache ready",
            "end": ("target transformer body output ready on all cohort participants"
                    if scope == "body"
                    else "full model step (embedding, target body, draft if attached, lm_head) ready"),
            "excluded": (["embedding", "final_norm", "lm_head", "draft_generation",
                          "sampling", "host_submission", "scheduler", "pd_handoff",
                          "inter_step_wait"] if scope == "body"
                         else ["sampling", "host_submission", "scheduler", "pd_handoff",
                               "inter_step_wait"]),
            "speculation": "tokens includes every target verification row; acceptance does not scale work",
            "prefix_cache": "context KV is ready; query rows still execute",
            "deployment": "first independent DP cohort; no request admission or capacity guarantee",
        },
    }


def pd_body_forward(raw, **options):
    common = {k: v for k, v in raw.items() if k not in {"pools", "connector"}}
    resolved_scope = str(options.get("scope", "body"))
    pools = {name: body_forward(
        {**common, **raw["pools"][name], "model": raw["model"], "workload": raw["workload"]},
        **options) for name in ("prefill", "decode")}
    from mindie_sim.coverage import combine_coverage
    models = [parse_case({**common, **raw["pools"][name], "model": raw["model"],
                          "workload": raw["workload"]}).model for name in pools]
    return {
        "schema": "mindie-sim.body-forward/1", "status": "modeled", "scope": resolved_scope,
        "deployment": "disaggregated", **combine_coverage(models),
        "metric_boundary": (
            "independent_target_body_device_graph_per_pool" if resolved_scope == "body"
            else "independent_full_model_step_device_graph_per_pool"
        ),
        "included_components": {name: value["included_components"] for name, value in pools.items()},
        "body_forward_ms": {name: value["body_forward_ms"] for name, value in pools.items()},
        "pools": pools,
    }
