"""Reusable normalization and residual block programs.

[Role] Compose residual and hyper-connection sublayers from explicit tensors.
[Invariants] Saved residuals survive the sublayer. Hyper connections carry
  separate projection, normalization, mixing and post-mixing dependencies.
[Timeouts & Locks] Pure finite operator construction; no I/O or locks.
[Side-Effect Policy] Returns records only.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import norm_role
from mindie_sim.schema.common import dtype_bytes
from .structure_groups import declare_group, source_reference

ZERO = 0
ONE = 1
TWO = 2
FOUR = 4
FIVE = 5
SIXTEEN = 16
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
        elements=tokens * width, ops_per_element=5, dtype=model.activation_dtype,
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
        op = vector_op(
            op_id=c["prefix"] + ".residual", deps=(output, saved), stage=c["stage"],
            layer=c["plan"].index, requests=c["requests"], elements=c["tokens"] * c["model"].hidden_size,
            ops_per_element=1, dtype=c["model"].activation_dtype,
            family="elementwise", phase=c["mode"], recipe="residual_add",
            read_bytes_override=c["tokens"] * c["model"].hidden_size * 4,
            write_bytes_override=c["tokens"] * c["model"].hidden_size * 2,
        )
        return [op], op.op_id


class HyperConnection:
    def before(self, context, ready):
        c = context
        plan, model = c["plan"], c["model"]
        h, hc, tokens = model.hidden_size, plan.hc_mult, c["tokens"]
        mixed = (2 + hc) * hc
        common = dict(stage=c["stage"], layer=plan.index, requests=c["requests"], phase=c["mode"])
        values = vector_op(
            op_id=c["prefix"] + ".hc_values_fp32", deps=(ready,),
            elements=tokens * hc * h, ops_per_element=1, dtype="fp32",
            family="cast", recipe="hc_values_fp32", shape=(tokens, hc * h),
            read_bytes_override=tokens * hc * h * dtype_bytes(model.activation_dtype),
            write_bytes_override=tokens * hc * h * 4, **common,
        )
        project = gemm_op(
            op_id=c["prefix"] + ".hc_project", deps=(values.op_id,), m=tokens, n=mixed, k=hc * h,
            quant=model.quant, category="mhc", recipe="hc_projection", family="gemm", **common,
        )
        inverse_rms = vector_op(
            op_id=c["prefix"] + ".hc_inv_rms", deps=(values.op_id,),
            elements=tokens * hc * h, ops_per_element=2, dtype="fp32",
            vector_ops_override=tokens * (2 * hc * h + 3),
            family="mhc", recipe="hc_inverse_rms", shape=(tokens,),
            read_bytes_override=tokens * hc * h * 4,
            write_bytes_override=tokens * 4, **common,
        )
        affine = vector_op(
            op_id=c["prefix"] + ".hc_affine", deps=(project.op_id, inverse_rms.op_id),
            elements=tokens * mixed, ops_per_element=5, dtype="fp32", family="mhc",
            recipe="hc_normalize_scale_bias_gate", shape=(tokens, mixed),
            read_bytes_override=(tokens * (mixed + 1) + mixed + 3) * 4,
            write_bytes_override=tokens * mixed * 4, **common,
        )
        sinkhorn = vector_op(
            op_id=c["prefix"] + ".hc_sinkhorn", deps=(affine.op_id,), elements=tokens * mixed,
            ops_per_element=4 * plan.hc_sinkhorn_iters, dtype="fp32", family="mhc",
            recipe="hc_sinkhorn", returns=("post", "comb"),
            read_bytes_override=tokens * mixed * 4, **common,
        )
        mix = vector_op(
            op_id=c["prefix"] + ".hc_mix", deps=(ready, sinkhorn.op_id), elements=tokens * h,
            ops_per_element=2 * hc, dtype=model.activation_dtype, family="mhc",
            recipe="hc_pre_mix", read_bytes_override=tokens * hc * (h * 2 + 4),
            write_bytes_override=tokens * h * 2, **common,
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
        op = vector_op(
            op_id=c["prefix"] + ".hc_post", deps=(output, *saved), stage=c["stage"],
            layer=c["plan"].index, requests=c["requests"], elements=tokens * hc * h,
            ops_per_element=2 * hc + 1, dtype=c["model"].activation_dtype,
            family="mhc", phase=c["mode"], recipe="hc_post_mix",
            read_bytes_override=tokens * (h * 2 + hc * h * 2 + hc * 4 + hc * hc * 4),
            write_bytes_override=tokens * hc * h * 2,
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
        hc = max(ONE, int(plan.hc_mult or ONE))
        lr = int(getattr(plan, "hc_lowrank", ZERO) or ZERO)
        tokens = c["tokens"]
        hc_dim = hc * h
        act_bytes = dtype_bytes(model.activation_dtype)
        down_out = ((lr + hc + SIXTEEN - ONE) // SIXTEEN) * SIXTEEN
        common = dict(stage=c["stage"], layer=plan.index,
                      requests=c["requests"], phase=c["mode"])
        hc_norm = vector_op(
            op_id=c["prefix"] + ".hc_norm", deps=(ready,),
            elements=tokens * hc_dim, ops_per_element=FIVE,
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
            elements=tokens * lr, ops_per_element=FOUR,
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
            elements=tokens * h, ops_per_element=FOUR * hc,
            dtype=model.activation_dtype, family="mhc",
            recipe="hc_gate_mix", shape=(tokens, h),
            read_bytes_override=TWO * tokens * hc_dim * act_bytes,
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
        hc = max(ONE, int(c["plan"].hc_mult or ONE))
        tokens = c["tokens"]
        hc_dim = hc * h
        act_bytes = dtype_bytes(c["model"].activation_dtype)
        op = vector_op(
            op_id=c["prefix"] + ".hc_post", deps=(output, *saved),
            stage=c["stage"], layer=c["plan"].index, requests=c["requests"],
            elements=tokens * hc_dim, ops_per_element=FOUR,
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
    hc = max((p.hc_mult for p in model.layer_plans), default=ONE)
    lr = max((getattr(p, "hc_lowrank", ZERO) for p in model.layer_plans),
             default=ONE)
    h = model.hidden_size
    hc_dim = hc * h
    act_bytes = dtype_bytes(model.activation_dtype)
    ready = deps if isinstance(deps, tuple) else (deps,)
    common = dict(stage=stage, layer=FINAL_LAYER, requests=requests, phase=mode)
    hc_head_norm = vector_op(
        op_id=prefix + ".hc_head_norm", deps=ready,
        elements=tokens * hc_dim, ops_per_element=FIVE,
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
        elements=tokens * lr, ops_per_element=FOUR,
        dtype=model.activation_dtype, family="activation", recipe="hc_silu",
        shape=(tokens, lr), read_bytes_override=tokens * lr * act_bytes,
        write_bytes_override=tokens * lr * act_bytes, **common)
    hc_head_up = gemm_op(
        op_id=prefix + ".hc_head_up", deps=(hc_head_silu.op_id,),
        m=tokens, n=hc_dim, k=lr, quant=model.quant, category="linear_param_bf16",
        recipe="hc_lowrank_head_up", family="gemm", **common)
    hc_head_gate_mix = vector_op(
        op_id=prefix + ".hc_head_gate_mix", deps=(hc_head_norm.op_id, hc_head_up.op_id),
        elements=tokens * h, ops_per_element=FOUR * hc,
        dtype=model.activation_dtype, family="mhc", recipe="hc_gate_mix",
        shape=(tokens, h), read_bytes_override=TWO * tokens * hc_dim * act_bytes,
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
        values = vector_op(
            op_id=c["prefix"] + ".residual_values_fp32", deps=sources,
            elements=elements, ops_per_element=1, dtype="fp32", family="cast",
            recipe="residual_values_fp32", shape=(tokens, len(sources), model.hidden_size),
            read_bytes_override=elements * dtype_bytes(model.activation_dtype),
            write_bytes_override=elements * 4, **common)
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
            elements=tokens * len(sources), ops_per_element=5, dtype="fp32",
            family="softmax", recipe="attention_residual_softmax",
            shape=(tokens, len(sources)), **common)
        mix = vector_op(
            op_id=c["prefix"] + ".residual_mix", deps=(probabilities.op_id, values.op_id),
            stage=c["stage"], layer=c["plan"].index, requests=c["requests"],
            elements=tokens * model.hidden_size * len(sources), ops_per_element=2,
            dtype="fp32", family="elementwise", phase=c["mode"],
            recipe="attention_residual_mix",
            read_bytes_override=tokens * len(sources) * (model.hidden_size + 1) * 4,
            write_bytes_override=tokens * model.hidden_size * 4)
        cast = vector_op(
            op_id=c["prefix"] + ".residual_output_cast", deps=(mix.op_id,),
            elements=tokens * model.hidden_size, ops_per_element=1,
            dtype=model.activation_dtype, family="cast", recipe="attention_residual_output_cast",
            read_bytes_override=tokens * model.hidden_size * 4,
            write_bytes_override=tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
            **common)
        ops = [values, normalized, scores, probabilities, mix, cast]
        declare_group(
            ops, key=c["prefix"] + ".attention_residual_group",
            role="attention_residual", family="attention_residual",
            members=tuple(op.op_id for op in ops), entries=(values.op_id,),
            anchor=cast.op_id, block=c.get("structure_block"), scope_role="residual",
            view={"tokens": tokens, "hidden": model.hidden_size, "sources": len(sources),
                  "dtype": model.activation_dtype, "compute_dtype": "fp32"},
            implementation={"native_kind": "AttnResFwd",
                            "conditions": {"scope": "normalized_learned_residual_mixture"}},
            source=source_reference("vllm_ascend/models/kimi_k3.py", "_apply_attention_residual"),
            coverage={"kind": "complete_subgraph",
                      "limitations": ["native_residual_slot_order_requires_descriptor_evidence"]},
        )
        return ops, cast.op_id, ready


RESIDUAL_BUILDERS = {"plain": Residual(), "hyper": HyperConnection(),
                     "lowrank_hyper": LowRankHyperConnection(),
                     "attention": AttentionResidual()}
