"""Draft generation as a program preceding target verification.

[Role] Derive proposal work from a declared draft structure.
[Invariants] Proposals are computed, verified and then committed. Draft heads
  have their own dimensions and weights; acceptance never discounts target work.
[Timeouts & Locks] Bounded by the scheduled proposal count and draft layers.
[Side-Effect Policy] Constructs operator records only.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.catalog.draft import optional_draft_spec, draft_quant
from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op
from mindie_sim.schema.model import LayerPlan
from .structure import stage_map


def draft_program(frontend, model, runtime, views, ready, *, global_views=None, prefill=False,
                  replica=0):
    kind = dict(runtime.metadata).get("draft_mode", "mtp")
    draft = optional_draft_spec(model, kind)
    selected_view = lambda view: view.phase == "prefill" if prefill else bool(view.draft_proposal_tokens)
    active = tuple(view for view in views if selected_view(view))
    all_active = tuple(view for view in (global_views or views) if selected_view(view))
    # Autoregressive MTP computes one proposal at a time. A parallel block
    # drafter computes its declared query block in one transformer pass.
    serial = not prefill and not draft.noncausal_query_block and draft.kind == "mtp"
    steps = max(view.draft_proposal_tokens for view in all_active) if serial else 1
    plan_fields = {
        key: getattr(draft, key) for key in (
            "num_heads", "num_kv_heads", "head_dim", "q_lora_rank", "kv_lora_rank",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "o_lora_rank",
            "o_groups", "num_experts", "top_k", "num_shared_experts", "intermediate_size",
        )
    }
    plans = tuple(LayerPlan(
        index=index, attention=draft.attention_kind, ffn="moe" if draft.num_experts else "dense",
        moe_intermediate_size=draft.expert_intermediate_size, scope="draft",
        noncausal_attention=draft.noncausal_query_block,
        mla_variant=draft.mla_variant, kv_cache_elements=draft.kv_cache_elements,
        qk_norm=draft.qk_norm, q_output_gate=draft.q_output_gate,
        kv_owner_layer_id=index, owns_long_kv=True,
        hyper_connections=bool(draft.mhc_mult), hc_mult=draft.mhc_mult,
        hc_sinkhorn_iters=draft.mhc_sinkhorn_iters, **plan_fields,
    ) for index in range(draft.layers))
    draft_model = replace(
        model, name="draft", layers=draft.layers, hidden_size=draft.hidden_size,
        num_heads=draft.num_heads, num_kv_heads=draft.num_kv_heads, head_dim=draft.head_dim,
        intermediate_size=draft.intermediate_size, layer_plans=plans, auxiliary=None,
        expert_routing=(), num_experts=draft.num_experts, top_k=draft.top_k,
        vocab_size=draft.vocab_size or model.vocab_size,
        quant=draft_quant(model, draft), activation_dtype=draft.activation_dtype,
        compute_dtype=draft.activation_dtype,
    )
    ops, previous = [], ready
    stage = runtime.parallel.pp - 1
    # Feature projection shards its output; the draft body consumes a complete
    # hidden vector, so TP must gather before entering the first layer.
    tp = runtime.parallel.tp
    for step in range(steps):
        selected = tuple(view for view in active if not serial or view.draft_proposal_tokens > step)
        proposal_views = tuple(replace(
            view, q_len=view.q_len if prefill else 1 if serial else view.draft_proposal_tokens,
            samples=not prefill, phase="prefill" if prefill else "decode",
            context=view.context + (step if serial else 0),
            draft_verify=False, draft_proposal_tokens=0,
        ) for view in selected)
        rows = sum(view.q_len for view in proposal_views)
        requests = tuple(view.request_id for view in proposal_views)
        prefix = "draft_prefill." if prefill else f"draft{step}."
        feature_deps = [previous]
        targets = stage_map(model, runtime)
        for layer_id in draft.target_layer_ids:
            source_stage = targets[layer_id]
            if source_stage != stage:
                transfer = comm_op(
                    op_id=prefix + f"features_from{layer_id}", deps=(previous,), stage=stage,
                    layer=-1, requests=requests, family="pp_p2p", phase="draft",
                    recipe="draft_feature_transfer", link_bytes=rows * model.hidden_size * 2,
                    shape=(rows, model.hidden_size),
                    metadata=(("group_axis", "pp"), ("source_stage", source_stage),
                              ("destination_stage", stage)))
                ops.append(transfer)
                feature_deps.append(transfer.op_id)
        projection = gemm_op(
            op_id=prefix + "features", deps=tuple(feature_deps), stage=stage, layer=-1, requests=requests,
            m=rows, n=(draft.hidden_size + tp - 1) // tp, k=draft.input_width, quant=draft_model.quant,
            category=draft.input_projection_category, phase="draft", family="gemm",
            recipe="draft_features",
        )
        if rows and draft.input_mode == "dual_normalized":
            projected = []
            for component in ("embedding", "target_hidden"):
                normalized = vector_op(
                    op_id=prefix + component + "_norm", deps=tuple(feature_deps), stage=stage,
                    layer=-1, requests=requests, elements=rows * draft.hidden_size,
                    ops_per_element=5, dtype=draft.activation_dtype, family="rmsnorm",
                    phase="draft", recipe="draft_input_norm")
                projection = gemm_op(
                    op_id=prefix + component + "_projection", deps=(normalized.op_id,), stage=stage,
                    layer=-1, requests=requests, m=rows, n=draft.hidden_size, k=draft.hidden_size,
                    quant=draft_model.quant, category="linear_param_bf16", phase="draft",
                    family="gemm", recipe="draft_feature_projection")
                ops.extend((normalized, projection))
                projected.append(projection.op_id)
            projection = vector_op(
                op_id=prefix + "features", deps=tuple(projected), stage=stage, layer=-1,
                requests=requests, elements=rows * draft.hidden_size, ops_per_element=1,
                dtype=draft.activation_dtype, family="elementwise", phase="draft",
                recipe="draft_feature_add")
            ops.append(projection)
        elif rows:
            ops.append(projection)
            if tp > 1:
                gathered = comm_op(
                    op_id=prefix + "features_gather", deps=(projection.op_id,), stage=stage,
                    layer=-1, requests=requests, family="allgather", phase="draft",
                    recipe="draft_features_gather", link_bytes=rows * draft.hidden_size * 2 * (tp - 1) / tp,
                    shape=(rows, draft.hidden_size), metadata=(("group_axis", "tp"), ("steps", tp - 1)))
                ops.append(gathered)
                projection = gathered
        global_tokens = sum(view.q_len if prefill else 1 if serial else view.draft_proposal_tokens
                            for view in all_active if not serial or view.draft_proposal_tokens > step)
        body, previous = frontend.body(
            draft_model, runtime, proposal_views, prefix=prefix, ready=projection.op_id if rows else previous,
            tokens=rows, mode="draft_prefill" if prefill else "draft",
            stages=(stage,) * draft.layers, global_tokens=global_tokens, replica=replica,
        )
        ops.extend(body)
        if prefill:
            continue
        heads, previous = frontend.logits(draft_model, runtime, proposal_views, previous,
                                         prefix=prefix, mode="draft", stage=stage, sample=False)
        ops.extend(heads)
        if draft.markov_rank and rows:
            lookup = vector_op(
                op_id=prefix + "markov_lookup", deps=(previous,), stage=stage, layer=-1,
                requests=requests, elements=rows * draft.markov_rank, ops_per_element=1,
                dtype=draft.activation_dtype, family="embedding", phase="draft",
                recipe="markov_embedding")
            markov = gemm_op(
                op_id=prefix + "markov", deps=(lookup.op_id,), stage=stage, layer=-1, requests=requests,
                m=rows, n=draft.vocab_size, k=draft.markov_rank, quant=draft_model.quant,
                category="lm_head", phase="draft", family="logits", recipe="draft_markov_head",
            )
            correction = vector_op(
                op_id=prefix + "markov_bias", deps=(previous, markov.op_id), stage=stage,
                layer=-1, requests=requests, elements=rows * draft.vocab_size,
                ops_per_element=1, dtype="fp32", family="elementwise", phase="draft",
                recipe="markov_bias")
            ops.extend((lookup, markov, correction))
            previous = correction.op_id
        if rows and (draft.confidence_executes or (runtime.dspark and runtime.dspark.confidence)):
            confidence = vector_op(
                op_id=prefix + "confidence", deps=(previous,), stage=stage, layer=-1, requests=requests,
                elements=rows * (draft.hidden_size + draft.markov_rank), ops_per_element=2,
                dtype=model.activation_dtype, family="confidence", phase="draft",
                recipe="draft_confidence",
            )
            ops.append(confidence)
            previous = confidence.op_id
        if rows:
            sample = vector_op(
                op_id=prefix + "sample", deps=(previous,), stage=stage, layer=-1,
                requests=requests, elements=rows * draft_model.vocab_size,
                ops_per_element=5, dtype="fp32", family="sampling", phase="draft",
                recipe="draft_sample")
            ops.append(sample)
            previous = sample.op_id
    return ops, previous
