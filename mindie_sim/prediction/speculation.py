"""Draft generation as a program preceding target verification.

[Role] Derive proposal work from a declared draft structure.
[Invariants] Proposals are computed, verified and then committed. Draft heads
  have their own dimensions and weights; acceptance never discounts target work.
[Timeouts & Locks] Bounded by the scheduled proposal count and draft layers.
[Side-Effect Policy] Constructs operator records only.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.catalog.draft import optional_draft_spec, draft_quant
from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan
from .structure import stage_map
from .implementation_selection import resolve, LoweringRegion


PROPOSAL_ID_BYTES = 8
PROLOG_ROLE = "mla_prolog"
FUSION_OPTION_ROLES = "roles"
FUSION_OPTION_PROFILES = ("program_profiles", "program_options")


def compute_speculative_accept_step(accept_rate, draft_tokens) -> float:
    """Expected accepted tokens in one verification step.

    ``(1 - a**(d+1)) / (1 - a)`` with an exact ``a == 1`` branch.
    """
    if isinstance(accept_rate, bool) or not isinstance(accept_rate, (int, float)):
        raise ModelingError("speculative_accept_rate_invalid")
    rate = float(accept_rate)
    if not math.isfinite(rate) or rate < 0 or rate > 1.0:
        raise ModelingError("speculative_accept_rate_invalid")
    if (not isinstance(draft_tokens, int) or isinstance(draft_tokens, bool)
            or draft_tokens < 0):
        raise ModelingError("speculative_draft_tokens_invalid")
    if math.isclose(rate, 1.0):
        return float(draft_tokens + 1)
    return (1.0 - rate ** (draft_tokens + 1)) / (1.0 - rate)


def _tag_draft_ops(ops, rows_by_request, *, step=0):
    """Attach the actual draft producer identity and materialized row counts.

    A target layer and a draft layer can share a numeric index.  Offload and
    state consumers must use this scope, not the layer number alone, to choose
    the producer and completion handle.
    """
    rows = tuple(sorted((str(request), int(count)) for request, count in rows_by_request.items()))
    return tuple(replace(op, metadata=tuple({
        **dict(op.metadata),
        "layer_scope": "draft",
        "producer_scope": "draft",
        "draft_step": int(step),
        "draft_rows_by_request": rows,
    }.items())) for op in ops)


from mindie_sim.catalog.draft_geometry import (  # noqa: F401
    _draft_layer_plan, _reference_attention_plan, draft_layer_plans,
)


def _independent_draft_runtime(runtime, selections=()):
    """Use the independent draft's declared writer for its body prolog."""
    kernels = dict(getattr(runtime, "kernels", {}) or ())
    lowerings = dict(kernels.get("lowerings", {}) or ())
    if "mla_prolog" not in lowerings:
        return runtime
    draft_binding = any(
        choice["selector"] == PROLOG_ROLE
        and choice["implementation"] != "analytical"
        for choice in selections)
    if not draft_binding:
        lowerings.pop(PROLOG_ROLE)
        return replace(runtime, kernels={**kernels, "lowerings": lowerings})
    fusion = lowerings.get("native_fusion")
    if isinstance(fusion, dict):
        # The target's fusion inventory can name grouped/hyper groups that a
        # dense DSpark block does not contain.  Keep only the prolog role so
        # the draft never claims a group it cannot materialize; its recipe is
        # already registered by the case's declaration.
        options = dict(fusion.get("options") or {})
        options[FUSION_OPTION_ROLES] = (PROLOG_ROLE,)
        for key in FUSION_OPTION_PROFILES:
            value = options.get(key)
            if isinstance(value, dict):
                options[key] = {name: item for name, item in value.items()
                                if name == PROLOG_ROLE}
        traffic = options.get("traffic_profile")
        if isinstance(traffic, dict):
            options["traffic_profile"] = {
                name: item for name, item in traffic.items() if name == PROLOG_ROLE}
        lowerings["native_fusion"] = {**fusion, "options": options}
    kernels["lowerings"] = lowerings
    return replace(runtime, kernels=kernels)


def _cyclic_draft_model(model, plans, step):
    """One declared NextN block for the proposal step it owns."""
    plan = plans[step % len(plans)]
    return replace(model, layers=1, layer_plans=(plan,))


def draft_program(frontend, model, runtime, views, ready, *, global_views=None, prefill=False,
                  replica=0, replay_scope="none", machine=None):
    kind = dict(runtime.metadata).get("draft_mode", "mtp")
    draft = optional_draft_spec(model, kind)
    from mindie_sim.catalog.paired_sfa import independent_draft
    external_context = independent_draft(draft)
    selected_view = lambda view: view.phase == "prefill" if prefill else bool(view.draft_proposal_tokens)
    active = tuple(view for view in views if selected_view(view))
    all_active = tuple(view for view in (global_views or views) if selected_view(view))
    # Autoregressive MTP computes one proposal at a time. A parallel block
    # drafter computes its declared query block in one transformer pass.
    serial = (
        not prefill
        and not draft.noncausal_query_block
        and (
            draft.kind == "mtp"
            or draft.execution_mode in {"sequential_mtp", "cyclic_nextn", "eagle3_serial"}
        )
    )
    steps = max(view.draft_proposal_tokens for view in all_active) if serial else 1
    plans = draft_layer_plans(model, kind)
    draft_model = replace(
        model, name="draft", layers=draft.layers, hidden_size=draft.hidden_size,
        num_heads=draft.num_heads, num_kv_heads=draft.num_kv_heads, head_dim=draft.head_dim,
        intermediate_size=draft.intermediate_size, layer_plans=plans, auxiliary=None,
        expert_routing=(), num_experts=draft.num_experts, top_k=draft.top_k,
        vocab_size=draft.vocab_size or model.vocab_size,
        quant=draft_quant(model, draft), activation_dtype=draft.activation_dtype,
        compute_dtype=draft.activation_dtype,
    )
    context_prolog_v3 = True
    if external_context:
        from .draft_context import draft_cache_requires_prolog_v3
        draft_kv_dtype = str(getattr(draft_model.quant, "kv_dtype", "") or "bf16")
        if draft.attention_kind == "mla":
            context_prolog_v3 = draft_cache_requires_prolog_v3(
                model, runtime=runtime, machine=machine)
            cache_reps = {"mla_kv": {"dtype": draft_kv_dtype}}
        else:
            context_prolog_v3 = False
            cache_reps = {
                "gqa_kv": {"dtype": draft_kv_dtype},
                "kv_cache": {"dtype": draft_kv_dtype},
            }
        metadata = dict(draft_model.metadata)
        metadata.pop("cache_allocator", None)
        metadata["cache_representations"] = cache_reps
        metadata["attention_execution"] = {
            **metadata.get("attention_execution", {}), "dcp_indexer_sharded": False}
        draft_model = replace(
            draft_model, kv_dtype=draft_kv_dtype, metadata=tuple(metadata.items())
        )
    regions, previous = [], ready
    stage = runtime.parallel.pp - 1
    # ``index_share_for_mtp_iteration`` reuses one TopK selection across the
    # autoregressive proposal steps while every step still updates its own
    # KPool cache row.  ``mtp_topks`` carries step 0's op ids into later steps.
    share_selection = bool(
        (draft.index_share_for_mtp_iteration
         or any(getattr(plan, "index_share_for_mtp_iteration", False)
                for plan in plans))
        and serial and draft.attention_kind == "mla")
    mtp_topks: dict[int, str] = {}
    # Feature projection shards its output; the draft body consumes a complete
    # hidden vector, so TP must gather before entering the first layer.
    tp = runtime.parallel.tp
    last_active = None
    for step in range(steps):
        ops = []
        selected = tuple(view for view in active if not serial or view.draft_proposal_tokens > step)

        def proposal(view):
            return replace(
                view, q_len=view.q_len if prefill else 1 if serial else view.draft_proposal_tokens,
                samples=not prefill, phase="prefill" if prefill else "decode",
                context=view.context + (step if serial else 0),
                draft_verify=False, draft_proposal_tokens=0,
            )
        proposal_views = tuple(proposal(view) for view in selected)
        global_proposals = tuple(proposal(view) for view in all_active
                                 if not serial or view.draft_proposal_tokens > step)
        rows = sum(view.q_len for view in proposal_views)
        feature_rows = (sum(view.q_len if prefill else view.draft_proposal_tokens + 1
                            for view in selected) if external_context else rows)
        selected_runtime, selections = resolve(
            runtime, proposal_views, padded_tokens=rows, replay_scope=replay_scope,
            global_views=global_proposals, model_scope="draft")
        from .frontend import mark_analytical_runtime
        selected_runtime = mark_analytical_runtime(selected_runtime, selections)
        if external_context and draft.attention_kind == "mla":
            selected_runtime = _independent_draft_runtime(selected_runtime, selections)
        if external_context and not draft.dcp_sharded and selected_runtime.parallel.cp > 1:
            selected_runtime = replace(
                selected_runtime,
                parallel=replace(selected_runtime.parallel, cp=1),
            )
        # Sequence-parallel lowering is body-scoped; draft_program manages TP
        # feature gather explicitly before entering the draft body.
        if "sequence_parallel" in selected_runtime.kernels:
            selected_runtime = replace(
                selected_runtime,
                kernels={k: v for k, v in selected_runtime.kernels.items()
                         if k != "sequence_parallel"})
        cyclic = bool(serial and draft.execution_mode == "cyclic_nextn"
                      and len(plans) > 1)
        step_model = (_cyclic_draft_model(draft_model, plans, step)
                      if cyclic else draft_model)
        pcp_prefill = bool(prefill and selected_runtime.parallel.pcp > 1)
        if pcp_prefill:
            # The target partitions each PCP rank's prefilling query rows.  The
            # draft must own the same shard while every cache access keeps the
            # declared global length (``global_q_len``) for its gather.
            proposal_views, rows, _ = frontend.partition_views(
                proposal_views, selected_runtime, rows, model=step_model)
            feature_rows = (sum(view.q_len for view in proposal_views)
                            if external_context else rows)
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
                    recipe="draft_feature_transfer",
                    link_bytes=feature_rows * model.hidden_size
                    * dtype_bytes(model.activation_dtype),
                    shape=(feature_rows, model.hidden_size),
                    metadata=(("group_axis", "pp"), ("source_stage", source_stage),
                              ("destination_stage", stage)))
                ops.append(transfer)
                feature_deps.append(transfer.op_id)
        projection = gemm_op(
            op_id=prefix + "features", deps=tuple(feature_deps), stage=stage, layer=-1, requests=requests,
            m=feature_rows, n=(step_model.hidden_size + tp - 1) // tp,
            k=draft.input_width, quant=step_model.quant,
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
                    recipe="draft_features_gather",
                    link_bytes=feature_rows * draft.hidden_size
                    * dtype_bytes(draft.activation_dtype) * (tp - 1) / tp,
                    shape=(feature_rows, draft.hidden_size), metadata=(("group_axis", "tp"), ("steps", tp - 1)))
                ops.append(gathered)
                projection = gathered
        global_tokens = sum(view.q_len if prefill else 1 if serial else view.draft_proposal_tokens
                            for view in all_active if not serial or view.draft_proposal_tokens > step)
        if pcp_prefill:
            global_total = sum(view.q_len for view in global_proposals)
            global_tokens = frontend.pcp_cohort_tokens(
                global_proposals, proposal_views, rows, selected_runtime,
                global_tokens=global_total, global_views=global_proposals)
        body_ready = projection.op_id if rows else previous
        if external_context and rows:
            from .draft_context import prepare
            body_ready = prepare(
                ops, projection=projection, model=draft_model, draft=draft,
                rows=feature_rows, query_rows=rows, prefix=prefix,
                stage=stage, requests=requests, parallel=runtime.parallel,
                rows_by_view=[view.q_len + (0 if prefill else 1)
                              for view in proposal_views],
                views=proposal_views,
                contexts=[view.context for view in proposal_views],
                block=int(getattr(runtime, "kv_block_size", 1) or 1),
                use_prolog_v3=context_prolog_v3)
        body, previous = frontend.body(
            step_model, selected_runtime, proposal_views, prefix=prefix,
            ready=body_ready,
            tokens=rows, mode="draft_prefill" if prefill else "draft",
            stages=(stage,) * step_model.layers, global_tokens=global_tokens,
            replica=replica,
            machine=machine,
            topk_scope=mtp_topks if (share_selection and not prefill) else None,
            reuse_index_selection=bool(share_selection and not prefill and step > 0),
        )
        ops.extend(body)
        if prefill:
            rows_by_request = {view.request_id: view.q_len for view in proposal_views}
            regions.append(LoweringRegion(
                _tag_draft_ops(ops, rows_by_request),
                selected_runtime, draft_model.activation_dtype, rows, selections))
            continue
        heads, previous = frontend.logits(step_model, selected_runtime, proposal_views, previous,
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
        rows_by_request = (
            {view.request_id: view.q_len for view in proposal_views}
            if prefill else {
                view.request_id: (1 if serial else view.draft_proposal_tokens)
                for view in selected
            })
        regions.append(LoweringRegion(
            _tag_draft_ops(ops, rows_by_request, step=step),
            selected_runtime, draft_model.activation_dtype, rows, selections))
        if rows:
            last_active = (len(regions) - 1, previous)
    # Draft runs on the last pipeline stage. Send every accepted proposal id
    # back to stage 0 from the last locally active proposal step so a shorter
    # replica still declares its own transfer.
    proposal_views_local = tuple(view for view in views
                                 if view.draft_proposal_tokens > 0)
    if (not prefill and runtime.parallel.pp > 1 and last_active is not None
            and proposal_views_local):
        region_index, dependency = last_active
        total_proposals = sum(view.draft_proposal_tokens
                              for view in proposal_views_local)
        xfer = comm_op(
            op_id="draft_proposal_transfer",
            deps=(dependency,),
            stage=runtime.parallel.pp - 1,
            layer=-1,
            requests=tuple(view.request_id for view in proposal_views_local),
            family="pp_p2p",
            phase="draft",
            recipe="pp_p2p",
            link_bytes=total_proposals * PROPOSAL_ID_BYTES,
            shape=(total_proposals,),
            metadata=(
                ("group_axis", "pp"),
                ("source_stage", runtime.parallel.pp - 1),
                ("destination_stage", 0),
            ),
        )
        xfer = _tag_draft_ops((xfer,), {
            view.request_id: view.draft_proposal_tokens
            for view in proposal_views_local
        })[0]
        region = regions[region_index]
        regions[region_index] = replace(region, ops=region.ops + (xfer,))
        previous = xfer.op_id
    return regions, previous
