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
from mindie_sim.catalog import draft_spec as _draft_spec_mod
from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan
from mindie_sim.prediction.structure import stage_map
from mindie_sim.prediction.implementation_selection import resolve, LoweringRegion


PROPOSAL_ID_BYTES = 8
_DRAFT_RMSNORM_OPS_PER_ELEMENT = 5
_DRAFT_SAMPLING_OPS_PER_ELEMENT = 5
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


def _draft_enforce_eager(runtime) -> bool:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if bool(meta.get("draft_enforce_eager", False)):
        return True
    spec_cfg = meta.get("speculative_config")
    return isinstance(spec_cfg, dict) and bool(spec_cfg.get("enforce_eager", False))


def _target_prolog_has_custom_cache_layout(binding) -> bool:
    if not isinstance(binding, dict):
        return False
    opts = binding.get("options")
    return isinstance(opts, dict) and bool(opts.get("cache_layout"))


def _effective_draft_context_cap(draft, runtime) -> int:
    """Resolve explicit draft context cap from DraftSpec or runtime metadata."""
    helper = getattr(_draft_spec_mod, "effective_draft_context_cap", None)
    if callable(helper):
        return int(helper(draft, runtime))
    caps: list[int] = []
    spec_cap = int(getattr(draft, "max_position_embeddings", 0) or 0)
    if spec_cap > 0:
        caps.append(spec_cap)
    meta = dict(getattr(runtime, "metadata", ()) or ())
    spec_cfg = meta.get("speculative_config")
    for raw in (
        meta.get("draft_max_model_len"),
        spec_cfg.get("max_model_len") if isinstance(spec_cfg, dict) else None,
    ):
        val = int(raw or 0)
        if val > 0:
            caps.append(val)
    return min(caps) if caps else 0


def _draft_mla_prolog_eligible(draft_model, runtime) -> bool:
    """True iff every draft MLA layer satisfies vllm_ascend_mla_prolog_v3 validation."""
    if draft_model is None:
        return True
    from mindie_sim.prediction.operators.mla_prolog import family_mla_prolog
    mla_plans = [p for p in getattr(draft_model, "layer_plans", ()) if p.attention == "mla"]
    if not mla_plans:
        return False
    for plan in mla_plans:
        try:
            family_mla_prolog.validate(
                plan=plan,
                parallel=runtime.parallel,
                model=draft_model,
                quant=draft_model.quant,
                lowering_passes=runtime.lowering_passes,
            )
        except ModelingError:
            return False
    return True


def _independent_draft_runtime(runtime, selections=(), *, draft_model=None):
    """Use the independent draft's declared writer for its body prolog."""
    kernels = dict(getattr(runtime, "kernels", {}) or ())
    lowerings = dict(kernels.get("lowerings", {}) or ())
    eager_draft = _draft_enforce_eager(runtime)
    if eager_draft:
        lowerings.pop(PROLOG_ROLE, None)
        lowerings.pop("native_fusion", None)
        return replace(
            runtime,
            graph_capture_mode="none",
            kernels={**kernels, "lowerings": lowerings},
        )
    if PROLOG_ROLE not in lowerings:
        return runtime
    draft_binding = any(
        choice["selector"] == PROLOG_ROLE
        and choice["implementation"] != "analytical"
        for choice in selections)
    prolog_decl = lowerings.get(PROLOG_ROLE)
    # When target's MLA prolog binding has no incompatible packed cache layout
    # override, an independent MLA drafter may run its own BF16 MLAPO kernel
    # iff its MLA layers satisfy MLAPO V3 structural validation.
    compatible_plain_prolog = (
        not draft_binding
        and not _target_prolog_has_custom_cache_layout(prolog_decl)
        and (isinstance(prolog_decl, str) or (
            isinstance(prolog_decl, dict) and "cases" not in prolog_decl
        ))
        and _draft_mla_prolog_eligible(draft_model, runtime)
    )
    if not draft_binding and not compatible_plain_prolog:
        lowerings.pop(PROLOG_ROLE)
        return replace(runtime, kernels={**kernels, "lowerings": lowerings})
    if compatible_plain_prolog:
        lowerings[PROLOG_ROLE] = "vllm_ascend_mla_prolog_v3"
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


def _draft_cache_model(model, draft, draft_model, runtime, machine):
    from mindie_sim.prediction.operators.mla_prolog.draft_context import draft_cache_requires_prolog_v3
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
    return draft_model, context_prolog_v3


def _draft_features(model, draft, draft_model, step_model, runtime, *, previous, prefix,
                    stage, requests, rows, feature_rows, tp):
    ops = []
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
                ops_per_element=_DRAFT_RMSNORM_OPS_PER_ELEMENT,
                dtype=draft.activation_dtype, family="rmsnorm",
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
    elif rows and draft.input_mode == "normalized_concat":
        norm_ids = []
        for component in ("enorm", "hnorm"):
            norm_op = vector_op(
                op_id=prefix + component, deps=tuple(feature_deps), stage=stage,
                layer=-1, requests=requests, elements=rows * draft.hidden_size,
                ops_per_element=_DRAFT_RMSNORM_OPS_PER_ELEMENT,
                dtype=draft.activation_dtype, family="rmsnorm",
                phase="draft", recipe="draft_input_norm",
                shape=(rows, draft.hidden_size),
            )
            ops.append(norm_op)
            norm_ids.append(norm_op.op_id)
        projection = replace(projection, deps=tuple(norm_ids))
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
    return ops, projection


def _append_draft_decode_head(ops, frontend, model, draft, step_model, draft_model,
                              runtime, selected_runtime, proposal_views, previous, *,
                              prefix, stage, requests, rows):
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
            ops_per_element=_DRAFT_SAMPLING_OPS_PER_ELEMENT, dtype="fp32",
            family="sampling", phase="draft",
            recipe="draft_sample")
        ops.append(sample)
        previous = sample.op_id
    return previous


def _proposal_view(view, *, prefill: bool, serial: bool, step: int, draft_cap: int):
    q_len = view.q_len if prefill else (1 if serial else view.draft_proposal_tokens)
    raw_ctx = view.context + (step if serial else 0)
    eff_ctx = min(raw_ctx, max(0, draft_cap - q_len)) if draft_cap > 0 else raw_ctx
    return replace(
        view, q_len=q_len, samples=not prefill,
        phase="prefill" if prefill else "decode",
        context=eff_ctx, draft_verify=False, draft_proposal_tokens=0,
    )


def _draft_step_runtime(runtime, draft, proposal_views, global_proposals, rows, *,
                        replay_scope, external_context, draft_model=None):
    eff_scope = "none" if _draft_enforce_eager(runtime) else replay_scope
    selected_runtime, selections = resolve(
        runtime, proposal_views, padded_tokens=rows, replay_scope=eff_scope,
        global_views=global_proposals, model_scope="draft")
    from mindie_sim.prediction.frontend import mark_analytical_runtime
    selected_runtime = mark_analytical_runtime(selected_runtime, selections)
    if external_context and draft.attention_kind == "mla":
        selected_runtime = _independent_draft_runtime(
            selected_runtime, selections, draft_model=draft_model)
    elif _draft_enforce_eager(runtime):
        selected_runtime = replace(selected_runtime, graph_capture_mode="none")
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
    return selected_runtime, selections


def _transfer_draft_proposals(regions, previous, views, runtime, *, prefill, last_active):
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
    return previous


def draft_program(frontend, model, runtime, views, ready, *, global_views=None, prefill=False,
                  replica=0, replay_scope="none", machine=None):
    kind = dict(runtime.metadata).get("draft_mode", "mtp")
    draft = optional_draft_spec(model, kind)
    from mindie_sim.catalog.paired_sfa import independent_draft
    external_context = independent_draft(draft)

    def selected_view(view):
        return view.phase == "prefill" if prefill else bool(view.draft_proposal_tokens)

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
        draft_model, context_prolog_v3 = _draft_cache_model(
            model, draft, draft_model, runtime, machine)
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
    draft_cap = _effective_draft_context_cap(draft, runtime)
    eager_draft = _draft_enforce_eager(runtime)
    last_active = None
    for step in range(steps):
        selected = tuple(view for view in active if not serial or view.draft_proposal_tokens > step)
        proposal_views = tuple(
            _proposal_view(view, prefill=prefill, serial=serial, step=step, draft_cap=draft_cap)
            for view in selected
        )
        global_proposals = tuple(
            _proposal_view(view, prefill=prefill, serial=serial, step=step, draft_cap=draft_cap)
            for view in all_active if not serial or view.draft_proposal_tokens > step
        )
        rows = sum(view.q_len for view in proposal_views)
        feature_rows = (sum(view.q_len if prefill else view.draft_proposal_tokens + 1
                            for view in selected) if external_context else rows)
        selected_runtime, selections = _draft_step_runtime(
            runtime, draft, proposal_views, global_proposals, rows,
            replay_scope=replay_scope, external_context=external_context,
            draft_model=draft_model)
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
        ops, projection = _draft_features(
            model, draft, draft_model, step_model, runtime, previous=previous, prefix=prefix,
            stage=stage, requests=requests, rows=rows, feature_rows=feature_rows, tp=tp)
        global_tokens = sum(view.q_len if prefill else 1 if serial else view.draft_proposal_tokens
                            for view in all_active if not serial or view.draft_proposal_tokens > step)
        if pcp_prefill:
            global_total = sum(view.q_len for view in global_proposals)
            global_tokens = frontend.pcp_cohort_tokens(
                global_proposals, proposal_views, rows, selected_runtime,
                global_tokens=global_total, global_views=global_proposals)
        body_ready = projection.op_id if rows else previous
        if external_context and rows:
            from mindie_sim.prediction.operators.mla_prolog.draft_context import prepare
            body_ready = prepare(
                ops, projection=projection, model=draft_model, draft=draft,
                rows=feature_rows, query_rows=rows, prefix=prefix,
                stage=stage, requests=requests, parallel=runtime.parallel,
                rows_by_view=[view.q_len + (0 if prefill else 1)
                              for view in proposal_views],
                views=proposal_views,
                contexts=[view.context for view in proposal_views],
                block=int(getattr(runtime, "kv_block_size", 1) or 1),
                use_prolog_v3=context_prolog_v3 and not eager_draft)
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
        previous = _append_draft_decode_head(
            ops, frontend, model, draft, step_model, draft_model, runtime,
            selected_runtime, proposal_views, previous, prefix=prefix,
            stage=stage, requests=requests, rows=rows)
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
    previous = _transfer_draft_proposals(
        regions, previous, views, runtime, prefill=prefill, last_active=last_active)
    return regions, previous
