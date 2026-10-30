"""Gated delta attention with finite state recurrence.

[Role] Lower GDN and KDA projections, local chunk math and state scans.
[Invariants] Each request owns its state; chunks carry a serial state edge.
  Key/value dimensions and head counts remain independent.
[Timeouts & Locks] One aggregate scan per request, lazily expanded at execution.
[Side-Effect Policy] Creates operators only.
"""
import math
from dataclasses import replace

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import dtype_bytes, ModelingError
from .lowering import Builder
from .tensors import TensorLowering
from .structure_groups import declare_group, source_reference
from .implementations import select
from .projections import kda_projections, require_packed_slice


def linear(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
           padded_tokens, policy=None, runtime=None, **_):
    if (runtime and plan.attention == "gdn" and views
            and all(view.phase == "decode" for view in views)):
        selected = select(runtime.kernels.get("lowerings", {}), "gdn_decode")
        if selected:
            return selected.module.lower(
                prefix=prefix, plan=plan, views=views, stage=stage, deps=deps, model=model,
                parallel=parallel, mode=mode, padded_tokens=padded_tokens, policy=policy,
                runtime=runtime, **selected.options)
    binding = (runtime.kernels.get("lowerings", {}).get("gdn_prefill")
               if runtime and plan.attention == "gdn"
               and any(view.phase == "prefill" for view in views) else None)
    if binding:
        selected = select(runtime.kernels["lowerings"], "gdn_prefill")
        return selected.module.lower(
            prefix=prefix, plan=plan, views=views, stage=stage, deps=deps, model=model,
            parallel=parallel, mode=mode, padded_tokens=padded_tokens, policy=policy,
            runtime=runtime, **selected.options)
    b = Builder(prefix, plan, stage, views, model, mode)
    tensors = TensorLowering(b, runtime.tensor_layouts if runtime else None)
    tp, tokens, hidden = parallel.tp, padded_tokens, model.hidden_size
    key_heads = math.ceil((plan.linear_key_heads or plan.linear_heads) / tp)
    heads = math.ceil((plan.linear_value_heads or plan.linear_heads) / tp)
    dk, dv = plan.linear_key_head_dim or plan.linear_head_dim, plan.linear_head_dim
    key_width, value_width = key_heads * dk, heads * dv
    kda_source = (select(runtime.kernels.get("lowerings", {}), "kda")
                  if runtime and plan.attention == "kda" else None)
    kda_caps = frozenset(getattr(kda_source.module, "CAPABILITIES", ())
                         if kda_source else ())
    prefill_source = (kda_source if kda_caps & {"prefill", "prefill_fwd_h"}
                      else None)
    decode_source = (kda_source if "decode" in kda_caps else None)
    conv_source = (select(runtime.kernels.get("lowerings", {}), "causal_conv_update")
                   if runtime and plan.attention == "kda"
                   and any(view.phase == "decode" for view in views) else None)
    if kda_source:
        kda_source.module.validate(
            key_heads=key_heads, value_heads=heads, key_width=dk, value_width=dv, model=model)
    projection_source = (select(runtime.kernels.get("lowerings", {}), "kda_input_projection")
                         if runtime and plan.attention == "kda" else None)
    projection_group = None
    conv_width = 2 * key_width + value_width
    chunk = policy.recurrent_chunk if policy else 64
    if plan.attention == "gdn":
        # Checkpoints can retain gate/output projections in native precision
        # while quantizing QKV. Role overrides inherit the existing gdn parent.
        qkv = b.matrix("linear_qkvz", (deps,), tokens, conv_width + value_width, hidden,
                       category="gdn.qkvz")
        gates = b.matrix("linear_beta_alpha", (deps,), tokens, 2 * heads, hidden,
                         category="gdn.beta_alpha")
        projection = tensors.result(qkv, (tokens, conv_width + value_width))
        conv_input = tensors.require(
            projection.narrow(1, 0, conv_width), "short_convolution", "input",
            name="convolution_input_contiguous")
        gate_projection = tensors.result(gates, (tokens, 2 * heads))
        gate_inputs = [
            tensors.require(gate_projection.narrow(1, index * heads, heads), "gdn_gating", name,
                            name=f"gate_{name}_contiguous").producer
            for index, name in enumerate(("beta", "alpha"))
        ]
        output_gate_view = projection.narrow(1, conv_width, value_width)
        sources = [conv_input.producer]
    else:
        if projection_source:
            projection_group = projection_source.module.lower(
                b, tensors, plan=plan, tokens=tokens, hidden=hidden, key_width=key_width,
                value_width=value_width, heads=heads, dk=dk, deps=deps)
            sources, output_gate, beta, decay, raw_beta = (projection_group[key] for key in
                ("sources", "output_gate", "beta", "decay", "raw_beta"))
        else:
            sources, output_gate, beta, decay = kda_projections(
                b, plan=plan, tokens=tokens, hidden=hidden, key_width=key_width,
                value_width=value_width, heads=heads, dk=dk, deps=deps)
            if len(set(sources)) == 1 and not conv_source:
                # The native q1 convolution adapter already owns AutoContiguous.
                incoming = require_packed_slice(
                    b, tensors, sources[0], ("q", "k", "v"), "short_convolution", "input",
                    "convolution_input_contiguous")
                sources = [incoming] * 3
            beta = require_packed_slice(b, tensors, beta, ("b",), "kda_recurrent", "beta",
                                        "kda_beta_contiguous")
            raw_beta = beta
            beta = b.vector(
                "linear_beta_sigmoid", (beta,), tokens * heads, 5,
                dtype="fp32", family="sigmoid", shape=(tokens, heads),
                read_bytes_override=tokens * heads * dtype_bytes(model.activation_dtype),
                write_bytes_override=tokens * heads * 4)
    if conv_source:
        conv = conv_source.module.lower(
            b, tensors, plan=plan, views=views, tokens=tokens, sources=sources,
            key_width=key_width, value_width=value_width, **conv_source.options)
    else:
        conv = b.vector("short_convolution", tuple(dict.fromkeys(sources)),
                        tokens * conv_width, 2 * plan.conv_kernel,
                        read_bytes_override=(tokens + (plan.conv_kernel - 1) * len(views)) * conv_width * 2,
                        write_bytes_override=tokens * conv_width * 2)
    if projection_group:
        # Source returns from its auxiliary stream only after beta is ready.
        b.ops[-1] = replace(b.ops[-1], deps=tuple(dict.fromkeys(
            (*b.ops[-1].deps, projection_group["join"]))))
    state_copy = (select(runtime.kernels.get("lowerings", {}), "conv_state_copy")
                  if runtime and plan.attention == "kda"
                  and views and all(view.phase == "decode" for view in views) else None)
    if state_copy:
        conv = state_copy.module.wrap(b, conv=conv, views=views, width=conv_width,
                                      **state_copy.options)
    norm_rows = sum(view.q_len for view in views) if conv_source else tokens
    normalized = b.vector("qk_l2norm", (conv,), norm_rows * 2 * key_width, 5)
    if plan.attention == "gdn":
        # g and beta are scalars per value head, not per value channel. The
        # independent z projection gates the output after the recurrent core.
        gated = b.vector(
            "delta_gates", tuple(dict.fromkeys(gate_inputs)), tokens * heads, 12, family="gdn_gating",
            shape=(tokens, heads), read_bytes_override=tokens * heads * 4 + heads * 8,
            write_bytes_override=tokens * heads * 6,
            metadata=(("outputs", ("fp32_decay", "activation_beta")),
                      ("input_heads", key_heads), ("state_heads", heads)),
        )
    else:
        gated = b.vector(
            "delta_gates", (decay, beta), tokens * (heads * dk + heads), 5,
            shape=(tokens, heads, dk),
            read_bytes_override=tokens * heads * dk * dtype_bytes(model.activation_dtype)
                                + (tokens * heads + heads + heads * dk) * 4,
            metadata=(("A_log_dtype", "fp32"), ("dt_bias_dtype", "fp32"),
                      ("beta_dtype", "fp32"), ("raw_gate_dtype", model.activation_dtype)))
    outputs, prefill_scan_ids, prepare_ids = [], [], []
    for index, view in enumerate(views):
        q = view.q_len
        chunks = math.ceil(q / chunk) if view.phase == "prefill" else q
        span = chunk if view.phase == "prefill" else 1
        last = gated
        if view.phase == "prefill":
            pairs = sum(min(chunk, q - offset) ** 2 for offset in range(0, q, chunk))
            # KDA forms QK and KK before WY; only Aqk @ V belongs after the scan.
            product_flops = (4 if plan.attention == "kda" else 2) * pairs * heads * dk
            local = b.work(f"chunk_products{index}", (gated, normalized), (q, heads, dk, dv),
                           flops=product_flops, vector=pairs * heads * 4,
                           read=q * heads * 2 * dk * 2, write=pairs * heads * 2,
                           family="recurrent_local")
            last = b.work(f"triangular_solve{index}", (local,), (chunks, heads, span, span),
                          flops=2 / 3 * sum(min(chunk, q - offset) ** 3
                                           for offset in range(0, q, chunk)) * heads,
                          vector=pairs * heads * 2, read=pairs * heads * 2, write=pairs * heads * 2,
                          family="triangular_solve")
            prepare_ids.extend((local, last))
            last = b.work(f"wy_transform{index}", (last,), (q, heads, dk, dv),
                          flops=2 * pairs * heads * (dk + dv),
                          read=(pairs + q * (dk + dv)) * heads * 2,
                          write=q * heads * (dk + dv) * 2, family="recurrent_local")
        state_bytes = heads * dk * dv * dtype_bytes(plan.recurrent_state_dtype)
        if view.phase == "prefill" or plan.attention != "gdn":
            last = b.work(
                f"recurrent_scan{index}", (last, normalized), (q, heads, dk, dv),
                flops=(4 if view.phase == "prefill" else 6) * q * heads * dk * dv,
                vector=q * heads * (2 + 2 * dv) + chunks * 2 * heads * dk * dv,
                read=chunks * state_bytes + q * heads * (2 * dk + dv) * 2,
                write=chunks * state_bytes + q * value_width * 2, family="recurrent_scan",
                metadata=(("chunk", span), ("chunks", chunks), ("tokens", q),
                          ("heads", heads), ("dk", dk), ("dv", dv)),
            )
            if view.phase == "prefill":
                prefill_scan_ids.append(last)
        else:
            last = b.work(
                f"recurrent_scan{index}", (gated, normalized), (q, heads, dk, dv),
                vector=7 * q * heads * dk * dv,
                read=state_bytes + q * heads * ((2 * dk + dv) * 2 + 6),
                write=q * (state_bytes + value_width * 2), family="recurrent_vector",
                metadata=(("tokens", q), ("heads", heads), ("dk", dk), ("dv", dv),
                          ("sequence_tokens", (q,)),
                          ("key_heads", key_heads),
                          ("state_dtype", plan.recurrent_state_dtype),
                          ("state_element_bytes", dtype_bytes(plan.recurrent_state_dtype)),
                          ("state_checkpoints", q)),
            )
        if view.phase == "prefill":
            pair_width = dv if plan.attention == "kda" else dk + dv
            last = b.work(f"recurrent_output{index}", (last, normalized), (q, heads, dk, dv),
                          flops=2 * q * heads * dk * dv + 2 * pairs * heads * pair_width,
                          vector=pairs * heads,
                          read=chunks * state_bytes + q * heads * (2 * dk + dv) * 2,
                          write=q * value_width * 2, family="recurrent_output")
        outputs.append(last)
    decode_core = None
    decode_binder = (getattr(decode_source.module, "bind_decode_scans", None)
                     if decode_source else None)
    if decode_binder is not None and any(view.phase == "decode" for view in views):
        if any(view.phase != "decode" for view in views) or parallel.pcp != 1:
            raise ModelingError("kda_source_recurrent_requires_decode_only_pcp1")
        options = dict(decode_source.options)
        # This caller provides raw Q/K and decay. The selected kernel owns
        # normalization and gate evaluation, so their old nodes are removed.
        internal_beta = options.get("use_beta_sigmoid", False)
        if projection_group and internal_beta:
            raise ModelingError("kda_mixed_projection_supplies_preprocessed_beta")
        supplied_beta = raw_beta if internal_beta else beta
        beta_dtype = model.activation_dtype if internal_beta else "fp32"
        if options.get("beta_dtype", beta_dtype) != beta_dtype:
            raise ModelingError("kda_source_beta_dtype_must_match_selected_producer")
        if options.get("gate_dtype", model.activation_dtype) != model.activation_dtype:
            raise ModelingError("kda_source_raw_gate_dtype_must_match_projection")
        options.update(beta_dtype=beta_dtype, gate_dtype=model.activation_dtype)
        scans = tuple(op for op in b.ops if op.family == "recurrent_scan")
        decode_core = decode_binder(
            scans, lengths=tuple(view.q_len for view in views), heads=heads,
            key_heads=key_heads, dk=dk, dv=dv, dtype=model.activation_dtype,
            state_dtype=plan.recurrent_state_dtype, requests=b.common["requests"],
            deps=tuple(dict.fromkeys((conv, decay, supplied_beta))), **options)
        if projection_group:
            decode_core = replace(decode_core, metadata=decode_core.metadata + (
                ("projection_input_views", projection_group["views"]),))
        removed = {normalized, gated, *(op.op_id for op in scans)}
        if internal_beta:
            removed.add(beta)
        rewritten = []
        for op in b.ops:
            if op.op_id == scans[0].op_id:
                rewritten.append(decode_core)
            elif op.op_id not in removed:
                rewritten.append(op)
        b.ops, outputs = rewritten, [decode_core.op_id]
    if (kda_source and "prefill_gate_prepare" in kda_caps
            and prepare_ids):
        if any(view.phase != "prefill" for view in views):
            raise ModelingError(
                "kda_gate_prepare_source_requires_prefill_only_views")
        binder = getattr(kda_source.module, "bind_gate_prepare", None)
        if binder is None:
            raise ModelingError("kda_gate_prepare_source_binder_missing")
        target_ids = {gated, *prepare_ids}
        target_ops = tuple(op for op in b.ops if op.op_id in target_ids)
        gate_op = next(op for op in target_ops if op.op_id == gated)
        prepare_ops = tuple(op for op in target_ops if op.op_id != gated)
        b.ops, gated_op = binder(
            b.ops, gate=gate_op, local_ops=prepare_ops,
            lengths=tuple(view.q_len for view in views if view.phase == "prefill"),
            heads=heads, key_heads=key_heads, dk=dk, dv=dv, chunk=chunk,
            dtype=model.activation_dtype, raw_g_dtype=model.activation_dtype,
            gate_dtype="fp32", beta_dtype="fp32",
            requests=tuple(dict.fromkeys(
                view.request_id for view in views if view.phase == "prefill")),
            pcp=parallel.pcp,
            speculative=any(view.draft_proposal_tokens
                            for view in views if view.phase == "prefill"),
        )
        gated = gated_op.op_id
    if prefill_source and prefill_scan_ids:
        binder = getattr(prefill_source.module, "bind_prefill_scans", None)
        if binder is not None:
            # One source call owns every prefill sequence's FWD_H stage.  The
            # generic decode scans, caller gates and output members stay put.
            scans = tuple(op for op in b.ops if op.op_id in set(prefill_scan_ids))
            fused = binder(
                scans,
                lengths=tuple(view.q_len for view in views if view.phase == "prefill"),
                heads=heads, key_heads=key_heads, dk=dk, dv=dv, chunk=chunk,
                dtype=model.activation_dtype, state_dtype=plan.recurrent_state_dtype,
                requests=tuple(dict.fromkeys(
                    view.request_id for view in views if view.phase == "prefill")),
                pcp=parallel.pcp,
                speculative=any(view.draft_proposal_tokens
                                for view in views if view.phase == "prefill"),
            )
            ids = {op.op_id for op in scans}
            anchor = scans[-1].op_id
            rewritten, deferred_outputs = [], []
            inserted = False
            for op in b.ops:
                if op.op_id in ids:
                    if op.op_id == anchor:
                        rewritten.extend((fused, *deferred_outputs))
                        inserted = True
                    continue
                updated = replace(op, deps=tuple(
                    fused.op_id if dep in ids else dep for dep in op.deps))
                # Every sequence's WY operands precede the one fused FWD_H.
                # Earlier sequence outputs must move behind that launch too.
                if (not inserted and op.family == "recurrent_output"
                        and any(dep in ids for dep in op.deps)):
                    deferred_outputs.append(updated)
                else:
                    rewritten.append(updated)
            b.ops = rewritten
            outputs = [fused.op_id if output in ids else output for output in outputs]
    if plan.attention == "gdn" and policy and policy.fused_batch:
        # Independent sequences share the kernel's head-task launch. Their
        # recurrence lengths stay separate; concatenation is not one long state.
        scans = [op for op in b.ops if op.family == "recurrent_vector"]
        if len(scans) > 1:
            ids = {op.op_id for op in scans}
            tokens_in_scan = tuple(dict(op.metadata)["tokens"] for op in scans)
            total_tokens = sum(tokens_in_scan)
            first = scans[0]
            shape = (total_tokens, heads, dk, dv)
            fields = ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
            meta = {**dict(first.metadata), "tokens": total_tokens,
                    "sequence_tokens": tokens_in_scan, "state_checkpoints": total_tokens}
            fused = replace(
                first, semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
                deps=tuple(dict.fromkeys(dep for op in scans for dep in op.deps)),
                metadata=tuple(meta.items()), **{key: sum(getattr(op, key) for op in scans) for key in fields})
            b.ops = [fused if op.op_id == first.op_id else op for op in b.ops
                     if op.op_id not in ids or op.op_id == first.op_id]
            outputs = list(dict.fromkeys(first.op_id if output in ids else output for output in outputs))
    if plan.attention == "gdn":
        if views and all(view.phase == "prefill" for view in views):
            core = [op for op in b.ops if op.family in {
                "recurrent_local", "triangular_solve", "recurrent_scan", "recurrent_output"}]
            core_ids = {op.op_id for op in core}
            declare_group(
                b.ops, key=prefix + ".gdn_chunk", role="gdn_chunk", family="gdn",
                members=tuple(op.op_id for op in core),
                entries=tuple(op.op_id for op in core if not core_ids.intersection(op.deps)),
                anchor=core[-1].op_id,
                view={"tokens": sum(view.q_len for view in views), "heads": heads,
                      "key_heads": key_heads, "key_width": dk, "value_width": dv,
                      "dtype": model.activation_dtype, "state_dtype": plan.recurrent_state_dtype,
                      "sequence_tokens": tuple(view.q_len for view in views), "chunk": chunk},
                implementation={
                    "native_kind": "ChunkGatedDeltaRuleFwd",
                    "selection": "compatible_recurrent_algorithm; native_kernel_not_selected",
                    "conditions": {"qk_l2norm": "external", "delta_gates": "external",
                                   "native_wrapper_state_dtype": "bf16"}},
                source=source_reference("vllm_ascend/ops/gdn.py",
                                        "AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused"),
                coverage={
                    "kind": "native_boundary_unverified",
                    "limitations": [
                        "wrapper_and_retained_native_source_have_different_tensor_ABIs",
                        "native_kernel_launch_decomposition_not_verified",
                        "wrapper_state_cast_and_sequence_metadata_conversion_not_selected",
                    ],
                },
            )
        output_gate = tensors.reshape(output_gate_view, (tokens * heads, dv),
                                      name="output_gate_reshape")
        output_gate = tensors.require(output_gate, "linear_output_norm_gate", "gate",
                                      name="output_gate_contiguous").producer
    else:
        core = [op for op in b.ops if op.family in {
            "recurrent_local", "triangular_solve", "recurrent_scan", "recurrent_output"}]
        if core:
            prefill = any(view.phase == "prefill" for view in views)
            members = tuple(op.op_id for op in core)
            entries = (core[0].op_id,)
            phase_source = prefill_source if prefill else decode_source
            if decode_core is not None:
                entries = (decode_core.op_id,)
            elif phase_source or not prefill:
                members = tuple(dict.fromkeys((gated, *members)))
                entries = (gated, core[0].op_id)
            if not prefill and decode_core is None:
                members = (normalized, *members)
                entries = (normalized, gated)
            state_selection = (dict(decode_core.metadata).get("state_selection")
                               if decode_core is not None else None)
            declare_group(
                b.ops, key=prefix + ".kda_core", role="kda_chunk" if prefill else "kda_recurrent",
                family="kda", members=members, entries=entries, anchor=core[-1].op_id,
                view={"tokens": sum(view.q_len for view in views), "heads": heads,
                      "key_heads": key_heads, "key_width": dk, "value_width": dv,
                      "dtype": model.activation_dtype, "state_dtype": plan.recurrent_state_dtype,
                      "chunk": chunk, "sequence_tokens": tuple(view.q_len for view in views),
                      "in_kernel_gate_source": bool(
                          phase_source and "in_kernel_gates" in kda_caps),
                      "beta_dtype": (dict(decode_core.metadata)["kda_recurrent_params"]["beta_dtype"]
                                     if decode_core is not None else "fp32"),
                      **({"has_accepted_tokens": True} if state_selection else {})},
                implementation={"native_kind": "ChunkKdaFwd" if prefill else "RecurrentKda",
                                "selection": phase_source.name if phase_source else None,
                                "conditions": {"scope": (
                                    "source_recurrent_kernel"
                                    if decode_core is not None else
                                    "gating_and_recurrent_algorithm"
                                    if phase_source and ("in_kernel_gates" in kda_caps
                                                         or not prefill) else
                                    "source_gate_prepare_plus_recurrent_algorithm"
                                    if phase_source and "prefill_gate_prepare" in kda_caps
                                    else "recurrent_algorithm_with_external_gate_prefix"),
                                               "state_output": True,
                                               **({"state_slot_mapping":
                                                   (state_selection["kind"] if state_selection else
                                                    "independent_sequence_checkpoints_no_accepted_array")}
                                                  if decode_core is not None else {})}},
                source=phase_source.module.SOURCE if phase_source else source_reference(
                              "vllm_ascend/ops/kimi_kda.py",
                              "KimiGatedDeltaNetAttention._run_prefill" if prefill
                              else "KimiGatedDeltaNetAttention._run_recurrent"),
                coverage={
                    "kind": ("source_selected_kernel" if decode_core is not None else
                             "multi_launch_stage_unknown" if prefill else "complete_subgraph"),
                    "limitations": ([
                        "capture_ABI_has_identical_descriptors_for_multiple_stage_launches",
                        "native_stage_scalar_not_observed",
                        "source_wrapper_gate_placement_depends_on_implementation_revision",
                    ] if prefill else [
                        "native_backing_tensor_extent_does_not_identify_active_view",
                        "state_indices_and_gate_flags_not_observed"]),
                },
            )
    output = b.vector(
        "linear_output_norm_gate", outputs + [output_gate], tokens * value_width, 9,
        shape=(tokens * heads, dv),
        read_bytes_override=(2 * tokens * value_width + dv) * dtype_bytes(model.activation_dtype),
        write_bytes_override=tokens * value_width * dtype_bytes(model.activation_dtype),
        metadata=(("norm_kind", "rmsnorm"), ("norm_has_weight", True),
                  ("norm_has_bias", False), ("norm_group_width", dv),
                  ("gate_present", True), ("recurrent_attention_kind", plan.attention),
                  ("fused_from", ("rmsnorm", "gate"))) + (
                      (("gate_input_layout", projection_group["views"]["output_gate"]),)
                      if projection_group else ()))
    last = b.matrix("linear_output", (output,), tokens, hidden, value_width,
                    category="kda.o" if plan.attention == "kda" else "gdn.o")
    if tp > 1:
        last = b.communicate("linear_reduce", (last,), 2 * (tp - 1) / tp * tokens * hidden * 2,
                             "allreduce")
    return AttentionResult(ops=b.ops, last=last)
