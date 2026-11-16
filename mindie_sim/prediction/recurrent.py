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
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from . import gdn_recurrent
from .operators import kda_state_selection as pcp_state_selection
from .lowering import Builder
from .tensors import TensorLowering
from .structure_groups import declare_group, source_reference
from .implementations import select
from .projections import kda_projections, require_packed_slice
from .recover_ssm_work import emit_commit_op, record_head_bytes, recover_ssm_enabled

POWER_QUADRATIC = 2
PRODUCT_VECTOR_BYTES = 256
PRODUCT_READ_BYTES = 32
LINEAR_VECTOR_BYTES = 2
POWER_CUBIC = 3
TWO_THIRDS = 2.0 / 3.0


ONE = 1


def _sum_valid_powers(tokens: int, chunk: int, power: int) -> int:
    """Closed form of sum(min(chunk, tokens - i*chunk)**power)."""
    full, remainder = divmod(int(tokens), int(chunk))
    return int(full) * int(chunk) ** int(power) + int(remainder) ** int(power)


def _serving_kda_source(runtime, model, views, plan):
    """Shared A5 KDA source selection for ordinary serving."""
    from .kda_source import resolve
    if runtime is None:
        return None
    opts = resolve(plan, arch=dict(runtime.metadata).get("arch_family"),
                   phase=(views[0].phase if views else ""), kernels=runtime.kernels,
                   activation_dtype=model.activation_dtype, views=views)
    if opts is None:
        return None
    return select({"kda": {"implementation": "vllm_ascend_recurrent_kda_arch35",
                           "options": opts}}, "kda")


def serial_context_parallel(parallel):
    """Collapse context parallelism while preserving a declared PCP plan.

    KDA state is head-local, so a serial recurrence is valid under PCP. The
    parallel spec still requires the matching plan; passing ``cp_plan="none"``
    with ``pcp > 1`` is an invalid declaration, not a supported fallback.
    """
    next_plan = "pcp" if parallel.pcp > ONE else "none"
    return replace(parallel, cp=ONE, dcp=ONE, cp_plan=next_plan)


def linear(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
           padded_tokens, policy=None, runtime=None, **_):
    from mindie_sim.catalog.paired_sfa import execution
    from dataclasses import replace
    if execution(model).get("recurrent_context_sharded") is False:
        # KDA state is head-local under PCP; collapsing context parallelism must
        # retain the PCP plan or the parallel spec rejects the serial state.
        parallel = serial_context_parallel(parallel)
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
    use_recover_ssm = recover_ssm_enabled(runtime)
    rec_bytes = heads * record_head_bytes(dk, dv, model.activation_dtype)
    kda_source = (select(runtime.kernels.get("lowerings", {}), "kda")
                  if runtime and plan.attention == "kda" else None) or _serving_kda_source(runtime, model, views, plan)
    kda_caps = frozenset(getattr(kda_source.module, "CAPABILITIES", ()) if kda_source else ())
    prefill_source = kda_source if kda_caps & {"prefill", "prefill_fwd_h"} else None
    decode_source = kda_source if "decode" in kda_caps else None
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
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ModelingError("recurrent_chunk_must_be_positive")
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
        active_views = tuple(view for view in views if not view.padding)
        active_tokens = sum(view.q_len for view in active_views)
        history_bytes = (plan.conv_kernel - 1) * len(active_views) * conv_width * 2
        conv = b.vector("short_convolution", tuple(dict.fromkeys(sources)),
                        tokens * conv_width, 2 * plan.conv_kernel,
                        read_bytes_override=(active_tokens * conv_width * 2
                                             + history_bytes),
                        write_bytes_override=(active_tokens * conv_width * 2
                                              + history_bytes))
        if plan.attention == "gdn" and active_views:
            # One physical convolution may carry speculative decode rows and
            # prefill rows.  Each slot keeps its own event: a decode row writes
            # query candidates and commits only its accepted offset, while a
            # prefill row writes its sequence tail once.  A whole-batch
            # "decode_only" or "conv" branch would publish 4100 for a q4 decode
            # slice and ignore B's pending decision.
            directions = tuple("token" if view.phase == "decode" else "conv"
                               for view in views)
            cursor, conv_views = 0, []
            for view in views:
                conv_views.append(conv_input.narrow(0, cursor, view.q_len))
                cursor += view.q_len
            contract, effects = gdn_recurrent.state_contract(
                views=views, phase="mixed", state_dtype=None,
                conv_state_dtype=plan.conv_state_dtype,
                conv_kernel=plan.conv_kernel, target_prefix=prefix,
                initial_state=tuple(view.context > 0 and not view.padding
                                    for view in views),
                direction=directions[0], directions=directions,
                target_kind="conv",
                global_lengths=tuple(view.global_q_len or view.q_len
                                     for view in views),
                value_inputs=tuple(conv_views), heads=heads,
                remote_publish=parallel.pcp > 1)
            contract, _state_outputs = gdn_recurrent.bind_state_ports(
                contract, b.ops[-1].op_id, heads=heads, head_shards=1)
            b.ops[-1] = replace(b.ops[-1], metadata=tuple({
                **dict(b.ops[-1].metadata),
                "state_dtype": plan.conv_state_dtype,
                "state_layout": "slots_history_channels",
                "state_mutation": "inplace_history_writeback",
                "state_write_bytes": history_bytes,
                "state_ready": b.ops[-1].op_id,
                "state_value_inputs": contract["value_inputs"],
                "state_value_outputs": contract["value_outputs"],
                "state_candidate_stores": contract["candidate_stores"],
                "state_candidate_storage": ({
                    "kind": "history_ring+query_rows",
                    "recovery": "accepted_prefix_from_conv_history_and_query_inputs",
                    "history_slots": plan.conv_kernel - 1,
                    "kernel": plan.conv_kernel,
                    "query_rows": tuple(view.q_len for view in views
                                        if view.phase == "decode"),
                    "op": b.ops[-1].op_id,
                } if any(direction == "token"
                         for direction in directions) else {}),
                "gdn_state_contract": contract,
                "gdn_state_effects": effects}.items()))
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
    # DualChunkSwap owns 2*pcp chunks. Conv history crosses each boundary, so
    # each prefill request pays (2*pcp-1)/pcp of one state all-gather.
    if parallel.pcp > 1 and any(view.phase == "prefill" for view in views):
        conv = b.communicate(
            "pcp_conv_state_step", (conv,),
            sum(1 for view in views if view.phase == "prefill")
            * max(0, plan.conv_kernel - 1) * conv_width * 2
            * (2 * parallel.pcp - 1) / parallel.pcp,
            "allgather", "pcp")
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
    view_offsets, _cursor = [], 0
    for _view in views:
        view_offsets.append(_cursor)
        _cursor += _view.q_len
    for index, view in enumerate(views):
        q = view.q_len
        chunks = math.ceil(q / chunk) if view.phase == "prefill" else q
        span = chunk if view.phase == "prefill" else 1
        last = gated
        if view.phase == "prefill":
            pairs = _sum_valid_powers(q, chunk, POWER_QUADRATIC)
            # KDA forms QK and KK before WY with a dk-wide decay gate per head
            # (stored in activation_dtype from delta_gates, accumulated in FP32);
            # only Aqk @ V belongs after the scan.
            is_kda = plan.attention == "kda"
            act_bytes = dtype_bytes(model.activation_dtype)
            product_flops = (4 if is_kda else 2) * pairs * heads * dk
            prod_vec = pairs * heads * (3 * dk if is_kda else 4)
            prod_read = q * heads * (3 if is_kda else 2) * dk * act_bytes
            prod_write = pairs * heads * (8 if is_kda else 2)
            local = b.work(f"chunk_products{index}", (gated, normalized), (q, heads, dk, dv),
                           flops=product_flops, vector=prod_vec,
                           read=prod_read, write=prod_write,
                           family="recurrent_local")
            last = b.work(f"triangular_solve{index}", (local,), (chunks, heads, span, span),
                          flops=TWO_THIRDS * _sum_valid_powers(
                              q, chunk, POWER_CUBIC) * heads,
                          vector=pairs * heads * (PRODUCT_VECTOR_BYTES if is_kda else LINEAR_VECTOR_BYTES),
                          read=pairs * heads * (PRODUCT_READ_BYTES if is_kda else LINEAR_VECTOR_BYTES),
                          write=pairs * heads * (PRODUCT_READ_BYTES if is_kda else LINEAR_VECTOR_BYTES),
                          family="triangular_solve")
            prepare_ids.extend((local, last))
            wy_vec = q * heads * (2 * dk + dv) * 4 if is_kda else 0
            wy_read = (2 * pairs + q * (3 * dk + dv)) * heads * act_bytes if is_kda else (pairs + q * (dk + dv)) * heads * act_bytes
            wy_write = q * heads * (3 * dk + dv if is_kda else dk + dv) * act_bytes
            last = b.work(f"wy_transform{index}", (last,), (q, heads, dk, dv),
                          flops=2 * pairs * heads * (dk + dv), vector=wy_vec,
                          read=wy_read, write=wy_write, family="recurrent_local")
        state_bytes = heads * dk * dv * dtype_bytes(plan.recurrent_state_dtype)
        pcp_ranges = (plan.attention == "gdn" and view.phase == "prefill"
                      and parallel.pcp > 1 and len(view.query_ranges) > 1)
        if view.phase == "prefill" or plan.attention != "gdn":
            is_kda_prefill = view.phase == "prefill" and plan.attention == "kda"
            if pcp_ranges:
                version, previous, segment_index = view.context, last, 0
                total_global = view.global_q_len or view.q_len
                for range_index, (start, end) in enumerate(view.query_ranges):
                    length = int(end) - int(start)
                    if length <= 0:
                        continue
                    seg_chunks = math.ceil(length / chunk)
                    seg_vec = seg_chunks * 2 * heads * dk * dv + length * heads * (2 + 2 * dv)
                    seg_read = seg_chunks * state_bytes + length * heads * (2 * dk + dv) * 2
                    seg_write = seg_chunks * state_bytes + length * value_width * 2
                    seg = b.work(
                        f"recurrent_scan{index}_seg{segment_index}", (previous, normalized),
                        (length, heads, dk, dv), flops=4 * length * heads * dk * dv,
                        vector=seg_vec, read=seg_read, write=seg_write, family="recurrent_scan",
                        metadata=(("chunk", chunk), ("chunks", seg_chunks), ("tokens", length),
                                  ("heads", heads), ("dk", dk), ("dv", dv),
                                  ("chunk_vector_ops", float(seg_chunks * 2 * heads * dk * dv)),
                                  ("token_vector_ops", float(length * heads * (2 + 2 * dv))),
                                  ("pcp_segment", (int(start), int(end)))))
                    segment_view = replace(view, q_len=length, context=version,
                                           global_q_len=length, query_ranges=((0, length),))
                    direction = "conv" if range_index == len(view.query_ranges) - 1 else "h"
                    # A local tail segment that reaches the global end is not
                    # necessarily the global owner: a PCP rank may miss the
                    # middle ranges owned by its peers.  Publish only when the
                    # whole rank owns every global query token.
                    owner = not pcp_state_selection.pcp_remote_publish(
                        view.query_ranges, view.q_len, total_global, pcp=parallel.pcp, padding=view.padding)
                    contract, effects = gdn_recurrent.state_contract(
                        views=(segment_view,), phase="prefill", state_dtype=plan.recurrent_state_dtype,
                        conv_state_dtype=plan.conv_state_dtype, conv_kernel=plan.conv_kernel,
                        target_prefix=prefix, chunk=chunk, initial_state=(True,), direction=direction,
                        target_kind=("recurrent" if direction == "conv" else "local"),
                        global_lengths=(length,),
                        remote_publish=(direction == "conv" and parallel.pcp > 1 and not owner))
                    b.ops[-1] = replace(b.ops[-1], metadata=tuple({
                        **dict(b.ops[-1].metadata), "gdn_state_contract": contract,
                        "gdn_state_effects": effects, "state_segment_index": segment_index,
                        "state_segment_range": (int(start), int(end)),
                        "state_segment_from_version": version,
                        "state_segment_to_version": version + length}.items()))
                    previous, version = seg, version + length
                    if range_index < len(view.query_ranges) - 1:
                        gap = int(view.query_ranges[range_index + 1][0]) - int(end)
                        if gap:
                            previous = b.communicate(
                                f"pcp_recurrent_state_step{index}_{segment_index}", (previous,),
                                state_bytes * (2 * parallel.pcp - 1) / parallel.pcp,
                                "state_transfer", "pcp", state_scope="recurrent_local_bank",
                                remote_producer="external_rank", source_implementation="pcp1_only",
                                status="pending_external_rank", from_version=version, to_version=version + gap)
                            version += gap
                    segment_index += 1
                if previous is not None and parallel.pcp > 1:
                    pub_name = "pcp_remote_state_publish" if len(views) == 1 else f"pcp_remote_state_publish{index}"
                    previous = b.work(
                        pub_name, (previous,), (0,), recipe="pcp_remote_state_publish", family="state_publish",
                        metadata=(("status", "local_owner" if owner else "pending_external_rank"),
                                  ("producer_scope", "local_writeback" if owner else "external_rank"),
                                  ("completion_owner", "T04/final_writeback"),
                                  ("source_implementation", "pcp1_only"),
                                  ("global_expected_version", view.context + (view.global_q_len or view.q_len))))
                last = previous
            else:
                spec_dec = view.phase == "decode" and use_recover_ssm and q > 1
                chunk_vec = chunks * (6 if is_kda_prefill else 2) * heads * dk * dv
                token_vec = q * heads * (dk * dv if is_kda_prefill else 2 + 2 * dv)
                scan_vec = token_vec + chunk_vec
                scan_read = (
                    2 * chunks * state_bytes + q * heads * ((2 * dk + dv) * 2 + dk * 4)
                    if is_kda_prefill else
                    (state_bytes if spec_dec else chunks * state_bytes) + q * heads * (2 * dk + dv) * 2
                )
                scan_write = (
                    2 * chunks * state_bytes + q * (value_width + heads * dv) * 2
                    if is_kda_prefill else
                    q * (rec_bytes + value_width * 2) if spec_dec else chunks * state_bytes + q * value_width * 2
                )
                pcp_segment = plan.attention == "gdn" and view.phase == "prefill" and parallel.pcp > 1
                ranges = tuple(view.query_ranges)
                first_start = int(ranges[0][0]) if ranges else 0
                range_end = int(ranges[-1][1]) if ranges else q
                global_q = view.global_q_len or view.q_len
                remote_initial = remote_port = None
                if pcp_segment and first_start > 0:
                    remote_initial = b.work(
                        f"pcp_remote_initial_state{index}", (normalized,), (0,), family="state_transfer",
                        metadata=(("status", "pending_external_rank"), ("producer_scope", "external_rank"),
                                  ("completion_owner", "T09/COMM-LEAVES"), ("source_implementation", "pcp1_only"),
                                  ("request", view.request_id), ("from_version", view.context),
                                  ("to_version", view.context + first_start)))
                    remote_port = gdn_recurrent.tensor_port(TensorValue(
                        "", plan.recurrent_state_dtype or "fp32",
                        TensorLayout.contiguous((1, heads, dv, dk)), binding="external",
                        binding_name=f"pcp_peer:{view.request_id}:state@{view.context + first_start}",
                        state_version=view.context + first_start),
                        ready=str(remote_initial), role="remote_input", owner="external_rank")
                scan_deps = (last, normalized) + ((remote_initial,) if remote_initial else ())
                extra_meta = (
                    (("recover_ssm", True), ("recover_ssm_stage", "verify_compact_record"),
                     ("record_bytes_per_token", rec_bytes), ("state_checkpoints", 0))
                    if spec_dec else ()
                )
                last = b.work(
                    f"recurrent_scan{index}", scan_deps, (q, heads, dk, dv),
                    flops=(4 if view.phase == "prefill" else 6) * q * heads * dk * dv,
                    vector=scan_vec, read=scan_read, write=scan_write, family="recurrent_scan",
                    metadata=(("chunk", span), ("chunks", chunks), ("tokens", q),
                              ("heads", heads), ("dk", dk), ("dv", dv),
                              ("chunk_vector_ops", float(chunk_vec)),
                              ("token_vector_ops", float(token_vec))) + extra_meta)
                if plan.attention == "gdn":
                    contract, effects = gdn_recurrent.state_contract(
                        views=(view,), phase="prefill", state_dtype=plan.recurrent_state_dtype,
                        conv_state_dtype=plan.conv_state_dtype, conv_kernel=plan.conv_kernel,
                        target_prefix=prefix, chunk=chunk, initial_state=(view.context > 0,),
                        direction="conv", target_kind="recurrent",
                        global_lengths=(view.global_q_len or view.q_len,),
                        remote_publish=(pcp_segment and pcp_state_selection.pcp_remote_publish(
                            view.query_ranges, q, global_q, pcp=parallel.pcp, padding=view.padding)),
                        remote_input=(remote_port,), heads=heads)
                    contract, state_outputs = gdn_recurrent.bind_state_ports(
                        contract, last, heads=heads, head_shards=1, remote_input=(remote_port,))
                    b.ops[-1] = replace(b.ops[-1], metadata=tuple({
                        **dict(b.ops[-1].metadata), "state_ready": last,
                        "state_remote_inputs": ((remote_port,) if remote_port else ()),
                        "gdn_state_contract": contract, "gdn_state_effects": effects,
                        "state_value_outputs": state_outputs}.items()))
                    if pcp_segment:
                        state_bytes = heads * dk * dv * dtype_bytes(plan.recurrent_state_dtype or "fp32")
                        step = None
                        if first_start == 0 and range_end < global_q:
                            step = b.communicate(
                                f"pcp_recurrent_state_step{index}", (last,),
                                state_bytes * (2 * parallel.pcp - 1) / parallel.pcp,
                                "state_transfer", "pcp", request=view.request_id,
                                from_version=view.context + range_end, to_version=view.context + global_q,
                                remote_producer="external_rank", completion_owner="T09/COMM-LEAVES",
                                source_implementation="pcp1_only", status="pending_external_rank")
                        owner = "local_writeback" if range_end >= global_q else "external_rank"
                        b.work(
                            f"pcp_remote_state_publish{index}", (step or last,), (0,), family="state_publish",
                            metadata=(("status", "local_owner" if range_end >= global_q else "pending_external_rank"),
                                      ("producer_scope", owner), ("completion_owner", "T04/final_writeback"),
                                      ("source_implementation", "pcp1_only"), ("request", view.request_id),
                                      ("global_expected_version", view.context + global_q),
                                      ("publish_pending", "remote_producer")))
            if view.phase == "prefill":
                prefill_scan_ids.append(last)
                if parallel.pcp > 1 and not pcp_ranges and plan.attention != "gdn":
                    # Contiguous KDA ranges keep the single post-scan handoff.
                    state_bytes = heads * dk * dv * dtype_bytes(plan.recurrent_state_dtype or "fp32")
                    last = b.communicate(
                        f"pcp_recurrent_state_step{index}", (last,),
                        state_bytes * (2 * parallel.pcp - 1) / parallel.pcp, "state_transfer", "pcp",
                        remote_producer="external_rank", source_implementation="pcp1_only",
                        status="pending_external_rank")
        else:
            spec_gdn = use_recover_ssm and q > 1
            gdn_write = q * ((rec_bytes if spec_gdn else state_bytes) + value_width * 2)
            gdn_extra = (
                (("recover_ssm", True), ("recover_ssm_stage", "verify_compact_record"),
                 ("record_bytes_per_token", rec_bytes))
                if spec_gdn else ()
            )
            last = b.work(
                f"recurrent_scan{index}", (gated, normalized), (q, heads, dk, dv),
                vector=7 * q * heads * dk * dv,
                read=state_bytes + q * heads * ((2 * dk + dv) * 2 + 6),
                write=gdn_write, family="recurrent_vector",
                metadata=(("tokens", q), ("heads", heads), ("dk", dk), ("dv", dv),
                          ("sequence_tokens", (q,)), ("key_heads", key_heads),
                          ("state_dtype", plan.recurrent_state_dtype),
                          ("state_element_bytes", dtype_bytes(plan.recurrent_state_dtype)),
                          ("state_checkpoints", 0 if spec_gdn else q)) + gdn_extra,
            )
            if plan.attention == "gdn":
                offset, activation = view_offsets[index], model.activation_dtype
                decay_value = TensorValue(gated, activation, TensorLayout.contiguous((tokens, heads)))
                qk_value = TensorValue(normalized, activation, TensorLayout.contiguous((tokens, 2 * key_width)))
                conv_value = TensorValue(conv, activation, TensorLayout.contiguous((tokens, conv_width)))
                value_views = {
                    "decay": decay_value.narrow(0, offset, view.q_len),
                    "qk": qk_value.narrow(0, offset, view.q_len),
                    "conv": conv_value.narrow(0, offset, view.q_len),
                }
                contract, effects = gdn_recurrent.state_contract(
                    views=(view,), phase="decode", state_dtype=plan.recurrent_state_dtype,
                    conv_state_dtype=plan.conv_state_dtype, conv_kernel=plan.conv_kernel,
                    target_prefix=prefix, initial_state=(view.context > 0,),
                    direction="token", target_kind="recurrent",
                    value_inputs=(value_views["conv"],), heads=heads)
                contract, _state_outputs = gdn_recurrent.bind_state_ports(
                    contract, b.ops[-1].op_id, heads=heads, head_shards=1)
                b.ops[-1] = replace(b.ops[-1], metadata=tuple({
                    **dict(b.ops[-1].metadata), "state_ready": b.ops[-1].op_id,
                    "state_row_range": (offset, offset + view.q_len),
                    "state_value_views": {n: gdn_recurrent.tensor_port(v, role=n) for n, v in value_views.items()},
                    "state_value_inputs": contract["value_inputs"],
                    "state_value_outputs": contract["value_outputs"],
                    "state_candidate_stores": contract["candidate_stores"],
                    "state_candidate_storage": {
                        "kind": "token_checkpoint_store", "recovery": "accepted_prefix_from_query_candidates",
                        "history_slots": None, "kernel": None, "query_rows": (view.q_len,), "op": b.ops[-1].op_id,
                    },
                    "gdn_state_contract": contract, "gdn_state_effects": effects}.items()))
        if view.phase == "prefill":
            is_kda = plan.attention == "kda"
            pair_width = dv if is_kda else dk + dv
            out_vec = pairs * heads * 6 + q * heads * dv * 4 if is_kda else pairs * heads
            out_read = (chunks * state_bytes + (pairs + q * (2 * dk + dv)) * heads * 2
                        if is_kda else chunks * state_bytes + q * heads * (2 * dk + dv) * 2)
            last = b.work(f"recurrent_output{index}", (last, normalized), (q, heads, dk, dv),
                          flops=2 * q * heads * dk * dv + 2 * pairs * heads * pair_width,
                          vector=out_vec, read=out_read,
                          write=q * value_width * 2, family="recurrent_output")
        outputs.append(last)
    decode_core = None
    decode_binder = (getattr(decode_source.module, "bind_decode_scans", None)
                     if decode_source else None)
    if decode_binder is not None and views and all(view.phase == "decode" for view in views):
        if parallel.cp > 1:
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
        if use_recover_ssm and any(view.q_len > 1 for view in views):
            options["recover_ssm"] = True
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
            deferred_ids = set()
            inserted = False
            for op in b.ops:
                if op.op_id in ids:
                    if op.op_id == anchor:
                        rewritten.extend((fused, *deferred_outputs))
                        inserted = True
                    continue
                updated = replace(op, deps=tuple(
                    fused.op_id if dep in ids else dep for dep in op.deps))
                # WY operands precede the one fused FWD_H. Earlier outputs and
                # the PCP state transfers that feed them move behind that launch.
                waits_for_scan = any(dep in ids or dep in deferred_ids for dep in op.deps)
                if not inserted and waits_for_scan and (
                        op.family == "recurrent_output"
                        or op.recipe.startswith("pcp_recurrent_state_step")):
                    deferred_outputs.append(updated)
                    deferred_ids.add(op.op_id)
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
            contracts = [dict(op.metadata).get("gdn_state_contract") for op in scans]
            contracts = [contract for contract in contracts if contract is not None]
            effects = tuple(effect for op in scans
                            for effect in (dict(op.metadata).get("gdn_state_effects") or ()))
            offsets, cursor = [], 0
            for count in tokens_in_scan:
                offsets.append(cursor)
                cursor += count
            checkpoints_sum = sum(
                int(dict(op.metadata).get("state_checkpoints", dict(op.metadata)["tokens"]))
                for op in scans
            )
            meta = {**dict(first.metadata), "tokens": total_tokens,
                    "sequence_tokens": tokens_in_scan, "state_checkpoints": checkpoints_sum,
                    "state_request_offsets": tuple(offsets),
                    "state_request_ids": tuple(dict.fromkeys(
                        request for op in scans for request in op.requests)),
                    "gdn_state_effects": effects}
            if contracts:
                fused_contract = gdn_recurrent.merge_contracts(contracts)
                fused_contract, _fused_outputs = gdn_recurrent.bind_state_ports(
                    fused_contract, first.op_id, heads=heads, head_shards=1)
                meta["gdn_state_contract"] = fused_contract
                meta["state_ready"] = first.op_id
                meta["state_candidate_stores"] = fused_contract["candidate_stores"]
                meta["state_value_inputs"] = fused_contract["value_inputs"]
                meta["state_value_outputs"] = fused_contract["value_outputs"]
                meta["state_checkpoint_versions"] = tuple(
                    item for op in scans
                    for item in (dict(op.metadata).get("state_checkpoint_versions") or ()))
            fused = replace(
                first, semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
                requests=tuple(dict.fromkeys(request for op in scans for request in op.requests)),
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
    if use_recover_ssm and any(view.phase == "decode" and view.q_len > 1 for view in views):
        emit_commit_op(b, name="recover_ssm_commit", deps=tuple(outputs), views=views,
                       heads=heads, dk=dk, dv=dv, state_dtype=plan.recurrent_state_dtype,
                       kg_dtype=model.activation_dtype, runtime=runtime,
                       attention_kind=plan.attention)
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
