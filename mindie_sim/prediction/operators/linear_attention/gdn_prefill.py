"""Non-interleaved QKVZ/BA GDN caller using the native arch35 composite.

[Role] Lower contiguous BSND inputs, native Prepare/H/O and requested exports.
[Invariants] Normalization is inside Prepare. Gate scalars become FP32 before
  BSN->BNS conversion. Recurrent state is FP32 V,K; conv history is the
  activation dtype. State is independent per sequence; padding slots keep
  their physical rows but neither read nor write a state row. Gather reads the
  cache, H keeps a local chunk bank, and only scatter publishes the final
  request state. Non-spec decode is a prefix of
  the actual batch; padded rows stop outside the core. Physical kernels are
  shared with isolated operator analysis.
[Timeouts & Locks] O(sequence count) metadata, lazy kernel work; no locks.
[Side-Effect Policy] Appends operators only; no measurements or device access.
"""
from dataclasses import replace
from types import SimpleNamespace
import math

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.prediction.operators.linear_attention import gdn_composite
from .gdn_domain import GDN_CHUNK64_TOKENS
from .gdn_prefill_phases import (
    lower_conv_stage,
    state_slot_inputs as _state_slot_inputs,
)
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.structure_groups import declare_group, source_reference
from mindie_sim.prediction.operators.causal_conv import native_causal_conv_prefill
from mindie_sim.prediction.operators.kda import kda_state_selection as pcp_state_selection
from mindie_sim.prediction.operators.linear_attention import gdn_recurrent
from mindie_sim.prediction.tensors import TensorLowering

DECODE_QK_L2NORM_OPS_PER_ELEMENT = 5
DECODE_RECURRENCE_OPS_PER_STATE_ELEMENT = 7
DELTA_GATE_OPS_PER_HEAD = 12
OUTPUT_NORM_GATE_OPS_PER_ELEMENT = 9
STATE_INDEX_BYTES = 4
FP32_BYTES = 4


def recipes():
    result = gdn_composite.recipes()
    result[native_causal_conv_prefill.IMPLEMENTATION] = PlannedRecipe(native_causal_conv_prefill.build)
    return result


def rearrange_qkv(tensors, conv, tokens, heads, key_heads, dk, dv):
    """The caller packs the whole non-spec batch before slicing its two phases."""
    qkv = tensors.result(conv, (tokens, 2 * key_heads * dk + heads * dv))
    slices = []
    start = 0
    for name, count, width in (("q", key_heads, dk), ("k", key_heads, dk), ("v", heads, dv)):
        sliced = qkv.narrow(1, start, count * width)
        slices.append(tensors.reshape(sliced, (tokens * count * width,), name=name + "_flat"))
        start += count * width
    fused = tensors.concatenate_flat(slices, "qkv_concatenate")
    values, start = [], 0
    for name, count, width in (("q", key_heads, dk), ("k", key_heads, dk), ("v", heads, dv)):
        size = tokens * count * width
        values.append(tensors.reshape(fused.narrow(0, start, size),
                                     (1, tokens, count, width), name=name + "_view"))
        start += size
    return values


def _prefill_state_segments(b, *, views, pcp, state_dtype, state_bytes, chunk,
                            slot_indices, state_capacity, slot_source, initial_flags,
                            global_lengths, lengths, gather, gather_outputs, heads, dk, dv):
    state_segments = []
    contiguous = pcp == 1 and all(len(view.query_ranges) <= 1 for view in views)
    if contiguous:
        contract, effects = gdn_recurrent.state_contract(
            views=views, phase="prefill", state_dtype=state_dtype,
            conv_state_dtype=b.model.activation_dtype,
            conv_kernel=b.plan.conv_kernel, target_prefix=b.prefix,
            chunk=chunk, initial_state=initial_flags, direction="h",
            target_kind="local", global_lengths=global_lengths,
            slot_indices=slot_indices, state_capacity=state_capacity,
            slot_source=slot_source)
        state_segments.append({
            "length": sum(lengths), "lengths": tuple(lengths),
            "contract": contract, "effects": effects,
            "initial_dependency": gather, "exchange_bytes": 0,
            "value_inputs": gather_outputs,
        })
    for view_index, view in enumerate(views):
        if contiguous:
            break
        ranges = tuple(view.query_ranges) or ((0, view.q_len),)
        first_start = int(ranges[0][0]) if ranges else 0
        initial_dependency = gather
        version = view.context
        remote_initial = None
        if pcp > 1 and first_start > 0:
            rem_name = (
                "pcp_remote_initial_state" if view_index == 0
                else f"pcp_remote_initial_state_{view_index}"
            )
            remote_initial = b.work(
                rem_name, (gather,), (0,),
                family="state_transfer",
                metadata=(("status", "pending_external_rank"),
                          ("producer_scope", "external_rank"),
                          ("source_implementation", "pcp1_only"),
                          ("from_version", view.context),
                          ("to_version", view.context + first_start)))
            initial_dependency = remote_initial
            version += first_start
        previous_end = first_start
        for segment_index, (start, end) in enumerate(ranges):
            length = int(end) - int(start)
            if length <= 0:
                continue
            gap = 0 if previous_end is None else int(start) - previous_end
            if gap < 0:
                raise ValueError("gdn_prefill_query_ranges_not_ordered")
            exchange_bytes = 0
            from_version = to_version = None
            if gap:
                from_version = version
                version += gap
                to_version = version
            if pcp == 1 and gap:
                # A single-rank caller may still pack non-contiguous requested
                # ranges; the state advances over the gap without a collective.
                to_version = version
            segment_view = SimpleNamespace(
                request_id=f"{view.request_id}#seg{segment_index}",
                q_len=length, context=version, global_q_len=length,
                query_ranges=((0, length),), padding=False, phase="prefill",
                draft_accepted_tokens=0, draft_advanced_tokens=0)
            initial = (True,) if segment_index else (
                view.context > 0 and not view.padding,)
            contract, effects = gdn_recurrent.state_contract(
                views=(segment_view,), phase="prefill",
                state_dtype=state_dtype,
                conv_state_dtype=b.model.activation_dtype,
                conv_kernel=b.plan.conv_kernel, target_prefix=b.prefix,
                chunk=chunk, initial_state=initial, direction="h",
                target_kind="local", global_lengths=(length,),
                slot_indices=(None if slot_indices is None
                              else (slot_indices[view_index],)),
                state_capacity=state_capacity, slot_source=slot_source)
            segment_inputs = None
            if segment_index == 0:
                if remote_initial is not None:
                    segment_inputs = (gdn_recurrent.tensor_port(
                        TensorValue(
                            "", state_dtype,
                            TensorLayout.contiguous((1, heads, dv, dk)),
                            binding="external",
                            binding_name=(f"pcp_peer:{view.request_id}:state@"
                                          f"{view.context + first_start}"),
                            state_version=view.context + first_start),
                        ready=str(remote_initial), role="remote_input",
                        owner="external_rank"),)
                elif view_index < len(gather_outputs):
                    segment_inputs = (gather_outputs[view_index],)
            state_segments.append({
                "length": length, "lengths": (length,),
                "range": (int(start), int(end)),
                "contract": contract,
                "effects": effects,
                "initial_dependency": (initial_dependency
                                       if segment_index == 0 else None),
                "exchange_bytes": exchange_bytes,
                "from_version": from_version,
                "to_version": to_version,
                "value_inputs": segment_inputs,
            })
            version += length
            previous_end = int(end)
    if not state_segments:
        raise ValueError("gdn_prefill_requires_a_positive_state_segment")
    return state_segments


def core(*, builder, tensors, conv, gated, views, heads, key_heads, dk, dv,
          chunk, gate_values=None, qkv_values=None, pcp=1, pcp_rank=None,
          state_slot_indices=None, state_capacity=None, state_slot_source="", **_):
    b = builder
    tokens = sum(view.q_len for view in views)
    dtype = b.model.activation_dtype
    lengths = tuple(view.q_len for view in views)
    values = (qkv_values if qkv_values is not None else
              rearrange_qkv(tensors, conv, tokens, heads, key_heads, dk, dv))
    # gdn.py always gathers an initial tensor, including zero states at context0.
    # A graph-padded slot keeps its physical state row (initialized to zero) but
    # neither reads a committed state nor later scatters a fake final state.
    state_dtype = b.plan.recurrent_state_dtype or "fp32"
    state_byte = dtype_bytes(state_dtype)
    active = tuple(view for view in views if not view.padding)
    active_elements = len(active) * heads * dk * dv
    state_elements = len(views) * heads * dk * dv
    padding_slots = tuple(index for index, view in enumerate(views) if view.padding)
    prior = tuple(view.context > 0 for view in active)
    prior_elements = sum(prior) * heads * dk * dv
    state_blocks = math.ceil(heads * dk * dv / 1024)
    gather_metadata = state_blocks * (len(views) + STATE_INDEX_BYTES * sum(prior))
    initial_flags = tuple(view.context > 0 and not view.padding for view in views)
    global_lengths = tuple(view.global_q_len or view.q_len for view in views)
    slot_indices = state_slot_indices
    slot_source = state_slot_source
    gather_contract, gather_effects = gdn_recurrent.state_contract(
        views=views, phase="prefill", state_dtype=state_dtype,
        conv_state_dtype=b.model.activation_dtype, conv_kernel=b.plan.conv_kernel,
        target_prefix=b.prefix, chunk=chunk, initial_state=initial_flags,
        direction="gather", target_kind="recurrent",
        global_lengths=global_lengths, slot_indices=slot_indices,
        state_capacity=state_capacity, slot_source=slot_source)
    chunk_start_index = len(b.ops)
    gather = b.work(
        "gather_recurrent_state", (conv,), (len(views), heads, dv, dk),
        read=prior_elements * state_byte + gather_metadata,
        # H reads a physical initial row for every prefill slot, including a
        # padded slot with no persistent cache effect.  Materialize that row
        # (zero for fresh/pad) so the H input is actually defined.
        write=state_elements * state_byte,
        family="state_gather", dtype=state_dtype,
        metadata=(("state_layout", "V,K"), ("has_initial_state", initial_flags),
                  ("padding_slots", padding_slots),
                  ("active_state_rows", len(active)),
                  ("effect_role", "gather_read"),
                  ("state_value_output", "initial_state"),
                  ("state_input_producer", conv),
                  ("state_publish_point", gather_contract["publish_point"]),
                  ("gdn_state_contract", gather_contract),
                  ("gdn_state_effects", gather_effects),
                  ("index_dtype", "int32"), ("metadata_cache_reuse", False)),
    )
    gather_contract, gather_outputs = gdn_recurrent.bind_state_ports(
        gather_contract, gather, heads=heads, head_shards=1)
    b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted({
        **dict(b.ops[-1].metadata),
        "state_ready": gather,
        "state_value_outputs": gather_outputs,
        "gdn_state_contract": gather_contract,
    }.items())))
    gate, beta = gate_values or (
        tensors.result(gated, (1, tokens, heads), "fp32", output="decay"),
        tensors.result(gated, (1, tokens, heads), dtype, output="beta"))
    state_bytes = heads * dk * dv * dtype_bytes(state_dtype)
    state_segments = _prefill_state_segments(
        b, views=views, pcp=pcp, state_dtype=state_dtype, state_bytes=state_bytes, chunk=chunk,
        slot_indices=slot_indices, state_capacity=state_capacity, slot_source=slot_source,
        initial_flags=initial_flags, global_lengths=global_lengths, lengths=lengths,
        gather=gather, gather_outputs=gather_outputs, heads=heads, dk=dk, dv=dv)
    composite = gdn_composite.prefill(
        builder=b, tensors=tensors, q=values[0], k=values[1], v=values[2],
        gate=gate, beta=beta, lengths=lengths, heads=heads, key_heads=key_heads,
        dk=dk, dv=dv, chunk=chunk, initial_state_dependency=gather,
        native_requests=tuple(view.request_id for view in views),
        state_segments=state_segments)
    # A PCP rank publishes a global state commit only when it actually owns
    # the whole global sequence.  Covering the local head and tail of a
    # longer sequence (or a prefix/suffix) still leaves the final state to a
    # peer, so the scatter stays remote-pending and only its local version is
    # committed here.
    remote_flags = tuple(
        pcp_state_selection.pcp_remote_publish(
            view.query_ranges, view.q_len,
            view.global_q_len or view.q_len, pcp=pcp,
            padding=view.padding)
        for view in views)
    scatter_contract, scatter_effects = gdn_recurrent.state_contract(
        views=views, phase="prefill", state_dtype=state_dtype,
        conv_state_dtype=b.model.activation_dtype, conv_kernel=b.plan.conv_kernel,
        target_prefix=b.prefix, chunk=chunk, initial_state=initial_flags,
        direction="scatter", target_kind="recurrent",
        global_lengths=global_lengths, slot_indices=slot_indices,
        state_capacity=state_capacity, slot_source=slot_source,
        remote_publish=remote_flags)
    if pcp > 1:
        # Move every chunk_fwd_o / output-concat op that depends on composite.states
        # (or composite.output) AFTER the PCP state/phi AllGathers, (p-1)-step
        # prefix scan, and rank>0 fwd_h rerun so chunk_fwd_o consumes corrected state.
        state_set = set(composite.states)
        deferred_ops = []
        deferred_ids: set[str] = set()
        kept_ops = []
        for op in b.ops:
            if op.op_id in state_set:
                kept_ops.append(op)
            elif (op.op_id == composite.output
                    or any(d in state_set or d in deferred_ids for d in op.deps)):
                deferred_ops.append(op)
                deferred_ids.add(op.op_id)
            else:
                kept_ops.append(op)
        b.ops[:] = kept_ops
        n_pf = max(1, len(active))
        scan_seq_lens = tuple(int(v.q_len) for v in active if int(v.q_len) > 0)
        scan_tok = sum(scan_seq_lens) if scan_seq_lens else max(1, int(tokens))
        scan_chunk = max(1, int(chunk))
        scan_chunks = max(
            1, sum((q_i + scan_chunk - 1) // scan_chunk for q_i in scan_seq_lens),
        )
        scan_seq_meta = (("sequence_tokens", scan_seq_lens),) if scan_seq_lens else ()
        scan_heads = max(1, int(heads))
        prefix_steps = max(1, int(pcp) - 1) * n_pf
        phi_bytes = float(n_pf * scan_heads * dk * dk * FP32_BYTES)
        st_bytes = float(n_pf * scan_heads * dk * dv * FP32_BYTES)
        hupdate = b.work(
            "pcp_gdn_hupdate", tuple(composite.states), (n_pf, scan_heads, dk, dk),
            flops=float(2 * scan_tok * scan_heads * dk * dk),
            vector=float(scan_chunks * scan_heads * dk * dk),
            read=phi_bytes + float(scan_tok * scan_heads * dk * 2), write=phi_bytes,
            family="recurrent_scan", dtype="fp32",
            metadata=(
                ("pcp", int(pcp)), ("chunks", scan_chunks), ("chunk", scan_chunk),
                ("tokens", scan_tok), ("heads", scan_heads),
                ("source", "ops/triton/fla/chunk.py:fwd_hupdate"),
            ) + scan_seq_meta,
        )
        ag_state = b.communicate(
            "pcp_gdn_state_gather", tuple(composite.states),
            st_bytes * (pcp - 1), "allgather", "pcp", dtype="fp32",
        )
        ag_phi = b.communicate(
            "pcp_gdn_hupdate_gather", (hupdate,),
            phi_bytes * (pcp - 1), "allgather", "pcp", dtype="fp32",
        )
        prefix_scan = b.work(
            "pcp_gdn_prefix_scan", (ag_state, ag_phi), (pcp - 1, n_pf, scan_heads, dk, dv),
            flops=float(2 * prefix_steps * scan_heads * dk * dk * dv),
            vector=float(prefix_steps * scan_heads * dk * dv),
            read=(st_bytes + phi_bytes) * pcp, write=st_bytes,
            family="recurrent_scan", dtype="fp32",
            metadata=(
                ("pcp", int(pcp)), ("chunks", prefix_steps), ("chunk", 1),
                ("tokens", prefix_steps), ("heads", scan_heads),
                ("source", "ops/triton/fla/chunk.py:prefix_scan"),
            ),
        )
        needs_rerun = pcp_rank is None or int(pcp_rank) > 0
        if needs_rerun:
            final_state_op = b.work(
                "pcp_gdn_fwd_h_rerun", (prefix_scan,), (scan_tok, scan_heads, dk, dv),
                flops=float(2 * 2 * scan_tok * scan_heads * dk * dv),
                vector=float(scan_chunks * 2 * scan_heads * dk * dv),
                read=st_bytes, write=st_bytes, family="recurrent_scan", dtype="fp32",
                metadata=(
                    ("pcp", int(pcp)), ("chunks", scan_chunks), ("chunk", scan_chunk),
                    ("tokens", scan_tok), ("heads", scan_heads),
                    ("pcp_rank_gt_zero_rerun", True),
                ) + scan_seq_meta,
            )
            for def_op in deferred_ops:
                if any(d in state_set for d in def_op.deps):
                    new_deps = tuple(dict.fromkeys(
                        [*(d for d in def_op.deps if d not in state_set), final_state_op]
                    ))
                    meta_d = dict(def_op.metadata)
                    sv_in = meta_d.get("state_value_inputs")
                    if sv_in:
                        meta_d["state_value_inputs"] = tuple(
                            ({**p, "producer": final_state_op, "ready": final_state_op,
                              "owner": final_state_op}
                             if isinstance(p, dict) and p.get("producer") in state_set else p)
                            for p in sv_in
                        )
                    if meta_d.get("state_ready") in state_set:
                        meta_d["state_ready"] = final_state_op
                    b.ops.append(replace(def_op, deps=new_deps,
                                         metadata=tuple(meta_d.items())))
                else:
                    b.ops.append(def_op)
        else:
            b.ops.extend(deferred_ops)
        # Upstream ``ops/triton/fla/chunk.py:198,210,240``: prefix_scan yields the
        # global final state for persistent cache scatter on ALL PCP ranks; the
        # rank>0 rerun discards its final-state return and feeds local gdn_output*.
        scatter_deps = (prefix_scan,)
    else:
        scatter_deps = tuple(composite.states)
    scatter = b.work("scatter_recurrent_state", scatter_deps,
           (len(views), heads, dv, dk),
           read=active_elements * state_byte + len(active) * state_blocks * STATE_INDEX_BYTES,
           write=active_elements * state_byte,
           family="state_scatter", dtype=state_dtype,
           metadata=(("state_layout", "V,K"), ("padding_slots", padding_slots),
                     ("active_state_rows", len(active)), ("effect_role", "scatter_publish"),
                     ("state_value_input", scatter_deps),
                     ("state_value_output", "persistent_cache"),
                     ("state_publish_versions", scatter_contract["publish_versions"]),
                     ("state_publish_point", scatter_contract["publish_point"]),
                     ("gdn_state_contract", scatter_contract),
                     ("gdn_state_effects", scatter_effects),
                     ("index_dtype", "int32"), ("metadata_cache_reuse", False)))
    scatter_inputs = tuple(
        gdn_recurrent.tensor_port(TensorValue(
            str(state), state_dtype,
            TensorLayout.contiguous((len(views), heads, dv, dk)),
            state_version=(scatter_contract.get("local_commit_versions") or (None,))[0]),
            ready=str(state), role="state_input", owner=str(state))
        for state in scatter_deps)
    scatter_contract, scatter_outputs = gdn_recurrent.bind_state_ports(
        scatter_contract, scatter, heads=heads, head_shards=1)
    b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted({
        **dict(b.ops[-1].metadata),
        "state_ready": scatter,
        "state_value_inputs": scatter_inputs,
        "state_value_outputs": scatter_outputs,
        "gdn_state_contract": scatter_contract,
    }.items())))
    region_ops = b.ops[chunk_start_index:]
    if region_ops:
        region_ids = {op.op_id for op in region_ops}
        entries = tuple(
            op.op_id for op in region_ops
            if any(dep not in region_ids for dep in op.deps) or not op.deps
        ) or (region_ops[0].op_id,)
        anchor_id = composite.output if composite.output in region_ids else region_ops[-1].op_id
        declare_group(
            b.ops, key=b.prefix + ".gdn_chunk", role="gdn_chunk", family="gdn",
            members=tuple(op.op_id for op in region_ops), entries=entries,
            anchor=anchor_id, block="attention",
            view={"tokens": tokens, "heads": heads, "key_heads": key_heads,
                  "key_width": dk, "value_width": dv, "dtype": dtype,
                  "state_dtype": state_dtype, "sequence_tokens": lengths,
                  "chunk": chunk, "pcp": int(pcp)},
            implementation={
                "native_kind": "ChunkGatedDeltaRuleFwd",
                "native_kinds": ("ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO"),
                "selection": "arch35_gdn_prefill_composite",
                "conditions": {"pcp": int(pcp), "chunk": chunk,
                               "state_dtype": state_dtype},
            },
            source=source_reference("vllm_ascend/ops/gdn.py",
                                    "AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused"),
            fusion_status="fused_existing",
            fusion_reason="arch35_native_gdn_prefill_composite",
            native_fusion_eligible=False,
        )
    return composite.output


def decode_core(b, tensors, values, gate_values, views, heads, keys, dk, dv, *,
                physical_tokens=None, physical_sequence_tokens=None,
                physical_row_offsets=()):
    """Explicit analytical decode until the native recurrence is accepted.

    ``views`` are active query rows; padding physical slots remain in the
    buffer and host partition but never consume or produce a state row.
    """
    tokens = sum(view.q_len for view in views)
    physical = tokens if physical_tokens is None else physical_tokens
    lengths = tuple(view.q_len for view in views)
    normalized = [b.vector(
        "decode_" + name + "_l2norm", (value.producer,), physical * keys * dk, DECODE_QK_L2NORM_OPS_PER_ELEMENT,
        shape=(1, physical, keys, dk),
        metadata=(("source_layout", value.layout.to_dict()),
                  ("implementation_scope", "analytical_decode_l2norm")))
        for name, value in zip(("q", "k"), values[:2])]
    state_bytes = heads * dk * dv * dtype_bytes(b.plan.recurrent_state_dtype)
    input_slices, row_ranges, value_views = [], [], []
    if len(physical_row_offsets) == len(views):
        for view, offset in zip(views, physical_row_offsets):
            q_view = values[0].narrow(1, offset, view.q_len)
            k_view = values[1].narrow(1, offset, view.q_len)
            v_view = values[2].narrow(1, offset, view.q_len)
            gate_views = tuple(value.narrow(1, offset, view.q_len)
                               for value in gate_values)
            # The narrowed TensorValue is the actual stage input: its layout
            # carries the physical byte offset ([32,33) after a pad slot), and
            # its producer/output identity survives instead of being reduced
            # to the original full-buffer producer string.
            value_views.append((q_view, k_view, v_view, *gate_views))
            input_slices.extend(value.producer
                                for value in (q_view, k_view, v_view, *gate_views))
            row_ranges.append((int(offset), int(offset) + view.q_len))
    if input_slices:
        scan_inputs = (*normalized, *input_slices)
    else:
        scan_inputs = (*normalized, values[2].producer,
                       *[value.producer for value in gate_values])
    output = b.work(
        "decode_recurrent_scan", scan_inputs,
        (tokens, heads, dk, dv), family="recurrent_vector",
        vector=DECODE_RECURRENCE_OPS_PER_STATE_ELEMENT * tokens * heads * dk * dv,
        read=len(views) * state_bytes + tokens * heads * ((2 * dk + dv) * dtype_bytes("bf16") + FP32_BYTES + dtype_bytes("bf16")),
        write=tokens * (state_bytes + heads * dv * 2),
        metadata=(("tokens", tokens), ("heads", heads), ("key_heads", keys),
                  ("dk", dk), ("dv", dv), ("sequence_tokens", lengths),
                  ("physical_query_rows", physical),
                  ("physical_sequence_tokens", physical_sequence_tokens or lengths),
                  ("physical_row_offsets", tuple(physical_row_offsets)),
                  ("input_row_ranges", tuple(row_ranges)),
                  ("state_value_views", tuple(
                      tuple(gdn_recurrent.tensor_port(value, role=name,
                                                      ready=value.producer)
                            for name, value in zip(
                                ("q", "k", "v", "decay", "beta"), view_values))
                      for view_values in value_views)),
                  ("state_input_views_actual", tuple(
                      tuple({"producer": value.producer,
                             "output": value.output,
                             "offset": value.layout.offset,
                             "shape": value.layout.shape,
                             "strides": value.layout.strides,
                             "storage_shape": value.layout.storage_shape,
                             "alias_of": value.alias_of,
                             "storage": value.storage,
                             "state_version": value.state_version,
                             "binding": value.binding,
                             "binding_name": value.binding_name}
                            for value in view_values)
                      for view_values in value_views)),
                  ("state_dtype", b.plan.recurrent_state_dtype),
                  ("state_element_bytes", dtype_bytes(b.plan.recurrent_state_dtype)),
                  ("state_checkpoints", tokens),
                  ("source_layouts", tuple(v.layout.to_dict() for v in values)),
                  ("implementation_scope", "analytical_decode_recurrent_vector")))
    b.ops[-1] = replace(b.ops[-1], padded_shape=(physical, heads, dk, dv))
    return tensors.result(output, (physical, heads, dv))


def _lower_prefill_convolution(b, tensors, *, parallel, ordered, prefill, decode, conv_input,
                               padding_slots, initial_flags, global_lengths, cache_indices,
                               state_rows, index_source, null_block_id, runtime, chunk,
                               heads, width, actual):
    prefix, plan, model = b.prefix, b.plan, b.model
    # Physical row positions and qsl stay intact.  FN resolves the cache index
    # per sequence and skips the null/pad *work* while preserving the request's
    # physical offset (the next request's qsl must still start at this row).
    conv_lengths = tuple(view.q_len for view in ordered)
    conv_params = {
        "lengths": conv_lengths,
        "dim": width, "width": plan.conv_kernel, "dtype": model.activation_dtype,
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": initial_flags,
        "cache_indices": cache_indices,
        "state_history_slots": plan.conv_kernel - 1 + int(
            dict(runtime.metadata).get("draft_proposal_tokens", 0)),
        "null_block_id": null_block_id,
    }
    if state_rows is not None:
        conv_params["state_rows"] = state_rows
    conv_op = native_causal_conv_prefill.make_op(conv_params)
    conv_parts = []
    conv_cursor = 0
    for index, view in enumerate(ordered):
        part_direction = "token" if view.phase == "decode" else "conv"
        conv_view = conv_input.narrow(0, conv_cursor, view.q_len)
        conv_cursor += view.q_len
        part, part_effects = gdn_recurrent.state_contract(
            views=(view,), phase=view.phase,
            state_dtype=plan.recurrent_state_dtype,
            conv_state_dtype=plan.conv_state_dtype,
            conv_kernel=plan.conv_kernel, target_prefix=prefix, chunk=chunk,
            initial_state=(initial_flags[index],),
            direction=part_direction, target_kind="conv",
            global_lengths=(global_lengths[index],),
            slot_indices=(None if cache_indices is None else cache_indices[index],),
            state_capacity=state_rows, slot_source=index_source,
            value_inputs=(conv_view,), heads=heads,
            remote_publish=(parallel.pcp > 1
                            and view.phase == "prefill"))
        conv_parts.append((part, part_effects))
    mixed_contract = gdn_recurrent.merge_contracts(
        [part for part, _ in conv_parts], allow_mixed_directions=True)
    mixed_conv = tuple(effect for _, effects in conv_parts for effect in effects)
    conv_meta = {
        "padding_slots": padding_slots,
        "state_dtype": plan.conv_state_dtype,
        "state_layout": "slots_history_channels",
        "state_mutation": "inplace_history_writeback",
        "state_write_bytes": (len(ordered) - len(padding_slots))
                             * (plan.conv_kernel - 1) * width * 2,
        "state_slots": tuple(index for index in range(len(ordered))
                             if index not in padding_slots),
        "state_copy_boundary": "caller_state_rows_to_conv_history_slots",
        "state_index_source": index_source,
        "state_capacity_source": ("caller_metadata" if state_rows is not None
                                  else "unbound"),
        "state_rows": state_rows,
        "state_rows_bound": state_rows is not None,
        "cache_indices": cache_indices,
        "cache_index_present": cache_indices is not None,
        "cache_index_bound": (cache_indices is not None
                              and all(value is not None for value in cache_indices)),
        "null_block_id": null_block_id,
    }
    conv = lower_conv_stage(
        builder=b, tensors=tensors, prefix=prefix, plan=plan,
        parallel=parallel, ordered=ordered, prefill=prefill, decode=decode,
        conv_input=conv_input, conv_params=conv_params, conv_op=conv_op,
        conv_meta=conv_meta, mixed_contract=mixed_contract,
        mixed_conv=mixed_conv, cache_indices=cache_indices,
        state_rows=state_rows, index_source=index_source, chunk=chunk,
        heads=heads, width=width, actual=actual,
    )
    return conv


def _lower_idle_pcp_peer(
    b, tensors, *, projection, zero, prefill, model, plan, parallel,
    tokens, tp, heads, dk, dv, width, value_width,
):
    """Idle PCP peer owns 0 local tokens of a live prefill: join PCP collectives."""
    pcp = int(parallel.pcp)
    n_pf = max(1, len(prefill))
    phi_bytes = float(n_pf * heads * dk * dk * FP32_BYTES)
    st_bytes = float(n_pf * heads * dk * dv * FP32_BYTES)
    idle_start = len(b.ops)
    ag_state = b.communicate(
        "pcp_gdn_state_gather", (zero,),
        st_bytes * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    ag_phi = b.communicate(
        "pcp_gdn_hupdate_gather", (zero,),
        phi_bytes * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    prefix_steps = max(1, pcp - 1) * n_pf
    prefix_scan = b.work(
        "pcp_gdn_prefix_scan", (ag_state, ag_phi), (pcp - 1, n_pf, heads, dk, dv),
        flops=float(2 * prefix_steps * heads * dk * dk * dv),
        vector=float(prefix_steps * heads * dk * dv),
        read=(st_bytes + phi_bytes) * pcp, write=st_bytes,
        family="recurrent_scan", dtype="fp32",
        metadata=(
            ("pcp", pcp), ("chunks", prefix_steps), ("chunk", 1),
            ("tokens", prefix_steps), ("heads", heads),
            ("source", "ops/triton/fla/chunk.py:prefix_scan"),
        ),
    )
    scatter = b.work(
        "scatter_recurrent_state", (prefix_scan,), (len(prefill), heads, dv, dk),
        read=st_bytes, write=st_bytes, family="state_scatter",
        dtype=plan.recurrent_state_dtype,
        metadata=(("state_layout", "V,K"), ("effect_role", "scatter_publish"),
                  ("state_value_input", (prefix_scan,)),
                  ("state_value_output", "persistent_cache")),
    )
    idle_ops = b.ops[idle_start:]
    idle_ids = {op.op_id for op in idle_ops}
    declare_group(
        b.ops, key=b.prefix + ".gdn_chunk", role="gdn_chunk", family="gdn",
        members=tuple(op.op_id for op in idle_ops),
        entries=tuple(op.op_id for op in idle_ops
                      if any(dep not in idle_ids for dep in op.deps) or not op.deps),
        anchor=scatter, block="attention",
        view={"tokens": 0, "heads": heads, "key_heads": heads,
              "key_width": dk, "value_width": dv,
              "dtype": model.activation_dtype, "state_dtype": plan.recurrent_state_dtype,
              "sequence_tokens": (), "pcp": pcp},
        implementation={"native_kind": "ChunkGatedDeltaRuleFwd",
                        "conditions": {"pcp": pcp, "idle_pcp_peer": True}},
        source=source_reference("vllm_ascend/ops/gdn.py",
                                "AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused"),
        fusion_status="fused_existing",
        fusion_reason="arch35_gdn_idle_pcp_peer_sync",
        native_fusion_eligible=False,
    )
    gate = tensors.reshape(projection.narrow(1, width, value_width), (tokens * heads, dv),
                           name="output_gate_reshape")
    gate = tensors.require(gate, "linear_output_norm_gate", "gate", name="output_gate_contiguous")
    normalized = b.vector(
        "linear_output_norm_gate", (zero, gate.producer, prefix_scan),
        tokens * value_width, OUTPUT_NORM_GATE_OPS_PER_ELEMENT,
        shape=(tokens * heads, dv), read_bytes_override=(2 * tokens * value_width + dv) * 2,
        write_bytes_override=tokens * value_width * 2,
        metadata=(("norm_kind", "rmsnorm"), ("norm_has_weight", True),
                  ("norm_has_bias", False), ("norm_group_width", dv),
                  ("gate_present", True), ("recurrent_attention_kind", plan.attention),
                  ("fused_from", ("rmsnorm", "gate"))))
    last = b.matrix("linear_output", (normalized,), tokens, model.hidden_size, value_width,
                    category="gdn.o")
    if tp > 1:
        last = b.communicate("linear_reduce", (last,), 2 * (tp - 1) / tp * tokens * model.hidden_size * 2,
                             "allreduce")
    return AttentionResult(ops=b.ops, last=last)


def lower(*, prefix, plan, views, stage, deps, model, parallel, mode,
          padded_tokens, policy, runtime, physical_rank=None):
    if plan.recurrent_state_dtype != "fp32":
        raise ValueError("gdn_prefill_arch35_state_precision_branch_not_implemented:"
                         + plan.recurrent_state_dtype)
    if plan.conv_state_dtype != model.activation_dtype:
        raise ValueError("gdn_prefill_conv_state_dtype_must_match_activation")
    prefill = tuple(view for view in views if view.phase == "prefill")
    decode = tuple(view for view in views if view.phase == "decode")
    if len(prefill) + len(decode) != len(views) or not prefill:
        raise ValueError("gdn_prefill_arch35_requires_non_spec_prefill")
    # The selected serving caller packs decode rows before prefill rows.
    ordered = decode + prefill
    b = Builder(prefix, plan, stage, ordered, model, mode)
    tensors = TensorLowering(b, runtime.tensor_layouts)
    tp, tokens = parallel.tp, padded_tokens
    actual = sum(view.q_len for view in ordered)
    decode_tokens = sum(view.q_len for view in decode)
    prefill_tokens = actual - decode_tokens
    keys = math.ceil((plan.linear_key_heads or plan.linear_heads) / tp)
    heads = math.ceil((plan.linear_value_heads or plan.linear_heads) / tp)
    dk, dv = plan.linear_key_head_dim or plan.linear_head_dim, plan.linear_head_dim
    width, value_width = 2 * keys * dk + heads * dv, heads * dv
    caller = (("caller_branch", "qkvz_ba_noninterleaved_non_spec"),)
    qkvz = b.matrix("linear_qkvz", (deps,), tokens, width + value_width,
                    model.hidden_size, category="gdn.qkvz", metadata=caller)
    ba = b.matrix("linear_beta_alpha", (deps,), tokens, 2 * heads, model.hidden_size,
                  category="gdn.beta_alpha", metadata=caller)
    projection = tensors.result(qkvz, (tokens, width + value_width))
    gates = tensors.result(ba, (tokens, 2 * heads))
    gate_inputs = [tensors.contiguous(gates.narrow(1, index * heads, heads),
                                     name + "_contiguous", reason="gdn_split_ba.contiguous")
                   for index, name in enumerate(("beta", "alpha"))]
    zero = b.work("core_output_zero", (deps,), (tokens, heads, dv),
                  vector=tokens * value_width, write=tokens * value_width * 2,
                  family="tensor_fill", dtype=model.activation_dtype)
    if actual == 0 and int(parallel.pcp) > 1:
        return _lower_idle_pcp_peer(
            b, tensors, projection=projection, zero=zero, prefill=prefill,
            model=model, plan=plan, parallel=parallel, tokens=tokens, tp=tp,
            heads=heads, dk=dk, dv=dv, width=width, value_width=value_width,
        )
    conv_input = tensors.contiguous(projection.narrow(0, 0, actual).narrow(1, 0, width),
                                    "convolution_input_contiguous",
                                    reason="causal_conv1d_fn.contiguous")
    padding_slots = tuple(index for index, view in enumerate(ordered) if view.padding)
    cache_indices, state_rows, index_source, null_block_id = _state_slot_inputs(
        runtime, ordered)
    chunk = policy.recurrent_chunk if policy else GDN_CHUNK64_TOKENS
    global_lengths = tuple(view.global_q_len or view.q_len for view in ordered)
    initial_flags = tuple(view.context > 0 and not view.padding for view in ordered)
    conv = _lower_prefill_convolution(
        b, tensors, parallel=parallel, ordered=ordered, prefill=prefill, decode=decode,
        conv_input=conv_input, padding_slots=padding_slots, initial_flags=initial_flags,
        global_lengths=global_lengths, cache_indices=cache_indices, state_rows=state_rows,
        index_source=index_source, null_block_id=null_block_id, runtime=runtime, chunk=chunk,
        heads=heads, width=width, actual=actual)
    gated = b.vector(
        "delta_gates", tuple(value.producer for value in gate_inputs), actual * heads, DELTA_GATE_OPS_PER_HEAD,
        family="gdn_gating", shape=(actual, heads),
        read_bytes_override=(actual + 2) * heads * FP32_BYTES,
        write_bytes_override=actual * heads * (FP32_BYTES + dtype_bytes("bf16")),
        metadata=(("outputs", ("fp32_decay", "activation_beta")),
                  ("input_layouts", tuple(v.narrow(0, 0, actual).layout.to_dict()
                                         for v in gate_inputs))))
    qkv = rearrange_qkv(tensors, conv, actual, heads, keys, dk, dv)
    gate_values = (tensors.result(gated, (1, actual, heads), "fp32", output="decay"),
                   tensors.result(gated, (1, actual, heads), output="beta"))
    outputs = []
    if decode:
        decode_active = tuple(view for view in decode if not view.padding)
        decode_offsets, cursor = [], 0
        for view in decode:
            if not view.padding:
                decode_offsets.append(cursor)
            cursor += view.q_len
        decode_start_index = len(b.ops)
        outputs.append(decode_core(
            b, tensors, tuple(v.narrow(1, 0, decode_tokens) for v in qkv),
            tuple(v.narrow(1, 0, decode_tokens) for v in gate_values),
            decode_active, heads, keys, dk, dv,
            physical_tokens=decode_tokens,
            physical_sequence_tokens=tuple(view.q_len for view in decode),
            physical_row_offsets=tuple(decode_offsets)))
        decode_contract, decode_effects = gdn_recurrent.state_contract(
            views=decode, phase="decode", state_dtype=plan.recurrent_state_dtype,
            conv_state_dtype=plan.conv_state_dtype, conv_kernel=plan.conv_kernel,
            target_prefix=prefix, chunk=chunk,
            initial_state=tuple(view.context > 0 and not view.padding
                                for view in decode),
            direction="token", target_kind="recurrent",
            global_lengths=tuple(view.global_q_len or view.q_len for view in decode),
            slot_indices=(None if cache_indices is None else
                          tuple(cache_indices[index] for index, view in enumerate(ordered)
                                if view.phase == "decode")),
            state_capacity=state_rows, slot_source=index_source)
        scan = b.ops[-1]
        scan_meta = dict(scan.metadata)
        decode_value_inputs = tuple(
            view_values[2] for view_values in scan_meta.get("state_value_views", ()))
        decode_contract, decode_outputs = gdn_recurrent.bind_state_ports(
            decode_contract, scan.op_id, heads=heads, head_shards=1,
            value_outputs=None)
        b.ops[-1] = replace(scan, metadata=tuple({**scan_meta,
            "state_ready": scan.op_id,
            "state_value_inputs": decode_value_inputs,
            "state_value_outputs": decode_outputs,
            "gdn_state_contract": decode_contract,
            "gdn_state_effects": decode_effects,
            "padding_slots": decode_contract["padding_slots"],
            "state_checkpoint_versions": tuple(
                None if count <= 0 else (context + 1,)
                for context, count in zip(decode_contract["contexts"],
                                          decode_contract["checkpoints"]))}.items()))
        dec_ops = b.ops[decode_start_index:]
        dec_ids = {op.op_id for op in dec_ops}
        declare_group(
            b.ops, key=prefix + ".gdn_recurrent", role="gdn_recurrent", family="gdn",
            members=tuple(op.op_id for op in dec_ops),
            entries=tuple(op.op_id for op in dec_ops
                          if any(dep not in dec_ids for dep in op.deps) or not op.deps),
            anchor=scan.op_id, block="attention",
            view={"tokens": decode_tokens, "heads": heads, "key_heads": keys,
                  "key_width": dk, "value_width": dv, "dtype": model.activation_dtype,
                  "state_dtype": plan.recurrent_state_dtype,
                  "sequence_tokens": tuple(view.q_len for view in decode)},
            implementation={"native_kind": "RecurrentGatedDeltaRule",
                            "conditions": {"mixed_batch_decode": True,
                                           "state_dtype": plan.recurrent_state_dtype}},
            source=source_reference("vllm_ascend/ops/gdn.py",
                                    "AscendGatedDeltaNetAttention._recurrent_gated_delta_rule"),
            fusion_status="unfused_decomposed",
            fusion_reason="mixed_batch_analytical_gdn_decode",
            native_fusion_eligible=False,
        )
    prefill_slots = (None if cache_indices is None else
                     tuple(cache_indices[len(decode) + index]
                           for index in range(len(prefill))))
    pcp_rank = (
        (int(physical_rank) % max(1, int(parallel.tp) * int(parallel.pcp))) // max(1, int(parallel.tp))
        if physical_rank is not None and int(parallel.pcp) > 1 else None
    )
    output = core(builder=b, tensors=tensors, conv=conv, gated=gated, views=prefill,
                  heads=heads, key_heads=keys, dk=dk, dv=dv, chunk=chunk,
                  state_slot_indices=prefill_slots, state_capacity=state_rows,
                  state_slot_source=index_source,
                  qkv_values=tuple(v.narrow(1, decode_tokens, prefill_tokens) for v in qkv),
                  gate_values=tuple(v.narrow(1, decode_tokens, prefill_tokens) for v in gate_values),
                  pcp=parallel.pcp, pcp_rank=pcp_rank)
    outputs.append(tensors.result(output, (prefill_tokens, heads, dv)))
    if decode:
        flat = [tensors.reshape(value, (value.layout.elements,), name="mixed_output_view")
                for value in outputs]
        output_value = tensors.reshape(tensors.concatenate_flat(flat, "mixed_output_concatenate"),
                                        (actual, heads, dv), name="mixed_output_view")
    else:
        output_value = outputs[0]
    copied = tensors.copy_into(output_value,
                               tensors.result(zero, (tokens, heads, dv)).narrow(0, 0, actual),
                               "core_output_copy")
    gate = tensors.reshape(projection.narrow(1, width, value_width), (tokens * heads, dv),
                           name="output_gate_reshape")
    gate = tensors.require(gate, "linear_output_norm_gate", "gate", name="output_gate_contiguous")
    normalized = b.vector(
        "linear_output_norm_gate", (copied.producer, gate.producer), tokens * value_width, OUTPUT_NORM_GATE_OPS_PER_ELEMENT,
        shape=(tokens * heads, dv), read_bytes_override=(2 * tokens * value_width + dv) * 2,
        write_bytes_override=tokens * value_width * 2,
        metadata=(("norm_kind", "rmsnorm"), ("norm_has_weight", True),
                  ("norm_has_bias", False), ("norm_group_width", dv),
                  ("gate_present", True), ("recurrent_attention_kind", plan.attention),
                  ("fused_from", ("rmsnorm", "gate"))))
    declare_group(
        b.ops, key=prefix + ".gated_rmsnorm", role="gated_rmsnorm",
        family="norm_gate", members=(normalized,), entries=(normalized,),
        anchor=normalized, block="attention",
        view={"tokens": tokens, "heads": heads, "width": dv,
              "dtype": model.activation_dtype, "attention_kind": plan.attention},
        implementation={"native_kind": "_layer_norm_fwd_1pass_kernel_npu",
                        "native_kinds": ("_layer_norm_fwd_1pass_kernel_npu", "FusedRMSNormGated")},
        source=source_reference("vllm_ascend/ops/gdn.py", "FusedRMSNormGated"),
        fusion_status="fused_existing",
        fusion_reason="triton_fused_rmsnorm_gated_kernel",
        native_fusion_eligible=False,
    )
    last = b.matrix("linear_output", (normalized,), tokens, model.hidden_size, value_width,
                    category="gdn.o")
    if tp > 1:
        last = b.communicate("linear_reduce", (last,), 2 * (tp - 1) / tp * tokens * model.hidden_size * 2,
                             "allreduce")
    return AttentionResult(ops=b.ops, last=last)
