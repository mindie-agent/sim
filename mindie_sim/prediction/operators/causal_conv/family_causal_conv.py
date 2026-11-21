"""Explicit stateful causal-convolution update on the shared native program.

[Role] Connect typed KDA producers to one native decode/verify call.
[Invariants] Graph/SP padding rows execute without owning a state slot.
  Speculative verification writes an accepted prefix and rejects later
  candidates; every candidate still performs its real query/history work.
  Prepared weights coexist with original checkpoint weights. Input aliases
  stay views, while separate producers require a real row pack. A state
  history capacity smaller than the source ABI is an error, never a silent
  non-speculative fallback.
[Timeouts & Locks] O(sequence/projection descriptors); no locks or external I/O.
[Side-Effect Policy] Creates model metadata, weight terms and graph operators.
"""
from collections.abc import Mapping
from dataclasses import replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import (
    ProgramFragment as ProgramFragment, SemanticAttr as SemanticAttr,
    StateEffect as StateEffect,
)
from mindie_sim.schema.subgraphs import PortSet as PortSet
from mindie_sim.schema.tensors import TensorLayout as TensorLayout, TensorValue

from .family_causal_conv_state import (
    _conv_bindings, _decision_name as _decision_name, _fragment as _fragment,
    _narrow_window, _per_sequence, _slot_storage as _slot_storage,
    _state_contract, state_accesses as state_accesses,
    state_fragments as state_fragments,
)
from mindie_sim.prediction.operators.linear_attention._gdn_decode_programs import (
    CONV, CONV_SOURCE, CONV_SPECULATIVE_WIDTH, CONV_SUPPORTED_WIDTHS, conv_plan,
)
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe
from mindie_sim.prediction.operators.kda.kda_state_selection import (
    bind_state_value, binding_record, binding_value, decision_ports as decision_ports,
    state_ports as state_ports, state_records as state_records, state_value as state_value,
)
from mindie_sim.prediction.program import KernelProgram, Stage

IMPLEMENTATION = CONV
SOURCE = {
    "caller": {"path": "vllm_ascend/ops/kimi_kda.py",
               "sha256": "35365b3b353fc492b4dc5d75b88916431f6802c21bf51b61b4893d39d54f11f3",
               "lines": "233-255,386-430,544-555,635-648"},
    "kernel": CONV_SOURCE,
    "scope": "source_compatible_update_program; capture_binary_identity_unverified",
    "autocontiguous": "csrc/moe/causal_conv1d/op_host/causal_conv1d_def.cpp:24-28",
    "upstream_projection": {
        "path": "vllm/models/kimi_k3/nvidia/kda.py", "lines": "470-499",
        "sha256": "3663c3b273abf41e9727e2b1724e87628b64b6a9dc048a2c48d002b1f9085420"},
}


def recipes():
    return {CONV: PlannedRecipe(_planned)}


def prepare_model(model, runtime):
    prepared = {}
    for layer in model.layer_plans:
        if layer.attention != "kda":
            continue
        heads = (layer.linear_key_heads or layer.linear_heads,
                 layer.linear_value_heads or layer.linear_heads)
        if any(count <= 0 or count % runtime.parallel.tp for count in heads):
            # The retained KDA producer asserts integral head shards. A
            # channel-wise ceil would undercount replicated whole heads.
            raise ModelingError("causal_conv_update_source_requires_integral_local_heads")
        if model.activation_dtype not in {"bf16", "fp16"}:
            raise ModelingError("causal_conv_update_requires_16bit_activation")
        if layer.conv_state_dtype != model.activation_dtype:
            raise ModelingError("causal_conv_update_state_dtype_must_match_activation")
        if layer.conv_kernel not in CONV_SUPPORTED_WIDTHS:
            raise ModelingError("causal_conv_update_requires_width2to4")
        conv_width = int(getattr(layer, "conv_width", 0) or 0)
        state_elements = int(getattr(layer, "conv_state_elements", 0) or 0)
        if conv_width < 1 or state_elements < 0 or state_elements % conv_width:
            raise ModelingError(
                "causal_conv_update_state_capacity_not_a_whole_history_row")
        history_slots = state_elements // conv_width
        if history_slots < layer.conv_kernel - 1:
            raise ModelingError("causal_conv_update_state_history_below_kernel_width")
        prepared[str(layer.index)] = {
            "dtype": model.activation_dtype,
            "prepared_weight_policy": "retain_original_fp32_checkpoint_weight",
            "state_layout": "slots_history_channels",
            "state_history_slots": history_slots,
            "conv_width": conv_width,
        }
        declared_history = dict(model.metadata).get(
            "causal_conv_history_acceptance", {})
        if str(layer.index) in declared_history:
            prepared[str(layer.index)]["history_acceptance"] = tuple(
                declared_history[str(layer.index)])
    metadata = {**dict(model.metadata), "causal_conv_update": {
        "implementation": IMPLEMENTATION, "layers": prepared, "source": SOURCE}}
    return replace(model, metadata=tuple(sorted(metadata.items())))


def _history_slots(model, plan):
    """Declared whole history rows for this rank's state allocation."""
    prepared = dict(dict(model.metadata).get("causal_conv_update", {}))
    entry = dict(prepared.get("layers", {})).get(str(plan.index), {})
    slots = entry.get("state_history_slots")
    if slots is None:
        width = int(getattr(plan, "conv_width", 0) or 0)
        elements = int(getattr(plan, "conv_state_elements", 0) or 0)
        if width < 1 or elements < 0 or elements % width:
            raise ModelingError(
                "causal_conv_update_state_capacity_not_a_whole_history_row")
        slots = elements // width
    if type(slots) is not int or slots < plan.conv_kernel - 1:
        raise ModelingError("causal_conv_update_state_history_below_kernel_width")
    return slots


def _planned(op, geometry, *, compact=True):
    """Turn the shared conv program into a plan that carries state semantics.

    The physical program remains ``_gdn_decode_programs.conv_plan``.  Only
    zero-work read/write boundary stages and plan metadata are added; no
    convolution or state work is duplicated.
    """
    plan = conv_plan(op, geometry, compact=compact)
    contract = dict(op.metadata).get("conv_state_contract")
    if contract is None:
        return plan
    fragments = state_fragments(op)

    def program(context):
        inner = plan.program(context)
        stages = []
        if fragments["read"] is not None and fragments["read"].state_effects:
            stages.append(Stage("state_read", fragment=fragments["read"]))
            stages.append(Stage("conv_owners", after=("state_read",),
                                program=inner))
        else:
            stages.append(Stage("conv_owners", program=inner))
        if fragments["write"] is not None and fragments["write"].state_effects:
            stages.append(Stage("state_publish", after=("conv_owners",),
                                fragment=fragments["write"]))
        return KernelProgram(tuple(stages))

    metadata = dict(plan.metadata)
    metadata.update({
        "state_contract": contract,
        "state_accesses": state_accesses(op),
        "state_effects": dict(op.metadata).get("conv_state_effects", ()),
        "conv_state_contract": contract,
        "conv_state_effects": dict(op.metadata).get("conv_state_effects", ()),
        "state_fragment_counts": {
            "read": 1 if fragments["read"] is not None else 0,
            "write": 1 if fragments["write"] is not None else 0,
            "commits": len(fragments["commits"]),
            "rollbacks": len(fragments["rollbacks"]),
        },
    })
    return OperatorPlan(plan.work, program, metadata=metadata)


def _attach_rebase_accesses(b, *, index, view, plan, width, context, target,
                            layer_target, binding, staged, staging_id, staging_grant,
                            rebase_id, source_offset, copy_rows, copy_bytes, hold_grant,
                            state_ready, hold_event, protected, source_view, dest_view):
    source_producer = binding["producer"]
    source_ready = binding.get("ready") or source_producer
    staging_accesses = ()
    if staged:
        staging_read = {
            "occurrence": f"{staging_id}:read:{index}",
            "producer": source_producer, "request": str(
                getattr(view, "request_id", index)),
            "stage": b.common["stage"], "layer": plan.index,
            "direction": "read", "position": "state",
            "output": target,
            "storage": binding["storage"],
            "owner": layer_target, "representation": "history",
            "persistent": False, "temporary": True,
            "state_version": context,
            "span": ((source_offset, source_offset + copy_rows),
                     (0, width)), "bytes": float(copy_bytes),
            "source": source_ready, "reader": "",
            "grant": staging_grant,
            "decision": None, "last_use": "before_staging_write",
            "rank": None, "scope": "target",
            "generation": context, "slot": binding.get("slot"),
            "offset": binding.get("offset"),
            "physical_key": f"{binding['storage']}:staging",
            "request_slot": binding.get("slot"), "rows": copy_rows,
            "row_stride": width, "tier": "hbm",
            "source_ready": source_ready,
            "grant_ready": staging_grant,
            "decision_event": None,
            "last_use_event": "before_staging_write",
        }
        staging_write = {
            **staging_read,
            "occurrence": f"{staging_id}:write:{index}",
            "producer": staging_id, "direction": "write",
            "position": "state", "temporary": True,
            "storage": f"{binding['storage']}:staging",
            "source": staging_id, "decision": "staging_complete",
            "last_use": "before_rebase_write",
        }
        staging_accesses = (staging_read, staging_write)
        for position, staged_op in enumerate(b.ops):
            if staged_op.op_id == staging_id:
                b.ops[position] = replace(
                    staged_op,
                    metadata=staged_op.metadata + (
                        ("state_accesses", staging_accesses),))
                break
    rebase_read = {
        "occurrence": f"{rebase_id}:read:{index}",
        "producer": (staging_id if staged else source_producer),
        "request": str(getattr(view, "request_id", index)),
        "stage": b.common["stage"], "layer": plan.index,
        "direction": "read", "position": "state",
        "output": target,
        "storage": (f"{binding['storage']}:staging" if staged
                    else binding["storage"]),
        "owner": layer_target, "representation": "history",
        "persistent": True, "temporary": False,
        "state_version": context,
        "span": ((source_offset, source_offset + copy_rows), (0, width)),
        "bytes": float(copy_bytes),
        "source": (staging_id if staged else source_ready),
        "reader": "", "grant": hold_grant, "decision": None,
        "last_use": "before_short_convolution", "rank": None,
        "scope": "target", "generation": context,
        "slot": binding.get("slot"), "offset": binding.get("offset"),
        "physical_key": binding["storage"],
        "request_slot": binding.get("slot"), "rows": copy_rows,
        "row_stride": width, "tier": "hbm",
        "source_ready": (staging_id if staged else source_ready),
        "grant_ready": hold_grant, "decision_event": None,
        "last_use_event": "before_short_convolution",
    }
    rebase_write = {
        **rebase_read,
        "occurrence": f"{rebase_id}:write:{index}",
        "producer": rebase_id, "direction": "write",
        "storage": binding["storage"], "source": rebase_id,
        "span": ((0, copy_rows), (0, width)),
        "window": (0, copy_rows),
        "decision": "copy_complete",
        "last_use": "before_short_convolution",
    }
    b.ops[-1] = replace(b.ops[-1], metadata=b.ops[-1].metadata + (
        ("state_accesses", (*staging_accesses, rebase_read)),
        ("state_accesses_write", (rebase_write,)),
        ("state_ready", tuple(state_ready)),
        ("reader_hold", hold_event),
        ("grant", hold_grant),
        ("protected_rows", protected),
        ("copy_source_value", source_view.to_dict()),
        ("copy_dest_value", dest_view.to_dict()),
    ))


def _rebase_history(b, *, index, view, plan, dtype, history_slots, width, byte,
                    accepted, binding, hold, state_staging, state_ready,
                    context, target, layer_target,
                    rebase_outputs, rebase_records):
    if plan.conv_kernel != CONV_SPECULATIVE_WIDTH:
        raise ModelingError(
            "causal_conv_update_history_rebase_requires_width4")
    if accepted - 1 + plan.conv_kernel - 1 > history_slots:
        raise ModelingError(
            "causal_conv_update_history_rebase_exceeds_state")
    source_value = binding_value(binding)
    source_offset = accepted - 1
    copy_rows = plan.conv_kernel - 1
    source_view = _narrow_window(source_value, source_offset, copy_rows)
    dest_view = _narrow_window(source_value, 0, copy_rows)
    hold_window = None
    hold_event = None
    hold_grant = None
    if isinstance(hold, Mapping):
        window = hold.get("window", hold.get("rows"))
        if window is not None:
            hold_window = tuple(int(item) for item in window)
        hold_event = hold.get("event", hold.get("last_use"))
        hold_grant = hold.get("grant")
    protected = ()
    if hold_window is not None:
        start = max(hold_window[0], 0)
        end = min(hold_window[1], copy_rows)
        if end > start:
            protected = (start, end)
    staged = bool(state_staging or (protected and not hold_event))
    copy_bytes = copy_rows * width * byte
    copy_deps = tuple(state_ready)
    if hold_event:
        copy_deps = tuple(dict.fromkeys((*copy_deps, hold_event)))
    staging_id = None
    staging_value = None
    staging_grant = hold_grant or f"{b.prefix}.rebase_grant"
    if staged:
        staging_id = b.work(
            "conv_history_staging", copy_deps, (copy_rows, width),
            read=copy_bytes, write=copy_bytes, family="tensor_copy",
            dtype=dtype,
            metadata=(("materialization_reason",
                       "independent_overlap_staging"),
                      ("source_offset_rows", (source_offset,)),
                      ("dest_offset_rows", (0,)),
                      ("history_rows", copy_rows),
                      ("allocation_bytes", copy_bytes),
                      ("grant", staging_grant),
                      ("temporary", True),
                      ("caller_boundary", "old_reader_may_overlap_destination")))
        staging_value = replace(
            source_view, producer=staging_id, output=0,
            storage=f"{binding['storage']}:staging",
            state_version=context, binding="static",
            binding_name="")
        copy_deps = (staging_id,)
    rebase_id = b.work(
        "conv_history_rebase", copy_deps, (copy_rows, width),
        read=copy_bytes, write=copy_bytes, family="tensor_copy", dtype=dtype,
        metadata=(("materialization_reason",
                   "committed_history_window_normalize"),
                  ("source_offset_rows", (source_offset,)),
                  ("dest_offset_rows", (0,)),
                  ("history_rows", copy_rows),
                  ("copy_order", ("staged_then_write" if staged
                                  else "descending_overlap_in_place")),
                  ("state_alias", "same_conv_state_buffer"),
                  ("allocation", ("independent_staging" if staged
                                  else "in_place_no_new_slot")),
                  ("staging_bytes", copy_bytes if staged else 0),
                  ("last_use", "before_short_convolution"),
                  ("state_layout", "slots_history_channels"),
                  ("state_dtype", plan.conv_state_dtype),
                  ("caller_boundary", "before_ordinary_q1_update"),
                  ("invocation_class", "vector")))
    copy_value = TensorValue(
        rebase_id, plan.conv_state_dtype, dest_view.layout, output=0,
        storage=binding["storage"], alias_of=(source_value.alias_of
                                               or binding["storage"]),
        state_version=context, binding="static", binding_name="")
    copy_binding = bind_state_value(
        copy_value, target=target, owner=layer_target,
        slot=binding.get("slot"), version=context,
        ready=rebase_id, grant=hold_grant, generation=context,
        required=True, representation="history",
        window=(0, copy_rows),
        span=((0, copy_rows), (0, width)),
        readiness="producer_complete",
        last_use="before_short_convolution")
    rebase_outputs[index] = copy_binding
    _attach_rebase_accesses(
        b, index=index, view=view, plan=plan, width=width, context=context,
        target=target, layer_target=layer_target, binding=binding,
        staged=staged, staging_id=staging_id, staging_grant=staging_grant, rebase_id=rebase_id,
        source_offset=source_offset, copy_rows=copy_rows, copy_bytes=copy_bytes,
        hold_grant=hold_grant, state_ready=state_ready, hold_event=hold_event,
        protected=protected, source_view=source_view, dest_view=dest_view)
    rebase_records.append({
        "sequence": index,
        "request": str(getattr(view, "request_id", index)),
        "accepted_tokens": accepted,
        "source_context": context,
        "source_version": context,
        "dest_version": context,
        "source_offset": source_offset,
        "dest_offset": 0,
        "rows": copy_rows,
        "bytes": copy_bytes,
        "source_window": (source_offset, source_offset + copy_rows),
        "dest_window": (0, copy_rows),
        "storage": binding["storage"],
        "slot": binding.get("slot"),
        "copy_op": rebase_id,
        "copy_output": copy_value.to_dict(),
        "staging_op": staging_id,
        "staging_bytes": copy_bytes if staged else 0,
        "order": ("staged_then_write" if staged
                  else "descending_overlap_in_place"),
        "allocation": ("independent_staging" if staged
                       else "in_place_no_new_slot"),
        "last_use": "before_short_convolution",
        "reader_hold": hold_event,
        "grant": hold_grant,
        "held_rows": hold_window,
        "protected_rows": protected,
    })
    return rebase_id


def _prepare_conv_input(b, tensors, *, sources, tokens, rows, width, key_width,
                        value_width, dtype, byte):
    by_id = {op.op_id: op for op in b.ops}
    producers = tuple(dict.fromkeys(sources))
    if any(dict(by_id[name].metadata).get("output_format", dtype) != dtype
           for name in producers):
        raise ModelingError("causal_conv_update_projection_dtype_mismatch")
    if len(producers) == 1:
        projection = by_id[producers[0]]
        slices = dict(projection.metadata).get("packed_projection_slices", ())
        qkv = [item for item in slices if item["role"] in {"q", "k", "v"}]
        if (tuple(item["role"] for item in qkv) != ("q", "k", "v")
                or tuple(item["offset_elements"] for item in qkv) != (0, key_width, 2 * key_width)):
            raise ModelingError("causal_conv_update_requires_declared_contiguous_qkv_slice")
        value = tensors.result(producers[0], (tokens, projection.local_shape[1]), dtype)
        value = value.narrow(0, 0, rows).narrow(1, 0, width)
        incoming = tensors.contiguous(value, "convolution_input_contiguous",
                                      reason="CausalConv1d.x.AutoContiguous").producer
        adaptation = "packed_projection_qkv_slice; AutoContiguous_if_strided"
    else:
        if len(sources) != 3 or len(producers) != 3:
            raise ModelingError("causal_conv_update_requires_three_qkv_producers_or_one_packed")
        incoming = b.work(
            "convolution_input_pack", producers, (rows, width),
            read=rows * width * byte, write=rows * width * byte,
            family="tensor_copy", dtype=dtype,
            metadata=(("materialization_reason", "rowwise_qkv_concatenation"),
                      ("source_row_widths", (key_width, key_width, value_width)),
                      ("caller_boundary", "declared_separate_producer_adapter_not_native_projection_claim"),
                      ("invocation_class", "vector")))
        adaptation = "explicit_row_pack_of_separate_semantic_producers"
    return incoming, adaptation


def _validate_update_window(model, plan, *, views, q_lens, active, history_slots,
                            history_acceptance):
    if any(q_lens[index] == 1 and (
            int(getattr(views[index], "draft_proposal_tokens", 0) or 0)
            or int(getattr(views[index], "draft_accepted_tokens", 0) or 0)
            or int(getattr(views[index], "draft_advanced_tokens", 0) or 0) > 1)
           for index in active):
        raise ModelingError("causal_conv_update_requires_q1_non_speculative_decode")
    for index in active:
        q = q_lens[index]
        if q <= 1:
            continue
        accepted = int(getattr(views[index], "draft_accepted_tokens", 0) or 0)
        advance = int(getattr(views[index], "draft_advanced_tokens", 0) or 0)
        if advance > 0:
            if not 1 <= advance <= q:
                raise ModelingError("causal_conv_update_advance_outside_query")
            if accepted not in (0, advance - 1):
                raise ModelingError("causal_conv_update_accepted_advance_mismatch")
    if history_acceptance is None:
        prepared = dict(dict(model.metadata).get(
            "causal_conv_update", {})).get("layers", {})
        history_acceptance = dict(prepared.get(str(plan.index), {})).get(
            "history_acceptance")
    if history_acceptance is None:
        history_acceptance = dict(
            dict(model.metadata).get("causal_conv_history_acceptance", {})
        ).get(str(plan.index))
    if history_acceptance is not None:
        history_acceptance = tuple(int(value) for value in history_acceptance)
        if len(history_acceptance) != len(views) or any(
                value < 0 for value in history_acceptance):
            raise ModelingError(
                "causal_conv_update_history_acceptance_invalid")
    speculative = any(
        q_lens[index] > 1
        or int(getattr(views[index], "draft_proposal_tokens", 0) or 0) > 0
        or int(getattr(views[index], "draft_advanced_tokens", 0) or 0) > 1
        for index in active)
    if speculative:
        if plan.conv_kernel != CONV_SPECULATIVE_WIDTH:
            raise ModelingError(
                "causal_conv_update_speculative_requires_width4_history")
        max_query = max((q_lens[index] for index in active), default=1)
        if history_slots < 2 + max_query:
            raise ModelingError(
                "causal_conv_update_speculative_requires_extended_history:"
                f"{history_slots}<{2 + max_query}")
        if history_acceptance is None:
            raise ModelingError(
                "causal_conv_update_history_acceptance_required")
    return history_acceptance, speculative


def lower(b, tensors, *, plan, views, tokens, sources, key_width, value_width,
          history_acceptance=None, state_ready=(), state_values=None,
          state_slots=None, state_grants=(), state_reader_holds=None,
          state_targets=None, state_staging=False, state_decisions=()):
    """Bind one native update; the selected recipe owns the state writeback."""
    if not views:
        raise ModelingError("causal_conv_update_requires_decode_views")
    if any(view.phase != "decode" for view in views):
        raise ModelingError("causal_conv_update_requires_decode_only_views")
    q_lens = tuple(int(view.q_len) for view in views)
    if any(q < 1 for q in q_lens):
        raise ModelingError("causal_conv_update_requires_positive_query_rows")
    rows = sum(q_lens)
    if tokens < rows:
        raise ModelingError(
            "causal_conv_update_tokens_must_equal_sequence_partition_rows")
    width = 2 * key_width + value_width
    dtype = b.model.activation_dtype
    byte = dtype_bytes(dtype)
    history_slots = _history_slots(b.model, plan)
    padding_slots = tuple(index for index, view in enumerate(views)
                          if getattr(view, "padding", False))
    active = tuple(index for index in range(len(views))
                   if index not in set(padding_slots))
    history_acceptance, speculative = _validate_update_window(
        b.model, plan, views=views, q_lens=q_lens, active=active, history_slots=history_slots,
        history_acceptance=history_acceptance)
    state_ready = ((state_ready,) if isinstance(state_ready, str)
                   else tuple(state_ready))
    state_grants = tuple(state_grants) if state_grants else ()
    contexts = tuple(int(getattr(view, "context", 0)) for view in views)
    layer_target = (f"kda.layer[{plan.index}]" if plan.index >= 0
                    else "kda.call")
    if state_targets is None:
        targets_by_view = tuple(
            None if index in set(padding_slots)
            else f"{layer_target}.conv_state["
                 f"{getattr(views[index], 'request_id', index)}]"
            for index in range(len(views)))
    else:
        state_targets = tuple(state_targets)
        if len(state_targets) != len(views):
            raise ModelingError("causal_conv_update_targets_must_match_views")
        targets_by_view = tuple(
            None if index in set(padding_slots) else state_targets[index]
            for index in range(len(views)))
    slots = _per_sequence(state_slots, len(views))
    if all(slot is None for slot in slots):
        slots = tuple(index for index in range(len(views)))
    bindings = _conv_bindings(
        views=views, active=active, targets=targets_by_view,
        contexts=contexts, history_slots=history_slots, width=width,
        dtype=plan.conv_state_dtype, values=state_values, slots=slots,
        owner=layer_target, ready=state_ready, grants=state_grants)
    # A q1 following a speculative q4 has no source numAcceptedTokens branch
    # for its own one-row query.  The committed window is normalized by an
    # ordered history copy before the ordinary UPDATE.  In a mixed speculative
    # call the native branch is selected once for the whole call; every
    # sequence then applies its own accepted-1 offset exactly once and no
    # sequence may additionally rebase.
    rebase = tuple(
        index for index in active
        if not speculative and q_lens[index] == 1
        and history_acceptance is not None
        and history_acceptance[index] > 0)
    rebase_records = []
    rebase_outputs = {}
    rebase_ids = ()
    reader_holds = _per_sequence(state_reader_holds, len(views))
    for index in rebase:
        rebase_id = _rebase_history(
            b, index=index, view=views[index], plan=plan, dtype=dtype, history_slots=history_slots,
            width=width, byte=byte, accepted=history_acceptance[index], binding=bindings[index],
            hold=reader_holds[index], state_staging=state_staging, state_ready=state_ready,
            context=contexts[index], target=targets_by_view[index], layer_target=layer_target,
            rebase_outputs=rebase_outputs, rebase_records=rebase_records)
        rebase_ids = (*rebase_ids, rebase_id)
    contract, effects, state_read, state_write = _state_contract(
        views=views, q_lens=q_lens, padding_slots=padding_slots, width=width,
        byte=byte, kernel_width=plan.conv_kernel, history_slots=history_slots,
        speculative=speculative, layer=plan.index, bindings=bindings,
        history_acceptance=history_acceptance, reader_holds=reader_holds,
        slots=slots, targets=targets_by_view, rebase_outputs=rebase_outputs,
        state_decisions=state_decisions)
    contract = {**contract, "state_dtype": plan.conv_state_dtype,
                "conv_width": width, "state_ready": tuple(state_ready),
                "state_grants": tuple(state_grants),
                "slots": tuple(slots),
                "reader_holds": tuple(reader_holds)}
    incoming, adaptation = _prepare_conv_input(
        b, tensors, sources=sources, tokens=tokens, rows=rows, width=width, key_width=key_width,
        value_width=value_width, dtype=dtype, byte=byte)
    num_accepted_values = history_acceptance if speculative else None
    metadata = (
        ("sequence_tokens", q_lens), ("physical_sequence_tokens", q_lens),
        ("padding_slots", padding_slots),
        ("kernel_width", plan.conv_kernel),
        ("speculative", speculative), ("state_history_slots", history_slots),
        ("state_dtype", plan.conv_state_dtype), ("state_layout", "slots_history_channels"),
        ("state_mutation", "inplace_history_writeback"),
        ("state_read_bytes", state_read), ("state_write_bytes", state_write),
        ("state_read_versions", contract["read_versions"]),
        ("state_commit_versions", contract["commit_versions"]),
        ("state_checkpoint_versions", contract["checkpoint_versions"]),
        ("conv_state_contract", contract), ("conv_state_effects", effects),
        ("state_bindings", tuple(
            binding_record(binding) for binding in bindings if binding)),
        ("num_accepted_tokens_present", speculative),
        ("num_accepted_tokens_dtype", "int32"),
        ("num_accepted_tokens_shape", (len(views),)),
        ("num_accepted_tokens_values", tuple(num_accepted_values or ())),
        ("state_token_offsets", tuple(
            max(0, value - 1) for value in (num_accepted_values or ()))),
        ("history_acceptance_basis",
         "caller_declared" if history_acceptance is not None else "absent"),
        ("has_initial_state", True), ("has_conv_states", bool(active)),
        ("cache_index_dtype", "int32"), ("cache_indices", "one_valid_slot_per_active_sequence"),
        ("query_start_loc_dtype", "int32"), ("bias", False),
        ("activation", "silu"), ("run_mode", 1), ("source", SOURCE),
        ("speculative_source_selector",
         "num_accepted_tokens_presence_width4" if speculative else "none"),
        ("history_rebase", tuple(rebase_records)),
        ("history_rebase_bytes", sum(record["bytes"]
                                     for record in rebase_records)),
        ("history_rebase_op", rebase_ids[0] if rebase_ids else None),
        ("state_ready", tuple(state_ready)),
        ("state_slots", tuple(slots)),
        ("state_decisions", tuple(state_decisions)),
        ("state_grants", tuple(state_grants)),
        ("reader_holds", tuple(reader_holds)),
        ("input_adaptation", adaptation), ("prepared_weight_dtype", dtype),
        ("prepared_weight_layout", "kernel_width_channels"),
        ("native_kind", "CausalConv1d"), ("invocation_class", "vector"),
        ("implementation", IMPLEMENTATION),
        ("limitations", ("scalar_metadata_issue_and_cache_latency_unmodeled",
                          "source_update_kernel_revision_not_capture_binary_identity",
                          "same_shape_owner_runs_use_existing_fluid_compaction",
                          "speculative_history_requires_declared_state_capacity")),
    )
    op_id = b.work("short_convolution",
                   (incoming, *state_ready, *rebase_ids), (rows, width),
                   family="short_convolution", dtype=dtype, metadata=metadata)
    op = b.ops[-1]
    b.ops[-1] = replace(
        op, recipe=CONV, source_boundary="kernel",
        metadata=op.metadata + (("state_accesses", state_accesses(op)),))
    return op_id
