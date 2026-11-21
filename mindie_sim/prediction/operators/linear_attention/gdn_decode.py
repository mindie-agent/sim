"""Non-interleaved QKVZ/BA caller with explicit GDN decode kernel boundaries.

[Role] Bind update convolution, gating, Q/K normalization and state recurrence.
[Invariants] Caller copies replace generic work. Query rows do not shrink with
  speculative acceptance; full capture retains physical buffers while padding
  slots own no state and dummy zero-length slots perform no convolution or
  recurrent state work. Conv history is activation-dtype; recurrent state is
  FP32. Each active query produces a candidate checkpoint; only the accepted
  offset commits at token commit, and unaccepted candidates are rejected.
  Each named copy/fill is a declared vector invocation, not an extra host gap.
[Timeouts & Locks] O(sequence descriptors), lazily executed physical kernels.
[Side-Effect Policy] Builds operators only; no measurements or device contexts.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from ...graph import GraphPolicy
from ...lowering import Builder
from ...mechanism import PlannedRecipe
from ...recover_ssm_work import (
    GDN_ARCH35_RECOVER_SSM_RECIPE,
    emit_commit_op, recover_ssm_enabled, rewrite_gdn_analytical_scan,
)
from ...tensors import TensorLowering
from . import gdn_recurrent
from ._gdn_decode_programs import CONV, GATING, CONV_SOURCE, conv_plan, gating_plan

SOURCE = {"path": "data/observations/sources/gdn-image-c3d00a196905/gdn.py.txt",
          "sha256": "e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98",
          "caller_lines": ((298, 379), (505, 575), (705, 737))}
CACHE_INDEX_SOURCE_LINES = ((138, 143), (435, 445))
OUTPUT_NORM_GATE_OPS_PER_ELEMENT = 9


def _build_conv(op, geometry):
    if "conv_state_contract" in dict(op.metadata):
        from ...family_causal_conv import _planned

        return _planned(op, geometry)
    return conv_plan(op, geometry)


def _build_recover_ssm(op, geometry):
    from ...recover_ssm_work import build_gdn_arch35_recover_ssm

    if any(int(rows) > 1 for rows in dict(op.metadata)["sequence_tokens"]):
        return build_gdn_arch35_recover_ssm(op, geometry)
    return gdn_recurrent.build(op, geometry)


_BUILDERS = {
    CONV: _build_conv,
    GATING: gating_plan,
    gdn_recurrent.IMPLEMENTATION: gdn_recurrent.build,
    GDN_ARCH35_RECOVER_SSM_RECIPE: _build_recover_ssm,
}
RECIPE_IDS = frozenset(_BUILDERS)


def build(op, geometry):
    """Bind a native GDN decode recipe to its exact kernel program."""
    return _BUILDERS[op.recipe](op, geometry)


def recipes():
    return dict.fromkeys(_BUILDERS, PlannedRecipe(build))


def lower(*, prefix, plan, views, stage, deps, model, parallel, mode,
          padded_tokens, policy, runtime, gating_beta_store=None, gating_programs=None,
          recurrent_implementation="analytical", cache_index_row_stride=1):
    from ...call_gdn import GATING_SOURCE
    from ...gdn_prefill import decode_core, rearrange_qkv

    if not views or any(view.phase != "decode" for view in views):
        raise ModelingError("gdn_decode_selected_caller_requires_decode_only")
    if model.activation_dtype != "bf16" or plan.recurrent_state_dtype != "fp32":
        raise ModelingError("gdn_decode_requires_bf16_inputs_fp32_state")
    if (not isinstance(cache_index_row_stride, int) or isinstance(cache_index_row_stride, bool)
            or cache_index_row_stride < 1):
        raise ModelingError("gdn_decode_cache_index_row_stride_requires_positive_integer")
    conv_dtype = plan.conv_state_dtype
    if conv_dtype != model.activation_dtype:
        raise ModelingError("gdn_decode_conv_state_dtype_must_match_activation")
    b = Builder(prefix, plan, stage, views, model, mode)
    tensors = TensorLowering(b, runtime.tensor_layouts)
    tp, tokens = parallel.tp, padded_tokens
    physical = sum(view.q_len for view in views)
    active = tuple(view for view in views if not view.padding)
    active_rows = sum(view.q_len for view in active)
    _, capture_scope = GraphPolicy().select(views, runtime)
    core_tokens = tokens if capture_scope == "full" else physical
    if core_tokens < physical:
        raise ModelingError(f"gdn_decode_capture_rows_below_physical:{core_tokens}:{physical}")
    extra = core_tokens - physical
    zero_slots = 0
    if extra:
        # A captured graph owns physical request slots, not a fluid share of
        # leftover tokens.  The caller must declare the real partition (padding
        # views or an exact uniform slot width); a remainder is not rounded into
        # a guessed balanced batch.
        if (any(view.padding for view in views)
                or len({view.q_len for view in views}) != 1
                or views[0].q_len <= 0 or extra % views[0].q_len):
            raise ModelingError(
                f"gdn_decode_capture_requires_exact_uniform_partition:{extra}")
        zero_slots = extra // views[0].q_len
    lengths = tuple(view.q_len for view in views) + (0,) * zero_slots
    # A padding view retains its physical rows but owns no state: it adds a
    # zero-length host-grid slot rather than a fake request/state row.
    padding_slots = (tuple(index for index, view in enumerate(views) if view.padding)
                     + tuple(range(len(views), len(views) + zero_slots)))
    speculative = any(view.q_len > 1 for view in active)
    max_query = max((view.q_len for view in active), default=1)
    initial_flags = tuple(view.context > 0 and not view.padding for view in views)
    runtime_meta = dict(runtime.metadata)
    state_capacity = runtime_meta.get(
        "gdn_state_rows", runtime_meta.get("gdn_state_capacity"))
    raw_slots = runtime_meta.get("gdn_decode_state_indices",
                                 runtime_meta.get("gdn_state_indices"))
    slot_source = "caller_tensor"
    slot_indices = None
    if raw_slots is not None:
        if isinstance(raw_slots, dict):
            try:
                slot_indices = tuple(
                    None if raw_slots[view.request_id] is None
                    else int(raw_slots[view.request_id]) for view in views)
            except (KeyError, TypeError, ValueError):
                raise ModelingError("gdn_decode_state_indices_missing_request")
        elif isinstance(raw_slots, (list, tuple)) and len(raw_slots) == len(views):
            slot_indices = tuple(None if value is None else int(value)
                                 for value in raw_slots)
        else:
            raise ModelingError("gdn_decode_state_indices_length_mismatch")
        slot_source = "caller_metadata"
    contract, recurrent_effects = gdn_recurrent.state_contract(
        views=views, phase="decode", state_dtype=plan.recurrent_state_dtype,
        conv_state_dtype=conv_dtype, conv_kernel=plan.conv_kernel,
        target_prefix=prefix, zero_slots=zero_slots,
        index_row_stride=cache_index_row_stride, index_slots=len(lengths),
        initial_state=initial_flags, direction="token",
        target_kind="recurrent", slot_source=slot_source,
        state_capacity=state_capacity, slot_indices=slot_indices)
    conv_contract, conv_effects = gdn_recurrent.state_contract(
        views=views, phase="decode", state_dtype=plan.recurrent_state_dtype,
        conv_state_dtype=conv_dtype, conv_kernel=plan.conv_kernel,
        target_prefix=prefix, zero_slots=zero_slots,
        index_row_stride=cache_index_row_stride, index_slots=len(lengths),
        initial_state=initial_flags, direction="token", target_kind="conv",
        slot_source=slot_source, state_capacity=state_capacity,
        slot_indices=slot_indices)
    keys = math.ceil((plan.linear_key_heads or plan.linear_heads) / tp)
    heads = math.ceil((plan.linear_value_heads or plan.linear_heads) / tp)
    dk, dv = plan.linear_key_head_dim or plan.linear_head_dim, plan.linear_head_dim
    if heads % keys:
        raise ModelingError("gdn_decode_requires_integer_value_to_key_head_ratio")
    width, value_width = 2 * keys * dk + heads * dv, heads * dv
    qkvz = b.matrix("linear_qkvz", (deps,), tokens, width + value_width, model.hidden_size, category="gdn.qkvz")
    ba = b.matrix("linear_beta_alpha", (deps,), tokens, 2 * heads, model.hidden_size, category="gdn.beta_alpha")
    projection = tensors.result(qkvz, (tokens, width + value_width))
    gates = tensors.result(ba, (tokens, 2 * heads))
    gate_inputs = [tensors.contiguous(gates.narrow(1, index * heads, heads),
                                     name + "_contiguous", reason="gdn_split_ba.contiguous")
                   for index, name in enumerate(("beta", "alpha"))]
    zero = b.work("core_output_zero", (deps,), (tokens, heads, dv),
                  vector=tokens * value_width, write=tokens * value_width * 2,
                  family="tensor_fill", dtype="bf16")
    conv_input = tensors.contiguous(projection.narrow(0, 0, core_tokens).narrow(1, 0, width),
                                    "convolution_input_contiguous", reason="causal_conv1d_update.contiguous")
    if speculative and any(view.q_len == 1 for view in active):
        raise ModelingError("gdn_decode_mixed_spec_non_spec_caller_not_selected")
    # The caller selects one state index per physical request slot before
    # normalizing the view. Proposal-table stride is independent of query count
    # and padding; a multi-column table needs the real one-column copy.
    cache_indices = tensors.result(deps, (len(lengths), cache_index_row_stride), "int32",
                                    output="causal_cache_indices").narrow(1, 0, 1)
    cache_indices = tensors.contiguous(cache_indices, "state_indices_contiguous",
                                       reason="_normalize_causal_cache_indices.contiguous")
    state_write_bytes = sum(
        (2 + view.q_len if speculative else plan.conv_kernel - 1) * width * 2
        for view in active)
    conv_deps = tuple(dict.fromkeys((conv_input.producer, cache_indices.producer)))
    conv_views, cursor = [], 0
    for view in views:
        conv_views.append(conv_input.narrow(0, cursor, view.q_len))
        cursor += view.q_len
    conv = b.work("short_convolution", conv_deps, (active_rows, width),
                  family="short_convolution", dtype=model.activation_dtype,
                  metadata=(("sequence_tokens", lengths), ("kernel_width", plan.conv_kernel),
                            ("physical_sequence_tokens", lengths),
                            ("physical_query_rows", core_tokens),
                            ("active_query_rows", active_rows),
                            ("padding_slots", padding_slots),
                            ("speculative", speculative),
                            ("state_history_slots", plan.conv_kernel - 1 + (max_query - 1 if speculative else 0)),
                            ("state_dtype", conv_dtype), ("state_layout", "slots_history_channels"),
                            ("state_mutation", "inplace_history_writeback"),
                            ("state_write_bytes", state_write_bytes),
                            ("state_slots", tuple(slot for slot in range(len(lengths)) if slot not in padding_slots)),
                            ("cache_index_row_stride", cache_index_row_stride),
                            ("cache_index_dtype", "int32"),
                            ("cache_index_source", {**SOURCE, "caller_lines": CACHE_INDEX_SOURCE_LINES}),
                            ("state_copy_boundary", "cache_index_single_column_contiguous"),
                            ("state_contract", "conv_history_slots_history_channels"),
                            ("gdn_state_contract", conv_contract),
                            ("gdn_state_effects", conv_effects),
                            ("bias", False), ("source", CONV_SOURCE)))
    conv_contract, conv_outputs = gdn_recurrent.bind_state_ports(
        {**conv_contract,
         "candidate_storage": ("history_ring+query_rows" if speculative else ""),
         "candidate_recovery": ("accepted_prefix_from_conv_history_and_query_inputs"
                                if speculative else ""),
         "value_inputs": tuple(gdn_recurrent.tensor_port(
             value, role="conv_input", ready=value.producer,
             owner=value.producer) for value in conv_views)},
        conv, heads=heads, head_shards=1)
    b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted({
        **dict(b.ops[-1].metadata),
        "state_ready": conv,
        "state_value_inputs": conv_contract["value_inputs"],
        "state_value_outputs": conv_outputs,
        "state_candidate_stores": conv_contract["candidate_stores"],
        "state_candidate_storage": {
            "kind": conv_contract.get("candidate_storage", ""),
            "recovery": conv_contract.get("candidate_recovery", ""),
            "history_slots": plan.conv_kernel - 1 + (max_query - 1
                                                    if speculative else 0),
            "query_rows": tuple(view.q_len for view in active),
            "cache_index_view": cache_indices.to_dict(),
            "op": conv,
        },
        "gdn_state_contract": conv_contract,
        "gdn_state_effects": conv_effects,
    }.items())))
    b.ops[-1] = replace(b.ops[-1], recipe=CONV, source_boundary="kernel",
                        padded_shape=(core_tokens, width))
    values = rearrange_qkv(tensors, conv, core_tokens, heads, keys, dk, dv)
    if gating_beta_store is None:
        declared = runtime.kernels.get("families", {}).get("gdn_gating", {})
        outputs = declared.get("outputs", ()) if declared.get("kind") == "tiled_pointwise" else ()
        gating_beta_store = dict(outputs[1]) if len(outputs) == 2 and "queue" in outputs[1] else {}
    if gating_beta_store and gating_beta_store.get("bytes_per_element", 2) != 2:
        raise ModelingError("gdn_decode_scalar_beta_store_requires_bf16_payload")
    if set(gating_beta_store) - {"bytes_per_element", "resource", "queue", "from_hardware"}:
        raise ModelingError("gdn_decode_beta_store_unknown_fields")
    if gating_beta_store:
        hardware_queue = gating_beta_store.get("from_hardware", False)
        if ((not hardware_queue and "queue" not in gating_beta_store)
                or not gating_beta_store.get("resource")):
            raise ModelingError("gdn_decode_beta_store_requires_queue_and_issue_resource")
        if gating_beta_store["resource"] in {"hbm", "mte2", "mte2_aiv", "mte2_aic", "mte3", "vector", "cube", "fixpipe"}:
            raise ModelingError("gdn_decode_beta_store_requires_scalar_issue_units")
        if hardware_queue:
            if hardware_queue is not True or "queue" in gating_beta_store:
                raise ModelingError("gdn_decode_store_queue_requires_one_parameter_source")
        else:
            if not {"outstanding", "granule_bytes", "response_ms"} <= gating_beta_store["queue"].keys():
                raise ModelingError("gdn_decode_beta_store_queue_parameters_required")
            if set(gating_beta_store["queue"]) - {"outstanding", "granule_bytes", "response_ms", "row_control_ms"}:
                raise ModelingError("gdn_decode_beta_store_unknown_queue_fields")
    if gating_programs is not None and (not isinstance(gating_programs, int) or isinstance(gating_programs, bool) or gating_programs < 1):
        raise ModelingError("gdn_decode_gating_programs_requires_positive_integer")
    beta_store = {key: value for key, value in gating_beta_store.items() if key != "bytes_per_element"}
    gated = b.work("delta_gates", tuple(value.producer for value in gate_inputs), (core_tokens, heads),
                   family="gdn_gating", dtype="bf16",
                   metadata=(("source", GATING_SOURCE), ("beta_store", beta_store),
                             ("logical_programs", gating_programs),
                             ("g_store_selection", "retained_binary_positive_stride_dma_inference"),
                             ("compiled_identity", "new_capture_binary_unverified"),
                             ("outputs", ("fp32_decay", "bf16_beta"))))
    b.ops[-1] = replace(b.ops[-1], recipe=GATING, source_boundary="kernel")
    gate_values = (tensors.result(gated, (1, core_tokens, heads), "fp32", output="decay"),
                   tensors.result(gated, (1, core_tokens, heads), "bf16", output="beta"))
    decode_offsets, offset_cursor = [], 0
    for view in views:
        if not view.padding:
            decode_offsets.append(offset_cursor)
        offset_cursor += view.q_len
    core = decode_core(b, tensors, values, gate_values, active, heads, keys, dk, dv,
                       physical_tokens=core_tokens, physical_sequence_tokens=lengths,
                       physical_row_offsets=tuple(decode_offsets))
    scan = b.ops[-1]
    scan_meta = dict(scan.metadata)
    decode_value_inputs = tuple(
        view_values[2] for view_values in scan_meta.get("state_value_views", ()))
    checkpoint_versions = tuple(
        None if count <= 0
        else tuple(context + offset + 1 for offset in range(count))
        for context, count in zip(contract["contexts"], contract["checkpoints"]))
    contract, recurrent_outputs = gdn_recurrent.bind_state_ports(
        contract, scan.op_id, heads=heads, head_shards=1)
    state_metadata = tuple({**scan_meta,
                            "padding_slots": padding_slots,
                            "physical_query_rows": core_tokens,
                            "state_ready": scan.op_id,
                            "state_value_inputs": decode_value_inputs,
                            "state_value_outputs": recurrent_outputs,
                            "state_candidate_stores": contract["candidate_stores"],
                            "state_candidate_storage": {
                                "kind": "token_checkpoint_store",
                                "recovery": "accepted_prefix_from_query_candidates",
                                "history_slots": None,
                                "query_rows": tuple(view.q_len for view in active),
                                "op": scan.op_id,
                            },
                            "gdn_state_contract": contract,
                            "gdn_state_effects": recurrent_effects,
                            "state_checkpoint_versions": checkpoint_versions}.items())
    b.ops[-1] = replace(scan, metadata=state_metadata)
    scan = b.ops[-1]
    use_recover_ssm = recover_ssm_enabled(runtime) and speculative
    if recurrent_implementation == "arch35_fp32":
        recipe_name = GDN_ARCH35_RECOVER_SSM_RECIPE if use_recover_ssm else gdn_recurrent.IMPLEMENTATION
        extra_meta = {"recover_ssm": True, "recover_ssm_stage": "verify_compact_record"} if use_recover_ssm else {}
        b.ops[-1] = replace(scan, recipe=recipe_name, source_boundary="kernel",
                           metadata=tuple({**dict(scan.metadata), "source": gdn_recurrent.SOURCE,
                               "implementation_scope": "source_resident_qkv_and_state_pipeline",
                               "recurrence_selection": recurrent_implementation,
                               "unmodeled_features": ("scalar_metadata_issue_and_cache_latency",),
                               **extra_meta,
                           }.items()))
    elif recurrent_implementation == "analytical":
        updated_scan = replace(scan, metadata=tuple({**dict(scan.metadata),
                               "recurrence_selection": recurrent_implementation}.items()))
        if use_recover_ssm:
            updated_scan = rewrite_gdn_analytical_scan(
                updated_scan, active_views=active, heads=heads, dk=dk, dv=dv,
                state_dtype=plan.recurrent_state_dtype, kg_dtype=model.activation_dtype)
        b.ops[-1] = updated_scan
    else:
        raise ModelingError("gdn_decode_recurrent_implementation_unknown:" + recurrent_implementation)
    if use_recover_ssm:
        emit_commit_op(b, name="recover_ssm_commit", deps=(scan.op_id,), views=active,
                       heads=heads, dk=dk, dv=dv, state_dtype=plan.recurrent_state_dtype,
                       kg_dtype=model.activation_dtype, runtime=runtime, attention_kind="gdn")
    copied = tensors.copy_into(core, tensors.result(zero, (tokens, heads, dv)).narrow(0, 0, core_tokens),
                               "core_output_copy")
    # gdn.py flattens token and head axes before norm. This crosses the
    # QKVZ row gap for multi-token/multi-head z and needs a materialization;
    # singleton/view-compatible layouts remain free through TensorLowering.
    gate = projection.narrow(1, width, value_width)
    gate = tensors.reshape(gate, (tokens * heads, dv), name="z_norm_materialize")
    normalized = b.vector(
        "linear_output_norm_gate", (copied.producer, gate.producer), tokens * value_width,
        OUTPUT_NORM_GATE_OPS_PER_ELEMENT,
        shape=(tokens * heads, dv), read_bytes_override=(2 * tokens * value_width + dv) * 2,
        write_bytes_override=tokens * value_width * 2,
        metadata=(("norm_kind", "rmsnorm"), ("norm_has_weight", True), ("norm_has_bias", False),
                  ("norm_group_width", dv), ("gate_present", True), ("gate_layout", gate.layout.to_dict()),
                  ("recurrent_attention_kind", "gdn"), ("fused_from", ("rmsnorm", "gate"))))
    last = b.matrix("linear_output", (normalized,), tokens, model.hidden_size, value_width, category="gdn.o")
    if tp > 1:
        last = b.communicate("linear_reduce", (last,), 2 * (tp - 1) / tp * tokens * model.hidden_size * 2, "allreduce")
    classes = {"tensor_copy", "tensor_fill", "short_convolution", "gdn_gating", "recurrent_vector", "elementwise"}
    b.ops = [replace(op, metadata=tuple({**dict(op.metadata), "invocation_class": "vector",
                                        "caller_implementation": "vllm_ascend_gdn_decode"}.items()))
             if op.family in classes else op for op in b.ops]
    return AttentionResult(ops=b.ops, last=last)
