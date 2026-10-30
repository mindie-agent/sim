"""Non-interleaved QKVZ/BA caller with explicit GDN decode kernel boundaries.

[Role] Bind update convolution, gating, Q/K normalization and state recurrence.
[Invariants] Caller copies replace generic work. Query rows do not shrink with
  speculative acceptance; graph padding stops before the active native core.
  Each named copy/fill is a declared vector invocation, not an extra host gap.
[Timeouts & Locks] O(sequence descriptors), lazily executed physical kernels.
[Side-Effect Policy] Builds operators only; no measurements or device contexts.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import ModelingError
from .call_gdn import GATING_SOURCE
from . import gdn_recurrent
from .gdn_decode_programs import CONV, GATING, CONV_SOURCE, conv_plan, gating_plan
from .gdn_prefill import decode_core, rearrange_qkv
from .lowering import Builder
from .mechanism import PlannedRecipe
from .tensors import TensorLowering

SOURCE = {"path": "data/observations/sources/gdn-image-c3d00a196905/gdn.py.txt",
          "sha256": "e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98",
          "caller_lines": ((298, 379), (505, 575), (705, 737))}


def recipes():
    return {CONV: PlannedRecipe(conv_plan), GATING: PlannedRecipe(gating_plan),
            gdn_recurrent.IMPLEMENTATION: PlannedRecipe(gdn_recurrent.build)}


def lower(*, prefix, plan, views, stage, deps, model, parallel, mode,
          padded_tokens, policy, runtime, gating_beta_store=None, gating_programs=None,
          recurrent_implementation="analytical", cache_index_row_stride=1):
    if not views or any(view.phase != "decode" for view in views):
        raise ModelingError("gdn_decode_selected_caller_requires_decode_only")
    if parallel.pcp != 1 or model.activation_dtype != "bf16" or plan.recurrent_state_dtype != "fp32":
        raise ModelingError("gdn_decode_requires_pcp1_bf16_inputs_fp32_state")
    if (not isinstance(cache_index_row_stride, int) or isinstance(cache_index_row_stride, bool)
            or cache_index_row_stride < 1):
        raise ModelingError("gdn_decode_cache_index_row_stride_requires_positive_integer")
    b = Builder(prefix, plan, stage, views, model, mode)
    tensors = TensorLowering(b, runtime.tensor_layouts)
    tp, tokens, actual = parallel.tp, padded_tokens, sum(view.q_len for view in views)
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
    conv_input = tensors.contiguous(projection.narrow(0, 0, actual).narrow(1, 0, width),
                                    "convolution_input_contiguous", reason="causal_conv1d_update.contiguous")
    lengths = tuple(view.q_len for view in views)
    speculative = any(length > 1 for length in lengths)
    if speculative and any(length == 1 for length in lengths):
        raise ModelingError("gdn_decode_mixed_spec_non_spec_caller_not_selected")
    # The caller selects one state index per request before normalizing the
    # view. Proposal-table stride is independent of query count and padding.
    cache_indices = tensors.result(deps, (len(views), cache_index_row_stride), "int32",
                                    output="causal_cache_indices").narrow(1, 0, 1)
    cache_indices = tensors.contiguous(cache_indices, "state_indices_contiguous",
                                       reason="_normalize_causal_cache_indices.contiguous")
    conv = b.work("short_convolution", tuple(dict.fromkeys((conv_input.producer, cache_indices.producer))), (actual, width),
                  family="short_convolution", dtype="bf16",
                  metadata=(("sequence_tokens", lengths), ("kernel_width", plan.conv_kernel),
                            ("speculative", speculative), ("state_history_slots", plan.conv_kernel - 1 + max(lengths) - 1),
                            ("cache_index_row_stride", cache_index_row_stride),
                            ("cache_index_dtype", "int32"),
                            ("cache_index_source", {**SOURCE, "caller_lines": ((138, 143), (435, 445))}),
                            ("bias", False), ("source", CONV_SOURCE)))
    b.ops[-1] = replace(b.ops[-1], recipe=CONV, source_boundary="kernel")
    values = rearrange_qkv(tensors, conv, actual, heads, keys, dk, dv)
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
    gated = b.work("delta_gates", tuple(value.producer for value in gate_inputs), (actual, heads),
                   family="gdn_gating", dtype="bf16",
                   metadata=(("source", GATING_SOURCE), ("beta_store", beta_store),
                             ("logical_programs", gating_programs),
                             ("g_store_selection", "retained_binary_positive_stride_dma_inference"),
                             ("compiled_identity", "new_capture_binary_unverified"),
                             ("outputs", ("fp32_decay", "bf16_beta"))))
    b.ops[-1] = replace(b.ops[-1], recipe=GATING, source_boundary="kernel")
    gate_values = (tensors.result(gated, (1, actual, heads), "fp32", output="decay"),
                   tensors.result(gated, (1, actual, heads), "bf16", output="beta"))
    core = decode_core(b, tensors, values, gate_values, views, heads, keys, dk, dv)
    scan = b.ops[-1]
    if recurrent_implementation == "arch35_fp32":
        b.ops[-1] = replace(scan, recipe=gdn_recurrent.IMPLEMENTATION, source_boundary="kernel",
                           metadata=tuple({**dict(scan.metadata), "source": gdn_recurrent.SOURCE,
                               "implementation_scope": "source_resident_qkv_and_state_pipeline",
                               "recurrence_selection": recurrent_implementation,
                               "unmodeled_features": ("scalar_metadata_issue_and_cache_latency",),
                           }.items()))
    elif recurrent_implementation == "analytical":
        b.ops[-1] = replace(scan, metadata=tuple({**dict(scan.metadata),
                           "recurrence_selection": recurrent_implementation}.items()))
    else:
        raise ModelingError("gdn_decode_recurrent_implementation_unknown:" + recurrent_implementation)
    copied = tensors.copy_into(core, tensors.result(zero, (tokens, heads, dv)).narrow(0, 0, actual),
                               "core_output_copy")
    # gdn.py flattens token and head axes before norm. This crosses the
    # QKVZ row gap for multi-token/multi-head z and needs a materialization;
    # singleton/view-compatible layouts remain free through TensorLowering.
    gate = projection.narrow(1, width, value_width)
    gate = tensors.reshape(gate, (tokens * heads, dv), name="z_norm_materialize")
    normalized = b.vector(
        "linear_output_norm_gate", (copied.producer, gate.producer), tokens * value_width, 9,
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
