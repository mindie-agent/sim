"""Non-interleaved QKVZ/BA GDN caller using the native arch35 composite.

[Role] Lower contiguous BSND inputs, native Prepare/H/O and requested exports.
[Invariants] Normalization is inside Prepare. Gate scalars become FP32 before
  BSN->BNS conversion. State is V,K and remains independent per sequence.
  Non-spec decode is a prefix of the actual batch; padded rows stop outside
  the core. Physical kernels are shared with isolated operator analysis.
[Timeouts & Locks] O(sequence count) metadata, lazy kernel work; no locks.
[Side-Effect Policy] Appends operators only; no measurements or device access.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import dtype_bytes
from . import gdn_composite
from .lowering import Builder
from .mechanism import PlannedRecipe
from .operators import native_causal_conv_prefill
from .tensors import TensorLowering


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


def core(*, builder, tensors, conv, gated, views, heads, key_heads, dk, dv,
          chunk, gate_values=None, qkv_values=None, **_):
    b = builder
    tokens = sum(view.q_len for view in views)
    dtype = b.model.activation_dtype
    lengths = tuple(view.q_len for view in views)
    values = (qkv_values if qkv_values is not None else
              rearrange_qkv(tensors, conv, tokens, heads, key_heads, dk, dv))
    # gdn.py always gathers an initial tensor, including zero states at context0.
    state_elements = len(views) * heads * dk * dv
    prior_elements = sum(view.context > 0 for view in views) * heads * dk * dv
    state_blocks = math.ceil(heads * dk * dv / 1024)
    gather_metadata = state_blocks * (len(views) + 4 * sum(view.context > 0 for view in views))
    gather = b.work(
        "gather_recurrent_state", (conv,), (len(views), heads, dv, dk),
        read=prior_elements * 4 + gather_metadata, write=state_elements * 4,
        family="state_gather", dtype="fp32",
        metadata=(("state_layout", "V,K"), ("has_initial_state", tuple(v.context > 0 for v in views)),
                  ("index_dtype", "int32"), ("metadata_cache_reuse", False)),
    )
    gate, beta = gate_values or (
        tensors.result(gated, (1, tokens, heads), "fp32", output="decay"),
        tensors.result(gated, (1, tokens, heads), dtype, output="beta"))
    composite = gdn_composite.prefill(
        builder=b, tensors=tensors, q=values[0], k=values[1], v=values[2],
        gate=gate, beta=beta, lengths=lengths, heads=heads, key_heads=key_heads,
        dk=dk, dv=dv, chunk=chunk, initial_state_dependency=gather,
        native_requests=tuple(view.request_id for view in views))
    b.work("scatter_recurrent_state", (composite.state,), (len(views), heads, dv, dk),
           read=state_elements * 4 + len(views) * state_blocks * 4, write=state_elements * 4,
           family="state_scatter", dtype="fp32", metadata=(("state_layout", "V,K"),
                  ("index_dtype", "int32"), ("metadata_cache_reuse", False)))
    return composite.output


def decode_core(b, tensors, values, gate_values, views, heads, keys, dk, dv):
    """Explicit analytical decode until the native recurrence is accepted."""
    tokens = sum(view.q_len for view in views)
    normalized = [b.vector(
        "decode_" + name + "_l2norm", (value.producer,), tokens * keys * dk, 5,
        shape=(1, tokens, keys, dk),
        metadata=(("source_layout", value.layout.to_dict()),
                  ("implementation_scope", "analytical_decode_l2norm")))
        for name, value in zip(("q", "k"), values[:2])]
    state_bytes = heads * dk * dv * dtype_bytes(b.plan.recurrent_state_dtype)
    output = b.work(
        "decode_recurrent_scan", (*normalized, values[2].producer,
                                  *[value.producer for value in gate_values]),
        (tokens, heads, dk, dv), family="recurrent_vector",
        vector=7 * tokens * heads * dk * dv,
        read=len(views) * state_bytes + tokens * heads * ((2 * dk + dv) * 2 + 6),
        write=tokens * (state_bytes + heads * dv * 2),
        metadata=(("tokens", tokens), ("heads", heads), ("key_heads", keys),
                  ("dk", dk), ("dv", dv), ("sequence_tokens", tuple(v.q_len for v in views)),
                  ("state_dtype", b.plan.recurrent_state_dtype),
                  ("state_element_bytes", dtype_bytes(b.plan.recurrent_state_dtype)),
                  ("state_checkpoints", tokens),
                  ("source_layouts", tuple(v.layout.to_dict() for v in values)),
                  ("implementation_scope", "analytical_decode_recurrent_vector")))
    return tensors.result(output, (tokens, heads, dv))


def lower(*, prefix, plan, views, stage, deps, model, parallel, mode,
          padded_tokens, policy, runtime):
    if parallel.pcp != 1:
        raise ValueError("gdn_prefill_arch35_selected_caller_branch_requires_pcp1")
    if plan.recurrent_state_dtype != "fp32":
        raise ValueError("gdn_prefill_arch35_state_precision_branch_not_implemented:"
                         + plan.recurrent_state_dtype)
    prefill = tuple(view for view in views if view.phase == "prefill")
    decode = tuple(view for view in views if view.phase == "decode")
    if any(view.q_len != 1 or view.draft_proposal_tokens for view in decode):
        raise ValueError("gdn_prefill_arch35_speculative_mixed_caller_not_implemented")
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
    conv_input = tensors.contiguous(projection.narrow(0, 0, actual).narrow(1, 0, width),
                                    "convolution_input_contiguous",
                                    reason="causal_conv1d_fn.contiguous")
    conv_op = native_causal_conv_prefill.make_op({
        "lengths": tuple(view.q_len for view in ordered), "dim": width,
        "width": plan.conv_kernel, "dtype": model.activation_dtype,
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": tuple(view.context > 0 for view in ordered),
        "cache_indices": tuple(range(1, len(ordered) + 1)),
        "state_rows": len(ordered) + 1,
        "state_history_slots": plan.conv_kernel - 1 + int(
            dict(runtime.metadata).get("draft_proposal_tokens", 0)),
        "null_block_id": 0,
    })
    conv = b.add(replace(conv_op, op_id=prefix + ".short_convolution",
                         deps=(conv_input.producer,), **b.common))
    gated = b.vector(
        "delta_gates", tuple(value.producer for value in gate_inputs), actual * heads, 12,
        family="gdn_gating", shape=(actual, heads),
        read_bytes_override=actual * heads * 4 + heads * 8, write_bytes_override=actual * heads * 6,
        metadata=(("outputs", ("fp32_decay", "activation_beta")),
                  ("input_layouts", tuple(v.narrow(0, 0, actual).layout.to_dict()
                                         for v in gate_inputs))))
    qkv = rearrange_qkv(tensors, conv, actual, heads, keys, dk, dv)
    gate_values = (tensors.result(gated, (1, actual, heads), "fp32", output="decay"),
                   tensors.result(gated, (1, actual, heads), output="beta"))
    outputs = []
    if decode:
        outputs.append(decode_core(
            b, tensors, tuple(v.narrow(1, 0, decode_tokens) for v in qkv),
            tuple(v.narrow(1, 0, decode_tokens) for v in gate_values),
            decode, heads, keys, dk, dv))
    output = core(builder=b, tensors=tensors, conv=conv, gated=gated, views=prefill,
                  heads=heads, key_heads=keys, dk=dk, dv=dv,
                  chunk=policy.recurrent_chunk if policy else 64,
                  qkv_values=tuple(v.narrow(1, decode_tokens, prefill_tokens) for v in qkv),
                  gate_values=tuple(v.narrow(1, decode_tokens, prefill_tokens) for v in gate_values))
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
        "linear_output_norm_gate", (copied.producer, gate.producer), tokens * value_width, 9,
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
