"""Explicit stateful causal-convolution update on the shared native program.

[Role] Connect typed KDA producers to one native non-speculative decode call.
[Invariants] Active queries exclude graph/SP padding. State history is read
  and written once; prepared weights coexist with original checkpoint weights.
  Input aliases stay views, while separate producers require a real row pack.
[Timeouts & Locks] O(sequence/projection descriptors); no locks or external I/O.
[Side-Effect Policy] Creates model metadata, weight terms and graph operators.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .gdn_decode_programs import CONV, CONV_SOURCE, conv_plan
from .mechanism import PlannedRecipe

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
    return {CONV: PlannedRecipe(conv_plan)}


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
        if layer.conv_kernel not in {2, 3, 4}:
            raise ModelingError("causal_conv_update_requires_width2to4")
        prepared[str(layer.index)] = {
            "dtype": model.activation_dtype,
            "prepared_weight_policy": "retain_original_fp32_checkpoint_weight",
            "state_layout": "slots_history_channels",
        }
    metadata = {**dict(model.metadata), "causal_conv_update": {
        "implementation": IMPLEMENTATION, "layers": prepared, "source": SOURCE}}
    return replace(model, metadata=tuple(sorted(metadata.items())))


def lower(b, tensors, *, plan, views, tokens, sources, key_width, value_width):
    """Bind q1 update; the selected recipe owns the complete state writeback."""
    # The scheduler records one committed token for ordinary q1 decode;
    # proposal/accepted rows or a larger advance require a speculative ABI.
    if not views or any(view.phase != "decode" or view.q_len != 1 or any(
            getattr(view, field) for field in ("draft_proposal_tokens",
                                             "draft_accepted_tokens"))
                        or view.draft_advanced_tokens not in (0, 1)
                        for view in views):
        raise ModelingError("causal_conv_update_requires_q1_non_speculative_decode")
    actual, width = len(views), 2 * key_width + value_width
    dtype = b.model.activation_dtype
    byte = dtype_bytes(dtype)
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
        value = value.narrow(0, 0, actual).narrow(1, 0, width)
        incoming = tensors.contiguous(value, "convolution_input_contiguous",
                                      reason="CausalConv1d.x.AutoContiguous").producer
        adaptation = "packed_projection_qkv_slice; AutoContiguous_if_strided"
    else:
        if len(sources) != 3 or len(producers) != 3:
            raise ModelingError("causal_conv_update_requires_three_qkv_producers_or_one_packed")
        incoming = b.work(
            "convolution_input_pack", producers, (actual, width),
            read=actual * width * byte, write=actual * width * byte,
            family="tensor_copy", dtype=dtype,
            metadata=(("materialization_reason", "rowwise_qkv_concatenation"),
                      ("source_row_widths", (key_width, key_width, value_width)),
                      ("caller_boundary", "declared_separate_producer_adapter_not_native_projection_claim"),
                      ("invocation_class", "vector")))
        adaptation = "explicit_row_pack_of_separate_semantic_producers"
    metadata = (
        ("sequence_tokens", (1,) * actual), ("kernel_width", plan.conv_kernel),
        ("speculative", False), ("state_history_slots", plan.conv_kernel - 1),
        ("state_dtype", plan.conv_state_dtype), ("state_layout", "slots_history_channels"),
        ("state_mutation", "inplace_history_writeback"),
        ("state_write_bytes", actual * (plan.conv_kernel - 1) * width * byte),
        ("cache_index_dtype", "int32"), ("cache_indices", "one_valid_slot_per_active_sequence"),
        ("query_start_loc_dtype", "int32"), ("bias", False),
        ("activation", "silu"), ("run_mode", 1), ("source", SOURCE),
        ("input_adaptation", adaptation), ("prepared_weight_dtype", dtype),
        ("prepared_weight_layout", "kernel_width_channels"),
        ("native_kind", "CausalConv1d"), ("invocation_class", "vector"),
        ("implementation", IMPLEMENTATION),
        ("limitations", ("scalar_metadata_issue_and_cache_latency_unmodeled",
                          "source_update_kernel_revision_not_capture_binary_identity",
                          "same_shape_owner_runs_use_existing_fluid_compaction")),
    )
    op_id = b.work("short_convolution", (incoming,), (actual, width),
                   family="short_convolution", dtype=dtype, metadata=metadata)
    b.ops[-1] = replace(b.ops[-1], recipe=CONV, source_boundary="kernel")
    return op_id
