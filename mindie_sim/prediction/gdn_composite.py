"""Reusable native arch35 GDN prefill composite.

[Role] Build caller materializations, Prepare/H/O and requested exports for an
  already prepared varlen prefill batch.
[Invariants] Q/K/V are BF16 BSND; gates are FP32 BSN; beta is BF16 BSN and is
  materialized to FP32 before its BNS view. State is FP32 V,K and is read from
  the caller's initial tensor and written once to the caller's final tensor.
  Prepare/H/O are the shared source planners. QKV packing and cache
  gather/scatter are caller concerns and never appear here.
[Timeouts & Locks] O(sequence count) metadata; lazy kernel work; no locks.
[Side-Effect Policy] Appends operators to the caller's Builder; no device I/O.
"""
from dataclasses import dataclass, replace

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .mechanism import PlannedRecipe
from . import gdn_chunk32, gdn_recurrent
from .gdn_domain import (
    GDN_CHUNK32_HEAD_DIM_V,
    GDN_CHUNK32_TOKENS,
    GDN_CHUNK64_TOKENS,
    GDN_HEAD_DIM_K,
)
from .operators import gdn_output, gdn_prepare, gdn_state


def recipes():
    """Recipes for native Chunk64 and Chunk32 GDN prefill kernels."""
    params = {
        "lengths": (1,), "heads": 1, "key_heads": 1,
        "dk": GDN_HEAD_DIM_K, "dv": GDN_CHUNK32_HEAD_DIM_V,
        "chunk": GDN_CHUNK64_TOKENS,
    }
    base = {module.make_op(params).recipe: PlannedRecipe(module.build)
            for module in (gdn_prepare, gdn_state, gdn_output)}
    base.update(gdn_chunk32.recipes())
    return base


@dataclass(frozen=True)
class PrefillComposite:
    """Declared native prefill call plus the caller-side materializations."""

    prepare: str
    state: str
    output: str
    ops: tuple[OpSpec, ...]
    states: tuple[str, ...] = ()
    prepares: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    segment_ranges: tuple = ()
    segment_lengths: tuple = ()
    segmented: bool = False


def _external_copy(builder, value, name, *, reason, dtype=None, source,
                   source_stream="current"):
    """Materialize a tensor whose payload was prepared outside the timed call."""
    dtype = dtype or value.dtype
    layout = value.layout
    if layout.is_contiguous and dtype == value.dtype:
        return replace(value, producer="")
    read = layout.elements * dtype_bytes(value.dtype)
    write = layout.elements * dtype_bytes(dtype)
    producer = builder.work(
        name, (), layout.shape, read=read, write=write,
        vector=layout.elements if dtype != value.dtype else 0,
        family="tensor_copy", dtype=dtype,
        metadata=(("source_layout", layout.to_dict()),
                  ("target_layout", TensorLayout.contiguous(layout.shape).to_dict()),
                  ("input_format", value.dtype), ("output_format", dtype),
                  ("source_output", value.output),
                  ("source_input", source), ("source_stream", source_stream),
                  ("materialization_reason", reason)),
    )
    return TensorValue(producer, dtype, TensorLayout.contiguous(layout.shape))


def _materialize(builder, tensors, value, name, *, reason, dtype=None, source=None):
    if source is None:
        return tensors.contiguous(value, name, reason=reason, dtype=dtype)
    return _external_copy(builder, value, name, reason=reason,
                          dtype=dtype, source=source)


def _concat_axis(builder, values, shape, axis, name):
    """Materialize a token-axis concatenation whose target strides differ."""
    dtype = values[0].dtype
    elements = sum(value.layout.elements for value in values)
    total = 1
    for size in shape:
        total *= int(size)
    producer = builder.work(
        name, tuple(dict.fromkeys(value.producer for value in values
                                  if value.producer)), shape,
        read=elements * dtype_bytes(dtype), write=total * dtype_bytes(dtype),
        family="tensor_copy", dtype=dtype,
        metadata=(("materialization_reason", "segmented_axis_concatenate"),
                  ("concatenate_axis", int(axis)),
                  ("source_layouts", tuple(value.layout.to_dict() for value in values)),
                  ("target_layout", TensorLayout.contiguous(shape).to_dict())))
    return TensorValue(producer, dtype, TensorLayout.contiguous(shape))


def _concat_token_blocks(tensors, values, shape, name):
    """Concatenate segment token blocks whose flattened order is the target."""
    flat = tensors.concatenate_flat(
        [tensors.reshape(value, (value.layout.elements,), name=name + f"_part{index}")
         for index, value in enumerate(values)], name)
    return tensors.reshape(flat, shape, name=name + "_view")


def prefill(*, builder, tensors, q, k, v, gate, beta, lengths, heads,
            key_heads, dk, dv, chunk, initial_state_dependency=None,
            external=None, native_requests=None, state_contract=None,
            state_effects=(), state_segments=None):
    """Append one native prefill call to ``builder``.

    ``external`` maps logical input names to caller-owned source labels. Those
    inputs have no producer operator in this call, so their materializations
    carry an explicit source/stream edge in metadata instead of a fabricated
    dependency. ``initial_state_dependency`` is the serving gather op when the
    caller owns cache movement; it is absent for a standalone initial tensor.

    When ``state_segments`` carries more than one physical segment (PCP query
    ranges), every Prepare/WY, H and O phase uses the same segment lengths and
    the same local token offset.  The previous segment's real H output (or the
    remote transfer event) is the next H input; O_i waits for H_i.  No phase
    falls back to the merged local token count.
    """
    b = builder
    external = dict(external or {})
    lengths = tuple(lengths)
    tokens = sum(lengths)
    dtype = b.model.activation_dtype
    is_chunk32 = int(chunk) == GDN_CHUNK32_TOKENS
    if (dtype != "bf16"
            or (dk, dv) != (GDN_HEAD_DIM_K, GDN_CHUNK32_HEAD_DIM_V)
            or int(chunk) not in {GDN_CHUNK32_TOKENS, GDN_CHUNK64_TOKENS}):
        raise ValueError("gdn_prefill_arch35_requires_bf16_chunk32_or_64_k128_v128")
    prepare_mod = gdn_chunk32.PREPARE if is_chunk32 else gdn_prepare
    state_mod = gdn_chunk32.STATE if is_chunk32 else gdn_state
    output_mod = gdn_chunk32.OUTPUT if is_chunk32 else gdn_output
    params = {"lengths": lengths, "heads": heads, "key_heads": key_heads,
              "dk": dk, "dv": dv, "chunk": int(chunk), "dtype": dtype,
              "gate_dtype": "fp32", "use_exp2": not is_chunk32}

    # Keep the caller's actual token-range views before any compacting
    # materialization.  The typed offset is the evidence for [32,33) / [33,97)
    # (or another caller range); the materialized value is a separate op.
    source_inputs = (q, k, v, gate, beta)
    if not is_chunk32:
        # Chunk64 wrapper promotes BF16 beta to FP32 before BSN->BNS conversion;
        # Chunk32 folds beta handling directly inside Prepare.
        beta = _materialize(
            b, tensors, beta, "beta_fp32", reason="fla_common_gate_precision",
            dtype="fp32", source=external.get("beta"))
    gate = _materialize(
        b, tensors, gate.permute((0, 2, 1)), "g_head_first",
        reason="fla_BSN_to_BNS", source=external.get("g"))
    beta = _materialize(
        b, tensors, beta.permute((0, 2, 1)), "beta_head_first",
        reason="fla_BSN_to_BNS",
        source=external.get("beta") if is_chunk32 else None)
    inputs = [
        _materialize(b, tensors, value.permute((0, 2, 1, 3)),
                     name + "_head_first", reason="fla_BSND_to_BNSD",
                     source=external.get(name))
        for name, value in zip(("q", "k", "v"), (q, k, v))]

    def bind(module, name, dependencies, flags, lengths=None):
        op = module.make_op({**params, **({"lengths": tuple(lengths)}
                                         if lengths is not None else {}), **flags})
        common = {**b.common, "phase": "prefill"}
        if native_requests is not None:
            common["requests"] = tuple(native_requests)
        return b.add(replace(op, op_id=b.prefix + "." + name,
                             deps=tuple(dep for dep in dependencies if dep), **common))

    if state_segments is None:
        state_segments = ({"length": tokens, "lengths": tuple(lengths),
                           "contract": state_contract,
                           "effects": tuple(state_effects),
                           "initial_dependency": initial_state_dependency},)
    segments = tuple(state_segments)
    segment_lengths = tuple(tuple(segment.get("lengths",
                                              (segment.get("length", tokens),)))
                            for segment in segments)
    all_ranges = tuple(segment.get("range") for segment in segments)
    segmented = len(segments) > 1
    tokens_per_segment = tuple(sum(value) for value in segment_lengths)
    if sum(tokens_per_segment) != tokens:
        raise ValueError("gdn_prefill_segments_do_not_cover_local_tokens")
    offsets, cursor = [], 0
    for count in tokens_per_segment:
        offsets.append(cursor)
        cursor += count

    def slice_segment(value, index):
        # Inputs have already been converted to head-first BNSD/BNS, so the
        # physical token axis is 2; the local offset is the actual byte offset
        # inside the caller's token range.
        return value.narrow(2, offsets[index], tokens_per_segment[index])

    prepares, states, outputs = [], [], []
    for index, segment in enumerate(segments):
        lengths_i = segment_lengths[index]
        count_i = tokens_per_segment[index]
        if segmented:
            q_i, k_i, v_i = (slice_segment(value, index) for value in inputs)
            gate_i = slice_segment(gate, index)
            beta_i = slice_segment(beta, index)
            input_views = (q_i, k_i, v_i, gate_i, beta_i)
            source_views = tuple(
                value.narrow(1 if len(value.layout.shape) == 4 else 1,
                             offsets[index], tokens_per_segment[index])
                for value in source_inputs)
        else:
            q_i, k_i, v_i = inputs
            gate_i, beta_i = gate, beta
            input_views = (q_i, k_i, v_i, gate_i, beta_i)
            source_views = source_inputs
        prepare_name = "gdn_prepare" if not segmented else f"gdn_prepare{index}"
        prepare = bind(
            prepare_mod, prepare_name,
            (*[value.producer for value in (q_i, k_i, v_i)],
             gate_i.producer, beta_i.producer),
            {"use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": False,
             "use_beta_sigmoid_in_kernel": is_chunk32, "output_a": True,
             "output_beta": False,
             **({"beta_dtype": beta_i.dtype} if is_chunk32 else {})},
            lengths=lengths_i)
        prepare_meta = {
            "gdn_prepare_inputs": tuple(
                gdn_recurrent.tensor_port(value, role=name, ready=value.producer)
                for name, value in zip(("q", "k", "v", "gate", "beta"), input_views)),
            "gdn_prepare_source_inputs": tuple(
                gdn_recurrent.tensor_port(value, role=name,
                                          ready=value.producer or "caller")
                for name, value in zip(("q", "k", "v", "gate", "beta"),
                                       source_views)),
            "gdn_prepare_source_offsets": tuple(
                int(value.layout.offset) for value in source_views),
            "gdn_prepare_input_offsets": tuple(int(value.layout.offset)
                                               for value in input_views),
            "state_segment_index": index if segmented else None,
            "state_segment_range": segment.get("range"),
            "state_segment_length": count_i,
            "segment_ranges": all_ranges,
            "segment_lengths": segment_lengths,
        }
        if segmented:
            prepare_meta["prepare_partition"] = "pcp_segments"
        b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted({
            **dict(b.ops[-1].metadata), **prepare_meta}.items())))
        prepares.append(prepare)

        dependency = segment.get("initial_dependency", initial_state_dependency)
        if index > 0:
            dependency = None
            if segment.get("exchange_bytes"):
                exchange = b.communicate(
                    f"pcp_recurrent_state_step{index}", (states[-1],),
                    segment["exchange_bytes"], "state_transfer", "pcp",
                    from_version=segment.get("from_version"),
                    to_version=segment.get("to_version"),
                    state_scope="recurrent_local_bank",
                    remote_producer="external_rank",
                    source_implementation="pcp1_only",
                    status="pending_external_rank")
                dependency = exchange
            else:
                dependency = states[-1]
        contract = segment.get("contract")
        segment_effects = tuple(segment.get("effects", ()))
        state_name = "gdn_state" if not segmented else f"gdn_state{index}"
        deps = ((prepare, dependency) if dependency else (prepare,))
        state = bind(state_mod, state_name, deps,
                     {"use_initial_state": True, "output_final_state": True,
                      "state_v_first": True},
                     lengths=lengths_i)
        if contract is not None:
            explicit_inputs = tuple(segment.get("value_inputs") or ())
            if explicit_inputs:
                input_port = explicit_inputs[0]
            elif dependency:
                input_port = gdn_recurrent.tensor_port(TensorValue(
                    str(dependency), contract.get("state_dtype") or "fp32",
                    TensorLayout.contiguous((len(lengths_i), heads, dv, dk)),
                    state_version=segment.get("to_version")),
                    ready=str(dependency), role="state_input", owner=str(dependency))
            else:
                input_port = gdn_recurrent.tensor_port(TensorValue(
                    "", contract.get("state_dtype") or "fp32",
                    TensorLayout.contiguous((len(lengths_i), heads, dv, dk)),
                    binding="external", binding_name="caller:initial_state",
                    state_version=(contract.get("read_versions") or (None,))[0]),
                    ready="caller:initial_state", role="state_input",
                    owner="caller")
            contract, state_outputs = gdn_recurrent.bind_state_ports(
                contract, state, heads=heads, head_shards=1)
            state_meta = dict(b.ops[-1].metadata)
            state_meta.update({
                "effect_role": ("h_return"
                                if contract["direction"] == "h"
                                else "call_return"),
                "state_input_producer": dependency or "",
                "state_ready": state,
                "state_value_input": input_port,
                "state_value_inputs": (input_port, *explicit_inputs[1:]),
                "state_value_outputs": state_outputs,
                "state_value_output": ("chunk_bank"
                                       if contract["direction"] == "h"
                                       else "final_state"),
                "gdn_state_contract": contract,
                "gdn_state_effects": segment_effects,
                "state_dtype": contract["state_dtype"],
                "state_layout": contract["state_layout"],
                "state_read_versions": contract["read_versions"],
                "state_commit_versions": contract["commit_versions"],
                "state_local_bank_versions": contract["local_bank_versions"],
                "state_local_chunk_versions": contract["local_chunk_versions"],
                "state_publish_point": contract["publish_point"],
                "state_no_io_slots": contract["no_state_io_slots"],
                "state_internal_commit_versions": contract["internal_commit_versions"],
                "state_segment_index": index if segmented else None,
                "state_segment_length": count_i,
                "state_segment_range": segment.get("range"),
                "state_segment_from_version": segment.get("from_version"),
                "state_segment_to_version": segment.get("to_version"),
                "state_segment_lengths": lengths_i,
            })
            b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted(state_meta.items())))
        states.append(state)

    for index, segment in enumerate(segments):
        lengths_i = segment_lengths[index]
        name = "gdn_output" if not segmented else f"gdn_output{index}"
        output_op = bind(output_mod, name, (prepares[index], states[index]),
                         {"state_v_first": True}, lengths=lengths_i)
        if segmented:
            b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted({
                **dict(b.ops[-1].metadata),
                "output_partition": "pcp_segments",
                "state_segment_index": index,
                "state_segment_range": segment.get("range"),
                "state_segment_length": tokens_per_segment[index],
                "segment_ranges": all_ranges,
                "segment_lengths": segment_lengths,
                "state_value_inputs": (
                    gdn_recurrent.tensor_port(TensorValue(
                        states[index], "fp32",
                        TensorLayout.contiguous((len(lengths_i), heads, dv, dk))),
                        ready=states[index], role="state_input",
                        owner=states[index]),),
                "state_value_output": "o",
                "state_ready": states[index],
            }.items())))
        outputs.append(output_op)

    if segmented:
        output_values = [
            tensors.result(op, (1, tokens_per_segment[index], heads, dv))
            for index, op in enumerate(outputs)]
        output = _concat_token_blocks(
            tensors, output_values, (1, tokens, heads, dv),
            "pcp_output_concatenate").producer
    else:
        output = outputs[0]
    state = states[-1]
    # Internal allocations and the wrapper's requested output buffers differ.
    # TransposeContiguous may be a view, but ViewCopy still writes that buffer.
    def export(value, name):
        destination = TensorValue("", value.dtype, TensorLayout.contiguous(value.layout.shape),
                                  output="caller:" + name)
        tensors.copy_into(value, destination, name + "_caller_copy")

    def prepare_values(name, count, width):
        values = []
        for index, op in enumerate(prepares):
            value = tensors.result(op, (1, count, tokens_per_segment[index], width),
                                   output=name)
            value = _materialize(b, tensors, value.permute((0, 2, 1, 3)),
                                 name + f"_export{index}",
                                 reason="fla_requested_normalized_output")
            values.append(value)
        return values

    if segmented:
        q_values = prepare_values("q_hat", key_heads, dk)
        k_values = prepare_values("k_hat", key_heads, dk)
        export(_concat_token_blocks(tensors, q_values,
                                    (1, tokens, key_heads, dk), "q_hat_export_concat"),
               "q_hat")
        export(_concat_token_blocks(tensors, k_values,
                                    (1, tokens, key_heads, dk), "k_hat_export_concat"),
               "k_hat")
        for name in ("q_rstd", "k_rstd"):
            values = [tensors.result(op, (1, key_heads, tokens_per_segment[index]),
                                     "fp32", output=name)
                      for index, op in enumerate(prepares)]
            export(_concat_axis(b, values, (1, key_heads, tokens), 2,
                                name + "_concat"), name)
        a_values = [tensors.result(op, (1, heads, tokens_per_segment[index], chunk),
                                   output="A")
                    for index, op in enumerate(prepares)]
        export(_concat_axis(b, a_values, (1, heads, tokens, chunk), 2, "A_concat"), "A")
        g_values = []
        for index, op in enumerate(prepares):
            value = tensors.result(op, (1, heads, tokens_per_segment[index]), "fp32",
                                   output="g_cumsum")
            g_values.append(_materialize(
                b, tensors, value.permute((0, 2, 1)), f"g_cumsum_export{index}",
                reason="fla_requested_cumsum_output"))
        export(_concat_token_blocks(tensors, g_values, (1, tokens, heads),
                                    "g_cumsum_export_concat"), "g_cumsum")
    else:
        for name, count, width in (("q_hat", key_heads, dk), ("k_hat", key_heads, dk)):
            value = tensors.result(prepares[0], (1, count, tokens, width), output=name)
            value = _materialize(b, tensors, value.permute((0, 2, 1, 3)), name + "_export",
                                 reason="fla_requested_normalized_output")
            export(value, name)
        for name in ("q_rstd", "k_rstd"):
            export(tensors.result(prepares[0], (1, key_heads, tokens), "fp32",
                                  output=name), name)
        export(tensors.result(prepares[0], (1, heads, tokens, chunk), output="A"), "A")
        value = tensors.result(prepares[0], (1, heads, tokens), "fp32",
                               output="g_cumsum")
        value = _materialize(b, tensors, value.permute((0, 2, 1)), "g_cumsum_export",
                             reason="fla_requested_cumsum_output")
        export(value, "g_cumsum")
    return PrefillComposite(
        prepares[0], state, output, tuple(b.ops), tuple(states),
        prepares=tuple(prepares), outputs=tuple(outputs),
        segment_ranges=all_ranges, segment_lengths=segment_lengths,
        segmented=segmented)
