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
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators.linear_attention import gdn_chunk32
from .gdn_domain import (
    GDN_CHUNK32_HEAD_DIM_V,
    GDN_CHUNK32_TOKENS,
    GDN_CHUNK64_TOKENS,
    GDN_HEAD_DIM_K,
)
from mindie_sim.prediction.operators.linear_attention import gdn_output, gdn_prepare, gdn_state
from mindie_sim.prediction.operators.linear_attention import gdn_recurrent

SWAP_SEQUENCE_AND_HEAD_AXES = (0, 2, 1, 3)
FP32_BYTES = 4
TRITON_L2NORM_OPS = 5
TRITON_KKT_EPILOGUE_OPS = 4
TRITON_SOLVE_BLOCK = 16
TRITON_SOLVE_MERGE_16_GEMMS = 4
TRITON_SOLVE_MERGE_32_GEMMS = 2
TRITON_SOLVE_HALF_CHUNK = 32
TRITON_SOLVE_BLOCK_CUBE = TRITON_SOLVE_BLOCK * TRITON_SOLVE_BLOCK * TRITON_SOLVE_BLOCK
TRITON_SOLVE_HALF_CUBE = TRITON_SOLVE_HALF_CHUNK * TRITON_SOLVE_HALF_CHUNK * TRITON_SOLVE_HALF_CHUNK
TRITON_DIAG_ROW_UPDATES = TRITON_SOLVE_BLOCK - 1
TRITON_DIAG_OPS_PER_BLOCK = (
    TRITON_SOLVE_BLOCK * TRITON_SOLVE_BLOCK
    + TRITON_DIAG_ROW_UPDATES * (2 * TRITON_SOLVE_BLOCK * TRITON_SOLVE_BLOCK)
    + TRITON_SOLVE_BLOCK
)
TRITON_MERGE_NEGATE_OPS = (
    2 * TRITON_SOLVE_BLOCK * TRITON_SOLVE_BLOCK
    + TRITON_SOLVE_HALF_CHUNK * TRITON_SOLVE_HALF_CHUNK
)
TRITON_CHUNK64_THIRD_QUARTER = TRITON_SOLVE_HALF_CHUNK + TRITON_SOLVE_BLOCK


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


def _export_prepare_values(b, tensors, *, prepares, segmented, tokens_per_segment,
                           tokens, heads, key_heads, dk, chunk):
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
            value = _materialize(b, tensors, value.permute(SWAP_SEQUENCE_AND_HEAD_AXES),
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
            value = _materialize(b, tensors, value.permute(SWAP_SEQUENCE_AND_HEAD_AXES), name + "_export",
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


def _append_prefill_state(b, bind, state_mod, prepare_deps, *, segment, index, segmented,
                          lengths_i, count_i, initial_state_dependency, previous_state,
                          heads, dv, dk, state_v_first=True):
    dependency = segment.get("initial_dependency", initial_state_dependency)
    if index > 0 and dependency is None:
        if segment.get("exchange_bytes"):
            exchange = b.communicate(
                f"pcp_recurrent_state_step{index}", (previous_state,),
                segment["exchange_bytes"], "state_transfer", "pcp",
                from_version=segment.get("from_version"),
                to_version=segment.get("to_version"),
                state_scope="recurrent_local_bank",
                remote_producer="external_rank",
                source_implementation="pcp1_only",
                status="pending_external_rank")
            dependency = exchange
        else:
            dependency = previous_state
    contract = segment.get("contract")
    segment_effects = tuple(segment.get("effects", ()))
    state_name = "gdn_state" if not segmented else f"gdn_state{index}"
    base_deps = (prepare_deps,) if isinstance(prepare_deps, str) else tuple(prepare_deps)
    deps = tuple(dict.fromkeys((*base_deps, dependency) if dependency else base_deps))
    state = bind(state_mod, state_name, deps,
                 {"use_initial_state": True, "output_final_state": True,
                  "state_v_first": state_v_first},
                 lengths=lengths_i)
    state_shape = (len(lengths_i), heads, dv, dk) if state_v_first else (len(lengths_i), heads, dk, dv)
    if contract is not None:
        explicit_inputs = tuple(segment.get("value_inputs") or ())
        if explicit_inputs:
            input_port = explicit_inputs[0]
        elif dependency:
            input_port = gdn_recurrent.tensor_port(TensorValue(
                str(dependency), contract.get("state_dtype") or "fp32",
                TensorLayout.contiguous(state_shape),
                state_version=segment.get("to_version")),
                ready=str(dependency), role="state_input", owner=str(dependency))
        else:
            input_port = gdn_recurrent.tensor_port(TensorValue(
                "", contract.get("state_dtype") or "fp32",
                TensorLayout.contiguous(state_shape),
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
            "state_layout": "V,K" if state_v_first else "K,V",
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
    return state


def _append_prefill_prepare(b, bind, prepare_mod, *, inputs, gate, beta, source_inputs,
                            segment, index, segmented, lengths_i, count_i, offset,
                            all_ranges, segment_lengths, is_chunk32):
    def slice_segment(value):
        # Inputs have already been converted to head-first BNSD/BNS, so the
        # physical token axis is 2; the local offset is the actual byte offset
        # inside the caller's token range.
        return value.narrow(2, offset, count_i)

    if segmented:
        q_i, k_i, v_i = (slice_segment(value) for value in inputs)
        gate_i = slice_segment(gate)
        beta_i = slice_segment(beta)
        input_views = (q_i, k_i, v_i, gate_i, beta_i)
        source_views = tuple(
            value.narrow(1 if len(value.layout.shape) == 4 else 1,
                         offset, count_i)
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
    return prepare


def _triton_merge_offdiag_elems(lengths) -> int:
    """Exact masked off-diagonal A elements read by merge_16x16_to_64x64_inverse_kernel."""
    total = 0
    for q_len in lengths:
        remaining = int(q_len)
        while remaining > 0:
            rows = min(GDN_CHUNK64_TOKENS, remaining)
            total += max(0, min(rows, TRITON_SOLVE_HALF_CHUNK) - TRITON_SOLVE_BLOCK) * TRITON_SOLVE_BLOCK
            total += max(0, rows - TRITON_CHUNK64_THIRD_QUARTER) * TRITON_SOLVE_BLOCK
            total += max(0, rows - TRITON_SOLVE_HALF_CHUNK) * TRITON_SOLVE_HALF_CHUNK
            remaining -= rows
    return total


def _lower_triton_pcp_prepare(
    b, tensors, *, q, k, v, gate, beta, source_inputs, external,
    all_lengths, tokens, all_ranges, segment_lengths, heads, key_heads,
    dk, dv, dtype, pcp, native_requests,
):
    """[Cohesive-Pipeline] Lower ONE varlen batched Triton prepare sequence for PCP > 1 (chunk.py:53-155)."""
    req_override = tuple(native_requests) if native_requests is not None else b.common["requests"]
    elem_bytes = int(dtype_bytes(dtype))
    beta_bytes = int(dtype_bytes(beta.dtype))
    bt = GDN_CHUNK64_TOKENS
    total_chunks = max(1, sum((int(q_len) + bt - 1) // bt for q_len in all_lengths))
    diag_blocks = max(1, sum((int(q_len) + TRITON_SOLVE_BLOCK - 1) // TRITON_SOLVE_BLOCK for q_len in all_lengths))
    cu_seqlens = [0]
    for q_len in all_lengths:
        cu_seqlens.append(cu_seqlens[-1] + int(q_len))

    def _stamp_prefill(op_id: str) -> str:
        b.ops[-1] = replace(b.ops[-1], phase="prefill", requests=req_override)
        return op_id

    # 1. External Q/K L2-norm on token-major BSND inputs (gdn.py:940, chunk.py:264-266).
    q_norm_id = _stamp_prefill(b.vector(
        "gdn_q_l2norm", tuple(dep for dep in (q.producer,) if dep),
        tokens * key_heads * dk, TRITON_L2NORM_OPS,
        shape=(1, tokens, key_heads, dk), dtype=dtype,
        metadata=(("native_kind", "l2norm_fwd"),
                  ("source", "vllm_ascend/ops/triton/fla/l2norm.py:l2norm_fwd")),
    ))
    q_norm = tensors.result(q_norm_id, (1, tokens, key_heads, dk), dtype)
    k_norm_id = _stamp_prefill(b.vector(
        "gdn_k_l2norm", tuple(dep for dep in (k.producer,) if dep),
        tokens * key_heads * dk, TRITON_L2NORM_OPS,
        shape=(1, tokens, key_heads, dk), dtype=dtype,
        metadata=(("native_kind", "l2norm_fwd"),
                  ("source", "vllm_ascend/ops/triton/fla/l2norm.py:l2norm_fwd")),
    ))
    k_norm = tensors.result(k_norm_id, (1, tokens, key_heads, dk), dtype)

    # 2. chunk_local_cumsum on token-major g [1, T, H] FP32 (chunk.py:62, cumsum.py:62).
    g_cumsum_id = _stamp_prefill(b.work(
        "gdn_gate_cumsum",
        tuple(dep for dep in (gate.producer,) if dep),
        (1, tokens, heads),
        vector=float(tokens * heads),
        read=float(tokens * heads * FP32_BYTES),
        write=float(tokens * heads * FP32_BYTES),
        family="elementwise",
        dtype="fp32",
        recipe="gdn_gate_cumsum",
        metadata=(
            ("native_kind", "chunk_local_cumsum"),
            ("source", "vllm_ascend/ops/triton/fla/cumsum.py:chunk_local_cumsum"),
            ("chunk", bt), ("pcp", int(pcp)),
            ("cu_seqlens", tuple(cu_seqlens)),
        ),
    ))
    g_cumsum = tensors.result(g_cumsum_id, (1, tokens, heads), "fp32")

    # 3. Head-first transposes at their upstream consumers (chunk_scaled_dot_kkt.py:142-143, chunk.py:84).
    g_hf = _materialize(
        b, tensors, g_cumsum.permute((0, 2, 1)), "g_head_first",
        reason="fla_BSN_to_BNS", source=external.get("g"))
    beta_hf = _materialize(
        b, tensors, beta.permute((0, 2, 1)), "beta_head_first",
        reason="fla_BSN_to_BNS", source=external.get("beta"))

    # 4. chunk_scaled_dot_kkt_fwd on token-major k (chunk_scaled_dot_kkt.py:64-87):
    #    fixed BT=64 tl.dot Cube work; boundary_check masks loads and stores to valid rows.
    kkt_flops = float(2 * total_chunks * heads * bt * bt * dk)
    kkt = _stamp_prefill(b.work(
        "chunk_products",
        tuple(dict.fromkeys(dep for dep in (k_norm.producer, beta_hf.producer, g_hf.producer) if dep)),
        (1, tokens, heads, bt),
        flops=kkt_flops,
        matrix_flops=((dtype, kkt_flops),),
        vector=float(total_chunks * heads * bt * bt * TRITON_KKT_EPILOGUE_OPS),
        read=float(tokens * heads * (dk * elem_bytes + FP32_BYTES + beta_bytes)),
        write=float(tokens * heads * bt * FP32_BYTES),
        family="recurrent_local",
        dtype=dtype,
        recipe="chunk_products",
        metadata=(
            ("native_kind", "chunk_scaled_dot_kkt_fwd"),
            ("source", "vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py:chunk_scaled_dot_kkt_fwd"),
            ("chunk", bt), ("pcp", int(pcp)),
            ("cu_seqlens", tuple(cu_seqlens)),
        ),
    ))

    # 5a. solve_tril launch 1: solve_tril_16x16_kernel (solve_tril.py:97-134, 15 row updates/block).
    tri_diag = _stamp_prefill(b.work(
        "triangular_solve_diag",
        (kkt,),
        (diag_blocks, heads, TRITON_SOLVE_BLOCK, TRITON_SOLVE_BLOCK),
        vector=float(diag_blocks * heads * TRITON_DIAG_OPS_PER_BLOCK),
        read=float(tokens * heads * TRITON_SOLVE_BLOCK * FP32_BYTES),
        write=float(tokens * heads * TRITON_SOLVE_BLOCK * FP32_BYTES),
        family="triangular_solve",
        dtype="fp32",
        recipe="triangular_solve_diag",
        metadata=(
            ("native_kind", "solve_tril_16x16_kernel"),
            ("source", "vllm_ascend/ops/triton/fla/solve_tril.py:solve_tril_16x16_kernel"),
            ("chunk", bt), ("pcp", int(pcp)),
            ("diag_blocks", diag_blocks),
            ("cu_seqlens", tuple(cu_seqlens)),
        ),
    ))

    # 5b. solve_tril launch 2: merge_16x16_to_64x64_inverse_kernel (solve_tril.py:240-328, FP32 IEEE dots).
    merge_flops_per_chunk = 2 * (
        TRITON_SOLVE_MERGE_16_GEMMS * TRITON_SOLVE_BLOCK_CUBE
        + TRITON_SOLVE_MERGE_32_GEMMS * TRITON_SOLVE_HALF_CUBE
    )
    merge_flops = float(merge_flops_per_chunk * total_chunks * heads)
    merge_read_elems = tokens * TRITON_SOLVE_BLOCK + _triton_merge_offdiag_elems(all_lengths)
    tri = _stamp_prefill(b.work(
        "triangular_solve",
        (kkt, tri_diag),
        (total_chunks, heads, bt, bt),
        flops=merge_flops,
        matrix_flops=(("fp32", merge_flops),),
        vector=float(total_chunks * heads * TRITON_MERGE_NEGATE_OPS),
        read=float(merge_read_elems * heads * FP32_BYTES),
        write=float(tokens * heads * bt * elem_bytes),
        family="triangular_solve",
        dtype="fp32",
        recipe="triangular_solve",
        metadata=(
            ("native_kind", "merge_16x16_to_64x64_inverse_kernel"),
            ("source", "vllm_ascend/ops/triton/fla/solve_tril.py:merge_16x16_to_64x64_inverse_kernel"),
            ("chunk", bt), ("pcp", int(pcp)),
            ("cu_seqlens", tuple(cu_seqlens)),
        ),
    ))

    # 6. recompute_w_u_fwd on token-major k, v (wy_fast.py:60-95): fixed BT=64 FP32 tl.dot tiles.
    wy_flops = float(2 * total_chunks * heads * bt * bt * (dk + dv))
    input_views = (q_norm, k_norm, v, g_hf, beta_hf)
    wy = _stamp_prefill(b.work(
        "wy_transform",
        tuple(dict.fromkeys(dep for dep in (tri, v.producer, k_norm.producer, beta_hf.producer, g_hf.producer) if dep)),
        (1, tokens, heads, dk + dv),
        flops=wy_flops,
        matrix_flops=(("fp32", wy_flops),),
        vector=float(tokens * heads * (dv + 2 * dk + 1)),
        read=float(tokens * heads * ((bt + dk + dv) * elem_bytes + FP32_BYTES + beta_bytes)),
        write=float(tokens * heads * (dk + dv) * elem_bytes),
        family="recurrent_local",
        dtype="fp32",
        recipe="wy_transform",
        metadata=tuple(sorted({
            "native_kind": "recompute_w_u_fwd",
            "source": "vllm_ascend/ops/triton/fla/wy_fast.py:recompute_w_u_fwd",
            "chunk": bt, "pcp": int(pcp),
            "cu_seqlens": tuple(cu_seqlens),
            "gdn_prepare_inputs": tuple(
                gdn_recurrent.tensor_port(value, role=name, ready=value.producer)
                for name, value in zip(("q", "k", "v", "gate", "beta"), input_views)),
            "gdn_prepare_source_inputs": tuple(
                gdn_recurrent.tensor_port(value, role=name, ready=value.producer or "caller")
                for name, value in zip(("q", "k", "v", "gate", "beta"), source_inputs)),
            "gdn_prepare_source_offsets": tuple(int(value.layout.offset) for value in source_inputs),
            "gdn_prepare_input_offsets": tuple(int(value.layout.offset) for value in input_views),
            "segment_ranges": all_ranges,
            "segment_lengths": segment_lengths,
        }.items())),
    ))

    # 7. AscendC H/O head-first transposes at their exact upstream consumers (chunk.py:95-96,140-141).
    q_hf = _materialize(
        b, tensors, q_norm.permute(SWAP_SEQUENCE_AND_HEAD_AXES),
        "q_head_first", reason="fla_BSND_to_BNSD")
    k_hf = _materialize(
        b, tensors, k_norm.permute(SWAP_SEQUENCE_AND_HEAD_AXES),
        "k_head_first", reason="fla_BSND_to_BNSD")
    w_tok = tensors.result(wy, (1, tokens, heads, dk), dtype, output="w")
    u_tok = tensors.result(wy, (1, tokens, heads, dv), dtype, output="u")
    w_hf = _materialize(
        b, tensors, w_tok.permute(SWAP_SEQUENCE_AND_HEAD_AXES),
        "w_head_first", reason="fla_BSND_to_BNSD")
    u_hf = _materialize(
        b, tensors, u_tok.permute(SWAP_SEQUENCE_AND_HEAD_AXES),
        "u_head_first", reason="fla_BSND_to_BNSD")
    return wy, q_hf, k_hf, w_hf, u_hf, g_hf


def prefill(*, builder, tensors, q, k, v, gate, beta, lengths, heads,
            key_heads, dk, dv, chunk, initial_state_dependency=None,
            external=None, native_requests=None, state_contract=None,
            state_effects=(), state_segments=None, pcp=1):
    """[Cohesive-Pipeline] Append one native prefill call to ``builder``.

    ``external`` maps logical input names to caller-owned source labels. Those
    inputs have no producer operator in this call, so their materializations
    carry an explicit source/stream edge in metadata instead of a fabricated
    dependency. ``initial_state_dependency`` is the serving gather op when the
    caller owns cache movement; it is absent for a standalone initial tensor.

    When ``state_segments`` carries more than one physical segment (PCP query
    ranges), every H and O phase uses the same segment lengths and local token
    offset. Independent requests start from their own gathered state; within a
    single request, the previous segment's H output feeds the next segment's H.
    """
    b = builder
    external = dict(external or {})
    lengths = tuple(lengths)
    tokens = sum(lengths)
    dtype = b.model.activation_dtype
    use_fused_prepare = int(pcp) == 1
    # Upstream chunk.py:53,154-155,235 always runs PCP on chunk_size=64,
    # natural exponentials (use_exp2=False), and K,V state (state_v_first=False).
    effective_chunk = int(chunk) if use_fused_prepare else GDN_CHUNK64_TOKENS
    is_chunk32 = use_fused_prepare and effective_chunk == GDN_CHUNK32_TOKENS
    state_v_first = use_fused_prepare
    use_exp2 = use_fused_prepare and not is_chunk32
    if (dtype != "bf16"
            or (dk, dv) != (GDN_HEAD_DIM_K, GDN_CHUNK32_HEAD_DIM_V)
            or effective_chunk not in {GDN_CHUNK32_TOKENS, GDN_CHUNK64_TOKENS}):
        raise ValueError("gdn_prefill_arch35_requires_bf16_chunk32_or_64_k128_v128")
    prepare_mod = gdn_chunk32.PREPARE if is_chunk32 else gdn_prepare
    state_mod = gdn_chunk32.STATE if is_chunk32 else gdn_state
    output_mod = gdn_chunk32.OUTPUT if is_chunk32 else gdn_output
    params = {"lengths": lengths, "heads": heads, "key_heads": key_heads,
              "dk": dk, "dv": dv, "chunk": effective_chunk, "dtype": dtype,
              "gate_dtype": "fp32", "use_exp2": use_exp2}

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

    source_inputs = (q, k, v, gate, beta)

    def bind(module, name, dependencies, flags, lengths=None):
        op = module.make_op({**params, **({"lengths": tuple(lengths)}
                                         if lengths is not None else {}), **flags})
        common = {**b.common, "phase": "prefill"}
        if native_requests is not None:
            common["requests"] = tuple(native_requests)
        return b.add(replace(op, op_id=b.prefix + "." + name,
                             deps=tuple(dep for dep in dependencies if dep), **common))

    prepares, states, outputs = [], [], []
    if use_fused_prepare:
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
            reason="fla_BSN_to_BNS", source=external.get("beta") if is_chunk32 else None)
        inputs = [
            _materialize(b, tensors, value.permute(SWAP_SEQUENCE_AND_HEAD_AXES),
                         name + "_head_first", reason="fla_BSND_to_BNSD",
                         source=external.get(name))
            for name, value in zip(("q", "k", "v"), (q, k, v))]
        for index, segment in enumerate(segments):
            lengths_i = segment_lengths[index]
            count_i = tokens_per_segment[index]
            prepare = _append_prefill_prepare(
                b, bind, prepare_mod, inputs=inputs, gate=gate, beta=beta,
                source_inputs=source_inputs, segment=segment, index=index,
                segmented=segmented, lengths_i=lengths_i, count_i=count_i,
                offset=offsets[index], all_ranges=all_ranges,
                segment_lengths=segment_lengths, is_chunk32=is_chunk32)
            prepares.append(prepare)
            state = _append_prefill_state(
                b, bind, state_mod, prepare, segment=segment, index=index,
                segmented=segmented, lengths_i=lengths_i, count_i=count_i,
                initial_state_dependency=initial_state_dependency,
                previous_state=states[-1] if states else None,
                heads=heads, dv=dv, dk=dk, state_v_first=True)
            states.append(state)
    else:
        all_lengths = tuple(q_i for seg_lens in segment_lengths for q_i in seg_lens)
        wy, q_hf, k_hf, w_hf, u_hf, g_hf = _lower_triton_pcp_prepare(
            b, tensors, q=q, k=k, v=v, gate=gate, beta=beta,
            source_inputs=source_inputs, external=external,
            all_lengths=all_lengths, tokens=tokens, all_ranges=all_ranges,
            segment_lengths=segment_lengths, heads=heads, key_heads=key_heads,
            dk=dk, dv=dv, dtype=dtype, pcp=pcp, native_requests=native_requests)
        prepares.append(wy)
        state_prepare_deps = (w_hf.producer, u_hf.producer, k_hf.producer, g_hf.producer)
        for index, segment in enumerate(segments):
            lengths_i = segment_lengths[index]
            count_i = tokens_per_segment[index]
            state = _append_prefill_state(
                b, bind, state_mod, state_prepare_deps, segment=segment, index=index,
                segmented=segmented, lengths_i=lengths_i, count_i=count_i,
                initial_state_dependency=initial_state_dependency,
                previous_state=states[-1] if states else None,
                heads=heads, dv=dv, dk=dk, state_v_first=False)
            states.append(state)

    for index, segment in enumerate(segments):
        lengths_i = segment_lengths[index]
        name = "gdn_output" if not segmented else f"gdn_output{index}"
        out_deps = (
            (prepares[index], states[index])
            if use_fused_prepare
            else tuple(dict.fromkeys((q_hf.producer, k_hf.producer, g_hf.producer, states[index])))
        )
        output_op = bind(output_mod, name, out_deps,
                         {"state_v_first": state_v_first}, lengths=lengths_i)
        if segmented:
            out_state_shape = (
                (len(lengths_i), heads, dv, dk) if state_v_first
                else (len(lengths_i), heads, dk, dv)
            )
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
                        TensorLayout.contiguous(out_state_shape)),
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
    if use_fused_prepare:
        _export_prepare_values(
            b, tensors, prepares=prepares, segmented=segmented, tokens_per_segment=tokens_per_segment,
            tokens=tokens, heads=heads, key_heads=key_heads, dk=dk, chunk=effective_chunk)
    return PrefillComposite(
        prepares[0], state, output, tuple(b.ops), tuple(states),
        prepares=tuple(prepares), outputs=tuple(outputs),
        segment_ranges=all_ranges, segment_lengths=segment_lengths,
        segmented=segmented)
