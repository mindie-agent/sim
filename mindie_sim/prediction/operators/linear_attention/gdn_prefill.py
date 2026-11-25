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
from typing import Literal
import math

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.prediction.operators.linear_attention import gdn_composite
from .gdn_decode import _resolve_local_gdn_heads
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
from mindie_sim.prediction.operators.recurrent.recover_ssm_work import (
    emit_commit_op,
    recover_ssm_enabled,
    rewrite_gdn_analytical_scan,
)
from mindie_sim.prediction.tensors import TensorLowering

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_REV = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
_OPS_TRANSFORMER_REV = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/gdn.py",
        source_sha256="b170b30626cf46bc4ec1e47080f9487c1e5443e51add42bb8c67bfd20a79fac4",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/fla/chunk.py",
        source_sha256="d624a189029371ddbcbf7fa8a602734fc758f341f445ff9bb9677043f127b9f9",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/fla/chunk_delta_hupdate.py",
        source_sha256="19a25d233a68f34cfe733e12c96ced18b7243ee913f66e4fe527553cec42b224",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/device/device_op.py",
        source_sha256="d9b682e08f292ec71cdb7877f755b0adafb3eed50acd0db353df6729529dae24",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_TRANSFORMER_REV,
        relative_path="mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_fn.h",
        source_sha256="a63d194254fab4c00db1f1265a52482517fd094a260f76e09165caf91b2119d3",
    ),
)

DECODE_QK_L2NORM_OPS_PER_ELEMENT = 5
DECODE_RECURRENCE_OPS_PER_STATE_ELEMENT = 7
DELTA_GATE_OPS_PER_HEAD = 12
OUTPUT_NORM_GATE_OPS_PER_ELEMENT = 9
STATE_INDEX_BYTES = 4
FP32_BYTES = 4
PCP_PREFIX_SCAN_VECTOR_PASSES = 2
PCP_PREFIX_STEP_STATE_READS = 6
PCP_PREFIX_STEP_STATE_WRITES = 4
PCP_HUPDATE_DIAG_HALVES = 2
STATE_BLOCK_ELEMENTS = 1024


def pcp_gdn_prefix_work(
    *, n_pf: int, tokens: int, chunks: int, chunk: int, heads: int,
    dk: int, dv: int, pcp: int, act_dtype: str = "bf16",
) -> dict[str, float | int | tuple[tuple[str, float], ...]]:
    """Canonical PCP GDN ``hupdate`` + ``prefix_scan`` work (chunk_delta_hupdate.py & chunk.py)."""
    n_seq = max(0, int(n_pf))
    scan_tok = max(0, int(tokens))
    scan_chunks = max(0, int(chunks))
    scan_chunk = max(1, int(chunk))
    scan_heads = max(1, int(heads))
    elem_bytes = int(dtype_bytes(act_dtype))
    per_chunk_phi = float(scan_heads * dk * dk * FP32_BYTES)
    phi_bytes = float(n_seq) * per_chunk_phi
    st_bytes = float(n_seq * scan_heads * dk * dv * FP32_BYTES)
    pcp_steps = max(0, int(pcp) - 1)
    prefix_steps = pcp_steps * n_seq
    # chunk_delta_hupdate.py:118-143: 4 local [K/2,BT]@[BT,K/2] act_dtype GEMMs +
    # 8 composition [K/2,K/2]@[K/2,K/2] FP32 GEMMs per chunk-head, plus 1 identity eye(K) write/seq.
    hupdate_act_flops = float(2 * scan_chunks * scan_heads * dk * dk * scan_chunk)
    hupdate_fp32_flops = float(2 * scan_chunks * scan_heads * dk * dk * dk)
    hupdate_flops = hupdate_act_flops + hupdate_fp32_flops
    if hupdate_flops <= 0.0:
        hupdate_matrix_flops: tuple[tuple[str, float], ...] = ()
    elif act_dtype == "fp32":
        hupdate_matrix_flops = (("fp32", hupdate_flops),)
    else:
        hupdate_matrix_flops = ((act_dtype, hupdate_act_flops), ("fp32", hupdate_fp32_flops))
    hupdate_vector = float(
        scan_chunks * scan_heads * (
            2 * dk * dk + (dk * dk) // PCP_HUPDATE_DIAG_HALVES + scan_chunk * dk + 2 * scan_chunk
        )
    )
    hupdate_read = (
        float(scan_chunks) * per_chunk_phi
        + float(scan_tok * scan_heads * (2 * dk * elem_bytes + FP32_BYTES))
        if scan_chunks > 0 else 0.0
    )
    hupdate_write = float(n_seq + scan_chunks) * per_chunk_phi
    # chunk.py:189-196: initial slice copy + (pcp-1) steps of sub, matmul, add, slice copy.
    prefix_flops = float(2 * prefix_steps * scan_heads * dk * dk * dv)
    prefix_vector = float(PCP_PREFIX_SCAN_VECTOR_PASSES * prefix_steps * scan_heads * dk * dv)
    prefix_read = (
        (1 + pcp_steps * PCP_PREFIX_STEP_STATE_READS) * st_bytes + pcp_steps * phi_bytes
        if (n_seq > 0 and pcp_steps > 0) else 0.0
    )
    prefix_write = (
        (1 + pcp_steps * PCP_PREFIX_STEP_STATE_WRITES) * st_bytes
        if (n_seq > 0 and pcp_steps > 0) else 0.0
    )
    return {
        "n_pf": n_seq,
        "scan_tok": scan_tok,
        "scan_chunks": scan_chunks,
        "scan_chunk": scan_chunk,
        "scan_heads": scan_heads,
        "phi_bytes": phi_bytes,
        "st_bytes": st_bytes,
        "hupdate_flops": hupdate_flops,
        "hupdate_matrix_flops": hupdate_matrix_flops,
        "hupdate_vector": hupdate_vector,
        "hupdate_read": hupdate_read,
        "hupdate_write": hupdate_write,
        "prefix_steps": prefix_steps,
        "prefix_flops": prefix_flops,
        "prefix_vector": prefix_vector,
        "prefix_read": prefix_read,
        "prefix_write": prefix_write,
    }


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


def _lower_gdn_pcp_prefix_sync(
    b, *, composite, active, tokens, chunk, heads, dk, dv, pcp, pcp_rank,
):
    """Reorder post-state ops after PCP state/phi AllGathers, prefix scan, and rank>0 rerun."""
    n_pf = len(active)
    if n_pf == 0:
        return tuple(composite.states)
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
    scan_seq_lens = tuple(int(v.q_len) for v in active if int(v.q_len) > 0)
    scan_tok = sum(scan_seq_lens)
    scan_chunk = GDN_CHUNK64_TOKENS
    scan_chunks = sum((q_i + scan_chunk - 1) // scan_chunk for q_i in scan_seq_lens)
    scan_seq_meta = (("sequence_tokens", scan_seq_lens),) if scan_seq_lens else ()
    scan_heads = max(1, int(heads))
    pw = pcp_gdn_prefix_work(
        n_pf=n_pf, tokens=scan_tok, chunks=scan_chunks, chunk=scan_chunk,
        heads=scan_heads, dk=dk, dv=dv, pcp=pcp, act_dtype=b.model.activation_dtype,
    )
    hupdate = b.work(
        "pcp_gdn_hupdate", tuple(composite.states), (n_pf, scan_heads, dk, dk),
        flops=pw["hupdate_flops"],
        matrix_flops=pw["hupdate_matrix_flops"],
        vector=pw["hupdate_vector"],
        read=pw["hupdate_read"], write=pw["hupdate_write"],
        family="recurrent_scan", dtype="fp32",
        metadata=(
            ("pcp", int(pcp)), ("chunks", scan_chunks), ("chunk", scan_chunk),
            ("tokens", scan_tok), ("heads", scan_heads),
            ("source", "ops/triton/fla/chunk.py:fwd_hupdate"),
        ) + scan_seq_meta,
    )
    ag_state = b.communicate(
        "pcp_gdn_state_gather", tuple(composite.states),
        pw["st_bytes"] * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    ag_phi = b.communicate(
        "pcp_gdn_hupdate_gather", (hupdate,),
        pw["phi_bytes"] * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    prefix_steps = int(pw["prefix_steps"])
    prefix_scan = b.work(
        "pcp_gdn_prefix_scan", (ag_state, ag_phi), (pcp - 1, n_pf, scan_heads, dk, dv),
        flops=pw["prefix_flops"],
        vector=pw["prefix_vector"],
        read=pw["prefix_read"], write=pw["prefix_write"],
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
            read=pw["st_bytes"], write=pw["st_bytes"], family="recurrent_scan", dtype="fp32",
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
    return (prefix_scan,)


def _declare_gdn_chunk_group(
    b, *, chunk_start_index, composite, tokens, heads, key_heads, dk, dv,
    dtype, state_dtype, lengths, chunk, pcp,
):
    region_ops = b.ops[chunk_start_index:]
    if not region_ops:
        return
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
            "native_kinds": (
                ("ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO")
                if int(pcp) == 1
                else ("chunk_local_cumsum", "chunk_scaled_dot_kkt_fwd", "solve_tril",
                      "recompute_w_u_fwd", "ChunkFwdH", "ChunkFwdO")
            ),
            "selection": (
                "arch35_gdn_prefill_composite"
                if int(pcp) == 1
                else "arch35_gdn_pcp_triton_prepare_composite"
            ),
            "conditions": {"pcp": int(pcp), "chunk": chunk,
                           "state_dtype": state_dtype},
        },
        source=source_reference("vllm_ascend/ops/gdn.py",
                                "AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused"),
        fusion_status="fused_existing",
        fusion_reason="arch35_native_gdn_prefill_composite",
        native_fusion_eligible=False,
    )


def core(*, builder, tensors, conv, gated, views, heads, key_heads, dk, dv,
          chunk, gate_values=None, qkv_values=None, pcp=1, pcp_rank=None,
          state_slot_indices=None, state_capacity=None, state_slot_source="", **_):
    b = builder
    tokens = sum(view.q_len for view in views)
    dtype = b.model.activation_dtype
    effective_chunk = GDN_CHUNK64_TOKENS if int(pcp) > 1 else int(chunk)
    lengths = tuple(view.q_len for view in views if view.q_len > 0)
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
    state_blocks = math.ceil(heads * dk * dv / STATE_BLOCK_ELEMENTS)
    gather_metadata = state_blocks * (len(views) + STATE_INDEX_BYTES * sum(prior))
    initial_flags = tuple(view.context > 0 and not view.padding for view in views)
    global_lengths = tuple(view.global_q_len or view.q_len for view in views)
    slot_indices = state_slot_indices
    slot_source = state_slot_source
    gather_contract, gather_effects = gdn_recurrent.state_contract(
        views=views, phase="prefill", state_dtype=state_dtype,
        conv_state_dtype=b.model.activation_dtype, conv_kernel=b.plan.conv_kernel,
        target_prefix=b.prefix, chunk=effective_chunk, initial_state=initial_flags,
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
    initial_state_dep = gather
    if int(pcp) > 1:
        # Upstream vllm_ascend/ops/gdn.py:935 transposes cache V,K -> K,V for Triton/PCP path.
        initial_state_dep = b.work(
            "initial_state_kv_transpose", (gather,), (len(views), heads, dk, dv),
            read=state_elements * state_byte, write=state_elements * state_byte,
            family="tensor_copy", dtype=state_dtype,
            metadata=(("state_layout", "K,V"), ("source", "vllm_ascend/ops/gdn.py:935")),
        )
    gate, beta = gate_values or (
        tensors.result(gated, (1, tokens, heads), "fp32", output="decay"),
        tensors.result(gated, (1, tokens, heads), dtype, output="beta"))
    state_bytes = heads * dk * dv * dtype_bytes(state_dtype)
    state_segments = _prefill_state_segments(
        b, views=views, pcp=pcp, state_dtype=state_dtype, state_bytes=state_bytes, chunk=effective_chunk,
        slot_indices=slot_indices, state_capacity=state_capacity, slot_source=slot_source,
        initial_flags=initial_flags, global_lengths=global_lengths, lengths=lengths,
        gather=initial_state_dep, gather_outputs=gather_outputs, heads=heads, dk=dk, dv=dv)
    composite = gdn_composite.prefill(
        builder=b, tensors=tensors, q=values[0], k=values[1], v=values[2],
        gate=gate, beta=beta, lengths=lengths, heads=heads, key_heads=key_heads,
        dk=dk, dv=dv, chunk=effective_chunk, initial_state_dependency=initial_state_dep,
        native_requests=tuple(view.request_id for view in views),
        state_segments=state_segments, pcp=pcp)
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
        target_prefix=b.prefix, chunk=effective_chunk, initial_state=initial_flags,
        direction="scatter", target_kind="recurrent",
        global_lengths=global_lengths, slot_indices=slot_indices,
        state_capacity=state_capacity, slot_source=slot_source,
        remote_publish=remote_flags)
    if pcp > 1:
        # Upstream ``ops/triton/fla/chunk.py:198,210,240``: prefix_scan yields the
        # global final state for persistent cache scatter on ALL PCP ranks; the
        # rank>0 rerun discards its final-state return and feeds local gdn_output*.
        scatter_deps = _lower_gdn_pcp_prefix_sync(
            b, composite=composite, active=active, tokens=tokens, chunk=effective_chunk,
            heads=heads, dk=dk, dv=dv, pcp=pcp, pcp_rank=pcp_rank)
        scatter_op_deps = scatter_deps
        if active_elements > 0:
            # Upstream vllm_ascend/ops/gdn.py:942 transposes final K,V -> V,K ahead of cache scatter.
            vk_transpose = b.work(
                "final_state_vk_transpose", scatter_deps, (len(views), heads, dv, dk),
                read=active_elements * state_byte, write=active_elements * state_byte,
                family="tensor_copy", dtype=state_dtype,
                metadata=(("state_layout", "V,K"), ("source", "vllm_ascend/ops/gdn.py:942")),
            )
            scatter_op_deps = (*scatter_deps, vk_transpose)
    else:
        scatter_deps = tuple(composite.states)
        scatter_op_deps = scatter_deps
    scatter = b.work("scatter_recurrent_state", scatter_op_deps,
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
    _declare_gdn_chunk_group(
        b, chunk_start_index=chunk_start_index, composite=composite, tokens=tokens,
        heads=heads, key_heads=key_heads, dk=dk, dv=dv, dtype=dtype,
        state_dtype=state_dtype, lengths=lengths, chunk=effective_chunk, pcp=pcp)
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
                               heads, width, actual, pcp_rank=None):
    prefix, plan, model = b.prefix, b.plan, b.model
    # Physical row positions and qsl stay intact.  FN resolves the cache index
    # per sequence and skips the null/pad *work* while preserving the request's
    # physical offset (the next request's qsl must still start at this row).
    positive_idx = tuple(i for i, view in enumerate(ordered) if view.q_len > 0)
    conv_ordered = tuple(ordered[i] for i in positive_idx)
    conv_lengths = tuple(view.q_len for view in conv_ordered)
    conv_initial = tuple(initial_flags[i] for i in positive_idx)
    conv_cache = None if cache_indices is None else tuple(cache_indices[i] for i in positive_idx)
    conv_global = tuple(global_lengths[i] for i in positive_idx)
    conv_padding = tuple(i for i, view in enumerate(conv_ordered) if view.padding)
    conv_params = {
        "lengths": conv_lengths,
        "dim": width, "width": plan.conv_kernel, "dtype": model.activation_dtype,
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": conv_initial,
        "cache_indices": conv_cache,
        "state_history_slots": plan.conv_kernel - 1 + int(
            dict(runtime.metadata).get("draft_proposal_tokens", 0)),
        "null_block_id": null_block_id,
    }
    if state_rows is not None:
        conv_params["state_rows"] = state_rows
    conv_op = native_causal_conv_prefill.make_op(conv_params)
    conv_parts = []
    conv_cursor = 0
    for index, view in enumerate(conv_ordered):
        part_direction = "token" if view.phase == "decode" else "conv"
        conv_view = conv_input.narrow(0, conv_cursor, view.q_len)
        conv_cursor += view.q_len
        part, part_effects = gdn_recurrent.state_contract(
            views=(view,), phase=view.phase,
            state_dtype=plan.recurrent_state_dtype,
            conv_state_dtype=plan.conv_state_dtype,
            conv_kernel=plan.conv_kernel, target_prefix=prefix, chunk=chunk,
            initial_state=(conv_initial[index],),
            direction=part_direction, target_kind="conv",
            global_lengths=(conv_global[index],),
            slot_indices=(None if conv_cache is None else conv_cache[index],),
            state_capacity=state_rows, slot_source=index_source,
            value_inputs=(conv_view,), heads=heads,
            remote_publish=(parallel.pcp > 1
                            and view.phase == "prefill"))
        conv_parts.append((part, part_effects))
    mixed_contract = gdn_recurrent.merge_contracts(
        [part for part, _ in conv_parts], allow_mixed_directions=True)
    mixed_conv = tuple(effect for _, effects in conv_parts for effect in effects)
    conv_meta = {
        "padding_slots": conv_padding,
        "state_dtype": plan.conv_state_dtype,
        "state_layout": "slots_history_channels",
        "state_mutation": "inplace_history_writeback",
        "state_write_bytes": (len(conv_ordered) - len(conv_padding))
                             * (plan.conv_kernel - 1) * width * 2,
        "state_slots": tuple(index for index in range(len(conv_ordered))
                             if index not in conv_padding),
        "state_copy_boundary": "caller_state_rows_to_conv_history_slots",
        "state_index_source": index_source,
        "state_capacity_source": ("caller_metadata" if state_rows is not None
                                  else "unbound"),
        "state_rows": state_rows,
        "state_rows_bound": state_rows is not None,
        "cache_indices": conv_cache,
        "cache_index_present": conv_cache is not None,
        "cache_index_bound": (conv_cache is not None
                              and all(value is not None for value in conv_cache)),
        "null_block_id": null_block_id,
    }
    conv = lower_conv_stage(
        builder=b, tensors=tensors, prefix=prefix, plan=plan,
        parallel=parallel, ordered=conv_ordered, prefill=prefill, decode=decode,
        conv_input=conv_input, conv_params=conv_params, conv_op=conv_op,
        conv_meta=conv_meta, mixed_contract=mixed_contract,
        mixed_conv=mixed_conv, cache_indices=conv_cache,
        state_rows=state_rows, index_source=index_source, chunk=chunk,
        heads=heads, width=width, actual=actual, pcp_rank=pcp_rank,
    )
    return conv


def _lower_idle_prefill_pcp_collectives(
    b, *, root_dep, prefill, model, plan, parallel, heads, dk, dv, width,
    emit_conv_tail: bool,
) -> str | None:
    """Join PCP prefill collectives for live prefill slots that own 0 local tokens."""
    pcp = int(parallel.pcp)
    active = tuple(view for view in prefill if not view.padding)
    n_pf = len(active)
    if pcp <= 1 or n_pf == 0:
        return None
    dep = root_dep
    if emit_conv_tail:
        conv_state_dtype = plan.conv_state_dtype or model.activation_dtype
        state_len = max(0, plan.conv_kernel - 1)
        tail_copy_bytes = n_pf * state_len * width * dtype_bytes(conv_state_dtype)
        tail_bytes = (pcp - 1) * tail_copy_bytes
        conv_tail = b.communicate(
            "pcp_conv_tail_gather", (dep,), tail_bytes,
            "allgather", "pcp", dtype=conv_state_dtype,
            state_scope="conv_history", source="vllm_ascend.ops.gdn",
        )
        if state_len > 0:
            b.work(
                "pcp_conv_tail_publish", (conv_tail,), (n_pf, state_len, width),
                vector=0, read=tail_copy_bytes, write=tail_copy_bytes,
                family="tensor_copy", dtype=conv_state_dtype,
                metadata=(("state_scope", "conv_history"), ("effect_role", "conv_tail_publish"),
                          ("source", "vllm_ascend/ops/gdn.py:608")),
            )
        dep = conv_tail
    state_dtype = plan.recurrent_state_dtype or "fp32"
    state_byte = dtype_bytes(state_dtype)
    active_elements = n_pf * heads * dk * dv
    prior_elements = sum(view.context > 0 for view in active) * heads * dk * dv
    state_blocks = math.ceil(heads * dk * dv / STATE_BLOCK_ELEMENTS)
    gather_meta = state_blocks * (n_pf + STATE_INDEX_BYTES * sum(view.context > 0 for view in active))
    pw = pcp_gdn_prefix_work(
        n_pf=n_pf, tokens=0, chunks=0, chunk=GDN_CHUNK64_TOKENS,
        heads=heads, dk=dk, dv=dv, pcp=pcp, act_dtype=model.activation_dtype,
    )
    idle_start = len(b.ops)
    # Idle live prefill slots supply their initial state and identity eye(K) update to the gathers.
    gather = b.work(
        "gather_recurrent_state", (dep,), (n_pf, heads, dv, dk),
        read=prior_elements * state_byte + gather_meta,
        write=active_elements * state_byte,
        family="state_gather", dtype=state_dtype,
        metadata=(("state_layout", "V,K"), ("active_state_rows", n_pf), ("effect_role", "gather_read")),
    )
    kv_transpose = b.work(
        "initial_state_kv_transpose", (gather,), (n_pf, heads, dk, dv),
        read=active_elements * state_byte, write=active_elements * state_byte,
        family="tensor_copy", dtype=state_dtype,
        metadata=(("state_layout", "K,V"), ("source", "vllm_ascend/ops/gdn.py:935")),
    )
    hupdate = b.work(
        "pcp_gdn_hupdate", (kv_transpose,), (n_pf, heads, dk, dk),
        flops=pw["hupdate_flops"], matrix_flops=pw["hupdate_matrix_flops"],
        vector=pw["hupdate_vector"], read=pw["hupdate_read"], write=pw["hupdate_write"],
        family="recurrent_scan", dtype="fp32",
        metadata=(("pcp", pcp), ("chunks", 0), ("chunk", GDN_CHUNK64_TOKENS),
                  ("tokens", 0), ("heads", heads),
                  ("source", "ops/triton/fla/chunk.py:fwd_hupdate")),
    )
    ag_state = b.communicate(
        "pcp_gdn_state_gather", (kv_transpose,),
        pw["st_bytes"] * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    ag_phi = b.communicate(
        "pcp_gdn_hupdate_gather", (hupdate,),
        pw["phi_bytes"] * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    prefix_steps = int(pw["prefix_steps"])
    prefix_scan = b.work(
        "pcp_gdn_prefix_scan", (ag_state, ag_phi), (pcp - 1, n_pf, heads, dk, dv),
        flops=pw["prefix_flops"], vector=pw["prefix_vector"],
        read=pw["prefix_read"], write=pw["prefix_write"],
        family="recurrent_scan", dtype="fp32",
        metadata=(("pcp", pcp), ("chunks", prefix_steps), ("chunk", 1),
                  ("tokens", prefix_steps), ("heads", heads),
                  ("source", "ops/triton/fla/chunk.py:prefix_scan")),
    )
    vk_transpose = b.work(
        "final_state_vk_transpose", (prefix_scan,), (n_pf, heads, dv, dk),
        read=active_elements * state_byte, write=active_elements * state_byte,
        family="tensor_copy", dtype=state_dtype,
        metadata=(("state_layout", "V,K"), ("source", "vllm_ascend/ops/gdn.py:942")),
    )
    scatter = b.work(
        "scatter_recurrent_state", (prefix_scan, vk_transpose), (n_pf, heads, dv, dk),
        read=active_elements * state_byte + n_pf * state_blocks * STATE_INDEX_BYTES,
        write=active_elements * state_byte, family="state_scatter", dtype=state_dtype,
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
                      if any(d not in idle_ids for d in op.deps) or not op.deps),
        anchor=scatter, block="attention",
        view={"tokens": 0, "heads": heads, "key_heads": heads,
              "key_width": dk, "value_width": dv,
              "dtype": model.activation_dtype, "state_dtype": plan.recurrent_state_dtype,
              "sequence_tokens": (), "chunk": GDN_CHUNK64_TOKENS, "pcp": pcp},
        implementation={"native_kind": "ChunkGatedDeltaRuleFwd",
                        "conditions": {"pcp": pcp, "idle_pcp_peer": True}},
        source=source_reference("vllm_ascend/ops/gdn.py",
                                "AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused"),
        fusion_status="fused_existing",
        fusion_reason="arch35_gdn_idle_pcp_peer_sync",
        native_fusion_eligible=False,
    )
    return prefix_scan


def _lower_idle_pcp_peer(
    b, tensors, *, projection, zero, prefill, model, plan, parallel,
    tokens, tp, heads, dk, dv, width, value_width,
):
    """Idle PCP peer owns 0 local tokens of a batch: join live prefill collectives only."""
    prefix_scan = _lower_idle_prefill_pcp_collectives(
        b, root_dep=zero, prefill=prefill, model=model, plan=plan,
        parallel=parallel, heads=heads, dk=dk, dv=dv, width=width,
        emit_conv_tail=True,
    )
    gate = tensors.reshape(projection, (tokens * heads, dv),
                           name="output_gate_reshape")
    gate = tensors.require(gate, "linear_output_norm_gate", "gate", name="output_gate_contiguous")
    norm_deps = (zero, gate.producer, prefix_scan) if prefix_scan is not None else (zero, gate.producer)
    normalized = b.vector(
        "linear_output_norm_gate", norm_deps,
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


def _lower_mixed_decode_branch(
    b, tensors, *, prefix, plan, model, ordered, decode, decode_tokens,
    qkv, gate_values, heads, keys, dk, dv, chunk, cache_indices, state_rows, index_source,
    runtime=None,
):
    decode_active = tuple(view for view in decode if not view.padding)
    decode_offsets, cursor = [], 0
    for view in decode:
        if not view.padding:
            decode_offsets.append(cursor)
        cursor += view.q_len
    decode_start_index = len(b.ops)
    decoded = decode_core(
        b, tensors, tuple(v.narrow(1, 0, decode_tokens) for v in qkv),
        tuple(v.narrow(1, 0, decode_tokens) for v in gate_values),
        decode_active, heads, keys, dk, dv,
        physical_tokens=decode_tokens,
        physical_sequence_tokens=tuple(view.q_len for view in decode),
        physical_row_offsets=tuple(decode_offsets))
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
    updated_scan = replace(scan, metadata=tuple({**scan_meta,
        "state_ready": scan.op_id,
        "state_value_inputs": decode_value_inputs,
        "state_value_outputs": decode_outputs,
        "gdn_state_contract": decode_contract,
        "gdn_state_effects": decode_effects,
        "padding_slots": decode_contract["padding_slots"],
        "recurrent_attention_kind": "gdn",
        "state_checkpoint_versions": tuple(
            None if count <= 0 else tuple(context + offset + 1 for offset in range(count))
            for context, count in zip(decode_contract["contexts"],
                                      decode_contract["checkpoints"]))}.items()))
    speculative_decode = any(int(view.q_len) > 1 for view in decode_active)
    use_recover_ssm = recover_ssm_enabled(runtime) and speculative_decode
    if use_recover_ssm:
        updated_scan = rewrite_gdn_analytical_scan(
            updated_scan, active_views=decode_active, heads=heads, dk=dk, dv=dv,
            state_dtype=plan.recurrent_state_dtype, kg_dtype=model.activation_dtype,
            attention_kind="gdn",
        )
    b.ops[-1] = updated_scan
    if use_recover_ssm:
        emit_commit_op(
            b, name="recover_ssm_commit", deps=(updated_scan.op_id,), views=decode_active,
            heads=heads, dk=dk, dv=dv, state_dtype=plan.recurrent_state_dtype,
            kg_dtype=model.activation_dtype, runtime=runtime, attention_kind="gdn",
        )
    dec_ops = [op for op in b.ops[decode_start_index:] if not dict(op.metadata).get("recover_ssm_intent")]
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
    return decoded


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
    keys, heads, dk, dv, key_replication = _resolve_local_gdn_heads(plan, tp)
    from mindie_sim.prediction.projections import gdn_input_projections

    width, value_width = 2 * keys * dk + heads * dv, heads * dv
    proj_qk_width = (2 * keys * dk) // key_replication
    proj_conv_width = proj_qk_width + value_width
    caller = (("caller_branch", "qkvz_ba_noninterleaved_non_spec"),)
    proj_meta = (
        (("gdn_key_head_replication", key_replication),
         ("checkpoint_qkvz_width", proj_conv_width + value_width))
        if key_replication > 1 else ()
    )
    conv_view, z_view, (beta_view, alpha_view) = gdn_input_projections(
        b, tensors, deps=deps, tokens=tokens, hidden=model.hidden_size,
        conv_width=proj_conv_width, value_width=value_width, heads=heads,
        metadata=caller + proj_meta,
    )
    if key_replication > 1:
        elem_bytes = dtype_bytes(model.activation_dtype)
        rep_bytes = (
            tokens * 2 * keys * dk * elem_bytes
            * (key_replication - 1) / key_replication
        )
        rep_id = b.communicate(
            "gdn_key_head_replicate", (conv_view.producer,), rep_bytes, "allgather",
            "local_tp",
            shape=(tokens, width),
            comm_group_size=key_replication,
            local_tp_width=key_replication,
            local_tp_stride=1,
            steps=key_replication - 1,
            gdn_key_head_replication=key_replication,
        )
        conv_view = tensors.result(rep_id, (tokens, width))
    gate_inputs = [tensors.contiguous(view, name + "_contiguous", reason="gdn_split_ba.contiguous")
                   for view, name in ((beta_view, "beta"), (alpha_view, "alpha"))]
    zero = b.work("core_output_zero", (deps,), (tokens, heads, dv),
                  vector=tokens * value_width, write=tokens * value_width * 2,
                  family="tensor_fill", dtype=model.activation_dtype)
    if actual == 0 and int(parallel.pcp) > 1:
        return _lower_idle_pcp_peer(
            b, tensors, projection=z_view, zero=zero, prefill=prefill,
            model=model, plan=plan, parallel=parallel, tokens=tokens, tp=tp,
            heads=heads, dk=dk, dv=dv, width=width, value_width=value_width,
        )
    conv_input = tensors.contiguous(conv_view.narrow(0, 0, actual),
                                    "convolution_input_contiguous",
                                    reason="causal_conv1d_fn.contiguous")
    padding_slots = tuple(index for index, view in enumerate(ordered) if view.padding)
    cache_indices, state_rows, index_source, null_block_id = _state_slot_inputs(
        runtime, ordered)
    chunk = policy.recurrent_chunk if policy else GDN_CHUNK64_TOKENS
    pcp_rank = (
        (int(physical_rank) % max(1, int(parallel.tp) * int(parallel.pcp))) // max(1, int(parallel.tp))
        if physical_rank is not None and int(parallel.pcp) > 1 else None
    )
    global_lengths = tuple(view.global_q_len or view.q_len for view in ordered)
    initial_flags = tuple(view.context > 0 and not view.padding for view in ordered)
    conv = _lower_prefill_convolution(
        b, tensors, parallel=parallel, ordered=ordered, prefill=prefill, decode=decode,
        conv_input=conv_input, padding_slots=padding_slots, initial_flags=initial_flags,
        global_lengths=global_lengths, cache_indices=cache_indices, state_rows=state_rows,
        index_source=index_source, null_block_id=null_block_id, runtime=runtime, chunk=chunk,
        heads=heads, width=width, actual=actual, pcp_rank=pcp_rank)
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
        outputs.append(_lower_mixed_decode_branch(
            b, tensors, prefix=prefix, plan=plan, model=model, ordered=ordered,
            decode=decode, decode_tokens=decode_tokens, qkv=qkv, gate_values=gate_values,
            heads=heads, keys=keys, dk=dk, dv=dv, chunk=chunk, cache_indices=cache_indices,
            state_rows=state_rows, index_source=index_source, runtime=runtime))
    idle_prefix_sync = None
    if prefill_tokens == 0 and int(parallel.pcp) > 1:
        idle_prefix_sync = _lower_idle_prefill_pcp_collectives(
            b, root_dep=conv, prefill=prefill, model=model, plan=plan,
            parallel=parallel, heads=heads, dk=dk, dv=dv, width=width,
            emit_conv_tail=False,
        )
    else:
        prefill_slots = (None if cache_indices is None else
                         tuple(cache_indices[len(decode) + index]
                               for index in range(len(prefill))))
        output = core(builder=b, tensors=tensors, conv=conv, gated=gated, views=prefill,
                      heads=heads, key_heads=keys, dk=dk, dv=dv, chunk=chunk,
                      state_slot_indices=prefill_slots, state_capacity=state_rows,
                      state_slot_source=index_source,
                      qkv_values=tuple(v.narrow(1, decode_tokens, prefill_tokens) for v in qkv),
                      gate_values=tuple(v.narrow(1, decode_tokens, prefill_tokens) for v in gate_values),
                      pcp=parallel.pcp, pcp_rank=pcp_rank)
        outputs.append(tensors.result(output, (prefill_tokens, heads, dv)))
    if len(outputs) > 1:
        flat = [tensors.reshape(value, (value.layout.elements,), name="mixed_output_view")
                for value in outputs]
        output_value = tensors.reshape(tensors.concatenate_flat(flat, "mixed_output_concatenate"),
                                        (actual, heads, dv), name="mixed_output_view")
    else:
        output_value = outputs[0]
    copied = tensors.copy_into(output_value,
                               tensors.result(zero, (tokens, heads, dv)).narrow(0, 0, actual),
                               "core_output_copy")
    gate = tensors.reshape(z_view, (tokens * heads, dv),
                           name="output_gate_reshape")
    gate = tensors.require(gate, "linear_output_norm_gate", "gate", name="output_gate_contiguous")
    norm_deps = (
        (copied.producer, gate.producer, idle_prefix_sync)
        if idle_prefix_sync is not None else (copied.producer, gate.producer)
    )
    normalized = b.vector(
        "linear_output_norm_gate", norm_deps, tokens * value_width, OUTPUT_NORM_GATE_OPS_PER_ELEMENT,
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
