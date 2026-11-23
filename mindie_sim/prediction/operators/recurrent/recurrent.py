"""Gated delta attention with finite state recurrence.

[Role] Lower GDN and KDA projections, local chunk math and state scans.
[Invariants] Each request owns its state; chunks carry a serial state edge.
  Key/value dimensions and head counts remain independent.
[Timeouts & Locks] One aggregate scan per request, lazily expanded at execution.
[Side-Effect Policy] Creates operators only.
"""
import math
from dataclasses import replace
from itertools import accumulate

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import dtype_bytes, ModelingError
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.prediction.operators.linear_attention import gdn_recurrent
from mindie_sim.prediction.operators.kda import kda_state_selection as pcp_state_selection
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.tensors import TensorLowering
from mindie_sim.prediction.structure_groups import declare_group, source_reference
from mindie_sim.prediction.implementations import select
from mindie_sim.prediction.projections import kda_projections, require_packed_slice
from .recover_ssm_work import emit_commit_op, record_head_bytes, recover_ssm_enabled
from mindie_sim.prediction.sharding.context_parallel import (
    emit_pcp_hidden_gather_restore,
    emit_pcp_restore_unpermute,
)
from mindie_sim.prediction.operators.kda.kda_source import serial_context_parallel, serving_source

DEFAULT_RECURRENT_CHUNK_TOKENS = 64
QKV_PROJECTION_INPUTS = 3
BETA_SIGMOID_OPS_PER_ELEMENT = 5
FP32_BYTES = 4
GDN_GATE_OPS_PER_HEAD = 12
KDA_GATE_OPS_PER_ELEMENT = 5
KDA_SCORE_GEMMS = 2
KDA_SCORE_GATE_OPS_PER_LANE = 3
GDN_SCORE_EPILOGUE_OPS_PER_PAIR = 4
KDA_SCORE_INPUT_TENSORS = 3
KDA_WY_SCALE_OPS_PER_ELEMENT = 4
KDA_WY_KEY_WIDTH_TENSORS = 3
PREFILL_SCAN_FLOPS_PER_STATE_ELEMENT = 4
DECODE_SCAN_FLOPS_PER_STATE_ELEMENT = 6
KDA_CHUNK_OPS_PER_STATE_ELEMENT = 6
GDN_DECODE_OPS_PER_STATE_ELEMENT = 7
KDA_OUTPUT_OPS_PER_PAIR = 6
KDA_OUTPUT_OPS_PER_VALUE_ELEMENT = 4
OUTPUT_NORM_GATE_OPS_PER_ELEMENT = 9

POWER_QUADRATIC = 2
PRODUCT_VECTOR_BYTES = 256
PRODUCT_READ_BYTES = 32
LINEAR_VECTOR_BYTES = 2
POWER_CUBIC = 3
TRIANGULAR_SOLVE_FLOP_COEFFICIENT = 2.0 / POWER_CUBIC
QK_INPUT_COUNT = 2
L2_NORM_OPS = 5
EAGER_NORM_CAST_OPS = 1
EAGER_NORM_POW_OPS = 1
EAGER_NORM_MEAN_OPS = 2
EAGER_NORM_ADD_OPS = 1
EAGER_NORM_RSQRT_OPS = 1
EAGER_NORM_MUL_OPS = 1
EAGER_NORM_SIGMOID_OPS = 4
EAGER_NORM_SILU_OPS = 5
EAGER_CONV_COPY_OPS = 1
EAGER_CONV_SILU_OPS = 5
EAGER_GATE_CAST_OPS = 1
EAGER_GATE_ADD_OPS = 1
EAGER_GATE_SOFTPLUS_OPS = 4
EAGER_GATE_MUL_OPS = 2
EAGER_GATE_SIGMOID_OPS = 4
EAGER_CONV_TORCH_SOURCE = "vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py:causal_conv1d_update_torch"
EAGER_GDN_GATING_SOURCE = "vllm_ascend/_310p/ops/fla/fused_gdn_gating.py:fused_gdn_gating_pytorch"


def _sum_valid_powers(tokens: int, chunk: int, power: int) -> int:
    """Closed form of sum(min(chunk, tokens - i*chunk)**power)."""
    full, remainder = divmod(int(tokens), int(chunk))
    return int(full) * int(chunk) ** int(power) + int(remainder) ** int(power)


def _lower_gdn_projections(b, tensors, *, deps, tokens, hidden, conv_width, value_width, heads):
    # Checkpoints can retain gate/output projections in native precision
    # while quantizing QKV. Role overrides inherit the existing gdn parent.
    from mindie_sim.prediction.projections import gdn_input_projections

    conv_view, output_gate_view, (beta_view, alpha_view) = gdn_input_projections(
        b, tensors, deps=deps, tokens=tokens, hidden=hidden,
        conv_width=conv_width, value_width=value_width, heads=heads,
    )
    conv_input = tensors.require(
        conv_view, "short_convolution", "input",
        name="convolution_input_contiguous")
    gate_inputs = [
        tensors.require(view, "gdn_gating", name,
                        name=f"gate_{name}_contiguous").producer
        for view, name in ((beta_view, "beta"), (alpha_view, "alpha"))
    ]
    return conv_input, gate_inputs, output_gate_view


def _lower_kda_projections(b, tensors, *, projection_source, conv_source, deps, tokens,
                           hidden, key_width, value_width, heads, dk):
    plan, model = b.plan, b.model
    projection_group = None
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
            sources = [incoming] * QKV_PROJECTION_INPUTS
        beta = require_packed_slice(b, tensors, beta, ("b",), "kda_recurrent", "beta",
                                    "kda_beta_contiguous")
        raw_beta = beta
        beta = b.vector(
            "linear_beta_sigmoid", (beta,), tokens * heads, BETA_SIGMOID_OPS_PER_ELEMENT,
            dtype="fp32", family="sigmoid", shape=(tokens, heads),
            read_bytes_override=tokens * heads * dtype_bytes(model.activation_dtype),
            write_bytes_override=tokens * heads * FP32_BYTES)
    return sources, output_gate, beta, decay, raw_beta, projection_group


def _lower_convolution(b, tensors, *, conv_source, projection_group, conv_input, sources,
                       views, tokens, heads, key_width, value_width, conv_width, parallel, runtime):
    prefix, plan = b.prefix, b.plan
    conv_deps = tuple(dict.fromkeys(sources))
    # GDN under PCP exchanges the pre-conv [n_pf, conv_width, kernel-1] boundary
    # tail via one pcp AllGather BEFORE short convolution; KDA under PCP operates
    # on gathered global sequences with tp*pcp head sharding.
    if (plan.attention == "gdn" and parallel.pcp > 1
            and any(view.phase == "prefill" for view in views)):
        n_pf = sum(1 for view in views if view.phase == "prefill")
        tail_bytes = (
            n_pf * max(0, plan.conv_kernel - 1) * conv_width
            * dtype_bytes(b.model.activation_dtype)
            * (parallel.pcp - 1)
        )
        tail_op = b.communicate(
            "pcp_conv_tail_gather", conv_deps, tail_bytes, "allgather", "pcp",
            dtype=b.model.activation_dtype, source="vllm_ascend.ops.gdn")
        conv_deps = (*conv_deps, tail_op)
    if conv_source:
        conv = conv_source.module.lower(
            b, tensors, plan=plan, views=views, tokens=tokens, sources=sources,
            key_width=key_width, value_width=value_width, **conv_source.options)
    else:
        active_views = tuple(view for view in views if not view.padding)
        active_tokens = sum(view.q_len for view in active_views)
        act_bytes = dtype_bytes(b.model.activation_dtype)
        history_rows = max(0, plan.conv_kernel - 1) * len(active_views)
        history_bytes = history_rows * conv_width * act_bytes
        eager_gdn_decode = (
            plan.attention == "gdn" and _is_raw_eager(runtime)
            and bool(views) and all(view.phase == "decode" for view in views)
        )
        if eager_gdn_decode:
            cat_rows = active_tokens + history_rows
            cat_bytes = cat_rows * conv_width * act_bytes
            conv_cat = b.vector(
                "conv_state_cat", conv_deps, max(1, cat_rows) * conv_width, EAGER_CONV_COPY_OPS,
                family="tensor_copy", shape=(max(1, cat_rows), conv_width),
                read_bytes_override=active_tokens * conv_width * act_bytes + history_bytes,
                write_bytes_override=cat_bytes,
                metadata=(("native_kind", "ConcatD"), ("native_source", EAGER_CONV_TORCH_SOURCE)))
            state_upd = b.vector(
                "conv_state_update", (conv_cat,), max(1, history_rows) * conv_width,
                EAGER_CONV_COPY_OPS, family="tensor_copy",
                shape=(max(1, history_rows), conv_width),
                read_bytes_override=history_bytes, write_bytes_override=history_bytes,
                metadata=(("native_kind", "InplaceCopy"), ("native_source", EAGER_CONV_TORCH_SOURCE)))
            conv = b.vector(
                "short_convolution", (conv_cat, state_upd),
                tokens * conv_width, QK_INPUT_COUNT * plan.conv_kernel,
                read_bytes_override=cat_bytes + plan.conv_kernel * conv_width * act_bytes,
                write_bytes_override=active_tokens * conv_width * act_bytes,
                metadata=(("native_kind", "Convolution"), ("native_source", EAGER_CONV_TORCH_SOURCE)))
        else:
            conv = b.vector("short_convolution", conv_deps,
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
        if eager_gdn_decode:
            out_bytes = tokens * conv_width * act_bytes
            conv = b.vector(
                "conv_silu", (conv,), tokens * conv_width, EAGER_CONV_SILU_OPS,
                family="elementwise", shape=(tokens, conv_width),
                read_bytes_override=out_bytes, write_bytes_override=out_bytes,
                metadata=(("native_kind", "Swish"), ("native_source", EAGER_CONV_TORCH_SOURCE)))
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
    return conv


def _lower_chunk_prepare(b, *, index, q, chunk, chunks, span, heads, dk, dv,
                         gated, normalized, prepare_ids):
    plan, model = b.plan, b.model
    pairs = _sum_valid_powers(q, chunk, POWER_QUADRATIC)
    # KDA forms QK and KK before WY with a dk-wide decay gate per head
    # (stored in activation_dtype from delta_gates, accumulated in FP32);
    # only Aqk @ V belongs after the scan.
    is_kda = plan.attention == "kda"
    act_bytes = dtype_bytes(model.activation_dtype)
    product_flops = 2 * (KDA_SCORE_GEMMS if is_kda else 1) * pairs * heads * dk
    prod_vec = pairs * heads * (KDA_SCORE_GATE_OPS_PER_LANE * dk if is_kda else GDN_SCORE_EPILOGUE_OPS_PER_PAIR)
    prod_read = q * heads * (KDA_SCORE_INPUT_TENSORS if is_kda else QK_INPUT_COUNT) * dk * act_bytes
    prod_write = pairs * heads * (KDA_SCORE_GEMMS * FP32_BYTES if is_kda else dtype_bytes("bf16"))
    local = b.work(f"chunk_products{index}", (gated, normalized), (q, heads, dk, dv),
                   flops=product_flops, vector=prod_vec,
                   read=prod_read, write=prod_write,
                   family="recurrent_local")
    last = b.work(f"triangular_solve{index}", (local,), (chunks, heads, span, span),
                  flops=TRIANGULAR_SOLVE_FLOP_COEFFICIENT * _sum_valid_powers(
                      q, chunk, POWER_CUBIC) * heads,
                  vector=pairs * heads * (PRODUCT_VECTOR_BYTES if is_kda else LINEAR_VECTOR_BYTES),
                  read=pairs * heads * (PRODUCT_READ_BYTES if is_kda else LINEAR_VECTOR_BYTES),
                  write=pairs * heads * (PRODUCT_READ_BYTES if is_kda else LINEAR_VECTOR_BYTES),
                  family="triangular_solve")
    prepare_ids.extend((local, last))
    wy_vec = q * heads * (2 * dk + dv) * KDA_WY_SCALE_OPS_PER_ELEMENT if is_kda else 0
    wy_width = (KDA_WY_KEY_WIDTH_TENSORS if is_kda else 1) * dk + dv
    wy_read = ((2 if is_kda else 1) * pairs + q * wy_width) * heads * act_bytes
    wy_write = q * heads * wy_width * act_bytes
    last = b.work(f"wy_transform{index}", (last,), (q, heads, dk, dv),
                  flops=2 * pairs * heads * (dk + dv), vector=wy_vec,
                  read=wy_read, write=wy_write, family="recurrent_local")
    return pairs, last


def _emit_gdn_pcp_prefix_scan(b, *, index, view, dep, heads, dk, dv, pcp, chunk):
    """Exact ``ops/triton/fla/chunk.py`` GDN PCP: fwd_hupdate + 2 FP32 AGs + prefix scan."""
    q_local = max(1, int(view.q_len))
    scan_chunk = max(1, int(chunk))
    chunks = max(1, math.ceil(q_local / scan_chunk))
    scan_heads = max(1, int(heads))
    phi_bytes = float(scan_heads * dk * dk * FP32_BYTES)
    state_bytes = float(scan_heads * dk * dv * FP32_BYTES)
    hupdate = b.work(
        f"pcp_gdn_hupdate{index}", (dep,), (1, scan_heads, dk, dk),
        flops=float(2 * q_local * scan_heads * dk * dk),
        vector=float(chunks * scan_heads * dk * dk),
        read=phi_bytes + float(q_local * scan_heads * dk * 2),
        write=phi_bytes, family="recurrent_scan", dtype="fp32",
        metadata=(
            ("pcp", int(pcp)), ("chunks", chunks), ("chunk", scan_chunk),
            ("tokens", q_local), ("heads", scan_heads),
            ("source", "ops/triton/fla/chunk.py:fwd_hupdate"),
        ),
    )
    ag_state = b.communicate(
        f"pcp_gdn_state_gather{index}", (dep,),
        state_bytes * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    ag_phi = b.communicate(
        f"pcp_gdn_hupdate_gather{index}", (hupdate,),
        phi_bytes * (pcp - 1), "allgather", "pcp", dtype="fp32",
    )
    steps = max(1, int(pcp) - 1)
    prefix = b.work(
        f"pcp_gdn_prefix_scan{index}", (ag_state, ag_phi), (steps, scan_heads, dk, dv),
        flops=float(2 * steps * scan_heads * dk * dk * dv),
        vector=float(steps * scan_heads * dk * dv),
        read=(state_bytes + phi_bytes) * pcp, write=state_bytes,
        family="recurrent_scan", dtype="fp32",
        metadata=(
            ("pcp", int(pcp)), ("chunks", steps), ("chunk", 1),
            ("tokens", steps), ("heads", scan_heads),
            ("source", "ops/triton/fla/chunk.py:prefix_scan"),
        ),
    )
    rerun = b.work(
        f"pcp_gdn_fwd_h_rerun{index}", (prefix,), (q_local, scan_heads, dk, dv),
        flops=float(PREFILL_SCAN_FLOPS_PER_STATE_ELEMENT * q_local * scan_heads * dk * dv),
        vector=float(chunks * 2 * scan_heads * dk * dv),
        read=chunks * state_bytes, write=chunks * state_bytes,
        family="recurrent_scan", dtype="fp32",
        metadata=(
            ("pcp", int(pcp)), ("chunks", chunks), ("chunk", scan_chunk),
            ("tokens", q_local), ("heads", scan_heads),
            ("pcp_rank_gt_zero_rerun", True),
        ),
    )
    return rerun


def _lower_segmented_gdn_scan(b, *, view, index, views, last, normalized, chunk,
                              heads, dk, dv, state_bytes, value_width, parallel):
    prefix, plan = b.prefix, b.plan
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
            (length, heads, dk, dv), flops=PREFILL_SCAN_FLOPS_PER_STATE_ELEMENT * length * heads * dk * dv,
            vector=seg_vec, read=seg_read, write=seg_write, family="recurrent_scan",
            metadata=(("chunk", chunk), ("chunks", seg_chunks), ("tokens", length),
                      ("heads", heads), ("dk", dk), ("dv", dv),
                      ("chunk_vector_ops", float(seg_chunks * 2 * heads * dk * dv)),
                      ("token_vector_ops", float(length * heads * (2 + 2 * dv))),
                      ("pcp_segment", (int(start), int(end)))))
        segment_view = replace(view, q_len=length, context=version,
                               global_q_len=length, query_ranges=((0, length),))
        direction = "conv" if range_index == len(view.query_ranges) - 1 else "h"
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
                version += gap
        segment_index += 1
    if previous is not None and parallel.pcp > 1:
        previous = _emit_gdn_pcp_prefix_scan(
            b, index=index, view=view, dep=previous, heads=heads, dk=dk, dv=dv,
            pcp=parallel.pcp, chunk=chunk)
    return previous


def _lower_recurrent_scan(b, *, view, index, q, chunks, span, chunk, heads, dk, dv,
                          last, normalized, state_bytes, value_width, parallel, is_kda_prefill,
                          use_recover_ssm, rec_bytes):
    prefix, plan = b.prefix, b.plan
    spec_dec = view.phase == "decode" and use_recover_ssm and q > 1
    chunk_vec = chunks * (KDA_CHUNK_OPS_PER_STATE_ELEMENT if is_kda_prefill else 2) * heads * dk * dv
    token_vec = q * heads * (dk * dv if is_kda_prefill else 2 + 2 * dv)
    scan_vec = token_vec + chunk_vec
    scan_read = (
        2 * chunks * state_bytes + q * heads * ((2 * dk + dv) * 2 + dk * FP32_BYTES)
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
        flops=(PREFILL_SCAN_FLOPS_PER_STATE_ELEMENT if view.phase == "prefill"
               else DECODE_SCAN_FLOPS_PER_STATE_ELEMENT) * q * heads * dk * dv,
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
            last = _emit_gdn_pcp_prefix_scan(
                b, index=index, view=view, dep=last, heads=heads, dk=dk, dv=dv,
                pcp=parallel.pcp, chunk=chunk)
    return last, state_bytes


def _lower_gdn_decode_scan(b, *, view, index, offset, q, conv, gated, normalized,
                           heads, key_heads, dk, dv, state_bytes, key_width, value_width,
                           conv_width, tokens, use_recover_ssm, rec_bytes):
    prefix, plan, model = b.prefix, b.plan, b.model
    spec_gdn = use_recover_ssm and q > 1
    gdn_write = q * ((rec_bytes if spec_gdn else state_bytes) + value_width * 2)
    gdn_extra = (
        (("recover_ssm", True), ("recover_ssm_stage", "verify_compact_record"),
         ("record_bytes_per_token", rec_bytes))
        if spec_gdn else ()
    )
    last = b.work(
        f"recurrent_scan{index}", (gated, normalized), (q, heads, dk, dv),
        vector=GDN_DECODE_OPS_PER_STATE_ELEMENT * q * heads * dk * dv,
        read=state_bytes + q * heads * ((2 * dk + dv) * 2 + FP32_BYTES + dtype_bytes("bf16")),
        write=gdn_write, family="recurrent_vector",
        metadata=(("tokens", q), ("heads", heads), ("dk", dk), ("dv", dv),
                  ("sequence_tokens", (q,)), ("key_heads", key_heads),
                  ("state_dtype", plan.recurrent_state_dtype),
                  ("state_element_bytes", dtype_bytes(plan.recurrent_state_dtype)),
                  ("state_checkpoints", 0 if spec_gdn else q)) + gdn_extra,
    )
    if plan.attention == "gdn":
        activation = model.activation_dtype
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
    return last


def _lower_recurrent_views(b, *, views, chunk, heads, key_heads, dk, dv, key_width,
                           value_width, conv_width, tokens, conv, normalized, gated,
                           parallel, use_recover_ssm, rec_bytes):
    plan = b.plan
    outputs, prefill_scan_ids, prepare_ids = [], [], []
    view_offsets = tuple(accumulate((view.q_len for view in views), initial=0))
    for index, view in enumerate(views):
        q = view.q_len
        chunks = math.ceil(q / chunk) if view.phase == "prefill" else q
        span = chunk if view.phase == "prefill" else 1
        last = gated
        if view.phase == "prefill":
            pairs, last = _lower_chunk_prepare(
                b, index=index, q=q, chunk=chunk, chunks=chunks, span=span, heads=heads,
                dk=dk, dv=dv, gated=gated, normalized=normalized, prepare_ids=prepare_ids)
        state_bytes = heads * dk * dv * dtype_bytes(plan.recurrent_state_dtype)
        pcp_ranges = (plan.attention == "gdn" and view.phase == "prefill"
                      and parallel.pcp > 1 and len(view.query_ranges) > 1)
        if view.phase == "prefill" or plan.attention != "gdn":
            is_kda_prefill = view.phase == "prefill" and plan.attention == "kda"
            if pcp_ranges:
                last = _lower_segmented_gdn_scan(
                    b, view=view, index=index, views=views, last=last, normalized=normalized,
                    chunk=chunk, heads=heads, dk=dk, dv=dv, state_bytes=state_bytes,
                    value_width=value_width, parallel=parallel)
            else:
                last, state_bytes = _lower_recurrent_scan(
                    b, view=view, index=index, q=q, chunks=chunks, span=span, chunk=chunk,
                    heads=heads, dk=dk, dv=dv, last=last, normalized=normalized,
                    state_bytes=state_bytes, value_width=value_width, parallel=parallel,
                    is_kda_prefill=is_kda_prefill, use_recover_ssm=use_recover_ssm, rec_bytes=rec_bytes)
            if view.phase == "prefill":
                prefill_scan_ids.append(last)
                if parallel.pcp > 1 and not pcp_ranges and plan.attention not in {"gdn", "kda"}:
                    # Non-head-sharded recurrent families keep the single post-scan handoff.
                    state_bytes = heads * dk * dv * dtype_bytes(plan.recurrent_state_dtype or "fp32")
                    last = b.communicate(
                        f"pcp_recurrent_state_step{index}", (last,),
                        state_bytes * (2 * parallel.pcp - 1) / parallel.pcp, "state_transfer", "pcp",
                        remote_producer="external_rank", source_implementation="pcp1_only",
                        status="pending_external_rank")
        else:
            last = _lower_gdn_decode_scan(
                b, view=view, index=index, offset=view_offsets[index], q=q,
                conv=conv, gated=gated, normalized=normalized, heads=heads, key_heads=key_heads,
                dk=dk, dv=dv, state_bytes=state_bytes, key_width=key_width, value_width=value_width,
                conv_width=conv_width, tokens=tokens, use_recover_ssm=use_recover_ssm,
                rec_bytes=rec_bytes)
        if view.phase == "prefill":
            is_kda = plan.attention == "kda"
            pair_width = dv if is_kda else dk + dv
            out_vec = pairs * heads * KDA_OUTPUT_OPS_PER_PAIR + q * heads * dv * KDA_OUTPUT_OPS_PER_VALUE_ELEMENT if is_kda else pairs * heads
            out_read = (chunks * state_bytes + (pairs + q * (2 * dk + dv)) * heads * 2
                        if is_kda else chunks * state_bytes + q * heads * (2 * dk + dv) * 2)
            last = b.work(f"recurrent_output{index}", (last, normalized), (q, heads, dk, dv),
                          flops=2 * q * heads * dk * dv + 2 * pairs * heads * pair_width,
                          vector=out_vec, read=out_read,
                          write=q * value_width * 2, family="recurrent_output")
        outputs.append(last)
    return outputs, prefill_scan_ids, prepare_ids


def _bind_kda_decode(b, *, decode_source, views, parallel, projection_group, beta, raw_beta,
                     use_recover_ssm, heads, key_heads, dk, dv, conv, decay, normalized, gated, outputs):
    plan, model = b.plan, b.model
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
            # The mixed input-projection caller materializes the FP32 beta
            # activation itself; the recurrent kernel consumes that producer
            # instead of applying a second in-kernel sigmoid.
            internal_beta = False
            options = {**options, "use_beta_sigmoid": False, "beta_dtype": "fp32"}
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
    return decode_core, outputs


def _bind_kda_prefill(b, *, kda_source, kda_caps, prefill_source, prefill_scan_ids, prepare_ids,
                      views, gated, outputs, heads, key_heads, dk, dv, chunk, parallel):
    plan, model = b.plan, b.model
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
    return gated, outputs


def _fuse_gdn_decode_scans(b, outputs, *, heads, dk, dv):
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
    return outputs


def _is_raw_eager(runtime) -> bool:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    return bool(
        meta.get("raw_eager")
        or (meta.get("execution_policy") or {}).get("raw_eager")
    )


def _lower_output_norm_gate(b, *, outputs, output_gate, tokens, heads, dv, value_width,
                            plan, model, projection_group, runtime):
    """Emit 1 fused Triton gated-RMSNorm kernel or its 11-op PyTorch eager chain."""
    act_dtype = model.activation_dtype
    act_bytes = dtype_bytes(act_dtype)
    elems = tokens * value_width
    rows = tokens * heads
    shape = (rows, dv)
    if _is_raw_eager(runtime):
        fp32_full = elems * FP32_BYTES
        fp32_row = rows * FP32_BYTES
        act_full = elems * act_bytes
        x_f32 = b.vector(
            "linear_output_norm_cast_in", outputs, elems, EAGER_NORM_CAST_OPS,
            family="cast", dtype="fp32", shape=shape,
            read_bytes_override=act_full, write_bytes_override=fp32_full,
            metadata=(("input_format", act_dtype), ("output_format", "fp32"),
                      ("eager_norm_gate_step", "cast_x")))
        sq = b.vector(
            "linear_output_norm_pow", (x_f32,), elems, EAGER_NORM_POW_OPS,
            family="elementwise", dtype="fp32", shape=shape,
            read_bytes_override=fp32_full, write_bytes_override=fp32_full,
            metadata=(("native_kind", "Pows"), ("eager_norm_gate_step", "pow2")))
        mean = b.vector(
            "linear_output_norm_mean", (sq,), elems, EAGER_NORM_MEAN_OPS,
            family="elementwise", dtype="fp32", shape=(rows, 1),
            read_bytes_override=fp32_full, write_bytes_override=fp32_row,
            metadata=(("native_kind", "ReduceMean"), ("eager_norm_gate_step", "mean")))
        eps = b.vector(
            "linear_output_norm_add_eps", (mean,), rows, EAGER_NORM_ADD_OPS,
            family="elementwise", dtype="fp32", shape=(rows, 1),
            read_bytes_override=fp32_row, write_bytes_override=fp32_row,
            metadata=(("native_kind", "Adds"), ("eager_norm_gate_step", "add_eps")))
        rsq = b.vector(
            "linear_output_norm_rsqrt", (eps,), rows, EAGER_NORM_RSQRT_OPS,
            family="elementwise", dtype="fp32", shape=(rows, 1),
            read_bytes_override=fp32_row, write_bytes_override=fp32_row,
            metadata=(("native_kind", "Rsqrt"), ("eager_norm_gate_step", "rsqrt")))
        normed = b.vector(
            "linear_output_norm_mul", (x_f32, rsq), elems, EAGER_NORM_MUL_OPS,
            family="elementwise", dtype="fp32", shape=shape,
            read_bytes_override=fp32_full + fp32_row, write_bytes_override=fp32_full,
            metadata=(("native_kind", "Mul"), ("eager_norm_gate_step", "norm_mul")))
        weighted = b.vector(
            "linear_output_norm_weight_mul", (normed,), elems, EAGER_NORM_MUL_OPS,
            family="elementwise", dtype="fp32", shape=shape,
            read_bytes_override=fp32_full + dv * FP32_BYTES, write_bytes_override=fp32_full,
            metadata=(("native_kind", "Mul"), ("eager_norm_gate_step", "weight_mul")))
        g_f32 = b.vector(
            "linear_output_gate_cast_in", (output_gate,), elems, EAGER_NORM_CAST_OPS,
            family="cast", dtype="fp32", shape=shape,
            read_bytes_override=act_full, write_bytes_override=fp32_full,
            metadata=(("input_format", act_dtype), ("output_format", "fp32"),
                      ("eager_norm_gate_step", "cast_gate")))
        is_gdn = plan.attention == "gdn"
        sig = b.vector(
            "linear_output_gate_silu" if is_gdn else "linear_output_gate_sigmoid",
            (g_f32,), elems,
            EAGER_NORM_SILU_OPS if is_gdn else EAGER_NORM_SIGMOID_OPS,
            family="elementwise", dtype="fp32", shape=shape,
            read_bytes_override=fp32_full, write_bytes_override=fp32_full,
            metadata=(
                ("native_kind", "Swish" if is_gdn else "Sigmoid"),
                ("eager_norm_gate_step", "silu" if is_gdn else "sigmoid"),
            ))
        gated = b.vector(
            "linear_output_gate_mul", (weighted, sig), elems, EAGER_NORM_MUL_OPS,
            family="elementwise", dtype="fp32", shape=shape,
            read_bytes_override=2 * fp32_full, write_bytes_override=fp32_full,
            metadata=(("native_kind", "Mul"), ("eager_norm_gate_step", "gate_mul")))
        cast_out = b.vector(
            "linear_output_norm_cast_out", (gated,), elems, EAGER_NORM_CAST_OPS,
            family="cast", dtype=act_dtype, shape=shape,
            read_bytes_override=fp32_full, write_bytes_override=act_full,
            metadata=(("input_format", "fp32"), ("output_format", act_dtype),
                      ("eager_norm_gate_step", "cast_out")))
        chain = (x_f32, sq, mean, eps, rsq, normed, weighted, g_f32, sig, gated, cast_out)
        native_kind = ("layer_norm_gated_fwd_kernel" if plan.attention == "kda"
                       else "_layer_norm_fwd_1pass_kernel_npu")
        declare_group(
            b.ops, key=b.prefix + ".gated_rmsnorm", role="gated_rmsnorm",
            family="norm_gate", members=chain, entries=(x_f32, g_f32),
            anchor=cast_out, block="attention",
            view={"tokens": tokens, "heads": heads, "width": dv,
                  "dtype": act_dtype, "attention_kind": plan.attention},
            implementation={"native_kind": native_kind,
                            "native_kinds": (native_kind, "FusedRMSNormGated")},
            source=source_reference(
                "vllm_ascend/ops/kimi_kda.py" if plan.attention == "kda"
                else "vllm_ascend/ops/gdn.py", "FusedRMSNormGated"),
            fusion_status="unfused_decomposed",
            fusion_reason="torch_npu_eager_rmsnorm_gated_11_kernels",
            native_fusion_eligible=False,
        )
        return cast_out
    output = b.vector(
        "linear_output_norm_gate", outputs + [output_gate], elems, OUTPUT_NORM_GATE_OPS_PER_ELEMENT,
        shape=shape,
        read_bytes_override=(2 * elems + dv) * act_bytes,
        write_bytes_override=elems * act_bytes,
        metadata=(("norm_kind", "rmsnorm"), ("norm_has_weight", True),
                  ("norm_has_bias", False), ("norm_group_width", dv),
                  ("gate_present", True), ("recurrent_attention_kind", plan.attention),
                  ("fused_from", ("rmsnorm", "gate"))) + (
                      (("gate_input_layout", projection_group["views"]["output_gate"]),)
                      if projection_group else ()))
    native_kind = ("layer_norm_gated_fwd_kernel" if plan.attention == "kda"
                   else "_layer_norm_fwd_1pass_kernel_npu")
    declare_group(
        b.ops, key=b.prefix + ".gated_rmsnorm", role="gated_rmsnorm",
        family="norm_gate", members=(output,), entries=(output,),
        anchor=output, block="attention",
        view={"tokens": tokens, "heads": heads, "width": dv,
              "dtype": act_dtype, "attention_kind": plan.attention},
        implementation={"native_kind": native_kind,
                        "native_kinds": (native_kind, "FusedRMSNormGated")},
        source=source_reference(
            "vllm_ascend/ops/kimi_kda.py" if plan.attention == "kda"
            else "vllm_ascend/ops/gdn.py", "FusedRMSNormGated"),
        fusion_status="fused_existing",
        fusion_reason="triton_fused_rmsnorm_gated_kernel",
        native_fusion_eligible=False,
    )
    return output


def _declare_recurrent_group(b, tensors, *, views, heads, key_heads, dk, dv, chunk,
                             tokens, output_gate, output_gate_view, prefill_source, decode_source,
                             decode_core, kda_caps, gated, normalized,
                             prefill_scan_ids=(), prepare_ids=()):
    prefix, plan, model = b.prefix, b.plan, b.model
    if plan.attention == "gdn":
        core = [op for op in b.ops if op.family in {
            "recurrent_local", "triangular_solve", "recurrent_scan",
            "recurrent_output", "recurrent_vector"}]
        has_prefill = any(view.phase == "prefill" for view in views)
        has_decode = any(view.phase == "decode" for view in views)
        prefill_ids = set(prefill_scan_ids) | set(prepare_ids)
        if has_prefill and has_decode:
            prefill_core = [op for op in core
                            if op.op_id in prefill_ids or op.family in {
                                "recurrent_local", "triangular_solve", "recurrent_output"}]
            prefill_set = {op.op_id for op in prefill_core}
            decode_core_ops = [op for op in core if op.op_id not in prefill_set]
        elif has_prefill:
            prefill_core, decode_core_ops = core, []
        else:
            prefill_core, decode_core_ops = [], core
        if prefill_core:
            pf_ids = {op.op_id for op in prefill_core}
            pf_views = tuple(view for view in views if view.phase == "prefill") or tuple(views)
            declare_group(
                b.ops, key=prefix + ".gdn_chunk", role="gdn_chunk", family="gdn",
                members=tuple(op.op_id for op in prefill_core),
                entries=tuple(op.op_id for op in prefill_core
                              if any(dep not in pf_ids for dep in op.deps) or not op.deps),
                anchor=prefill_core[-1].op_id,
                view={"tokens": sum(view.q_len for view in pf_views), "heads": heads,
                      "key_heads": key_heads, "key_width": dk, "value_width": dv,
                      "dtype": model.activation_dtype, "state_dtype": plan.recurrent_state_dtype,
                      "sequence_tokens": tuple(view.q_len for view in pf_views), "chunk": chunk},
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
                fusion_status="unfused_decomposed",
                fusion_reason="analytical_gdn_chunk_decomposition",
            )
        if decode_core_ops:
            dec_ids = {op.op_id for op in decode_core_ops}
            dec_views = tuple(view for view in views if view.phase == "decode") or tuple(views)
            declare_group(
                b.ops, key=prefix + ".gdn_recurrent", role="gdn_recurrent", family="gdn",
                members=tuple(op.op_id for op in decode_core_ops),
                entries=tuple(op.op_id for op in decode_core_ops
                              if any(dep not in dec_ids for dep in op.deps) or not op.deps),
                anchor=decode_core_ops[-1].op_id,
                view={"tokens": sum(view.q_len for view in dec_views), "heads": heads,
                      "key_heads": key_heads, "key_width": dk, "value_width": dv,
                      "dtype": model.activation_dtype, "state_dtype": plan.recurrent_state_dtype,
                      "sequence_tokens": tuple(view.q_len for view in dec_views), "chunk": chunk},
                implementation={
                    "native_kind": "RecurrentGatedDeltaRule",
                    "selection": "compatible_recurrent_algorithm; native_kernel_not_selected",
                    "conditions": {"qk_l2norm": "external", "delta_gates": "external",
                                   "state_dtype": plan.recurrent_state_dtype}},
                source=source_reference("vllm_ascend/ops/gdn.py",
                                        "AscendGatedDeltaNetAttention._recurrent_gated_delta_rule"),
                fusion_status="unfused_decomposed",
                fusion_reason="analytical_gdn_decode_recurrence",
                native_fusion_eligible=False,
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
                fusion_status=("fused_existing" if decode_core is not None
                               else "fused_framework_enable" if phase_source
                               else "unfused_decomposed"),
                fusion_reason=("source_selected_kda_recurrent_kernel" if decode_core is not None
                               else "source_selected_kda_chunk_kernel" if phase_source
                               else "analytical_kda_decomposition"),
            )
    return output_gate


def _kda_pcp_enabled(plan, parallel, runtime):
    if plan.attention != "kda" or int(parallel.pcp) <= 1:
        return False
    axis = str(dict(getattr(runtime, "metadata", ()) or ()).get(
        "kda_parallel_axis", "")).strip().lower()
    return axis != "tp"


def _apply_kda_pcp_entry(b, *, deps, views, tokens, hidden, parallel, model, runtime):
    """Gather hidden states for head-sharded KDA PCP.

    GLM5Next (``kda_batch_chunk_norm=True``) gathers/restores hidden states only
    when prefill tokens are present; Kimi KDA (``ops/kimi_kda.py``) gathers,
    restores/zero-masks padding, and scatters unconditionally across both prefill
    and decode when ``pcp > 1``.
    """
    from mindie_sim.prediction.sharding.context_parallel import pcp_uniform_peer_rows
    pcp = int(parallel.pcp)
    has_prefill = any(v.phase == "prefill" for v in views)
    uses_pcp_row_parallel = bool(b.plan.kda_batch_chunk_norm)
    if uses_pcp_row_parallel and not has_prefill:
        return deps, views, tokens, tokens, False
    global_views = tuple(
        replace(v, q_len=int(v.global_q_len or v.q_len),
                query_ranges=((0, int(v.global_q_len or v.q_len)),))
        if v.phase == "prefill" else v
        for v in views
    )
    if has_prefill:
        common_peer_rows, global_tokens = pcp_uniform_peer_rows(views, pcp, local_rows=0)
        common_peer_rows = max(1, common_peer_rows)
        global_tokens = max(1, global_tokens)
    else:
        common_peer_rows = max(1, int(tokens))
        global_tokens = common_peer_rows * pcp
    # Kimi KDA (``ops/kda_pcp.py:54,108-109``) retains all ``pcp * common_peer_rows``
    # rows (with padding zero-masked) through projection and AllReduce, slicing back
    # only at ``kda_pcp_scatter``; GLM row-parallel KDA compacts to ``global_tokens``.
    restored_rows = global_tokens if uses_pcp_row_parallel else (common_peer_rows * pcp)
    kimi_mask_meta = (() if uses_pcp_row_parallel
                      else (("pcp_zero_mask_padding", True),
                            ("pcp_restore_mask", True)))
    restored = emit_pcp_hidden_gather_restore(
        b.ops, prefix=b.prefix,
        gather_name="kda_pcp_gather", restore_name="kda_pcp_restore",
        stage=b.common["stage"], layer=b.plan.index,
        requests=b.common["requests"], mode=b.common["phase"],
        dep=deps, local_rows=common_peer_rows, global_rows=restored_rows,
        hidden=hidden, pcp=pcp, act_dtype=model.activation_dtype,
        runtime=runtime, source="vllm_ascend.ops.kda_pcp",
        restore=True, per_peer_rows=common_peer_rows,
        restore_meta=kimi_mask_meta,
    )
    return restored, global_views, restored_rows, common_peer_rows, True


def _lower_delta_gates(b, *, plan, model, runtime, tokens, heads, key_heads, dk,
                       gate_inputs, decay, beta):
    """Emit fused GDN/KDA delta gating or the 7-kernel PyTorch eager GDN chain."""
    if plan.attention == "gdn":
        deps = tuple(dict.fromkeys(gate_inputs))
        if _is_raw_eager(runtime):
            act_dtype = model.activation_dtype
            act_bytes = dtype_bytes(act_dtype)
            elems = tokens * heads
            act_total = elems * act_bytes
            fp32_total = elems * FP32_BYTES
            head_param = heads * FP32_BYTES
            alpha_dep = (gate_inputs[-1],) if gate_inputs else deps
            beta_dep = (gate_inputs[0],) if gate_inputs else deps
            a_f32 = b.vector(
                "gate_alpha_cast_fp32", alpha_dep, elems, EAGER_GATE_CAST_OPS,
                family="cast", dtype="fp32", shape=(tokens, heads),
                read_bytes_override=act_total, write_bytes_override=fp32_total,
                metadata=(("input_format", act_dtype), ("output_format", "fp32"),
                          ("native_source", EAGER_GDN_GATING_SOURCE)))
            a_bias = b.vector(
                "gate_alpha_add_bias", (a_f32,), elems, EAGER_GATE_ADD_OPS,
                family="elementwise", dtype="fp32", shape=(tokens, heads),
                read_bytes_override=fp32_total + head_param, write_bytes_override=fp32_total,
                metadata=(("native_kind", "Add"), ("native_source", EAGER_GDN_GATING_SOURCE)))
            a_sp = b.vector(
                "gate_alpha_softplus", (a_bias,), elems, EAGER_GATE_SOFTPLUS_OPS,
                family="elementwise", dtype="fp32", shape=(tokens, heads),
                read_bytes_override=fp32_total, write_bytes_override=fp32_total,
                metadata=(("native_kind", "Softplus"), ("native_source", EAGER_GDN_GATING_SOURCE)))
            b_f32 = b.vector(
                "gate_beta_cast_fp32", beta_dep, elems, EAGER_GATE_CAST_OPS,
                family="cast", dtype="fp32", shape=(tokens, heads),
                read_bytes_override=act_total, write_bytes_override=fp32_total,
                metadata=(("input_format", act_dtype), ("output_format", "fp32"),
                          ("native_source", EAGER_GDN_GATING_SOURCE)))
            b_sig = b.vector(
                "gate_beta_sigmoid", (b_f32,), elems, EAGER_GATE_SIGMOID_OPS,
                family="elementwise", dtype="fp32", shape=(tokens, heads),
                read_bytes_override=fp32_total, write_bytes_override=fp32_total,
                metadata=(("native_kind", "Sigmoid"), ("native_source", EAGER_GDN_GATING_SOURCE)))
            b_out = b.vector(
                "gate_beta_cast_out", (b_sig,), elems, EAGER_GATE_CAST_OPS,
                family="cast", dtype=act_dtype, shape=(tokens, heads),
                read_bytes_override=fp32_total, write_bytes_override=act_total,
                metadata=(("input_format", "fp32"), ("output_format", act_dtype),
                          ("native_source", EAGER_GDN_GATING_SOURCE)))
            alog_exp = b.vector(
                "gate_alog_exp", alpha_dep, heads, EAGER_GATE_SOFTPLUS_OPS,
                family="elementwise", dtype="fp32", shape=(heads,),
                read_bytes_override=head_param, write_bytes_override=head_param,
                metadata=(("native_kind", "Exp"), ("native_source", EAGER_GDN_GATING_SOURCE)))
            alog_neg = b.vector(
                "gate_alog_neg", (alog_exp,), heads, EAGER_GATE_CAST_OPS,
                family="elementwise", dtype="fp32", shape=(heads,),
                read_bytes_override=head_param, write_bytes_override=head_param,
                metadata=(("native_kind", "Neg"), ("native_source", EAGER_GDN_GATING_SOURCE)))
            return b.vector(
                "delta_gates", (a_sp, alog_neg, b_out), elems, EAGER_GATE_MUL_OPS,
                family="gdn_gating", dtype="fp32", shape=(tokens, heads),
                read_bytes_override=fp32_total + head_param, write_bytes_override=fp32_total,
                metadata=(("outputs", ("fp32_decay", "activation_beta")),
                          ("input_heads", key_heads), ("state_heads", heads),
                          ("native_kind", "Mul"), ("native_source", EAGER_GDN_GATING_SOURCE)),
            )
        return b.vector(
            "delta_gates", deps, tokens * heads, GDN_GATE_OPS_PER_HEAD, family="gdn_gating",
            shape=(tokens, heads), read_bytes_override=(tokens + QK_INPUT_COUNT) * heads * FP32_BYTES,
            write_bytes_override=tokens * heads * (FP32_BYTES + dtype_bytes("bf16")),
            metadata=(("outputs", ("fp32_decay", "activation_beta")),
                      ("input_heads", key_heads), ("state_heads", heads)),
        )
    return b.vector(
        "delta_gates", (decay, beta), tokens * (heads * dk + heads), KDA_GATE_OPS_PER_ELEMENT,
        shape=(tokens, heads, dk),
        read_bytes_override=tokens * heads * dk * dtype_bytes(model.activation_dtype)
                            + (tokens * heads + heads + heads * dk) * FP32_BYTES,
        metadata=(("A_log_dtype", "fp32"), ("dt_bias_dtype", "fp32"),
                  ("beta_dtype", "fp32"), ("raw_gate_dtype", model.activation_dtype)))


def _lower_kda_pcp_exit(b, *, last, plan, model, parallel, views, tokens, local_tokens,
                        hidden, kda_pcp, kda_gathered, kda_common_peer, tp):
    act_bytes = dtype_bytes(model.activation_dtype)
    pure_prefill_rs = (
        kda_pcp and kda_gathered and bool(plan.kda_batch_chunk_norm)
        and bool(views) and all(v.phase == "prefill" for v in views)
    )
    if kda_pcp:
        pcp = int(parallel.pcp)
        if pure_prefill_rs:
            last = b.communicate(
                "kda_pcp_reduce_scatter", (last,),
                float(kda_common_peer * hidden) * act_bytes * float(pcp - 1),
                "reduce_scatter", "pcp", dtype=model.activation_dtype,
                shape=(kda_common_peer, hidden),
                source="vllm_ascend.models.glm5next.kda")
        else:
            last = b.communicate(
                "kda_pcp_allreduce", (last,),
                QK_INPUT_COUNT * (pcp - 1) / pcp * tokens * hidden * act_bytes,
                "allreduce", "pcp", dtype=model.activation_dtype,
                shape=(tokens, hidden),
                source="vllm_ascend.ops.kda_parallel")
    if kda_gathered and not pure_prefill_rs:
        scatter_meta = (() if bool(plan.kda_batch_chunk_norm)
                        else (("pcp_zero_mask_padding", True),
                              ("pcp_local_mask", True)))
        last = emit_pcp_restore_unpermute(
            b.ops, prefix=b.prefix, name="kda_pcp_scatter",
            stage=b.common["stage"], layer=plan.index,
            requests=b.common["requests"], mode=b.common["phase"],
            dep=last, global_rows=local_tokens,
            row_bytes=float(hidden) * act_bytes, dtype=model.activation_dtype,
            meta_extra=scatter_meta)
    if tp > 1:
        tp_tokens = local_tokens if kda_gathered else tokens
        last = b.communicate(
            "linear_reduce", (last,),
            QK_INPUT_COUNT * (tp - 1) / tp * tp_tokens * hidden * act_bytes,
            "allreduce", dtype=model.activation_dtype)
    return last


def linear(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
           padded_tokens, policy=None, runtime=None, physical_rank=None,
           explicit_physical_rank=False, **_):
    from mindie_sim.catalog.paired_sfa import execution
    if execution(model).get("recurrent_context_sharded") is False:
        # Preserve the PCP plan for head-local KDA state.
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
            runtime=runtime,
            physical_rank=physical_rank if explicit_physical_rank else None,
            **selected.options)
    b = Builder(prefix, plan, stage, views, model, mode)
    tensors = TensorLowering(b, runtime.tensor_layouts if runtime else None)
    tp, local_tokens, hidden = parallel.tp, padded_tokens, model.hidden_size
    kda_pcp = _kda_pcp_enabled(plan, parallel, runtime)
    head_shards = tp * int(parallel.pcp) if kda_pcp else tp
    kda_gathered = False
    tokens = local_tokens
    kda_common_peer = local_tokens
    if kda_pcp:
        deps, views, tokens, kda_common_peer, kda_gathered = _apply_kda_pcp_entry(
            b, deps=deps, views=views, tokens=local_tokens, hidden=hidden,
            parallel=parallel, model=model, runtime=runtime)
    key_heads = math.ceil((plan.linear_key_heads or plan.linear_heads) / head_shards)
    heads = math.ceil((plan.linear_value_heads or plan.linear_heads) / head_shards)
    dk, dv = plan.linear_key_head_dim or plan.linear_head_dim, plan.linear_head_dim
    key_width, value_width = key_heads * dk, heads * dv
    use_recover_ssm = recover_ssm_enabled(runtime)
    from .recover_ssm_work import gdn_record_head_bytes
    rec_bytes = heads * (
        gdn_record_head_bytes(dk, dv)
        if plan.attention == "gdn"
        else record_head_bytes(dk, dv, model.activation_dtype)
    )
    kda_source = (select(runtime.kernels.get("lowerings", {}), "kda")
                  if runtime and plan.attention == "kda" else None
                  ) or serving_source(runtime, model, views, plan)
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
    chunk = policy.recurrent_chunk if policy else DEFAULT_RECURRENT_CHUNK_TOKENS
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ModelingError("recurrent_chunk_must_be_positive")
    conv_input = output_gate_view = output_gate = raw_beta = beta = decay = None
    gate_inputs: list[str] = []
    if plan.attention == "gdn":
        conv_input, gate_inputs, output_gate_view = _lower_gdn_projections(
            b, tensors, deps=deps, tokens=tokens, hidden=hidden, conv_width=conv_width,
            value_width=value_width, heads=heads)
        sources = [conv_input.producer]
    else:
        sources, output_gate, beta, decay, raw_beta, projection_group = _lower_kda_projections(
            b, tensors, projection_source=projection_source, conv_source=conv_source,
            deps=deps, tokens=tokens, hidden=hidden, key_width=key_width, value_width=value_width,
            heads=heads, dk=dk)
    conv = _lower_convolution(
        b, tensors, conv_source=conv_source, projection_group=projection_group,
        conv_input=conv_input, sources=sources, views=views, tokens=tokens, heads=heads,
        key_width=key_width, value_width=value_width, conv_width=conv_width, parallel=parallel,
        runtime=runtime)
    norm_rows = sum(view.q_len for view in views) if conv_source else tokens
    normalized = b.vector("qk_l2norm", (conv,), norm_rows * QK_INPUT_COUNT * key_width, L2_NORM_OPS,
                          metadata=(("batch_chunk_norm", plan.kda_batch_chunk_norm
                                     and any(v.phase == "prefill" for v in views)),))
    gated = _lower_delta_gates(
        b, plan=plan, model=model, runtime=runtime, tokens=tokens, heads=heads,
        key_heads=key_heads, dk=dk, gate_inputs=gate_inputs, decay=decay, beta=beta)
    outputs, prefill_scan_ids, prepare_ids = _lower_recurrent_views(
        b, views=views, chunk=chunk, heads=heads, key_heads=key_heads, dk=dk, dv=dv,
        key_width=key_width, value_width=value_width, conv_width=conv_width, tokens=tokens,
        conv=conv, normalized=normalized, gated=gated, parallel=parallel,
        use_recover_ssm=use_recover_ssm, rec_bytes=rec_bytes)
    decode_core, outputs = _bind_kda_decode(
        b, decode_source=decode_source, views=views, parallel=parallel, projection_group=projection_group,
        beta=beta, raw_beta=raw_beta, use_recover_ssm=use_recover_ssm, heads=heads,
        key_heads=key_heads, dk=dk, dv=dv, conv=conv, decay=decay, normalized=normalized,
        gated=gated, outputs=outputs)
    gated, outputs = _bind_kda_prefill(
        b, kda_source=kda_source, kda_caps=kda_caps, prefill_source=prefill_source,
        prefill_scan_ids=prefill_scan_ids, prepare_ids=prepare_ids, views=views, gated=gated,
        outputs=outputs, heads=heads, key_heads=key_heads, dk=dk, dv=dv, chunk=chunk, parallel=parallel)
    if plan.attention == "gdn" and policy and policy.fused_batch:
        outputs = _fuse_gdn_decode_scans(b, outputs, heads=heads, dk=dk, dv=dv)
    output_gate = _declare_recurrent_group(
        b, tensors, views=views, heads=heads, key_heads=key_heads, dk=dk, dv=dv,
        chunk=chunk, tokens=tokens, output_gate=output_gate, output_gate_view=output_gate_view,
        prefill_source=prefill_source, decode_source=decode_source, decode_core=decode_core,
        kda_caps=kda_caps, gated=gated, normalized=normalized,
        prefill_scan_ids=prefill_scan_ids, prepare_ids=prepare_ids)
    if use_recover_ssm and any(view.phase == "decode" and view.q_len > 1 for view in views):
        emit_commit_op(b, name="recover_ssm_commit", deps=tuple(outputs), views=views,
                       heads=heads, dk=dk, dv=dv, state_dtype=plan.recurrent_state_dtype,
                       kg_dtype=model.activation_dtype, runtime=runtime,
                       attention_kind=plan.attention)
    output = _lower_output_norm_gate(
        b, outputs=outputs, output_gate=output_gate, tokens=tokens, heads=heads,
        dv=dv, value_width=value_width, plan=plan, model=model,
        projection_group=projection_group, runtime=runtime)
    last = b.matrix("linear_output", (output,), tokens, hidden, value_width,
                    category="kda.o" if plan.attention == "kda" else "gdn.o")
    last = _lower_kda_pcp_exit(
        b, last=last, plan=plan, model=model, parallel=parallel, views=views,
        tokens=tokens, local_tokens=local_tokens, hidden=hidden,
        kda_pcp=kda_pcp, kda_gathered=kda_gathered, kda_common_peer=kda_common_peer, tp=tp)
    return AttentionResult(ops=b.ops, last=last)
