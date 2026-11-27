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
from typing import Literal

from mindie_sim.catalog.attention_common import AttentionResult
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from ...graph import GraphPolicy
from ...lowering import Builder
from ...mechanism import PlannedRecipe
from ...structure_groups import declare_group, source_reference
from mindie_sim.prediction.operators.recurrent.recover_ssm_work import (
    GDN_ARCH35_RECOVER_SSM_RECIPE,
    emit_commit_op, recover_ssm_enabled, rewrite_gdn_analytical_scan, stamp_gdn_kernel_spec,
)
from ...tensors import TensorLowering
from ...values.rewrite import add_input_ports
from ..contracts import OperatorPlanningContext
from . import gdn_recurrent
from ._gdn_decode_programs import (
    CONV, GATING, CONV_SOURCE, conv_plan, conv_spec_terms, gating_plan, gating_spec_terms,
)
from .common import request_layout_metadata

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
        relative_path="vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape.py",
        source_sha256="3b51906217f16ba69e711a5694cdd9012b7fd2fcc0730761bf1373607de3c389",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_TRANSFORMER_REV,
        relative_path="attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule_arch35.h",
        source_sha256="2a0085a85082d74d7cf6a72c26c78b2f65cac597509506c1d28ca9c4a8767ed8",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_TRANSFORMER_REV,
        relative_path="mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_update.h",
        source_sha256="812584ad8fd5b8dbb76fa53fb15648a15d2942fec7c6d23a1fa866355fea13c3",
    ),
)

SOURCE = {"path": "data/observations/sources/gdn-image-c3d00a196905/gdn.py.txt",
          "sha256": "e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98",
          "caller_lines": ((298, 379), (505, 575), (705, 737))}
CACHE_INDEX_SOURCE_LINES = ((138, 143), (435, 445))
OUTPUT_NORM_GATE_OPS_PER_ELEMENT = 9
FUSED_QKVZBA_RECIPE = "fused_qkvzba_split_reshape_cat"
FUSED_QKVZBA_OUTPUT_COUNT = 4
FUSED_QKVZBA_OPS_PER_ELEMENT = 1


def _is_interleaved_qkvzba(model, plan=None) -> bool:
    """True strictly when the layer plan declares interleaved QKVZ/BA layout."""
    del model
    if plan is None:
        return False
    return str(getattr(plan, "gdn_layout", "") or "").strip().lower() == "interleaved"


def _split_conv_qkv(tensors, conv, tokens, heads, key_heads, dk, dv):
    """Return three contiguous Q/K/V views directly from conv output (no qkv_concatenate)."""
    qkv = tensors.result(conv, (tokens, 2 * key_heads * dk + heads * dv))
    values, start = [], 0
    for name, count, width in (("q", key_heads, dk), ("k", key_heads, dk), ("v", heads, dv)):
        sliced = qkv.narrow(1, start, count * width)
        values.append(tensors.reshape(sliced, (1, tokens, count, width), name=name + "_view"))
        start += count * width
    return values


def _resolve_local_gdn_heads(plan, tp: int) -> tuple[int, int, int, int, int]:
    """Resolve local (keys, heads, dk, dv, key_replication) with exact TP>H_k replication."""
    total_k_heads = int(plan.linear_key_heads or plan.linear_heads)
    total_v_heads = int(plan.linear_value_heads or plan.linear_heads)
    dk = int(plan.linear_key_head_dim or plan.linear_head_dim)
    dv = int(plan.linear_head_dim)
    if tp > total_k_heads:
        if tp % total_k_heads or (2 * total_k_heads * dk) % tp:
            raise ModelingError(
                f"gdn_decode_requires_divisible_key_head_replication:{total_k_heads}:{tp}")
        key_replication = tp // total_k_heads
        keys = 1
    else:
        key_replication = 1
        keys = math.ceil(total_k_heads / tp)
    heads = math.ceil(total_v_heads / tp)
    if heads % keys:
        raise ModelingError("gdn_decode_requires_integer_value_to_key_head_ratio")
    return keys, heads, dk, dv, key_replication


def _build_conv(op, geometry):
    if "conv_state_contract" in dict(op.metadata):
        from mindie_sim.prediction.operators.causal_conv.family_causal_conv import _planned

        return _planned(op, geometry)
    return conv_plan(op, geometry)


def _build_recover_ssm(op, geometry, *, ctx: OperatorPlanningContext | None = None):
    from mindie_sim.prediction.operators.recurrent.recover_ssm_work import build_gdn_arch35_recover_ssm

    return build_gdn_arch35_recover_ssm(op, geometry, ctx=ctx)


_BUILDERS = {
    CONV: _build_conv,
    GATING: gating_plan,
    gdn_recurrent.IMPLEMENTATION: gdn_recurrent.build,
    GDN_ARCH35_RECOVER_SSM_RECIPE: _build_recover_ssm,
}
RECIPE_IDS = frozenset(_BUILDERS)


def build(op, geometry, *, ctx: OperatorPlanningContext | None = None):
    """Bind a native GDN decode recipe to its exact kernel program."""
    builder = _BUILDERS[op.recipe]
    if builder in (gdn_recurrent.build, _build_recover_ssm):
        return builder(op, geometry, ctx=ctx)
    plan = builder(op, geometry)
    if ctx is not None and ctx.requests:
        plan = replace(plan, metadata={**plan.metadata, **request_layout_metadata(ctx)})
    return plan


def recipes():
    return dict.fromkeys(_BUILDERS, PlannedRecipe(build))


def _lower_convolution(b, tensors, *, conv_input, deps, views, active, active_rows,
                       core_tokens, lengths, padding_slots, speculative, max_query,
                       conv_dtype, conv_contract, conv_effects, heads, width,
                       cache_index_row_stride):
    plan, model = b.plan, b.model
    if speculative and any(view.q_len == 1 for view in active):
        raise ModelingError("gdn_decode_mixed_spec_non_spec_caller_not_selected")
    # The caller selects one state index per physical request slot before
    # normalizing the view. Proposal-table stride is independent of query count
    # and padding; a multi-column table needs the real one-column copy.
    cache_indices = tensors.result(deps, (len(lengths), cache_index_row_stride), "int32",
                                    output="causal_cache_indices").narrow(1, 0, 1)
    cache_indices = tensors.contiguous(cache_indices, "state_indices_contiguous",
                                       reason="_normalize_causal_cache_indices.contiguous")
    conv_elem_bytes = int(dtype_bytes(conv_dtype))
    state_write_bytes = sum(
        ((plan.conv_kernel - 1 + max(0, view.q_len - 1)) if speculative else (plan.conv_kernel - 1))
        * width * conv_elem_bytes
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
        "source_layouts": (TensorLayout.contiguous((core_tokens, width)).to_dict(),),
    }.items())))
    external, reads, writes, vector = conv_spec_terms(
        lengths, padding_slots, plan.conv_kernel, width, False, speculative)
    b.ops[-1] = replace(b.ops[-1], recipe=CONV, source_boundary="kernel",
                        padded_shape=(core_tokens, width), hbm_bytes=float(external),
                        mte2_bytes=float(reads), mte3_bytes=float(writes), vector_ops=float(vector))
    if isinstance(conv_input.output, str) and conv_input.output:
        b.ops[-1] = add_input_ports(
            b.ops[-1], ({"op": conv_input.producer, "output": conv_input.output},),
        )
    return conv


def _lower_recurrence(b, tensors, *, decode_core, values, gate_values, views,
                      active, heads, keys, dk, dv, core_tokens, lengths, padding_slots,
                      contract, recurrent_effects, speculative, runtime,
                      recurrent_implementation):
    plan, model = b.plan, b.model
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
        b.ops[-1] = stamp_gdn_kernel_spec(replace(
            scan, recipe=recipe_name, source_boundary="kernel",
            metadata=tuple({**dict(scan.metadata), "source": gdn_recurrent.SOURCE,
                            "implementation_scope": "source_resident_qkv_and_state_pipeline",
                            "recurrence_selection": recurrent_implementation,
                            "recurrent_attention_kind": "gdn",
                            "unmodeled_features": ("scalar_metadata_issue_and_cache_latency",),
                            **extra_meta,
                            }.items())), compact_records=use_recover_ssm)
    elif recurrent_implementation == "analytical":
        updated_scan = replace(scan, metadata=tuple({**dict(scan.metadata),
                               "recurrence_selection": recurrent_implementation}.items()))
        if use_recover_ssm:
            updated_scan = rewrite_gdn_analytical_scan(
                updated_scan, active_views=active, heads=heads, dk=dk, dv=dv,
                state_dtype=plan.recurrent_state_dtype, kg_dtype=model.activation_dtype,
                attention_kind="gdn")
        b.ops[-1] = updated_scan
    else:
        raise ModelingError("gdn_decode_recurrent_implementation_unknown:" + recurrent_implementation)
    if use_recover_ssm:
        emit_commit_op(b, name="recover_ssm_commit", deps=(scan.op_id,), views=active,
                       heads=heads, dk=dk, dv=dv, state_dtype=plan.recurrent_state_dtype,
                       kg_dtype=model.activation_dtype, runtime=runtime, attention_kind="gdn")
    is_kernel = recurrent_implementation == "arch35_fp32"
    if is_kernel:
        rec_members = (scan.op_id,)
        rec_entries = (scan.op_id,)
    else:
        rec_ops = [op for op in b.ops if op.op_id.endswith((
            ".decode_q_l2norm", ".decode_k_l2norm", ".decode_recurrent_scan"))]
        rec_ids = {op.op_id for op in rec_ops}
        rec_members = tuple(op.op_id for op in rec_ops) or (scan.op_id,)
        rec_entries = tuple(
            op.op_id for op in rec_ops
            if any(dep not in rec_ids for dep in op.deps) or not op.deps
        ) or (scan.op_id,)
    declare_group(
        b.ops, key=b.prefix + ".gdn_recurrent", role="gdn_recurrent", family="gdn",
        members=rec_members, entries=rec_entries, anchor=scan.op_id, block="attention",
        view={"tokens": sum(view.q_len for view in active), "heads": heads,
              "key_heads": keys, "key_width": dk, "value_width": dv,
              "dtype": model.activation_dtype, "state_dtype": plan.recurrent_state_dtype,
              "sequence_tokens": lengths, "speculative": speculative},
        implementation={
            "native_kind": "RecurrentGatedDeltaRule",
            "selection": (gdn_recurrent.IMPLEMENTATION if is_kernel
                          else "compatible_recurrent_algorithm; native_kernel_not_selected"),
            "conditions": {"recurrence_selection": recurrent_implementation,
                           "speculative": speculative, "state_dtype": plan.recurrent_state_dtype},
        },
        source=gdn_recurrent.SOURCE if is_kernel else source_reference(
            "vllm_ascend/ops/gdn.py", "AscendGatedDeltaNetAttention._recurrent_gated_delta_rule"),
        fusion_status="fused_existing" if is_kernel else "unfused_decomposed",
        fusion_reason=("arch35_gdn_recurrent_kernel" if is_kernel
                       else "analytical_gdn_decode_recurrence"),
        native_fusion_eligible=False,
    )
    return core


def _resolve_beta_store(runtime, gating_beta_store, gating_programs):
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
    return beta_store


def _declare_gated_rmsnorm(b, *, normalized, tokens, heads, dv, dtype):
    declare_group(
        b.ops, key=b.prefix + ".gated_rmsnorm", role="gated_rmsnorm",
        family="norm_gate", members=(normalized,), entries=(normalized,),
        anchor=normalized, block="attention",
        view={"tokens": tokens, "heads": heads, "width": dv,
              "dtype": dtype, "attention_kind": "gdn"},
        implementation={"native_kind": "_layer_norm_fwd_1pass_kernel_npu",
                        "native_kinds": ("_layer_norm_fwd_1pass_kernel_npu", "FusedRMSNormGated")},
        source=source_reference("vllm_ascend/ops/gdn.py", "FusedRMSNormGated"),
        fusion_status="fused_existing",
        fusion_reason="triton_fused_rmsnorm_gated_kernel",
        native_fusion_eligible=False,
    )


def _resolve_decode_slot_indices(raw_slots, views):
    if raw_slots is None:
        return "caller_tensor", None
    if isinstance(raw_slots, dict):
        try:
            indices = tuple(
                None if raw_slots[view.request_id] is None
                else int(raw_slots[view.request_id]) for view in views
            )
        except (KeyError, TypeError, ValueError):
            raise ModelingError("gdn_decode_state_indices_missing_request")
    elif isinstance(raw_slots, (list, tuple)) and len(raw_slots) == len(views):
        indices = tuple(None if value is None else int(value) for value in raw_slots)
    else:
        raise ModelingError("gdn_decode_state_indices_length_mismatch")
    return "caller_metadata", indices


def _lower_gdn_inputs(b, tensors, *, deps, model, plan, tp: int, tokens: int, core_tokens: int):
    """Project GDN inputs with optional TP>H_k replication and interleaved split."""
    from mindie_sim.prediction.projections import gdn_input_projections

    keys, heads, dk, dv, key_replication = _resolve_local_gdn_heads(plan, tp)
    width, value_width = 2 * keys * dk + heads * dv, heads * dv
    proj_qk_width = (2 * keys * dk) // key_replication
    proj_conv_width = proj_qk_width + value_width
    proj_meta = (
        (("gdn_key_head_replication", key_replication),
         ("checkpoint_qkvz_width", proj_conv_width + value_width))
        if key_replication > 1 else ()
    )
    conv_view, z_view, (beta_view, alpha_view) = gdn_input_projections(
        b, tensors, deps=deps, tokens=tokens, hidden=model.hidden_size,
        conv_width=proj_conv_width, value_width=value_width, heads=heads,
        metadata=proj_meta,
    )
    if key_replication > 1:
        elem_bytes = dtype_bytes(model.activation_dtype)
        rep_bytes = (tokens * 2 * keys * dk * elem_bytes
                     * (key_replication - 1) / key_replication)
        rep_id = b.communicate(
            "gdn_key_head_replicate", (conv_view.producer,), rep_bytes, "allgather",
            axis="local_tp",
            shape=(tokens, width),
            comm_group_size=key_replication,
            local_tp_width=key_replication,
            local_tp_stride=1,
            steps=key_replication - 1,
            gdn_key_head_replication=key_replication,
        )
        conv_view = tensors.result(rep_id, (tokens, width))
    if _is_interleaved_qkvzba(model, plan):
        elem_bytes = dtype_bytes(model.activation_dtype)
        split_elems = tokens * (width + value_width + 2 * heads)
        split_deps = tuple(dict.fromkeys((
            conv_view.producer, z_view.producer, beta_view.producer, alpha_view.producer,
        )))
        split_id = b.vector(
            FUSED_QKVZBA_RECIPE, split_deps, split_elems, FUSED_QKVZBA_OPS_PER_ELEMENT,
            family="tensor_copy",
            shape=(tokens, width + value_width + 2 * heads),
            read_bytes_override=split_elems * elem_bytes,
            write_bytes_override=split_elems * elem_bytes,
            returns=("mixed_qkv", "z", "beta", "alpha"),
            metadata=(("native_kind", FUSED_QKVZBA_RECIPE),
                      ("interleaved_qkvzba", True),
                      ("output_count", FUSED_QKVZBA_OUTPUT_COUNT),
                      ("mixed_qkv_shape", (tokens, width)),
                      ("mixed_qkv_write_bytes", float(tokens * width * elem_bytes)),
                      ("z_shape", (tokens, value_width)),
                      ("z_write_bytes", float(tokens * value_width * elem_bytes)),
                      ("beta_shape", (tokens, heads)),
                      ("beta_write_bytes", float(tokens * heads * elem_bytes)),
                      ("alpha_shape", (tokens, heads)),
                      ("alpha_write_bytes", float(tokens * heads * elem_bytes))),
        )
        conv_input = tensors.result(
            split_id, (tokens, width), output="mixed_qkv").narrow(0, 0, core_tokens)
        z_view = tensors.result(split_id, (tokens, value_width), output="z")
        gate_inputs = [
            tensors.result(split_id, (tokens, heads), output="beta"),
            tensors.result(split_id, (tokens, heads), output="alpha"),
        ]
    else:
        # Live A3+A5 GDN decode captures confirm zero standalone contiguous
        # TensorMoves for beta, alpha, or causal_conv1d input views.
        gate_inputs = [beta_view, alpha_view]
        conv_input = conv_view.narrow(0, 0, core_tokens)
    return keys, heads, dk, dv, width, value_width, conv_input, z_view, gate_inputs


def lower(*, prefix, plan, views, stage, deps, model, parallel, mode,
          padded_tokens, policy, runtime, gating_beta_store=None, gating_programs=None,
          recurrent_implementation="analytical", cache_index_row_stride=1):
    from mindie_sim.prediction.operators.linear_attention.call_gdn import GATING_SOURCE
    from mindie_sim.prediction.operators.linear_attention.gdn_prefill import decode_core

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
    slot_source, slot_indices = _resolve_decode_slot_indices(raw_slots, views)
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
    (keys, heads, dk, dv, width, value_width, conv_input, z_view,
     gate_inputs) = _lower_gdn_inputs(
        b, tensors, deps=deps, model=model, plan=plan, tp=tp,
        tokens=tokens, core_tokens=core_tokens)
    conv = _lower_convolution(
        b, tensors, conv_input=conv_input, deps=deps, views=views, active=active,
        active_rows=active_rows, core_tokens=core_tokens, lengths=lengths,
        padding_slots=padding_slots, speculative=speculative, max_query=max_query,
        conv_dtype=conv_dtype, conv_contract=conv_contract, conv_effects=conv_effects,
        heads=heads, width=width, cache_index_row_stride=cache_index_row_stride)
    values = _split_conv_qkv(tensors, conv, core_tokens, heads, keys, dk, dv)
    beta_store = _resolve_beta_store(runtime, gating_beta_store, gating_programs)
    gate_deps = tuple(dict.fromkeys(value.producer for value in gate_inputs))
    gated = b.work("delta_gates", gate_deps, (core_tokens, heads),
                   family="gdn_gating", dtype="bf16",
                   metadata=(("source", GATING_SOURCE), ("beta_store", beta_store),
                             ("logical_programs", gating_programs),
                             ("g_store_selection", "retained_binary_positive_stride_dma_inference"),
                             ("compiled_identity", "new_capture_binary_unverified"),
                             ("outputs", ("fp32_decay", "bf16_beta"))))
    gate_reads, gate_writes, gate_vector = gating_spec_terms(core_tokens, heads)
    b.ops[-1] = replace(b.ops[-1], recipe=GATING, source_boundary="kernel",
                        hbm_bytes=float(gate_reads + gate_writes), mte2_bytes=float(gate_reads),
                        mte3_bytes=float(gate_writes), vector_ops=float(gate_vector))
    gate_ports = tuple(
        {"op": value.producer, "output": value.output}
        for value in gate_inputs
        if isinstance(value.output, str) and value.output
    )
    if gate_ports:
        b.ops[-1] = add_input_ports(b.ops[-1], gate_ports)
    gate_values = (tensors.result(gated, (1, core_tokens, heads), "fp32", output="decay"),
                   tensors.result(gated, (1, core_tokens, heads), "bf16", output="beta"))
    core = _lower_recurrence(
        b, tensors, decode_core=decode_core, values=values, gate_values=gate_values,
        views=views, active=active, heads=heads, keys=keys, dk=dk, dv=dv,
        core_tokens=core_tokens, lengths=lengths, padding_slots=padding_slots,
        contract=contract, recurrent_effects=recurrent_effects, speculative=speculative,
        runtime=runtime, recurrent_implementation=recurrent_implementation)
    # Live A3+A5 captures confirm layer_norm_gated consumes recurrence output
    # and gate z directly without core_output_copy or z_norm_materialize moves.
    gate = z_view
    normalized = b.vector(
        "linear_output_norm_gate", (core.producer, gate.producer), tokens * value_width,
        OUTPUT_NORM_GATE_OPS_PER_ELEMENT,
        shape=(tokens * heads, dv), read_bytes_override=(2 * tokens * value_width + dv) * 2,
        write_bytes_override=tokens * value_width * 2,
        metadata=(("norm_kind", "rmsnorm"), ("norm_has_weight", True), ("norm_has_bias", False),
                  ("norm_group_width", dv), ("gate_present", True), ("gate_layout", gate.layout.to_dict()),
                  ("recurrent_attention_kind", "gdn"), ("fused_from", ("rmsnorm", "gate"))))
    if isinstance(gate.output, str) and gate.output:
        b.ops[-1] = add_input_ports(
            b.ops[-1], ({"op": gate.producer, "output": gate.output},),
        )
    _declare_gated_rmsnorm(
        b, normalized=normalized, tokens=tokens, heads=heads, dv=dv, dtype=model.activation_dtype)
    last = b.matrix("linear_output", (normalized,), tokens, model.hidden_size, value_width, category="gdn.o")
    if tp > 1:
        last = b.communicate("linear_reduce", (last,), 2 * (tp - 1) / tp * tokens * model.hidden_size * 2, "allreduce")
    classes = {"tensor_copy", "tensor_fill", "short_convolution", "gdn_gating", "recurrent_vector", "elementwise"}
    b.ops = [replace(op, metadata=tuple({**dict(op.metadata), "invocation_class": "vector",
                                        "caller_implementation": "vllm_ascend_gdn_decode"}.items()))
             if op.family in classes else op for op in b.ops]
    return AttentionResult(ops=b.ops, last=last)
