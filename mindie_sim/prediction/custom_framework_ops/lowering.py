"""xLLM custom operator lowering.

[Role] Lower declared xLLM semantic occurrences.
[Invariants] Each occurrence retains its source identity and charges its typed outputs once.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .cache import _DEFAULT_KV_PAGE_SIZE, _SLOT_INDEX_BYTES, _cache_row_bytes
from .contracts import (
    FUSED_GROUPED_RECIPE,
    XLLM_CUSTOM_OPS,
    XLLM_VENDOR,
    _attention_heads,
    _check_family,
    _declared_dtype,
    _fused_declared,
    _resolve_op_name,
    _shape,
    _tokens_width,
)
from .gdn import (
    _CONV_OPS_PER_ELEMENT,
    _DECLARED_READOUT_OPS_PER_ELEMENT,
    _QKV_NORM_OPS_PER_ELEMENT,
    _convolution_traffic,
    _mega_gdn_config,
)
from .grouped_matmul import _fused_config, _fused_quant_contract

_ATTENTION_INDEX_BYTES = int(dtype_bytes("int32"))
_SOFTMAX_STAT_BYTES = int(dtype_bytes("fp32"))
_BEAM_TOKEN_ID_BYTES = int(dtype_bytes("int32"))
_BEAM_LOG_PROB_BYTES = int(dtype_bytes("fp32"))
_BEAM_SELECT_OPS_PER_CANDIDATE = 4


def lower_op(
    op: OpSpec,
    *,
    op_name: str | None = None,
    soc_family: str | None = None,
    return_lse: bool = True,
    beam_width: int | None = None,
    kv_page_size: int = _DEFAULT_KV_PAGE_SIZE,
) -> OpSpec:
    """Lower one complete semantic occurrence into an xLLM custom operator."""
    resolved = _resolve_op_name(op, op_name)
    spec = XLLM_CUSTOM_OPS[resolved]
    _check_family(op, spec, resolved)
    if soc_family is not None and soc_family not in spec["supported_socs"]:
        raise ModelingError(f"xllm_custom_op_soc_unsupported:{resolved}:{soc_family}")
    if type(kv_page_size) is not int or kv_page_size <= 0:
        raise ModelingError(f"xllm_custom_op_invalid_page_size:{kv_page_size}")

    meta = dict(op.metadata)
    dtype = _declared_dtype(op, meta)
    if resolved == FUSED_GROUPED_RECIPE and dtype != "int8":
        _fused_quant_contract(op, meta, dtype)
    if dtype not in spec["input_dtypes"]:
        raise ModelingError(f"xllm_custom_op_input_dtype_unsupported:{resolved}:{dtype}")
    result_dtype = dtype
    local_shape = tuple(op.local_shape)
    if type(return_lse) is not bool:
        raise ModelingError("xllm_sparse_flash_attention_requires_boolean_return_lse")

    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    vec_ops = float(op.vector_ops)
    returns = tuple(op.returns)

    if resolved in {"smla_va", "sparse_flash_attention_lse"} and op.family == "attention":
        recipe = str(op.recipe).lower()
        if not any(token in recipe for token in ("latent", "mla", "sparse", "smla")):
            raise ModelingError(
                f"xllm_{resolved}_requires_latent_sparse_or_mla_attention:{op.recipe}")

    if resolved == "smla_va":
        shape = _shape(op)
        q_shape = tuple(meta.get("q_shape", shape))
        if len(q_shape) == 4:
            batch, seq, heads, dim = q_shape
            tokens = batch * seq
        elif len(q_shape) == 3:
            tokens, heads, dim = q_shape
            batch, seq = 1, tokens
        else:
            tokens, heads = shape[0], _attention_heads(op, meta)
            batch, seq, dim = 1, tokens, shape[-1]
        dv = int(meta.get("value_head_dim", dim))
        read_bytes = tokens * heads * (dim + dv) * 2
        for key in ("ori_kv_shape", "cmp_kv_shape"):
            if meta.get(key) is not None:
                read_bytes += prod(int(v) for v in meta[key]) * 2
        read_bytes += int(meta.get("sparse_index_count", 0)) * _ATTENTION_INDEX_BYTES
        write_bytes = tokens * heads * dv * 2
        meta["xllm_attention_geometry"] = {
            "query_tokens": tokens, "heads": heads, "batch": batch,
            "sequence": seq, "value_head_dim": dv,
            "mode": "variable_absorption"}
        meta["xllm_ports"] = {
            "inputs": ("q", "ori_kv", "cmp_kv", "ori_sparse_indices",
                       "cmp_sparse_indices", "block_tables", "seqlens"),
            "outputs": ("attn_out", "softmax_lse"),
        }
        if "attn_out" not in returns:
            returns = (*returns, "attn_out")
        if bool(meta.get("return_softmax_lse", False)):
            lse = tokens * heads * _SOFTMAX_STAT_BYTES
            write_bytes += lse
            if "softmax_lse" not in returns:
                returns = (*returns, "softmax_lse")
    elif resolved == "sparse_flash_attention_lse":
        shape = _shape(op)
        heads = _attention_heads(op, meta)
        q_shape = tuple(meta.get("q_shape", shape))
        if len(q_shape) == 4:
            batch, seq = q_shape[0], q_shape[1]
            tokens = batch * seq
        elif len(q_shape) == 3:
            tokens = q_shape[0]
        else:
            tokens = shape[0]
        stats = float(2 * tokens * heads * _SOFTMAX_STAT_BYTES)
        write_bytes += stats
        read_bytes = float(tokens * heads * shape[-1] * 2
                           + int(meta.get("sparse_index_count", 0)) * _ATTENTION_INDEX_BYTES)
        for key in ("key_shape", "value_shape"):
            if meta.get(key) is not None:
                read_bytes += prod(int(v) for v in meta[key]) * 2
        meta["xllm_ports"] = {
            "inputs": ("query", "key", "value", "sparse_indices",
                       "block_table", "actual_seq_lengths"),
            "outputs": ("attention_out", "softmax_max", "softmax_sum",
                        "softmax_lse"),
        }
        for item in ("softmax_max", "softmax_sum"):
            if item not in returns:
                returns = (*returns, item)
        if return_lse and "softmax_lse" not in returns:
            returns = (*returns, "softmax_lse")
        meta["xllm_attention_geometry"] = {
            "query_tokens": tokens, "heads": heads, "return_lse": return_lse,
            "softmax_stats_bytes": int(stats),
            "lse_representation": "softmax_max_plus_sum",
        }
    elif resolved == "moe_grouped_matmul_swiglu_quant":
        if not _fused_declared(op, meta, resolved):
            raise ModelingError(
                "xllm_fused_grouped_matmul_requires_declared_fusion_region")
        config = _fused_config(op, meta)
        rows, output_width = config["rows"], config["output_width"]
        read_bytes = float(config["read_bytes"] + config["workspace_bytes"])
        write_bytes = float(config["write_bytes"] + config["workspace_bytes"])
        vec_ops = 0.0
        local_shape = (rows, output_width)
        meta["xllm_swiglu_quant_fused"] = True
        meta["xllm_fused_config"] = config
        meta["xllm_grouped_geometry"] = config
        # The complete fused occurrence owns its group ledger in the config;
        # the generic grouped-matrix row adapter must not reinterpret the
        # now-output-shaped occurrence as an MxNxK matrix.
        meta.pop("matrix_group_rows", None)
        meta["xllm_fused_output_shape"] = (rows, output_width)
        meta["xllm_fused_scale_shape"] = (rows,)
        returns = ("y", "y_scale")
        result_dtype = "int8"
    elif resolved == "mega_gdn_decode":
        if dtype != "bf16":
            raise ModelingError(
                f"xllm_mega_gdn_requires_bf16_inputs:{dtype}")
        if (op.family in {"recurrent_vector", "recurrent_scan"}
                and not _fused_declared(op, meta, resolved)):
            raise ModelingError(
                "xllm_mega_gdn_decode_requires_declared_fusion_region")
        config = _mega_gdn_config(op, meta)
        meta["xllm_mega_gdn_config"] = config
        meta["xllm_gdn_geometry"] = {
            "batch": config["batch"], "k_heads": config["k_heads"],
            "v_heads": config["v_heads"], "head_dim": config["head_dim"],
            "conv_dim": config["conv_dim"],
            "state_dtype": str(meta.get("state_dtype", "fp32")),
            "conv_state_shape": config["conv_state_shape"],
            "ssm_state_shape": config["ssm_state_shape"],
        }
        local_shape = config["z_shape"]
        returns = ("convOut", "convStateOut", "ssmStateOut", "out")
        conv_dim = config["conv_dim"]
        batch, dim = config["batch"], config["head_dim"]
        bf16 = dtype_bytes("bf16")
        conv_read, conv_write = _convolution_traffic(batch, conv_dim, bf16)
        norm_read = batch * config["v_heads"] * dim * bf16 + dim * bf16
        read_bytes = float(conv_read + norm_read)
        write_bytes = float(conv_write)
        vec_ops = float(batch * conv_dim * _CONV_OPS_PER_ELEMENT
                        + batch * (2 * config["k_heads"] + config["v_heads"])
                        * dim * _QKV_NORM_OPS_PER_ELEMENT
                        + batch * config["v_heads"] * dim * _DECLARED_READOUT_OPS_PER_ELEMENT)
        meta["xllm_mega_gdn_declared_work"] = {
            "conv_read_bytes": int(conv_read),
            "conv_write_bytes": int(conv_write),
            "norm_read_bytes": int(norm_read),
            "basis": "source_conv_norm_stage_work_before_recurrent_plan",
        }
    elif resolved == "onerec_final_beam_select":
        rows, width = _tokens_width(op)
        k_beam = int(beam_width if beam_width is not None else
                     meta.get("active_beam", meta.get("beam_width", 0)))
        if k_beam <= 0:
            raise ModelingError(f"xllm_onerec_requires_positive_beam_width:{k_beam}")
        # Source outputs: three result-width scalars (INT32 ids/index, FP32
        # log-probs) plus an INT32 sequence tensor.  The candidate top-k is an
        # input width, not the final result width.
        result_width = int(meta.get("result_width",
                                    meta.get("num_return_sequences", k_beam)))
        topk = int(meta.get("topk", meta.get("candidate_topk", width)))
        batch = int(meta.get("batch", meta.get("request_batch",
                                               max(1, len(op.requests)))))
        current_step = meta.get("current_step")
        sequence_capacity = int(meta.get("sequence_capacity", 0))
        if current_step is not None:
            current_step = int(current_step)
            if sequence_capacity < 1:
                sequence_capacity = current_step + 1
            if current_step < 0 or sequence_capacity < current_step + 1:
                raise ModelingError("xllm_onerec_sequence_capacity_too_small")
            sequence_shape = (batch, result_width, current_step + 1)
        else:
            sequence_shape = meta.get("sequence_shape")
        sequence_bytes = (prod(int(v) for v in sequence_shape) * _BEAM_TOKEN_ID_BYTES
                          if sequence_shape is not None else 0)
        scalar_bytes = float(result_width * (2 * _BEAM_TOKEN_ID_BYTES + _BEAM_LOG_PROB_BYTES))
        beam_write = scalar_bytes + sequence_bytes
        beam_read = float(rows * topk * _BEAM_LOG_PROB_BYTES
                          + rows * k_beam * (_BEAM_TOKEN_ID_BYTES + _BEAM_LOG_PROB_BYTES)
                          + rows * _BEAM_LOG_PROB_BYTES + sequence_bytes)
        declared_write = write_bytes
        # OneRec owns its public outputs; inherited shape bytes are not an
        # additional old-output read/write.
        write_bytes = beam_write
        read_bytes = beam_read
        vec_ops = max(vec_ops, float(rows * k_beam * _BEAM_SELECT_OPS_PER_CANDIDATE))
        for item in ("out_token_ids", "out_token_index", "out_log_probs",
                     "out_sequence"):
            if item not in returns:
                returns = (*returns, item)
        meta["beam_width"] = k_beam
        meta["xllm_ports"] = {
            "inputs": ("log_probs", "top_tokens", "top_probs", "sequence"),
            "outputs": ("out_token_ids", "out_token_index", "out_log_probs",
                        "out_sequence"),
        }
        meta["xllm_beam_selection"] = {
            "rows": rows, "input_width": width, "beam_width": k_beam,
            "selected_write_bytes": int(beam_write),
            "selected_read_bytes": int(beam_read),
            "active_beam": k_beam, "result_width": result_width, "topk": topk,
            "batch": batch, "sequence_capacity": sequence_capacity,
            "current_step": current_step,
            "sequence_shape": sequence_shape,
            "declared_semantic_write_bytes": int(declared_write)}
    elif resolved == "reshape_and_cache_a5":
        tokens = _shape(op)[0]
        cache_view = dict(meta.get("cache_tensor_view") or {})
        cache_dtype = str(
            meta.get("kv_dtype") or meta.get("cache_dtype")
            or cache_view.get("key_dtype") or cache_view.get("value_dtype") or dtype)
        if cache_dtype not in spec["input_dtypes"]:
            raise ModelingError(
                f"xllm_cache_write_dtype_unsupported:{cache_dtype}")
        row_bytes, row_basis = _cache_row_bytes(op, meta, cache_dtype)
        slot_mapping = meta.get("slot_mapping")
        total_slots = meta.get("cache_capacity_slots", meta.get("total_slots"))
        capacity_inferred = False
        mapping_known = slot_mapping is not None
        if slot_mapping is None:
            valid_tokens = None
            invalid_tokens = None
            if type(total_slots) is not int or total_slots < 1:
                total_slots = tokens
            # The external INT32 index tensor is real even when its values are
            # not yet bound.  It is read now; the K/V write remains conditional.
            slot_index_bytes = tokens * _SLOT_INDEX_BYTES
        else:
            if (isinstance(slot_mapping, (str, bytes))
                    or not isinstance(slot_mapping, (tuple, list))
                    or len(slot_mapping) != tokens
                    or any(type(slot) is not int for slot in slot_mapping)):
                raise ModelingError("xllm_cache_slot_mapping_requires_one_integer_per_token")
            if type(total_slots) is not int or total_slots < 1:
                total_slots = max(1, max(slot_mapping, default=-1) + 1)
                capacity_inferred = True
            # Source skips slot < 0 and slot >= total_slots; it does not reject
            # the call.  Those rows are not written.
            valid_tokens = sum(1 for slot in slot_mapping
                               if 0 <= slot < total_slots)
            invalid_tokens = tokens - valid_tokens
            slot_index_bytes = tokens * _SLOT_INDEX_BYTES
        # Page capacity is the actual allocated slot count, not a page-aligned
        # guess; written K/V rows remain the valid token count.
        declared_write = write_bytes
        page_capacity_bytes = float(total_slots * row_bytes)
        if valid_tokens is None:
            # Binding happens later.  No ordinary all-valid K/V write is issued;
            # only the index read and readiness are known now.
            write_bytes = 0.0
            read_bytes = float(slot_index_bytes)
            write_basis = "deferred_slot_binding"
        else:
            write_bytes = float(valid_tokens * row_bytes)
            read_bytes = float(valid_tokens * row_bytes + slot_index_bytes)
            write_basis = "valid_token_rows_not_page_capacity"
        meta["kv_page_size"] = int(kv_page_size)
        # Page mapping span is recorded, but it does not size the issued write.
        meta["aligned_page_rows"] = int(align_up(tokens, kv_page_size))
        meta["cache_write_dtype"] = cache_dtype
        meta["cache_row_bytes"] = int(row_bytes)
        meta["cache_row_basis"] = row_basis
        meta["cache_capacity_slots"] = int(total_slots)
        meta["cache_page_capacity_bytes"] = page_capacity_bytes
        meta["cache_valid_write_rows"] = (None if valid_tokens is None
                                          else int(valid_tokens))
        meta["cache_invalid_slot_rows"] = (None if invalid_tokens is None
                                           else int(invalid_tokens))
        meta["cache_slot_mapping_known"] = mapping_known
        meta["cache_capacity_inferred_from_slots"] = capacity_inferred
        meta["cache_conditional_write_upper_bound"] = (
            float(tokens * row_bytes) if valid_tokens is None else None)
        meta["cache_update_ready"] = bool(valid_tokens is not None)
        meta["cache_slot_index_read_bytes"] = int(slot_index_bytes)
        meta["declared_semantic_write_bytes"] = int(declared_write)
        meta["cache_work_basis"] = write_basis
        if "xtensor_kv_page" not in returns:
            returns = (*returns, "xtensor_kv_page")
        for item in ("key_cache", "value_cache"):
            if item not in returns:
                returns = (*returns, item)

    meta.update({
        "xllm_custom_op": resolved,
        "native_vendor": XLLM_VENDOR,
        "native_kind": spec["native_kind"],
        "native_source": {
            "repository": "xllm",
            "vendor": XLLM_VENDOR,
            "path": spec["source_path"],
            "scope": "independent_source_semantics; custom_xllm_math_lowering",
        },
        "native_implementation": {
            "framework": "xllm",
            "vendor": XLLM_VENDOR,
            "native_kinds": (spec["native_kind"],),
            "op_name": resolved,
            "selection": "explicit_caller_implementation",
            "soc_family": soc_family,
            "atomic_occurrence": bool(spec["atomic_occurrence"]),
        },
    })

    if resolved == FUSED_GROUPED_RECIPE:
        matrix_flops = (("int8", float(dict(meta["xllm_fused_config"])["matrix_flops"])),)
        flops = 0.0
        semantic_shape = local_shape
        padded_shape = local_shape
        launch_shape = local_shape
    elif resolved == "mega_gdn_decode":
        matrix_flops = ()
        flops = 0.0
        semantic_shape = local_shape
        padded_shape = local_shape
        launch_shape = local_shape
    else:
        matrix_flops = op.matrix_flops
        flops = op.flops
        semantic_shape = op.semantic_shape
        padded_shape = op.padded_shape
        launch_shape = op.launch_shape
    return replace(
        op,
        recipe=resolved,
        vector_ops=vec_ops,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        hbm_bytes=read_bytes + write_bytes,
        returns=returns,
        dtype=result_dtype,
        flops=flops,
        matrix_flops=matrix_flops,
        semantic_shape=semantic_shape,
        local_shape=local_shape,
        padded_shape=padded_shape,
        launch_shape=launch_shape,
        metadata=tuple(meta.items()),
    )
