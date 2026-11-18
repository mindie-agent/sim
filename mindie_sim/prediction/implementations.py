"""Explicit block and operator-family implementation registry.

[Role] Couple a frontend implementation's lowering with its physical recipes.
[Invariants] Selection is a declared software implementation, never a model
  label, hardware name or measurement lookup. One binding serves both paths.
  Recipe selectors narrow declared coverage; selected domain misses are errors.
  Typed value descriptors and transform audit records share one vocabulary;
  occurrence ids live in structural annotations, not physical-work axes.
[Timeouts & Locks] Small local module imports; no subprocesses or locks.
[Side-Effect Policy] No device contexts, configuration writes or source reads.
"""
from dataclasses import dataclass, replace
from importlib import import_module
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import FieldDisposition
from mindie_sim.schema.tensors import TensorLayout, TensorValue


@dataclass(frozen=True)
class Implementation:
    module: str
    operators: tuple[str, ...] = ()
    blocks: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    after_activation_quantization: bool = False


@dataclass(frozen=True)
class Selection:
    name: str
    module: object
    options: dict
    recipes: tuple[str, ...] = ()

    def matches(self, op):
        return not self.recipes or op.recipe in self.recipes


IMPLEMENTATIONS = {
    "cann_bf16_lightning_indexer_v2_complete": Implementation(
        "mindie_sim.prediction.family_lightning_indexer", blocks=("lightning_indexer",)),
    "cann_quant_lightning_indexer_v2_complete": Implementation(
        "mindie_sim.prediction.family_quant_indexer", blocks=("quant_lightning_indexer",)),
    "vllm_ascend_group_list_prefix": Implementation(
        "mindie_sim.prediction.moe_group_list", blocks=("group_list_prefix",)),
    "vllm_ascend_cann_keypool_caller": Implementation(
        "mindie_sim.prediction.index_pool_native", blocks=("pool_key_indexer",),
        options=("score_mode", "tail_capacity", "block_table_columns")),
    "causal_pool_indexer_analytic": Implementation(
        "mindie_sim.prediction.index_pool", blocks=("pool_key_indexer",),
        options=("score_mode",)),
    "vllm_ascend_sparse_mla_metadata": Implementation(
        "mindie_sim.prediction.sparse_mla_metadata", blocks=("sparse_mla_metadata",),
        options=("layer_groups", "schedule")),
    "ascend_moe_init_routing_v3_mxfp8": Implementation(
        "mindie_sim.prediction.family_moe_init", blocks=("moe_init_routing",)),
    "triton_noncontiguous_conv_state_copy": Implementation(
        "mindie_sim.prediction.conv_state_copy", blocks=("conv_state_copy",),
        options=("state_slots", "cache_index_dtype", "query_start_dtype", "joint_program_service")),
    "ascend_grouped_int32_dequant_swiglu": Implementation(
        "mindie_sim.prediction.family_grouped_quant", blocks=("grouped_int8_caller",),
        options=("dequant_program", "input_capacity_rows")),
    "vllm_ascend_mc2_unfused_w8a8": Implementation(
        "mindie_sim.prediction.family_mc2_unfused", blocks=("grouped_int8_caller",),
        options=("global_bs",)),
    "ascend_prepared_mx32_matrix": Implementation(
        "mindie_sim.prediction.family_mx_matrix", blocks=("mx_matrix",),
        options=("projection_recipes", "scale_storage")),
    "v41_partial_rotary": Implementation(
        "mindie_sim.prediction.family_v41_rotary", blocks=("v41_rotary",)),
    "ascend_causal_conv1d_update": Implementation(
        "mindie_sim.prediction.family_causal_conv", blocks=("causal_conv_update",)),
    "ascend_paged_mla_decode_arch35": Implementation(
        "mindie_sim.prediction.family_paged_mla", blocks=("paged_mla_attention",),
        options=("kv_splits", "dispatch", "native_query_capacity")),
    "ascend_paged_mla_decode_arch22": Implementation(
        "mindie_sim.prediction.family_paged_mla_arch22", blocks=("paged_mla_attention",),
        options=("dispatch", "retained_kv_rows", "native_query_capacity")),
    "vllm_ascend_mla_head_padding": Implementation(
        "mindie_sim.prediction.mla_head_padding", blocks=("mla_head_padding",)),
    # CANN 9.2.1 official mixed-quant sparse MLA.  The retired ``cannbot-dsl``
    # prototype stays registered as an explicit alias for retained callers.
    "aclnn_mixed_quant_sparse_flash_mla": Implementation(
        "mindie_sim.prediction.family_mqsmla", blocks=("mqsmla_attention",),
        options=("metadata_support_fd", "addressing", "return_softmax_lse")),
    "mixed_quant_sparse_flash_mla": Implementation(
        "mindie_sim.prediction.family_mqsmla", blocks=("mqsmla_attention",),
        options=("metadata_support_fd", "addressing", "return_softmax_lse")),
    "cannbot_mixed_quant_sparse_flash_mla": Implementation(
        "mindie_sim.prediction.family_mqsmla", blocks=("mqsmla_attention",),
        options=("metadata_support_fd", "addressing", "return_softmax_lse")),
    "vllm_ascend_cann_compressor": Implementation(
        "mindie_sim.prediction.compression", blocks=("compressor",)),
    "vllm_ascend_draft_context_prolog_v3": Implementation(
        "mindie_sim.prediction.draft_context", blocks=("draft_context",)),
    "vllm_ascend_kda_mixed_projection": Implementation(
        "mindie_sim.prediction.family_kda_projection", blocks=("kda_input_projection",)),
    "vllm_ascend_recurrent_kda_arch22": Implementation(
        "mindie_sim.prediction.family_kda_recurrent_arch22", blocks=("kda",),
        options=("safe_gate", "use_beta_sigmoid", "gate_dtype", "beta_dtype",
                 "has_a_log", "has_dt_bias", "allow_neg_eigval", "state_selection")),
    "vllm_ascend_recurrent_kda_arch35": Implementation(
        "mindie_sim.prediction.family_kda_recurrent", blocks=("kda",),
        options=("safe_gate", "use_beta_sigmoid", "gate_dtype", "beta_dtype",
                 "has_a_log", "has_dt_bias", "allow_neg_eigval", "state_selection")),
    "vllm_ascend_fla_kda_recurrent": Implementation(
        "mindie_sim.prediction.family_kda_recurrent", blocks=("kda",),
        options=("safe_gate", "use_beta_sigmoid", "gate_dtype", "beta_dtype",
                 "has_a_log", "has_dt_bias", "allow_neg_eigval", "state_selection")),
    "vllm_ascend_gdn_decode": Implementation(
        "mindie_sim.prediction.gdn_decode", blocks=("gdn_decode",),
        options=("gating_beta_store", "gating_programs", "recurrent_implementation", "cache_index_row_stride")),
    "ascend_sparse_flash_attention_arch22": Implementation(
        "mindie_sim.prediction.indexed_attention", blocks=("indexed_attention",),
        options=("scratch_policy",)),
    "vllm_ascend_sharedkv_scfa": Implementation(
        "mindie_sim.prediction.family_sharedkv", blocks=("sharedkv_attention",),
        options=("scratch_policy", "owner_schedule", "selected_rows", "sinks_dtype",
                 "metadata_s2_split", "return_softmax_lse")),
    "vllm_ascend_mxfp8_single_group_eager": Implementation(
        "mindie_sim.prediction.matrix_wrappers", blocks=("matrix_wrapper",),
        options=("projection_recipes",)),
    "ascend_shared_gate_up_int32": Implementation(
        "mindie_sim.prediction.family_shared_quant", blocks=("shared_expert",)),
    "vllm_ascend_native_norm": Implementation(
        "mindie_sim.prediction.family_ascend_norm",
        operators=("add_rmsnorm", "rmsnorm_quant", "add_rmsnorm_quant"),
        options=("has_bias", "external_gamma_shift")),
    "vllm_ascend_attn_res_prefix": Implementation(
        "mindie_sim.prediction.family_attnres", blocks=("attention_residual",),
        options=("optimize_prefill",)),
    "declared_native_fusion": Implementation(
        "mindie_sim.prediction.native_fusion", blocks=("native_fusion",),
        options=("roles", "traffic_profile", "program_profiles", "program_options")),
    "vllm_ascend_kda_in_kernel_gates": Implementation(
        "mindie_sim.prediction.family_kda", blocks=("kda",)),
    "vllm_ascend_kda_chunk_state": Implementation(
        "mindie_sim.prediction.family_kda_state", blocks=("kda",)),
    "vllm_ascend_kda_chunk_prepare_state": Implementation(
        "mindie_sim.prediction.family_kda_prepare", blocks=("kda",)),
    "vllm_ascend_mla_prolog_v3": Implementation(
        "mindie_sim.prediction.family_mla_prolog", blocks=("mla_prolog",),
        options=("cache_layout", "cache_group_size")),
    "gdn_prefill_arch35": Implementation(
        "mindie_sim.prediction.gdn_prefill", blocks=("gdn_prefill",)),
    "vllm_triton_unified_attention_2d_pointer": Implementation(
        "mindie_sim.prediction.family_attention", blocks=("gqa_attention",),
        options=("compute_capability",)),
    "vllm_cuda_rms_norm": Implementation(
        "mindie_sim.prediction.family_norm", operators=("rmsnorm", "add_rmsnorm"),
        options=("aligned", "batch_invariant")),
    "ascendc_grouped_nd_bf16": Implementation(
        "mindie_sim.prediction.family_grouped_gemm", operators=("grouped_gemm",)),
    "ascend_w4a8_msd_grouped_down": Implementation(
        "mindie_sim.prediction.w4a8_down", operators=("grouped_gemm",),
        after_activation_quantization=True),
    "vllm_triton_mla_decode_tuple": Implementation(
        "mindie_sim.prediction.family_mla", blocks=("mla_decode",),
        options=("num_kv_splits",)),
    "cuda_dynamic_int8_quant_symmetric_no_azp": Implementation(
        "mindie_sim.prediction.family_quant", operators=("quantize",),
        options=("input_byte_offset", "output_byte_offset", "zero_absmax")),
    "cuda_moe_align_block_size": Implementation(
        "mindie_sim.prediction.family_moe", operators=("route_pack",),
        options=("block_size", "pad_sorted_ids", "ignore_invalid_experts",
                 "topk_ids_dtype")),
    "triton_moe_fused_mul_sum": Implementation(
        "mindie_sim.prediction.family_moe", operators=("route_unpack",),
        options=("compute_capability", "weight_dtype")),
    "xllm_custom_xllm_math": Implementation(
        "mindie_sim.prediction.family_xllm_ops",
        operators=(
            "smla_va", "sparse_flash_attention_lse", "moe_grouped_matmul_swiglu_quant",
            "mega_gdn_decode", "onerec_final_beam_select", "reshape_and_cache_a5",
            "mla_attention", "paged_mla_attention", "indexed_attention", "grouped_gemm",
            "gdn_decode", "beam_select", "elementwise", "kv_cache_write",
        ),
        options=("op_name", "soc_family", "return_lse", "beam_width", "kv_page_size")),
    "xllm_smla_va": Implementation(
        "mindie_sim.prediction.family_xllm_ops",
        operators=("smla_va", "mla_attention", "paged_mla_attention"),
        options=("op_name", "soc_family")),
    "xllm_sparse_flash_attention_lse": Implementation(
        "mindie_sim.prediction.family_xllm_ops",
        operators=("sparse_flash_attention_lse", "indexed_attention", "mla_attention"),
        options=("op_name", "soc_family", "return_lse")),
    "xllm_moe_grouped_matmul_swiglu_quant": Implementation(
        "mindie_sim.prediction.family_xllm_ops",
        operators=("moe_grouped_matmul_swiglu_quant", "grouped_gemm"),
        options=("op_name", "soc_family")),
    "xllm_mega_gdn_decode": Implementation(
        "mindie_sim.prediction.family_xllm_ops",
        operators=("mega_gdn_decode", "gdn_decode"),
        options=("op_name", "soc_family")),
    "xllm_onerec_final_beam_select": Implementation(
        "mindie_sim.prediction.family_xllm_ops",
        operators=("onerec_final_beam_select", "beam_select", "elementwise"),
        options=("op_name", "soc_family", "beam_width")),
    "xllm_reshape_and_cache_a5": Implementation(
        "mindie_sim.prediction.family_xllm_ops",
        operators=("reshape_and_cache_a5", "kv_cache_write"),
        options=("op_name", "soc_family", "kv_page_size")),
    "mindie_sd_custom_ops": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=(
            "rainfusion", "ada_bsa", "eagle_bsa", "eagle_qbsa", "adalayernorm",
            "norm_rope_concat", "laurel", "quant_four_over_six_a5", "eagle_ffn",
            "mindie_sd_rainfusion", "mindie_sd_ada_bsa", "mindie_sd_eagle_bsa",
            "mindie_sd_eagle_qbsa", "mindie_sd_adalayernorm", "mindie_sd_norm_rope_concat",
            "mindie_sd_laurel", "mindie_sd_quant_four_over_six_a5", "mindie_sd_eagle_ffn",
            "indexed_attention", "gqa_attention", "rmsnorm", "add_rmsnorm",
            "elementwise", "quantize", "gemm", "ffn",
        ),
        options=("op_name", "soc_family", "sparsity_ratio", "low_rank", "block_size")),
    "mindie_sd_rainfusion": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("rainfusion", "mindie_sd_rainfusion", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size")),
    "mindie_sd_ada_bsa": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("ada_bsa", "mindie_sd_ada_bsa", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size")),
    "mindie_sd_eagle_bsa": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("eagle_bsa", "mindie_sd_eagle_bsa", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size")),
    "mindie_sd_eagle_qbsa": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("eagle_qbsa", "mindie_sd_eagle_qbsa", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size")),
    "mindie_sd_adalayernorm": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("adalayernorm", "mindie_sd_adalayernorm", "rmsnorm", "add_rmsnorm"),
        options=("op_name", "soc_family")),
    "mindie_sd_norm_rope_concat": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("norm_rope_concat", "mindie_sd_norm_rope_concat", "rmsnorm", "elementwise"),
        options=("op_name", "soc_family")),
    "mindie_sd_laurel": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("laurel", "mindie_sd_laurel", "gemm", "elementwise"),
        options=("op_name", "soc_family", "low_rank")),
    "mindie_sd_quant_four_over_six_a5": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5", "quantize", "elementwise"),
        options=("op_name", "soc_family")),
    "mindie_sd_eagle_ffn": Implementation(
        "mindie_sim.prediction.family_mindie_sd_ops",
        operators=("eagle_ffn", "mindie_sd_eagle_ffn", "gemm", "ffn"),
        options=("op_name", "soc_family")),
    "vllm_ascend_sparse_kv_offload": Implementation(
        "mindie_sim.prediction.sparse_kv_offload",
        blocks=("sparse_kv_offload",),
        operators=(
            "memfabric_sparse_copy_d2h", "memfabric_sparse_copy_h2d",
            "memfabric_kv_exchange", "memfabric_entry_gather",
            "fused_lightning_indexer_manage", "copy_sfa_attention",
            "fused_sparse_attention_overlap",
        ),
        options=(
            "dram_size_per_dp_gb", "topk_buffer_size", "miss_rate",
            "use_fused_copy_sfa", "use_fused_overlap", "keep_device_kv_cache",
            "store_index_k_on_host", "layerwise_kv_offload", "num_shared_layers",
            "soc_family", "page_size",
        )),
    "hisparse_kv_offload": Implementation(
        "mindie_sim.prediction.sparse_kv_offload",
        blocks=("sparse_kv_offload",),
        operators=(
            "memfabric_sparse_copy_d2h", "memfabric_sparse_copy_h2d",
            "memfabric_kv_exchange", "memfabric_entry_gather",
            "fused_lightning_indexer_manage", "copy_sfa_attention",
            "fused_sparse_attention_overlap",
        ),
        options=(
            "dram_size_per_dp_gb", "topk_buffer_size", "miss_rate",
            "use_fused_copy_sfa", "use_fused_overlap", "keep_device_kv_cache",
            "store_index_k_on_host", "layerwise_kv_offload", "num_shared_layers",
            "soc_family", "page_size",
        )),
}


def implementation(name):
    return import_module(IMPLEMENTATIONS[name].module)


def select(bindings, selector):
    """Resolve one software declaration, independent of shape and measurements."""
    raw = bindings.get(selector)
    if raw is None:
        return None
    if isinstance(raw, dict) and "cases" in raw:
        from .implementation_selection import alternatives
        tuple(alternatives(raw, selector))
        raise ModelingError(f"implementation_policy_requires_batch_geometry:{selector}")
    declaration = {"implementation": raw} if isinstance(raw, str) else raw
    if set(declaration) - {"implementation", "options", "recipes"}:
        raise ModelingError(f"implementation_binding_fields_unknown:{selector}")
    name = declaration.get("implementation")
    if name not in IMPLEMENTATIONS:
        raise ModelingError(f"implementation_not_registered:{selector}:{name}")
    registered = IMPLEMENTATIONS[name]
    if selector not in (*registered.operators, *registered.blocks):
        raise ModelingError(f"implementation_selector_mismatch:{selector}:{name}")
    options = dict(declaration.get("options", {}))
    if isinstance(name, str) and name.startswith("xllm_") and name != "xllm_custom_xllm_math":
        options.setdefault("op_name", name.removeprefix("xllm_"))
    elif isinstance(name, str) and name.startswith("mindie_sd_") and name != "mindie_sd_custom_ops":
        options.setdefault("op_name", name.removeprefix("mindie_sd_"))
    unknown_options = options.keys() - set(registered.options)
    if unknown_options:
        raise ModelingError(
            f"implementation_options_unknown:{selector}:" + ",".join(sorted(unknown_options)))
    selected_recipes = declaration.get("recipes", ())
    if isinstance(selected_recipes, str):
        raise ModelingError(f"implementation_recipes_require_sequence:{selector}")
    selected_recipes = tuple(selected_recipes)
    if selector in registered.blocks and selected_recipes:
        raise ModelingError(f"block_implementation_has_no_recipe_selector:{selector}")
    return Selection(name, implementation(name), options, selected_recipes)


def recipes(bindings):
    from .implementation_selection import alternatives

    result, loaded = {}, set()
    for selector, raw in bindings.items():
        for binding in alternatives(raw, selector):
            if binding == "analytical":
                continue
            selected = select({selector: binding}, selector)
            if selected.name not in loaded:
                result.update(selected.module.recipes())
                loaded.add(selected.name)
    return result


def lower_operators(ops, bindings, *, activation_quantized=False, final=False):
    selections = {}
    for selector in bindings:
        selected = select(bindings, selector)
        if selector in IMPLEMENTATIONS[selected.name].operators:
            selections[selector] = selected
    result = []
    for op in ops:
        selector = op.family if op.family in selections else op.recipe
        selected = selections.get(selector)
        # A binding applied before a generic pass must not be applied again to
        # an output that intentionally retains its semantic recipe (the MoE
        # pack copy consumes alignment indices but is not replaced by them).
        if (op.plan_required or "implementation_binding" in dict(op.metadata)
                or selected is None or not selected.matches(op)):
            result.append(op)
            continue
        if (IMPLEMENTATIONS[selected.name].after_activation_quantization
                and not activation_quantized):
            if final:
                raise ModelingError(
                    "implementation_requires_activation_quantization:" + selected.name)
            result.append(op)
            continue
        lowered = selected.module.lower_op(op, **selected.options)
        binding = ("implementation_binding", {"selector": selector,
                                              "implementation": selected.name})
        if isinstance(lowered, (tuple, list)):
            result.extend(replace(item, metadata=item.metadata + (binding,))
                          for item in lowered)
        else:
            result.append(replace(lowered, metadata=lowered.metadata + (binding,)))
    return tuple(result)


def preserve_graph(source, semantic):
    """A physical replacement keeps callers, outputs and graph placement.

    The selected source owns its dispatch/ABI metadata, but a semantic field
    that only the replaced record declared (dtype/format/scale/state/attr) is
    not dropped merely because the source planner does not repeat it.  Source
    keys take precedence; the complete semantic record remains available under
    ``semantic_metadata`` for exact provenance.
    """
    metadata = dict(source.metadata)
    for key, value in semantic.metadata:
        if key in {"graph_captured", "graph_segment"}:
            metadata.setdefault(key, value)
        else:
            metadata.setdefault(key, value)
    metadata["semantic_recipe"] = semantic.recipe
    metadata["semantic_metadata"] = dict(semantic.metadata)
    metadata["source_binding"] = {
        "implementation_recipe": source.recipe,
        "semantic_recipe": semantic.recipe,
        "semantic_input_dtype": semantic.dtype,
        "semantic_input_format": dict(semantic.metadata).get("input_format", semantic.dtype),
        "semantic_output_format": dict(semantic.metadata).get("output_format", semantic.dtype),
        "semantic_returns": list(semantic.returns),
        "source_boundary": source.source_boundary,
    }
    explicit_semantic = _audit_outputs(semantic)
    if explicit_semantic and not _audit_outputs(source):
        append_audit(
            metadata, kind="source_value_contract", op_id=semantic.op_id,
            boundary={"typed_outputs": tuple(
                _normalize_output(value, semantic, None, native=native)
                for value, native in explicit_semantic)},
            layer=semantic.layer, stage=semantic.stage)
    return replace(
        source, op_id=semantic.op_id, family=semantic.family, phase=semantic.phase,
        stage=semantic.stage, layer=semantic.layer, requests=semantic.requests,
        deps=semantic.deps, stream=semantic.stream, semantic_shape=semantic.semantic_shape,
        metadata=tuple(metadata.items()),
        notes=semantic.notes + source.notes,
    )


def plan_record(selected):
    """Describe the selected software plan without executing its planner."""
    if selected is None:
        return {"kind": "analytical_semantic_work", "implementation": "analytical",
                "options": {}, "recipes": []}
    return {
        "kind": "declared_software_plan",
        "implementation": selected.name,
        "module": selected.module.__name__,
        "options": dict(selected.options),
        "recipes": list(selected.recipes),
    }


def _contiguous_strides(shape):
    stride, strides = 1, []
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(1, size)
    return tuple(reversed(strides))


def typed_value(*, producer, dtype, shape, output, role, source, quant="",
                materialized=True, layout_kind="strided", storage="", alias_of="",
                state_version=None, binding="static", binding_name="",
                storage_shape=None, write_bytes=None, declared_bytes=None,
                contract="", layout=None, strides=None, offset=0,
                semantic_producer="", semantic_output="", layout_name="",
                source_format=""):
    """One explicit value descriptor, independent of the op-level ``dtype``.

    ``shape`` is the logical value shape.  ``storage_shape`` is the actual
    allocation view (for example four captured rows for three valid rows).
    A matmul output is the (M,N) value, not the operator's (M,N,K) geometry.
    """
    if not isinstance(dtype, str) or not dtype:
        raise ModelingError(f"typed_value_dtype_required:{producer}:{output}")
    logical = tuple(int(size) for size in shape)
    physical = tuple(int(size) for size in (storage_shape or logical))
    if any(size < 0 for size in logical) or any(size < 0 for size in physical):
        raise ModelingError(f"typed_value_shape_negative:{producer}:{output}")
    if len(physical) < len(logical):
        physical = logical
    if isinstance(layout, str):
        layout_name = layout_name or layout
        layout = None
    if isinstance(layout, dict):
        for key in ("layout_name", "semantic_layout", "name", "layout"):
            if not layout_name and isinstance(layout.get(key), str) and layout[key]:
                layout_name = str(layout[key])
    if layout is not None:
        if isinstance(layout, dict):
            layout_shape = tuple(int(size) for size in
                                 (layout.get("shape") or logical))
            strides = tuple(int(size) for size in
                            (layout.get("strides") or _contiguous_strides(layout_shape)))
            offset = int(layout.get("offset", 0) or 0)
            physical = tuple(int(size) for size in
                             (layout.get("storage_shape") or physical))
            logical = layout_shape
        else:
            logical = tuple(int(size) for size in layout.shape)
            strides = tuple(int(size) for size in layout.strides)
            offset = int(layout.offset)
            physical = tuple(int(size) for size in (layout.storage_shape or physical))
    if strides is None:
        strides = _contiguous_strides(logical)
    if len(strides) != len(logical):
        raise ModelingError(f"typed_value_strides_rank_mismatch:{producer}:{output}")
    value = TensorValue(
        producer=producer, dtype=dtype,
        layout=TensorLayout(logical, tuple(strides), int(offset), physical),
        output=output, quant=quant, layout_kind=layout_kind, storage=storage,
        alias_of=alias_of, state_version=state_version, binding=binding,
        binding_name=binding_name)
    descriptor = value.to_dict()
    logical_bytes = None
    storage_bytes = None
    try:
        from mindie_sim.schema.common import dtype_bytes
        logical_bytes = float(math.prod(logical) * dtype_bytes(dtype))
        storage_bytes = float(math.prod(physical) * dtype_bytes(dtype))
    except (KeyError, ValueError):
        pass
    descriptor.update({
        "shape": list(logical),
        "logical_shape": list(logical),
        "storage_shape": list(physical),
        "role": role,
        "source": source,
        "materialized": bool(materialized),
        "contract": contract,
        "value_id": f"{producer}.{output}",
        "semantic_producer": semantic_producer or producer,
        "semantic_output": semantic_output or output,
    })
    if logical_bytes is not None:
        descriptor["bytes"] = logical_bytes
        descriptor["storage_bytes"] = storage_bytes
    if write_bytes is not None:
        descriptor["write_bytes"] = float(write_bytes)
    elif descriptor["materialized"] and logical_bytes is not None:
        descriptor["write_bytes"] = logical_bytes
    if declared_bytes is not None:
        descriptor["declared_bytes"] = float(declared_bytes)
    if layout_name:
        descriptor["layout_name"] = str(layout_name)
        descriptor["semantic_layout"] = str(layout_name)
    if source_format:
        descriptor["source_format"] = str(source_format)
    return descriptor


def field_disposition(field, action, source="", detail=""):
    """Validate and freeze one preserve/rewrite/consume decision."""
    return FieldDisposition(field, action, source, detail).to_dict()


def typed_roles(source):
    """Separate declared input/scale/state/index/accumulator/output types.

    ``source`` is an OpSpec or a mapping (a fusion tensor view).  Only declared
    dtype/format fields are reported; a missing field stays missing rather
    than being inferred from the model's activation dtype.
    """
    metadata = dict(getattr(source, "metadata", source) or {})
    roles = {}

    def add(role, field, dtype, shape=None):
        if not isinstance(dtype, str) or not dtype:
            return
        record = {"field": field, "dtype": dtype}
        if shape:
            record["shape"] = list(shape)
        bucket = roles.setdefault(role, [])
        if record not in bucket:
            bucket.append(record)

    for key, value in metadata.items():
        if not isinstance(value, str) or not value or key == "dtype":
            continue
        lower = key.lower()
        if not (lower.endswith("dtype") or lower.endswith("format")):
            continue
        if "scale" in lower or "amax" in lower:
            role = "scale"
        elif "index" in lower or "topk" in lower:
            role = "index"
        elif "accum" in lower or "compute" in lower:
            role = "accumulator"
        elif "state" in lower or lower in {"cache_dtype", "kv_dtype", "kv_cache_dtype"}:
            role = "state"
        elif "gamma" in lower or "parameter" in lower or "coefficient" in lower or "bias" in lower:
            role = "parameter"
        elif "weight" in lower or "router" in lower:
            role = "weight"
        elif "input" in lower or "activation" in lower:
            role = "input"
        elif "output" in lower:
            role = "output"
        else:
            continue
        shape = None
        if key == "activation_scale_format":
            shape = metadata.get("activation_scale_shape")
        elif key in {"output_format", "dtype"} and hasattr(source, "local_shape"):
            if hasattr(source, "op_id"):
                produced = produced_values(source)
                if produced:
                    value = produced[0]["dtype"]
                    shape = produced[0]["logical_shape"]
                else:
                    shape = getattr(source, "local_shape", None)
            else:
                shape = getattr(source, "local_shape", None)
        add(role, key, value, shape)
    if hasattr(source, "dtype") and source.dtype:
        add("output", "output_format", metadata.get("output_format") or source.dtype,
            getattr(source, "local_shape", None))
    for name in getattr(source, "returns", ()) or ():
        if name == "activation_scales":
            add("scale", "returns.activation_scales",
                metadata.get("activation_scale_format") or "fp32",
                metadata.get("activation_scale_shape") or
                ((source.local_shape[0], 1) if getattr(source, "local_shape", ()) else None))
        elif name == "rstd":
            add("auxiliary", "returns.rstd", "fp32")
        elif name:
            produced = next((value for value in produced_values(source)
                             if value.get("output") == name), None)
            add("output", f"returns.{name}",
                (produced or {}).get("dtype") or metadata.get(f"{name}_dtype") or
                (metadata.get("output_format") or getattr(source, "dtype", "")),
                (produced or {}).get("logical_shape"))
    return {role: tuple(records) for role, records in roles.items()}


def merge_typed_roles(*sources):
    merged = {}
    for source in sources:
        for role, records in typed_roles(source).items():
            bucket = merged.setdefault(role, [])
            for record in records:
                if record not in bucket:
                    bucket.append(record)
    return {role: tuple(records) for role, records in merged.items()}


def _row_shape(metadata, shape):
    logical, storage = tuple(shape), tuple(shape)
    if logical:
        active = metadata.get("graph_active_query_rows")
        physical = metadata.get("graph_physical_query_rows")
        if (isinstance(active, int) and not isinstance(active, bool)
                and isinstance(physical, int) and not isinstance(physical, bool)
                and 0 <= active <= physical
                and shape[0] == physical
                and "expected_rows" not in metadata
                and "routing_basis" not in metadata
                and "packed_layout" not in metadata
                and not metadata.get("activation_input_segments")):
            logical, storage = (active,) + logical[1:], (physical,) + storage[1:]
    return logical, storage


def _primary_output_name(op, metadata):
    declared = metadata.get("primary_output")
    if isinstance(declared, str) and declared:
        return declared
    if op.recipe in {"dynamic_quantize", "rmsnorm_quantize",
                     "add_rmsnorm_quantize"} or op.family in {
            "quantize", "rmsnorm_quant", "add_rmsnorm_quant"}:
        return "payload"
    if op.family in {"rmsnorm", "add_rmsnorm", "layernorm", "normalization"}:
        return "norm"
    return "output"


def _row_geometry(op, metadata):
    shape = tuple(op.local_shape)
    if metadata.get("program_kind") == "matmul" and len(shape) >= 2:
        return shape[:2]
    return shape


def _audit_outputs(op):
    for record in audit_records(op):
        if record.get("kind") not in {"transform_boundary", "activation_materialization",
                                      "native_fusion_boundary", "source_value_contract"}:
            continue
        boundary = record.get("boundary")
        if not isinstance(boundary, dict):
            continue
        if boundary.get("typed_outputs"):
            return tuple((value, False) for value in boundary["typed_outputs"])
        typed = boundary.get("typed")
        if isinstance(typed, dict):
            for key in ("outputs", "external_outputs"):
                if typed.get(key):
                    return tuple((value, True) for value in typed[key])
        if boundary.get("outputs"):
            return tuple((value, True) for value in boundary["outputs"])
    return ()


def _json_identity(value, default="static"):
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _declared_output_geometry(op, metadata):
    """Prefer actual declared output views over operator input/KV geometry."""
    for key in ("output_shape", "combine_output_shape", "public_output_shape"):
        shape = metadata.get(key)
        if isinstance(shape, (list, tuple)) and shape:
            return tuple(int(value) for value in shape), None, ""
    for key in ("output_view", "result_view", "public_view"):
        view = metadata.get(key)
        if isinstance(view, dict):
            layout = view.get("layout") if isinstance(view.get("layout"), dict) else None
            shape = view.get("shape") or (layout or {}).get("shape")
            if isinstance(shape, (list, tuple)) and shape:
                return (tuple(int(value) for value in shape), layout,
                        str(view.get("layout_name") or view.get("semantic_layout") or ""))
    for key in ("output_ports", "typed_ports"):
        ports = metadata.get(key)
        if isinstance(ports, (list, tuple)):
            for port in ports:
                if not isinstance(port, dict):
                    continue
                shape = port.get("shape") or port.get("logical_shape")
                if port.get("output") in (None, "output", "q") and shape:
                    layout = port.get("layout") if isinstance(port.get("layout"), dict) else None
                    return (tuple(int(value) for value in shape), layout,
                            str(port.get("layout_name")
                                or port.get("semantic_layout") or ""))
    attention = metadata.get("attention_tensor_view")
    if isinstance(attention, dict):
        shape = attention.get("output_shape") or attention.get("query_shape")
        if isinstance(shape, (list, tuple)) and shape:
            layout_name = str(attention.get("output_layout")
                              or attention.get("logical_layout") or "")
            return tuple(int(value) for value in shape), None, layout_name
    target = metadata.get("target_layout")
    if isinstance(target, dict) and target.get("shape"):
        return (tuple(int(value) for value in target["shape"]), target,
                str(metadata.get("target_layout_name") or ""))
    return None, None, ""


def _normalize_output(value, op, materialized, *, native=False):
    raw_layout = value.get("layout")
    layout_name = (value.get("layout_name") or value.get("semantic_layout")
                   or (raw_layout if isinstance(raw_layout, str) else ""))
    if not layout_name and isinstance(raw_layout, dict):
        for key in ("layout_name", "semantic_layout", "name", "layout"):
            if isinstance(raw_layout.get(key), str) and raw_layout[key]:
                layout_name = str(raw_layout[key])
                break
    layout = raw_layout if isinstance(raw_layout, (dict, TensorLayout)) else None
    logical = tuple(value.get("logical_shape") or value.get("shape") or ())
    storage = tuple(value.get("storage_shape") or logical)
    if isinstance(layout, dict):
        logical = tuple(layout.get("shape") or logical)
        storage = tuple(layout.get("storage_shape") or storage)
    dtype = value.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ModelingError(f"typed_output_dtype_unresolved:{op.op_id}:{value.get('output')}")
    if native:
        producer = op.op_id
        output = value.get("port") or value.get("output") or "output"
        semantic_producer = value.get("member") or value.get("producer") or producer
        semantic_output = value.get("output") or output
    else:
        producer = value.get("producer") or op.op_id
        output = value.get("output") or "output"
        semantic_producer = value.get("semantic_producer") or producer
        semantic_output = value.get("semantic_output") or output
    return typed_value(
        producer=producer, dtype=dtype, shape=logical,
        storage_shape=storage, output=output,
        role=value.get("role") or "declared_output",
        source=value.get("source") or op.op_id, quant=value.get("quant") or "",
        materialized=(value.get("materialized", True) if materialized is None
                      else materialized),
        layout_kind=value.get("layout_kind") or "strided",
        storage=value.get("storage") or "", alias_of=value.get("alias_of") or "",
        state_version=_json_identity(value.get("state_version"), None),
        binding=_json_identity(value.get("binding"), "static"),
        binding_name=_json_identity(value.get("binding_name"), ""),
        write_bytes=value.get("write_bytes"),
        declared_bytes=value.get("declared_bytes"),
        contract=value.get("contract") or "",
        layout=layout,
        semantic_producer=semantic_producer,
        semantic_output=semantic_output,
        layout_name=layout_name,
        source_format=value.get("source_format") or "",
    )


def produced_values(op, *, materialized=None):
    """All actual values written by ``op``, with logical and storage shapes.

    Produced values are derived from the operator's real output contract:
    a matmul writes ``(M,N)`` even though its operand geometry is ``(M,N,K)``;
    quantized values carry the declared payload format, not the consumer's
    activation format; scales are separate values.
    """
    explicit = _audit_outputs(op)
    if explicit:
        return tuple(_normalize_output(value, op, materialized, native=native)
                     for value, native in explicit)
    metadata = dict(op.metadata)
    declared_shape, declared_layout, declared_layout_name = _declared_output_geometry(
        op, metadata)
    geometry = declared_shape or _row_geometry(op, metadata)
    if not geometry:
        return ()
    logical, storage = _row_shape(metadata, geometry)
    dtype = str(metadata.get("output_format") or metadata.get("output_dtype")
                or op.dtype)
    if not dtype:
        raise ModelingError(f"typed_output_primary_dtype_unresolved:{op.op_id}")
    name = _primary_output_name(op, metadata)
    quant = str(metadata.get("output_quant") or metadata.get("quantization") or "")
    layout_kind = str(metadata.get("output_layout_kind") or
                      ("strided" if op.layout in {"", "contiguous", "strided"} else op.layout))
    layout_name = str(declared_layout_name
                      or metadata.get("output_layout_name")
                      or metadata.get("semantic_layout")
                      or (metadata.get("output_layout")
                          if isinstance(metadata.get("output_layout"), str) else ""))
    try:
        payload_bytes = float(math.prod(logical) * dtype_bytes(dtype))
    except (KeyError, ValueError):
        payload_bytes = None
    declared_materialized = metadata.get("materialized")
    is_materialized = (bool(declared_materialized)
                       if declared_materialized is not None else True)
    if materialized is not None:
        is_materialized = materialized
    alias_of = str(metadata.get("alias_of") or "")
    write_bytes = metadata.get("write_bytes")
    if write_bytes is None and is_materialized:
        write_bytes = payload_bytes
    primary = typed_value(
        producer=op.op_id, dtype=dtype, shape=logical, storage_shape=storage,
        output=name, role="activation", source=f"{op.op_id}:output_format",
        quant=quant, materialized=is_materialized,
        layout_kind=layout_kind, write_bytes=write_bytes,
        declared_bytes=op.mte3_bytes, alias_of=alias_of,
        storage=str(metadata.get("storage") or ""),
        layout=(declared_layout or (metadata.get("output_layout")
                                    if isinstance(metadata.get("output_layout"), dict) else None)),
        layout_name=layout_name,
        source_format=str(metadata.get("source_format") or ""),
        contract="operator_output_contract")
    if metadata.get("source_outputs"):
        primary["source_outputs"] = tuple(metadata["source_outputs"])
    if metadata.get("source_layouts"):
        primary["source_layouts"] = tuple(metadata["source_layouts"])
    if op.family == "tensor_copy" and not (
            metadata.get("source_layout") or metadata.get("source_layouts")
            or metadata.get("target_layout") or metadata.get("output_layout")):
        primary["layout_gap"] = "missing_source_layout"
    outputs = [primary]
    for returned in op.returns:
        if returned == "activation_scales":
            dtype = str(metadata.get("activation_scale_format") or "fp32")
            shape = tuple(metadata.get("activation_scale_shape") or
                          ((logical[0], 1) if logical else ()))
            logical_scale, storage_scale = _row_shape(metadata, shape)
            outputs.append(typed_value(
                producer=op.op_id, dtype=dtype, shape=logical_scale,
                storage_shape=storage_scale, output=returned, role="scale",
                source=f"{op.op_id}:returns.{returned}",
                materialized=(True if materialized is None else materialized),
                write_bytes=metadata.get("activation_scale_bytes"),
                contract="declared_scale_value"))
        elif returned == "rstd":
            rows = logical[0] if logical else 0
            outputs.append(typed_value(
                producer=op.op_id, dtype="fp32", shape=(rows,),
                storage_shape=(storage[0],) if storage else (rows,),
                output=returned, role="auxiliary",
                source=f"{op.op_id}:returns.{returned}",
                materialized=True, contract="declared_auxiliary_value"))
        else:
            declared_shape = metadata.get(f"{returned}_shape")
            shape = tuple(declared_shape) if declared_shape else logical
            if returned == "residual":
                dtype = str(metadata.get("residual_dtype")
                            or metadata.get("input_format")
                            or metadata.get("activation_format") or op.dtype)
            else:
                dtype = str(metadata.get(f"{returned}_dtype")
                            or metadata.get("output_format") or op.dtype)
            outputs.append(typed_value(
                producer=op.op_id, dtype=dtype, shape=shape, storage_shape=shape,
                output=returned, role="declared_output",
                source=f"{op.op_id}:returns.{returned}",
                materialized=True,
                write_bytes=None,
                contract="declared_return_value"))
    return tuple(outputs)


def typed_outputs(op, *, materialized=True):
    """Compatibility wrapper for the produced value contract."""
    return produced_values(op, materialized=materialized)


def first_value(values):
    return values[0] if values else None


def _access_bytes(value, *, shape=None, repeats=1, explicit=None,
                  expected_elements=None,
                  contract="declared_operand_geometry"):
    shape = tuple(shape or value.get("logical_shape") or value.get("shape") or ())
    dtype = value.get("dtype")
    elements = int(math.prod(shape)) if shape else 0
    effective_elements = elements if expected_elements is None else float(expected_elements)
    try:
        base_read = float(effective_elements * dtype_bytes(dtype))
    except (KeyError, ValueError):
        base_read = None
    if explicit is not None:
        # Convention: operand_read_bytes is total issued bytes for this
        # operand and already includes repeats.
        read_bytes = float(explicit)
        contract = "explicit_operand_access"
    else:
        read_bytes = base_read
    result = dict(value)
    result.update({
        "access_shape": list(shape),
        "read_elements": elements,
        "repeats": int(repeats),
        "base_read_bytes": base_read,
        "contract": contract,
    })
    if read_bytes is not None:
        result["read_bytes"] = (read_bytes if explicit is not None
                                else read_bytes * int(repeats))
    return result


def _select_named_value(op, dependency, requested, outputs_by_id):
    available = list((outputs_by_id or {}).get(dependency, ()))
    if not available:
        return None
    if requested:
        exact = next((value for value in available
                      if value.get("output") == requested), None)
        if exact is not None:
            return exact
        if requested in {"payload", "output", "activation"}:
            return available[0]
        return None
    return available[0]


def producer_input_values(op, outputs_by_id=None, source_dtype=None):
    """Resolve every consumer operand by value identity, not by first output.

    Named ``input_ports`` select the exact producer output (for example a BF16
    residual port of an INT8-payload fusion).  A bare dependency is only the
    explicit primary-output compatibility rule.  Each returned value carries a
    per-operand access shape, repeat count and issued read bytes; a producer's
    write bytes are never substituted for the consumer's access plan.
    """
    metadata = dict(op.metadata)
    is_matmul = metadata.get("program_kind") == "matmul" and len(op.local_shape) >= 3
    if not op.deps and not (is_matmul and (source_dtype or metadata.get("input_format"))):
        return ()
    produced_by_dep = outputs_by_id or {}
    ports_by_dep = {}
    for entry in input_ports(op):
        producer = entry.get("op")
        if isinstance(producer, str):
            ports_by_dep.setdefault(producer, []).append(entry)

    repeats = metadata.get("input_repeats") or {}
    explicit = metadata.get("operand_read_bytes") or {}
    if not isinstance(repeats, dict):
        repeats = {}
    if not isinstance(explicit, dict):
        explicit = {}

    def requested(dependency):
        entries = ports_by_dep.get(dependency, ())
        for entry in entries:
            output = entry.get("output")
            if output and output not in {"activation_scales", "scale"}:
                return output
        return None

    def selected(dependency):
        available = list(produced_by_dep.get(dependency, ()))
        requested_value = requested(dependency)
        if requested_value:
            value = _select_named_value(op, dependency, requested_value, produced_by_dep)
            if value is not None:
                return value, requested_value
        for entry in ports_by_dep.get(dependency, ()):
            value = _select_named_value(op, dependency, entry.get("output"),
                                        produced_by_dep)
            if value is not None:
                return value, entry.get("output")
        return (available[0], available[0].get("output")) if available else (None, None)

    def add(value, *, role, field, shape=None, dependency="", repeats_value=1,
            explicit_bytes=None, expected_elements=None):
        value = _access_bytes(value, shape=shape, repeats=repeats_value,
                              explicit=explicit_bytes,
                              expected_elements=expected_elements)
        value.update({"role": role, "consumer": op.op_id,
                      "consumer_field": field, "producer": dependency or
                      value.get("producer", "")})
        return value

    if is_matmul:
        m, _, k = op.local_shape
        expected_rows = float(metadata.get("expected_rows", m))
        shape_rows = int(math.ceil(expected_rows)) if expected_rows > 0 else 0
        dependency = op.deps[0] if op.deps else ""
        primary, primary_output = selected(dependency) if dependency else (None, None)
        logical, storage = _row_shape(metadata, (shape_rows, k))
        expected_elements = (
            expected_rows * k
            if "expected_rows" in metadata and expected_rows != shape_rows
            else None
        )
        if primary is None:
            activation_dtype = str(
                (source_dtype if not produced_by_dep.get(dependency) and source_dtype else "")
                or metadata.get("input_format")
                or metadata.get("activation_format") or "")
            if not activation_dtype:
                return ()
            primary = typed_value(
                producer=dependency, dtype=activation_dtype, shape=logical,
                storage_shape=storage, output="activation", role="activation",
                source=f"{op.op_id}:activation_format",
                materialized=True, contract="declared_operand_value")
            primary_output = "activation"
        primary_repeats = repeats.get(primary_output, repeats.get(dependency, 1))
        explicit_primary = explicit.get(primary_output, explicit.get(dependency))
        values = [add(primary, role="activation", field=primary_output or "activation",
                      shape=logical, dependency=dependency,
                      repeats_value=primary_repeats, explicit_bytes=explicit_primary,
                      expected_elements=expected_elements)]
        seen_outputs = {primary_output}
        for entry in ports_by_dep.get(dependency, ()):
            if entry.get("output") in {"payload", "output", "activation", None}:
                continue
            if entry.get("output") in seen_outputs:
                continue
            seen_outputs.add(entry.get("output"))
            value = _select_named_value(op, dependency, entry.get("output"),
                                        produced_by_dep)
            if value is not None:
                values.append(add(value, role="input",
                                  field=entry.get("output") or "input",
                                  shape=tuple(value.get("logical_shape") or ()),
                                  dependency=dependency,
                                  repeats_value=repeats.get(entry.get("output"), 1),
                                  explicit_bytes=explicit.get(entry.get("output"))))
        scale = next((value for value in produced_by_dep.get(dependency, ())
                      if value.get("output") == "activation_scales"), None)
        if scale is None and metadata.get("activation_scale_format"):
            shape = tuple(metadata.get("activation_scale_shape") or ())
            scale = typed_value(
                producer=dependency,
                dtype=str(metadata["activation_scale_format"]), shape=shape,
                storage_shape=_row_shape(metadata, shape)[1],
                output="activation_scales", role="scale",
                source=f"{op.op_id}:activation_scale_format",
                materialized=True,
                write_bytes=metadata.get("activation_scale_bytes"),
                contract="declared_scale_operand")
        if scale is not None and "activation_scales" not in seen_outputs:
            values.append(add(scale, role="scale", field="activation_scales",
                              shape=tuple(scale.get("logical_shape") or ()),
                              dependency=dependency,
                              repeats_value=repeats.get("activation_scales", 1),
                              explicit_bytes=explicit.get("activation_scales")))
        weight_dtype = metadata.get("weight_format") or metadata.get("weight_dtype")
        if weight_dtype:
            shape = (op.local_shape[1], op.local_shape[2])
            weight = typed_value(
                producer="", dtype=str(weight_dtype), shape=shape,
                storage_shape=shape, output="weight", role="weight",
                source=f"{op.op_id}:weight_format", materialized=True,
                write_bytes=metadata.get("weight_bytes")
                or metadata.get("packed_weight_bytes"),
                contract="parameter_value")
            values.append(add(weight, role="weight", field="weight", shape=shape,
                              dependency="", repeats_value=repeats.get("weight", 1),
                              explicit_bytes=explicit.get("weight")))
        return tuple(values)

    values = []
    seen = set()
    for dependency in op.deps:
        entries = ports_by_dep.get(dependency, ())
        selected_entries = []
        for entry in entries:
            output = entry.get("output")
            value = _select_named_value(op, dependency, output, produced_by_dep)
            if value is not None:
                selected_entries.append((value, output))
        if not selected_entries:
            value, output = selected(dependency)
            if value is not None:
                selected_entries.append((value, output))
        for value, output in selected_entries:
            key = (dependency, output, value.get("value_id"))
            if key in seen:
                continue
            seen.add(key)
            values.append(add(
                value, role="input", field=output or "input",
                shape=tuple(value.get("logical_shape") or ()),
                dependency=dependency,
                repeats_value=repeats.get(output, repeats.get(dependency, 1)),
                explicit_bytes=explicit.get(output, explicit.get(dependency))))
        if selected_entries:
            continue
        activation_dtype = str(
            (source_dtype if dependency not in produced_by_dep and source_dtype else "")
            or metadata.get("input_format") or op.dtype or "")
        if not activation_dtype:
            continue
        logical, storage = _row_shape(metadata, tuple(op.local_shape))
        fallback = typed_value(
            producer=dependency, dtype=activation_dtype, shape=logical,
            storage_shape=storage, output="activation", role="activation",
            source=f"{op.op_id}:input_format", materialized=True,
            contract="declared_operand_value")
        values.append(add(fallback, role="input", field="activation",
                          shape=logical, dependency=dependency,
                          repeats_value=repeats.get(dependency, 1),
                          explicit_bytes=explicit.get(dependency)))
    return tuple(values)


def capture_partition(ops, runtime, replay_scope=None):
    """Predict the capture split before transforms run.

    ``GraphPolicy.apply`` normally runs after generic fusion.  The transforms
    need the same split rule now so a split (eager) family is not folded into
    a captured neighbour and then relabelled as a fused captured node.
    """
    if replay_scope == "none":
        return None
    if replay_scope == "full":
        return None
    mode = str(getattr(runtime, "graph_mode", "auto")).lower()
    if mode in {"none", "off", "eager", "disabled"}:
        return None
    phases = {op.phase for op in ops if op.phase}
    if replay_scope is None:
        full = (str(getattr(runtime, "graph_capture_mode", "full")) == "full"
                and not bool(getattr(runtime, "invalid_full_modes", False))
                and phases <= {"decode"})
        if full:
            return None
        if not (phases <= {"decode"}
                or str(getattr(runtime, "mixed_mode", "none")) not in {"none", "off"}):
            return None
        if not tuple(getattr(runtime, "piecewise_capture_keys", ())):
            return None
    split = set(getattr(runtime, "splitting_ops", None) or {
        "attention", "attention_qk", "attention_pv", "attention_softmax",
        "attention_merge",
    })
    state, segment, last_stage, last_captured = {}, 0, None, False
    for op in ops:
        captured = bool(replay_scope == "full"
                        or (op.family not in split and op.recipe not in split))
        if op.stage != last_stage or captured != last_captured:
            segment += 1
        state[op.op_id] = (captured, segment)
        last_stage, last_captured = op.stage, captured
    return state


def capture_fusion_overlap(ops, *, kind):
    """Keep capture, fusion and overlap as separate declared dimensions."""
    records = [dict(op.metadata) for op in ops]

    def common(key):
        values = tuple(record.get(key) for record in records)
        return values[0] if values and all(value == values[0] for value in values) else "mixed"

    streams = sorted({op.stream or "main" for op in ops})
    return {
        "capture": {
            "graph_captured": common("graph_captured"),
            "graph_segment": common("graph_segment"),
            "declared": any(record.get("graph_captured") is not None
                            for record in records),
        },
        "fusion": {"kind": kind, "members": len(ops),
                   "member_occurrences": tuple(op.op_id for op in ops)},
        "overlap": {
            "streams": tuple(streams),
            "cross_stream": len(streams) > 1,
            "declared_concurrency": "not_implied_by_fusion",
        },
    }


def audit_records(op):
    """Return the identity-bearing transform/port audit records.

    These records live inside ``fused_origin_scopes``, which is a structural
    annotation rather than a physical-work axis.  Keeping exact occurrence and
    consumer ids there lets the relation be exported without making repeated
    layers look physically different.
    """
    for key, value in reversed(op.metadata):
        if key == "fused_origin_scopes":
            return tuple(value) if value else ()
    return ()


def audit_boundary(op):
    for record in audit_records(op):
        if record.get("kind") in {"transform_boundary", "native_fusion_boundary",
                                  "activation_materialization"}:
            boundary = record.get("boundary")
            if isinstance(boundary, dict):
                return boundary
    return {}


def input_ports(op):
    result = []
    for record in audit_records(op):
        if record.get("kind") != "input_ports":
            continue
        for entry in record.get("ports", ()):
            if entry not in result:
                result.append(entry)
    return tuple(result)


_PRODUCER_FIELDS = {
    "activation_scale_producer": "activation_scales",
    "input_scale_producer": "activation_scales",
    "activation_payload_producer": None,
    "payload_producer": None,
    "topk_ids_producer": "topk_ids",
    "topk_weights_producer": "topk_weights",
    "inverse_index_producer": "inverse_index",
}


def _op_producer_refs(op):
    refs = set(op.deps)
    for record in audit_records(op):
        if record.get("kind") == "input_ports":
            for entry in record.get("ports", ()):
                target = entry.get("op")
                if target:
                    refs.add(target)
    for key, value in op.metadata:
        if key in _PRODUCER_FIELDS and isinstance(value, str):
            refs.add(value)
    return refs


_LAST_ENTRY = -1; _FIRST_SLOT = 0; _SINGLE_DEP = 1


def consumer_index(ops):
    """Index candidate consumers by producer op_id while preserving ``ops`` order."""
    by_producer = {}
    by_id = {}
    for index, op in enumerate(ops):
        by_id[op.op_id] = op
        item = (index, op)
        for producer in _op_producer_refs(op):
            bucket = by_producer.get(producer)
            if bucket is None:
                by_producer[producer] = [item]
            elif bucket[_LAST_ENTRY][_FIRST_SLOT] != index:
                bucket.append(item)
    return by_id, by_producer


def consumer_value_uses(op, member_ids, values_by_id):
    """Resolve the actual ``(member, output)`` values an external reader uses.

    Explicit ports are authoritative.  Producer-reference metadata is used
    next, and a bare dependency falls back to that producer's primary output.
    A reader of one member output never implicitly marks the member's other
    returns external.
    """
    if not isinstance(member_ids, (set, frozenset)):
        member_ids = set(member_ids)
    if not member_ids:
        return ()
    ports = input_ports(op)
    has_port_hit = any(entry.get("op") in member_ids for entry in ports)
    has_prod_hit = any(
        key in _PRODUCER_FIELDS and isinstance(value, str) and value in member_ids
        for key, value in op.metadata
    )
    has_dep_hit = any(dependency in member_ids for dependency in op.deps)
    if not has_port_hit and not has_prod_hit and not has_dep_hit:
        return ()
    metadata = dict(op.metadata) if has_prod_hit else None
    uses = []

    def names(member):
        return {value.get("output") for value in values_by_id.get(member, ())}

    def add(member, requested, *, fallback=True):
        if member not in member_ids:
            return
        available = names(member)
        if requested and requested in available:
            output = requested
        elif (requested in {None, "payload", "output", "activation"}
              and values_by_id.get(member)):
            output = values_by_id[member][0].get("output")
        elif fallback and values_by_id.get(member):
            output = values_by_id[member][0].get("output")
        else:
            return
        if not output:
            return
        pair = (member, output)
        if pair not in uses:
            uses.append(pair)

    if has_port_hit:
        for entry in ports:
            add(entry.get("op"), entry.get("output"), fallback=False)
    if metadata is not None:
        for field, requested in _PRODUCER_FIELDS.items():
            producer = metadata.get(field)
            if isinstance(producer, str):
                add(producer, requested, fallback=requested is None)
    if has_dep_hit:
        for dependency in op.deps:
            if dependency in member_ids and not any(
                    member == dependency for member, _ in uses):
                add(dependency, None)
    return tuple(uses)


def external_value_plan(ops, members, values_by_id, public_outputs=(), *,
                        consumer_map=None):
    """Exact externally-live values for one selected fusion group.

    ``ops`` includes the region around the group.  A member with no consumer
    at all is a call/region public root: all of its produced values remain
    public even when no node follows it (independent native/subgraph calls).
    Consumers inside the group are internal edges, not public readers.
    """
    member_ids = {member.op_id for member in members}
    specs, consumers = [], []
    consumed_members = set()
    if consumer_map is not None:
        by_id, by_producer = consumer_map
        cand_by_pos = {}
        for m_id in member_ids:
            for pos, op in by_producer.get(m_id, ()):
                if op.op_id not in member_ids:
                    cand_by_pos[pos] = op
        outside = [cand_by_pos[pos] for pos in sorted(cand_by_pos)]
        group_ops = [by_id[m_id] for m_id in member_ids if m_id in by_id]
    else:
        outside = [op for op in ops if op.op_id not in member_ids]
        group_ops = [op for op in ops if op.op_id in member_ids]
    for op in outside:
        for member_id, output in consumer_value_uses(op, member_ids, values_by_id):
            consumed_members.add(member_id)
            pair = (member_id, output)
            if pair not in specs:
                specs.append(pair)
            entry = {"consumer": op.op_id, "member": member_id,
                     "output": output}
            if entry not in consumers:
                consumers.append(entry)
    public_refs = set()
    for entry in public_outputs or ():
        if isinstance(entry, str):
            member_id, output = entry.rsplit(".", 1) if "." in entry else (entry, "output")
        else:
            member_id, output = entry[:2]
        if member_id in member_ids:
            public_refs.add((member_id, output))
            if (member_id, output) not in specs:
                specs.append((member_id, output))
    internal_uses = [
        (op.op_id, {m_id for m_id, _ in consumer_value_uses(op, member_ids, values_by_id)})
        for op in group_ops
    ]
    for member in members:
        member_public = {pair for pair in public_refs if pair[0] == member.op_id}
        if member_public:
            for pair in member_public:
                if pair not in specs:
                    specs.append(pair)
            continue
        if (member.op_id in consumed_members
                or any(member.op_id in used for op_id, used in internal_uses
                       if op_id != member.op_id)):
            continue
        for value in values_by_id.get(member.op_id, ()):
            pair = (member.op_id, value["output"])
            if pair not in specs:
                specs.append(pair)
    counts = {}
    for _, output in specs:
        counts[output] = counts.get(output, 0) + 1
    seen = {}
    port_for = {}
    values = []
    for member_id, output in specs:
        value = next((item for item in values_by_id.get(member_id, ())
                      if item.get("output") == output), None)
        if value is None:
            continue
        duplicate = counts[output]
        if duplicate == 1:
            port = output
        else:
            ordinal = seen.get(output, 0)
            seen[output] = ordinal + 1
            port = f"{output}__{ordinal}"
        port_for[(member_id, output)] = port
        descriptor = {**value, "member": member_id, "output": output,
                      "port": port,
                      "materialized": bool(value.get("materialized", True))}
        values.append(descriptor)
    member_ids_out = tuple(dict.fromkeys(member_id for member_id, _ in specs))
    anchor = members[-1].op_id if members else None
    anchor_primary = (values_by_id.get(anchor) or [{}])[0].get("output")
    explicit_public = {(value["member"], value["output"])
                       for value in values
                       if (value["member"], value["output"]) in public_refs}
    returns = tuple(
        value["port"] for value in values
        if (value["member"], value["output"]) in explicit_public
        or not (value["member"] == anchor and value["output"] == anchor_primary)
    )
    return {
        "member_ids": member_ids_out,
        "values": tuple(values),
        "references": tuple((value["member"], value["output"]) for value in values),
        "ports": port_for,
        "returns": tuple(dict.fromkeys(returns)),
        "consumers": tuple(consumers),
    }


def append_audit(metadata, *, kind, op_id, boundary=None, ports=(),
                 layer=None, stage=None, block=None, role=None):
    """Append/replace one audit record without adding a physical axis."""
    origins = [record for record in metadata.get("fused_origin_scopes", ())
               if not (record.get("kind") == kind and record.get("op_id") == op_id)]
    record = {"op_id": op_id, "kind": kind, "layer": layer,
              "stage": stage, "block": block, "role": role}
    if boundary is not None:
        record["boundary"] = boundary
    if ports:
        record["ports"] = tuple(ports)
    origins.append(record)
    metadata["fused_origin_scopes"] = tuple(origins)


def set_input_ports(op, entries):
    """Replace named consumer ports, including removal of revoked producers."""
    entries = tuple(entry for entry in entries if entry)
    metadata = dict(op.metadata)
    origins = [record for record in metadata.get("fused_origin_scopes", ())
               if not (record.get("kind") == "input_ports"
                       and record.get("op_id") == op.op_id)]
    metadata["fused_origin_scopes"] = tuple(origins)
    op = replace(op, metadata=tuple(metadata.items()))
    return add_input_ports(op, entries)


def add_input_ports(op, entries):
    """Attach named output ports to an existing consumer edge."""
    entries = tuple(entry for entry in entries if entry)
    if not entries:
        return op
    metadata = dict(op.metadata)
    existing = list(input_ports(op))
    merged = list(existing)
    for entry in entries:
        if entry not in merged:
            merged.append(entry)
    if merged == existing:
        return op
    append_audit(metadata, kind="input_ports", op_id=op.op_id, ports=merged,
                 layer=op.layer, stage=op.stage,
                 block=dict(op.metadata).get("structure_block"),
                 role=dict(op.metadata).get("structure_role"))
    return replace(op, metadata=tuple(metadata.items()))


def replaced_occurrences(*ops):
    """Flatten any nested transform replacement into the current occurrence set."""
    result = []
    for op in ops:
        boundary = audit_boundary(op) or dict(op.metadata).get(
            "transform_boundary", {})
        values = (boundary.get("replaced_occurrences", (op.op_id,))
                  if isinstance(boundary, dict) else (op.op_id,))
        for value in values:
            if value not in result:
                result.append(value)
    return tuple(result)


def nested_boundary(op):
    boundary = audit_boundary(op) or dict(op.metadata).get("transform_boundary")
    return boundary if isinstance(boundary, dict) else {}


def member_contracts(ops):
    """Preserve each replaced member's own typed fields and provenance."""
    return tuple({
        "op_id": op.op_id,
        "family": op.family,
        "recipe": op.recipe,
        "dtype": op.dtype,
        "layout": op.layout,
        "returns": list(op.returns),
        "metadata": dict(op.metadata),
    } for op in ops)


def semantic_attrs(ops, keys):
    attrs = []
    for op in ops:
        metadata = dict(op.metadata)
        for key in keys:
            if key in metadata:
                attrs.append({"name": key, "value": metadata[key],
                              "unit": "", "source": op.op_id})
    return tuple(attrs)


def attach_quantized_ports(ops, replacements):
    """Name payload and scale ports for consumers of a fused quantizer.

    A matrix consumer reads both the low-precision payload and its scale; an
    unrelated norm/residual consumer is left to its already named port.
    """
    result = list(ops)
    for index, op in enumerate(result):
        metadata = dict(op.metadata)
        if not (metadata.get("activation_scale_bytes")
                or metadata.get("activation_scale_in_mte2")):
            continue
        entries = []
        for dep in op.deps:
            if dep not in replacements:
                continue
            entries.extend((
                {"op": dep, "output": "payload", "source_op": dep},
                {"op": dep, "output": "activation_scales", "source_op": dep},
            ))
        if entries:
            result[index] = add_input_ports(op, entries)
    return result


def _move_access_slots(metadata, uses, port_map, aliases):
    """Move consumer-slot repeats/issued bytes when the producer port changes."""
    changed = False
    for key in ("input_repeats", "operand_read_bytes"):
        mapping = metadata.get(key)
        if not isinstance(mapping, dict):
            continue
        moved = dict(mapping)
        for member, output in uses:
            target = port_map.get((member, output))
            if target is None:
                continue
            for old_key in (member, output):
                if old_key in moved:
                    moved.setdefault(target[1], moved.pop(old_key))
                    changed = True
        if moved != mapping:
            metadata[key] = moved
    return changed


def rewire_value_graph(ops, *, replacements, aliases, removed, port_map,
                      values_by_id):
    """One replacement result for ordinary nodes and replacement nodes.

    ``port_map`` maps ``(old producer, output)`` to ``(new producer, port,
    value)``.  The same pass rewrites deps, named ports, producer references
    and unrelated ports; a bare primary edge is implicit only when it is the
    sole old-primary edge folded into one replacement.
    """
    member_ids = set(aliases)
    values_by_id = values_by_id or {}
    replacement_ports = {}
    for (_, _), target in port_map.items():
        replacement_ports.setdefault(target[0], set()).add(target[1])

    def rewire_node(op):
        has_dep_alias = any(dep in aliases for dep in op.deps)
        old_ports = input_ports(op)
        has_port_alias = any(entry.get("op") in aliases for entry in old_ports)
        has_prod_alias = any(
            key in _PRODUCER_FIELDS and isinstance(value, str) and value in aliases
            for key, value in op.metadata
        )
        if not has_dep_alias and not has_port_alias and not has_prod_alias:
            if len(op.deps) > _SINGLE_DEP:
                dedup = tuple(dict.fromkeys(op.deps))
                if dedup != op.deps:
                    return replace(op, deps=dedup)
            return op
        deps = tuple(dict.fromkeys(aliases.get(dep, dep) for dep in op.deps))
        uses = consumer_value_uses(op, member_ids, values_by_id)
        existing = [entry for entry in old_ports
                    if entry.get("op") not in aliases]
        explicit_old = {entry.get("op") for entry in old_ports}
        producer_fields = {value for value in dict(op.metadata).values()
                           if isinstance(value, str)}
        target_counts = {}
        for member, output in uses:
            target = port_map.get((member, output))
            if target is not None:
                target_counts[target[0]] = target_counts.get(target[0], 0) + 1
        mapped = []
        for member, output in uses:
            target = port_map.get((member, output))
            if target is None:
                continue
            primary = (values_by_id.get(member) or [{}])[0].get("output")
            folded = (target_counts.get(target[0], 0) > 1
                      or len(replacement_ports.get(target[0], ())) > 1)
            if (output == primary and member not in explicit_old
                    and member not in producer_fields and not folded):
                continue
            value = target[2] if len(target) > 2 else {}
            entry = {"op": target[0], "output": target[1], "source_op": member}
            for key in ("state_version", "binding", "binding_name", "quant",
                        "role", "access_shape", "repeats", "read_bytes",
                        "base_read_bytes", "storage", "alias_of"):
                if value.get(key) not in (None, ""):
                    entry[key] = (_json_identity(value[key], "")
                                  if key in {"state_version", "binding",
                                             "binding_name"} else value[key])
            if entry not in existing and entry not in mapped:
                mapped.append(entry)
        metadata = dict(op.metadata)
        changed = _move_access_slots(metadata, uses, port_map, aliases)
        for field, requested in _PRODUCER_FIELDS.items():
            producer = metadata.get(field)
            if producer not in aliases:
                continue
            primary = (values_by_id.get(producer) or [{}])[0].get("output")
            target = port_map.get((producer, requested or primary))
            if target is None:
                continue
            metadata[field] = target[0]
            metadata[f"{field}_output"] = target[1]
            changed = True
        rewired = replace(op, deps=deps,
                          metadata=tuple(metadata.items()) if changed else op.metadata)
        ports = tuple(existing + mapped)
        if existing != list(old_ports) or mapped:
            return set_input_ports(rewired, ports)
        return rewired

    result = []
    for op in ops:
        if op.op_id in replacements:
            result.append(rewire_node(replacements[op.op_id]))
            continue
        if op.op_id in removed:
            continue
        result.append(rewire_node(op))
    return result


def rewire_replacements(ops, *, replacements, aliases, removed, ports):
    """Apply one shared value map to any number of generic replacements."""
    values_by_id = {}
    for candidate in ops:
        try:
            values_by_id[candidate.op_id] = produced_values(candidate)
        except ModelingError:
            values_by_id[candidate.op_id] = ()
    port_map = {}
    targets = {}
    for old_id in set(replacements) | set(removed):
        fused = replacements.get(old_id)
        if fused is None:
            fused = replacements.get(aliases.get(old_id))
        if fused is None:
            continue
        targets[old_id] = fused
        fused_values = {value["output"]: value for value in produced_values(fused)}
        requested = ports.get(old_id)
        for value in values_by_id.get(old_id, ()):
            if isinstance(requested, dict):
                target_name = requested.get(value["output"], value["output"])
            else:
                target_name = requested or value["output"]
            target = fused_values.get(target_name)
            if target is not None:
                port_map[(old_id, value["output"])] = (
                    fused.op_id, target["output"], target)
    return rewire_value_graph(
        ops, replacements=replacements, aliases=aliases,
        removed=set(removed), port_map=port_map, values_by_id=values_by_id)


def rewire_replacement(ops, *, replacement_key, fused, removed, ports):
    """Generic-transform adapter for the shared replacement mapping."""
    removed = set(removed)
    aliases = {key: replacement_key for key in removed}
    aliases[replacement_key] = replacement_key
    values_by_id = {}
    for candidate in ops:
        try:
            values_by_id[candidate.op_id] = produced_values(candidate)
        except ModelingError:
            values_by_id[candidate.op_id] = ()
    fused_values = {value["output"]: value for value in produced_values(fused)}
    port_map = {}
    for old in ops:
        if old.op_id != replacement_key and old.op_id not in removed:
            continue
        requested = ports.get(old.op_id)
        for value in values_by_id.get(old.op_id, ()):
            target_name = requested or value["output"]
            target = fused_values.get(target_name)
            if target is None and requested is None:
                target = next(iter(fused_values.values()), None)
            if target is not None:
                port_map[(old.op_id, value["output"])] = (
                    fused.op_id, target["output"], target)
    return rewire_value_graph(
        ops, replacements={replacement_key: fused}, aliases=aliases,
        removed=removed, port_map=port_map, values_by_id=values_by_id)
