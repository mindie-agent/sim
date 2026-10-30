"""Explicit block and operator-family implementation registry.

[Role] Couple a frontend implementation's lowering with its physical recipes.
[Invariants] Selection is a declared software implementation, never a model
  label, hardware name or measurement lookup. One binding serves both paths.
  Recipe selectors narrow declared coverage; selected domain misses are errors.
[Timeouts & Locks] Small local module imports; no subprocesses or locks.
[Side-Effect Policy] No device contexts, configuration writes or source reads.
"""
from dataclasses import dataclass, replace
from importlib import import_module

from mindie_sim.schema.common import ModelingError


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
    "ascend_prepared_mx32_matrix": Implementation(
        "mindie_sim.prediction.family_mx_matrix", blocks=("mx_matrix",),
        options=("projection_recipes", "scale_storage")),
    "v41_partial_rotary": Implementation(
        "mindie_sim.prediction.family_v41_rotary", blocks=("v41_rotary",)),
    "ascend_causal_conv1d_update": Implementation(
        "mindie_sim.prediction.family_causal_conv", blocks=("causal_conv_update",)),
    "ascend_paged_mla_decode_arch35": Implementation(
        "mindie_sim.prediction.family_paged_mla", blocks=("paged_mla_attention",),
        options=("kv_splits", "dispatch")),
    "ascend_paged_mla_decode_arch22": Implementation(
        "mindie_sim.prediction.family_paged_mla_arch22", blocks=("paged_mla_attention",),
        options=("dispatch", "retained_kv_rows", "native_query_capacity")),
    "vllm_ascend_mla_head_padding": Implementation(
        "mindie_sim.prediction.mla_head_padding", blocks=("mla_head_padding",)),
    "cannbot_mixed_quant_sparse_flash_mla": Implementation(
        "mindie_sim.prediction.family_mqsmla", blocks=("mqsmla_attention",),
        options=("metadata_support_fd", "addressing", "return_softmax_lse")),
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
    """A physical replacement keeps callers, outputs and graph placement."""
    return replace(
        source, op_id=semantic.op_id, family=semantic.family, phase=semantic.phase,
        stage=semantic.stage, layer=semantic.layer, requests=semantic.requests,
        deps=semantic.deps, stream=semantic.stream, semantic_shape=semantic.semantic_shape,
        metadata=tuple({**dict(source.metadata), "semantic_recipe": semantic.recipe,
                        "semantic_metadata": dict(semantic.metadata)}.items()),
        notes=semantic.notes + source.notes,
    )
