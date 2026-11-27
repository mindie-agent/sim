"""Declared block and operator-family implementations.

[Role] Bind frontend implementation selection to lowering and physical recipes.
[Invariants] Selectors identify declared software and recipe coverage. A domain
  miss raises an error, and activation quantization precedes dependent lowering.
[Timeouts & Locks] Local planner imports on selection. No subprocesses or locks.
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
    "cann_bf16_lightning_indexer_v2_complete": Implementation(
        "mindie_sim.prediction.operators.sparse_indexer.bf16_lightning", blocks=("lightning_indexer",)),
    "cann_quant_lightning_indexer_v2_complete": Implementation(
        "mindie_sim.prediction.operators.sparse_indexer.family_quant_indexer", blocks=("quant_lightning_indexer",)),
    "vllm_ascend_group_list_prefix": Implementation(
        "mindie_sim.prediction.operators.moe_routing.moe_group_list", blocks=("group_list_prefix",)),
    "vllm_ascend_cann_keypool_caller": Implementation(
        "mindie_sim.prediction.operators.sparse_indexer.index_pool_native", blocks=("pool_key_indexer",),
        options=("score_mode", "tail_capacity", "block_table_columns")),
    "vllm_ascend_sparse_mla_metadata": Implementation(
        "mindie_sim.prediction.operators.sparse_indexer.sparse_mla_metadata", blocks=("sparse_mla_metadata",),
        options=("layer_groups", "schedule")),
    "ascend_moe_init_routing_v3_mxfp8": Implementation(
        "mindie_sim.prediction.operators.moe_routing.family_moe_init", blocks=("moe_init_routing",)),
    "triton_noncontiguous_conv_state_copy": Implementation(
        "mindie_sim.prediction.operators.causal_conv.conv_state_copy", blocks=("conv_state_copy",),
        options=("state_slots", "cache_index_dtype", "query_start_dtype", "joint_program_service")),
    "ascend_grouped_int32_dequant_swiglu": Implementation(
        "mindie_sim.prediction.operators.grouped_moe.family_grouped_quant", blocks=("grouped_int8_caller",),
        options=("dequant_program", "input_capacity_rows")),
    "vllm_ascend_mc2_unfused_w8a8": Implementation(
        "mindie_sim.prediction.operators.mc2.family_mc2_unfused", blocks=("grouped_int8_caller",),
        options=("global_bs",)),
    "ascend_prepared_mx32_matrix": Implementation(
        "mindie_sim.prediction.operators.mx_quant_matrix.family_mx_matrix", blocks=("mx_matrix",),
        options=("projection_recipes", "scale_storage")),
    "v41_partial_rotary": Implementation(
        "mindie_sim.prediction.operators.rotary.family_v41_rotary", blocks=("v41_rotary",)),
    "ascend_causal_conv1d_update": Implementation(
        "mindie_sim.prediction.operators.causal_conv.family_causal_conv", blocks=("causal_conv_update",)),
    "ascend_paged_mla_decode_arch35": Implementation(
        "mindie_sim.prediction.operators.paged_mla.family_paged_mla", blocks=("paged_mla_attention",),
        options=("kv_splits", "dispatch", "native_query_capacity")),
    "ascend_paged_mla_decode_arch22": Implementation(
        "mindie_sim.prediction.operators.paged_mla.family_paged_mla_membase", blocks=("paged_mla_attention",),
        options=("dispatch", "retained_kv_rows", "native_query_capacity")),
    "vllm_ascend_mla_head_padding": Implementation(
        "mindie_sim.prediction.operators.mla.mla_head_padding", blocks=("mla_head_padding",)),
    # CANN 9.2.1 official mixed-quant sparse MLA.
    "aclnn_mixed_quant_sparse_flash_mla": Implementation(
        "mindie_sim.prediction.operators.paged_mla.family_mqsmla", blocks=("mqsmla_attention",),
        options=("metadata_support_fd", "addressing", "return_softmax_lse")),
    "mixed_quant_sparse_flash_mla": Implementation(
        "mindie_sim.prediction.operators.paged_mla.family_mqsmla", blocks=("mqsmla_attention",),
        options=("metadata_support_fd", "addressing", "return_softmax_lse")),
    "vllm_ascend_cann_compressor": Implementation(
        "mindie_sim.prediction.operators.mla_prolog.compression", blocks=("compressor",)),
    "vllm_ascend_draft_context_prolog_v3": Implementation(
        "mindie_sim.prediction.operators.mla_prolog.draft_context", blocks=("draft_context",)),
    "vllm_ascend_kda_mixed_projection": Implementation(
        "mindie_sim.prediction.operators.kda.family_kda_projection", blocks=("kda_input_projection",)),
    "vllm_ascend_recurrent_kda_arch22": Implementation(
        "mindie_sim.prediction.operators.kda.family_kda_recurrent_membase", blocks=("kda",),
        options=("safe_gate", "use_beta_sigmoid", "gate_dtype", "beta_dtype",
                 "has_a_log", "has_dt_bias", "allow_neg_eigval", "state_selection")),
    "vllm_ascend_recurrent_kda_arch35": Implementation(
        "mindie_sim.prediction.operators.kda.family_kda_recurrent", blocks=("kda",),
        options=("safe_gate", "use_beta_sigmoid", "gate_dtype", "beta_dtype",
                 "has_a_log", "has_dt_bias", "allow_neg_eigval", "state_selection")),
    "vllm_ascend_fla_kda_recurrent": Implementation(
        "mindie_sim.prediction.operators.kda.family_kda_recurrent", blocks=("kda",),
        options=("safe_gate", "use_beta_sigmoid", "gate_dtype", "beta_dtype",
                 "has_a_log", "has_dt_bias", "allow_neg_eigval", "state_selection")),
    "vllm_ascend_gdn_decode": Implementation(
        "mindie_sim.prediction.operators.linear_attention.gdn_decode", blocks=("gdn_decode",),
        options=("gating_beta_store", "gating_programs", "recurrent_implementation", "cache_index_row_stride")),
    "ascend_sparse_flash_attention_arch22": Implementation(
        "mindie_sim.prediction.operators.paged_mla.indexed_attention", blocks=("indexed_attention",),
        options=("scratch_policy",)),
    "vllm_ascend_sharedkv_scfa": Implementation(
        "mindie_sim.prediction.operators.sharedkv.family_sharedkv", blocks=("sharedkv_attention",),
        options=("scratch_policy", "owner_schedule", "selected_rows", "sinks_dtype",
                 "metadata_s2_split", "return_softmax_lse")),
    "vllm_ascend_mxfp8_single_group_eager": Implementation(
        "mindie_sim.prediction.operators.matrix.matrix_wrappers", blocks=("matrix_wrapper",),
        options=("projection_recipes",)),
    "ascend_shared_gate_up_int32": Implementation(
        "mindie_sim.prediction.operators.mx_quant_matrix.family_shared_quant", blocks=("shared_expert",)),
    "vllm_ascend_native_norm": Implementation(
        "mindie_sim.prediction.operators.vector.family_ascend_norm",
        operators=("add_rmsnorm", "rmsnorm_quant", "add_rmsnorm_quant"),
        options=("has_bias", "external_gamma_shift")),
    "vllm_ascend_attn_res_prefix": Implementation(
        "mindie_sim.prediction.operators.hyper_connections.family_attnres", blocks=("attention_residual",),
        options=("optimize_prefill",)),
    "declared_native_fusion": Implementation(
        "mindie_sim.prediction.operators.hyper_connections.native_fusion", blocks=("native_fusion",),
        options=("roles", "traffic_profile", "program_profiles", "program_options")),
    "vllm_ascend_kda_in_kernel_gates": Implementation(
        "mindie_sim.prediction.operators.kda.family_kda", blocks=("kda",)),
    "vllm_ascend_kda_chunk_state": Implementation(
        "mindie_sim.prediction.operators.kda.family_kda_state", blocks=("kda",)),
    "vllm_ascend_kda_chunk_prepare_state": Implementation(
        "mindie_sim.prediction.operators.kda.family_kda_prepare", blocks=("kda",)),
    "vllm_ascend_mla_prolog_v3": Implementation(
        "mindie_sim.prediction.operators.mla_prolog.family_mla_prolog", blocks=("mla_prolog",),
        options=("cache_layout", "cache_group_size")),
    "gdn_prefill_arch35": Implementation(
        "mindie_sim.prediction.operators.linear_attention.gdn_prefill", blocks=("gdn_prefill",)),
    "ascendc_grouped_nd_bf16": Implementation(
        "mindie_sim.prediction.operators.grouped_moe.family_grouped_gemm", operators=("grouped_gemm",)),
    "ascend_w4a8_msd_grouped_down": Implementation(
        "mindie_sim.prediction.operators.mx_quant_matrix.w4a8_down", operators=("grouped_gemm",),
        after_activation_quantization=True),
    "mindie_sd_custom_ops": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=(
            "rainfusion", "ada_bsa", "eagle_bsa", "eagle_qbsa", "adalayernorm",
            "norm_rope_concat", "laurel", "quant_four_over_six_a5", "eagle_ffn",
            "mindie_sd_rainfusion", "mindie_sd_ada_bsa", "mindie_sd_eagle_bsa",
            "mindie_sd_eagle_qbsa", "mindie_sd_adalayernorm", "mindie_sd_norm_rope_concat",
            "mindie_sd_laurel", "mindie_sd_quant_four_over_six_a5", "mindie_sd_eagle_ffn",
            "indexed_attention", "gqa_attention", "rmsnorm", "add_rmsnorm",
            "elementwise", "quantize", "gemm", "ffn",
        ),
        options=("op_name", "soc_family", "sparsity_ratio", "low_rank", "block_size", "kv_dtype")),
    "mindie_sd_rainfusion": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("rainfusion", "mindie_sd_rainfusion", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size")),
    "mindie_sd_ada_bsa": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("ada_bsa", "mindie_sd_ada_bsa", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size")),
    "mindie_sd_eagle_bsa": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("eagle_bsa", "mindie_sd_eagle_bsa", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size")),
    "mindie_sd_eagle_qbsa": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("eagle_qbsa", "mindie_sd_eagle_qbsa", "indexed_attention", "gqa_attention"),
        options=("op_name", "soc_family", "sparsity_ratio", "block_size", "kv_dtype")),
    "mindie_sd_adalayernorm": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("adalayernorm", "mindie_sd_adalayernorm", "rmsnorm", "add_rmsnorm"),
        options=("op_name", "soc_family")),
    "mindie_sd_norm_rope_concat": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("norm_rope_concat", "mindie_sd_norm_rope_concat", "rmsnorm", "elementwise"),
        options=("op_name", "soc_family")),
    "mindie_sd_laurel": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("laurel", "mindie_sd_laurel", "gemm", "elementwise"),
        options=("op_name", "soc_family", "low_rank")),
    "mindie_sd_quant_four_over_six_a5": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5", "quantize", "elementwise"),
        options=("op_name", "soc_family")),
    "mindie_sd_eagle_ffn": Implementation(
        "mindie_sim.prediction.operators.diffusion",
        operators=("eagle_ffn", "mindie_sd_eagle_ffn", "gemm", "ffn"),
        options=("op_name", "soc_family")),
    "vllm_ascend_sparse_kv_offload": Implementation(
        "mindie_sim.prediction.operators.sparse_indexer.sparse_kv_offload",
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


_GEOMETRY = {"valid_tokens", "padded_tokens", "sequences", "min_query_tokens", "max_query_tokens",
             "max_context_tokens", "min_context_tokens"}
_STATIC_SELECTORS = {"mla_prolog", "mx_matrix"}
_CATEGORIES = {"phases": {"prefill", "decode"},
               "replay_scopes": {"none", "full", "piecewise"},
               "model_scopes": {"target", "draft"}}


def is_policy(raw):
    return isinstance(raw, dict) and "cases" in raw


def _validate_domain(domain, selector):
    if not domain or set(domain) - (_GEOMETRY | _CATEGORIES.keys()):
        raise ModelingError(f"implementation_domain_unknown_or_empty:{selector}")
    for key, value in domain.items():
        if key in _CATEGORIES:
            allowed = _CATEGORIES[key]
            if (not isinstance(value, (tuple, list)) or not value
                    or set(value) - allowed or len(set(value)) != len(value)):
                raise ModelingError(f"implementation_domain_invalid:{selector}:{key}")
        elif (not isinstance(value, dict) or not value or set(value) - {"min", "max"}
              or any(type(bound) is not int or bound < 0 for bound in value.values())
              or value.get("min", 0) > value.get("max", float("inf"))):
            raise ModelingError(f"implementation_domain_invalid:{selector}:{key}")


def alternatives(raw, selector):
    """Enumerate all declared recipe providers without executing their planners."""
    if not is_policy(raw):
        yield raw
        return
    if selector in _STATIC_SELECTORS and not all(
            isinstance(case, dict) and set(case.get("domain") or ()) == {"model_scopes"}
            for case in (raw.get("cases") or ())):
        raise ModelingError(f"implementation_policy_requires_static_model_preparation:{selector}")
    if set(raw) - {"cases", "otherwise"} or not raw["cases"]:
        raise ModelingError(f"implementation_policy_requires_cases:{selector}")
    for case in raw["cases"]:
        if set(case) != {"domain", "binding"}:
            raise ModelingError(f"implementation_case_requires_domain_and_binding:{selector}")
        _validate_domain(case["domain"], selector)
        if is_policy(case["binding"]) or case["binding"] == "error":
            raise ModelingError(f"implementation_case_requires_leaf_binding:{selector}")
        yield case["binding"]
    otherwise = raw.get("otherwise", "error")
    if is_policy(otherwise):
        raise ModelingError(f"implementation_case_requires_leaf_binding:{selector}")
    if otherwise != "error":
        yield otherwise


def implementation(name):
    return import_module(IMPLEMENTATIONS[name].module)


def select(bindings, selector):
    """Resolve one software declaration, independent of shape and measurements."""
    raw = bindings.get(selector)
    if raw is None:
        return None
    if is_policy(raw):
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
    if isinstance(name, str) and name.startswith("mindie_sd_") and name != "mindie_sd_custom_ops":
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
