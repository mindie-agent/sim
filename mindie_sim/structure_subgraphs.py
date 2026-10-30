"""Export declared fusion groups without adding execution nodes.

[Role] Resolve subgraph membership after ordinary lowering and call expansion.
[Invariants] The final DAG defines inputs and externally visible outputs.
  Quantization nodes remain inside or outside according to actual dependencies.
  A group carries one candidate contract, not a second copy of member work.
[Timeouts & Locks] Linear in exported nodes plus finite group slices; no I/O.
[Side-Effect Policy] Pure dictionaries; no profiler or device imports.
"""
from collections import defaultdict

from .structure_matrix_contracts import mxfp4_reference_coverage
from .structure_native_contracts import grouped_situ_contract


def metadata(op):
    meta = dict(op.metadata)
    return {**meta.get("semantic_metadata", {}), **meta}


def layer_structure(model, layer):
    plans = model.get("layer_plans", ())
    if layer < 0 or layer >= len(plans):
        return {}
    plan = plans[layer]
    return {
        "attention_kind": plan["attention"],
        "attention_architecture": plan.get("architecture"),
        "ffn_kind": plan["ffn"],
        "residual_kind": (
            "mhc" if plan.get("hyper_connections") else
            "attention_residual" if plan.get("residual_block_size") else "residual"),
        "cache_kind": plan.get("cache_kind"),
        "compress_ratio": plan.get("compress_ratio", 0),
        "window": plan.get("window", 0),
        "indexer": plan.get("indexer", False),
        "owns_topk": plan.get("owns_topk", False),
        "index_query_per_layer": plan.get("index_query_per_layer", False),
        "topk_owner_layer_id": plan.get("topk_owner_layer_id"),
    }


def subgraph_contract(declaration):
    if declaration.get("coverage", {}).get("kind") == "multi_launch_stage_unknown":
        return None
    view, role = declaration["tensor_view"], declaration["role"]
    if role == "grouped_gate_up_situ_quant":
        return grouped_situ_contract(view)
    if role == "situ_quant":
        rows, width = view["rows"], view["width"]
        if (rows != int(rows) or width % 64 or view["input_dtype"] != "bf16"
                or view["output_dtype"] not in {"fp8_e4m3", "fp8_e5m2"}
                or view["scale_dtype"] != "fp8_e8m0"
                or tuple(view["scale_shape"]) != (rows, width // 64, 2)):
            return None
        return (
            "situ_mxquant_active", "SituMxQuant",
            ((int(rows), 2 * width), view["input_dtype"], "ND"),
            ((int(rows), width), view["output_dtype"], "ND"),
            ((int(rows), width // 64, 2), view["scale_dtype"], "ND"),
        )
    if role == "router_selection":
        rows, experts, topk = (view[key] for key in ("tokens", "experts", "topk"))
        dtype = view["dtype"]
        return (
            "router_selection_active", view["native_kind"], view["selection_mode"],
            ((rows, experts), dtype, "ND"), ((rows, topk), dtype, "ND"),
            ((rows, topk), "int32", "ND"),
        )
    if role == "lightning_indexer":
        rows, heads, width = (view[key] for key in ("tokens", "heads", "width"))
        if view["query_dtype"] not in {"bf16", "fp16"}:
            return None
        return (
            "lightning_indexer_active", "LightningIndexer",
            (((rows, heads, width), view["query_dtype"], "ND"),
             ((rows, heads), view["weight_dtype"], "ND")),
            ((1, width), view["cache_dtype"], "ND"),
            ((rows, 1, view["topk"]), view["selection_output_dtype"], "ND"),
        )
    if role == "mla_prolog":
        rows, hidden, heads, qr, latent, nope, rope = (
            view[key] for key in
            ("tokens", "hidden", "heads", "query_rank", "latent_rank", "nope_width", "rope_width"))
        dq, uq, dkv = view["projection_weight_dtypes"]
        norm, cache = view["norm_dtype"], view["cache_dtype"]
        query_norm = (((rows, qr), view["query_norm_dtype"], "ND")
                      if view["query_norm_output"] else ((), None, None))
        cache_contract = (((1, latent), cache, "ND"), ((1, rope), cache, "ND"))
        if "packed_cache" in view:
            cache_contract = (((1, view["packed_cache"]["row_bytes"]), "int8", "ND"),
                              ((), None, None))
        return (
            "mla_prolog_active", "MlaPrologV3",
            ((rows, hidden), view["token_dtype"], "ND"),
            (((hidden, qr), dq), ((qr, heads * (nope + rope)), uq),
             ((heads, nope, latent), view["absorbed_weight_dtype"]),
             ((hidden, latent + rope), dkv), ((qr,), norm), ((latent,), norm)),
            cache_contract,
            (((rows, heads, latent), norm, "ND"),
             ((rows, heads, rope), norm, "ND"), query_norm),
        )
    if role == "quant_lightning_indexer" and view["query_dtype"] == "int8":
        rows, heads, width = (view[key] for key in ("tokens", "heads", "width"))
        return (
            "quant_lightning_indexer_active", "VllmAscendQuantLightningIndexerV2",
            (((rows, heads, width), view["query_dtype"], "ND"),
             ((rows, heads), view["weight_dtype"], "ND"),
             ((rows, heads), view["scale_dtype"], "ND")),
            (((1, width), view["cache_dtype"], "ND"),
             ((1,), view["scale_dtype"], "ND")),
            ((rows, 1, view["topk"]), view["selection_output_dtype"], "ND"),
        )
    if role in {"hc_pre", "hc_post"}:
        rows, hidden, hc, dtype = (view[key] for key in ("tokens", "hidden", "hc", "dtype"))
        x, out = ((rows, hc, hidden), dtype, "ND"), ((rows, hidden), dtype, "ND")
        post, comb = ((rows, hc), "fp32", "ND"), ((rows, hc, hc), "fp32", "ND")
        if role == "hc_post":
            return ("hyper_connection_active", "HcPost", (out, x, post, comb), (x,))
        mixed = view["mixed"]
        return (
            "hyper_connection_active", "HcPre",
            (x, ((mixed, hc * hidden), "fp32", "ND"),
             ((3,), "fp32", "ND"), ((mixed,), "fp32", "ND")),
            (out, post, comb),
        )
    if role == "compressor":
        rows, hidden, width, expansion = (
            view[key] for key in ("tokens", "hidden", "width", "coff"))
        slot = lambda shape, dtype: (shape, dtype, "ND")
        return (
            "compressor_active", "Compressor", slot((rows, hidden), view["dtype"]),
            (slot((expansion * width, hidden), view["weight_dtype"]),
             slot((expansion * width, hidden), view["weight_dtype"]),
             slot((view["ratio"], expansion * width), view["ape_dtype"]),
             slot((width,), view["norm_dtype"])),
            slot((2 * expansion * width,), view["state_dtype"]),
            (slot((view["rope_width"],), view["rope_dtype"]),) * 2,
            slot((width,), view["dtype"]),
        )
    if role == "grouped_gate_up_activation_quant":
        weight, activation = view["weight_dtype"], view["input_dtype"]
        kind = ("GroupedMatmulSwigluQuantV2" if weight == "int4" else
                "GroupedMatmulSwigluQuant" if weight == "int8" else None)
        if kind and activation == view["output_dtype"] == "int8":
            return (
                "grouped_mlp_parameters_active", kind,
                (view["local_experts"], view["input_width"], view["intermediate_width"]),
                (activation, weight, view["output_dtype"]),
            )
    if role == "kda_recurrent" and view.get("in_kernel_gate_source"):
        rows, heads, width = (view[key] for key in ("tokens", "heads", "value_width"))
        dtype = view["dtype"]
        return (
            "recurrent_kda_parameters_active", "RecurrentKda",
            (((1, rows, heads, width), dtype, "ND"),
             ((1, rows, heads), view["beta_dtype"], "ND"),
             ((heads,), "fp32", "ND"), ((heads * width,), "fp32", "ND")),
            ((dtype, "ND"),) * 3, (view["state_dtype"], "ND"), (dtype, "ND"),
        )
    return None


def subgraph_scope(role):
    if role == "grouped_gate_up_situ_quant":
        return {
            "kind": "source_verified_MX_A8W4_expert_bank_and_channel_precision",
            "not_compared": ["routed_rows", "group_list_values", "beta", "linear_beta",
                             "native_dispatch"],
            "work_assumption": "expected_routing_work_is_declared_not_measured",
            "excluded_semantic_work": ["input_quantization", "routing",
                                       "down_projection", "combine"],
            "occurrence_identity": False,
            "prediction_coverage": mxfp4_reference_coverage(),
        }
    if role == "situ_quant":
        return ("active_input_output_MX_scale_shape_dtype_layout; "
                "beta_linear_beta_activate_left_axis_rounding_and_dispatch_not_observed; "
                "no_occurrence_proof")
    if role == "router_selection":
        return (
            "active_logits_selected_weights_and_ids_shape_dtype_layout_and_hash_table_presence; "
            "score_function_renorm_groups_bias_scaling_eps_and_table_values_not_observed; "
            "logits_gemm_excluded; Hash_renorm_external_TopK_declared_renorm_internal; "
            "no_dispatch_or_occurrence_proof")
    if role == "mla_prolog":
        return (
            "active_token_rows_logical_parameters_cache_heads_and_query_outputs; "
            "query_norm_output_uses_actual_lowered_activation_format_and_quantization_boundary; "
            "physical_parameter_storage_leading_axis_grouping_rope_inputs_scales_cache_capacity_"
            "slots_and_scalar_flags_not_compared; no_dispatch_or_occurrence_proof")
    if role == "lightning_indexer":
        return (
            "active_query_weights_cache_heads_and_selected_index_shape_dtype; "
            "cache_capacity_page_size_sequence_lengths_index_values_scores_scales_"
            "and_scalar_flags_not_compared; no_dispatch_or_occurrence_proof")
    if role == "quant_lightning_indexer":
        return ("quantized_query_weights_scales_cache_head_and_indices_shape_dtype; "
                "cache_capacity_blocksize_sequence_metadata_values_and_scalar_modes_not_compared; "
                "no_dispatch_or_occurrence_proof")
    if role in {"hc_pre", "hc_post"}:
        return ("active_hyper_connection_tensor_descriptors; leading_token_grouping_"
                "epsilon_sinkhorn_scalar_optional_pre_coefficients_not_compared; "
                "no_dispatch_or_occurrence_proof")
    if role == "compressor":
        return ("active_input_parameters_state_width_rope_width_output_width_dtype; "
                "output_reserved_rows_state_capacity_page_size_slots_rope_values_sequence_"
                "metadata_and_scalar_flags_not_compared; no_dispatch_or_occurrence_proof")
    if role == "grouped_gate_up_activation_quant":
        return ("local_expert_parameter_geometry_and_operand_precision; "
                "routed_rows_group_values_weight_packing_quant_scales_and_native_version_selection_"
                "not_compared; no_dispatch_or_occurrence_proof")
    if role == "kda_recurrent":
        return ("declared_in_kernel_gate_source_parameter_domain; "
                "gate_beta_Alog_dtbias_descriptors_and_QKV_state_output_precision; "
                "QKV_and_output_active_shapes_state_shapes_strides_indices_and_scalar_flags_"
                "not_compared; no_dispatch_or_occurrence_proof")
    return "declared_subgraph; no_observed_contract"


def subgraph_comparison_scope(role, members, *, resource=None):
    """Fusion preserves missing physical work unless its selected program supplies it."""
    scope = subgraph_scope(role)
    scopes = [scope, *(member.get("contract_scope") for member in members)]
    partial = []
    msd = (
        role == "grouped_gate_up_activation_quant" and resource is not None
        and resource["recipe"] == "ascend_w4a8_msd"
        and resource["operator"]["metadata"].get("fusion_program_profile") == "ascend_w4a8_msd"
    )
    for item in scopes:
        coverage = item.get("prediction_coverage", {}) if isinstance(item, dict) else {}
        if coverage.get("kind") != "partial_program":
            continue
        if (msd and coverage.get("reason") == "quantized_epilogue_unmodeled"
                and set(coverage.get("missing_work", ())) == {
                    "packed_int4_output_rescale", "quantization_offset_correction"}):
            # The MSD pre/matrix/post ledger replaces this incomplete matrix
            # program; unrelated missing work and MXFP4 assumptions still apply.
            continue
        if coverage not in partial:
            partial.append(coverage)
    if not partial:
        return scope
    combined = partial[0] if len(partial) == 1 else {
        "kind": "partial_program", "reason": "incomplete_member_programs",
        "missing_work": sorted({work for item in partial for work in item.get("missing_work", ())}),
        "components": partial,
    }
    return {**(scope if isinstance(scope, dict) else {"comparison_scope": scope}),
            "prediction_coverage": combined}


def export_subgraphs(nodes, declarations, memberships, entry_dependencies):
    """Take dependency closure between declared members and their actual inputs."""
    by_id = {node["node_id"]: node for node in nodes}
    position = {node["node_id"]: index for index, node in enumerate(nodes)}
    layer_structures = {
        (node["replica"], node["stage"], node["layer"]): node["structure"] for node in nodes}
    users = defaultdict(set)
    for node in nodes:
        for dependency in node["deps"]:
            users[dependency].add(node["node_id"])
    result = []
    for key, declaration in sorted(declarations.items()):
        replica, local_id = key
        seed = memberships[key]
        if not seed:
            raise ValueError(f"structure_subgraph_has_no_members:{local_id}")
        stop = entry_dependencies[key]
        members, pending = set(seed), list(seed)
        while pending:
            current = pending.pop()
            for dependency in by_id[current]["deps"]:
                if dependency not in stop and dependency in by_id and dependency not in members:
                    members.add(dependency)
                    pending.append(dependency)
        ordered = [by_id[node_id] for node_id in sorted(members, key=position.__getitem__)]
        first = ordered[0]
        if any((node["replica"], node["stage"], node["phase"]) !=
               (first["replica"], first["stage"], first["phase"])
               for node in ordered):
            raise ValueError(f"structure_subgraph_crosses_placement:{local_id}")
        owners = set()
        for node in ordered:
            owners.add((replica, node["stage"], node["layer"], node["block"]))
            for origin in node["operator"]["metadata"].get("fused_origin_scopes", ()):
                owners.add((replica, origin["stage"], origin["layer"], origin["block"]))
        scopes = [{"replica": r, "stage": s, "layer": layer, "block": block}
                  for r, s, layer, block in sorted(
                      owners, key=lambda value: (value[0], value[1], value[2], value[3] or ""))]
        layers = sorted({scope["layer"] for scope in scopes})
        blocks = {scope["block"] for scope in scopes}
        common_structure = {
            key: value for key, value in first["structure"].items()
            if all(layer_structures.get(owner[:3], {}).get(key) == value for owner in owners)
        }
        inputs = sorted({dep for node in ordered for dep in node["deps"] if dep not in members})
        outputs = [node["node_id"] for node in ordered
                   if users[node["node_id"]] - members or not users[node["node_id"]] & members]
        source_calls = sorted({node["lineage"]["source_call_id"] for node in ordered
                               if node["lineage"]["source_call_id"]})
        result.append({
            "subgraph_id": f"rank:{replica}/subgraph:{local_id}",
            "node_ids": [node["node_id"] for node in ordered],
            "role": declaration["role"], "family": declaration["family"],
            "replica": replica, "stage": first["stage"],
            "layer": layers[0] if len(layers) == 1 else None,
            "layer_ids": layers, "owner_scopes": scopes,
            "phase": first["phase"], "block": next(iter(blocks)) if len(blocks) == 1 else None,
            "scope_role": first["role"],
            "structure": common_structure, "boundary": "semantic_subgraph",
            "external_inputs": inputs, "external_outputs": outputs,
            "contract": subgraph_contract(declaration),
            "contract_scope": subgraph_comparison_scope(declaration["role"], ordered),
            "implementation": declaration.get("implementation", {}),
            "coverage": declaration.get("coverage", {
                "kind": "complete_subgraph", "limitations": [],
            }),
            "lineage": {
                "declared_op_ids": declaration["op_ids"],
                "source_call_ids": source_calls,
            },
            "source": declaration.get("source"),
        })
    return result
