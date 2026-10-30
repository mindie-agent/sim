"""Structural input normalization for the mechanism simulator.

[Role] Turn catalog or explicit geometry into the same layer and quant records.
[Invariants] A catalog supplies structure, never latency. Layer count and
  dimensions are calculation inputs. Measured pools and serving floors do not
  participate in admission or in model normalization.
[Timeouts & Locks] Metadata-only parsing; no weights or device contexts.
[Side-Effect Policy] Creates immutable records without modifying caller inputs.
"""
from __future__ import annotations

from dataclasses import fields, replace
from typing import Mapping

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.model import LayerPlan, ModelSpec, SloSpec, WorkloadSpec
from mindie_sim.schema.common import ModelingError, _unknown
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.hardware import Hardware


def normalize_model(model: ModelSpec) -> ModelSpec:
    quant = model.quant or QuantSpec(
        name=model.quantization, weight_format=model.weight_dtype,
        expert_weight_format=model.weight_dtype, activation_format=model.activation_dtype,
        kv_dtype=model.kv_dtype,
    )
    if model.layer_plans:
        if len(model.layer_plans) != model.layers:
            raise ModelingError("layer_plan_count_differs_from_layers")
        return replace(model, quant=quant)
    dimensions = {
        "num_heads": model.num_heads, "num_kv_heads": model.num_kv_heads or model.num_heads,
        "head_dim": model.head_dim, "intermediate_size": model.intermediate_size,
        "num_experts": model.num_experts, "top_k": model.top_k,
        "moe_intermediate_size": model.moe_intermediate_size,
        "num_shared_experts": model.num_shared_experts,
    }
    if model.mla:
        dimensions.update({
            name: getattr(model.mla, name) for name in
            ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
             "o_lora_rank", "o_groups")
        })
        dimensions.update(mla_variant="latent_mla", index_heads=model.mla.index_n_heads,
                          index_dim=model.mla.index_head_dim, index_topk=model.mla.index_topk,
                          indexer=bool(model.mla.index_topk), owns_index_k=bool(model.mla.index_topk),
                          index_query_per_layer=bool(model.mla.index_topk))
        dimensions.update(compress_ratio=model.mla.compress_ratio, window=model.mla.sliding_window,
                          cache_kind="window_compressed" if model.mla.compress_ratio else "dense",
                          compressor_kind="weighted_pool" if model.mla.compress_ratio else "none",
                          compressor_state_rows=model.mla.compress_ratio)
    attention = {"mha": "gqa", "dsa": "mla", "csa": "mla"}.get(model.attention, model.attention)
    plans = tuple(LayerPlan(
        index=index, attention=attention, ffn=model.ffn, architecture=attention,
        kv_owner_layer_id=index, index_k_owner_layer_id=index, owns_long_kv=True, **dimensions,
    ) for index in range(model.layers))
    return replace(model, quant=quant, layer_plans=plans)


def parse_case(value: Case | Mapping) -> Case:
    if not isinstance(value, Mapping):
        case = value
        if not isinstance(case.machine, Hardware):
            case = Case(Hardware.from_mapping(case.machine.to_dict()), case.model, case.workload,
                        Software.from_mapping(case.runtime.to_dict()), case.slo)
    else:
        _unknown(value, {"machine", "model", "workload", "runtime", "slo",
                         "schema", "structure_variant"}, "case")
        missing = {"machine", "model", "workload"} - value.keys()
        if missing:
            raise ModelingError("case_missing_fields:" + ",".join(sorted(missing)))
        raw_model = value["model"]
        model = ModelSpec.from_mapping(raw_model)
        if value.get("structure_variant"):
            raise ModelingError("structure_variant_retired:use_model_layer_plans_or_layer_overrides")
        case = Case(
            machine=Hardware.from_mapping(value["machine"]), model=model,
            workload=WorkloadSpec.from_mapping(value["workload"]),
            runtime=Software.from_mapping(value.get("runtime", {})),
            slo=SloSpec.from_mapping(value.get("slo")),
        )
    model = normalize_model(case.model)
    from .implementations import select

    prolog = select(case.runtime.kernels.get("lowerings", {}), "mla_prolog")
    if prolog:
        model = prolog.module.prepare_model(model, case.runtime)
    head_padding = select(case.runtime.kernels.get("lowerings", {}), "mla_head_padding")
    if head_padding:
        model = head_padding.module.prepare_model(model, case.runtime)
    elif "mla_head_padding" in dict(model.metadata):
        # Parsed cases can be reused with another explicit implementation.
        # Derived prepared-weight/head state must not survive its selection.
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "mla_head_padding"))
    conv = select(case.runtime.kernels.get("lowerings", {}), "causal_conv_update")
    if conv:
        model = conv.module.prepare_model(model, case.runtime)
    elif "causal_conv_update" in dict(model.metadata):
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "causal_conv_update"))
    mx_matrix = select(case.runtime.kernels.get("lowerings", {}), "mx_matrix")
    if mx_matrix:
        model = mx_matrix.module.prepare_model(model, case.runtime, mx_matrix)
    elif "prepared_mx32_matrix" in dict(model.metadata):
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "prepared_mx32_matrix"))
    for index, layer in enumerate(model.layer_plans):
        if layer.index != index:
            raise ModelingError("layer_indices_must_be_contiguous")
        for owner in (layer.kv_owner_layer_id, layer.index_k_owner_layer_id,
                      layer.topk_owner_layer_id, layer.candidate_source_layer_id):
            if owner > index:
                raise ModelingError(f"layer_owner_must_precede_consumer:{index}:{owner}")
    p = case.runtime.parallel
    counts = case.runtime.pp_layer_partition or tuple(
        model.layers // p.pp + int(stage < model.layers % p.pp) for stage in range(p.pp)
    )
    p.validate(model, layer_partition=counts)
    stages = tuple(stage for stage, count in enumerate(counts) for _ in range(count))
    if any(layer.attention in {"gqa", "projection_only"}
           and 0 <= layer.kv_owner_layer_id < layer.index
           and stages[layer.kv_owner_layer_id] != stages[layer.index]
           for layer in model.layer_plans):
        metadata = dict(model.metadata)
        gaps = set(metadata.get("unmodeled_features", ()))
        gaps.add("cross_pp_shared_kv_receive_residency_and_copy_endpoints")
        degraded = set(metadata.get("degraded_submodules", ()))
        degraded.add("cross_pp_shared_kv:transport_only_proxy")
        metadata.update(model_coverage="partial", unmodeled_features=tuple(sorted(gaps)),
                        degraded_submodules=tuple(sorted(degraded)))
        model = replace(model, metadata=tuple(sorted(metadata.items())))
    case.machine.validate_placement(p)
    case.workload.validate_model(model, case.runtime)
    metadata = dict(case.runtime.metadata)
    media = {request.request_id: request.media_tokens for request in case.workload.requests if request.media_tokens}
    if media:
        metadata["media_tokens_by_request"] = media
    return replace(case, model=model, runtime=replace(
        case.runtime, pp_layer_partition=counts, metadata=tuple(sorted(metadata.items()))))


def stage_map(model, runtime) -> tuple[int, ...]:
    counts = runtime.pp_layer_partition or tuple(
        model.layers // runtime.parallel.pp + int(i < model.layers % runtime.parallel.pp)
        for i in range(runtime.parallel.pp)
    )
    return tuple(stage for stage, count in enumerate(counts) for _ in range(count))
