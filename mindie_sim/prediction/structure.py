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
import json
from collections import OrderedDict

from mindie_sim.model_catalog import normalize_model
from mindie_sim.schema.model import ModelSpec, SloSpec, WorkloadSpec
from mindie_sim.schema.common import ModelingError, _unknown
from mindie_sim.schema.case import Case, Software, _decode_raw_case, register_case_normalizer
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema import runtime_features


#: Bounded identity memo of already-parsed declarations.  ``Case`` is frozen and
#: parsing is a pure function of the declaration, so an unchanged declaration
#: object always yields the same parsed case.  The memo holds a strong reference
#: to the producing input, so a recycled ``id`` misses instead of mis-hitting.
_PARSE_MEMO: "OrderedDict[int, tuple]" = OrderedDict()
_PARSE_MEMO_LIMIT = 8
MEMO_INPUT = 0
MEMO_PARSED = 1

def _memoized_parse(value: Case) -> Case:
    key = id(value)
    entry = _PARSE_MEMO.get(key)
    if entry is not None and entry[MEMO_INPUT] is value:
        _PARSE_MEMO.move_to_end(key)
        return entry[MEMO_PARSED]
    parsed = _parse_case_uncached(value)
    _PARSE_MEMO[key] = (value, parsed)
    # A caller that re-parses the already-parsed value gets it back in O(1):
    # normalisation is idempotent, so this is the same declaration.
    if parsed is not value:
        _PARSE_MEMO[id(parsed)] = (parsed, parsed)
    _PARSE_MEMO.move_to_end(key)
    while len(_PARSE_MEMO) > _PARSE_MEMO_LIMIT:
        _PARSE_MEMO.popitem(last=False)
    return parsed


def _parse_case_uncached(value: Case | Mapping) -> Case:
    return _normalize_parsed_case(_decode_raw_case(value))


def parse_case(value: Case | Mapping) -> Case:
    """Parse a declaration into the frozen internal Case.

    A ``Case`` object is frozen and parsing is a pure function of it, so a
    repeated parse of the same declaration object reuses the identical parsed
    value.  A ``Mapping`` declaration may be mutated by its owner and is never
    memoized.
    """
    if isinstance(value, Mapping):
        # A raw mapping is mutable and must never be retained: parsing it is a
        # pure function whose result belongs to the caller's active work only.
        return _parse_case_uncached(value)
    return _memoized_parse(value)


def parse_model(value: Case | Mapping) -> Case:
    """Parse a pure-model declaration without fabricating a workload.

    ``model`` + ``runtime`` + ``machine``/``hardware`` are sufficient.  The
    returned Case carries an empty WorkloadSpec; callers pass explicit
    RequestView geometry to the forward/assembly entry.  A complete Case still
    parses through :func:`parse_case`.
    """
    if isinstance(value, Case):
        case = value
        if not isinstance(case.machine, Hardware):
            case = Case(Hardware.from_mapping(case.machine.to_dict()), case.model,
                        case.workload, Software.from_mapping(case.runtime.to_dict()),
                        case.slo)
    else:
        _unknown(value, {"machine", "hardware", "model", "runtime", "slo",
                         "schema", "structure_variant"}, "pure_model")
        missing = {"model"} | ({"machine"} if "hardware" not in value else set())
        missing = {key for key in missing if key not in value}
        if missing:
            raise ModelingError("pure_model_missing_fields:" + ",".join(sorted(missing)))
        if value.get("structure_variant"):
            raise ModelingError("structure_variant_retired:use_model_layer_plans_or_layer_overrides")
        machine = value.get("machine", value.get("hardware"))
        case = Case(
            machine=Hardware.from_mapping(machine),
            model=ModelSpec.from_mapping(value["model"]),
            workload=WorkloadSpec._pure_model_sentinel(),
            runtime=Software.from_mapping(value.get("runtime", {})),
            slo=SloSpec.from_mapping(value.get("slo")),
        )
    return _normalize_parsed_case(replace(case, workload=WorkloadSpec._pure_model_sentinel()))


def _normalize_parsed_case(case: Case) -> Case:
    model = normalize_model(case.model)
    # Every declared model acquires its explicit native auxiliary plans here,
    # once, before geometry, residency or execution can consume them.  Adapters
    # that already declared plans keep them; an undeclared auxiliary still
    # raises ``draft_explicit_layer_plans_required`` at the geometry entry.
    from mindie_sim.catalog.draft import declare_native_auxiliary_plans
    model = declare_native_auxiliary_plans(model)
    from .implementations import select
    from .implementation_selection import prepared_selection

    pool = prepared_selection(case.runtime.kernels.get("lowerings", {}),
                              "pool_key_indexer")
    if pool and pool.name == "vllm_ascend_cann_keypool_caller":
        model = pool.module.prepare_model(model, case.runtime, pool)
    elif "native_keypool_caller" in dict(model.metadata):
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "native_keypool_caller"))
    prolog = prepared_selection(case.runtime.kernels.get("lowerings", {}), "mla_prolog")
    if prolog:
        model = prolog.module.prepare_model(model, case.runtime)
    head_padding = prepared_selection(case.runtime.kernels.get("lowerings", {}), "mla_head_padding")
    if head_padding:
        model = head_padding.module.prepare_model(model, case.runtime)
    elif "mla_head_padding" in dict(model.metadata):
        # Parsed cases can be reused with another explicit implementation.
        # Derived prepared-weight/head state must not survive its selection.
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "mla_head_padding"))
    conv = prepared_selection(case.runtime.kernels.get("lowerings", {}), "causal_conv_update")
    if conv:
        model = conv.module.prepare_model(model, case.runtime)
    elif "causal_conv_update" in dict(model.metadata):
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "causal_conv_update"))
    kda_projection = select(case.runtime.kernels.get("lowerings", {}), "kda_input_projection")
    if kda_projection:
        model = kda_projection.module.prepare_model(model, case.runtime)
    elif "kda_input_projection" in dict(model.metadata):
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "kda_input_projection"))
    mx_matrix = prepared_selection(case.runtime.kernels.get("lowerings", {}), "mx_matrix")
    if mx_matrix:
        model = mx_matrix.module.prepare_model(model, case.runtime, mx_matrix)
    elif "prepared_mx32_matrix" in dict(model.metadata):
        model = replace(model, metadata=tuple((key, value) for key, value in model.metadata
                                             if key != "prepared_mx32_matrix"))
    runtime = case.runtime
    sparse_offload = select(runtime.kernels.get("lowerings", {}), "sparse_kv_offload")
    if sparse_offload:
        model, runtime = sparse_offload.module.prepare_case(
            model, runtime, case.machine, case.workload, sparse_offload)
    elif "sparse_kv_offload" in dict(model.metadata) or "sparse_kv_offload" in dict(runtime.metadata):
        from mindie_sim.prediction.operators.sparse_indexer.sparse_kv_offload import strip_case
        model, runtime = strip_case(model, runtime)
    for index, layer in enumerate(model.layer_plans):
        if layer.index != index:
            raise ModelingError("layer_indices_must_be_contiguous")
        for owner in (layer.kv_owner_layer_id, layer.index_k_owner_layer_id,
                      layer.topk_owner_layer_id, layer.candidate_source_layer_id):
            if owner > index:
                raise ModelingError(f"layer_owner_must_precede_consumer:{index}:{owner}")
    p = runtime.parallel
    counts = runtime.pp_layer_partition or tuple(
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
    if p.has_local_tp:
        from .sharding.local_tp import validate_local_projection_tp
        validate_local_projection_tp(case)
    case.workload.validate_model(model, runtime)
    metadata = dict(runtime.metadata)
    aux = getattr(model, "auxiliary", None)
    if aux is not None and any(
        bool(getattr(d, "is_synthetic", False)) for d in (aux.mtp, aux.dspark) if d is not None
    ):
        spine = metadata.get("single_spine")
        if isinstance(spine, Mapping) and (
            spine.get("optimization_mode") == "simulation"
            or spine.get("stack_track") in {"eager", "shipping", "proposed"}
        ):
            raise ModelingError(
                "simulation_mode_forbids_synthetic_gqa_draft:"
                "actual_deployment_requires_real_speculative_checkpoint"
            )
    # Carry the unaltered case machine/model identity with the parsed runtime so
    # every ordinary FeatureConfig/compiled feature record receives it.  This is
    # the deployment identity, not a measured supply or model-label branch.
    from .frontend.features import _identity_metadata
    metadata.update(_identity_metadata(case.machine, model))
    from .sharding.context_parallel import validate_quantized_allreduce_runtime
    validate_quantized_allreduce_runtime(
        metadata, arch=str(case.machine.arch_family or ""), tp=int(p.tp),
    )
    if case.workload.requests:
        media = {
            request.request_id: request.media_tokens
            for request in case.workload.requests
            if request.media_tokens and not getattr(request, "media_encoded", False)
        }
        if media:
            metadata["media_tokens_by_request"] = media
        else:
            metadata.pop("media_tokens_by_request", None)
    moe_cfg = case.runtime.kernels.get("moe_transport")
    deepep_enabled = (
        (isinstance(moe_cfg, Mapping) and moe_cfg.get("mode") == "deepep")
        or bool(metadata.get("enable_deepep", False))
    )
    if (deepep_enabled and "deepep_ranks_per_node" not in metadata
            and not (isinstance(moe_cfg, Mapping) and moe_cfg.get("ranks_per_node"))):
        metadata["deepep_ranks_per_node"] = int(case.machine.ranks_per_node)
    binding = runtime.kernels.get("lowerings", {}).get("gdn_decode")
    options = binding.get("options") if isinstance(binding, Mapping) else None
    if isinstance(binding, Mapping):
        assumptions = dict(case.machine.assumptions)
        assumptions["gdn_decode_selected"] = "1"
        if isinstance(options, Mapping) and options.get("recurrent_implementation"):
            assumptions["gdn_recurrent_implementation"] = str(options["recurrent_implementation"])
        case = replace(case, machine=replace(case.machine, assumptions=tuple(sorted(assumptions.items()))))
    if isinstance(options, Mapping) and "gating_beta_store" in options:
        store = options.get("gating_beta_store")
        if store in ({}, None):
            store = {"disabled": True}
        elif isinstance(store, Mapping):
            store = dict(store)
            if not store.get("from_hardware"):
                store.pop("bytes_per_element", None)
        assumptions = dict(case.machine.assumptions)
        assumptions["gdn_gating_beta_store"] = json.dumps(store, sort_keys=True)
        case = replace(case, machine=replace(case.machine, assumptions=tuple(sorted(assumptions.items()))))
    return replace(case, model=model, runtime=replace(
        runtime, pp_layer_partition=counts, metadata=tuple(sorted(metadata.items()))))


register_case_normalizer(_normalize_parsed_case)


def stage_map(model, runtime) -> tuple[int, ...]:
    counts = runtime.pp_layer_partition or tuple(
        model.layers // runtime.parallel.pp + int(i < model.layers % runtime.parallel.pp)
        for i in range(runtime.parallel.pp)
    )
    return tuple(stage for stage, count in enumerate(counts) for _ in range(count))
