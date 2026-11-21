"""Export shared model and operator structure without executing a cost model.

[Role] Reuse the semantic frontend and implementation expansion for profiling.
[Invariants] Node identities preserve layer/replica instances. Declared tensor
  contracts and software boundaries are not evidence that kernels were observed.
  Every node keeps per-value dtype/layout/quant/attrs, nullable semantic refs and
  separate native candidates/selection; candidates never become executed work.
  Source calls retain their children; unknown logical origins are not invented.
[Timeouts & Locks] One finite target cohort, metadata only; no device or locks.
[Side-Effect Policy] The CLI reads one bounded case and writes requested JSON.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from .prediction.implementations import produced_values
from .prediction.operator_analysis import target_graphs
from .prediction.native_binding import (
    bind_native_call, coverage_from_records, member_physical_view,
)
from .prediction.program import RecipeBook
from .prediction.structure import parse_case
from .structure_contracts import comparison_scope, predicted_contract
from .structure_implementations import (
    native_implementation, operator_structure, region_structure,
)
from .structure_physical import node_physical, subgraph_physical
from .subgraph_math import attach_subgraph_math
from .structure_subgraphs import (
    export_subgraphs, layer_structure, metadata, subgraph_comparison_scope,
)


SCHEMA_VERSION = "mindie-model-structure/1"
MAX_CASE_BYTES = 50 * 1024 * 1024
_SOFTWARE_FIELDS = (
    "parallel", "pp_layer_partition", "pipeline_microbatches", "graph_mode",
    "graph_capture_mode", "graph_buckets", "graph_capture_keys",
    "piecewise_capture_keys", "mixed_mode", "splitting_ops", "kv_block_size",
    "moe_recipe", "mc2_chunks", "multistream_overlap_shared_expert",
    "vllm_impl", "vllm_version", "kernel_format", "attention_policy", "routing",
    "tensor_layouts", "final_norm_scope", "lowering_passes",
)
_FAMILY_BLOCKS = {
    "qkv_projection": "attention", "attention": "attention",
    "rotary": "attention", "cache_store": "attention",
    "kv_compress_epilog": "attention", "indexer_quant_cache": "attention",
    "kv_cache_write": "attention",
    "gate_up_projection": "ffn", "down_projection": "ffn", "swiglu": "ffn",
    "grouped_gemm": "ffn", "embedding": "embedding", "logits": "logits",
}


def _intern_tree(value):
    """Intern decoded strings without changing mapping order or values.

    ``json.loads(json.dumps(...))`` creates a distinct ``str`` for every key
    and value occurrence.  A decode report repeats the same small vocabulary
    across tens of thousands of nodes, so retaining those duplicates dominates
    the report footprint.  Interning is identity-only: equality, hashing, JSON
    encoding and insertion order are unchanged.
    """
    if isinstance(value, str):
        return sys.intern(value)
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            new_key = sys.intern(key) if isinstance(key, str) else key
            items.append((new_key, _intern_tree(item)))
        value.clear()
        value.update(items)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _intern_tree(item)
        return value
    if isinstance(value, tuple):
        return tuple(_intern_tree(item) for item in value)
    return value


def _json_value(value):
    t = type(value)
    if t is str:
        return sys.intern(value)
    if value is None or t is int or t is bool:
        return value
    if t is float:
        if not math.isfinite(value):
            raise ValueError("Out of range float values are not JSON compliant")
        return value
    if t is list or t is tuple:
        return [_json_value(item) for item in value]
    if t is dict:
        out = {}
        for key, item in value.items():
            if type(key) is not str:
                return _intern_tree(json.loads(json.dumps(value, allow_nan=False)))
            out[sys.intern(key)] = _json_value(item)
        return out
    return _intern_tree(json.loads(json.dumps(value, allow_nan=False)))


_NATIVE_EVIDENCE_KEYS = frozenset(("native_call", "native_call_binding", "observed_aliases"))


def _without_native_call_evidence(op):
    """Structural view: derived native-call provenance is not identity."""
    meta = dict(op.metadata)
    semantic = meta.get("semantic_metadata")
    nested = (isinstance(semantic, Mapping)
              and bool(_NATIVE_EVIDENCE_KEYS.intersection(semantic)))
    if not _NATIVE_EVIDENCE_KEYS.intersection(meta) and not nested:
        return op
    for key in _NATIVE_EVIDENCE_KEYS:
        meta.pop(key, None)
    if nested:
        meta["semantic_metadata"] = {
            key: value for key, value in semantic.items()
            if key not in _NATIVE_EVIDENCE_KEYS}
    return replace(op, metadata=tuple(meta.items()))


def _node_id(replica, op_id):
    return f"rank:{replica}/{op_id}"


def _block(op):
    if metadata(op).get("structure_block"):
        return metadata(op)["structure_block"]
    return _FAMILY_BLOCKS.get(op.family)


def _semantic_refs(op, replica, *, parent=None, block=None):
    """Nullable semantic relations; none decides whether this node executes."""
    meta = metadata(op)
    refs = []
    if op.layer is not None and op.layer >= 0:
        refs.append({"kind": "layer", "name": str(op.layer),
                     "role": block or "layer"})
    for group in meta.get("structure_subgraph_memberships", ()):
        refs.append({"kind": "subgraph",
                     "name": f"rank:{replica}/subgraph:{group}", "role": "member"})
    for origin in meta.get("fused_origin_scopes", ()):
        if isinstance(origin, dict) and origin.get("layer") is not None:
            refs.append({"kind": "layer", "name": str(origin["layer"]),
                         "role": origin.get("block") or "layer"})
    if parent is not None:
        refs.append({"kind": "operator", "name": parent, "role": "source_call"})
    unique = []
    for ref in refs:
        if ref not in unique:
            unique.append(ref)
    return unique


def _node(op, replica, *, source_call=None, block=None, role=None, structure=None,
          kernels=None, book=None, geometry=None, shared_call=None, native_sink=None,
          semantic_refs=None, outputs_by_id=None, metadata_by_id=None):
    plan = None
    if geometry is not None and op.source_boundary != "call":
        # Source planners own physical declarations. Preparing a plan does not
        # start its program, consult timings, or resolve captured ABI identity.
        op, plan = book.prepare(op, geometry)
    node_id = _node_id(replica, op.op_id)
    full_operator = op.to_dict(include_work=False)
    structural_op = _without_native_call_evidence(op)
    operator = full_operator if structural_op is op else structural_op.to_dict(include_work=False)
    contract = predicted_contract(operator)
    outputs_by_id = {} if outputs_by_id is None else outputs_by_id
    metadata_by_id = {} if metadata_by_id is None else metadata_by_id
    try:
        produced = tuple(produced_values(op))
    except (KeyError, ValueError, TypeError):
        # A producer helper may refuse an unsupported declaration.  The
        # declared contract/audit remains authoritative; this is not a value
        # extraction path and must not hide a malformed T11 layout.
        produced = ()
    structure_record = operator_structure(
        full_operator, contract, semantic_refs=semantic_refs, owner_id=node_id,
        context={"op": op, "outputs_by_id": outputs_by_id,
                 "metadata_by_id": metadata_by_id, "produced": produced})
    outputs_by_id[op.op_id] = tuple(structure_record["values"]["outputs"])
    metadata_by_id[op.op_id] = dict(op.metadata)
    meta = dict(op.metadata)
    semantic = meta.get("semantic_metadata", {})
    binding = None
    if native_sink is not None or any(
        k in meta for k in (
            "native_call", "native_call_binding", "native_api",
            "aclnn_symbol", "execution_core", "invocation_class",
        )
    ):
        if shared_call is not None:
            # A fused member keeps the parent call identity and forwards only its
            # own existing plan work as a view; it never becomes a second charge.
            binding = member_physical_view(shared_call, plan, op if plan is not None else None)
        else:
            binding = bind_native_call(op, contract=contract, plan=plan,
                                       prepared=op if plan is not None else None)

    payload = {
        "node_id": node_id,
        "op_id": op.op_id,
        "replica": replica,
        "layer": op.layer,
        "stage": op.stage,
        "block": block if block is not None else _block(op),
        "role": role if role is not None else metadata(op).get("structure_role"),
        "structure": structure or {},
        "family": op.family,
        "phase": op.phase,
        "recipe": op.recipe,
        "deps": [_node_id(replica, dep) for dep in op.deps],
        "operator": operator,
        "contract": contract,
        "values": structure_record["values"],
        "semantic_refs": structure_record["semantic_refs"],
        "native_candidates": structure_record["native_candidates"],
        "native_execution": structure_record["native_execution"],
        "representation_gaps": structure_record["gaps"],
        "implementation": native_implementation(operator, contract) or {},
        **node_physical(structural_op, recipe_bindings=kernels, book=book),
        "physical_work_basis": ("prepared_source_plan" if geometry is not None else
                                "source_declaration_without_resource_geometry")
                               if op.source_boundary == "kernel" else "operator_declaration",
        "contract_scope": comparison_scope(operator),
        "boundary": op.source_boundary or "semantic",
        "lineage": {
            "source_call_id": source_call,
            "fused_from": meta.get("fused_from", semantic.get("fused_from")),
            "semantic_recipe": meta.get("semantic_recipe"),
            **(
                {"absorbed_source_ops": list(meta.get("absorbed_source_ops") or semantic.get("absorbed_source_ops"))}
                if (meta.get("absorbed_source_ops") or semantic.get("absorbed_source_ops"))
                else {}
            ),
        },
    }
    if binding is not None and native_sink is not None:
        # BIND1 evidence travels beside the node identity, not inside the
        # structural content hash: it is derived provenance, not structure.
        native_sink.append({"node_id": node_id, **binding.to_dict()})
    return _json_value(payload)


def export_op_structure(ops, *, model, execution, kernels=None, source=None,
                        expand_calls=True, resource_graphs=None, resource_geometries=None,
                        native_evidence=False):
    """Export a declared OpSpec DAG without a hardware Case or timing execution.

    A sequence selects replica 0; a mapping supplies explicit replica -> DAG.
    ``model`` and ``execution`` contain structural declarations, never observed
    timings. Execution must declare the common ``phase``; placement scopes are
    caller declarations rather than assumptions about physical capture ranks.
    ``kernels`` uses the existing RecipeBook mapping for source-call expansion.
    ``resource_graphs`` is the same declaration after resource fusion, retaining
    semantic nodes while exposing their selected executable boundaries.
    ``resource_geometries`` optionally maps (replica, stage) to declared Geometry
    for source-kernel planning. It contains ownership/capacity, not resource rates.
    Caller ``source`` is provenance excluded from the content hash.
    ``native_evidence=True`` attaches derived ``native_calls`` and
    ``native_call_coverage`` beside the identity. It is opt-in because released
    consumers (profiler <= 0.5.11) verify ``graph_id`` over every top-level key
    except ``graph_id``/``source``; the default report stays verifiable by them.
    """
    graphs = ops if isinstance(ops, Mapping) else {0: ops}
    graphs = {replica: tuple(operators) for replica, operators in graphs.items()}
    if (resource_graphs is None
            and (kernels or {}).get("lowerings", {}).get("native_fusion")):
        from mindie_sim.prediction.operators.hyper_connections.native_fusion import lower_native_fusions

        runtime = SimpleNamespace(kernels=kernels)
        resource_graphs = {replica: lower_native_fusions(operators, runtime)
                           for replica, operators in graphs.items()}
    phase = execution["phase"]
    draft_prepare = bool(execution.get("draft_prepare", False))
    allowed_phases = {phase}
    if phase == "decode":
        allowed_phases.add("draft")
    elif phase == "prefill" and draft_prepare:
        allowed_phases.update({"draft", "draft_prefill"})
    book = RecipeBook.from_mapping(kernels or {})
    resource_geometries = resource_geometries or {}
    nodes, source_calls, external = [], [], set()
    native_calls = [] if native_evidence else None
    declarations, entry_dependencies, memberships = {}, defaultdict(set), defaultdict(set)
    for replica, operators in sorted(graphs.items()):
        outputs_by_id, metadata_by_id = {}, {}
        for op in operators:
            if op.phase not in allowed_phases:
                raise ValueError(f"structure_phase_mismatch:{op.op_id}:{op.phase}:{phase}")
            expanded = book.expand((op,)) if expand_calls else (op,)
            meta = metadata(op)
            for group in meta.get("structure_subgraph_entries", ()):
                entry_dependencies[replica, group].update(
                    _node_id(replica, dep) for dep in op.deps)
            for declaration in meta.get("structure_subgraphs", ()):
                key = replica, declaration["key"]
                if key in declarations and declarations[key] != declaration:
                    raise ValueError(f"structure_subgraph_conflicting_declaration:{key}")
                declarations[key] = declaration
            for group in meta.get("structure_subgraph_memberships", ()):
                memberships[replica, group].update(_node_id(replica, child.op_id) for child in expanded)
            is_expanded = op.source_boundary == "call" and expand_calls
            parent = _node_id(replica, op.op_id) if is_expanded else None
            # One shared actual-call descriptor for a fused source call: every
            # declared child kernel keeps this same call_id and one timing owner.
            call_binding = (
                bind_native_call(op, children=expanded)
                if is_expanded and (native_evidence or any(
                    k in meta for k in (
                        "native_call", "native_call_binding", "native_api",
                        "aclnn_symbol", "execution_core", "invocation_class",
                    )
                ))
                else None
            )
            if is_expanded:
                source_calls.append(_json_value({
                    "node_id": parent,
                    "operator": _without_native_call_evidence(op).to_dict(include_work=False),
                    "children": [
                        _node_id(replica, child.op_id)
                        for child in expanded if child.op_id != op.op_id
                    ],
                }))
            nodes.extend(_node(child, replica, source_call=parent, block=_block(op),
                               role=meta.get("structure_role"),
                               structure=layer_structure(model, op.layer, op=op, execution=execution),
                               kernels=kernels,
                               book=book, geometry=resource_geometries.get((replica, child.stage)),
                               shared_call=(None if child.family == "barrier" else call_binding),
                               native_sink=native_calls,
                               outputs_by_id=outputs_by_id,
                               metadata_by_id=metadata_by_id,
                               semantic_refs=_semantic_refs(
                                   child, replica, parent=parent, block=_block(op)))
                         for child in expanded)
    ids = {node["node_id"] for node in nodes}
    if len(ids) != len(nodes):
        raise ValueError("structure_duplicate_node_id")
    for node in nodes:
        external.update(dep for dep in node["deps"] if dep not in ids)
    subgraphs = export_subgraphs(nodes, declarations, memberships, entry_dependencies)
    by_id = {node["node_id"]: node for node in nodes}
    for group in subgraphs:
        group.update(subgraph_physical(by_id, group))
        member_refs = [{"kind": "operator", "name": key,
                        "role": by_id[key]["family"]}
                       for key in group["node_ids"]]
        group["semantic_refs"] = member_refs or None
        group.update(region_structure(group.get("implementation") or {},
                                      region_id=group["subgraph_id"],
                                      semantic_refs=member_refs))
        group.setdefault("representation_gaps", [])
    resource_operators = []
    if resource_graphs is not None:
        groups = {group["subgraph_id"]: group for group in subgraphs}
        subgraph_member_sets = {id(other): set(other["node_ids"]) for other in subgraphs}
        subgraphs_by_first_member = defaultdict(list)
        for other in subgraphs:
            if other["node_ids"]:
                subgraphs_by_first_member[other["node_ids"][0]].append(other)
        for replica, operators in sorted(resource_graphs.items()):
            outputs_by_id, metadata_by_id = {}, {}
            for op in operators:
                meta = dict(op.metadata)
                if "fusion_group_key" not in meta or not isinstance(meta.get("fusion_boundary"), Mapping):
                    continue
                key = f"rank:{replica}/subgraph:{meta['fusion_group_key']}"
                group = groups[key]
                resource = _node(op, replica, block=group["block"], role=group["role"],
                                 structure=group["structure"], kernels=kernels, book=book,
                                 geometry=resource_geometries.get((replica, op.stage)),
                                 semantic_refs=group["semantic_refs"],
                                 outputs_by_id=outputs_by_id,
                                 metadata_by_id=metadata_by_id)
                resource_operators.append(resource)
                boundary = {**meta["fusion_boundary"],
                            "operator_ids": [resource["node_id"]],
                            "kind": "selected_native_fusion"}
                group["resource_boundary"] = boundary
                group.update({field: resource[field]
                              for field in ("physical_work", "physical_signature")})
                group["contract_scope"] = subgraph_comparison_scope(
                    group["role"], [by_id[key] for key in group["node_ids"]], resource=resource)
                # The region owns one physical boundary.  A native name is a
                # correspondence; only a selection fact may mark it chosen.
                candidates = [dict(candidate)
                              for candidate in resource["native_candidates"]]
                selected = (resource["native_execution"] or {}).get("selected")
                selected_candidate = None
                if selected and selected.get("native"):
                    selected_candidate = next(
                        (candidate for candidate in candidates
                         if candidate["native"] == selected["native"]), None)
                    if selected_candidate is not None:
                        selected_candidate.update(selected=True,
                                                  status="selected_declared")
                    fusion_execution = {
                        "kind": "selected_native_fusion",
                        "selection_state": "selected_declared",
                        "selected": {
                            **selected,
                            "candidate_id": (selected_candidate or {}).get(
                                "candidate_id")},
                        "observed": False,
                        "reason": "one selected region boundary covers every semantic member",
                    }
                elif selected:
                    fusion_execution = {
                        "kind": "selected_native_fusion_boundary",
                        "selection_state": "selected_declared",
                        "selected": {**selected, "candidate_id": None},
                        "observed": False,
                        "reason": "a plan is selected but its native API is not unique",
                    }
                else:
                    fusion_execution = {
                        "kind": "selected_native_fusion_boundary",
                        "selection_state": ("multiple_compatible_unselected"
                                            if candidates else "unresolved"),
                        "selected": None,
                        "observed": False,
                        "reason": "fusion boundary is replaced but no native API is selected",
                    }
                group["native_candidates"] = candidates
                group["native_execution"] = fusion_execution
                region_gaps = list(group.get("representation_gaps") or ())
                if selected and selected.get("native") and selected_candidate is None:
                    region_gaps.append({
                        "code": "selected_native_not_in_candidates",
                        "detail": {"selected": selected["native"]}})
                group["representation_gaps"] = region_gaps
                members = set(group["node_ids"])
                internal = {"kind": "internal_to_selected_native_fusion",
                            "subgraph_id": key, "operator_ids": [resource["node_id"]]}
                covered = {
                    "kind": "covered_by_fused_region",
                    "selection_state": "covered",
                    "selected": None,
                    "observed": False,
                    "region": key,
                    "reason": "member work belongs to the region's selected native boundary",
                }
                for member in members:
                    by_id[member].update(event_matchable=False, resource_boundary=internal,
                                         native_execution=covered)
                    for other in subgraphs_by_first_member.get(member, ()):
                        if other is not group and subgraph_member_sets[id(other)] <= members:
                            other.update(event_matchable=False, resource_boundary=internal,
                                         native_execution=covered)
    # Normalize each owned record separately. A whole-payload JSON round trip
    # retained the DAG, its encoded string and a second DAG at the same time.
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": _json_value(model),
        "execution": _json_value({
            **execution, "expand_calls": expand_calls,
            "replicas": sorted(graphs),
            "kernel_bindings": kernels or {},
        }),
        "nodes": nodes,
        "source_calls": source_calls,
        "subgraphs": [_json_value(group) for group in subgraphs],
        "resource_operators": resource_operators,
        "external_inputs": sorted(external),
    }
    # Structural math belongs to the structural record: attach it before the
    # digest so a verifier re-hashes exactly these fields. It reuses this graph;
    # no causal Engine and no second export runs here.
    attach_subgraph_math(payload)
    # Onboarding hints for genuine core holes travel *inside* the hashed
    # payload: a verifier that re-hashes this report must see them, so the
    # graph identity stays self-verifying. Structure export itself succeeds.
    from .prediction.unmodeled_gate import HINT_KEYS, structure_hints
    all_ops = tuple(op for operators in graphs.values() for op in operators)
    hints = structure_hints(all_ops, {"model": model})
    if hints:
        payload[HINT_KEYS] = [dict(hint) for hint in hints]
    encoded_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest_hex = sha256(encoded_payload).hexdigest()
    del encoded_payload
    report = {
        **payload,
        "graph_id": "sha256:" + digest_hex,
        "source": _json_value(source),
    }
    if native_evidence:
        # Derived actual-invocation evidence: carried beside the structural
        # identity, excluded from its hash exactly like caller source.
        report["native_calls"] = _json_value(native_calls)
        report["native_call_coverage"] = _json_value(
            coverage_from_records(native_calls, emitted=len(nodes)))
    return report


def export_structure(case, *, phase, tokens, context=0, sequences=1,
                     scope="model", source=None, expand_calls=True, apply_body_defaults=True,
                     replicas=None, native_evidence=False, draft_prepare=False):
    """Export the first independent DP cohort, without evaluating its timing.

    ``case`` is the same complete declared case accepted by ``target_graphs``.
    The optional source is caller provenance and does not change ``graph_id``.
    Nodes describe declared work; a profiler must separately bind observed events.
    ``replicas`` projects serialization only, after the complete cohort frontend
    has bound peer traffic. It never changes the case or the cost-model workload.
    """
    graphs, options = _structure_inputs(
        case, phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope=scope, source=source, expand_calls=expand_calls,
        apply_body_defaults=apply_body_defaults, replicas=replicas,
        draft_prepare=draft_prepare)
    return export_op_structure(graphs, native_evidence=native_evidence, **options)


def structure_inputs_from_graphs(
    case, graphs, *, phase, tokens, context=0, sequences=1,
    scope="model", source=None, expand_calls=True, apply_body_defaults=True,
    replicas=None, draft_prepare=False, physical_graphs=None,
):
    """Build ``export_op_structure`` inputs from already-lowered cohort graphs."""
    from .prediction.body_defaults import consume_target_declaration
    case = consume_target_declaration(case, graphs)
    cohort_replicas = sorted(graphs)
    selection = getattr(graphs, "default_selection", None)
    projection = {}
    if replicas is not None:
        chosen = sorted(set(replicas))
        if not chosen or any(type(r) is not int or r not in graphs for r in chosen):
            raise ValueError("structure_replica_projection_requires_cohort_members")
        projection = {
            "cohort_replicas": cohort_replicas,
            "replica_projection": {
                "selected": chosen,
                "basis": "serialization_only_after_full_cohort_traffic_binding",
            },
        }
        graphs = {replica: graphs[replica] for replica in chosen}
    from mindie_sim.prediction.operators.hyper_connections.native_fusion import lower_native_fusions
    from .prediction.mechanism import Geometry

    if physical_graphs is None:
        physical_graphs = {replica: lower_native_fusions(ops, case.runtime)
                           for replica, ops in graphs.items()}
    else:
        physical_graphs = {replica: physical_graphs[replica] for replica in graphs}
    geometries = {}
    parallel = case.runtime.parallel
    width = parallel.tp * parallel.pcp
    for replica, operators in graphs.items():
        for op in (*operators, *physical_graphs[replica]):
            if op.plan_required or "fusion_group_key" in dict(op.metadata):
                key = replica, op.stage
                if key not in geometries:
                    start = (replica * parallel.pp + op.stage) * width
                    geometries[key] = Geometry.from_devices(
                        case.machine.dies[start:start + width],
                        links=case.machine.links,
                        transfer_channels=case.machine.transfer_channels,
                    )
    software = case.runtime.to_dict()
    selected = {key: software[key] for key in _SOFTWARE_FIELDS if key in software}
    selected["kernels"] = case.runtime.kernels
    return graphs, dict(
        model=case.model.to_dict(), source=source,
        kernels=case.runtime.kernels, expand_calls=expand_calls,
        resource_graphs=physical_graphs, resource_geometries=geometries,
        execution={
            "phase": phase, "tokens": tokens, "context": context,
            "sequences": sequences, "scope": scope, "software": selected,
            **({"draft_prepare": True} if draft_prepare else {}),
            **({"apply_body_defaults": apply_body_defaults,
                "default_mechanisms": selection} if scope == "body" else {}),
            **projection,
            "replica_scope": "first_independent_dp_cohort",
            "replica_axis": "data_parallel",
            "shard_scope": "representative_tp_pcp_shard",
            "stage_scope": "all_pipeline_stages",
            **({"resource_geometry": {
                f"replica:{replica}/stage:{stage}": {
                    key: field for key, field in asdict(value).items()
                    if key not in {"rates", "compute_peaks", "compute_arch", "datapath"}
                }
                for (replica, stage), value in geometries.items()}}
               if geometries else {}),
        },
    )


def _structure_inputs(case, *, phase, tokens, context=0, sequences=1,
                      scope="model", source=None, expand_calls=True,
                      apply_body_defaults=True, replicas=None,
                      draft_prepare=False):
    """Bind the complete cohort once before choosing an export consumer."""
    if not isinstance(draft_prepare, bool):
        raise ValueError("structure_draft_prepare_must_be_bool")
    case = parse_case(case)
    graphs = target_graphs(
        case, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope, resource_fusion=False,
        draft_prepare=draft_prepare,
        apply_body_defaults=apply_body_defaults,
    )
    return structure_inputs_from_graphs(
        case, graphs, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope, source=source,
        expand_calls=expand_calls, apply_body_defaults=apply_body_defaults,
        replicas=replicas, draft_prepare=draft_prepare,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--phase", required=True, choices=("prefill", "decode"))
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--scope", choices=("body", "model"), default="model")
    parser.add_argument("--replicas", type=int, nargs="+",
                        help="Serialize these DP replicas after full-cohort traffic binding")
    parser.add_argument("--draft-prepare", action="store_true",
                        help="Include declared draft cache preparation after model prefill")
    parser.add_argument("--no-body-defaults", dest="apply_body_defaults", action="store_false")
    parser.add_argument("--no-expand-calls", dest="expand_calls", action="store_false")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    with args.case.open("rb") as handle:
        content = handle.read(MAX_CASE_BYTES + 1)
    if len(content) > MAX_CASE_BYTES:
        parser.error("case metadata exceeds 50 MiB")
    result = export_structure(
        json.loads(content), phase=args.phase, tokens=args.tokens,
        context=args.context, sequences=args.sequences, scope=args.scope,
        expand_calls=args.expand_calls, apply_body_defaults=args.apply_body_defaults,
        replicas=args.replicas, draft_prepare=args.draft_prepare,
        source={"kind": "case_file", "sha256": sha256(content).hexdigest()},
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
