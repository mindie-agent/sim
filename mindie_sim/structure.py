"""Export shared model and operator structure without executing a cost model.

[Role] Reuse the semantic frontend and implementation expansion for profiling.
[Invariants] Node identities preserve layer/replica instances. Declared tensor
  contracts and software boundaries are not evidence that kernels were observed.
  Source calls retain their children; unknown logical origins are not invented.
[Timeouts & Locks] One finite target cohort, metadata only; no device or locks.
[Side-Effect Policy] The CLI reads one bounded case and writes requested JSON.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from .prediction.operator_analysis import target_graphs
from .prediction.program import RecipeBook
from .prediction.structure import parse_case
from .structure_contracts import comparison_scope, predicted_contract
from .structure_implementations import native_implementation
from .structure_physical import node_physical, subgraph_physical
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
    "gate_up_projection": "ffn", "down_projection": "ffn", "swiglu": "ffn",
    "grouped_gemm": "ffn", "embedding": "embedding", "logits": "logits",
}


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _node_id(replica, op_id):
    return f"rank:{replica}/{op_id}"


def _block(op):
    if metadata(op).get("structure_block"):
        return metadata(op)["structure_block"]
    return _FAMILY_BLOCKS.get(op.family)


def _node(op, replica, *, source_call=None, block=None, role=None, structure=None,
          kernels=None, book=None, geometry=None):
    if geometry is not None and op.source_boundary != "call":
        # Source planners own physical declarations. Preparing a plan does not
        # start its program, consult timings, or resolve captured ABI identity.
        op, _ = book.prepare(op, geometry)
    operator = op.to_dict(include_work=False)
    contract = predicted_contract(operator)
    meta = dict(op.metadata)
    semantic = meta.get("semantic_metadata", {})
    return _json_value({
        "node_id": _node_id(replica, op.op_id),
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
        "implementation": native_implementation(operator, contract) or {},
        **node_physical(op, recipe_bindings=kernels),
        "physical_work_basis": ("prepared_source_plan" if geometry is not None else
                                "source_declaration_without_resource_geometry")
                               if op.source_boundary == "kernel" else "operator_declaration",
        "contract_scope": comparison_scope(operator),
        "boundary": op.source_boundary or "semantic",
        "lineage": {
            "source_call_id": source_call,
            "fused_from": meta.get("fused_from", semantic.get("fused_from")),
            "semantic_recipe": meta.get("semantic_recipe"),
        },
    })


def export_op_structure(ops, *, model, execution, kernels=None, source=None,
                        expand_calls=True, resource_graphs=None, resource_geometries=None):
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
    """
    graphs = ops if isinstance(ops, Mapping) else {0: ops}
    graphs = {replica: tuple(operators) for replica, operators in graphs.items()}
    if (resource_graphs is None
            and (kernels or {}).get("lowerings", {}).get("native_fusion")):
        from .prediction.native_fusion import lower_native_fusions

        runtime = SimpleNamespace(kernels=kernels)
        resource_graphs = {replica: lower_native_fusions(operators, runtime)
                           for replica, operators in graphs.items()}
    phase = execution["phase"]
    book = RecipeBook.from_mapping(kernels or {})
    resource_geometries = resource_geometries or {}
    nodes, source_calls, external = [], [], set()
    declarations, entry_dependencies, memberships = {}, defaultdict(set), defaultdict(set)
    for replica, operators in sorted(graphs.items()):
        for op in operators:
            if op.phase != phase:
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
            if is_expanded:
                source_calls.append(_json_value({
                    "node_id": parent,
                    "operator": op.to_dict(include_work=False),
                    "children": [
                        _node_id(replica, child.op_id)
                        for child in expanded if child.op_id != op.op_id
                    ],
                }))
            nodes.extend(_node(child, replica, source_call=parent, block=_block(op),
                               role=meta.get("structure_role"),
                               structure=layer_structure(model, op.layer), kernels=kernels,
                               book=book, geometry=resource_geometries.get((replica, child.stage)))
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
    resource_operators = []
    if resource_graphs is not None:
        groups = {group["subgraph_id"]: group for group in subgraphs}
        for replica, operators in sorted(resource_graphs.items()):
            for op in operators:
                meta = dict(op.metadata)
                if "fusion_group_key" not in meta:
                    continue
                key = f"rank:{replica}/subgraph:{meta['fusion_group_key']}"
                group = groups[key]
                resource = _node(op, replica, block=group["block"], role=group["role"],
                                 structure=group["structure"], kernels=kernels, book=book,
                                 geometry=resource_geometries.get((replica, op.stage)))
                resource_operators.append(resource)
                boundary = {**meta["fusion_boundary"],
                            "operator_ids": [resource["node_id"]],
                            "kind": "selected_native_fusion"}
                group["resource_boundary"] = boundary
                group.update({field: resource[field]
                              for field in ("physical_work", "physical_signature")})
                group["contract_scope"] = subgraph_comparison_scope(
                    group["role"], [by_id[key] for key in group["node_ids"]], resource=resource)
                members = set(group["node_ids"])
                internal = {"kind": "internal_to_selected_native_fusion",
                            "subgraph_id": key, "operator_ids": [resource["node_id"]]}
                for member in members:
                    by_id[member].update(event_matchable=False, resource_boundary=internal)
                for other in subgraphs:
                    if other is not group and set(other["node_ids"]) <= members:
                        other.update(event_matchable=False, resource_boundary=internal)
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
    digest = sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(payload):
        digest.update(chunk.encode("utf-8"))
    return {
        **payload,
        "graph_id": "sha256:" + digest.hexdigest(),
        "source": _json_value(source),
    }


def export_structure(case, *, phase, tokens, context=0, sequences=1,
                     scope="model", source=None, expand_calls=True, apply_body_defaults=True,
                     replicas=None):
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
        apply_body_defaults=apply_body_defaults, replicas=replicas)
    return export_op_structure(graphs, **options)


def _structure_inputs(case, *, phase, tokens, context=0, sequences=1,
                      scope="model", source=None, expand_calls=True,
                      apply_body_defaults=True, replicas=None):
    """Bind the complete cohort once before choosing an export consumer."""
    case = parse_case(case)
    graphs = target_graphs(
        case, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope, resource_fusion=False,
        apply_body_defaults=apply_body_defaults,
    )
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
    from .prediction.native_fusion import lower_native_fusions
    from .prediction.mechanism import Geometry

    physical_graphs = {replica: lower_native_fusions(ops, case.runtime)
                       for replica, ops in graphs.items()}
    geometries = {}
    parallel = case.runtime.parallel
    width = parallel.tp * parallel.pcp
    for replica, operators in graphs.items():
        for op in (*operators, *physical_graphs[replica]):
            if op.plan_required or "fusion_group_key" in dict(op.metadata):
                key = replica, op.stage
                if key not in geometries:
                    start = (replica * parallel.pp + op.stage) * width
                    geometries[key] = Geometry.from_devices(case.machine.dies[start:start + width])
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
            **({"apply_body_defaults": apply_body_defaults,
                "default_mechanisms": selection} if scope == "body" else {}),
            **projection,
            "replica_scope": "first_independent_dp_cohort",
            "replica_axis": "data_parallel",
            "shard_scope": "representative_tp_pcp_shard",
            "stage_scope": "all_pipeline_stages",
            **({"resource_geometry": {f"replica:{replica}/stage:{stage}": asdict(value)
                                      for (replica, stage), value in geometries.items()}}
               if geometries else {}),
        },
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
        replicas=args.replicas,
        source={"kind": "case_json", "path": str(args.case.resolve()),
                "sha256": sha256(content).hexdigest()},
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
