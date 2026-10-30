"""Identity-free physical declarations for repeated-structure reconciliation.

[Role] Export resource work and an implementation-sensitive cohort fingerprint.
[Invariants] Equal tensor contracts alone do not establish equal physical work.
  Unbound catalog labels resolve through the actual recipe selection; explicit
  bindings, work, implementation, stream and internal dependencies remain.
[Timeouts & Locks] Finite operator/subgraph metadata; no I/O or locks.
[Side-Effect Policy] Pure records; no timing engine or profiler dependency.
"""
from hashlib import sha256
from functools import lru_cache
import json


_WORK = ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "link_bytes")
_ANNOTATIONS = {
    "structure_subgraphs", "structure_subgraph_entries", "structure_subgraph_memberships",
    "input_conversion_subgraphs", "fused_origin_scopes", "source",
    "structure_block", "structure_role", "module_id",
    "fusion_group_key", "fusion_semantic_ids", "fusion_external_outputs",
    "graph_segment",
    "gathered_slice",  # Equal-sized rank slices differ in address, not copy work.
}
_IDENTITY = {"op_id", "deps", "layer", "stage", "requests", "notes"}
_PRODUCER_REFERENCES = {
    "topk_ids_producer", "topk_weights_producer", "payload_producer", "inverse_index_producer",
    "activation_payload_producer", "activation_scale_producer",
}


def _signature(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + sha256(encoded.encode()).hexdigest()


def _metadata(value, references=None):
    if not isinstance(value, dict):
        return value
    references = {} if references is None else references
    result = {}
    for key, item in sorted(value.items()):
        if key in _ANNOTATIONS:
            continue
        if key == "fusion_semantic_recipes" and isinstance(item, list):
            # Source fusion records keep ordered member programs; their layer
            # IDs identify occurrences, while recipe/shape/order remain physical.
            item = [{**entry, "op_id": f"member:{index}"}
                    if isinstance(entry, dict) and "op_id" in entry else entry
                    for index, entry in enumerate(item)]
        if key in _PRODUCER_REFERENCES and isinstance(item, str):
            # A repeated layer changes producer IDs, not the routing program.
            # Keep whether two operands alias the same producer and preserve
            # absent operands; do not erase the physical relationship itself.
            item = references.setdefault(item, f"producer:{len(references)}")
        result[key] = _metadata(item, references) if isinstance(item, dict) else item
    return result


@lru_cache(maxsize=16)
def _recipe_book(encoded):
    from .prediction.program import RecipeBook

    return RecipeBook.from_mapping(json.loads(encoded))


def _recipe_selection(op, bindings):
    """Unbound semantic labels do not distinguish the fallback resource program."""
    from .prediction.services import ResourceService

    encoded = json.dumps(bindings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    book = _recipe_book(encoded)
    if op.plan_required or op.recipe in book.recipes:
        return op.recipe, False
    selected = book.select(op)
    if isinstance(selected, ResourceService):
        return op.recipe, False
    if any({"recipe", "kernel"} & rule.get("match", {}).keys()
           for rule in bindings.get("invocations", ())):
        # Invocation rules and service curves can use the semantic recipe as
        # an explicit kernel identity even when resource selection falls back.
        return op.recipe, False
    generic_matrix = (
        dict(op.metadata).get("program_kind") == "matmul"
        and op.family not in book.families
        and not any("family" in rule.get("match", {})
                    for rule in bindings.get("invocations", ())))
    return ({"fallback_program": f"{type(selected).__module__}.{type(selected).__qualname__}"},
            generic_matrix)


def node_physical(op, *, recipe_bindings=None):
    """Keep every physical axis in OpSpec except its occurrence identity."""
    declaration = op.to_dict()
    for key in _IDENTITY:
        declaration.pop(key, None)
    declaration["metadata"] = _metadata(declaration["metadata"])
    declaration["recipe"], generic_matrix = _recipe_selection(op, recipe_bindings or {})
    if generic_matrix:
        # catalog.ops has already resolved these selection labels into operand
        # precisions, layouts, scale traffic and work. The fallback matmul
        # program reads those physical fields, not the originating model role.
        declaration["family"] = "program:matmul"
        for key in ("semantic_family", "quant_category", "catalog_category",
                    "ffn_category", "ffn_role"):
            declaration["metadata"].pop(key, None)
    if recipe_bindings:
        declaration["recipe_bindings"] = recipe_bindings
    work = {key: getattr(op, key) for key in _WORK}
    work["matrix_flops"] = dict(op.matrix_flops)
    # Directional copy, operand and output pipes remain independent physical
    # services; folding them into HBM or generic MTE2 would double-count bytes.
    work["resource_work"] = dict(dict(op.metadata).get("resource_work", {}))
    work["launches"] = int(op.family != "barrier")
    return {"physical_signature": _signature(declaration), "physical_work": work}


def subgraph_physical(by_id, group):
    """A repeated subgraph requires equal work and equal internal connectivity."""
    members = group["node_ids"]
    positions = {key: index for index, key in enumerate(members)}
    work = {key: 0 for key in (*_WORK, "launches")}
    matrix = {}
    resources = {}
    topology = []
    for key in members:
        node = by_id[key]
        for field in work:
            work[field] += node["physical_work"][field]
        for dtype, amount in node["physical_work"]["matrix_flops"].items():
            matrix[dtype] = matrix.get(dtype, 0) + amount
        for role, amount in node["physical_work"].get("resource_work", {}).items():
            resources[role] = resources.get(role, 0) + amount
        topology.append({
            "physical_signature": node["physical_signature"],
            "internal_dependencies": sorted(positions[dep] for dep in node["deps"] if dep in positions),
            "external_dependency_count": sum(dep not in positions for dep in node["deps"]),
            "external_output": key in group["external_outputs"],
        })
    work["matrix_flops"] = matrix
    work["resource_work"] = resources
    return {
        "physical_signature": _signature({
            "members": topology, "implementation": group["implementation"],
            "coverage": group["coverage"],
        }),
        "physical_work": work,
    }
