"""Structure physical projections, target selection and indexing.

[Role] Compute node physical metrics, selective target graph exports, and
  indexed structure queries over lowered execution graphs.
[Invariants] Selective target hashes match full-model graph identity.
[Timeouts & Locks] Bounded in-memory graph traversal; no locks.
[Side-Effect Policy] Pure read-only analysis."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from functools import lru_cache
from hashlib import sha256
import json
import re
from tempfile import SpooledTemporaryFile


_WORK = ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "link_bytes")
_ANNOTATIONS = {
    "structure_subgraphs", "structure_subgraph_entries", "structure_subgraph_memberships",
    "input_conversion_subgraphs", "fused_origin_scopes", "source",
    "structure_block", "structure_role", "module_id",
    "fusion_group_key", "fusion_semantic_ids", "fusion_external_outputs",
    "graph_segment",
    "gathered_slice",  # Equal-sized rank slices differ in address, not copy work.
    "ep_rank",  # Rank placement label; uneven slot/work counts stay hashed.
    "norm_gamma_role",  # Role label (attn_norm/ffn_norm); format and bytes stay hashed.
}
_IDENTITY = {"op_id", "deps", "layer", "stage", "requests", "notes"}
_PRODUCER_REFERENCES = {
    "topk_ids_producer", "topk_weights_producer", "payload_producer", "inverse_index_producer",
    "activation_payload_producer", "activation_scale_producer",
}
_OCCURRENCE_PREFIX = re.compile(
    r"(?<![\w.])(?:draft(?:\d+|_prefill)\.)?s\d+\.l\d+(?=\.|$)")


def _signature(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + sha256(encoded.encode()).hexdigest()


def _scrub_occurrence(value, occurrences):
    if isinstance(value, str):
        return _OCCURRENCE_PREFIX.sub(
            lambda m: occurrences.setdefault(
                m.group(0), f"occ{len(occurrences)}"),
            value,
        )
    if isinstance(value, (list, tuple)):
        return [_scrub_occurrence(item, occurrences) for item in value]
    if isinstance(value, dict):
        items = sorted(
            value.items(),
            key=lambda kv: (_OCCURRENCE_PREFIX.sub("", kv[0]), kv[0]) if isinstance(kv[0], str) else ("", str(kv[0])),
        )
        return {
            (_scrub_occurrence(k, occurrences) if isinstance(k, str) else k): _scrub_occurrence(v, occurrences)
            for k, v in items
        }
    return value


def _metadata(value, references=None, occurrences=None):
    if not isinstance(value, dict):
        return value
    references = {} if references is None else references
    occurrences = {} if occurrences is None else occurrences
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
        elif isinstance(item, dict):
            item = _metadata(item, references, occurrences)
        elif isinstance(item, (list, tuple)):
            item = [_metadata(elem, references, occurrences)
                    if isinstance(elem, dict) else elem for elem in item]
        result[key] = _scrub_occurrence(item, occurrences)
    return result


@lru_cache(maxsize=16)
def _recipe_book(encoded):
    from mindie_sim.prediction.program import RecipeBook

    return RecipeBook.from_mapping(json.loads(encoded))


def _recipe_selection(op, bindings, *, book=None):
    """Unbound semantic labels do not distinguish the fallback resource program."""
    from mindie_sim.prediction.service_regime_domains.services import ResourceService

    if book is None:
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


def node_physical(op, *, recipe_bindings=None, book=None):
    """Keep every physical axis in OpSpec except its occurrence identity."""
    declaration = op.to_dict()
    for key in _IDENTITY:
        declaration.pop(key, None)
    declaration["metadata"] = _metadata(declaration["metadata"])
    declaration["recipe"], generic_matrix = _recipe_selection(op, recipe_bindings or {}, book=book)
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


ARRAYS = ('nodes', 'source_calls', 'subgraphs', 'resource_operators')


def select_structure_targets(case, target_ids, *, draft_prepare: bool = False, **geometry):
    requested = set(target_ids)
    targets = {}

    def collect(key, row):
        identifier = row.get('subgraph_id', row.get('node_id'))
        if key in {'nodes', 'subgraphs'} and identifier in requested:
            targets[identifier] = row

    identity = _scan_structure(case, collect, draft_prepare=draft_prepare, **geometry)
    return {**identity, 'targets': targets}


def _merge_math_rollup(total, part):
    """Sum per-replica family unions into the complete-export roll-up."""
    if total is None:
        return {
            **part,
            'families': {key: dict(row) for key, row in part['families'].items()},
            'uncovered_node_ids': list(part['uncovered_node_ids']),
        }
    for family, row in part['families'].items():
        current = total['families'].get(family)
        if current is None:
            total['families'][family] = dict(row)
            continue
        for key in ('subgraph_occurrences', 'member_node_count'):
            current[key] += row[key]
        for key in ('realized_hbm_bytes', 'parallel_comm_bytes',
                    'algebraic_cube_flops', 'algebraic_vector_ops',
                    'minimal_boundary_hbm_bytes'):
            current[key] = (None if current[key] is None or row[key] is None
                            else current[key] + row[key])
        current['unavailable_reasons'] = list(dict.fromkeys(
            (*current['unavailable_reasons'], *row['unavailable_reasons'])))
    total['uncovered_node_ids'] = sorted(
        set(total['uncovered_node_ids']) | set(part['uncovered_node_ids']))
    return total


def _merge_blockers(found, records):
    """Merge onboarding blockers with the full export's key/union rules."""
    for blocker in records:
        key = (int(blocker["layer_index"]), blocker["architecture"])
        if key not in found and key[0] < 0:
            same_core = next(
                (item for item in found if item[1] == key[1]), None)
            if same_core is not None:
                key = same_core
        record = found.get(key)
        if record is None:
            found[key] = {
                **blocker,
                "recipes": tuple(blocker["recipes"]),
                "op_ids": tuple(blocker["op_ids"]),
            }
            continue
        record["recipes"] = tuple(sorted(
            set(record["recipes"]) | set(blocker["recipes"])))
        record["op_ids"] = tuple(sorted(
            set(record["op_ids"]) | set(blocker["op_ids"])))


def _scan_structure(case, consume, *, draft_prepare: bool = False, **geometry):
    """Hash the complete export while visiting one replica's records at a time."""
    from mindie_sim.prediction.efficiency.unmodeled_gate import (
        HINT_KEYS, _hint_for, collect_blockers,
    )
    from mindie_sim.structure.export import (
        SCHEMA_VERSION as _STRUCTURE_SCHEMA_VERSION,
        _json_value, _structure_inputs, export_op_structure,
    )

    graphs, options = _structure_inputs(case, draft_prepare=draft_prepare, **geometry)
    external, all_ids = set(), set()
    blockers: dict[tuple[int, str], dict] = {}
    _merge_blockers(blockers, collect_blockers({"model": options['model']}))
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    execution = _json_value({**options['execution'], 'expand_calls': options['expand_calls'],
                             'replicas': sorted(graphs), 'kernel_bindings': options['kernels']})
    scalar = {'schema_version': _STRUCTURE_SCHEMA_VERSION, 'model': _json_value(options['model']),
              'execution': execution}
    math_rollup = None
    with ExitStack() as stack:
        streams = {key: stack.enter_context(SpooledTemporaryFile(max_size=1024**2))
                   for key in ARRAYS}
        counts = dict.fromkeys(ARRAYS, 0)
        for replica, operators in sorted(graphs.items()):
            # Executed core holes are part of the hashed identity just like a
            # declared layer hole; collect them from this replica's operators
            # before serialization drops occurrence metadata.
            _merge_blockers(blockers, collect_blockers(None, operators))
            part = export_op_structure({replica: operators}, **{
                **options, 'resource_graphs': {replica: options['resource_graphs'][replica]}})
            if 'subgraph_math_by_family' in part:
                math_rollup = _merge_math_rollup(
                    math_rollup, part['subgraph_math_by_family'])
            external.update(part['external_inputs'])
            for row in part['nodes']:
                all_ids.add(row['node_id'])
            for key, stream in streams.items():
                for row in part[key]:
                    if counts[key]:
                        stream.write(b',')
                    stream.write(encoder.encode(row).encode('utf-8'))
                    counts[key] += 1
                    consume(key, row)
            del part
        # Per-replica exports see cross-replica dependencies as external. Full
        # identity excludes every declared node regardless of its replica.
        scalar['external_inputs'] = sorted(external - all_ids)
        if math_rollup is not None:
            scalar['subgraph_math_by_family'] = math_rollup
        if blockers:
            scalar[HINT_KEYS] = [
                _hint_for(blockers[key]) for key in sorted(blockers)]
        digest = sha256()
        digest.update(b'{')
        for index, key in enumerate(sorted((*scalar, *ARRAYS))):
            if index:
                digest.update(b',')
            digest.update(encoder.encode(key).encode('utf-8') + b':')
            if key in streams:
                digest.update(b'[')
                streams[key].seek(0)
                while chunk := streams[key].read(1024**2):
                    digest.update(chunk)
                digest.update(b']')
            else:
                for chunk in encoder.iterencode(scalar[key]):
                    digest.update(chunk.encode('utf-8'))
        digest.update(b'}')
    return {'graph_id': 'sha256:' + digest.hexdigest(), 'execution': execution,
            'record_counts': counts}


SCHEMA_VERSION = 'mindie-model-structure-index/1'
# Source-role selection uses contracts, geometry, precision and lineage. The
# executable stage tree is already represented by physical_signature and work;
# retaining it for every layer/rank duplicates the largest metadata component.
_OMITTED_METADATA = frozenset({'program_execution'})


def _project_metadata(value):
    if isinstance(value, dict):
        return {key: _project_metadata(item) for key, item in value.items()
                if key not in _OMITTED_METADATA}
    if isinstance(value, list):
        return [_project_metadata(item) for item in value]
    return value


def _project_record(row):
    if 'operator' not in row:
        return row
    operator = row['operator']
    return {**row, 'operator': {
        **operator, 'metadata': _project_metadata(operator.get('metadata', {}))}}


class _SharedValues:
    """Bounded hash-consing of read-only index values across layers and ranks."""
    def __init__(self):
        self.values = OrderedDict()

    def share(self, value):
        if isinstance(value, dict):
            value = {key: self.share(item) for key, item in value.items()}
        elif isinstance(value, list):
            value = [self.share(item) for item in value]
        elif not isinstance(value, str):
            return value
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        key = sha256(encoded.encode('utf-8')).digest()
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        self.values[key] = value
        if len(self.values) > 8192:
            self.values.popitem(last=False)
        return value


def index_digest(index):
    """Recompute index integrity independently of the retained full graph digest."""
    digest = sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    for chunk in encoder.iterencode({k: v for k, v in index.items() if k != 'index_id'}):
        digest.update(chunk.encode('utf-8'))
    return 'sha256:' + digest.hexdigest()


def export_structure_index(case, **geometry):
    """Retain every node/subgraph and every peer, omitting only execution trees.

    Consumers verify ``index_id`` and retain ``graph_id`` as the full declaration
    identity. They must not verify this projection using the full export schema
    or reconstruct a cost model from its intentionally incomplete operators.
    """
    records = {key: [] for key in ARRAYS}
    shared = _SharedValues()

    def collect(key, row):
        records[key].append(shared.share(_project_record(row)))

    identity = _scan_structure(case, collect, **geometry)
    payload = {'schema_version': SCHEMA_VERSION, **identity, **records,
               'projection': {
                   'basis': 'complete_cohort_hash_before_record_projection',
                   'omitted_operator_metadata_keys': sorted(_OMITTED_METADATA),
                   'cost_reconstruction_supported': False,
                   'record_values': 'read_only_shared_values; deepcopy_before_mutation',
               }}
    return {**payload, 'index_id': index_digest(payload)}
