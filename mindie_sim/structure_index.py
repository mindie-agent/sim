"""Verified matching index without retaining repeated executable stage ledgers.

[Role] Export contracts and source-role metadata for bounded profiler matching.
[Invariants] All DP peers are bound and hashed before projection. Full graph_id
  hashes the original complete structure; index_id hashes this projected payload.
  Program details are omitted explicitly, never used as a replacement cost graph.
[Timeouts & Locks] One replica of full records plus the index; bounded hash spools.
  No subprocesses, locks, device access or network.
[Side-Effect Policy] Ephemeral serialization scratch only; returns JSON records.
"""
from hashlib import sha256
import json
from collections import OrderedDict

from .structure_selection import ARRAYS, _scan_structure


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
