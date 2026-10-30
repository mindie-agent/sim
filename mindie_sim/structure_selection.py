"""Complete structure identity with a bounded selected-target working set.

[Role] Regenerate every replica while retaining only requested cohort records.
[Invariants] Hash bytes match the full structure export exactly. Every declared
  peer is bound before serialization; candidate selection cannot alter identity.
[Timeouts & Locks] One replica of JSON records at a time; four bounded spools.
  No subprocesses, locks or network. Temporary bytes are removed on return.
[Side-Effect Policy] Ephemeral serialization scratch only; no persistent output.
"""
from contextlib import ExitStack
from hashlib import sha256
import json
from tempfile import SpooledTemporaryFile

from .structure import SCHEMA_VERSION, _json_value, _structure_inputs, export_op_structure

ARRAYS = ('nodes', 'source_calls', 'subgraphs', 'resource_operators')


def select_structure_targets(case, target_ids, **geometry):
    requested = set(target_ids)
    targets = {}

    def collect(key, row):
        identifier = row.get('subgraph_id', row.get('node_id'))
        if key in {'nodes', 'subgraphs'} and identifier in requested:
            targets[identifier] = row

    identity = _scan_structure(case, collect, **geometry)
    return {**identity, 'targets': targets}


def _scan_structure(case, consume, **geometry):
    """Hash the complete export while visiting one replica's records at a time."""
    graphs, options = _structure_inputs(case, **geometry)
    external, all_ids = set(), set()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    execution = _json_value({**options['execution'], 'expand_calls': options['expand_calls'],
                             'replicas': sorted(graphs), 'kernel_bindings': options['kernels']})
    scalar = {'schema_version': SCHEMA_VERSION, 'model': _json_value(options['model']),
              'execution': execution}
    with ExitStack() as stack:
        streams = {key: stack.enter_context(SpooledTemporaryFile(max_size=1024**2))
                   for key in ARRAYS}
        counts = dict.fromkeys(ARRAYS, 0)
        for replica, operators in sorted(graphs.items()):
            part = export_op_structure({replica: operators}, **{
                **options, 'resource_graphs': {replica: options['resource_graphs'][replica]}})
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
