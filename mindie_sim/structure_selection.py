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


def _scan_structure(case, consume, **geometry):
    """Hash the complete export while visiting one replica's records at a time."""
    from .prediction.unmodeled_gate import (
        HINT_KEYS, _hint_for, collect_blockers,
    )

    graphs, options = _structure_inputs(case, **geometry)
    external, all_ids = set(), set()
    blockers: dict[tuple[int, str], dict] = {}
    _merge_blockers(blockers, collect_blockers({"model": options['model']}))
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    execution = _json_value({**options['execution'], 'expand_calls': options['expand_calls'],
                             'replicas': sorted(graphs), 'kernel_bindings': options['kernels']})
    scalar = {'schema_version': SCHEMA_VERSION, 'model': _json_value(options['model']),
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
