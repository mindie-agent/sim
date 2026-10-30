"""Explicit three-level mesh RS/AG composition, independent of native selectors.

[Role] Build rank groups, equal shards and remote events on the common phase DAG.
[Invariants] RS shrinks one rank dimension at a time; reverse AG restores it.
  Directed payload counts once on wire and once at each local HBM endpoint.
  Analytic scratch copies and control events are declarations, not source claims.
[Timeouts & Locks] Six bounded O(R squared) exchanges; a 16-entry immutable cache.
[Side-Effect Policy] No native dispatch inference, observations, I/O or mutation.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import prod

from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HcclPhase, HcclPlan
from .hccl_hierarchy import _notify

RECIPE = 'analytic_hierarchy_phase'


@dataclass(frozen=True)
class MeshLevel:
    reduce_scatter: str
    allgather: str
    protocol: str


@dataclass(frozen=True)
class HierarchySelection:
    ranks: int
    payload_bytes: int
    rank_dimensions: tuple[int, int, int]
    rank_order: tuple[int, ...]
    levels: tuple[MeshLevel, ...]
    dtype_bytes: int = 2
    signal_bytes: int | None = None
    collective: str = 'allreduce'
    expansion: str = 'ANALYTIC'
    template: str = 'three_level_mesh'
    selection_basis: str = 'explicit_analytic_composition'

    def to_dict(self):
        return asdict(self)


def select(raw, *, collective, ranks, payload_bytes):
    allowed = {'rank_dimensions', 'rank_order', 'levels', 'dtype_bytes', 'signal_bytes'}
    if set(raw) - allowed or not {'rank_dimensions', 'levels'} <= set(raw):
        raise ModelingError('analytic_hierarchy_requires_rank_dimensions_and_level_templates')
    dims = tuple(raw['rank_dimensions'])
    if (len(dims) != 3 or any(type(d) is not int or d < 2 for d in dims)
            or prod(dims) != ranks or collective != 'allreduce'):
        raise ModelingError('analytic_hierarchy_requires_three_nontrivial_allreduce_dimensions')
    order = tuple(raw.get('rank_order', range(ranks)))
    if (len(order) != ranks or any(type(r) is not int for r in order)
            or set(order) != set(range(ranks))):
        raise ModelingError('analytic_hierarchy_rank_order_must_permute_communicator_positions')
    levels = []
    for raw_level in raw['levels']:
        if set(raw_level) != {'reduce_scatter', 'allgather', 'protocol'}:
            raise ModelingError('analytic_hierarchy_level_requires_explicit_templates_and_protocol')
        level = MeshLevel(**raw_level)
        if (level.reduce_scatter != 'mesh' or level.allgather != 'mesh'
                or level.protocol not in {'ack_data', 'data_signal'}):
            raise ModelingError('analytic_hierarchy_only_mesh_ack_data_or_data_signal_implemented')
        levels.append(level)
    if len(levels) != 3:
        raise ModelingError('analytic_hierarchy_requires_three_level_declarations')
    dtype = raw.get('dtype_bytes', 2)
    if (type(payload_bytes) is not int or payload_bytes < 0
            or dtype not in {1, 2, 4, 8} or payload_bytes % (ranks * dtype)):
        raise ModelingError('analytic_hierarchy_requires_equal_integer_element_shards')
    signal = raw.get('signal_bytes')
    if signal is not None and (type(signal) is not int or signal < 0):
        raise ModelingError('analytic_hierarchy_signal_frame_requires_nonnegative_bytes')
    return HierarchySelection(ranks, payload_bytes, dims, order, tuple(levels), dtype, signal)


def groups(selection, level):
    """Dimension zero varies fastest; rank_order maps slots to communicator ranks."""
    stride = prod(selection.rank_dimensions[:level])
    width = selection.rank_dimensions[level]
    return tuple(tuple(selection.rank_order[start + offset + k * stride]
                       for k in range(width))
                 for start in range(0, selection.ranks, stride * width)
                 for offset in range(stride))


def stages(selection):
    """(level, primitive, local input bytes, local output bytes) in DAG order."""
    result, size = [], selection.payload_bytes
    for level, width in enumerate(selection.rank_dimensions):
        result.append((level, 'rs', size, size // width))
        size //= width
    for level in reversed(range(3)):
        result.append((level, 'ag', size, size * selection.rank_dimensions[level]))
        size *= selection.rank_dimensions[level]
    return tuple(result)


@lru_cache(maxsize=16)
def contract(selection, rank=0):
    s = selection
    if not 0 <= rank < s.ranks:
        raise ModelingError('analytic_hierarchy_rank_outside_communicator')
    if not s.payload_bytes:
        return HcclPlan(s, rank, (HcclPhase('empty'),), 1, 0)
    result, previous = [], ()
    scratch = 0
    for level, kind, input_size, output_size in stages(s):
        name = f'l{level}_{kind}'
        size = output_size if kind == 'rs' else input_size
        peers = s.rank_dimensions[level] - 1
        matrix = [[0] * s.ranks for _ in range(s.ranks)]
        for group in groups(s, level):
            for src in group:
                for dst in group:
                    if src != dst:
                        matrix[src][dst] = size
        edges = tuple((src, dst) for src, row in enumerate(matrix)
                      for dst, amount in enumerate(row) if amount)
        # Each primitive has its own output region: copy the resident shard,
        # receive the other shards, then reduce (RS) or retain ordered chunks (AG).
        copied = output_size if kind == 'rs' else input_size
        result.append(HcclPhase(name + '.copy', previous,
            (('hbm', 2 * copied), ('hccl_copy', copied))))
        deps = previous
        if s.levels[level].protocol == 'ack_data':
            ack = _notify(name + '.ack', deps,
                          tuple((dst, src) for src, dst in edges), rank)
            result.extend(ack)
            deps = (ack[-1].name,)
        result.append(HcclPhase(name + '.exchange', deps,
            (('hbm', 2 * peers * size),), exchange=tuple(map(tuple, matrix))))
        signal = _notify(name + '.complete', (name + '.exchange',), edges, rank)
        result.extend(signal)
        deps = (name + '.copy', signal[-1].name)
        if kind == 'rs':
            # Separate receive scratch + serial local SUM, unlike fused remote
            # reduce. The bytes are explicit rather than a latency multiplier.
            steps = tuple((f'reduce_{peer}', (('hbm', 3 * size), ('hccl_reduce', size)))
                          for peer in range(peers))
            result.append(HcclPhase(name + '.reduce', deps,
                (('hbm', 3 * peers * size), ('hccl_reduce', peers * size)), steps))
            previous = (name + '.reduce',)
            scratch = max(scratch, input_size + peers * size)
        else:
            result.append(HcclPhase(name + '.ready', deps))
            previous = (name + '.ready',)
            scratch = max(scratch, output_size)
    return HcclPlan(s, rank, tuple(result), max(s.rank_dimensions), scratch,
        unmodeled=(
            'analytic_composition_not_native_HCCL_executor_or_selector_identity',
            'whole_phase_peer_barriers_without_interlevel_chunk_pipeline',
            'scratch_bytes_ledger_not_physical_pool_reservation',
            'controller_issue_requires_declared_hccl_control_rate',
            'signal_frame_unknown_propagation_only' if s.signal_bytes is None else
                'signal_frame_bytes_explicitly_declared',
        ))


def build(op, geometry):
    from .hccl_aicpu import phase_plan
    raw = dict(op.metadata)['hccl_phase']
    spec = raw['selection']
    selection = select({k: spec[k] for k in ('rank_dimensions', 'rank_order', 'levels',
        'dtype_bytes', 'signal_bytes')}, collective=spec['collective'], ranks=spec['ranks'],
        payload_bytes=spec['payload_bytes'])
    plan = contract(selection, raw.get('rank', 0))
    return phase_plan(op, geometry, plan, {
        'model_kind': 'analytic_hierarchy',
        'rank_groups': [groups(selection, i) for i in range(3)],
        'stage_geometry': stages(selection),
    }, 'analytic_hierarchy_phase')
