"""Finite BF16 paged MLA decode with shared latent K/V and native tile buffers.

[Role] Lower explicit dense MLA tile and KV-split declarations onto the Engine.
[Invariants] K_nope is V: one HBM load feeds both products. Scores go directly
  through Fixpipe to UB, probabilities through MTE3 to the same L1 ring. Each
  sequence has separate online state; split partials merge behind a barrier.
  Host split selection is declared, not inferred from measured duration.
[Timeouts & Locks] O(batch + active owners + finite tile windows), no locks.
[Side-Effect Policy] Immutable plans and virtual tasks only; no device or I/O.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .refill_steady import compact_periodic_pipeline
from .paged_mla_dispatch import Dispatch
from . import paged_mla_host_v2

IMPLEMENTATION = 'ascend_paged_mla_decode_arch35'
SOURCE = {
    'repository': 'cann/ops-transformer',
    'branch': 'arch35 BF16 noquant paged MLA',
    'kernel': 'attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_mla.h',
    'cube': 'attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h',
    'vector': 'attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_noquant_mla.h',
    'sha256': {'kernel': 'e2de9d747702b384de60e67e62d584dced476e47ace5f68a235af61e018dbbbb',
               'cube': 'c7ecef63fb1754133d8b2588b297cb4621a72a82d3faea6945d10a440ee21574',
               'vector': '176bd97f6286611570cf99549395a6b23d74ecd212fc9cd5d8d13ebb417b87b8'},
    'scope': 'M64/S2_128/D512/rope64; declared split count; source-compatible not binary identity',
}
KEY_TILE, HEAD_TILE = 128, 64
# Exact no-mask/no-LSE allocations: kernel CV queues plus Vector-owned queues.
BUFFERS = {'l1': 3*128*576*2 + 64*576*2, 'l0a': 65536, 'l0b': 65536,
           'l0c': 262144, 'ub': 2*32*(128+512)*4 + 6*256 + 2*1024 + 512
           + 32*512*4 + 33*128*2}


def aligned(value, step=16):
    return (value + step - 1) // step * step


def segments(contexts, requested):
    """Each record is one sequence's contiguous range of actual KV tiles."""
    result = []
    for sequence, context in enumerate(contexts):
        tiles = math.ceil((context + 1) / KEY_TILE)
        count = min(requested, tiles)
        small, extra = divmod(tiles, count)
        start = 0
        for split in range(count):
            size = small + (split < extra)
            result.append((sequence, start, size, count))
            start += size
    return tuple(result)


def tile_work(keys, heads, *, first, last):
    """Source memory directions and executed matrix footprint, without timers."""
    hm, kn = aligned(heads), aligned(keys)
    score = aligned(heads, 2) * aligned(keys, 8) * 4
    pv = aligned(heads, 2) * 512 * 4
    return (
        ('load', Work(resources={'hbm': keys*576*2, 'mte2_aic': keys*576*2})),
        ('qk_operands', Work(resources={'mte1': (hm + kn)*576*2})),
        ('qk', Work(matrix={'bf16': 2*hm*kn*576})),
        ('scores', Work(resources={'fixpipe': score})),
        ('softmax', Work(resources={'vector': 8*heads*keys + (26 if first else 32)*heads})),
        ('probability', Work(resources={'mte3': heads*KEY_TILE*2})),
        # MatmulN loads the declared M64 probability matrix once, then reuses
        # L0A across four N128 products; realM bounds the actual Mmad rows.
        ('pv_operands', Work(resources={'mte1': (HEAD_TILE + 512)*kn*2})),
        ('pv', Work(matrix={'bf16': 2*hm*kn*512})),
        ('partial', Work(resources={'fixpipe': pv})),
        ('update', Work(resources={'vector': heads*512*(2 if first else 3)
                                  + (heads*512 if last else 0)})),
    )


@dataclass(frozen=True)
class Spec:
    contexts: tuple
    heads: int
    kv_splits: int
    page_size: int
    dispatch: Dispatch | None = None
    host_policy: str = 'declared'

    @classmethod
    def parse(cls, op, cores):
        raw = dict(op.metadata)['paged_mla_geometry']
        contexts = tuple(raw['contexts'])
        result = cls(contexts, raw['heads'], raw['kv_splits'], raw['page_size'])
        if (not result.contexts or any(type(x) is not int or x < 0 for x in result.contexts)
                or type(result.kv_splits) is not int or result.kv_splits < 1
                or type(result.heads) is not int or not 1 <= result.heads <= HEAD_TILE
                or result.page_size < 1 or KEY_TILE % result.page_size):
            raise ModelingError('paged_mla_requires_q1_heads_le64_and_page_divisor_of128')
        dispatch = raw.get('dispatch')
        policy = paged_mla_host_v2.POLICY if dispatch == paged_mla_host_v2.POLICY else 'declared'
        if policy == paged_mla_host_v2.POLICY:
            capacity = raw.get('native_query_capacity', len(contexts))
            if capacity < len(contexts):
                raise ModelingError('paged_mla_query_capacity_below_active_sequences')
            dispatch = paged_mla_host_v2.source_dispatch(contexts, result.heads, cores, capacity)
        return cls(contexts, result.heads, result.kv_splits, result.page_size,
                   Dispatch.parse(dispatch, contexts, result.heads), policy)


def _ends(spec, segment):
    sequence, start, count, splits = segment
    query = Work(resources={'hbm': spec.heads*576*2, 'mte2_aic': spec.heads*576*2})
    # One 32-bit physical page index per encountered page; scalar issue and
    # translation latency remain declared omissions, not a timing floor.
    first = start*KEY_TILE
    end = min((start+count)*KEY_TILE, spec.contexts[sequence]+1)
    pages = (end + spec.page_size-1)//spec.page_size - first//spec.page_size
    query += Work(resources={'hbm': pages*4, 'mte2_aic': pages*4})
    out_bytes = spec.heads * (512*4+8 if splits > 1 else 512*2)
    return query, Work(resources={'hbm': out_bytes, 'mte3': out_bytes})


def merge_work(spec, count, heads=None):
    heads = spec.heads if heads is None else heads
    read = count * heads * (512*4+8)
    write = heads*512*2
    return (Work(resources={'hbm': read, 'mte2_aiv': read}),
            Work(resources={'vector': heads*(count*(3*512+6)+512)}),
            Work(resources={'hbm': write, 'mte3': write}))


def build(op, geometry):
    spec = Spec.parse(op, geometry.compute_units)
    for pool, needed in BUFFERS.items():
        if geometry.buffers.get(pool, 0) < needed:
            raise ModelingError(f'paged_mla_source_pool_capacity:{pool}:{needed}')
    required = {'mte1': 1, 'mte2_aic': 1, 'fixpipe': 1,
                'vector': 2, 'mte2_aiv': 2, 'mte3': 2}
    if any(geometry.resource_units.get(role) != geometry.compute_units*ratio
           for role, ratio in required.items()):
        raise ModelingError('paged_mla_requires_paired_aic_aiv_physical_pipes')
    if spec.dispatch and len(spec.dispatch.owners) > geometry.compute_units:
        raise ModelingError('paged_mla_dispatch_exceeds_physical_aic_owners')
    return _plan(spec, geometry.compute_units)


@lru_cache(maxsize=64)
def _plan(spec, cores):
    if spec.dispatch:
        by_owner = spec.dispatch.jobs()
        jobs = tuple(job for rows in by_owner for job in rows)
    else:
        jobs = segments(spec.contexts, spec.kv_splits)
        by_owner = tuple(jobs[owner::cores] for owner in range(min(cores,len(jobs))))
    total = Work()
    split_counts = {}
    for job in jobs:
        sequence, start, count, splits = job
        split_counts[sequence] = splits
        begin, end = _ends(spec, job)
        total += begin + end
        boundary_tiles = (0,) if count == 1 else (0, count - 1) if count > 1 else ()
        for tile in boundary_tiles:
            keys = min(KEY_TILE, spec.contexts[sequence]+1-(start+tile)*KEY_TILE)
            for _, work in tile_work(keys, spec.heads, first=tile == 0, last=tile+1 == count):
                total += work
        if count > 2:
            for _, work in tile_work(KEY_TILE, spec.heads, first=False, last=False):
                total += work.scaled(count - 2)
    for count in split_counts.values():
        if count > 1:
            for work in merge_work(spec, count):
                total += work

    def lower(ctx):
        active = len(by_owner)

        def stage(name, work, owner, *, aiv=None, **kwargs):
            roles = set(work.resources) | ({'cube'} if work.matrix else set())
            placement = ctx.owned(owner, cores, roles-{'hbm'})
            vector_roles = roles & {'vector', 'mte2_aiv', 'mte3'}
            if aiv is not None:
                for role in vector_roles:
                    placement[ctx.resource_names[role]] = (aiv,)
                return Stage(name,ctx.bind(work),placement=placement,**kwargs)
            if vector_roles and spec.heads > 1:
                # InitTask uses actualM=q*g (q=1 here), then ceil/floor halves.
                # M64 is the capacity, not a fixed first-AIV M32 assignment.
                # Each half owns its physical pipes; no idle sibling borrowing.
                children = []
                for subblock, rows in enumerate(((spec.heads+1)//2, spec.heads//2)):
                    own = dict(placement)
                    for role in vector_roles:
                        own[ctx.resource_names[role]] = (2*owner+subblock,)
                    children.append(KernelProgram((Stage('subblock',
                        ctx.bind(work.scaled(rows/spec.heads)), placement=own),)))
                return Stage(name, program=ParallelProgram(tuple(children)), **kwargs)
            if spec.heads == 1:
                for role in vector_roles:
                    key = ctx.resource_names[role]
                    placement[key] = placement[key][:1]
            return Stage(name, ctx.bind(work), placement=placement, **kwargs)

        def owner_program(owner):
            local = by_owner[owner]
            def job_program(job):
                sequence, start, count, _ = job
                begin, end = _ends(spec, job)

                @lru_cache(maxsize=8)
                def cycle_template(keys, old_keys, first, old_first, old_last, parity):
                    # ExecuteTask issues C1(i), C2(i-1), and independently
                    # V1(i), V2(i-1). Distinct physical pipes may overlap, but
                    # the next driver iteration cannot reorder their MMAs.
                    result = []
                    if keys is not None:
                        work = dict(tile_work(keys, spec.heads, first=first, last=False))
                        result.extend((
                            stage('load', work['load'], owner, previous=('qk','pv')),
                            stage('qk_operands', work['qk_operands'], owner, after=('load',)),
                            stage('qk', work['qk'], owner, after=('qk_operands',)),
                            stage('scores', work['scores'], owner, after=('qk',),
                                  previous=(f'score_release{parity}',)),
                            stage('softmax', work['softmax'], owner, after=('scores',),
                                  previous=('update',)),
                            stage('probability', work['probability'], owner, after=('softmax',)),
                            Stage(f'score_release{parity}', after=('softmax',)),
                        ))
                    else:
                        # Source flushes C2/V2 of the last valid task without
                        # fabricating a further QK or score/probability tile.
                        result.extend((Stage('qk'), Stage('softmax')))
                    if old_keys is not None:
                        work = dict(tile_work(old_keys, spec.heads, first=old_first,
                                              last=old_last))
                        result.extend((
                            stage('pv_operands', work['pv_operands'], owner,
                                  after=('qk',), previous=('probability',)),
                            stage('pv', work['pv'], owner, after=('pv_operands',)),
                            stage('partial', work['partial'], owner, after=('pv',),
                                  previous=(f'partial_release{1-parity}',)),
                            stage('update', work['update'], owner, after=('partial','softmax')),
                            Stage(f'partial_release{1-parity}', after=('update',)),
                        ))
                    return tuple(result)

                def cycle(index):
                    # Only prologue, two ring parities and tail/flush differ.
                    # Reuse immutable physical stages; every original event
                    # and dependency still executes in the same order.
                    end = spec.contexts[sequence]+1
                    keys = min(KEY_TILE, end-(start+index)*KEY_TILE) if index < count else None
                    old_keys = min(KEY_TILE, end-(start+index-1)*KEY_TILE) if index else None
                    return cycle_template(keys, old_keys, index == 0, index == 1,
                                          index == count, index % 2)
                return KernelProgram((stage('query', begin, owner),
                    Stage('tiles', after=('query',), program=compact_periodic_pipeline(
                        ctx, count + 1, 3, cycle, head=2, tail=2)),
                    stage('output', end, owner, after=('tiles',))))

            return PipelineProgram(len(local), 1,
                lambda i: (Stage('job', program=job_program(local[i])),))

        merged = []
        def merge_program(count, owner, heads, aiv=None):
            read, _, write = merge_work(spec, count, heads)
            update = Work(resources={'vector': heads*(3*512+6)})
            return KernelProgram((Stage('partials', program=PipelineProgram(count, 1, lambda _: (
                    stage('read', read.scaled(1/count), owner,aiv=aiv),
                    stage('accumulate', update, owner,aiv=aiv, after=('read',))))),
                stage('normalize', Work(resources={'vector':heads*512}), owner,aiv=aiv,
                      after=('partials',)), stage('write', write, owner,aiv=aiv, after=('normalize',))))

        if spec.dispatch:
            for aiv,(sequence,_,heads) in enumerate(spec.dispatch.merges):
                merged.append(merge_program(split_counts[sequence],aiv//2,heads,aiv))
        else:
            for owner in range(active):
                counts = tuple(count for sequence, count in split_counts.items()
                               if count > 1 and sequence % cores == owner)
                if not counts:
                    continue
                # Native FD owner metadata is not inferred; this declared
                # owner processes each sequence serially using bounded UB.
                merged.append(PipelineProgram(len(counts), 1,
                    lambda i, counts=counts, owner=owner: (
                        Stage('sequence', program=merge_program(counts[i], owner,spec.heads)),)))
        allocations = tuple(ctx.allocate(f'{pool}_{owner}', pool, size,
                            tuple(range(2*owner,2*owner+2)) if pool == 'ub' else (owner,))
                            for owner in range(active) for pool, size in BUFFERS.items())
        return KernelProgram((Stage('reserve', acquire=allocations),
            Stage('producers', after=('reserve',), program=ParallelProgram(tuple(
            owner_program(i) for i in range(active)))),
            Stage('merge', after=('producers',), program=ParallelProgram(tuple(merged))),
            Stage('release', after=('merge',), free=allocations)))

    return OperatorPlan(total, lower, {
        'implementation': IMPLEMENTATION, 'source': SOURCE,
        'source_execution_domain': 'cube_vector', 'query_sequences': len(spec.contexts),
        'requested_kv_splits': spec.kv_splits if spec.host_policy == 'declared' else None,
        'physical_segments': len(jobs),
        'active_aic_owners': len(by_owner), 'buffers_per_owner': BUFFERS,
        'tile_sequence': 'C1(i)_before_C2(i-1); V1(i)_before_V2(i-1)',
        'host_dispatch': (spec.host_policy if spec.host_policy != 'declared' else
            'explicit_AIC_KV_and_AIV_head_metadata' if spec.dispatch else
            'explicit_equal_tile_splits_and_round_robin_owners; native_SplitCore_not_reconstructed'),
        **({'host_source': paged_mla_host_v2.SOURCE}
           if spec.host_policy == paged_mla_host_v2.POLICY else {}),
        'omitted_mechanisms': ['scalar_page_translation_latency', 'cross_core_flag_latency',
                               'vector_instruction_issue_detail',
                               'inner_L0_instruction_issue'],
    })
