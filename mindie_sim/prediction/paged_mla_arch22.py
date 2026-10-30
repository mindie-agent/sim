"""Finite arch22 BF16 paged MLA with explicit GM communication and source owners.

[Role] Execute declared common SplitCore ownership through the shared Engine.
[Invariants] Aliased K/V are separately read. Q is reloaded per S2 tile; QK,
  probability and PV cross GM. Atomic updates retain read/write traffic.
  C1(i), V1/C2(i-1), V2(i-2) have two real drains and finite source buffers.
[Timeouts & Locks] O(batch + owners + finite active tile windows), no locks.
[Side-Effect Policy] Immutable virtual work only; no device, source or timing I/O.
"""
from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .paged_mla_arch22_dispatch import Dispatch, source_dispatch
from .paged_mla_arch22_matrix import matrix_program

IMPLEMENTATION = 'ascend_paged_mla_decode_arch22'
SOURCE = {
    'repository': 'cann/ops-transformer',
    'revision': 'a3925fd49cd256ad478d2faf983bf2961b8ae2c8',
    'directory': 'attention/fused_infer_attention_score/op_kernel/arch22',
    'sha256': {
        'fia_kernel_nonquant_mla.h': 'd76c338e39c6845d0c0ada93291cb0839016fb8142d5293ce6d1fd237d6d2a18',
        'fia_block_cube_nonquant_mla.h': '5b6e2459788367a0e122dc7edb5d31797abf3a8c1972e1b9d46cf6bed0868a2c',
        'fia_block_vec_nonquant_mla.h': '14737c7bca3cad6d8500d777fbc61997dbbb2d3cfa70e80e308fa09995d9c037',
        'fia_block_vec_flashdecode_arch22.h': '55e67b72af3d2365e3868d308e85f4586570ebbdd0fde40ae2fbacc491169eb9',
        'attention/common/op_host/split_core.cpp': '7a45c1e33258acd1b85b9a4919f1dd3d315f35201431983ddfdd79fa19710452',
    },
    'scope': 'q1/physical_heads8_or16/S2_512/D512/rope64; source compatible, binary identity unestablished',
}
KEY_TILE = 512
# Source allocation maxima; FD resets UB to 122112 bytes after the FA barrier.
BUFFERS = {'l1': 7*128*288*2, 'l0a': 65536, 'l0b': 65536,
           'l0c': 131072, 'ub': 170*1024}


def align(value, step=16):
    return (value+step-1)//step*step


@dataclass(frozen=True)
class Spec:
    contexts: tuple
    heads: int
    page_size: int
    dispatch: Dispatch
    dispatch_origin: str


def tile_work(keys, heads, *, first, last, splits, page_size):
    """Source transfers and aligned matrix work; no measured service or split fit."""
    hm, kn, columns = align(heads), align(keys), align(keys, 32)
    blocks = (keys+127)//128
    # QK has two K288 panels per N128. PV reloads P from L1 for each N128,
    # but each P row enters L1 only once; latent V is a separate GM source.
    query_key = (heads+keys)*576*2
    probability = heads*columns*2
    value_probability = keys*(512+heads)*2
    score_store, score_read = hm*kn*4, heads*columns*4
    pv = hm*512*4
    # K and Krope share the page lookup within each K288 call; V has four
    # N128 calls. Scalar instruction latency is deliberately not assigned.
    pages = sum((min(128,keys-start)+page_size-1)//page_size for start in range(0,keys,128))
    query_key += pages*2*4
    value_probability += pages*4*4
    works = {
        'qk_load': Work(resources={'hbm':query_key, 'mte2_aic':query_key}),
        'qk_operands': Work(resources={'mte1':(blocks*hm+kn)*576*2}),
        'qk': Work(matrix={'bf16':2*hm*kn*576}),
        'score_store': Work(resources={'hbm':score_store, 'fixpipe':score_store}),
        'score_read': Work(resources={'hbm':score_read, 'mte2_aiv':score_read}),
        # Scale/reduce/exp/update, AMLA row rescaling and BF16 cast. These
        # are algebraic vector operations, not reconstructed instruction issue.
        'softmax': Work(resources={'vector':8*heads*columns + heads*(30+(1 if last else 0))}),
        'probability': Work(resources={'hbm':probability, 'mte3':probability}),
        'nupdate_compute': Work(resources={} if first else {'vector':hm*128}),
        'nupdate': Work(resources={} if first else {'hbm':2*pv, 'mte3':pv}),
        'pv_load': Work(resources={'hbm':value_probability, 'mte2_aic':value_probability}),
        'pv_operands': Work(resources={'mte1':(4*hm+512)*kn*2}),
        'pv': Work(matrix={'bf16':2*hm*kn*512}),
        'partial': Work(resources={'hbm':pv*(1 if first else 2), 'fixpipe':pv}),
    }
    if last:
        read = heads*512*4
        write = heads*(512*4+64 if splits > 1 else 512*2)
        works.update({
            'output_read': Work(resources={'hbm':read,'mte2_aiv':read}),
            'output_normalize': Work(resources={'vector':heads*512*(4 if splits>1 else 5)}),
            'output_write': Work(resources={'hbm':write,'mte3':write}),
        })
    return works


def merge_work(heads, splits):
    read = splits*heads*(512*4+64)
    write = heads*512*2
    return (
        Work(resources={'hbm':read, 'mte2_aiv':read}),
        Work(resources={'vector':heads*(16+48*splits+1024*splits)}),
        Work(resources={'hbm':write, 'mte3':write}),
    )


def build(op, geometry):
    raw = dict(op.metadata)['paged_mla_geometry']
    contexts, heads, page = tuple(raw['contexts']), raw['heads'], raw['page_size']
    if (not contexts or any(type(c) is not int or c < 512 for c in contexts)
            or heads not in (8,16) or page < 1 or KEY_TILE % page):
        raise ModelingError('paged_mla_arch22_requires_q1_heads8_or16_context_ge512_page_divisor512')
    declared = raw.get('dispatch')
    dispatch = Dispatch.parse(declared if declared is not None else source_dispatch(
        contexts,heads,geometry.compute_units),contexts,heads,geometry.compute_units)
    for pool, needed in BUFFERS.items():
        if geometry.buffers.get(pool,0) < needed:
            raise ModelingError(f'paged_mla_arch22_source_pool_capacity:{pool}:{needed}')
    ratios = {'mte1':1, 'mte2_aic':1, 'fixpipe':1,
              'vector':2, 'mte2_aiv':2, 'mte3':2}
    if any(geometry.resource_units.get(role) != geometry.compute_units*ratio
           for role,ratio in ratios.items()):
        raise ModelingError('paged_mla_arch22_requires_paired_aic_aiv_physical_pipes')
    origin = 'source_common_SplitCore_q1_geometry' if declared is None else 'declared_common_SplitCore_endpoints'
    return _plan(Spec(contexts,heads,page,dispatch,origin),geometry.compute_units)


@lru_cache(maxsize=64)
def _plan(spec, cores):
    owners = spec.dispatch.owners
    total = Work()
    for jobs in owners:
        for seq,start,count,splits in jobs:
            for tile in range(count):
                keys = min(KEY_TILE,spec.contexts[seq]+1-(start+tile)*KEY_TILE)
                for work in tile_work(keys,spec.heads,first=tile==0,last=tile+1==count,
                                      splits=splits,page_size=spec.page_size).values():
                    total += work
    for jobs in spec.dispatch.merges:
        for _,_,heads,splits in jobs:
            for work in merge_work(heads,splits):
                total += work

    def lower(ctx):
        def stage(name,work,owner,*,aiv=None,**kwargs):
            roles = set(work.resources) | ({'cube'} if work.matrix else set())
            placement = ctx.owned(owner,cores,roles-{'hbm'})
            # Source vecDealM<=16 gives all live rows to subblock zero.
            for role in roles & {'vector','mte2_aiv','mte3'}:
                placement[ctx.resource_names[role]] = (2*owner if aiv is None else aiv,)
            return Stage(name,ctx.bind(work),placement=placement,**kwargs)

        @lru_cache(maxsize=256)
        def matrix(kind,keys,owner):
            return matrix_program(ctx,kind,keys,spec.heads,spec.page_size,owner,cores,len(owners))

        def matrix_steps(kind,keys,owner,*,after=(),previous=()):
            # The cached compact IR is a three-stage chain. Put that chain in
            # the existing tile window; zero-time nested wrappers add no work
            # or dependency and otherwise dominate large-batch event counts.
            load,steady,cube = matrix(kind,keys,owner).stages
            return (replace(load,name=kind+'_load',after=after,previous=previous),
                    replace(steady,name=kind+'_steady',after=(kind+'_load',)),
                    replace(cube,name=kind,after=(kind+'_steady',)))

        def owner_program(owner):
            jobs = owners[owner]
            prefix = [0]
            for _,_,count,_ in jobs:
                prefix.append(prefix[-1]+count)
            count = prefix[-1]

            def keys_at(index):
                job = bisect_right(prefix,index)-1
                seq,start,size,splits = jobs[job]
                offset = index-prefix[job]
                keys = min(KEY_TILE,spec.contexts[seq]+1-(start+offset)*KEY_TILE)
                return keys,offset,size,splits

            def work_at(index):
                keys,offset,size,splits = keys_at(index)
                return tile_work(keys,spec.heads,first=offset==0,last=offset+1==size,
                                 splits=splits,page_size=spec.page_size)

            def cycle(index):
                steps = []
                if index < count:
                    w = work_at(index)
                    steps.extend(matrix_steps('qk',keys_at(index)[0],owner,previous=('qk','pv')))
                    steps.extend((
                        stage('score_store',w['score_store'],owner,after=('qk',),
                              previous=('probability',)),
                    ))
                else:
                    steps.append(Stage('qk'))
                if 1 <= index <= count:
                    w = work_at(index-1)
                    steps.extend((
                        stage('score_read',w['score_read'],owner,
                              previous=('score_store','nupdate','output_write')),
                        stage('softmax',w['softmax'],owner,after=('score_read',)),
                        stage('probability',w['probability'],owner,after=('softmax',)),
                        stage('nupdate_compute',w['nupdate_compute'],owner,
                              after=('probability',),previous=('partial',)),
                        stage('nupdate',w['nupdate'],owner,after=('nupdate_compute',)),
                    ))
                    steps.extend(matrix_steps('pv',keys_at(index-1)[0],owner,
                                              after=('qk','probability'),previous=('pv',)))
                    steps.extend((
                        stage('partial',w['partial'],owner,after=('pv','nupdate'),
                              previous=(f'output_release{(bisect_right(prefix,index-1)-1)%2}',)),
                    ))
                else:
                    steps.append(Stage('nupdate'))
                if index >= 2:
                    w = work_at(index-2)
                    if 'output_read' in w:
                        steps.extend((
                            stage('output_read',w['output_read'],owner,
                                  after=('nupdate',),previous=('partial',)),
                            stage('output_normalize',w['output_normalize'],owner,after=('output_read',)),
                            stage('output_write',w['output_write'],owner,after=('output_normalize',)),
                            Stage(f'output_release{(bisect_right(prefix,index-2)-1)%2}',
                                  after=('output_write',)),
                        ))
                return tuple(steps)
            return PipelineProgram(count+2,3,cycle)

        def merge_program(aiv,jobs):
            def sequence(index):
                _,_,heads,splits = jobs[index]
                read,compute,write = merge_work(heads,splits)
                return (
                    stage('read',read,aiv//2,aiv=aiv),
                    stage('reduce',compute,aiv//2,aiv=aiv,after=('read',)),
                    stage('write',write,aiv//2,aiv=aiv,after=('reduce',)),
                )
            return PipelineProgram(len(jobs),1,sequence)

        allocations = tuple(ctx.allocate(f'{pool}_{owner}',pool,size,
            (2*owner,2*owner+1) if pool=='ub' else (owner,))
            for owner in range(len(owners)) for pool,size in BUFFERS.items())
        return KernelProgram((
            Stage('reserve',acquire=allocations),
            Stage('producers',after=('reserve',),program=ParallelProgram(tuple(
                owner_program(owner) for owner in range(len(owners))))),
            Stage('merge',after=('producers',),program=ParallelProgram(tuple(
                merge_program(aiv,jobs) for aiv,jobs in enumerate(spec.dispatch.merges)))),
            Stage('release',after=('merge',),free=allocations),
        ))
    return OperatorPlan(total,lower,{
        'implementation':IMPLEMENTATION,'source':SOURCE,'source_execution_domain':'cube_vector',
        'host_dispatch':spec.dispatch_origin,
        'active_aic_owners':len(owners),'active_fd_aiv_owners':len(spec.dispatch.merges),
        'physical_segments':sum(map(len,owners)),'buffers_per_owner':BUFFERS,
        'tile_sequence':'C1(i);V1_C2(i-1);V2(i-2);two_drains',
        'matrix_micro_geometry':{'qk_n':128,'qk_k_l1':288,'qk_k_l0':96,
                                 'pv_n':128,'pv_k_l1':256,'pv_k_l0':128},
        'matrix_pipeline':'three_L1_refill_slots_two_L0_consumer_slots; compact_nominal_recurrence',
        'gm_traffic':'source_transfers_and_atomic_read_write; no cache_hit_credit',
        'omitted_mechanisms':['cross_core_flag_latency','scalar_page_translation_latency',
                              'inner_instruction_issue','arbitrary_contention_pipeline_phase_alignment',
                              'vector_instruction_issue'],
    })
