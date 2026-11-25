"""Finite BF16 paged MLA decode with shared latent K/V and native tile buffers.

[Role] Lower explicit dense MLA tile and KV-split declarations onto the Engine.
[Invariants] K_nope is V: one HBM load feeds both products. Scores go directly
  through Fixpipe to UB, probabilities through MTE3 to the same L1 ring. Each
  sequence has separate online state; split partials merge behind a barrier.
  Host split selection is declared, not inferred from measured duration.
[Timeouts & Locks] O(batch + active owners + finite tile windows), no locks.
[Side-Effect Policy] Immutable plans and virtual tasks only; no device or I/O.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelinePhase, PipelineProgram, PipelineTemplate,
    PipelineTemplateProgram, Stage,
)
from .common import (
    DENSE_MAX_PAGE_SIZE,
    DENSE_MAX_PHYSICAL_HEADS,
    DENSE_PAGE_ALIGNMENT,
    FIA_ABI,
    FLASH_ABI,
)
from .paged_mla_dispatch import Dispatch
from mindie_sim.prediction.operators.paged_mla import paged_mla_host_v2

TARGET_BACKEND: Literal["ascend"] = "ascend"
IMPLEMENTATION = 'ascend_paged_mla_decode_arch35'
NATIVE_KIND = 'FusedInferAttentionScore'
FLASH_NATIVE_KIND = 'FlashMla'
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        repository='cann/ops-transformer',
        revision='2d3f4f6db0231ac2e9e487cf62e94f6e173eb946',
        relative_path='attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h',
        source_sha256='0ce3c62b573478790b78fe057b5a697d45042c1b7f3d7a9ee582410fe1d1c811',
    ),
    SourceRef(
        repository='cann/ops-transformer',
        revision='2d3f4f6db0231ac2e9e487cf62e94f6e173eb946',
        relative_path='attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_noquant_mla.h',
        source_sha256='e62a9d1704937b5a67b1b4e532318d81bff5bd39d76fd2cf7f17e6b5ceb8ea7d',
    ),
    SourceRef(
        repository='cann/ops-transformer',
        revision='2d3f4f6db0231ac2e9e487cf62e94f6e173eb946',
        relative_path='attention/flash_mla/op_kernel/arch35/flash_mla_block_cube.h',
        source_sha256='a6369c827ddca2d5d6a03e8cd058c0331eb9c63b416e3ebb6d83f633b5bf01da',
    ),
    SourceRef(
        repository='cann/ops-transformer',
        revision='2d3f4f6db0231ac2e9e487cf62e94f6e173eb946',
        relative_path='attention/flash_mla/op_kernel/arch35/flash_mla_block_vec.h',
        source_sha256='e57170b54c8a97e60d2500604c24a4623d7aef3356be49b55a98411d5e7e7a3f',
    ),
)
SOURCE = {
    'repository': 'cann/ops-transformer',
    'revision': '2d3f4f6db0231ac2e9e487cf62e94f6e173eb946',
    'branch': 'arch35 BF16 noquant FIA MLA',
    'relative_path': 'attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h',
    'source_sha256': '0ce3c62b573478790b78fe057b5a697d45042c1b7f3d7a9ee582410fe1d1c811',
    'cube': 'attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h',
    'vector': 'attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_noquant_mla.h',
    'sha256': {'cube': '0ce3c62b573478790b78fe057b5a697d45042c1b7f3d7a9ee582410fe1d1c811',
               'vector': 'e62a9d1704937b5a67b1b4e532318d81bff5bd39d76fd2cf7f17e6b5ceb8ea7d'},
    'scope': 'M64/S2_128/D512/rope64; declared split count; source-compatible not binary identity',
}
FLASH_SOURCE = {
    'repository': 'cann/ops-transformer',
    'revision': '2d3f4f6db0231ac2e9e487cf62e94f6e173eb946',
    'branch': 'arch35 BF16 noquant FlashMLA',
    'relative_path': 'attention/flash_mla/op_kernel/arch35/flash_mla_block_cube.h',
    'source_sha256': 'a6369c827ddca2d5d6a03e8cd058c0331eb9c63b416e3ebb6d83f633b5bf01da',
    'cube': 'attention/flash_mla/op_kernel/arch35/flash_mla_block_cube.h',
    'vector': 'attention/flash_mla/op_kernel/arch35/flash_mla_block_vec.h',
    'sha256': {'cube': 'a6369c827ddca2d5d6a03e8cd058c0331eb9c63b416e3ebb6d83f633b5bf01da',
               'vector': 'e57170b54c8a97e60d2500604c24a4623d7aef3356be49b55a98411d5e7e7a3f'},
    'scope': 'M96/S2_112/D512/rope64; fused TND-576; source-compatible not binary identity',
}
KEY_TILE = 128
HEAD_TILE = 64
FLASH_KEY_TILE = 112
FLASH_HEAD_TILE = 96

_LATENT_WIDTH = 512
_ROPE_WIDTH = 64
_QK_WIDTH = _LATENT_WIDTH + _ROPE_WIDTH
_FLASH_KV_L1_WIDTH = 608
_FLASH_UB_HALF_HEADS = FLASH_HEAD_TILE // 2
# Exact compiled UbLayout in flash_mla_block_vec.h:734-750 (252,928 B <= 248 KiB).
_FLASH_UB_MM2_SLOTS = 2
_FLASH_UB_MM2_SLOT_BYTES = 49152
_FLASH_UB_MM1_BYTES = 24576
_FLASH_UB_VEC2_BYTES = 98304
_FLASH_UB_VEC1_ROWS = _FLASH_UB_HALF_HEADS + 1
_FLASH_UB_VEC1_COLS = 128
_FLASH_UB_MASK_SLOTS = 2
_FLASH_UB_MASK_SLOT_BYTES = 6144
_FLASH_UB_SOFTMAX_ARRAYS = 3
_FLASH_UB_SOFTMAX_SLOTS = 3
_FLASH_UB_SOFTMAX_SLOT_BYTES = 256
_FLASH_UB_LSE_BYTES = 1536
_FLASH_UB_BRDCST_BYTES = 2048
_FLASH_UB_TMP_BYTES = 512
_FLASH_UB_SINK_TMP_BYTES = 512
_BF16_BYTES = 2
_CUBE_ALIGNMENT = 16
_SCORE_COLUMN_ALIGNMENT = 8
_FP32_BYTES = 4
_PAGE_INDEX_BYTES = 4
_SPLIT_STATS_BYTES = 2 * _FP32_BYTES
_SOFTMAX_OPS_PER_SCORE = 8
_TEMPLATE_BOUNDARY_CYCLES = 2
_STEADY_CYCLE_PERIOD = 2
_TEMPLATE_BOUNDARY_TOTAL = 2 * _TEMPLATE_BOUNDARY_CYCLES
_MIN_TEMPLATE_CYCLES = _TEMPLATE_BOUNDARY_TOTAL + _STEADY_CYCLE_PERIOD
_SOFTMAX_FIRST_OPS_PER_HEAD = 26
_SOFTMAX_NEXT_OPS_PER_HEAD = 32
_LATER_UPDATE_OPS_PER_ELEM = 3
_MERGE_OPS_PER_ELEM = 3
_MERGE_OPS_PER_PARTIAL = 6
_MERGE_OPS_PER_HEAD = _MERGE_OPS_PER_ELEM * _LATENT_WIDTH + _MERGE_OPS_PER_PARTIAL
_PLAN_CACHE_SIZE = 64
_CYCLE_TEMPLATE_CACHE_SIZE = 8
_PIPELINE_SLOTS = 3
_L0AB_BYTES = 65536
_L0C_BYTES = 262144
_FLASH_L1_BYTES = (
    _PIPELINE_SLOTS * FLASH_KEY_TILE * _FLASH_KV_L1_WIDTH * _BF16_BYTES
    + FLASH_HEAD_TILE * _QK_WIDTH * _BF16_BYTES
)
_FLASH_UB_BYTES = (
    _FLASH_UB_MM2_SLOTS * _FLASH_UB_MM2_SLOT_BYTES
    + _FLASH_UB_MM1_BYTES
    + _FLASH_UB_VEC2_BYTES
    + _FLASH_UB_VEC1_ROWS * _FLASH_UB_VEC1_COLS * _BF16_BYTES
    + _FLASH_UB_MASK_SLOTS * _FLASH_UB_MASK_SLOT_BYTES
    + _FLASH_UB_SOFTMAX_ARRAYS * _FLASH_UB_SOFTMAX_SLOTS * _FLASH_UB_SOFTMAX_SLOT_BYTES
    + _FLASH_UB_LSE_BYTES
    + _FLASH_UB_BRDCST_BYTES
    + _FLASH_UB_TMP_BYTES
    + _FLASH_UB_SINK_TMP_BYTES
)

# Exact no-mask/no-LSE allocations: kernel CV queues plus Vector-owned queues.
BUFFERS = {'l1': 3*128*576*2 + 64*576*2, 'l0a': _L0AB_BYTES, 'l0b': _L0AB_BYTES,
           'l0c': _L0C_BYTES, 'ub': 2*32*(128+512)*4 + 6*256 + 2*1024 + 512
           + 32*512*4 + 33*128*2}
FLASH_BUFFERS = {
    'l1': _FLASH_L1_BYTES, 'l0a': _L0AB_BYTES, 'l0b': _L0AB_BYTES,
    'l0c': _L0C_BYTES, 'ub': _FLASH_UB_BYTES,
}


def aligned(value, step=_CUBE_ALIGNMENT):
    return (value + step - 1) // step * step


def segments(contexts, requested, *, key_tile=KEY_TILE):
    """Each record is one sequence's contiguous range of actual KV tiles."""
    result = []
    for sequence, context in enumerate(contexts):
        tiles = math.ceil((context + 1) / key_tile)
        count = min(requested, tiles)
        small, extra = divmod(tiles, count)
        start = 0
        for split in range(count):
            size = small + (split < extra)
            result.append((sequence, start, size, count))
            start += size
    return tuple(result)


def tile_work(keys, heads, *, first, last, key_tile=KEY_TILE, head_tile=HEAD_TILE, is_flash=False):
    """Source memory directions and executed matrix footprint, without timers."""
    hm, kn = aligned(heads), aligned(keys)
    score = aligned(heads, 2) * aligned(keys, _SCORE_COLUMN_ALIGNMENT) * _FP32_BYTES
    pv = aligned(heads, 2) * _LATENT_WIDTH * _FP32_BYTES
    softmax_ops = _SOFTMAX_FIRST_OPS_PER_HEAD if first else _SOFTMAX_NEXT_OPS_PER_HEAD
    # FlashMLA uses head_tile as source stride and aligned realM (hm) as MTE1 transfer
    # extent (flash_mla_block_cube.h:304); FIA MatmulN loads the declared probability tile.
    pv_m = hm if is_flash else head_tile
    return (
        ('load', Work(resources={'hbm': keys*_QK_WIDTH*_BF16_BYTES, 'mte2_aic': keys*_QK_WIDTH*_BF16_BYTES})),
        ('qk_operands', Work(resources={'mte1': (hm + kn)*_QK_WIDTH*_BF16_BYTES})),
        ('qk', Work(matrix={'bf16': _BF16_BYTES*hm*kn*_QK_WIDTH})),
        ('scores', Work(resources={'fixpipe': score})),
        ('softmax', Work(resources={'vector': heads * (_SOFTMAX_OPS_PER_SCORE * keys + softmax_ops)})),
        ('probability', Work(resources={'mte3': heads*key_tile*_BF16_BYTES})),
        ('pv_operands', Work(resources={'mte1': (pv_m + _LATENT_WIDTH)*kn*_BF16_BYTES})),
        ('pv', Work(matrix={'bf16': _BF16_BYTES*hm*kn*_LATENT_WIDTH})),
        ('partial', Work(resources={'fixpipe': pv})),
        ('update', Work(resources={'vector': heads*_LATENT_WIDTH*(2 if first else _LATER_UPDATE_OPS_PER_ELEM)
                                  + (heads*_LATENT_WIDTH if last else 0)})),
    )


@dataclass(frozen=True)
class Spec:
    contexts: tuple
    heads: int
    kv_splits: int
    page_size: int
    dispatch: Dispatch | None = None
    host_policy: str = 'declared'
    native_abi: str = FIA_ABI
    key_tile: int = KEY_TILE
    head_tile: int = HEAD_TILE

    @classmethod
    def parse(cls, op, cores):
        meta = dict(op.metadata)
        raw = meta['paged_mla_geometry']
        contexts = tuple(raw['contexts'])
        native_abi = str(raw.get('native_abi') or meta.get('aclnn_symbol') or FIA_ABI)
        is_flash = native_abi == FLASH_ABI
        key_tile = FLASH_KEY_TILE if is_flash else KEY_TILE
        head_tile = FLASH_HEAD_TILE if is_flash else HEAD_TILE
        result = cls(contexts, raw['heads'], raw['kv_splits'], raw['page_size'],
                     native_abi=native_abi, key_tile=key_tile, head_tile=head_tile)
        if is_flash:
            if (not result.contexts or any(type(x) is not int or x < 0 for x in result.contexts)
                    or type(result.kv_splits) is not int or result.kv_splits < 1
                    or type(result.heads) is not int or not 1 <= result.heads <= DENSE_MAX_PHYSICAL_HEADS
                    or type(result.page_size) is not int or result.page_size < DENSE_PAGE_ALIGNMENT
                    or result.page_size % DENSE_PAGE_ALIGNMENT
                    or result.page_size > DENSE_MAX_PAGE_SIZE):
                raise ModelingError('paged_mla_flash_mla_requires_heads_le128_and_page_mult16')
            if raw.get('dispatch') is not None:
                raise ModelingError('paged_mla_flash_mla_does_not_support_fia_dispatch')
            return result
        if (not result.contexts or any(type(x) is not int or x < 0 for x in result.contexts)
                or type(result.kv_splits) is not int or result.kv_splits < 1
                or type(result.heads) is not int or not 1 <= result.heads <= HEAD_TILE
                or type(result.page_size) is not int or result.page_size < DENSE_PAGE_ALIGNMENT
                or result.page_size % DENSE_PAGE_ALIGNMENT
                or result.page_size > DENSE_MAX_PAGE_SIZE):
            raise ModelingError('paged_mla_requires_q1_heads_le64_and_page_mult16')
        dispatch = raw.get('dispatch')
        policy = paged_mla_host_v2.POLICY if dispatch == paged_mla_host_v2.POLICY else 'declared'
        if policy == paged_mla_host_v2.POLICY:
            capacity = raw.get('native_query_capacity', len(contexts))
            if capacity < len(contexts):
                raise ModelingError('paged_mla_query_capacity_below_active_sequences')
            dispatch = paged_mla_host_v2.source_dispatch(contexts, result.heads, cores, capacity)
        return cls(contexts, result.heads, result.kv_splits, result.page_size,
                   Dispatch.parse(dispatch, contexts, result.heads), policy,
                   FIA_ABI, KEY_TILE, HEAD_TILE)


def _ends(spec, segment):
    sequence, start, count, splits, heads = segment
    query = Work(resources={'hbm': heads*_QK_WIDTH*_BF16_BYTES, 'mte2_aic': heads*_QK_WIDTH*_BF16_BYTES})
    # One 32-bit physical page index per encountered page; scalar issue and
    # translation latency remain declared omissions, not a timing floor.
    first = start * spec.key_tile
    end = min((start + count) * spec.key_tile, spec.contexts[sequence] + 1)
    pages = (end + spec.page_size-1)//spec.page_size - first//spec.page_size
    query += Work(resources={'hbm': pages*_PAGE_INDEX_BYTES, 'mte2_aic': pages*_PAGE_INDEX_BYTES})
    out_bytes = heads * (_LATENT_WIDTH*_FP32_BYTES+_SPLIT_STATS_BYTES if splits > 1 else _LATENT_WIDTH*_BF16_BYTES)
    return query, Work(resources={'hbm': out_bytes, 'mte3': out_bytes})


def merge_work(spec, count, heads=None):
    heads = spec.heads if heads is None else heads
    read = count * heads * (_LATENT_WIDTH*_FP32_BYTES+_SPLIT_STATS_BYTES)
    write = heads*_LATENT_WIDTH*_BF16_BYTES
    return (Work(resources={'hbm': read, 'mte2_aiv': read}),
            Work(resources={'vector': heads * (count * _MERGE_OPS_PER_HEAD + _LATENT_WIDTH)}),
            Work(resources={'hbm': write, 'mte3': write}))


def build(op, geometry):
    spec = Spec.parse(op, geometry.compute_units)
    buffers = FLASH_BUFFERS if spec.native_abi == FLASH_ABI else BUFFERS
    for pool, needed in buffers.items():
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


@lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _plan(spec, cores):
    is_flash = spec.native_abi == FLASH_ABI
    buffers = FLASH_BUFFERS if is_flash else BUFFERS
    source = FLASH_SOURCE if is_flash else SOURCE
    native_kind = FLASH_NATIVE_KIND if is_flash else NATIVE_KIND
    head_groups = tuple(
        min(spec.head_tile, spec.heads - offset)
        for offset in range(0, spec.heads, spec.head_tile)
    )
    if spec.dispatch:
        by_owner = tuple(
            tuple((*job, spec.heads) for job in rows)
            for rows in spec.dispatch.jobs()
        )
        jobs = tuple(job for rows in by_owner for job in rows)
    else:
        base_jobs = segments(spec.contexts, spec.kv_splits, key_tile=spec.key_tile)
        jobs = tuple((*seg, gh) for seg in base_jobs for gh in head_groups)
        by_owner = tuple(jobs[owner::cores] for owner in range(min(cores, len(jobs))))
    total = Work()
    split_counts = {}
    for job in jobs:
        sequence, start, count, splits, job_heads = job
        split_counts[sequence] = splits
        begin, end = _ends(spec, job)
        total += begin + end
        boundary_tiles = (0,) if count == 1 else (0, count - 1) if count > 1 else ()
        for tile in boundary_tiles:
            keys = min(spec.key_tile, spec.contexts[sequence]+1-(start+tile)*spec.key_tile)
            for _, work in tile_work(keys, job_heads, first=tile == 0, last=tile+1 == count,
                                     key_tile=spec.key_tile, head_tile=spec.head_tile,
                                     is_flash=is_flash):
                total += work
        if count > 2:
            for _, work in tile_work(spec.key_tile, job_heads, first=False, last=False,
                                     key_tile=spec.key_tile, head_tile=spec.head_tile,
                                     is_flash=is_flash):
                total += work.scaled(count - 2)
    for count in split_counts.values():
        if count > 1:
            for work in merge_work(spec, count):
                total += work

    def lower(ctx):
        active = len(by_owner)

        def stage(name, work, owner, *, aiv=None, heads=None, **kwargs):
            eff_heads = spec.heads if heads is None else heads
            roles = set(work.resources) | ({'cube'} if work.matrix else set())
            placement = ctx.owned(owner, cores, roles-{'hbm'})
            vector_roles = roles & {'vector', 'mte2_aiv', 'mte3'}
            if aiv is not None:
                for role in vector_roles:
                    placement[ctx.resource_names[role]] = (aiv,)
                return Stage(name,ctx.bind(work),placement=placement,**kwargs)
            if vector_roles and eff_heads > 1:
                # InitTask uses actualM=q*g (q=1 here), then ceil/floor halves.
                # Each half owns its physical pipes; no idle sibling borrowing.
                children = []
                for subblock, rows in enumerate(((eff_heads+1)//2, eff_heads//2)):
                    own = dict(placement)
                    for role in vector_roles:
                        own[ctx.resource_names[role]] = (2*owner+subblock,)
                    children.append(KernelProgram((Stage('subblock',
                        ctx.bind(work.scaled(rows/eff_heads)), placement=own),)))
                return Stage(name, program=ParallelProgram(tuple(children)), **kwargs)
            if eff_heads == 1:
                for role in vector_roles:
                    key = ctx.resource_names[role]
                    placement[key] = placement[key][:1]
            return Stage(name, ctx.bind(work), placement=placement, **kwargs)

        merged = []
        def merge_program(count, owner, heads, aiv=None):
            read, _, write = merge_work(spec, count, heads)
            update = Work(resources={'vector': heads * _MERGE_OPS_PER_HEAD})
            return KernelProgram((Stage('partials', program=PipelineProgram(count, 1, lambda _: (
                    stage('read', read.scaled(1/count), owner, aiv=aiv, heads=heads),
                    stage('accumulate', update, owner, aiv=aiv, heads=heads, after=('read',))))),
                stage('normalize', Work(resources={'vector':heads*_LATENT_WIDTH}), owner, aiv=aiv, heads=heads,
                      after=('partials',)), stage('write', write, owner, aiv=aiv, heads=heads, after=('normalize',))))

        if spec.dispatch:
            for aiv,(sequence,_,heads) in enumerate(spec.dispatch.merges):
                merged.append(merge_program(split_counts[sequence],aiv//2,heads,aiv))
        else:
            for owner in range(active):
                entries = tuple(
                    (count, gh)
                    for sequence, count in split_counts.items()
                    if count > 1 and sequence % cores == owner
                    for gh in head_groups
                )
                if not entries:
                    continue
                # Native FD owner metadata is not inferred; this declared
                # owner processes each sequence serially using bounded UB.
                merged.append(PipelineProgram(len(entries), 1,
                    lambda i, entries=entries, owner=owner: (
                        Stage('sequence', program=merge_program(entries[i][0], owner, entries[i][1])),)))
        allocations = tuple(ctx.allocate(f'{pool}_{owner}', pool, size,
                            tuple(range(2*owner,2*owner+2)) if pool == 'ub' else (owner,))
                            for owner in range(active) for pool, size in buffers.items())
        producers = _producer_group(by_owner, spec, stage)
        return KernelProgram((Stage('reserve', acquire=allocations),
            Stage('producers', after=('reserve',), program=ParallelProgram(producers)),
            Stage('merge', after=('producers',), program=ParallelProgram(tuple(merged))),
            Stage('release', after=('merge',), free=allocations)))

    return OperatorPlan(total, lower, {
        'implementation': IMPLEMENTATION, 'source': source,
        'native_kind': native_kind, 'aclnn_symbol': spec.native_abi,
        'source_execution_domain': 'cube_vector', 'query_sequences': len(spec.contexts),
        'requested_kv_splits': spec.kv_splits if spec.host_policy == 'declared' else None,
        'physical_segments': len(jobs),
        'active_aic_owners': len(by_owner), 'buffers_per_owner': buffers,
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


def _producer_group(by_owner, spec, stage):
    def producer(owner, jobs):
        def job_program(index):
            job = jobs[index]
            begin, end = _ends(spec, job)
            job_heads = job[4]
            # The engine can skip steady cycles up to the next release.
            pipe = _template_pipeline(owner, _job_pipeline(owner, job, spec, stage))
            return (Stage('job', program=KernelProgram((
                stage('query', begin, owner, heads=job_heads),
                Stage('tiles', after=('query',), program=pipe),
                stage('output', end, owner, heads=job_heads, after=('tiles',)),
            ))),)

        return PipelineProgram(len(jobs), 1, job_program)

    return tuple(producer(owner, jobs) for owner, jobs in enumerate(by_owner))


def _template_pipeline(owner, pipe):
    if pipe.count <= _MIN_TEMPLATE_CYCLES:
        return pipe
    steady_start = _TEMPLATE_BOUNDARY_CYCLES
    tail_start = pipe.count - _TEMPLATE_BOUNDARY_CYCLES
    steady = tuple(pipe.tile(index) for index in range(steady_start, steady_start + _STEADY_CYCLE_PERIOD))
    return PipelineTemplateProgram(PipelineTemplate(owner, pipe.slots, (
        PipelinePhase('head0', pipe.tile(0), 1),
        PipelinePhase('head1', pipe.tile(1), 1),
        PipelinePhase('steady', steady[0], pipe.count - _TEMPLATE_BOUNDARY_TOTAL, residue_stages=steady),
        PipelinePhase('tail0', pipe.tile(tail_start), 1),
        PipelinePhase('tail1', pipe.tile(tail_start + 1), 1),
    )))


def _flat_cycle_stage(stage):
    if stage.program is None:
        yield stage
        return
    # The nested stage serialized both subblocks behind the prior join.
    previous = (*stage.previous, stage.name)
    names = []
    for subblock, program in enumerate(stage.program.programs):
        name = f'{stage.name}_sb{subblock}'
        yield replace(program.stages[0], name=name, after=stage.after, previous=previous)
        names.append(name)
    yield Stage(stage.name, after=tuple(names))


def _job_pipeline(owner, job, spec, stage):
    sequence, start, count, _, job_heads = job
    is_flash = spec.native_abi == FLASH_ABI

    @lru_cache(maxsize=_CYCLE_TEMPLATE_CACHE_SIZE)
    def cycle_template(keys, old_keys, first, old_first, old_last, parity):
        # ExecuteTask issues C1(i), C2(i-1), and independently
        # V1(i), V2(i-1). Distinct physical pipes may overlap, but
        # the next driver iteration cannot reorder their MMAs.
        result = []
        # FlashMLA allocates one score slot in UB (flash_mla_block_cube.h:361), so tile i+1
        # must wait for both AIVs to release slot 0 after tile i's softmax.
        score_slot = 0 if is_flash else parity
        if keys is not None:
            work = dict(tile_work(keys, job_heads, first=first, last=False,
                                  key_tile=spec.key_tile, head_tile=spec.head_tile,
                                  is_flash=is_flash))
            result.extend((
                stage('load', work['load'], owner, heads=job_heads, previous=('qk','pv')),
                stage('qk_operands', work['qk_operands'], owner, heads=job_heads, after=('load',)),
                stage('qk', work['qk'], owner, heads=job_heads, after=('qk_operands',)),
                stage('scores', work['scores'], owner, heads=job_heads, after=('qk',),
                      previous=(f'score_release{score_slot}',)),
                stage('softmax', work['softmax'], owner, heads=job_heads, after=('scores',),
                      previous=('update',)),
                stage('probability', work['probability'], owner, heads=job_heads, after=('softmax',)),
                Stage(f'score_release{score_slot}', after=('softmax',)),
            ))
        else:
            # Source flushes C2/V2 of the last valid task without
            # fabricating a further QK or score/probability tile.
            result.extend((Stage('qk'), Stage('softmax')))
        if old_keys is not None:
            work = dict(tile_work(old_keys, job_heads, first=old_first,
                                  last=old_last,
                                  key_tile=spec.key_tile, head_tile=spec.head_tile,
                                  is_flash=is_flash))
            result.extend((
                stage('pv_operands', work['pv_operands'], owner, heads=job_heads,
                      after=('qk',), previous=('probability',)),
                stage('pv', work['pv'], owner, heads=job_heads, after=('pv_operands',)),
                stage('partial', work['partial'], owner, heads=job_heads, after=('pv',),
                      previous=(f'partial_release{1-parity}',)),
                stage('update', work['update'], owner, heads=job_heads, after=('partial','softmax')),
                Stage(f'partial_release{1-parity}', after=('update',)),
            ))
        return tuple(flat for stage in result for flat in _flat_cycle_stage(stage))

    def cycle(index):
        # Only prologue, two ring parities and tail/flush differ.
        # Reuse immutable physical stages; every original event
        # and dependency still executes in the same order.
        end = spec.contexts[sequence]+1
        keys = min(spec.key_tile, end-(start+index)*spec.key_tile) if index < count else None
        old_keys = min(spec.key_tile, end-(start+index-1)*spec.key_tile) if index else None
        return cycle_template(keys, old_keys, index == 0, index == 1,
                              index == count, index % _STEADY_CYCLE_PERIOD)
    return PipelineProgram(count + 1, _PIPELINE_SLOTS, cycle)
