"""MXFP8 grouped gate/up with FP32 Cube-to-Vector and local SwiGLU/MX.

[Role] Execute the declared CGMCT mixed AIC/AIV gate/up source boundary.
[Invariants] Two matrix products feed separate UB tensors, never a BF16 GM
  intermediate. Both fixed AIV row partitions complete before owner reuse.
  Integer bank rows and explicit L1 K determine work; measurements are absent.
[Timeouts & Locks] O(output tiles) metadata and finite per-owner buffers.
[Side-Effect Policy] Ordinary Engine programs only; no I/O, device or locks.
"""
from collections import Counter
from dataclasses import dataclass
from math import ceil

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .mxfp8_grouped import Mapping, align, panel_work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .refill import RefillPipeline
from .refill_steady import compact_refill

PROFILE = RECIPE = 'ascend_mxfp8_grouped_swiglu_quant'
ROLE = 'grouped_gate_up_activation_quant'
SOURCE = {
    'repository': 'https://gitcode.com/cann/ops-transformer',
    'revision': 'a3925fd49cd256ad478d2faf983bf2961b8ae2c8',
    'kernel': 'gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h',
    'epilogue': 'gmm/common/cgmct/epilogue/block_epilogue_swiglu_mx_quant.h',
    'matrix_api': 'gmm/common/cgmct/block/block_quant_with_tile_mmad_multi_block.h',
    'sha256': {'kernel':'a5e7f842809cbc3d526ed68ab06ad83ffd4508ee13e48326684f03f19b9d1815',
               'epilogue':'50d67ce6942a43fa37f40c9c47f28c1df7276d96210fd39fe11fd0c4db3b14f5',
               'matrix_api':'f5acb01f3f81189c90b40e640f55ebe894a87983911f5a2661f61b70632e16d1'},
    'domain': 'CGMCT MX E4M3/NZ, AIC-to-AIV FP32, no bias, nontransposed A/B',
    'capture_binary_identity': 'not_established',
}


@dataclass(frozen=True)
class Config:
    rows: tuple
    width: int
    k: int
    l1_k: int
    compact: bool = True


def owner_tiles(rows, width, tile, cores):
    """The fused caller uses a fixed base M and rolling ASWT, no tail split."""
    owners, cursor = [[] for _ in range(cores)], 0
    for bank, rows_in_bank in enumerate(rows):
        if not rows_in_bank:
            continue
        nm, nn = ceil(rows_in_bank/tile.m), ceil(width/tile.n)
        window = min(4, nm)
        last = nm//window-1
        for wi in range(last+1):
            for ni in range(nn):
                ni = nn-1-ni if wi % 2 else ni
                for mi in range(wi*window, nm if wi == last else (wi+1)*window):
                    owners[cursor % cores].append((bank, min(tile.m, rows_in_bank-mi*tile.m),
                        min(tile.n, width-ni*tile.n), mi*tile.m, ni*tile.n))
                    cursor += 1
    return tuple(tuple(items) for items in owners)


def vector_work(rows, width):
    """Source scalar lane inventory; register ports/issue cycles are not inferred."""
    valid = rows*width
    lanes = rows*align(width,64)
    groups = lanes//32
    # SwiGLU: neg, exp, add, div, mul, BF16 cast. MX: exponent AND,
    # paired/reduced max (2*lanes-groups), 13 scale expressions/group;
    # conversion multiply/casts/pack-adds follows the unmasked 256-lane loop.
    quant_lanes = align(lanes,256)
    operations = 6*valid + 2*lanes-groups + 13*groups + 6*quant_lanes
    writes = rows*(width+2*ceil(width/64))
    return Work(resources={'vector': operations}), Work(resources={'hbm': writes, 'mte3': writes})


def matrix_work(m, n, k, tile):
    total = Work(resources={'fixpipe': 4*m*n})
    for start in range(0,k,tile.l1_k):
        load, steps = panel_work(m,n,min(tile.l1_k,k-start),tile.k)
        total += load
        for operand, compute in steps:
            total += operand+compute
    return total


def build(op, geometry):
    raw = dict(op.metadata)['mxfp8_swiglu_config']
    cfg = Config(tuple(raw['rows']), raw['width'],raw['k'],raw['l1_k'],raw.get('compact',True))
    if (not cfg.rows or any(type(x) is not int or x < 0 for x in cfg.rows)
            or not sum(cfg.rows) or cfg.width < 64 or cfg.width % 64 or cfg.k % 64
            or cfg.l1_k < 128 or cfg.l1_k % 128):
        raise ModelingError('mxfp8_swiglu_requires_integer_rows_N64_K64_declared_L1K128')
    cores = geometry.compute_units
    for role in ('mte1','mte2_aic','fixpipe','vector','mte3'):
        expected = cores*(2 if role in {'vector','mte3'} else 1)
        if geometry.resource_units.get(role) != expected:
            raise ModelingError('mxfp8_swiglu_requires_paired_AIC_AIV_pipes:'+role)
    bm,bn,bk = align(min(sum(cfg.rows),128),16), align(min(2*cfg.width,256),128),128
    tile = Mapping(bm,bn,bk,min(cfg.l1_k,align(cfg.k,bk)),2 if 2*bm*bn*4 <= geometry.buffers['l0c'] else 1)
    # Two 64x256 FP32 inputs + FP8 output/scale + BF16 SwiGLU + two
    # uint16 exponent/scale temporaries + per-row 32-byte scale layout.
    ub = 2*16384*4+16384+512+16384*2+2*512*2+(bm//2)*32
    buffers = {**tile.buffers,'ub':ub}
    for pool,size in buffers.items():
        if (size > geometry.buffers.get(pool,0)
                or geometry.buffer_units.get(pool) != cores*(2 if pool == 'ub' else 1)):
            raise ModelingError('mxfp8_swiglu_buffer_capacity_or_owners:'+pool)
    owners=owner_tiles(cfg.rows,cfg.width,tile,cores)
    # Each AIC and its two AIVs executes the bank loop and reads int64 group_list.
    scalar=Work(resources={'hbm':len(cfg.rows)*8*3})
    total=scalar.scaled(cores)
    for (m,n),count in Counter((item[1],item[2]) for jobs in owners for item in jobs).items():
        work=matrix_work(m,n,cfg.k,tile).scaled(2)
        for rows in ((m+1)//2,m//2):
            if rows:
                vector,write=vector_work(rows,n);work += vector+write
        total += work.scaled(count)

    def lower(ctx):
        active=sum(bool(items) for items in owners)
        def owner(owner_id,items):
            def stage(name,work,*,aiv=None,**kwargs):
                roles=set(work.resources)-{'hbm'}|({'cube'} if work.matrix else set())
                placement=ctx.owned(owner_id,cores,roles)
                if aiv is not None:
                    for role in roles & {'vector','mte3'}:
                        placement[ctx.resource_names[role]]=(aiv,)
                return Stage(name,ctx.bind(work),placement=placement,**kwargs)
            def matrix(m,n):
                count,tail=divmod(cfg.k,tile.l1_k)
                def parts(kd):
                    load,steps=panel_work(m,n,kd,tile.k)
                    return (stage('load',load),),tuple((stage('operand',a),stage('cube',b,after=('operand',))) for a,b in steps)
                full=(parts(tile.l1_k),) if count else ()
                last=(parts(tail),) if tail else ()
                if cfg.compact:
                    program=compact_refill(ctx,full,count,last,shared_clients={ctx.resource_names['hbm']:active})
                else:
                    def panel(i):
                        loads,steps=parts(tail if tail and i == count else tile.l1_k)
                        return loads[0],iter((pair,index+1 == len(steps)) for index,pair in enumerate(steps)),()
                    program=RefillPipeline(count+bool(tail),2,2,panel)
                return KernelProgram((Stage('K',program=program),
                    stage('fp32_to_ub',Work(resources={'fixpipe':4*m*n}),after=('K',))))
            def output(index):
                _,m,n,_,_=items[index]
                vec=[]
                for part,rows in enumerate(((m+1)//2,m//2)):
                    if rows:
                        work,write=vector_work(rows,n)
                        vec.append(KernelProgram((stage('swiglu_mx',work,aiv=2*owner_id+part),
                            stage('output',write,aiv=2*owner_id+part,after=('swiglu_mx',)))))
                return (Stage('left',program=matrix(m,n)),
                        Stage('right',after=('left',),program=matrix(m,n)),
                        Stage('epilogue',after=('right',),program=ParallelProgram(tuple(vec))))
            allocations=tuple(ctx.allocate(f'owner{owner_id}:{pool}',pool,size,
                (2*owner_id,2*owner_id+1) if pool=='ub' else (owner_id,))
                for pool,size in buffers.items()) if items else ()
            return KernelProgram((stage('metadata',scalar,acquire=allocations),
                Stage('outputs',after=('metadata',),program=PipelineProgram(len(items),1,output)),
                Stage('release',after=('outputs',),free=allocations)))
        return ParallelProgram(tuple(owner(i,items) for i,items in enumerate(owners)))
    return OperatorPlan(total,lower,{'implementation':PROFILE,'source':SOURCE,
        'source_execution_domain':'cube_vector','source_output_tiles':sum(map(len,owners)),
        'active_aic_owners':sum(bool(items) for items in owners),'buffers_per_owner':buffers,
        'matrix_base_tile':(bm,bn,bk),'l1_k':tile.l1_k,
        'intermediate':'two_FP32_UB_tensors; no_HBM_workspace',
        'tiling_basis':'source_base_M128_cap/N256/K128; explicit_L1_K',
        'limitations':['capture_binary_and_host_L1_K_not_established',
                      'matrix_API_K_refill_uses_shared_MX_NZ_source_abstraction',
                      'compact_K_interior_is_fluid_contention_approximation',
                      'cross_output_tiles_serial; PIPE_FIX_wait_broadened_to_whole_next_tile',
                      'scalar_group_list_latency_and_cross_core_flag_latency_unmodeled',
                      'vector_lane_inventory_not_register_issue_model']})
