"""Source-sized MLA Cube refills and L0 double-buffer consumers.

[Role] Compile QK N128/K288/K96 and PV N128/K256/K128 onto shared refill programs.
[Invariants] Three KV L1 slots overlap two L0 operand/compute slots. Query and
  probability enter L1 once per S2 tile, then reread through MTE1 for each N128.
  Every byte/FLOP matches the enclosing arch22 ledger, including real tails.
[Timeouts & Locks] At most eight L1 panels per call; no locks, device or I/O.
[Side-Effect Policy] Immutable programs only; nominal compaction uses shared Engine.
"""
from .mechanism import Work
from .program import KernelProgram, Stage
from .refill import RefillPipeline
from .refill_steady import compact_refill


def _aligned(value):
    return (value+15)//16*16


def matrix_panels(kind, keys, heads, page_size, *, retained_kv_rows=0):
    """Yield one GM refill and its ordered L0 operand/Cube consumers."""
    hm = _aligned(heads)
    if kind == 'qk':
        for start in range(0,keys,128):
            columns = min(128,keys-start)
            kn = _aligned(columns)
            pages = (columns+page_size-1)//page_size
            for panel in range(2):
                # The first K288 copy also packs the complete Qrope64;
                # the second copy supplies the remaining latent Q256.
                query = heads*(320 if panel == 0 else 256)*2 if start == 0 else 0
                load = query+columns*288*2+pages*4
                resources = {'hbm':load,'mte2_aic':load}
                if retained_kv_rows:
                    retained = min(columns,max(0,retained_kv_rows-start))
                    resources['l2_write'] = retained*288*2
                operand = Work(resources={'mte1':(hm+kn)*96*2})
                cube = Work(matrix={'bf16':2*hm*kn*96})
                yield Work(resources=resources), ((operand,cube),)*3
    else:
        for column in range(4):
            for start in range(0,keys,256):
                width = min(256,keys-start)
                pages = sum((min(128,width-offset)+page_size-1)//page_size
                            for offset in range(0,width,128))
                probability = heads*width*2 if column == 0 else 0
                load = probability+width*128*2+pages*4
                cached = min(width,max(0,retained_kv_rows-start))*128*2
                resources = {'hbm':load-cached,'mte2_aic':load-cached}
                if cached:
                    resources.update(mte2_aic_cached=cached,l2_read=cached)
                consumers = []
                for offset in range(0,width,128):
                    kd = _aligned(min(128,width-offset))
                    consumers.append((Work(resources={'mte1':(hm+128)*kd*2}),
                                      Work(matrix={'bf16':2*hm*128*kd})))
                yield Work(resources=resources), tuple(consumers)


def matrix_program(ctx, kind, keys, heads, page_size, owner, cores, active, *, compact=True,
                   retained_kv_rows=0):
    def stage(name,work,**kwargs):
        roles=set(work.resources)-{'hbm','l2_read','l2_write'}|({'cube'} if work.matrix else set())
        return Stage(name,ctx.bind(work),placement=ctx.owned(owner,cores,roles),**kwargs)

    panels = tuple(((stage('load',load),),tuple(
        (stage('operand',operand),stage('cube',cube,after=('operand',)))
        for operand,cube in consumers)) for load,consumers in matrix_panels(
            kind,keys,heads,page_size,retained_kv_rows=retained_kv_rows))
    if compact:
        shared = {ctx.resource_names['hbm']:active}
        if retained_kv_rows:
            shared[ctx.resource_names[ctx.physical_role('l2_read')]] = active
        program=compact_refill(ctx,(),0,panels,refill_slots=3,consumer_slots=2,
                               shared_clients=shared)
        # This source's producer is one DMA stage. Inline its single-stage
        # program so the cached IR needs no nested readiness/completion events.
        return KernelProgram((panels[0][0][0],*program.stages[1:]))

    def panel(index):
        loads,steps=panels[index]
        return loads[0],iter((pair,i+1==len(steps)) for i,pair in enumerate(steps)),()
    return RefillPipeline(len(panels),3,2,panel)
