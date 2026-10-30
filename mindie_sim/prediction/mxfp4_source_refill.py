"""Source-selected overlapping MXA8W4 producer compilation.

[Role] Bind source dependency rings to existing typed resources and compact IR.
[Invariants] Reuses panel_work/panel_parts; no new bytes, rates or duration fit.
  Only explicit dynamic_buffer4 selects this plan. Compact interior contention
  is fluid, while finite nominal dependencies and first/final phases remain.
[Timeouts & Locks] Bounded periodic descriptors and source ring state; no I/O.
[Side-Effect Policy] Pure source IR plus existing bounded nominal Engine control.
"""
from dataclasses import replace

from .mechanism import Work
from .mxfp4_geometry import panel_sizes, panel_work, Config
from .mxfp4_pipeline import Panel, Dependencies, pipeline_cycle
from .program import KernelProgram, Stage
from .refill_steady import _duration, producer_duration


def source_panel(context, *, k, m, n, repeat, index, owner, cores, aivs, directional,
                 ownership=None):
    from .mxfp4_panels import panel_parts
    loads, micros = panel_parts(context, k=k, m=m, n=n, repeat=repeat, index=index,
                               owner=owner, cores=cores, aivs=aivs,
                               directional=directional, ownership=ownership)
    kb, ka, _ = panel_sizes(m, n)
    offset = index * kb
    a = m * min(ka, k-offset) if offset % ka == 0 else 0
    scale = m * min(4096, k-offset) // 32 if offset % 4096 == 0 else 0
    _, vectors, _ = panel_work(Config(k,n,(m,)),m,n,offset)
    def transfer(name, amount):
        return Stage(name, context.bind(Work(resources={'hbm': amount*repeat,
                     'mte2_aic': amount*repeat})), placement=loads[0].placement) if amount else None
    return Panel(transfer('activation',a), transfer('activation_scale',scale),
                 (loads[1:7],loads[7:13]), tuple(v['scale'] > 0 for v in vectors), micros)


def producer_program(panel, slots):
    # A one-panel producer has no resident earlier L1/L0 content. Zero-work
    # consumers expose its completion without borrowing real operand work.
    first = replace(panel, micros=((Stage('operand'),Stage('cube')),))
    state, stages = Dependencies('input', slots), []
    def emit(name, stage, deps):
        stages.append(replace(stage, name=name,
                              after=tuple(dict.fromkeys(d for d in deps if d != 'input'))))
        return name
    state.advance(first, emit)
    return KernelProgram(tuple(stages))


def compact_source(context, pattern, repeats, tail, *, slots, shared_clients):
    active = pattern if repeats else ()
    first, last = (active or tail)[0], (tail or active)[-1]
    prologue = producer_program(first, slots)
    final = last.micros[-1][1]
    nominal = replace(context, rates={key:value/shared_clients.get(key,1)
                                     for key,value in context.rates.items()})
    durations = lambda p: p.mapped(lambda stage:_duration(nominal,stage))
    cycle = pipeline_cycle(tuple(map(durations,active)),repeats,
                           tail=tuple(map(durations,tail)),slots=slots)
    work, owners = {}, {}
    for count, panels in ((repeats,active),(1,tail)):
        for panel in panels:
            stages = (*(() if panel.activation is None else (panel.activation,)),
                      *(() if panel.activation_scale is None else (panel.activation_scale,)),
                      *(s for vector in panel.vectors for s in vector),
                      *(s for pair in panel.micros for s in pair))
            for stage in stages:
                for key,value in stage.work.items():
                    work[key]=work.get(key,0)+count*value
                for key,units in stage.placement.items():
                    owners.setdefault(key,set()).update(units)
    for stage in (*prologue.stages, final):
        for key,value in stage.work.items():
            work[key]-=value
    work={key:value for key,value in work.items() if value>0}
    placement={key:tuple(sorted(units)) for key,units in owners.items() if key in work}
    interior=Stage('steady',work,placement=placement,coupled=True)
    span=max(_duration(context,interior),cycle.span_ms-
             producer_duration(nominal,prologue.stages)-_duration(context,final))
    return KernelProgram((Stage('load',program=prologue),
        Stage('steady',work,after=('load',),placement=placement,coupled=True,
              caps={key:value/span for key,value in work.items()} if span else {}),
        replace(final,name='cube',after=('steady',))))
