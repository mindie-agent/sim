"""Opt-in finite-DAG compression of one HCPre L1 panel per Cube owner.

[Role] Retain source input, RMS and projection publication while compressing
  private finite AIV and MTE1/Cube pipelines on the ordinary Engine.
[Invariants] Source work/owners and fixed buffers are unchanged. A source DAG
  supplies a nominal period, not a measured span. Coupled interiors approximate
  external contention; each prologue and terminal publication remain explicit.
[Timeouts & Locks] Finite source DAGs; bounded shared nominal cache, no locks.
[Side-Effect Policy] Constructs programs and isolated nominal controls only.
"""
from dataclasses import replace

from .program import KernelProgram, Stage
from .refill_steady import _duration, producer_duration
from .hc_arch35_program import phase_program


def capsule(context, stages, prefix, after=()):
    """Compress a closed single-entry/single-exit source branch.

    In this source domain the first read dominates every interior task and
    the last store/Cube consumes them all. The original finite-slot DAG sets
    the isolated service span; its resource demand still runs on the caller's
    shared Engine. This is not an arbitrary-contention equivalence claim.
    """
    if len(stages) <= 2:
        names={stage.name:f'{prefix}.{stage.name}' for stage in stages}
        return tuple(replace(stage,name=names[stage.name],
            after=tuple(names[key] for key in stage.after) or tuple(after))
            for stage in stages)
    first,last=stages[0],stages[-1]
    work,owners={},{}
    for stage in stages[1:-1]:
        for key,amount in stage.work.items():
            work[key]=work.get(key,0)+amount
        for key,units in stage.placement.items():
            owners.setdefault(key,set()).update(units)
    work={key:amount for key,amount in work.items() if amount}
    placement={key:tuple(sorted(units)) for key,units in owners.items() if key in work}
    interior=Stage('interior',work,placement=placement,coupled=True)
    span=max(_duration(context,interior),producer_duration(context,stages)
        -_duration(context,first)-_duration(context,last))
    return (
        replace(first,name=prefix+'.prologue',after=tuple(after)),
        replace(interior,name=prefix+'.interior',after=(prefix+'.prologue',),
            caps={key:amount/span for key,amount in work.items()} if span else {}),
        replace(last,name=prefix+'.epilogue',after=(prefix+'.interior',)),
    )


def _branch(phases, context):
    names={phase.name for phase in phases}
    local=tuple(replace(phase,after=tuple(key for key in phase.after if key in names))
                for phase in phases)
    return phase_program(local,context).stages


def first_owner(g, phases, context):
    """Source one-panel branches publish RMS before the Cube projection ends."""
    result=[]
    weight=next(phase for phase in phases if phase.name=='p0.weight')
    result.extend(_branch((weight,),context))
    vector_ends=[]
    for sub in (0,1):
        branch=tuple(phase for phase in phases if phase.name.startswith(f'p0.v{sub}.'))
        compact=capsule(context,_branch(branch,context),f'aiv{sub}')
        result.extend(compact)
        vector_ends.append(compact[-1].name)
    matrix=tuple(phase for phase in phases if phase.name.startswith('p0.k'))
    compact=capsule(context,_branch(matrix,context),'matrix',('p0.weight',*vector_ends))
    result.extend(compact)
    projection=next(phase for phase in phases if phase.name=='projection_out')
    result.extend(replace(stage,after=(compact[-1].name,))
                  for stage in _branch((projection,),context))
    for sub in (0,1):
        rms=next(phase for phase in phases if phase.name==f'rms_out{sub}')
        result.extend(replace(stage,after=(vector_ends[sub],))
                      for stage in _branch((rms,),context))
    return KernelProgram(tuple(result))


def eligible(g):
    # Multiple L1 panels expose intermediate MTE1 releases and output-ring
    # reuse across panels. Keep their original DAG, with visible diagnostics.
    return g.k_span <= g.k_l1
