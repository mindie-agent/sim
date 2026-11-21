"""Opt-in finite-DAG lowering of one HCPre owner on the ordinary Engine.

[Role] Preserve source input slots, UB publication, L1 release, matrix micro
  slots and terminal RMS/projection order while exposing the same finite
  per-owner graph for compact-profile selection.
[Invariants] The source chunk DAG is retained exactly: ``read`` waits the cast
  two chunks back, ``transpose`` waits the publish two chunks back, publish
  waits the previous write and panel ``p-2`` MTE1 release, RMS waits the last
  vector and write FIFO, and matrix operands/cube keep operand/cube FIFO and
  ``cube[m-2]`` L0 reuse. No measured span or nominal cap enters the program.
[Timeouts & Locks] Finite source panels; no threads, locks or I/O.
[Side-Effect Policy] Constructs programs only; no files, observations or
  device contexts.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError

from .program import PipelinePhase, PipelineTemplate, Stage
from .native_fusion_hc_program import MK_SPLIT_TILING_KEY, M_SPLIT_TILING_KEY, phase_program


def capsule(context, stages, prefix, after=()):
    """Compress a closed bound-stage branch for the retained hand control.

    Production HCPre does not use this helper: ``first_owner`` below retains
    every source slot and FIFO boundary. This control remains for the existing
    hand-computed two-chunk test only.
    """
    from .refill_steady import _duration, producer_duration
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


def _roles(work):
    return tuple(sorted(work.resources.keys()-{'hbm'}))+(('cube',) if work.matrix else ())


def _class_stage(phase, name, context, after=(), previous=()):
    return Stage(name,context.bind(phase.work),after=tuple(after),
                 previous=tuple(previous),
                 placement=context.owned(phase.owner,phase.domain,_roles(phase.work)))


def finite_template_available():
    """True only when the actual T04 ordered-template consumer is importable."""
    try:
        from .scheduling import start_pipeline_template
    except ImportError:
        return False
    return callable(start_pipeline_template)


def finite_owner_template(g, phases, context, *, terminal='projection_out', slots=2):
    """Build a finite ordered PipelineTemplate for one key1000 owner.

    Panels are grouped into adjacent pairs.  Every source dependency inside a
    pair is a local ``after``; dependencies on the preceding pair become
    ``Stage.previous`` references, so the two-panel L1/L0/input/output slots
    and FIFOs remain real cross-tile edges.  A final odd panel gets its own
    one-panel class.  Identical classes share one Stage set and a phase count.
    This is the ordered-phase mode of the existing PipelineTemplate; it does
    not claim the indexed catalog ABI.
    """
    panels={}
    for phase in phases:
        head=phase.name.split('.',1)[0]
        if head.startswith('p') and head[1:].isdigit() and '.' in phase.name:
            panels.setdefault(int(head[1:]),[]).append(phase)
    order=sorted(panels)
    if not order:
        raise ModelingError('hc_arch35_finite_template_requires_panels')
    groups=[order[index:index+2] for index in range(0,len(order),2)]
    classes=[]          # [signature, count, stages]
    previous_map={}
    last_map={}
    for group in groups:
        local_map={}
        stages=[]
        for slot,panel in enumerate(group):
            role='ab'[slot]
            for phase in panels[panel]:
                suffix=phase.name.split('.',1)[1]
                local_map[phase.name]=f'{role}.{suffix}'
        for slot,panel in enumerate(group):
            role='ab'[slot]
            for phase in panels[panel]:
                after=[];previous=[]
                for dep in phase.after:
                    if dep in local_map:
                        after.append(local_map[dep])
                    elif dep in previous_map:
                        previous.append(previous_map[dep])
                    else:
                        raise ModelingError(
                            f'hc_arch35_finite_template_unsupported_dependency:{dep}')
                stages.append(_class_stage(phase,local_map[phase.name],context,
                                           after,previous))
        signature=tuple((stage.name,dict(stage.work),stage.after,stage.previous,
                         dict(stage.placement),stage.coupled) for stage in stages)
        if classes and classes[-1][0]==signature:
            classes[-1][1]+=1
        else:
            classes.append([signature,1,stages])
        previous_map=local_map
        last_map=local_map
    terms={phase.name:phase for phase in phases
           if phase.name in (terminal,'rms_out0','rms_out1')}
    drain=[]
    for name in (terminal,'rms_out0','rms_out1'):
        phase=terms.get(name)
        if phase is None:
            raise ModelingError(f'hc_arch35_finite_template_missing_terminal:{name}')
        previous=[]
        for dep in phase.after:
            if dep not in last_map:
                raise ModelingError(
                    f'hc_arch35_finite_template_unsupported_terminal:{dep}')
            previous.append(last_map[dep])
        drain.append(_class_stage(phase,name,context,(),previous))
    phase_objects=len(classes)
    stage_objects=sum(len(stages) for _,_,stages in classes)+len(drain)
    classes=tuple(PipelinePhase(kind='hc_panel_pair',stages=tuple(stages),
                                count=count)
                  for _,count,stages in classes)
    return PipelineTemplate(
        owner=panels[order[0]][0].owner,slots=slots,phases=classes,
        drain=tuple(drain),phase_objects=phase_objects,
        stage_objects=stage_objects,static_phase_classes=phase_objects,
        dynamic_instances=len(order),tile_factory_calls=phase_objects,
    )


def first_owner(g, phases, context, *, terminal='projection_out'):
    """Lower one source owner without crossing a source slot boundary.

    The production path keeps the complete source chunk DAG: each chunk's
    ``read`` waits the cast two chunks back (input ring), each ``transpose``
    waits the publish two chunks back (UB ring), each ``publish_l1`` waits the
    previous write and panel ``p-2`` last MTE1 operands (L1 slot), and each RMS
    stage waits the last vector plus the last publish (write FIFO). The matrix
    micro DAG (weight, per-micro operands/cube, ``cube[m-2]`` L0 reuse and
    operand/cube FIFO) is likewise the source dependency graph. Repeating this
    for every legal panel uses the existing ``Phase``/``Stage`` objects and
    creates fresh events per occurrence; static ``Phase`` tuples may be shared.
    """
    return phase_program(phases, context)


def eligible(g):
    # Both host keys lower the same finite source stage1 graph. key1001 keeps
    # its local vector stage2 explicit because it owns public materialization.
    return g.tiling_key in (MK_SPLIT_TILING_KEY, M_SPLIT_TILING_KEY)
