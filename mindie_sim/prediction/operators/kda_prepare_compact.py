"""Bounded owner runs for the selected KDA Prepare phase program.

[Role] Compress repeated source phases while retaining owner-local nominal time.
[Invariants] The shared Engine computes finite isolated nominal schedules.
  Cached values depend only on geometry, physical rates and ownership.
  Main execution retains physical work, per-unit shares and the global gate
  barrier. External contention is coupled fluid progress, not discrete replay.
[Timeouts & Locks] Bounded 64-entry scalar cache; nominal execution is lazy.
[Side-Effect Policy] No observations, file I/O, hardware or persistent state.
"""
from collections import OrderedDict
from dataclasses import replace
import math

from ..engine import Engine, Resource
from ..program import KernelProgram, ParallelProgram, Stage
from .mixed_core import placement

_NOMINAL = OrderedDict()


def _signature(context):
    return (context.core_count, *(tuple(sorted(value.items())) for value in (
        context.resource_names, context.rates, context.resource_units,
        context.compute_peaks, context.service_modes)))


def _ownership(context, core, lanes):
    groups = [placement(context, core, lane) for lane in lanes]
    resources = set().union(*(group.keys() for group in groups))
    result = []
    for resource in sorted(resources):
        union = sorted(set().union(*(set(group.get(resource, ())) for group in groups)))
        result.append((resource, tuple(tuple(union.index(unit) for unit in
                           group.get(resource, ())) for group in groups)))
    return tuple(result)


def _nominal(key, program, context):
    if key not in _NOMINAL:
        units = {name: context.resource_units[role]
                 for role, name in context.resource_names.items()}
        engine = Engine(tuple(Resource(name, rate, units=units[name])
                              for name, rate in context.rates.items()
                              if name in units))
        ready = engine.submit("ready")
        end = program.start(engine, "nominal", ready)
        engine.advance(until=(end,))
        _NOMINAL[key] = engine.now
        if len(_NOMINAL) > 64:
            _NOMINAL.popitem(last=False)
    _NOMINAL.move_to_end(key)
    return _NOMINAL[key]


def _ledger(program):
    """Enumerate one finite template, retaining each physical unit's share."""
    work, shares, unplaced = {}, {}, set()

    def collect(stages):
        for stage in stages:
            for resource, amount in stage.work.items():
                work[resource] = work.get(resource, 0) + amount
                units = stage.placement.get(resource, ())
                if not units:
                    unplaced.add(resource)
                else:
                    for unit in units:
                        key = resource, unit
                        shares[key] = shares.get(key, 0) + amount / len(units)
            if stage.program is not None:
                child = stage.program
                if isinstance(child, KernelProgram):
                    collect(child.stages)
                else:
                    raise ValueError("kda_prepare_compact_template_must_be_finite")

    collect(program.stages)
    owned = {}
    for resource in work:
        values = [(unit, amount) for (name, unit), amount in shares.items()
                  if name == resource and amount]
        if not values:
            continue
        if resource in unplaced or not all(
                math.isclose(amount, values[0][1], rel_tol=1e-12)
                for _, amount in values):
            raise ValueError("kda_prepare_compact_requires_uniform_owner_partition")
        owned[resource] = tuple(sorted(unit for unit, _ in values))
    return work, owned


def _run(name, work, owned, elapsed):
    return KernelProgram((Stage(
        name, work, placement=owned, coupled=True,
        caps={key: amount / elapsed for key, amount in work.items() if amount}
        if elapsed else {}),))


def program(params, context, cores):
    from . import kda_prepare as source

    signature = _signature(context)
    gates = []
    for aiv in range(min(params.gate_tasks, 2 * cores)):
        lengths = tuple(params.lengths[task // params.heads]
                        for task in range(aiv, params.gate_tasks, 2 * cores))
        expanded = source._gate_pipeline(aiv, params, context, 2 * cores)
        key = ("gate", lengths, signature,
               _ownership(context, aiv // 2, (aiv % 2,)))
        nominal = _nominal(key, expanded, context)
        # Reuse only arithmetic; a one-head Params is an owner ledger, not
        # a supported public odd-head source invocation.
        ledger_params = replace(params, lengths=lengths, heads=1, key_heads=1)
        work = source.gate_work(ledger_params)
        if "mte2_aiv" in context.resource_names:
            resources = dict(work.resources)
            resources["mte2_aiv"] = resources.pop("mte2")
            work = source.Work(work.matrix, resources)
        gates.append(_run("owner", context.bind(work),
                          source._aiv(context, aiv), nominal))

    prepares = []
    for core, count in source.task_ownership(params, cores):
        expanded = source._paired_tile(core, params, context)
        template = KernelProgram(tuple(expanded.tile(0)))
        key = ("paired", signature, _ownership(context, core, (None, 0, 1)))
        nominal = _nominal(key, template, context)
        work, owned = _ledger(template)
        prepares.append(_run("owner", {name: amount * count for name, amount in work.items()},
                             owned, nominal * count))

    if context.record_mapping:
        context.record_mapping({
            "program": "kda_prepare_owner_runs",
            "nominal": "same_Engine_finite_owner_gate_and_single_paired_task",
            "contention": "coupled_fluid_owner_progress",
            "gate_owners": len(gates), "prepare_owners": len(prepares),
            "paired_tasks": params.paired_tasks,
            "gate_barrier": "global_gate_completion_before_any_prepare",
            "expanded_control": "compact=False source parameter",
            "unmodeled": ["native_deferred_overlap", "scalar_issue_and_flag_latency",
                          "external_cache_residency"],
        })
    return KernelProgram((
        Stage("gate", program=ParallelProgram(tuple(gates))),
        Stage("syncall", after=("gate",)),
        Stage("prepare", after=("syncall",), program=ParallelProgram(tuple(prepares))),
    ))
