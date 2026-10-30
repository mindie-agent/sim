"""Compact resource phases for the explicitly selected MC2 fullmesh protocol.

[Role] Bind the source row ledger to physical AIV owners on the common Engine.
[Invariants] A phase is not a kernel launch. Frame whole rows before scaling
  expected assignments. Eligible idle AIVs cannot supply a payload phase.
  Count and payload branches are ordered by the caller DAG, not this planner.
[Timeouts & Locks] At most 128 immutable O(AIV count) physical templates;
  no locks, file reads or device contexts.
[Side-Effect Policy] Pure plans and virtual Engine tasks only.
"""
from dataclasses import asdict
from functools import lru_cache
import math
from types import MappingProxyType

from .mechanism import OperatorPlan, Work
from .mc2_instructions import phase_events, SOURCE_RANGES
from .program import KernelProgram, PipelineProgram, Stage
from .operators.mc2_fullmesh import dispatch_layout, combine_layout, dispatch_plan, combine_plan

RECIPE = "arch35_fullmesh_resource_phase"


class _FrozenDict(dict):
    """Shared diagnostics remain JSON mappings without mutable cache aliases."""

    def _readonly(self, *args, **kwargs):
        raise TypeError("mc2_physical_template_is_immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _frozen_work(work):
    return Work(MappingProxyType(dict(work.matrix)), MappingProxyType(dict(work.resources)))


def contract(raw, aiv_count):
    kind = raw["kind"]
    layout = (dispatch_layout(raw["width"], input_dtype=raw["dtype"],
                              payload_dtype=raw.get("payload_dtype", raw["dtype"]),
                              quant_mode=raw.get("dispatch_quant", "none"))
              if kind == "dispatch" else combine_layout(raw["width"], dtype=raw["dtype"]))
    args = {key: raw[key] for key in
            ("source_assignments", "received_assignments", "experts", "ep")}
    args["aiv_count"] = aiv_count
    args.update(source_tokens=raw["source_tokens"], top_k=raw["top_k"])
    return (dispatch_plan if kind == "dispatch" else combine_plan)(layout, **args)


def build(op, geometry):
    # Source geometry and declared owner roles determine this IR. Layer/rank
    # identities, rates and Engine callbacks are bound only by program(context).
    raw = dict(op.metadata)["mc2_phase"]
    return _template(tuple(sorted(raw.items())), geometry.compute_units,
                     tuple(sorted(geometry.resource_units.items())))


@lru_cache(maxsize=128)
def _template(raw_items, compute_units, resource_items):
    raw = _FrozenDict(raw_items)
    resource_units = dict(resource_items)
    units = resource_units.get("vector", compute_units)
    plan = contract(raw, units)
    phase = next(p for p in plan.phases if p.name == raw["phase"])
    work = Work(resources=dict(phase.work))
    domain = phase.owner_domain
    prefix = dict(phase.details).get("assignment_prefix_scan", {})
    prefix_by_owner = dict(prefix.get("owner_vector_work") or ())
    if phase.owner_basis == "domain":
        weights = (1 / len(domain),) * len(domain)
    else:
        rows = raw["source_tokens"] if phase.owner_basis == "output_tokens" else (
            raw["source_assignments"] if phase.name in {
            "send_read_quantize", "frame_payload", "return_read_frame", "payload_exchange"
        } else raw["received_assignments"])
        # A fractional expected row is distributed only after each whole row's
        # frame was established. This is an explicit expected row-grid closure,
        # not an inferred destination permutation or native owner schedule.
        active = min(len(domain), math.ceil(rows))
        domain = domain[:active]
        if active:
            base, tail = divmod(rows, active)
            counts = [base + min(1.0, max(0.0, tail - i)) for i in range(active)]
            weights = tuple(value / rows for value in counts)
        else:
            weights = ()
        if not active and any(work.resources.values()):
            # An idle sender can still receive network-window HBM stores. They
            # have no local MTE3 owner; retain memory work on the shared service.
            domain, weights = (0,), (1.0,)

    inventory = {owner: phase_events(raw, phase, owner, weight, plan.layout)
                 for owner, weight in zip(domain, weights)}
    services = MappingProxyType({owner: MappingProxyType({
        key: _frozen_work(item.partition(resource_units)[0])
        for key, item in items.items()}) for owner, items in inventory.items()})
    for items in services.values():
        for demand in items.values():
            work += demand

    def program(context):
        stages = []
        for owner, weight in zip(domain, weights):
            previous = ()
            steps = phase.steps or (("service", phase.work),)

            def append_primitive(target, name, when, after, *, share=1.0, owner=owner, prefix=None):
                demand = services[owner].get(when + ':' + name, Work()).scaled(share)
                if not demand.resources:
                    return after
                step_id = (f"owner{owner}:" if prefix is None else prefix) + name + ':' + when + ':primitive'
                target.append(Stage(step_id, context.bind(demand), after=after,
                    placement=context.owned(owner, units, demand.resources)))
                return (step_id,)
            if phase.name == "weighted_reduce" and raw["source_tokens"] > 0:
                # ExpertScaleCopy loads this owner's weights once. Each output
                # token then executes its own dependent top-k slot sequence.
                name, values = steps[0]
                demand = Work(resources=dict(values)).scaled(weight)
                step_id = f"owner{owner}:{name}"
                stages.append(Stage(step_id, context.bind(demand),
                    placement=context.owned(owner, units, demand.resources)))
                owned_rows = raw["source_tokens"] * weight

                def token_stages(token, *, owner=owner, owned_rows=owned_rows):
                    fraction = min(1.0, owned_rows - token) / raw["source_tokens"]
                    token_chain, previous = [], ()
                    for name, values in steps[1:]:
                        demand = Work(resources=dict(values)).scaled(fraction)
                        share = min(1.0, owned_rows-token) / owned_rows
                        previous = append_primitive(token_chain, name, 'before', previous,
                                                    share=share, owner=owner, prefix='')
                        token_chain.append(Stage(name, context.bind(demand), after=previous,
                            placement=context.owned(owner, units, demand.resources)))
                        previous = append_primitive(token_chain, name, 'after', (name,),
                                                    share=share, owner=owner, prefix='')
                    return tuple(token_chain)

                stages.append(Stage(f"owner{owner}:tokens", after=(step_id,),
                    program=PipelineProgram(math.ceil(owned_rows), 1, token_stages)))
            else:
                for name, values in steps:
                    # Sender prefix lengths depend on the assignment's global
                    # index, not merely this owner's number of payload rows.
                    demand = (Work(resources={"vector": prefix_by_owner.get(owner, 0)})
                              if name == "assignment_prefix_scan" else
                              Work(resources=dict(values)).scaled(weight))
                    step_id = f"owner{owner}:{name}"
                    previous = append_primitive(stages, name, 'before', previous)
                    stages.append(Stage(step_id, context.bind(demand), after=previous,
                        placement=context.owned(owner, units, demand.resources)))
                    previous = append_primitive(stages, name, 'after', (step_id,))
        return KernelProgram(tuple(stages))

    return OperatorPlan(
        _frozen_work(work),
        program, _freeze({
        "execution_class": "vector", "mc2_phase": phase.name,
        "primitive_services": {
            "source_ranges": SOURCE_RANGES[raw['kind']],
            "owners": {owner: {key: item.to_dict(resource_units)
                                for key, item in items.items()}
                       for owner, items in inventory.items()},
            "coverage": "selected_source_operations; not_complete_compiled_ISA",
            "excluded": ("unobserved failed polls and arrival wait latency",
                         "SyncFunc/queue event issue and unlisted scalar address/control operations"),
            "barrier_placement": "conservative_after_containing_whole_vector_step",
        },
        "row_layout": asdict(plan.layout), "active_owner_domain": domain,
        "owner_approximation": "count_expert_intervals; payload_expected_round_robin_row_grid",
        "count_basis": plan.count_basis,
        "unmodeled_services": plan.unmodeled_features,
        "owner_basis": phase.owner_basis,
        "source_service_details": dict(phase.details),
        "assignment_prefix_owner_vector_work": prefix_by_owner,
        "phase_overlap": ("serial_topk_slots; native_bufferNum_unidentified"
                          if phase.name == "weighted_reduce" else
                          "source_ordered_whole_phase_services; native_token_window_not_reconstructed"
                          if phase.steps else "whole_phase_envelope; native_token_window_not_reconstructed"),
    }))
