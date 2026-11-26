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
from types import MappingProxyType, SimpleNamespace

from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.schema.datapath import SourceRef
from .mc2_instructions import phase_events, SOURCE_RANGES
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
from .mc2_fullmesh import (
    combine_layout, combine_plan, directed_transport_cells, dispatch_layout, dispatch_plan,
    resolve_ep_local_rank, resolve_group_send_enabled,
)

TARGET_BACKEND = "ascend"
_OPS_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h",
        source_sha256="3194ed49e10aab74976053840a5e771051be946d0945664fd5a3ca34a32bdb6b",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h",
        source_sha256="4e2a3f571df2e0e3718c8e61145638483be15445383ed77c3873381874ffb5a9",
    ),
)

RECIPE = "arch35_fullmesh_resource_phase"
_PHASE_TEMPLATE_CACHE_ENTRIES = 128
#: The native plan frames rows over at least two AIV owners (``max(2, units)``).
MIN_PROTOCOL_AIV_OWNERS = 2


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
    if "global_bs" in raw:
        args["global_bs"] = raw["global_bs"]
    if "enable_group_send" in raw:
        args["enable_group_send"] = bool(raw["enable_group_send"])
    if kind == "dispatch":
        if "is_scaling_down" in raw:
            args["is_scaling_down"] = bool(raw["is_scaling_down"])
        if "shared_expert_ranks" in raw:
            args["shared_expert_ranks"] = int(raw["shared_expert_ranks"])
        if "has_smooth_scale" in raw:
            args["has_smooth_scale"] = bool(raw["has_smooth_scale"])
    return (dispatch_plan if kind == "dispatch" else combine_plan)(layout, **args)


def build(op, geometry):
    # Source geometry and declared owner roles determine this IR. Layer/rank
    # identities, rates and Engine callbacks are bound only by program(context).
    meta = dict(op.metadata)
    raw = meta["mc2_phase"]
    if "phase" not in raw:
        # The ordinary public graph binds one whole native dispatch/combine
        # call, not a dozen separately scheduled semantic phase operators.
        return _native_plan(raw, geometry, op_metadata=meta)
    return _template(tuple(sorted(raw.items())), geometry.compute_units,
                     tuple(sorted(geometry.resource_units.items())))


def declared_floor_work(raw, *, ep_rank: int | None = None):
    """Owner-count floor of one native dispatch/combine call for its ``OpSpec``.

    Per-owner flag, frame and count descriptors only grow with the AIV owner
    count, so the protocol's minimum owner set yields the compulsory payload,
    vector and link work the executed program can never undercut.
    """
    floor = SimpleNamespace(resource_units={"vector": MIN_PROTOCOL_AIV_OWNERS},
                            compute_units=MIN_PROTOCOL_AIV_OWNERS,
                            rank=ep_rank)
    resources = dict(_native_plan(raw, floor).work.resources)
    reads = sum(float(resources.get(role, 0.0)) for role in ("mte2", "mte2_aiv", "mte2_aic"))
    declared = {"hbm": resources.get("hbm", 0.0), "mte2": reads, "mte3": resources.get("mte3", 0.0),
                "vector": resources.get("vector", 0.0), "link": resources.get("link", 0.0)}
    return Work(resources={role: float(amount) for role, amount in declared.items() if amount})


def _native_plan(raw, geometry, *, op_metadata=None):
    """One real native call over the existing phase contract.

    The phase ledger remains attributable metadata; it is not scheduled a
    second time as separate Work operators.  The common Execution launches the
    returned peer program on its own Engine.
    """
    units = max(MIN_PROTOCOL_AIV_OWNERS, int(geometry.resource_units.get(
        "vector", geometry.compute_units)))
    ep = max(1, int(raw["ep"]))
    effective_raw = dict(raw)
    if "enable_group_send" not in effective_raw:
        if resolve_group_send_enabled(effective_raw, ep=ep, geometry=geometry):
            effective_raw["enable_group_send"] = True
    plan = contract(effective_raw, units)
    resources = {}
    for phase in plan.phases:
        for role, amount in phase.work:
            resources[role] = resources.get(role, 0.0) + float(amount)
    rank = resolve_ep_local_rank(effective_raw, ep=ep, op_metadata=op_metadata, geometry=geometry)
    explicit_ep_rank = (
        op_metadata.get("ep_rank")
        if (op_metadata is not None and isinstance(op_metadata.get("ep_rank"), int)
            and not isinstance(op_metadata.get("ep_rank"), bool))
        else None
    )
    geo_physical_rank = (
        getattr(geometry, "rank", None)
        if (isinstance(getattr(geometry, "rank", None), int)
            and not isinstance(getattr(geometry, "rank", None), bool))
        else None
    )
    group = tuple(range(ep))
    sources = (rank,) if 0 <= rank < ep else ()
    uniform_bytes = (
        float(effective_raw["uniform_payload_bytes"])
        if "uniform_payload_bytes" in effective_raw
        else float(effective_raw["source_assignments"]) * float(plan.layout.packet_bytes)
    )
    payload_cells = directed_transport_cells(
        group=group,
        sources=sources,
        packet_bytes=plan.layout.packet_bytes,
        route_matrix=effective_raw.get("route_matrix"),
        route_reverse=bool(effective_raw.get("route_reverse", False)),
        route_matrix_bytes=effective_raw.get("route_matrix_bytes"),
        uniform_payload_bytes=uniform_bytes,
        count=False,
    )
    count_cells = (
        directed_transport_cells(
            group=group,
            sources=sources,
            experts=int(effective_raw.get("experts", 0)),
            count=True,
        )
        if effective_raw["kind"] == "dispatch"
        else ()
    )
    link = float(
        sum(useful for _si, _di, _src, _dst, useful, _frames in payload_cells)
        + sum(useful for _si, _di, _src, _dst, useful, _frames in count_cells)
    )
    if link:
        resources["link"] = resources.get("link", 0.0) + link
    kind = str(effective_raw["kind"])

    def program(context):
        from .mc2_program import native_program_from_contract
        return native_program_from_contract(
            effective_raw, context, rank=explicit_ep_rank, physical_rank=geo_physical_rank,
        )

    return OperatorPlan(
        Work(resources=resources), program, {
            "execution_class": "vector",
            "mc2_kind": kind,
            "mc2_phases": tuple(phase.name for phase in plan.phases),
            "mc2_phase_contract": "source_ordered; one native launch",
            "coverage": "complete_native_call_from_declared_source_contract",
        }, native=True)


@lru_cache(maxsize=_PHASE_TEMPLATE_CACHE_ENTRIES)
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
                source = services[owner].get(when + ':' + name)
                if source is None or not source.resources:
                    return after
                demand = source.scaled(share)
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

                def token_stages(rows):
                    fraction = rows / raw["source_tokens"]
                    token_chain, previous = [], ()
                    for name, values in steps[1:]:
                        demand = Work(resources=dict(values)).scaled(fraction)
                        share = rows / owned_rows
                        previous = append_primitive(token_chain, name, 'before', previous,
                                                    share=share, owner=owner, prefix='')
                        token_chain.append(Stage(name, context.bind(demand), after=previous,
                            placement=context.owned(owner, units, demand.resources)))
                        previous = append_primitive(token_chain, name, 'after', (name,),
                                                    share=share, owner=owner, prefix='')
                    return tuple(token_chain)

                # Full rows repeat the same source instruction DAG. Bind its
                # immutable stage template once per owner, with one additional
                # fractional expected-row tail; PipelineProgram still issues
                # every token and retains the original one-slot dependencies.
                full_count = math.floor(owned_rows)
                tail_rows = owned_rows - full_count
                full = token_stages(1.0) if full_count else ()
                tail = token_stages(tail_rows) if tail_rows else ()
                stages.append(Stage(f"owner{owner}:tokens", after=(step_id,),
                    program=PipelineProgram(math.ceil(owned_rows), 1,
                        lambda token, full=full, tail=tail, count=full_count:
                        full if token < count else tail,
                        interior_fold=False)))
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
