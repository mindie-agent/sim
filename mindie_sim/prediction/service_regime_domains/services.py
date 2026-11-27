"""Work-dependent primitive service on the common physical resource engine.

[Role] Interpolate a declared primitive response and constrain its service supply.
[Invariants] Categorical identity is fixed; only work is interpolated. Native
  work and shared capacity are unchanged. Startup is charged once per invocation.
[Timeouts & Locks] Bounded tables, binary search, no locks or I/O.
[Side-Effect Policy] Builds programs and records bounded calibration diagnostics.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.collectives.topology import Topology, _declared_peer_traffic
from mindie_sim.prediction.program.stages import KernelProgram, Stage


_COMMUNICATION_FAMILIES = frozenset({
    "collective", "alltoall", "alltoallv", "allreduce", "allgather",
    "all_to_all", "all_to_allv", "all_reduce", "all_gather",
    "reduce_scatter", "reducescatter", "broadcast", "send", "recv",
    "send_recv", "p2p", "p2p_send_recv", "pp_p2p",
    "mc2_dispatch", "mc2_combine"})
_COMMUNICATION_KEYS = (
    "collective", "a5_collective", "comm_dtype", "group_axis", "steps",
    "world_size", "comm_group_size", "group_size", "ranks", "group_ranks",
    "placement_scope", "p2p_pair", "expansion_mode", "expansion_regime",
    "algorithm", "protocol", "stack", "cann", "transport_rail_id")
_TOPOLOGY_GROUP_AXES = frozenset({
    "tp", "ep", "dp", "cp", "dcp", "pcp", "dcp_tp", "dcp_pcp", "dcp_dp",
    "dcp_a2a", "etp", "local_tp",
})
_PAIRWISE_FAMILIES = frozenset({
    "alltoall", "alltoallv", "all_to_all", "all_to_allv",
    "mc2_dispatch", "mc2_combine",
})
_PAIRWISE_RECIPES = frozenset({"expert_dispatch", "expert_combine"})
_FULL_PAIRWISE_ROUTE_LIMIT = 8
_ROUTE_SHARE_DIGITS = 12


def _parse_rank_pair(value) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        src, dst = value.get("source"), value.get("destination")
        if src is not None and dst is not None:
            return int(src), int(dst)
    text = str(value or "").strip()
    if "->" in text:
        left, right = text.split("->", 1)
        if left.strip().isdigit() and right.strip().isdigit():
            return int(left.strip()), int(right.strip())
    return None


def _route_pairs_for_group(group: tuple[int, ...], topo: Topology, *,
                           directed: bool, pairwise: bool) -> list[tuple[int, int]]:
    if len(group) < 2:
        return []
    if len(group) == 2:
        return [(group[0], group[1])] if directed else [(group[0], group[1]), (group[1], group[0])]
    if pairwise and len(group) <= _FULL_PAIRWISE_ROUTE_LIMIT:
        return [(src, dst) for src in group for dst in group if src != dst]
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for idx, src in enumerate(group):
        dst = group[(idx + 1) % len(group)]
        if src != dst and (src, dst) not in seen:
            seen.add((src, dst))
            pairs.append((src, dst))
    representatives: dict[tuple[int, int], int] = {}
    for rank in group:
        loc = topo.location(rank)
        representatives.setdefault((loc["node"], loc["package"]), rank)
    rep_ranks = list(representatives.values())
    for src in rep_ranks:
        for dst in rep_ranks:
            if src != dst and (src, dst) not in seen:
                seen.add((src, dst))
                pairs.append((src, dst))
    return pairs


def _explicit_group_ranks(metadata: Mapping, topo: Topology) -> tuple[int, ...] | None:
    for key in ("group_ranks", "ranks"):
        raw = metadata.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            if all(isinstance(r, int) and not isinstance(r, bool) and r in topo._rank_node for r in raw):
                return tuple(int(r) for r in raw)
    return None


def _peer_traffic_pairs(group: tuple[int, ...], metadata: Mapping) -> list[tuple[int, int, float]] | None:
    if "peer_traffic" not in metadata or len(group) < 2:
        return None
    peer = _declared_peer_traffic(metadata, len(group))
    active = [
        (int(group[i]), int(group[j]), float(payload))
        for i in range(len(group))
        for j in range(len(group))
        if i != j and (payload := peer.payload(i, j)) > 0
    ]
    total = sum(weight for _, _, weight in active)
    if total <= 0.0:
        return []
    return [(src, dst, round(weight / total, _ROUTE_SHARE_DIGITS)) for src, dst, weight in active]


def _resolved_transport_route(case, op, metadata: Mapping, device, *,
                              device_explicit: bool = False) -> dict | None:
    dies = tuple(getattr(case.machine, "dies", ()) or ())
    if not dies:
        return None
    parallel = case.runtime.parallel
    stage = int(metadata.get("source_stage", getattr(op, "stage", 0) or 0))
    width = max(1, int(parallel.tp) * int(parallel.pcp))
    stage_span = max(1, int(parallel.pp) * width)
    die_idx = next(
        (i for i, die in enumerate(dies) if die is device or die.rank == device.rank),
        stage * width,
    )
    replica = die_idx // stage_span
    total_replicas = max(1, int(parallel.dp) * int(getattr(parallel, "external_dp", 1) or 1))
    max_replicas = min(total_replicas, len(dies) // stage_span)
    topo = Topology(case.machine, parallel)
    family = str(metadata.get("collective", op.family)).strip().lower()
    axis = metadata.get("group_axis", "tp")
    groups: tuple[tuple[int, ...], ...] = ()
    directed = False
    explicit_ranks = _explicit_group_ranks(metadata, topo)
    if family == "pp_p2p" or axis == "pp":
        dest_stage = (
            int(metadata["destination_stage"])
            if "destination_stage" in metadata
            else stage + 1
        )
        if 0 <= stage < parallel.pp and 0 <= dest_stage < parallel.pp and dest_stage != stage:
            groups = tuple(
                (src, dst)
                for src, dst in zip(topo.ranks(replica, stage), topo.ranks(replica, dest_stage))
                if src != dst
            )
            directed = True
    elif family in {"p2p", "p2p_send_recv", "send", "recv", "send_recv"}:
        pair = _parse_rank_pair(metadata.get("p2p_pair"))
        if pair is not None and all(r in topo._rank_node for r in pair):
            groups = (pair,)
            directed = True
    elif explicit_ranks is not None:
        groups = (explicit_ranks,)
    elif 0 <= replica < max_replicas and 0 <= stage < parallel.pp and axis in _TOPOLOGY_GROUP_AXES:
        if axis != "local_tp" or metadata.get("local_tp_width"):
            groups = tuple(topo.groups(axis, replica, stage, metadata=metadata))
            if device_explicit:
                device_groups = tuple(grp for grp in groups if device.rank in grp)
                if device_groups:
                    groups = device_groups
    if not groups:
        return None
    linked_ranks = {int(rank) for link in case.machine.links for rank in link.members}
    pairwise = (
        family in _PAIRWISE_FAMILIES
        or "peer_traffic" in metadata
        or getattr(op, "recipe", "") in _PAIRWISE_RECIPES
    )
    rail_id = metadata.get("transport_rail_id")
    routes = []
    for grp in groups:
        peer_pairs = _peer_traffic_pairs(grp, metadata)
        weighted_pairs = (
            peer_pairs
            if peer_pairs is not None
            else [(src, dst, None) for src, dst in _route_pairs_for_group(
                grp, topo, directed=directed, pairwise=pairwise)]
        )
        for src, dst, share in weighted_pairs:
            if src not in linked_ranks or dst not in linked_ranks:
                continue
            path = topo.route(src, dst, rail_id=rail_id)
            entry = {
                "source": int(src),
                "destination": int(dst),
                "hops": [
                    {
                        "link": link_name,
                        "channel": topo.links[link_name].channel,
                        "scope": topo.link_scope(link_name),
                    }
                    for link_name, _, _ in path
                ],
            }
            if share is not None:
                entry["share"] = share
            routes.append(entry)
    result = {
        "groups": [[int(rank) for rank in grp] for grp in groups],
        "routes": routes,
    }
    if rail_id is not None:
        result["rail_id"] = rail_id
    return result


def _zero_resource_node(op) -> bool:
    """A pure sync/join has no physical work and no native precision."""
    return not (
        float(getattr(op, "flops", 0.0) or 0.0)
        or tuple(getattr(op, "matrix_flops", ()) or ())
        or float(getattr(op, "hbm_bytes", 0.0) or 0.0)
        or float(getattr(op, "mte2_bytes", 0.0) or 0.0)
        or float(getattr(op, "mte3_bytes", 0.0) or 0.0)
        or float(getattr(op, "vector_ops", 0.0) or 0.0)
        or float(getattr(op, "link_bytes", 0.0) or 0.0)
    )


def _zero_resource_identity(case, op, metadata, scope, device):
    """Identity for a barrier/join: no model or dtype lookup is attempted."""
    runtime = getattr(case, "runtime", None) if case is not None else None
    machine = getattr(case, "machine", None) if case is not None else None
    hardware = {}
    if machine is not None:
        hardware = {"key": getattr(machine, "hardware_key", "") or
                    getattr(machine, "sku", ""),
                    "architecture": getattr(machine, "arch_family", "")}
        dies = tuple(getattr(machine, "dies", ()) or ())
        selected = device
        if selected is None and dies:
            try:
                selected = dies[op.stage]
            except (AttributeError, IndexError, TypeError):
                selected = None
        if selected is not None:
            hardware.update({
                "compute_units": getattr(selected, "compute_units", 0),
                "resource_units": dict(getattr(selected, "resource_units", {}) or {}),
            })
    return {
        "hardware": hardware,
        "runtime": getattr(runtime, "vllm_impl", "") if runtime is not None else "",
        "revision": getattr(runtime, "vllm_version", "") if runtime is not None else "",
        "format": getattr(runtime, "kernel_format", "") if runtime is not None else "",
        "kernel": metadata.get("kernel", metadata.get("kernel_name", op.recipe)),
        "implementation": metadata.get("implementation", {}),
        "dtype": "",
        "layout": op.layout,
        "replay_scope": scope or "none",
        "deps": list(getattr(op, "deps", ()) or ()),
        "stream": getattr(op, "stream", ""),
        "axes": metadata.get("service_axes", {}),
    }


def service_identity(case, op, scope="none", *, device=None):
    metadata = dict(op.metadata)
    zero = (_zero_resource_node(op)
            and not getattr(op, "dtype", "")
            and op.family not in _COMMUNICATION_FAMILIES
            and not metadata.get("hccl_selection")
            and not any(key in metadata for key in _COMMUNICATION_KEYS))
    if zero:
        return json.loads(json.dumps(
            _zero_resource_identity(case, op, metadata, scope, device),
            sort_keys=True, allow_nan=False))
    device_explicit = device is not None
    device = device or case.machine.dies[
        op.stage * case.runtime.parallel.tp * case.runtime.parallel.pcp]
    identity = {
        "hardware": {
            "key": case.machine.hardware_key or case.machine.sku,
            "architecture": case.machine.arch_family,
            "compute_units": device.compute_units,
            "resource_units": dict(device.resource_units),
        },
        "runtime": case.runtime.vllm_impl,
        "revision": case.runtime.vllm_version,
        "format": case.runtime.kernel_format,
        "kernel": metadata.get("kernel", metadata.get("kernel_name", op.recipe)),
        "implementation": metadata.get("implementation", {}),
        "dtype": op.dtype or case.model.compute_dtype,
        "layout": op.layout, "replay_scope": scope or "none",
        "axes": metadata.get("service_axes", {}),
    }
    if (op.family in _COMMUNICATION_FAMILIES
            or metadata.get("hccl_selection")
            or any(key in metadata for key in _COMMUNICATION_KEYS)):
        # A native communication envelope includes the declared physical
        # transport domain. Equal SKU/local pipes do not identify changed
        # link bandwidth, latency, framing, hops or rank placement.
        parallel = case.runtime.parallel
        identity["transport"] = {
            "links": [link.to_dict() for link in case.machine.links],
            "placements": [{"rank": die.rank, "node": die.node, "package": die.package}
                           for die in case.machine.dies],
            "ranks_per_node": case.machine.ranks_per_node,
            "dies_per_package": case.machine.dies_per_package,
            # The declared parallel decomposition decides which ranks share a
            # link.  A curve bound at one axis split cannot describe another,
            # even when the SKU and die set are identical.
            "parallel": {
                "tp": parallel.tp, "dp": parallel.dp, "pp": parallel.pp,
                "ep": parallel.ep, "cp": parallel.cp, "pcp": parallel.pcp,
                "sp": bool(getattr(parallel, "sp", dict(
                    getattr(case.runtime, "metadata", ()) or ()).get("sp", False))),
                **({"rank_table": [list(group) for group in parallel.rank_table]}
                   if getattr(parallel, "rank_table", None) is not None else {}),
            },
        }
        resolved_route = _resolved_transport_route(
            case, op, metadata, device, device_explicit=device_explicit)
        if resolved_route is not None:
            identity["transport"].update(resolved_route)
        for key in _COMMUNICATION_KEYS:
            if key in metadata:
                identity["transport"][key] = metadata[key]
        if (op.family in _COMMUNICATION_FAMILIES
                and not metadata.get("device_kernel_group")
                and "collective" not in identity["transport"]
                and "a5_collective" not in identity["transport"]):
            identity["transport"]["collective"] = op.family
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class ServiceEstimate:
    work: float
    startup_ms: float
    service_ms: float
    basis: str

    @property
    def duration_ms(self):
        return self.startup_ms + self.service_ms


@dataclass(frozen=True)
class ServiceCurve:
    resource: str
    work_unit: str
    identity: Mapping
    boundary: str
    active_units: int
    startup_ms: float
    knots: tuple[tuple[float, float], ...]
    provenance: Mapping = field(default_factory=dict)
    model_id: str = field(init=False)
    _coordinates: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if self.boundary not in {"device_kernel", "resource_phase"}:
            raise ModelingError("service_curve_requires_device_or_resource_phase_boundary")
        if self.active_units < 1:
            raise ModelingError("service_curve_requires_positive_active_units")
        if (len(self.knots) < 2 or not math.isfinite(self.startup_ms)
                or self.startup_ms < 0):
            raise ModelingError("service_curve_requires_two_knots_and_nonnegative_startup")
        xs = tuple(x for x, _ in self.knots)
        if (any(not math.isfinite(x) or x <= 0 for x in xs)
                or any(b <= a for a, b in zip(xs, xs[1:]))
                or any(not math.isfinite(t) or t <= self.startup_ms for _, t in self.knots)):
            raise ModelingError("service_curve_requires_ordered_work_and_positive_service")
        object.__setattr__(self, "identity", json.loads(json.dumps(self.identity, allow_nan=False)))
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        object.__setattr__(self, "_coordinates", xs)
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "model_id", hashlib.sha256(payload.encode()).hexdigest())

    @classmethod
    def from_mapping(cls, data):
        if data["schema"] != "mindie-sim.service-curve/1":
            raise ModelingError("service_curve_schema_unknown")
        return cls(data["resource"], data["work_unit"], data["identity"], data["boundary"],
                   data["active_units"], data["startup_ms"], tuple(tuple(p) for p in data["knots"]),
                   data.get("provenance", {}))

    def to_dict(self):
        return {"schema": "mindie-sim.service-curve/1", "resource": self.resource,
                "work_unit": self.work_unit, "identity": deepcopy(self.identity),
                "boundary": self.boundary, "active_units": self.active_units,
                "startup_ms": self.startup_ms,
                "knots": self.knots, "provenance": deepcopy(self.provenance)}

    def estimate(self, work):
        """No extrapolation or categorical substitution is hidden in this lookup."""
        if not self._coordinates[0] <= work <= self._coordinates[-1]:
            return None
        right = bisect_left(self._coordinates, work)
        if self._coordinates[right] == work:
            duration, basis = self.knots[right][1], "anchor"
        else:
            x0, t0 = self.knots[right - 1]
            x1, t1 = self.knots[right]
            duration = t0 + (t1 - t0) * (work - x0) / (x1 - x0)
            basis = "interpolated"
        return ServiceEstimate(work, self.startup_ms, duration - self.startup_ms, basis)


def resolved_phase_work(context, work):
    """Resolve one native phase demand onto physical pipes exactly once.

    ``RecipeContext.for_phase`` already binds its declared phase work.  Reusing
    that normalized demand prevents a service curve from multiplying a mode's
    physical-rate conversion twice.  A context without a prepared phase keeps
    the ordinary one-time bind for standalone use.
    """
    native = {**work.resources, "cube": sum(work.matrix.values())}
    native = {key: value for key, value in native.items() if value}
    if context.work and native and native == dict(context.native_work):
        # The phase context already carries the normalized result of exactly
        # this native demand. Reuse it instead of applying the mode conversion
        # a second time.
        return dict(context.work)
    expected = context.bind(work)
    if not context.work:
        return expected
    if (set(expected) != set(context.work)
            or any(not math.isclose(expected[key], context.work[key],
                                    rel_tol=1e-12, abs_tol=1e-12)
                   for key in expected)):
        raise ModelingError("service_work_does_not_match_phase_context")
    return dict(context.work)


@dataclass(frozen=True)
class ResourceService:
    """One isolated resource invocation; phase composition stays in its caller."""

    model: ServiceCurve | Mapping
    outside_domain: str = "analytic"
    units: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.model, ServiceCurve):
            object.__setattr__(self, "model", ServiceCurve.from_mapping(self.model))
        if self.outside_domain not in {"analytic", "error"}:
            raise ModelingError("service_domain_policy_requires_analytic_or_error")

    def lower(self, context):
        from mindie_sim.prediction.mechanism import Work
        curve = self.model
        if not context.work:
            return KernelProgram((Stage("empty_boundary", keep_empty=True),),
                                 minimum_ms=context.minimum_ms)
        if curve.identity != context.service_identity:
            raise ModelingError(f"service_identity_mismatch:{curve.model_id}")
        if curve.boundary != context.service_boundary:
            raise ModelingError("service_boundary_mismatch")
        role = curve.resource
        if curve.work_unit != context.work_units.get(role, "work"):
            raise ModelingError(f"service_work_unit_mismatch:{role}")
        if set(context.native_work) != {role}:
            raise ModelingError("service_curve_requires_one_explicit_resource_phase")
        amount = context.native_work.get(role, 0)
        if role == "cube":
            portions = dict(context.op.matrix_flops) or {curve.identity["dtype"]: amount}
            if len(portions) != 1 or next(iter(portions)) != curve.identity["dtype"]:
                raise ModelingError("service_curve_requires_one_compute_precision")
            work = Work(matrix=portions)
        elif role == "link" and role not in context.resource_names:
            expected = {name: val for name, val in context.work.items()
                        if name not in context.resource_names.values()}
            if not expected or expected != context.work:
                raise ModelingError("service_curve_requires_one_explicit_resource_phase")
            owner_count = context.resource_units.get(
                context.physical_role(role), context.core_count)
            active_units = len(self.units) if self.units else owner_count
            if active_units != curve.active_units:
                raise ModelingError("service_active_units_mismatch")
            estimate = curve.estimate(amount)
            if estimate is None and self.outside_domain == "error":
                raise ModelingError(f"service_work_outside_domain:{amount}")
            caps = ({res: val / estimate.service_ms for res, val in expected.items()}
                    if estimate else {})
            capacity_limited = estimate is not None and any(
                caps[res] > context.rates[res] * active_units / owner_count
                for res in expected)
            if context.record_service is not None:
                context.record_service(
                    curve, estimate.basis if estimate else "analytic_outside_domain",
                    capacity_limited=capacity_limited,
                )
            if estimate is None:
                return KernelProgram((
                    Stage("setup", latency=context.setup_ms),
                    Stage("service", context.work, after=("setup",)),
                ), minimum_ms=context.minimum_ms)
            return KernelProgram((
                Stage("setup", latency=context.setup_ms + estimate.startup_ms),
                Stage("service", context.work, after=("setup",), caps=caps),
            ), minimum_ms=context.minimum_ms)
        else:
            work = Work(resources={role: amount})
        expected = resolved_phase_work(context, work)
        # One measured response cannot identify the occupancy of several pipes,
        # or several instruction modes sharing a pipe. The caller must split it.
        if len(expected) != 1 or expected != context.work:
            raise ModelingError("service_curve_requires_one_explicit_resource_phase")
        resource = next(iter(expected))
        owner_count = context.resource_units.get(context.physical_role(role), context.core_count)
        active_units = len(self.units) if self.units else owner_count
        if active_units != curve.active_units:
            raise ModelingError("service_active_units_mismatch")
        placement = {resource: tuple(self.units)} if self.units else {}
        estimate = curve.estimate(amount)
        if estimate is None and self.outside_domain == "error":
            raise ModelingError(f"service_work_outside_domain:{amount}")
        supply = expected[resource] / estimate.service_ms if estimate else math.inf
        physical_supply = context.rates[resource] * active_units / owner_count
        if context.record_service is not None:
            context.record_service(
                curve, estimate.basis if estimate else "analytic_outside_domain",
                capacity_limited=estimate is not None and supply > physical_supply,
            )
        if estimate is None:
            return KernelProgram((
                Stage("setup", latency=context.setup_ms),
                Stage("service", context.work, after=("setup",), placement=placement),
            ), minimum_ms=context.minimum_ms)
        return KernelProgram((
            Stage("setup", latency=context.setup_ms + estimate.startup_ms),
            Stage("service", context.work, after=("setup",),
                  caps={resource: supply}, placement=placement),
        ), minimum_ms=context.minimum_ms)
