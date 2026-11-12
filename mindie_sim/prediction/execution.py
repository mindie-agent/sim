"""Compile iteration DAGs onto the shared physical execution engine.

[Role] Bind rank placement, host submission, kernel recipes and device streams.
[Invariants] Model names do not select execution. Same-stream order and data
  dependencies are distinct from resource contention. Host submission can run
  ahead of device completion. Link work is never reduced by overlap.
[Timeouts & Locks] Finite virtual time; at most active iteration DAGs are live.
[Side-Effect Policy] Mutates only the simulation Engine and aggregate counters.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
import re
from typing import Any, Mapping

from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from .cost import CostModel
from .cohort_traffic import bind_cohort_traffic
from .collectives import plan_options as collective_options
from .engine import Engine, Event, Resource
from .mechanism import Geometry, Work
from .program import RecipeBook, RecipeContext
from .topology import Topology
from .serving_io import ServingIO
from .step_metrics import StepSummary


def _freeze_key(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_key(item) for item in value)
    if isinstance(value, list):
        return ("__list__", tuple(_freeze_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("__set__", tuple(sorted((_freeze_key(item) for item in value), key=repr)))
    if isinstance(value, Mapping):
        return ("__map__", tuple(sorted(((str(k), _freeze_key(v)) for k, v in value.items()), key=lambda pair: pair[0])))
    try:
        hash(value)
        return value
    except TypeError:
        return ("__repr__", type(value), repr(value))


_LAYER_PREFIX_RE = re.compile(r"\bs\d+\.l\d+\.")
_DIAGNOSTIC_ONLY_META_KEYS = frozenset({
    "structure_subgraphs", "fusion_semantic_ids", "fusion_external_outputs",
})


def _op_meta_key(op) -> Any:
    def _norm(value: Any) -> Any:
        if isinstance(value, str):
            return _LAYER_PREFIX_RE.sub("s_.l_.", value)
        if isinstance(value, (int, float, bool, bytes, type(None))):
            return value
        if isinstance(value, tuple):
            return tuple(_norm(item) for item in value)
        if isinstance(value, list):
            return ("__list__", tuple(_norm(item) for item in value))
        if isinstance(value, (set, frozenset)):
            return ("__set__", tuple(sorted((_norm(item) for item in value), key=repr)))
        if isinstance(value, Mapping):
            return ("__map__", tuple(sorted(
                ((str(k), _norm(v)) for k, v in value.items()
                 if k not in _DIAGNOSTIC_ONLY_META_KEYS),
                key=lambda pair: pair[0],
            )))
        try:
            hash(value)
            return value
        except TypeError:
            return ("__repr__", type(value), repr(value))

    return _norm(dict(op.metadata))


def _rank_token(context) -> str | None:
    """Replica marker embedded in physical resource names, such as ``@r3``."""
    for name in context.resource_names.values():
        start = name.find("@r")
        if start < 0:
            continue
        end = start + 2
        while end < len(name) and name[end].isdigit():
            end += 1
        if end > start + 2:
            return name[start:end]
    return None


def _collective_native_basis(op, group_size):
    """Logical per-rank native input admitted by a TP/DP collective op."""
    metadata = dict(op.metadata)
    collective = str(metadata.get("collective", op.family)).lower()
    collective = {"allgather": "all_gather", "reducescatter": "reduce_scatter",
                  "alltoall": "all_to_all"}.get(collective, collective)
    count = max(1, int(group_size) - 1)
    if collective == "all_gather" and op.link_bytes > 0:
        expected = Fraction(str(op.link_bytes)) / count
    elif collective == "reduce_scatter" and op.link_bytes > 0:
        expected = Fraction(str(op.link_bytes * group_size)) / count
    else:
        expected = Fraction(op.link_bytes) if op.link_bytes > 0 else None
    dtype = (metadata.get("payload_dtype") or metadata.get("dtype")
             or metadata.get("activation_dtype") or op.dtype or None)
    return collective, expected, dtype


def _normalize_token(value, token: str):
    if isinstance(value, str):
        return value.replace(token, "@r*")
    if isinstance(value, tuple):
        return tuple(_normalize_token(item, token) for item in value)
    if isinstance(value, list):
        return [_normalize_token(item, token) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_token(item, token) for key, item in value.items()}
    return value


def _isolated_script_key(prog_key, context, token: str):
    """Physical identity shared by equal ranks. Replica ids are not part of it."""
    rates = tuple(sorted(
        (role, context.rates[name])
        for role, name in context.resource_names.items()
        if name in context.rates
    ))
    pools = tuple(sorted(
        (str(key), float(value)) for key, value in context.pool_capacities.items()
    ))
    return (_normalize_token(prog_key[1:], token), rates, pools, int(context.core_count))


def _meta_diff(orig: Any, res: Any) -> Any:
    if isinstance(orig, Mapping) and isinstance(res, Mapping):
        removed = frozenset(orig.keys() - res.keys())
        updated = {
            k: (_meta_diff(orig[k], v) if k in orig else ("__set__", v))
            for k, v in res.items() if k not in orig or orig[k] != v
        }
        return ("__dict__", removed, updated)
    if (isinstance(orig, tuple) and isinstance(res, tuple) and len(orig) == len(res)
            and orig and all(isinstance(x, Mapping) for x in orig)
            and all(isinstance(x, Mapping) for x in res)):
        return ("__tuple_map__", tuple(_meta_diff(a, b) for a, b in zip(orig, res)))
    return ("__set__", res)


def _meta_patch(orig: Any, diff: Any) -> Any:
    kind = diff[0]
    if kind == "__set__":
        return diff[1]
    if kind == "__dict__" and isinstance(orig, Mapping):
        _, removed, updated = diff
        out = {k: v for k, v in orig.items() if k not in removed}
        for k, sub in updated.items():
            out[k] = _meta_patch(out[k], sub) if k in out else sub[1]
        return out
    if kind == "__tuple_map__" and isinstance(orig, tuple) and len(orig) == len(diff[1]):
        return tuple(_meta_patch(a, sub) for a, sub in zip(orig, diff[1]))
    return diff[-1]


class ResourceRates(Mapping):
    """Read-only live rate view; PD extends the same Engine with new namespaces."""

    def __init__(self, resources):
        self.resources = resources

    def __getitem__(self, key):
        return self.resources[key].rate

    def __iter__(self):
        return iter(self.resources)

    def __len__(self):
        return len(self.resources)


def _execution_geometry(device):
    """A representative shard requires identical physical owner topology."""
    roles = dict(device.rates).keys() | {"cube", "host"}
    return (
        device.compute_units,
        tuple(sorted((role, device.units(role)) for role in roles)),
        tuple(sorted((pool, device.pool_units(pool)) for pool, _ in device.buffers)),
        tuple(sorted((mode, role) for mode, (role, _) in device.service_modes)),
        tuple(sorted(dict(device.compute_peaks))),
        tuple(sorted((role, dict(device.work_units).get(role, NATIVE_WORK_UNITS.get(role, "work")))
                     for role in roles | dict(device.service_modes).keys())),
    )


@dataclass(frozen=True)
class SubmissionPolicy:
    graph_replay_ms: float | None = None
    scheduler_ms: float = 0
    completion_ms: float = 0
    kernel_launch_ms: float | None = None

    def launch_ms(self, dies, *, captured=False):
        """Software submission overrides the explicit legacy device parameter."""
        eager = (self.kernel_launch_ms if self.kernel_launch_ms is not None
                 else max(d.launch_us for d in dies) / 1000)
        return (self.graph_replay_ms if captured and self.graph_replay_ms is not None
                else eager)


@dataclass(frozen=True)
class ScheduledIteration:
    replica: int
    iteration: int
    completion: Event
    start_ms: float
    mode: str
    valid_tokens: int
    padded_tokens: int
    graph_bucket: int | None
    replay_scope: str
    operators: int
    commits: Mapping[str, Event]
    device_complete: Event
    host_results: Mapping[str, Event]
    client_deliveries: Mapping[str, Event]
    implementation_selections: tuple[dict, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica": self.replica, "iteration": self.iteration,
            "start_ms": self.start_ms, "end_ms": self.completion.time,
            "duration_ms": self.completion.time - self.start_ms,
            "mode": self.mode, "valid_tokens": self.valid_tokens,
            "padded_tokens": self.padded_tokens, "graph_bucket": self.graph_bucket,
            "replay_scope": self.replay_scope, "operators": self.operators,
            "device_end_ms": self.device_complete.time,
            "host_result_ready_ms": max((event.time for event in self.host_results.values()), default=None),
            "implementation_selections": list(self.implementation_selections),
        }


@dataclass(frozen=True)
class CollectiveParticipant:
    replica: int
    ready: Event
    context: RecipeContext
    recipe: Any
    name: str
    completion: Event


class _RankEngine:
    """Name-mapping view over the one shared Engine.

    It does not schedule, allocate a second clock, or own tasks.  It only
    translates a launch's native pipe names (including hardcoded source keys
    such as ``pipe_s_issue``) to that rank's physical resources before the
    shared Engine prepares the task.
    """

    def __init__(self, base, resource_names):
        self.base = base
        self.resource_names = {str(key): str(value)
                               for key, value in dict(resource_names).items()}

    def _work(self, work):
        return {self.resource_names.get(str(key), str(key)): value
                for key, value in dict(work or {}).items()}

    def _keys(self, values):
        return {self.resource_names.get(str(key), str(key)): value
                for key, value in dict(values or {}).items()}

    def submit(self, name, work=None, **kwargs):
        return self.base.submit(
            name, self._work(work) if work is not None else None, **kwargs)

    def _prepare(self, name, work=None, *, latency=0.0, caps=None,
                 acquire=(), free=(), placement=None, coupled=False):
        return self.base._prepare(
            name, self._work(work) if work is not None else None,
            latency=latency, caps=self._keys(caps), acquire=acquire,
            free=free, placement=self._keys(placement), coupled=coupled)

    def _submit_prepared(self, prepared, name, **kwargs):
        return self.base._submit_prepared(prepared, name, **kwargs)

    def future(self, name):
        return self.base.future(name)

    def resolve(self, event):
        return self.base.resolve(event)

    def __getattr__(self, name):
        return getattr(self.base, name)


def _bind_native_resource_view(program, resource_names, engine):
    """Shallow launch-local resource view; cached program is not mutated."""
    if not resource_names:
        return program
    from .piecewise.compiler import NativeProgram
    inner = copy(program.program)
    names = dict(getattr(inner, "names", {}) or {})
    rates = dict(getattr(inner, "rates", {}) or {})
    units = dict(getattr(inner, "units", {}) or {})
    for pipe, physical in dict(resource_names).items():
        names[str(pipe)] = str(physical)
        resource = engine.resources.get(str(physical))
        if resource is not None:
            rates[str(pipe)] = resource.rate
            rates[str(physical)] = resource.rate
            units[str(pipe)] = resource.units
            units[str(physical)] = resource.units
    inner.names = names
    inner.rates = rates
    inner.units = units
    return NativeProgram(inner, program.work, program.duration)


def launch_native_cohort(engine, launches, *, name_prefix="native_cohort",
                         owner="native_cohort", defer=False):
    """Two-phase launch of native peer programs on one shared Engine.

    Each launch receives a private launch-local rank/topology view; the cached
    native program is never mutated.  Directed ``(phase, src_rank, dst_rank)``
    futures are allocated before any program starts.  After every program has
    produced its real per-phase send handles, each future is bound to the
    matching sender's actual send Event.  Receivers wait the future, so only the
    named peer/phase join is consumed.  Per-rank parent Events depend only on
    that rank's own program finish; the cohort join waits every real finish.

    ``launches`` is a sequence of mappings (or attribute objects) with:
      program: piecewise.compiler.NativeProgram
      name: unique launch name
      ready: actual local ready Event
      rank_index: launch-local EP rank index
      peer_count / peer_labels / node_ranks / peer_classes / peer_weights:
        optional launch-local geometry.  Absent fields stay the program's own
        declared values; no scalar wait or measured timestamp is accepted.

    Returns finishes, per-rank parents, the genuine global join, launch-local
    handles, and unresolved peer labels.
    """
    from .piecewise.compiler import NativeProgram
    from .piecewise.families_moe_mc2 import Mc2LaunchView

    def field(raw, key, default=None):
        return (raw.get(key, default) if isinstance(raw, Mapping)
                else getattr(raw, key, default))

    infos, by_rank, names = [], {}, set()
    phases = None
    for raw in launches:
        program = field(raw, "program")
        if not isinstance(program, NativeProgram):
            raise ModelingError("native_cohort_requires_native_program")
        name = str(field(raw, "name") or "")
        if not name or name in names:
            raise ModelingError(f"native_cohort_launch_name_invalid:{name}")
        names.add(name)
        rank = field(raw, "rank_index")
        if rank is None:
            raise ModelingError(f"native_cohort_rank_missing:{name}")
        rank = int(rank)
        if rank in by_rank:
            raise ModelingError(f"native_cohort_rank_duplicate:{rank}")
        inner = program.program
        source_view = getattr(inner, "view", None)
        peer_count = int(field(raw, "peer_count")
                         or getattr(inner, "peer_count", 0) or 0)
        if peer_count <= 0:
            raise ModelingError(f"native_cohort_peer_count_missing:{name}")
        if not 0 <= rank < peer_count:
            raise ModelingError(f"native_cohort_rank_out_of_range:{name}:{rank}")
        labels = field(raw, "peer_labels")
        labels = (tuple(int(item) for item in labels)
                  if labels is not None else
                  tuple(item for item in range(peer_count) if item != rank))
        if len(set(labels)) != len(labels) or (
                rank in labels
                or any(not 0 <= item < peer_count for item in labels)):
            raise ModelingError(f"native_cohort_peer_labels_invalid:{name}")
        node_ranks = field(raw, "node_ranks")
        node = int(node_ranks if node_ranks is not None else
                   (getattr(source_view, "node_ranks", 0)
                    if source_view is not None else 0))
        peer_classes = field(raw, "peer_classes")
        classes = tuple(peer_classes or ())
        if not classes and source_view is not None and source_view.rank == rank:
            classes = tuple(source_view.peer_classes)
        peer_weights = field(raw, "peer_weights")
        weights = tuple(peer_weights or ())
        if not weights and source_view is not None and source_view.rank == rank:
            weights = tuple(source_view.peer_weights)
        intra = field(raw, "intra_latency_ms")
        inter = field(raw, "inter_latency_ms")
        resource_names = field(raw, "resource_names")
        frames_raw = field(raw, "payload_frames", None)
        if frames_raw is None:
            # An exact source ledger is indexed by physical source rank, not
            # by the representative graph's rank. The same program is launched
            # once per replica with its own whole routed frames.
            frames_for = getattr(inner, "frames_for", None)
            if callable(frames_for):
                frames_raw = frames_for(rank)
        if frames_raw is None and source_view is not None:
            frames_raw = getattr(source_view, "payload_frames", None)
        if isinstance(frames_raw, Mapping):
            payload_frames = tuple(sorted(
                (int(peer), int(frames)) for peer, frames in frames_raw.items()
                if int(frames) > 0))
        elif frames_raw:
            payload_frames = tuple(sorted(
                (int(item[0]), int(item[1])) for item in frames_raw
                if int(item[1]) > 0))
        else:
            payload_frames = ()
        packet = field(raw, "payload_packet_bytes", None)
        if packet is None and source_view is not None:
            packet = getattr(source_view, "payload_packet_bytes", 0)
        payload_packet = int(packet or 0)
        basis_raw = field(raw, "payload_basis", None)
        if basis_raw is None:
            basis_for = getattr(inner, "payload_basis_for", None)
            if callable(basis_for):
                basis_raw = basis_for(rank)
        if basis_raw is None:
            basis_raw = (getattr(source_view, "payload_basis", "unknown")
                         if source_view is not None else "unknown")
        payload_basis = str(basis_raw)
        local_frames = field(raw, "payload_local_frames", None)
        if local_frames is None:
            local_for = getattr(inner, "local_frames_for", None)
            if callable(local_for):
                local_frames = local_for(rank)
        if local_frames is None and source_view is not None:
            local_frames = getattr(source_view, "payload_local_frames", 0)
        payload_local = int(local_frames or 0)
        view = Mc2LaunchView(
            rank=rank, peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=node,
            intra_latency_ms=(getattr(source_view, "intra_latency_ms", 0)
                              if intra is None and source_view is not None
                              else (intra or 0)),
            inter_latency_ms=(getattr(source_view, "inter_latency_ms", 0)
                              if inter is None and source_view is not None
                              else (inter or 0)),
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(
                    resource_names
                    or (source_view.resource_names if source_view is not None
                        else ())).items())),
            payload_frames=payload_frames,
            payload_packet_bytes=payload_packet,
            payload_basis=payload_basis,
            payload_local_frames=payload_local)
        launch_phases = tuple(getattr(inner, "phases", ()))
        phase_names = tuple(getattr(phase, "name", str(phase))
                            for phase in launch_phases)
        if phases is None:
            phases = phase_names
        elif set(phases) != set(phase_names):
            raise ModelingError(
                f"native_cohort_phase_mismatch:{name}:"
                f"{sorted(set(phases) ^ set(phase_names))}")
        infos.append({
            "name": name, "rank": rank, "view": view,
            "program": program, "ready": field(raw, "ready"),
            "operator_id": str(field(raw, "operator_id", "mc2")),
            "resource_names": dict(resource_names or {}),
            "payload_frames": dict(payload_frames),
            "payload_packet_bytes": payload_packet,
            "payload_basis": payload_basis,
            "payload_local_frames": payload_local,
            "transport": field(raw, "transport", None),
        })
        by_rank[rank] = infos[-1]
    if not infos:
        raise ModelingError("native_cohort_empty")
    link_phases = tuple(
        phase for phase in (phases or ())
        if any(getattr(info["program"].program, "link_by_phase", {}).get(phase)
               for info in infos))
    for info in infos:
        info["program"] = _bind_native_resource_view(
            info["program"], info["resource_names"], engine)
    programs_by_rank = {info["rank"]: info["program"] for info in infos}
    names_by_rank = {info["rank"]: info["name"] for info in infos}
    unresolved, partial = [], []
    placeholders = {}
    # Expected directed cells are real local receive prerequisites even when
    # no witnessed integer frame count exists.  Prebind the same futures for
    # every cell a launch program will resolve; receivers wait these handles
    # instead of a global phase barrier.
    for info in infos:
        transport = info.get("transport")
        inner = info["program"].program
        if transport is None or not hasattr(inner, "transport_cells"):
            continue
        for phase_name, count_phase in (
                ("payload_exchange", False), ("count_status_exchange", True)):
            if phase_name not in (phases or ()):
                continue
            for _si, di, _src, _dst, useful, _frames in inner.transport_cells(
                    transport, count=count_phase):
                if useful <= 0 or _si == di:
                    continue
                placeholders.setdefault(
                    (_si, di, phase_name),
                    engine.future(f"{name_prefix}:{phase_name}:{_si}->{di}"))
    protocol_edges = 0
    directed_edges = 0
    directed_frames = 0
    fullmesh_by_rank = {}
    payload_packet_by_rank = {}
    for info in infos:
        source_inner = info["program"].program
        view = info["view"]
        rank = int(info["rank"])
        source_phases = tuple(
            (str(phase), links) for phase, links
            in getattr(source_inner, "link_by_phase", {}).items() if links)
        fullmesh = tuple(sorted(
            phase for phase, _links in source_phases
            if phase != "payload_exchange"))
        fullmesh_by_rank[info["name"]] = fullmesh
        for phase, _links in source_phases:
            if phase != "payload_exchange":
                # Count/control is a real fullmesh protocol: every expected
                # peer participates.
                for peer in view.peer_labels:
                    if peer in programs_by_rank:
                        key = (rank, peer, phase)
                        if key not in placeholders:
                            placeholders[key] = engine.future(
                                f"{name_prefix}:{phase}:{rank}->{peer}")
                        protocol_edges += 1
                    else:
                        edge = (phase, rank, peer)
                        if edge not in unresolved:
                            unresolved.append(edge)
                continue
            frames = dict(info.get("payload_frames") or {})
            basis = str(info.get("payload_basis") or "unknown")
            packet = int(info.get("payload_packet_bytes") or 0)
            payload_packet_by_rank[info["name"]] = packet
            if frames:
                if packet <= 0:
                    partial.append({
                        "rank": rank, "name": info["name"],
                        "phase": "payload_exchange", "basis": basis,
                        "reason": "payload_packet_bytes_missing"})
                    continue
                for peer, whole_frames in frames.items():
                    peer = int(peer)
                    whole_frames = int(whole_frames)
                    if whole_frames <= 0:
                        continue
                    if peer not in view.peer_labels:
                        partial.append({
                            "rank": rank, "name": info["name"],
                            "phase": "payload_exchange", "peer": peer,
                            "basis": basis,
                            "reason": "destination_not_declared"})
                        continue
                    if peer in programs_by_rank:
                        key = (rank, peer, "payload_exchange")
                        if key not in placeholders:
                            placeholders[key] = engine.future(
                                f"{name_prefix}:payload_exchange:"
                                f"{rank}->{peer}")
                        directed_edges += 1
                        directed_frames += whole_frames
                    else:
                        edge = ("payload_exchange", rank, peer)
                        if edge not in unresolved:
                            unresolved.append(edge)
                continue
            if "zero" in basis.lower():
                # Explicit known zero: no future, no invented arrival.
                continue
            partial.append({
                "rank": rank, "name": info["name"],
                "phase": "payload_exchange", "basis": basis,
                "reason": "payload_destinations_unknown"})
        if not any(phase == "payload_exchange"
                   for phase, _links in source_phases):
            frames = dict(info.get("payload_frames") or {})
            basis = str(info.get("payload_basis") or "unknown")
            if not frames and "zero" not in basis.lower():
                partial.append({
                    "rank": rank, "name": info["name"],
                    "phase": "payload_exchange", "basis": basis,
                    "reason": "payload_destinations_unknown"})
    def _declared_metrics(info):
        inner = info["program"].program
        transport = info.get("transport") or {}
        rank = int(info["rank"])
        payload, count = (inner.transport_account(transport)
                          if hasattr(inner, "transport_account") else ({}, {}))
        phases = {phase.name for phase in getattr(inner, "phases", ())}
        records = {}
        if payload:
            records["payload_exchange"] = {
                **payload,
                "basis": (inner.payload_basis_for(rank)
                          if hasattr(inner, "payload_basis_for")
                          else info["payload_basis"]),
                "count": False,
            }
        if count and "count_status_exchange" in phases:
            count_basis = (inner.payload_basis_for(rank)
                           if hasattr(inner, "payload_basis_for")
                           else info["payload_basis"])
            records["count_status_exchange"] = {
                **count,
                "basis": count_basis,
                "count": True,
            }
        links_for_rank = getattr(inner, "link_by_phase_for", None)
        links = (dict(links_for_rank(rank)) if callable(links_for_rank)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        work_for_rank = getattr(inner, "work_for_rank", None)
        work = (tuple(work_for_rank(rank)) if callable(work_for_rank)
                else tuple(getattr(info["program"], "work", ()) or ()))
        transport_work = dict(payload.get("work", {}))
        if "count_status_exchange" in phases:
            for pipe, amount in (count.get("work", {}) or {}).items():
                transport_work[pipe] = transport_work.get(pipe, 0.0) + float(amount)
        return records, links, work, transport_work

    cohort = {}
    if defer:
        launch_by_rank = {info["rank"]: info for info in infos}
        parents, starts = {}, {}
        remaining_starts = [len(infos)]
        for info in infos:
            name = info["name"]
            parent = engine.future(name + ":program_complete")
            parents[name] = parent
            starts[name] = info["ready"]
            records, links, work, transport_work = _declared_metrics(info)
            cohort[info["rank"]] = {
                "parent": parent, "start": info["ready"],
                "operators": {}, "phases": {},
                "handles": {"deferred": True},
                "work": work,
                "link_by_phase": {
                    phase: {str(pipe): float(amount)
                            for pipe, amount in values.items()}
                    for phase, values in links.items()},
                "transport": records,
                "transport_work": transport_work,
            }

            def begin(_event, item=info, promise=parent):
                rank_engine = _RankEngine(engine, item["resource_names"])
                finish, _handles = item["program"].start_with_handles(
                    rank_engine, item["name"], item["ready"],
                    launch=item["view"], send_placeholders=placeholders,
                    transport=item.get("transport"))
                engine.submit(item["name"] + ":parent_finish", after=(finish,),
                              owner=owner,
                              on_complete=lambda _e: engine.resolve(promise))
                remaining_starts[0] -= 1
                if remaining_starts[0] == 0:
                    # The handles are now owned by the real receive/retire
                    # events; do not retain the per-cohort sentinel registry.
                    placeholders.clear()

            engine.submit(name + ":begin", after=(info["ready"],), owner=owner,
                          on_complete=begin)
        parents = {names_by_rank[rank]: item["parent"]
                   for rank, item in cohort.items()}
        starts = {names_by_rank[rank]: item["start"]
                  for rank, item in cohort.items()}
        operators = {names_by_rank[rank]: item["operators"]
                     for rank, item in cohort.items()}
        handles = {names_by_rank[rank]: item["handles"]
                   for rank, item in cohort.items()}
        phases_by_rank = {names_by_rank[rank]: item["phases"]
                          for rank, item in cohort.items()}
        work_by_rank = {names_by_rank[rank]: item["work"]
                        for rank, item in cohort.items()}
        link_by_phase_by_rank = {
            names_by_rank[rank]: item["link_by_phase"]
            for rank, item in cohort.items()}
        transport_by_rank = {
            names_by_rank[rank]: item["transport"]
            for rank, item in cohort.items()}
        transport_work_by_rank = {
            names_by_rank[rank]: item["transport_work"]
            for rank, item in cohort.items()}
        transport_tx = sum(
            float(phase.get("tx_bytes", 0.0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_rx = sum(
            float(phase.get("rx_bytes", 0.0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_frames = sum(
            int(phase.get("frames", 0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_edges = sum(
            int(phase.get("edges", 0))
            for details in transport_by_rank.values()
            for phase in details.values())
        join = engine.submit(f"{name_prefix}:join", after=tuple(parents.values()),
                             owner=owner)
        return {
            "schema": "mindie-sim/native-cohort/3",
            "finishes": parents, "parents": parents, "join": join,
            "starts": starts, "operators": operators, "handles": handles,
            "phases_by_rank": phases_by_rank, "work_by_rank": work_by_rank,
            "link_by_phase_by_rank": link_by_phase_by_rank,
            "transport_by_rank": transport_by_rank,
            "transport_work_by_rank": transport_work_by_rank,
            "transport_tx_bytes": transport_tx,
            "transport_rx_bytes": transport_rx,
            "transport_frames": transport_frames,
            "transport_edges": transport_edges,
            "directed_edges": sum(
                int(value.get("edges", 0))
                for details in transport_by_rank.values()
                for phase, value in details.items()
                if isinstance(value, Mapping) and not value.get("count")),
            "directed_frames": transport_frames,
            "rank_indices": {names_by_rank[rank]: rank for rank in cohort},
            "payload_basis": {
                names_by_rank[rank]: str(
                    getattr(launch_by_rank[rank]["program"].program,
                            "payload_basis_for",
                            lambda _r, info=launch_by_rank[rank]:
                            info["payload_basis"])(rank))
                for rank in cohort},
            "payload_packet_bytes": {
                names_by_rank[rank]: int(
                    getattr(launch_by_rank[rank]["program"].program, "view", None).payload_packet_bytes
                    if getattr(launch_by_rank[rank]["program"].program, "view", None) is not None else 0)
                for rank in cohort},
            "payload_frames": {
                names_by_rank[rank]: dict(
                    getattr(launch_by_rank[rank]["program"].program,
                            "frames_for", lambda _r: ())(rank))
                for rank in cohort},
            "protocol_edges": sum(
                1 for key in placeholders
                if key[2] not in {"payload_exchange"}),
            "unresolved_edges": len(unresolved),
            "partial": tuple(partial),
        }
    for info in infos:
        rank_engine = _RankEngine(engine, info["resource_names"])
        parent, launch_handles = info["program"].start_with_handles(
            rank_engine, info["name"], info["ready"],
            launch=info["view"], send_placeholders=placeholders,
            transport=info.get("transport"))
        op_id = info.get("operator_id", "mc2")
        operators = dict(launch_handles.get("operators") or {})
        if op_id != "mc2" and "mc2" in operators:
            operators = {op_id: operators["mc2"]}
        inner = getattr(info["program"], "program", None)
        work_for_rank = getattr(inner, "work_for_rank", None)
        work = (tuple(work_for_rank(info["rank"]))
                if callable(work_for_rank)
                else tuple(getattr(info["program"], "work", ()) or ()))
        links_for_rank = getattr(inner, "link_by_phase_for", None)
        links = (dict(links_for_rank(info["rank"]))
                 if callable(links_for_rank)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        cohort[info["rank"]] = {
            "parent": parent,
            "start": launch_handles.get("start", info["ready"]),
            "operators": operators,
            "phases": dict(launch_handles.get("phase_complete") or {}),
            "handles": launch_handles,
            "work": work,
            "link_by_phase": {
                phase: {str(pipe): float(amount) for pipe, amount in values.items()}
                for phase, values in links.items()},
            "transport": dict(launch_handles.get("transport") or {}),
            "transport_work": dict(launch_handles.get("transport_work") or {}),
        }
    parents = {names_by_rank[rank]: item["parent"]
               for rank, item in cohort.items()}
    starts = {names_by_rank[rank]: item["start"]
              for rank, item in cohort.items()}
    operators = {names_by_rank[rank]: item["operators"]
                 for rank, item in cohort.items()}
    handles = {names_by_rank[rank]: item["handles"]
               for rank, item in cohort.items()}
    phases_by_rank = {names_by_rank[rank]: item["phases"]
                      for rank, item in cohort.items()}
    work_by_rank = {names_by_rank[rank]: item["work"]
                    for rank, item in cohort.items()}
    link_by_phase_by_rank = {
        names_by_rank[rank]: item["link_by_phase"]
        for rank, item in cohort.items()}
    transport_by_rank = {
        names_by_rank[rank]: item["transport"]
        for rank, item in cohort.items()}
    transport_work_by_rank = {
        names_by_rank[rank]: item["transport_work"]
        for rank, item in cohort.items()}
    transport_tx = sum(
        float(phase.get("tx_bytes", 0.0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_rx = sum(
        float(phase.get("rx_bytes", 0.0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_frames = sum(
        int(phase.get("frames", 0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_edges = sum(
        int(phase.get("edges", 0))
        for details in transport_by_rank.values()
        for phase in details.values())
    join = engine.submit(f"{name_prefix}:join", after=tuple(parents.values()),
                         owner=owner)
    return {
        "schema": "mindie-sim/native-cohort/3",
        "finishes": parents, "parents": parents, "join": join,
        "starts": starts, "operators": operators, "handles": handles,
        "phases_by_rank": phases_by_rank, "work_by_rank": work_by_rank,
        "link_by_phase_by_rank": link_by_phase_by_rank,
        "transport_by_rank": transport_by_rank,
        "transport_work_by_rank": transport_work_by_rank,
        "transport_tx_bytes": transport_tx,
        "transport_rx_bytes": transport_rx,
        "transport_frames": transport_frames,
        "transport_edges": transport_edges,
        "by_rank": {rank: dict(item) for rank, item in cohort.items()},
        "names_by_rank": dict(names_by_rank),
        "phases": tuple(phases or ()), "link_phases": link_phases,
        "fullmesh_phases": fullmesh_by_rank,
        "payload_frames": {
            info["name"]: dict(info.get("payload_frames") or {})
            for info in infos},
        "payload_basis": {
            info["name"]: str(info.get("payload_basis") or "unknown")
            for info in infos},
        "payload_packet_bytes": payload_packet_by_rank,
        "partial": tuple(partial),
        "protocol_edges": protocol_edges,
        "directed_edges": directed_edges,
        "directed_frames": directed_frames,
        "unresolved_peers": tuple(sorted({peer for _, _, peer in unresolved})),
        "unresolved_edges": len(unresolved),
        "rank_indices": {info["name"]: info["rank"] for info in infos},
    }


class Execution:
    # Identical private schedules replay across ranks and later layers.
    # Experimental: off in the shipped default. Opt-in only; no shape or input auto-enable.
    schedule_replay = False
    _isolated_scripts: dict = {}

    def __init__(
        self, case: CaseSpec, cost: CostModel, *, recipes: RecipeBook | None = None,
        submission: SubmissionPolicy | None = None, trace_limit: int = 0,
        engine: Engine | None = None, namespace: str = "",
        include_submission: bool = True, on_operator=None,
        include_serving_io: bool = False,
        client_tails=None, memory_policy=None,
    ):
        self.case, self.cost = case, cost
        self.namespace = namespace
        self.include_submission = include_submission
        self.on_operator = on_operator
        self.recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        from .invocation_binding import InvocationPolicy
        self.invocations = InvocationPolicy(case.runtime.kernels.get("invocations", ()))
        from .service_regimes import RegimePolicy
        self.regimes = RegimePolicy(case.runtime.kernels.get("service_regimes", ()))
        self.submission = submission or SubmissionPolicy(**case.runtime.submission)
        self.submission_source = "api.submission" if submission is not None else "runtime.submission"
        self.serving_io = ServingIO(case.runtime.serving_io if include_serving_io else {},
                                    namespace, case.runtime.parallel.dp, client_tails)
        from .offload import offload_resources, resolve_case_offload_policy
        self.offload_policy = resolve_case_offload_policy(case, memory_policy)
        self._offload_memory = None
        self.step_summary = StepSummary()
        self.stage_names: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_dies: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.streams: dict[tuple[int, int, str], Event] = {}
        self.hosts: dict[tuple[int, int], Event] = {}
        self.submission_counts = {"eager_invocations": 0, "graph_replays": 0}
        self.prepared = {}
        self.schedules = {}
        resources, pools, pool_units = [], {}, {}
        resources.extend(self.serving_io.resources())
        resources.extend(offload_resources(
            self.offload_policy, namespace, case.runtime.parallel.dp, case.runtime.parallel.pp))
        p = case.runtime.parallel
        width = p.tp * p.pcp
        for replica in range(p.dp):
            resources.append(Resource(f"{namespace}scheduler@r{replica}", 1, "ms", "fifo"))
            for stage in range(p.pp):
                start = (replica * p.pp + stage) * width
                dies = tuple(case.machine.dies[start:start + width])
                if len(dies) != width:
                    raise ModelingError("placement_exceeds_physical_devices")
                geometry = _execution_geometry(dies[0])
                if any(_execution_geometry(die) != geometry for die in dies[1:]):
                    raise ModelingError(
                        "representative_shard_requires_matching_resource_geometry:"
                        f"replica={replica}:stage={stage};"
                        "heterogeneous_owners_require_per_rank_execution")
                self.stage_dies[replica, stage] = dies
                roles = dict(dies[0].rates)
                work_units = {**NATIVE_WORK_UNITS, **dict(dies[0].work_units)}
                rates = {role: (min(dict(d.rates)[role] for d in dies) / 1000,
                                work_units.get(role, "work")) for role in roles}
                rates.update({"cube": (1, "peak_ms"), "host": (1, "ms")})
                names = {}
                for kind, (rate, unit) in rates.items():
                    label = {"cube": "matrix", "hbm": "memory"}.get(kind, kind)
                    name = f"{namespace}{label}@r{replica}:s{stage}"
                    resources.append(Resource(
                        name, rate, unit, "fifo" if kind == "host" else "fair",
                        dies[0].units(kind)))
                    names[kind] = name
                self.stage_names[replica, stage] = names
                for pool in dict(dies[0].buffers):
                    key = f"{namespace}{pool}@r{replica}:s{stage}"
                    pools[key] = min(dict(d.buffers)[pool] for d in dies)
                    pool_units[key] = dies[0].pool_units(pool)
        self.topology = Topology(case.machine, p, namespace)
        resources.extend(self.topology.resources())
        if engine is not None:
            engine.extend(resources, pools, pool_units)
        self.engine = engine if engine is not None else Engine(
            resources, pools=pools, pool_units=pool_units, trace_limit=trace_limit)
        self.resource_rates = ResourceRates(self.engine.resources)
        # Hardware bindings are stage properties. Thousands of pending kernels
        # retain their own work/readiness, but share these immutable inputs.
        self.stage_geometry = {key: Geometry.from_devices(dies)
                               for key, dies in self.stage_dies.items()}
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self.work = dict.fromkeys(("flops", "vector_ops", "hbm_bytes", "link_bytes"), 0.0)
        self.conflicts = 0
        self.recipes_used: dict[str, int] = {}
        self._collectives = None
        self.native_group_joins: dict = {}
        # Bounded launch-local evidence: ranks, actual frames/edges, the same
        # per-rank link ledger and the work really submitted by the program.
        self.native_group_launches: dict = {}
        self._program_cache: dict = {}
        self._prepare_cache: dict = {}
        self._service_identity_cache: dict = {}
        self.replay_counts = {"record": 0, "shadow": 0, "replay": 0, "real": 0}
        self._trace_key = None
        self._trace_shadows: list = []
        metadata = getattr(case.runtime, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = dict(metadata)
        self.readiness_report = metadata.get("collective_readiness")
        self.readiness_observations: list = []
        self.readiness_unresolved: list = []
        # Joint-collective observation is an opt-in bounded trace.  It retains
        # the first occurrence of each distinct joint collective, not per-step
        # history, and copies numeric boundaries rather than live Event handles.
        self._joint_collective_limit = max(0, int(trace_limit))
        self._joint_collective_trace = (
            {} if self._joint_collective_limit else None)
        self.joint_collective_stats = ({"observations": 0, "retained": 0,
                                        "max_ready_spread_ms": 0.0,
                                        "max_rendezvous_delay_ms": 0.0,
                                        "critical_rank": None}
                                       if self._joint_collective_trace is not None
                                       else None)
        self._joint_collective_ordinals: dict = {}
        from .participant_profile import normalise as _normalise_profile
        self.participant_profile = _normalise_profile(
            metadata.get("collective_participant_profile"))

    def add_group(self, works, release_ms, *, external_inputs=None, emitted_tokens=None):
        """Compile an EP cohort before releasing any of its collective events."""
        graphs = bind_cohort_traffic(self.case, {r: work.ops for r, work in works.items()})
        works = {r: replace(work, ops=graphs[r]) for r, work in works.items()}
        self._collectives = {}
        steps = {replica: self.add(
            work, replica, release_ms,
            external_inputs=(external_inputs or {}).get(replica),
            emitted_tokens=(emitted_tokens or {}).get(replica),
        ) for replica, work in works.items()}
        for key, participants in self._collectives.items():
            if len(participants) != len(works):
                raise ValueError(f"collective_participants_mismatch:{key}")
            if any(getattr(item.recipe, "algorithm", None) for item in participants):
                self._start_joint_collective(key, participants)
                continue
            native = any(getattr(item.recipe, "native", False)
                         for item in participants)
            if native:
                self._schedule_native_cohort(key, participants)
                continue
            if all(item.context.op.family == "mc2_transport"
                   for item in participants):
                programs = {item.replica: self.lower_program(
                    item.context, item.recipe) for item in participants}
                from .piecewise.compiler import NativeProgram
                if all(isinstance(program, NativeProgram)
                       and hasattr(program.program, "start_with_handles")
                       for program in programs.values()):
                    if self._start_native_cohort(key, participants, programs):
                        continue
            source_stage = int(dict(participants[0].context.op.metadata).get(
                "source_stage", participants[0].context.op.stage))
            local_ready_by_rank = {}
            for item in participants:
                local_ready_by_rank.update({
                    rank: item.ready for rank in
                    self.topology.ranks(item.replica, source_stage)})
            observation = self._begin_joint_observation(
                participants[0].context.op, key, participants,
                local_ready_by_rank, source_stage)
            dependencies = tuple(item.ready for item in participants)
            ready = self.engine.submit(f"group:{key}:ready", after=dependencies, owner="expert_group")
            # Aggregate recipes keep their common cohort barrier.  A native
            # peer program with named send phases is launched by
            # _start_native_cohort above; a flat prework ready is never used
            # as a phase send.
            completions, promises = [], []
            for item in participants:
                completions.append(self._start_program(
                    item.context, item.recipe, item.name, ready,
                    owner="expert_group", replica=item.replica))
                promises.append(item.completion)

            def release(event, events=tuple(promises), observation=observation,
                        local_ready_by_rank=local_ready_by_rank,
                        completions=tuple(completions), ready=ready,
                        participants=tuple(participants),
                        source_stage=source_stage):
                if observation is not None:
                    # The common ready event is the actual cohort barrier; the
                    # local events remain the genuine per-rank upstream bounds.
                    observation["barrier_ms"] = ready.time
                    observation["participants"] = []
                    for item, completion in zip(participants, completions):
                        ranks = self.topology.ranks(item.replica, source_stage)
                        local = {rank: local_ready_by_rank[rank].time
                                 for rank in ranks}
                        done = {rank: completion.time for rank in ranks}
                        observation["participants"].append({
                            "replica": item.replica, "ranks": list(ranks),
                            "ready_ms": local, "completion_ms": done,
                            "wait_ms": {rank: ready.time - local[rank]
                                        for rank in ranks},
                            "parent_complete_ms": event.time,
                        })
                        observation["ready_ms_by_rank"].update(local)
                        observation["completion_ms_by_rank"].update(done)
                        observation["parent_complete_ms"][item.replica] = event.time
                    self._finish_joint_collective(observation)
                for item in events:
                    self.engine.resolve(item)

            self.engine.submit(f"group:{key}", after=completions,
                               on_complete=release, owner="expert_group")
        self._collectives = None
        return steps

    def _begin_joint_observation(self, op, key, participants, ready_by_rank,
                                 source_stage):
        """Allocate one bounded record for an actual cohort rendezvous."""
        if self._joint_collective_trace is None:
            return None
        collective, expected_bytes, expected_dtype = _collective_native_basis(
            op, len(ready_by_rank))
        axis = dict(op.metadata).get("group_axis", "tp")
        occurrence = self._joint_collective_ordinals.get((op.family, axis), 0)
        self._joint_collective_ordinals[(op.family, axis)] = occurrence + 1
        link_by_rank, native_by_rank = {}, {}
        for item in participants:
            ranks = self.topology.ranks(item.replica, source_stage)
            _, native, _ = _collective_native_basis(
                item.context.op, len(ready_by_rank))
            value = (int(native) if native is not None
                     and native.denominator == 1 else None)
            link_value = item.context.native_work.get(
                "link", item.context.op.link_bytes)
            for rank in ranks:
                link_by_rank[rank] = link_value
                native_by_rank[rank] = value
        native_total = (sum(native_by_rank.values())
                        if native_by_rank
                        and all(value is not None
                                for value in native_by_rank.values())
                        else None)
        return {
            "op_id": op.op_id, "group": key,
            "family": op.family, "group_axis": axis,
            "collective": collective, "dtype": expected_dtype,
            "native_input_bytes": native_total,
            "native_input_bytes_by_rank": native_by_rank,
            "link_bytes": sum(link_by_rank.values()),
            "link_bytes_by_rank": link_by_rank,
            "source_stage": source_stage,
            "rank_count": len(ready_by_rank), "occurrence": occurrence,
            "participants": [], "ready_ms_by_rank": {},
            "completion_ms_by_rank": {}, "parent_complete_ms": {},
        }

    def _start_joint_collective(self, key, participants):
        """Bind one physical cohort while retaining each replica's readiness."""
        first = participants[0]
        algorithm = getattr(first.recipe, "algorithm", None)
        plan_options = collective_options(first.recipe, first.context.op)
        path_model = getattr(first.recipe, "path_model", None)
        op = first.context.op
        def declaration(item):
            return (
                dict(item.metadata).get("collective", item.family),
                dict(item.metadata).get("group_axis", "tp"),
                dict(item.metadata).get("source_stage", item.stage),
                dict(item.metadata).get("steps"),
                dict(item.metadata).get("peer_traffic"),
                dict(item.metadata).get("peer_volume_basis", "per_rank"),
            )
        for item in participants:
            if (getattr(item.recipe, "algorithm", None) != algorithm
                    or collective_options(item.recipe, item.context.op) != plan_options
                    or getattr(item.recipe, "path_model", None) != path_model
                    or declaration(item.context.op) != declaration(op)):
                raise ModelingError(f"joint_collective_declarations_differ:{key}")
            if any(value for role, value in item.context.native_work.items() if role != "link"):
                raise ModelingError(f"{algorithm}_collective_requires_transport_only_work")
        plan = self.topology.joint_plan(
            op, {item.replica: item.context.native_work.get("link", 0)
                 for item in participants}, algorithm=algorithm, **plan_options)
        plan.validate_path_model(path_model)
        source_stage = int(dict(op.metadata).get("source_stage", op.stage))
        ready_by_rank = {}
        for item in participants:
            timing = item.context.invocation_timing
            ready = item.ready
            if timing is not None and timing.startup_ms:
                ready = self.engine.submit(
                    item.name + ":device_startup", after=(ready,), latency=timing.startup_ms,
                    owner=f"replica:{item.replica}")
            ready_by_rank.update({rank: ready for rank in
                                  self.topology.ranks(item.replica, source_stage)})
        if self.readiness_report:
            from .collective_readiness import bind_pipeline_ready
            try:
                collective, expected_bytes, expected_dtype = _collective_native_basis(
                    op, len(ready_by_rank))
                context = bind_pipeline_ready(
                    self.readiness_report, ready_by_rank,
                    expected_collective=collective,
                    expected_bytes=(int(expected_bytes)
                                    if expected_bytes is not None
                                    and expected_bytes.denominator == 1
                                    else None),
                    expected_dtype=expected_dtype,
                    expected_group_size=len(ready_by_rank),
                    connection_id=dict(op.metadata).get("collective_connection_id"),
                    ordinal=dict(op.metadata).get("collective_call_ordinal"),
                    call_id=dict(op.metadata).get("collective_call_id"))
            except ModelingError as exc:
                self.readiness_unresolved.append({
                    "op_id": op.op_id, "group": key, "reason": str(exc)})
            else:
                self.readiness_observations.append({
                    "op_id": op.op_id, "group": key,
                    "ready_by_rank": tuple(sorted(ready_by_rank)),
                    "context": context,
                })
        completions = plan.start_participants(
            self.engine, first.name + ":cohort", ready_by_rank, owner="expert_group")
        observation = self._begin_joint_observation(
            op, key, participants, ready_by_rank, source_stage)
        pending = [len(participants)] if observation is not None else None
        for item in participants:
            ranks = self.topology.ranks(item.replica, source_stage)
            dependencies = [completions[rank] for rank in ranks]
            timing = item.context.invocation_timing
            minimum = max(item.context.minimum_ms, timing.minimum_ms if timing else 0)
            if minimum:
                dependencies.append(self.engine.submit(
                    item.name + ":observation", after=(item.ready,),
                    latency=minimum, owner=f"replica:{item.replica}"))

            def cohort_complete(event, item=item, ranks=ranks,
                                observation=observation, pending=pending):
                # The callback already runs on the physical cohort-complete
                # event.  Copy numeric boundaries; never retain Events.
                if observation is not None:
                    ready = {rank: ready_by_rank[rank].time for rank in ranks}
                    completion = {rank: completions[rank].time for rank in ranks}
                    observation["participants"].append({
                        "replica": item.replica, "ranks": list(ranks),
                        "ready_ms": ready, "completion_ms": completion,
                        "parent_complete_ms": event.time,
                    })
                    observation["ready_ms_by_rank"].update(ready)
                    observation["completion_ms_by_rank"].update(completion)
                    observation["parent_complete_ms"][item.replica] = event.time
                    pending[0] -= 1
                    if pending[0] == 0:
                        self._finish_joint_collective(observation)
                self.engine.resolve(item.completion)

            self.engine.submit(
                item.name + ":cohort_complete", after=dependencies,
                owner=f"replica:{item.replica}", on_complete=cohort_complete)
        return completions

    def _finish_joint_collective(self, observation: dict) -> None:
        """Reduce one completed cohort to bounded numeric observations."""
        observation["participants"].sort(key=lambda item: item["replica"])
        ready = [value for value in observation["ready_ms_by_rank"].values()
                 if value is not None]
        completion = [value for value in observation["completion_ms_by_rank"].values()
                      if value is not None]
        if ready:
            observation["rendezvous_ready_ms"] = observation.get(
                "barrier_ms", max(ready))
            observation["ready_spread_ms"] = max(ready) - min(ready)
            observation["critical_rank"] = max(
                observation["ready_ms_by_rank"],
                key=lambda rank: (observation["ready_ms_by_rank"][rank]
                                  if observation["ready_ms_by_rank"][rank] is not None
                                  else float("-inf")))
        if completion:
            observation["last_completion_ms"] = max(completion)
        from .participant_profile import select_call
        observation["source"] = select_call(
            self.participant_profile, primitive=observation["collective"],
            occurrence=observation["occurrence"],
            group_size=observation["rank_count"], dtype=observation["dtype"])
        stats = self.joint_collective_stats
        stats["observations"] += 1
        spread = observation.get("ready_spread_ms") or 0.0
        if spread > stats["max_ready_spread_ms"]:
            stats["max_ready_spread_ms"] = spread
            stats["max_rendezvous_delay_ms"] = spread
            stats["critical_rank"] = observation.get("critical_rank")
        trace = self._joint_collective_trace
        trace_key = (observation["op_id"], observation["group"])
        if trace_key in trace:
            trace[trace_key]["occurrences"] += 1
            trace[trace_key]["last_ready_spread_ms"] = spread
            trace[trace_key]["last_parent_complete_ms"] = max(
                observation["parent_complete_ms"].values(), default=None)
        else:
            observation["occurrences"] = 1
            if len(trace) >= self._joint_collective_limit:
                trace.pop(next(iter(trace)))
            trace[trace_key] = observation
        stats["retained"] = len(trace)

    def _outside_trace(self, engine) -> bool:
        prefix = engine._trace_prefix
        if prefix is None:
            return False
        return any(
            (task.remaining or task.allocation_demands)
            and not task.event.name.startswith(prefix)
            for task in engine._tasks.values()
        )

    def _join_isolated_replay(self, engine, script_key, cached, token, complete) -> bool:
        steps, peaks = cached
        pending = engine._script_pending
        if pending is None:
            if not engine.replay_ready():
                return False
            engine.queue_script(script_key, steps, [(token, complete)], peaks)
            return True
        if engine._script_pending_key != script_key:
            return False
        pending[1].append((token, complete))
        return True

    def _seal_trace(self, engine, complete) -> None:
        key = self._trace_key
        shadows = self._trace_shadows
        self._trace_key = None
        self._trace_shadows = []
        if engine._trace_prefix is None:
            engine.resolve(complete)
            return
        clean, steps, peaks = engine.end_trace()
        if key is not None and clean:
            Execution._isolated_scripts[key] = (steps, peaks)
            if len(Execution._isolated_scripts) > 1024:
                Execution._isolated_scripts.pop(next(iter(Execution._isolated_scripts)))
            # The recording rank already advanced time and its own counters.
            # Sibling ranks only receive the same increments.
            for token, shadow in shadows:
                self._apply_private_account(engine, steps, peaks, token)
                engine.resolve(shadow)
        else:
            for _token, shadow in shadows:
                engine.resolve(shadow)
        engine.resolve(complete)

    def _apply_private_account(self, engine, steps, peaks, token) -> None:
        for dt, busy, adds in steps:
            for name in busy:
                concrete = name.replace("@r*", token, 1)
                if concrete in engine.busy_ms:
                    engine.busy_ms[concrete] += dt
            for name, used in adds:
                concrete = name.replace("@r*", token, 1)
                if concrete in engine.serviced:
                    engine.serviced[concrete] += used
        for pool, unit, value in peaks:
            concrete = pool.replace("@r*", token, 1)
            if concrete not in engine.peak_allocated:
                continue
            pair = concrete, unit
            if value > engine.peak_unit_allocated.get(pair, 0.0):
                engine.peak_unit_allocated[pair] = value
            if value > engine.peak_allocated[concrete]:
                engine.peak_allocated[concrete] = value

    def _schedule_native_cohort(self, key, participants):
        """Defer the whole native cohort until its first genuine ready.

        The first callback prepares the bounded placeholder registry for this
        layer only, then each participant materializes its own program at its
        own ready Event.  No future layer contributes peer tasks before its
        first source is ready.
        """
        started = [False]
        def begin(_event):
            if started[0]:
                return
            started[0] = True
            programs = {item.replica: self.lower_program(
                item.context, item.recipe) for item in participants}
            from .piecewise.compiler import NativeProgram
            if not all(isinstance(program, NativeProgram)
                       and hasattr(program.program, "start_with_handles")
                       for program in programs.values()):
                raise ModelingError(
                    f"native_peer_plan_requires_executable_cohort:{key}")
            if not self._start_native_cohort(key, participants, programs,
                                             defer=True):
                raise ModelingError(
                    f"native_peer_plan_requires_executable_cohort:{key}")
        for item in participants:
            self.engine.submit(
                f"group:{key}:native_wait:r{item.replica}",
                after=(item.ready,), owner="expert_group",
                on_complete=begin)
        return True

    def _start_native_cohort(self, key, participants, programs, *, defer=False):
        """Two-phase native launch for one ordinary EP cohort."""
        source_stage = int(dict(participants[0].context.op.metadata).get(
            "source_stage", participants[0].context.op.stage))
        launches = []
        for item in participants:
            program = programs[item.replica]
            rank = self.participant_rank_index(item.replica, source_stage)
            if rank is None:
                return False
            launch = {
                "program": program,
                "name": f"group:{key}:r{item.replica}",
                "ready": item.ready,
                "rank_index": rank,
                "peer_count": int(getattr(program.program, "peer_count", 0)),
                "node_ranks": int(getattr(self.case.machine, "ranks_per_node", 0)),
                "resource_names": self.native_resource_names(
                    item.context, program, rank=rank,
                    replica=item.replica),
                "transport": self.native_transport(
                    item.context.op, item.replica),
            }
            launches.append(launch)
        result = launch_native_cohort(
            self.engine, launches, name_prefix=f"group:{key}",
            owner="expert_group", defer=defer)
        for item in participants:
            name = f"group:{key}:r{item.replica}"
            parent = result["parents"][name]
            self.engine.submit(
                name + ":release", after=(parent,),
                owner=f"replica:{item.replica}",
                on_complete=lambda _event, promise=item.completion:
                self.engine.resolve(promise))
        self.native_group_joins[key] = result["join"]
        raw = dict(participants[0].context.op.metadata).get("mc2_phase", {})
        payload_edges = sum(
            int(phases.get("payload_exchange", {}).get("edges", 0))
            for phases in result["transport_by_rank"].values())
        count_edges = sum(
            int(phases.get("count_status_exchange", {}).get("edges", 0))
            for phases in result["transport_by_rank"].values())
        payload_frames = sum(
            int(phases.get("payload_exchange", {}).get("frames", 0))
            for phases in result["transport_by_rank"].values())
        merged_work = {}
        for name, work in result["work_by_rank"].items():
            merged = {pipe: amount for pipe, amount in work
                      if "link" not in str(pipe).lower()}
            for pipe, amount in result["transport_work_by_rank"].get(
                    name, {}).items():
                merged[pipe] = merged.get(pipe, 0.0) + float(amount)
            merged_work[name] = tuple(sorted(merged.items()))
        self.native_group_launches[key] = {
            "op_id": key,
            "kind": str(raw.get("kind", "")),
            "launch_mode": "cohort",
            "ranks": tuple(sorted(result["rank_indices"].values())),
            "payload_basis": dict(result["payload_basis"]),
            "payload_packet_bytes": dict(result["payload_packet_bytes"]),
            "payload_frames": dict(result["payload_frames"]),
            "link_by_phase_by_rank": dict(result["link_by_phase_by_rank"]),
            "work_by_rank": merged_work,
            "transport_by_rank": {
                name: dict(phases)
                for name, phases in result["transport_by_rank"].items()},
            "transport_work_by_rank": {
                name: dict(work)
                for name, work in result["transport_work_by_rank"].items()},
            "transport_tx_bytes": float(result["transport_tx_bytes"]),
            "transport_rx_bytes": float(result["transport_rx_bytes"]),
            "transport_frames": int(result["transport_frames"]),
            "transport_edges": payload_edges,
            "total_edges": int(result["transport_edges"]),
            "payload_edges": payload_edges,
            "count_edges": count_edges,
            "directed_edges": (payload_edges or int(result["directed_edges"])),
            "directed_frames": (payload_frames
                                or int(result["directed_frames"])),
            "protocol_edges": int(result["protocol_edges"]),
            "unresolved_edges": int(result["unresolved_edges"]),
            "partial": tuple(result["partial"]),
        }
        return True

    def participant_peer_ready(self, replica, source_stage, ready_by_rank):
        """Actual peer ready Events for one participant, keyed by EP rank.

        These are the same shared-Engine handles exposed by
        ``CollectiveParticipant.ready``; no skew, timestamp or inferred wait is
        introduced.  A native peer program uses each handle only for its named
        peer receive/check/retire edge.
        """
        local_ranks = set(self.topology.ranks(replica, source_stage))
        for group in self.topology.groups("ep", replica, source_stage):
            if local_ranks & set(group):
                return {index: ready_by_rank[rank]
                        for index, rank in enumerate(group)
                        if rank not in local_ranks}
        return {rank: event for rank, event in ready_by_rank.items()
                if rank not in local_ranks}

    def participant_rank_index(self, replica, source_stage):
        """Local EP rank index for one participant (TP1/MC2 ingress)."""
        local_ranks = tuple(self.topology.ranks(replica, source_stage))
        for group in self.topology.groups("ep", replica, source_stage):
            for rank in local_ranks:
                if rank in group:
                    return group.index(rank)
        return None

    def _peer_ready_for(self, item, local_ready_by_rank, source_stage):
        return self.participant_peer_ready(
            item.replica, source_stage, local_ready_by_rank)

    def _topology_class_resource(self, rank, pipe):
        """Actual TX port for this rank/class, or None when not port-scoped."""
        rank = int(rank)
        kind = "roce" if pipe == "link_roce" else "hccs"
        links = list(getattr(self.topology, "links", {}).values())
        preferred = [link for link in links
                     if (kind == "roce") == ("roce" in link.name.lower()
                                             or "inter" in link.name.lower())]
        for link in preferred or links:
            if rank not in link.members:
                continue
            if link.semantics == "per_pair":
                peer = next((member for member in link.members
                             if member != rank), None)
                if peer is None:
                    continue
                return self.topology.resource(link, rank, peer)
            if link.semantics in {"per_port", "fully_connected", "mesh"}:
                return f"{self.topology.namespace}link:{link.name}:tx:port{rank}"
        return None

    def native_transport(self, op, replica):
        """Whole actual EP membership and source cells for this launch.

        TP8/DP1 has eight physical EP members even though the graph has one DP
        replica.  The replica owns the transport cells whose source rank is
        local; peer ranks are endpoints, not cloned expert/FFN compute.
        """
        metadata = dict(op.metadata)
        source_stage = int(metadata.get("source_stage", op.stage))
        local_ranks = tuple(self.topology.ranks(replica, source_stage))
        groups = tuple(self.topology.groups("ep", replica, source_stage))
        group = next(
            (members for members in groups
             if set(members) & set(local_ranks)), ())
        if not group:
            group = local_ranks
        sources = tuple(index for index, rank in enumerate(group)
                        if rank in local_ranks)
        destination_memory = {
            int(rank): self.destination_window_resource(int(rank), source_stage)
            for rank in group}
        return {"group": tuple(group), "sources": sources,
                "route": self.topology.route_message,
                "destination_memory": destination_memory}

    def native_resource_names(self, context, program, *, rank=None,
                              replica=None):
        """Public launch-resource binding for canonical cohort drivers."""
        return self._native_resource_names(
            context, program, rank=rank, replica=replica)

    def _native_resource_names(self, context, program, *, rank=None,
                               replica=None):
        """Bind native pipe names to this rank's physical stage resources."""
        inner = getattr(program, "program", program)
        native_names = getattr(inner, "names", {}) or {}
        available = dict(getattr(context, "resource_names", {}) or {})
        bound = {str(role): str(physical)
                 for role, physical in available.items()}
        for pipe, native_name in native_names.items():
            physical = available.get(pipe) or available.get(native_name)
            if rank is not None and pipe in {"link", "link_hccs", "link_roce"}:
                route = self._topology_class_resource(rank, str(pipe))
                if route is not None:
                    physical = route
            if physical is not None:
                bound[str(pipe)] = str(physical)
        if rank is not None:
            # A source phase names the fabric class directly even when the
            # stage resource table exposes only per-rank topology ports.
            for pipe in ("link", "link_hccs", "link_roce"):
                if pipe in bound:
                    continue
                route = self._topology_class_resource(rank, pipe)
                if route is not None:
                    bound[pipe] = route
        return bound

    @staticmethod
    def _build_launch_view(inner, rank_index, *, node_ranks=None,
                           peer_classes=None, peer_weights=None,
                           intra_latency_ms=None, inter_latency_ms=None,
                           resource_names=None):
        """Immutable per-launch rank/topology view; cached program untouched."""
        from .piecewise.families_moe_mc2 import Mc2LaunchView
        peer_count = int(getattr(inner, "peer_count", 0) or 0)
        if peer_count <= 0 or rank_index is None:
            return None
        rank = int(rank_index)
        if not 0 <= rank < peer_count:
            raise ModelingError(f"native_launch_rank_out_of_range:{rank}")
        source = getattr(inner, "view", None)
        labels = tuple(item for item in range(peer_count) if item != rank)
        classes = tuple(peer_classes or ())
        weights = tuple(peer_weights or ())
        if source is not None and source.rank == rank:
            if not classes:
                classes = tuple(source.peer_classes)
            if not weights:
                weights = tuple(source.peer_weights)
        node = int(node_ranks if node_ranks is not None else
                   (getattr(source, "node_ranks", 0) if source else 0))
        intra = (getattr(source, "intra_latency_ms", 0)
                 if intra_latency_ms is None and source
                 else (intra_latency_ms or 0))
        inter = (getattr(source, "inter_latency_ms", 0)
                 if inter_latency_ms is None and source
                 else (inter_latency_ms or 0))
        return Mc2LaunchView(
            rank=rank, peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=node,
            intra_latency_ms=intra, inter_latency_ms=inter,
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(resource_names or {}).items())))

    @staticmethod
    def _phase_scope(program, mapping):
        """Normalise a flat peer map to MOE10's named-phase mapping."""
        if mapping is None:
            return None
        inner = getattr(program, "program", program)
        phases = getattr(inner, "phases", None)
        link_by_phase = getattr(inner, "link_by_phase", None)
        if not phases or not link_by_phase:
            return dict(mapping)
        names = {phase.name for phase in phases}
        if mapping and all(isinstance(key, str) and key in names
                           for key in mapping):
            return dict(mapping)
        peers = dict(mapping)
        return {phase.name: dict(peers) for phase in phases
                if phase.name in link_by_phase}

    def start_native_participant_with_handles(self, program, name, ready, *,
                                              owner="", local_ready=None,
                                              peer_ready=None, peer_send=None,
                                              rank_index=None, launch=None,
                                              source_send=None,
                                              send_placeholders=None,
                                              resource_names=None,
                                              transport=None):
        """Start a native peer program and return its actual phase handles.

        ``peer_ready`` and ``peer_send`` accept either the flat
        ``{ep_rank: Event}`` form (applied to every source link phase) or
        MOE10's exact ``{phase_name: {ep_rank: Event}}`` form.  The returned
        ``handles`` are the program's launch-local peer send/receive/check/
        retire and phase-complete Events on this same Engine.
        """
        from .piecewise.compiler import NativeProgram
        phase_ready = self._phase_scope(program, peer_ready)
        phase_send = self._phase_scope(program, peer_send)
        start = local_ready if local_ready is not None else ready
        if isinstance(program, NativeProgram):
            inner = program.program
            if launch is None and rank_index is not None:
                launch = Execution._build_launch_view(
                    inner, rank_index, resource_names=resource_names)
            if hasattr(inner, "start_with_handles"):
                return program.start_with_handles(
                    self.engine, name, start, peer_ready=phase_ready,
                    peer_send=phase_send, launch=launch,
                    source_send=source_send,
                    send_placeholders=send_placeholders,
                    transport=transport)
            if peer_send is not None:
                raise ModelingError(
                    "native_peer_send_requires_start_with_handles")
            return program.start(self.engine, name, start,
                                 peer_ready=phase_ready), {}
        event = self._start_engine_program(
            program, self.engine, name, ready, owner,
            local_ready=local_ready, peer_ready=peer_ready,
            peer_send=peer_send)
        return event, {}

    def start_native_participant(self, program, name, ready, *, owner="",
                                 local_ready=None, peer_ready=None,
                                 peer_send=None, rank_index=None,
                                 launch=None, source_send=None,
                                 send_placeholders=None, transport=None):
        """Start a lowered native program on this Execution's shared Engine.

        MOE10 calls this with the same Engine Events used by
        ``CollectiveParticipant``.  A ``NativeProgram`` starts at its own real
        ``local_ready`` and receives phase-scoped ``peer_ready``/``peer_send``
        for its named peer receive/check/retire edges.  Any non-native program
        keeps the supplied common ``ready`` barrier.
        """
        event, _handles = self.start_native_participant_with_handles(
            program, name, ready, owner=owner, local_ready=local_ready,
            peer_ready=peer_ready, peer_send=peer_send,
            rank_index=rank_index, launch=launch, source_send=source_send,
            send_placeholders=send_placeholders, transport=transport)
        return event

    @staticmethod
    def _start_engine_program(program, engine, name, ready, owner,
                              local_ready=None, peer_ready=None,
                              peer_send=None, rank_index=None, launch=None,
                              source_send=None, send_placeholders=None,
                              resource_names=None, transport=None):
        """Start one lowered program on the caller's shared Engine.

        A native peer program starts at its own real ready and receives the
        actual peer Events.  Every other program keeps the common cohort
        barrier, so ordinary aggregate recipes are unchanged.
        """
        from .piecewise.compiler import NativeProgram
        if isinstance(program, NativeProgram) and (
                peer_ready is not None or peer_send is not None):
            start = local_ready if local_ready is not None else ready
            inner = program.program
            if launch is None and rank_index is not None:
                launch = Execution._build_launch_view(
                    inner, rank_index, resource_names=resource_names)
            phase_ready = Execution._phase_scope(program, peer_ready)
            phase_send = Execution._phase_scope(program, peer_send)
            if peer_send is not None:
                if not hasattr(program.program, "start_with_handles"):
                    raise ModelingError(
                        "native_peer_send_requires_start_with_handles")
                event, _handles = program.start_with_handles(
                    engine, name, start, peer_ready=phase_ready,
                    peer_send=phase_send, launch=launch,
                    source_send=source_send,
                    send_placeholders=send_placeholders,
                    transport=transport)
                return event
            return program.start(engine, name, start, peer_ready=phase_ready,
                                 launch=launch, source_send=source_send,
                                 transport=transport)
        start = ready
        if local_ready is not None and local_ready is not ready:
            start = engine.submit(name + ":cohort_gate", after=(ready,))
        if isinstance(program, NativeProgram):
            return program.start(engine, name, start, transport=transport)
        return program.start(engine, name, start, owner=owner)

    def _record_single_native(self, context, program, replica, name,
                              transport=None):
        """Retain an actual non-cohort native launch as public evidence.

        A representative single-replica graph still executes its native peer
        program on this Execution.  It is a real launch, but it has no
        placeholder cohort, so it is reported separately from a complete
        cross-DP rendezvous instead of being labelled by family.
        """
        from .piecewise.compiler import NativeProgram
        if not isinstance(program, NativeProgram):
            return
        inner = program.program
        if not hasattr(inner, "start_with_handles"):
            return
        key = context.op.op_id
        if key in self.native_group_launches:
            return
        meta = dict(context.op.metadata)
        raw = meta.get("mc2_phase", {})
        source_stage = int(meta.get("source_stage", context.op.stage))
        rank = (self.participant_rank_index(replica, source_stage)
                if replica is not None else None)
        if rank is None:
            rank = 0
        frames_for = getattr(inner, "frames_for", None)
        frames = (tuple(frames_for(rank)) if callable(frames_for) else
                  tuple(getattr(getattr(inner, "view", None),
                                "payload_frames", ()) or ()))
        basis_for = getattr(inner, "payload_basis_for", None)
        basis = (basis_for(rank) if callable(basis_for) else
                 str(getattr(getattr(inner, "view", None),
                             "payload_basis", "unknown")))
        packet = int(getattr(getattr(inner, "view", None),
                            "payload_packet_bytes", 0) or 0)
        links_for = getattr(inner, "link_by_phase_for", None)
        links = (links_for(rank) if callable(links_for)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        work_for = getattr(inner, "work_for_rank", None)
        work = dict(work_for(rank) if callable(work_for)
                    else tuple(getattr(program, "work", ()) or ()))
        transport_details = {}
        transport_work = {}
        if transport is not None and hasattr(inner, "transport_account"):
            payload_account, count_account = inner.transport_account(transport)
            has_count = any(getattr(item, "name", "") == "count_status_exchange"
                            for item in getattr(inner, "phases", ()))
            phases = (("payload_exchange", payload_account),
                      ("count_status_exchange", count_account))
            for phase_name, account in phases:
                if phase_name == "count_status_exchange" and not has_count:
                    continue
                if not account.get("edges"):
                    continue
                transport_details[phase_name] = {
                    "frames": int(account.get("frames", 0)),
                    "edges": int(account.get("edges", 0)),
                    "tx_bytes": float(account.get("tx_bytes", 0.0)),
                    "rx_bytes": float(account.get("rx_bytes", 0.0)),
                    "basis": str(getattr(inner, "transport_basis", basis)),
                    "group": list(account.get("group") or ()),
                    "sources": list(account.get("sources") or ()),
                    "count": phase_name == "count_status_exchange",
                }
                for pipe, amount in account.get("work", {}).items():
                    transport_work[pipe] = (
                        transport_work.get(pipe, 0.0) + float(amount))
        # Replace the logical per-rank link estimate with the actual coupled
        # TX/RX hop demand when a real route was scheduled.
        if transport_work:
            work = {pipe: amount for pipe, amount in work.items()
                    if "link" not in str(pipe).lower()}
            for pipe, amount in transport_work.items():
                work[pipe] = work.get(pipe, 0.0) + amount
        total_edges = sum(item["edges"] for item in transport_details.values())
        transport_frames = sum(item["frames"] for item in transport_details.values())
        payload_edges = int(transport_details.get(
            "payload_exchange", {}).get("edges", 0))
        count_edges = int(transport_details.get(
            "count_status_exchange", {}).get("edges", 0))
        self.native_group_launches[key] = {
            "op_id": key, "kind": str(raw.get("kind", "")),
            "launch_mode": "single_program",
            "ranks": (rank,),
            "payload_basis": {name: str(basis)},
            "payload_packet_bytes": {name: packet},
            "payload_frames": {name: dict(frames)},
            "link_by_phase_by_rank": {
                name: {phase: {str(pipe): float(amount)
                               for pipe, amount in values.items()}
                       for phase, values in links.items()}},
            "work_by_rank": {name: tuple(sorted(work.items()))},
            "transport_by_rank": {name: transport_details},
            "transport_work_by_rank": {name: transport_work},
            "transport_tx_bytes": sum(
                item["tx_bytes"] for item in transport_details.values()),
            "transport_rx_bytes": sum(
                item["rx_bytes"] for item in transport_details.values()),
            "transport_frames": transport_frames,
            "transport_edges": payload_edges,
            "total_edges": total_edges,
            "payload_edges": payload_edges,
            "count_edges": count_edges,
            "directed_edges": (transport_details.get(
                "payload_exchange", {}).get("edges") or len(frames)),
            "directed_frames": (transport_details.get(
                "payload_exchange", {}).get("frames")
                or sum(int(count) for _peer, count in frames)),
            "protocol_edges": 0,
            "unresolved_edges": 0,
            "partial": (),
        }

    def _start_program(self, context, recipe, name, ready, *, owner,
                       local_ready=None, peer_ready=None, rank_index=None,
                       launch=None, source_send=None,
                       send_placeholders=None, replica=None):
        """Keep per-core programs resident only while their operator can run."""
        engine = self.engine
        complete = engine.future(name + ":program_complete")

        def begin(_):
            op = context.op
            transport = (
                self.native_transport(op, replica)
                if replica is not None and getattr(recipe, "native", False)
                else None)
            live_rates = tuple(
                (role, context.rates.get(res_name))
                for role, res_name in sorted(context.resource_names.items())
                if res_name in context.rates
            )
            inv_t = context.invocation_timing
            prog_key = (
                owner, _freeze_key(recipe), op.family, op.recipe, op.phase, op.stage,
                op.dtype, op.layout, op.semantic_shape, op.local_shape, op.padded_shape,
                op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
                op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
                op.link_scope, op.source_boundary, op.returns,
                _op_meta_key(op), context.setup_ms, context.minimum_ms,
                (inv_t.startup_ms, inv_t.minimum_ms) if inv_t is not None else None,
                _freeze_key(context.joint_native_service),
                tuple(sorted(context.work.items())),
                live_rates,
                context.group_buffers,
            )
            token = _rank_token(context)
            replayable = (
                Execution.schedule_replay and token is not None
                and not op.link_bytes and not getattr(recipe, "algorithm", None)
                and dict(op.metadata).get("group_axis") != "ep"
            )
            if replayable:
                script_key = _isolated_script_key(prog_key, context, token)
                # Later layers run one rank for real so latency fill is recomputed
                # at the current virtual time. Replaying stored steps is not
                # bit-exact once absolute time changes a latency remainder.
                if (engine._trace_prefix is not None and engine._trace_private
                        and engine._trace_clean and engine._trace_token != token
                        and not self._outside_trace(engine)):
                    self._trace_shadows.append((token, complete))
                    self.replay_counts["shadow"] += 1
                    return
                if engine._trace_prefix is None and engine.replay_ready():
                    engine.begin_trace(name, token)
                    self._trace_key = script_key
                    self._trace_shadows = []
                    self.replay_counts["record"] += 1
                else:
                    self.replay_counts["real"] += 1
            else:
                self.replay_counts["real"] += 1

            recording = engine._trace_prefix == name

            def finished(_event=None):
                if recording and engine._trace_prefix == name:
                    self._seal_trace(engine, complete)
                else:
                    engine.resolve(complete)

            cached_prog = self._program_cache.get(prog_key)
            if cached_prog is not None:
                program, mapped_recs, svc_calls = cached_prog
                native_resources = self._native_resource_names(context, program)
                if mapped_recs and context.record_mapping is not None:
                    for mrec in mapped_recs:
                        context.record_mapping(deepcopy(mrec))
                if svc_calls and context.record_service is not None:
                    for sargs, skwargs in svc_calls:
                        context.record_service(*deepcopy(sargs), **deepcopy(skwargs))
                self._record_single_native(
                    context, program, replica, name, transport=transport)
                finish = self._start_engine_program(
                    program, engine, name, ready, owner,
                    local_ready=local_ready, peer_ready=peer_ready,
                    rank_index=rank_index, launch=launch,
                    source_send=source_send,
                    send_placeholders=send_placeholders,
                    resource_names=native_resources, transport=transport)
                engine.submit(name + ":resolve", after=(finish,), owner=owner,
                              on_complete=finished)
                return

            mapped_list = []
            svc_list = []
            orig_rec = context.record_mapping
            orig_svc = context.record_service

            def cap_map(rec):
                mapped_list.append(deepcopy(rec))
                if orig_rec is not None:
                    orig_rec(rec)

            def cap_svc(*args, **kwargs):
                svc_list.append((deepcopy(args), deepcopy(kwargs)))
                if orig_svc is not None:
                    orig_svc(*args, **kwargs)

            run_ctx = replace(
                context,
                record_mapping=cap_map if orig_rec is not None else None,
                record_service=cap_svc if orig_svc is not None else None,
            )
            program = self.lower_program(run_ctx, recipe)
            native_resources = self._native_resource_names(context, program)
            if len(self._program_cache) >= 768:
                self._program_cache.pop(next(iter(self._program_cache)))
            self._program_cache[prog_key] = (program, tuple(mapped_list), tuple(svc_list))
            self._record_single_native(
                context, program, replica, name, transport=transport)
            finish = self._start_engine_program(
                program, engine, name, ready, owner,
                local_ready=local_ready, peer_ready=peer_ready,
                rank_index=rank_index, launch=launch,
                source_send=source_send,
                send_placeholders=send_placeholders,
                resource_names=native_resources, transport=transport)
            engine.submit(name + ":resolve", after=(finish,), owner=owner,
                          on_complete=finished)

        # Host submission and graph dependencies are already declared. Expanding
        # every future layer's per-core pipelines here multiplies inactive state
        # by the device's core count without changing any physical work.
        begin_after = local_ready if local_ready is not None else ready
        engine.submit(name + ":begin", after=(begin_after,), owner=owner,
                      on_complete=begin)
        return complete

    @staticmethod
    def lower_program(context, recipe):
        recipe = context.program_recipe or recipe
        if context.joint_native_service is not None:
            from .program import KernelProgram, Stage
            return KernelProgram((Stage("joint_native_service", context.work),))
        from .invocation_binding import lower_invocation
        return lower_invocation(recipe, context)

    def prepare(self, op, replica=0):
        geom = self.stage_geometry[replica, op.stage]
        prep_key = (
            id(self.recipes), _freeze_key(geom), op.stage, op.family, op.recipe, op.phase,
            op.dtype, op.layout, op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
            _op_meta_key(op),
        )
        if prep_key in self._prepare_cache:
            cached_op, meta_diff, recipe = self._prepare_cache[prep_key]
            meta = op.metadata
            if meta_diff is not None:
                patched = _meta_patch(dict(op.metadata), meta_diff)
                meta = tuple(sorted(patched.items()))
            return replace(
                cached_op, op_id=op.op_id, layer=op.layer, deps=op.deps,
                requests=op.requests, stream=op.stream, notes=op.notes,
                metadata=meta,
            ), recipe
        res_op, recipe = self.recipes.prepare(op, geom)
        orig_meta = dict(op.metadata)
        res_meta = dict(res_op.metadata)
        meta_diff = _meta_diff(orig_meta, res_meta) if orig_meta != res_meta else None
        if len(self._prepare_cache) >= 512:
            self._prepare_cache.pop(next(iter(self._prepare_cache)))
        self._prepare_cache[prep_key] = (res_op, meta_diff, recipe)
        return res_op, recipe

    def destination_window_resource(self, destination_rank: int, stage: int) -> str | None:
        """Return the declared resident memory resource for one represented rank.

        The topology defines which physical rank is the first die of each
        ordinary execution replica. Only those declared representative ranks
        own a resident ``memory@rN:sS`` resource; other physical EP endpoints are
        network-only placeholders and are not given invented local capacity.
        """
        for (replica, owner_stage), names in self.stage_names.items():
            if owner_stage != stage:
                continue
            ranks = self.topology.ranks(replica, stage)
            if ranks and ranks[0] == destination_rank:
                return names.get("hbm")
        return None

    def _stage_context(self, key, dies):
        replica, stage = key
        names = self.stage_names[key]
        return RecipeContext(
            None, {}, names, self.resource_rates,
            dies[0].aic_count,
            pool_names={pool: f"{self.namespace}{pool}@r{replica}:s{stage}"
                        for pool in dict(dies[0].buffers)},
            resource_units={role: dies[0].units(role) for role in names},
            compute_peaks={dtype: min(d.peak(dtype) for d in dies)
                           for dtype, _ in dies[0].compute_peaks},
            pool_units={pool: dies[0].pool_units(pool)
                        for pool in dict(dies[0].buffers)},
            pool_capacities={pool: min(dict(d.buffers)[pool] for d in dies)
                             for pool in dict(dies[0].buffers)},
            record_mapping=self.cost.record_mapping,
            service_modes={mode: (role, min(dict(d.service_modes)[mode][1] for d in dies))
                           for mode, (role, _) in dies[0].service_modes},
            record_service=self.cost.record_service,
            work_units={**NATIVE_WORK_UNITS, **dict(dies[0].work_units)},
            store_queues={name: dict(queue) for name, queue in dies[0].store_queues
                          if all(dict(d.store_queues).get(name) == queue for d in dies)},
        )

    def _context(self, op: Any, replica: int, scope: str, *, recipe=None) -> RecipeContext:
        from .services import service_identity
        from .kernel_groups import declaration
        cost = self.cost.price(op, scope)
        dies = self.stage_dies[replica, op.stage]
        template = self.stage_contexts[replica, op.stage]
        peaks = template.compute_peaks
        additional = {dtype for dtype, _ in op.matrix_flops} - peaks.keys()
        if additional:
            peaks = {**peaks, **{dtype: min(d.peak(dtype) for d in dies) for dtype in additional}}
        meta = dict(op.metadata)
        sid_key = (
            replica, op.stage, dies[0].rank, dies[0].compute_units,
            _freeze_key(dict(dies[0].resource_units)),
            op.family, op.recipe, op.dtype, op.layout, scope,
            meta.get("kernel"), meta.get("kernel_name"),
            _freeze_key(meta.get("implementation")),
            _freeze_key(meta.get("service_axes")),
            bool(meta.get("hccl_selection")),
            _freeze_key(meta),
        )
        sid = self._service_identity_cache.get(sid_key)
        if sid is None:
            sid = service_identity(self.case, op, scope, device=dies[0])
            if len(self._service_identity_cache) >= 1024:
                self._service_identity_cache.pop(next(iter(self._service_identity_cache)))
            self._service_identity_cache[sid_key] = sid
        context = replace(template, op=op, native_work=cost.work, compute_peaks=peaks,
            service_identity=sid,
            service_boundary="resource_phase" if declaration(op) else "device_kernel")
        demands, frames, setup = {}, {}, 0.0
        if "peer_traffic" in dict(op.metadata) and not cost.work.get("link", 0):
            # A zero sender still participates and receives; validate its row.
            self.topology.accounting(op, replica, volume=cost.work.get("link", 0))
        for resource, amount in cost.work.items():
            if resource == "link":
                routed = self.topology.accounting(op, replica, volume=amount)
                demands.update(routed.wire_by_resource)
                frames.update(routed.transaction_frames)
                # Envelope transport honors the declared round count. Explicit
                # collective programs own their hop arrivals and ignore setup.
                setup = routed.setup_ms
                continue
            if resource == "cube":
                portions = op.matrix_flops or ((op.dtype or self.case.model.compute_dtype, amount),)
                bound = context.bind(Work(matrix=dict(portions)))
            else:
                bound = context.bind(Work(resources={resource: amount}))
            for name, value in bound.items():
                demands[name] = demands.get(name, 0) + value
        measured = cost.timing.duration_ms if cost.timing is not None else 0
        physical = max((value / self.engine.resources[key].rate for key, value in demands.items()), default=0)
        # Context binding also serves independently constructed source phases;
        # native binding must not require an unrelated source-dispatch registration.
        algorithm = getattr(recipe, "algorithm", "ring")
        context = replace(context, work=demands, minimum_ms=measured, setup_ms=setup,
                          transaction_frames=frames,
                          collective_plan=lambda: self.topology.joint_plan(
                              op, {replica: cost.work.get("link", 0)}, algorithm=algorithm,
                              **collective_options(recipe, op)))
        response = self.regimes.resolve(context, self.cost.record_regime)
        if response is not None and response.mode == "joint_native":
            if getattr(recipe, "algorithm", None):
                raise ModelingError("joint_native_service_requires_single_local_kernel")
            demand = response.bind_joint(context)
            self.cost.record_joint_native(response, cost)
            declaration = {**response.execution, **response.diagnostics(),
                           "reference_program": type(recipe).__name__,
                           "reference_primitive_ms": measured if cost.timing is not None else None}
            return replace(context, work=demand, minimum_ms=0, setup_ms=0,
                           invocation_timing=None, joint_native_service=declaration)
        if response is not None and response.mode == "mac_service":
            matrix_role = str(response.execution.get("resource") or "cube")
            matrix_name = context.resource_names.get(matrix_role)
            if (not matrix_name or response.issued_work is None
                    or response.rate_flops_per_s is None
                    or not response.active_owners):
                raise ModelingError("mac_service_requires_bound_cube_work_and_rate")
            issued = float(response.issued_work)
            owners = int(response.active_owners)
            # ``cube`` work is physical time at peak (unit ``peak_ms``, rate 1).
            # The measured MAC time is per active owner; the fair-shared cube
            # resource yields that per-owner time while the tiled program
            # distributes it across the owner's tiles.  Do not multiply by the
            # owner count: that would serialize independent owners.
            mac_work_ms = float(response.duration_ms)
            peak_work_per_ms = float(context.compute_peaks.get(context.op.dtype, 0))
            physical_ms = (issued / (peak_work_per_ms / 1000.0)
                           if peak_work_per_ms > 0 else None)
            work = dict(context.work)
            work[matrix_name] = mac_work_ms
            declaration = {
                **response.execution, **response.diagnostics(),
                "stage": "mac", "resource_name": matrix_name,
                "issued_work": issued,
                "logical_work": float(cost.work.get("cube", 0.0)),
                "rate_flops_per_s": float(response.rate_flops_per_s),
                "active_owners": owners,
                "measured_mac_per_owner_ms": mac_work_ms,
                "mac_work_ms_total": mac_work_ms,
                "physical_mac_ms_at_peak": physical_ms,
                "reference_program": type(recipe).__name__,
            }
            program_recipe = None
            from .matrix import TiledMatmul
            if isinstance(recipe, TiledMatmul):
                source = TiledMatmul(
                    tile_m=int(response.execution["tile_m"]),
                    tile_n=int(response.execution["tile_n"]),
                    tile_k=int(response.execution["tile_k"]),
                    slots=int(response.execution["slots"]),
                )
                if source != recipe:
                    program_recipe = source
                    declaration = {**declaration,
                                   "source_selected_tiling": {
                                       "tile_m": source.tile_m, "tile_n": source.tile_n,
                                       "tile_k": source.tile_k, "slots": source.slots}}
            context = replace(context, work=work, mac_caps={},
                              mac_service=declaration,
                              program_recipe=program_recipe)
        self.conflicts += int(bool(measured and measured < physical))
        # A native peer plan owns one real invocation boundary in KernelGroups:
        # startup/minimum are applied once at its actual exit. Wrapping the
        # native program here would both duplicate that floor and hide its
        # launch handles from the shared cohort launcher.
        selected = (None if getattr(recipe, "native", False)
                    else self.invocations.select(context, recipe, devices=dies))
        if selected is not None:
            timing, record = selected
            self.cost.record_invocation(record)
            context = replace(context, invocation_timing=timing)
        if response is not None:
            if response.mode == "mac_service":
                context = replace(context, service_regime=response.diagnostics())
            else:
                context = replace(context,
                                  minimum_ms=max(context.minimum_ms, response.duration_ms),
                                  service_regime=response.diagnostics())
        return context

    def add(self, work: IterationWork, replica: int, release_ms: float, *,
            external_inputs=None, emitted_tokens=None) -> ScheduledIteration:
        if self._collectives is None:
            work = replace(work, ops=bind_cohort_traffic(
                self.case, {replica: work.ops})[replica])
        engine = self.engine
        prefix = f"{self.namespace}r{replica}:i{work.plan.iteration}"
        ids: dict[str, Event] = dict(external_inputs or {})
        commits: dict[str, Event] = {}
        scope = work.replay_scope or ("full" if work.graph_bucket is not None else "none")
        if self.include_submission and self.case.runtime.async_scheduling and replica in self.prepared:
            schedule = self.prepared.pop(replica)
        else:
            scheduler_ms = self.submission.scheduler_ms if self.include_submission else 0
            schedule = engine.submit(prefix + ":schedule", {f"{self.namespace}scheduler@r{replica}": scheduler_ms},
                                     release=release_ms, owner=f"replica:{replica}")
        self.schedules[replica] = schedule
        prepared, input_ready = self.serving_io.input(self, prefix, replica, schedule, work)
        graph_launch: dict[int, Event] = {}
        layer_entries: dict[int, Event] = {}
        expert_layer_events: dict[tuple[int, int], Event | None] = {}
        last_expert_done: dict[int, Event] = {}
        kv_layer_events: dict[tuple[int, int], tuple[Event, ...]] = {}
        last_kv_done: dict[int, Event] = {}
        kv_writeback_layers: set[tuple[int, int]] = set()
        kv_writeback_events: list[Event] = []
        physical_ops = self.recipes.expand(work.ops)
        from .kernel_groups import KernelGroups, declaration
        kernel_groups = KernelGroups(self, physical_ops, replica, prefix)
        for op in physical_ops:
            original = op
            op, recipe = self.prepare(op, replica)
            stage_key = replica, op.stage
            name = prefix + ":" + op.op_id
            names = self.stage_names[stage_key]
            deps = [input_ready, *(ids[key] for key in op.deps)]
            if self.offload_policy is not None and self.offload_policy.expert_ddr_ratio > 0.0:
                from .offload import is_routed_expert_entry_op, schedule_expert_layer_offload
                if is_routed_expert_entry_op(op):
                    exp_key = (op.stage, op.layer)
                    if exp_key not in expert_layer_events:
                        expert_layer_events[exp_key] = schedule_expert_layer_offload(
                            self, prefix, replica, op.stage, op.layer, work.valid_tokens,
                            last_expert_done.get(op.stage, input_ready), op=op)
                    if expert_layer_events[exp_key] is not None:
                        deps.append(expert_layer_events[exp_key])
            if (self.offload_policy is not None
                    and (self.offload_policy.kvc_ddr_ratio + self.offload_policy.kvc_ssd_ratio) > 0.0):
                from .offload import is_kv_or_state_entry_op, schedule_kv_layer_offload
                if is_kv_or_state_entry_op(op):
                    kv_key = (op.stage, op.layer)
                    if kv_key not in kv_layer_events:
                        kv_layer_events[kv_key] = schedule_kv_layer_offload(
                            self, prefix, replica, op.stage, op.layer, work,
                            input_ready, tuple(deps), layer_entries,
                            prev_kv_done=last_kv_done.get(op.stage))
                    deps.extend(kv_layer_events[kv_key])
            deps.extend(kernel_groups.entry_dependencies(op, ids))
            stream_key = replica, op.stage, op.stream or "main"
            stream_tail = kernel_groups.stream_tail(op, stream_key)
            if stream_tail is not None:
                deps.append(stream_tail)
            context = self._context(op, replica, scope, recipe=recipe)
            context = replace(context, group_buffers=kernel_groups.buffer_scope(op))
            owner = f"replica:{replica}"
            captured = scope == "full" or (
                scope == "piecewise" and bool(dict(op.metadata).get("graph_captured"))
            )
            capture_key = op.stage, dict(op.metadata).get("graph_segment", 0) if scope == "piecewise" else 0
            kernel_group = declaration(op)
            submit = (bool(kernel_group.get("entry")) if kernel_group else
                      bool(context.work or getattr(recipe, "algorithm", None)))
            if submit and self.include_submission:
                if captured and capture_key in graph_launch:
                    launch = graph_launch[capture_key]
                else:
                    dies = self.stage_dies[stage_key]
                    duration = self.submission.launch_ms(dies, captured=captured)
                    host_deps = (prepared,) + ((self.hosts[stage_key],) if stage_key in self.hosts else ())
                    launch = engine.submit(name + ":submit", {names["host"]: duration},
                                           after=host_deps, owner=owner)
                    self.hosts[stage_key] = launch
                    self.submission_counts[
                        "graph_replays" if captured else "eager_invocations"] += 1
                    if captured:
                        graph_launch[capture_key] = launch
                deps.append(launch)
            ready = engine.submit(name + ":ready", after=tuple(deps) or (schedule,), owner=owner)
            if (self.readiness_report
                    and op.family in {"collective", "allgather", "reduce_scatter",
                                      "allreduce", "transport"}
                    and op.link_bytes > 0
                    and dict(op.metadata).get("group_axis") != "ep"):
                from .collective_readiness import bind_pipeline_ready
                ranks = self.topology.ranks(replica, op.stage)
                ready_by_rank = {rank: ready for rank in ranks}
                collective, expected_bytes, expected_dtype = _collective_native_basis(
                    op, len(ranks))
                try:
                    readiness_context = bind_pipeline_ready(
                        self.readiness_report, ready_by_rank,
                        expected_collective=collective,
                        expected_bytes=(int(expected_bytes)
                                        if expected_bytes is not None
                                        and expected_bytes.denominator == 1
                                        else None),
                        expected_dtype=expected_dtype,
                        expected_group_size=len(ranks),
                        connection_id=dict(op.metadata).get(
                            "collective_connection_id"),
                        ordinal=dict(op.metadata).get("collective_call_ordinal"),
                        call_id=dict(op.metadata).get("collective_call_id"))
                except ModelingError as exc:
                    self.readiness_unresolved.append({
                        "op_id": op.op_id, "reason": str(exc)})
                else:
                    self.readiness_observations.append({
                        "op_id": op.op_id, "replica": replica,
                        "ready_by_rank": tuple(sorted(ready_by_rank)),
                        "context": readiness_context,
                    })
            if op.layer >= 0 and op.layer not in layer_entries:
                layer_entries[op.layer] = ready
            ready = kernel_groups.phase_ready(op, ready, name)
            recipe_name = op.recipe or type(recipe).__name__
            self.recipes_used[recipe_name] = self.recipes_used.get(recipe_name, 0) + 1
            cross_dp = (dict(op.metadata).get("group_axis") == "ep"
                        and self.case.runtime.parallel.ep_spans_dp)
            if cross_dp and getattr(recipe, "algorithm", None) and self._collectives is None:
                raise ModelingError("joint_collective_requires_complete_cohort")
            grouped = self._collectives is not None and cross_dp
            if grouped:
                complete = engine.future(name + ":collective")
                self._collectives.setdefault(op.op_id, []).append(CollectiveParticipant(
                    replica, ready, context, recipe, name, complete))
            else:
                complete = self._start_program(
                    context, recipe, name, ready, owner=owner, replica=replica)
            complete = kernel_groups.finish(op, context, recipe, complete, stream_key)
            ids[op.op_id] = complete
            if self.offload_policy is not None and self.offload_policy.expert_ddr_ratio > 0.0:
                from .offload import is_routed_expert_entry_op
                if is_routed_expert_entry_op(op) or op.recipe in {"grouped_down", "balanced_grouped_down"}:
                    last_expert_done[op.stage] = complete
            if (self.offload_policy is not None
                    and (self.offload_policy.kvc_ddr_ratio + self.offload_policy.kvc_ssd_ratio) > 0.0):
                from .offload import is_kv_or_state_entry_op, schedule_kv_layer_writeback
                if is_kv_or_state_entry_op(op):
                    last_kv_done[op.stage] = complete
                    wb_key = (op.stage, op.layer)
                    if wb_key not in kv_writeback_layers:
                        kv_writeback_layers.add(wb_key)
                        kv_writeback_events.extend(schedule_kv_layer_writeback(
                            self, prefix, replica, op.stage, op.layer, work, complete))
            if self.on_operator is not None:
                self.on_operator(original, replica, "JointNativeService" if context.joint_native_service
                                 is not None else type(recipe).__name__, ready, complete)
            if op.family == "token_commit":
                commits.update((request, complete) for request in op.requests)
            for key in self.work:
                self.work[key] += float(getattr(op, key))
        offload_tails: tuple[Event, ...] = tuple(
            ev for evs in kv_layer_events.values() for ev in evs
        ) + tuple(kv_writeback_events)
        if self.offload_policy is not None and not offload_tails:
            from .offload import schedule_kv_iteration_offload
            offload_tails = schedule_kv_iteration_offload(
                self, prefix, replica, work, input_ready, physical_ops, ids,
                scheduled_layers=set(kv_layer_events) | kv_writeback_layers)
        device_complete = engine.submit(
            prefix + ":device_complete", owner=f"replica:{replica}",
            after=(tuple(ids[op.op_id] for op in physical_ops) + offload_tails) or (input_ready,))
        completion = engine.submit(prefix + ":complete", after=(device_complete,),
                                   latency=self.submission.completion_ms if self.include_submission else 0,
                                   owner=f"replica:{replica}")
        for view in work.plan.requests:
            commits.setdefault(view.request_id, completion)
        host_results, clients, host_complete = self.serving_io.output(
            self, prefix, replica, work, commits, completion,
            emitted_tokens if emitted_tokens is not None else {
                view.request_id: int(view.draft_advanced_tokens or 1)
                for view in work.plan.requests if view.samples})
        if self.serving_io.enabled:
            completion = engine.submit(prefix + ":step_ready", after=(completion, host_complete),
                                       owner=f"replica:{replica}")
        step = ScheduledIteration(replica, work.plan.iteration, completion, release_ms,
                                  work.mode, work.valid_tokens, work.padded_tokens,
                                  work.graph_bucket, scope, len(physical_ops), commits,
                                  device_complete, host_results, clients, work.implementation_selections)
        engine.submit(prefix + ":step_metrics", after=(completion,),
                      on_complete=lambda _: self.step_summary.record(step))
        return step

    def prepare_next(self, replica):
        if self.include_submission and self.case.runtime.async_scheduling:
            self.prepared[replica] = self.engine.submit(
                f"{self.namespace}r{replica}:prepare_next",
                {f"{self.namespace}scheduler@r{replica}": self.submission.scheduler_ms},
                after=(self.schedules[replica],), owner=f"replica:{replica}")

    def submission_summary(self) -> dict:
        return {
            "included": self.include_submission,
            **self.submission_counts,
            "host_service_ms": sum(
                self.engine.serviced.get(names["host"], 0)
                for names in self.stage_names.values()),
            "stages": [
                {"replica": replica, "stage": stage,
                 "kernel_launch_ms": self.submission.launch_ms(dies),
                 "graph_replay_ms": self.submission.launch_ms(dies, captured=True),
                 "kernel_parameter": (f"{self.submission_source}.kernel_launch_ms"
                                      if self.submission.kernel_launch_ms is not None
                                      else "machine.device.launch_us"),
                 "replay_parameter": (f"{self.submission_source}.graph_replay_ms"
                                      if self.submission.graph_replay_ms is not None
                                      else "kernel_launch_parameter")}
                for (replica, stage), dies in self.stage_dies.items()
            ],
            "boundary": "host enqueue service; may overlap ready device work",
            "zero_parameter": "excludes undeclared host cost; not measured zero overhead",
        }

    def native_group_summary(self) -> list:
        """Bounded actual launch evidence from the shared native cohorts."""
        groups = []
        for key in sorted(self.native_group_launches):
            item = self.native_group_launches[key]
            groups.append({
                "op_id": item["op_id"], "kind": item["kind"],
                "launch_mode": item.get("launch_mode", ""),
                "ranks": list(item["ranks"]),
                "payload_basis": dict(sorted(item["payload_basis"].items())),
                "payload_packet_bytes": dict(sorted(
                    item["payload_packet_bytes"].items())),
                "payload_frames": {
                    name: dict(sorted(frames.items()))
                    for name, frames in sorted(item["payload_frames"].items())},
                "link_by_phase_by_rank": {
                    name: {phase: {pipe: float(amount)
                                   for pipe, amount in values.items()}
                           for phase, values in sorted(phases.items())}
                    for name, phases in sorted(
                        item["link_by_phase_by_rank"].items())},
                "work_by_rank": {
                    name: [[str(pipe), float(amount)] for pipe, amount in work]
                    for name, work in sorted(item["work_by_rank"].items())},
                "transport_by_rank": {
                    name: {phase: dict(details)
                           for phase, details in sorted(phases.items())}
                    for name, phases in sorted(
                        item.get("transport_by_rank", {}).items())},
                "transport_work_by_rank": {
                    name: [[str(pipe), float(amount)]
                           for pipe, amount in sorted(work.items())]
                    for name, work in sorted(
                        item.get("transport_work_by_rank", {}).items())},
                "transport_tx_bytes": float(item.get(
                    "transport_tx_bytes", 0.0)),
                "transport_rx_bytes": float(item.get(
                    "transport_rx_bytes", 0.0)),
                "transport_frames": int(item.get("transport_frames", 0)),
                "transport_edges": int(item.get("transport_edges", 0)),
                "total_edges": int(item.get("total_edges", 0)),
                "payload_edges": int(item.get("payload_edges", 0)),
                "count_edges": int(item.get("count_edges", 0)),
                "directed_edges": item["directed_edges"],
                "directed_frames": item["directed_frames"],
                "protocol_edges": item["protocol_edges"],
                "unresolved_edges": item["unresolved_edges"],
                "partial": list(item["partial"]),
            })
            if len(groups) >= 64:
                break
        return groups

    def summary(self) -> dict:
        native_groups = self.native_group_summary()
        return {
            **self.engine.summary(), "work": dict(self.work),
            **({"work_basis": "logical_and_analytical_reference; executed_service_is_in_serviced"}
               if self.cost.joint_native else {}),
            "recipes": dict(self.recipes_used),
            "submission": self.submission_summary(),
            "serving_io": self.serving_io.summary(),
            "step_summary": self.step_summary.to_dict(),
            "rank_model": "representative_local_shard_per_stage",
            **({"native_group_count": len(self.native_group_launches),
                "native_groups": native_groups} if native_groups else {}),
            **({"collective_readiness_observations": list(self.readiness_observations),
                "collective_readiness_unresolved": list(self.readiness_unresolved)}
               if (self.readiness_observations or self.readiness_unresolved) else {}),
            **({"joint_collective_observations": list(
                    self._joint_collective_trace.values()),
                "joint_collective_stats": dict(self.joint_collective_stats)}
               if (self._joint_collective_trace is not None
                   and self.joint_collective_stats["observations"]) else {}),
        }
