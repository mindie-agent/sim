"""Lower MC2 peer routes and bind the shared phase ledger.

[Role] Describe per-hop transfers and bind analytical phases to engine resources.
[Invariants] Ordered physical hops each carry the wire bytes. SHMEM transfers
  share rank injection capacity and keep their declared per-peer rate caps.
[Timeouts & Locks] Finite phase and peer traversal without locks.
[Side-Effect Policy] Binds analytical stages and declares capacity on the
  caller's engine without I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from ...engine import Resource
from ...mechanism import Work
from ...program import KernelProgram, RecipeContext, Stage

if TYPE_CHECKING:
    from .common import Mc2CohortSlice, Mc2Contract

DISPATCH = "dispatch_v2"
FP32_SCALE_BYTES = 4
INT32_BYTES = 4
INT64_BYTES = 8
TRIPLE_META_BYTES = 12
INT8_QUANT_OPS = 7
FP8_QUANT_OPS = 5
SIO_HCCS = "a3_sio_hccs"
A2_MTE = "a2_mte"
SHMEM = "a5_shmem"
CCU = "a5_ccu"
A5_ROCE = "a5_roce"
MILLISECONDS_PER_SECOND = 1000

@dataclass(frozen=True, slots=True)
class PeerRoute:
    source: int
    destination: int
    hops: tuple[str, ...]


def declared_incoming_routes(rank: int, peers: tuple[int, ...], hops: dict) -> tuple[PeerRoute, ...]:
    return tuple(PeerRoute(peer, rank, hops[peer, rank]) for peer in peers if (peer, rank) in hops)


@dataclass(frozen=True, slots=True)
class Transport:
    abi: str
    algorithm: str
    routes: tuple[PeerRoute, ...]
    wire_alignment: int = 1
    window_alignment: int = 1
    window_offset: int = 0
    credits: int = 1
    slot_depth: int = 1
    peer_limit: int = 0
    ccu: bool = False
    aiv_staging: bool = True
    direct_remote_gm: bool = False
    gm_bounce_hbm: bool = False
    incoming_routes: tuple[PeerRoute, ...] = ()

    def peer_bytes(self, payload: int | float) -> int | float:
        return align_up(math.ceil(payload), self.wire_alignment) if self.wire_alignment > 1 else payload


@dataclass(frozen=True, slots=True)
class Phase:
    name: str
    after: tuple[str, ...] = ()
    work: Work = field(default_factory=Work)
    owner: str = ""
    metadata: tuple[tuple[str, bool | int | float | str | bytes | None | tuple[int, ...]], ...] = ()
    service_mode: str = ""
    physical_hop: str = ""
    shmem: bool = False
    outgoing: bool = True


def read_work(amount: int | float, *, aic: bool = False, l2: bool = False) -> Work:
    pipe = "mte2_aic" if aic else "mte2_aiv"
    return Work(resources={pipe: amount, "hbm": amount, **({"l2": amount} if l2 else {})})


def store_work(amount: int | float, *, fixpipe: bool = False, l2: bool = False) -> Work:
    pipe = "fixpipe" if fixpipe else "mte3"
    return Work(resources={pipe: amount, "hbm": amount, **({"l2": amount} if l2 else {})})


def peer_stages(name: str, after: tuple[str, ...], route: PeerRoute, transport: Transport,
                amount: int | float, metadata: tuple = (), *, outgoing: bool = True,
                couple_gm_bounce: bool = False) -> tuple[Phase, ...]:
    wire = transport.peer_bytes(amount)
    info = (
        ("source_rank", route.source), ("destination_rank", route.destination),
        ("topology_hops", "/".join(route.hops)), ("peer_limit", transport.peer_limit),
        ("wire_padding_bytes", wire - amount),
    )
    extra_hbm = {"hbm": wire} if (couple_gm_bounce and transport.gm_bounce_hbm and wire) else {}
    if transport.abi != SIO_HCCS:
        return (Phase(name, after, Work(resources={"link": wire, **extra_hbm}), metadata=metadata + info,
                      service_mode="mc2_peer_" + "_".join(route.hops).lower(),
                      shmem=transport.abi == SHMEM, outgoing=outgoing),)
    phases = []
    for index, hop in enumerate(route.hops):
        stage = name if index == len(route.hops) - 1 else f"{name}_hop_{index}"
        hop_hbm = extra_hbm if index == 0 else {}
        phases.append(Phase(stage, after, Work(resources={"link": wire, **hop_hbm}), metadata=metadata + info + (
            ("physical_hop", hop), ("hop_index", index), ("hop_bytes", wire),
        ), service_mode=f"mc2_peer_{hop.lower()}", physical_hop=hop))
        after = (stage,)
    return tuple(phases)


def cohort_stages(
    contract: Mc2Contract, cohort_slice: Mc2CohortSlice, transport: Transport,
    framing: tuple[int, int, int, int], *,
    after: tuple[str, ...] = (), peer_after: tuple[str, ...] = (),
) -> tuple[Phase, ...]:
    """Lower one caller-owned cohort with explicit data, status and consumption."""
    part, layout = cohort_slice, contract.row_layout
    cid = part.cohort_id
    prefix = f"cohort{cid}_"
    tag = (("cohort_id", cid), ("row_start", part.row_start), ("row_end", part.row_end),
           ("cohort_row_ids", part.row_ids))
    if part.row_ids:
        tag += (("cohort_compact_indices", tuple(range(part.row_count))), ("cohort_restore_map", part.row_ids))
    receiving = part.received_assignments
    working = part.source_assignments + part.masked_padding_rows
    dispatch = contract.abi == DISPATCH
    counts = {(src, dst): count for cohort, src, dst, count in contract.assignments if cohort == cid}
    meta_bytes = layout.expert_scale_bytes + INT32_BYTES if dispatch else TRIPLE_META_BYTES
    pack = working * (layout.input_bytes + layout.input_scale_bytes + meta_bytes)
    vector = working  # One validity test per packed or masked assignment.
    if dispatch and layout.quant_mode != "none":
        ops = INT8_QUANT_OPS if layout.payload_dtype == "int8" else FP8_QUANT_OPS
        vector += working * (ops * layout.width - 1)
    pack_name = prefix + "pack"
    phases = [Phase(pack_name, after, read_work(pack) + Work(resources={"vector": vector}), "aiv", tag + (
        ("pack_input_bytes", pack), ("masked_padding_rows", part.masked_padding_rows),
    ))]
    send_done, recv_done = [], []
    row_bytes, status_bytes, padding_bytes, flag_bytes = framing
    count_every_peer = transport.ccu and dispatch
    rank = contract.rank
    for route in transport.routes:
        peer = route.destination
        rows = counts.get((rank, peer), 0)
        if not rows and not count_every_peer:
            continue
        payload = rows * row_bytes
        data = prefix + f"send_{peer}"
        credit = (send_done[-transport.credits],) if len(send_done) >= transport.credits else ()
        deps = (pack_name, *credit, *peer_after)
        if payload:
            window = transport.peer_bytes(payload + status_bytes)
            if transport.aiv_staging:
                staged = data + "_stage_ub"
                phases.append(Phase(staged, deps, Work(resources={"mte3": payload}), "aiv", tag))
                deps = (staged,)
            phases.extend(peer_stages(data, deps, route, transport, payload, tag + (
                ("payload_bytes", payload), ("protocol_padding_bytes", rows * padding_bytes),
                ("frame_flag_bytes", rows * flag_bytes),
                ("window_bytes", align_up(math.ceil(window), transport.window_alignment)),
                ("window_rank", peer), ("window_offset_bytes", transport.window_offset),
                ("direct_remote_gm", transport.direct_remote_gm),
                ("transfer", ("write_nbi" if dispatch else "read_nbi") if transport.ccu else "peer_window"),
            )))
            deps = (data,)
        if status_bytes:
            status = prefix + f"status_{peer}"
            phases.extend(peer_stages(status, deps, route, transport, status_bytes,
                                      tag + (("status_bytes", status_bytes),)))
            data = status
        send_done.append(data)
    incoming_routes = {
        route.destination: PeerRoute(route.destination, rank, tuple(reversed(route.hops)))
        for route in transport.routes
    }
    incoming_routes.update((route.source, route) for route in transport.incoming_routes)
    for peer, incoming in incoming_routes.items():
        rows = counts.get((peer, rank), 0)
        if not rows and not count_every_peer:
            continue
        receive = rows * row_bytes
        data = prefix + f"recv_link_{peer}"
        credit = (recv_done[-transport.credits],) if len(recv_done) >= transport.credits else ()
        phases.extend(peer_stages(data, (*after, *peer_after, *credit), incoming, transport, receive + status_bytes,
                                  tag + (("payload_bytes", receive), ("status_bytes", status_bytes),
                                         ("protocol_padding_bytes", rows * padding_bytes),
                                         ("frame_flag_bytes", rows * flag_bytes)), outgoing=False))
        if not receive:
            recv_done.append(data)
            continue
        visible, consume = prefix + f"recv_{peer}_ready", prefix + f"recv_{peer}_consume"
        phases.append(Phase(visible, (data,), Work(resources={"hbm": receive}), metadata=tag))
        phases.append(Phase(consume, (visible,), read_work(receive), "aiv", tag))
        recv_done.append(consume)
    count_bytes = status_bytes * len(recv_done)
    readiness = prefix + ("cross_core_ready" if transport.ccu else "readiness")
    phases.append(Phase(readiness, (pack_name, *recv_done), read_work(count_bytes) + store_work(2 * count_bytes), "aiv", tag))
    local_bytes = part.local_assignments * layout.materialized_bytes
    if dispatch:
        output = receiving * layout.materialized_bytes
        if contract.experts:
            output += contract.experts * INT32_BYTES + math.ceil(contract.experts / contract.world_size) * INT64_BYTES
        unpack = read_work(local_bytes) + store_work(local_bytes + output)
    else:
        tokens = part.row_count
        weighted = 2 * receiving * layout.width  # Multiply and accumulate into each source row.
        output = tokens * math.ceil(layout.width * dtype_bytes(contract.output_dtype))
        unpack = read_work(local_bytes + receiving * FP32_SCALE_BYTES) + store_work(local_bytes + output)
        unpack += Work(resources={"vector": weighted})
    phases.append(Phase(prefix + "unpack", (readiness,), unpack, "aiv", tag))
    phases.append(Phase(prefix + "slot_release", (prefix + "unpack", *send_done), metadata=tag))
    return tuple(phases)


def _bind_transfer(phase: Phase, context: RecipeContext, routed_pair: str | None = None) -> tuple[dict, dict]:
    names = context.resource_names
    candidates = (
        context.physical_role(phase.physical_hop.lower()),
        context.physical_role(phase.physical_hop),
        context.physical_role(phase.service_mode),
        context.physical_role("mc2_peer"),
        context.physical_role("link"),
    )
    resource = routed_pair or next((names[role] for role in candidates if role and role in names), None)
    if resource is None:
        routed = sorted(set(context.work) - set(names.values()))
        if len(routed) == 1:
            resource = routed[0]
        elif routed:
            tx_ports = [item for item in routed if ":tx:port" in item]
            pool = tx_ports or routed
            src_idx = int(dict(phase.metadata).get("source_rank", 0) or 0)
            resource = pool[src_idx % len(pool)]
        else:
            raise ModelingError(f"mc2_peer_resource_binding_required:{phase.service_mode}")
    rate = context.service_rate(phase.service_mode)
    if rate is None:
        rate = context.service_rate("mc2_peer")
    caps = {resource: rate / MILLISECONDS_PER_SECOND} if rate is not None else {}
    return {resource: phase.work.resources["link"]}, caps


def _shmem_pair(context: RecipeContext, source: int, destination: int) -> str | None:
    suffix = f":pair{source}>{destination}"
    candidates = [name for name in context.rates if name.endswith(suffix)]
    return next((name for name in candidates if name in context.work),
                candidates[0] if len(candidates) == 1 else None)


def _bind_injection(
    source: int, pair: str | None, context: RecipeContext, extra: dict[str, Resource],
) -> str | None:
    role = context.physical_role("mc2_injection")
    bound = context.resource_names.get(role) if role != "link" else None
    if bound is None and pair is not None:
        fabric, _ = pair.rsplit(":pair", 1)
        port = f"{fabric}:tx:port{source}"
        if port in context.rates:
            bound = port
    rate = context.service_rate("mc2_injection")
    if rate is None:
        return bound
    rate /= MILLISECONDS_PER_SECOND
    if bound is not None and context.rates[bound] <= rate:
        return bound
    # A service mode can cap a faster backing link. The cap must belong to
    # one fair resource for the source rank, so concurrent peers share it.
    scope = context.resource_names.get("link", context.resource_names.get("hbm", "link"))
    name = f"{scope}:mc2_injection:{source}"
    extra[name] = Resource(name, rate, unit="bytes")
    return name


@dataclass(frozen=True)
class _ShmemProgram(KernelProgram):
    injection_resources: tuple[Resource, ...] = ()

    def start(self, engine, name, ready, *, owner=""):
        missing = tuple(resource for resource in self.injection_resources
                        if engine.resources.get(resource.name) != resource)
        if missing:
            engine.extend(missing)
        return super().start(engine, name, ready, owner=owner)


@dataclass(frozen=True, slots=True)
class Mc2Program:
    phases: tuple[Phase, ...]

    def __call__(self, context: RecipeContext) -> KernelProgram:
        stages = []
        injection_resources = {}
        encoded_rows: dict[int, str] = {}
        for phase in self.phases:
            info = dict(phase.metadata)
            non_link_coupled = False
            if phase.service_mode and phase.work.resources["link"]:
                pair = _shmem_pair(context, info["source_rank"], info["destination_rank"]) if phase.shmem else None
                demand, caps = _bind_transfer(phase, context, pair)
                if phase.shmem and phase.outgoing:
                    injection = _bind_injection(info["source_rank"], pair, context, injection_resources)
                    if injection is not None:
                        demand[injection] = phase.work.resources["link"]
                non_link = {k: v for k, v in phase.work.resources.items() if k != "link" and v}
                if non_link:
                    non_link_coupled = True
                    for r_name, r_val in context.bind(Work(resources=non_link)).items():
                        demand[r_name] = demand.get(r_name, 0.0) + r_val
            else:
                demand, caps = context.bind(phase.work), {}
            placement = {}
            if phase.owner:
                units = context.effective_aic_cores if phase.owner == "aic" else context.effective_aiv_cores
                roles = ("cube", "mte2_aic", "mte1", "fixpipe") if phase.owner == "aic" else ("mte2_aiv", "vector", "mte3")
                placement = {
                    name: owners for name, owners in context.owned(0, 1, roles).items() if name in demand
                }
                placement = {name: owners[:units] for name, owners in placement.items()}
            # Every peer phase of a cohort shares the same row sequences.
            for key in ("cohort_row_ids", "cohort_compact_indices", "cohort_restore_map"):
                rows = info.get(key)
                if type(rows) is tuple:
                    identity = id(rows)
                    if identity not in encoded_rows:
                        encoded_rows[identity] = ",".join(map(str, rows))
                    info[key] = encoded_rows[identity]
            stages.append(Stage(phase.name, demand, phase.after, caps=caps, placement=placement,
                                coupled=(phase.shmem or non_link_coupled) and len(demand) > 1,
                                metadata=info, keep_empty=True))
        if injection_resources:
            return _ShmemProgram(tuple(stages), injection_resources=tuple(injection_resources.values()))
        return KernelProgram(tuple(stages))
