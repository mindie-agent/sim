"""Physical connector service for P/D handoffs.

[Role] Submit causal transfers and account for wire, memory and receiver residency.
[Invariants] Source HBM reads, wire transmission, and post-propagation destination HBM
  writes stream causally per chunk; receive buffers share one physical Engine pool per
  buffer id; simultaneous live reservations and arrivals/consumptions conserve bytes.
[Timeouts & Locks] Chunk issuance is bounded. All waits are virtual Engine events.
[Side-Effect Policy] Submits Engine tasks and updates connector counters only after preflight.
"""
import math
import re
from collections import defaultdict
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link

from ..collectives.ports import PortKey
from ..collectives.transport import chunk_size_at
from ..communication import MessageFlow
from ..topology import Topology, TransportOptions
from .handoff import PDHandoff, _non_negative_number, _normalize_releases
from .transport import _AliasedTopology, _alias_plan

_MAX_PHYSICAL_CHUNKS = 4096
_MS_PER_S = 1000.0
_FLOAT_ABS_TOL = 1e-9
_FLOAT_REL_TOL = 1e-12
_SOURCE_PREFIXES = ("prefill:", "prefill_", "src:", "src_", "source:", "source_")
_DEST_PREFIXES = (
    "decode:", "decode_", "dst:", "dst_", "dest:", "dest_", "destination:", "destination_",
)
_REPLICA_TOKEN_RE = re.compile(r"(?:^|[@:_/-])r(\d+)(?=$|[@:_/-])")
_STAGE_TOKEN_RE = re.compile(r"(?:^|[@:_/-])s(\d+)(?=$|[@:_/-])")
_TIER_SPLIT_RE = re.compile(r"[:@._/-]+")
_HOST_TIER_TOKENS = frozenset({"host", "dram", "ddr", "cpu"})
_HBM_TIER_TOKENS = frozenset({"hbm", "memory"})


def _parse_tier_token(text: str) -> str | None:
    tokens = set(_TIER_SPLIT_RE.split(str(text).lower()))
    if tokens & _HOST_TIER_TOKENS:
        return "host"
    if tokens & _HBM_TIER_TOKENS:
        return "hbm"
    return None


def _parse_replica_token(text: str) -> int | None:
    matches = list(_REPLICA_TOKEN_RE.finditer(str(text).lower()))
    return int(matches[-1].group(1)) if matches else None


def _parse_stage_token(text: str) -> int | None:
    matches = list(_STAGE_TOKEN_RE.finditer(str(text).lower()))
    return int(matches[-1].group(1)) if matches else None


def _classify_memory_resource(name: str) -> tuple[str, tuple[int | None, int | None, str | None]]:
    """Classify a memory resource name into role and ``(replica, stage, tier)`` scope."""
    text = str(name)
    low = text.lower()
    scope = (_parse_replica_token(low), _parse_stage_token(low), _parse_tier_token(low))
    if low in {"src", "source", "prefill"} or low.startswith(_SOURCE_PREFIXES):
        return "source", scope
    if low in {"dst", "dest", "destination", "decode"} or low.startswith(_DEST_PREFIXES):
        return "destination", scope
    return "shared", scope


def _partition_memory_work(memory_work) -> tuple[list, list, list]:
    """Split memory work mapping/items into source, destination, and shared entries."""
    items = memory_work.items() if hasattr(memory_work, "items") else memory_work
    src_items, dst_items, shared_items = [], [], []
    for key, raw_value in items:
        amount = float(raw_value)
        if amount <= 0.0:
            continue
        role, scope = _classify_memory_resource(str(key))
        if role == "source":
            src_items.append((str(key), amount, scope))
        elif role == "destination":
            dst_items.append((str(key), amount, scope))
        else:
            shared_items.append((str(key), amount))
    return src_items, dst_items, shared_items


def _transfer_scope(transfer, side: str) -> tuple[int | None, int, str]:
    if side == "source":
        rep = _parse_replica_token(transfer.source_scope)
        if rep is None:
            rep = _parse_replica_token(transfer.source_storage)
        src_scope_tier = _parse_tier_token(transfer.source_scope)
        if src_scope_tier is not None:
            tier = src_scope_tier
        else:
            src_storage_low = str(transfer.source_storage).lower()
            tier = (
                "host" if src_storage_low.startswith(("pd:host:", "host:", "host_"))
                else "hbm"
            )
        return rep, int(transfer.source_stage), tier
    rep = _parse_replica_token(transfer.destination_scope)
    if rep is None:
        rep = _parse_replica_token(transfer.destination_storage)
    tier = str(transfer.destination_tier or "hbm").lower()
    return rep, int(transfer.destination_stage), tier


def _matching_transfer_indices(
    transfers, side: str, scope: tuple[int | None, int | None, str | None],
) -> list[int]:
    if transfers is None:
        return []
    replica, stage, tier = scope
    scopes = [_transfer_scope(t, side) for t in transfers]

    def _filter(req_rep: int | None, req_stage: int | None, req_tier: str | None) -> list[int]:
        return [
            idx for idx, (t_rep, t_stage, t_tier) in enumerate(scopes)
            if (req_stage is None or t_stage == req_stage)
            and (req_tier is None or t_tier == req_tier)
            and (req_rep is None or t_rep == req_rep)
        ]

    for req_rep, req_stage, req_tier in (
        (replica, stage, tier),
        (None, stage, tier),
        (None, stage, None),
    ):
        matched = _filter(req_rep, req_stage, req_tier)
        if matched:
            return matched
    return []


def _plan_wire_and_transactions(plan, *, track_transactions: bool):
    wire = sum(
        hop.wire_bytes for path in plan.flows
        for chunk in path.chunks for hop in chunk.hops
    )
    transactions = (
        sum(
            hop.transactions or 0 for path in plan.flows
            for chunk in path.chunks for hop in chunk.hops
        )
        if track_transactions
        else None
    )
    chunks = sum(len(path.chunks) for path in plan.flows)
    return wire, transactions, chunks


def _validate_plan_chunks(plan) -> tuple[int, ...]:
    """Force bounded-chunk validation before creating pools or submitting tasks."""
    return tuple(len(path.chunks) for path in plan.flows)


def _build_flow_memory_weights(plan, memory_work, transfers=None) -> list[dict[str, float]]:
    """Allocate declared memory work proportionally across matching flows (0-byte drift)."""
    n_flows = len(plan.flows)
    per_flow: list[dict[str, float]] = [{} for _ in range(n_flows)]
    if not n_flows:
        return per_flow
    src_items, dst_items, shared_items = _partition_memory_work(memory_work)
    payloads = [float(path.flow.payload_bytes) for path in plan.flows]
    all_indices = list(range(n_flows))

    def _assign(res_name: str, amount: float, matching: list[int]) -> None:
        targets = matching if matching else all_indices
        total = sum(payloads[i] for i in targets)
        if total <= 0.0:
            share = amount / len(targets)
            for idx in targets:
                per_flow[idx][res_name] = per_flow[idx].get(res_name, 0.0) + share
            return
        for idx in targets:
            per_flow[idx][res_name] = (
                per_flow[idx].get(res_name, 0.0) + amount * (payloads[idx] / total)
            )

    for res_name, amount, scope in src_items:
        matching = (
            _matching_transfer_indices(transfers, "source", scope)
            if transfers is not None and any(part is not None for part in scope)
            else all_indices
        )
        _assign(res_name, amount, matching)
    for res_name, amount, scope in dst_items:
        matching = (
            _matching_transfer_indices(transfers, "destination", scope)
            if transfers is not None and any(part is not None for part in scope)
            else all_indices
        )
        _assign(res_name, amount, matching)
    for res_name, amount in shared_items:
        _assign(res_name, amount, all_indices)
    return per_flow


class PDConnector:
    def __init__(self, raw):
        explicit = "link" in raw
        declaration = dict(raw["link"] if explicit else raw)
        if not explicit:
            declaration["capacity_bytes_per_ms"] = declaration.pop("bytes_per_s") / _MS_PER_S
        declaration.setdefault("name", "connector")
        declaration.setdefault("members", (0, 1))
        declaration["members"] = tuple(declaration["members"])
        self.chunk_bytes = raw.get("chunk_bytes")
        if self.chunk_bytes is not None:
            self.chunk_bytes = _non_negative_number(
                self.chunk_bytes, "pd_connector_chunk_bytes")
            if not self.chunk_bytes:
                raise ModelingError("pd_connector_chunk_bytes_invalid:0")
        self.receive_capacity_bytes = raw.get("receive_capacity_bytes")
        if self.receive_capacity_bytes is not None:
            self.receive_capacity_bytes = _non_negative_number(
                self.receive_capacity_bytes, "pd_connector_receive_capacity_bytes")
            if not self.receive_capacity_bytes:
                raise ModelingError("pd_connector_receive_capacity_bytes_invalid:0")
        self.hop_startup_us = _non_negative_number(
            raw.get("hop_startup_us", 0.0), "pd_connector_hop_startup_us")
        self.receive_buffer_id = str(raw.get("receive_buffer_id", "connector"))
        self.shared_receive_buffer = bool(raw.get("shared_receive_buffer", False))
        for key in ("chunk_bytes", "receive_capacity_bytes", "hop_startup_us",
                    "receive_buffer_id", "shared_receive_buffer"):
            declaration.pop(key, None)
        self.transport_declared = (
            self.chunk_bytes is not None
            or self.receive_capacity_bytes is not None
            or bool(self.hop_startup_us))
        self.link = Link(**declaration)
        if set(self.link.members) != {0, 1}:
            raise ModelingError("pd_connector_members_require_pool_endpoints_0_1")
        if self.link.semantics not in {
                "shared", "direct", "per_pair", "per_port", "fully_connected", "mesh"}:
            raise ModelingError(f"link_semantics_unknown:{self.link.semantics}")
        self.topology = Topology(Hardware("pd_connector", (), (self.link,)), None, "pd:")
        self.alias = ({f"pd:link:{self.link.name}": "pd_link"}
                      if not explicit else {})
        self.wire_bytes = 0.0
        self.transactions = 0 if self.link.transaction_bytes is not None else None
        self._transport = {
            "chunks": 0, "credit_waits": 0,
            "reservation_waits": 0, "reservation_wait_ms": 0.0,
            "resident": {}, "peak_resident": {},
            "reserved": {}, "peak_reserved": {},
            "unknowns": set(),
        }
        self._tracked_residency = False
        self._engine = None
        self._handoff_count = 0
        self._handoff_flows = 0
        self._buffer_ranks: dict[str, set[int]] = defaultdict(set)
        self._handoff_ranks = set()
        self._last_receive_capacity = {}
        self._last_receive_buffers = {}

    @staticmethod
    def _receive_pool_name(buffer_id: str) -> str:
        return f"transport:rx:{buffer_id}"

    def _role_adjusted_resources(self, topology):
        """Apply connector aliases and pool-role TX/RX endpoint rates."""
        src_rate = self.link.endpoint_capacity_bytes_per_ms(0)
        dst_rate = self.link.endpoint_capacity_bytes_per_ms(1)
        tx_marker = f"link:{self.link.name}:tx:port"
        rx_marker = f"link:{self.link.name}:rx:port"
        has_endpoint_rates = bool(self.link.endpoint_rates)
        adjusted = []
        for resource in topology.resources():
            rate = resource.rate
            if has_endpoint_rates:
                if tx_marker in resource.name:
                    rate = (
                        float(src_rate) if src_rate is not None
                        else float(self.link.capacity_bytes_per_ms)
                    )
                elif rx_marker in resource.name:
                    rate = (
                        float(dst_rate) if dst_rate is not None
                        else float(self.link.capacity_bytes_per_ms)
                    )
            adjusted.append(
                replace(
                    resource,
                    name=self.alias.get(resource.name, resource.name),
                    rate=rate,
                )
            )
        return tuple(adjusted)

    def resources(self):
        return self._role_adjusted_resources(self.topology)

    def _alias_plan(self, plan):
        """Keep the public resource identity while using the shared planner."""
        aliased = _alias_plan(self, plan)
        if self.alias and plan.topology is not self.topology:
            aliased = replace(
                aliased, topology=_AliasedTopology(plan.topology, self.alias))
        return aliased

    def _topology_for_handoff(self, handoff):
        """Bind declared pool-endpoint rates/ids onto actual physical ranks."""
        src_ranks = {int(item.source_rank) for item in handoff.transfers}
        dst_ranks = {int(item.destination_rank) for item in handoff.transfers}
        if src_ranks <= {0} and dst_ranks <= {1}:
            return self.topology
        all_ranks = tuple(sorted(src_ranks | dst_ranks))
        src_rate = self.link.endpoint_capacity_bytes_per_ms(0)
        dst_rate = self.link.endpoint_capacity_bytes_per_ms(1)
        mapped_rates = []
        for rank in all_ranks:
            if rank in src_ranks and rank in dst_ranks:
                both = [r for r in (src_rate, dst_rate) if r is not None]
                if both:
                    mapped_rates.append((rank, float(min(both))))
            elif rank in src_ranks and src_rate is not None:
                mapped_rates.append((rank, float(src_rate)))
            elif rank in dst_ranks and dst_rate is not None:
                mapped_rates.append((rank, float(dst_rate)))
        ep_ids = dict(self.link.endpoint_ids)
        src_ep = ep_ids.get(0)
        dst_ep = ep_ids.get(1)
        mapped_ids = []
        for rank in all_ranks:
            if rank in src_ranks and rank in dst_ranks:
                if src_ep is not None and (dst_ep is None or src_ep == dst_ep):
                    mapped_ids.append((rank, src_ep))
                elif dst_ep is not None and src_ep is None:
                    mapped_ids.append((rank, dst_ep))
            elif rank in src_ranks and src_ep is not None:
                mapped_ids.append((rank, src_ep))
            elif rank in dst_ranks and dst_ep is not None:
                mapped_ids.append((rank, dst_ep))
        link = replace(
            self.link,
            members=all_ranks,
            endpoint_rates=tuple(mapped_rates),
            endpoint_ids=tuple(mapped_ids),
        )
        return Topology(Hardware("pd_connector", (), (link,)), None, "pd:")

    def _ensure_engine_resources(self, engine, topology):
        missing = [resource for resource in self._role_adjusted_resources(topology)
                   if resource.name not in engine.resources]
        if missing:
            engine.extend(resources=missing)

    def _check_pool_capacity(self, engine, pool: str, capacity: float, label):
        if pool in engine.pools and not math.isclose(
                engine.pools[pool], capacity,
                rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL):
            raise ModelingError(
                f"pd_receive_pool_capacity_conflict:{label}:{engine.pools[pool]}!={capacity}")

    def _preflight_handoff_pools(self, engine, handoff):
        """Atomically validate per-rank and shared-pool footprints before submission."""
        per_rank, by_buffer = handoff.resolved_receive_pools()
        for destination, (_buf_id, footprint, pool_cap) in sorted(per_rank.items()):
            if pool_cap < footprint:
                raise ModelingError(
                    "pd_whole_message_footprint_exceeds_pool:"
                    f"{destination}:{footprint}>{pool_cap}")
        for buffer_id, (total_footprint, pool_cap, ranks) in sorted(by_buffer.items()):
            for rank in ranks:
                rank_cap = per_rank[rank][2]
                if rank_cap != pool_cap:
                    raise ModelingError(
                        "pd_receive_pool_capacity_conflict:"
                        f"{rank}:{pool_cap}!={rank_cap}")
            pool = self._receive_pool_name(buffer_id)
            if (self.receive_capacity_bytes is not None
                    and buffer_id == self.receive_buffer_id):
                conn_cap = float(self.receive_capacity_bytes)
                if float(total_footprint) > conn_cap + _FLOAT_ABS_TOL:
                    raise ModelingError(
                        "pd_whole_message_footprint_exceeds_pool:"
                        f"{pool}:{total_footprint}>{conn_cap}")
                if not math.isclose(
                        float(pool_cap), conn_cap,
                        rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL):
                    raise ModelingError(
                        "pd_receive_pool_capacity_conflict:"
                        f"{buffer_id}:{conn_cap}!={pool_cap}")
            self._check_pool_capacity(engine, pool, float(pool_cap), ranks[0])
            if pool_cap < total_footprint:
                raise ModelingError(
                    "pd_whole_message_footprint_exceeds_pool:"
                    f"{pool}:{total_footprint}>{pool_cap}")
        return per_rank

    def _expand_handoff_releases(self, handoff, release_events):
        normalized = _normalize_releases(release_events) or {}
        chunk = self.chunk_bytes
        counts = defaultdict(int)
        for transfer in handoff.transfers:
            count = (1 if chunk is None else max(1, math.ceil(
                float(transfer.payload_bytes) / float(chunk) - _FLOAT_REL_TOL)))
            counts[transfer.destination_rank] += count
        result = {}
        for rank, count in counts.items():
            events = tuple(normalized.get(rank, ()))
            if not events:
                continue
            if len(events) == 1:
                events = events * count
            elif len(events) < count:
                raise ModelingError(
                    "pd_handoff_release_event_shortage:"
                    f"{rank}:{len(events)}<{count}")
            result[rank] = events
        return result

    def _declared_port_demand(self, engine, topo, src: int, dst: int,
                              wire_bytes: float) -> dict[str, float]:
        """Charge externally registered physical PortKey resources matching endpoint_ids."""
        ep_ids = dict(self.link.endpoint_ids)
        src_ep, dst_ep = ep_ids.get(0), ep_ids.get(1)
        if src_ep is None and dst_ep is None:
            return {}
        rail = getattr(self.link, "rail_id", 0)
        ns = getattr(topo, "namespace", "pd:")
        extra: dict[str, float] = {}
        if src_ep is not None:
            tx_name = PortKey(int(topo._node_of(src)), str(src_ep), rail, "tx").resource_name(ns)
            if tx_name in engine.resources:
                extra[tx_name] = wire_bytes
        if dst_ep is not None:
            rx_name = PortKey(int(topo._node_of(dst)), str(dst_ep), rail, "rx").resource_name(ns)
            if rx_name in engine.resources:
                extra[rx_name] = wire_bytes
        return extra

    def _chunk_coupled_demand(self, engine, plan, flow_index: int, chunk,
                              flow_mem: dict[str, float]) -> tuple[dict[str, float], float]:
        """Build coupled wire + declared physical endpoint port + memory demand for one chunk."""
        path = plan.flows[flow_index]
        demand: dict[str, float] = {}
        prop_ms = 0.0
        for hop in chunk.hops:
            prop_ms += float(hop.latency_ms)
            for res_name, amt in hop.demand.items():
                demand[res_name] = demand.get(res_name, 0.0) + float(amt)
        if chunk.hops:
            extra_ports = self._declared_port_demand(
                engine, plan.topology, path.flow.source, path.flow.destination,
                float(chunk.hops[0].wire_bytes))
            for port_name, amt in extra_ports.items():
                demand.setdefault(port_name, amt)
        if flow_mem:
            total_payload = float(path.flow.payload_bytes)
            frac = (float(chunk.flow.useful_bytes) / total_payload) if total_payload > 0.0 else 1.0
            for res_name, total_amt in flow_mem.items():
                share = float(total_amt) * frac
                if share > 0.0:
                    demand[res_name] = demand.get(res_name, 0.0) + share
        return demand, prop_ms

    def _execute_plan(
        self, engine, name, plan, ready_by_rank, *,
        memory_work=(), transfers=None, owner="",
        per_chunk_pool: tuple[str, float] | None = None,
        per_chunk_releases=None,
        on_chunk_arrived=None,
    ):
        """Execute framed flows with coupled src/wire/dst streaming and causal credits."""
        from ..engine import Allocation

        expected = _validate_plan_chunks(plan)
        n_flows = len(plan.flows)
        flow_mems = _build_flow_memory_weights(plan, memory_work, transfers=transfers)
        completions = [engine.future(f"{name}:flow{i}:complete") for i in range(n_flows)]
        arrived_count = [0] * n_flows
        issued = [0] * n_flows
        prev_grant: list = [None] * n_flows
        prev_service: list = [None] * n_flows
        completed = [False] * n_flows
        pool_name = per_chunk_pool[0] if per_chunk_pool is not None else None
        pool_cap = float(per_chunk_pool[1]) if per_chunk_pool is not None else None
        if pool_name is not None and pool_name not in engine.pools:
            engine.extend(pools={pool_name: pool_cap})
        releases_map = per_chunk_releases or {}
        release_cursor = defaultdict(int)
        windows: list[int | None] = []
        for idx, path in enumerate(plan.flows):
            if not expected[idx]:
                windows.append(0)
            elif pool_cap is None:
                windows.append(None)
            else:
                largest = chunk_size_at(path.flow.payload_bytes, plan.chunk_bytes, 0)
                windows.append(max(1, int(pool_cap // largest + _FLOAT_ABS_TOL)))

        def complete_flow(fi: int):
            if not completed[fi]:
                completed[fi] = True
                if not completions[fi].done:
                    engine.resolve(completions[fi])

        def schedule_per_chunk_release(prefix, dest, useful, done_ev, alloc):
            events = tuple(releases_map.get(dest, ()))
            explicit = dest in releases_map
            cursor = release_cursor[dest]
            while cursor < len(events) and events[cursor].done:
                cursor += 1
            release_cursor[dest] = cursor
            if explicit and cursor >= len(events):
                return
            after = (done_ev, events[cursor]) if cursor < len(events) else (done_ev,)
            def on_rel(_ev, d=dest, u=useful, a=alloc):
                self._track_consumption(d, u)
                if a is not None:
                    self._track_reserved_release(d, a.size)
            free_kw = {"free": (alloc,)} if alloc is not None else {}
            engine.submit(
                prefix + ":release", after=after, **free_kw,
                owner=owner, on_complete=on_rel)

        def on_chunk_done(fi: int, dest: int, useful: float, prefix: str, alloc, done_ev):
            arrived_count[fi] += 1
            if on_chunk_arrived is not None:
                on_chunk_arrived(fi, dest, useful, done_ev)
            elif pool_name is not None:
                self._track_arrival(dest, useful)
                schedule_per_chunk_release(prefix, dest, useful, done_ev, alloc)
            pump(fi)
            if arrived_count[fi] == expected[fi]:
                complete_flow(fi)

        def issue_chunk(fi: int, ci: int):
            path = plan.flows[fi]
            chunk = path.chunks[ci]
            dest = int(path.flow.destination)
            src = int(path.flow.source)
            useful = float(chunk.flow.useful_bytes)
            prefix = f"{name}:flow{fi}:chunk{ci}"
            alloc = None
            if pool_name is not None:
                grant_deps = [ready_by_rank[src], ready_by_rank[dest]]
                if prev_grant[fi] is not None:
                    grant_deps.append(prev_grant[fi])
                alloc = Allocation(prefix + ":credit", pool_name, useful, (0,))
                s_ev, d_ev = ready_by_rank[src], ready_by_rank[dest]
                def on_grant(_ev, d=dest, u=useful, se=s_ev, de=d_ev):
                    self._track_reserved_acquire(d, u)
                    ready_at = max(se.time or 0.0, de.time or 0.0)
                    if float(engine.now) - ready_at > _FLOAT_REL_TOL:
                        self._transport["credit_waits"] += 1
                grant_ev = engine.submit(
                    prefix + ":grant", after=tuple(grant_deps), acquire=(alloc,),
                    owner=owner, on_complete=on_grant)
                prev_grant[fi] = grant_ev
            else:
                gate_deps = [ready_by_rank[src], ready_by_rank[dest]]
                if prev_service[fi] is not None:
                    gate_deps.append(prev_service[fi])
                grant_ev = engine.submit(prefix + ":gate", after=tuple(gate_deps), owner=owner)
            demand, prop_ms = self._chunk_coupled_demand(engine, plan, fi, chunk, flow_mems[fi])
            startup_ms = (
                plan._port_stages(path) * float(plan.hop_startup_us) / _MS_PER_S
                if ci == 0 and plan.hop_startup_us else 0.0
            )
            delay_ms = (prop_ms if (pool_name is not None or ci == 0) else 0.0) + startup_ms
            if delay_ms > 0.0:
                gate_ev = engine.submit(
                    prefix + ":prop", after=(grant_ev,), latency=delay_ms, owner=owner)
            else:
                gate_ev = grant_ev
            svc_deps = (gate_ev, prev_service[fi]) if prev_service[fi] is not None else (gate_ev,)
            svc = engine.submit(
                prefix + ":service", demand, after=svc_deps, owner=owner, coupled=True,
                on_complete=lambda ev, f=fi, d=dest, u=useful, p=prefix, a=alloc:
                    on_chunk_done(f, d, u, p, a, ev))
            prev_service[fi] = svc

        def pump(fi: int):
            if expected[fi] == 0:
                return
            win = windows[fi]
            limit = (
                min(expected[fi], arrived_count[fi] + 1)
                if win is None
                else min(expected[fi], arrived_count[fi] + win + 1)
            )
            while issued[fi] < limit:
                ci = issued[fi]
                issued[fi] += 1
                issue_chunk(fi, ci)

        for fi, path in enumerate(plan.flows):
            if not expected[fi]:
                engine.submit(
                    f"{name}:flow{fi}:empty",
                    after=(ready_by_rank[path.flow.source], ready_by_rank[path.flow.destination]),
                    owner=owner, on_complete=lambda _ev, f=fi: complete_flow(f))
            else:
                pump(fi)
        return tuple(completions)

    def submit_handoff(self, engine, handoff, *, source_ready_by_rank,
                       destination_ready_by_rank, release_events=None,
                       source_last_use_by_rank=None,
                       source_release_callbacks=None,
                       release_callbacks_by_rank=None,
                       name=None):
        """Submit an actual request/rank/stage handoff binding."""
        if not isinstance(handoff, PDHandoff):
            raise ModelingError("pd_submit_handoff_requires_PDHandoff")
        name = name or f"pd:{handoff.request_id}"
        if not handoff.transfers:
            return engine.submit(
                name + ":empty", owner=f"request:{handoff.request_id}",
                metadata={"boundary": "pd_empty_handoff",
                          "request_id": handoff.request_id})
        if any(item.source_rank == item.destination_rank
               for item in handoff.transfers):
            raise ModelingError(
                "pd_handoff_self_flow_requires_distinct_pool_ranks")
        sources = {int(rank): event for rank, event in
                   dict(source_ready_by_rank).items()}
        destinations = {int(rank): event for rank, event in
                        dict(destination_ready_by_rank).items()}
        missing_sources = sorted({item.source_rank for item in handoff.transfers}
                                 - set(sources))
        missing_destinations = sorted({item.destination_rank for item in handoff.transfers}
                                      - set(destinations))
        if missing_sources or missing_destinations:
            raise ModelingError(
                "pd_handoff_missing_rank_ready:"
                f"source={missing_sources}:destination={missing_destinations}")
        per_rank_pools = self._preflight_handoff_pools(engine, handoff)
        topology = self._topology_for_handoff(handoff)
        transport = TransportOptions(
            chunk_bytes=self.chunk_bytes,
            hop_startup_us=self.hop_startup_us or 0.0,
            receive_buffer_ids=handoff.receive_buffer_ids,
            max_physical_chunks=_MAX_PHYSICAL_CHUNKS)
        plan = self._alias_plan(topology.transport_plan(handoff.flows(), transport=transport))
        _validate_plan_chunks(plan)
        self._ensure_engine_resources(engine, topology)

        source_last_use = {
            int(rank): (tuple(val) if isinstance(val, (tuple, list)) else (val,))
            for rank, val in dict(source_last_use_by_rank or {}).items()
        }
        source_callbacks = {int(rank): cb for rank, cb
                            in dict(source_release_callbacks or {}).items()}
        release_callbacks = {int(rank): cb for rank, cb
                             in dict(release_callbacks_by_rank or {}).items()}

        rank_ready, reservations = self._reserve_handoff_receivers(
            engine, handoff, name, sources, destinations, per_rank_pools)
        release_map = _normalize_releases(release_events) or {}
        flow_index_by_destination = defaultdict(list)
        hbm_flow_indices = set()
        for index, transfer in enumerate(handoff.transfers):
            if transfer.destination_tier == "hbm":
                flow_index_by_destination[transfer.destination_rank].append(index)
                hbm_flow_indices.add(index)
        chunk_handlers: dict[int, object] = {}

        def dispatch_chunk(fi: int, dest: int, useful: float, ev):
            if fi in hbm_flow_indices:
                handler = chunk_handlers.get(dest)
                if handler is not None:
                    handler(useful, float(ev.time if ev.time is not None else engine.now))

        completions = self._execute_plan(
            engine, name + ":transport", plan, rank_ready,
            memory_work=handoff.memory_work, transfers=handoff.transfers,
            owner=f"request:{handoff.request_id}",
            on_chunk_arrived=dispatch_chunk)
        source_flow_events = defaultdict(list)
        for index, event in enumerate(completions):
            source_flow_events[handoff.transfers[index].source_rank].append(event)
        for destination, allocation in reservations.items():
            indices = tuple(flow_index_by_destination.get(destination, ()))
            flow_events = tuple(completions[index] for index in indices)
            consumer_events = tuple(release_map.get(destination, ()))
            amount = sum(float(handoff.transfers[index].payload_bytes)
                         for index in indices)
            _, on_chunk = self._schedule_residency(
                engine, f"{name}:r{destination}", destination, amount,
                flow_events, consumer_events, allocation=allocation,
                release_name=f"{name}:receive_release:r{destination}",
                explicit_consumer=True,
                release_callback=release_callbacks.get(destination))
            chunk_handlers[destination] = on_chunk
        source_release_events = {}
        for rank in sorted(source_flow_events):
            explicit = source_last_use.get(rank, ())
            release = engine.submit(
                f"{name}:source_last_use:r{rank}",
                after=(sources[rank], *source_flow_events[rank], *explicit),
                owner=f"request:{handoff.request_id}",
                metadata={"boundary": "pd_source_last_use_release",
                          "request_id": handoff.request_id,
                          "source_rank": rank,
                          "explicit_local_readers": len(explicit)},
                on_complete=(
                    (lambda _event, callback=source_callbacks.get(rank):
                     callback() if callback is not None else None)
                    if rank in source_callbacks else None))
            source_release_events[rank] = release
        return self._finish_handoff(
            engine, name, handoff, plan, completions, source_release_events)

    def _reserve_handoff_receivers(
        self, engine, handoff, name, sources, destinations, per_rank_pools,
    ):
        pair_ready = defaultdict(list)
        for transfer in handoff.transfers:
            pair = engine.submit(
                f"{name}:pair{transfer.source_rank}>{transfer.destination_rank}:ready",
                after=(sources[transfer.source_rank],
                       destinations[transfer.destination_rank]),
                owner=f"request:{handoff.request_id}")
            pair_ready[transfer.source_rank].append(pair)
            pair_ready[transfer.destination_rank].append(pair)
        from ..engine import Allocation
        reservations = {}
        allocations = []
        for destination, (buffer_id, footprint, pool_cap) in sorted(per_rank_pools.items()):
            pool = self._receive_pool_name(buffer_id)
            if pool not in engine.pools:
                engine.extend(pools={pool: float(pool_cap)})
            self._buffer_ranks[buffer_id].add(int(destination))
            allocation = Allocation(
                f"{name}:rx:r{destination}", pool, float(footprint), (0,))
            reservations[destination] = allocation
            allocations.append(allocation)
        if allocations:
            reserve_ready = engine.submit(
                f"{name}:receive_ready",
                after=tuple(event for destination in sorted(per_rank_pools)
                            for event in pair_ready[destination]),
                owner=f"request:{handoff.request_id}")
            reserved_pairs = tuple(
                (destination, allocation.size)
                for destination, allocation in sorted(reservations.items())
            )
            def on_grant(_event):
                self._track_reservation(reserve_ready)
                for dest, amt in reserved_pairs:
                    self._track_reserved_acquire(dest, amt)
            grant = engine.submit(
                f"{name}:receive_grant", after=(reserve_ready,),
                acquire=tuple(allocations),
                owner=f"request:{handoff.request_id}",
                on_complete=on_grant,
                metadata={"boundary": "pd_whole_message_receive_grant",
                          "request_id": handoff.request_id,
                          "destination_ranks": sorted(per_rank_pools),
                          "receive_buffer_ids": {
                              str(rank): per_rank_pools[rank][0]
                              for rank in sorted(per_rank_pools)},
                          "footprint_bytes": {
                              str(rank): per_rank_pools[rank][1]
                              for rank in sorted(per_rank_pools)},
                          "pool_capacity_bytes": {
                              str(rank): per_rank_pools[rank][2]
                              for rank in sorted(per_rank_pools)}})
            for rank in pair_ready:
                pair_ready[rank].append(grant)
        rank_ready = {
            rank: engine.submit(f"{name}:ready:r{rank}",
                                after=tuple(pair_ready[rank]),
                                owner=f"request:{handoff.request_id}")
            for rank in sorted(pair_ready)}
        return rank_ready, reservations

    def _finish_handoff(self, engine, name, handoff, plan, completions, source_release_events):
        self._engine = engine
        wire, transactions, chunks = _plan_wire_and_transactions(
            plan, track_transactions=self.transactions is not None)
        self.wire_bytes += wire
        if self.transactions is not None:
            self.transactions += transactions
        self._transport["chunks"] += chunks
        self._transport["unknowns"].update(key for key, _ in plan.unknowns)
        self._handoff_count += 1
        self._handoff_flows += len(handoff.transfers)
        self._handoff_ranks.update(handoff.ranks)
        self._last_receive_capacity = dict(handoff.receive_capacity_bytes)
        self._last_receive_buffers = dict(handoff.receive_buffer_ids)
        for rank, buffer_id in plan.receive_buffer_ids:
            self._buffer_ranks[buffer_id].add(int(rank))
        return engine.submit(
            name, after=completions,
            owner=f"request:{handoff.request_id}",
            metadata={
                "boundary": "pd_pool_connector_handoff",
                "request_id": handoff.request_id,
                "useful_bytes": sum(float(item.payload_bytes)
                                    for item in handoff.transfers),
                "host_bytes": sum(float(item.payload_bytes)
                                  for item in handoff.host_transfers()),
                "hbm_bytes": sum(float(item.payload_bytes)
                                 for item in handoff.hbm_transfers()),
                "wire_bytes": wire,
                "transactions": transactions,
                "chunks": chunks,
                "flows": len(handoff.transfers),
                "receive_buffer_ids": dict(handoff.receive_buffer_ids),
                "source_release_ranks": sorted(source_release_events),
                "transport": "common_topology_transport_plan",
            })

    def submit(self, engine, name, volume, memory_work, *,
               ready=None, destination_ready=None, release_events=None):
        """Submit one handoff under one common source/destination boundary."""
        releases = _normalize_releases(release_events)
        declared_consumer = release_events is not None
        pool = self._receive_pool_name(self.receive_buffer_id)
        effective_rx_cap = self.receive_capacity_bytes
        if effective_rx_cap is not None:
            self._check_pool_capacity(engine, pool, float(effective_rx_cap), self.receive_buffer_id)
        elif pool in engine.pools:
            effective_rx_cap = float(engine.pools[pool])
        if (declared_consumer and effective_rx_cap is not None
                and float(effective_rx_cap) + _FLOAT_ABS_TOL < float(volume)):
            raise ModelingError(
                "pd_finite_receive_credit_requires_per_chunk_consumer:"
                f"capacity={effective_rx_cap}:volume={volume}")
        single_chunk = self.chunk_bytes is None or float(self.chunk_bytes) >= float(volume)
        if (not declared_consumer and effective_rx_cap is not None):
            largest = float(volume) if single_chunk else float(self.chunk_bytes)
            if float(effective_rx_cap) + _FLOAT_ABS_TOL < largest:
                raise ModelingError(
                    "pd_whole_message_footprint_exceeds_pool:"
                    f"{pool}:{largest}>{effective_rx_cap}")
        use_transport = self.transport_declared or effective_rx_cap is not None
        if not use_transport:
            self._engine = engine
            boundary = self._submit_common_boundary(engine, name, ready, destination_ready)
            return self._submit_legacy(
                engine, name, volume, memory_work, boundary=boundary, releases=releases)

        use_reserved = (
            effective_rx_cap is not None
            and float(effective_rx_cap) + _FLOAT_ABS_TOL >= float(volume)
            and (declared_consumer or single_chunk)
        )
        transport = TransportOptions(
            chunk_bytes=self.chunk_bytes,
            hop_startup_us=self.hop_startup_us or 0.0,
            receive_capacity_bytes=(
                ((1, float(effective_rx_cap)),)
                if (not use_reserved and effective_rx_cap is not None) else ()
            ),
            receive_buffer_ids=((1, self.receive_buffer_id),),
            max_physical_chunks=_MAX_PHYSICAL_CHUNKS)
        plan = self._alias_plan(self.topology.transport_plan(
            [MessageFlow(0, 1, volume)], transport=transport))
        _validate_plan_chunks(plan)

        self._engine = engine
        boundary = self._submit_common_boundary(engine, name, ready, destination_ready)
        if use_reserved:
            return self._submit_reserved_single(
                engine, name, volume, memory_work, plan=plan,
                pool_capacity=float(effective_rx_cap), boundary=boundary,
                releases=releases, declared_consumer=declared_consumer)

        per_chunk_pool = (
            (pool, float(effective_rx_cap)) if effective_rx_cap is not None else None
        )
        if per_chunk_pool is not None:
            self._buffer_ranks[self.receive_buffer_id].add(1)
        completions = self._execute_plan(
            engine, name + ":transport", plan, {0: boundary, 1: boundary},
            memory_work=memory_work, per_chunk_pool=per_chunk_pool,
            per_chunk_releases=releases)
        wire, transactions, chunks = _plan_wire_and_transactions(
            plan, track_transactions=self.transactions is not None)
        self.wire_bytes += wire
        if self.transactions is not None:
            self.transactions += transactions
        self._transport["chunks"] += chunks
        self._transport["unknowns"].update(key for key, _ in plan.unknowns)
        for rank, buffer_id in plan.receive_buffer_ids:
            self._buffer_ranks[buffer_id].add(int(rank))
        return engine.submit(
            name, after=completions, metadata={
                "boundary": "pd_pool_connector",
                "useful_bytes": volume,
                "wire_bytes": wire,
                "chunks": chunks,
                "transactions": transactions,
                "transport": "common_topology_transport_plan",
            })

    @staticmethod
    def _submit_common_boundary(engine, name, ready, destination_ready):
        source_ready = ready if ready is not None else engine.submit(name + ":source_ready")
        receiver_ready = destination_ready if destination_ready is not None else source_ready
        return engine.submit(
            name + ":ready", after=(source_ready, receiver_ready),
            metadata={"boundary": "pd_common_source_destination_ready",
                      "source_ready": source_ready.name,
                      "destination_ready": receiver_ready.name})

    def _submit_reserved_single(self, engine, name, volume, memory_work, *,
                                plan, pool_capacity, boundary, releases,
                                declared_consumer=False):
        """Hold one complete receive footprint on the shared physical receive pool."""
        from ..engine import Allocation
        pool = self._receive_pool_name(self.receive_buffer_id)
        if pool not in engine.pools:
            engine.extend(pools={pool: pool_capacity})
        self._buffer_ranks[self.receive_buffer_id].add(1)
        allocation = Allocation(f"{name}:rx", pool, float(volume), (0,))
        def on_grant(_event):
            self._track_reservation(boundary)
            self._track_reserved_acquire(1, allocation.size)
        grant = engine.submit(
            name + ":receive_grant", after=(boundary,), acquire=(allocation,),
            metadata={"boundary": "pd_whole_message_receive_grant",
                      "receive_buffer_id": self.receive_buffer_id,
                      "pool_capacity_bytes": pool_capacity,
                      "footprint_bytes": volume},
            on_complete=on_grant)
        chunk_cb = [None]
        def dispatch_chunk(_fi, _dest, useful, ev):
            if chunk_cb[0] is not None:
                chunk_cb[0](useful, float(ev.time if ev.time is not None else engine.now))
        completions = self._execute_plan(
            engine, name + ":transport", plan, {0: grant, 1: grant},
            memory_work=memory_work, on_chunk_arrived=dispatch_chunk)
        consumer_events = tuple((releases or {}).get(1, ()))
        _, on_chunk = self._schedule_residency(
            engine, name, 1, float(volume), completions, consumer_events,
            allocation=allocation,
            release_name=f"{name}:receive_release:r1",
            explicit_consumer=declared_consumer)
        chunk_cb[0] = on_chunk
        wire, transactions, chunks = _plan_wire_and_transactions(
            plan, track_transactions=self.transactions is not None)
        self.wire_bytes += wire
        if self.transactions is not None:
            self.transactions += transactions
        self._transport["chunks"] += chunks
        self._transport["unknowns"].update(key for key, _ in plan.unknowns)
        self._handoff_count += 1
        self._handoff_flows += 1
        self._handoff_ranks.update((0, 1))
        self._last_receive_capacity = {1: pool_capacity}
        self._last_receive_buffers = {1: self.receive_buffer_id}
        for rank, buffer_id in plan.receive_buffer_ids:
            self._buffer_ranks[buffer_id].add(int(rank))
        return engine.submit(
            name, after=completions, metadata={
                "boundary": "pd_pool_connector_reserved",
                "useful_bytes": volume, "wire_bytes": wire,
                "transactions": transactions, "chunks": chunks,
                "receive_buffer_id": self.receive_buffer_id,
                "transport": "whole_message_receive_reservation",
            })

    def _submit_legacy(self, engine, name, volume, memory_work, *,
                       boundary, releases=None):
        hop, = self.topology.route_message(MessageFlow(0, 1, volume)).hops
        propagation = engine.submit(
            name + ":latency", after=(boundary,), latency=hop.latency_ms,
            metadata={"boundary": "pd_link_propagation_once"})
        demand = {self.alias.get(key, key): float(value) for key, value in hop.demand.items()}
        for port_name, amt in self._declared_port_demand(
                engine, self.topology, 0, 1, float(hop.wire_bytes)).items():
            demand.setdefault(port_name, amt)
        src_items, dst_items, shared_items = _partition_memory_work(memory_work)
        for key, amount, _ in (*src_items, *dst_items):
            demand[key] = demand.get(key, 0.0) + amount
        for key, amount in shared_items:
            demand[key] = demand.get(key, 0.0) + amount
        self.wire_bytes += hop.wire_bytes
        if self.transactions is not None:
            self.transactions += hop.transactions
        wire = engine.submit(name + ":wire", demand, after=(propagation,), coupled=True)
        complete = engine.submit(name, after=(wire,), metadata={
            "boundary": "pd_pool_connector", "useful_bytes": volume,
            "wire_bytes": hop.wire_bytes, "transactions": hop.transactions,
        })
        destination_releases = (releases or {}).get(1, ())
        if destination_releases:
            engine.submit(name + ":consume", after=(wire, *destination_releases),
                          metadata={"boundary": "pd_consumer_release",
                                    "consumer_events": len(destination_releases)})
        return complete

    def summary(self):
        result = {
            "boundary": "pd_pool_connector", "endpoints": {"0": "prefill", "1": "decode"},
            "message_boundary": "one_complete_destination_sharded_handoff",
            "memory_boundary": "one_local_shard_per_pool_replica_stage",
            "link": self.link.to_dict(), "wire_bytes": self.wire_bytes,
            "transactions": self.transactions,
            "chunk_bytes": self.chunk_bytes,
            "receive_capacity_bytes": self.receive_capacity_bytes,
            "hop_startup_us": self.hop_startup_us,
            "handoffs": self._handoff_count,
            "handoff_flows": self._handoff_flows,
            "handoff_ranks": sorted(self._handoff_ranks),
            "handoff_receive_capacity_bytes": {
                str(rank): value for rank, value in
                sorted(self._last_receive_capacity.items())},
            "handoff_receive_buffer_ids": {
                str(rank): value for rank, value in
                sorted(self._last_receive_buffers.items())},
        }
        if self._transport["chunks"] or self._transport["credit_waits"]:
            reserved = {
                str(rank): value for rank, value in
                self._transport["peak_reserved"].items()
            }
            shared_pool_seen = False
            if self._engine is not None:
                for pool, value in self._engine.peak_allocated.items():
                    if str(pool).startswith("transport:rx:"):
                        shared_pool_seen = True
                        buffer_id = str(pool)[len("transport:rx:"):]
                        ranks = self._buffer_ranks.get(buffer_id)
                        if ranks and len(ranks) == 1:
                            rank_key = str(next(iter(ranks)))
                            if value > reserved.get(rank_key, 0.0):
                                reserved[rank_key] = value
            result["transport"] = {
                "boundary": "common_topology_transport_plan",
                "capacity_scope": ("engine_pool_transport:rx:<destination>; "
                                   "shared by plans on the same Engine"),
                "reserved_peak_basis": ("engine_pool_peak_granted" if shared_pool_seen
                                        else "max_single_handoff_granted"),
                "residency_peak_basis": (
                    "pd_receive_arrival_minus_consumer"
                    if self._tracked_residency else
                    "max_single_handoff_not_concurrent_sum"),
                "chunks": self._transport["chunks"],
                "credit_waits": self._transport["credit_waits"],
                "reservation_waits": self._transport["reservation_waits"],
                "reservation_wait_ms": self._transport["reservation_wait_ms"],
                "resident_bytes": dict(sorted(
                    (str(rank), value) for rank, value in
                    self._transport["resident"].items())),
                "peak_resident_bytes": dict(sorted(
                    (str(rank), value) for rank, value in
                    self._transport["peak_resident"].items())),
                "peak_reserved_bytes": dict(sorted(reserved.items())),
                "unknowns": sorted(self._transport["unknowns"]),
            }
        return result

    def _track_arrival(self, destination, amount):
        key = int(destination)
        value = self._transport["resident"].get(key, 0.0) + float(amount)
        self._transport["resident"][key] = value
        if value > self._transport["peak_resident"].get(key, 0.0):
            self._transport["peak_resident"][key] = value

    def _track_consumption(self, destination, amount):
        key = int(destination)
        self._transport["resident"][key] = max(
            0.0, self._transport["resident"].get(key, 0.0) - float(amount))

    def _track_reserved_acquire(self, destination, amount):
        key = int(destination)
        value = self._transport["reserved"].get(key, 0.0) + float(amount)
        self._transport["reserved"][key] = value
        if value > self._transport["peak_reserved"].get(key, 0.0):
            self._transport["peak_reserved"][key] = value

    def _track_reserved_release(self, destination, amount):
        key = int(destination)
        self._transport["reserved"][key] = max(
            0.0, self._transport["reserved"].get(key, 0.0) - float(amount))

    def _schedule_residency(self, engine, name, destination, amount,
                            arrivals, consumer_events, allocation=None,
                            release_name=None, explicit_consumer=False,
                            release_callback=None):
        """Drive chunk arrival, bounded early/full consumption, and pool release."""
        self._tracked_residency = True
        events = tuple(consumer_events)
        total = float(amount)
        early_cap = float(self.chunk_bytes) if self.chunk_bytes is not None else total
        state = {"arrived": 0.0, "consumed": 0.0, "released": False}
        completed_consumers: list[dict] = []

        def sync_arrived(new_arrived):
            clamped = min(total, max(state["arrived"], float(new_arrived)))
            delta = clamped - state["arrived"]
            if delta > 0.0:
                state["arrived"] = clamped
                self._track_arrival(destination, delta)

        def apply_consume(delta):
            available = max(0.0, state["arrived"] - state["consumed"])
            step = min(max(0.0, float(delta)), available)
            if step > 0.0:
                state["consumed"] += step
                self._track_consumption(destination, step)
            return step

        def maybe_release():
            if state["released"] or allocation is None:
                return
            if state["consumed"] + _FLOAT_ABS_TOL < total:
                return
            state["released"] = True
            def on_released(_event):
                self._track_reserved_release(destination, allocation.size)
                if release_callback is not None:
                    release_callback()
            engine.submit(
                release_name or f"{name}:receive_release", free=(allocation,),
                metadata={"boundary": "pd_whole_message_receive_release",
                          "destination_rank": destination,
                          "bytes": amount},
                on_complete=on_released)

        def on_chunk_arrived(chunk_bytes: float, at_ms: float):
            sync_arrived(state["arrived"] + float(chunk_bytes))
            for record in completed_consumers:
                ev_time = record["event"].time
                if ev_time is None or ev_time < at_ms - _FLOAT_REL_TOL:
                    continue
                rem_quota = max(0.0, early_cap - record["used"])
                if rem_quota > 0.0:
                    record["used"] += apply_consume(rem_quota)
            maybe_release()

        def on_arrival(_event):
            sync_arrived(total)
            arrival_t = float(arrival_event.time or 0.0)
            if not explicit_consumer and not events:
                apply_consume(total - state["consumed"])
            elif any(rec["event"].time is not None and rec["event"].time >= arrival_t - _FLOAT_REL_TOL
                     for rec in completed_consumers):
                apply_consume(total - state["consumed"])
            maybe_release()

        arrival_event = engine.submit(
            f"{name}:arrival", after=tuple(arrivals),
            metadata={"boundary": "pd_receive_arrival",
                      "destination_rank": destination,
                      "bytes": amount},
            on_complete=on_arrival)
        if explicit_consumer and not events:
            return None, on_chunk_arrived
        if not events:
            return arrival_event, on_chunk_arrived

        def on_consumer(_task, index):
            original = events[index]
            record = {"event": original, "used": 0.0}
            completed_consumers.append(record)
            if arrival_event.done:
                sync_arrived(total)
                record["used"] += apply_consume(total - state["consumed"])
            else:
                record["used"] += apply_consume(early_cap)
            maybe_release()

        for index, event in enumerate(events):
            engine.submit(
                f"{name}:consume{index}", after=(event,),
                metadata={"boundary": "pd_receive_consumer_share",
                          "destination_rank": destination,
                          "consumer_index": index},
                on_complete=lambda _event, i=index: on_consumer(_event, i))
        if allocation is None:
            return None, on_chunk_arrived
        decision = engine.submit(
            f"{name}:release_decision", after=(arrival_event, *events),
            metadata={"boundary": "pd_receive_consumer_coverage",
                      "destination_rank": destination,
                      "bytes": amount},
            on_complete=lambda _event: maybe_release())
        return decision, on_chunk_arrived

    def _track_reservation(self, ready_event):
        ready_ms = ready_event.time or 0.0
        wait = max(0.0, float(self._engine.now) - float(ready_ms))
        if wait > _FLOAT_REL_TOL:
            self._transport["reservation_waits"] += 1
            self._transport["reservation_wait_ms"] += wait
