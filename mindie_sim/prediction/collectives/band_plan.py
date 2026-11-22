"""Unified band-selected collective plan for Engine execution and accounting.

[Role]
  Bridge piecewise ``CollectiveBandSpec`` lowering (``protocol.py``) with
  production ``Execution`` / ``Topology`` so every band-declared collective
  builds ONE ``BandCollectivePlan`` driving both Engine submission and wire
  accounting (zero split-brain between ``joint_plan`` and ``accounting``).

[Invariants]
  * ``start`` / ``start_participants`` execute the exact ``LoweredCollective``
    (including declared ``band.algorithm``, ``band.startup_us`` per
    ``startup_scope``, ``band.credit_bytes``, and per-hop pipelining).
  * ``accounting`` derives ``wire_by_resource``, ``useful_by_resource``,
    ``transaction_frames``, and ``setup_ms`` directly from the same lowered
    transfers—never re-routing independently.
  * Preserves ``validate_path_model`` and ``ranks``/``groups`` interfaces
    expected by ``RingCollective``, ``PairwiseCollective``, and cohort launch.

[Timeouts & Locks]
  Pure in-memory lowering/accounting and Engine task submission; no I/O.

[Side-Effect Policy]
  Submits tasks/pools only to the caller-supplied Engine.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from mindie_sim.schema.common import ModelingError
from ..communication import CommunicationWork
from .protocol import (
    MS_PER_S,
    LoweredCollective,
    lower_collective,
    submit_lowered_collective,
)
from .ring_pairwise import _joint_complete, _ready_by_rank

TWO_PHASES = 2


def api_bytes_from_link_volume(
    family: str, group_size: int, link_bytes: float, metadata: Mapping | None = None
) -> float:
    """Invert per-rank wire volume ``link_bytes`` to caller API bytes/rank."""
    if metadata and metadata.get("api_bytes_per_rank") is not None:
        return float(metadata["api_bytes_per_rank"])
    vol = float(link_bytes)
    if vol <= 0.0 or group_size <= 1:
        return 0.0
    if family == "allreduce":
        return vol * float(group_size) / float(TWO_PHASES * (group_size - 1))
    if family in {"reduce_scatter", "alltoall"}:
        return vol * float(group_size) / float(group_size - 1)
    if family == "allgather":
        return vol / float(group_size - 1)
    return vol


@dataclass(frozen=True)
class BandCollectiveGroup:
    """One physical group inside a ``BandCollectivePlan``."""

    ranks: tuple[int, ...]
    lowered: LoweredCollective
    topology: object = field(repr=False, compare=False)

    @property
    def rounds(self) -> int:
        return sum(phase.rounds for phase in self.lowered.phases)

    def start_participants(self, engine, name: str, ready_by_rank, *, owner: str = ""):
        return submit_lowered_collective(
            engine, name, self.lowered, ready_by_rank, owner=owner
        )


@dataclass(frozen=True)
class BandCollectivePlan:
    """Single band-selected collective plan owning both execution & accounting."""

    family: str
    groups: tuple[BandCollectiveGroup, ...]
    topology: object = field(repr=False, compare=False)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(sorted({rank for group in self.groups for rank in group.ranks}))

    def start(self, engine, name: str, ready, *, owner: str = ""):
        events = self.start_participants(
            engine, name, {rank: ready for rank in self.ranks}, owner=owner
        )
        return _joint_complete(engine, name, events.values(), ready, owner)

    def start_participants(self, engine, name: str, ready_by_rank, *, owner: str = ""):
        _ready_by_rank(self.ranks, ready_by_rank)
        events = {}
        for index, group in enumerate(self.groups):
            part = group.start_participants(
                engine,
                f"{name}:group{index}",
                {rank: ready_by_rank[rank] for rank in group.ranks},
                owner=owner,
            )
            overlap = set(events) & set(part)
            if overlap:
                raise ModelingError(
                    "band_collective_rank_accounted_twice:"
                    + ",".join(map(str, sorted(overlap)))
                )
            events.update(part)
        if set(events) != set(self.ranks):
            raise ModelingError("band_collective_group_completion_missing")
        return events

    def validate_path_model(self, path_model: str) -> None:
        if path_model not in {"direct", "store_forward"}:
            raise ModelingError(f"band_collective_path_model_unsupported:{path_model}")
        if path_model == "direct":
            for group in self.groups:
                for phase in group.lowered.phases:
                    for transfer in phase.transfers:
                        if len(transfer.hops) != 1:
                            raise ModelingError(
                                "ring_collective_multihop_requires_store_forward_declaration"
                            )

    def accounting(self):
        """Derive wire, useful, framing, and setup accounting from this plan."""
        useful_by_rank: dict[int, float] = {}
        useful_by_res: dict[str, float] = {}
        wire_by_res: dict[str, float] = {}
        transactions: dict[str, int] = {}
        transaction_frames: dict[str, tuple[int, int]] = {}
        message_count = 0
        routed_useful = 0.0
        physical_wire = 0.0
        setup_ms = 0.0
        max_path_latency = 0.0

        for group in self.groups:
            lowered = group.lowered
            band = lowered.band
            frame_pair = (
                int(band.payload_bytes_per_frame),
                int(band.wire_bytes_per_frame),
            )
            hop_startup_ms = (
                0.0
                if lowered.uses_calibrated_startup
                else (band.startup_us / MS_PER_S if band.startup_scope == "hop" else 0.0)
            )
            group_setup = lowered.collective_startup_us / MS_PER_S
            for phase in lowered.phases:
                group_setup += phase.startup_us / MS_PER_S
                round_latencies: dict[int, float] = {}
                for transfer in phase.transfers:
                    message_count += 1
                    useful_by_rank[transfer.source] = (
                        useful_by_rank.get(transfer.source, 0.0)
                        + transfer.payload_bytes
                    )
                    path_lat = 0.0
                    for hop in transfer.hops:
                        routed_useful += transfer.payload_bytes
                        physical_wire += transfer.wire_bytes
                        for res in hop.resources:
                            useful_by_res[res] = (
                                useful_by_res.get(res, 0.0) + transfer.payload_bytes
                            )
                            wire_by_res[res] = (
                                wire_by_res.get(res, 0.0) + transfer.wire_bytes
                            )
                            transactions[res] = (
                                transactions.get(res, 0) + transfer.frames
                            )
                            transaction_frames[res] = frame_pair
                        if lowered.uses_calibrated_startup:
                            hop_lat = hop.switch_propagation_ms
                        elif band.startup_us > 0.0:
                            hop_lat = hop_startup_ms + hop.switch_propagation_ms
                        else:
                            hop_lat = hop.propagation_ms
                        path_lat += hop_lat
                    max_path_latency = max(max_path_latency, path_lat)
                    round_latencies[transfer.round_index] = max(
                        round_latencies.get(transfer.round_index, 0.0), path_lat
                    )
                group_setup += sum(round_latencies.values())
            setup_ms = max(setup_ms, group_setup)

        work = CommunicationWork(
            useful_by_rank,
            useful_by_res,
            wire_by_res,
            transactions,
            transaction_frames,
            message_count,
            setup_ms,
            routed_useful,
            physical_wire,
            max_path_latency,
        )
        return self.topology._work_with_capacity(work)

    def cache_key(self) -> tuple:
        """Return a stable structural key for Execution program caching."""
        return (
            self.family,
            tuple(
                (group.ranks, group.lowered.band, group.lowered.api_bytes_per_rank,
                 group.lowered.total_wire_bytes, group.lowered.collective_startup_us,
                 group.lowered.local_hbm_bytes_per_rank)
                for group in self.groups
            ),
        )

    def lower_with_local_work(self, context, base_program):
        """Wrap this lowered collective plan inside preserved local stages."""
        from ..mechanism import Work
        from ..program import KernelProgram, Stage

        if any(stg.program is self for stg in base_program.stages):
            return base_program
        link_names = frozenset(self.accounting().wire_by_resource)
        stages_in = base_program.stages
        # Unwrap outer setup -> body container from TopologyCollectiveRecipe
        if (len(stages_in) == TWO_PHASES
                and stages_in[0].name == "setup"
                and isinstance(stages_in[1].program, KernelProgram)):
            stages_in = stages_in[1].program.stages

        def _strip_links(stg: Stage, after: tuple[str, ...]) -> Stage | None:
            local_work = {k: v for k, v in stg.work.items() if k not in link_names}
            local_caps = {k: v for k, v in stg.caps.items() if k not in link_names}
            local_place = {k: v for k, v in stg.placement.items() if k not in link_names}
            local_units = {k: v for k, v in stg.work_units.items() if k not in link_names}
            local_widened = tuple(k for k in stg.cohort_widened if k not in link_names)
            local_members = {k: v for k, v in stg.cohort_members.items() if k not in link_names}
            local_deliv = tuple((k, v) for k, v in stg.deliveries if k not in link_names)
            has_local = bool(
                local_work or stg.metadata or local_deliv or stg.acquire
                or stg.free or stg.latency > 0.0 or stg.program is not None
                or stg.fragment is not None
            )
            if not has_local:
                return None
            return replace(
                stg,
                work=local_work,
                caps=local_caps,
                placement=local_place,
                work_units=local_units,
                cohort_widened=local_widened,
                cohort_members=local_members,
                deliveries=local_deliv,
                coupled=stg.coupled and bool(local_work),
                after=after,
                keep_empty=True,
            )

        def _remap_after(raw: tuple[str, ...], inherited: dict[str, tuple[str, ...]], valid: set[str]) -> tuple[str, ...]:
            deps: list[str] = []
            for dep in raw:
                for target in inherited.get(dep, (dep,)):
                    if target in valid and target not in deps:
                        deps.append(target)
            return tuple(deps)

        pre_mid = [s for s in stages_in if s.name != "unpack_store"]
        unpack_stages = [s for s in stages_in if s.name == "unpack_store"]

        out: list[Stage] = []
        valid: set[str] = set()
        inherited: dict[str, tuple[str, ...]] = {}
        consumed_by_downstream: set[str] = set()
        for stg in pre_mid:
            eff_after = _remap_after(stg.after, inherited, valid)
            kept = _strip_links(stg, eff_after)
            if kept is None:
                inherited[stg.name] = eff_after
                continue
            out.append(kept)
            valid.add(kept.name)
            consumed_by_downstream.update(eff_after)

        terminals = tuple(s.name for s in out if s.name not in consumed_by_downstream)
        reserved = valid | {s.name for s in unpack_stages}
        plan_name = "band_collective" if "band_collective" not in reserved else "band_collective:plan"
        out.append(Stage(plan_name, after=terminals, program=self, keep_empty=True))

        post_root = plan_name
        local_hbm = max(
            (float(g.lowered.local_hbm_bytes_per_rank) for g in self.groups),
            default=0.0,
        )
        op_hbm = float(getattr(getattr(context, "op", None), "hbm_bytes", 0.0) or 0.0)
        bind_fn = getattr(context, "bind", None)
        res_names = getattr(context, "resource_names", ()) or ()
        if local_hbm > 0.0 and op_hbm <= 0.0 and callable(bind_fn) and "hbm" in res_names:
            hbm_stage_name = "band_collective:local_hbm"
            out.append(Stage(
                hbm_stage_name,
                work=bind_fn(Work(resources={"hbm": local_hbm})),
                after=(plan_name,),
            ))
            post_root = hbm_stage_name

        post_valid: set[str] = {post_root}
        post_inherited: dict[str, tuple[str, ...]] = {}
        for stg in unpack_stages:
            internal = _remap_after(stg.after, post_inherited, post_valid)
            eff_after = internal if post_root in internal else (post_root, *internal)
            kept = _strip_links(stg, eff_after)
            if kept is None:
                post_inherited[stg.name] = eff_after
                continue
            out.append(kept)
            post_valid.add(kept.name)

        return KernelProgram(tuple(out), minimum_ms=context.minimum_ms)


def build_band_collective_plan(
    topology,
    family: str,
    unique_groups: tuple[tuple[int, ...], ...],
    volume_by_rank: Mapping[int, float],
    metadata: Mapping,
    *,
    peer=None,
    stack: str = "default",
    expansion_mode: str | None = None,
) -> BandCollectivePlan:
    """Lower all groups in a cohort through ``lower_collective`` into one plan."""
    stack = str(metadata.get("collective_stack", stack) or stack)
    groups: list[BandCollectiveGroup] = []
    for idx, ranks in enumerate(unique_groups):
        if family == "alltoallv" or peer is not None:
            if peer is None:
                raise ModelingError("pairwise_collective_alltoallv_requires_explicit_peer_matrix")
            matrix = tuple(
                tuple(int(round(peer.payload(r, c))) for c in range(len(ranks)))
                for r in range(len(ranks))
            )
            send_splits = tuple(sum(row) for row in matrix)
            recv_splits = tuple(
                sum(matrix[r][c] for r in range(len(ranks)))
                for c in range(len(ranks))
            )
            api_bytes = float(max(send_splits)) if send_splits else 0.0
            lowered = lower_collective(
                "alltoallv", ranks, topology, api_bytes,
                stack=stack, expansion_mode=expansion_mode,
                peer_matrix=matrix, send_splits=send_splits, recv_splits=recv_splits,
                communicator_id=str(metadata.get("collective_id", "")),
                coset_id=f"g{idx}",
            )
        else:
            volumes = {rank: volume_by_rank[rank] for rank in ranks}
            if len(set(volumes.values())) > 1:
                if family != "alltoall":
                    raise ModelingError("ring_collective_requires_equal_per_rank_volume")
                n_ranks = len(ranks)
                matrix = tuple(
                    tuple(
                        0 if r_i == c_i else int(round(volumes[ranks[r_i]] / float(n_ranks - 1)))
                        for c_i in range(n_ranks)
                    )
                    for r_i in range(n_ranks)
                )
                send_splits = tuple(sum(row) for row in matrix)
                recv_splits = tuple(
                    sum(matrix[r_i][c_i] for r_i in range(n_ranks))
                    for c_i in range(n_ranks)
                )
                api_bytes = float(max(send_splits)) if send_splits else 0.0
                lowered = lower_collective(
                    "alltoallv", ranks, topology, api_bytes,
                    stack=stack, expansion_mode=expansion_mode,
                    peer_matrix=matrix, send_splits=send_splits, recv_splits=recv_splits,
                    communicator_id=str(metadata.get("collective_id", "")),
                    coset_id=f"g{idx}",
                )
            else:
                vol = next(iter(volumes.values()), 0.0)
                api_bytes = api_bytes_from_link_volume(family, len(ranks), vol, metadata)
                lowered = lower_collective(
                    family, ranks, topology, api_bytes,
                    stack=stack, expansion_mode=expansion_mode,
                    communicator_id=str(metadata.get("collective_id", "")),
                    coset_id=f"g{idx}",
                )
        groups.append(BandCollectiveGroup(ranks=ranks, lowered=lowered, topology=topology))
    return BandCollectivePlan(family=family, groups=tuple(groups), topology=topology)
