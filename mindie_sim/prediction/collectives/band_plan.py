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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math

from mindie_sim.schema.common import ModelingError
from ..communication import CommunicationWork
from ..mechanism import Work
from .protocol import (
    MS_PER_S,
    LoweredCollective,
    lower_collective,
    submit_lowered_collective,
)
from .ring_pairwise import _joint_complete, _ready_by_rank

TWO_PHASES = 2
_POST_NETWORK_STAGE_NAMES = frozenset({
    "reduce_scatter", "all_gather", "peer_exchange", "unpack_store",
})


def _unwrap_collective_stages(stages_in):
    from ..program.stages import KernelProgram

    if (len(stages_in) == TWO_PHASES
            and stages_in[0].name == "setup"
            and isinstance(stages_in[1].program, KernelProgram)):
        return stages_in[1].program.stages
    return stages_in


def _strip_stage_links(
    stg, after: tuple[str, ...], link_names: frozenset[str], *, exclude_band_plan: bool = False,
):
    if exclude_band_plan and (
        isinstance(stg.program, BandCollectivePlan) or stg.name == "band_collective:local_hbm"
    ):
        return None
    local_work = {k: v for k, v in stg.work.items() if k not in link_names}
    local_caps = {k: v for k, v in stg.caps.items() if k not in link_names}
    local_place = {k: v for k, v in stg.placement.items() if k not in link_names}
    local_units = {k: v for k, v in stg.work_units.items() if k not in link_names}
    local_widened = tuple(k for k in stg.cohort_widened if k not in link_names)
    local_members = {k: v for k, v in stg.cohort_members.items() if k not in link_names}
    local_deliv = tuple((k, v) for k, v in stg.deliveries if k not in link_names)
    has_prog = (
        stg.program is not None and not isinstance(stg.program, BandCollectivePlan)
        if exclude_band_plan else stg.program is not None
    )
    has_local = bool(
        local_work or stg.metadata or local_deliv or stg.acquire
        or stg.free or stg.latency > 0.0 or has_prog or stg.fragment is not None
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


def _remap_stage_after(
    raw: tuple[str, ...], inherited: dict[str, tuple[str, ...]], valid: set[str]
) -> tuple[str, ...]:
    deps: list[str] = []
    for dep in raw:
        for target in inherited.get(dep, (dep,)):
            if target in valid and target not in deps:
                deps.append(target)
    return tuple(deps)


def _partition_pre_post_stages(stages_in, link_names: frozenset[str]):
    """Split stages into pre-network vs link-coupled/post-network stages."""
    pre_stages = []
    post_stages = []
    post_names: set[str] = set()
    for stg in stages_in:
        is_post = (
            isinstance(stg.program, BandCollectivePlan)
            or stg.name == "band_collective:local_hbm"
            or stg.name in _POST_NETWORK_STAGE_NAMES
            or bool(set(stg.work) & link_names)
            or any(dep in post_names for dep in stg.after)
        )
        if is_post:
            post_names.add(stg.name)
            post_stages.append(stg)
        else:
            pre_stages.append(stg)
    return pre_stages, post_stages


def _with_recomputed_phases(
    lowered: LoweredCollective, phases, **extra,
) -> LoweredCollective:
    """Replace phases on ``lowered`` and recompute wire/payload totals."""
    per_rank = {rank: 0.0 for rank in lowered.group}
    total_wire = 0.0
    for phase in phases:
        for transfer in phase.transfers:
            if transfer.source in per_rank:
                per_rank[transfer.source] += transfer.payload_bytes
            total_wire += transfer.wire_bytes
    return replace(
        lowered,
        phases=tuple(phases),
        total_payload_bytes=math.fsum(per_rank.values()),
        total_wire_bytes=total_wire,
        per_rank_payload_bytes=tuple((r, per_rank[r]) for r in lowered.group),
        **extra,
    )


def _phase_slice_plan(plan: BandCollectivePlan, phase_index: int) -> BandCollectivePlan:
    """Slice one phase out of a multi-phase plan, charging collective startup once."""
    sliced_groups = tuple(
        replace(
            group,
            lowered=_with_recomputed_phases(
                group.lowered,
                (group.lowered.phases[phase_index],),
                collective_startup_us=(
                    group.lowered.collective_startup_us if phase_index == 0 else 0.0
                ),
            ),
        )
        for group in plan.groups
    )
    return replace(plan, groups=sliced_groups)


def _integer_peer_row(raw_row: Sequence[float], row_idx: int) -> tuple[int, ...]:
    """Apportion non-negative peer cells to integers conserving exact row sum."""
    n_ranks = len(raw_row)
    result = [0] * n_ranks
    diag = int(round(raw_row[row_idx])) if n_ranks else 0
    if n_ranks:
        result[row_idx] = diag
    off_cols = [c for c in range(n_ranks) if c != row_idx]
    if not off_cols:
        return tuple(result)
    off_vals = [float(raw_row[c]) for c in off_cols]
    target = int(round(math.fsum(off_vals)))
    floors = [int(math.floor(v)) for v in off_vals]
    rem = target - sum(floors)
    for c, base_val in zip(off_cols, floors):
        result[c] = base_val
    if rem > 0:
        order = sorted(
            range(len(off_cols)),
            key=lambda i: (-(off_vals[i] - floors[i]), (off_cols[i] - row_idx) % n_ranks),
        )
        for i in order[:rem]:
            result[off_cols[i]] += 1
    return tuple(result)


def _snap_integral(value: float) -> float:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-9):
        return float(nearest)
    return float(value)


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
        return _snap_integral(vol * float(group_size) / float(TWO_PHASES * (group_size - 1)))
    if family in {"reduce_scatter", "alltoall"}:
        return _snap_integral(vol * float(group_size) / float(group_size - 1))
    if family == "allgather":
        return _snap_integral(vol / float(group_size - 1))
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
                 group.lowered.local_hbm_bytes_per_rank,
                 tuple(
                     (p.name, len(p.transfers), tuple(sorted({t.source for t in p.transfers})))
                     for p in group.lowered.phases
                 ))
                for group in self.groups
            ),
        )

    def scoped_to_ranks(
        self,
        local_ranks: frozenset[int],
        *,
        available_resources: Mapping[str, object] | frozenset[str] | set[str],
    ) -> BandCollectivePlan | None:
        """Retain local-rank/relay transfers and member HBM staging on an isolated engine."""
        scoped_groups: list[BandCollectiveGroup] = []
        for group in self.groups:
            has_member = bool(set(group.ranks) & local_ranks)
            lowered = group.lowered
            scoped_phases = []
            for phase in lowered.phases:
                kept_transfers = []
                for transfer in phase.transfers:
                    endpoint_local = (
                        transfer.source in local_ranks
                        or transfer.destination in local_ranks
                    )
                    kept_hops = []
                    for hop in transfer.hops:
                        if not (
                            endpoint_local
                            or hop.source in local_ranks
                            or hop.destination in local_ranks
                        ):
                            continue
                        res = tuple(r for r in hop.resources if r in available_resources)
                        if res:
                            kept_hops.append(replace(hop, resources=res))
                    if kept_hops:
                        orig_has_per_hop = any(
                            h.flow_cap_bytes_per_ms is not None for h in transfer.hops
                        )
                        kept_caps = [
                            h.flow_cap_bytes_per_ms for h in kept_hops
                            if h.flow_cap_bytes_per_ms is not None
                        ]
                        kept_cap = (
                            min(kept_caps) if kept_caps
                            else (None if orig_has_per_hop else transfer.flow_cap_bytes_per_ms)
                        )
                        kept_transfers.append(replace(
                            transfer,
                            hops=tuple(kept_hops),
                            flow_cap_bytes_per_ms=kept_cap,
                        ))
                scoped_phases.append(replace(phase, transfers=tuple(kept_transfers)))
            if not has_member and not any(p.transfers for p in scoped_phases):
                continue
            scoped_lowered = _with_recomputed_phases(lowered, scoped_phases)
            if not has_member:
                scoped_lowered = replace(scoped_lowered, local_hbm_bytes_per_rank=0.0)
            scoped_groups.append(replace(group, lowered=scoped_lowered))
        if not scoped_groups:
            return None
        return replace(self, groups=tuple(scoped_groups))

    def _append_local_hbm(self, context, out: list, after: tuple[str, ...]) -> tuple[str, ...]:
        from ..program.stages import Stage

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
                after=after,
            ))
            return (hbm_stage_name,)
        return after

    def _can_interleave_two_phase_ar(self, post_stages, link_names: frozenset[str]) -> bool:
        if not self.groups:
            return False
        if not all(
            len(g.lowered.phases) == TWO_PHASES
            and not any(p.pipelined_slices for p in g.lowered.phases)
            and g.lowered.phases[0].name == "reduce_scatter"
            and g.lowered.phases[1].name == "all_gather"
            for g in self.groups
        ):
            return False
        return any(
            stg.name == "reduce_scatter"
            and _strip_stage_links(stg, (), link_names) is not None
            for stg in post_stages
        )

    def lower_with_local_work(self, context, base_program):
        """Wrap this lowered collective plan inside causally ordered local stages."""
        from ..program.stages import KernelProgram, Stage

        if any(isinstance(stg.program, BandCollectivePlan) for stg in base_program.stages):
            return base_program
        link_names = frozenset(self.accounting().wire_by_resource)
        stages_in = _unwrap_collective_stages(base_program.stages)
        pre_stages, post_stages = _partition_pre_post_stages(stages_in, link_names)
        mid_stages = [s for s in post_stages if s.name != "unpack_store"]
        unpack_stages = [s for s in post_stages if s.name == "unpack_store"]

        out: list[Stage] = []
        valid: set[str] = set()
        inherited: dict[str, tuple[str, ...]] = {}
        consumed_by_downstream: set[str] = set()
        for stg in pre_stages:
            eff_after = _remap_stage_after(stg.after, inherited, valid)
            kept = _strip_stage_links(stg, eff_after, link_names)
            if kept is None:
                inherited[stg.name] = eff_after
                continue
            out.append(kept)
            valid.add(kept.name)
            consumed_by_downstream.update(eff_after)

        pre_terminals = tuple(s.name for s in out if s.name not in consumed_by_downstream)
        reserved = valid | {s.name for s in post_stages}
        if self._can_interleave_two_phase_ar(mid_stages, link_names):
            rs_net = "band_collective:rs_net"
            ag_net = "band_collective:ag_net"
            rs_idx = next(i for i, s in enumerate(mid_stages) if s.name == "reduce_scatter")
            out.append(Stage(
                rs_net, after=pre_terminals, program=_phase_slice_plan(self, 0), keep_empty=True,
            ))
            rs_valid: set[str] = {rs_net}
            rs_inherited: dict[str, tuple[str, ...]] = {}
            rs_consumed: set[str] = set()
            rs_added: list[str] = []
            for stg in mid_stages[:rs_idx + 1]:
                internal = _remap_stage_after(stg.after, rs_inherited, rs_valid)
                eff_after = internal or (rs_net,)
                kept = _strip_stage_links(stg, eff_after, link_names)
                if kept is None:
                    rs_inherited[stg.name] = eff_after
                    continue
                out.append(kept)
                rs_valid.add(kept.name)
                rs_consumed.update(eff_after)
                rs_added.append(kept.name)
            rs_terminals = tuple(name for name in rs_added if name not in rs_consumed) or (rs_net,)
            out.append(Stage(
                ag_net, after=rs_terminals, program=_phase_slice_plan(self, 1), keep_empty=True,
            ))
            net_root = ag_net
            remaining_mid = mid_stages[rs_idx + 1:]
        else:
            plan_name = "band_collective" if "band_collective" not in reserved else "band_collective:plan"
            out.append(Stage(plan_name, after=pre_terminals, program=self, keep_empty=True))
            net_root = plan_name
            remaining_mid = mid_stages

        post_valid: set[str] = {net_root}
        post_inherited: dict[str, tuple[str, ...]] = {}
        post_consumed: set[str] = set()
        mid_added: list[str] = []
        for stg in remaining_mid:
            internal = _remap_stage_after(stg.after, post_inherited, post_valid)
            eff_after = internal or (net_root,)
            kept = _strip_stage_links(stg, eff_after, link_names)
            if kept is None:
                post_inherited[stg.name] = eff_after
                continue
            out.append(kept)
            post_valid.add(kept.name)
            post_consumed.update(eff_after)
            mid_added.append(kept.name)

        mid_terminals = tuple(name for name in mid_added if name not in post_consumed) or (net_root,)
        hbm_roots = self._append_local_hbm(context, out, mid_terminals)
        post_valid.update(hbm_roots)
        for stg in unpack_stages:
            internal = _remap_stage_after(stg.after, post_inherited, post_valid)
            eff_after = tuple(dict.fromkeys((*hbm_roots, *internal)))
            kept = _strip_stage_links(stg, eff_after, link_names)
            if kept is None:
                post_inherited[stg.name] = eff_after
                continue
            out.append(kept)
            post_valid.add(kept.name)

        return KernelProgram(tuple(out), minimum_ms=context.minimum_ms)

    def split_local_programs(
        self, context, base_program, *, extra_link_names: frozenset[str] = frozenset(),
    ):
        """Split a collective program into link-stripped ``(pre_program, post_program)``.

        Used by cohort-wide scheduling so each replica executes its local pack
        stages before ONE shared ``BandCollectivePlan`` network submission and
        runs its local reduction/HBM/unpack stages strictly after rank arrivals.
        """
        from ..program.stages import KernelProgram

        link_names = frozenset(self.accounting().wire_by_resource) | frozenset(extra_link_names)
        stages_in = _unwrap_collective_stages(base_program.stages)
        pre_stages, post_stages = _partition_pre_post_stages(stages_in, link_names)
        mid_stages = [s for s in post_stages if s.name != "unpack_store"]
        unpack_stages = [s for s in post_stages if s.name == "unpack_store"]

        pre_out = []
        pre_valid: set[str] = set()
        pre_inherited: dict[str, tuple[str, ...]] = {}
        for stg in pre_stages:
            eff_after = _remap_stage_after(stg.after, pre_inherited, pre_valid)
            kept = _strip_stage_links(stg, eff_after, link_names, exclude_band_plan=True)
            if kept is None:
                pre_inherited[stg.name] = eff_after
                continue
            pre_out.append(kept)
            pre_valid.add(kept.name)

        post_out = []
        post_valid: set[str] = set()
        post_inherited: dict[str, tuple[str, ...]] = {}
        post_consumed: set[str] = set()
        mid_added: list[str] = []
        for stg in mid_stages:
            eff_after = _remap_stage_after(stg.after, post_inherited, post_valid)
            kept = _strip_stage_links(stg, eff_after, link_names, exclude_band_plan=True)
            if kept is None:
                post_inherited[stg.name] = eff_after
                continue
            post_out.append(kept)
            post_valid.add(kept.name)
            post_consumed.update(eff_after)
            mid_added.append(kept.name)

        mid_terminals = tuple(name for name in mid_added if name not in post_consumed)
        hbm_roots = self._append_local_hbm(context, post_out, mid_terminals)
        post_valid.update(hbm_roots)
        for stg in unpack_stages:
            internal = _remap_stage_after(stg.after, post_inherited, post_valid)
            eff_after = tuple(dict.fromkeys((*hbm_roots, *internal)))
            kept = _strip_stage_links(stg, eff_after, link_names, exclude_band_plan=True)
            if kept is None:
                post_inherited[stg.name] = eff_after
                continue
            post_out.append(kept)
            post_valid.add(kept.name)

        pre_prog = KernelProgram(tuple(pre_out), minimum_ms=0.0) if pre_out else None
        post_prog = KernelProgram(tuple(post_out), minimum_ms=0.0) if post_out else None
        return pre_prog, post_prog


def _lower_group_with_sibling_fallback(
    topology,
    family: str,
    ranks: tuple[int, ...],
    api_bytes: float,
    *,
    stack: str,
    expansion_mode: str | None,
    peer_matrix=None,
    send_splits=None,
    recv_splits=None,
    communicator_id: str,
    coset_id: str,
    allow_sibling_fallback: bool,
    dtype: str | None = None,
) -> LoweredCollective:
    del allow_sibling_fallback
    return lower_collective(
        family, ranks, topology, api_bytes,
        stack=stack, expansion_mode=expansion_mode,
        peer_matrix=peer_matrix, send_splits=send_splits, recv_splits=recv_splits,
        communicator_id=communicator_id, coset_id=coset_id, dtype=dtype,
    )


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
    dtype: str | None = None,
) -> BandCollectivePlan:
    """Lower all groups in a cohort through ``lower_collective`` into one plan."""
    stack = str(metadata.get("collective_stack", stack) or stack)
    resolved_dtype = dtype or metadata.get("dtype") or metadata.get("payload_dtype")
    multi_group = len(unique_groups) > 1
    groups: list[BandCollectiveGroup] = []
    for idx, ranks in enumerate(unique_groups):
        if family == "alltoallv" or peer is not None:
            if peer is None:
                raise ModelingError("pairwise_collective_alltoallv_requires_explicit_peer_matrix")
            n_ranks = len(ranks)
            if getattr(peer, "direction", "forward") == "transpose":
                fwd_peer = peer.transposed()
                fwd_matrix = tuple(
                    _integer_peer_row(
                        tuple(fwd_peer.payload(r, c) for c in range(n_ranks)), r
                    )
                    for r in range(n_ranks)
                )
                matrix = tuple(
                    tuple(fwd_matrix[c][r] for c in range(n_ranks))
                    for r in range(n_ranks)
                )
            else:
                matrix = tuple(
                    _integer_peer_row(
                        tuple(peer.payload(r, c) for c in range(n_ranks)), r
                    )
                    for r in range(n_ranks)
                )
            send_splits = tuple(sum(row) for row in matrix)
            recv_splits = tuple(
                sum(matrix[r][c] for r in range(len(ranks)))
                for c in range(len(ranks))
            )
            api_bytes = float(max(send_splits)) if send_splits else 0.0
            lowered = _lower_group_with_sibling_fallback(
                topology, "alltoallv", ranks, api_bytes,
                stack=stack, expansion_mode=expansion_mode,
                peer_matrix=matrix, send_splits=send_splits, recv_splits=recv_splits,
                communicator_id=str(metadata.get("collective_id", "")),
                coset_id=f"g{idx}",
                allow_sibling_fallback=multi_group,
                dtype=resolved_dtype,
            )
        else:
            volumes = {rank: volume_by_rank[rank] for rank in ranks}
            if len(set(volumes.values())) > 1:
                if family != "alltoall":
                    raise ModelingError("ring_collective_requires_equal_per_rank_volume")
                n_ranks = len(ranks)
                matrix = tuple(
                    _integer_peer_row(
                        tuple(
                            0.0 if r_i == c_i else volumes[ranks[r_i]] / float(n_ranks - 1)
                            for c_i in range(n_ranks)
                        ),
                        r_i,
                    )
                    for r_i in range(n_ranks)
                )
                send_splits = tuple(sum(row) for row in matrix)
                recv_splits = tuple(
                    sum(matrix[r_i][c_i] for r_i in range(n_ranks))
                    for c_i in range(n_ranks)
                )
                api_bytes = float(max(send_splits)) if send_splits else 0.0
                lowered = _lower_group_with_sibling_fallback(
                    topology, "alltoallv", ranks, api_bytes,
                    stack=stack, expansion_mode=expansion_mode,
                    peer_matrix=matrix, send_splits=send_splits, recv_splits=recv_splits,
                    communicator_id=str(metadata.get("collective_id", "")),
                    coset_id=f"g{idx}",
                    allow_sibling_fallback=multi_group,
                    dtype=resolved_dtype,
                )
            else:
                vol = next(iter(volumes.values()), 0.0)
                api_bytes = api_bytes_from_link_volume(family, len(ranks), vol, metadata)
                lowered = _lower_group_with_sibling_fallback(
                    topology, family, ranks, api_bytes,
                    stack=stack, expansion_mode=expansion_mode,
                    communicator_id=str(metadata.get("collective_id", "")),
                    coset_id=f"g{idx}",
                    allow_sibling_fallback=multi_group,
                    dtype=resolved_dtype,
                )
        groups.append(BandCollectiveGroup(ranks=ranks, lowered=lowered, topology=topology))
    return BandCollectivePlan(family=family, groups=tuple(groups), topology=topology)
