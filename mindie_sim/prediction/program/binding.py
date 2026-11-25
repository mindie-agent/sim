"""Recipe binding context, 3-ledger access/vector binding, and RecipeBook.

[Role] Provide ``RecipeContext`` (including ``bind``, ``bind_access``, and
  ``bind_vector``), ``Recipe``, ``ResourceEnvelope``, ``TransferRecipe``,
  ``HostStaging``, ``RecurrentScan``, and ``RecipeBook`` dispatch.
[Invariants] ``ResourceEnvelope.lower`` unconditionally rejects legacy envelope
  bindings. ``bind_access`` conserves 3-ledger byte accounting; ``bind_vector``
  lowers onto ``vec_issue`` and sub-pipes without a competing total vector row.
[Timeouts & Locks] Selection/expansion caches are bounded by
  ``_SELECT_CACHE_LIMIT`` and ``_EXPAND_CACHE_LIMIT``; no I/O or locks.
[Side-Effect Policy] Pure recipe dispatch and binding; state mutations occur
  only on caller-supplied ``AdmissionMemoryView`` passed to ``bind_access``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from ..engine import Allocation
from ..ownership import unit_group
from ..runtime.memory_physics import (
    AdmissionMemoryView,
    BoundAccess,
    BurstTurnRule,
    HbmBusSpec,
    MemoryAccessPattern,
    bind_memory_access,
    cache_policy_for_arch,
)
from ..runtime.prepared_terms import (
    BoundVectorOp,
    VectorInstructionGroup,
    bind_vector_groups,
)
from .stages import KernelProgram, ParallelProgram, Program, Stage

_MS_PER_SECOND = 1000.0
_DATACOPY_PRIMITIVE = "P01_datacopy"


def _burst_rule(item) -> BurstTurnRule:
    if isinstance(item, BurstTurnRule):
        return item
    if not isinstance(item, Mapping):
        item = item.to_dict()
    return BurstTurnRule(
        item.get("min_row_bytes", 0), item.get("max_row_bytes"),
        item.get("min_stride_bytes", 0), item.get("max_stride_bytes"),
        item["penalty_bytes"], item.get("evidence_id", ""),
    )


def proportional_coupled_placement(
    aic_units: Sequence[int],
    aiv_units: Sequence[int],
    aic_total: int,
    aiv_total: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Trim ``(aic_units, aiv_units)`` to the largest proportional AIC/AIV subset."""
    if aic_total < 1 or aiv_total < 1:
        raise ModelingError(f"proportional_totals_invalid:{aic_total}:{aiv_total}")
    aic = tuple(sorted(set(int(u) for u in aic_units)))
    aiv = tuple(sorted(set(int(u) for u in aiv_units)))
    g = math.gcd(aic_total, aiv_total)
    step_c = aic_total // g
    step_v = aiv_total // g
    groups = min(len(aic) // step_c, len(aiv) // step_v)
    if groups < 1:
        raise ModelingError(
            f"stream_proportional_subset_empty:{len(aic)}/{aic_total}:{len(aiv)}/{aiv_total}"
        )
    return aic[: groups * step_c], aiv[: groups * step_v]


def _transport_identity(value):
    """Stable T09 field projection; never stringifies or drops explicit None."""
    from ..topology import TransportOptions
    if isinstance(value, TransportOptions):
        return {
            "chunk_bytes": value.chunk_bytes,
            "hop_startup_us": value.hop_startup_us,
            "receive_capacity_bytes": [
                [int(rank), float(amount)]
                for rank, amount in sorted(value.receive_capacity_bytes)
            ],
            "receive_buffer_ids": [
                [int(rank), str(name)]
                for rank, name in sorted(value.receive_buffer_ids)
            ],
            "receive_capacity_bytes_per_peer": value.receive_capacity_bytes_per_peer,
            "receive_buffer_id": value.receive_buffer_id,
            "max_inflight_per_rank": value.max_inflight_per_rank,
            "max_physical_chunks": value.max_physical_chunks,
        }
    if isinstance(value, dict):
        return {str(key): _transport_identity(item) for key, item in value.items()}
    return value


def _stable_identity(value):
    """Make a service identity JSON-serializable without default=str."""
    from ..topology import TransportOptions
    if isinstance(value, TransportOptions):
        return _transport_identity(value)
    if isinstance(value, dict):
        return {str(key): _stable_identity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_stable_identity(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"service_identity_value_not_serializable:{type(value).__name__}")


@dataclass(frozen=True)
class RecipeContext:
    op: object
    work: Mapping[str, float]
    resource_names: Mapping[str, str]
    rates: Mapping[str, float]
    core_count: int
    minimum_ms: float = 0.0
    minimum_provenance: Mapping = field(default_factory=dict)
    setup_ms: float = 0.0
    pool_names: Mapping[str, str] = field(default_factory=dict)
    resource_units: Mapping[str, int] = field(default_factory=dict)
    compute_peaks: Mapping[str, float] = field(default_factory=dict)
    pool_units: Mapping[str, int] = field(default_factory=dict)
    service_modes: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    native_work: Mapping[str, float] = field(default_factory=dict)
    service_identity: Mapping = field(default_factory=dict)
    record_service: Callable | None = None
    work_units: Mapping[str, str] = field(default_factory=lambda: dict(NATIVE_WORK_UNITS))
    service_boundary: str = "device_kernel"
    transaction_frames: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    collective_plan: Callable[[], object] | None = None
    pool_capacities: Mapping[str, float] = field(default_factory=dict)
    record_mapping: Callable | None = None
    invocation_timing: object = None
    store_queues: Mapping[str, Mapping] = field(default_factory=dict)
    group_buffers: tuple[str, ...] = ()
    joint_native_service: Mapping | None = None
    service_regime: Mapping | None = None
    mac_caps: Mapping[str, float] = field(default_factory=dict)
    mac_service: Mapping | None = None
    program_recipe: object = None
    transport_options: object = None
    transport_identity: Mapping | None = None
    physical_mechanisms: object = None
    compute_arch: object = None
    datapath: object = None
    stream_owners: object = None
    primitive_table: object = None
    primitive_coverage_gaps: tuple[str, ...] = ()
    declared_physical_rates: Mapping[str, Any] | None = None
    admission_memory_view: object = None
    allocation_versions: Mapping[str, int] = field(default_factory=dict)
    requests: tuple[Any, ...] = ()
    padded_tokens: int = 0
    layout: str = ""
    service_caps: Mapping[str, float] = field(default_factory=dict)
    phase_link_work: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def stage_caps_for(
        self,
        demand: Mapping[str, float],
        *,
        share: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return declared sustainable rate ceilings for active physical resources."""
        if not self.service_caps or not demand:
            return {}
        caps: dict[str, float] = {}
        for name, ceiling in self.service_caps.items():
            if float(demand.get(name, 0.0)) <= 0.0:
                continue
            rate = float(ceiling)
            if rate <= 0.0:
                continue
            factor = float(share[name]) if (share is not None and name in share) else 1.0
            if factor > 0.0:
                caps[name] = rate * factor
        return caps

    def wave_count(self, tiles: int, *, kind: str = "aic") -> int:
        """Return ``ceil(tiles / active_cores)`` under this stream's core budget."""
        if tiles <= 0:
            return 0
        active = self.effective_aic_cores if kind == "aic" else self.effective_aiv_cores
        if active <= 0:
            raise ModelingError(f"stream_zero_{kind}_budget:wave_count")
        return (int(tiles) + active - 1) // active

    def __post_init__(self):
        identity = dict(self.service_identity)
        transport = identity.pop("transport_options", None)
        if self.transport_options is None and transport is not None:
            object.__setattr__(self, "transport_options", transport)
        object.__setattr__(self, "service_identity", identity)
        object.__setattr__(self, "_owned_cache", {})

    def for_phase(self, work, *, kernel, dtype=None, layout=None, axes=None):
        """Bind explicitly identified phase work without repeating kernel setup."""
        op = work.apply(self.op)
        identity = {**self.service_identity, "kernel": kernel}
        if dtype is not None:
            op = replace(op, dtype=dtype)
            identity["dtype"] = dtype
        if layout is not None:
            op = replace(op, layout=layout)
            identity["layout"] = layout
        if axes is not None:
            identity["axes"] = axes
        transport = self.transport_options
        if transport is None:
            transport = identity.get("transport_options")
        identity.pop("transport_options", None)
        transport_identity = (
            _transport_identity(transport) if transport is not None else None
        )
        identity = json.loads(json.dumps(_stable_identity(identity), allow_nan=False))
        native = {**work.resources, "cube": sum(work.matrix.values())}
        return replace(
            self, op=op, work=self.bind(work),
            native_work={role: amount for role, amount in native.items() if amount},
            setup_ms=0, minimum_ms=0, service_identity=identity,
            transport_options=transport, transport_identity=transport_identity,
            service_boundary="resource_phase", invocation_timing=None,
        )

    def physical_role(self, role):
        try:
            return self.service_modes[role][0]
        except KeyError:
            return role

    def service_rate(self, role: str) -> float | None:
        """Return the declared physical service rate for a named service mode."""
        try:
            return float(self.service_modes[role][1])
        except KeyError:
            return None

    def placement(self, role, units):
        return {self.resource_names[self.physical_role(role)]: tuple(units)}

    def allocate(self, name, pool, size, units):
        return Allocation(name, self.pool_names[pool], size, tuple(units))

    def owned(self, owner, owners, roles):
        cache = self._owned_cache
        roles_key = roles if roles.__class__ is tuple else tuple(roles)
        key = (owner, owners, roles_key)
        hit = cache.get(key)
        if hit is None:
            phys_roles = {self.physical_role(role) for role in roles_key}
            res_names = self.resource_names
            res_units = self.resource_units
            core_cnt = self.core_count
            hit = tuple(
                (res_names[role], unit_group(owner, owners, res_units.get(role, core_cnt)))
                for role in phys_roles if role in res_names
            )
            cache[key] = hit
        return dict(hit)

    def read_pipe(self, *, aiv: bool | None = None) -> str:
        """Declared operand-read pipe for this op's consumer owner."""
        from ..ownership import read_role
        names = self.resource_names
        if not {"mte2_aic", "mte2_aiv"}.intersection(names):
            return "mte2"
        declared = str(dict(self.op.metadata or {}).get("read_pipe", "") or "")
        if declared in {"aic", "aiv"}:
            return read_role(names, aiv=0 if declared == "aiv" else None)
        if aiv is None:
            matrix = tuple(getattr(self.op, "matrix_flops", ()) or ())
            flops = float(getattr(self.op, "flops", 0.0) or 0.0)
            vector = float(getattr(self.op, "vector_ops", 0.0) or 0.0)
            aiv = not matrix and flops <= 0.0 and vector > 0.0
        return read_role(names, aiv=0 if aiv else None)

    def bind(self, work) -> dict[str, float]:
        """Resolve typed stage work onto shared physical resources, in ms units."""
        demand: dict[str, float] = {}
        for dtype, flops in work.matrix.items():
            if not flops:
                continue
            try:
                peak = self.compute_peaks[dtype]
            except KeyError:
                raise ModelingError(f"compute_precision_rate_required:{dtype}") from None
            name = self.resource_names["cube"]
            demand[name] = demand.get(name, 0) + _MS_PER_SECOND * flops / peak
        matrix_consumer = any(float(value) > 0.0 for value in work.matrix.values())
        for role, amount in work.resources.items():
            if not amount:
                continue
            try:
                resource, rate = self.service_modes[role]
            except KeyError:
                pass
            else:
                name = self.resource_names[resource]
                amount *= self.rates[name] * _MS_PER_SECOND / rate
                role = resource
            if role == "mte2" and role not in self.resource_names:
                if not {"mte2_aic", "mte2_aiv"}.intersection(self.resource_names):
                    continue
                role = self.read_pipe(aiv=False if matrix_consumer else None)
            if role == "mte3" and role not in self.resource_names:
                continue
            if role not in self.resource_names:
                raise ModelingError(
                    f"physical_resource_rate_missing:{role}:"
                    f"{getattr(self.op, 'family', '')}:"
                    f"{getattr(self.op, 'recipe', '')}"
                )
            name = self.resource_names[role]
            demand[name] = demand.get(name, 0) + amount
        return demand

    @property
    def effective_aic_cores(self) -> int:
        owners = self.stream_owners
        if owners is not None and getattr(owners, "aic_limit", None) is not None:
            return int(owners.aic_limit)
        if self.datapath is not None and getattr(self.datapath, "aic_count", None) is not None:
            return int(self.datapath.aic_count)
        return int(self.core_count)

    def _units_for(self, role: str, physical: str, default: int) -> int:
        if physical in self.resource_units:
            return max(1, int(self.resource_units[physical]))
        if role in self.resource_units:
            return max(1, int(self.resource_units[role]))
        if role.startswith("mte2") and "mte2" in self.resource_units:
            return max(1, int(self.resource_units["mte2"]))
        return max(1, int(default))

    @property
    def effective_aiv_cores(self) -> int:
        owners = self.stream_owners
        if owners is not None and getattr(owners, "aiv_limit", None) is not None:
            return int(owners.aiv_limit)
        if self.datapath is not None and getattr(self.datapath, "aiv_count", None) is not None:
            return int(self.datapath.aiv_count)
        return self._units_for("vector", self.resource_names.get("vector", "vector"), self.core_count)

    def bind_coupled_stream_placement(
        self, aic_role: str = "cube", aiv_role: str = "vector",
    ) -> dict[str, tuple[int, ...]]:
        """Bind largest proportional AIC/AIV subset from active ``stream_owners``."""
        owners = self.stream_owners
        if owners is None or getattr(owners, "isolation", "shared_cap") != "disjoint":
            return {}
        aic_name = self.resource_names.get(aic_role, aic_role)
        aiv_name = self.resource_names.get(aiv_role, aiv_role)
        aic_total = self._units_for(aic_role, aic_name, self.core_count)
        aiv_total = self._units_for(aiv_role, aiv_name, self.core_count)
        aic_sub, aiv_sub = proportional_coupled_placement(
            owners.aic, owners.aiv, aic_total, aiv_total,
        )
        return {aic_name: aic_sub, aiv_name: aiv_sub}

    def stream_owner_units(self, role: str, *, buffer_pool: bool = False) -> tuple[int, ...]:
        """Return physical core IDs owned by the active stream for ``role``."""
        owners = self.stream_owners
        if owners is None:
            return ()
        return owners.pool_for_role(role, datapath=self.datapath, buffer_pool=buffer_pool)

    def stream_aic_aiv_pairs(
        self, *, require_full_pair: bool = False,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Resolve physical ``(aic, paired_aivs)`` groups from ``datapath.aic_aiv_pairs``."""
        if self.datapath is None:
            return ()
        from ..streams import resolve_aic_aiv_pairs

        return resolve_aic_aiv_pairs(
            self.datapath, self.stream_owners, require_full_pair=require_full_pair,
        )

    def validate_stage_placement(
        self, stage: Stage, *, stream_owners: Any = None, require_placement: bool = True,
    ) -> None:
        """Validate post-binding per-owner placement and private buffer units on one ``Stage``."""
        from ..streams import MAIN_STREAM, validate_stage_owners

        owners = self.stream_owners if stream_owners is None else stream_owners
        stream_id = getattr(self.op, "stream", None) or MAIN_STREAM
        validate_stage_owners(
            stage,
            self.datapath,
            resource_names=self.resource_names,
            resource_units=self.resource_units,
            owners=owners,
            stream_id=stream_id,
            pool_names=self.pool_names,
            pool_units=self.pool_units,
            require_placement=require_placement,
        )

    def validate_program_placement(
        self, program: Any, *, stream_owners: Any = None, require_placement: bool = True,
    ) -> None:
        """Validate post-binding per-owner placements and grant/release pairs on a ``Program``."""
        from ..streams import MAIN_STREAM, validate_program_owners

        owners = self.stream_owners if stream_owners is None else stream_owners
        stream_id = getattr(self.op, "stream", None) or MAIN_STREAM
        validate_program_owners(
            program,
            self.datapath,
            resource_names=self.resource_names,
            resource_units=self.resource_units,
            owners=owners,
            stream_id=stream_id,
            pool_names=self.pool_names,
            pool_units=self.pool_units,
            require_placement=require_placement,
        )

    def check_live_buffers(
        self,
        buffers: Sequence[Any],
        *,
        family: str = "",
        uses_simt: bool = False,
        fused: bool = False,
        kernel_abi: str = "",
        simt_ub_carveout_bytes: int = 0,
    ) -> tuple[Any, ...]:
        """Verify live on-chip/L2 buffer peaks via ``tile_solver.check_live_buffers``."""
        from ..runtime.tile_solver import check_live_buffers

        return check_live_buffers(
            buffers,
            self,
            family=family or str(getattr(self.op, "family", "") or "buffers"),
            uses_simt=uses_simt,
            fused=fused,
            kernel_abi=kernel_abi,
            simt_ub_carveout_bytes=simt_ub_carveout_bytes,
        )

    def _datacopy_cell(self):
        cell = getattr(self.primitive_table, "cell", None)
        if cell is not None:
            try:
                return cell(_DATACOPY_PRIMITIVE)
            except ModelingError:
                pass
        return None

    def _copy_rate(self, pipe, phase, count, nominal, primitive):
        measured = None
        if primitive is not None and primitive.supported:
            band = primitive.metrics.get(f"{pipe}_{phase}_bytes_per_s_per_core")
            measured = band.p50 if band is not None else None
        if measured is not None and measured > 0:
            return float(measured) / _MS_PER_SECOND
        mode = f"{pipe}_{'l2' if phase == 'hit' else 'dram'}"
        rate = self.service_rate(mode)
        return nominal if rate is None else rate / count / _MS_PER_SECOND

    def bind_access(
        self,
        pattern: MemoryAccessPattern,
        state: AdmissionMemoryView | None = None,
        *,
        hbm_bus: HbmBusSpec | None = None,
        burst_rules: Sequence[BurstTurnRule] | None = None,
        copy_hit_bytes_per_ms: float | None = None,
        copy_miss_bytes_per_ms: float | None = None,
        l2_bytes_per_ms: float | None = None,
        commit: bool = False,
    ) -> BoundAccess:
        """Bind a ``MemoryAccessPattern`` into exact 3-ledger byte accounting and row terms."""
        pipe_role = pattern.pipe
        if pipe_role not in self.resource_names and pipe_role.startswith("mte2"):
            pipe_role = "mte2"
        if pipe_role not in self.resource_names:
            raise ModelingError(f"bind_access_pipe_missing:{pattern.pipe}")
        pipe_row = self.resource_names[pipe_role]
        pipe_units = self._units_for(pattern.pipe, pipe_row, self._units_for(pipe_role, pipe_row, 1))
        pipe_rate = float(self.rates[pipe_row]) / float(pipe_units)
        primitive = self._datacopy_cell()
        hit_rate = (self._copy_rate(pattern.pipe, "hit", pipe_units, pipe_rate, primitive)
                    if copy_hit_bytes_per_ms is None else float(copy_hit_bytes_per_ms))
        miss_rate = (self._copy_rate(pattern.pipe, "miss", pipe_units, pipe_rate, primitive)
                     if copy_miss_bytes_per_ms is None else float(copy_miss_bytes_per_ms))
        memory = getattr(self.datapath, "memory_access", None)
        if burst_rules is None:
            rules = getattr(memory, "burst_turn_penalty", ())
            if not rules and primitive is not None and primitive.supported:
                rules = primitive.parameters.get("burst_turn_penalty", ())
            burst_rules = tuple(_burst_rule(item) for item in rules)
        arch = getattr(self.datapath, "arch_id", "generic")
        if arch != "generic":
            cache_policy_for_arch(arch, pattern.cache_policy)
        l2_row = self.resource_names.get("l2", "l2")
        if l2_bytes_per_ms is not None:
            l2_rate = float(l2_bytes_per_ms)
        elif l2_row in self.rates:
            l2_rate = float(self.rates[l2_row])
        else:
            l2_rate = hit_rate
        bus = hbm_bus
        if bus is None:
            if "hbm" not in self.resource_names:
                raise ModelingError("bind_access_hbm_rate_missing")
            hbm_name = self.resource_names["hbm"]
            hbm_rate = float(self.rates[hbm_name])
            alpha = getattr(self.physical_mechanisms, "hbm_rw_alpha", None)
            if alpha is None:
                alpha = getattr(self.datapath, "hbm_rw_alpha", 0.0)
                if not alpha and primitive is not None and primitive.supported:
                    band = primitive.metrics.get("hbm_rw_alpha")
                    alpha = band.p50 if band is not None else alpha
            bus = HbmBusSpec(
                die=pattern.die,
                read_bytes_per_ms=hbm_rate,
                write_bytes_per_ms=hbm_rate,
                alpha_rw=float(alpha or 0.0),
                row_r=f"{hbm_name}:r",
                row_w=f"{hbm_name}:w",
                ctr_r=f"{hbm_name}:ctr_r",
                ctr_w=f"{hbm_name}:ctr_w",
            )
        if memory is not None and bus.memory_access is None:
            bus = replace(bus, memory_access=memory)
        return bind_memory_access(
            pattern,
            state=state,
            hbm_bus=bus,
            copy_hit_bytes_per_ms=hit_rate,
            copy_miss_bytes_per_ms=miss_rate,
            l2_bytes_per_ms=l2_rate,
            nominal_pipe_bytes_per_ms=pipe_rate,
            pipe_row_id=pipe_row,
            l2_row_id=l2_row,
            burst_rules=burst_rules,
            commit=commit,
        )

    def bind_vector(
        self,
        groups: Sequence[VectorInstructionGroup],
        *,
        clock_hz: float | None = None,
        target_eff: float = 1.0,
    ) -> BoundVectorOp:
        """Lower typed AIV instruction groups onto ``vec_issue`` and sub-pipes."""
        issue_name = self.resource_names.get("vec_issue", "vec_issue")
        if issue_name not in self.rates:
            raise ModelingError("bind_vector_issue_rate_missing")
        pipe_rates: dict[str, float] = {}
        res_map: dict[str, str] = {"vec_issue": issue_name}
        for pipe in ("vec_alu", "vec_sfu", "vec_reduce"):
            rname = self.resource_names.get(pipe, pipe)
            if rname in self.rates:
                pipe_rates[pipe] = float(self.rates[rname])
                res_map[pipe] = rname
        raw_clock = (
            clock_hz
            if clock_hz is not None
            else getattr(self.compute_arch, "clock_hz", None)
        )
        if raw_clock is None and self.datapath is not None:
            raw_clock = getattr(self.datapath, "clock_hz", None)
        if raw_clock is None or isinstance(raw_clock, bool):
            raise ModelingError("bind_vector_clock_hz_missing")
        resolved_clock = float(raw_clock)
        if not math.isfinite(resolved_clock) or resolved_clock <= 0.0:
            raise ModelingError("bind_vector_clock_hz_missing")
        return bind_vector_groups(
            groups,
            issue_rate_per_ms=float(self.rates[issue_name]),
            pipe_rates_per_ms=pipe_rates,
            clock_hz=resolved_clock,
            target_eff=target_eff,
            resource_names=res_map,
        )


class Recipe(Protocol):
    def lower(self, context: RecipeContext) -> Program: ...


class ResourceEnvelope:
    """Removed production fallback; retained only to reject legacy bindings."""

    def lower(self, context: RecipeContext) -> KernelProgram:
        raise ModelingError("resource_envelope_banned")


@dataclass(frozen=True)
class TransferRecipe:
    """Per-path transactions with slots held through post-service response."""

    transaction_bytes: int | None = None
    outstanding: int | None = None
    response_ms: float = 0.0

    def lower(self, context: RecipeContext) -> KernelProgram:
        from ..collectives.transfer import FiniteTransfer
        paths = []
        for resource, amount in context.work.items():
            frame = context.transaction_frames.get(resource)
            if frame is not None:
                payload, wire = frame
                if self.transaction_bytes is not None and self.transaction_bytes != payload:
                    raise ModelingError(f"transfer_granule_conflicts_with_link:{resource}")
                granule = wire
            else:
                granule = self.transaction_bytes if self.transaction_bytes is not None else 1
            paths.append(
                FiniteTransfer({resource: amount}, granule, self.outstanding, self.response_ms)
            )
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("transactions", after=("setup",), program=ParallelProgram(tuple(paths))),
        ), minimum_ms=context.minimum_ms)
