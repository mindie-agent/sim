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
)
from ..runtime.prepared_terms import (
    BoundVectorOp,
    VectorInstructionGroup,
    bind_vector_groups,
)
from .stages import KernelProgram, ParallelProgram, Program, Stage

_MS_PER_SECOND = 1000.0
_DEFAULT_VECTOR_CLOCK_HZ = 1_000_000_000.0


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

    @property
    def effective_aic_cores(self) -> int:
        """Effective AIC core limit for this operator's stream."""
        if self.stream_owners is not None:
            return int(self.stream_owners.aic_limit)
        if self.datapath is not None:
            return int(self.datapath.aic_count)
        return int(self.core_count)

    @property
    def effective_aiv_cores(self) -> int:
        """Effective AIV core limit for this operator's stream."""
        if self.stream_owners is not None:
            return int(self.stream_owners.aiv_limit)
        if self.datapath is not None:
            return int(self.datapath.aiv_count)
        return int(self.resource_units.get("vector", self.core_count))

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

    def bind_access(
        self,
        pattern: MemoryAccessPattern,
        state: AdmissionMemoryView | None = None,
        *,
        hbm_bus: HbmBusSpec | None = None,
        burst_rules: Sequence[BurstTurnRule] = (),
        copy_hit_bytes_per_ms: float | None = None,
        copy_miss_bytes_per_ms: float | None = None,
        l2_bytes_per_ms: float | None = None,
    ) -> BoundAccess:
        """Bind a ``MemoryAccessPattern`` into exact 3-ledger byte accounting and row terms."""
        pipe_role = pattern.pipe
        if pipe_role not in self.resource_names and pipe_role.startswith("mte2"):
            pipe_role = "mte2"
        if pipe_role not in self.resource_names:
            raise ModelingError(f"bind_access_pipe_missing:{pattern.pipe}")
        pipe_row = self.resource_names[pipe_role]
        pipe_rate = float(self.rates[pipe_row])
        hit_rate = pipe_rate if copy_hit_bytes_per_ms is None else float(copy_hit_bytes_per_ms)
        miss_rate = pipe_rate if copy_miss_bytes_per_ms is None else float(copy_miss_bytes_per_ms)
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
            alpha_rw = float(getattr(self.physical_mechanisms, "hbm_alpha_rw", 0.0) or 0.0)
            bus = HbmBusSpec(
                die=pattern.die,
                read_bytes_per_ms=hbm_rate,
                write_bytes_per_ms=hbm_rate,
                alpha_rw=alpha_rw,
                row_r=f"{hbm_name}:r",
                row_w=f"{hbm_name}:w",
                ctr_r=f"{hbm_name}:ctr_r",
                ctr_w=f"{hbm_name}:ctr_w",
            )
        return bind_memory_access(
            pattern,
            state=state,
            hbm_bus=bus,
            copy_hit_bytes_per_ms=hit_rate,
            copy_miss_bytes_per_ms=miss_rate,
            l2_bytes_per_ms=l2_rate,
            pipe_row_id=pipe_row,
            l2_row_id=l2_row,
            burst_rules=burst_rules,
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
        resolved_clock = (
            float(clock_hz)
            if clock_hz is not None
            else float(getattr(self.compute_arch, "clock_hz", _DEFAULT_VECTOR_CLOCK_HZ))
        )
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
        from ..transfer import FiniteTransfer
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

