"""Stage hardware profiles, recipe context binding, and program lowering helpers.

[Role]
  Build immutable per-stage hardware profiles (including per-unique
  ``(node, root_complex_id, direction)`` Kunpeng Host-HCCS contention
  resources), bind ``RecipeContext`` objects and ``CostLedger`` views, and
  handle operator preparation and lowered-program caching.

[Invariants]
  * ``build_stage_profiles`` validates homogeneous owner topology across a
    representative shard and builds host root-complex resources for every
    unique ``(die.node, die.root_complex_id)`` in the stage (never collapsing
    multi-node stages onto ``dies[0].node``).
  * Lowered program and preparation caches are strictly bounded and never store
    simulated durations.
  * Architecture capabilities flow through ``ComputeArchSpec`` /
    ``DatapathCapabilitySpec`` with zero service-mode or peak-threshold sniffing.

[Timeouts & Locks]
  Bounded in-memory dictionary merges and cache lookups; no I/O or locks.

[Side-Effect Policy]
  Mutates only per-Execution bounded preparation/lowering caches and CostLedger.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping
import weakref

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import datapath_for_devices
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from mindie_sim.schema.invocation import InvocationTiming
from mindie_sim.schema.operators import IterationWork
from ..collectives.ports import root_complex_resources_for_dies
from ..collectives.transport import TransportOptions
from ..engine import Resource
from ..mechanism import Work
from ..program import KernelProgram, RecipeContext, Stage
from .streams import (
    _DEVICE_UNITS_CACHE,
    ResourceRates,
    _CollectivePlanBinding,
    _LedgerBoundCost,
    _ServiceSetupProgram,
    _declaration_of,
    _freeze_key,
    _ledger_recorder,
    _meta_diff,
    _meta_patch,
    _metadata_dict,
    _rank_token,
    apply_stream_owners,
)

MS_PER_S = 1000.0
_RATE_EQ_REL_TOL = 1e-9
PREPARE_CACHE_LIMIT = 512
PREPARED_OPS_CACHE_LIMIT = 16384
DEVICE_UNITS_CACHE_LIMIT = 64
SERVICE_IDENTITY_CACHE_LIMIT = 1024
INLINE_FRONTIER_TASK_CAP = 49152
PROGRAM_CACHE_ENTRY_LIMIT = 768
DIGEST_PREFIX_CHARS = 12
_EXECUTION_CLASS_HARDWARE_ROLES: dict[str, tuple[str, ...]] = {
    "mixed_aic_aiv": ("cube", "vector"),
    "mixed_aiv_aic_overlap": ("cube", "vector"),
    "mixed_aic_aiv_mc2": ("cube", "vector"),
    "aiv_only": ("vector",),
    "vector_aiv": ("vector",),
    "aic_only": ("cube",),
    "cube_aic": ("cube",),
}
_ORDINARY_HARDWARE_CLASSES = frozenset({"cube", "vector"})
_CATALOG_VECTOR_BASELINE_BASES = frozenset({
    "arch2201_baseline_extrapolation",
    "arch2201_baseline_declaration",
    "stream_saturated_event_median",
})
_NON_KERNEL_FAMILIES = frozenset({
    "barrier", "token_commit", "collective", "allreduce", "allgather",
    "reduce_scatter", "alltoall", "alltoallv", "transport",
})


def _at_registry_mode_rate(reg_modes: Mapping[str, Any], mode: str, rate: float) -> bool:
    """Whether ``rate`` still equals the SKU registry default for service ``mode``."""
    reg_mode = reg_modes.get(mode)
    reg_rate = float(reg_mode.get("rate", 0.0) or 0.0) if isinstance(reg_mode, Mapping) else 0.0
    return reg_rate > 0.0 and abs(rate - reg_rate) <= _RATE_EQ_REL_TOL * reg_rate


def _is_catalog_vector_baseline(timing: InvocationTiming) -> bool:
    """Return True only when every selected device vector declaration came from a catalog baseline."""
    prov = timing.provenance if isinstance(timing.provenance, _AbcMapping) else {}
    declarations = prov.get("declarations")
    if not isinstance(declarations, (list, tuple)) or not declarations:
        return False
    for entry in declarations:
        if not isinstance(entry, _AbcMapping):
            return False
        entry_prov = entry.get("provenance")
        basis = entry_prov.get("basis") if isinstance(entry_prov, _AbcMapping) else None
        if basis not in _CATALOG_VECTOR_BASELINE_BASES:
            return False
    return True


_MAPPED_INVOCATION_CACHE_MAX = 64
_MAPPED_INVOCATION_CACHE: dict[tuple, tuple[tuple, tuple[InvocationTiming, dict[str, Any]] | None]] = {}
_UPGRADED_INVOCATION_CACHE_MAX = 128
_UPGRADED_INVOCATION_CACHE: dict[tuple, tuple[InvocationTiming, dict[str, Any]]] = {}


def _resolve_mapped_hardware_invocation(dies, roles: tuple[str, ...]):
    """Resolve hardware invocation timing across ``dies`` for canonical role aliases."""
    dies_tuple = tuple(dies)
    cache_key = (tuple((device.rank, id(device), id(device.kernel_invocations)) for device in dies_tuple), roles)
    cached = _MAPPED_INVOCATION_CACHE.get(cache_key)
    if cached is not None:
        _, result = cached
        return None if result is None else (result[0], dict(result[1]))
    selected = [
        {"rank": device.rank, "class": role, **timing.to_dict()}
        for device in dies_tuple for role, timing in device.kernel_invocations if role in roles
    ]
    if not selected:
        result = None
    else:
        timing = InvocationTiming(
            startup_ms=max(row["startup_ms"] for row in selected),
            minimum_ms=max(row["minimum_ms"] for row in selected),
            provenance={"basis": "hardware_execution_class", "declarations": selected},
        )
        payload = json.dumps(timing.to_dict(), sort_keys=True, separators=(",", ":"))
        result = (
            timing,
            {
                "id": hashlib.sha256(payload.encode()).hexdigest(),
                "match": {"execution_classes": roles},
                "source": "device.kernel_invocations",
                **timing.to_dict(),
            },
        )
    if len(_MAPPED_INVOCATION_CACHE) >= _MAPPED_INVOCATION_CACHE_MAX:
        _MAPPED_INVOCATION_CACHE.pop(next(iter(_MAPPED_INVOCATION_CACHE)))
    _MAPPED_INVOCATION_CACHE[cache_key] = (dies_tuple, result)
    return None if result is None else (result[0], dict(result[1]))


def _active_aiv_blocks(op, meta: Mapping[str, Any], envelope) -> int:
    """Resolve active AIV block count from declared launch grid or I/O tile footprint."""
    from mindie_sim.primitives.loader import AIV_MIN_TILE_BYTES

    prog_exec = meta.get("program_execution")
    declared = (
        prog_exec.get("launched_blocks") or prog_exec.get("active_aivs")
        if isinstance(prog_exec, _AbcMapping) else None
    ) or meta.get("launched_blocks") or meta.get("active_aivs")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return max(1, min(envelope.aiv_count, declared))
    elem_bytes = float(dtype_bytes(getattr(op, "dtype", None) or "bf16"))
    io_bytes = max(
        float(getattr(op, "mte2_bytes", 0.0) or 0.0),
        float(getattr(op, "mte3_bytes", 0.0) or 0.0),
        float(getattr(op, "hbm_bytes", 0.0) or 0.0),
        float(getattr(op, "vector_ops", 0.0) or 0.0) * elem_bytes,
    )
    if io_bytes <= 0.0:
        return 1
    return max(1, min(envelope.aiv_count, math.ceil(io_bytes / float(AIV_MIN_TILE_BYTES))))


def _apply_calibrated_dispatch_floor(context, dies, selected, envelope):
    """Apply calibrated P12/P13/P15/P16 block-dispatch floor without overriding explicit rules."""
    op = context.op
    if context.service_boundary != "device_kernel" or not context.work:
        return selected
    if op.family in _NON_KERNEL_FAMILIES or context.native_work.get("link", 0):
        return selected
    meta = _metadata_dict(op)
    prog_exec = meta.get("program_execution")
    explicit_class = meta.get("invocation_class") or (
        prog_exec.get("execution_class") if isinstance(prog_exec, _AbcMapping) else None
    )
    mapped_roles = _EXECUTION_CLASS_HARDWARE_ROLES.get(str(explicit_class)) if explicit_class else None
    if selected is None and mapped_roles is not None:
        selected = _resolve_mapped_hardware_invocation(dies, mapped_roles)
    if selected is None or envelope is None:
        return selected
    timing, record = selected
    if record.get("source") != "device.kernel_invocations":
        return selected
    if (explicit_class is not None
            and explicit_class not in _ORDINARY_HARDWARE_CLASSES
            and mapped_roles is None):
        return selected
    has_cube = float(context.native_work.get("cube", 0.0) or 0.0) > 0.0
    has_vec = float(context.native_work.get("vector", 0.0) or 0.0) > 0.0
    target_startup_ms = timing.startup_ms
    if mapped_roles == ("cube", "vector") or (explicit_class is None and has_cube and has_vec):
        cube_floor_ms = max(
            (t.minimum_ms for d in dies for role, t in d.kernel_invocations if role == "cube"),
            default=timing.minimum_ms,
        )
        floor_ms = envelope.mix_aic_floor_ms(cube_floor_ms)
        kind = "mixed_aic_aiv"
    elif mapped_roles == ("vector",) or (explicit_class in (None, "vector") and has_vec and not has_cube):
        active_blocks = _active_aiv_blocks(op, meta, envelope)
        needs_sync = bool(
            meta.get("cross_core_sync")
            or (isinstance(prog_exec, _AbcMapping) and prog_exec.get("cross_core_sync"))
        )
        floor_ms = (
            envelope.aiv_sync_floor_ms(active_blocks)
            if needs_sync else envelope.aiv_floor_ms(active_blocks)
        )
        kind = "vector_sync" if needs_sync else "vector"
        if (
            target_startup_ms <= 0.0
            and str(meta.get("vector_submode") or "") in {"broadcast_elementwise", "outer_axis_reduction"}
        ):
            # Non-streaming CANN BroadcastSch / ReduceOpTilingDataNonContiguous
            # AIV kernels serialize block tiling & scalar coordinate setup before
            # the steady-state UB pipe loop (ops-math/math/{mul,reduce_sum}/op_kernel).
            target_startup_ms = floor_ms
    else:
        return selected
    if kind in ("vector", "vector_sync"):
        has_declared_blocks = (
            (isinstance(prog_exec, _AbcMapping)
             and bool(prog_exec.get("launched_blocks") or prog_exec.get("active_aivs")))
            or bool(meta.get("launched_blocks") or meta.get("active_aivs"))
        )
        if has_declared_blocks and _is_catalog_vector_baseline(timing):
            target_min_ms = floor_ms
        else:
            target_min_ms = max(timing.minimum_ms, floor_ms)
    else:
        target_min_ms = max(timing.minimum_ms, floor_ms)
    if target_min_ms == timing.minimum_ms and target_startup_ms <= timing.startup_ms:
        return selected
    evidence = f"p15_p13:{envelope.sku}:{envelope.envelope_sha256[:DIGEST_PREFIX_CHARS]}:{kind}"
    up_key = (record.get("id"), target_startup_ms, target_min_ms, evidence)
    cached_up = _UPGRADED_INVOCATION_CACHE.get(up_key)
    if cached_up is not None:
        return cached_up[0], dict(cached_up[1])
    upgraded = InvocationTiming(
        startup_ms=target_startup_ms,
        minimum_ms=target_min_ms,
        provenance={**dict(timing.provenance), "dispatch_envelope": evidence},
    )
    payload = json.dumps(upgraded.to_dict(), sort_keys=True, separators=(",", ":"))
    upgraded_record = {
        **record,
        "id": hashlib.sha256(payload.encode()).hexdigest(),
        "startup_ms": target_startup_ms,
        "minimum_ms": target_min_ms,
        "provenance": upgraded.provenance,
    }
    if len(_UPGRADED_INVOCATION_CACHE) >= _UPGRADED_INVOCATION_CACHE_MAX:
        _UPGRADED_INVOCATION_CACHE.pop(next(iter(_UPGRADED_INVOCATION_CACHE)))
    _UPGRADED_INVOCATION_CACHE[up_key] = (upgraded, upgraded_record)
    return upgraded, dict(upgraded_record)


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
class StageProfile:
    """Immutable cluster-level rates, pools and die geometry for one stage."""

    replica: int
    stage: int
    dies: tuple
    geometry: tuple
    rates: dict
    names: dict
    resources: tuple
    stage_pools: dict
    pools: dict
    pool_units: dict
    pool_units_by_name: dict
    units: dict
    compute_peaks: dict
    pool_capacities: dict
    service_modes: dict
    work_units: dict
    store_queues: dict
    physical_mechanisms: Any
    compute_arch: Any = None
    datapath: Any = None
    host_rc_resources: tuple = ()
    primitive_table: Any = None
    primitive_identity: tuple = ()
    primitive_coverage_gaps: tuple[str, ...] = ()
    dispatch_envelope: Any = None
    declared_physical_rates: Mapping[str, Any] | None = None

    @property
    def profile_key(self) -> tuple:
        stage_ranks = frozenset(int(d.rank) for d in self.dies)
        die_sig = tuple(
            (
                str(d.sku), float(d.launch_us), _freeze_key(d.kernel_invocations),
                _freeze_key(d.memory_access),
                d.hbm_read_bytes_per_s, d.hbm_write_bytes_per_s, d.hbm_rw_alpha,
                int(d.simt_ub_carveout_bytes), _freeze_key(d.subtile_wave),
                _freeze_key(d.dispatch_sync),
            )
            for d in self.dies
        )
        dp_sig = None
        if self.datapath is not None:
            ub_edges = getattr(self.datapath, "ub_peer_edges", ()) or ()
            in_stage_ub = sum(1 for src, dst in ub_edges if src in stage_ranks and dst in stage_ranks)
            out_stage_ub = sum(1 for src, dst in ub_edges if src in stage_ranks and dst not in stage_ranks)
            dp_sig = self.datapath._signature((in_stage_ub, out_stage_ub))
        arch_sig = None
        if self.compute_arch is not None:
            from mindie_sim.schema.datapath import _arch_capability_key
            arch_sig = (
                _arch_capability_key(self.compute_arch),
                tuple(
                    (rt.name, rt.provides, _freeze_key(rt.conditions))
                    for rt in self.compute_arch.runtimes
                ),
            )
        rc_sig = tuple(sorted(
            (
                str(res.name).rsplit(":", 1)[-1],
                float(res.rate), str(res.unit), str(res.policy), int(res.units),
            )
            for res in self.host_rc_resources
        ))
        return (
            int(self.stage),
            self.geometry,
            self.primitive_identity,
            tuple(sorted((str(k), (float(r), str(u))) for k, (r, u) in self.rates.items())),
            tuple(sorted((str(k), float(v)) for k, v in self.compute_peaks.items())),
            tuple(sorted((str(k), float(v)) for k, v in self.pool_capacities.items())),
            tuple(sorted((str(k), (str(role), float(r))) for k, (role, r) in self.service_modes.items())),
            _freeze_key(self.store_queues),
            getattr(self.physical_mechanisms, "signature", None),
            tuple(sorted((str(k), int(v)) for k, v in self.units.items())),
            tuple(sorted((str(k), int(v)) for k, v in self.pool_units_by_name.items())),
            rc_sig,
            die_sig,
            dp_sig,
            arch_sig,
        )


def _resolve_stage_service_modes(
    dies,
    orig_rates_s: Mapping[str, float],
    orig_service_modes: Mapping[str, tuple[str, float]],
    overlay_rates: Mapping[str, float],
    overlay: Any,
    names: Mapping[str, str],
) -> dict[str, tuple[str, float]]:
    reg_entry = None
    if dies[0].sku:
        from mindie_sim.backends.ascend import profiles as ascend_profiles
        reg_entry = ascend_profiles().get(str(dies[0].sku))
    reg_dev = reg_entry.get("device") if isinstance(reg_entry, Mapping) else None
    reg_rates = dict(reg_dev.get("rates") or {}) if isinstance(reg_dev, Mapping) else {}
    reg_modes = dict(reg_dev.get("service_modes") or {}) if isinstance(reg_dev, Mapping) else {}
    mte1_modes_at_registry = all(
        _at_registry_mode_rate(reg_modes, mode, m_rate)
        for mode, (role, m_rate) in orig_service_modes.items() if role == "mte1"
    )
    service_modes: dict[str, tuple[str, float]] = {}
    for mode, (role, m_rate) in orig_service_modes.items():
        if mode in {"mte2_aic_dram", "mte2_aiv_dram"} and overlay and overlay.hbm_read_bytes_per_s is not None:
            m_rate = min(m_rate, overlay.hbm_read_bytes_per_s)
        elif role in overlay_rates and orig_rates_s.get(role, 0.0) > 0.0:
            m_rate = m_rate * (overlay_rates[role] / orig_rates_s[role])
        elif (role == "mte1" and mte1_modes_at_registry and orig_rates_s.get(role, 0.0) > 0.0
              and float(reg_rates.get(role, 0.0) or 0.0) > 0.0):
            m_rate = m_rate * (orig_rates_s[role] / float(reg_rates[role]))
        service_modes[mode] = (role, m_rate)
    hbm_cap_s = orig_rates_s.get("hbm", 0.0)
    if "mte2_aiv_dram" not in service_modes and overlay and overlay.hbm_read_bytes_per_s is not None:
        hbm_r = overlay.hbm_read_bytes_per_s
        service_modes["mte2_aiv_dram"] = ("mte2_aiv", min(hbm_r, hbm_cap_s) if hbm_cap_s > 0.0 else hbm_r)
    for attr, mode_name in (
        ("hbm_write_bytes_per_s", "hbm_write_dram"),
        ("hbm_rw11_bytes_per_s", "hbm_rw11_dram"),
        ("hbm_rw21_bytes_per_s", "hbm_rw21_dram"),
    ):
        val = getattr(overlay, attr, None) if overlay is not None else None
        if val is not None:
            service_modes[mode_name] = ("hbm", min(val, hbm_cap_s) if hbm_cap_s > 0.0 else val)
    l2_pipe = "mte2_aiv" if "mte2_aiv" in names else ("mte2" if "mte2" in names else None)
    if l2_pipe is not None and overlay and overlay.l2_die_bytes_per_s is not None:
        service_modes["l2_die_bw"] = (l2_pipe, overlay.l2_die_bytes_per_s)
    if l2_pipe is not None and overlay and overlay.l2_contiguous_bytes_per_s is not None:
        service_modes["l2_contiguous_bw"] = (l2_pipe, overlay.l2_contiguous_bytes_per_s)
    return service_modes


def build_stage_profiles(case: CaseSpec, namespace: str = "", *,
                         replicas=None, primitive_table: Any = None) -> tuple[StageProfile, ...]:
    """Resolve every stage's immutable hardware binding exactly once."""
    p = case.runtime.parallel
    placement = case.machine.validate_placement(p)
    width = p.tp * p.pcp
    selected = range(p.dp) if replicas is None else tuple(replicas)
    total_replicas = max(1, int(p.external_dp) * int(p.dp))
    profiles: list[StageProfile] = []
    for replica in selected:
        rep_i = int(replica)
        if rep_i < 0 or rep_i >= total_replicas:
            raise ModelingError("placement_exceeds_physical_devices")
        for stage in range(p.pp):
            ranks = placement.stage_ranks(rep_i, stage)
            dies = tuple(placement.die_for_rank(rank) for rank in ranks)
            if len(dies) != width:
                raise ModelingError("placement_exceeds_physical_devices")
            geometry = _execution_geometry(dies[0])
            if any(_execution_geometry(die) != geometry for die in dies[1:]):
                raise ModelingError(
                    "representative_shard_requires_matching_resource_geometry:"
                    f"replica={replica}:stage={stage};"
                    "heterogeneous_owners_require_per_rank_execution")
            compute_arch, datapath = datapath_for_devices(
                dies, links=case.machine.links,
                transfer_channels=case.machine.transfer_channels,
            )
            peak_aliases = dict(dies[0].compute_peak_aliases)
            roles = dict(dies[0].rates)
            work_units = {**NATIVE_WORK_UNITS, **dict(dies[0].work_units)}
            orig_rates_s = {
                role: min(dict(die.rates)[role] for die in dies)
                for role in roles
            }
            overlay = None
            dispatch_envelope = None
            overlay_rates: dict[str, float] = {}
            overlay_peaks: dict[str, float] = {}
            if primitive_table is not None:
                from mindie_sim.primitives.loader import resolve_dispatch_envelope, resolve_sim_rate_overlay
                overlay = resolve_sim_rate_overlay(primitive_table, datapath, sku=dies[0].sku)
                dispatch_envelope = resolve_dispatch_envelope(primitive_table, datapath)
                overlay_rates = {
                    role: (min(val, orig_rates_s[role]) if orig_rates_s.get(role, 0.0) > 0.0 else val)
                    for role, val in overlay.rates
                }
                overlay_peaks = {
                    dt: pk for dt, pk in dict(overlay.compute_peaks).items()
                    if dt not in peak_aliases
                    and (not compute_arch.has_native_modes() or compute_arch.supports_native("matrix", dt))
                }
                hbm_cap_s = orig_rates_s.get("hbm", 0.0)
                dp_updates: dict[str, Any] = {}
                if overlay.hbm_read_bytes_per_s is not None:
                    hbm_r = overlay.hbm_read_bytes_per_s
                    dp_updates["hbm_read_bytes_per_s"] = (
                        min(hbm_r, hbm_cap_s) if hbm_cap_s > 0.0 else hbm_r
                    )
                if overlay.hbm_write_bytes_per_s is not None:
                    hbm_w = overlay.hbm_write_bytes_per_s
                    dp_updates["hbm_write_bytes_per_s"] = (
                        min(hbm_w, hbm_cap_s) if hbm_cap_s > 0.0 else hbm_w
                    )
                if overlay.hbm_rw_alpha is not None:
                    dp_updates["hbm_rw_alpha"] = overlay.hbm_rw_alpha
                if overlay_peaks:
                    merged_caps = {**dict(datapath.compute_peaks), **overlay_peaks}
                    dp_updates["compute_peaks"] = tuple(sorted(merged_caps.items()))
                datapath = replace(datapath, **dp_updates)
            rates = {
                role: (overlay_rates.get(role, orig_rates_s[role]) / MS_PER_S,
                       work_units.get(role, "work"))
                for role in roles
            }
            rates.update({"cube": (1, "peak_ms"), "host": (1, "ms")})
            names: dict[str, str] = {}
            stage_resources: list[Resource] = []
            for kind, (rate, unit) in rates.items():
                label = {"cube": "matrix", "hbm": "memory"}.get(kind, kind)
                name = f"{namespace}{label}@r{replica}:s{stage}"
                stage_resources.append(Resource(
                    name, rate, unit, "fifo" if kind == "host" else "fair",
                    dies[0].units(kind)))
                names[kind] = name
            host_rc = root_complex_resources_for_dies(
                case.machine, dies, namespace=namespace
            )
            stage_pools: dict[str, str] = {}
            pools: dict[str, float] = {}
            pool_units: dict[str, int] = {}
            for pool in dict(dies[0].buffers):
                key = f"{namespace}{pool}@r{replica}:s{stage}"
                pools[key] = min(dict(die.buffers)[pool] for die in dies)
                pool_units[key] = dies[0].pool_units(pool)
                stage_pools[str(pool)] = key
            orig_compute_peaks = {
                dtype: min(die.peak(dtype) for die in dies)
                for dtype, _ in dies[0].compute_peaks
                if dtype not in peak_aliases
            }
            compute_peaks = {**orig_compute_peaks, **overlay_peaks}
            orig_service_modes = {
                mode: (role, min(dict(die.service_modes)[mode][1] for die in dies))
                for mode, (role, _) in dies[0].service_modes
            }
            declared_physical_rates = {
                "resources": {**orig_rates_s, "cube": MS_PER_S, "host": MS_PER_S},
                "compute_peaks": orig_compute_peaks,
                "service_modes": orig_service_modes,
            }
            service_modes = _resolve_stage_service_modes(
                dies, orig_rates_s, orig_service_modes, overlay_rates, overlay, names,
            )
            store_queues = {
                name: dict(queue) for name, queue in dies[0].store_queues
                if all(dict(die.store_queues).get(name) == queue for die in dies)
            }
            profiles.append(StageProfile(
                replica=int(replica), stage=int(stage), dies=dies, geometry=geometry,
                rates=rates, names=names, resources=tuple(stage_resources),
                stage_pools=stage_pools, pools=pools, pool_units=pool_units,
                units={role: dies[0].units(role) for role in names},
                pool_units_by_name={pool: dies[0].pool_units(pool) for pool in dict(dies[0].buffers)},
                compute_peaks=compute_peaks,
                pool_capacities={pool: min(dict(die.buffers)[pool] for die in dies) for pool in dict(dies[0].buffers)},
                service_modes=service_modes, work_units=work_units, store_queues=store_queues,
                physical_mechanisms=dies[0].physical_mechanisms, compute_arch=compute_arch,
                datapath=datapath, host_rc_resources=host_rc, primitive_table=primitive_table,
                primitive_identity=() if overlay is None else overlay.identity,
                primitive_coverage_gaps=() if overlay is None else overlay.coverage_gaps,
                dispatch_envelope=dispatch_envelope,
                declared_physical_rates=declared_physical_rates,
            ))
    return tuple(profiles)


def _admission_l2_cache_key(
    view: Any, die_id: int,
) -> tuple[int | None, str | None] | None:
    """Live per-die L2 budget and residency fingerprint for prepare/program caches."""
    if view is None:
        return None
    eff_fn = getattr(view, "effective_l2_bytes", None)
    fp_fn = getattr(view, "residency_fingerprint", None)
    caps = getattr(view, "_l2_capacity", None)
    eff = (
        int(eff_fn(die_id))
        if callable(eff_fn) and (caps is None or die_id in caps)
        else None
    )
    fp = str(fp_fn(die_id)) if callable(fp_fn) else None
    return (eff, fp)


def _live_residency_op_scope(view: Any, die_id: int, op: Any) -> Any:
    """Distinguish absolute producer bindings when ``view`` has live resident tensors on ``die_id``."""
    resident = getattr(view, "_resident", None)
    if not resident or not any(
        entry.key.die == die_id and not entry.evicted for entry in resident.values()
    ):
        return None
    from ..program.implementation_audit import input_ports

    return (op.stage, op.layer, op.deps, _freeze_key(input_ports(op)))


class _ContextMixin:
    """Operator preparation, stage context binding, and program lowering."""

    @staticmethod
    def lower_program(context, recipe):
        recipe = context.program_recipe or recipe
        if context.joint_native_service is not None:
            return KernelProgram((Stage("joint_native_service", context.work),))
        from ..program.invocation import lower_invocation
        return lower_invocation(recipe, context)

    def transport_options_for(self, op, replica: int = 0, scope: str = "none"):
        """Return the exact TransportOptions bound for this occurrence."""
        return self._transport_options.get((str(scope), op.op_id, replica))

    def _apply_resource_service_program(self, context, program):
        binding = dict(context.op.metadata).get("resource_service")
        if binding is None:
            return program
        if not isinstance(binding, _AbcMapping):
            raise ModelingError("resource_service_binding_requires_mapping")
        service_id = binding.get("service_id")
        service = self.resource_services.get(service_id)
        if service is None:
            raise ModelingError(f"resource_service_binding_unknown:{service_id}")
        if service.scope != "independent_resource" or service.kind != "resource_rate":
            raise ModelingError("resource_service_binding_requires_independent_rate")
        role = binding.get("resource_role")
        physical = context.resource_names.get(role)
        if physical is None:
            raise ModelingError(f"resource_service_role_unbound:{role}")
        work_unit = binding.get("work_unit", service.work_unit)
        expected = context.work_units.get(role, service.work_unit)
        if service.work_unit != work_unit or expected != work_unit:
            raise ModelingError(
                f"resource_service_work_unit_mismatch:{service.service_id}:"
                f"{service.work_unit}:{work_unit}:{expected}")
        if (service.phase and binding.get("phase")
                and service.phase != binding.get("phase")):
            raise ModelingError(
                f"resource_service_phase_mismatch:{service.service_id}")
        if not isinstance(program, KernelProgram):
            raise ModelingError("resource_service_requires_kernel_program")
        cap = service.per_owner_rate / MS_PER_S
        stages = []
        for stage in program.stages:
            if physical in stage.work:
                caps = dict(stage.caps)
                caps[physical] = min(caps.get(physical, cap), cap)
                stage = replace(stage, caps=caps)
            stages.append(stage)
        body = KernelProgram(tuple(stages), program.minimum_ms)
        if service.fixed_us > 0:
            return _ServiceSetupProgram(body, service.fixed_us / MS_PER_S)
        return body

    def _apply_stream_owners_program(self, context, program):
        return apply_stream_owners(self.stream_quota_by_res, context, program)

    def _record_prepared_op(self, replica: int, op: Any) -> None:
        """Persist the prepared executable OpSpec on Execution and EvaluationContext."""
        op_id = getattr(op, "op_id", None)
        if not op_id:
            return
        key = (int(replica), str(op_id))
        own_map = getattr(self, "prepared_ops", None)
        if own_map is None:
            own_map = {}
            self.prepared_ops = own_map
        if len(own_map) >= PREPARED_OPS_CACHE_LIMIT and key not in own_map:
            own_map.pop(next(iter(own_map)))
        own_map[key] = op
        evaluation = getattr(self, "evaluation", None)
        if evaluation is not None:
            eval_map = getattr(evaluation, "prepared_ops", None)
            if eval_map is None:
                eval_map = {}
                evaluation.prepared_ops = eval_map
            if len(eval_map) >= PREPARED_OPS_CACHE_LIMIT and key not in eval_map:
                eval_map.pop(next(iter(eval_map)))
            eval_map[key] = op

    def _live_admission_memory_view(self):
        """Return the live engine-attached AdmissionMemoryView, falling back to self."""
        engine = getattr(self, "engine", None)
        view = getattr(engine, "admission_memory_view", None) if engine is not None else None
        return view if view is not None else getattr(self, "admission_memory_view", None)

    def _admission_view_identity(self, view: Any) -> int | None:
        """Return a monotonic per-Execution identity token immune to Python id() reuse."""
        if view is None:
            return None
        tokens = getattr(self, "_admission_view_tokens", None)
        if tokens is None:
            tokens = weakref.WeakKeyDictionary()
            self._admission_view_tokens = tokens
        try:
            token = tokens.get(view)
        except TypeError:
            current = getattr(self, "_last_admission_view", None)
            if current is view:
                return int(self._last_admission_view_token)
            token = int(getattr(self, "_next_admission_view_token", 0)) + 1
            self._next_admission_view_token = token
            self._last_admission_view = view
            self._last_admission_view_token = token
            return token
        if token is None:
            token = int(getattr(self, "_next_admission_view_token", 0)) + 1
            self._next_admission_view_token = token
            tokens[view] = token
        return int(token)

    def prepare(self, op, replica=0):
        from ..efficiency.unmodeled_gate import assert_timing_supported
        assert_timing_supported(None, (op,), entry="execution_prepare")
        stage_key = (replica, op.stage)
        geom = self.stage_geometry[stage_key]
        stream_owners = self.stream_owners.get((replica, op.stage, op.stream or "main"))
        active_reqs = tuple(getattr(self, "_active_requests", ()) or ())
        active_padded = int(getattr(self, "_active_padded_tokens", 0) or 0)
        active_layout = str(getattr(self, "_active_layout", "") or "")
        admission_view = self._live_admission_memory_view()
        if (stream_owners is not None or geom.replica != int(replica)
                or admission_view is not None or active_reqs or active_padded or active_layout):
            geom = replace(
                geom, replica=int(replica), stream_owners=stream_owners,
                admission_memory_view=admission_view,
                requests=active_reqs, padded_tokens=active_padded, layout=active_layout,
            )
        geom_keys = getattr(self, "_stage_geom_keys", None)
        if geom_keys is None:
            geom_keys = {}
            self._stage_geom_keys = geom_keys
        gcache_key = (stage_key, stream_owners, active_reqs, active_padded, active_layout)
        frozen_geom = geom_keys.get(gcache_key)
        if frozen_geom is None:
            frozen_geom = _freeze_key(replace(geom, admission_memory_view=None))
            geom_keys[gcache_key] = frozen_geom
        startup_mode = self.case.runtime.collective_startup_mode
        die_id = 0 if getattr(geom, "rank", None) is None else int(geom.rank)
        l2_state_key = _admission_l2_cache_key(admission_view, die_id)
        l2_op_scope = _live_residency_op_scope(admission_view, die_id, op)
        view_token = self._admission_view_identity(admission_view)
        prep_key = (
            id(self.recipes), frozen_geom, view_token, l2_state_key, l2_op_scope,
            stream_owners, startup_mode, op.stage,
            op.family, op.recipe, op.phase, op.dtype, op.layout, op.requests,
            op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
            _freeze_key(op.precision_abi),
            _freeze_key(op.kernel_binding),
            _freeze_key(op.owner_schedule),
            _freeze_key(op.logical_precision_reqs),
            op.coverage,
            self._op_key(op),
            getattr(geom, "physical_mechanisms").signature
            if getattr(geom, "physical_mechanisms", None) is not None else None,
        )
        if prep_key in self._prepare_cache:
            cached_op, meta_diff, recipe = self._prepare_cache[prep_key]
            meta = op.metadata
            if meta_diff is not None:
                patched = _meta_patch(dict(op.metadata), meta_diff)
                meta = tuple(sorted(patched.items()))
            resolved = replace(
                cached_op, op_id=op.op_id, layer=op.layer, deps=op.deps,
                requests=op.requests, stream=op.stream, notes=op.notes,
                metadata=meta,
            )
            self._record_prepared_op(replica, resolved)
            return resolved, recipe
        res_op, recipe = self.recipes.prepare(op, geom)
        from ..collectives.waves import startup_waves_recipe
        recipe = startup_waves_recipe(res_op, recipe, startup_mode)
        stage_dies = self.stage_dies.get((replica, res_op.stage))
        if stage_dies:
            from mindie_sim.prediction.operators.matrix.matrix_precision import resolve_matrix_precision
            res_op = resolve_matrix_precision(
                res_op, devices=stage_dies,
                policy=self.case.runtime.matrix_compute_policy, recipe=recipe)
        orig_meta = dict(op.metadata)
        res_meta = dict(res_op.metadata)
        meta_diff = _meta_diff(orig_meta, res_meta) if orig_meta != res_meta else None
        if len(self._prepare_cache) >= PREPARE_CACHE_LIMIT:
            self._prepare_cache.pop(next(iter(self._prepare_cache)))
        self._prepare_cache[prep_key] = (res_op, meta_diff, recipe)
        self._record_prepared_op(replica, res_op)
        return res_op, recipe

    def destination_window_resource(self, destination_rank: int, stage: int) -> str | None:
        """Return the declared resident memory resource for one represented rank."""
        for (replica, owner_stage), names in self.stage_names.items():
            if owner_stage != stage:
                continue
            ranks = self.topology.ranks(replica, stage)
            if ranks and ranks[0] == destination_rank:
                return names.get("hbm")
        return None

    def _stage_context(self, key, dies):
        replica, stage = key
        profile = self.stage_profiles.get(key)
        if profile is None:
            names = self.stage_names[key]
            compute_arch, datapath = datapath_for_devices(
                dies, links=self.case.machine.links,
                transfer_channels=self.case.machine.transfer_channels,
            )
            return RecipeContext(
                None, {}, names, self.resource_rates,
                dies[0].aic_count,
                pool_names={pool: f"{self.namespace}{pool}@r{replica}:s{stage}"
                            for pool in dict(dies[0].buffers)},
                resource_units={role: dies[0].units(role) for role in names},
                compute_peaks={dtype: min(d.peak(dtype) for d in dies)
                               for dtype, _ in dies[0].compute_peaks
                               if dtype not in dict(dies[0].compute_peak_aliases)},
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
                physical_mechanisms=dies[0].physical_mechanisms,
                compute_arch=compute_arch,
                datapath=datapath,
                admission_memory_view=getattr(self, "admission_memory_view", None),
                allocation_versions=getattr(self, "allocation_versions", {}),
            )
        return RecipeContext(
            None, {}, profile.names, self.resource_rates,
            profile.dies[0].aic_count,
            pool_names=dict(profile.stage_pools),
            resource_units=dict(profile.units),
            compute_peaks=dict(profile.compute_peaks),
            pool_units=dict(profile.pool_units_by_name),
            pool_capacities=dict(profile.pool_capacities),
            record_mapping=self.cost.record_mapping,
            service_modes=dict(profile.service_modes),
            record_service=self.cost.record_service,
            work_units=dict(profile.work_units),
            store_queues={name: dict(queue)
                          for name, queue in profile.store_queues.items()},
            physical_mechanisms=profile.physical_mechanisms,
            compute_arch=profile.compute_arch,
            datapath=profile.datapath,
            primitive_table=profile.primitive_table,
            primitive_coverage_gaps=profile.primitive_coverage_gaps,
            declared_physical_rates=profile.declared_physical_rates,
            admission_memory_view=getattr(self, "admission_memory_view", None),
            allocation_versions=getattr(self, "allocation_versions", {}),
        )

    def _context(self, op: Any, replica: int, scope: str, *, recipe=None) -> RecipeContext:
        from ..collectives.waves import startup_waves_recipe
        from mindie_sim.prediction.operators.collectives.direct_peer import resolve_transport_options
        from mindie_sim.prediction.operators.matrix.matrix_precision import execution_compute_peaks, resolve_matrix_precision

        recipe = startup_waves_recipe(
            op, recipe, self.case.runtime.collective_startup_mode)
        dies = self.stage_dies[replica, op.stage]
        orig_op = op
        op = resolve_matrix_precision(
            op, devices=dies,
            policy=self.case.runtime.matrix_compute_policy, recipe=recipe)
        if op is not orig_op or (replica, op.op_id) not in getattr(self, "prepared_ops", ()):
            self._record_prepared_op(replica, op)
        cost = self.cost.price(op, scope)
        template = self.stage_contexts[replica, op.stage]
        transport_options = (resolve_transport_options(recipe, op)
                             if recipe is not None else TransportOptions())
        self._transport_options[(str(scope), op.op_id, replica)] = transport_options
        peaks = execution_compute_peaks(
            op, devices=dies, base_peaks=template.compute_peaks)
        sid = self._cached_service_identity(op, replica, scope, dies[0])
        declared = _declaration_of(op)
        stream_owners = self.stream_owners.get((replica, op.stage, op.stream or "main"))
        live_admission_view = self._live_admission_memory_view()
        context = replace(
            template, op=op, native_work=cost.work, compute_peaks=peaks,
            service_identity=sid,
            service_boundary="resource_phase" if declared else "device_kernel",
            stream_owners=stream_owners,
            admission_memory_view=(
                template.admission_memory_view if live_admission_view is None
                else live_admission_view
            ),
            requests=tuple(getattr(self, "_active_requests", ()) or ()),
            padded_tokens=int(getattr(self, "_active_padded_tokens", 0) or 0),
            layout=str(getattr(self, "_active_layout", "") or ""))
        bind_context = getattr(recipe, "bind_context", None)
        if bind_context is not None:
            context = bind_context(context)
        demands, frames, setup, plan_binding, phase_links = self._bind_context_work(
            context, op, replica, recipe, cost.work, transport_options)
        diagnostic = (self.cost.lookup_diagnostic(op, scope)
                      if self.legacy_diagnostics else None)
        diagnostic_timing = (diagnostic or {}).get("timing")
        measured = (diagnostic_timing.duration_ms
                    if diagnostic_timing is not None else 0)
        physical = max((value / self.engine.resources[key].rate
                        for key, value in demands.items()
                        if key in self.engine.resources), default=0)
        context = replace(
            context, work=demands,
            minimum_ms=measured if self.legacy_diagnostics else 0.0,
            setup_ms=setup,
            service_identity={**dict(context.service_identity),
                              "transport_options": transport_options},
            transaction_frames=frames,
            collective_plan=plan_binding,
            phase_link_work=phase_links)
        response = (self.regimes.resolve(context, self.cost.record_regime)
                    if self.legacy_diagnostics or self.regimes.rules else None)
        if response is not None and response.mode == "joint_native":
            if getattr(recipe, "algorithm", None):
                raise ModelingError("joint_native_service_requires_single_local_kernel")
            demand = response.bind_joint(context)
            self.cost.record_joint_native(response, cost)
            declaration = {**response.execution, **response.diagnostics(),
                           "reference_program": type(recipe).__name__,
                           "reference_primitive_ms": (
                               measured if diagnostic_timing is not None else None)}
            return replace(context, work=demand, minimum_ms=0, setup_ms=0,
                           invocation_timing=None, joint_native_service=declaration)
        if (response is not None and response.mode == "mac_service"
                and self.legacy_diagnostics):
            context = self._bind_mac_service(context, recipe, cost, response)
        self.conflicts += int(bool(measured and measured < physical))
        self._last_invocation_record = None
        selected = (None if getattr(recipe, "native", False)
                    else self.invocations.select(context, recipe, devices=dies))
        if not getattr(recipe, "native", False):
            profile = self.stage_profiles.get((replica, op.stage))
            envelope = getattr(profile, "dispatch_envelope", None) if profile is not None else None
            selected = _apply_calibrated_dispatch_floor(context, dies, selected, envelope)
        if selected is not None:
            timing, record = selected
            from ..program.invocation import _declared_launch_count
            launch_count = _declared_launch_count(context.program_recipe or recipe, context)
            if launch_count > 0:
                self._last_invocation_record = record
                for _ in range(launch_count):
                    self.cost.record_invocation(record)
                context = replace(context, invocation_timing=timing)
        if response is not None:
            if response.mode == "mac_service":
                context = replace(context, service_regime=response.diagnostics())
            else:
                existing = float(context.minimum_ms or 0.0)
                resp_prov = getattr(response, "provenance", None)
                provenance = (
                    dict(resp_prov or {})
                    if float(response.duration_ms) >= existing
                    else dict(getattr(context, "minimum_provenance", {}) or {})
                )
                context = replace(
                    context,
                    minimum_ms=max(existing, response.duration_ms),
                    minimum_provenance=provenance,
                    service_regime=response.diagnostics(),
                )
        return context

    def _cached_service_identity(self, op, replica, scope, device):
        from ..service_regime_domains.services import service_identity

        meta = _metadata_dict(op)
        units_key = _DEVICE_UNITS_CACHE.get(id(device))
        if units_key is None or units_key[0] is not device:
            units_key = (device, _freeze_key(dict(device.resource_units)))
            if len(_DEVICE_UNITS_CACHE) >= DEVICE_UNITS_CACHE_LIMIT:
                _DEVICE_UNITS_CACHE.clear()
            _DEVICE_UNITS_CACHE[id(device)] = units_key
        sid_key = (
            replica, op.stage, device.rank, device.compute_units,
            units_key[1],
            op.family, op.recipe, op.dtype, op.layout, scope,
            meta.get("kernel"), meta.get("kernel_name"),
            _freeze_key(meta.get("implementation")),
            _freeze_key(meta.get("service_axes")),
            bool(meta.get("hccl_selection")),
            self._op_key(op),
        )
        sid = self._service_identity_cache.get(sid_key)
        if sid is None:
            sid = service_identity(self.case, op, scope, device=device)
            if len(self._service_identity_cache) >= SERVICE_IDENTITY_CACHE_LIMIT:
                self._service_identity_cache.pop(next(iter(self._service_identity_cache)))
            self._service_identity_cache[sid_key] = sid
        return sid

    def _route_link_phase(self, phase_op, replica, volume, transport_options, *, algorithm):
        band_plan = None
        if volume and algorithm in {"ring", "pairwise", "auto"}:
            try:
                band_plan = self.topology.band_collective_plan(
                    phase_op, {replica: volume}, algorithm=algorithm,
                    transport=transport_options,
                )
            except ModelingError as exc:
                if not str(exc).startswith("joint_collective_group_membership"):
                    raise
        if band_plan is not None:
            band_plan.register_die_injection_resources(self.engine)
        if (band_plan is not None
                and not set(band_plan.accounting().wire_by_resource).issubset(self.engine.resources)):
            local_ranks = frozenset(
                int(die.rank) for die in self.stage_dies[replica, phase_op.stage]
            )
            band_plan = band_plan.scoped_to_ranks(
                local_ranks, available_resources=self.engine.resources
            )
        routed = (
            band_plan.accounting()
            if band_plan is not None
            else self.topology.accounting(
                phase_op, replica, volume=volume, transport=transport_options
            )
        )
        wire = dict(routed.wire_by_resource)
        if not set(wire).issubset(self.engine.resources):
            wire = {k: v for k, v in wire.items() if k in self.engine.resources}
        return wire, routed.transaction_frames, float(routed.setup_ms)

    def _route_megamoe_link_phases(self, op, replica, amount, transport_options):
        meta = _metadata_dict(op)
        mega = dict(meta.get("megamoe_geometry") or {})
        disp_vol = float(mega.get("dispatch_link_bytes", 0.0) or 0.0)
        comb_vol = float(mega.get("combine_link_bytes", 0.0) or 0.0)
        disp_peer = mega.get("dispatch_peer_traffic", meta.get("dispatch_peer_traffic"))
        comb_peer = mega.get("combine_peer_traffic", meta.get("combine_peer_traffic"))
        declared_sum = disp_vol + comb_vol
        if declared_sum <= 0.0:
            disp_vol, comb_vol = float(amount), 0.0
        elif (disp_peer is None and comb_peer is None
              and not math.isclose(declared_sum, float(amount), rel_tol=1e-9, abs_tol=1e-9)):
            scale = float(amount) / declared_sum
            disp_vol *= scale
            comb_vol *= scale
        axis = str(mega.get("group_axis") or meta.get("group_axis") or "ep")
        collective = str(mega.get("collective") or meta.get("collective") or "alltoall")
        demands: dict[str, float] = {}
        frames: dict[str, tuple[int, int]] = {}
        phase_wires: dict[str, dict[str, float]] = {}
        setup = 0.0
        for phase_name, phase_vol, phase_peer in (
            ("dispatch", disp_vol, disp_peer),
            ("combine", comb_vol, comb_peer),
        ):
            phase_meta = dict(meta)
            phase_meta["group_axis"] = axis
            phase_meta["collective"] = collective
            phase_meta.pop("peer_traffic", None)
            if phase_peer is not None:
                phase_meta["peer_traffic"] = phase_peer
            phase_op = replace(
                op, family=collective, link_bytes=phase_vol,
                metadata=tuple(sorted(phase_meta.items())),
            )
            if phase_vol <= 0.0:
                if phase_peer is not None:
                    self.topology.accounting(phase_op, replica, volume=0.0)
                phase_wires[phase_name] = {}
                continue
            alg = "pairwise" if phase_peer is not None else "ring"
            wire, p_frames, p_setup = self._route_link_phase(
                phase_op, replica, phase_vol, transport_options, algorithm=alg,
            )
            phase_wires[phase_name] = wire
            for res_name, val in wire.items():
                demands[res_name] = demands.get(res_name, 0.0) + val
            frames.update(p_frames)
            setup += p_setup
        return demands, frames, setup, phase_wires

    def _bind_context_work(self, context, op, replica, recipe, work, transport_options):
        meta = _metadata_dict(op)
        is_megamoe = isinstance(meta.get("megamoe_geometry"), _AbcMapping)
        explicit_algorithm = getattr(recipe, "algorithm", None)
        algorithm = explicit_algorithm or "ring"
        plan_algorithm = explicit_algorithm or "auto"
        if (not is_megamoe and algorithm == "ring" and getattr(recipe, "mesh_capable", False)
                and self.topology.declares_direct_mesh(op, replica)):
            algorithm = "direct_mesh"
        band_plan = None
        link_vol = work.get("link", 0)
        if is_megamoe:
            plan_binding = None
        elif algorithm == "direct_mesh":
            plan = self.topology.joint_plan(
                op, {replica: link_vol},
                algorithm=algorithm, transport=transport_options)
            self.topology.bind_direct_mesh(op, replica, plan)
            plan_binding = _CollectivePlanBinding(algorithm, plan=plan)
        else:
            if link_vol and plan_algorithm in {"ring", "pairwise", "auto"}:
                try:
                    band_plan = self.topology.band_collective_plan(
                        op, {replica: link_vol}, algorithm=plan_algorithm,
                        transport=transport_options,
                    )
                except ModelingError as exc:
                    if not str(exc).startswith("joint_collective_group_membership"):
                        raise
            if band_plan is not None:
                band_plan.register_die_injection_resources(self.engine)
            if (band_plan is not None
                    and not set(band_plan.accounting().wire_by_resource).issubset(self.engine.resources)):
                local_ranks = frozenset(
                    int(die.rank) for die in self.stage_dies[replica, op.stage]
                )
                band_plan = band_plan.scoped_to_ranks(
                    local_ranks, available_resources=self.engine.resources
                )
            if band_plan is not None:
                plan_binding = _CollectivePlanBinding(algorithm, plan=band_plan)
            else:
                plan_binding = _CollectivePlanBinding(
                    algorithm,
                    factory=lambda: self.topology.joint_plan(
                        op, {replica: link_vol},
                        algorithm=algorithm, transport=transport_options))
        demands, frames, setup, phase_links = {}, {}, 0.0, {}
        if "peer_traffic" in meta and not link_vol and not is_megamoe:
            self.topology.accounting(op, replica, volume=link_vol)
        for resource, amount in work.items():
            if resource == "link":
                if is_megamoe:
                    link_demands, link_frames, link_setup, phase_links = (
                        self._route_megamoe_link_phases(
                            op, replica, amount, transport_options,
                        )
                    )
                    for name, value in link_demands.items():
                        demands[name] = demands.get(name, 0.0) + value
                    frames.update(link_frames)
                    setup += link_setup
                    continue
                routed = (
                    band_plan.accounting()
                    if band_plan is not None
                    else self.topology.accounting(
                        op, replica, volume=amount, transport=transport_options)
                )
                demands.update(routed.wire_by_resource)
                frames.update(routed.transaction_frames)
                setup = routed.setup_ms
                continue
            if resource == "cube":
                portions = op.matrix_flops or ((op.dtype or self.case.model.compute_dtype, amount),)
                bound = context.bind(Work(matrix=dict(portions)))
            else:
                bound = context.bind(Work(resources={resource: amount}))
            for name, value in bound.items():
                demands[name] = demands.get(name, 0) + value
        return demands, frames, setup, plan_binding, phase_links

    def _bind_mac_service(self, context, recipe, cost, response):
        matrix_role = str(response.execution.get("resource") or "cube")
        matrix_name = context.resource_names.get(matrix_role)
        if (not matrix_name or response.issued_work is None
                or response.rate_flops_per_s is None
                or not response.active_owners):
            raise ModelingError("mac_service_requires_bound_cube_work_and_rate")
        issued = float(response.issued_work)
        owners = int(response.active_owners)
        mac_work_ms = float(response.duration_ms)
        dtype_rate = float(context.compute_peaks.get(context.op.dtype, 0.0))
        try:
            physical_ms = issued / (dtype_rate / MS_PER_S)
        except ZeroDivisionError:
            physical_ms = None
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
        from mindie_sim.prediction.operators.matrix.matrix import TiledMatmul
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
        return replace(context, work=work, mac_caps={},
                       mac_service=declaration, program_recipe=program_recipe)

    def _program_inputs(self, context, recipe, engine, replica, op, owner):
        """Virtual-time-free lowering inputs of one operator occurrence."""
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
        do_compact = not self.legacy_diagnostics and (
            not engine.trace.maxlen or self._compact_step
        )
        cp_binding = getattr(context, "collective_plan", None)
        cp_plan = getattr(cp_binding, "plan", None) if cp_binding is not None else None
        band_key = cp_plan.cache_key() if hasattr(cp_plan, "cache_key") else None
        stage_geom = (
            self.stage_geometry.get((replica, op.stage))
            if replica is not None else None
        )
        die_rank = getattr(stage_geom, "rank", None)
        if die_rank is None and replica is not None:
            dies = self.stage_dies.get((replica, op.stage))
            if dies:
                die_rank = int(dies[0].rank)
        die_id = 0 if die_rank is None else int(die_rank)
        l2_state_key = _admission_l2_cache_key(
            getattr(context, "admission_memory_view", None), die_id,
        )
        prog_key = (
            owner, do_compact, _freeze_key(recipe), op.family, op.recipe, op.phase, op.stage,
            op.dtype, op.layout, op.requests, op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
            _freeze_key(op.precision_abi),
            _freeze_key(op.kernel_binding),
            _freeze_key(op.owner_schedule),
            _freeze_key(op.logical_precision_reqs),
            op.coverage,
            self._op_key(op), context.setup_ms, context.minimum_ms,
            (inv_t.startup_ms, inv_t.minimum_ms) if inv_t is not None else None,
            _freeze_key(context.joint_native_service),
            tuple(sorted(context.work.items())),
            tuple(sorted(context.service_caps.items())),
            tuple((k, tuple(sorted(v.items()))) for k, v in sorted(context.phase_link_work.items())),
            live_rates,
            context.group_buffers,
            getattr(context.physical_mechanisms, "signature", None),
            op.stream or "main",
            context.stream_owners,
            band_key,
            l2_state_key,
        )
        token = _rank_token(context)
        replayable = (
            type(self).schedule_replay and token is not None
            and not op.link_bytes and not getattr(recipe, "algorithm", None)
            and dict(op.metadata).get("group_axis") not in {"ep", "local_tp"}
        )
        return transport, prog_key, do_compact, replayable

    def _lower_operator_program(self, context, recipe, engine, replica, name,
                                prog_key, do_compact):
        """Return ``((program, mapped, services), from_cache)`` for one operator."""
        hit = self._program_cache.get(prog_key)
        if hit is None and self._shared_eval_cache is not None:
            hit = self._shared_eval_cache.get(prog_key)
            if hit is not None:
                self._program_cache[prog_key] = hit
        if hit is not None:
            return hit, True
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
        cp_binding = getattr(context, "collective_plan", None)
        cp_plan = getattr(cp_binding, "plan", None) if cp_binding is not None else None
        if (cp_plan is not None
                and hasattr(cp_plan, "lower_with_local_work")
                and getattr(recipe, "algorithm", None) is None
                and getattr(cp_binding, "algorithm", None) != "direct_mesh"
                and context.joint_native_service is None):
            program = cp_plan.lower_with_local_work(context, program)
        program = self._apply_resource_service_program(context, program)
        program = self._apply_stream_owners_program(context, program)
        if do_compact:
            from ..program import coalesce_execution_program
            program = coalesce_execution_program(program)
        entry = (program, tuple(mapped_list), tuple(svc_list))
        if len(self._program_cache) >= PROGRAM_CACHE_ENTRY_LIMIT:
            self._program_cache.pop(next(iter(self._program_cache)))
        self._program_cache[prog_key] = entry
        if self._shared_eval_cache is not None:
            self._shared_eval_cache.put(prog_key, entry)
        return entry, False

    def _program_uses_fifo(self, context, engine) -> bool:
        """True when FIFO arbitration could order this program by sequence."""
        names = engine.fifo_resource_names()
        if not names:
            return False
        if engine._fifo_live > 0:
            return True
        for resource in context.resource_names.values():
            if resource in names:
                return True
        return False

    def _post_local_tp_transform(self, ops):
        """Apply L3 TP/Local-TP MC2 overlap fusion after L2 Local-TP sharding."""
        if (
            "tp_mc2_overlap" in tuple(getattr(self.case.runtime, "lowering_passes", ()) or ())
            or bool(dict(getattr(self.case.runtime, "metadata", ()) or ()).get("tp_mc2_overlap", False))
        ):
            from ..operators.mc2.mc2_tp_overlap import tp_mc2_overlap_fusion
            act_dt = str(getattr(self.case.model, "activation_dtype", "bf16") or "bf16")
            return tuple(tp_mc2_overlap_fusion(ops, act_dt))
        return ops

    def _lower_local_tp_work(self, work: IterationWork, replica: int) -> IterationWork:
        """Apply the declared node-local dense placement once per graph."""
        parallel = self.case.runtime.parallel
        if not parallel.has_local_tp:
            return work
        from ..sharding.local_tp import lower_local_projection_tp

        cohort_views = self._cohort_views
        if cohort_views is None:
            local_width = max(
                int(parallel.tp), int(parallel.q_proj_tp), int(parallel.o_proj_tp),
                int(parallel.linear_proj_tp), int(parallel.dcp_linear_tp),
                int(parallel.mlp_tp), int(parallel.shared_expert_tp), int(parallel.lm_head_tp),
            )
            if (int(parallel.tp) == 1 and local_width > 1
                    and (work.plan.requests or work.mode != "graph")):
                from ..sharding.local_tp import local_tp_replica_cohorts
                cohorts = local_tp_replica_cohorts(
                    self.case.machine, parallel)
                peers = next((group for group in cohorts if replica in group),
                             (replica,))
                cohort_views = {
                    int(rank): work.plan.requests
                    for peer in peers
                    for rank in self.topology.ranks(peer, 0)
                }
            else:
                cohort_views = (
                    {int(rank): work.plan.requests
                     for rank in self.topology.ranks(replica, 0)}
                    if work.plan.requests or work.mode != "graph"
                    else {}
                )
        ops = lower_local_projection_tp(
            work.ops, model=self.case.model, runtime=self.case.runtime,
            machine=self.case.machine, replica=replica,
            cohort_views=cohort_views, scope="target")
        ops = self._post_local_tp_transform(ops)
        return work if ops is work.ops else replace(work, ops=ops)
