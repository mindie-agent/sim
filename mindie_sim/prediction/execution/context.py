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
PREPARE_CACHE_LIMIT = 512
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
_NON_KERNEL_FAMILIES = frozenset({
    "barrier", "token_commit", "collective", "allreduce", "allgather",
    "reduce_scatter", "alltoall", "alltoallv", "transport",
})


def _resolve_mapped_hardware_invocation(dies, roles: tuple[str, ...]):
    """Resolve hardware invocation timing across ``dies`` for canonical role aliases."""
    selected = [
        {"rank": device.rank, "class": role, **timing.to_dict()}
        for device in dies for role, timing in device.kernel_invocations if role in roles
    ]
    if not selected:
        return None
    timing = InvocationTiming(
        startup_ms=max(row["startup_ms"] for row in selected),
        minimum_ms=max(row["minimum_ms"] for row in selected),
        provenance={"basis": "hardware_execution_class", "declarations": selected},
    )
    payload = json.dumps(timing.to_dict(), sort_keys=True, separators=(",", ":"))
    return timing, {
        "id": hashlib.sha256(payload.encode()).hexdigest(),
        "match": {"execution_classes": roles},
        "source": "device.kernel_invocations",
        **timing.to_dict(),
    }


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
        floor_ms = envelope.aiv_floor_ms(_active_aiv_blocks(op, meta, envelope))
        kind = "vector"
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
    target_min_ms = max(timing.minimum_ms, floor_ms)
    if target_min_ms <= timing.minimum_ms and target_startup_ms <= timing.startup_ms:
        return selected
    evidence = f"p15_p13:{envelope.sku}:{envelope.envelope_sha256[:DIGEST_PREFIX_CHARS]}:{kind}"
    upgraded = InvocationTiming(
        startup_ms=target_startup_ms,
        minimum_ms=target_min_ms,
        provenance={**dict(timing.provenance), "dispatch_envelope": evidence},
    )
    payload = json.dumps(upgraded.to_dict(), sort_keys=True, separators=(",", ":"))
    return upgraded, {
        **record,
        "id": hashlib.sha256(payload.encode()).hexdigest(),
        "startup_ms": target_startup_ms,
        "minimum_ms": target_min_ms,
        "provenance": upgraded.provenance,
    }


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

    @property
    def profile_key(self) -> tuple:
        return (int(self.stage), self.geometry, self.primitive_identity)


def build_stage_profiles(case: CaseSpec, namespace: str = "", *,
                         replicas=None, primitive_table: Any = None) -> tuple[StageProfile, ...]:
    """Resolve every stage's immutable hardware binding exactly once."""
    p = case.runtime.parallel
    width = p.tp * p.pcp
    selected = range(p.dp) if replicas is None else tuple(replicas)
    profiles: list[StageProfile] = []
    for replica in selected:
        for stage in range(p.pp):
            start = (int(replica) * p.pp + stage) * width
            dies = tuple(case.machine.dies[start:start + width])
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
                overlay_peaks = dict(overlay.compute_peaks)
                dp_updates: dict[str, Any] = {}
                if overlay.hbm_read_bytes_per_s is not None:
                    dp_updates["hbm_read_bytes_per_s"] = overlay.hbm_read_bytes_per_s
                if overlay.hbm_write_bytes_per_s is not None:
                    dp_updates["hbm_write_bytes_per_s"] = overlay.hbm_write_bytes_per_s
                if overlay.hbm_rw_alpha is not None:
                    dp_updates["hbm_rw_alpha"] = overlay.hbm_rw_alpha
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
            peak_aliases = dict(dies[0].compute_peak_aliases)
            compute_peaks = {
                dtype: min(die.peak(dtype) for die in dies)
                for dtype, _ in dies[0].compute_peaks
                if dtype not in peak_aliases
            }
            for dt, peak in overlay_peaks.items():
                if dt in compute_peaks or dt not in peak_aliases:
                    compute_peaks[dt] = peak
            service_modes = {}
            for mode, (role, _) in dies[0].service_modes:
                m_rate = min(dict(die.service_modes)[mode][1] for die in dies)
                if mode in {"mte2_aic_dram", "mte2_aiv_dram"} and overlay and overlay.hbm_read_bytes_per_s is not None:
                    m_rate = min(m_rate, overlay.hbm_read_bytes_per_s)
                elif role in overlay_rates and orig_rates_s.get(role, 0.0) > 0.0:
                    m_rate = m_rate * (overlay_rates[role] / orig_rates_s[role])
                service_modes[mode] = (role, m_rate)
            if "mte2_aiv_dram" not in service_modes and overlay and overlay.hbm_read_bytes_per_s is not None:
                service_modes["mte2_aiv_dram"] = ("mte2_aiv", overlay.hbm_read_bytes_per_s)
            if overlay and overlay.hbm_write_bytes_per_s is not None:
                service_modes["hbm_write_dram"] = ("hbm", overlay.hbm_write_bytes_per_s)
            if overlay and overlay.hbm_rw11_bytes_per_s is not None:
                service_modes["hbm_rw11_dram"] = ("hbm", overlay.hbm_rw11_bytes_per_s)
            if overlay and overlay.hbm_rw21_bytes_per_s is not None:
                service_modes["hbm_rw21_dram"] = ("hbm", overlay.hbm_rw21_bytes_per_s)
            l2_pipe = "mte2_aiv" if "mte2_aiv" in names else ("mte2" if "mte2" in names else None)
            if l2_pipe is not None and overlay and overlay.l2_die_bytes_per_s is not None:
                service_modes["l2_die_bw"] = (l2_pipe, overlay.l2_die_bytes_per_s)
            if l2_pipe is not None and overlay and overlay.l2_contiguous_bytes_per_s is not None:
                service_modes["l2_contiguous_bw"] = (l2_pipe, overlay.l2_contiguous_bytes_per_s)
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
            ))
    return tuple(profiles)


class _ContextMixin:
    """Operator preparation, stage context binding, and program lowering."""

    @staticmethod
    def lower_program(context, recipe):
        recipe = context.program_recipe or recipe
        if context.joint_native_service is not None:
            return KernelProgram((Stage("joint_native_service", context.work),))
        from ..invocation_binding import lower_invocation
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

    def prepare(self, op, replica=0):
        from ..unmodeled_gate import assert_timing_supported
        assert_timing_supported(None, (op,), entry="execution_prepare")
        stage_key = (replica, op.stage)
        geom = self.stage_geometry[stage_key]
        stream_owners = self.stream_owners.get((replica, op.stage, op.stream or "main"))
        active_reqs = tuple(getattr(self, "_active_requests", ()) or ())
        active_padded = int(getattr(self, "_active_padded_tokens", 0) or 0)
        active_layout = str(getattr(self, "_active_layout", "") or "")
        admission_view = getattr(self, "admission_memory_view", None)
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
        prep_key = (
            id(self.recipes), frozen_geom, stream_owners, startup_mode, op.stage,
            op.family, op.recipe, op.phase, op.dtype, op.layout,
            op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
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
            return replace(
                cached_op, op_id=op.op_id, layer=op.layer, deps=op.deps,
                requests=op.requests, stream=op.stream, notes=op.notes,
                metadata=meta,
            ), recipe
        res_op, recipe = self.recipes.prepare(op, geom)
        from ..collective_waves import startup_waves_recipe
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
            admission_memory_view=getattr(self, "admission_memory_view", None),
            allocation_versions=getattr(self, "allocation_versions", {}),
        )

    def _context(self, op: Any, replica: int, scope: str, *, recipe=None) -> RecipeContext:
        from ..collective_waves import startup_waves_recipe
        from mindie_sim.prediction.operators.collectives.direct_peer import resolve_transport_options
        from mindie_sim.prediction.operators.matrix.matrix_precision import execution_compute_peaks, resolve_matrix_precision

        recipe = startup_waves_recipe(
            op, recipe, self.case.runtime.collective_startup_mode)
        dies = self.stage_dies[replica, op.stage]
        op = resolve_matrix_precision(
            op, devices=dies,
            policy=self.case.runtime.matrix_compute_policy, recipe=recipe)
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
        context = replace(
            template, op=op, native_work=cost.work, compute_peaks=peaks,
            service_identity=sid,
            service_boundary="resource_phase" if declared else "device_kernel",
            stream_owners=stream_owners,
            requests=tuple(getattr(self, "_active_requests", ()) or ()),
            padded_tokens=int(getattr(self, "_active_padded_tokens", 0) or 0),
            layout=str(getattr(self, "_active_layout", "") or ""))
        bind_context = getattr(recipe, "bind_context", None)
        if bind_context is not None:
            context = bind_context(context)
        demands, frames, setup, plan_binding = self._bind_context_work(
            context, op, replica, recipe, cost.work, transport_options)
        diagnostic = (self.cost.lookup_diagnostic(op, scope)
                      if self.legacy_diagnostics else None)
        diagnostic_timing = (diagnostic or {}).get("timing")
        measured = (diagnostic_timing.duration_ms
                    if diagnostic_timing is not None else 0)
        physical = max((value / self.engine.resources[key].rate
                        for key, value in demands.items()), default=0)
        context = replace(
            context, work=demands,
            minimum_ms=measured if self.legacy_diagnostics else 0.0,
            setup_ms=setup,
            service_identity={**dict(context.service_identity),
                              "transport_options": transport_options},
            transaction_frames=frames,
            collective_plan=plan_binding)
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
            self._last_invocation_record = record
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
        from ..services import service_identity

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

    def _bind_context_work(self, context, op, replica, recipe, work, transport_options):
        algorithm = getattr(recipe, "algorithm", "ring")
        if (algorithm == "ring" and getattr(recipe, "mesh_capable", False)
                and self.topology.declares_direct_mesh(op, replica)):
            algorithm = "direct_mesh"
        band_plan = None
        link_vol = work.get("link", 0)
        if algorithm == "direct_mesh":
            plan = self.topology.joint_plan(
                op, {replica: link_vol},
                algorithm=algorithm, transport=transport_options)
            self.topology.bind_direct_mesh(op, replica, plan)
            plan_binding = _CollectivePlanBinding(algorithm, plan=plan)
        else:
            if link_vol and algorithm in {"ring", "pairwise"}:
                try:
                    band_plan = self.topology.band_collective_plan(
                        op, {replica: link_vol}, algorithm=algorithm,
                        transport=transport_options,
                    )
                except ModelingError as exc:
                    if not str(exc).startswith("joint_collective_group_membership"):
                        raise
            if band_plan is not None:
                plan_binding = _CollectivePlanBinding(algorithm, plan=band_plan)
            else:
                plan_binding = _CollectivePlanBinding(
                    algorithm,
                    factory=lambda: self.topology.joint_plan(
                        op, {replica: link_vol},
                        algorithm=algorithm, transport=transport_options))
        demands, frames, setup = {}, {}, 0.0
        if "peer_traffic" in dict(op.metadata) and not link_vol:
            self.topology.accounting(op, replica, volume=link_vol)
        for resource, amount in work.items():
            if resource == "link":
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
        return demands, frames, setup, plan_binding

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
        prog_key = (
            owner, do_compact, _freeze_key(recipe), op.family, op.recipe, op.phase, op.stage,
            op.dtype, op.layout, op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
            self._op_key(op), context.setup_ms, context.minimum_ms,
            (inv_t.startup_ms, inv_t.minimum_ms) if inv_t is not None else None,
            _freeze_key(context.joint_native_service),
            tuple(sorted(context.work.items())),
            live_rates,
            context.group_buffers,
            getattr(context.physical_mechanisms, "signature", None),
            op.stream or "main",
            context.stream_owners,
            band_key,
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

    def _lower_local_tp_work(self, work: IterationWork, replica: int) -> IterationWork:
        """Apply the declared node-local dense placement once per graph."""
        parallel = self.case.runtime.parallel
        if max(int(parallel.o_proj_tp), int(parallel.linear_proj_tp),
               int(parallel.lm_head_tp)) <= 1:
            return work
        from ..local_projection_tp import lower_local_projection_tp

        cohort_views = self._cohort_views
        if cohort_views is None:
            local_width = max(int(parallel.tp), int(parallel.o_proj_tp),
                              int(parallel.linear_proj_tp),
                              int(parallel.lm_head_tp))
            if (int(parallel.tp) == 1 and local_width > 1
                    and (work.plan.requests or work.mode != "graph")):
                from ..local_tp_groups import local_tp_replica_cohorts
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
        return work if ops is work.ops else replace(work, ops=ops)
