"""Canonical per-evaluation context for one prediction run.

[Role]
  Freeze the declared hardware representation once per evaluation, build the
  immutable execution bindings (topology, recipes, structural resource
  signature), and own the bounded evaluation-local caches used by physical
  grouping and isolated programs.  This module contains infrastructure only: it
  never prices an operator, lowers a program or executes device work.

[Invariants]
  * ``Device.to_dict()`` runs at most once per declared device during setup.
  * One evaluation constructs exactly one ``CostModel`` and one
    ``ExecutionBindings``; every isolated execution receives a fresh
    ``CostLedger`` from the caller so row ``basis`` records stay isolated.
  * Caches are evaluation-local and bounded.  There is no process-global
    ``id(object)`` prediction cache and no cross-case result cache.
  * A changed hardware counterfactual (rates, pools, topology, instruction
    services, quantization, runtime policy or recipe selection) produces a new
    evaluation context; the structural signature is explicit and hashable.

  * With no caller table, the shipped calibrated ``PrimitiveTable`` of the
    declared SKU is bound to the declared dies: hardware what-ifs rescale its
    rate bands by declared/registered supply ratios instead of dropping it.

[Timeouts & Locks]
  Pure construction from already-parsed declarations.  The only read is the
  shipped read-only primitive table, loaded once per SKU (``lru_cache``).  No
  locks and no device context initialization.

[Side-Effect Policy]
  Reads the parsed Case; never mutates it, the catalog, the filesystem or any
  global state.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from contextvars import ContextVar
from functools import lru_cache
import time
import hashlib
import json
from typing import Any, Mapping

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import (
    DatapathCapabilitySpec,
    datapath_for_die,
    ub_peer_edges_for_rank,
)


BOUNDED_CACHE_LIMIT = 4096
SIGNATURE_JSON_FLAGS = {"sort_keys": True, "separators": (",", ":"), "default": str}


def _canonical(value: Any) -> str:
    return json.dumps(value, **SIGNATURE_JSON_FLAGS)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class BoundedCache:
    """Small insertion-ordered cache with a hard size bound.

    Evaluation-local by construction: one instance is created by
    ``EvaluationContext`` and is never shared between cases.
    """

    __slots__ = ("_data", "limit", "hits", "misses", "evictions")

    def __init__(self, limit: int = BOUNDED_CACHE_LIMIT) -> None:
        self.limit = max(1, int(limit))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            value = self._data.pop(key)
        except KeyError:
            self.misses += 1
            return default
        self._data[key] = value
        self.hits += 1
        return value

    def put(self, key: Any, value: Any) -> Any:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.limit:
            self._data.popitem(last=False)
            self.evictions += 1
        return value

    __setitem__ = put

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._data), "limit": self.limit,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}


@dataclass(frozen=True)
class FrozenHardwareView:
    """Immutable hardware representation for one evaluation.

    ``_die_payloads`` holds one ``to_dict()`` result per declared device; the
    tuple is built during setup and is the only materialization of the mutable
    ``Device`` mapping in the evaluation.  ``dies`` exposes those same dict
    objects read-only to pricing/topology code that needs a mapping.
    """

    identity: str
    architecture: str
    hardware_key: str
    compute_peaks: Mapping[str, float]
    dies: tuple[Mapping[str, Any], ...]
    die_signatures: tuple[str, ...]
    signature: tuple

    @classmethod
    def from_machine(cls, machine: Any) -> "FrozenHardwareView":
        payloads: list[Mapping[str, Any]] = []
        signatures: list[str] = []
        for die in machine.dies:
            payload = die.to_dict()
            payloads.append(payload)
            signatures.append(_canonical(payload))
        identity = str(machine.hardware_key or machine.sku)
        peaks = dict(machine.compute_peaks)
        signature = (
            identity, str(machine.arch_family), str(machine.sku),
            tuple(signatures), _canonical(peaks),
        )
        return cls(identity=identity, architecture=str(machine.arch_family),
                   hardware_key=str(machine.hardware_key or machine.sku),
                   compute_peaks=peaks, dies=tuple(payloads),
                   die_signatures=tuple(signatures), signature=signature)

    def payload(self) -> dict[str, Any]:
        """The mapping expected by ``primitive_key`` and topology construction."""
        return {"identity": self.identity,
                "dies": list(self.dies),
                "compute_peaks": dict(self.compute_peaks)}

    def digest(self) -> str:
        return _digest(self.signature)


def _stage_geometry(case: CaseSpec,
                    hardware: FrozenHardwareView) -> tuple:
    """Per (replica, stage) declared die geometry, from the frozen payloads."""
    parallel = case.runtime.parallel
    width = parallel.tp * parallel.pcp
    geometry = []
    for replica in range(int(parallel.dp)):
        for stage in range(int(parallel.pp)):
            start = (replica * parallel.pp + stage) * width
            if start + width > len(hardware.die_signatures):
                raise ModelingError("placement_exceeds_physical_devices")
            indexes = tuple(range(start, start + width))
            geometry.append((
                replica, stage, indexes,
                tuple(hardware.die_signatures[index] for index in indexes),
            ))
    return tuple(geometry)


def _runtime_signature(case: CaseSpec) -> tuple:
    runtime = case.runtime
    quantization = getattr(runtime, "quantization", None)
    return (
        str(runtime.vllm_impl), str(runtime.vllm_version),
        str(runtime.kernel_format), str(runtime.graph_capture_mode),
        str(runtime.collective_startup_mode), bool(runtime.async_scheduling),
        _canonical(quantization if isinstance(quantization, Mapping) else {}),
        _canonical({str(key): value for key, value in dict(runtime.kernels).items()
                    if key != "recipes"}),
    )


#: Cooperative per-candidate/outer sweep deadline.  Every pricing loop polls
#: this value; no signal handler, thread or subprocess is involved.
ACTIVE_DEADLINE_MONO: ContextVar[float | None] = ContextVar(
    "mindie_sim_deadline_mono", default=None)


def check_active_deadline() -> None:
    deadline = ACTIVE_DEADLINE_MONO.get()
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("simulation_deadline_exceeded")


@dataclass(frozen=True)
class ExecutionBindings:
    """Immutable bindings shared by graph, operator, cohort and efficiency runs."""

    hardware: FrozenHardwareView
    resource_signature: tuple
    stage_geometry: tuple
    resources: tuple = ()
    pools: tuple = ()
    topology: Any = None
    recipes: Any = None
    stage_profiles: tuple = ()
    datapaths: tuple[tuple[int, DatapathCapabilitySpec], ...] = ()
    compute_arch: Any = None
    primitive_table: Any = None
    primitive_coverage_gaps: tuple[str, ...] = ()

    @classmethod
    def create(cls, case: CaseSpec, recipes: Any = None, *,
               primitive_table: Any = None) -> "ExecutionBindings":
        from .topology import Topology
        hardware = FrozenHardwareView.from_machine(case.machine)
        geometry = _stage_geometry(case, hardware)
        topology = Topology(
            case.machine, case.runtime.parallel, "",
            collective_startup_mode=case.runtime.collective_startup_mode,
            primitive_table=primitive_table)
        datapaths = tuple(
            (
                int(die.rank),
                datapath_for_die(
                    die,
                    ub_peer_edges=ub_peer_edges_for_rank(
                        die.rank, case.machine.links, case.machine.transfer_channels
                    ),
                ),
            )
            for die in case.machine.dies
        )
        from .execution import build_stage_profiles
        profiles = build_stage_profiles(case, "", primitive_table=primitive_table)
        prim_sig = tuple(getattr(p, "primitive_identity", ()) for p in profiles)
        coverage_gaps = tuple(sorted({
            gap for p in profiles for gap in getattr(p, "primitive_coverage_gaps", ())
        }))
        signature = (
            hardware.digest(), geometry, _runtime_signature(case),
            tuple(getattr(item, "name", str(item)) for item in topology.resources()),
            prim_sig,
        )
        compute_arch = (
            case.machine.dies[0].compute_arch
            if case.machine.dies and case.machine.homogeneous
            else None
        )
        return cls(hardware=hardware, resource_signature=signature,
                   stage_geometry=geometry, topology=topology, recipes=recipes,
                   stage_profiles=profiles, datapaths=datapaths,
                   compute_arch=compute_arch, primitive_table=primitive_table,
                   primitive_coverage_gaps=coverage_gaps)

    def datapath_for_rank(self, rank: int, /) -> DatapathCapabilitySpec:
        """Return the precomputed immutable datapath spec for one physical rank."""
        target = int(rank)
        for die_rank, spec in self.datapaths:
            if die_rank == target:
                return spec
        raise ModelingError(f"datapath_rank_unknown:{target}")

    def datapath_for_stage(self, replica: int, stage: int, /) -> DatapathCapabilitySpec:
        """Return the homogeneity-checked datapath spec for ``(replica, stage)``."""
        target_replica, target_stage = int(replica), int(stage)
        for profile in self.stage_profiles:
            if profile.replica == target_replica and profile.stage == target_stage:
                return profile.datapath
        raise ModelingError(f"datapath_stage_unknown:{target_replica}:{target_stage}")

    def signature_for(self, stage: int, replica: int) -> tuple:
        return (self.resource_signature, int(stage), int(replica))

    def profile_key(self, replica: int, stage: int) -> tuple:
        for profile in self.stage_profiles:
            if profile.replica == int(replica) and profile.stage == int(stage):
                return profile.profile_key
        return (self.resource_signature, int(replica), int(stage))

    def with_recipes(self, recipes: Any) -> "ExecutionBindings":
        if recipes is self.recipes:
            return self
        return ExecutionBindings(
            hardware=self.hardware, resource_signature=self.resource_signature,
            stage_geometry=self.stage_geometry, resources=self.resources,
            pools=self.pools, topology=self.topology, recipes=recipes,
            stage_profiles=self.stage_profiles, datapaths=self.datapaths,
            compute_arch=self.compute_arch,
            primitive_table=self.primitive_table,
            primitive_coverage_gaps=self.primitive_coverage_gaps)


PRIMITIVE_TABLE_CACHE_MAXSIZE = 32


@lru_cache(maxsize=PRIMITIVE_TABLE_CACHE_MAXSIZE)
def _bundled_calibrated_primitive_table(sku: str) -> Any:
    """Load and cache the shipped calibrated L0 primitive table for ``sku``."""
    if not sku:
        return None
    from mindie_sim.primitives.loader import load_primitives

    try:
        return load_primitives(sku, require_calibrated=True)
    except ModelingError:
        return None


#: Relative tolerance under which a declared die rate equals the registered
#: SKU rate the bundled primitive table was measured against.
PRIMITIVE_RESCALE_TOLERANCE = 1e-9
PRIMITIVE_RESCALED_WARNING = "primitive_table_rescaled_to_declared_die"
_MIB_BYTES = 1024 * 1024
#: ``(primitive, metric) -> die role`` for whole-die bus rates, and per-owner
#: rates whose table value is already normalized by the owning core count.
_DIE_BUS_METRICS = {
    ("P01_datacopy", "hbm_read_bytes_per_s"): "hbm_read",
    ("P01_datacopy", "hbm_write_bytes_per_s"): "hbm_write",
    ("P01_datacopy", "hbm_rw11_bytes_per_s"): "hbm",
    ("P01_datacopy", "hbm_rw21_bytes_per_s"): "hbm",
    ("P04_mte3_store", "atomic_rmw_bytes_per_s"): "hbm",
    ("P05_l2_residency_reuse", "l2_die_bytes_per_s"): "l2",
    ("P10_sort_topk", "sort_elems_per_s"): "vector_die",
}
_PER_OWNER_METRICS = {
    ("P01_datacopy", "mte2_aic_hit_bytes_per_s_per_core"): "mte2_aic",
    ("P01_datacopy", "mte2_aiv_hit_bytes_per_s_per_core"): "mte2_aiv",
    ("P02_load_data", "l1_to_l0a_bytes_per_s_per_aic"): "mte1",
    ("P02_load_data", "l1_to_l0a_transpose_bytes_per_s_per_aic"): "mte1",
    ("P02_load_data", "l1_to_l0b_bytes_per_s_per_aic"): "mte1",
    ("P02_load_data", "l1_to_l0b_transpose_bytes_per_s_per_aic"): "mte1",
    ("P03_fixpipe", "fixpipe_in_bytes_per_s_per_aic"): "fixpipe",
    ("P03_fixpipe", "fixpipe_out_bytes_per_s_per_aic"): "fixpipe",
    ("P04_mte3_store", "mte3_store_bytes_per_s_per_aiv"): "mte3",
    ("P07_vec_alu", "vec_alu_ops_per_s_per_aiv"): "vector",
    ("P08_vec_sfu", "vec_sfu_ops_per_s_per_aiv"): "vector",
    ("P09_vec_reduce_cast", "vec_reduce_ops_per_s_per_aiv"): "vector",
    ("P09_vec_reduce_cast", "cast_bytes_per_s_per_aiv"): "vector",
    ("P10_sort_topk", "sort_elems_per_s_per_aiv"): "vector",
    ("P11_simt", "simt_ops_per_s_per_aiv"): "simt",
    ("P11_simt", "simt_warp_instructions_per_s_per_aiv"): "simt",
}
_CUBE_PRIMITIVE = "P06_mmad"


def _hbm_direction_rate(explicit: Any, rates: Mapping[str, Any]) -> float:
    return float(explicit if explicit is not None else rates.get("hbm", 0.0) or 0.0)


def _per_owner(rate: float, units: int) -> float:
    return rate / units if units > 0 else 0.0


def _ratio(declared: float, registered: float) -> float:
    return declared / registered if declared > 0.0 and registered > 0.0 else 1.0


def _die_rescale_factors(die: Any, registered: Mapping[str, Any]) -> dict[str, float]:
    """Declared-die / registered-SKU supply ratios, one per table-bound role.

    Whole-die buses compare total rates; per-owner pipes compare the rate per
    owning core, because the overlay multiplies per-core table values by the
    declared core count itself.  Cube modes compare the per-AIC peak.
    """
    reg_rates = dict(registered.get("rates") or {})
    die_rates = dict(die.rates)
    reg_units = dict(registered.get("resource_units") or {})
    reg_cu = int(registered.get("compute_units", 1) or 1)
    reg_hbm = float(reg_rates.get("hbm", 0.0) or 0.0)
    die_hbm = float(die_rates.get("hbm", 0.0) or 0.0)
    hbm_factor = _ratio(die_hbm, reg_hbm)
    reg_hbm_read = _hbm_direction_rate(registered.get("hbm_read_bytes_per_s"), reg_rates)
    reg_hbm_write = _hbm_direction_rate(registered.get("hbm_write_bytes_per_s"), reg_rates)
    die_hbm_read = _hbm_direction_rate(die.hbm_read_bytes_per_s, die_rates)
    die_hbm_write = _hbm_direction_rate(die.hbm_write_bytes_per_s, die_rates)
    # Callers perturbing ``hbm_bw_bytes_per_s`` or ``rates["hbm"]`` leave
    # explicit SKU read/write fields untouched; follow ``hbm`` when they were
    # not independently perturbed, and never let read/write exceed ``die_hbm``.
    hbm_read_factor = (
        hbm_factor
        if abs(_ratio(die_hbm_read, reg_hbm_read) - 1.0) <= PRIMITIVE_RESCALE_TOLERANCE
        else _ratio(die_hbm_read, reg_hbm_read)
    )
    hbm_write_factor = (
        hbm_factor
        if abs(_ratio(die_hbm_write, reg_hbm_write) - 1.0) <= PRIMITIVE_RESCALE_TOLERANCE
        else _ratio(die_hbm_write, reg_hbm_write)
    )
    factors = {
        "hbm": hbm_factor,
        "hbm_read": hbm_read_factor,
        "hbm_write": hbm_write_factor,
        "l2": _ratio(float(die_rates.get("l2", 0.0) or 0.0), float(reg_rates.get("l2", 0.0) or 0.0)),
        "vector_die": _ratio(float(die_rates.get("vector", 0.0) or 0.0), float(reg_rates.get("vector", 0.0) or 0.0)),
    }
    for role in set(_PER_OWNER_METRICS.values()):
        factors[role] = _ratio(
            _per_owner(float(die_rates.get(role, 0.0) or 0.0), int(die.units(role))),
            _per_owner(float(reg_rates.get(role, 0.0) or 0.0), int(reg_units.get(role, reg_cu))))
    die_peaks = dict(die.compute_peaks)
    for dtype, reg_peak in dict(registered.get("compute_peaks") or {}).items():
        factors[f"cube:{dtype}"] = _ratio(
            _per_owner(float(die_peaks.get(dtype, 0.0) or 0.0), int(die.compute_units)),
            _per_owner(float(reg_peak), reg_cu))
    return factors


def _scaled_band(band: Any, factor: float) -> Any:
    from dataclasses import replace as _replace

    if factor == 1.0 or getattr(band, "bound_kind", "") != "rate_upper":
        return band
    scaled = {name: (getattr(band, name) * factor if getattr(band, name) is not None else None)
              for name in ("certified_upper_capacity", "p05", "p50", "p95", "p99", "trimmed_mean")}
    return _replace(band, **scaled)


def _scaled_metrics(values: Mapping[str, Any], factor_for: Any) -> dict[str, Any]:
    """Rebuild one metric mapping, rescaling only bands with a declared role factor."""
    from mindie_sim.primitives.schema import UncertaintyBand

    out: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, UncertaintyBand):
            out[name] = _scaled_band(value, factor_for(name))
        elif isinstance(value, Mapping):
            out[name] = _scaled_metrics(value, factor_for)
        else:
            out[name] = value
    return out


def _scale_bps_mapping(values: Any, factor: float) -> Any:
    if factor == 1.0 or not isinstance(values, Mapping):
        return values
    return {
        k: (float(v) * factor if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        for k, v in values.items()
    }


def _scale_bps_records(records: Any, factor: float, field: str = "bps_p50") -> Any:
    if factor == 1.0 or not isinstance(records, (list, tuple)):
        return records
    return tuple(
        {**item, field: float(item[field]) * factor}
        if isinstance(item, Mapping) and isinstance(item.get(field), (int, float))
        else item
        for item in records
    )


def _scale_nested_bps(mapping: Any, field_factors: Mapping[str, float]) -> Any:
    if not isinstance(mapping, Mapping) or all(f == 1.0 for f in field_factors.values()):
        return mapping
    out: dict[str, Any] = {}
    for key, entry in mapping.items():
        if not isinstance(entry, Mapping):
            out[key] = entry
            continue
        updated = dict(entry)
        for metric_name, factor in field_factors.items():
            if factor != 1.0 and isinstance(updated.get(metric_name), (int, float)):
                updated[metric_name] = float(updated[metric_name]) * factor
        out[key] = updated
    return out


def _parse_ws_size_bytes(key: Any) -> int | None:
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    if isinstance(key, str):
        text = key.strip()
        if text.endswith("MiB") and text[:-len("MiB")].isdigit():
            return int(text[:-len("MiB")]) * _MIB_BYTES
        if text.isdigit():
            return int(text)
    return None


def _scale_ws_sweep(
    sweep: Any, orig_l2_bytes: int, l2_factor: float, hbm_r_factor: float,
) -> Any:
    """Rescale P05 working-set sweep via harmonic L2/HBM blend so both L2 and HBM perturbations propagate."""
    if not isinstance(sweep, Mapping) or (l2_factor == 1.0 and hbm_r_factor == 1.0):
        return sweep
    if abs(l2_factor - hbm_r_factor) <= PRIMITIVE_RESCALE_TOLERANCE:
        return _scale_bps_mapping(sweep, l2_factor)
    parsed: list[tuple[Any, int | None, float]] = []
    for k, v in sweep.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0.0:
            parsed.append((k, _parse_ws_size_bytes(k), float(v)))
    fitting = [rate for _, sz, rate in parsed if sz is not None and orig_l2_bytes > 0 and sz <= orig_l2_bytes]
    spill = [rate for _, sz, rate in parsed if sz is not None and orig_l2_bytes > 0 and sz > orig_l2_bytes]
    warm = max(fitting) if fitting else 0.0
    cold = min(spill) if spill else 0.0
    has_blend = 0.0 < cold < warm
    out: dict[Any, Any] = dict(sweep)
    for k, sz, rate in parsed:
        if sz is not None and orig_l2_bytes > 0 and sz <= orig_l2_bytes:
            out[k] = rate * l2_factor
        elif has_blend:
            hit = min(1.0, max(0.0, (1.0 / cold - 1.0 / rate) / (1.0 / cold - 1.0 / warm)))
            inv_rate = hit / (warm * l2_factor) + (1.0 - hit) / (cold * hbm_r_factor)
            out[k] = (1.0 / inv_rate) if inv_rate > 0.0 else rate * hbm_r_factor
        else:
            out[k] = rate * hbm_r_factor
    return out


def _rescaled_parameters(
    pid: str, params: Mapping[str, Any], factors: Mapping[str, float],
    buffers: Mapping[str, int],
) -> dict[str, Any]:
    """Scale embedded bandwidth lookup surfaces inside P01/P05/P06/P10/P11 cell parameters."""
    out = dict(params)
    hbm = float(factors.get("hbm", 1.0))
    hbm_r = float(factors.get("hbm_read", hbm))
    hbm_w = float(factors.get("hbm_write", hbm))
    l2 = float(factors.get("l2", 1.0))
    mte2_aic = float(factors.get("mte2_aic", 1.0))
    if pid == "P01_datacopy":
        if isinstance(out.get("hbm_read_datacopy_blocked_bps_p50"), (int, float)) and hbm_r != 1.0:
            out["hbm_read_datacopy_blocked_bps_p50"] = float(out["hbm_read_datacopy_blocked_bps_p50"]) * hbm_r
        out["gm_alignment_cold_bps"] = _scale_bps_mapping(out.get("gm_alignment_cold_bps"), hbm_r)
        out["mte2_aic_cold_scaling_bps"] = _scale_bps_mapping(out.get("mte2_aic_cold_scaling_bps"), hbm_r)
        out["mte2_aic_warm_scaling_bps"] = _scale_bps_mapping(out.get("mte2_aic_warm_scaling_bps"), mte2_aic)
        out["strided_cold_surface"] = _scale_bps_records(out.get("strided_cold_surface"), hbm_r)
        out["gather_v2_cold_surface"] = _scale_nested_bps(
            out.get("gather_v2_cold_surface"),
            {"random_scatter_rw_bps": hbm, "sequential_rw_bps": hbm})
        out["payload_ramp_bytes_per_s"] = _scale_nested_bps(
            out.get("payload_ramp_bytes_per_s"),
            {"read_bps": hbm_r, "write_bps": hbm_w, "rw11_bps": hbm})
    elif pid == "P05_l2_residency_reuse":
        orig_l2 = int(out.get("l2_bytes") or buffers.get("l2") or 0)
        if "ws_sweep_bytes_per_s" in out:
            out["ws_sweep_bytes_per_s"] = _scale_ws_sweep(out.get("ws_sweep_bytes_per_s"), orig_l2, l2, hbm_r)
        if "l2_service_curve" in out:
            out["l2_service_curve"] = _scale_ws_sweep(out.get("l2_service_curve"), orig_l2, l2, hbm_r)
        if "l2" in buffers:
            out["l2_bytes"] = int(buffers["l2"])
        out["l2_stride_conflict"] = _scale_bps_records(out.get("l2_stride_conflict"), l2)
    elif pid == "P06_mmad":
        cube_bf16 = float(factors.get("cube:bf16", 1.0))
        if isinstance(out.get("gemm_e2e_flops_per_s_per_aic"), (int, float)) and cube_bf16 != 1.0:
            out["gemm_e2e_flops_per_s_per_aic"] = float(out["gemm_e2e_flops_per_s_per_aic"]) * cube_bf16
    elif pid == "P10_sort_topk":
        vec_die = float(factors.get("vector_die", 1.0))
        if isinstance(out.get("sort_die_elems_per_s_p50"), (int, float)) and vec_die != 1.0:
            out["sort_die_elems_per_s_p50"] = float(out["sort_die_elems_per_s_p50"]) * vec_die
    elif pid == "P11_simt":
        simt = float(factors.get("simt", 1.0))
        if isinstance(out.get("simt_scalar_flops_per_s_per_aiv_p50"), (int, float)) and simt != 1.0:
            out["simt_scalar_flops_per_s_per_aiv_p50"] = float(out["simt_scalar_flops_per_s_per_aiv_p50"]) * simt
    return out


def _rescaled_cell(
    cell: Any, factors: Mapping[str, float], buffers: Mapping[str, int],
) -> Any:
    from dataclasses import replace as _replace

    pid = cell.primitive_id

    def metric_factor(metric: str) -> float:
        role = _DIE_BUS_METRICS.get((pid, metric)) or _PER_OWNER_METRICS.get((pid, metric))
        return float(factors.get(role, 1.0)) if role else 1.0

    modes = {}
    for mode, entry in cell.modes.items():
        if pid == _CUBE_PRIMITIVE:
            cube = float(factors.get(f"cube:{mode}", 1.0))
            modes[mode] = _scaled_metrics(entry, lambda metric, cube=cube: cube if metric == "cube_only_rate" else 1.0)
        else:
            modes[mode] = _scaled_metrics(entry, metric_factor)
    return _replace(
        cell,
        metrics=_scaled_metrics(cell.metrics, metric_factor),
        modes=modes,
        parameters=_rescaled_parameters(pid, cell.parameters, factors, buffers),
    )


def _rebind_primitive_table(table: Any, die: Any, registered: Mapping[str, Any]) -> Any:
    """Bind a measured SKU table to a declared (possibly perturbed) die.

    A hardware what-if (e.g. 4.0 -> 1.4 TB/s HBM, fewer cores, another clock)
    keeps every measured per-mechanism efficiency: bus rates scale by the
    declared/registered supply ratio, per-core rates by the per-core ratio, and
    the silicon fingerprint adopts the declared core counts and buffers.  Launch
    and latency bands are unchanged.  A different architecture is unbound.
    """
    from dataclasses import replace as _replace

    from mindie_sim.primitives.loader import _check_datapath

    datapath = datapath_for_die(die)
    factors = _die_rescale_factors(die, registered)
    silicon = table.provenance.silicon_fingerprint
    buffers = dict(datapath.buffers)
    fingerprint = _replace(
        silicon, aic_count=int(datapath.aic_count), aiv_count=int(datapath.aiv_count),
        **{f"{pool}_bytes": int(buffers[pool]) for pool in ("l1", "l0a", "l0b", "l0c", "ub", "l2")
           if pool in buffers})
    unchanged = fingerprint == silicon and all(
        abs(value - 1.0) <= PRIMITIVE_RESCALE_TOLERANCE for value in factors.values())
    candidate = table
    if not unchanged:
        candidate = _replace(
            table,
            provenance=_replace(table.provenance, silicon_fingerprint=fingerprint),
            primitives={key: _rescaled_cell(cell, factors, buffers) for key, cell in table.primitives.items()},
            warnings=tuple(dict.fromkeys((*table.warnings, PRIMITIVE_RESCALED_WARNING))))
    try:
        _check_datapath(candidate, datapath)
    except ModelingError:
        return None
    return candidate


def _resolve_default_primitive_table(case: CaseSpec) -> Any:
    """Auto-bind the calibrated bundled SKU primitive table to the declared dies."""
    if not case.machine.dies or not case.machine.homogeneous:
        return None
    sku = str(case.machine.sku or "")
    table = _bundled_calibrated_primitive_table(sku)
    if table is None and sku:
        from mindie_sim.backends.ascend import DERIVED_ASCEND_SKUS

        derived = DERIVED_ASCEND_SKUS.get(sku)
        base_sku = str(derived.get("base", "")) if isinstance(derived, Mapping) else ""
        if base_sku:
            table = _bundled_calibrated_primitive_table(base_sku)
    if table is None:
        return None
    from mindie_sim.backends.ascend import profiles as ascend_profiles

    # Ratios are taken against the silicon the table was measured on, which can
    # be a manifest-compatible sibling of the declared SKU.
    registry = ascend_profiles()
    entry = registry.get(str(table.sku)) or registry.get(sku)
    registered = entry.get("device") if isinstance(entry, Mapping) else None
    if not isinstance(registered, Mapping):
        return None
    return _rebind_primitive_table(table, case.machine.dies[0], registered)


@dataclass
class EvaluationContext:
    """One evaluation's frozen hardware, bindings, cost model and caches."""

    case: CaseSpec
    bindings: ExecutionBindings
    cost: Any
    physical_keys: BoundedCache = field(default_factory=BoundedCache)
    prepared_programs: BoundedCache = field(default_factory=BoundedCache)
    isolated_results: BoundedCache = field(default_factory=BoundedCache)
    #: ``case`` when the recipe book is the one the case declares, ``caller``
    #: when a caller supplied an override.  Only a case-declared book may be
    #: reused across evaluations of the same declaration.
    recipe_origin: str = "caller"
    construction_counts: dict[str, int] = field(
        default_factory=lambda: {"cost_models": 1, "bindings": 1,
                                 "contexts": 1})

    @classmethod
    def create(cls, case: CaseSpec, recipes: Any = None,
               primitives: Any = None, *, primitive_table: Any = None) -> "EvaluationContext":
        from .efficiency.cost import CostModel
        from .program import RecipeBook
        book = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        if primitive_table is False:
            eff_table = None
        elif primitive_table is not None:
            eff_table = primitive_table
        else:
            eff_table = _resolve_default_primitive_table(case)
        bindings = ExecutionBindings.create(case, book, primitive_table=eff_table)
        cost = CostModel(
            case, primitives, hardware=bindings.hardware,
            primitive_table=bindings.primitive_table,
        )
        return cls(case=case, bindings=bindings, cost=cost,
                   recipe_origin="case" if recipes is None else "caller")

    def cost_summary(self) -> dict[str, Any]:
        return self.cost.summary()

    def hardware_payload(self) -> dict[str, Any]:
        return self.bindings.hardware.payload()


def context_for(case: CaseSpec, context: EvaluationContext | None,
                recipes: Any = None) -> EvaluationContext:
    """Reuse an explicit evaluation context or create one for a standalone call.

    The recipe book of an explicit context describes the case it was created
    from.  A caller that has consumed target declarations passes its own
    ``recipes`` to execution; it does not silently rebuild the frozen bindings.
    """
    if context is None:
        return EvaluationContext.create(case, recipes)
    return context
