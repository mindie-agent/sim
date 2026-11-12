"""Operator-piecewise compiler over fixed static ``Theta`` and dynamic ``X,Z``.

[Role]
  Compile a bound canonical operator into guarded, multi-stage analytical
  resource functions.  Static weight dimensions, head/expert geometry,
  quantisation roles, parallel axes and tiles are frozen; sequence/batch
  geometry and explicit state remain dynamic.  The same expressions produce
  exact numeric costs for a bound request vector and symbolic segment
  formulas for a bounded analysis domain.

[Invariants]
  * Work and bytes come from declared structural formulas, never from a model
    name, request id, measurement presence or a fitted latency.
  * A stage serves its independent pipes concurrently (max); stages inside one
    operator are serial (sum).  A tail wave cannot borrow idle core supply.
  * Discrete regimes select exactly one guard cell; a domain miss or a
    non-polynomial crossover raises instead of silently widening.
  * Logical work (FLOPs, valid pairs) is reported separately from padded
    physical tile work and from weight residency.
  * Segment enumeration is bounded and reports truncation; no whole-model
    Cartesian table is materialised.

[Timeouts & Locks]
  Pure bounded arithmetic and finite segment enumeration.  No waits or locks.

[Side-Effect Policy]
  Returns immutable cost records; no I/O, device work or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from typing import Any, Callable, Iterable, Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.subgraphs import BoundOperator, StaticSignature, SubgraphInstance
from mindie_sim.prediction.subgraphs.state import StateView
from .geometry import AnalysisDomain, RequestGeometry, WorkloadGeometry
from .expressions import (
    Add, CeilDiv, Const, Div, Expr, FloorDiv, Interval, Max, Min, Mul, Sub,
    Sum, Var, as_fraction, evaluate, simplify,
)
from .guards import (
    CoreWaveGuard, Crossover, GraphBucketGuard, GuardCell, PipeDemand,
    PipeDominanceGuard, SplitKGuard, SplitKPolicy, StepGuard,
)
from .physical import (
    A3_ANALYTICAL_CORES, ANALYTICAL_MAX_SPLITS, ANALYTICAL_SPLIT_TARGET,
    BF16_ELEMENT_BYTES, SOFTMAX_OPS_PER_PAIR, SOFTWARE_KV_TILE,
    SOFTWARE_PIPELINE_SLOTS, SOFTWARE_TILE_K, SOFTWARE_TILE_M, SOFTWARE_TILE_N,
    fill_undeclared_byte_pipes,
)

_PIPE_UNITS = {
    "cube": "flops", "vector": "ops", "hbm": "bytes", "mte1": "bytes",
    "mte2": "bytes", "mte3": "bytes", "fixpipe": "bytes", "link": "bytes",
    "host": "ms",
}

# Derived family coordinates whose declared analysis-domain bound must be
# enforced even when they are not themselves a segment axis coordinate.
_DERIVED_DOMAIN_KEYS = (
    ("moe_rows", "max_moe_rows"),
    ("index_rows", "max_index_rows"),
    ("query_tiles", "max_query_tiles"),
    ("kv_tiles", "max_kv_tiles"),
    ("gemm_work", "max_work"),
)

def _frac(value: Any) -> Fraction:
    return as_fraction(value)

def _ceil(value: Fraction) -> int:
    return int(math.ceil(value))

@dataclass(frozen=True)
class PipeSupply:
    """Native supply per physical pipe, in work units per millisecond."""

    rates: tuple[tuple[str, float], ...]
    structure_caps: tuple[tuple[str, float], ...] = ()
    invocations: tuple[tuple[str, float], ...] = ()
    resource_units: tuple[tuple[str, int], ...] = ()
    flops_by_dtype: tuple[tuple[str, float], ...] = ()
    undeclared_pipes: tuple[str, ...] = ()
    store_queues: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    gdn_beta_store: tuple[tuple[str, Any], ...] = ()
    gdn_decode_selected: bool = False
    gdn_recurrent_implementation: str = ""
    buffers: tuple[tuple[str, int], ...] = ()
    pool_units: tuple[tuple[str, int], ...] = ()
    service_modes: tuple[tuple[str, tuple[str, float]], ...] = ()
    compute_units: int = 1
    compute_peaks: tuple[tuple[str, float], ...] = ()
    arch_family: str = ""

    def __post_init__(self) -> None:
        rates = tuple(sorted((str(pipe), float(rate)) for pipe, rate in self.rates))
        if not rates:
            raise ModelingError("pipe_supply_empty")
        for pipe, rate in rates:
            if not pipe or not math.isfinite(rate) or rate <= 0:
                raise ModelingError(f"pipe_supply_invalid:{pipe}")
        object.__setattr__(self, "rates", rates)
        queues = tuple(sorted(
            (str(name), tuple(sorted((str(k), v) for k, v in dict(queue).items())))
            for name, queue in (self.store_queues or ())))
        object.__setattr__(self, "store_queues", queues)
        caps = tuple(sorted((str(pipe), float(cap)) for pipe, cap in self.structure_caps))
        for pipe, cap in caps:
            if not math.isfinite(cap) or cap <= 0:
                raise ModelingError(f"pipe_structure_cap_invalid:{pipe}")
        object.__setattr__(self, "structure_caps", caps)

    def rate(self, pipe: str) -> float:
        for name, value in self.rates:
            if name == pipe:
                return value
        raise ModelingError(f"pipe_supply_missing:{pipe}")

    def cap(self, pipe: str) -> float | None: return dict(self.structure_caps).get(pipe)

    def effective(self, pipe: str) -> float:
        rate, cap = self.rate(pipe), self.cap(pipe)
        return min(rate, cap) if cap is not None else rate

    def to_dict(self) -> dict[str, float]: return dict(self.rates)

    @classmethod
    def from_hardware(cls, hardware: Any, *, dtype: str = "bf16",
                      time_scale: float = 1e-3, link_rate: float | None = None) -> "PipeSupply":
        if not hardware.dies:
            raise ModelingError("hardware_requires_device")
        import json
        die = hardware.dies[0]
        rates = {name: float(value) * float(time_scale) for name, value in die.rates}
        rates["cube"] = float(die.peak(dtype)) * float(time_scale)
        undeclared = fill_undeclared_byte_pipes(rates)
        if "memory" in rates and "hbm" not in dict(die.rates):
            rates["hbm"] = rates.pop("memory")
        if link_rate is not None:
            rates["link"] = float(link_rate)
        else:
            rates.update(_device_link_rates(hardware))
        inv, units, peaks = _extract_hardware_meta(hardware, float(time_scale))
        queues = {}
        for name, queue in dies_store_queues(hardware):
            if name in rates and all(
                    dict(dict(d.store_queues).get(name, ())) == queue
                    for d in hardware.dies):
                queues[name] = dict(queue)
        assumptions = dict(getattr(hardware, "assumptions", ()))
        selected = assumptions.get("gdn_decode_selected") == "1"
        pool_units = tuple(sorted((str(name), int(die.pool_units(name)))
                                  for name in dict(die.buffers)))
        buffers = tuple(sorted((str(name), int(value))
                               for name, value in dict(die.buffers).items()))
        recurrence = str(assumptions.get("gdn_recurrent_implementation", ""))
        modes = tuple(sorted((str(name), (str(role), float(rate)))
                             for name, (role, rate) in die.service_modes))
        compute_units = int(die.compute_units)
        peaks = tuple(sorted((str(name), float(value) * float(time_scale))
                             for name, value in dict(die.compute_peaks).items()))
        store = ()
        declared_store = dict(getattr(hardware, "assumptions", ())).get("gdn_gating_beta_store")
        if declared_store:
            import json
            try:
                parsed = json.loads(declared_store)
            except (TypeError, ValueError) as exc:
                raise ModelingError("gdn_beta_store_declaration_invalid") from exc
            store = tuple(sorted(dict(parsed).items())) if isinstance(parsed, Mapping) else ()
        return cls(tuple(rates.items()), (), inv, units, peaks, undeclared,
                   tuple(sorted(queues.items())), store, selected, recurrence,
                   buffers, pool_units, modes, compute_units, peaks,
                   str(getattr(hardware, "arch_family", "") or ""))

def dies_store_queues(hardware: Any):
    """Uniform declared die queues; heterogeneous owners are not averaged."""
    if not hardware.dies:
        return ()
    first = dict(hardware.dies[0].store_queues)
    for name in first:
        queue = first[name]
        if all(dict(d.store_queues).get(name) == queue for d in hardware.dies[1:]):
            yield name, dict(queue)


def _device_link_rates(hardware: Any) -> dict[str, float]:
    """Declared per-route link capacities, not one global minimum.

    ``link_hccs`` / ``link`` are the on-node route and ``link_roce`` is the
    cross-node route.  A route class uses its own weakest link.  An idle
    cross-node link must not replace the on-node rate.
    """
    intra = inter = 0.0
    for link in (getattr(hardware, "links", ()) or ()):
        rate = float(getattr(link, "capacity_bytes_per_ms", 0.0) or 0.0)
        if rate <= 0:
            continue
        text = f"{getattr(link, 'channel', '')} {getattr(link, 'name', '')}".lower()
        channel = str(getattr(link, "channel", "")).lower()
        # CPU/host I/O channels are distinct from device d2d links.  A link
        # named ub-host0 on channel d2d_intra_node is a real intra-node route;
        # a bare host substring must not discard it.
        if ("h2d" in text or "d2h" in text or "pcie" in text
                or "cpu" in text or "host_to_device" in text
                or "device_to_host" in text):
            continue
        if ("host" in str(getattr(link, "name", "")).lower()
                and not any(token in channel for token in
                            ("d2d", "intra", "ub", "package"))):
            continue
        # Cross-node fabrics only.  An on-node mesh keeps its own capacity.
        if "inter" in text or "roce" in text or "scale-out" in text or "scale_out" in text:
            inter = rate if inter <= 0 else min(inter, rate)
        else:
            intra = rate if intra <= 0 else min(intra, rate)
    rates: dict[str, float] = {}
    if intra > 0:
        rates["link_hccs"] = intra
        # Callers that still emit a single ``link`` pipe are on-node unless
        # the only declared fabric is the inter-node route.
        rates["link"] = intra
    if inter > 0:
        rates["link_roce"] = inter
        if "link" not in rates:
            rates["link"] = inter
    return rates

@dataclass(frozen=True)
class QuantLedger:
    """Separate packed payload, scale, zero-point and residency accounting."""

    role: str
    weight_format: str
    activation_format: str
    payload_bytes: Fraction
    scale_bytes: Fraction
    zero_point_bytes: Fraction
    residency_bytes: Fraction
    scale_group: int = 0

    @property
    def weight_bytes(self) -> Fraction:
        return self.payload_bytes + self.scale_bytes + self.zero_point_bytes

    @property
    def metadata_bytes(self) -> Fraction:
        return self.scale_bytes + self.zero_point_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "weight_format": self.weight_format,
            "activation_format": self.activation_format,
            "payload_bytes": str(self.payload_bytes),
            "scale_bytes": str(self.scale_bytes),
            "zero_point_bytes": str(self.zero_point_bytes),
            "weight_bytes": str(self.weight_bytes),
            "residency_bytes": str(self.residency_bytes),
            "scale_group": self.scale_group,
        }

def quant_ledger(quant: Any, category: str, rows: int, cols: int) -> QuantLedger:
    """Resolve one quantisation role into independent byte accounts."""
    rows, cols = int(rows), int(cols)
    if rows < 0 or cols < 0:
        raise ModelingError(f"quant_ledger_shape_invalid:{rows}x{cols}")
    if quant is None:
        return QuantLedger(category, "bf16", "bf16",
                           Fraction(rows * cols * BF16_ELEMENT_BYTES),
                           Fraction(0), Fraction(0),
                           Fraction(rows * cols * BF16_ELEMENT_BYTES))
    resolved = quant.for_category(category)
    payload = Fraction(str(resolved.matrix_payload_bytes(rows, cols, category)))
    total = Fraction(str(resolved.matrix_bytes(rows, cols, category)))
    fmt = resolved.format_for(category)
    zero = Fraction(0)
    if fmt in {"int8", "w8a8"} and resolved.int8_weight_metadata == "scale_offset":
        zero = Fraction(rows) * Fraction(str(resolved.scale_bytes))
    scale = max(Fraction(0), total - payload - zero)
    group = 0
    if fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        group = int(resolved.weight_block[1] if resolved.weight_block else 0)
        if not group:
            group = int(resolved.expert_block or 0)
    return QuantLedger(category, fmt, resolved.activation_format or "bf16",
                       payload, scale, zero, total, group)

@dataclass(frozen=True)
class ImplementationPolicy:
    """Declared tile/split policy; shape-dependent choices stay explicit."""

    name: str = "analytical"
    mt: int = SOFTWARE_TILE_M
    nt: int = SOFTWARE_TILE_N
    kt: int = SOFTWARE_TILE_K
    kv_tile: int = SOFTWARE_KV_TILE
    cores: int = A3_ANALYTICAL_CORES
    moe_row_tile: int = SOFTWARE_TILE_M
    split_target: int = ANALYTICAL_SPLIT_TARGET
    max_splits: int = ANALYTICAL_MAX_SPLITS
    softmax_ops_per_pair: int = SOFTMAX_OPS_PER_PAIR
    pipeline_slots: int = SOFTWARE_PIPELINE_SLOTS

    def __post_init__(self) -> None:
        for name in ("mt", "nt", "kt", "kv_tile", "cores", "moe_row_tile",
                     "split_target", "max_splits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"policy_geometry_invalid:{name}")
        if self.pipeline_slots < 1:
            raise ModelingError("policy_pipeline_slots_invalid")

    def split_policy(self, *, target: int | None = None, max_splits: int | None = None) -> SplitKPolicy:
        return SplitKPolicy(target or self.split_target, max_splits or self.max_splits)


@dataclass(frozen=True)
class SegmentAxis:
    """One discrete guard axis of a compiled family."""

    name: str
    coordinate: str
    guard: Any
    domain_key: str
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.coordinate or not self.domain_key:
            raise ModelingError("segment_axis_invalid")

    def index(self, value: Any) -> int:
        if isinstance(self.guard, GraphBucketGuard):
            return self.guard.select_index(value)
        return int(self.guard.select(value))

    def cell(self, index: int) -> GuardCell:
        return self.guard.cell(index)

    def index_range(self, domain: AnalysisDomain) -> range:
        """Lazy cell-index range; enumeration must not materialise it.

        ``max_query_tiles``/``max_kv_tiles`` are maximum tile *indices*, the
        same unit as the runtime ``QUERY_TILES``/``KV_TILES`` regime.  Other
        keys bound the raw coordinate and are converted through the guard.
        """
        upper = domain.bound(self.domain_key)
        if isinstance(self.guard, GraphBucketGuard):
            return range(len(self.guard.buckets))
        if isinstance(self.guard, SplitKGuard):
            return range(1, self.guard.policy.max_splits + 1)
        first = 0 if isinstance(self.guard, CoreWaveGuard) or getattr(
            self.guard, "lower_bound", 1) <= 0 else 1
        if self.domain_key in {"max_query_tiles", "max_kv_tiles"}:
            return range(first, upper + 1)
        return range(first, self.guard.select(upper) + 1)

    def indices(self, domain: AnalysisDomain) -> tuple[int, ...]:
        return tuple(self.index_range(domain))

    def position(self, index: int, domain: AnalysisDomain) -> int:
        """Zero-based normalised position shared by runtime and export."""
        return index - self.index_range(domain).start

@dataclass(frozen=True)
class FamilyTemplate:
    axes: tuple[SegmentAxis, ...]
    stages: tuple[tuple[PipeDemand, ...], ...]
    materialization: tuple[tuple[str, Expr], ...]
    internal: tuple[tuple[str, Expr], ...]
    residency: Expr
    derived: tuple[tuple[str, Expr], ...]
    regime_exprs: tuple[tuple[str, Expr], ...]
    binder: Callable[[WorkloadGeometry, StateView], dict[str, Any]]
    state_effects: tuple[tuple[str, str], ...] = ()
    annotations: tuple[tuple[str, str], ...] = ()
    active: Expr | None = None
    program_duration: Callable[[Any, Any], Fraction] | None = None
    program_stages: Callable[[Any, Any], tuple[tuple[str, Fraction], ...]] | None = None
    minimum_duration: Callable[[Any, Any], Fraction] | None = None

@dataclass(frozen=True)
class NativeProgram:
    """The original executable native program plus its exact source ledger."""
    program: Any
    work: tuple[tuple[str, Fraction], ...]
    duration: Fraction
    fidelity: str = "native_program"

    def start(self, engine: Any, name: str, ready: Any, *,
              peer_ready: Any = None, peer_send: Any = None,
              launch: Any = None, source_send: Any = None,
              send_placeholders: Any = None, transport: Any = None) -> Any:
        if (peer_ready is None and peer_send is None and launch is None
                and source_send is None and send_placeholders is None
                and transport is None):
            return self.program.start(engine, name, ready)
        arguments = dict(
            peer_ready=peer_ready, peer_send=peer_send, launch=launch,
            source_send=source_send, send_placeholders=send_placeholders)
        if transport is not None:
            arguments["transport"] = transport
        return self.program.start(engine, name, ready, **arguments)

    def start_with_handles(self, engine: Any, name: str, ready: Any, *,
                           peer_ready: Any = None, peer_send: Any = None,
                           launch: Any = None, source_send: Any = None,
                           send_placeholders: Any = None,
                           transport: Any = None):
        arguments = dict(
            peer_ready=peer_ready, peer_send=peer_send, launch=launch,
            source_send=source_send, send_placeholders=send_placeholders)
        if transport is not None:
            arguments["transport"] = transport
        return self.program.start_with_handles(
            engine, name, ready, **arguments)


@dataclass(frozen=True)
class NativeTask:
    """One native execution task: resource, work and its actual executed cap."""
    resource: str
    work: Fraction
    cap: Fraction | None = None


@dataclass(frozen=True)
class NativeSchedule:
    """Public native owner-program schedule retained for shared-engine use.

    ``stages`` are concurrent task groups in native order; ``startup`` is the
    non-overlapped wait before the first body task; caps are the native
    executed rates (work / busy).  Source work stays in task units, never
    re-derived from the analytical stage aggregate.
    """
    stages: tuple[tuple[NativeTask, ...], ...]
    startup: Fraction = Fraction(0)
    dependency: Fraction = Fraction(0)
    floor: Fraction = Fraction(0)
    duration: Fraction = Fraction(0)
    fidelity: str = "partial_aggregate"


@dataclass(frozen=True)
class OperatorCost:
    name: str
    family: str
    kind: str
    segment_index: int
    regime: tuple[tuple[str, int], ...]
    guard_cells: tuple[GuardCell, ...]
    work: tuple[tuple[str, Fraction], ...]
    times: tuple[tuple[str, Fraction], ...]
    stage_times: tuple[Fraction, ...]
    duration_ms: Fraction
    dominant_pipe: str
    tied_pipes: tuple[str, ...]
    materialization: tuple[tuple[str, Fraction], ...]
    internal: tuple[tuple[str, Fraction], ...]
    residency_bytes: Fraction
    derived: tuple[tuple[str, Fraction], ...]
    formula: str
    fidelity: str = "isolated_device_time"
    stage_work: tuple[tuple[tuple[str, Fraction], ...], ...] = ()
    stage_stats: tuple[tuple[str, Fraction], ...] = ()
    native_call: dict | None = None
    native_occurrence: dict | None = None
    native_schedule: NativeSchedule | None = None
    native_program: NativeProgram | None = None

    def pipe_time(self, pipe: str) -> Fraction: return dict(self.times).get(pipe, Fraction(0))
    def pipe_work(self, pipe: str) -> Fraction:
        work = dict(self.work)
        if pipe == "link":
            # ``link`` is the aggregate transport class; route-specific pipes
            # (link_hccs/link_roce) preserve the selected topology.
            return sum((value for name, value in work.items()
                        if name == "link" or name.startswith("link_")), Fraction(0))
        return work.get(pipe, Fraction(0))
    def get(self, name: str, default: Fraction = Fraction(0)) -> Fraction: return dict(self.derived).get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "family": self.family, "kind": self.kind,
            "segment": self.segment_index,
            "regime": dict(self.regime),
            "guard_cells": [cell.to_dict() for cell in self.guard_cells],
            "work": {key: str(value) for key, value in self.work},
            "times_ms": {key: str(value) for key, value in self.times},
            "stage_times_ms": [str(value) for value in self.stage_times],
            "stage_work": [{pipe: str(value) for pipe, value in stage}
                           for stage in self.stage_work],
            "stage_stats_ms": {name: str(value) for name, value in self.stage_stats},
            "duration_ms": str(self.duration_ms), "dominant": self.dominant_pipe,
            "tied": list(self.tied_pipes),
            "materialization": {key: str(value) for key, value in self.materialization},
            "internal": {key: str(value) for key, value in self.internal},
            "residency_bytes": str(self.residency_bytes),
            "derived": {key: str(value) for key, value in self.derived},
            "formula": self.formula, "fidelity": self.fidelity,
            "native_call": self.native_call,
            "native_occurrence": self.native_occurrence,
        }

@dataclass(frozen=True)
class SegmentFormula:
    index: int
    regime: tuple[tuple[str, int], ...]
    guard_cells: tuple[GuardCell, ...]
    stages: tuple[tuple[PipeDemand, ...], ...]
    materialization: tuple[tuple[str, Expr], ...]
    derived: tuple[tuple[str, Expr], ...]
    formula: str
    domain: AnalysisDomain | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "regime": dict(self.regime),
            "guards": [cell.to_dict() for cell in self.guard_cells],
            "stages": [[demand.to_dict() for demand in stage] for stage in self.stages],
            "materialization": {key: str(value) for key, value in self.materialization},
            "derived": {key: str(value) for key, value in self.derived},
            "formula": self.formula,
            "domain": None if self.domain is None else self.domain.to_dict(),
        }

@dataclass(frozen=True)
class SegmentTable:
    segments: tuple[SegmentFormula, ...]
    total_cells: int
    truncated: bool = False
    axes: tuple[str, ...] = ()
    domain: AnalysisDomain | None = None

    def __len__(self) -> int: return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {"segments": [segment.to_dict() for segment in self.segments], "total_cells": self.total_cells,
                "truncated": self.truncated, "axes": list(self.axes),
                "domain": None if self.domain is None else self.domain.to_dict()}

@dataclass(frozen=True)
class SegmentHit:
    index: int; regime: tuple[tuple[str, int], ...]; guard_cells: tuple[GuardCell, ...]; env: tuple[tuple[str, Fraction], ...]

_FAMILY_COMPILERS: dict[str, Any] = {}

def measured_mac_rate(rule: Mapping[str, Any], cube_work: Any, *,
                      k: int | None = None, n: int | None = None) -> Fraction | None:
    """Return the measured cube FLOP/ms for the physically equivalent class.

    The underfilled class is the exact M16-fractal issued footprint from the
    independent C7.  The saturated class is the full-wave native macro grid:
    an integer ``m_issued`` with at least four full 128-row tiles (M512+).
    Other tails (M17/M128/M129) are unsupported and return None.
    """
    response = rule.get("response") or {}
    issue = response.get("issue") or {}
    k = int(k if k is not None else issue["k"])
    n = int(n if n is not None else issue["n"])
    work = Fraction(cube_work)
    loads = {str(item["id"]): item for item in response.get("loads", ())}
    under = loads.get("underfilled")
    saturated = loads.get("saturated")
    if under is not None and work == Fraction(int(under["measured_issued_flops"])):
        return Fraction(str(under["rate_flops_per_s"])) / 1000
    if saturated is None or k <= 0 or n <= 0:
        return None
    full_work = Fraction(int(saturated["measured_issued_flops"]))
    if full_work > 0:
        multiple = work / full_work
        if multiple.denominator == 1 and int(multiple) >= 1:
            return Fraction(str(saturated["rate_flops_per_s"])) / 1000
    return None


class CompiledOperator:
    """One bound operator compiled to a guarded family function."""

    def __init__(self, operator: BoundOperator, supply: PipeSupply,
                 policy: ImplementationPolicy, *, kind: str = "", layer: int = 0,
                 domain: AnalysisDomain | None = None,
                 siblings: tuple[BoundOperator, ...] = (),
                 variant: str = "",
                 feature_metadata: Iterable[tuple[str, Any]] = ()):
        if not isinstance(operator, BoundOperator):
            raise ModelingError("compiled_operator_bound_required")
        self.operator = operator
        self.siblings = tuple(siblings)
        self.supply = supply
        self.policy = policy
        self.kind = kind
        self.layer = int(layer)
        self.variant = str(variant or "")
        self.feature_metadata = tuple(feature_metadata or ())
        self.domain = domain or AnalysisDomain()
        self._native_bindings: dict[str, Any] = {}
        self._native_payloads: dict[str, Any] = {}
        self.template = _FAMILY_COMPILERS.get(operator.family)
        if self.template is None: raise ModelingError(f"operator_family_unsupported:{operator.family}")
        base_program: FamilyTemplate = self.template(self, operator)
        # Declared peaks, invocation floors and link latency are service
        # constraints.  They never rewrite the structural work program:
        # a higher peak divides elapsed time, and a floor is latency.
        self._raw_stages = base_program.stages
        self.program = base_program
        self._rate_scale: dict[str, Fraction] = {}
        self._service_floor = Fraction(0)
        self._service_startup = Fraction(0)
        self._service_dependency = Fraction(0)
        self._formula, self._theta_env_cached = _formula_text(self.program), self._theta_env()
        self._supply_ref = None
        self._eff_supply = {}
        self._has_hw_service = False
        self._measured_mac_rates: dict[str, Fraction] | None = None
        self._measured_mac_underfilled_flops = 0
        self._measured_mac_saturated_flops = 0
        self._measured_mac_k = 0
        self._measured_mac_n = 0
        self._selected_mac_rate: Fraction | None = None
        self._use_supply()

    def select_measured_mac(self, rule: Mapping[str, Any], *, hardware: str,
                            phase: str = "") -> bool:
        """Select a measured cube service rate for this exact native operator.

        This is a service-rate override only.  It leaves the operator's
        physical work, stage work, state effects, projections and every other
        pipe untouched.  The caller must supply the query hardware key; the
        native call itself carries the selected role, wrapper/kernel source,
        dtype, layout and K/N identity.
        """
        native = self.native_call_for(phase)
        native = native.to_dict() if hasattr(native, "to_dict") else native
        if not isinstance(native, Mapping):
            return False
        match = rule.get("match") or {}
        response = rule.get("response") or {}
        issue = response.get("issue") or {}
        if str(hardware) != str(match.get("hardware")):
            return False
        if str(native.get("recipe")) != str(match.get("recipe")):
            return False
        typed = native.get("typed") or {}
        attrs = typed.get("attrs") or {}
        if typed.get("dtype") != issue.get("dtype"):
            return False
        if attrs.get("weight_layout") != issue.get("weight_layout"):
            return False
        if (attrs.get("matrix_k"), attrs.get("matrix_n")) != (issue.get("k"), issue.get("n")):
            return False
        source = native.get("source") or {}
        if source.get("sha256") != issue.get("wrapper_source_sha256"):
            return False
        if source.get("revision") != issue.get("wrapper_source_revision"):
            return False
        kernels = native.get("kernels") or ()
        if not any(item.get("name") == issue.get("native_type")
                   and item.get("source_sha256") == issue.get("source_sha256")
                   for item in kernels):
            return False
        loads = {str(item["id"]): Fraction(str(item["rate_flops_per_s"])) / 1000
                 for item in response.get("loads", ())}
        if set(loads) != {"saturated", "underfilled"}:
            return False
        under = next(item for item in response["loads"] if item["id"] == "underfilled")
        saturated = next(item for item in response["loads"] if item["id"] == "saturated")
        self._measured_mac_rates = loads
        self._measured_mac_underfilled_flops = int(under["measured_issued_flops"])
        self._measured_mac_saturated_flops = int(saturated["measured_issued_flops"])
        self._measured_mac_k = int(issue["k"])
        self._measured_mac_n = int(issue["n"])
        self._selected_mac_rate = None
        return True

    def clear_measured_mac(self) -> None:
        self._measured_mac_rates = None
        self._measured_mac_underfilled_flops = 0
        self._measured_mac_saturated_flops = 0
        self._selected_mac_rate = None

    @property
    def name(self) -> str: return self.operator.name
    @property
    def family(self) -> str: return self.operator.family
    def native_call_for(self, phase: str = ""):
        """The immutable selected native call for one evaluated phase."""
        key = str(phase or "")
        if key not in self._native_bindings:
            from .native_calls import bind_compiled_operator
            self._native_bindings[key] = bind_compiled_operator(self, phase=key)
        return self._native_bindings[key]
    def native_payload_for(self, phase: str = ""):
        """One shared serialized call record; occurrences stay per row."""
        key = str(phase or "")
        if key not in self._native_payloads:
            binding = self.native_call_for(key)
            self._native_payloads[key] = binding.to_dict() if binding is not None else None
        return self._native_payloads[key]
    @property
    def native_call(self):
        return self.native_call_for("")
    def _native_cost(self, cost: OperatorCost, geometry: WorkloadGeometry,
                     env: Mapping[str, Any]) -> OperatorCost:
        from .native_calls import attach_native_occurrence
        return attach_native_occurrence(self, cost, geometry, env)
    @property
    def axes(self) -> tuple[SegmentAxis, ...]: return self.program.axes
    @property
    def piecewise_axes(self) -> tuple[str, ...]: return tuple(axis.name for axis in self.program.axes)

    def _validate_domain(self, geometry: WorkloadGeometry) -> None:
        geometry.validate(self.domain)

    def _theta_env(self) -> dict[str, Any]:
        theta = self.operator.theta
        buckets = theta.find("graph_buckets", ()) or ()
        max_b = int(theta.find("max_bucket", 0) or (max(int(x) for x in buckets) if buckets else 0))
        return {"MAX_BUCKET": max_b, "KV_BF16": int(theta.find("kv_bf16", 0) or 0),
                "F3_WFMT": str(theta.find("routed_weight_format", "") or "")}

    def _use_supply(self) -> None:
        """Bind effective rates to the current supply object.

        Callers rebind ``operator.supply`` for resource sweeps.  The cache
        follows that object; it does not freeze the supply seen at compile.
        """
        supply = self.supply
        if self._supply_ref is supply:
            return
        self._supply_ref = supply
        self._has_hw_service = bool(
            supply.invocations or supply.resource_units or supply.flops_by_dtype)
        self._eff_supply = {
            pipe: (as_fraction(supply.effective(pipe)) if self._has_hw_service
                   else supply.effective(pipe))
            for pipe, _rate in supply.rates}

    def _pipe_rate(self, pipe: str) -> Fraction:
        """Structural supply times the declared executed-peak promotion."""
        scale = self._rate_scale.get(pipe, Fraction(1))
        base = self._eff_supply.get(pipe)
        if base is None:
            base = as_fraction(self.supply.effective(pipe))
        return base * scale

    def _elapsed_rate(self, pipe: str, env: Mapping[str, Any]):
        """Denominator for one pipe's elapsed time.

        A binder may set ``HW_RATE_<PIPE>`` to the share of the declared
        supply this kernel can actually keep busy (active cores, tail wave).
        Absent means the whole supply.  The factor never scales work.
        """
        if pipe == "cube" and self._selected_mac_rate is not None:
            return self._selected_mac_rate
        key = f"HW_RATE_{pipe.upper()}"
        if self._has_hw_service:
            base = self._pipe_rate(pipe)
        else:
            base = self.supply.effective(pipe)
        if key not in env:
            return base
        cap = as_fraction(env[key])
        if cap <= 0:
            raise ModelingError(f"hw_rate_invalid:{pipe}:{cap}")
        return base * cap

    def _env(self, geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        self._use_supply()
        geometry = geometry.for_subgraph(self.kind)
        env = {**geometry.bindings(), **self.program.binder(geometry, state),
               **self._theta_env_cached, **dict(getattr(state, "env_extra", ()))}
        env.setdefault("IS_PREFILL", 0)
        for axis in self.program.axes:
            if axis.coordinate not in env: raise ModelingError(f"segment_coordinate_missing:{self.name}:{axis.coordinate}")
            env[axis.name] = axis.index(env[axis.coordinate])
        for name, expr in self.program.regime_exprs: env[name] = evaluate(expr, env)
        for axis in self.program.axes:
            coord_val, bound = env[axis.coordinate], self.domain.bound(axis.domain_key)
            measured = (env[axis.name] if axis.domain_key in {"max_query_tiles", "max_kv_tiles"} else coord_val)
            if measured > bound: raise ModelingError(f"analysis_domain_miss:{axis.coordinate}:{coord_val}>{bound}")
        for key, bound_key in _DERIVED_DOMAIN_KEYS:
            if key in env and env[key] > self.domain.bound(bound_key):
                raise ModelingError(f"analysis_domain_miss:{key}:{env[key]}>{self.domain.bound(bound_key)}")
        self._validate_domain(geometry)
        self._rate_scale = {}
        self._service_floor = Fraction(0)
        self._service_startup = Fraction(0)
        self._service_dependency = Fraction(0)
        if self._has_hw_service:
            raw_work: dict[str, Fraction] = {}
            for stage in self._raw_stages:
                for demand in stage:
                    value = evaluate(demand.work, env)
                    raw_work[demand.pipe] = raw_work.get(demand.pipe, Fraction(0)) + value
            scale_pipes, add_span, _target_pipe = _compute_hardware_service_factors(
                self, geometry, env, raw_work)
            # ``scale_pipes`` is declared_peak / default supply.  It promotes
            # the clock.  Multiplying it into FLOPs or bytes makes a faster
            # peak slower and invents work.
            self._rate_scale = {pipe: scale for pipe, scale in scale_pipes.items() if scale != 1}
            self._service_floor = add_span
            self._service_startup = as_fraction(env.get("HW_STARTUP_MS", 0))
            self._service_dependency = as_fraction(env.get("HW_DEPENDENCY_MS", 0))
            for pipe in _PIPE_UNITS:
                env[f"HW_SCALE_{pipe.upper()}"] = scale_pipes.get(pipe, Fraction(1))
            env["HW_SPAN_WORK"] = Fraction(0)
        return env

    def _segment_index(self, env: Mapping[str, Any],
                       domain: AnalysisDomain | None = None) -> int:
        domain = self.domain if domain is None else domain
        index = 0
        for axis in self.program.axes:
            position = axis.position(int(env[axis.name]), domain)
            index = index * max(1, len(axis.index_range(domain))) + position
        return index

    def segment(self, geometry: WorkloadGeometry, state: StateView | None = None) -> SegmentHit:
        state = state or StateView()
        env = self._env(geometry, state)
        cells = tuple(axis.cell(env[axis.name]) for axis in self.program.axes)
        regime = tuple((axis.name, int(env[axis.name])) for axis in self.program.axes)
        return SegmentHit(self._segment_index(env), regime, cells,
                          tuple((key, _frac(value)) for key, value in sorted(env.items())
                                if isinstance(value, (int, Fraction))))

    def _check_readiness(self, geometry: WorkloadGeometry,
                         state: StateView) -> None:
        """Enforce the producer ports this operator actually declares."""
        if self.kind == "I1" or self.name in {"index_score", "topk", "index_filter"}:
            # Only the cache-consuming score/filter declares require_index.
            # ``topk`` produces the selection and must not invent a requirement
            # on the consumer layer when the cache owner is an earlier layer.
            raw_owner = self.operator.theta.find("require_index", None)
            if raw_owner is not None:
                owner = int(raw_owner)
                state.require_index(owner if owner >= 0 else self.layer)
        if self.kind == "A1" and self.name == "mla_attention":
            theta = self.operator.theta
            for port, fn in (("require_index", state.require_index), ("require_selection", state.require_selection)):
                raw = theta.find(port, None)
                if raw is not None:
                    fn(int(raw) if int(raw) >= 0 else self.layer)
        if self.kind == "A2" and self.name in {"window_attention", "compressed_attention"}:
            theta = self.operator.theta
            kv_owner = theta.find("require_kv", None)
            if kv_owner is not None:
                state.require_kv(int(kv_owner) if int(kv_owner) >= 0 else self.layer)
            index_owner = theta.find("require_index", None)
            if index_owner is not None:
                state.require_index(int(index_owner) if int(index_owner) >= 0 else self.layer)
            selection_owner = theta.find("require_selection", None)
            if selection_owner is not None:
                state.require_selection(
                    int(selection_owner) if int(selection_owner) >= 0 else self.layer)
            ratio = int(theta.find("require_compression", 0) or 0)
            owner = int(theta.find("compression_owner", self.layer))
            owner = self.layer if owner < 0 else owner
            if (ratio > 0 and geometry.s_max >= ratio
                    and not (_compression_recorded(state, owner) or _selection_ready(state, self.layer))):
                raise ModelingError(f"state_port_not_ready:compressed_selection:layer={owner}")

    def evaluate(self, geometry: WorkloadGeometry,
                 state: StateView | None = None, *,
                 readiness: bool = True) -> OperatorCost:
        geometry = geometry.for_subgraph(self.kind)
        geometry = geometry.operator_layout(
            self.kind, str(getattr(self.operator.theta, "role", "") or ""),
            self.name, self.family)
        check = readiness and state is not None
        state = state or StateView()
        if check:
            self._check_readiness(geometry, state)
        env = self._env(geometry, state)
        active = (True if self.program.active is None else
                  evaluate(self.program.active, env) != 0)
        if not active:
            # No matching owner/phase: do not charge a kernel floor or
            # materialize an empty result.  Work, staged time and residency
            # are all zero; the descriptor remains explicit in derived rows.
            derived = tuple((name, evaluate(expr, env))
                            for name, expr in self.program.derived)
            return self._native_cost(OperatorCost(
                name=self.name, family=self.family, kind=self.kind,
                segment_index=self._segment_index(env),
                regime=tuple((axis.name, int(env[axis.name]))
                             for axis in self.program.axes),
                guard_cells=tuple(axis.cell(env[axis.name])
                                  for axis in self.program.axes),
                work=(), times=(), stage_times=(), stage_work=(),
                duration_ms=Fraction(0), dominant_pipe="", tied_pipes=(),
                materialization=tuple((name, Fraction(0))
                                      for name, _ in self.program.materialization),
                internal=tuple((name, Fraction(0))
                               for name, _ in self.program.internal),
                residency_bytes=Fraction(0), derived=derived,
                formula=self._formula, fidelity="inactive"), geometry, env)
        self._selected_mac_rate = None
        if self._measured_mac_rates is not None:
            cube_total = sum(
                (evaluate(demand.work, env)
                 for stage in self.program.stages for demand in stage
                 if demand.pipe == "cube"),
                Fraction(0),
            )
            self._selected_mac_rate = measured_mac_rate(
                {"match": {}, "response": {"issue": {
                    "k": self._measured_mac_k, "n": self._measured_mac_n,
                }, "loads": [
                    {"id": "saturated", "measured_issued_flops": self._measured_mac_saturated_flops,
                     "rate_flops_per_s": self._measured_mac_rates["saturated"] * 1000},
                    {"id": "underfilled", "measured_issued_flops": self._measured_mac_underfilled_flops,
                     "rate_flops_per_s": self._measured_mac_rates["underfilled"] * 1000},
                ]}},
                cube_total, k=self._measured_mac_k, n=self._measured_mac_n)
        stage_work: list[dict[str, Fraction]] = []
        stage_times: list[Fraction] = []
        all_work: dict[str, Fraction] = {}
        all_times: dict[str, Fraction] = {}
        peak_times: dict[str, Fraction] = {}
        for stage in self.program.stages:
            # One demand per pipe is the concurrent stage; a repeated pipe
            # keeps the last expression, matching the stage dictionary.
            work = {demand.pipe: evaluate(demand.work, env) for demand in stage}
            times = {pipe: value / self._elapsed_rate(pipe, env) for pipe, value in work.items()}
            stage_work.append(work)
            stage_times.append(max(times.values()) if times else Fraction(0))
            for pipe, value in work.items():
                all_work[pipe] = all_work.get(pipe, Fraction(0)) + value
            for pipe, value in times.items():
                all_times[pipe] = all_times.get(pipe, Fraction(0)) + value
                peak_times[pipe] = max(peak_times.get(pipe, Fraction(0)), value)
        program_work = None
        program_result = (None if self.program.program_duration is None else
                          self.program.program_duration(geometry, state))
        if program_result is not None:
            # The selected physical program owns the elapsed time; its plan
            # work remains the resource ledger, never a second roofline charge.
            program_work = None
            if isinstance(program_result, tuple):
                duration, dominant_pipe, tied_pipes, busy = program_result[:4]
                if len(program_result) > 4:
                    program_work = dict(program_result[4])
                all_times = dict(busy)
                peak_times = dict(busy)
                stage_times = []
            else:
                duration = Fraction(program_result)
                all_times = {}
                peak_times = {}
                stage_times = []
                dominant_pipe, tied_pipes = "", ()
                program_work = None
        else:
            duration = max(
                self._service_floor,
                self._service_startup + sum(stage_times, Fraction(0)) + self._service_dependency,
            )
        if program_work is not None:
            # The plan's Work is the exact resource ledger; the old aggregate
            # stage expressions are not added alongside it.  The native plan
            # is one concurrent owner stage, so its exported stage_work must
            # carry that ledger (including register issue/ALU/UB work), not
            # the analytical aggregate with zero register entries.
            all_work = dict(program_work)
            stage_work = [dict(all_work)]
        native_program = None
        if (program_result is not None and isinstance(program_result, tuple)
                and len(program_result) > 5
                and isinstance(program_result[5], NativeProgram)):
            native_program = program_result[5]
        native_schedule = None
        if (native_program is None and program_result is not None
                and program_work is not None):
            # Retain the native owner program as explicit concurrent tasks.
            # The analytical stage aggregate can differ (e.g. register work
            # collapsed to zero), so it must not drive the shared engine.
            tasks = []
            modes = dict(self.supply.service_modes)
            rates = dict(self.supply.rates)
            for pipe, amount in sorted(all_work.items()):
                if amount <= 0:
                    continue
                busy = all_times.get(pipe)
                if busy and busy > 0:
                    tasks.append(NativeTask(str(pipe), as_fraction(amount),
                                            as_fraction(amount) / as_fraction(busy)))
                elif pipe in modes:
                    role, rate = modes[pipe]
                    if role in rates:
                        tasks.append(NativeTask(str(role), as_fraction(amount),
                                                as_fraction(rate)))
                elif pipe in rates:
                    tasks.append(NativeTask(str(pipe), as_fraction(amount)))
            if tasks:
                peak = max(all_times.values(), default=Fraction(0))
                startup = max(Fraction(0), as_fraction(duration) - peak)
                native_schedule = NativeSchedule(
                    (tuple(tasks),), startup=startup,
                    floor=self._service_floor, duration=as_fraction(duration),
                    fidelity="partial_aggregate")
        stage_stats = (tuple(self.program.program_stages(geometry, state))
                       if self.program.program_stages is not None else ())
        if self.program.minimum_duration is not None:
            duration = max(duration, Fraction(
                self.program.minimum_duration(geometry, state)))
        if not peak_times:
            dominant, tied = "", ()
        else:
            peak = max(peak_times.values())
            dominant = next(pipe for pipe in peak_times if peak_times[pipe] == peak)
            tied = tuple(pipe for pipe in peak_times if peak_times[pipe] == peak)
        materialization = tuple((name, evaluate(expr, env))
                                for name, expr in self.program.materialization)
        internal = tuple((name, evaluate(expr, env))
                         for name, expr in self.program.internal)
        derived = tuple((name, evaluate(expr, env)) for name, expr in self.program.derived) + (
            (("kv_bf16", Fraction(1)),) if int(env.get("KV_BF16", 0) or 0) > 0 else ()
        )
        if self._service_floor > 0 or self._service_startup > 0 or self._service_dependency > 0:
            derived = derived + (
                ("service_floor_ms", self._service_floor),
                ("service_startup_ms", self._service_startup),
                ("service_dependency_ms", self._service_dependency),
            )
        cells = tuple(axis.cell(env[axis.name]) for axis in self.program.axes)
        regime = tuple((axis.name, int(env[axis.name])) for axis in self.program.axes)
        return self._native_cost(OperatorCost(
            name=self.name, family=self.family, kind=self.kind,
            segment_index=self._segment_index(env), regime=regime, guard_cells=cells,
            work=tuple(sorted(all_work.items())), times=tuple(sorted(all_times.items())),
            stage_times=tuple(stage_times),
            stage_work=tuple(tuple(sorted(stage.items())) for stage in stage_work),
            stage_stats=stage_stats,
            duration_ms=duration,
            dominant_pipe=dominant, tied_pipes=tied,
            materialization=materialization, internal=internal,
            residency_bytes=evaluate(self.program.residency, env),
            derived=derived, formula=self._formula,
            native_schedule=native_schedule,
            native_program=native_program,
        ), geometry, env)

    def cell_count(self, domain: AnalysisDomain) -> int:
        return math.prod(len(axis.index_range(domain))
                         for axis in self.program.axes)

    def piecewise(self, domain: AnalysisDomain, *, limit: int = 4096) -> SegmentTable:
        if limit < 1:
            raise ModelingError("segment_limit_invalid")
        axes = self.program.axes
        if axes:
            ranges = [axis.index_range(domain) for axis in axes]
            total = math.prod(len(values) for values in ranges)
        else:
            ranges, total = [], 1
        segments: list[SegmentFormula] = []
        truncated = False
        for combo in _product_indices(ranges):
            if len(segments) >= limit:
                truncated = True
                break
            substitutions = {axis.name: Const(value) for axis, value in zip(axes, combo)}
            stages = tuple(tuple(PipeDemand(demand.pipe,
                                            simplify(_substitute(demand.work, substitutions)),
                                            demand.supply, demand.unit)
                                 for demand in stage) for stage in self.program.stages)
            materialization = tuple(
                (name, simplify(_substitute(expr, substitutions)))
                for name, expr in self.program.materialization)
            derived = tuple((name, simplify(_substitute(expr, substitutions)))
                            for name, expr in self.program.derived)
            regime = dict((axis.name, int(value)) for axis, value in zip(axes, combo))
            cells = tuple(axis.cell(value) for axis, value in zip(axes, combo))
            # The exported identity belongs to the supplied export domain,
            # never to the compiler's runtime domain.
            index = self._segment_index(regime, domain)
            segments.append(SegmentFormula(
                index, tuple(regime.items()), cells, stages, materialization,
                derived, _formula_text(self.program), domain))
        return SegmentTable(tuple(segments), total, truncated,
                            tuple(axis.name for axis in axes), domain)

    def dominance(self, geometry: WorkloadGeometry,
                  state: StateView | None = None, *, stage: int = 0) -> PipeDominanceGuard:
        state = state or StateView()
        env = self._env(geometry, state)
        stages = self.program.stages
        if not 0 <= stage < len(stages):
            raise ModelingError(f"operator_stage_missing:{self.name}:{stage}")
        return PipeDominanceGuard(tuple(
            PipeDemand(demand.pipe, demand.work, demand.supply, demand.unit)
            for demand in stages[stage]))

@dataclass(frozen=True)
class SubgraphEvaluation:
    instance_id: str
    kind: str
    operators: tuple[OperatorCost, ...]
    total_ms: Fraction
    state_out: StateView

    def to_dict(self) -> dict[str, Any]:
        return {"instance_id": self.instance_id, "kind": self.kind,
                "total_ms": str(self.total_ms),
                "operators": [cost.to_dict() for cost in self.operators]}

class CompiledSubgraph:
    def __init__(self, instance: SubgraphInstance, supply: PipeSupply,
                 policy: ImplementationPolicy, *,
                 domain: AnalysisDomain | None = None,
                 feature_metadata: Iterable[tuple[str, Any]] = ()):
        self.instance = instance
        self.supply = supply
        self.policy = policy
        self.domain = domain or AnalysisDomain()
        self.feature_metadata = tuple(feature_metadata or ())
        self.operators = tuple(
            CompiledOperator(operator, supply, policy, kind=instance.kind,
                             layer=instance.layer, domain=self.domain,
                             siblings=instance.operators, variant=instance.variant,
                             feature_metadata=self.feature_metadata)
            for operator in instance.operators)

    def operator(self, name: str) -> CompiledOperator:
        for operator in self.operators:
            if operator.name == name:
                return operator
        raise ModelingError(f"compiled_operator_missing:{self.instance.instance_id}:{name}")

    def select_measured_mac(self, rule: Mapping[str, Any], *, hardware: str,
                            phase: str = "") -> bool:
        """Apply a measured cube service rate to every matching operator."""
        applied = False
        for operator in self.operators:
            applied = operator.select_measured_mac(
                rule, hardware=hardware, phase=phase) or applied
        return applied

    def clear_measured_mac(self) -> None:
        for operator in self.operators:
            operator.clear_measured_mac()

    def evaluate(self, geometry: WorkloadGeometry,
                 state: StateView | None = None) -> SubgraphEvaluation:
        geometry = geometry.for_subgraph(self.instance.kind)
        geometry.validate(self.domain)
        supplied = state is not None
        state = state or StateView()
        costs = []
        for operator in self.operators:
            if supplied:
                operator._check_readiness(geometry, state)
            cost = operator.evaluate(geometry, state, readiness=supplied)
            costs.append(cost)
            state = _apply_operator_state(state, operator, cost, self.instance, geometry)
        return SubgraphEvaluation(self.instance.instance_id, self.instance.kind,
                                  tuple(costs), sum((cost.duration_ms for cost in costs),
                                                    Fraction(0)), state)

    def piecewise(self, domain: AnalysisDomain, *, limit: int = 4096) -> SegmentTable:
        if limit < 1:
            raise ModelingError("segment_limit_invalid")
        total = sum(operator.cell_count(domain) for operator in self.operators)
        segments: list[SegmentFormula] = []
        for operator in self.operators:
            remaining = limit - len(segments)
            if remaining <= 0:
                break
            table = operator.piecewise(domain, limit=remaining)
            segments.extend(table.segments)
            if table.truncated:
                break
        axes = tuple(dict.fromkeys(
            name for operator in self.operators for name in operator.piecewise_axes))
        return SegmentTable(tuple(segments), total, total > len(segments), axes,
                            domain)

class OperatorPiecewiseCompiler:
    """Compile bound subgraph instances against one declared hardware supply."""

    def __init__(self, supply: PipeSupply, policy: ImplementationPolicy | None = None,
                 domain: AnalysisDomain | None = None):
        self.supply = supply
        self.policy = policy or ImplementationPolicy()
        self.domain = domain or AnalysisDomain()

    def compile(self, instance: SubgraphInstance, *,
                feature_metadata: Iterable[tuple[str, Any]] = ()
                ) -> CompiledSubgraph:
        return CompiledSubgraph(instance, self.supply, self.policy,
                                domain=self.domain,
                                feature_metadata=feature_metadata)

    def compile_many(self, instances: Iterable[SubgraphInstance], *,
                     feature_metadata: Iterable[tuple[str, Any]] = ()
                     ) -> tuple[CompiledSubgraph, ...]:
        return tuple(self.compile(instance, feature_metadata=feature_metadata)
                     for instance in instances)

    def compile_operator(self, operator: BoundOperator, *, kind: str = "",
                         variant: str = "",
                         feature_metadata: Iterable[tuple[str, Any]] = ()
                         ) -> CompiledOperator:
        return CompiledOperator(operator, self.supply, self.policy, kind=kind,
                                variant=variant, domain=self.domain,
                                feature_metadata=feature_metadata)

# ---------------------------------------------------------------------------
# Shared helpers

def _substitute(expr: Expr, mapping: Mapping[str, Expr]) -> Expr:
    from .expressions import substitute
    return substitute(expr, mapping)

def _formula_text(program: FamilyTemplate) -> str:
    if not program.stages:
        return ""
    parts = []
    for index, stage in enumerate(program.stages):
        text = ", ".join(f"{demand.work}/{demand.supply}" for demand in stage)
        parts.append(f"stage{index} max({text})")
    return " ; ".join(parts)

def _product_indices(ranges: list[tuple[int, ...]]):
    if not ranges:
        yield ()
        return
    for value in ranges[0]:
        for tail in _product_indices(ranges[1:]):
            yield (value, *tail)

def _demand(ctx: CompiledOperator, pipe: str, work: Expr, unit: str = "") -> PipeDemand:
    return PipeDemand(pipe, work, ctx.supply.effective(pipe),
                      unit or _PIPE_UNITS.get(pipe, "work"))

def _out_name(ctx: CompiledOperator) -> str:
    name = ctx.name
    mapping = {
        "qkv_projection": "qkv_activation", "q_down": "q_lora_activation",
        "q_up": "q_activation", "q_direct": "q_activation",
        "gate_up": "ffn_activation", "shared_gate_up": "shared_ffn_activation",
        "activation": "activation_intermediate",
        "shared_activation": "shared_activation_intermediate",
        "lm_head": "logits", "final_norm": "hidden_out",
    }
    return mapping.get(name, name + "_output")

# ---------------------------------------------------------------------------
# Family compilers

def _zero() -> Expr:
    return Const(0)

def _uniform_counts(tokens: int, top_k: int, local: int) -> tuple[int, ...]:
    total = max(0, int(tokens)) * max(1, int(top_k))
    base, extra = divmod(total, max(1, local))
    return tuple(base + (1 if index < extra else 0) for index in range(local))

def _compression_recorded(state: StateView, layer: int) -> bool:
    records = tuple(state.compression) + tuple(
        (owner, record) for (owner, _), record in state.compression_requests)
    return any(key == layer and record.consumed > 0 for key, record in records)

def _selection_ready(state: StateView, layer: int) -> bool:
    return (any(key == layer and value in {"index_selection", "selection"}
                for key, value in state.index_ready)
            or any(key == layer for key, _ in getattr(state, "selection_ready", ())))

def _apply_operator_state(state: StateView, operator: CompiledOperator,
                          cost: OperatorCost, instance: SubgraphInstance,
                          geometry: WorkloadGeometry | None = None) -> StateView:
    """Apply only write effects; readiness checks live in family binders."""
    for effect, value in operator.program.state_effects:
        if effect == "index":
            state = state.mark_index(instance.layer)
        elif effect == "kv":
            if value == "kv_store":
                state = state.mark_kv(instance.layer)
            elif value == "kv_read":
                layer = int(operator.operator.theta.find("owner_layer", instance.layer)
                            or instance.layer)
                if instance.owner_layer is not None:
                    layer = instance.owner_layer
                state.require_kv(layer)
            elif (value in {"index_store", "index_cache_store"}
                  or getattr(operator.operator, "role", "") == "index_store"):
                state = state.mark_index(instance.layer)
        elif effect == "residual":
            state = state.push_residual(instance.layer)
        elif effect == "compress" and operator.name == "compress":
            ratio = 1
            for key, item in operator.operator.theta.extras:
                if key == "ratio":
                    ratio = int(item)
            if geometry is not None:
                requests = tuple(
                    (index, request.context, request.q_len)
                    for index, request in enumerate(geometry.requests))
                state = state.update_compression(
                    instance.layer, max(1, ratio), requests)[0]
    if state.compression_requests:
        # Model-level K1 advancement may record per-request progress only;
        # keep every aggregate owner cursor aligned for downstream readiness.
        owners = dict(state.compression)
        for (owner, _), record in state.compression_requests:
            current = owners.get(owner)
            if current is None or record.consumed > current.consumed:
                owners[owner] = record
        state = replace(state, compression=tuple(sorted(owners.items())))
    if instance.kind == "I1" and operator.name == "topk":
        # Top-k produces selection readiness, never an index-cache row.  Marking
        # the consumer layer's index cache here would hide a missing producer
        # when a shared layer reuses an earlier cache owner.
        owner = int(operator.operator.theta.find("owner_layer", instance.layer))
        state = state.mark_selection(owner if owner >= 0 else instance.layer,
                                     "index_selection")
    return state

from .families import (  # noqa: E402
    FAMILY_COMPILERS as _FAMILY_IMPLEMENTATIONS,
    compute_hardware_service_factors as _compute_hardware_service_factors,
    extract_hardware_meta as _extract_hardware_meta,
)
_FAMILY_COMPILERS.update(_FAMILY_IMPLEMENTATIONS)
