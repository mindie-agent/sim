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

    def __post_init__(self) -> None:
        rates = tuple(sorted((str(pipe), float(rate)) for pipe, rate in self.rates))
        if not rates:
            raise ModelingError("pipe_supply_empty")
        for pipe, rate in rates:
            if not pipe or not math.isfinite(rate) or rate <= 0:
                raise ModelingError(f"pipe_supply_invalid:{pipe}")
        object.__setattr__(self, "rates", rates)
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
        die = hardware.dies[0]
        rates = {name: float(value) * float(time_scale) for name, value in die.rates}
        rates["cube"] = float(die.peak(dtype)) * float(time_scale)
        undeclared = fill_undeclared_byte_pipes(rates)
        if "memory" in rates and "hbm" not in dict(die.rates):
            rates["hbm"] = rates.pop("memory")
        if link_rate is not None:
            rates["link"] = float(link_rate)
        elif hardware.links:
            rates["link"] = min(link.capacity_bytes_per_ms for link in hardware.links)
        inv, units, peaks = _extract_hardware_meta(hardware, float(time_scale))
        return cls(tuple(rates.items()), (), inv, units, peaks, undeclared)

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

    def pipe_time(self, pipe: str) -> Fraction: return dict(self.times).get(pipe, Fraction(0))
    def pipe_work(self, pipe: str) -> Fraction: return dict(self.work).get(pipe, Fraction(0))
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
            "duration_ms": str(self.duration_ms), "dominant": self.dominant_pipe,
            "tied": list(self.tied_pipes),
            "materialization": {key: str(value) for key, value in self.materialization},
            "internal": {key: str(value) for key, value in self.internal},
            "residency_bytes": str(self.residency_bytes),
            "derived": {key: str(value) for key, value in self.derived},
            "formula": self.formula, "fidelity": self.fidelity,
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

class CompiledOperator:
    """One bound operator compiled to a guarded family function."""

    def __init__(self, operator: BoundOperator, supply: PipeSupply,
                 policy: ImplementationPolicy, *, kind: str = "", layer: int = 0,
                 domain: AnalysisDomain | None = None,
                 siblings: tuple[BoundOperator, ...] = ()):
        if not isinstance(operator, BoundOperator):
            raise ModelingError("compiled_operator_bound_required")
        self.operator = operator
        self.siblings = tuple(siblings)
        self.supply = supply
        self.policy = policy
        self.kind = kind
        self.layer = int(layer)
        self.domain = domain or AnalysisDomain()
        self.template = _FAMILY_COMPILERS.get(operator.family)
        if self.template is None: raise ModelingError(f"operator_family_unsupported:{operator.family}")
        base_program: FamilyTemplate = self.template(self, operator)
        self._raw_stages, self._has_hw_service = base_program.stages, bool(self.supply.invocations or self.supply.resource_units)
        if self._has_hw_service and base_program.stages:
            pipes = {d.pipe for s in base_program.stages for d in s}
            t_pipe = ("vector" if self.family in {"collective", "transport"} and "vector" in pipes else ("link" if self.family in {"collective", "transport"} and "link" in pipes else ("cube" if "cube" in pipes else ("vector" if "vector" in pipes else next(iter(pipes), "hbm")))))
            scaled = tuple(tuple(PipeDemand(d.pipe, simplify(d.work * Var(f"HW_SCALE_{d.pipe.upper()}")), d.supply, d.unit) for d in s) for s in base_program.stages)
            span_stage = (PipeDemand(t_pipe, Var("HW_SPAN_WORK"), self.supply.effective(t_pipe), _PIPE_UNITS.get(t_pipe, "work")),)
            self.program = replace(base_program, stages=(*scaled, span_stage))
        else: self.program = base_program

    @property
    def name(self) -> str: return self.operator.name
    @property
    def family(self) -> str: return self.operator.family
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

    def _env(self, geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        geometry = geometry.for_subgraph(self.kind)
        env = {**geometry.bindings(), **self.program.binder(geometry, state),
               **self._theta_env(), **dict(getattr(state, "env_extra", ()))}
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
        if self._has_hw_service and self._raw_stages:
            raw_work: dict[str, Fraction] = {}
            for stage in self._raw_stages:
                for d in stage: raw_work[d.pipe] = raw_work.get(d.pipe, Fraction(0)) + evaluate(d.work, env)
            scale_pipes, add_span, target_pipe = _compute_hardware_service_factors(self, geometry, env, raw_work)
            for pipe in _PIPE_UNITS: env[f"HW_SCALE_{pipe.upper()}"] = scale_pipes.get(pipe, Fraction(1))
            env["HW_SPAN_WORK"] = add_span * as_fraction(self.supply.effective(target_pipe))
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
            owner = int(self.operator.theta.find("require_index", self.layer))
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
        check = readiness and state is not None
        state = state or StateView()
        if check:
            self._check_readiness(geometry, state)
        env = self._env(geometry, state)
        stage_work: list[dict[str, Fraction]] = []
        stage_times: list[Fraction] = []
        all_work: dict[str, Fraction] = {}
        all_times: dict[str, Fraction] = {}
        peak_times: dict[str, Fraction] = {}
        for stage in self.program.stages:
            work = {demand.pipe: evaluate(demand.work, env) for demand in stage}
            times = {pipe: value / (as_fraction(self.supply.effective(pipe)) if self._has_hw_service else self.supply.effective(pipe))
                     for pipe, value in work.items()}
            stage_work.append(work)
            stage_times.append(max(times.values()) if times else Fraction(0))
            for pipe, value in work.items():
                all_work[pipe] = all_work.get(pipe, Fraction(0)) + value
            for pipe, value in times.items():
                all_times[pipe] = all_times.get(pipe, Fraction(0)) + value
                peak_times[pipe] = max(peak_times.get(pipe, Fraction(0)), value)
        duration = sum(stage_times, Fraction(0))
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
        derived = tuple((name, evaluate(expr, env)) for name, expr in self.program.derived)
        cells = tuple(axis.cell(env[axis.name]) for axis in self.program.axes)
        regime = tuple((axis.name, int(env[axis.name])) for axis in self.program.axes)
        return OperatorCost(
            name=self.name, family=self.family, kind=self.kind,
            segment_index=self._segment_index(env), regime=regime, guard_cells=cells,
            work=tuple(sorted(all_work.items())), times=tuple(sorted(all_times.items())),
            stage_times=tuple(stage_times),
            stage_work=tuple(tuple(sorted(stage.items())) for stage in stage_work),
            duration_ms=duration,
            dominant_pipe=dominant, tied_pipes=tied,
            materialization=materialization, internal=internal,
            residency_bytes=evaluate(self.program.residency, env),
            derived=derived, formula=_formula_text(self.program),
        )

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
                 domain: AnalysisDomain | None = None):
        self.instance = instance
        self.supply = supply
        self.policy = policy
        self.domain = domain or AnalysisDomain()
        self.operators = tuple(
            CompiledOperator(operator, supply, policy, kind=instance.kind,
                             layer=instance.layer, domain=self.domain,
                             siblings=instance.operators)
            for operator in instance.operators)

    def operator(self, name: str) -> CompiledOperator:
        for operator in self.operators:
            if operator.name == name:
                return operator
        raise ModelingError(f"compiled_operator_missing:{self.instance.instance_id}:{name}")

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

    def compile(self, instance: SubgraphInstance) -> CompiledSubgraph:
        return CompiledSubgraph(instance, self.supply, self.policy,
                                domain=self.domain)

    def compile_many(self, instances: Iterable[SubgraphInstance]) -> tuple[CompiledSubgraph, ...]:
        return tuple(self.compile(instance) for instance in instances)

    def compile_operator(self, operator: BoundOperator, *, kind: str = "") -> CompiledOperator:
        return CompiledOperator(operator, self.supply, self.policy, kind=kind,
                                domain=self.domain)

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
        state = state.mark_index(instance.layer, "index_selection")
    return state

from .families import (  # noqa: E402
    FAMILY_COMPILERS as _FAMILY_IMPLEMENTATIONS,
    compute_hardware_service_factors as _compute_hardware_service_factors,
    extract_hardware_meta as _extract_hardware_meta,
)
_FAMILY_COMPILERS.update(_FAMILY_IMPLEMENTATIONS)
