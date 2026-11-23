"""Canonical typed Python SDK backed by the single internal ``PredictionSpine``.

[Role] Expose the seven typed top-level simulation entrypoints
  (``compose_case``, ``evaluate_step``, ``evaluate_three_tiers``,
  ``evaluate_service``, ``solve_sla``, ``sweep_topology``, ``export_structure``)
  consuming immutable domain requests and returning frozen domain results.
[Invariants] Every entrypoint delegates directly to ``default_spine()`` and
  the single ``runtime.Engine``. Caller mappings are snapshotted and never
  mutated. ``compose_case`` validates options against the real composer
  signature and returns a parsed ``Case``.
[Timeouts & Locks] Pure bounded delegation honoring ``ReplayOptions`` deadlines;
  no locks or device contexts.
[Side-Effect Policy] Read-only over caller inputs; no filesystem writes.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from mindie_sim.compose import compose_case as _compose_case_dict
from mindie_sim.prediction.offload.policy import missing_scenario_error
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.results import (
    CaseInput,
    HardwareInput,
    ModelInput,
    ReplayOptions,
    ServiceEvaluation,
    SlaEvaluation,
    SlaRequest,
    StepEvaluation,
    StepRequest,
    StructureExport,
    StructureRequest,
    SweepRequest,
    ThreeTierEvaluation,
    TopologySweepResult,
)
from mindie_sim.prediction.spine import DEFAULT_OPTIMIZE_TRIALS, default_spine
from mindie_sim.prediction.streams import StreamPlan
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.stack_registry import LoweringRequest


def _to_case(case: CaseInput) -> Case:
    if isinstance(case, Case):
        return case
    return parse_case(deepcopy(dict(case)))


def compose_case(
    model: ModelInput,
    hardware: HardwareInput = "a3",
    *,
    options: Mapping[str, object] | None = None,
) -> Case:
    """Synthesize and parse a complete immutable ``Case``."""
    model_arg: Any = str(model) if isinstance(model, Path) else (
        deepcopy(dict(model)) if isinstance(model, Mapping) else model
    )
    hw_arg: Any = (
        deepcopy(dict(hardware)) if isinstance(hardware, Mapping) else hardware
    )
    kwargs = (
        {key: deepcopy(val) for key, val in options.items()}
        if options is not None else {}
    )
    try:
        raw = _compose_case_dict(model=model_arg, hardware=hw_arg, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" in str(exc):
            raise ModelingError(f"compose_case_unknown_options:{exc}") from exc
        raise
    return parse_case(raw)


def evaluate_step(
    case: CaseInput,
    request: StepRequest,
    /,
    *,
    replay: ReplayOptions = ReplayOptions(),
    primitives: PrimitiveStore | None = None,
    primitive_table: Any = None,
    streams: tuple[StreamPlan, ...] = (),
) -> StepEvaluation:
    """Lower and evaluate one fixed step on ``runtime.Engine``."""
    spine = default_spine()
    prepared = spine.lower(
        _to_case(case), LoweringRequest(step=request), streams=streams
    )
    return spine.run_step(
        prepared, replay, primitives=primitives, primitive_table=primitive_table
    )


def evaluate_three_tiers(
    case: CaseInput,
    request: StepRequest,
    /,
    *,
    replay: ReplayOptions = ReplayOptions(),
    primitives: PrimitiveStore | None = None,
    primitive_table: Any = None,
) -> ThreeTierEvaluation:
    """Evaluate ``unmerge``, ``mainline``, and constructive ``should_achieve`` on ``runtime.Engine``."""
    return default_spine().evaluate_three_tiers(
        _to_case(case),
        request,
        replay=replay,
        primitives=primitives,
        primitive_table=primitive_table,
    )


def optimize_ladder(
    case: CaseInput,
    request: StepRequest,
    /,
    *,
    replay: ReplayOptions = ReplayOptions(),
    primitives: PrimitiveStore | None = None,
    primitive_table: Any = None,
    stack_id: str = "cann93_va_main",
    max_trials: int = DEFAULT_OPTIMIZE_TRIALS,
    allow_unbound_formula: bool = False,
) -> Any:
    """Solve the certified 5-level primal-dual performance ladder for one step."""
    return default_spine().optimize_ladder(
        _to_case(case),
        request,
        replay=replay,
        primitives=primitives,
        primitive_table=primitive_table,
        stack_id=stack_id,
        max_trials=max_trials,
        allow_unbound_formula=allow_unbound_formula,
    )


def evaluate_service(
    case: CaseInput,
    /,
    *,
    replay: ReplayOptions = ReplayOptions(),
    primitives: PrimitiveStore | None = None,
    primitive_table: Any = None,
) -> ServiceEvaluation:
    """Evaluate a complete serving workload through ``PredictionSpine``."""
    parsed = _to_case(case)
    if dict(parsed.runtime.metadata).get("scenario_declared") is False:
        raise missing_scenario_error(("prompt_tokens", "output_tokens"))
    return default_spine().run_service(
        parsed, replay, primitives=primitives, primitive_table=primitive_table
    )


def solve_sla(case: CaseInput, request: SlaRequest, /) -> SlaEvaluation:
    """Solve SLA-constrained concurrency sizing."""
    return default_spine().solve_sla(_to_case(case), request)


def solve_optimal_deployment(
    model: ModelInput,
    hardware: HardwareInput = "a3",
    **kwargs: Any,
) -> dict[str, Any]:
    """Jointly optimize PD Disaggregated vs Colocated + MTP under Domain 1/2/3."""
    from mindie_sim.sweep.pd import solve_optimal_deployment as _solve_optimal

    model_arg: Any = str(model) if isinstance(model, Path) else (
        deepcopy(dict(model)) if isinstance(model, Mapping) else model
    )
    hw_arg: Any = (
        deepcopy(dict(hardware)) if isinstance(hardware, Mapping) else hardware
    )
    return _solve_optimal(model_arg, hw_arg, **kwargs)


def solve_deployment(
    model: ModelInput,
    hardware: HardwareInput = "a3",
    **kwargs: Any,
) -> dict[str, Any]:
    """One-shot agent entrypoint: capacity ledger, PD/colocated optimum, 4-track waterfall, roofline."""
    from mindie_sim.solve import solve_deployment as _solve_deployment

    model_arg: Any = str(model) if isinstance(model, Path) else (
        deepcopy(dict(model)) if isinstance(model, Mapping) else model
    )
    hw_arg: Any = (
        deepcopy(dict(hardware)) if isinstance(hardware, Mapping) else hardware
    )
    return _solve_deployment(model_arg, hw_arg, **kwargs)


def model_target_ceiling(
    model: ModelInput,
    hardware: HardwareInput = "a5",
    **kwargs: Any,
) -> dict[str, Any]:
    """First-class Modeling Mode entrypoint solving Before vs After Iso-SLA ceiling."""
    from mindie_sim.public.formula_sdk import model_target_ceiling as _model_target_ceiling

    model_arg: Any = str(model) if isinstance(model, Path) else (
        deepcopy(dict(model)) if isinstance(model, Mapping) else model
    )
    hw_arg: Any = (
        deepcopy(dict(hardware)) if isinstance(hardware, Mapping) else hardware
    )
    return _model_target_ceiling(model_arg, hw_arg, **kwargs)


def sweep_topology(
    case: CaseInput,
    request: SweepRequest,
    /,
    *,
    decode_case: CaseInput | None = None,
) -> TopologySweepResult:
    """Sweep parallel topologies or P/D disaggregation ratios."""
    parsed = _to_case(case)
    if str(getattr(request, "mode", "")).strip().lower() == "auto":
        from mindie_sim.prediction.results import TopologyPoint
        from mindie_sim.sweep.pd import solve_optimal_deployment as _solve_optimal

        total_dies = (
            int(request.device_counts[0])
            if request.device_counts else len(parsed.machine.dies)
        )
        prompt = int(request.sla.step.context or request.sla.step.tokens)
        output = int(request.sla.output_budget)
        ttft = parsed.slo.ttft_ms
        tpot = float(request.sla.latency_ms)
        solved = _solve_optimal(
            parsed.model.to_dict(),
            parsed.machine.to_dict(),
            num_devices=total_dies,
            prompt_tokens=prompt,
            output_tokens=output,
            ttft_sla_ms=ttft,
            tpot_sla_ms=tpot,
        )
        frontier: list[TopologyPoint] = []
        for row in solved.get("pareto_frontier", ()):
            par = row.get("prefill_parallel") or row.get("parallel") or {}
            d_par = row.get("decode_parallel") or {}
            frontier.append(TopologyPoint(
                device_count=int(row.get("used_dies", total_dies)),
                parallel=tuple((str(k), int(v)) for k, v in sorted(par.items()) if isinstance(v, int)),
                decode_parallel=tuple((str(k), int(v)) for k, v in sorted(d_par.items()) if isinstance(v, int)),
                span_ms=float(row.get("effective_tpot_ms", 0.0) or 0.0),
                throughput_tokens_s=float(row.get("sustainable_tps", 0.0) or 0.0),
                fits_memory=bool(row.get("feasible", False)),
            ))
        return TopologySweepResult(
            evaluated_count=len(solved.get("candidates", ())),
            frontier=tuple(frontier),
        )
    parsed_decode = _to_case(decode_case) if decode_case is not None else None
    return default_spine().sweep_topology(
        parsed, request, decode_case=parsed_decode
    )


def export_structure(
    case: CaseInput,
    request: StructureRequest,
    /,
) -> StructureExport:
    """Export canonical hashed structure sealed by ``graph_id`` without timing replay."""
    spine = default_spine()
    prepared = spine.lower(_to_case(case), LoweringRequest(step=request.step))
    return spine.project_structure(prepared, request)


__all__ = [
    "compose_case",
    "evaluate_service",
    "evaluate_step",
    "evaluate_three_tiers",
    "export_structure",
    "model_target_ceiling",
    "optimize_ladder",
    "solve_deployment",
    "solve_optimal_deployment",
    "solve_sla",
    "sweep_topology",
]
