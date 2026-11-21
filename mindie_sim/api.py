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
from mindie_sim.prediction.primitives import PrimitiveStore
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
    streams: tuple[StreamPlan, ...] = (),
) -> StepEvaluation:
    """Lower and evaluate one fixed step on ``runtime.Engine``."""
    spine = default_spine()
    prepared = spine.lower(
        _to_case(case), LoweringRequest(step=request), streams=streams
    )
    return spine.run_step(prepared, replay, primitives=primitives)


def evaluate_three_tiers(
    case: CaseInput,
    request: StepRequest,
    /,
    *,
    replay: ReplayOptions = ReplayOptions(),
    primitives: PrimitiveStore | None = None,
) -> ThreeTierEvaluation:
    """Evaluate ``unmerge``, ``mainline``, and constructive ``should_achieve`` on ``runtime.Engine``."""
    return default_spine().evaluate_three_tiers(
        _to_case(case),
        request,
        replay=replay,
        primitives=primitives,
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
    )


def evaluate_service(
    case: CaseInput,
    /,
    *,
    replay: ReplayOptions = ReplayOptions(),
    primitives: PrimitiveStore | None = None,
) -> ServiceEvaluation:
    """Evaluate a complete serving workload through ``PredictionSpine``."""
    return default_spine().run_service(
        _to_case(case), replay, primitives=primitives
    )


def solve_sla(case: CaseInput, request: SlaRequest, /) -> SlaEvaluation:
    """Solve SLA-constrained concurrency sizing."""
    return default_spine().solve_sla(_to_case(case), request)


def sweep_topology(
    case: CaseInput,
    request: SweepRequest,
    /,
    *,
    decode_case: CaseInput | None = None,
) -> TopologySweepResult:
    """Sweep parallel topologies or P/D disaggregation ratios."""
    parsed_decode = _to_case(decode_case) if decode_case is not None else None
    return default_spine().sweep_topology(
        _to_case(case), request, decode_case=parsed_decode
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
    "optimize_ladder",
    "solve_sla",
    "sweep_topology",
]
