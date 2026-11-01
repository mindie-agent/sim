"""Closed-world counterfactual advisor for MindIE simulation Cases.

[Role]
  Public facade for ``load_case_from_profile`` and ``advise_case``: ingest
  measured profiling artifacts, evaluate the baseline causal simulator, expose
  four-dimensional attribution and calibration residuals, then rank guarded
  counterfactual runtime mutations.

[Invariants]
  * Every counterfactual is validated and evaluated by the causal engine; no
    model-name multiplier or observed-latency leakage enters prediction.
  * Caller Cases, mappings, and profile records are never mutated in place.
  * Missing observations stay ``None``; emitted report floats are finite.

[Timeouts & Locks]
  Bounded local profile reads and in-memory graph evaluation; no network, NPU,
  subprocess, sleep, or file locks.

[Side-Effect Policy]
  Pure in-memory report construction; the CLI writes only an explicit ``-o``
  path through the shared JSON emitter.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .advise_catalog import _evaluate_intervention, candidate_specs, rank_key
from .advise_core import _baseline, _calibration, _config_for
from .prediction.structure import parse_case
from .advise_util import _has_observed_measurement
from .profile_ingest import _merge_observed_profiles, load_case_from_profile
from .schema.case import Case as CaseSpec

ADVISE_SCHEMA = "mindie-sim.advise/1"


def _merge_observed(case_dict: dict[str, Any], observed: Mapping[str, Any]) -> None:
    if not observed or not _has_observed_measurement(observed):
        return
    runtime = case_dict.setdefault("runtime", {})
    raw_meta = runtime.get("metadata") or {}
    meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else dict(raw_meta)
    merged = _merge_observed_profiles(meta.get("observed_profile"), observed)
    if merged:
        meta["observed_profile"] = merged
    runtime["metadata"] = meta


def advise_case(
    case_or_path: CaseSpec | Mapping[str, Any] | str | Path | None = None,
    *,
    from_profile: str | Path | Mapping[str, Any] | None = None,
    phase: str = "decode",
    scope: str = "body",
    context: int | None = None,
    sequences: int | None = None,
    tokens: int | None = None,
    sla_tpot_ms: float | None = None,
    sla_ttft_ms: float | None = None,
    max_interventions: int = 10,
    **profile_overrides: Any,
) -> dict[str, Any]:
    """Evaluate a baseline Case and rank closed-world counterfactual interventions."""
    if isinstance(max_interventions, bool) or not isinstance(max_interventions, int) or max_interventions < 0:
        raise ValueError(f"max_interventions_must_be_nonnegative_int:{max_interventions}")
    if case_or_path is None and from_profile is None:
        raise ValueError("advise_requires_case_or_from_profile")
    profile_case = None
    if from_profile is not None:
        profile_case = load_case_from_profile(from_profile, **profile_overrides)
    if case_or_path is None:
        source = profile_case
    elif isinstance(case_or_path, (str, Path)):
        source = load_case_from_profile(case_or_path, **profile_overrides)
    elif isinstance(case_or_path, CaseSpec):
        source = (load_case_from_profile(case_or_path, **profile_overrides)
                  if profile_overrides else case_or_path.to_dict())
    elif isinstance(case_or_path, Mapping):
        if {"machine", "model", "workload"} <= set(case_or_path) and not profile_overrides:
            source = deepcopy(dict(case_or_path))
        else:
            source = load_case_from_profile(case_or_path, **profile_overrides)
    else:
        raise ValueError(f"advise_invalid_case_source:{type(case_or_path).__name__}")
    if source is None:
        raise ValueError("advise_requires_case_or_from_profile")
    case_dict = deepcopy(dict(source)) if isinstance(source, Mapping) else dict(source)
    if profile_case is not None and source is not profile_case:
        raw_meta = dict(profile_case.get("runtime", {}).get("metadata") or ())
        observed = raw_meta.get("observed_profile")
        if isinstance(observed, Mapping):
            _merge_observed(case_dict, observed)
        prof_prov = raw_meta.get("profiler_provenance")
        if isinstance(prof_prov, Mapping) and prof_prov:
            rt_meta = dict(case_dict.setdefault("runtime", {}).get("metadata") or {})
            existing_prov = rt_meta.get("profiler_provenance")
            rt_meta["profiler_provenance"] = {
                **(dict(existing_prov) if isinstance(existing_prov, Mapping) else {}),
                **dict(prof_prov),
            }
            case_dict["runtime"]["metadata"] = rt_meta
    case = parse_case(case_dict)
    cfg = _config_for(case, phase, scope, context, sequences, tokens, sla_tpot_ms, sla_ttft_ms)
    baseline = _baseline(case, cfg)
    raw_meta = dict(case.runtime.metadata)
    observed = raw_meta.get("observed_profile")
    observed_vs_predicted = _calibration(
        baseline, dict(observed), cfg=cfg, dp=int(case.runtime.parallel.dp), case=case,
    ) if isinstance(observed, Mapping) and observed else {
        "available": False, "reason": "no_observed_profile_metadata"}
    inventory = candidate_specs(case, baseline, cfg, case_dict)[:max_interventions]
    recommendations = [_evaluate_intervention(spec, case_dict, baseline, case, cfg)
                       for spec in inventory]
    recommendations.sort(key=rank_key)
    top = next((record for record in recommendations
                if record.get("applicable_and_recommended")), None)
    if top is None and recommendations:
        top = recommendations[0]
    public_baseline = {key: baseline[key] for key in (
        "step_duration_ms", "operator_sum_ms", "tpot_p50_ms", "ttft_p50_ms",
        "sla_goodput_tok_s", "b_vram_max_sync", "b_vram_max_async", "dominant_bottleneck")}
    provenance = raw_meta.get("profiler_provenance")
    return {
        "schema": ADVISE_SCHEMA,
        "status": "completed",
        "geometry": {"phase": cfg.phase, "scope": cfg.scope, "context": cfg.context,
                     "sequences": cfg.sequences, "tokens": cfg.tokens,
                     "output_tokens": cfg.output_tokens, "prefix_tokens": cfg.prefix_tokens,
                     "sla_tpot_ms": cfg.sla_tpot_ms, "sla_ttft_ms": cfg.sla_ttft_ms},
        "case": {"model": case.model.name,
                 "hardware": case.machine.sku or case.machine.name,
                 "parallel": {"tp": case.runtime.parallel.tp, "dp": case.runtime.parallel.dp,
                              "ep": case.runtime.parallel.ep, "pp": case.runtime.parallel.pp}},
        "baseline": public_baseline,
        "attribution_4d": baseline["attribution"],
        "observed_vs_predicted": observed_vs_predicted,
        "recommendations": recommendations,
        "top_recommendation": top,
        "profile_provenance": dict(provenance) if isinstance(provenance, Mapping) else {},
        "max_interventions": max_interventions,
        "limitations": [
            "counterfactual timing is causal-simulation output, not a measured hardware claim",
            "SLA goodput uses VRAM capacity and step latency; the exact solver is capped for large contexts",
        ],
    }


__all__ = ["advise_case", "load_case_from_profile"]
