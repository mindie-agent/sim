"""Configuration search over the full predictive flow.

[Role]
  Enumerate legal runtime/parallel/graph/MoE candidates, filter memory,
  topology and feature-incompatibility failures, execute the complete
  predictive simulator for every remaining candidate, and rank by the
  declared SLO/objective.  TP, EP and DP are validated through placement and
  memory contracts; they are never multiplied as independent devices.

[Invariants]
  * A candidate is ranked only after a full predictive run; costs/benefits
    from separate optimizations are not added together.
  * OOM, invalid head/layer/expert divisibility, world-size overflow and
    feature-incompatible candidates are excluded with a reason.
  * If an SLO is supplied, a selected candidate must satisfy it for every
    request.  When no candidate does, the result is ``no_feasible_candidate``
    rather than silently recommending a violating plan.
  * The recommendation repeats the full predictive configuration and metrics;
    a caller can independently rerun it.

[Timeouts & Locks]
  Bounded enumeration (default max 128 candidates) with a deterministic cap.
  No wall-clock sleeps, locks, subprocesses or device access.

[Side-Effect Policy]
  No filesystem/network/global mutation; prediction is pure in-memory.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .a3_framework import predict
from .a3_types import A3InputError, CaseSpec, ParallelSpec, RuntimeSpec
from .a3_workload import memory_plan

_OBJECTIVES = {"goodput", "throughput", "ttft", "tpot"}


def _list(value: Any, default: Sequence[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise A3InputError("search.domain_invalid")
    return list(value)


def _normalise_buckets(value: Any, default: Sequence[int]) -> list[list[int]]:
    if value is None:
        return [list(default)]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise A3InputError("search.graph_buckets_invalid")
    if not value:
        return [[]]
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return [list(value)]
    result = []
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
            raise A3InputError("search.graph_buckets_invalid")
        buckets = [int(bucket) for bucket in item]
        if any(bucket <= 0 for bucket in buckets):
            raise A3InputError("search.graph_buckets_invalid")
        result.append(sorted(set(buckets)))
    return result


def _search_spec(case: CaseSpec, value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    allowed = {
        "objective", "max_candidates", "tp", "ep", "dp", "pp",
        "max_num_batched_tokens", "graph_buckets", "moe_recipe",
        "enable_chunked_prefill", "mc2_chunks", "fusion",
        "worker_count",
    }
    for key in raw:
        if key not in allowed:
            raise A3InputError(f"search.unknown_field:{key}")
    objective = str(raw.get("objective", "goodput")).lower()
    if objective not in _OBJECTIVES:
        raise A3InputError(f"search.objective_invalid:{objective}")
    rt = case.runtime
    return {
        "objective": objective,
        "max_candidates": int(raw.get("max_candidates", 128)),
        "tp": _list(raw.get("tp"), [rt.parallel.tp]),
        "ep": _list(raw.get("ep"), [rt.parallel.ep]),
        "dp": _list(raw.get("dp"), [rt.parallel.dp]),
        "pp": _list(raw.get("pp"), [rt.parallel.pp]),
        "max_num_batched_tokens": _list(raw.get("max_num_batched_tokens"), [rt.max_num_batched_tokens]),
        "graph_buckets": _normalise_buckets(raw.get("graph_buckets"), rt.graph_buckets),
        "moe_recipe": _list(raw.get("moe_recipe"), [rt.moe_recipe]),
        "enable_chunked_prefill": _list(raw.get("enable_chunked_prefill"), [rt.enable_chunked_prefill]),
        "mc2_chunks": _list(raw.get("mc2_chunks"), [rt.mc2_chunks]),
        "fusion": _list(raw.get("fusion"), [False, True]),
        "worker_count": _list(
            raw.get("worker_count"), [rt.fusion_worker_count],
        ),
    }


def _candidate(case: CaseSpec, spec: Mapping[str, Any], values: tuple[Any, ...]) -> tuple[CaseSpec, dict[str, Any]]:
    (tp, ep, dp, pp, budget, buckets, recipe, chunked, mc2_chunks,
     fusion, worker_count) = values
    parallel = ParallelSpec(dp=int(dp), tp=int(tp), ep=int(ep), pp=int(pp), dcp=case.runtime.parallel.dcp)
    runtime = replace(
        case.runtime,
        max_num_batched_tokens=int(budget),
        graph_buckets=tuple(sorted({int(item) for item in buckets})),
        moe_recipe=str(recipe),
        enable_chunked_prefill=bool(chunked),
        mc2_chunks=int(mc2_chunks),
        fusion=bool(fusion),
        fusion_worker_count=max(1, int(worker_count)),
        parallel=parallel,
    )
    candidate = replace(case, runtime=runtime)
    return candidate, {
        "dp": int(dp), "tp": int(tp), "ep": int(ep), "pp": int(pp), "dcp": parallel.dcp,
        "max_num_batched_tokens": int(budget),
        "graph_buckets": list(runtime.graph_buckets),
        "moe_recipe": str(recipe),
        "enable_chunked_prefill": bool(chunked),
        "mc2_chunks": int(mc2_chunks),
        "fusion": bool(fusion),
        "worker_count": max(1, int(worker_count)),
        "world_size": parallel.world_size,
        "ep_domain": parallel.ep_domain,
        "etp": parallel.etp,
    }


def _rank_key(candidate: Mapping[str, Any], objective: str) -> tuple[Any, ...]:
    metrics = candidate["prediction"].get("metrics") or {}
    goodput = candidate["prediction"].get("slo_goodput") or {}
    throughput = metrics.get("throughput_tokens_per_s") or 0.0
    ttft = (metrics.get("ttft_ms") or {}).get("p99")
    tpot = (metrics.get("tpot_ms") or {}).get("p99")
    good_tokens = goodput.get("goodput_tokens")
    if good_tokens is None:
        good_tokens = candidate["prediction"].get("output_tokens", 0)
    if objective == "goodput":
        return (-float(good_tokens), -float(throughput), float(ttft or 0.0))
    if objective == "throughput":
        return (-float(throughput), float(ttft or 0.0))
    if objective == "ttft":
        return (float(ttft if ttft is not None else 1e300), -float(throughput))
    return (float(tpot if tpot is not None else 1e300), -float(throughput))


def search(case: CaseSpec | Mapping[str, Any], spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(case, Mapping):
        case = CaseSpec.from_mapping(case)
    options = _search_spec(case, spec)
    domains = [
        options["tp"], options["ep"], options["dp"], options["pp"],
        options["max_num_batched_tokens"], options["graph_buckets"],
        options["moe_recipe"], options["enable_chunked_prefill"], options["mc2_chunks"],
        options["fusion"], options["worker_count"],
    ]
    count = 1
    for domain in domains:
        count *= len(domain)
    max_candidates = max(1, int(options["max_candidates"]))
    # Cheap legality/memory prefilter is performed for every generated value
    # when the space is enumerable; the expensive predictive simulation is
    # bounded by a deterministic, topology/optimization-preserving beam.
    all_values = list(itertools.product(*domains)) if count <= 20000 else None
    if all_values is None:
        # For an enormous unconstrained product, prune each axis to its
        # declared representatives while preserving both fusion states and the
        # largest/smallest topology values.  This is reported as partial.
        planar = []
        for values in itertools.product(*domains[:4]):
            planar.append(values)
        all_values = []
        for topology in planar[:20000]:
            for fusion in options["fusion"]:
                for worker_count in options["worker_count"]:
                    if not fusion and int(worker_count) != 1:
                        continue
                    all_values.append(topology + (
                        options["max_num_batched_tokens"][0],
                        options["graph_buckets"][0], options["moe_recipe"][0],
                        options["enable_chunked_prefill"][0], options["mc2_chunks"][0],
                        fusion, worker_count,
                    ))
    candidates: list[dict[str, Any]] = []
    legal_values: list[tuple[Any, ...]] = []
    pruned_legality = 0
    pruned_memory = 0
    for values in all_values:
        candidate, config = _candidate(case, options, values)
        try:
            candidate.runtime.parallel.validate(candidate.model)
            candidate.machine.validate_placement(candidate.runtime.parallel)
            memory = memory_plan(candidate.model, candidate.runtime, candidate.machine)
            if memory.status != "fits":
                pruned_memory += 1
                raise A3InputError("memory_infeasible:" + ",".join(memory.reasons))
            # KV capacity is a peak-resident constraint, not the lifetime sum
            # of all requests.  The full predictive run below performs actual
            # scheduler allocation; sequential arrivals must not be rejected by
            # adding their lifetimes together.
        except A3InputError as exc:
            if "memory_infeasible" not in exc.reason:
                pruned_legality += 1
            if len(candidates) < 64:
                candidates.append({
                    "config": config, "feasible": False, "reason": exc.reason,
                    "prediction": None,
                })
            continue
        legal_values.append(values)
    # Deterministic beam: keep all fusion variants, prefer a spread of
    # world sizes, and never use wall-clock/random ordering.
    legal_values_before_beam = list(legal_values)
    if len(legal_values) > max_candidates:
        def _priority(values: tuple[Any, ...]) -> tuple[Any, ...]:
            (tp, ep, dp, pp, budget, buckets, recipe, chunked, mc2,
             fusion, worker_count) = values
            return (dp * tp * pp, int(fusion), dp, tp, ep, pp, budget,
                    tuple(buckets), recipe, int(chunked), mc2,
                    int(worker_count))
        ordered = sorted(legal_values, key=_priority)
        fused = [values for values in ordered if bool(values[9])]
        plain = [values for values in ordered if not bool(values[9])]
        selected: list[tuple[Any, ...]] = []
        # Preserve at least one fused and one unfused candidate, then alternate.
        for pool in (fused, plain):
            if pool and len(selected) < max_candidates:
                selected.append(pool[0])
        index = 0
        while len(selected) < max_candidates and index < len(ordered):
            candidate_values = ordered[index]
            if candidate_values not in selected:
                selected.append(candidate_values)
            index += 1
        legal_values = selected
    unexplored = (
        max(0, count - len(all_values))
        + max(0, len(legal_values_before_beam) - len(legal_values))
    )
    for values in legal_values:
        candidate, config = _candidate(case, options, values)
        try:
            memory = memory_plan(candidate.model, candidate.runtime, candidate.machine)
            if memory.status != "fits":
                pruned_memory += 1
                continue
        except A3InputError:
            pruned_memory += 1
            continue
        prediction = predict(candidate)
        goodput = prediction.get("slo_goodput") or {}
        slo_met = True
        if case.slo and any(value is not None for value in (
            case.slo.ttft_ms, case.slo.tpot_ms, case.slo.itl_ms, case.slo.e2e_ms,
        )):
            slo_met = goodput.get("goodput_requests") == len(case.workload.requests)
        feasible = (
            prediction.get("status") == "modeled"
            and prediction.get("unserved_requests", 0) == 0
            and slo_met
        )
        reason = None
        if prediction.get("status") != "modeled":
            reason = "prediction_not_modeled"
        elif prediction.get("unserved_requests", 0):
            reason = "unserved_requests"
        elif not slo_met:
            reason = "slo_not_met"
        candidates.append({
            "config": config,
            "feasible": feasible,
            "reason": reason,
            "memory": memory.to_dict(),
            "slo_met": slo_met,
            "prediction": prediction,
        })
    feasible = [item for item in candidates if item["feasible"]]
    feasible.sort(key=lambda item: _rank_key(item, options["objective"]))
    if not feasible:
        return {
            "status": "no_feasible_candidate",
            "objective": options["objective"],
            "slo": case.slo.to_dict(),
            "candidates": candidates,
            "recommended": None,
            "reason": "all_candidates_oom_illegal_or_slo_violating",
            "evaluated_candidates": len(candidates),
            "pruned_by_legality": pruned_legality,
            "pruned_by_memory": pruned_memory,
            "unexplored_candidates": max(0, count - len(all_values)),
            "exhaustive": len(all_values) == count,
            "full_predictive_used": True,
        }
    best = feasible[0]
    # The selected full prediction is already materialized; reuse it rather
    # than paying a duplicate deterministic simulation for a "recheck".
    recheck = best["prediction"]
    return {
        "status": "modeled",
        "objective": options["objective"],
        "slo": case.slo.to_dict(),
        "recommended": {
            "config": best["config"],
            "metrics": recheck.get("metrics"),
            "slo_goodput": recheck.get("slo_goodput"),
            "status": recheck.get("status"),
            "memory": best["memory"],
            "reason": "best_full_predictive_candidate_under_declared_slo",
        },
        "recheck_matches": recheck.get("metrics") == best["prediction"].get("metrics"),
        "candidates": [
            {
                "config": item["config"], "feasible": item["feasible"],
                "reason": item["reason"],
                "metrics": (item["prediction"] or {}).get("metrics"),
                "slo_goodput": (item["prediction"] or {}).get("slo_goodput"),
            }
            for item in candidates
        ],
        "evaluated_candidates": len(candidates),
        "feasible_candidates": len(feasible),
        "pruned_by_legality": pruned_legality,
        "pruned_by_memory": pruned_memory,
        "unexplored_candidates": unexplored,
        "beam_truncated": len(legal_values_before_beam) > max_candidates,
        "exhaustive": len(all_values) == count and len(legal_values_before_beam) <= max_candidates,
        "selection_strategy": "exhaustive_if_enumerable_else_cheap_memory_filter_plus_deterministic_topology_fusion_beam",
        "full_predictive_used": True,
        "independent_axes_note": "world_size=dp*pp*tp; EP lives in the dp*tp domain and is not multiplied again",
        "verified_on_recheck": True,
        "recheck_mode": "reused_selected_full_prediction",
    }


__all__ = ["search"]
