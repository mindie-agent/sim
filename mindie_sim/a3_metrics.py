"""Request/replica metric aggregation and SLO goodput."""
from __future__ import annotations

import math
from typing import Any, Sequence

from .a3_types import SloSpec


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)





def _merge_replicas(replica_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    requests = [item for result in replica_results for item in result["requests"]]
    traces = [item for result in replica_results for item in result["traces"]]
    served = [item for item in requests if item["status"] == "served"]
    ttft = [item["ttft_ms"] for item in served if item["ttft_ms"] is not None]
    e2e = [item["e2e_ms"] for item in served if item["e2e_ms"] is not None]
    tpot = [item["tpot_ms"] for item in served if item["tpot_ms"] is not None]
    itl = [value for item in served for value in item["itl_ms"]]
    makespan = max((item["finish_ms"] or 0.0 for item in served), default=0.0)
    arrivals = [item["arrival_ms"] for item in requests]
    horizon = max(1e-9, makespan - min(arrivals)) if arrivals else 1e-9
    output_tokens = sum(item["generated_tokens"] for item in served)
    return {
        "requests": requests,
        "traces": traces,
        "makespan_ms": makespan,
        "horizon_ms": horizon,
        "served_requests": len(served),
        "unserved_requests": len(requests) - len(served),
        "output_tokens": output_tokens,
        "metrics": {
            "ttft_ms": {
                "mean": (sum(ttft) / len(ttft)) if ttft else None,
                "p50": _percentile(ttft, 0.50), "p90": _percentile(ttft, 0.90),
                "p99": _percentile(ttft, 0.99),
            },
            "itl_ms": {
                "mean": (sum(itl) / len(itl)) if itl else None,
                "p50": _percentile(itl, 0.50), "p90": _percentile(itl, 0.90),
                "p99": _percentile(itl, 0.99), "max": max(itl) if itl else None,
            },
            "tpot_ms": {
                "mean": (sum(tpot) / len(tpot)) if tpot else None,
                "p50": _percentile(tpot, 0.50), "p90": _percentile(tpot, 0.90),
                "p99": _percentile(tpot, 0.99),
            },
            "e2e_ms": {
                "mean": (sum(e2e) / len(e2e)) if e2e else None,
                "p50": _percentile(e2e, 0.50), "p90": _percentile(e2e, 0.90),
                "p99": _percentile(e2e, 0.99),
            },
            "throughput_tokens_per_s": output_tokens / (horizon / 1000.0),
            "throughput_requests_per_s": len(served) / (horizon / 1000.0),
        },
    }


def _slo_goodput(requests: Sequence[dict[str, Any]], slo: SloSpec) -> dict[str, Any]:
    if all(value is None for value in (slo.ttft_ms, slo.tpot_ms, slo.itl_ms, slo.e2e_ms)):
        return {
            "slo": slo.to_dict(), "goodput_requests": None, "goodput_tokens": None,
            "goodput_ratio": None, "reason": "no_slo_declared",
        }
    good_requests = 0
    good_tokens = 0
    failures: dict[str, int] = {}
    for item in requests:
        if item["status"] != "served":
            failures["unserved"] = failures.get("unserved", 0) + 1
            continue
        reasons = []
        if slo.ttft_ms is not None and (item["ttft_ms"] is None or item["ttft_ms"] > slo.ttft_ms):
            reasons.append("ttft")
        if slo.tpot_ms is not None and (item["tpot_ms"] is None or item["tpot_ms"] > slo.tpot_ms):
            reasons.append("tpot")
        if slo.itl_ms is not None and any(value > slo.itl_ms for value in item["itl_ms"]):
            reasons.append("itl")
        if slo.e2e_ms is not None and (item["e2e_ms"] is None or item["e2e_ms"] > slo.e2e_ms):
            reasons.append("e2e")
        if reasons:
            for reason in reasons:
                failures[reason] = failures.get(reason, 0) + 1
            continue
        good_requests += 1
        good_tokens += item["generated_tokens"]
    return {
        "slo": slo.to_dict(),
        "goodput_requests": good_requests,
        "goodput_tokens": good_tokens,
        "goodput_ratio": good_requests / len(requests) if requests else None,
        "slo_failures": failures,
    }


__all__ = ["_percentile", "_merge_replicas", "_slo_goodput"]
