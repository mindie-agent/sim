"""Metrics computed exclusively from request commit events.

[Role] Aggregate latency, throughput and request-level SLO outcomes.
[Invariants] Queue time is part of TTFT. Tokens credited by speculative batches
  share a commit time. Unserved requests remain in SLO denominators.
[Timeouts & Locks] O(requests) storage, no token history, I/O or locks.
[Side-Effect Policy] Does not change request progress or model parameters.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def distribution(values: Sequence[float]) -> dict:
    ordered = sorted(values)

    def percentile(fraction: float):
        if not ordered:
            return None
        index = fraction * (len(ordered) - 1)
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered), "mean": sum(ordered) / len(ordered) if ordered else None,
        "min": ordered[0] if ordered else None,
        "p50": percentile(.5), "p90": percentile(.9), "p99": percentile(.99),
        "max": ordered[-1] if ordered else None,
    }


def summarize(rows: Sequence[Mapping], slo, *, now_ms=None) -> dict:
    served = [row for row in rows if row["status"] == "served"]
    start = min((row["arrival_ms"] for row in rows), default=0)
    finish = max((row.get("last_token_abs_ms") or row.get("finish_ms") or start for row in rows), default=start)
    finish = max(finish, now_ms if now_ms is not None else finish)
    seconds = (finish - start) / 1000
    tokens = sum(row["generated_tokens"] for row in rows)
    configured = {key: value for key, value in slo.to_dict().items() if value is not None}
    passing, failures = [], {}
    for row in rows:
        reasons = [] if row["status"] == "served" else ["unserved"]
        for key, limit in configured.items():
            value = row["itl"]["max_ms"] if key == "itl_ms" else row[key]
            # A one-token response has no inter-token interval.
            if value is None and key in {"itl_ms", "tpot_ms"} and row["output_tokens"] == 1:
                continue
            if value is None or value > limit:
                reasons.append(key)
        if not reasons:
            passing.append(row)
        else:
            failures[row["request_id"]] = reasons
    return {
        **{key: distribution([row[key] for row in rows if row[key] is not None])
           for key in ("ttft_ms", "tpot_ms", "e2e_ms")},
        "makespan_ms": finish - start, "window_start_ms": start, "window_end_ms": finish,
        "total_requests": len(rows), "served_requests": len(served),
        "unserved_requests": len(rows) - len(served),
        "output_tokens": tokens, "output_tps": tokens / seconds if seconds > 0 else None,
        "request_tps": len(served) / seconds if seconds > 0 else None,
        "slo": {
            "limits": configured, "passing_requests": len(passing) if configured else None,
            "passing_fraction": len(passing) / len(rows) if configured and rows else None,
            "goodput_tps": (sum(row["generated_tokens"] for row in passing) / seconds
                            if configured and seconds > 0 else None),
            "failures": failures if configured else {},
        },
    }
