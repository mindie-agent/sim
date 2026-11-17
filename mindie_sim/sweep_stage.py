"""Finalist promotion and killable evaluation stage for bounded sweeps.

[Role] Retain the global top-K screening candidates and evaluate only those
  against the complete graph/SLA solver.
[Invariants] A screen score is never an SLA verdict; the stage writes each
  completed row before yielding it and never mutates a candidate record.
[Timeouts & Locks] One cooperative deadline per finalist; an explicitly
  requested ``isolate_workers`` run is bounded by a killable child process.
[Side-Effect Policy] Appends only to the caller's checkpoint stream.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from mindie_sim.prediction.evaluation import ACTIVE_DEADLINE_MONO
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep_checkpoint import STAGE_FULL, write_checkpoint_row
from mindie_sim.sweep_engine import (
    CANDIDATE_ID_KEY, FULL_STAGE, select_finalists, set_deadline, static_row,
    timeout_row,
)


PARETO_EPS = 1e-12
FINALIST_COMMON_KEYS = (
    CANDIDATE_ID_KEY, "parallel", "parallel_spec", "num_devices",
    "ranks_per_node",
)


def retain_finalist(sink: list, item: Mapping[str, Any], top_k: int) -> list:
    """Keep the best ``top_k`` screen scores, independent of visit order."""
    if item.get("operator_sum_screen_ms") is None:
        return sink
    sink.append(item)
    sink.sort(key=lambda row: (float(row["operator_sum_screen_ms"]),
                               str(row.get(CANDIDATE_ID_KEY, ""))))
    del sink[int(top_k):]
    return sink


def pareto_screen(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legacy B1 Pareto promotion used when no global top-K is requested."""
    candidates = [row for row in rows if row.get("b1_sla_pass")]
    front: list[dict[str, Any]] = []
    for row in candidates:
        vector = (
            float(row.get("tps_per_die", 0.0) or 0.0),
            float(row.get("upper_tps_per_die", 0.0) or 0.0),
            float(row.get("max_concurrency_bound", 0.0) or 0.0),
            -float(row.get("step_ms_at_b1", 0.0) or 0.0),
        )
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_vector = (
                float(other.get("tps_per_die", 0.0) or 0.0),
                float(other.get("upper_tps_per_die", 0.0) or 0.0),
                float(other.get("max_concurrency_bound", 0.0) or 0.0),
                -float(other.get("step_ms_at_b1", 0.0) or 0.0),
            )
            if all(a >= b - PARETO_EPS for a, b in zip(other_vector, vector)) \
                    and any(a > b + PARETO_EPS
                            for a, b in zip(other_vector, vector)):
                dominated = True
                break
        if not dominated:
            row["b1_pareto_optimal"] = True
            front.append(row)
    return front



def screen_row_meta(row: dict[str, Any], cache: Any,
                    meta: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the sweep metadata and derived-hardware fields to one row."""
    merged = dict(meta)
    row.update(merged.pop("derived_row_fields", {}))
    merged["cache_stats"] = cache.stats()
    row["sweep_meta"] = merged
    return row


def isolated_row(parsed, cand_p, cand_devices, part, common, phase, scope,
                 context, q_tokens, emit_tokens, sla_ms, output_budget,
                 max_sequences_per_dp, *, stage, timeout_s,
                 deadline_monotonic):
    """Evaluate one candidate behind a killable process boundary."""
    from mindie_sim._sweep_worker import evaluate_sweep_job

    budgets = [float(timeout_s)]
    if deadline_monotonic is not None:
        budgets.append(max(0.0, float(deadline_monotonic) - time.monotonic()))
    job = {
        "_case": parsed,
        "parallel_spec": cand_p.to_dict(),
        "num_devices": int(cand_devices),
        "part": tuple(part or ()),
        "common": dict(common),
        "probe": {
            "phase": phase, "scope": scope, "context": int(context),
            "q_tokens": int(q_tokens), "emit_tokens": float(emit_tokens),
            "sla_ms": float(sla_ms), "output_budget": int(output_budget),
            "max_sequences_per_dp": int(max_sequences_per_dp),
        },
    }
    return evaluate_sweep_job(job, stage, min(budgets))


def evaluate_finalists(items, *, finalist_top_k, cache, stream, full_row_fn,
                       progress_callback, isolate_workers, candidate_timeout_s,
                       deadline_monotonic, phase, scope, context, q_tokens,
                       emit_tokens, sla_ms, output_budget,
                       max_sequences_per_dp, meta=None):
    """Evaluate the retained finalists and stream their complete rows."""
    for item in select_finalists(items, finalist_top_k):
        common = {key: item[key] for key in FINALIST_COMMON_KEYS}
        cand_p = ParallelSpec.from_mapping(item["parallel_spec"])
        item_meta = item.get("meta") or meta or {}
        started = time.monotonic()
        token = set_deadline(candidate_timeout_s, deadline_monotonic)
        try:
            if isolate_workers and candidate_timeout_s is not None:
                row = isolated_row(
                    item["parsed"], cand_p, int(item["num_devices"]),
                    item.get("part", ()), common, phase, scope, context,
                    q_tokens, emit_tokens, sla_ms, output_budget,
                    max_sequences_per_dp, stage=STAGE_FULL,
                    timeout_s=candidate_timeout_s,
                    deadline_monotonic=deadline_monotonic)
            else:
                row = full_row_fn(
                    item["parsed"], item["cand_case"], cand_p, common=common,
                    ctx=item["ctx"], phase=phase, scope=scope, context=context,
                    q_tokens=q_tokens, emit_tokens=emit_tokens, sla_ms=sla_ms,
                    output_budget=output_budget,
                    max_sequences_per_dp=max_sequences_per_dp,
                    b_vram_max=int(item["b_vram_max"]),
                    static={"weight_bytes": item.get("weight_bytes_per_die"),
                            "kv_bytes_per_token":
                                item.get("kv_bytes_per_token_per_die")},
                    screening=True)
            screen_row_meta(row, cache, item_meta)
            write_checkpoint_row(stream, row)
            if progress_callback is not None:
                progress_callback(row)
            yield row
        except TimeoutError:
            if (deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic):
                raise
            if candidate_timeout_s is None:
                raise
            row = timeout_row(
                common, elapsed_s=time.monotonic() - started,
                candidate_timeout_s=candidate_timeout_s)
            screen_row_meta(row, cache, item_meta)
            write_checkpoint_row(stream, row)
            if progress_callback is not None:
                progress_callback(row)
            yield row
        except (ModelingError, ValueError) as exc:
            row = static_row(common, reason=str(exc), kind="topology",
                             elapsed_ms=0.0)
            row["evaluation_stage"] = FULL_STAGE
            screen_row_meta(row, cache, item_meta)
            write_checkpoint_row(stream, row)
            if progress_callback is not None:
                progress_callback(row)
            yield row
        finally:
            ACTIVE_DEADLINE_MONO.reset(token)


__all__ = [
    "evaluate_finalists", "isolated_row", "pareto_screen", "retain_finalist",
    "screen_row_meta",
]
