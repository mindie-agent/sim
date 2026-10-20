"""Truthful result book, raw-clock summaries and atomic partial persistence.

[Role]
  Own the versioned results document for the A3 graph timing lane: requested
  conditions, per-trial raw observations, three independent clock channels,
  explicitly labelled per-invocation derivations, capture/allocation evidence,
  correctness and measurement-completeness status plus failure text.  Write the
  complete document atomically after each trial or terminal condition so partial
  evidence survives.

[Invariants]
  * Raw CPU-submit, device-event and host-wall clocks are retained per block.
    Per-invocation values live under ``derived`` and are never analytic time.
  * No empty-event subtraction, kernel-time claim or analytic prediction exists.
  * A condition is measurement-complete only when it is measured, every declared
    trial and sample exists, correctness passed and no timing anomaly is recorded.
    Completeness is not prediction, ingestion, fit or accuracy evidence.
  * Failed/pending conditions keep exact errors; no retry is represented.
  * Requested condition counts stay separate from terminal/measured counts; an
    empty or unexecuted document is never completed.
  * Atomic replacement prevents a partial write from truncating prior evidence.

[Timeouts & Locks]
  Serial in-process bookkeeping and one atomic file replacement at a time.  No
  threads or locks; device budgets are enforced by the runtime.

[Side-Effect Policy]
  Writes only the caller-supplied results path (via a same-directory temporary
  file) when ``write`` is called.  Importing this module performs no I/O.
"""
from __future__ import annotations

import copy
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a3_operator_graph_plan import N_KIND, OUTER_SUPERVISOR_SECONDS_MIN

CLOCK_KEYS = ("cpu_submit_ms", "device_interval_ms", "wall_complete_ms")
CLOCK_SEMANTICS = {
    "cpu_submit_ms": (
        "host perf_counter_ns interval surrounding only the eager call/replay "
        "loop; start/end event record, elapsed_time and synchronization are excluded"
    ),
    "device_interval_ms": (
        "precreated timing Event interval recorded start->end on the same explicit "
        "measurement stream; ordered device markers, not proven busy-kernel time"
    ),
    "wall_complete_ms": (
        "host perf_counter_ns interval from before the start event submission "
        "through the end-event completion wait"
    ),
}
DERIVED_LABEL = "derived"
DERIVED_KIND = N_KIND
TERMINAL_STATUSES = ("measured", "failed", "pending", "not_run")


class ResultBookError(RuntimeError):
    """Internal result-book contract violation."""


class OutputPathError(RuntimeError):
    """Unsafe results path or refusal to overwrite existing evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z")


def _clock(value: Any, key: str, anomalies: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultBookError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        anomalies.append(f"nonfinite_{key}")
        return None
    if number < 0.0:
        anomalies.append(f"negative_{key}")
    return number


def make_raw_observation(condition: Mapping[str, Any], *, trial_index: int,
                         sample_index: int, cpu_submit_ms: float,
                         device_interval_ms: float,
                         wall_complete_ms: float) -> dict[str, Any]:
    """Build one raw block observation with three independent clocks."""
    if isinstance(trial_index, bool) or not isinstance(trial_index, int) or trial_index < 0:
        raise ResultBookError("trial_index must be a non-negative integer")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ResultBookError("sample_index must be a non-negative integer")
    n = int(condition["block_invocations"])
    if n <= 0:
        raise ResultBookError("block_invocations must be positive")
    anomalies: list[str] = []
    cpu = _clock(cpu_submit_ms, "cpu_submit_ms", anomalies)
    device = _clock(device_interval_ms, "device_interval_ms", anomalies)
    wall = _clock(wall_complete_ms, "wall_complete_ms", anomalies)
    if cpu is not None and wall is not None and cpu > wall:
        anomalies.append("cpu_submit_exceeds_wall_complete")
    if device is not None and wall is not None and device > wall:
        anomalies.append("device_interval_exceeds_wall_complete")
    derived = {
        "label": DERIVED_LABEL,
        "denominator": n,
        "denominator_kind": DERIVED_KIND,
        "cpu_submit_per_invocation_ms": (cpu / n) if cpu is not None else None,
        "device_interval_per_invocation_ms": (
            device / n) if device is not None else None,
        "wall_complete_per_invocation_ms": (wall / n) if wall is not None else None,
        "warning": "derived block/N; not a direct kernel timer or analytic prediction",
    }
    return {
        "condition_id": condition["condition_id"],
        "case_id": condition["case_id"],
        "operator": condition["operator"],
        "mode": condition["mode"],
        "n": n,
        "trial_index": trial_index,
        "sample_index": sample_index,
        "block_invocations": n,
        "n_kind": condition.get("n_kind", DERIVED_KIND),
        "capture_invocations": int(condition["capture_invocations"]),
        "host_calls": int(condition["host_calls_per_block"]),
        "replay_calls": int(condition["replay_calls_per_block"]),
        "cpu_submit_ms": cpu,
        "device_interval_ms": device,
        "wall_complete_ms": wall,
        "derived": derived,
        "anomalies": anomalies,
        "clock_units": "ms",
        "clock_semantics": dict(CLOCK_SEMANTICS),
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "min_ms": None,
                "max_ms": None, "stdev_ms": None}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe raw samples without replacing them or inventing analytic time."""
    clock_stats: dict[str, Any] = {}
    for key in CLOCK_KEYS:
        values = [float(sample[key]) for sample in samples
                  if sample.get(key) is not None]
        clock_stats[key] = _stats(values)
    derived_stats: dict[str, Any] = {}
    derived_keys = (
        "cpu_submit_per_invocation_ms",
        "device_interval_per_invocation_ms",
        "wall_complete_per_invocation_ms",
    )
    for key in derived_keys:
        values = [
            float(sample["derived"][key])
            for sample in samples
            if sample.get("derived", {}).get(key) is not None
        ]
        derived_stats[key] = _stats(values)
    anomaly_codes = sorted({
        code for sample in samples for code in sample.get("anomalies", [])
    })
    anomaly_count = sum(1 for sample in samples if sample.get("anomalies"))
    block_invocations = sorted({int(sample["block_invocations"]) for sample in samples})
    return {
        "sample_count": len(samples),
        "block_invocations": block_invocations,
        "clocks": clock_stats,
        "derived_per_invocation": {
            "label": DERIVED_LABEL,
            "denominator_kind": DERIVED_KIND,
            "statistics": derived_stats,
            "warning": "block/N descriptions only; no analytic prediction",
        },
        "anomaly_codes": anomaly_codes,
        "anomaly_count": anomaly_count,
        "raw_observations_preserved_in": "trials[].samples[]",
        "interpretation": "descriptive_only_not_analytic_prediction",
    }


def _condition_record(condition: Mapping[str, Any],
                      protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": condition["condition_id"],
        "case_id": condition["case_id"],
        "operator": condition["operator"],
        "mode": condition["mode"],
        "n": condition["n"],
        "requested": True,
        "status": "requested",
        "measurement_complete": False,
        "reason": None,
        "condition_identity": {
            "case_id": condition["case_id"],
            "operator": condition["operator"],
            "dtype": condition["dtype"],
            "layout": condition["layout"],
            "shape": copy.deepcopy(condition["shape"]),
            "epsilon": condition["epsilon"],
            "data_seed": condition["data_seed"],
            "mode": condition["mode"],
            "mode_semantics": condition["mode_semantics"],
            "n": condition["n"],
            "n_kind": condition["n_kind"],
            "block_invocations": condition["block_invocations"],
            "capture_invocations": condition["capture_invocations"],
            "host_calls_per_block": condition["host_calls_per_block"],
            "replay_calls_per_block": condition["replay_calls_per_block"],
            "callable_source": condition["callable_source"],
            "output_policy": condition["output_policy"],
            "preallocated_output": condition["preallocated_output"],
            "declared_capture_output_bytes": condition["declared_capture_output_bytes"],
            "declared_capture_output_bytes_kind":
                condition["declared_capture_output_bytes_kind"],
        },
        "geometry": copy.deepcopy(condition["geometry"]),
        "reference_plan": copy.deepcopy(condition["reference_plan"]),
        "data_contract": copy.deepcopy(condition["data_contract"]),
        "timing_protocol": {
            "warmup_iterations": int(protocol["warmup_iterations"]),
            "trials": int(protocol["trials"]),
            "samples_per_condition": int(protocol["samples_per_condition"]),
            "counterbalance": "deterministic complete-permutation trials",
            "between_blocks": "explicit measurement-stream synchronize",
        },
        "execution": {
            "measurement_stream": None,
            "stream_policy": "one_explicit_measurement_stream_created_once",
            "inference_mode": True,
            "timing_events": {"status": "not_created", "enable_timing": True},
            "cache_policy": None,
            "timed_body_extra_work": False,
            "output_lifetime": "retain_last_only",
            "pool_policy": "graph_private_pool_no_sharing",
            "callable_source": condition["callable_source"],
        },
        "graph_capture": {
            "attempted": False,
            "status": "not_attempted",
            "timing_excluded": True,
            "no_eager_fallback": True,
        },
        "allocation": None,
        "correctness": None,
        "warmup": {
            "policy": "rewarm_exact_callable_or_replay_before_each_trial",
            "timed": False,
            "trials": [],
        },
        "debug_dump": {"status": "not_attempted"},
        "active_case_seconds": 0.0,
        "trials": [],
        "summary": None,
        "completeness": {
            "complete": False,
            "blockers": ["not_finished"],
            "requirements": {
                "status_measured": False,
                "correctness_passed": False,
                "all_trials_present": False,
                "all_samples_present": False,
                "no_timing_anomalies": False,
            },
        },
        "errors": [],
    }


class ResultBook:
    """Mutable result document with explicit requested/terminal accounting."""

    def __init__(self, plan: Mapping[str, Any], *, run_id: str,
                 results_path: str | Path, manifest_path: str,
                 manifest_sha256: str, device: str,
                 source_files: Mapping[str, str] | None = None) -> None:
        conditions = list(plan["conditions"])
        if not conditions:
            raise ResultBookError("a results document cannot be initialized "
                                  "without requested conditions")
        self.plan = plan
        self.run_id = run_id
        self.results_path = str(results_path)
        self._conditions: dict[str, dict[str, Any]] = {
            condition["condition_id"]: _condition_record(condition,
                                                          plan["protocol"])
            for condition in conditions
        }
        self._failed = False
        self._finalized = False
        self.doc: dict[str, Any] = {
            "kind": "a3_operator_graph_timing_results",
            "run_id": run_id,
            "manifest": {
                "manifest_id": plan["manifest_id"],
                "path": manifest_path,
                "sha256": manifest_sha256,
            },
            "worker": {
                "module": "mindie_sim.a3_operator_graph",
                "source_files_sha256": dict(source_files or {}),
                "device_owner": "external_coordinator_sim_audit",
                "lease_implemented_here": False,
            },
            "device": {"requested": device, "selected": None,
                       "logical_index": 0,
                       "selection": "coordinator_provided_explicit_npu_0"},
            "environment": None,
            "protocol": copy.deepcopy(plan["protocol"]),
            "budgets": copy.deepcopy(plan["budgets"]),
            "requested": {
                "conditions": list(plan["condition_ids"]),
                "condition_count": int(plan["counts"]["condition_count"]),
                "trials": int(plan["counts"]["trials"]),
                "samples_per_condition": int(
                    plan["counts"]["samples_per_condition"]),
                "planned_condition_executions": int(
                    plan["counts"]["planned_condition_executions"]),
                "planned_raw_blocks": int(plan["counts"]["planned_raw_blocks"]),
                "warmup_iterations": int(
                    plan["counts"]["warmup_iterations"]),
            },
            "progress": {},
            "schedule": copy.deepcopy(plan["schedule"]),
            "setup": {"status": "not_started"},
            "cleanup": {"status": "not_started"},
            "cleanup_ok": None,
            "persistence": {"ok": None, "errors": []},
            "persistence_ok": None,
            "conditions": list(self._conditions.values()),
            "failures": [],
            "stop_reason": None,
            "status": "running",
            "started_at_utc": utc_now(),
            "updated_at_utc": None,
            "finished_at_utc": None,
            "interpretation": {
                "clock_semantics": dict(CLOCK_SEMANTICS),
                "derived": (
                    "per-invocation values are derived block/N descriptions; N is "
                    "declared API invocations, not measured kernel launches"
                ),
                "no_analytic_prediction": True,
                "no_empty_event_subtraction": True,
                "native_outputs": (
                    "RMSNorm/SwiGLU use native default outputs only; no .out or "
                    "preallocated-output overload is claimed"
                ),
                "allocation_pool": (
                    "native allocation addresses and graph private-pool reuse are "
                    "recorded separately from eager allocator diagnostics; timing "
                    "differences are not labelled pure host dispatch cost"
                ),
                "copy": (
                    "one same preallocated destination is written repeatedly; this "
                    "is local contiguous copy, not a D2D/cache-scatter claim"
                ),
                "debug_dump": (
                    "an optional structural dump is not a measured task count and "
                    "its failure does not invalidate otherwise good replay timing"
                ),
                "outer_supervisor_seconds_min": OUTER_SUPERVISOR_SECONDS_MIN,
            },
        }
        self.refresh()

    def condition(self, condition_id: str) -> dict[str, Any]:
        try:
            return self._conditions[condition_id]
        except KeyError as exc:
            raise ResultBookError(f"unknown condition {condition_id!r}") from exc

    def set_environment(self, environment: Mapping[str, Any]) -> None:
        self.doc["environment"] = copy.deepcopy(dict(environment))
        selected = environment.get("device", {}).get("selected")
        if selected is not None:
            self.doc["device"]["selected"] = selected
        self.refresh()

    def set_setup(self, setup: Mapping[str, Any]) -> None:
        self.doc["setup"] = copy.deepcopy(dict(setup))
        self.refresh()

    def set_cleanup(self, cleanup: Mapping[str, Any]) -> None:
        self.doc["cleanup"] = copy.deepcopy(dict(cleanup))
        self.doc["cleanup_ok"] = self.doc["cleanup"].get("ok")
        self.refresh()

    def update(self, condition_id: str, **fields: Any) -> None:
        record = self.condition(condition_id)
        allowed = {
            "execution", "graph_capture", "allocation", "correctness", "warmup",
            "debug_dump", "active_case_seconds", "reason",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ResultBookError(f"unsupported condition fields {sorted(unknown)}")
        for key, value in fields.items():
            if key == "active_case_seconds":
                if value is None or float(value) < 0.0:
                    raise ResultBookError("active_case_seconds must be non-negative")
                record[key] = float(value)
            else:
                record[key] = copy.deepcopy(value)
        self.refresh()

    def record_trial(self, condition_id: str, trial: Mapping[str, Any]) -> None:
        record = self.condition(condition_id)
        if record["status"] in ("failed", "pending", "not_run"):
            raise ResultBookError(
                f"cannot record a trial for terminal condition {condition_id!r}")
        trial_index = trial.get("trial_index")
        if (isinstance(trial_index, bool) or not isinstance(trial_index, int)
                or trial_index != len(record["trials"])):
            raise ResultBookError(
                f"trial_index must equal next index {len(record['trials'])}")
        declared = int(self.plan["protocol"]["trials"])
        if trial_index >= declared:
            raise ResultBookError("trial_index exceeds declared trials")
        samples = trial.get("samples")
        if not isinstance(samples, list):
            raise ResultBookError("trial samples must be a list")
        sample_limit = int(self.plan["protocol"]["samples_per_condition"])
        if len(samples) > sample_limit:
            raise ResultBookError("trial has more samples than declared")
        status = trial.get("status")
        if status not in ("measured", "failed"):
            raise ResultBookError("trial status must be measured or failed")
        copied_trial = copy.deepcopy(dict(trial))
        copied_trial["run_id"] = self.run_id
        for sample in copied_trial.get("samples", []):
            if isinstance(sample, dict):
                sample["run_id"] = self.run_id
        record["trials"].append(copied_trial)
        warmup = copied_trial.get("warmup")
        if isinstance(warmup, dict):
            if not isinstance(record.get("warmup"), dict):
                record["warmup"] = {
                    "policy": "rewarm_exact_callable_or_replay_before_each_trial",
                    "timed": False,
                    "trials": [],
                }
            record["warmup"]["trials"].append(copy.deepcopy(warmup))
        record["summary"] = summarize_samples(
            [sample for existing in record["trials"]
             for sample in existing.get("samples", [])])
        all_trials = len(record["trials"]) == declared
        all_measured = all(existing.get("status") == "measured"
                           and len(existing.get("samples", [])) == sample_limit
                           for existing in record["trials"])
        if all_trials:
            if all_measured:
                record["status"] = "measured"
                record["reason"] = None
            else:
                record["status"] = "failed"
                record["reason"] = record["reason"] or "trial_incomplete_or_failed"
        else:
            record["status"] = "measuring"
        self.refresh()

    def fail(self, condition_id: str, phase: str, error: Mapping[str, Any], *,
             status: str = "failed", reason: str | None = None) -> None:
        if status not in ("failed", "pending"):
            raise ResultBookError("failure status must be failed or pending")
        record = self.condition(condition_id)
        if record["status"] == "measured":
            raise ResultBookError("refusing to overwrite a measured condition")
        record["status"] = status
        record["reason"] = reason or error.get("message") or phase
        entry = {
            "phase": phase,
            "at_utc": utc_now(),
            "error": copy.deepcopy(dict(error)),
            "retry": "not_attempted",
        }
        record["errors"].append(entry)
        self.doc["failures"].append({
            "condition_id": condition_id,
            "case_id": record["case_id"],
            "mode": record["mode"],
            "n": record["n"],
            **copy.deepcopy(entry),
        })
        self._failed = True
        self.refresh()

    def attach_recovery(self, condition_id: str, recovery: Mapping[str, Any]) -> None:
        """Attach an observed post-failure stream synchronization result."""
        record = self.condition(condition_id)
        if not record["errors"]:
            raise ResultBookError("no failure exists to attach recovery evidence")
        payload = copy.deepcopy(dict(recovery))
        record["errors"][-1]["post_failure_sync"] = payload
        for failure in reversed(self.doc["failures"]):
            if failure.get("condition_id") == condition_id:
                failure["post_failure_sync"] = payload
                break
        self.refresh()

    def mark_not_run(self, condition_id: str, reason: str) -> None:
        record = self.condition(condition_id)
        if record["status"] != "requested":
            raise ResultBookError(
                f"only requested conditions may be marked not_run, got "
                f"{record['status']!r}")
        record["status"] = "not_run"
        record["reason"] = reason
        self.refresh()

    def mark_setup_failure(self, error: Mapping[str, Any]) -> None:
        self.doc["setup"] = {
            "status": "failed",
            "error": copy.deepcopy(dict(error)),
            "at_utc": utc_now(),
        }
        self._failed = True
        for condition_id, record in self._conditions.items():
            if record["status"] == "requested":
                record["status"] = "not_run"
                record["reason"] = "setup_failed"
        self.refresh()

    def mark_remaining_not_run(self, reason: str) -> None:
        for record in self._conditions.values():
            if record["status"] == "requested":
                record["status"] = "not_run"
                record["reason"] = reason
        self.refresh()

    def _completeness(self, record: Mapping[str, Any]) -> tuple[bool, list[str],
                                                                dict[str, bool]]:
        protocol = self.plan["protocol"]
        all_trials = len(record["trials"]) == int(protocol["trials"])
        all_samples = all_trials and all(
            trial.get("status") == "measured"
            and len(trial.get("samples", [])) == int(protocol["samples_per_condition"])
            for trial in record["trials"])
        correctness = record.get("correctness") or {}
        correctness_passed = correctness.get("passed") is True
        anomalies = [
            anomaly
            for trial in record["trials"]
            for sample in trial.get("samples", [])
            for anomaly in sample.get("anomalies", [])
        ]
        requirements = {
            "status_measured": record["status"] == "measured",
            "correctness_passed": correctness_passed,
            "all_trials_present": all_trials,
            "all_samples_present": all_samples,
            "no_timing_anomalies": not anomalies,
        }
        blockers = [name for name, passed in requirements.items() if not passed]
        if record["status"] in ("failed", "pending", "not_run"):
            blockers.append(f"condition_{record['status']}")
        return not blockers, blockers, requirements

    def refresh(self) -> None:
        measured = failed = pending = not_run = complete = 0
        completed_trials = attempted = raw_blocks = 0
        for record in self._conditions.values():
            if record["status"] in ("measuring", "measured", "failed", "pending"):
                attempted += 1
            if record["status"] == "measured":
                measured += 1
            elif record["status"] == "failed":
                failed += 1
            elif record["status"] == "pending":
                pending += 1
            elif record["status"] == "not_run":
                not_run += 1
            completed_trials += sum(
                1 for trial in record["trials"] if trial.get("status") == "measured")
            raw_blocks += sum(len(trial.get("samples", []))
                              for trial in record["trials"])
            is_complete, blockers, requirements = self._completeness(record)
            record["measurement_complete"] = is_complete
            record["completeness"] = {
                "complete": is_complete,
                "blockers": blockers,
                "requirements": requirements,
            }
            if is_complete:
                complete += 1
        terminal = measured + failed + pending + not_run
        requested = len(self._conditions)
        if requested == 0:
            status = "invalid_empty"
        elif terminal == requested:
            if measured == requested and failed == 0 and pending == 0 and not_run == 0:
                status = ("completed" if complete == requested
                          else "completed_with_incomplete_measurements")
            else:
                status = "completed_with_failures"
        elif attempted:
            status = "partial"
        else:
            status = "running"
        self.doc["progress"] = {
            "requested_conditions": requested,
            "attempted_conditions": attempted,
            "terminal_conditions": terminal,
            "measured_conditions": measured,
            "failed_conditions": failed,
            "pending_conditions": pending,
            "not_run_conditions": not_run,
            "complete_conditions": complete,
            "completed_trials": completed_trials,
            "planned_condition_executions": int(
                self.plan["counts"]["planned_condition_executions"]),
            "completed_condition_executions": sum(
                1 for record in self._conditions.values()
                for trial in record["trials"]
                if trial.get("status") == "measured"),
            "recorded_condition_executions": sum(
                len(record["trials"]) for record in self._conditions.values()),
            "raw_blocks": raw_blocks,
            "planned_raw_blocks": int(
                self.plan["counts"]["planned_raw_blocks"]),
            "fraction_of_requested_conditions_terminal": (
                terminal / requested if requested else 0.0),
            "fraction_of_planned_raw_blocks_collected": (
                raw_blocks / int(self.plan["counts"]["planned_raw_blocks"])
                if int(self.plan["counts"]["planned_raw_blocks"]) else 0.0),
        }
        if not self._finalized:
            self.doc["status"] = status
        self.doc["updated_at_utc"] = utc_now()

    def set_persistence(self, errors: Sequence[Mapping[str, Any]]) -> None:
        payload = [copy.deepcopy(dict(error)) for error in errors]
        self.doc["persistence"] = {"ok": not payload, "errors": payload}
        self.doc["persistence_ok"] = not payload
        self.refresh()

    def finalize(self, *, stop_reason: str | None = None,
                 fatal: bool = False, cleanup_ok: bool | None = None,
                 persistence_ok: bool | None = None) -> None:
        if stop_reason:
            self.doc["stop_reason"] = stop_reason
        if cleanup_ok is None:
            cleanup_ok = self.doc.get("cleanup_ok")
        if persistence_ok is None:
            persistence_ok = self.doc.get("persistence_ok")
        self.doc["cleanup_ok"] = cleanup_ok
        self.doc["persistence_ok"] = persistence_ok
        self.doc["cleanup"]["ok"] = cleanup_ok
        before = dict(self.doc["progress"])
        requested = int(before["requested_conditions"])
        attempted_before = int(before["attempted_conditions"])
        terminal_before = int(before["terminal_conditions"])
        for condition_id, record in self._conditions.items():
            if record["status"] != "measuring":
                continue
            record["status"] = "failed"
            record["reason"] = "run_stopped_before_condition_completion"
            entry = {
                "phase": "finalize",
                "at_utc": utc_now(),
                "error": {
                    "type": "IncompleteConditionAtFinalize",
                    "module": __name__,
                    "message": (
                        "the run stopped while this condition was measuring; "
                        "collected samples are retained and no retry was made"),
                    "repr": "IncompleteConditionAtFinalize()",
                    "traceback": "",
                    "retry": "not_attempted",
                },
                "retry": "not_attempted",
            }
            record["errors"].append(entry)
            self.doc["failures"].append({
                "condition_id": condition_id,
                "case_id": record["case_id"],
                "mode": record["mode"],
                "n": record["n"],
                **copy.deepcopy(entry),
            })
        self.mark_remaining_not_run("run_stopped_before_condition")
        self.refresh()
        if fatal:
            status = "failed"
        elif terminal_before < requested:
            status = "partial" if attempted_before else "failed"
        else:
            status = self.doc["status"]
        if status == "completed" and persistence_ok is False:
            status = "completed_with_persistence_failure"
        elif status == "completed" and cleanup_ok is False:
            status = "completed_with_cleanup_failure"
        self.doc["status"] = status
        self.doc["finished_at_utc"] = utc_now()
        self.doc["updated_at_utc"] = self.doc["finished_at_utc"]
        self._finalized = True

    def to_dict(self) -> dict[str, Any]:
        return self.doc


class AtomicResultWriter:
    """Write one results document by fsync+replace in its target directory."""

    def __init__(self, path: str | Path, *, allow_overwrite: bool = False) -> None:
        self.path = Path(path)
        self.allow_overwrite = allow_overwrite

    def preflight(self) -> None:
        if self.path.exists() and not self.allow_overwrite:
            raise OutputPathError(
                f"refusing to overwrite existing results evidence: {self.path}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputPathError(
                f"cannot create results directory {self.path.parent}: {exc}") from exc

    def write(self, document: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document, indent=2, ensure_ascii=False,
                             allow_nan=False)
        temporary = self.path.with_name(
            f".{self.path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            try:
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                pass
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
