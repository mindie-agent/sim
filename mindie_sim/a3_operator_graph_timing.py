"""Measured-block timing, partial-trial evidence and post-work cleanup.

[Role]
  Implement the timing half of the device runner: isolate each block with a
  stream synchronize, precreate/reuse events, record CPU-submit/device-interval/
  host-wall clocks, preserve partial trials on failure, observe stream
  responsiveness after failure without retry, and synchronize/reset graphs only
  after the declared work loop ends.

[Invariants]
  * Events are precreated on the same explicit measurement stream.  CPU submit
    excludes event record/elapsed/synchronize APIs; wall spans start submission
    through end completion; device time is event start->end.
  * Timed bodies contain only N eager calls, N replays, or one N-call replay.
  * The exact callable/replay is rewarmed before every trial, outside all timing
    arrays and inside case/global budget accounting; per-trial metadata is kept.
  * Eager output ownership releases the previous block's output before a new
    block and retains only the current final output.  Graph captured output stays
    retained.
  * No empty-event subtraction, analytic time, `.item()`, copies or reference
    math occurs in a timed block.
  * Failures retain all samples already collected and exact error text; no
    uncertain device work is retried.
  * Reset requires successful completion synchronization.  If synchronization
    fails, reset is skipped and cleanup is failed/unknown, never success.

[Timeouts & Locks]
  Cooperative per-sample case/global checks with no locks/threads.  Cleanup is
  bounded at 120 s; the outer supervisor remains the hard bound.

[Side-Effect Policy]
  Enqueues only the declared invoked callable/replay in measured blocks and
  performs synchronize/reset only in post-work cleanup.  Writes no files.
"""
import time
from typing import Any, Mapping

from .a3_operator_graph_plan import OUTER_SUPERVISOR_SECONDS_MIN
from .a3_operator_graph_reference import output_facts
from .a3_operator_graph_results import (
    make_raw_observation,
    summarize_samples,
    utc_now,
)
from .a3_operator_graph_device_common import (
    BudgetClock,
    CaseBudgetExceeded,
    CleanupBudgetExceeded,
    ConditionArtifact,
    GlobalWorkBudgetExceeded,
    TrialFailure,
    exception_info,
)

def warmup_plan(condition: Mapping[str, Any], iterations: int) -> dict[str, Any]:
    """Declare exactly which callable/replay is warmed before every trial."""
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("warmup iterations must be positive")
    mode = condition["mode"]
    n = int(condition["block_invocations"])
    if mode == "eager_block":
        calls = iterations
        replays = 0
        captured_per_replay = 0
        total_captured = 0
        scope = "exact_eager_callable_calls_not_block_sized"
    elif mode == "graph_single_repeated":
        calls = 0
        replays = iterations
        captured_per_replay = 1
        total_captured = replays * captured_per_replay
        scope = "exact_one_call_graph_replays"
    else:
        calls = 0
        replays = iterations
        captured_per_replay = n
        total_captured = replays * captured_per_replay
        scope = "exact_n_call_graph_replays"
    return {
        "policy": "rewarm_exact_callable_or_replay_before_each_trial",
        "timed": False,
        "excluded_from_samples": True,
        "mode": mode,
        "requested_iterations": iterations,
        "scope": scope,
        "block_invocations": n,
        "n_kind": condition.get(
            "n_kind",
            "declared_operator_api_invocations_not_measured_kernel_launches"),
        "calls": calls,
        "replays": replays,
        "captured_invocations_per_replay": captured_per_replay,
        "captured_invocations_warmed": total_captured,
    }


class TimingMixin:
    def _warmup(self, artifact: ConditionArtifact, trial_index: int,
                iterations: int) -> dict[str, Any]:
        plan = warmup_plan(artifact.condition, iterations)
        started = time.monotonic()
        last = None
        with self.torch.npu.stream(self.stream):
            if artifact.graph is not None:
                for _ in range(plan["replays"]):
                    artifact.graph.replay()
            else:
                for _ in range(plan["calls"]):
                    last = artifact.tensors.callable()
        self._sync()
        plan.update({
            "status": "ok",
            "trial_index": int(trial_index),
            "duration_s": time.monotonic() - started,
            "synchronized": True,
            "warmup_output_retained": False,
        })
        if last is not None:
            plan["last_warmup_output"] = output_facts(self.torch, last)
            del last
        return plan

    def measure_trial(self, artifact: ConditionArtifact, trial_index: int,
                      protocol: Mapping[str, Any]) -> dict[str, Any]:
        active_before = float(artifact.metadata["active_case_seconds"])
        case_budget = float(self.budgets["case"])
        samples_expected = int(protocol["samples_per_condition"])
        trial: dict[str, Any] = {
            "trial_index": int(trial_index),
            "condition_id": artifact.condition["condition_id"],
            "mode": artifact.mode,
            "n": int(artifact.condition["n"]),
            "status": "measuring",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "active_case_seconds_at_start": active_before,
            "warmup": None,
            "samples": [],
            "summary": None,
            "errors": [],
        }
        start = time.monotonic()
        try:
            warmup = self._warmup(
                artifact, trial_index, int(protocol["warmup_iterations"]))
            trial["warmup"] = warmup
            active_after_warmup = active_before + (time.monotonic() - start)
            trial["active_case_seconds_after_warmup"] = active_after_warmup
            trial["warmup_duration_s"] = warmup["duration_s"]
            artifact.metadata["active_case_seconds"] = active_after_warmup
            self._check_deadlines(
                start, max(case_budget - active_before, 0.0), "warmup")
            if active_after_warmup > case_budget:
                raise CaseBudgetExceeded(
                    "warmup", active_after_warmup, case_budget)
            for sample_index in range(samples_expected):
                self._check_deadlines(
                    start, max(case_budget - active_before, 0.0), "sample")
                remaining = case_budget - active_before
                if remaining <= 0.0:
                    raise CaseBudgetExceeded(
                        "sample", active_before, case_budget)
                sample = self._one_block(artifact, trial_index, sample_index)
                trial["samples"].append(sample)
                active = active_before + (time.monotonic() - start)
                if active > case_budget:
                    raise CaseBudgetExceeded("sample", active, case_budget)
            trial["status"] = "measured"
        except GlobalWorkBudgetExceeded as exc:
            trial["status"] = "failed"
            trial["errors"].append({
                "phase": "measurement",
                "error": exception_info(exc),
                "retry": "not_attempted",
            })
            trial["finished_at_utc"] = utc_now()
            trial["active_case_seconds_at_end"] = (
                active_before + (time.monotonic() - start))
            trial["summary"] = summarize_samples(trial["samples"])
            artifact.metadata["active_case_seconds"] = (
                trial["active_case_seconds_at_end"])
            raise TrialFailure(trial, exc, stop_run=True) from exc
        except Exception as exc:
            trial["status"] = "failed"
            if trial.get("warmup") is None:
                trial["warmup"] = {
                    "status": "failed",
                    "timed": False,
                    "excluded_from_samples": True,
                    "error": exception_info(exc),
                    "policy": "rewarm_exact_callable_or_replay_before_each_trial",
                }
            trial["errors"].append({
                "phase": "measurement",
                "error": exception_info(exc),
                "retry": "not_attempted",
            })
            trial["finished_at_utc"] = utc_now()
            trial["active_case_seconds_at_end"] = (
                active_before + (time.monotonic() - start))
            trial["summary"] = summarize_samples(trial["samples"])
            artifact.metadata["active_case_seconds"] = (
                trial["active_case_seconds_at_end"])
            raise TrialFailure(trial, exc) from exc
        trial["finished_at_utc"] = utc_now()
        active = active_before + (time.monotonic() - start)
        trial["active_case_seconds_at_end"] = active
        trial["summary"] = summarize_samples(trial["samples"])
        artifact.metadata["active_case_seconds"] = active
        if active > case_budget:
            exc = CaseBudgetExceeded("trial_complete", active, case_budget)
            trial["status"] = "failed"
            trial["errors"].append({
                "phase": "case_budget",
                "error": exception_info(exc),
                "retry": "not_attempted",
            })
            raise TrialFailure(trial, exc)
        return trial

    def _one_block(self, artifact: ConditionArtifact, trial_index: int,
                   sample_index: int) -> dict[str, Any]:
        if artifact.graph is None:
            artifact.final_output = None
        self._sync()
        start_event, end_event = artifact.events
        last_output = None
        with self.torch.npu.stream(self.stream):
            wall_start = time.perf_counter_ns()
            start_event.record(self.stream)
            cpu_start = time.perf_counter_ns()
            if artifact.graph is not None:
                for _ in range(int(artifact.condition["replay_calls_per_block"])):
                    artifact.graph.replay()
            else:
                for _ in range(int(artifact.condition["host_calls_per_block"])):
                    last_output = artifact.tensors.callable()
            cpu_end = time.perf_counter_ns()
            end_event.record(self.stream)
            end_event.synchronize()
            wall_end = time.perf_counter_ns()
            device_ms = float(start_event.elapsed_time(end_event))
        if last_output is not None:
            artifact.final_output = last_output
            artifact.metadata["allocation"]["final_output"] = output_facts(
                self.torch, last_output)
        raw = make_raw_observation(
            artifact.condition,
            trial_index=trial_index,
            sample_index=sample_index,
            cpu_submit_ms=(cpu_end - cpu_start) / 1_000_000.0,
            device_interval_ms=device_ms,
            wall_complete_ms=(wall_end - wall_start) / 1_000_000.0,
        )
        raw["measurement_stream_repr"] = repr(self.stream)
        raw["events_reused"] = True
        raw["synchronize_between_blocks_outside_clocks"] = True
        raw["eager_previous_output_released_before_block"] = (
            artifact.graph is None)
        return raw

    def post_failure_sync(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            self._sync()
            return {
                "status": "ok",
                "duration_s": time.monotonic() - started,
                "policy": "no device work retried; only responsiveness observed",
            }
        except Exception as exc:
            return {
                "status": "failed",
                "duration_s": time.monotonic() - started,
                "error": exception_info(exc),
                "policy": "run stops rather than retrying uncertain device work",
            }

    def cleanup(self, clock: BudgetClock) -> dict[str, Any]:
        started = time.monotonic()
        info: dict[str, Any] = {
            "status": "running",
            "ok": False,
            "started_at_utc": utc_now(),
            "graphs_seen": len(self._graphs),
            "graphs_reset": 0,
            "reset_errors": [],
            "synchronized_before_reset": False,
            "final_synchronize": False,
        }
        try:
            clock.check_cleanup("cleanup_start")
        except CleanupBudgetExceeded as exc:
            info.update({"status": "timeout", "ok": False,
                         "duration_s": time.monotonic() - started,
                         "error": exception_info(exc)})
            return info
        if self.torch is None or self.stream is None:
            if not self._graphs:
                info.update({
                    "status": "not_needed_no_device_resources",
                    "ok": True,
                    "duration_s": time.monotonic() - started,
                    "finished_at_utc": utc_now(),
                })
                return info
            info.update({
                "status": "failed_no_stream_with_graphs",
                "ok": False,
                "duration_s": time.monotonic() - started,
                "error": {
                    "type": "MissingMeasurementStream",
                    "message": "graphs exist but no measurement stream is available",
                },
                "finished_at_utc": utc_now(),
            })
            return info
        try:
            self.stream.synchronize()
            info["synchronized_before_reset"] = True
            info["synchronize"] = {
                "status": "ok",
                "duration_s": time.monotonic() - started,
            }
        except Exception as exc:
            info.update({
                "status": "failed_synchronize_before_reset",
                "ok": False,
                "graphs_reset_skipped": True,
                "synchronize": {"status": "failed",
                                "error": exception_info(exc)},
                "duration_s": time.monotonic() - started,
                "finished_at_utc": utc_now(),
                "policy": "reset is unsafe because completion sync failed",
            })
            return info
        try:
            self.torch.npu.synchronize()
            info["global_synchronize"] = {
                "status": "ok",
                "duration_s": time.monotonic() - started,
            }
        except Exception as exc:
            info.update({
                "status": "failed_global_synchronize_before_reset",
                "ok": False,
                "graphs_reset_skipped": True,
                "global_synchronize": {
                    "status": "failed", "error": exception_info(exc)},
                "duration_s": time.monotonic() - started,
                "finished_at_utc": utc_now(),
                "policy": "reset is unsafe because completion sync failed",
            })
            return info
        for condition_id, graph in self._graphs:
            try:
                clock.check_cleanup(f"reset:{condition_id}")
            except CleanupBudgetExceeded as exc:
                info.update({"status": "timeout", "ok": False,
                             "error": exception_info(exc)})
                break
            try:
                graph.reset()
                info["graphs_reset"] += 1
            except Exception as exc:
                info["reset_errors"].append({
                    "condition_id": condition_id,
                    "error": exception_info(exc),
                })
        if info["status"] == "timeout":
            info["duration_s"] = time.monotonic() - started
            info["finished_at_utc"] = utc_now()
            return info
        final_error: dict[str, Any] | None = None
        try:
            clock.check_cleanup("final_synchronize")
            self.stream.synchronize()
            info["final_synchronize"] = True
        except CleanupBudgetExceeded as exc:
            final_error = exception_info(exc)
            info["status"] = "timeout"
        except Exception as exc:
            final_error = exception_info(exc)
            info["final_synchronize"] = {"status": "failed",
                                         "error": final_error}
        if final_error is None:
            try:
                self.torch.npu.synchronize()
                info["final_global_synchronize"] = True
            except Exception as exc:
                final_error = exception_info(exc)
                info["final_global_synchronize"] = {
                    "status": "failed", "error": final_error}
        if final_error is None:
            try:
                clock.check_cleanup("cleanup_complete")
            except CleanupBudgetExceeded as exc:
                final_error = exception_info(exc)
                info["status"] = "timeout"
                info["cleanup_complete_deadline_exceeded"] = True
        if final_error is not None:
            info["status"] = ("timeout" if info["status"] == "timeout"
                              else "failed_final_synchronize")
            info["ok"] = False
            info["error"] = final_error
        elif info["reset_errors"]:
            info["status"] = "completed_with_reset_errors"
            info["ok"] = False
        else:
            info["status"] = "ok"
            info["ok"] = True
        info["duration_s"] = time.monotonic() - started
        info["finished_at_utc"] = utc_now()
        info["policy"] = (
            "cleanup synchronize/reset occurred only after the declared work "
            "loop ended; reset required successful completion sync; no lease or "
            "queue operation is performed")
        info["within_cleanup_budget"] = (
            info["status"] != "timeout"
            and info["duration_s"] <= float(self.budgets["cleanup"]))
        info["outer_supervisor_seconds_min"] = OUTER_SUPERVISOR_SECONDS_MIN
        return info


