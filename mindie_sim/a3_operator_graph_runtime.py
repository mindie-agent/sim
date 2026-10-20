"""CLI execution loop for the isolated A3 native-operator graph timing worker.

[Role]
  Consume a validated manifest/plan, write requested conditions first, run the
  counterbalanced trial permutations through the device runner, persist after
  every trial and terminal condition, and finalize cleanup/status.  No torch
  import occurs unless this module is invoked for a real run.

[Invariants]
  * Requested condition records exist before setup or any device work.
  * Every trial is a complete permutation; condition artifacts are reused across
    trials after one setup/capture; condition failures never retry.
  * Partial evidence persists atomically after each trial/failure; later trials
    skip terminal conditions.  Failed persistence stops new device work, performs
    a best final write attempt and forces a nonzero result.
  * Cleanup synchronize/reset occurs only after the work loop, and reset requires
    successful completion synchronization.  Cleanup or persistence failure is
    surfaced separately without invalidating otherwise successful measurements.
  * The external coordinator owns the npu:0 lease; this module implements no
    queue/SSH.  Exit 0 requires every requested condition measured and
    measurement-complete plus successful cleanup and persistence; invalid input
    is 2, interrupt 130.

[Timeouts & Locks]
  Cooperative setup/work/case/cleanup limits with no locks/threads; the outer
  supervisor of at least 1380 s is the hard bound for a hung native call.

[Side-Effect Policy]
  Reads the validated manifest/plan, writes only the explicit results path and
  optional debug directory, and allocates device state only in a real run.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .a3_operator_graph_plan import OUTER_SUPERVISOR_SECONDS_MIN, build_conditions
from .a3_operator_graph_results import AtomicResultWriter, ResultBook, utc_now
from .a3_operator_graph_device import DeviceRunner
from .a3_operator_graph_device_common import (
    BudgetClock,
    ConditionArtifact,
    ConditionFailure,
    GlobalWorkBudgetExceeded,
    TrialFailure,
    exception_info,
)

def _source_hashes() -> dict[str, str]:
    paths = {
        "a3_operator_graph": Path(__file__).with_name("a3_operator_graph.py"),
        "a3_operator_graph_plan": Path(__file__).with_name(
            "a3_operator_graph_plan.py"),
        "a3_operator_graph_results": Path(__file__).with_name(
            "a3_operator_graph_results.py"),
        "a3_operator_graph_reference": Path(__file__).with_name(
            "a3_operator_graph_reference.py"),
        "a3_operator_graph_runtime": Path(__file__),
        "a3_operator_graph_device": Path(__file__).with_name(
            "a3_operator_graph_device.py"),
        "a3_operator_graph_device_common": Path(__file__).with_name(
            "a3_operator_graph_device_common.py"),
        "a3_operator_graph_timing": Path(__file__).with_name(
            "a3_operator_graph_timing.py"),
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        try:
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result[name] = "unavailable"
    return result



def _plan_conditions(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    conditions = {condition["condition_id"]: condition
                  for condition in build_conditions(plan["cases"])}
    if set(conditions) != set(plan["condition_ids"]):
        raise RuntimeError("plan condition list is inconsistent")
    return conditions


def _exit_code(book: ResultBook) -> int:
    progress = book.doc["progress"]
    if (book.doc["status"] == "completed"
            and book.doc.get("cleanup_ok") is True
            and book.doc.get("persistence_ok") is True
            and progress["measured_conditions"] == progress["requested_conditions"]
            and progress["complete_conditions"] == progress["measured_conditions"]):
        return 0
    return 1


def run_manifest(plan: Mapping[str, Any], *, manifest_path: str,
                 manifest_sha256: str, run_id: str, device: str,
                 results_path: str | Path, debug_dir: str | None = None,
                 allow_overwrite: bool = False) -> int:
    """Execute one validated plan on the coordinator-provided npu:0."""
    if device != "npu:0":
        raise ValueError("this isolated lane requires the coordinator-visible "
                         "logical device npu:0")
    writer = AtomicResultWriter(results_path, allow_overwrite=allow_overwrite)
    writer.preflight()
    book = ResultBook(
        plan, run_id=run_id, results_path=results_path,
        manifest_path=manifest_path, manifest_sha256=manifest_sha256,
        device=device, source_files=_source_hashes())
    persistence_errors: list[dict[str, Any]] = []

    def persist(stage: str) -> bool:
        try:
            writer.write(book.to_dict())
            return True
        except Exception as exc:
            persistence_errors.append({
                "stage": stage,
                "path": str(results_path),
                "at_utc": utc_now(),
                "error": exception_info(exc),
            })
            book.set_persistence(persistence_errors)
            return False

    initial_persisted = persist("initial_requested_conditions")
    clock = BudgetClock(plan["budgets"])
    conditions = _plan_conditions(plan)
    runner: DeviceRunner | None = None
    artifacts: dict[str, ConditionArtifact] = {}
    stop_reason: str | None = None
    fatal = False
    interrupted = False
    unexpected_error: dict[str, Any] | None = None
    if not initial_persisted:
        stop_reason = "initial_persistence_failure"
        fatal = True
    try:
        if initial_persisted:
            try:
                setup_torch = importlib.import_module("torch")
            except Exception:
                setup_torch = None
            inference_context = (setup_torch.inference_mode()
                                 if setup_torch is not None else _NullContext())
            with inference_context:
                try:
                    runner = DeviceRunner(
                        device=device, device_index=0, budgets=plan["budgets"],
                        debug_dir=debug_dir)
                    setup = runner.setup(clock)
                    book.set_setup(setup)
                    if runner.environment:
                        book.set_environment(runner.environment)
                    if not persist("setup"):
                        stop_reason = "persistence_failure:setup"
                except Exception as exc:
                    error = exception_info(exc)
                    book.mark_setup_failure(error)
                    stop_reason = "setup_failed"
                    fatal = True
                    if not persist("setup_failure"):
                        stop_reason = (
                            "setup_failed; persistence_failure:setup_failure")
                if not fatal and stop_reason is None and runner is not None:
                    clock.start_work()
                    for trial_index, order in enumerate(
                            plan["schedule"]["trial_orders"]):
                        for condition_id in order:
                            record = book.condition(condition_id)
                            if record["status"] in (
                                    "failed", "pending", "not_run", "measured"):
                                continue
                            try:
                                clock.check_work("before_condition")
                            except GlobalWorkBudgetExceeded as exc:
                                stop_reason = f"global_work_budget: {exc}"
                                break
                            artifact = artifacts.get(condition_id)
                            if artifact is None:
                                active_before = float(
                                    record.get("active_case_seconds") or 0.0)
                                try:
                                    artifact = runner.prepare(
                                        conditions[condition_id],
                                        plan["protocol"], active_before)
                                except ConditionFailure as exc:
                                    book.update(
                                        condition_id,
                                        active_case_seconds=(
                                            exc.metadata.get(
                                                "active_case_seconds")
                                            or active_before),
                                        **{key: value for key, value in
                                           exc.metadata.items()
                                           if key != "active_case_seconds"})
                                    book.fail(condition_id, exc.phase,
                                              exception_info(exc.cause),
                                              status=exc.status,
                                              reason=exc.reason)
                                    if not persist("condition_failure"):
                                        stop_reason = (
                                            "persistence_failure:"
                                            "condition_failure")
                                        break
                                    if exc.stop_run:
                                        stop_reason = (
                                            f"global_work_budget: {exc}")
                                        break
                                    recovery = runner.post_failure_sync()
                                    book.attach_recovery(condition_id, recovery)
                                    if not persist("condition_failure_recovery"):
                                        stop_reason = (
                                            "persistence_failure:"
                                            "condition_failure_recovery")
                                        break
                                    if recovery.get("status") != "ok":
                                        stop_reason = (
                                            "device_unresponsive_after_"
                                            "condition_failure")
                                        break
                                    continue
                                artifacts[condition_id] = artifact
                                book.update(
                                    condition_id,
                                    active_case_seconds=(
                                        artifact.active_case_seconds),
                                    **{key: value for key, value in
                                       artifact.metadata.items()
                                       if key != "active_case_seconds"})
                                if not persist("condition_prepared"):
                                    stop_reason = (
                                        "persistence_failure:condition_prepared")
                                    break
                            try:
                                trial = runner.measure_trial(
                                    artifact, trial_index, plan["protocol"])
                            except TrialFailure as exc:
                                book.record_trial(condition_id, exc.trial)
                                meta = exc.trial.get(
                                    "active_case_seconds_at_end",
                                    float(record.get("active_case_seconds")
                                          or 0.0))
                                book.update(condition_id,
                                            active_case_seconds=meta)
                                book.fail(condition_id, "measurement",
                                          exc.error)
                                if not persist("trial_failure"):
                                    stop_reason = (
                                        "persistence_failure:trial_failure")
                                    break
                                if exc.stop_run:
                                    stop_reason = (
                                        f"global_work_budget: {exc.cause}")
                                    break
                                recovery = runner.post_failure_sync()
                                book.attach_recovery(condition_id, recovery)
                                if not persist("trial_failure_recovery"):
                                    stop_reason = (
                                        "persistence_failure:"
                                        "trial_failure_recovery")
                                    break
                                if recovery.get("status") != "ok":
                                    stop_reason = (
                                        "device_unresponsive_after_"
                                        "condition_failure")
                                    break
                                continue
                            book.record_trial(condition_id, trial)
                            book.update(
                                condition_id, active_case_seconds=float(
                                    trial["active_case_seconds_at_end"]))
                            if not persist("trial"):
                                stop_reason = "persistence_failure:trial"
                                break
                        if stop_reason:
                            break
                        if not persist(f"trial_pass_{trial_index}_complete"):
                            stop_reason = (
                                f"persistence_failure:trial_pass_"
                                f"{trial_index}_complete")
                            break
                    if stop_reason is None:
                        try:
                            clock.check_work("work_complete")
                        except GlobalWorkBudgetExceeded as exc:
                            stop_reason = f"global_work_budget: {exc}"
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "keyboard_interrupt"
    except Exception as exc:
        unexpected_error = exception_info(exc)
        stop_reason = f"unexpected_runtime_error: {type(exc).__name__}"
    finally:
        clock.start_cleanup()
        if runner is not None:
            cleanup = runner.cleanup(clock)
        else:
            cleanup = {
                "status": "not_started",
                "ok": True,
                "reason": "device runner was not created",
                "outer_supervisor_seconds_min": OUTER_SUPERVISOR_SECONDS_MIN,
            }
        book.set_cleanup(cleanup)
        if unexpected_error is not None:
            book.doc["runtime_error"] = unexpected_error
        book.set_persistence(persistence_errors)
        cleanup_ok = cleanup.get("ok") is True
        persistence_ok = not persistence_errors
        book.finalize(stop_reason=stop_reason, fatal=fatal,
                      cleanup_ok=cleanup_ok,
                      persistence_ok=persistence_ok)
        final_write_error: dict[str, Any] | None = None
        try:
            writer.write(book.to_dict())
        except Exception as exc:
            final_write_error = {
                "stage": "final",
                "path": str(results_path),
                "at_utc": utc_now(),
                "error": exception_info(exc),
            }
            persistence_errors.append(final_write_error)
            persistence_ok = False
            book.set_persistence(persistence_errors)
            book.doc["cleanup_ok"] = cleanup_ok
            book.doc["persistence_ok"] = False
            book.doc["status"] = (
                "completed_with_persistence_failure"
                if book.doc["status"] == "completed"
                else book.doc["status"])
    if persistence_errors:
        payload = {
            "error": "persistence_failure",
            "durable_final_write": final_write_error is None,
            "results_path": str(results_path),
            "errors": persistence_errors,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False,
                         allow_nan=False), file=sys.stderr)
    if interrupted:
        return 130
    return _exit_code(book)


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None
