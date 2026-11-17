"""Killable candidate evaluation worker for bounded topology sweeps.

[Role] Evaluate one sweep candidate in a separate process so an uncooperative
  lowering or engine run can be killed at its declared deadline.
[Invariants] The request carries only the parsed case, the candidate parallel
  spec and the phase probe.  The reply is either one complete row or the
  original exception; no partial row crosses the boundary.
[Timeouts & Locks] The parent enforces the deadline; this process holds none.
[Side-Effect Policy] No files, no device work, no network; stdin/stdout only.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec


WORKER_MODULE = "mindie_sim._sweep_worker"
EMPTY_BUDGET_S = 0.0
MAX_DIAGNOSTIC_BYTES = 2048
STAGE_SCREEN = "screen"


def _evaluate(job: Mapping[str, Any], stage: str) -> dict[str, Any]:
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.sweep_engine import (
        b1_screen, build_candidate_case, full_row, static_row,
        validate_candidate_statically,
    )
    from mindie_sim.sweep_resize import materialize_machine_for_devices

    base = job["_case"]
    common = dict(job["common"])
    probe = dict(job["probe"])
    devices = int(job["num_devices"])
    part = tuple(job["part"] or ())
    parallel = ParallelSpec.from_mapping(job["parallel_spec"])
    machine = materialize_machine_for_devices(base, base, devices)
    case = build_candidate_case(base, machine, parallel, part)
    try:
        static = validate_candidate_statically(
            base, parallel, cand_devices=devices, cand_machine=machine,
            part=part, phase=probe["phase"], context=probe["context"],
            q_tokens=probe["q_tokens"], emit_tokens=probe["emit_tokens"],
            output_budget=probe["output_budget"],
            max_sequences_per_dp=probe["max_sequences_per_dp"])
    except (ModelingError, ValueError) as exc:
        return static_row(common, reason=str(exc), kind="topology",
                          elapsed_ms=EMPTY_BUDGET_S)
    if not static["ok"]:
        return static_row(common, reason=str(static["reason"]),
                          kind=str(static["kind"]),
                          elapsed_ms=EMPTY_BUDGET_S)
    arguments = dict(
        common=common, ctx=EvaluationContext.create(case),
        phase=probe["phase"], scope=probe["scope"],
        context=probe["context"], q_tokens=probe["q_tokens"],
        emit_tokens=probe["emit_tokens"], sla_ms=probe["sla_ms"],
        output_budget=probe["output_budget"],
        max_sequences_per_dp=probe["max_sequences_per_dp"],
        b_vram_max=int(static["b_vram_max_joint"]), static=static)
    if stage == STAGE_SCREEN:
        return b1_screen(base, case, parallel, **arguments)
    return full_row(base, case, parallel, **arguments)


def evaluate_sweep_job(job: Mapping[str, Any], stage: str,
                       budget_s: float | None) -> dict[str, Any]:
    """Run one candidate in a killable child process."""
    if budget_s is not None and budget_s <= EMPTY_BUDGET_S:
        raise TimeoutError("candidate_timeout")
    payload = pickle.dumps((dict(job), stage))
    started = time.monotonic()
    with subprocess.Popen(
            [sys.executable, "-B", "-m", WORKER_MODULE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE) as process:
        try:
            remaining = (None if budget_s is None
                         else max(EMPTY_BUDGET_S,
                                  budget_s - (time.monotonic() - started)))
            output, errors = process.communicate(payload, timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise TimeoutError("candidate_timeout") from exc
        if process.returncode:
            detail = (errors or b"")[-MAX_DIAGNOSTIC_BYTES:].decode(
                "utf-8", "replace")
            raise ModelingError(
                f"candidate_worker_exit:{process.returncode}:{detail}".rstrip(":"))
    ok, result = pickle.loads(output)
    if not ok:
        raise result
    return result


def main() -> None:
    """Worker entry point: one request in, one reply out."""
    job, stage = pickle.load(sys.stdin.buffer)
    try:
        reply = (True, _evaluate(job, stage))
    except Exception as exc:  # noqa: BLE001 - transported to the parent process
        reply = (False, exc)
    pickle.dump(reply, sys.stdout.buffer)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
