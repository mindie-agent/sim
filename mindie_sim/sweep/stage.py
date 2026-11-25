"""Bounded stage execution, checkpointing, locked search and killable sweep worker.

[Role] Own stage-1/stage-2 candidate execution, JSONL checkpointing, locked-axis
  search, and killable subprocess candidate evaluation for topology sweeps.
[Invariants] Every candidate row is complete or carries an explicit error/timeout;
  checkpoint replay preserves exact stage ordering and determinism.
[Timeouts & Locks] Honors per-candidate and stage deadlines via killable child workers.
[Side-Effect Policy] Writes checkpoints only to caller-supplied paths."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from typing import Any

from mindie_sim.sweep.engine import B1_STAGE


MAX_CHECKPOINT_ROW_BYTES = 4 * 1024 * 1024
CHECKPOINT_INDEX_LIMIT = 100000
CANDIDATE_ID_KEY = "candidate_id"
STAGE_SCREEN = "screen"
STAGE_FULL = "full"
STAGES = frozenset({STAGE_SCREEN, STAGE_FULL})


@dataclass(frozen=True)
class CheckpointRef:
    """One committed row position and the evaluation stage it records."""

    offset: int
    stage: str


def _stage_of(row: Mapping[str, Any]) -> str:
    stage = row.get("search_stage")
    if stage in STAGES:
        return str(stage)
    return STAGE_SCREEN if row.get("evaluation_stage") == B1_STAGE else STAGE_FULL


def checkpoint_index(stream) -> dict[str, CheckpointRef]:
    """Index complete lines and truncate one incomplete trailing line."""
    index: dict[str, CheckpointRef] = {}
    while stream is not None:
        offset = stream.tell()
        line = stream.readline(MAX_CHECKPOINT_ROW_BYTES)
        if not line:
            break
        if not line.endswith(b"\n"):
            if len(line) == MAX_CHECKPOINT_ROW_BYTES:
                raise ValueError("checkpoint_row_too_large")
            # The writer flushes complete rows; a tail without its newline is
            # an interrupted append and is removed before new rows are written.
            stream.seek(offset)
            stream.truncate()
            break
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict)
                or not row.get(CANDIDATE_ID_KEY)
                or _stage_of(row) not in STAGES):
            raise ValueError("invalid_sweep_checkpoint_row")
        index[str(row[CANDIDATE_ID_KEY])] = CheckpointRef(
            offset, _stage_of(row))
        if len(index) > CHECKPOINT_INDEX_LIMIT:
            raise ValueError("checkpoint_index_limit_exceeded")
    return index


def read_checkpoint_row(stream, ref: CheckpointRef) -> dict[str, Any]:
    """Read exactly the indexed line, never the whole checkpoint."""
    stream.seek(ref.offset)
    return json.loads(stream.readline(MAX_CHECKPOINT_ROW_BYTES))


def write_checkpoint_row(stream, row: Mapping[str, Any]) -> CheckpointRef:
    """Append one compact row and return its committed position."""
    if stream is None:
        return CheckpointRef(0, _stage_of(dict(row)))
    payload = (json.dumps(dict(row), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False) + "\n").encode("utf-8")
    if len(payload) >= MAX_CHECKPOINT_ROW_BYTES:
        raise ValueError("checkpoint_row_too_large")
    stream.seek(0, os.SEEK_END)
    offset = stream.tell()
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    return CheckpointRef(offset, _stage_of(dict(row)))


def open_checkpoint(path, resume: bool):
    """Open a checkpoint for index/append, repairing only on resume."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a+b" if resume else "w+b")


from collections.abc import Mapping
import hashlib

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    evaluate_fixed_plan, prepare_fixed_plan, solve_sla_sizing,
)
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep.engine import BASELINE_SCENARIO, LOCKED_STAGE, parallel_report

MS_PER_S = 1000.0


def locked_plan_identity(plan: Mapping[str, object]) -> str:
    graphs = plan["graphs"]
    head = (getattr(graphs, "default_selection", None),
            tuple((str(op.op_id), str(op.recipe), tuple(op.local_shape))
                  for op in next(iter(graphs.values()))))
    return hashlib.sha256(repr(head).encode("utf-8")).hexdigest()[:16]


def prepare_locked_plan(parsed: CaseSpec, *, sla_ms: float, context: int,
                        phase: str, scope: str,
                        tokens_per_seq: int | None, output_budget: int,
                        max_sequences_per_dp: int) -> dict[str, object]:
    result = solve_sla_sizing(
        parsed, sla_ms=sla_ms, context=context, phase=phase, scope=scope,
        tokens_per_seq=tokens_per_seq, output_budget=output_budget,
        max_sequences_per_dp=max_sequences_per_dp,
        scenarios=BASELINE_SCENARIO)
    scenario = result["scenarios"][0]
    candidate = scenario.get("feasible_candidate")
    if candidate is None:
        raise ModelingError(
            f"lock_parallel_baseline_infeasible:"
            f"{scenario.get('feasibility_status')}")
    batch, tokens = int(candidate["b"]), int(result["tokens_per_seq"])
    plan = prepare_fixed_plan(parsed, phase=phase, tokens=tokens,
                              context=context, sequences=batch, scope=scope)
    return {
        "parallel": parsed.runtime.parallel,
        "pp_layer_partition": tuple(parsed.runtime.pp_layer_partition),
        "b": batch, "tokens": tokens,
        "emitted": float(scenario["emitted_tokens_per_seq"]), "plan": plan,
        "plan_identity": locked_plan_identity(plan),
        "baseline_status": scenario.get("feasibility_status"),
    }


def validate_locked_arguments(parsed: CaseSpec, *, candidates, fixed_parallel,
                              num_devices, candidate_tps, ep_candidates,
                              pp_candidates, pcp_candidates) -> None:
    base = parsed.runtime.parallel
    for label, value in (("fixed_parallel", fixed_parallel),
                         ("candidates", candidates)):
        if value is not None:
            specs = [value] if label == "fixed_parallel" else list(value)
            if len(specs) != 1 or ParallelSpec.from_mapping(
                    specs[0]) != base:
                raise ModelingError(
                    f"lock_parallel_{label}_conflicts_with_baseline")
    devices = (num_devices if num_devices is not None
               else len(parsed.machine.dies))
    if devices != base.world_size:
        raise ModelingError(
            f"lock_parallel_num_devices_conflicts_with_baseline:{num_devices}")
    for name, value, domain in (("tp", base.tp, candidate_tps),
                                ("ep", base.ep, ep_candidates),
                                ("pp", base.pp, pp_candidates),
                                ("pcp", base.pcp, pcp_candidates)):
        if domain is not None and value not in {int(item) for item in domain}:
            raise ModelingError(
                f"lock_parallel_{name}_candidates_conflict_with_baseline:{value}")


def locked_row(cand_case: CaseSpec, locked: Mapping[str, object], *,
               sla_ms: float, output_budget: int, devices: int,
               ranks_per_node: int) -> dict[str, object]:
    geometry = dict(locked["plan"]["geometry"])
    spec, batch = locked["parallel"], int(locked["b"])
    analysis = evaluate_fixed_plan(
        cand_case, plan=locked["plan"], include_capacity=True,
        output_budget=output_budget,
        emitted_tokens_per_seq=float(locked["emitted"]))
    capacity = analysis["capacity"]
    step = float(analysis["graph_duration_ms"])
    b_vram, sla_ok = int(capacity["b_vram_max_configured"]), step <= sla_ms
    feasible = bool(sla_ok and batch <= b_vram and capacity["is_vram_feasible"])
    tps = ((batch * spec.dp * float(locked["emitted"]) * MS_PER_S / step)
           if feasible and step > 0.0 else 0.0)
    probe = max(1, int(geometry["context"]))
    memory = Memory(cand_case, workspace_tokens=1)
    candidate = {
        "b": batch, "step_ms": step, "feasible": feasible,
        "source": "retained_baseline_plan", "input_boundary": {
            **geometry, "output_budget": output_budget,
            "emitted_tokens_per_seq": locked["emitted"]},
        "plan": {"kind": "prepared_fixed_plan",
                 "identity": locked["plan_identity"],
                 "selection": "baseline_case"},
    }
    return {
        "parallel": {key: int(getattr(spec, key))
                     for key in ("dp", "tp", "ep", "pp", "pcp")},
        "parallel_spec": parallel_report(spec),
        "pp_layer_partition": list(locked["pp_layer_partition"]),
        "status": ("feasible" if feasible
                   else "sla_or_vram_exceeded_at_retained_batch"),
        "feasible": feasible, "plan_mode": "fixed_parallel",
        "evaluation_stage": LOCKED_STAGE, "is_screening_estimate": False,
        "evaluation_boundary":
            "complete_prepared_plan_at_retained_parallel_and_batch",
        "reason": None if feasible else (
            "retained_batch_sla_exceeded" if not sla_ok
            else "retained_batch_vram_exceeded"),
        "rejection_reason": None if feasible else (
            "retained_batch_sla_exceeded" if not sla_ok
            else "retained_batch_vram_exceeded"),
        "num_devices": int(devices), "ranks_per_node": int(ranks_per_node),
        "weight_bytes_per_die": float(max(memory.weights)),
        "kv_bytes_per_token_per_die": float(
            max(memory.cache_bytes(probe, padded=False)) / probe),
        "b_sla": batch if sla_ok else 0,
        "b_sla_lower_bound": batch if sla_ok else 0,
        "b_sla_is_proven_max": False, "b_vram_max": b_vram,
        "b_feasible": batch if feasible else 0,
        "b_retained": batch, "b_evaluated": batch,
        "batch_geometry": {"b": batch, "context": int(geometry["context"]),
                           "tokens": int(geometry["tokens"]),
                           "scope": geometry["scope"],
                           "selection": "retained_baseline_batch"},
        "step_ms_at_b1": None, "step_ms_at_retained_b": step,
        "tps_at_b1": None,
        "latency_candidate": candidate if sla_ok else None,
        "feasible_candidate": candidate if feasible else None,
        "feasibility_status": ("feasible_retained_batch" if feasible
                               else "retained_batch_infeasible"),
        "search": {"method": "retained_baseline_batch_no_reselection",
                   "b_sla_is_proven_max": False,
                   "evaluated_candidates": [batch]},
        "total_feasible_concurrency": (batch if feasible else 0) * spec.dp,
        "step_ms_feasible": step if feasible else None,
        "tps_feasible": tps, "tps_per_die": tps / float(devices),
        "network_mbu": analysis.get("network_mbu"),
        "network_cube_mfu": analysis.get("network_cube_mfu"),
        "network_vec_mfu": analysis.get("network_vec_mfu"),
        "network_link_util": analysis.get("network_link_util"),
        "plan_identity": locked["plan_identity"],
        "metric_provenance":
            "one_prepared_baseline_plan_and_batch_geometry_per_hardware_variant",
    }


import pickle
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec


WORKER_MODULE = "mindie_sim.sweep.stage"
EMPTY_BUDGET_S = 0.0
MAX_DIAGNOSTIC_BYTES = 2048


def _evaluate(job: Mapping[str, Any], stage: str) -> dict[str, Any]:
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.sweep.engine import (
        b1_screen, build_candidate_case, full_row, static_row,
        validate_candidate_statically,
    )
    from mindie_sim.sweep.resize import materialize_machine_for_devices

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
    except UnmodeledCapabilityError:
        raise
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


import time
from collections.abc import Mapping
from typing import Any

from mindie_sim.prediction.evaluation import ACTIVE_DEADLINE_MONO
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep.engine import (
    CANDIDATE_ID_KEY, FULL_STAGE, select_finalists, set_deadline, static_row,
    timeout_row,
)


PARETO_EPS = 1e-12
FINALIST_COMMON_KEYS = (
    CANDIDATE_ID_KEY, "parallel", "parallel_spec", "num_devices",
    "ranks_per_node", "kv_cache_replicated_tp",
    "kv_cache_replication_factor", "canonical_kv_sharding",
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
    from mindie_sim.sweep.stage import evaluate_sweep_job

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
        common = {key: item[key] for key in FINALIST_COMMON_KEYS if key in item}
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
        except UnmodeledCapabilityError:
            raise
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


if __name__ == "__main__":
    main()
