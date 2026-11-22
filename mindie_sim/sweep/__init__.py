"""Streaming topology sweeps and disaggregated Prefill/Decode Pareto analysis.

[Role] Enumerate structural parallel candidates, stream deterministic rows,
  checkpoint/resume them as JSONL, statically reject impossible candidates
  before graph lowering, and compose P/D stage rates into a Pareto frontier.
[Invariants] ``sweep_topologies(iter(...))`` reconstructs the exact same report.
  Candidate identity is a pure hash of model/runtime, hardware, phase boundary
  and complete parallel spec. Screening estimates never masquerade as full SLA
  rows. JSONL rows are written before they are yielded.
[Timeouts & Locks]
  Cooperative monotonic deadlines in-process; an explicitly requested
  ``isolate_workers`` run wraps hardware materialization, capacity checks,
  lowering and execution in one killable child process.  No threads.
[Side-Effect Policy]
  Input cases are never mutated. Only the explicit ``output_jsonl`` path is
  written, one compact line per evaluated row.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    DEFAULT_SLA_SCENARIOS, _lower_and_run_graph, _resolve_speculative_tokens,
    prepare_fixed_plan, solve_sla_sizing,
)
from mindie_sim.prediction.evaluation import (
    ACTIVE_DEADLINE_MONO, EvaluationContext, check_active_deadline,
)
from mindie_sim.prediction.sharding.local_tp import local_tp_spans_dp
from mindie_sim.prediction.offload.policy import missing_scenario_error
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep.engine import (
    B1_STAGE, BASELINE_SCENARIO, CANDIDATE_ID_KEY,
    DEFAULT_MAX_SEQUENCES,
    FULL_STAGE, LOCKED_PROGRAM_ERROR, LOCKED_STAGE, STATIC_STAGE,
    TIMEOUT_STAGE,
    SweepEvaluationCache, _parallel_short, b1_screen, build_candidate_case,
    compute_pareto_frontier, compute_sweep_candidate_id,
    enumerate_case_candidates, evaluate_candidate_worker, full_row,
    kv_replication_fields, parallel_report, project_replica_row,
    set_deadline, static_row, timeout_row, validate_candidate_statically,
)
from mindie_sim.runtime_advisor import resolve_jobs
from mindie_sim.sweep.stage import (
    STAGE_FULL, STAGE_SCREEN, checkpoint_index, open_checkpoint,
    read_checkpoint_row, write_checkpoint_row,
)
from mindie_sim.sweep.stage import (
    locked_row as _locked_row, prepare_locked_plan as _prepare_locked_plan,
    validate_locked_arguments as _validate_locked_arguments,
)
from mindie_sim.sweep.stage import (
    evaluate_finalists, isolated_row as _isolated_row,
    pareto_screen as _b1_pareto, retain_finalist,
    screen_row_meta as _screen_row_meta,
)
from mindie_sim.sweep.pd import (
    solve_optimal_deployment,
    sweep_pd_disaggregation_ratios,
)
from mindie_sim.sweep.resize import (
    check_pos_float, check_pos_int, enumerate_parallel_candidates,
    materialize_machine_for_devices, resolve_ranks_per_node,
)

_materialize_machine_for_devices = materialize_machine_for_devices

MS_PER_S = 1000.0


def _validate_sweep_arguments(*, sla_ms: float, context: int, output_budget: int,
                              max_sequences_per_dp: int, phase: str,
                              scope: str, tokens_per_seq: int | None,
                              two_stage: bool, candidate_timeout_s: float | None,
                              resume: bool, output_jsonl: Any) -> None:
    check_pos_float("sla_ms", sla_ms)
    check_pos_int("context", context, minimum=0)
    check_pos_int("output_budget", output_budget, minimum=0)
    check_pos_int("max_sequences_per_dp", max_sequences_per_dp, minimum=1)
    if phase not in {"prefill", "decode"} or scope not in {"body", "model"}:
        raise ValueError(f"invalid_phase_or_scope:{phase}:{scope}")
    if tokens_per_seq is not None:
        check_pos_int("tokens_per_seq", tokens_per_seq, minimum=1)
    if candidate_timeout_s is not None:
        check_pos_float("candidate_timeout_s", candidate_timeout_s)
    if resume and output_jsonl is None:
        raise ValueError("resume_requires_output_jsonl")


def _derivation_key(parsed: CaseSpec) -> str:
    assumptions = dict(getattr(parsed.machine, "assumptions", ()) or ())
    return json.dumps({str(key): value for key, value in assumptions.items()
                       if str(key).startswith("derivation.")},
                      sort_keys=True, default=repr)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Index the committed rows of an existing checkpoint (streamed)."""
    if not path.exists():
        return {}
    with path.open("r+b") as stream:
        return checkpoint_index(stream)


def _open_stream(path: Path | None, resume: bool):
    return open_checkpoint(path, resume)


def _write_row(stream, row: Mapping[str, Any]):
    return write_checkpoint_row(stream, row)


def _checkpoint_path(output_jsonl, checkpoint_jsonl):
    """Treat ``checkpoint_jsonl`` as the validated alias of ``output_jsonl``."""
    if checkpoint_jsonl is None:
        return output_jsonl
    if (output_jsonl is not None
            and Path(checkpoint_jsonl) != Path(output_jsonl)):
        raise ValueError("conflicting_checkpoint_paths")
    return checkpoint_jsonl


def _run_finalist_stage(items, *, finalist_top_k, cache, output_jsonl,
                        progress_callback, isolate_workers,
                        candidate_timeout_s, deadline_monotonic, phase, scope,
                        context, q_tokens, emit_tokens, sla_ms, output_budget,
                        max_sequences_per_dp):
    """Evaluate the retained finalists against the complete solver."""
    path = Path(output_jsonl) if output_jsonl is not None else None
    stream = open_checkpoint(path, True) if path is not None else None
    try:
        yield from evaluate_finalists(
            items, finalist_top_k=finalist_top_k, cache=cache, stream=stream,
            full_row_fn=full_row, progress_callback=progress_callback,
            isolate_workers=isolate_workers,
            candidate_timeout_s=candidate_timeout_s,
            deadline_monotonic=deadline_monotonic, phase=phase, scope=scope,
            context=context, q_tokens=q_tokens, emit_tokens=emit_tokens,
            sla_ms=sla_ms, output_budget=output_budget,
            max_sequences_per_dp=max_sequences_per_dp)
    finally:
        if stream is not None:
            stream.close()


def _resolve_sweep_scenario_inputs(
    parsed: CaseSpec,
    *,
    sla_ms: float | None,
    context: int | None,
    output_budget: int | None,
    phase: str,
) -> tuple[float, int, int]:
    """Enforce Domain-1 fail-fast on sweep SLA/context/output_budget inputs."""
    resolved_sla = sla_ms
    if resolved_sla is None:
        resolved_sla = (
            parsed.slo.ttft_ms if phase == "prefill" else parsed.slo.tpot_ms
        )
    resolved_output = output_budget
    if resolved_output is None:
        meta = dict(parsed.runtime.metadata)
        declared = meta.get("scenario_declared", True) is not False
        reqs = tuple(parsed.workload.requests or ())
        if declared and reqs and reqs[0].request_id != "__structure_probe__":
            resolved_output = int(reqs[0].output_tokens)
    missing = [
        name for name, val in (
            ("sla_ms", resolved_sla),
            ("context", context),
            ("output_budget", resolved_output),
        ) if val is None
    ]
    if missing:
        raise missing_scenario_error(missing)
    return float(resolved_sla), int(context), int(resolved_output)


def _sweep_hardware_derivations(base, derivations, *, output_jsonl, resume, **sweep_options):
    for derivation_index, derivation in enumerate(derivations):
        from mindie_sim.hardware_derivation import derived_machine
        declare = dict(derivation)
        sku = str(declare.pop("sku", base.machine.sku or "derived"))
        derived = derived_machine(base.machine, {**declare, "sku": sku})
        sub_case = replace(base, machine=derived)
        for row in sweep_topologies_iter(
            sub_case, output_jsonl=output_jsonl,
            resume=(resume or (output_jsonl is not None and derivation_index > 0)),
            **sweep_options,
        ):
            row["derived_sku"] = sku
            row["derivation"] = {
                "sku": derived.sku,
                "aic_count": (derived.dies[0].compute_units
                              if derived.dies else None),
                "aiv_count": (dict(derived.dies[0].resource_units).get("vector")
                              if derived.dies else None),
                "memory_bytes": (derived.dies[0].memory_bytes
                                 if derived.dies else None),
                "assumptions": dict(derived.assumptions),
            }
            yield row


def sweep_topologies_iter(
    case: CaseSpec | Mapping[str, Any], *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    num_devices: int | None = None,
    num_devices_grid: Sequence[int] | None = None,
    sla_ms: float | None = None,
    context: int | None = None,
    phase: str = "decode",
    scope: str = "model",
    tokens_per_seq: int | None = None,
    output_budget: int | None = None,
    max_sequences_per_dp: int = DEFAULT_MAX_SEQUENCES,
    allow_dp: bool = True,
    allow_ep: bool = True,
    allow_pp: bool = False,
    allow_pcp: bool = False,
    allow_dcp: bool = False,
    candidate_tps: Sequence[int] | None = None,
    ep_candidates: Sequence[int] | None = None,
    pp_candidates: Sequence[int] | None = None,
    pcp_candidates: Sequence[int] | None = None,
    dcp_candidates: Sequence[int] | None = None,
    hardware_derivations: Sequence[Mapping[str, Any]] | None = None,
    lock_parallel: bool = False,
    fixed_parallel: Mapping[str, int] | None = None,
    lock_program: bool = False,
    two_stage: bool = False,
    candidate_timeout_s: float | None = None,
    deadline_monotonic: float | None = None,
    output_jsonl: str | Path | None = None,
    resume: bool = False,
    progress_callback: Any | None = None,
    _locked_plan: Mapping[str, Any] | None = None,
    finalist_top_k: int | None = None,
    checkpoint_jsonl: str | Path | None = None,
    isolate_workers: bool = False,
    jobs: int | str = 1,
    _finalist_sink: list | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream deterministic candidate rows, optionally checkpointing each row."""
    output_jsonl = _checkpoint_path(output_jsonl, checkpoint_jsonl)
    if lock_program and not lock_parallel:
        raise ModelingError(LOCKED_PROGRAM_ERROR)
    parsed_pre = parse_case(case)
    sla_ms, context, output_budget = _resolve_sweep_scenario_inputs(
        parsed_pre, sla_ms=sla_ms, context=context,
        output_budget=output_budget, phase=phase,
    )
    _validate_sweep_arguments(
        sla_ms=sla_ms, context=context, output_budget=output_budget,
        max_sequences_per_dp=max_sequences_per_dp, phase=phase, scope=scope,
        tokens_per_seq=tokens_per_seq, two_stage=two_stage,
        candidate_timeout_s=candidate_timeout_s, resume=resume,
        output_jsonl=output_jsonl)
    if finalist_top_k is not None and (
            type(finalist_top_k) is not int or finalist_top_k < 1):
        raise ValueError("finalist_top_k_must_be_positive")
    from mindie_sim.prediction.efficiency.unmodeled_gate import assert_timing_supported
    assert_timing_supported(case, entry="sweep_topologies")
    owns_finalists = (two_stage and finalist_top_k is not None
                      and _finalist_sink is None)
    finalist_sink = [] if owns_finalists else _finalist_sink
    devices_grid = tuple(int(item) for item in (num_devices_grid or ()))
    if devices_grid and num_devices is not None:
        raise ValueError("num_devices_and_grid_are_mutually_exclusive")
    if hardware_derivations:
        base = parse_case(case)
        base_q, base_emit = _resolve_speculative_tokens(
            base, output_budget=output_budget, tokens_per_seq=tokens_per_seq,
            emitted_tokens_per_seq=None)
        yield from _sweep_hardware_derivations(
            base, hardware_derivations, output_jsonl=output_jsonl, resume=resume,
            candidates=candidates, num_devices=num_devices, num_devices_grid=num_devices_grid,
            sla_ms=sla_ms, context=context, phase=phase, scope=scope,
            tokens_per_seq=tokens_per_seq, output_budget=output_budget,
            max_sequences_per_dp=max_sequences_per_dp,
            allow_dp=allow_dp, allow_ep=allow_ep, allow_pp=allow_pp,
            allow_pcp=allow_pcp, allow_dcp=allow_dcp,
            candidate_tps=candidate_tps, ep_candidates=ep_candidates,
            pp_candidates=pp_candidates, pcp_candidates=pcp_candidates,
            dcp_candidates=dcp_candidates, lock_parallel=lock_parallel,
            fixed_parallel=fixed_parallel, lock_program=lock_program,
            two_stage=two_stage, candidate_timeout_s=candidate_timeout_s,
            deadline_monotonic=deadline_monotonic, progress_callback=progress_callback,
            finalist_top_k=finalist_top_k, isolate_workers=isolate_workers,
            jobs=jobs, _finalist_sink=finalist_sink,
        )
        if owns_finalists:
            yield from _run_finalist_stage(
                finalist_sink, finalist_top_k=finalist_top_k,
                cache=SweepEvaluationCache(), output_jsonl=output_jsonl,
                progress_callback=progress_callback,
                isolate_workers=isolate_workers,
                candidate_timeout_s=candidate_timeout_s,
                deadline_monotonic=deadline_monotonic, phase=phase,
                scope=scope, context=context, q_tokens=base_q,
                emit_tokens=base_emit, sla_ms=sla_ms,
                output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp)
        return
    parsed = parse_case(case)
    cache = SweepEvaluationCache()
    base_q, base_emit = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=tokens_per_seq,
        emitted_tokens_per_seq=None)
    grids = devices_grid or ((num_devices,) if num_devices is not None
                             else (len(parsed.machine.dies),))
    for grid_index, grid in enumerate(grids):
        if grid is None:
            continue
        for row in _sweep_one_device_count(
                parsed, cache, devices=int(grid), candidates=candidates,
                sla_ms=sla_ms, context=context, phase=phase, scope=scope,
                output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp,
                allow_dp=allow_dp, allow_ep=allow_ep, allow_pp=allow_pp,
                allow_pcp=allow_pcp, allow_dcp=allow_dcp,
                dcp_candidates=dcp_candidates,
                candidate_tps=candidate_tps,
                ep_candidates=ep_candidates, pp_candidates=pp_candidates,
                pcp_candidates=pcp_candidates,
                lock_parallel=lock_parallel, fixed_parallel=fixed_parallel,
                two_stage=two_stage,
                candidate_timeout_s=candidate_timeout_s,
                deadline_monotonic=deadline_monotonic,
                output_jsonl=output_jsonl,
                resume=(resume or (output_jsonl is not None and grid_index > 0)),
                progress_callback=progress_callback, base_q=base_q,
                base_emit=base_emit, locked_plan=_locked_plan,
                finalist_top_k=finalist_top_k,
                isolate_workers=isolate_workers,
                jobs=jobs,
                finalist_sink=finalist_sink,
                pinned_devices=(num_devices is not None
                                or bool(devices_grid))):
            yield row
    if owns_finalists:
        yield from _run_finalist_stage(
            finalist_sink, finalist_top_k=finalist_top_k, cache=cache,
            output_jsonl=output_jsonl, progress_callback=progress_callback,
            isolate_workers=isolate_workers,
            candidate_timeout_s=candidate_timeout_s,
            deadline_monotonic=deadline_monotonic, phase=phase, scope=scope,
            context=context, q_tokens=base_q, emit_tokens=base_emit,
            sla_ms=sla_ms, output_budget=output_budget,
            max_sequences_per_dp=max_sequences_per_dp)


LOCAL_TP_CANDIDATE_KEYS = (
    "q_proj_tp", "o_proj_tp", "linear_proj_tp", "dcp_linear_tp",
    "mlp_tp", "shared_expert_tp", "lm_head_tp", "redundant_experts_per_rank",
)


def _parallel_batch_eval(
    parsed, cache, *, cand_list, devices, pinned_devices, phase, scope,
    context, base_q, base_emit, output_budget, sla_ms, max_sequences_per_dp,
    allow_dp, allow_ep, allow_pp, allow_pcp, two_stage, fixed_parallel,
    checkpoint, resume, stream, progress_callback, workers,
):
    """Evaluate independent candidates in parallel using pickle-safe plain dicts."""
    case_dict = parsed.to_dict()
    probe = {
        "phase": phase, "scope": scope, "context": int(context),
        "q_tokens": int(base_q), "emit_tokens": float(base_emit),
        "sla_ms": float(sla_ms), "output_budget": int(output_budget),
        "max_sequences_per_dp": int(max_sequences_per_dp),
    }
    prepared: list[tuple[str, Any, dict[str, Any]]] = []
    pending_indices: list[int] = []
    pending_tasks: list[dict[str, Any]] = []
    for raw_candidate in cand_list:
        cand_dict = {key: int(raw_candidate.get(key, 1))
                     for key in ("dp", "tp", "ep", "pp", "pcp")}
        cand_dict.update({key: raw_candidate[key]
                          for key in ("cp", "dcp", "cp_plan", *LOCAL_TP_CANDIDATE_KEYS)
                          if key in raw_candidate})
        cand_p = ParallelSpec.from_mapping(cand_dict)
        part = (parsed.runtime.pp_layer_partition
                if len(parsed.runtime.pp_layer_partition) == cand_p.pp else ())
        cand_devices = (
            int(devices) if pinned_devices
            else int(raw_candidate["num_devices"]) if raw_candidate.get("num_devices")
            else cand_p.world_size
        )
        ranks = resolve_ranks_per_node(parsed, cand_devices)
        cand_id = compute_sweep_candidate_id(
            parsed, cand_p, cand_devices=cand_devices, ranks_per_node=ranks,
            phase=phase, scope=scope, context=context, q_tokens=base_q,
            emit_tokens=base_emit, output_budget=output_budget, sla_ms=sla_ms,
            max_sequences_per_dp=max_sequences_per_dp, lock_parallel=False)
        meta = {
            "num_devices": int(cand_devices), "sla_ms": float(sla_ms),
            "context": int(context), "phase": phase, "scope": scope,
            "allow_dp": bool(allow_dp), "allow_ep": bool(allow_ep),
            "allow_pp": bool(allow_pp), "allow_pcp": bool(allow_pcp),
            "output_budget": int(output_budget),
            "emitted_tokens_per_seq": float(base_emit),
            "two_stage": bool(two_stage),
            "parallel_signature": str(fixed_parallel or {}),
        }
        if dict(parsed.machine.assumptions).get("derivation.base"):
            die = parsed.machine.dies[0] if parsed.machine.dies else None
            meta["derived_row_fields"] = {
                "derived_sku": parsed.machine.sku,
                "derivation": {
                    "sku": parsed.machine.sku,
                    "aic_count": die.compute_units if die else None,
                    "aiv_count": dict(die.resource_units).get("vector") if die else None,
                    "memory_bytes": die.memory_bytes if die else None,
                    "assumptions": dict(parsed.machine.assumptions),
                },
            }
        if resume and cand_id in checkpoint and checkpoint[cand_id].stage != STAGE_SCREEN:
            row = read_checkpoint_row(stream, checkpoint[cand_id])
            row["resumed_from_checkpoint"] = True
            row["elapsed_ms"] = 0.0
            prepared.append(("resumed", row, meta))
            continue
        common = {
            CANDIDATE_ID_KEY: cand_id,
            "parallel": _parallel_short(cand_p),
            "parallel_spec": parallel_report(cand_p),
            "num_devices": int(cand_devices),
            "ranks_per_node": int(ranks),
            **kv_replication_fields(parsed.model, cand_p),
        }
        pending_indices.append(len(prepared))
        pending_tasks.append({
            "case_dict": case_dict,
            "parallel_spec": cand_p.to_dict(),
            "num_devices": int(cand_devices),
            "part": tuple(part or ()),
            "common": common,
            "probe": probe,
        })
        prepared.append(("pending", None, meta))
    if pending_tasks:
        pool_size = min(workers, len(pending_tasks))
        with ProcessPoolExecutor(max_workers=pool_size) as pool:
            futures = [pool.submit(evaluate_candidate_worker, task)
                       for task in pending_tasks]
            fut_iter = iter(futures)
            computed_seen = 0
            for kind, row, meta in prepared:
                if kind == "pending":
                    row = next(fut_iter).result()
                    if computed_seen > 0:
                        cache.model_spec_reuses += 1
                    computed_seen += 1
                    if row.get("evaluation_stage") == STATIC_STAGE:
                        if row.get("feasibility_status") == "capacity_infeasible":
                            cache.static_capacity_rejections += 1
                        else:
                            cache.static_topology_rejections += 1
                    _screen_row_meta(row, cache, meta)
                    _write_row(stream, row)
                    if progress_callback is not None:
                        progress_callback(row)
                    yield row
                else:
                    _screen_row_meta(row, cache, meta)
                    yield row
        return
    for _, row, meta in prepared:
        _screen_row_meta(row, cache, meta)
        yield row


def _sweep_candidate_description(
    parsed, raw_candidate, locked, *, devices, pinned_devices, phase, scope, context,
    base_q, base_emit, output_budget, sla_ms, max_sequences_per_dp, lock_parallel,
    allow_dp, allow_ep, allow_pp, allow_pcp, two_stage, fixed_parallel,
):
    cand_dict = {key: int(raw_candidate.get(key, 1))
                 for key in ("dp", "tp", "ep", "pp", "pcp")}
    cand_dict.update({key: raw_candidate[key]
                      for key in ("cp", "dcp", "cp_plan",
                                  *LOCAL_TP_CANDIDATE_KEYS)
                      if key in raw_candidate})
    cand_p = locked["parallel"] if locked is not None else None
    if cand_p is None:
        cand_p = ParallelSpec.from_mapping(cand_dict)
        part = (parsed.runtime.pp_layer_partition
                if len(parsed.runtime.pp_layer_partition) == cand_p.pp
                else ())
        if locked is not None:
            cand_devices = cand_p.world_size
        elif pinned_devices:
            cand_devices = int(devices)
        elif raw_candidate.get("num_devices"):
            cand_devices = int(raw_candidate["num_devices"])
        else:
            cand_devices = cand_p.world_size
    else:
        cand_devices = cand_p.world_size
        part = locked["pp_layer_partition"]
    ranks = resolve_ranks_per_node(parsed, cand_devices)
    cand_id = compute_sweep_candidate_id(
        parsed, cand_p, cand_devices=cand_devices,
        ranks_per_node=ranks, phase=phase, scope=scope,
        context=context, q_tokens=base_q, emit_tokens=base_emit,
        output_budget=output_budget, sla_ms=sla_ms,
        max_sequences_per_dp=max_sequences_per_dp,
        lock_parallel=lock_parallel)
    meta = {
        "num_devices": int(cand_devices), "sla_ms": float(sla_ms),
        "context": int(context), "phase": phase, "scope": scope,
        "allow_dp": bool(allow_dp), "allow_ep": bool(allow_ep),
        "allow_pp": bool(allow_pp), "allow_pcp": bool(allow_pcp),
        "output_budget": int(output_budget),
        "emitted_tokens_per_seq": float(base_emit),
        "two_stage": bool(two_stage),
        "parallel_signature": str(fixed_parallel or {}),
    }
    if dict(parsed.machine.assumptions).get("derivation.base"):
        die = parsed.machine.dies[0] if parsed.machine.dies else None
        meta["derived_row_fields"] = {
            "derived_sku": parsed.machine.sku,
            "derivation": {
                "sku": parsed.machine.sku,
                "aic_count": die.compute_units if die else None,
                "aiv_count": dict(die.resource_units).get("vector") if die else None,
                "memory_bytes": die.memory_bytes if die else None,
                "assumptions": dict(parsed.machine.assumptions),
            },
        }
    return cand_p, cand_devices, part, ranks, cand_id, meta


def _evaluate_validated_sweep_candidate(
    parsed, cache, cand_case, cand_p, part, common, static, meta, promoted, *,
    phase, scope, context, base_q, base_emit, sla_ms, output_budget,
    max_sequences_per_dp, two_stage, cand_devices, ranks, cand_id, started,
):
    if not static["ok"]:
        if static["kind"] == "capacity":
            cache.static_capacity_rejections += 1
        else:
            cache.static_topology_rejections += 1
        row = static_row(
            common, reason=str(static["reason"]),
            kind=str(static["kind"]),
            elapsed_ms=(time.monotonic() - started) * MS_PER_S)
    else:
        b_vram = int(static["b_vram_max_joint"])
        eff_max = min(int(max_sequences_per_dp), b_vram)
        ctx = EvaluationContext.create(cand_case)
        cache.evaluation_context_misses += 1
        args = dict(
            common=common, ctx=ctx, phase=phase, scope=scope,
            context=context, q_tokens=base_q,
            emit_tokens=base_emit, sla_ms=sla_ms,
            output_budget=output_budget,
            max_sequences_per_dp=max_sequences_per_dp,
            b_vram_max=b_vram, static=static)
        if two_stage:
            row = b1_screen(
                parsed, cand_case, cand_p, **args)
            if row["b1_sla_pass"]:
                promoted.append(
                    {**dict(row), "cand_case": cand_case,
                     "ctx": ctx,
                     "part": tuple(part or ()),
                     "parsed": parsed,
                     "meta": dict(meta)})
        else:
            key = None
            if (not cand_p.ep_spans_dp
                    and not local_tp_spans_dp(cand_p)
                    and not cand_p.cp_spans_dp
                    and cand_p.rank_table is None):
                key = (
                    _derivation_key(parsed), phase, scope,
                    int(context), int(base_q), float(base_emit),
                    int(output_budget), float(sla_ms), eff_max,
                    int(cand_p.tp), int(cand_p.ep),
                    int(cand_p.pp), int(cand_p.pcp),
                    int(cand_p.cp),
                    tuple(cand_p.local_tp_widths),
                    int(cand_p.redundant_experts_per_rank),
                    tuple(part or ()),
                )
            base_row = (cache.replica_result(key)
                        if key is not None else None)
            if base_row is not None:
                row = project_replica_row(
                    base_row, cand_p=cand_p,
                    devices=cand_devices,
                    ranks_per_node=ranks,
                    candidate_id=cand_id)
                row["elapsed_ms"] = (
                    time.monotonic() - started) * MS_PER_S
            else:
                row = full_row(
                    parsed, cand_case, cand_p, **args)
                if key is not None:
                    cache.store_replica_result(key, row)
        cache.prepared_program_hits += int(
            ctx.prepared_programs.hits)
        cache.prepared_program_misses += int(
            ctx.prepared_programs.misses)
    return row


def _run_sweep_candidate(
    parsed, cache, cand_p, cand_devices, part, ranks, cand_id, meta, promoted, *,
    locked, index, phase, scope, context, base_q, base_emit, sla_ms, output_budget,
    max_sequences_per_dp, two_stage, isolate_workers, candidate_timeout_s,
    deadline_monotonic, stream, progress_callback,
):
    common = {
        CANDIDATE_ID_KEY: cand_id,
        "parallel": _parallel_short(cand_p),
        "parallel_spec": parallel_report(cand_p),
        "num_devices": int(cand_devices),
        "ranks_per_node": int(ranks),
        **kv_replication_fields(parsed.model, cand_p),
    }
    token = set_deadline(candidate_timeout_s, deadline_monotonic)
    started = time.monotonic()
    try:
        check_active_deadline()
        if locked is not None:
            row = _locked_row(
                parsed, locked, sla_ms=sla_ms,
                output_budget=output_budget, devices=cand_devices,
                ranks_per_node=ranks)
            row.update(common)
            row.update({"elapsed_ms": (time.monotonic() - started) * MS_PER_S})
        elif isolate_workers and candidate_timeout_s is not None:
            row = _isolated_row(
                parsed, cand_p, cand_devices, part, common, phase,
                scope, context, base_q, base_emit, sla_ms,
                output_budget, max_sequences_per_dp,
                stage=(STAGE_SCREEN if two_stage else STAGE_FULL),
                timeout_s=candidate_timeout_s,
                deadline_monotonic=deadline_monotonic)
            if row.get("evaluation_stage") == STATIC_STAGE:
                if row.get("feasibility_status") == "capacity_infeasible":
                    cache.static_capacity_rejections += 1
                else:
                    cache.static_topology_rejections += 1
            elif two_stage:
                promoted.append({**dict(row),
                                 "part": tuple(part or ()),
                                 "parsed": parsed,
                                 "meta": dict(meta)})
        else:
            machine = cache.machine(
                _derivation_key(parsed), cand_devices,
                lambda: materialize_machine_for_devices(
                    parsed, parsed, cand_devices))
            if index > 0:
                cache.model_spec_reuses += 1
            cand_case = build_candidate_case(
                parsed, machine, cand_p, part)
            try:
                static = validate_candidate_statically(
                    parsed, cand_p, cand_devices=cand_devices,
                    cand_machine=machine, part=part, phase=phase,
                    context=context, q_tokens=base_q,
                    emit_tokens=base_emit,
                    output_budget=output_budget,
                    max_sequences_per_dp=max_sequences_per_dp)
            except UnmodeledCapabilityError:
                # A core hole is a refused capability, never a rejected
                # or infeasible topology candidate.
                raise
            except (ModelingError, ValueError) as exc:
                cache.static_topology_rejections += 1
                row = static_row(
                    common, reason=str(exc), kind="topology",
                    elapsed_ms=(time.monotonic() - started) * MS_PER_S)
                _screen_row_meta(row, cache, meta)
                _write_row(stream, row)
                if progress_callback is not None:
                    progress_callback(row)
                yield row
                return
            check_active_deadline()
            row = _evaluate_validated_sweep_candidate(
                parsed, cache, cand_case, cand_p, part, common, static, meta, promoted,
                phase=phase, scope=scope, context=context, base_q=base_q, base_emit=base_emit,
                sla_ms=sla_ms, output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp, two_stage=two_stage,
                cand_devices=cand_devices, ranks=ranks, cand_id=cand_id, started=started)
        _screen_row_meta(row, cache, meta)
        _write_row(stream, row)
        if progress_callback is not None:
            progress_callback(row)
        yield row
    except TimeoutError:
        if (deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic):
            raise
        if candidate_timeout_s is None:
            raise
        elapsed = time.monotonic() - started
        row = timeout_row(common, elapsed_s=elapsed,
                          candidate_timeout_s=candidate_timeout_s)
        _screen_row_meta(row, cache, meta)
        _write_row(stream, row)
        if progress_callback is not None:
            progress_callback(row)
        yield row
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        row = static_row(
            common, reason=str(exc), kind="topology",
            elapsed_ms=(time.monotonic() - started) * MS_PER_S)
        row["evaluation_stage"] = FULL_STAGE
        _screen_row_meta(row, cache, meta)
        _write_row(stream, row)
        if progress_callback is not None:
            progress_callback(row)
        yield row
    finally:
        ACTIVE_DEADLINE_MONO.reset(token)


def _run_pareto_finalists(
    parsed, cache, promoted, meta, *, phase, scope, context, base_q, base_emit,
    sla_ms, output_budget, max_sequences_per_dp, isolate_workers,
    candidate_timeout_s, deadline_monotonic, stream, progress_callback,
):
    _b1_pareto([item for item in promoted
                if item.get("b1_sla_pass")])
    for item in promoted:
        if not item.get("b1_pareto_optimal"):
            continue
        common = {key: item[key] for key in
                  (CANDIDATE_ID_KEY, "parallel", "parallel_spec",
                   "num_devices", "ranks_per_node")}
        cand_p = ParallelSpec.from_mapping(item["parallel_spec"])
        started = time.monotonic()
        token = set_deadline(candidate_timeout_s, deadline_monotonic)
        try:
            if isolate_workers and candidate_timeout_s is not None:
                row = _isolated_row(
                    parsed, cand_p, int(item["num_devices"]),
                    item.get("part", ()), common, phase, scope,
                    context, base_q, base_emit, sla_ms,
                    output_budget, max_sequences_per_dp,
                    stage=STAGE_FULL, timeout_s=candidate_timeout_s,
                    deadline_monotonic=deadline_monotonic)
            else:
                row = full_row(
                    parsed, item["cand_case"], cand_p,
                    common=common, ctx=item["ctx"], phase=phase,
                    scope=scope, context=context, q_tokens=base_q,
                    emit_tokens=base_emit, sla_ms=sla_ms,
                    output_budget=output_budget,
                    max_sequences_per_dp=max_sequences_per_dp,
                    b_vram_max=int(item["b_vram_max"]),
                    static={"weight_bytes":
                            item.get("weight_bytes_per_die"),
                            "kv_bytes_per_token":
                                item.get("kv_bytes_per_token_per_die")},
                    screening=True)
            _screen_row_meta(row, cache, meta)
            _write_row(stream, row)
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
            _screen_row_meta(row, cache, meta)
            _write_row(stream, row)
            yield row
        finally:
            ACTIVE_DEADLINE_MONO.reset(token)


def _sweep_one_device_count(parsed, cache, *, devices, candidates, sla_ms,
                            context, phase, scope, output_budget,
                            max_sequences_per_dp, allow_dp, allow_ep,
                            allow_pp, allow_pcp, allow_dcp, dcp_candidates,
                            candidate_tps,
                            ep_candidates, pp_candidates, pcp_candidates,
                            lock_parallel, fixed_parallel, two_stage,
                            candidate_timeout_s, deadline_monotonic,
                            output_jsonl, resume, progress_callback, base_q,
                            base_emit, locked_plan, pinned_devices,
                            finalist_top_k=None, isolate_workers=False,
                            jobs=1, finalist_sink=None):
    path = Path(output_jsonl) if output_jsonl is not None else None
    checkpoint = _load_checkpoint(path) if (resume and path is not None) else {}
    stream = _open_stream(path, resume)
    locked = locked_plan
    if lock_parallel:
        _validate_locked_arguments(
            parsed, candidates=candidates, fixed_parallel=fixed_parallel,
            num_devices=devices, candidate_tps=candidate_tps,
            ep_candidates=ep_candidates, pp_candidates=pp_candidates,
            pcp_candidates=pcp_candidates)
        if locked is None:
            locked = _prepare_locked_plan(
                parsed, sla_ms=sla_ms, context=context, phase=phase,
                scope=scope, tokens_per_seq=base_q,
                output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp)
    if locked is not None:
        cand_list = [locked["parallel"].to_dict()]
    else:
        cand_list = enumerate_case_candidates(
            parsed, num_devices=devices, allow_dp=allow_dp, allow_ep=allow_ep,
            allow_pp=allow_pp, allow_pcp=allow_pcp, allow_dcp=allow_dcp,
            candidate_tps=candidate_tps, ep_candidates=ep_candidates,
            pp_candidates=pp_candidates, pcp_candidates=pcp_candidates,
            dcp_candidates=dcp_candidates, explicit=candidates)
    if candidates is not None and allow_dcp and dcp_candidates and locked is None:
        expanded: list[dict[str, Any]] = []
        for candidate in cand_list:
            for degree in dcp_candidates:
                degree = int(degree)
                if degree <= 1:
                    expanded.append(dict(candidate))
                else:
                    expanded.append({**dict(candidate), "cp": degree,
                                     "dcp": degree})
        cand_list = expanded
    if locked is None:
        for raw_candidate in cand_list:
            cand_dict = {key: int(raw_candidate.get(key, 1))
                         for key in ("dp", "tp", "ep", "pp", "pcp")}
            cand_dict.update({key: raw_candidate[key]
                              for key in ("cp", "dcp", "cp_plan", *LOCAL_TP_CANDIDATE_KEYS)
                              if key in raw_candidate})
            ParallelSpec.from_mapping(cand_dict)
    resolved_jobs = resolve_jobs(jobs, task_count=max(1, len(cand_list)))
    promoted: list[dict[str, Any]] = []
    try:
        if (resolved_jobs > 1 and locked is None and not two_stage
                and not isolate_workers and candidate_timeout_s is None
                and deadline_monotonic is None and len(cand_list) > 1):
            yield from _parallel_batch_eval(
                parsed, cache, cand_list=cand_list, devices=devices,
                pinned_devices=pinned_devices, phase=phase, scope=scope,
                context=context, base_q=base_q, base_emit=base_emit,
                output_budget=output_budget, sla_ms=sla_ms,
                max_sequences_per_dp=max_sequences_per_dp,
                allow_dp=allow_dp, allow_ep=allow_ep, allow_pp=allow_pp,
                allow_pcp=allow_pcp, two_stage=two_stage,
                fixed_parallel=fixed_parallel, checkpoint=checkpoint,
                resume=resume, stream=stream,
                progress_callback=progress_callback, workers=resolved_jobs)
            return
        for index, raw_candidate in enumerate(cand_list):
            if (deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic):
                raise TimeoutError("sweep_outer_deadline_exceeded")
            cand_p, cand_devices, part, ranks, cand_id, meta = _sweep_candidate_description(
                parsed, raw_candidate, locked, devices=devices, pinned_devices=pinned_devices,
                phase=phase, scope=scope, context=context, base_q=base_q, base_emit=base_emit,
                output_budget=output_budget, sla_ms=sla_ms, max_sequences_per_dp=max_sequences_per_dp,
                lock_parallel=lock_parallel, allow_dp=allow_dp, allow_ep=allow_ep,
                allow_pp=allow_pp, allow_pcp=allow_pcp, two_stage=two_stage,
                fixed_parallel=fixed_parallel,
            )
            if (resume and cand_id in checkpoint
                    and checkpoint[cand_id].stage != STAGE_SCREEN):
                row = read_checkpoint_row(stream, checkpoint[cand_id])
                row["resumed_from_checkpoint"] = True
                row["elapsed_ms"] = 0.0
                _screen_row_meta(row, cache, meta)
                yield row
                continue
            yield from _run_sweep_candidate(
                parsed, cache, cand_p, cand_devices, part, ranks, cand_id, meta, promoted,
                locked=locked, index=index, phase=phase, scope=scope, context=context,
                base_q=base_q, base_emit=base_emit, sla_ms=sla_ms, output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp, two_stage=two_stage,
                isolate_workers=isolate_workers, candidate_timeout_s=candidate_timeout_s,
                deadline_monotonic=deadline_monotonic, stream=stream, progress_callback=progress_callback,
            )
        if two_stage and promoted and finalist_sink is not None:
            for item in promoted:
                retain_finalist(finalist_sink, item, finalist_top_k)
        elif two_stage and promoted:
            yield from _run_pareto_finalists(
                parsed, cache, promoted, meta, phase=phase, scope=scope, context=context,
                base_q=base_q, base_emit=base_emit, sla_ms=sla_ms, output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp, isolate_workers=isolate_workers,
                candidate_timeout_s=candidate_timeout_s, deadline_monotonic=deadline_monotonic,
                stream=stream, progress_callback=progress_callback,
            )
    finally:
        if stream is not None:
            stream.close()


def num_devices_grid_hint(raw: Mapping[str, Any]) -> bool:
    return "num_devices" in raw


def reconstruct_sweep_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct the exact aggregate report from streamed/checkpoint rows."""
    compacted: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    meta: dict[str, Any] = {}
    final_meta: dict[str, Any] = {}
    for raw in rows:
        row = dict(raw)
        if isinstance(row.get("sweep_meta"), Mapping):
            if not meta:
                meta = dict(row["sweep_meta"])
            final_meta = dict(row["sweep_meta"])
        candidate_id = str(row.get(CANDIDATE_ID_KEY, ""))
        if not candidate_id:
            candidate_id = f"row{len(order)}"
            row[CANDIDATE_ID_KEY] = candidate_id
        if candidate_id not in compacted:
            order.append(candidate_id)
        compacted[candidate_id] = row
    final_rows = []
    for candidate_id in order:
        row = dict(compacted[candidate_id])
        row.pop("sweep_meta", None)
        # Wall-clock is a per-run diagnostic; the reconstructed report is a
        # deterministic function of the evaluated row content.
        row["elapsed_ms"] = 0.0
        final_rows.append(row)
    pareto = compute_pareto_frontier(final_rows)
    two_stage = bool(meta.get("two_stage", False))
    b1_rows = [dict(row) for row in final_rows
               if row.get("evaluation_stage") == B1_STAGE]
    b1_front = _b1_pareto(b1_rows) if two_stage else []
    static_rejected = sum(1 for row in final_rows
                          if row.get("evaluation_stage") == STATIC_STAGE)
    promoted = sum(1 for row in final_rows
                   if row.get("evaluation_stage") == FULL_STAGE)
    b1_evaluated = sum(1 for row in final_rows
                       if row.get("evaluation_stage") == B1_STAGE) + promoted
    b1_rejected = sum(1 for row in final_rows
                      if row.get("evaluation_stage") == B1_STAGE
                      and row.get("b1_sla_pass") is False)
    locked = any(row.get("evaluation_stage") == LOCKED_STAGE
                 for row in final_rows)
    report = _sweep_report(
        final_rows, num_devices=int(meta.get("num_devices", 0) or 0),
        sla_ms=float(meta.get("sla_ms", 0.0) or 0.0),
        context=int(meta.get("context", 0) or 0),
        phase=str(meta.get("phase", "decode")),
        scope=str(meta.get("scope", "body")),
        allow_dp=bool(meta.get("allow_dp", True)),
        allow_ep=bool(meta.get("allow_ep", True)),
        allow_pp=bool(meta.get("allow_pp", False)),
        allow_pcp=bool(meta.get("allow_pcp", False)),
        locked=True if locked else None,
        parallel_signature=str(meta.get("parallel_signature", "")))
    report["pareto_frontier"] = pareto
    if final_meta.get("cache_stats"):
        report["cache_stats"] = dict(final_meta["cache_stats"])
    if two_stage:
        report["b1_pareto_frontier"] = b1_front
        report["screening_summary"] = {
            "two_stage": True, "total_candidates": len(final_rows),
            "static_rejected": static_rejected, "b1_evaluated": b1_evaluated,
            "b1_sla_rejected": b1_rejected,
            "promoted_to_full_sla": promoted,
        }
    return report


def _sweep_report(rows, *, num_devices, sla_ms, context, phase, scope,
                  allow_dp, allow_ep, allow_pp, allow_pcp,
                  hardware_derivations=None, locked=None,
                  parallel_signature="") -> dict[str, Any]:
    domain = {"num_devices": int(num_devices), "allow_dp": allow_dp,
              "allow_ep": allow_ep, "allow_pp": allow_pp,
              "allow_pcp": allow_pcp, "candidate_count": len(rows)}
    if hardware_derivations:
        domain["hardware_derivations"] = hardware_derivations
    return {
        "schema": "mindie-sim.topology-sweep/1",
        "num_devices": int(num_devices), "sla_ms": float(sla_ms),
        "context": int(context), "phase": phase, "scope": scope,
        "search_domain": domain, "candidates": rows,
        "pareto_frontier": compute_pareto_frontier(rows),
        "plan_selection": ("fixed_parallel" if locked is not None
                           else "reselected_parallel_per_candidate"),
        "program_selection": "fixed" if locked is not None else "reselected",
        "program_selection_basis": (
            "retained_baseline_prepared_plan_reused" if locked is not None
            else "reselected_per_candidate"),
        "parallel_signature": parallel_signature,
        "changed_hardware_fields": (),
        "comparison_boundary": (
            "each_row_is_one_hardware_variant_with_retained_parallel_batch_and_plan"
            if locked is not None else
            "each_row_is_one_complete_candidate_no_cross_candidate_metric_stitching"),
        "pareto": {
            "basis": "exact_multi_objective_dominance_on_single_candidate_metrics",
            "frontier_is_selected_candidates_not_a_global_optimum_proof": True,
        },
    }


def sweep_topologies(case: CaseSpec | Mapping[str, Any], **kwargs) -> dict[str, Any]:
    """Evaluate a bounded candidate set and return the reconstructed report."""
    return reconstruct_sweep_summary(sweep_topologies_iter(case, **kwargs))


solve_hardware_sla_throughput = solve_sla_sizing


sweep_topology_grid_iter = sweep_topologies_iter
sweep_topology_grid = sweep_topologies
