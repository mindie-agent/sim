"""Streaming topology sweeps and disaggregated Prefill/Decode Pareto analysis.

[Role] Enumerate structural parallel candidates, stream deterministic rows,
  checkpoint/resume them as JSONL, statically reject impossible candidates
  before graph lowering, and compose P/D stage rates into a Pareto frontier.
[Invariants] ``sweep_topologies(iter(...))`` reconstructs the exact same report.
  Candidate identity is a pure hash of model/runtime, hardware, phase boundary
  and complete parallel spec. Screening estimates never masquerade as full SLA
  rows. JSONL rows are written before they are yielded.
[Timeouts & Locks]
  Cooperative monotonic deadlines only; no signals, threads or subprocesses.
[Side-Effect Policy]
  Input cases are never mutated. Only the explicit ``output_jsonl`` path is
  written, one compact line per evaluated row.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
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
from mindie_sim.prediction.local_tp_groups import local_tp_spans_dp
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep_engine import (
    B1_STAGE, BASELINE_SCENARIO, CANDIDATE_ID_KEY, DEFAULT_CONTEXT,
    DEFAULT_MAX_SEQUENCES, DEFAULT_OUTPUT_BUDGET, DEFAULT_SLA_MS,
    FULL_STAGE, LOCKED_PROGRAM_ERROR, LOCKED_STAGE, STATIC_STAGE,
    TIMEOUT_STAGE,
    SweepEvaluationCache, b1_screen, build_candidate_case,
    compute_pareto_frontier, compute_sweep_candidate_id,
    enumerate_case_candidates, full_row, parallel_report,
    project_replica_row, set_deadline, static_row, timeout_row,
    validate_candidate_statically,
)
from mindie_sim.sweep_locked import (
    locked_row as _locked_row, prepare_locked_plan as _prepare_locked_plan,
    validate_locked_arguments as _validate_locked_arguments,
)
from mindie_sim.sweep_pd import sweep_pd_disaggregation_ratios
from mindie_sim.sweep_resize import (
    check_pos_float, check_pos_int, enumerate_parallel_candidates,
    materialize_machine_for_devices, resolve_ranks_per_node,
)

_materialize_machine_for_devices = materialize_machine_for_devices

DEFAULT_PROMPT_TOKENS = 4096
DEFAULT_OUTPUT_TOKENS = 1024
DEFAULT_TTFT_SLA_MS = 3000.0
DEFAULT_KV_TRANSFER_BW_GBPS = 100.0
MS_PER_S = 1000.0
PARETO_EPS = 1e-12


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


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    records: dict[str, dict[str, Any]] = {}
    valid_parts: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break
            raise ValueError(
                f"corrupt_checkpoint_jsonl:line_{index + 1}") from exc
        if not isinstance(row, dict) or not row.get(CANDIDATE_ID_KEY):
            continue
        records[str(row[CANDIDATE_ID_KEY])] = row
        valid_parts.append(stripped + "\n")
    repaired = "".join(valid_parts)
    if repaired != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(repaired, encoding="utf-8")
    return records


def _open_stream(path: Path | None, resume: bool):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a" if resume else "w", encoding="utf-8")


def _write_row(stream, row: Mapping[str, Any]) -> None:
    if stream is None:
        return
    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False) + "\n")
    stream.flush()


def _b1_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _screen_row_meta(row: dict[str, Any], cache: SweepEvaluationCache,
                     meta: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(meta)
    merged["cache_stats"] = cache.stats()
    row["sweep_meta"] = merged
    return row


def sweep_topologies_iter(
    case: CaseSpec | Mapping[str, Any], *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    num_devices: int | None = None,
    num_devices_grid: Sequence[int] | None = None,
    sla_ms: float = DEFAULT_SLA_MS,
    context: int = DEFAULT_CONTEXT,
    phase: str = "decode",
    scope: str = "body",
    tokens_per_seq: int | None = None,
    output_budget: int = DEFAULT_OUTPUT_BUDGET,
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
) -> Iterator[dict[str, Any]]:
    """Stream deterministic candidate rows, optionally checkpointing each row."""
    _validate_sweep_arguments(
        sla_ms=sla_ms, context=context, output_budget=output_budget,
        max_sequences_per_dp=max_sequences_per_dp, phase=phase, scope=scope,
        tokens_per_seq=tokens_per_seq, two_stage=two_stage,
        candidate_timeout_s=candidate_timeout_s, resume=resume,
        output_jsonl=output_jsonl)
    if lock_program and not lock_parallel:
        raise ModelingError(LOCKED_PROGRAM_ERROR)
    devices_grid = tuple(int(item) for item in (num_devices_grid or ()))
    if devices_grid and num_devices is not None:
        raise ValueError("num_devices_and_grid_are_mutually_exclusive")
    if hardware_derivations:
        base = parse_case(case)
        for derivation in hardware_derivations:
            from mindie_sim.hardware_derivation import derived_machine
            declare = dict(derivation)
            sku = str(declare.pop("sku", base.machine.sku or "derived"))
            derived = derived_machine(base.machine, {**declare, "sku": sku})
            sub_case = replace(base, machine=derived)
            for row in sweep_topologies_iter(
                    sub_case, candidates=candidates,
                    num_devices=num_devices, num_devices_grid=num_devices_grid,
                    sla_ms=sla_ms, context=context, phase=phase, scope=scope,
                    tokens_per_seq=tokens_per_seq, output_budget=output_budget,
                    max_sequences_per_dp=max_sequences_per_dp,
                    allow_dp=allow_dp, allow_ep=allow_ep, allow_pp=allow_pp,
                    allow_pcp=allow_pcp, allow_dcp=allow_dcp,
                    candidate_tps=candidate_tps, ep_candidates=ep_candidates,
                    pp_candidates=pp_candidates, pcp_candidates=pcp_candidates,
                    dcp_candidates=dcp_candidates, lock_parallel=lock_parallel,
                    fixed_parallel=fixed_parallel, lock_program=lock_program,
                    two_stage=two_stage,
                    candidate_timeout_s=candidate_timeout_s,
                    deadline_monotonic=deadline_monotonic,
                    output_jsonl=None, resume=False,
                    progress_callback=progress_callback):
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
        return
    parsed = parse_case(case)
    cache = SweepEvaluationCache()
    base_q, base_emit = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=tokens_per_seq,
        emitted_tokens_per_seq=None)
    grids = devices_grid or ((num_devices,) if num_devices is not None
                             else (len(parsed.machine.dies),))
    for grid in grids:
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
                output_jsonl=output_jsonl, resume=resume,
                progress_callback=progress_callback, base_q=base_q,
                base_emit=base_emit, locked_plan=_locked_plan,
                pinned_devices=(num_devices is not None
                                or bool(devices_grid))):
            yield row


def _sweep_one_device_count(parsed, cache, *, devices, candidates, sla_ms,
                            context, phase, scope, output_budget,
                            max_sequences_per_dp, allow_dp, allow_ep,
                            allow_pp, allow_pcp, allow_dcp, dcp_candidates,
                            candidate_tps,
                            ep_candidates, pp_candidates, pcp_candidates,
                            lock_parallel, fixed_parallel, two_stage,
                            candidate_timeout_s, deadline_monotonic,
                            output_jsonl, resume, progress_callback, base_q,
                            base_emit, locked_plan, pinned_devices):
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
            allow_pp=allow_pp, allow_pcp=allow_pcp,
            candidate_tps=candidate_tps, ep_candidates=ep_candidates,
            pp_candidates=pp_candidates, pcp_candidates=pcp_candidates,
            explicit=candidates)
    if allow_dcp and dcp_candidates and locked is None:
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
    promoted: list[dict[str, Any]] = []
    try:
        for index, raw_candidate in enumerate(cand_list):
            if (deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic):
                raise TimeoutError("sweep_outer_deadline_exceeded")
            cand_dict = {key: int(raw_candidate.get(key, 1))
                         for key in ("dp", "tp", "ep", "pp", "pcp")}
            cand_dict.update({key: raw_candidate[key]
                              for key in ("cp", "dcp", "cp_plan")
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
            if resume and cand_id in checkpoint and not (
                    two_stage and checkpoint[cand_id].get("evaluation_stage")
                    == B1_STAGE):
                row = dict(checkpoint[cand_id])
                row["resumed_from_checkpoint"] = True
                row["elapsed_ms"] = 0.0
                _screen_row_meta(row, cache, meta)
                yield row
                continue
            common = {
                CANDIDATE_ID_KEY: cand_id,
                "parallel": {key: int(getattr(cand_p, key))
                             for key in ("dp", "tp", "ep", "pp", "pcp")},
                "parallel_spec": parallel_report(cand_p),
                "num_devices": int(cand_devices),
                "ranks_per_node": int(ranks),
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
                        continue
                    check_active_deadline()
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
                                     "ctx": ctx})
                        else:
                            key = None
                            if (not cand_p.ep_spans_dp
                                    and not local_tp_spans_dp(cand_p)
                                    and cand_p.rank_table is None):
                                key = (
                                    _derivation_key(parsed), phase, scope,
                                    int(context), int(base_q), float(base_emit),
                                    int(output_budget), float(sla_ms), eff_max,
                                    int(cand_p.tp), int(cand_p.ep),
                                    int(cand_p.pp), int(cand_p.pcp),
                                    int(cand_p.cp), int(cand_p.o_proj_tp),
                                    int(cand_p.linear_proj_tp),
                                    int(cand_p.lm_head_tp),
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
        if two_stage and promoted:
            _b1_pareto([item for item in promoted if item.get("b1_sla_pass")])
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
                    row = full_row(
                        parsed, item["cand_case"], cand_p,
                        common=common, ctx=item["ctx"], phase=phase,
                        scope=scope, context=context, q_tokens=base_q,
                        emit_tokens=base_emit, sla_ms=sla_ms,
                        output_budget=output_budget,
                        max_sequences_per_dp=max_sequences_per_dp,
                        b_vram_max=int(item["b_vram_max"]),
                        static={"weight_bytes": item.get("weight_bytes_per_die"),
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
                      and not row.get("b1_sla_pass"))
    locked = any(row.get("evaluation_stage") == LOCKED_STAGE
                 for row in final_rows)
    report = _sweep_report(
        final_rows, num_devices=int(meta.get("num_devices", 0) or 0),
        sla_ms=float(meta.get("sla_ms", DEFAULT_SLA_MS)),
        context=int(meta.get("context", DEFAULT_CONTEXT)),
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
